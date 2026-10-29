# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference (torch, no device) for the KDA prefill scan over a physical chunk whose tail holds FULLY padded
32-row recurrence chunks (Stage 4 v2 finding: a 65-token prompt in a 128-row chunk left a wrong KDA state).

The device path (`tt/kda.py` `_chunk_recurrence`, stage1 scan; the hoisted scan in `tt/fused_decoder.py` has the same
structure) is modelled exactly at the level that matters:
- `prepare_chunk_recurrence` zeroes k, v, g, beta of the padded rows of the PARTIAL chunk (identity steps) and SKIPS every
  chunk past actual_end (reader / compute / writer `continue`), so that chunk's t_inv / v_beta / kd / k_dec_t keep
  whatever the output buffers held (modelled as random values or NaN);
- the scan masks the decay of padded rows (chunk_decay = exp(0) = 1 for a dead chunk) and computes
  U_n = T_n^-1 (v_beta_n - kd_n S),  S <- chunk_decay_n * S + k_dec_n^T U_n.
Reference: the token recurrence over the valid tokens only (kda.py docstring):
  S <- diag(exp g_t) S;  S += k_t (beta_t (v_t - S^T k_t))^T.
Checks: the chunked scan equals the token recurrence when every chunk is live; with a dead chunk the UNGATED scan
diverges (the bug) and the GATED scan (`keep_if_dead`: keep S for a chunk with no valid row) equals the reference, also
when the stale terms are NaN.

run: python -m pytest -q tests/test_kda_padded_chunk_cpu.py   (or python tests/test_kda_padded_chunk_cpu.py)
"""

import pytest
import torch

CHUNK = 32
H, K, V = 2, 8, 8


def token_recurrence(k, v, g, beta, s0, n):
    """k [T,H,K] (l2-normalised), v [T,H,V], g [T,H,K] log-decay, beta [T,H]; over the first n rows."""
    s = s0.clone()
    for t in range(n):
        s = torch.exp(g[t])[:, :, None] * s  # [H,K,V]
        pred = torch.einsum("hkv,hk->hv", s, k[t])
        u = beta[t][:, None] * (v[t] - pred)  # [H,V]
        s = s + torch.einsum("hk,hv->hkv", k[t], u)
    return s


def chunk_terms(k, v, g, beta):
    """One 32-row chunk (all rows treated as given; padded rows already zeroed) -> (T^-1, v_beta, kd, k_dec, decay)
    with cumulative decays G_r = sum_{j<=r} g_j:
      kd_r = beta_r exp(G_r) k_r,  A_rs = beta_r sum_c k_s[c] k_r[c] exp(G_r[c] - G_s[c]) (s < r),  T = (I + A)^-1,
      v_beta_r = beta_r v_r,  k_dec_s = exp(G_C - G_s) k_s,  decay = exp(G_C)."""
    c = k.shape[0]
    G = torch.cumsum(g, dim=0)  # [C,H,K]
    out_t, out_vb, out_kd, out_kdec = [], [], [], []
    for h in range(H):
        Gh, kh, vh, bh = G[:, h], k[:, h], v[:, h], beta[:, h]
        diff = Gh[:, None, :] - Gh[None, :, :]  # [r,s,K]
        A = torch.einsum("sc,rc,rsc->rs", kh, kh, torch.exp(torch.clamp(diff, max=0.0)))
        A = torch.tril(A, diagonal=-1) * bh[:, None]
        out_t.append(torch.linalg.inv(torch.eye(c, dtype=k.dtype) + A))
        out_vb.append(bh[:, None] * vh)
        out_kd.append(bh[:, None] * torch.exp(Gh) * kh)
        out_kdec.append(torch.exp(Gh[-1][None, :] - Gh) * kh)
    decay = torch.exp(G[-1])  # [H,K]
    return torch.stack(out_t), torch.stack(out_vb), torch.stack(out_kd), torch.stack(out_kdec), decay


def chunked_scan(k, v, g, beta, s0, n_valid, *, gated, stale="random", seed=0):
    """Device-model scan over a physical chunk of k.shape[0] rows with n_valid valid rows."""
    phys = k.shape[0]
    assert phys % CHUNK == 0
    gen = torch.Generator().manual_seed(seed)
    k, v, g, beta = k.clone(), v.clone(), g.clone(), beta.clone()
    k[n_valid:], v[n_valid:], g[n_valid:], beta[n_valid:] = 0.0, 0.0, 0.0, 0.0  # reader zeroes padded rows
    s = s0.clone()
    for n in range(phys // CHUNK):
        lo, hi = n * CHUNK, (n + 1) * CHUNK
        live = lo < n_valid
        if live:
            t_inv, vb, kd, kdec, decay = chunk_terms(k[lo:hi], v[lo:hi], g[lo:hi], beta[lo:hi])
        else:  # the op skipped the chunk: stale output buffers; the scan's own decay mask gives exp(0) = 1
            fill = (
                (lambda *sh: torch.full(sh, float("nan"), dtype=k.dtype))
                if stale == "nan"
                else (lambda *sh: torch.randn(*sh, generator=gen, dtype=k.dtype))
            )
            t_inv, vb, kd, kdec = fill(H, CHUNK, CHUNK), fill(H, CHUNK, V), fill(H, CHUNK, K), fill(H, CHUNK, K)
            decay = torch.ones(H, K, dtype=k.dtype)
        u = torch.einsum("hrs,hsv->hrv", t_inv, vb - torch.einsum("hsk,hkv->hsv", kd, s))  # [H,C,V]
        new_s = decay[:, :, None] * s + torch.einsum("hsk,hsv->hkv", kdec, u)
        s = torch.where(torch.tensor(live), new_s, s) if gated else new_s  # keep_if_dead: a lane select
    return s


def inputs(phys, seed=1):
    gen = torch.Generator().manual_seed(seed)
    k = torch.nn.functional.normalize(torch.randn(phys, H, K, generator=gen, dtype=torch.float64), dim=-1)
    v = torch.randn(phys, H, V, generator=gen, dtype=torch.float64)
    g = -5.0 * torch.sigmoid(torch.randn(phys, H, K, generator=gen, dtype=torch.float64)) * 0.05  # slow decays
    beta = torch.sigmoid(torch.randn(phys, H, generator=gen, dtype=torch.float64))
    s0 = 0.1 * torch.randn(H, K, V, generator=gen, dtype=torch.float64)
    return k, v, g, beta, s0


def close(a, b):
    return bool(torch.isfinite(a).all()) and torch.allclose(a, b, rtol=1e-9, atol=1e-9)


def test_chunked_scan_matches_token_recurrence():
    k, v, g, beta, s0 = inputs(128)
    ref = token_recurrence(k, v, g, beta, s0, 128)
    assert close(chunked_scan(k, v, g, beta, s0, 128, gated=False), ref)


# (valid rows, physical rows, holds a fully padded 32-row chunk)
CASES = [(65, 128, True), (130, 256, True), (1999, 2048, True), (100, 128, False), (2047, 2048, False), (20, 32, False)]


@pytest.mark.parametrize("n,phys,dead", CASES)
def test_padded_chunk_state(n, phys, dead):
    k, v, g, beta, s0 = inputs(phys)
    ref = token_recurrence(k, v, g, beta, s0, n)
    gated = chunked_scan(k, v, g, beta, s0, n, gated=True)
    ungated = chunked_scan(k, v, g, beta, s0, n, gated=False)
    assert close(gated, ref), "gated scan must equal the token recurrence over the valid rows"
    if dead:
        assert not close(ungated, ref), "ungated scan must show the bug (stale terms added to the state)"
        # NaN in the stale buffers: the select keeps the state finite and exact
        assert close(chunked_scan(k, v, g, beta, s0, n, gated=True, stale="nan"), ref)
        assert not bool(torch.isfinite(chunked_scan(k, v, g, beta, s0, n, gated=False, stale="nan")).all())
    else:
        assert close(ungated, ref), "no dead chunk: gating changes nothing"


if __name__ == "__main__":
    test_chunked_scan_matches_token_recurrence()
    for n, phys, dead in CASES:
        test_padded_chunk_state(n, phys, dead)
        k, v, g, beta, s0 = inputs(phys)
        ref = token_recurrence(k, v, g, beta, s0, n)
        ung = chunked_scan(k, v, g, beta, s0, n, gated=False)
        rel = float((ung - ref).norm() / ref.norm())
        print(f"n={n:5d} phys={phys:5d} dead_chunk={dead!s:5}  ungated rel.err={rel:.3e}  gated == reference")
    print("ok")
