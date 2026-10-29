# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the host-p300 TP2 pair or the host-8chip 2x4 mesh): the CONTEXT-LENGTH AXIS of the served PMK stack
(program 2026-10-23 16:07:40Z: "per-chunk equivalence vs the op path ... at chunk positions 0 / 8,192 / 32,768 /
131,072 / 1,048,064 with the real KDA carry-in and the DSA pool selection from a populated store"). The PMK is per
chunk; what changes with the context length is the CARRY-IN: the KDA recurrent state S / C the chunk starts from and
the sparse_dsa pooled-key store / latent pages it selects from. Every PMK silicon row before this one started from
zeroed S / C at position 0.

Per position P (sorted): the slot is prefilled [0, P) through the OP chain chunk by chunk (traced replays: every chunk
replays the trace of its own key, TPPrefill.plan -> (bucket, window t)), so the carry-in is the real one; the KDA
state is snapshotted (export_kda_state); the chunk [P, P + rows) runs once through the op chain and once through the
PMK chain (GLM53_PF_MODE=pmk, the same TPPrefill, the snapshot re-imported before it); compared: every probed layer's
output residual, the logits row, the first token and every KDA layer's S / C after the chunk (the P9 state write).
The DSA side needs no restore: P is a multiple of 128, the chunk reads only pools / pages of positions < P (written by
the prefix, untouched by the chunk) and its own rows (each run writes them before reading them). After P the snapshot
is restored and the prefix continues through the op chain from P.

Tokens: the 32,768-token golden corpus tiled to ctx_end (real text at every depth). Windows: GLM53_PF_SPARSE_T (the
served list by default); the 2,048 bucket is the only one the PMK plans (pmk_stack.V1_BUCKET), so the deepest PMK
position of a 1M context is its last FULL chunk, 1,046,528 (the 512-token tail at 1,048,064 takes the 512 bucket = the
op graph on both sides, identical by construction). One live trace at a time; every program (both chains, every
window key, the state export / import) compiled before the first capture; capture_guarded asserts no program compiles
inside a capture; no host writes inside a capture; EagerGuard around every measured replay.
VARIANTS (program 17:39:49Z, the 1M L3 split): `variants` = {name: spec} of PMK-chain variants run at every P on
the same snapshot after the op chunk; spec "allop" = the stack attached with every slot of every layer on the op graph
(the tap reference), {L: {slot: mode}} = the served modes with those slots overridden (e.g. {"3": {"ffn": "op"}} =
L3's program B swapped to the op MoE). `taps` = the layers whose per-slot taps (y0 attention out, h1 FFN input, mb
FFN out, ids / sel the routing) are compared per variant vs "allop" (the routing agreement per token). The default
(one variant "pmk" = the served modes, no taps) is the unit as before; the row's top-level fields are the "pmk" variant.
ARGS: layers ([0..4]), rows (2048), positions ([0, 8192, 32768, 131072, 1046528]), windows (the served list),
ctx_end (1048576), pcc_bar (0.9999), probe_layers (= layers), b (2), mock_prefix (2: the mock's prefix chunks per
segment), variants ({"pmk": {}}), taps ([]), tag, record (True)."""
import gc
import math
import os
import sys
import time
import traceback

import torch

import ttnn

gc.collect()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pmk_harness as H  # noqa: E402
import s16common as S  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import pmk_stack as PS  # noqa: E402

C = S.C
mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
TPN = int(mesh.get_num_devices())
LAYERS = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
N = int(args.get("rows", 2048))
POSITIONS = sorted(int(v) for v in args.get("positions", [0, 8192, 32768, 131072, 1046528]))
WINDOWS = str(args.get("windows", "65536,131072,262144,524288,1048576"))
CTX_END = int(args.get("ctx_end", 1048576))
PCC_BAR = float(args.get("pcc_bar", 0.9999))
PROBE = [int(v) for v in args.get("probe_layers", LAYERS)]
B_BUILD = int(args.get("b", 2))
MOCK_PREFIX = int(args.get("mock_prefix", 2))
VARIANTS = {str(k): v for k, v in (args.get("variants") or {"pmk": {}}).items()}
assert "pmk" in VARIANTS and "op" not in VARIANTS, ("the row's top-level fields are the 'pmk' variant; 'op' is the reference", list(VARIANTS))
TAPS = [int(v) for v in args.get("taps", [])]
assert not TAPS or "allop" in VARIANTS, "the taps compare against the 'allop' variant"
tag = args.get("tag", f"pmk-stack-len-L{LAYERS[0]}-{LAYERS[-1]}-tp{TPN}")
rec = bool(args.get("record", True))
assert all(p % 128 == 0 for p in POSITIONS), ("sparse_dsa chunks start at multiples of 128", POSITIONS)
assert POSITIONS[-1] + N <= CTX_END, (POSITIONS[-1], N, CTX_END, "the deepest full chunk ends inside the context")
import pmk45_alias as A45  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

VOCAB = int(hf_config().vocab_size)

os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ",".join(str(v) for v in PROBE)
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["GLM53_PF_SPARSE_T"] = WINDOWS
os.environ["PMK_L1_CHECK"] = "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
t0 = time.time()
info, row = {}, {}
try:
    max_seq = max(int(w) for w in WINDOWS.split(",")) if WINDOWS else CTX_END
    max_seq = max(max_seq, POSITIONS[-1] + N) + 512
    ent = S.build(mesh, ctx, LAYERS, b=B_BUILD, buckets=(N,), max_seq=max_seq)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    pf.probe_layers = set(PROBE)
    info["page_table"] = A45.publish_page_table(backend, kv, io, pf, rows=B_BUILD)
    c = pf.bucket_for(N)
    sp = pf.mode(c) == "sp"
    assert sp and c == N, (sp, c, N)
    # the DSA selection from a populated store is half of the axis: a sparse_dsa layer must run the indexer path
    assert pf.has_dsa() or not any(pl.kind == PS.SPARSE_DSA for pl in pf.layers), "sparse_dsa layer without GLM53_PF_DSA=1"
    g32 = S.golden_prompt(32768).reshape(-1).long()
    corpus = g32.repeat(math.ceil(CTX_END / g32.numel()))[:CTX_END]
    opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")), taps=list(TAPS) or False)
    opts.update({k: v for k, v in args.items() if k.startswith("b_")})
    os.environ["GLM53_PF_MODE"] = "pmk"
    stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts)
    info["wired"] = {f"L{L_}": v for (_, L_), v in stack.wired().items()}
    natural = {L_: dict(stack.progs[(c, L_)].modes) for _, L_, _ in stack.layers}
    for L_ in TAPS:
        assert stack.progs[(c, L_)].taps, (f"L{L_}: no taps built", list(stack.progs[(c, L_)].taps))

    def variant_modes(v):
        spec = VARIANTS[v]
        if spec == "allop":
            return {L_: {s_: "op" for s_ in PS.SLOTS} for L_ in natural}
        out = {L_: dict(m_) for L_, m_ in natural.items()}
        for L_, ov in (spec or {}).items():
            assert int(L_) in out and set(ov) <= set(PS.SLOTS), (v, L_, ov)
            out[int(L_)].update(ov)
        return out

    info["variant_modes"] = {v: {f"L{L_}": m_ for L_, m_ in variant_modes(v).items() if m_ != natural[L_]} for v in VARIANTS}
    guard = H.EagerGuard(mesh, pf, backend, stack)
    keys = [(c, 0)] + pf.fresh_keys((c,)) + pf.sparse_keys((c,))
    info["keys"] = [list(k) for k in keys]
    H.log(f"{tag}: max_seq {max_seq}, positions {POSITIONS}, windows {WINDOWS}, keys {keys}; chain {info['wired']}")

    def pc():
        try:
            return int(mesh.num_program_cache_entries())
        except Exception:  # noqa: BLE001
            return -1

    def set_chain(v: str):
        """"op" = the stack detached (the op chain); else a PMK-chain variant: the stack attached, its slot modes set"""
        assert v == "op" or v in VARIANTS, ("set_chain takes 'op' or a variant name", v)
        if v != "op":
            for L_, m_ in variant_modes(v).items():
                stack.progs[(c, L_)].modes = dict(m_)
            os.environ["GLM53_PF_MODE"] = "pmk"
            stack.attach()
        else:
            stack.detach()
            assert pf.pmk_layers == {}
            os.environ["GLM53_PF_MODE"] = "op"

    # ---- every program before the first capture: both chains at every key (TPPrefill.compile), the state copies
    t1 = time.time()
    for v in ["op"] + list(VARIANTS):
        set_chain(v)
        pf.compile((c,))
    snap0 = backend.export_kda_state(kv, 0)
    backend.import_kda_state(kv, 0, snap0)
    ttnn.synchronize_device(mesh)
    info["compile_s"] = round(time.time() - t1, 1)
    progs0 = pc()
    H.log(f"{tag}: both chains compiled at {len(keys)} keys in {info['compile_s']} s; program cache {progs0}")

    live = dict(key=None, chain=None, tid=None)

    def release():
        if live["tid"] is not None:
            ttnn.release_trace(mesh, live["tid"])
            pf.traces.pop(live["key"], None)
        live.update(key=None, chain=None, tid=None)

    def run_chunk(cs: int, nv: int, chain: str):
        """one chunk [cs, cs + nv) of slot 0 through the op / PMK chain: its key's trace (captured on a key / chain
        change; one live trace at a time), the host inputs written OUTSIDE the capture, then one replay"""
        key = pf.plan(cs, nv)
        assert key[0] == c, (key, c, "the PMK plans the 2,048 bucket only")
        if C.MOCK:
            set_chain(chain)
            pf.write_inputs(c, corpus[cs:cs + nv], slot=0, chunk_start=cs, n_valid=nv)
            pf.forward(c, t=key[1])
            ttnn.synchronize_device(mesh)
            return key
        if live["key"] != key or live["chain"] != chain:
            release()
            set_chain(chain)
            n0 = pc()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                pf.forward(c, t=key[1])
            finally:
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
            assert pc() == n0, f"program cache grew inside the capture of {key} chain={chain}: {n0} -> {pc()}"
            pf.traces[key] = tid
            live.update(key=key, chain=chain, tid=tid)
        pf.write_inputs(c, corpus[cs:cs + nv], slot=0, chunk_start=cs, n_valid=nv)
        ttnn.execute_trace(mesh, live["tid"], cq_id=0, blocking=True)
        return key

    def outputs(taps=False):
        probes = {L_: pf.probe(c, L_).float().clone() for L_ in PROBE}
        tok = int(backend.read(io.tokens)[0]) if not C.MOCK else -1
        lg = backend.read(io.logits)[0:1].float().clone() if not C.MOCK else torch.zeros(1, 8)
        st = backend.export_kda_state(kv, 0)
        tp_ = {L_: {k_: H.rows_host(t_, sp if k_ in ("y0", "mb", "h1") else False).float().clone()
                    for k_, t_ in stack.progs[(c, L_)].taps.items() if k_ != "ffn_pre"} for L_ in TAPS} if taps else {}  # fmt: skip
        return probes, tok, lg, {int(d["L"]): (d["S"].float(), d["C"].float()) for d in st}, tp_

    def routing(t_, r_):
        """the routing agreement per token, variant t_ vs reference r_ (taps ids / sel, [C, top-k]): the top-k SETS
        per token (order-free), the slots that differ, the routed weights on the tokens with equal sets, and the
        smallest selected score of the flipped tokens (near the k-th / (k+1)-th boundary) vs every token's"""
        ia, ib = t_["ids"][:N].round().long(), r_["ids"][:N].round().long()
        sa, sb = t_["sel"][:N], r_["sel"][:N]
        oa, ob = ia.sort(-1), ib.sort(-1)
        wa, wb = sa.gather(-1, oa.indices), sb.gather(-1, ob.indices)
        same = (oa.values == ob.values).all(-1)
        k = int(ia.shape[-1])
        hit = sum((ia[:, j:j + 1] == ib).any(-1).long() for j in range(k))
        flipped = (~same).nonzero().reshape(-1)
        kth_ref = sb.min(-1).values
        return dict(tokens=N, top_k=k, tokens_same_set=int(same.sum()), tokens_flipped=int((~same).sum()),
                    slots_differ=int((k - hit).sum()), max_slots_differ_per_token=int((k - hit).max()) if N else 0,
                    order_equal=int((ia == ib).all(-1).sum()),
                    weights_same_set=H.cmp(wa[same], wb[same]) if bool(same.any()) else None,
                    kth_score_flipped=[round(float(v), 6) for v in kth_ref[flipped][:16]],
                    kth_score_median_all=round(float(kth_ref.median()), 6),
                    flipped_tokens=[int(v) for v in flipped[:32]])  # fmt: skip

    rows_out, cur = [], 0
    for P in POSITIONS:
        # the prefix [cur, P) through the op chain (the real carry-in)
        tp0 = time.time()
        starts = list(range(cur, P, N))
        if C.MOCK and len(starts) > MOCK_PREFIX:
            starts = starts[:1] + starts[-(MOCK_PREFIX - 1):] if MOCK_PREFIX > 1 else starts[:1]
        used = []
        for cs in starts:
            k_ = run_chunk(cs, min(N, P - cs), "op")
            if k_ not in used:
                used.append(k_)
        ttnn.synchronize_device(mesh)
        prefix_s = round(time.time() - tp0, 1)
        snap = backend.export_kda_state(kv, 0)
        # the chunk at P: op, then (the snapshot back) the PMK chain
        guard.snapshot()
        k_op = run_chunk(P, N, "op")
        ttnn.synchronize_device(mesh)
        p_op, t_op, l_op, s_op, _ = outputs()
        outs = {}
        for v in VARIANTS:
            backend.import_kda_state(kv, 0, snap)
            k_v = run_chunk(P, N, v)
            ttnn.synchronize_device(mesh)
            outs[v] = (k_v,) + outputs(taps=bool(TAPS))
        g_ = guard.check(f"position {P}") if not C.MOCK else None
        backend.import_kda_state(kv, 0, snap)  # the prefix continues from P through the op chain
        k_pmk, p_pmk, t_pmk, l_pmk, s_pmk, _ = outs["pmk"]
        per_layer = {L_: H.cmp(p_pmk[L_].reshape(N, -1), p_op[L_].reshape(N, -1)) for L_ in PROBE}
        state = {}
        for L_ in s_op:
            state[f"L{L_}.S"] = H.cmp(s_pmk[L_][0], s_op[L_][0])
            state[f"L{L_}.C"] = H.cmp(s_pmk[L_][1], s_op[L_][1])
        var_rows = {}
        for v, (k_v, p_v, t_v, l_v, s_v, tp_v) in outs.items():
            var_rows[v] = dict(key=list(k_v), per_layer_pcc={f"L{L_}": H.cmp(p_v[L_].reshape(N, -1), p_op[L_].reshape(N, -1))["pcc"] for L_ in PROBE},
                               per_layer_bitwise={f"L{L_}": bool(torch.equal(p_v[L_], p_op[L_])) for L_ in PROBE},
                               logits=H.cmp(l_v.reshape(-1), l_op.reshape(-1)), token=t_v, token_equal=t_v == t_op)  # fmt: skip
            if TAPS and v != "allop":
                tr = outs["allop"][5]
                var_rows[v]["taps_vs_allop"] = {f"L{L_}": dict({k_: H.cmp(tp_v[L_][k_].reshape(-1), tr[L_][k_].reshape(-1)) for k_ in ("y0", "h1", "mb")},
                                                               routing=routing(tp_v[L_], tr[L_]) if "ids" in tp_v[L_] else None)
                                                for L_ in TAPS}  # fmt: skip
        # the carry-in |S| max (P = 0 is a fresh chunk: its state_in zeroes the scratch, the snapshot is not read)
        carry = {f"L{int(d['L'])}": round(float(d["S"].float().abs().max()), 6) for d in snap} if P > 0 else "fresh"

        def top2(lg_):  # the first token's margin (fp32): top-1 minus top-2 logit -> near-ties are visible
            v_, i_ = torch.topk(lg_.reshape(-1)[:VOCAB].float(), 2)
            return dict(top1=int(i_[0]), top2=int(i_[1]), margin=round(float(v_[0] - v_[1]), 5))
        lg = H.cmp(l_pmk.reshape(-1), l_op.reshape(-1))
        ok_p = (all(d["pcc"] >= PCC_BAR or d["bitwise"] for d in per_layer.values()) and (lg["pcc"] >= PCC_BAR or lg["bitwise"])
                and t_op == t_pmk and all(d["pcc"] >= PCC_BAR or d["bitwise"] for d in state.values()))  # fmt: skip
        r_ = dict(position=P, key=list(k_pmk), key_op=list(k_op), prefix_chunks=len(starts), prefix_keys=[list(k) for k in used],
                  prefix_s=prefix_s, carry_in_S_absmax=carry, per_layer=per_layer, logits=lg,
                  first_token=dict(op=t_op, pmk=t_pmk, equal=t_op == t_pmk, op_top2=top2(l_op), pmk_top2=top2(l_pmk)),
                  kda_state_after=state, eager_guard=g_, variants=var_rows,
                  status="mock" if C.MOCK else ("pass" if ok_p else "fail"))  # fmt: skip
        H.log(f"{tag}: position {P}: {r_}")
        rows_out.append(r_)
        cur = P
    release()
    ok = all(r_["status"] == "pass" for r_ in rows_out)
    status = "mock" if C.MOCK else ("pass" if ok else "fail")
    row = dict(test=(f"PMK served stack CONTEXT-LENGTH axis: L{LAYERS[0]}-L{LAYERS[-1]} GLM53_PF_MODE=pmk vs the op chain on the same "
                     f"TPPrefill, the chunk at positions {POSITIONS} ({N} rows, tp{TPN}) after the REAL prefix (op chain, traced), "
                     f"same KDA carry-in; bar per-layer residual / logits / KDA S, C after the chunk PCC >= {PCC_BAR}, first token equal"),
               milestone="PMK-B5", status=status, tp=TPN, layers=LAYERS, bucket=c, positions=POSITIONS, windows=WINDOWS,
               max_seq=max_seq, rows=rows_out, wired=info["wired"], keys=info["keys"], compile_s=info["compile_s"],
               variants=VARIANTS, variant_modes=info["variant_modes"], taps=TAPS,
               page_table=info["page_table"], signature=stack.signature()["signature"], mock=C.MOCK,
               wall_s=round(time.time() - t0, 1), tag=tag,
               note="the 512-token tail of a 1M context (1,048,064) takes the 512 bucket = the op graph on both sides (the PMK plans 2,048 only)")  # fmt: skip
    H.log({k: v for k, v in row.items() if k not in ("test", "wired", "rows")})
    if rec and not C.MOCK:
        C.record("tests", row)
        for r_ in rows_out:
            C.record("pcc", dict(test=row["test"], milestone="PMK-B5", tp=TPN, bucket=c, position=r_["position"],
                                 pcc_min_layer=min(d["pcc"] for d in r_["per_layer"].values()), pcc_logits=r_["logits"]["pcc"],
                                 pcc_min_state=min((d["pcc"] for d in r_["kda_state_after"].values()), default=None),
                                 first_token_equal=r_["first_token"]["equal"], status=r_["status"], tag=tag))  # fmt: skip
except Exception:  # noqa: BLE001
    err = traceback.format_exc()[-6000:]
    H.log(f"FAILED:\n{err}")
    row = dict(test="PMK served stack context-length axis", milestone="PMK-B5", status="fail", error=err, tp=TPN, layers=LAYERS,
               mock=C.MOCK, wall_s=round(time.time() - t0, 1), tag=tag)  # fmt: skip
    if rec and not C.MOCK:
        C.record("tests", row)
RESULT = row
