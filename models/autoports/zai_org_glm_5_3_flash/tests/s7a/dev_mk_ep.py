# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 7a expert-parallel baseline for the expert-TP study (megakernel_plan §10e): the MoE FFN half as ONE program
(tt/mk_moe.py MkMoE) on ONE chip with only an EP slice of the routed experts — slice r of GLM53_EP (default 8) = experts
r * E / EP .. (r + 1) * E / EP - 1 (e_lo, num_experts = E = 288): the routed work chip r of a TP8 mesh runs (router and
shared expert in full here; on the mesh the shared expert is a 1/8 shard). Layer GLM53_S7A_LAYER (default 4), T =
GLM53_MK_TS (default 1,8,32), golden inputs (real routing). GLM53_EP_MODE=xtp: the slices are expert-TP slices
instead (mk_experts.py xtp = r: m tiles [8r, 8r + 8) of EVERY expert, read in place from the full tensors; balanced);
GLM53_EP_MODE=xsl: batched expert TP (mk_experts.py xsl = r, MKX_XSL: the same slices, the union in groups of 8).
Per T: the full program (all E experts) and every slice,
traced, N_CALLS calls per replay on the same input (per call = median replay / N_CALLS); the slices' routed partials
summed on the host vs the full program's y (PCC: bf16 partials); ids of every slice == the full program's; per slice
the routed (token, expert) pairs and the distinct experts it serves. Rows -> pcc.jsonl / tests.jsonl / perf.jsonl."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, golden, hf_config, pcc
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.dev_moe_parts import ffn_args
from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced, cached_decoder
from models.autoports.zai_org_glm_5_3_flash.tt.mk_experts import ReaderExperts
from models.autoports.zai_org_glm_5_3_flash.tt.mk_moe import MkMoE
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

MOCK = os.environ.get("GLM53_MOCK") == "1"
LAYER = int(os.environ.get("GLM53_S7A_LAYER", "4"))
OUT = os.environ.get("GLM53_STAGE_EVIDENCE", os.environ.get("GLM53_OUT_DIR", "/tmp"))
RUN = os.environ.get("GLM53_RUN_ID", time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
TS = [int(v) for v in os.environ.get("GLM53_MK_TS", "1,8,32").split(",")]
EP = int(os.environ.get("GLM53_EP", "8"))
MODE = os.environ.get("GLM53_EP_MODE", "ep")  # ep: expert slices; xtp / xsl: expert-TP m slices; placed: a placement
# GLM53_EP_MODE=placed: chip r's experts = the load-aware placement (scripts/ep_placement.py JSON, GLM53_EP_PLACEMENT)
PLACEMENT = os.environ.get("GLM53_EP_PLACEMENT", "")
# GLM53_EP_BATCH_OFFSET: the first golden row of the batch (rows OFF, OFF + 37, ...; default 301) -> other batches for
# a batch-averaged slice balance (EP replicas, 2026-10-20)
OFF = int(os.environ.get("GLM53_EP_BATCH_OFFSET", "301"))
N_CALLS, REPLAYS = 4, 20
ROWS = []


def row(name, **kw):
    r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), run=RUN, layer=LAYER, test=name,
             watcher=bool(os.environ.get("TT_METAL_WATCHER")), **kw)  # fmt: skip
    ROWS.append(r)
    print("ROW", json.dumps(r), flush=True)


def flush():
    for name, keys in (
        ("pcc.jsonl", None),
        ("tests.jsonl", ("time_utc", "run", "test", "status", "t")),
        ("perf.jsonl", ("time_utc", "run", "test", "t", "execution", "timing_us", "watcher", "ep", "mode")),
    ):
        with open(os.path.join(OUT, name), "a") as f:
            for r in ROWS:
                if name == "perf.jsonl" and not r.get("timing_us"):
                    continue
                f.write(json.dumps(r if keys is None else {k: r[k] for k in keys if k in r}) + "\n")


def assign_replicas(touched, home):
    """xw.cpp MKX_REP mirrored: single-copy experts count on their home, then every touched replicated expert in ascending
    id goes to the lighter of its two homes (ties: the primary) -> {expert: chip}."""
    cost, owner = [0] * 8, {}
    for x in touched:
        if home[x][1] < 0:
            cost[home[x][0]] += 1
            owner[x] = home[x][0]
    for x in touched:
        p_, s_ = home[x]
        if s_ >= 0:
            c = p_ if cost[p_] <= cost[s_] else s_
            cost[c] += 1
            owner[x] = c
    return owner


def host(t):
    return ttnn.to_torch(t).float()


def timed(tr):
    if not tr.enabled:
        return None
    d = []
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        tr.replay()
        d.append((time.perf_counter() - t0) * 1e6)
    return round(sorted(d)[len(d) // 2] / N_CALLS, 1)


def main(mesh=None, ctx=None):
    """mesh / ctx: run inside the long-lived device process (serve.py)."""
    own = mesh is None
    if ctx is not None and ctx.pop("dev_mk_layer_objs", None) is not None:  # its resident objects pin L1: release
        gc.collect()
    cfg = hf_config()
    if own:
        mesh = (ttnn.open_device(device_id=0) if MOCK
                else ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=200_000_000))  # fmt: skip
    tr = Traced(mesh)
    try:
        source = FdTileCacheSource(CACHE_DIR, MANIFEST)
        dec = cached_decoder(cfg, LAYER, mesh, ctx)
        mlp = dec.layer.mlp
        f_args = ffn_args(source, cfg, LAYER)
        n_e = mlp.num_experts
        per = n_e // EP
        assert per * EP == n_e == len(mlp.gate_up), (n_e, EP, len(mlp.gate_up))
        g = golden(LAYER, 2048)["streams_in"].float()
        s1s = {
            t: ttnn.from_torch(
                g[0, OFF : OFF + 37 * t : 37].reshape((1, 1, t, 16384) if t > 4 else (t, 1, 4, 4096)),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )  # fmt: skip
            for t in TS
        }
        plc = json.load(open(PLACEMENT)) if MODE == "placed" else {}
        sets = plc["layers"][str(LAYER)] if MODE == "placed" else None
        # EP replicas (ep_placement.py --replicas): every chip's slice gets the homes + its index (xw.cpp MKX_REP)
        home = plc.get("home", {}).get(str(LAYER))
        res = {t: {} for t in TS}  # t -> {slice (None = full): (us per call, y, ids)}
        inplace = {}  # xsl_sliced: (t, slice) -> (in-place us per call, y bitwise, max abs diff)
        for r in [None] + list(range(EP)):
            if (
                r is None
            ):  # the reference: its own ring depth (GLM53_MKX_RING sizes the slices' 8-tile-K expert TP ring)
                mk = MkMoE(mesh, f_args, mlp, expert_ring_slots=int(os.environ.get("GLM53_EP_FULL_RING", "4")))
            elif MODE == "placed":  # this chip's experts from the placement, in slot order (MKX_LOC)
                ex = sets[r]
                part = SimpleNamespace(router=mlp.router, shared=mlp.shared, gate_up=[mlp.gate_up[x] for x in ex],
                                       down=[mlp.down[x] for x in ex], top_k=mlp.top_k, scaling=mlp.scaling,
                                       limit=mlp.limit, intermediate=mlp.intermediate, hidden=mlp.hidden,
                                       num_experts=n_e)  # fmt: skip
                mk = MkMoE(mesh, f_args, part, num_experts=n_e, local_experts=ex, replica_home=home, chip=r)
            elif MODE == "xtp":
                mk = MkMoE(mesh, f_args, mlp, xtp=r)
            elif MODE == "xsl":  # batched expert TP (MKX_XSL): slice r of every selected expert, groups of 8
                mk = MkMoE(mesh, f_args, mlp, xsl=r, xsl_tp=EP)
            elif MODE == "xsl_sliced":  # the production layout: slice r as its own tensors (union experts only)
                union = sorted(
                    {
                        int(v)
                        for t_ in TS
                        for v in res[t_][None][2][:, : mlp.top_k].reshape(-1).tolist()
                        if 0 <= int(v) < n_e
                    }
                ) or [
                    0
                ]  # (a mock reads back garbage ids)
                gs, ds = ReaderExperts.xsl_slices(mlp.gate_up, mlp.down, EP, r, mesh, experts=union)
                part = SimpleNamespace(router=mlp.router, shared=mlp.shared, gate_up=gs, down=ds, top_k=mlp.top_k,
                                       scaling=mlp.scaling, limit=mlp.limit, intermediate=mlp.intermediate,
                                       hidden=mlp.hidden, num_experts=n_e)  # fmt: skip
                mk = MkMoE(mesh, f_args, part, xsl=r, xsl_tp=EP, xsl_sliced=True)
                mk_ip = MkMoE(mesh, f_args, mlp, xsl=r, xsl_tp=EP)  # the in-place slice: must match bitwise
            else:
                sl = slice(r * per, (r + 1) * per)
                part = SimpleNamespace(router=mlp.router, shared=mlp.shared, gate_up=mlp.gate_up[sl],
                                       down=mlp.down[sl], top_k=mlp.top_k, scaling=mlp.scaling, limit=mlp.limit,
                                       intermediate=mlp.intermediate, hidden=mlp.hidden, num_experts=n_e)  # fmt: skip
                mk = MkMoE(mesh, f_args, part, e_lo=r * per, num_experts=n_e)
            for t in TS:
                try:
                    outs = tr.run(lambda mk=mk, t=t: [mk(s1s[t], t=t) for _ in range(N_CALLS)])
                    o = outs[-1]
                    y = host(o["y"]).reshape(-1)
                    ids = host(o["ids"])[..., :t, :].reshape(t, -1)
                    res[t][r] = (timed(tr), y, ids)
                    tr.release()
                    if MODE == "xsl_sliced" and r is not None:
                        o2 = tr.run(lambda mk=mk_ip, t=t: [mk(s1s[t], t=t) for _ in range(N_CALLS)])[-1]
                        y2 = host(o2["y"]).reshape(-1)
                        inplace[(t, r)] = (timed(tr), bool(torch.equal(y, y2)), float((y - y2).abs().max()))
                        tr.release()
                except Exception:
                    traceback.print_exc()
                    tr.release()
                    err = traceback.format_exc()
                    row("mk_ep_slices", t=t, ep=EP, mode=MODE, slice=r, status="fail", error=err[-400:])
                    if "TIMEOUT" in err or "hang" in err:
                        flush()
                        print("DEV_MK_EP_ABORT device timeout", flush=True)
                        os._exit(3)
            del mk
            if MODE == "xsl_sliced" and r is not None:
                del mk_ip, gs, ds, part
            gc.collect()
        for t in TS:
            got = res[t]
            if len(got) != EP + 1:
                continue  # a failed case already has its row
            us_full, y_full, ids_full = got[None]
            e = ids_full[:, : mlp.top_k].long().reshape(-1)
            if MODE == "placed" and home is not None:  # the device's assignment, mirrored (xw.cpp MKX_REP)
                owner = assign_replicas(sorted(set(e.tolist())), home)
                on = [torch.tensor([owner[int(v)] == r for v in e.tolist()]) for r in range(EP)]
            else:
                on = [torch.isin(e, torch.tensor(sets[r])) if MODE == "placed" else (e >= r * per) & (e < (r + 1) * per)
                      for r in range(EP)]  # fmt: skip
            pairs = [int(on[r].sum()) for r in range(EP)]
            distinct = [int(torch.unique(e[on[r]]).numel()) for r in range(EP)]
            y_sum = sum(got[r][1] for r in range(EP))
            if os.environ.get("GLM53_EP_DUMP"):  # diagnostics: the full y, every slice's partial, the ids (host floats)
                torch.save(dict(full=y_full, slices=[got[r][1] for r in range(EP)], ids=ids_full, mode=MODE),
                           f"{os.environ['GLM53_EP_DUMP']}_t{t}.pt")  # fmt: skip
            ids_same = all(torch.equal(got[r][2], ids_full) for r in range(EP))
            us = [got[r][0] for r in range(EP)]
            timing = dict(full=us_full, slices=us)
            if all(v is not None for v in us):
                timing.update(slice_max=max(us), slice_mean=round(sum(us) / EP, 1))
            ok = ids_same and pcc(y_sum, y_full) >= 0.9999
            # expert TP at the default LoFi: the routing-weight multiply (FPU, LoFi) rounds the slices' partials, while
            # the full program rounds whole expert outputs (EP slices round the same values: their errors cancel), so
            # y_sum vs full is ~0.9998 by fidelity alone -> a timing record (still >= 0.999, else fail: a broken
            # program cannot pass as info); GLM53_MKX_FIDELITY=HiFi4 is the gate
            lofi_xtp = MODE in ("xtp", "xsl", "xsl_sliced") and os.environ.get("GLM53_MKX_FIDELITY", "LoFi") == "LoFi"
            if MODE == "xsl_sliced":  # the sliced tensors must reproduce the in-place slices bitwise
                ip = [inplace.get((t, r)) for r in range(EP)]
                timing["inplace_slices"] = [v[0] if v else None for v in ip]
                sliced_bitwise = all(v is not None and v[1] for v in ip)
                ids_same = ids_same and sliced_bitwise
                ok = ids_same and pcc(y_sum, y_full) >= 0.9999
            row("mk_ep_slices", t=t, ep=EP, mode=MODE, experts_per_slice=per, execution=tr.mode, n_calls=N_CALLS,
                timing_us=timing, pairs=pairs, distinct_experts=distinct,
                distinct_full=int(torch.unique(e).numel()), ids_bitwise_all_slices=ids_same,
                y_sum_pcc=pcc(y_sum, y_full), y_sum_maxabs=float((y_sum - y_full).abs().max()),
                sliced_vs_inplace=([v[1:] if v else None for v in ip] if MODE == "xsl_sliced" else None),
                fidelity=os.environ.get("GLM53_MKX_FIDELITY", "LoFi"),
                status="record" if MOCK else ("info" if lofi_xtp and ids_same and pcc(y_sum, y_full) >= 0.999
                                              else "pass" if ok else "fail"))  # fmt: skip
    finally:
        flush()
        if own:
            ttnn.close_device(mesh) if MOCK else ttnn.close_mesh_device(mesh)
    bad = [r for r in ROWS if r.get("status") == "fail"]
    print(f"DEV_MK_EP_DONE rows={len(ROWS)} fail={len(bad)}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
