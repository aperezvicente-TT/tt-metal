# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 14c job (serve.py): kda_rec v2 (GLM53_REC_MODE=v2, kernels/kda_rec/kda_rec_v2_sfpu.hpp) vs v1, device vs
device and vs the fp64 oracle (kda_rec.reference_step), on the standalone KdaRec program (TP8 shard: H heads on the
64-core grid, B users, user groups ug), SAME random inputs (q / k / v / g / beta / gate_out rows, a state snapshot).
Compile launch, then ONE traced call (state restored first) per mode for the numerics, traced replays for the timing.
Rows: pcc.jsonl (v2 vs v1 / oracle: on, state) + perf.jsonl (traced us per mode) when `record`.
ARGS: heads (8), b (32), ug (8), seed (0), reps (20), modes (["v1", "v2"]), record (False), tag; cases [{overrides}]."""
import gc
import importlib
import importlib.util
import os
from types import SimpleNamespace

import torch

import ttnn

gc.collect()


def _load(name, fname):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load("s14c_common", "s14common.py")
import models.autoports.zai_org_glm_5_3_flash.tt.kda_rec as KREC  # noqa: E402

importlib.reload(KREC)
mesh = MESH  # noqa: F821


def case(args):
    H, B, UG = int(args.get("heads", 8)), int(args.get("b", 32)), int(args.get("ug", 8))
    REPS = int(args.get("reps", 20))
    g = torch.Generator().manual_seed(int(args.get("seed", 0)))
    rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
    DK = 128
    rep = dict(device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), layout=ttnn.TILE_LAYOUT)
    dev = lambda t, dt=ttnn.float32, mem=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(t, dtype=dt, memory_config=mem, **rep)  # noqa: E731
    o_norm_h = 1.0 + 0.1 * rnd(1, 1, 1, DK)
    o_norm = dev(o_norm_h, ttnn.bfloat16)
    host = dict(q=rnd(1, 1, B, H * DK), k=rnd(1, 1, B, H * DK), v=rnd(1, 1, B, H * DK),
                g=-torch.rand(1, 1, B, H * DK, generator=g) * 0.2, beta=torch.rand(1, 1, B, H, generator=g),
                go=rnd(1, 1, B, H * DK), S=rnd(B, H, DK, DK) * 0.05)  # fmt: skip
    ins = {k: dev(v) for k, v in host.items() if k != "S"}
    S0 = dev(host["S"])
    # the oracle (fp64): the device's bf16 o_norm values
    ref = KREC.reference_step(host["q"].reshape(B, H, DK), host["k"].reshape(B, H, DK), host["v"].reshape(B, H, DK),
                              host["g"].reshape(B, H, DK), host["beta"].reshape(B, H), host["go"].reshape(B, H, DK),
                              host["S"], o_norm_h.to(torch.bfloat16).float().reshape(DK), 1e-6)  # fmt: skip


    def run(mode):
        os.environ.pop("GLM53_REC_MODE", None)
        os.environ.pop("GLM53_REC_PROBE", None)
        if mode in ("v2", "v2probe"):
            os.environ["GLM53_REC_MODE"] = "v2"
        if mode == "v2probe":
            os.environ["GLM53_REC_PROBE"] = "1"
        kda = SimpleNamespace(device=mesh, h=H, d=DK, eps=1e-6, o_norm=o_norm, kda_rec_spread=1, kda_rec_ugroups=UG)
        rec = KREC.KdaRec(kda)
        os.environ.pop("GLM53_REC_MODE", None)
        os.environ.pop("GLM53_REC_PROBE", None)
        on_buf = rec.output(B)  # user groups zero it with a host write: once, outside the capture
        rec.output = lambda rows, memory_config=None: on_buf
        S = dev(host["S"])

        def step():
            ttnn.copy(S0, S)
            return rec(ins["q"], ins["k"], ins["v"], ins["g"], ins["beta"], ins["go"], S, rows=B)

        tr = C.Traced(ttnn, mesh, step)
        tr.replay()
        on = ttnn.to_torch(ttnn.get_device_tensors(tr.out)[0]).float()[..., :B, :].reshape(B, H, DK)
        st = ttnn.to_torch(ttnn.get_device_tensors(S)[0]).float()
        walls = tr.replays(REPS)
        tr.release()
        return on, st, C.stats(walls)


    out = {}
    for mode in args.get("modes", ["v1", "v2"]):
        on, st, tm = run(mode)
        out[mode] = dict(on=on, st=st, us=tm["median_us"])
        print(f"REC {mode} B={B} H={H} ug={UG}: {tm['median_us']} us", flush=True)

    res = dict(heads=H, b=B, ug=UG, us={m: v["us"] for m, v in out.items() if m != "v2probe"})
    if "v2probe" in out:  # the cross-lane primitives on the raw v planes of item (b 0, h 0) vs the host emulator's
        import numpy as np

        E = _load("s14c_v2emu", "v2_emulator.py")
        vec = np.zeros(1024, np.float32)
        vrow = host["v"].reshape(B, H, DK)[0, 0].numpy().astype(np.float32)
        for j in range(DK):
            vec[E.plane_off(12 + (j >> 5), 8 * (2 * ((j >> 4) & 1) + (j & 1)) + ((j >> 1) & 7))] = vrow[j]
        vp = [vec[E.PLANE_IDX[12 + m]] for m in range(4)]
        mask3 = np.zeros(32, np.float32)
        mask3[[8 * i + 3 for i in range(4)]] = 1.0
        tr4 = E.transp(*vp)
        exp = [E.ror(vp[0]), E.rowsum8(vp[0]), tr4[0], tr4[1], tr4[2], tr4[3], mask3, E.rowsum8(vp[0] * mask3)]
        got_tile = E.to_tile(out["v2probe"]["st"][0, 0, :32, :32].numpy().astype(np.float32))
        got = [got_tile[E.PLANE_IDX[p]] for p in range(8)]
        names = ["ror1", "rowsum8", "transp0", "transp1", "transp2", "transp3", "mask3", "spread3"]
        res["probe"] = {n: dict(ok=bool(np.allclose(g, e, rtol=1e-6, atol=1e-6)), max_abs=float(np.abs(g - e).max()))
                        for n, g, e in zip(names, got, exp)}  # fmt: skip
        res["probe_ok"] = all(v_["ok"] for v_ in res["probe"].values())
        if not res["probe_ok"]:
            res["probe_raw"] = {n: dict(got=g.tolist(), exp=e.tolist()) for n, g, e in zip(names, got, exp)}
        print("REC_V2_PROBE", res["probe"], flush=True)
        out.pop("v2probe")
    for m, v in out.items():
        res[m] = dict(on_pcc_oracle=C.pcc(v["on"], ref["on"].float()), state_pcc_oracle=C.pcc(v["st"], ref["state"].float()),
                      on_max_abs_oracle=float((v["on"] - ref["on"].float()).abs().max()),
                      state_max_abs_oracle=float((v["st"] - ref["state"].float()).abs().max()))  # fmt: skip
    if "v1" in out and "v2" in out:
        a, b_ = out["v1"], out["v2"]
        res["v2_vs_v1"] = dict(on_pcc=C.pcc(b_["on"], a["on"]), state_pcc=C.pcc(b_["st"], a["st"]),
                               on_max_abs=float((b_["on"] - a["on"]).abs().max()),
                               state_max_abs=float((b_["st"] - a["st"]).abs().max()),
                               on_bitwise=bool(torch.equal(b_["on"], a["on"])), state_bitwise=bool(torch.equal(b_["st"], a["st"])))  # fmt: skip
    print("REC_V2", res, flush=True)
    if args.get("record"):
        if "v2_vs_v1" in res:
            ok = res["v2_vs_v1"]["on_pcc"] >= 0.9999 and res["v2_vs_v1"]["state_pcc"] >= 0.9999
            test = f"14c kda_rec v2 vs v1 standalone B={B} H={H} ug={UG}"
        else:  # a probe-only case (modes = [v2probe]): the cross-lane primitives vs the host emulator
            ok = bool(res.get("probe_ok"))
            test = f"14c kda_rec v2 cross-lane probe standalone B={B} H={H} ug={UG}"
        C.record("pcc", dict(test=test, tag=args.get("tag", "14c-rec-v2"),
                             status="pass" if ok else "fail", **{k: v for k, v in res.items() if k != "us"}))  # fmt: skip
        for m, us in (res["us"].items() if not os.environ.get("TT_METAL_WATCHER") else ()):  # watched: no perf rows
            C.record("perf", dict(test=f"14c kda_rec {m} standalone B={B} H={H} ug={UG}", tag=args.get("tag", "14c-rec-v2"),
                                  status="info", us=us, us_metric="traced_single_launch_median", traced=True))  # fmt: skip
    RESULT = res

    return res


BASE = {k: v for k, v in ARGS.items() if k != "cases"}  # noqa: F821
RESULT = [case({**BASE, **c}) for c in (ARGS.get("cases") or [{}])]  # noqa: F821
