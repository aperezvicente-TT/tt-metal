# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the host-p300 TP2 pair or the host-8chip 2x4 mesh; watched once then unwatched; ~15 min): THE
Stage 16 unit of the SERVED PMK STACK -- the L0-L4 stack (the three kinds) under GLM53_PF_MODE=pmk vs the op path on
the SAME TPPrefill, same golden prompt rows, both TRACED (one live trace at a time, every program compiled before the
first capture, no host writes inside a capture, EagerGuard around every replay):
  1. reference: the op-graph bucket forward (GLM53_PF_MODE=op: the chain detached) captured + replayed -> per-layer
     output residuals (GLM53_PF_PROBE_LAYERS: copied inside the trace) + the first token + the logits row;
  2. the chain attached (slots per GLM53_PMK_SLOTS / `slots`): compile forward (the programs' first launches),
     capture, replay -> the same probes; PCC per layer (FFN input = the layer output residual), the logits PCC and
     the first token (bitwise expected when the pieces are bitwise), the slot modes / pending per layer, the replay
     us of both (median of `replays`), the per-chain counters (launches / fillers; eager 0).
Criterion: status pass = every probed layer PCC >= pcc_bar (0.9999) AND logits PCC >= pcc_bar AND first token equal
AND eager guard clean; `info` when a layer has every slot pending (nothing on the PMK yet). ARGS: layers ([0..4]),
rows (2048), slots ("auto"), replays (50), pcc_bar (0.9999), env ({}), b_g (2), tag, record (True)."""
import gc
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
REPLAYS = int(args.get("replays", 50))
PCC_BAR = float(args.get("pcc_bar", 0.9999))
tag = args.get("tag", f"pmk-stack-serve-unit-L{LAYERS[0]}-{LAYERS[-1]}-tp{TPN}")
rec = bool(args.get("record", True))
# 14b stack45 (2026-10-23): probe_layers (default = layers) -- the residual probes copied inside the trace (16.8 MB per
# chip per layer at TP8: a 45-layer row may probe a subset); b (default 2 with page_table): the decode batch the build
# allocates (the replicated pool's non-identity page table needs a second slot's blocks); alias (default False): the
# per-kind weight alias of pmk45_alias (a same-weights A/B of the chain vs the op graph without a 45-layer load)
PROBE = [int(v) for v in args.get("probe_layers", LAYERS)]
PAGE_TABLE = bool(args.get("page_table", True))
B_BUILD = int(args.get("b", 2 if PAGE_TABLE else 1))
import pmk45_alias as A45  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config as _hfc  # noqa: E402

ALIAS = A45.install(_hfc()) if bool(args.get("alias", False)) else dict(on=False)
# share (the pair: 45 layers of distinct experts do not fit a TP2 chip): the layers of a kind SHARE the device weights
ALIAS.update(A45.install_device_share(_hfc(), ctx) if bool(args.get("share", False)) else dict(device_share=False))
os.environ["GLM53_PF_MHC"] = "comp"
os.environ["GLM53_PF_MHC_POST"] = args.get("mhc_post", "fused")
os.environ["GLM53_PF_MHC_PRE"] = args.get("mhc_pre", "fused7a")
os.environ["GLM53_PF_PACKED"] = ""
os.environ["GLM53_PF_PROBE_LAYERS"] = ",".join(str(v) for v in PROBE)
os.environ["GLM53_PMK_SLOTS"] = str(args.get("slots", "auto"))
os.environ["PMK_L1_CHECK"] = "1"
S.set_attn(args)
args = {"mla_form": "absorbed", "kda_scan": "op", "mm_nq": 4, "mm_kb": 640, "moe_v2": True, **args}
t0 = time.time()
err = None
info = {}
row = {}
try:
    ent = S.build(mesh, ctx, LAYERS, b=B_BUILD, buckets=(N,), max_seq=N + 512)
    pf, backend, kv, io = ent["pf"], ent["backend"], ent["kv"], ent["io"]
    pf.moe_impl = args.get("moe", "ep")
    S.set_moe_v2(pf, bool(args.get("moe_v2", True)))
    pf.kda_scan = args.get("kda_scan", "op")
    backend.tpp = pf
    # a standing server's ctx may hand back a TPPrefill built before this job's GLM53_PF_PROBE_LAYERS: the probes are
    # a TPPrefill attribute, allocated lazily at the next forward of each bucket -> set them here, explicitly
    pf.probe_layers = set(PROBE)
    # the owner-aware physical page table BEFORE any prefill, page_row asserted real (program 14:07Z harness rule)
    info["page_table"] = A45.publish_page_table(backend, kv, io, pf, rows=B_BUILD) if PAGE_TABLE else None
    info["alias"] = ALIAS
    c = pf.bucket_for(N)
    sp = pf.mode(c) == "sp"
    assert sp and c == N, (sp, c, N)
    prompt = S.golden_prompt(N).reshape(-1).long()
    opts = dict(pfs_park=int(args.get("pfs_park", 4)), router_in=str(args.get("router_in", "fp32")))
    opts.update({k: v for k, v in args.items() if k.startswith("b_")})  # program B: the record (b_config) + these overrides
    opts["taps"] = bool(args.get("taps", True))  # per-slot taps: the attention / FFN outputs per layer (slot PCC vs the all-op chain)
    os.environ["GLM53_PF_MODE"] = "pmk"
    t1 = time.time()
    stack = PS.PmkStack(pf, len(pf.layers), impl="served", slots=PS.slot_modes_from_env(), opts=opts)
    info["chain_build_s"] = round(time.time() - t1, 1)
    info["wired"] = {f"L{L_}": v for (_, L_), v in stack.wired().items()}
    H.log(f"{tag}: chain built in {info['chain_build_s']} s: {info['wired']}")
    guard = H.EagerGuard(mesh, pf, backend, stack)
    pc_rows = []

    def pc():
        try:
            return int(mesh.num_program_cache_entries())
        except Exception:  # noqa: BLE001
            return -1

    def capture_guarded(fn, label):
        """ttnn.begin/end_trace_capture around fn() with the program-cache count asserted UNCHANGED (d0 01:24Z: a
        program whose first run lands inside a capture cannot load its binaries; every program of a captured forward
        must have run before, with the same shapes / args)"""
        n0 = pc()
        tid_ = ttnn.begin_trace_capture(mesh, cq_id=0)
        try:
            fn()
        finally:
            ttnn.end_trace_capture(mesh, tid_, cq_id=0)
        n1 = pc()
        pc_rows.append(dict(label=label, before=n0, after=n1))
        assert n1 == n0 or n0 < 0, f"program cache grew during the capture of {label}: {n0} -> {n1} (a program first run inside the capture)"
        return tid_

    def forward_traced(label):
        """compile launch (eager, once: the programs' first launches), then ONE capture, then `replays` traced
        replays; returns (probes per layer, first token, logits row, replay stats)"""
        pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=N)
        pf.quiesce()  # the pmk pass follows the op trace's replays: zero the completion accounting first (TPPrefill.quiesce)
        pf.forward(c)  # the compile launch
        ttnn.synchronize_device(mesh)
        if C.MOCK:
            stats = dict(median_us=None, replays=0)
        else:
            tm0 = A45.trace_mib(mesh)
            tid = capture_guarded(lambda: pf.forward(c), f"chain forward {label}")
            pf.traces[(c, 0)] = tid
            info.setdefault("trace_mib", {})[label] = dict(before=tm0, after=A45.trace_mib(mesh))
            guard.snapshot()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            stats = S.replay_stats(mesh, tid, replays=REPLAYS, label=f"{tag} {label}")
            guard.check(label)
        probes = {L_: pf.probe(c, L_) for L_ in PROBE}
        tok = int(backend.read(io.tokens)[0]) if not C.MOCK else -1
        logits = backend.read(io.logits)[0:1].float().clone() if not C.MOCK else torch.zeros(1, 8)
        if not C.MOCK:
            pf.release()  # one live trace at a time: the next capture gets the region
        return probes, tok, logits, stats

    # 1. the reference: the op graph on the same TPPrefill (nothing installed)
    stack.detach()
    assert pf.pmk_layers == {}
    os.environ["GLM53_PF_MODE"] = "op"
    p_ref, tok_ref, lg_ref, st_ref = forward_traced("op")
    # 2. the chain
    os.environ["GLM53_PF_MODE"] = "pmk"
    stack.attach()
    p_pmk, tok_pmk, lg_pmk, st_pmk = forward_traced("pmk")
    per_layer = {}
    for L_ in PROBE:
        a, b = p_ref[L_].reshape(N, -1), p_pmk[L_].reshape(N, -1)
        d = H.cmp(a, b)
        d["modes"] = stack.progs[(c, L_)].modes
        d["pending"] = stack.progs[(c, L_)].pending
        per_layer[L_] = d
    lg = H.cmp(lg_ref.reshape(-1), lg_pmk.reshape(-1))
    chain = {L_: dict(stack.progs[(c, L_)].counters) for L_ in LAYERS}
    # per-slot numerics: the attention output (y0) and the FFN output (mb) of every layer, tapped inside the trace,
    # the chain (its modes) vs the SAME chain with every slot on the op graph (modes flipped, one more capture +
    # replay) -> which slot carries the logits remainder (d0 card0 r12: residuals >= 0.99992, logits 0.999848)
    def routing_agreement(a, b):
        """top-8 ids / selected scores per token (gathered [C, 8] fp32): tokens whose expert SET differs, the
        weight difference on the agreeing tokens"""
        ia, ib = a["ids"].round().long(), b["ids"].round().long()
        sa, sb = [set(r.tolist()) for r in ia], [set(r.tolist()) for r in ib]
        flips = [i for i, (x, y) in enumerate(zip(sa, sb)) if x != y]
        same = [i for i in range(len(sa)) if i not in set(flips)]
        wd = float((a["sel"][same] - b["sel"][same]).abs().max()) if same else None
        return dict(tokens=len(sa), set_flips=len(flips), flip_frac=round(len(flips) / max(1, len(sa)), 5), weights_max_abs_on_agreeing=wd,
                    ids_bitwise=bool(torch.equal(ia, ib)), first_flips=flips[:8])  # fmt: skip

    def assemble_tap_swap(t_sw, t_op, pre_sw_chips, pre_op_chips, mb_natural, abl_mb, t_nat):
        """the tap-swap row from HOST tensors only (the device work is done; a failure here must never cost a
        card again -- d0 card0 r17: a keyword collision in this assembly lost two runs): t_sw / t_op / t_nat =
        {y0, mb, h1, ids, sel} of the swapped pmk chain / the all-op chain / the natural pmk chain, pre_* = the
        pre-RS FFN output per chip, abl_mb = {name: mb without that term}"""
        r_ = dict(input_check=H.cmp(t_sw["h1"].reshape(-1), t_op["h1"].reshape(-1)),  # the swapped input IS the op's (bitwise expected)
                  mb_B_vs_op_ffn_same_input=H.cmp(t_sw["mb"].reshape(-1), t_op["mb"].reshape(-1)),
                  routing_same_input=routing_agreement(t_sw, t_op), routing_natural_chain=routing_agreement(t_nat, t_op),
                  mb_natural_chain=mb_natural)  # fmt: skip
        if pre_op_chips is not None:  # H.cmp carries norm_ratio already (r17's collision): nothing added to it
            r_["pre_rs_per_chip"] = [dict(chip=k_, **H.cmp(a_.reshape(-1), b_.reshape(-1))) for k_, (a_, b_) in enumerate(zip(pre_sw_chips, pre_op_chips))]
        tr_ = t_sw["mb"].norm(dim=-1) / t_op["mb"].norm(dim=-1).clamp_min(1e-30)
        r_["mb_token_norm_ratio"] = dict(mean=float(tr_.mean()), p05=float(tr_.quantile(0.05)), p50=float(tr_.median()), p95=float(tr_.quantile(0.95)),
                                         min=float(tr_.min()), max=float(tr_.max()))  # fmt: skip
        mb_full = t_sw["mb"]
        abl = {}
        for name_, mb_a in abl_mb.items():
            d_ = mb_full - mb_a
            w_share = float(t_sw["sel"][:, int(name_[4:])].mean() / t_sw["sel"].sum(-1).mean()) if name_.startswith("slot") else None
            abl[name_] = dict(delta_rel=float(d_.norm() / mb_full.norm().clamp_min(1e-30)),
                              delta_token_mean=float((d_.norm(dim=-1) / mb_full.norm(dim=-1).clamp_min(1e-30)).mean()), weight_share=w_share)  # fmt: skip
        r_["ablation"] = abl
        same_ids = r_["routing_same_input"]["set_flips"] == 0
        pcc_same = r_["mb_B_vs_op_ffn_same_input"]["pcc"]
        r_["verdict"] = ("combine: the ids agree on the same input and program B's output still misses the op ffn by "
                         f"PCC {pcc_same} -> program B's path in the chain (see pre_rs_per_chip / ablation: the bfp8 rounding is excluded by the host emulation)"
                         if same_ids and pcc_same < PCC_BAR else
                         ("input: on the same input program B matches the op ffn (PCC " + str(pcc_same) + "); the natural chain's "
                          f"{r_['routing_natural_chain']['set_flips']} re-routed tokens come from the pmk attention path's input difference "
                          "(inherent to the pmk attention path until its PCC rises)" if same_ids else
                          f"routing differs even on the same input ({r_['routing_same_input']['set_flips']} tokens): the router call is "
                          "not deterministic across the two chains -- investigate before attributing"))
        return r_

    # the tap-swap (program 02:5xZ): L4's program B on the OP chain's EXACT L4 input (the op chain's tapped fp32
    # x-hat_f staged on the device OUTSIDE the trace), inside one more guarded capture of the pmk chain -> mb vs the
    # op ffn on that input, and the routing agreement -- splits "input sensitivity" from "B numerics in the chain"
    slot_pcc, swap = {}, {}
    if opts.get("taps") and not C.MOCK:
        taps_pmk = {L_: {k_: H.rows_host(t_, sp if k_ in ("y0", "mb", "h1") else False).float().clone() for k_, t_ in stack.progs[(c, L_)].taps.items() if k_ != "ffn_pre"} for L_ in LAYERS}
        saved_modes = {L_: dict(stack.progs[(c, L_)].modes) for L_ in LAYERS}
        for L_ in LAYERS:
            stack.progs[(c, L_)].modes = {s_: "op" for s_ in PS.SLOTS}
        try:
            pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=N)
            pf.forward(c)
            ttnn.synchronize_device(mesh)
            tid_o = capture_guarded(lambda: pf.forward(c), "chain forward all-op (taps)")
            ttnn.execute_trace(mesh, tid_o, cq_id=0, blocking=True)
            ttnn.release_trace(mesh, tid_o)
            for L_ in LAYERS:
                pr = stack.progs[(c, L_)]
                slot_pcc[L_] = {k_: dict(H.cmp(taps_pmk[L_][k_].reshape(-1), H.rows_host(t_, sp if k_ in ("y0", "mb", "h1") else False).float().reshape(-1)),
                                         mode=saved_modes[L_]["attn" if k_ == "y0" else "ffn"]) for k_, t_ in pr.taps.items() if k_ in ("y0", "mb", "h1")}
            taps_op = {L_: {k_: H.rows_host(t_, sp if k_ in ("y0", "mb", "h1") else False).float().clone() for k_, t_ in stack.progs[(c, L_)].taps.items() if k_ != "ffn_pre"} for L_ in LAYERS}
            for L_ in LAYERS:
                t_ = stack.progs[(c, L_)].taps.get("ffn_pre")
                if t_ is not None:
                    parts_ = ttnn.get_device_tensors(t_) if TPN > 1 else [t_]
                    taps_op[L_]["ffn_pre_chips"] = [ttnn.to_torch(x_).float().reshape(-1, int(t_.shape[-1])).clone() for x_ in parts_]
        finally:
            for L_ in LAYERS:
                stack.progs[(c, L_)].modes = saved_modes[L_]
        H.log(f"{tag}: per-slot PCC vs the all-op chain {slot_pcc}")

        moe_layers = [L_ for L_ in LAYERS if stack.progs[(c, L_)].b is not None]
        for L_ in moe_layers:
            pr = stack.progs[(c, L_)]
            if "ids" not in taps_pmk[L_]:
                continue
            HID = int(pr.taps["mb"].shape[-1])
            swap_in = S.to_mesh_rows(mesh, taps_op[L_]["h1"], sp, ttnn.float32)  # staged before the capture
            pr.swap_in = {"h1_in": swap_in}
            try:
                pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=N)
                pf.forward(c)
                ttnn.synchronize_device(mesh)
                tid_s = capture_guarded(lambda: pf.forward(c), f"chain forward tap-swap L{L_}")
                ttnn.execute_trace(mesh, tid_s, cq_id=0, blocking=True)
                ttnn.release_trace(mesh, tid_s)
                t_sw = {k_: H.rows_host(t_, sp if k_ in ("y0", "mb", "h1") else False).float().clone() for k_, t_ in pr.taps.items() if k_ != "ffn_pre"}
                parts_ = ttnn.get_device_tensors(pr.taps["ffn_pre"]) if TPN > 1 else [pr.taps["ffn_pre"]]
                pre_sw = [ttnn.to_torch(x_).float().reshape(-1, HID).clone() for x_ in parts_]
            finally:
                pr.swap_in = {}
            # the ablation: program B without expert slot k (the weight column zeroed) / without the shared term, on
            # the same swapped input; 9 more guarded captures at this layer
            topk_ = int(pr.taps["ids"].shape[-1])
            masks = []
            for k_ in range(topk_):
                m_ = torch.ones(c, topk_); m_[:, k_] = 0.0
                masks.append(S.to_mesh_rows(mesh, m_, False, ttnn.float32))
            zeros_ys = ttnn.from_torch(torch.zeros(1, 1, c, HID), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                       memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh) if TPN > 1 else None)  # fmt: skip
            abl_mb = {}
            for name_, abl_ in [(f"slot{k_}", ("slot", masks[k_])) for k_ in range(topk_)] + [("shared", ("shared", zeros_ys))]:
                pr.swap_in = {"h1_in": swap_in}
                pr.b_ablate = abl_
                try:
                    pf.write_inputs(c, prompt, slot=0, chunk_start=0, n_valid=N)
                    pf.forward(c)
                    ttnn.synchronize_device(mesh)
                    tid_a = capture_guarded(lambda: pf.forward(c), f"chain forward ablate L{L_} {name_}")
                    ttnn.execute_trace(mesh, tid_a, cq_id=0, blocking=True)
                    ttnn.release_trace(mesh, tid_a)
                    abl_mb[name_] = H.rows_host(pr.taps["mb"], sp).float().clone()
                finally:
                    pr.b_ablate = None
                    pr.swap_in = {}
            # the raw device results to disk BEFORE any assembly (GLM53_OUT_DIR): a host-side failure keeps the card's work
            try:
                out_dir = os.environ.get("GLM53_OUT_DIR", ".")
                os.makedirs(out_dir, exist_ok=True)
                torch.save(dict(L=L_, t_sw=t_sw, t_op=taps_op[L_], t_nat=taps_pmk[L_], pre_sw=pre_sw, pre_op=taps_op[L_].get("ffn_pre_chips"),
                                abl_mb=abl_mb, slot_pcc=slot_pcc.get(L_)), os.path.join(out_dir, f"{tag}_tapswap_L{L_}.pt"))  # fmt: skip
                H.log(f"{tag}: raw tap-swap tensors saved to {out_dir}/{tag}_tapswap_L{L_}.pt")
            except Exception as e_:  # noqa: BLE001
                H.log(f"{tag}: raw dump failed: {e_!r}")
            swap[L_] = assemble_tap_swap(t_sw, taps_op[L_], pre_sw, taps_op[L_].get("ffn_pre_chips"), slot_pcc[L_]["mb"], abl_mb, taps_pmk[L_])
            swap[L_]["truth_note"] = ("the op reference is NOT the truth at TP2 (r18 dump vs the fp32 host reference on the real experts, "
                                      "logs/ffn_ref_L4_r18.py <dump.pt>: op routed 1.0346 of fp32 (moe_fused_swiglu), program B routed 0.9889); "
                                      "run that script on this row's dump for the truth ratios")
            H.log(f"{tag}: tap-swap L{L_}: {swap[L_]}")
    if opts.get("taps") and C.MOCK:  # the host-side assembly exercised on the mock with synthetic taps of the real shapes
        g_ = torch.Generator().manual_seed(0)
        for L_ in [L_ for L_ in LAYERS if stack.progs[(c, L_)].b is not None]:
            pr = stack.progs[(c, L_)]
            HID, topk_ = int(pr.taps["mb"].shape[-1]), int(pr.taps["ids"].shape[-1])
            mk_ = lambda: dict(y0=torch.randn(N, HID, generator=g_), mb=torch.randn(N, HID, generator=g_), h1=torch.randn(N, HID, generator=g_),  # noqa: E731
                               ids=torch.randint(0, 288, (c, topk_), generator=g_).float(), sel=torch.rand(c, topk_, generator=g_))  # fmt: skip
            t_op_ = mk_(); t_sw_ = {k_: v_.clone() for k_, v_ in t_op_.items()}; t_nat_ = mk_()
            pre_ = [torch.randn(c, HID, generator=g_) for _ in range(TPN)]
            abl_ = {f"slot{k_}": t_sw_["mb"] * 0.9 for k_ in range(topk_)}; abl_["shared"] = t_sw_["mb"] * 0.8
            swap[L_] = dict(selftest=True, **assemble_tap_swap(t_sw_, t_op_, pre_, pre_, dict(pcc=1.0), abl_, t_nat_))
            H.log(f"{tag}: tap-swap assembly SELFTEST on synthetic taps (mock): verdict '{swap[L_]['verdict'][:40]}...'")
    # the eager-fallback verdict belongs to the comparison replays (snapshot before them): taken HERE, before the
    # spans phase, whose compile-style step runs are launches outside a capture by design (d0 card2 r11: the guard's
    # 'final' check saw program_cache +1 = the spans-mode clone of L3's h0, a program the plain chain never runs,
    # compiled before its own guarded capture)
    eager_final = guard.check("final") if not C.MOCK else None
    # per-slot device spans (program 00:3xZ: the chain 122,026 vs op 108,781 us at TP2 with no per-slot view).
    # After the comparison replays the residual holds the last replay's output (a realistic residual: the spans are
    # timings, not numerics) and every chain tensor is freed; per layer, each step runs ONCE compile-style (its outputs
    # become the chain's tensors for the following steps), then ONCE MORE inside its own capture whose outputs are
    # trace-owned and DISCARDED (snapshot / restore: d0 01:38Z segfault = the next step's all_gather on a released
    # trace's buffer), then exactly `span_replays` traced replays (no soak: min_s 0). Every chain tensor a step reads
    # is asserted live (ServedLayer._get), on the device and on the mock alike.
    spans = {}
    spans_pc = dict(before=pc(), after=None, compiled_steps=[])  # the spans phase's own program-cache accounting
    if bool(args.get("spans", True)):
        SPAN_REPLAYS = int(args.get("span_replays", 20))
        ci_ = pf.chunk_inputs(c)
        for L_ in LAYERS:
            prog_ = stack.progs[(c, L_)]
            prog_.spans_mode = True
            try:
                prog_.prepare(stack.res[c], ci_)
                rows_ = []
                for slot_, name_, fn_ in prog_.steps(ci_):
                    n_before = pc()
                    fn_()  # compile-style: its outputs are the chain's tensors for the next steps
                    if pc() != n_before:  # a program first run here (outside any capture, as a compile launch should)
                        spans_pc["compiled_steps"].append(dict(L=L_, slot=slot_, step=name_, program_cache=f"{n_before} -> {pc()}"))
                    if C.MOCK:
                        snap = prog_.snapshot()
                        fn_()  # the mock's stand-in for the capture (every step twice, as on the device)
                        prog_.restore(snap)
                        rows_.append(dict(slot=slot_, step=name_, us=None))
                        continue
                    ttnn.synchronize_device(mesh)
                    snap = prog_.snapshot()
                    tid_ = capture_guarded(fn_, f"L{L_} {slot_}/{name_}")
                    prog_.restore(snap)  # the captured run's outputs belong to the trace
                    st_ = S.replay_stats(mesh, tid_, replays=SPAN_REPLAYS, min_s=0.0, label=f"{tag} L{L_} {slot_}/{name_}")
                    ttnn.release_trace(mesh, tid_)
                    rows_.append(dict(slot=slot_, step=name_, us=st_.get("median_us"), replays=st_.get("replays")))
                by_slot = {}
                for r_ in rows_:
                    if r_["us"] is not None:
                        by_slot[r_["slot"]] = by_slot.get(r_["slot"], 0.0) + r_["us"]
                spans[L_] = dict(steps=rows_, us_by_slot=by_slot, modes=prog_.modes)
                H.log(f"{tag}: L{L_} spans {by_slot} ({len(rows_)} steps)")
            finally:
                prog_.spans_mode = False
                prog_._free_owned()
    spans_pc["after"] = pc()
    any_pmk = any(m == "pmk" for L_ in LAYERS for m in stack.progs[(c, L_)].modes.values())
    bar_ok = all(per_layer[L_]["pcc"] >= PCC_BAR or per_layer[L_]["bitwise"] for L_ in LAYERS) and (lg["pcc"] >= PCC_BAR or lg["bitwise"])
    ok = bar_ok and tok_ref == tok_pmk and all(v["eager"] == 0 for v in chain.values())
    status = "mock" if C.MOCK else ("pass" if ok else ("info" if not any_pmk else "fail"))
    row = dict(test=f"PMK served stack unit: L{LAYERS[0]}-L{LAYERS[-1]} GLM53_PF_MODE=pmk (slots {os.environ['GLM53_PMK_SLOTS']}) vs the op path on the "
                    f"same TPPrefill, {N} rows tp{TPN}, both traced; bar per-layer residual / logits PCC >= {PCC_BAR}, first token equal",
               milestone="PMK-B5", status=status, tp=TPN, layers=LAYERS, bucket=c, pcc_bar=PCC_BAR,
               per_layer={L_: {k: v for k, v in d.items()} for L_, d in per_layer.items()}, logits=lg,
               first_token=dict(op=tok_ref, pmk=tok_pmk, equal=tok_ref == tok_pmk),
               us_op=st_ref.get("median_us"), us_pmk=st_pmk.get("median_us"), replays=st_ref.get("replays"),
               chain_counters=chain, wired=info["wired"], programs_per_replay=sum(v["programs"] for v in info["wired"].values()),
               census=stack.census(), eager_guard=eager_final,
               spans=spans, slot_pcc=slot_pcc, tap_swap=swap, b_config={L_: getattr(stack.progs[(c, L_)], "b_cfg", None) for L_ in LAYERS if stack.progs[(c, L_)].b is not None},
               b_dropped={L_: getattr(stack.progs[(c, L_)], "b_dropped", None) for L_ in LAYERS if stack.progs[(c, L_)].b is not None},
               program_cache=pc_rows, spans_program_cache=spans_pc, signature=stack.signature()["signature"],
               chain_build_s=info["chain_build_s"], mock=C.MOCK, wall_s=round(time.time() - t0, 1), tag=tag,
               probe_layers=PROBE, page_table=info.get("page_table"), alias=info.get("alias"), trace_mib=info.get("trace_mib"),
               launches_per_replay={L_: round(v["launches"] / max(1, v["calls"]), 2) for L_, v in chain.items()},
               fillers_per_replay={L_: round(v["fillers"] / max(1, v["calls"]), 2) for L_, v in chain.items()})  # fmt: skip
    H.log({k: v for k, v in row.items() if k not in ("test", "wired", "census")})
    if rec and not C.MOCK:
        C.record("tests", row)
        for L_, d in per_layer.items():
            C.record("pcc", dict(test=row["test"], milestone="PMK-B5", layer=L_, tp=TPN, bucket=c, criterion="layer output residual vs op",
                                 pcc=d["pcc"], bitwise=d["bitwise"], modes=d["modes"], status=status, tag=tag))  # fmt: skip
        C.record("perf", dict(test=f"PMK served stack L{LAYERS[0]}-{LAYERS[-1]} tp{TPN} traced bucket replay: chain vs op path",
                              milestone="PMK-B5", tp=TPN, bucket=c, us=st_pmk.get("median_us"), us_op=st_ref.get("median_us"),
                              replays=st_pmk.get("replays"), status=status, tag=tag))  # fmt: skip
except Exception:  # noqa: BLE001
    err = traceback.format_exc()[-6000:]
    H.log(f"FAILED:\n{err}")
    row = dict(test="PMK served stack unit", milestone="PMK-B5", status="fail", error=err, tp=TPN, layers=LAYERS, mock=C.MOCK,
               wall_s=round(time.time() - t0, 1), tag=tag)  # fmt: skip
    if rec and not C.MOCK:
        C.record("tests", row)
RESULT = row
