# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): M3 — the TP dense_kda layer (tt/tp8_layer.py: 7a's two megakernel programs on every chip on its
TP shard + the in-program fabric all-reduce after o_proj and down) vs the Stage 3 single-chip layer, device vs device:

  reference (jobs/m3_ref.py in a single-chip server, saved to work-s14/m3_ref/): the Stage 3 OptimizedDecoder layer,
      B users of golden prompt P (layer_check's Users: prefix n_i = P - 1 - i), traced prefill of every prefix, state
      snapshot (S, C), ONE traced decode step -> y_ref rows + the decoded state; the golden rows (fp32 reference
      model) of the same positions;
  TP: x = the users' input rows (replicated), S / C = the snapshot sharded per chip (tt/tp8.py), all users active;
      compile launch, then ONE traced call (numerics: state restored first) -> the layer output on every chip (must
      be identical), PCC vs y_ref (bar 0.9999) and vs golden (bar 0.995), the new state vs the reference's;
      timing: a capture WITHOUT the restore, replayed N times (state drifts; timing only) -> median us.
ARGS: layer (0), b (1; > 4 = chunks of 4 rows), prompt (128), reps (30), record (False), tag, probe (False),
 program (layer | halves), kda_spread (kda_pre cores per head's columns), rec_spread (kda_rec cores per head),
 chain (N > 1: also time N back-to-back launches in one trace -> steady state per layer), rev, vtp (shard as TP vtp
 on this mesh's chips: TP8-sized per-chip programs on the pair; numerics not comparable, rows "info")."""
import gc
import importlib
import os
import sys
import time

import torch

import ttnn

gc.collect()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.ds_mm as DSMM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.fused_decoder as FDEC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder as ODEC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.fabric_ar as FAR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_pre as KPRE  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mhc_post as MPOST  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mhc_pre as MPRE  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_readers as MKR  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.kda_rec as KREC  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_dense as MKD  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_ffn as MKF  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_layer as MKL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8 as TP  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_layer as TPL  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_experts as MKX  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_moe as MKM  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.mk_sparse as MKS  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.tp8_sparse as TPS  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST, hf_config, pcc  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import (  # noqa: E402
    FdTileCacheSource,
    OptimizedDecoderConfig,
)

LC_DIR = "/opt/glm53/kit/doc/ttnn/07a-fused-streaming-layer/scripts"
sys.path.insert(0, LC_DIR)
import layer_check as LC  # noqa: E402

mesh = MESH  # noqa: F821
JOB_ARGS = ARGS  # noqa: F821
args = JOB_ARGS
ctx = CTX  # noqa: F821
# 7a's megakernel knobs (read when the layer is built): reset every job (the server's environment persists), then the
# job's overrides
KNOBS = ("GLM53_MK_KB_QKV", "GLM53_MK_KB_O", "GLM53_MK_RING_KB_QKV", "GLM53_MK_RING_KB_O", "GLM53_MK_RING_KB",
         "GLM53_MK_READERS", "GLM53_MK_LR_FIDELITY", "GLM53_MK_DS_FIDELITY", "GLM53_MHC_ROWREAD", "GLM53_MK_DEFER_INV",
         "GLM53_MHC_POST_ROWWRITE", "GLM53_REC_EARLY_SIGNAL", "GLM53_MHC_SYNC_MARKS",
         "GLM53_REC_DEPTH", "GLM53_REC_ITEM_MARKS", "GLM53_REC_UGROUPS", "GLM53_MK_W_DELAY_US", "GLM53_MHC_PRE_EARLY", "GLM53_KDA_PREFETCH", "GLM53_MHC_HN_FIRST", "GLM53_TP_SHARED",
         "GLM53_REC_MODE", "GLM53_MHC_MODE", "GLM53_MK_ROUTER_FIDELITY", "GLM53_MK_KB_ROUTER",
         "GLM53_MK_ROUTER_KSPLIT", "GLM53_MK_FAR_CREDIT_NEXT", "GLM53_REC_V2_UG",
         "GLM53_EP_PLACEMENT", "GLM53_KDA_RSPLIT", "GLM53_TP_XSL", "GLM53_MKX_RING", "GLM53_MKX_XSL_INFLIGHT", "GLM53_TP_XSL_SLICED",
         "GLM53_MHC_FPU_COLLAPSE", "GLM53_MHC_DIAG", "GLM53_MK_SPARSE_DEFER_INV", "GLM53_MHC_PRE_SPLIT",
         "GLM53_MK_QKV_ALIAS", "GLM53_MK_QKV_R0")  # 7a's M1 (879 us): 3 readers per bank, GLM53_MK_RING_KB=750


def _tree_rev() -> str:
    """The model tree's code state (HEAD + a hash of uncommitted model-dir edits): the default S14_REV define, so a
    long-lived server never reuses a FILE_PATH kernel binary built from older sources (cached by path + args)."""
    import hashlib
    import subprocess

    home = os.environ.get("TT_METAL_HOME", ".")
    md = "models/autoports/zai_org_glm_5_3_flash"
    head = subprocess.run(["git", "-C", home, "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", home, "diff", "HEAD", "--", md], capture_output=True, text=True).stdout
    return head + ("_" + hashlib.sha1(diff.encode()).hexdigest()[:8] if diff else "")


REV = args.get("rev") or _tree_rev()
# Stage 14 job pipeline rule (program 23:52:53Z): resident state. The model modules are reloaded only when the
# tree's code revision changes (leaves first: a stale mk_readers with a fresh reader_a.cpp = CT mismatch); everything
# built with older code is dropped then. Within one revision: the device weights per (layer, kind, vtp, spreads), the
# chunk inputs / states per (layer, B, prompt, vtp) and the references stay in CTX; a knob change builds programs only.
if ctx.get("m3_rev") != REV:
    for mod in (FDEC, ODEC, DSMM, MKR, MPRE, MPOST, KREC, KPRE, MKD, MKF, MKL, MKX, MKM, MKS, FAR, TP, TPL, TPS):
        importlib.reload(mod)
    for k in [k for k in ctx if isinstance(k, tuple) and k[0] in ("tp", "tpw", "moe", "chunks")]:
        ctx.pop(k)
    gc.collect()
    ctx["m3_rev"] = REV
W = os.environ.get("GLM53_WORK", "/opt/glm53/work-s14")
cfg = hf_config()
source = FdTileCacheSource(CACHE_DIR, MANIFEST)


def log(*a):
    print(f"M3 {C.now()}", *a, flush=True)



def interleave_test(args, step, ys_c, CH, *, tag, layer, B, env):
    """ONE trace = [state restore -> TP layer (2 in-program fabric all-reduces) -> unit op chain on the same mesh
    (embedding, a [32, 4096] x [4096, 16384] matmul, softmax, top-k: ordinary ttnn programs whose CBs start at the bottom
    of the CB region on every core, the fabric lane cores included)] x 2; replayed `interleave` / 2 times, every output
    (both layer launches, the unit chain's top-k) compared bitwise with the first replay. The lane CB region is NOT
    reserved: under FAR_CREDIT_NEXT a peer writes a chip's landing CBs only after that chip's lane program started
    the launch, and the only lane state kept across programs is the counter tensor (an allocated L1 tensor)."""
    mesh_ = mesh
    rep = ttnn.ReplicateTensorToMesh(mesh_)
    g = torch.Generator().manual_seed(1234)
    key = ("units", CH)
    if key not in CTX:  # noqa: F821
        table = torch.randn(8192, 4096, generator=g).to(torch.bfloat16)
        ids = torch.randint(0, 8192, (1, 32), generator=g, dtype=torch.int32)
        w = (torch.randn(4096, 16384, generator=g) * 0.02).to(torch.bfloat16)
        CTX[key] = dict(  # noqa: F821
            table=ttnn.from_torch(table, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep),
            ids=ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep),
            w=ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep))  # fmt: skip
    u = CTX[key]  # noqa: F821

    # scrub_kb > 0: an L1 scrubber program after the unit chain — ONE CB of scrub_kb KiB from the bottom of the CB
    # region on every worker core (the fabric lane cores' own / landing / result CBs included), NaN-filled by BRISC
    scrub_kb = int(args.get("scrub_kb", 0))
    scrub = None
    if scrub_kb:
        grid = mesh_.compute_with_storage_grid_size()
        allc = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        pages = scrub_kb * 1024 // 2048
        src = ("#include <cstdint>\n#include \"api/dataflow/dataflow_api.h\"\n"
               "void kernel_main() {\n  constexpr uint32_t pages = get_compile_time_arg_val(0);\n"
               "  cb_reserve_back(0, pages);\n"
               "  volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));\n"
               "  for (uint32_t i = 0; i < pages * 512; ++i) { p[i] = 0x7FC07FC0u; }\n}\n")  # bf16 NaN pairs
        kd = ttnn.KernelDescriptor(kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                   core_ranges=allc, compile_time_args=[pages], runtime_args=ttnn.RuntimeArgs(),
                                   config=ttnn.ReaderConfigDescriptor())  # fmt: skip
        cb = ttnn.CBDescriptor(total_size=pages * 2048, core_ranges=allc, format_descriptors=[
            ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)])  # fmt: skip
        scrub = ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[cb])

    def units():
        e = ttnn.embedding(u["ids"], u["table"], layout=ttnn.TILE_LAYOUT)  # [1, 32, 4096]
        e = ttnn.reshape(e, (1, 1, 32, 4096))
        m = ttnn.matmul(e, u["w"], memory_config=ttnn.DRAM_MEMORY_CONFIG)  # [1, 1, 32, 16384]
        sm = ttnn.softmax(m, dim=-1)
        vals, idx = ttnn.topk(sm, k=32, dim=-1)
        if scrub is not None:
            ttnn.generic_op([u["ids"], u["ids"]], scrub)
        return [m, vals, idx]

    def step2():
        outs = []
        for _ in range(2):  # TP layer -> unit chain -> TP layer -> unit chain
            outs += step()
            outs += units()
        return outs

    tr2 = C.Traced(ttnn, mesh_, step2)
    n_rep = max(1, int(args["interleave"]) // 2)
    ref_ = None
    bad, events, t0 = 0, [], time.time()
    for i in range(n_rep):
        tr2.replay()
        cur = [[ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(o)] for o in tr2.out]
        if ref_ is None:
            ref_ = cur
            continue
        for oi, (a_, b_) in enumerate(zip(cur, ref_)):
            for d, (x_, y_) in enumerate(zip(a_, b_)):
                if not torch.equal(x_, y_):
                    bad += 1
                    if len(events) < 20:
                        events.append(dict(replay=i, output=oi, chip=d, n_diff=int((x_ != y_).sum()),
                                           nonfinite=int((~torch.isfinite(x_)).sum())))
                        log("INTERLEAVE_MISMATCH", events[-1])
    tr2.release()
    # the layer outputs of the interleaved trace vs the plain numerics trace's (same restored state, same program)
    n_layer = len(ys_c)
    layer_same = all(torch.equal(ref_[k][d].reshape(-1, 16384)[:CH], ys_c[k][d]) for k in range(n_layer)
                     for d in range(len(ys_c[k])))  # fmt: skip
    res = dict(alternations=2 * n_rep, replays=n_rep, mismatching_outputs=bad, events=events,
               layer_equal_plain_trace=layer_same, seconds=round(time.time() - t0, 1),
               unit_chain="embedding [8192, 4096] -> matmul [32, 4096] x [4096, 16384] (DRAM) -> softmax -> topk 32"
               + (f" -> L1 scrubber ({scrub_kb} KiB CB NaN-filled from the CB base on every core)" if scrub_kb else ""))
    ok = bad == 0 and layer_same
    row = dict(test=f"fabric interleave tp L{layer} B={B} tp{mesh_.get_num_devices()} {CTX['mgd']}",  # noqa: F821
               status="pass" if ok else "fail", tag=tag, layer=layer, B=B, interleave=res, knobs=env,
               watcher=bool(args.get("watcher")),
               note="TP layer -> unit op chain -> TP layer in ONE trace, replayed; bitwise vs the first replay")  # fmt: skip
    if args.get("record"):
        C.record("pcc", row)
    log("INTERLEAVE", res)
    return dict(interleave=res, ok=ok)


def run(args):
    """One configuration (the job's ARGS, or ARGS + one entry of ARGS["sweep"]) on the resident state -> its result;
    its pcc / perf rows are recorded as it finishes."""
    # every model knob is reset per config (the 8-chip server also runs other stages' jobs, whose env settings would
    # otherwise leak in): the listed ones and anything with a model-knob prefix (7a's / 14c's knobs included)
    MODEL_PREFIX = ("GLM53_MK_", "GLM53_MKX_", "GLM53_MHC_", "GLM53_REC_", "GLM53_KDA_", "GLM53_TP_")
    for k_ in set(KNOBS) | {k for k in os.environ if k.startswith(MODEL_PREFIX)}:
        os.environ.pop(k_, None)
    ENV = {k_: str(v_) for k_, v_ in (args.get("env") or {}).items()}
    assert all(k in KNOBS or k.startswith(MODEL_PREFIX) for k in ENV), ENV
    os.environ.update(ENV)
    LAYER, B, PROMPT = int(args.get("layer", 0)), int(args.get("b", 1)), int(args.get("prompt", 128))
    REPS = int(args.get("reps", 30))
    rec = bool(args.get("record", False))
    if args.get("watcher"):
        assert os.environ.get("TT_METAL_WATCHER"), "watcher job outside a watcher process"

    tag = args.get("tag", "m3")


    # ------------------------------------------------------------------ reference (Stage 3 single-chip layer)
    ref_key = ("ref", LAYER, B, PROMPT)
    ref_path = os.path.join(W, "m3_ref", f"L{LAYER}_B{B}_P{PROMPT}.pt")
    if ref_key not in ctx:
        if C.MOCK and not os.path.exists(ref_path):  # mock cluster (compile / build check only): synthetic reference
            g = torch.Generator().manual_seed(0)
            z = lambda *sh: torch.randn(*sh, generator=g)  # noqa: E731
            ctx[ref_key] = dict(x=z(B, 16384), y_ref=z(B, 16384), golden=z(B, 16384), S0=z(B, 64, 128, 128) * 0.1,
                                C0=z(1, 3, B, 24576).to(torch.bfloat16), S1=z(B, 64, 128, 128),
                                C1=z(1, 3, B, 24576).to(torch.bfloat16), positions=[0] * B, prefill_mode="mock")  # fmt: skip
        elif not os.path.exists(ref_path):  # computed by jobs/m3_ref.py in a single-chip server (never on a submesh)
            raise FileNotFoundError(f"reference {ref_path} missing: run jobs/m3_ref.py on a single-chip server first")
        else:
            ctx[ref_key] = torch.load(ref_path)
            log("reference loaded", ref_path)
    ref = ctx[ref_key]

    # ------------------------------------------------------------------ the TP layer
    # chunked (default): B > 4 users = B / 4 launches of 4 rows inside the ONE traced step (each chunk its own state /
    # history / output tensors; same programs, back-to-back); native (ARGS native, program 01:57:50Z): ONE launch of
    # B <= 32 rows (7a's native-t megakernels; kda_rec at t > 4 needs rec_spread 1 — user groups spread its rows)
    CH = B if args.get("native") else min(B, 4)
    assert CH <= 32, CH
    assert B % CH == 0, B
    NCH = B // CH
    lay_key = ("tp", LAYER, CH, args.get("program", "layer"), args.get("kda_spread"), bool(args.get("probe", False)),
               int(args.get("rec_spread", 1)), REV, tuple(sorted(ENV.items())), args.get("vtp"))
    SPARSE = cfg.mlp_layer_types[LAYER] == "sparse"  # sparse_kda (KDA + MoE): tt/tp8_sparse.py on 7a's MkSparseLayer
    assert cfg.layer_types[LAYER] == "linear_attention", f"layer {LAYER}: {cfg.layer_types[LAYER]} (KDA kinds only here)"
    LCLS = TPS.TPSparseKDALayer if SPARSE else TPL.TPDenseKDALayer
    wkey = ("tpw", LAYER, SPARSE, args.get("vtp"), int(args.get("rec_spread", 1)), args.get("kda_spread"))
    if lay_key not in ctx or type(ctx[lay_key]) is not LCLS:  # rebuilt after a code reload
        for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "tp"]:  # one TP program set at a time (L1)
            ctx.pop(k)
        if wkey not in ctx:  # one resident weight set at a time (DRAM)
            for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "tpw"]:
                ctx.pop(k)
        gc.collect()
        t0 = time.time()
        if SPARSE:  # the layer's MoE (router, router tail, shared expert, the 288 experts) as the decoder loads it
            mkey = ("moe", LAYER)
            if mkey not in ctx:
                for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "moe"]:
                    ctx.pop(k)
                gc.collect()
                t0 = time.time()
                ctx[mkey] = TPS.build_moe(mesh, cfg, source, LAYER, OptimizedDecoderConfig())
                log(f"MoE of layer {LAYER} built in {time.time() - t0:.1f}s")
            ctx[lay_key] = TPS.TPSparseKDALayer(mesh, cfg, LAYER, source, OptimizedDecoderConfig(), b=CH,
                                                moe=ctx[mkey], kda_spread=args.get("kda_spread"),
                                                rec_spread=int(args.get("rec_spread", 1)), rev=REV,
                                                vtp=args.get("vtp"), weights=ctx.get(wkey),
                                                probe=bool(args.get("probe", False)))  # fmt: skip
        else:
            ctx[lay_key] = TPL.TPDenseKDALayer(mesh, cfg, LAYER, source, OptimizedDecoderConfig(), b=CH,
                                               probe=bool(args.get("probe", False)),
                                               program=args.get("program", "layer"),
                                               kda_spread=args.get("kda_spread"),
                                               rec_spread=int(args.get("rec_spread", 1)),
                                               rev=REV, vtp=args.get("vtp"), weights=ctx.get(wkey))  # fmt: skip
        log(f"TP layer {'programs' if wkey in ctx else 'weights + programs'} built in {time.time() - t0:.1f}s, "
            f"lanes {ctx[lay_key].lanes}, fabric cb {getattr(ctx[lay_key].fabric, 'addr', None)}")
        ctx[wkey] = ctx[lay_key].w
    lay = ctx[lay_key]
    sh = lay.shards
    if args.get("build_only"):  # compile-ahead: weights + program objects resident, nothing launched
        return dict(built=lay_key[0:2], lanes=lay.lanes)
    ckey = ("chunks", LAYER, B, PROMPT, args.get("vtp"), int(args.get("rec_spread", 1)))
    if ckey not in ctx:
        for k in [k for k in ctx if isinstance(k, tuple) and k[0] == "chunks"]:
            ctx.pop(k)
        gc.collect()
        act = lay.masks(torch.ones(CH, dtype=torch.bool))
        chunks = []
        for c in range(NCH):
            sl = slice(c * CH, (c + 1) * CH)
            S0c, C0c = ref["S0"][sl], ref["C0"][:, :, sl]
            chunks.append(dict(x=lay.replicated(ref["x"][sl].reshape(1, 1, CH, 16384)),
                               snap=sh.shard_state(S0c, C0c),  # the snapshot (restored inside the numerics call)
                               live=sh.shard_state(S0c, C0c)))  # the live state  # fmt: skip
        ctx[ckey] = (act, chunks)
    act, chunks = ctx[ckey]


    # MoE layers: the router logits of the numerics call are kept (the near-tie rule below needs their margins)
    ROUTER_DIAG = SPARSE and args.get("router_diag", True) is not False
    LOGITS = []  # (a list mutated in place: step() is a closure of run())


    def step(restore=True):
        outs = []
        LOGITS.clear()
        for ch in chunks:
            (S0m, C0m), (Sm, Cm) = ch["snap"], ch["live"]
            if restore:
                ttnn.copy(S0m, Sm)
                ttnn.copy(C0m, Cm)
            r = lay(ch["x"], Cm, act, Sm, t=CH)
            if ROUTER_DIAG and restore:
                LOGITS.append(ttnn.to_memory_config(r[2]["logits"], ttnn.DRAM_MEMORY_CONFIG))
            outs.append(r[1])  # keep the DRAM layer output only: the chunk's L1 temporaries are freed before the next
            del r  # chunk (they would push the next chunk's L1 allocations into the static CB region)
        return outs


    tr = C.Traced(ttnn, mesh, step)  # compile launch first, then the capture
    pc0 = mesh.num_program_cache_entries()
    tr.replay()
    ys_c = [[ttnn.to_torch(d).float().reshape(-1, 16384)[:CH] for d in ttnn.get_device_tensors(o)] for o in tr.out]
    ys = [torch.cat([yc[d] for yc in ys_c]) for d in range(len(ys_c[0]))]
    if args.get("interleave"):  # FabricPhase vs other programs on the lane cores (program 14:24Z, 7a #7y(2))
        return interleave_test(args, step, ys_c, CH, tag=args.get("tag", ""), layer=LAYER, B=B, env=ENV)
    soak = None
    if args.get("soak_s"):  # soak: the numerics trace (state restored each time) replayed for soak_s seconds, every
        # output compared bitwise with the first replay's (one host read per replay; all chips' replicas)
        t_end, n_rep, bad, mx = time.time() + float(args["soak_s"]), 0, 0, 0.0
        events = []  # the first mismatches: replay index, chunk, chip, rows / columns hit, non-finite count, max-abs
        bad_t = []  # the first 6 mismatching outputs + their references (saved to m3_out for offline analysis)
        while time.time() < t_end:
            tr.replay()
            for c_, o_ in enumerate(tr.out):
                for d_, dt_ in enumerate(ttnn.get_device_tensors(o_)):
                    v_ = ttnn.to_torch(dt_).float().reshape(-1, 16384)[:CH]
                    if not torch.equal(v_, ys_c[c_][d_]):
                        bad += 1
                        dif = (v_ - ys_c[c_][d_]).abs()
                        dif = torch.where(torch.isnan(dif), torch.full_like(dif, float("inf")), dif)
                        mx = max(mx, float(dif.max()))
                        if len(events) < 20:
                            hit = (dif > 0).nonzero()
                            events.append(dict(replay=n_rep, chunk=c_, chip=d_, n_diff=int(hit.shape[0]),
                                               rows=sorted({int(r) for r in hit[:, 0].tolist()})[:8],
                                               cols=[int(hit[:, 1].min()), int(hit[:, 1].max())] if hit.numel() else None,
                                               nonfinite=int((~torch.isfinite(v_)).sum()), maxabs=float(dif.max())))  # fmt: skip
                            log("SOAK_MISMATCH", events[-1])
                        if len(bad_t) < 6:
                            bad_t.append(dict(replay=n_rep, chunk=c_, chip=d_, got=v_.clone(), ref=ys_c[c_][d_].clone()))
            n_rep += 1
        soak = dict(seconds=float(args["soak_s"]), replays=n_rep, mismatching_outputs=bad, mismatch_maxabs=mx,
                    events=events)
        if bad_t:
            os.makedirs(os.path.join(W, "m3_out"), exist_ok=True)
            soak["saved"] = os.path.join(W, "m3_out", f"soak_{args.get('tag', 'x')}_{time.strftime('%H%M%S')}.pt")
            torch.save(bad_t, soak["saved"])
        log("SOAK", soak)
    st_c = [sh.gather_state(*ch["live"]) for ch in chunks]
    router = None
    if ROUTER_DIAG:  # choice = sigmoid(logits) + bias (router_tail.reference_step): the 8th vs 9th choice margin per row
        bias = lay.mlp.router.tail.host_bias.float()
        lg = torch.cat([ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float().reshape(-1, t.shape[-1])[:CH] for t in LOGITS])
        ks_ = int(getattr(lay.mk, "router_ks", 1) or 1)  # GLM53_MK_ROUTER_KSPLIT: the K-chunk partials side by side
        if ks_ > 1:
            lg = sum(lg[:, c * 320:(c + 1) * 320] for c in range(ks_))
        choice = torch.sigmoid(lg[:, : bias.numel()]) + bias
        srt = choice.sort(dim=-1, descending=True).values
        margin = (srt[:, 7] - srt[:, 8]).tolist()
        router = dict(margin_8_9=[round(m, 7) for m in margin], min_margin=min(margin),
                      min_margin_row=min(range(len(margin)), key=lambda i: margin[i]))  # fmt: skip
        for t_ in LOGITS:  # made inside the numerics capture: freed before the timing captures (no live allocation)
            ttnn.deallocate(t_)
        LOGITS.clear()
    S1, C1 = torch.cat([v[0] for v in st_c]), torch.cat([v[1] for v in st_c], dim=2)
    if args.get("save"):  # device-vs-device: the layer output + new state of this program, for a bitwise comparison later
        torch.save(dict(y=ys[0], S1=S1, C1=C1), os.path.join(W, "m3_out", f"{args['save']}.pt")) if os.makedirs(
            os.path.join(W, "m3_out"), exist_ok=True) is None else None
    probes = None
    if getattr(lay, "mk", None) is not None and lay.mk.probe is not None:  # per-phase marks of the traced numerics call
        from models.autoports.zai_org_glm_5_3_flash.tt.mk_layer import decode_probe

        MHZ = 1350.0
        lanes = lay.fabric.lane_marks() if lay.fabric is not None else None  # tp1 (no fabric): none
        probes = {}
        for d, pt in enumerate(ttnn.get_device_tensors(lay.mk.probe)):
            ph = ttnn.to_torch(pt)
            p = ph.reshape(-1, 16).long() & 0xFFFFFFFF
            if not bool((p > 0).any()):  # mock cluster: nothing written
                continue
            t0 = int(p[p > 0].min())

            def us(v, t0=t0):  # signed 32-bit cycle difference vs the program's earliest mark
                dd = (int(v) - t0) & 0xFFFFFFFF
                return round((dd - (1 << 32) if dd >= (1 << 31) else dd) / MHZ, 1) if v else None

            probes[d] = dict(mk=decode_probe(ph, lay.mk.readers.assign, k_cores=len(lay.mk.place["k"]), mhz=MHZ),
                             lane_brisc=[[us(v) for v in ln[0]] for ln in lanes[d]] if lanes else None,
                             lane_ncrisc=[[us(v) for v in ln[1]] for ln in lanes[d]] if lanes else None,
                             # per owner core (k index): the raw marks (the medians above truncate to the shortest list,
                             # which drops the kda_rec marks of the cores that run the recurrence)
                             owner_brisc_all=[[us(v) for v in p[k] if int(v) > 0] for k in range(len(lay.mk.place["k"]))],
                             owner_ncrisc_all=[[us(v) for v in p[96 + k] if int(v) > 0]
                                               for k in range(len(lay.mk.place["k"]))])  # fmt: skip
            log(f"PROBE chip {d}", probes[d])
        # controlled steady metric (Stage 7a #7v, program 08:00Z: mk_one_span_restored_us): the probe span (earliest
        # to latest mark of the program, all cores) of a RESTORED launch — the numerics trace replayed span_replays
        # times, per chip the median span, the slowest chip reported
        if probes:
            def span(ph):  # 32-bit cycle marks: the span = 2^32 - the largest circular gap (wrap-safe)
                flat = ph.reshape(-1)
                v = sorted(set((flat.long() & 0xFFFFFFFF)[flat != 0].tolist()))
                if len(v) < 2:
                    return 0.0
                gaps = [y - x for x, y in zip(v, v[1:])] + [v[0] + (1 << 32) - v[-1]]
                return ((1 << 32) - max(gaps)) / MHZ
            per_chip = {d: [] for d in probes}
            for _ in range(int(args.get("span_replays", 5))):
                tr.replay()
                for d, pt in enumerate(ttnn.get_device_tensors(lay.mk.probe)):
                    if d in per_chip:
                        per_chip[d].append(span(ttnn.to_torch(pt)))
            med = {d: round(sorted(v)[len(v) // 2], 2) for d, v in per_chip.items() if v}
            probes["span_restored_us"] = max(med.values()) if med else None
            probes["span_restored_per_chip_us"] = med
            log("SPAN_RESTORED", med)
    tr.release()
    same = all(torch.equal(ys[0], v) for v in ys[1:])
    p_ref, p_gold = pcc(ys[0], ref["y_ref"]), pcc(ys[0], ref["golden"])
    p_ref_gold = pcc(ref["y_ref"], ref["golden"])
    p_state = pcc(S1, ref["S1"]) if not sh.virtual else None
    hist_ok = bool(torch.equal(C1, ref["C1"])) if not sh.virtual else None
    # not bitwise (e.g. GLM53_MK_DEFER_INV: qkv rounded after the deferred 1 / rms scaling): how far the history is
    hist_cmp = None if sh.virtual or hist_ok else dict(pcc=pcc(C1.float(), ref["C1"].float()),
                                                       max_abs=float((C1.float() - ref["C1"].float()).abs().max()))
    mae = float((ys[0] - ref["y_ref"]).abs().max())
    # per-row view (B > 1): one row far off with the others at ~1.0 points at a discrete flip (MoE top-k near-tie)
    rows_pcc = [pcc(ys[0][i], ref["y_ref"][i]) for i in range(ys[0].shape[0])] if ys[0].shape[0] > 1 else None
    rows = None
    if rows_pcc:
        worst = min(range(len(rows_pcc)), key=lambda i: rows_pcc[i])
        keep = [i for i in range(len(rows_pcc)) if i != worst]
        rows = dict(min=rows_pcc[worst], worst_row=worst, below_9999=sum(p < 0.9999 for p in rows_pcc),
                    pcc_without_worst=pcc(ys[0][keep], ref["y_ref"][keep]),
                    worst_row_pcc_vs_golden=pcc(ys[0][worst], ref["golden"][worst]),
                    worst_row_stage3_vs_golden=pcc(ref["y_ref"][worst], ref["golden"][worst]))  # fmt: skip
    # timing: capture without the restores (state drifts; timing only)
    trt = C.Traced(ttnn, mesh, lambda: step(False), warm=False)
    walls = trt.replays(REPS)
    batch_us = trt.batch(REPS)
    try:  # allocations made while a trace is alive (TT_METAL_TRACE_ALLOC_TRACKING=1): must be none at replay time
        from ttnn._ttnn.operations.trace import get_all_unsafe_tracked_ids

        unsafe = len(get_all_unsafe_tracked_ids())
    except Exception as e:  # noqa: BLE001
        unsafe = f"n/a: {e}"
    trt.release()
    st = C.stats(walls)
    pc1 = mesh.num_program_cache_entries()  # the timing captures + replays add no program
    # steady state per layer (7a's M1 convention): CHAIN launches of the layer back to back in ONE trace (what the
    # 45-layer step does: the next program's dispatch overlaps the running one) -> (chain - single) / (CHAIN - 1)
    CHAIN = int(args.get("chain", 0))
    chain = None
    if CHAIN > 1:
        trc = C.Traced(ttnn, mesh, lambda: [step(False) for _ in range(CHAIN)], warm=False)
        wc = C.stats(trc.replays(REPS))
        bc = trc.batch(REPS)
        trc.release()
        chain = dict(n=CHAIN, median_us=wc["median_us"], batch_mean_us=round(bc, 2) if bc else None,
                     steady_per_layer_us=round((wc["median_us"] - st["median_us"]) / (CHAIN - 1), 2),
                     mean_per_layer_us=round(wc["median_us"] / CHAIN, 2))  # fmt: skip
    VIRTUAL = bool(sh.virtual)  # TP-vtp shards on fewer chips: the all-reduce sums a subset -> timing only
    EXPLORE = bool(args.get("explore")) or any(k in ENV for k in ("GLM53_MK_DS_FIDELITY", "GLM53_MK_LR_FIDELITY", "GLM53_MK_ROUTER_FIDELITY"))
    ok = same and p_ref >= 0.9999 and p_gold >= 0.995 and (soak is None or soak["mismatching_outputs"] == 0)
    # MoE layer_check near-tie rule (Stage 14, 2026-10-20; evidence: L44 B=32 prompt 512 — its 3 rows < 0.9999 are
    # exactly its 3 rows with the smallest router margins, all < 1e-3): a row whose 8th and 9th router choices
    # (sigmoid(logit) + bias) are closer than TIE_MARGIN may pick the other expert under any rounding difference vs the
    # single-chip reference; such rows are reported and the 0.9999 bar applies to the other rows (at most B / 8 tie rows)
    TIE_MARGIN = 1e-3
    tie = None
    if router is not None and ys[0].shape[0] > 1:
        m_ = router["margin_8_9"]
        tie_rows = [i for i, v in enumerate(m_) if v < TIE_MARGIN]
        keep_ = [i for i in range(len(m_)) if i not in tie_rows]
        p_keep = pcc(ys[0][keep_], ref["y_ref"][keep_]) if keep_ else None
        tie = dict(margin=TIE_MARGIN, rows=tie_rows, rows_pcc=[rows_pcc[i] for i in tie_rows] if rows_pcc else None,
                   pcc_vs_stage3_other_rows=p_keep)  # fmt: skip
        if not ok and same and p_gold >= 0.995 and p_keep is not None and p_keep >= 0.9999 \
                and len(tie_rows) <= max(1, ys[0].shape[0] // 8):
            ok = True
    KIND = "sparse_kda" if SPARSE else "dense_kda"
    row = dict(test=f"tp {KIND} L{LAYER} B={B} tp{mesh.get_num_devices()} {CTX['mgd']}" + (f" virtual-tp{sh.tp}" if VIRTUAL else ""),  # noqa: F821
               # exploratory numerics-changing knobs (fidelity) are experiments, not deliverable tests: info + bar_met
               status="info" if (VIRTUAL or EXPLORE) else ("pass" if ok else "fail"), bar_met=ok,
               virtual_tp=sh.tp if VIRTUAL else None, explore=EXPLORE or None,
               layer=LAYER, B=B, tp=mesh.get_num_devices(), mgd=CTX["mgd"], replicas_identical=same,  # noqa: F821
               pcc_vs_stage3=p_ref, pcc_vs_golden=p_gold, stage3_vs_golden=p_ref_gold, max_abs_vs_stage3=mae,
               rows_pcc_vs_stage3=rows, router_diag=router, router_near_tie=tie, soak=soak, state_pcc_vs_stage3=p_state, conv_history_bitwise=hist_ok, conv_history_vs_stage3=hist_cmp, prefill_mode=ref["prefill_mode"], tag=tag,
               program_cache_after_capture=pc0, program_cache_after_replays=pc1, program_cache_stable=pc0 == pc1,
               unsafe_allocations_live_at_replay=unsafe,
               trace_alloc_tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"))  # fmt: skip
    prow = dict(test=f"tp {KIND} L{LAYER} B={B} step" + (f" virtual-tp{sh.tp}" if VIRTUAL else ""), status="info",
                milestone=(f"M3-vtp{sh.tp}" if VIRTUAL else ("M3" if CTX["mgd"] == "2x4" else "M3-dev")) + ("-sparse" if SPARSE else ""),  # noqa: F821
                virtual_tp=sh.tp if VIRTUAL else None,
                traced=True,
                # the layer's traced step: steady state per layer when chained (CHAIN launches in one trace / CHAIN, 7a's M1
                # convention accepted 19:58Z; includes 1 / CHAIN of the launch start-up), else the single-launch median
                us=chain["mean_per_layer_us"] if chain else st["median_us"],
                # the controlled span of a restored launch (probe runs only; the timing captures above drift the state)
                span_restored_us=(probes or {}).get("span_restored_us"),
                us_metric="steady_state_chain_mean" if chain else "single_launch_median",
                single_launch_us=st["median_us"], replay=st, batch_replay_mean_us=round(batch_us, 2) if batch_us else None,
                layer=LAYER, B=B, tp=mesh.get_num_devices(), tag=tag, launches_per_step=NCH, chain=chain,
                steady_per_layer_us=chain["steady_per_layer_us"] if chain else None,
                kda_pre_spread=lay.attn.kda_pre_spread, kda_rec_spread=lay.attn.kda_rec_spread, knobs=ENV,
                program=args.get("program", "layer"),
                note=("ONE program per chip (7a's MkLayer on the TP shard) with 2 in-program fabric all-reduces"
                      if args.get("program", "layer") == "layer" else
                      "2 programs per chip (MkStepB step D + MkFFN) with 2 in-program fabric all-reduces")
                     + ("; us = steady state per layer (chain of launches in one trace)" if chain else "; us = blocking replay of one traced step")
                     + (f"; B={B} = {NCH} back-to-back launches of {CH} rows (7a's megakernel serves t <= 4)" if NCH > 1 else ""))  # fmt: skip
    if rec:
        C.record("pcc", row)
        if not args.get("watcher"):  # a watcher process's timing is not a performance number
            C.record("perf", prow)
    else:
        import json

        with open(os.path.join(W, "dev_rows.jsonl"), "a") as f:
            f.write(json.dumps({"time_utc": C.now(), **row}, default=str) + "\n")
            f.write(json.dumps({"time_utc": C.now(), **prow}, default=str) + "\n")
    log("ROW", row)
    log("PERF", prow)
    return dict(pcc_ref=p_ref, pcc_gold=p_gold, same=same, us=st["median_us"], batch_us=batch_us, state=p_state,
                  hist=hist_ok, probes=probes, chain=chain)  # fmt: skip


# sweep-in-job: ARGS["sweep"] = [overrides, ...] (env replaces, the rest overrides key by key) -> one row set each
if JOB_ARGS.get("sweep"):
    base_args = {k: v for k, v in JOB_ARGS.items() if k != "sweep"}
    RESULT = []
    for i_, over in enumerate(JOB_ARGS["sweep"]):
        t_ = time.time()
        r_ = run({**base_args, **over})
        RESULT.append(dict(config=over, wall_s=round(time.time() - t_, 2), result=r_))
        log(f"SWEEP {i_ + 1}/{len(JOB_ARGS['sweep'])} {over.get('tag', '')} wall_s={time.time() - t_:.2f}")
else:
    RESULT = run(JOB_ARGS)
