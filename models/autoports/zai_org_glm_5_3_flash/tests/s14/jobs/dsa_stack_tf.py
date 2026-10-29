# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14 DSA in the persistent stack, GLM53_STACK_DSA=1; 16's #28 long-context blocker): the Stage 6
teacher-forcing harness (the plugin's readiness_check.run_teacher_forcing, unchanged; as 14b's m5c_tf.py) on
tt/stack_model.py's StackBackend with the sparse_dsa layers on MkMla(dsa) + the per-layer pooled-key store, PAST the
dense-exact 2,051 rows: a model prefix (default L0-L4: dense_kda L0-L2, sparse_dsa L3, sparse_kda L4) vs Stage 16's CPU
fp32 reference of the same prefix (default prefill4096_L0-4.refpt: a 4,064-token prompt + 32 forced tokens, so every
scored token of L3 selects its top-512 pools). Prefill: GLM53_STACK_PREFILL (default "decode": the decode program
writes every pooled key / ring row itself; "tp" + GLM53_PF_DSA=1: Stage 16's TP prefill writes the same store).
Pass (as m5c_tf): traced decode, top-5 >= 0.98 and top-100 = 1.0.
ARGS: layers ([0, 1, 2, 3, 4] or "all"), b (1), ref, max_seq_len (8192), native (False: 4-row chunks; True = the native
B-row step, rev 60's sharded B=32 gate), clear_program_cache (False: on a backend switch in one process, also drop the
program cache's references to the old backend), warm_only (build + warmup only), env (knobs set before the backend is built;
GLM53_STACK_DSA defaults to 1 here), tag, record (rows -> GLM53_STAGE_EVIDENCE pcc / perf)."""

import gc
import json
import os
import sys
import time
from pathlib import Path

gc.collect()
ENV = {"GLM53_STACK_DSA": "1", **{k: str(v) for k, v in (ARGS.get("env") or {}).items()}}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    import importlib

    import ttnn

    import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
    import models.autoports.zai_org_glm_5_3_flash.tt.mk_mla as MK
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM
    import models.autoports.zai_org_glm_5_3_flash.tt.tp_dsa as TPD
    import models.autoports.zai_org_glm_5_3_flash.tt.tp_mla as TPM
    from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    for m in (MK, TPM, TPD, SM, G):
        importlib.reload(m)
    WS = os.environ.get("GLM53_WS", "/opt/glm53/kit")
    sys.path.insert(0, os.path.join(WS, "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"))
    from readiness_check.run_teacher_forcing import run_teacher_forcing

    MODEL_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
    ids = (list(range(hf_config().num_hidden_layers)) if args.get("layers") == "all"
           else [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])])  # fmt: skip
    B = int(args.get("b", 1))
    ref = args.get("ref", "/opt/glm53/work-s16/refpt/prefill4096_L0-4.refpt")
    ms = int(args.get("max_seq_len", 8192))
    tag = args.get("tag", "dsa-stack-tf")
    native = bool(args.get("native", False))  # (default: the 4-row-chunk stack every earlier row of this job ran)
    tpn = int(mesh.get_num_devices())
    ev = os.environ.get("GLM53_STAGE_EVIDENCE")

    def log(*a):
        print(f"DSA-STACK-TF {time.strftime('%H:%M:%S')}", *a, flush=True)

    # the job's own knobs only: the stack's set_knobs() writes GLM53_* layer knobs into the environment during a run, so
    # hashing the whole environment rebuilt the backend in the next job (u31c u3: the rebuild after a close clashed)
    key = ("dsa_stack_tf", B, tuple(ids), ms, native, tuple(sorted(ENV.items())))
    switched = False
    for k in [k for k in ctx if isinstance(k, tuple) and k and k[0] == "dsa_stack_tf" and k != key]:
        be_old = ctx.pop(k)["backend"]  # one backend per process; the same (layers, B, knobs) is reused
        be_old.close() if hasattr(be_old, "close") else None
        del be_old
        switched = True
    gc.collect()

    def l1_per_bank():
        try:
            return int(ttnn.get_memory_view(mesh, ttnn.BufferType.L1).total_bytes_allocated_per_bank)
        except Exception:  # noqa: BLE001
            return None

    if switched:
        # W int3 (2026-10-23): a second backend in one serve process starts above the first one's leftover L1 (the
        # 8K -> 131K mock: +181 KB per bank -> "static CBs clash with L1 buffers" in the 131K build, which fits alone);
        # clear_program_cache (opt-in) drops the program cache's references to the old backend's tensors
        l1a = l1_per_bank()
        if args.get("clear_program_cache"):
            mesh.disable_and_clear_program_cache()
            mesh.enable_program_cache()
            gc.collect()
        log(f"backend switch in one process: L1 per bank after the old backend's close {l1a}, "
            f"after the program-cache clear {l1_per_bank() if args.get('clear_program_cache') else '-'} "
            "(a static-CB clash in this build may be the old backend's leftovers: run each build in its own session)")
    if key not in ctx:
        t0 = time.time()
        backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=hf_config(), native_t=native)
        gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=ms,
                                stop_token_ids=())  # fmt: skip
        t1 = time.time()
        gen.warmup()
        ctx[key] = dict(backend=backend, gen=gen)
        log(f"backend built {t1 - t0:.1f}s warmup {time.time() - t1:.1f}s dsa_decode={backend.dsa_decode} "
            f"supported_context={backend.supported_context} {backend.trace_report()}")  # fmt: skip
    backend, gen = ctx[key]["backend"], ctx[key]["gen"]
    if args.get("warm_only"):  # build + warmup (+ captures) only: the 2x4 full-model mock / a DRAM census at capacity
        rep = backend.trace_report()
        log("WARM_ONLY", json.dumps(dict(layers=len(ids), B=B, max_seq_len=ms, report=rep), default=str))
        RESULT = dict(ok=True, warm_only=True, dram_gib_per_chip=rep.get("dram_gib_per_chip"))
    else:
        c0 = dict(gen.counters)
        r0 = dict(backend.trace_report())
        tpp = getattr(backend, "tpp", None)
        p0 = dict(tpp.counters) if tpp is not None else {}
        t2 = time.time()
        per_entry = run_teacher_forcing(model_dir=MODEL_DIR, reference_path=Path(ref), mesh_device=mesh,
                                        build_kwargs=dict(generator=gen))  # fmt: skip
        wall = time.time() - t2
        c1 = dict(gen.counters)
        d = {k: c1.get(k, 0) - c0.get(k, 0) for k in c1}
        r1 = backend.trace_report()
        s = per_entry[0]
        traced = d.get("eager_decode_steps", 0) == 0 and r1.get("decode_eager", 0) == r0.get("decode_eager", 0) and \
            d.get("decode_trace_replays", 0) > 0  # fmt: skip
        names = [f"L{e['L']}" for e in backend.seq]
        tpp = getattr(backend, "tpp", None)  # the served TP prefill's own trace counters (every chunk a replay)
        pd = {k: int(tpp.counters.get(k, 0)) - int(p0.get(k, 0)) for k in ("replays", "eager", "captures")} if tpp else None
        ok = traced and (pd is None or pd["eager"] == 0) and s["top5"] >= 0.98 and s["top100"] >= 1.0
        r = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 test=os.environ.get("GLM53_ROW_PREFIX", "") + "dsa_stack_tf", status="pass" if ok else "fail",  # (d0: host-p300)
                 B=B, tp=tpn, prefix=names, reference=ref, max_seq_len=ms, rev=MK.MLA_REV, native=native,
                 dsa_decode=backend.dsa_decode, supported_context=backend.supported_context,
                 prefill=os.environ.get("GLM53_STACK_PREFILL", "decode"), top1=s["top1"], top5=s["top5"],
                 top100=s["top100"], matches=[s["matches_top1"], s["matches_top5"], s["matches_top100"], s["total"]],
                 decode_traced_no_eager=traced, prefill_trace_counters=pd, counters_delta=d, wall_s=round(wall, 1),
                 program_cache_entries=r1.get("program_cache_entries"),
                 program_cache_new=(int(r1.get("program_cache_entries", 0)) - int(r0.get("program_cache_entries", 0))),
                 dram_gib_per_chip=r1.get("dram_gib_per_chip"),
                 us_per_token=round(1e6 / s["decode_t/s/u"], 1) if s.get("decode_t/s/u") else None,
                 watcher=bool(os.environ.get("TT_METAL_WATCHER")), knob_env=ENV, tag=tag,
                 note="prefix model past 2,051 rows: scores the DSA decode numerics + the stack plumbing, not the model")  # fmt: skip
        if args.get("dump_pools") and os.environ.get("GLM53_MOCK") != "1":
            # Stage 16 (n29h7s v1, GLM53_DSA_PAGED_POOLS A/B): slot 0's pooled keys of every sparse_dsa layer in GLOBAL pool
            # order, from either store layout -- dense (per-slot unit pages: unit u on chip u % C, local unit u // C, its 32
            # pools in rows 0-31) or paged (page = the local latent block: block k of the slot = physical global block
            # g = row[k] on chip g % C, page g // C, its 16 pools in rows 0-15) -> torch.save; a dense run and a paged run
            # of the same prompt must give BITWISE equal dumps
            import torch  # noqa: PLC0415
            from readiness_check.schema import load_reference  # noqa: PLC0415

            kvb = gen._own_binding().kv_cache
            prow = backend.tpp.page_row(0).to(torch.int64).reshape(-1)
            n_tok = int(per_entry[0]["total"]) + len(load_reference(Path(ref)).entries[0].prompt_tokens.reshape(-1))
            dump = {}
            for i, st in kvb.extra.get("dsa", {}).items():
                chips = [ttnn.to_torch(t_).float() for t_ in ttnn.get_device_tensors(st["pools"])]
                C_ = len(chips)
                n_pools = n_tok // 4
                out = torch.zeros(n_pools, 128)
                if st.get("paged", False):
                    for k in range(-(-n_pools // 16)):
                        g = int(prow[k])
                        rows_ = chips[g % C_][g // C_, 0, :16]
                        n_ = min(16, n_pools - 16 * k)
                        out[16 * k : 16 * k + n_] = rows_[:n_]
                else:
                    for u in range(-(-n_pools // 32)):
                        rows_ = chips[u % C_][u // C_, 0, :32]
                        n_ = min(32, n_pools - 32 * u)
                        out[32 * u : 32 * u + n_] = rows_[:n_]
                dump[int(backend.seq[i]["L"])] = out
            torch.save(dict(pools=dump, n_tok=n_tok, paged=any(st.get("paged", False) for st in kvb.extra.get("dsa", {}).values()),
                            predicted=None), str(args["dump_pools"]))  # fmt: skip
            r.update(dump_pools=str(args["dump_pools"]), dump_layers=sorted(dump), dump_pools_n=n_tok // 4)
        if os.environ.get("GLM53_MOCK") == "1":
            r["status"] = "record"
        log("ROW", json.dumps(r, default=str))
        if args.get("record") and ev:
            os.makedirs(ev, exist_ok=True)  # (rev 60: a new evidence dir; the 6h watched row was lost to a missing one)
            with open(os.path.join(ev, "pcc.jsonl"), "a") as f:
                f.write(json.dumps(r, default=str) + "\n")
        RESULT = dict(ok=ok or r["status"] == "record", top1=s["top1"], top5=s["top5"], top100=s["top100"])
        if r["status"] == "fail":
            raise RuntimeError(f"dsa_stack_tf: top5 {s['top5']} top100 {s['top100']} traced {traced}")
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
