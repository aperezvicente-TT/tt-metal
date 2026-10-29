# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (MOCK diagnosis, Stage 14 GLM53_STACK_DSA): where the sparse_dsa layer's MkMla(dsa) program meets the
stack's L1 floor. Builds the StackBackend of ARGS layers with GLM53_STACK_DSA = ARGS dsa, runs the generator warm-up
(the first step; with dsa=1 the MLA launch may raise the static-CB clash), and prints: the MLA program's role cores
(latent / q-norm / index core), the static-CB placement of the captured MkMla ProgramDescriptor on ARGS cores
(tt/cb_guard.placement: tt-metal's allocation order), and every live L1 tensor whose shard grid covers those cores.
ARGS: dsa (1), layers ([2, 3]), cores ([[0, 1], [0, 2]]), max_seq_len (8192)."""

import gc
import importlib
import os

ENV = {"GLM53_STACK_DSA": str(int(ARGS.get("dsa", 1))), **{k: str(v) for k, v in (ARGS.get("env") or {}).items()}}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    import ttnn
    import models.autoports.zai_org_glm_5_3_flash.tt.cb_guard as CBG
    import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
    import models.autoports.zai_org_glm_5_3_flash.tt.mk_mla as MK
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM
    import models.autoports.zai_org_glm_5_3_flash.tt.tp_dsa as TPD
    import models.autoports.zai_org_glm_5_3_flash.tt.tp_mla as TPM
    from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    for m in (MK, TPM, TPD, SM, G):
        importlib.reload(m)
    mesh = MESH  # noqa: F821
    cores = [tuple(int(v) for v in c) for c in ARGS.get("cores", [[0, 1], [0, 2]])]  # noqa: F821
    ids = [int(v) for v in ARGS.get("layers", [2, 3])]  # noqa: F821
    seen = {}
    orig = TPM.TPMla._hooked

    def hooked(self, io, desc, t):
        seen.setdefault("descs", []).append((self, desc, t))
        return orig(self, io, desc, t)

    TPM.TPMla._hooked = hooked
    B, native = int(ARGS.get("b", 1)), bool(ARGS.get("native", False))  # noqa: F821
    backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=hf_config(), native_t=native)
    gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B,
                            max_seq_len=int(ARGS.get("max_seq_len", 8192)), stop_token_ids=())  # noqa: F821  # fmt: skip
    err = None
    try:
        gen.warmup()
    except Exception as e:  # noqa: BLE001
        err = str(e).splitlines()[-1][:300]
    print(f"L1PROBE dsa={ENV['GLM53_STACK_DSA']} warmup_error={err}", flush=True)
    for tpm, desc, t in seen.get("descs", [])[:1]:
        mk = tpm.mk
        roles = dict(lat=mk.lat_core, qn=list(mk.qn_cores), ix=mk.ix_core, lanes=tpm.lanes, root=mk.place["root"])
        print("L1PROBE roles", {k: (str(v) if not isinstance(v, list) else [str(x) for x in v]) for k, v in roles.items()},
              flush=True)  # fmt: skip
        rd = mk.last_readers
        for p, g in zip(rd.phases if hasattr(rd, "phases") else [], rd.geo):
            dt, block = rd.rings[g["ring"]]
            print(f"L1PROBE phase {p.w.role}: kb {g['kb']} nr {g['nr']} ring cb {g['ring']} block {block} "
                  f"slots {rd.ring_slots[g['ring']]}", flush=True)  # fmt: skip
        print(f"L1PROBE readers l1_bytes {rd.l1_bytes()} in0_cap {rd.in0_cap}", flush=True)
        pl = CBG.placement(list(desc.cbs))
        for c in cores:
            rows = [(a, a + s, idx) for a, s, rs, idx in pl
                    if any(r[0] <= c[0] <= r[2] and r[1] <= c[1] <= r[3] for r in rs)]  # fmt: skip
            end = max((b for _, b, _ in rows), default=0)
            print(f"L1PROBE core {c} t={t}: {len(rows)} CBs, region end (relative) {end}: "
                  f"{[(a, b, i) for a, b, i in rows]}", flush=True)  # fmt: skip
    gc.collect()
    for o in gc.get_objects():
        try:
            if not (isinstance(o, ttnn.Tensor) and o.is_allocated() and o.storage_type() == ttnn.StorageType.DEVICE):
                continue
            mc = o.memory_config()
            if mc.buffer_type != ttnn.BufferType.L1 or mc.shard_spec is None:
                continue
            hit = [c for c in cores if any(r.start.x <= c[0] <= r.end.x and r.start.y <= c[1] <= r.end.y
                                           for r in mc.shard_spec.grid.ranges())]  # fmt: skip
            if hit:
                print(f"L1PROBE tensor @{o.buffer_address()} {tuple(o.shape)} {o.dtype} shard {list(mc.shard_spec.shape)} "
                      f"grid {mc.shard_spec.grid} cores {hit}", flush=True)  # fmt: skip
        except Exception:  # noqa: BLE001
            continue
    TPM.TPMla._hooked = orig
    for k in ("backend",):
        backend.close() if hasattr(backend, "close") else None
    RESULT = dict(ok=True, err=err)
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
