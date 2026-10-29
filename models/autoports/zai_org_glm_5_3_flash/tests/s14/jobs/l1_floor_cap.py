# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (mock or device; Stage 14 W / int3, the 131K-capacity static-CB clash): which L1 buffers a StackBackend
holds at a given max_seq_len BEFORE its first decode program, and who allocated them. Builds the backend + generator
(native, ARGS layers / b / env as dsa_stack_tf.py), allocates the KV cache, then runs backend.prepare_prefill with the
ttnn tensor creators wrapped (from_torch / empty / zeros / ones / full / allocate_tensor_on_device / to_device /
to_memory_config: an L1 result records its Python call site) and lists the L1 buffers (ttnn reports.get_buffers, chip
0) that exist after each phase: the allocator view, the per-bank bytes, and per new buffer its recorded call site
(unrecorded = created inside an op's program). No decode program is built.
ARGS: layers ([0..4]), b (8), native (True), max_seq_len (8192), env, top (40)."""

import os
import traceback

ENV = {"GLM53_STACK_DSA": "1", **{k: str(v) for k, v in (ARGS.get("env") or {}).items()}}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    import importlib

    import ttnn

    import models.autoports.zai_org_glm_5_3_flash.tt.generator as G
    import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM
    from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config
    from models.autoports.zai_org_glm_5_3_flash.tt.tokenizer_chat import load_tokenizer

    for m in (SM, G):
        importlib.reload(m)
    mesh, args = MESH, ARGS  # noqa: F821
    ids = [int(v) for v in args.get("layers", [0, 1, 2, 3, 4])]
    B, ms, top = int(args.get("b", 8)), int(args.get("max_seq_len", 8192)), int(args.get("top", 40))

    def l1_bufs():
        out = {}
        for b_ in ttnn._ttnn.reports.get_buffers(mesh):
            if "L1" in str(b_.buffer_type) and int(b_.device_id) == 0:
                out[int(b_.address)] = (int(b_.max_size_per_bank), str(b_.buffer_layout).split(".")[-1])
        return out

    def view(tag):
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        bufs = l1_bufs()
        print(f"L1FLOOR {tag}: ms={ms} alloc_per_bank={int(mv.total_bytes_allocated_per_bank)} "
              f"largest_free={int(mv.largest_contiguous_bytes_free_per_bank)} buffers={len(bufs)} "
              f"lowest={min(bufs) if bufs else None}", flush=True)  # fmt: skip
        return bufs

    sites = {}
    names = ("from_torch", "empty", "zeros", "ones", "full", "allocate_tensor_on_device", "to_device", "to_memory_config")
    orig = {n: getattr(ttnn, n) for n in names if hasattr(ttnn, n)}

    def wrap(n, f):
        def g(*a, **k):
            r = f(*a, **k)
            try:
                if "L1" in str(r.memory_config().buffer_type):
                    st = [fr for fr in traceback.extract_stack()[:-1] if "zai_org_glm_5_3_flash" in fr.filename][-3:]
                    sites[int(r.buffer_address())] = f"{n} {list(r.shape)} " + " <- ".join(
                        f"{os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}" for fr in reversed(st))
            except Exception:  # noqa: BLE001
                pass
            return r

        return g

    for n, f in orig.items():
        setattr(ttnn, n, wrap(n, f))
    try:
        backend = SM.build_model_backend(None, mesh, max_batch=B, layers=ids, hf_config=hf_config(),
                                         native_t=bool(args.get("native", True)))  # fmt: skip
        gen = G.build_generator(None, mesh, backend=backend, tokenizer=load_tokenizer(), max_batch=B, max_seq_len=ms,
                                stop_token_ids=())  # fmt: skip
        b0 = view("built")
        bnd = gen._own_binding()
        b1 = view("kv_allocated")
        backend.prepare_prefill(bnd.kv_cache)
        b2 = view("prefill_prepared")
    finally:
        for n, f in orig.items():
            setattr(ttnn, n, f)
    for tag, lo, hi in (("kv_cache", b0, b1), ("prepare_prefill", b1, b2)):
        new = {a: v for a, v in hi.items() if a not in lo}
        print(f"L1FLOOR new in {tag}: {len(new)} buffers, {sum(v[0] for v in new.values())} B per bank", flush=True)
        for a, (sz, lay) in sorted(new.items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"L1FLOOR   {tag} addr={a} per_bank={sz} {lay} site={sites.get(a, '(op output)')}", flush=True)
    RESULT = dict(ok=True)
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
