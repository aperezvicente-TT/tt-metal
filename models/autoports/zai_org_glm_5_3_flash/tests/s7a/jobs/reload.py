# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job: pick up edited host code inside the long-lived device process (no close / reopen): reload the given
modules in order (dependencies first) and, with ARGS["clear_program_cache"], drop the in-memory program cache so
programs whose descriptor did not change but whose kernel sources did are rebuilt (the JIT's own dependency tracking
rebuilds the binaries). Globals from serve.py: MESH, ARGS, CTX; sets RESULT."""

import importlib
import sys

PKG = "models.autoports.zai_org_glm_5_3_flash."
DEFAULT = ["tt.mk_readers", "tt.mhc_pre", "tt.mhc_post", "tt.kda_pre", "tt.kda_rec", "tt.mk_dense", "tt.mk_ffn",
           "tt.mk_layer", "tests.s7a.traced_harness", "tests.s7a.dev_kda_pre", "tests.s7a.dev_mk_a",
           "tests.s7a.dev_mk_ffn", "tt.mk_experts", "tt.mk_moe", "tests.s7a.dev_moe_parts"]  # fmt: skip
done = []
for m in ARGS.get("modules", DEFAULT):  # noqa: F821 -- serve.py global
    name = PKG + m
    if name in sys.modules:
        importlib.reload(sys.modules[name])
        done.append(m)
if ARGS.get(
    "clear_kernel_cache"
):  # noqa: F821 -- every FILE_PATH kernel rebuilt (in-process binaries keyed by path + args)
    import ttnn

    ttnn.device.ClearKernelCache()
    done.append("kernel_cache cleared")
if ARGS.get("clear_program_cache"):  # noqa: F821
    MESH.disable_and_clear_program_cache()  # noqa: F821
    MESH.enable_program_cache()  # noqa: F821
    done.append("program_cache cleared")
RESULT = None if done is not None else 1
print(f"RELOADED {done}", flush=True)
