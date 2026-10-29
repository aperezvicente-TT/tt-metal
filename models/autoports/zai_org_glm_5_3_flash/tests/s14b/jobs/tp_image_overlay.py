# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; the first job of a process, before any backend is built): Stage 12's tp_image.py (git blob
8e04b366d9: GLM53_IMAGE_EMULATE_REAL opt-in, pull-first default) replaces the tree's tt/tp_image module for this process
(program 2026-10-21 06:21Z: open 2 = the true resident restart + Stage 12's fingerprint with EMULATE_REAL; Stage 12
mocked the same overlay on tt-metal-s14bx 6379da920d2). Python only: kernels and every other module stay the tree's.
ARGS: path (the module file), env (knobs to set, e.g. GLM53_IMAGE_EMULATE_REAL=1)."""
import importlib.util
import os
import sys

args = ARGS  # noqa: F821
path = args.get("path", "/opt/glm53/work-s14b/overlay/tp_image_8e04b366d9.py")
name = "models.autoports.zai_org_glm_5_3_flash.tt.tp_image"
assert name not in sys.modules or getattr(sys.modules[name], "__file__", "") == path, \
    ("tp_image already imported from the tree in this process", sys.modules[name].__file__)  # fmt: skip
os.environ.update({k: str(v) for k, v in (args.get("env") or {}).items()})
spec = importlib.util.spec_from_file_location(name, path)
mod = importlib.util.module_from_spec(spec)
sys.modules[name] = mod
spec.loader.exec_module(mod)
import models.autoports.zai_org_glm_5_3_flash.tt as pkg  # noqa: E402

pkg.tp_image = mod
from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI  # noqa: E402

assert TI is mod and TI.__file__ == path, TI.__file__
print("TP-IMAGE-OVERLAY", TI.__file__, "EMULATE_REAL", os.environ.get("GLM53_IMAGE_EMULATE_REAL"),
      "build_or_restore", hasattr(TI, "build_or_restore"), flush=True)  # fmt: skip
RESULT = dict(ok=True, module=TI.__file__, emulate_real=os.environ.get("GLM53_IMAGE_EMULATE_REAL"))
