# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mock-cluster rehearsal of the Stage 12 TP image restore in the vLLM server process (host-only; GLM53_BACKEND_MODULE
for a mock serve). build_model_backend runs the PRODUCTION restore: tt/generator.py restore_tp_backend (the
GLM53_IMAGE_REQUIRED loop, the GLM53_IMAGE_PIN filter, on_fail) -> tt/tp_image.py restore_backend (manifest / chip
checks, the replay of tt/stack_model.py's construction with the adapter's keywords: every upload at its recorded
address, the tensor map, the live placeholders, the post-replay hashes). Stubbed (_STUBS): the on-device block hashes
(the mock has no DRAM: they report the image's own, so nothing is stale or clobbered -> no pulls) and, in trees whose
tp_image has them, the image-file prefetch and the sidecar writes into the shared image dir (pull sets, fine hashes;
the mock runs with GLM53_IMAGE_FINE_BLOCK=0); a pull of anything raises -> the only image-file reads are the manifest /
map / host_small (~200 MB over NFS), no image block reads, no writes. Pass = this process (the plugin's mesh open,
the adapter's build keywords) constructs the backend exactly as the image's recorder did, so GLM53_WEIGHT_SOURCE=image
restores it on the device. The result -> $GLM53_IMAGE_MOCK_RESULT (JSON); a mismatch raises (the server does not
start)."""

from __future__ import annotations

import json
import os
import time

from models.autoports.zai_org_glm_5_3_flash.tt import generator as G
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM
from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI

serving_fabric_config = SM.serving_fabric_config


def _image_hashes(mesh, man: dict, block_bytes=None) -> dict:  # the mock's "device": DRAM == the image on every chip
    if block_bytes not in (None, man["block_bytes"]):  # fine blocks: the mock runs with GLM53_IMAGE_FINE_BLOCK=0
        raise RuntimeError(f"stack_image_mock: {block_bytes}-byte blocks; set GLM53_IMAGE_FINE_BLOCK=0 for the mock")
    return {int(dev): list(c["block_hashes"]) for dev, c in man["chips"].items()}


class _NoPrefetch:  # tp_image's page-cache prefetch reads image-file blocks over NFS: none on the mock
    def __init__(self, man, block_bytes, groups):
        self.bytes, self.t_done = 0, None

    def start(self, detach=False):
        return self

    def stop(self):
        return {"mock": "no prefetch"}


def _no_pull(mesh, d, man, only, **kw) -> dict:  # nothing is stale on the mock: any block to move is a bug (or a
    if any(v is None or v for v in only.values()):  # full pull: GLM53_IMAGE_RESIDENT=0) -> never read image blocks
        raise RuntimeError(f"stack_image_mock: a pull of {sum(len(v or [0]) for v in only.values())} block sets")
    return {}


# what the mock replaces in tt/tp_image.py (when the tree's tp_image has it): the on-device hashes; the image-file
# prefetch and pulls; the sidecar writes a verified restore leaves in the SHARED image dir (pull sets, fine hashes)
_STUBS = {"device_hashes": _image_hashes, "Prefetcher": _NoPrefetch, "pull": _no_pull,
          "save_pullset": lambda *a, **k: None, "save_fine_hashes": lambda *a, **k: None}  # fmt: skip


def build_model_backend(model_dir, mesh_device, **kw):
    # the required loop for this restore only: never the tensor-loader fallback on the mock (200 GB of reads); the
    # process env stays without it (tt/generator.py refuses GLM53_IMAGE_REQUIRED for a module outside TP_IMAGE_MODULES)
    prev = os.environ.get(G.IMAGE_REQUIRED_ENV)
    os.environ[G.IMAGE_REQUIRED_ENV] = "1"
    wc = kw.get("weight_cache")
    env = {k: v for k, v in sorted(os.environ.items()) if k.startswith(("GLM53_STACK_", "GLM53_TP_"))}
    res = {"pin": os.environ.get(G.IMAGE_PIN_ENV), "root": os.environ.get("GLM53_TP_IMAGE_ROOT", TI.DEFAULT_ROOT),
           "kw": {k: (v if k != "weight_cache" else f"dict({len(v)})") for k, v in kw.items()}, "stack_env": env}  # fmt: skip
    t0 = time.perf_counter()
    real = {k: getattr(TI, k) for k in _STUBS if hasattr(TI, k)}
    for k in real:
        setattr(TI, k, _STUBS[k])
    res["stubbed"] = sorted(real)
    try:
        be = G.restore_tp_backend(lambda: SM.build_model_backend(model_dir, mesh_device, **kw), mesh_device,
                                  signature={"mock": True}, on_fail=wc.clear if isinstance(wc, dict) else None)  # fmt: skip
        info = be.weight_source_info
        res.update(status="pass" if info.get("source") == "image" else "fail", image=info.get("image"),
                   uploads=info.get("replay_uploads"), host_misses=info.get("replay_host_misses"),
                   tried=info.get("tried"), stale_blocks=info.get("stale_blocks"),
                   clobbered_blocks=info.get("clobbered_blocks"), restore_s=info.get("restore_s"))  # fmt: skip
        be.weight_source_info = {**info, "mock": "device block hashes stubbed with the image's (no DRAM on the mock)"}
        return be
    except Exception as e:
        res.update(status="fail", error=f"{type(e).__name__}: {str(e)[:600]}")
        raise
    finally:
        for k, v in real.items():
            setattr(TI, k, v)
        if prev is None:
            os.environ.pop(G.IMAGE_REQUIRED_ENV, None)
        else:
            os.environ[G.IMAGE_REQUIRED_ENV] = prev
        res["wall_s"] = round(time.perf_counter() - t0, 2)
        out = os.environ.get("GLM53_IMAGE_MOCK_RESULT")
        if out:
            with open(out, "w") as f:
                json.dump(res, f, indent=1)
        print(f"[stack_image_mock] {json.dumps(res)[:1200]}", flush=True)
