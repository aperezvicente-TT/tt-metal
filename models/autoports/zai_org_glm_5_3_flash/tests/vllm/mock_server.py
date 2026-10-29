# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`vllm serve` over the TT plugin with a fake mesh and 06a's generator over the CPU MockModel (09a host-only HTTP checks).

Usage: python -m models.autoports.zai_org_glm_5_3_flash.tests.vllm.mock_server <vllm serve args...>
The engine core runs in its own (forked) process, as in production, so the plugin registration and the adapter
are resolved there; the fake mesh replaces only TTWorker's mesh open/close. Never opens a TT device.
"""

import os
import sys

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import engine_harness as H

if __name__ == "__main__":
    os.environ.setdefault("GLM53_VLLM_GENERATOR_FACTORY", H.MOCK_FACTORY)  # (a test may name another mock factory)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")  # the engine core inherits the fake mesh
    with H.fake_tt_mesh():
        from vllm.entrypoints.cli.main import main

        sys.argv = ["vllm", "serve", *sys.argv[1:]]
        main()
