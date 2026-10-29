# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM53_BACKEND_MODULE for LMCache ladder R1 on the device: the Stage 14b StackBackend (tt/stack_model.py)
unchanged, plus the KV-connector page-path probes (tests/vllm/r1_probe.py) when GLM53_R1_PROBE=1. The EngineCore
imports this module when it builds the model, before the connector moves any page."""

from models.autoports.zai_org_glm_5_3_flash.tests.vllm import r1_probe
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model


def serving_fabric_config() -> dict:
    return stack_model.serving_fabric_config()


def build_model_backend(model_dir, mesh_device, **kw):
    r1_probe.install_from_env()
    return stack_model.build_model_backend(model_dir, mesh_device, **kw)
