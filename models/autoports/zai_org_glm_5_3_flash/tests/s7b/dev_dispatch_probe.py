# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# mock-only: build MkMla once and print what one launch dispatches (kernel groups, ranges, rt words)
import ttnn
from models.autoports.zai_org_glm_5_3_flash.tests.s7b.mla_common import MlaLayer
from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK

mesh = ttnn.open_device(device_id=0, worker_l1_size=1264512)
L = MlaLayer(mesh, None, 3)
c = L.case(1, tag="d")
mk = MK.MkMla(mesh, L.a_args, L.attn)
mk(c["x"], c["cache"].latent, c["pt"], c["cur"], t=1)
print("DISPATCH_STATS", mk.last_dispatch)
ks = []
ttnn.close_device(mesh)
