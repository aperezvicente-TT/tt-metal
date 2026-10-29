# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job: does ttnn.from_torch(device=mesh) WITHOUT a mesh_mapper replicate on a multi-device mesh (7a's phase classes
create their constants that way)? And ttnn.zeros / ttnn.empty on the mesh."""
import torch

import ttnn

mesh = MESH  # noqa: F821
x = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
outs = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]
z = ttnn.zeros((1, 1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
zs = [float(ttnn.to_torch(d).abs().sum()) for d in ttnn.get_device_tensors(z)]
RESULT = dict(n_dev=len(outs), replicated=all(torch.equal(o.reshape(x.shape), x) for o in outs), zeros=zs,
              addr=t.buffer_address())
print("PROBE_MAPPER", RESULT, flush=True)
