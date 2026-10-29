# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, any mesh): a device health probe after an aborted process (e.g. a watcher-assert SIGABRT) -- one
small eltwise op and a matmul on every chip, read back and compared with the host. A healthy chip returns exact values
within a second; a chip left in a bad state shows wrong values or hits the op timeout (the hang watchdog then resets
the group). ARGS: tag."""
import torch

import ttnn

mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
n = int(mesh.get_num_devices())
repl = ttnn.ReplicateTensorToMesh(mesh) if n > 1 else None
a = torch.arange(32 * 64, dtype=torch.float32).reshape(1, 1, 32, 64) / 128.0
b = torch.full((1, 1, 64, 32), 0.5)
ta = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
tb = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=repl)
s = ttnn.add(ta, ta)
m = ttnn.matmul(ta, tb)
ttnn.synchronize_device(mesh)
outs_s = [ttnn.to_torch(t_).float() for t_ in (ttnn.get_device_tensors(s) if n > 1 else [s])]
outs_m = [ttnn.to_torch(t_).float() for t_ in (ttnn.get_device_tensors(m) if n > 1 else [m])]
ref_s = (a.bfloat16().float() * 2).bfloat16().float()
ref_m = a.bfloat16().float() @ b
ok_s = [bool(torch.equal(o, ref_s)) for o in outs_s]
err_m = [float((o - ref_m).abs().max()) for o in outs_m]
ok = all(ok_s) and all(e < 0.5 for e in err_m)
RESULT = dict(test=f"device health probe ({n} chips): add + matmul vs host", status="pass" if ok else "fail",
              add_exact=ok_s, matmul_max_abs_err=err_m, tag=args.get("tag", "health"))  # fmt: skip
print("S16", RESULT, flush=True)
