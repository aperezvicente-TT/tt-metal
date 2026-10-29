# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mock-cluster build check of the fused mhc_pre program (Stage 7a): every mode compiles (JIT kernels), the CBs fit
next to the decode path's L1 tensors, and the descriptors are accepted. Run under E/scripts/mock.sh (no hardware;
readback is garbage)."""

import sys
import time

import torch

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import mhc_pre as M


def main():
    mesh = ttnn.open_device(device_id=0)
    print("device type", type(mesh).__name__, flush=True)
    try:
        g = mesh.compute_with_storage_grid_size()
        print("grid", g.x, g.y, flush=True)
        torch.manual_seed(0)
        fn_t = (torch.randn(16384, 24) * 0.02).to(torch.bfloat16)
        scale = torch.tensor([0.5] * 4 + [0.7] * 4 + [1.1] * 16)
        bias = torch.randn(24) * 0.1
        w = (1 + 0.1 * torch.randn(4096)).to(torch.bfloat16)
        prog = M.MhcPre(mesh, fn_t, scale, bias, w, hc_eps=1e-6, x_eps=1e-6, h_eps=1e-5, iters=20)
        cases = [
            ("T1 flat DRAM -> stacked out, PC stacked", 1, M.X_FLAT, M.XOUT_STACKED, M.PC_STACKED, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 1, 16384)),
            ("T1 stacked L1, PC stacked", 1, M.X_STACKED, M.XOUT_NONE, M.PC_STACKED, ttnn.L1_MEMORY_CONFIG, (1, 1, 4, 4096)),
            ("T4 stacked L1, PC stacked", 4, M.X_STACKED, M.XOUT_NONE, M.PC_STACKED, ttnn.L1_MEMORY_CONFIG, (4, 1, 4, 4096)),
            ("T8 flat DRAM -> flat out, PC sm", 8, M.X_FLAT, M.XOUT_FLAT, M.PC_STREAM_MAJOR, ttnn.DRAM_MEMORY_CONFIG, (1, 1, 8, 16384)),
            ("T32 sm L1, PC sm", 32, M.X_FLAT, M.XOUT_NONE, M.PC_STREAM_MAJOR, ttnn.L1_MEMORY_CONFIG, (1, 4, 32, 4096)),
        ]
        for name, t, xm, xo, pc, mem, shape in cases:
            x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=mem)
            for debug, probe in ((False, False), (True, False), (True, True)):
                t0 = time.time()
                out = prog(x, t=t, x_mode=xm, xout_mode=xo, pc_mode=pc, debug=debug, probe=probe)
                ttnn.synchronize_device(mesh)
                print(f"OK {name} debug={debug} probe={probe} hn={tuple(out.hn.shape)} post={tuple(out.post.shape)} "
                      f"comb={tuple(out.comb.shape)} xout={None if out.x_out is None else tuple(out.x_out.shape)} "
                      f"{time.time() - t0:.1f}s", flush=True)
        print("MOCK_MHC_PRE_OK")
    finally:
        ttnn.close_device(mesh)


if __name__ == "__main__":
    sys.exit(main())
