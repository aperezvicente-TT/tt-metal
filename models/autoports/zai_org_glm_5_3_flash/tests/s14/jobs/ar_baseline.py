# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): the dispatched-CCL baseline the M2 fabric phase replaces — ttnn.all_reduce of a [1,1,B,4096] bf16
partial (different per chip) over the opened mesh, TRACED (one compile launch, then a 1-op trace replayed blocking and
a K-op trace), B = 1 / 8 / 32. Saves inputs + outputs (host fp32 sum reference) for the device-vs-device check of the
in-program phase. Rows -> perf.jsonl / pcc.jsonl."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s14common as C  # noqa: E402

mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
ndev = mesh.get_num_devices()
mgd = CTX["mgd"]  # noqa: F821
save_dir = os.path.join(os.environ.get("GLM53_WORK", "/opt/glm53/work-s14"), "ar_ref")
os.makedirs(save_dir, exist_ok=True)
res = {}
for B in args.get("bs", [1, 8, 32]):
    g = torch.Generator().manual_seed(1234 + B)
    x = (torch.randn(ndev, 1, B, 4096, generator=g) * 2.0).to(torch.bfloat16)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                         memory_config=ttnn.L1_MEMORY_CONFIG, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
    ref = x.float().sum(0, keepdim=True)
    fn = lambda: ttnn.all_reduce(xt, memory_config=ttnn.L1_MEMORY_CONFIG, topology=ttnn.Topology.Linear)
    tr = C.Traced(ttnn, mesh, fn)
    one = C.stats(tr.replays(50))
    outs = [ttnn.to_torch(t).float().reshape(1, 1, B, 4096) for t in ttnn.get_device_tensors(tr.out)]
    tr.release()
    K = 20

    def kfn():
        o = None
        for _ in range(K):
            o = ttnn.all_reduce(xt, memory_config=ttnn.L1_MEMORY_CONFIG, topology=ttnn.Topology.Linear)
        return o

    trk = C.Traced(ttnn, mesh, kfn, warm=False)
    kops = [v / K for v in trk.replays(10)]
    trk.release()
    same = all(torch.equal(outs[0], o) for o in outs[1:])
    p = C.pcc(outs[0], ref)
    mae = float((outs[0] - ref).abs().max())
    exact_vs_bf16ref = bool(torch.equal(outs[0], ref.to(torch.bfloat16).float()))
    torch.save(dict(x=x, outs=torch.stack(outs), ref=ref), os.path.join(save_dir, f"ttnn_all_reduce_{mgd}_B{B}.pt"))
    C.record("perf", dict(test=f"ttnn.all_reduce {mgd} B={B} 1-op trace", status="info", milestone="M2-baseline",
                          traced=True, bytes=B * 4096 * 2, us=one["median_us"], one_op=one, mgd=mgd, ndev=ndev,
                          note="dispatched CCL op (the path M2 replaces); blocking 1-op trace replay incl. launch + sync"))
    C.record("perf", dict(test=f"ttnn.all_reduce {mgd} B={B} K-op trace", status="info", milestone="M2-baseline",
                          traced=True, bytes=B * 4096 * 2, us=C.stats(kops)["median_us"], k=K, mgd=mgd, ndev=ndev,
                          note="per-op mean of a 20-op trace replay (launch amortised)"))
    C.record("pcc", dict(test=f"ttnn.all_reduce {mgd} B={B} vs host fp32 sum", status="info", mgd=mgd, pcc=p,
                         max_abs_err=mae, replicas_identical=same, bitwise_vs_bf16_of_fp32_sum=exact_vs_bf16ref))
    res[B] = dict(one=one["median_us"], kop=C.stats(kops)["median_us"], pcc=p, mae=mae, same=same)
    ttnn.deallocate(xt)
RESULT = res
