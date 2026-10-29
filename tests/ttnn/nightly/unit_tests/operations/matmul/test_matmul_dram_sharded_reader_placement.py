# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

MM = ttnn._ttnn.operations.matmul
P = MM.DramReaderPlacement
K = 4096
M = 32
PLACEMENTS = [P.NOC_HOPS, P.ROW_PLUS_X, P.MESH_ROW_PLUS_X]


def _grid(cores):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cores - 1, 0))})


@pytest.mark.parametrize("placement", PLACEMENTS, ids=lambda p: p.name)
@pytest.mark.parametrize("readers", [1, 2])
def test_reader_assignments(device, readers, placement):
    if readers > 1 and not is_blackhole():
        pytest.skip("Multiple DRAM-sharded matmul workers per bank are currently Blackhole-only")
    excluded = _grid(8)
    readers_list = MM.dram_bank_reader_assignments(
        device,
        num_workers_per_dram_bank=readers,
        secondary_reader_excluded_cores=excluded,
        dram_reader_placement=placement,
    )
    banks = device.dram_grid_size().x
    assert len(readers_list) == banks * readers
    cores = [(int(c.x), int(c.y)) for c, _, _ in readers_list]
    assert len(set(cores)) == len(cores)
    grid = device.compute_with_storage_grid_size()
    assert all(x < grid.x and y < grid.y for x, y in cores)
    optimal = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, ttnn.MeshCoordinate(0, 0))
    primaries = {}
    for core, bank, index in readers_list:
        if index == 0:
            primaries[int(bank)] = (int(core.x), int(core.y))
            assert primaries[int(bank)] == (int(optimal[int(bank)].x), int(optimal[int(bank)].y))
    if readers == 1 or placement == P.NOC_HOPS:
        return
    for core, bank, index in readers_list:
        if index == 0:
            continue
        px, py = primaries[int(bank)]
        row = [
            (px + dx, py)
            for dx in range(1, 5)
            if px + dx < grid.x and not excluded.contains(ttnn.CoreCoord(px + dx, py))
        ]
        if row:
            assert (int(core.x), int(core.y)) in row
        assert not excluded.contains(core)


def test_default_placement_is_noc_hops():
    config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(in0_block_w=4, per_core_M=1, per_core_N=16)
    assert config.dram_reader_placement == P.NOC_HOPS
    explicit = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=4, per_core_M=1, per_core_N=16, num_workers_per_dram_bank=2, dram_reader_placement=P.ROW_PLUS_X
    )
    assert explicit.dram_reader_placement == P.ROW_PLUS_X
    restored = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig.from_json(explicit.to_json())
    assert restored.dram_reader_placement == P.ROW_PLUS_X


@pytest.mark.parametrize("n", [4096, 8192, 12288])
@pytest.mark.parametrize(
    "dtype,fidelity",
    [(ttnn.bfloat8_b, ttnn.MathFidelity.HiFi2), (ttnn.bfloat4_b, ttnn.MathFidelity.LoFi)],
    ids=["bfp8", "bfp4"],
)
def test_two_readers_bitwise_equal_one_reader(device, n, dtype, fidelity):
    if not is_blackhole():
        pytest.skip("Multiple DRAM-sharded matmul workers per bank are currently Blackhole-only")
    torch.manual_seed(n)
    banks = device.dram_grid_size().x
    cores = 8
    act = torch.randn(1, 1, M, K).bfloat16()
    w = (torch.randn(1, 1, K, n) * 0.02).bfloat16()
    act_t = ttnn.from_torch(
        act,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.create_sharded_memory_config(
            (M, K // cores),
            ttnn.CoreGrid(x=cores, y=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
        (K, n // banks),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    w_t = ttnn.from_torch(
        w,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard),
    )
    compute = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    outputs = {}
    for readers, placement in [(1, P.NOC_HOPS)] + [(2, p) for p in PLACEMENTS]:
        config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=4,
            per_core_M=1,
            per_core_N=n // (32 * cores),
            num_workers_per_dram_bank=readers,
            dram_reader_placement=placement,
        )
        out = ttnn.linear(
            act_t,
            w_t,
            program_config=config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
        )
        outputs[(readers, placement.name)] = ttnn.to_torch(out)
    reference = outputs[(1, "NOC_HOPS")]
    reference_fp32 = act.float().reshape(M, K) @ ttnn.to_torch(w_t).float().reshape(K, n)
    pcc = torch.corrcoef(torch.stack([reference.float().flatten(), reference_fp32.flatten()]))[0, 1]
    assert pcc > (0.999 if dtype == ttnn.bfloat8_b else 0.99)
    for key, out in outputs.items():
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), key


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_mesh_placement_on_two_dies(mesh_device):
    if not is_blackhole() or mesh_device.get_num_devices() != 2:
        pytest.skip("needs a 1x2 Blackhole mesh")
    hops = ttnn._ttnn.multi_device.experimental.get_worker_noc_hop_distance
    coords = [ttnn.MeshCoordinate(0, c) for c in range(2)]
    optimal = [mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0, c) for c in coords]

    def worst(readers):
        return max(
            hops(mesh_device, c, optimal[i][int(b)], core, ttnn.NOC.NOC_0)
            for core, b, j in readers
            if j == 0
            for i, c in enumerate(coords)
        )

    shared = MM.dram_bank_reader_assignments(mesh_device, num_workers_per_dram_bank=2, dram_reader_placement=P.NOC_HOPS)
    mesh = MM.dram_bank_reader_assignments(
        mesh_device, num_workers_per_dram_bank=2, dram_reader_placement=P.MESH_ROW_PLUS_X
    )
    assert len(mesh) == 2 * mesh_device.dram_grid_size().x
    assert len({(int(c.x), int(c.y)) for c, _, _ in mesh}) == len(mesh)
    assert worst(mesh) <= worst(shared)
    for core, bank, index in mesh:
        if index == 0:
            assert any(
                (int(core.x), int(core.y)) == (int(o[int(bank)].x), int(o[int(bank)].y)) for o in optimal
            ), "a mesh-aware primary is one of the devices' optimal readers"


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_mesh_two_readers_bitwise_equal_one_reader(mesh_device):
    if not is_blackhole() or mesh_device.get_num_devices() != 2:
        pytest.skip("needs a 1x2 Blackhole mesh")
    torch.manual_seed(7)
    n, cores, banks = 8192, 8, mesh_device.dram_grid_size().x
    act = torch.randn(1, 1, M, K).bfloat16()
    w = (torch.randn(1, 1, K, n) * 0.02).bfloat16()
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    act_t = ttnn.from_torch(
        act,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate,
        memory_config=ttnn.create_sharded_memory_config(
            (M, K // cores),
            ttnn.CoreGrid(x=cores, y=1),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        ),
    )
    shard = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))}),
        (K, n // banks),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    w_t = ttnn.from_torch(
        w,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=replicate,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard),
    )
    compute = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    outputs = {}
    for readers, placement in [(1, P.NOC_HOPS), (1, P.MESH_ROW_PLUS_X)] + [(2, p) for p in PLACEMENTS]:
        config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=4,
            per_core_M=1,
            per_core_N=n // (32 * cores),
            num_workers_per_dram_bank=readers,
            dram_reader_placement=placement,
        )
        out = ttnn.linear(
            act_t,
            w_t,
            program_config=config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
        )
        outputs[(readers, placement.name)] = ttnn.to_torch(
            out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )
    reference = outputs[(1, "NOC_HOPS")]
    assert torch.equal(reference[0].view(torch.int16), reference[1].view(torch.int16))
    for key, out in outputs.items():
        assert torch.equal(out.view(torch.int16), reference.view(torch.int16)), key
    # P4 review F3: the cached MESH_ROW_PLUS_X two-reader program executed again on the real mesh with fresh output
    # buffers (program-cache hits, no new program) still matches the one-reader output bit for bit on both devices.
    config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=4,
        per_core_M=1,
        per_core_N=n // (32 * cores),
        num_workers_per_dram_bank=2,
        dram_reader_placement=P.MESH_ROW_PLUS_X,
    )
    entries = mesh_device.num_program_cache_entries()
    for _ in range(2):
        out = ttnn.linear(
            act_t,
            w_t,
            program_config=config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute,
        )
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        assert torch.equal(got.view(torch.int16), reference.view(torch.int16))
    assert mesh_device.num_program_cache_entries() == entries, "MESH_ROW_PLUS_X re-run compiled a new program"
