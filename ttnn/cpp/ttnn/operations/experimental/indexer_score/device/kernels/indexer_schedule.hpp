// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "indexer_ring_schedule.hpp"

namespace indexer_schedule {
struct CoreSchedule {
    uint32_t row;
    uint32_t col;
    uint32_t block;
    uint32_t row_group;
    uint32_t band_start;  // contiguous band offset, or ring lane identity
    uint32_t band_count;
};

// Shared core identity layout. Unfused work retains the two-level contiguous
// block/column split; fused work retains the ring-arrival lane distribution.
template <bool Ring>
inline constexpr CoreSchedule for_core(
    uint32_t core_id, uint32_t group_rows, const indexer_ring_schedule::Geometry& geometry) {
    const uint32_t row = core_id / geometry.cols;
    const uint32_t col = core_id % geometry.cols;
    const uint32_t block = row / group_rows;
    uint32_t start = 0, count = 0;
    if constexpr (Ring) {
        start = block + col * geometry.num_blocks;
        count = indexer_ring_schedule::band_count(geometry, start);
    } else {
        const uint32_t per_block = geometry.units_per_shard / geometry.num_blocks;
        const uint32_t extra_blocks = geometry.units_per_shard % geometry.num_blocks;
        const uint32_t block_start = block * per_block + (block < extra_blocks ? block : extra_blocks);
        const uint32_t block_size = per_block + (block < extra_blocks);
        const uint32_t per_col = block_size / geometry.cols;
        const uint32_t extra_cols = block_size % geometry.cols;
        start = block_start + col * per_col + (col < extra_cols ? col : extra_cols);
        count = per_col + (col < extra_cols);
    }
    return {row, col, block, row % group_rows, start, count};
}

// KV-BALANCED split (stage 3 M1, 2026-10-24). On the classic metadata path the causal extent kv_len is derived on
// device, but the static split above deals the units of the whole window, so the columns whose bands lie past kv_len
// idle (at the 1M key and chunk start 524,288, 6 of 11 columns had work). Deal only the units the extent covers,
// clamped to >= one band per (block, column) cell (the resident q / w row rendezvous runs at each core's band 0) and
// <= the window's units. Bands past kv_len stay unassigned (unwritten, as with the static split). Every cell's math is
// unchanged, so the score plane is bitwise the static split's.
inline constexpr CoreSchedule for_core_kv(
    uint32_t core_id,
    uint32_t group_rows,
    indexer_ring_schedule::Geometry geometry,
    uint32_t kv_len_tiles,
    uint32_t k_tiles_per_unit) {
    const uint32_t kv_units = (kv_len_tiles + k_tiles_per_unit - 1) / k_tiles_per_unit;
    const uint32_t cells = geometry.num_blocks * geometry.cols;
    uint32_t units = kv_units < geometry.units_per_shard ? kv_units : geometry.units_per_shard;
    units = units > cells ? units : cells;
    geometry.units_per_shard = units;
    return for_core<false>(core_id, group_rows, geometry);
}
}  // namespace indexer_schedule
