// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Batched router tail (PMK P12, GLM53_PMK_RTAIL_BATCH=1; batch_dm.cpp is the data movement) — compute: the member's 32
// token rows in their NATURAL tiles, tile j = experts 32 j .. 32 j + 31 of the 32 rows (j < SCORE_TILES: the padded
// experts 288 .. 319 are never scanned), against bias tile j (bias[32 j + c] replicated over the 32 rows): per tile the
// SAME SFPU sequence as compute.cpp's per-row packed tile -- scores = sigmoid(logits) (fp32, accurate), choice = scores
// + bias (fp32 SFPU add), scores rounded to bf16 by the SFPU typecast. Every op is elementwise, so element (t, e) gets
// the same bits in either layout. Out: SCORE_TILES fp32 choice tiles + SCORE_TILES bf16 score tiles, one tile per push.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/router_tail_common.hpp"

using namespace rtail;

#if RTAIL_KSPLIT != 1
#error "the batched router tail: one K chunk (the PMK P12 logits)"
#endif

void kernel_main() {
    const uint32_t arg = get_arg_val<uint32_t>(ph_rt0 + 0);
#if defined(MK_PHASED)
    if (arg == ROW_SKIP) {
        return;  // megakernel: an owner without a token row
    }
#endif
    compute_kernel_hw_startup(CB_LOG, CB_CHOICE);
    CircularBuffer cb_log(CB_LOG), cb_bias(CB_BIAS), cb_choice(CB_CHOICE), cb_scores(CB_SCORES);
    for (uint32_t j = 0; j < SCORE_TILES; ++j) {
        cb_log.wait_front(1);
        cb_bias.wait_front(1);
        cb_choice.reserve_back(1);
        cb_scores.reserve_back(1);
        tile_regs_acquire();
        copy_init(CB_LOG);
        copy_tile(CB_LOG, 0, 0);
        sigmoid_tile_init();
        sigmoid_tile(0);  // scores
        copy_dest_values_init();
        copy_dest_values(0, 1);
        copy_init(CB_BIAS);
        copy_tile(CB_BIAS, 0, 2);
        add_binary_tile_init();
        add_binary_tile(1, 2, 1);  // choice
        typecast_tile_init<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>();
        typecast_tile<static_cast<uint32_t>(DataFormat::Float32), static_cast<uint32_t>(DataFormat::Float16_b)>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(CB_CHOICE);
        pack_tile(1, CB_CHOICE);
        pack_reconfig_data_format(CB_SCORES);
        pack_tile(0, CB_SCORES);
        tile_regs_release();
        cb_choice.push_back(1);
        cb_scores.push_back(1);
        cb_log.pop_front(1);
        cb_bias.pop_front(1);
    }
}
