// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Page-DMA protocol: what the host and the persistent page-DMA kernel (kernels/page_dma.cpp) agree on. Shared by the
// RISC-V kernel, the host driver and the host emulator, so it is plain C++ with fixed-width fields only.
//
// One page-DMA core owns one L1 window (PDMA_WINDOW_BYTES at a fixed L1 address, the same on every core and chip):
//
//   +0      PdmaCtrl         64 B   host writes tail / stop; the kernel writes magic, head, done, err, heartbeat
//   +64     PdmaTensor[32]  512 B   the device buffers a descriptor may name (host writes before launch)
//   +576    PdmaSlab[64]    512 B   the pinned host slabs (1 GiB hugepages) a descriptor may name
//   +1088   (pad to 2048)
//   +2048   PdmaDesc[ring]          ring of descriptors, ring = PDMA_RING_ENTRIES
//   +...    staging                 2 x stage bytes (double buffer), 64 B aligned
//
// A descriptor moves n_tiles consecutive pages ("tiles") of one interleaved DRAM buffer to or from one contiguous
// byte run of one host slab: LOAD = host -> DRAM, SAVE = DRAM -> host. Page i of an interleaved buffer lives in bank
// i % banks at bank offset base + (i / banks) * tile_bytes (the allocator's interleaved rule). Everything the kernel
// touches is bounds-checked against the two tables, so a bad descriptor cannot write outside a registered device
// buffer or outside a pinned slab: an out-of-range descriptor sets err and is skipped (done still advances).
//
// Ordering: the host writes descriptor bodies, then (after a store fence) tail. The kernel completes descriptors in
// ring order; `done` counts descriptors whose data has landed (DRAM writes acked for LOAD, PCIe writes acked for SAVE)
// and is published to L1 and, if a completion slot is configured, to host memory (the host polls RAM, not the chip).

#pragma once

#include <stdint.h>

#define PDMA_MAGIC 0x50444D41u  // "PDMA"
#define PDMA_VERSION 1u
#define PDMA_TIMING_MARK 0x71AE0003u  // PdmaDone.rsvd[8]: 0x71AE0000 | kernel rev (2: timing rsvd[0..7]; 3: + burst rsvd[9..10])

#define PDMA_MAX_TENSORS 32u
#define PDMA_MAX_SLABS 64u
#define PDMA_SLAB_BYTES (1ull << 30)

#define PDMA_CTRL_OFF 0u
#define PDMA_TENSORS_OFF 64u
#define PDMA_SLABS_OFF (PDMA_TENSORS_OFF + PDMA_MAX_TENSORS * 16u)
#define PDMA_RING_OFF 2048u

enum PdmaOp : uint16_t {
    PDMA_NOP = 0,
    PDMA_LOAD = 1,  // host slab -> DRAM tiles
    PDMA_SAVE = 2,  // DRAM tiles -> host slab
    PDMA_STOP = 3,  // finish everything before it, publish, exit the kernel
};

enum PdmaErr : uint32_t {
    PDMA_OK = 0,
    PDMA_ERR_OP = 1,       // unknown op
    PDMA_ERR_TENSOR = 2,   // tensor index >= n_tensors or tile range outside the buffer
    PDMA_ERR_SLAB = 3,     // slab index >= n_slabs or byte range outside the slab
    PDMA_ERR_ALIGN = 4,    // slab_off not 64-B aligned
    PDMA_ERR_SIZE = 5,     // n_tiles == 0 or tile_bytes does not divide the staging buffer
};

struct PdmaCtrl {          // 64 B
    uint32_t magic;        // kernel: PDMA_MAGIC once the tables are read (the host waits for it after launch)
    uint32_t version;      // kernel: PDMA_VERSION
    uint32_t tail;         // host: number of descriptors written (monotonic; ring slot = n % ring)
    uint32_t head;         // kernel: number of descriptors taken
    uint32_t done;         // kernel: number of descriptors completed (data landed)
    uint32_t err;          // kernel: first error (PdmaErr), sticky
    uint32_t err_index;    // kernel: descriptor number of the first error
    uint32_t n_errors;     // kernel: descriptors skipped
    uint32_t heartbeat;    // kernel: idle polls (liveness)
    uint32_t n_tensors;    // host: entries of the tensor table
    uint32_t n_slabs;      // host: entries of the slab table
    uint32_t ring;         // host: ring entries (power of two)
    uint32_t stage;        // host: staging buffer bytes (one of two)
    uint32_t done_slab;    // host: slab of the host completion slot, or 0xffffffff for none
    uint32_t done_off;     // host: byte offset of the completion slot in that slab (64-B aligned, 64 B per core)
    uint32_t bytes_lo;     // kernel: bytes moved (low 32 bits; statistics)
};

struct PdmaTensor {        // 16 B: an interleaved DRAM buffer
    uint32_t base;         // bank offset of page 0 (the buffer address)
    uint32_t n_tiles;      // pages in the buffer
    uint32_t tile_bytes;   // page size (64-B multiple)
    uint32_t banks;        // number of DRAM banks it is interleaved over
};

struct PdmaSlab {          // 8 B: a pinned host slab as the chip addresses it
    uint32_t lo;
    uint32_t hi;
};

struct PdmaDesc {          // 32 B
    uint16_t op;           // PdmaOp
    uint16_t tensor;       // index into the tensor table
    uint32_t tile0;        // first page of the buffer
    uint32_t n_tiles;      // pages (consecutive)
    uint32_t slab;         // index into the slab table
    uint32_t slab_off;     // byte offset in the slab (64-B aligned)
    uint32_t tag;          // host's id (echoed into the completion slot)
    uint32_t rsvd[2];      // LOAD burst (kernel rev 3): rsvd[1] = block bytes != 0 -> up to k = max(1, rsvd[0]) reads of
                           // one block each in flight (T2 page-in); rsvd[1] == 0 = the double-buffered stage pipeline
};

struct PdmaDone {          // 64 B host completion slot per core (written by the kernel over PCIe)
    uint32_t done;         // == PdmaCtrl.done at the time of the write
    uint32_t last_tag;     // tag of the last completed descriptor
    uint32_t err;
    uint32_t n_errors;
    uint32_t rsvd[12];     // timing kernels (rsvd[8] == PDMA_TIMING_MARK): [0] first-transfer cycles of the last
                           // descriptor, [1] its total cycles, [2] descriptors timed, [3] sum of [0], [4] min, [5] max,
                           // [6] sum of [1], [7] the wall clock at the write, [9] the last descriptor's read-phase
                           // cycles (issue -> barrier, summed over its burst groups), [10] its burst groups
};

#if defined(__cplusplus) && !defined(PDMA_NO_STATIC_ASSERT)
static_assert(sizeof(PdmaCtrl) == 64, "PdmaCtrl");
static_assert(sizeof(PdmaTensor) == 16, "PdmaTensor");
static_assert(sizeof(PdmaSlab) == 8, "PdmaSlab");
static_assert(sizeof(PdmaDesc) == 32, "PdmaDesc");
static_assert(sizeof(PdmaDone) == 64, "PdmaDone");
static_assert(PDMA_SLABS_OFF + PDMA_MAX_SLABS * 8u <= PDMA_RING_OFF, "tables overlap the ring");
#endif

// Validation shared by the kernel and the host driver (the host refuses a descriptor the kernel would skip).
// t = the descriptor's tensor-table entry, or null when d.tensor >= the table size.
static inline uint32_t pdma_check(const PdmaDesc& d, const PdmaTensor* t, uint32_t n_slabs, uint32_t stage) {
    if (d.op != PDMA_LOAD && d.op != PDMA_SAVE) {
        return d.op == PDMA_NOP || d.op == PDMA_STOP ? PDMA_OK : PDMA_ERR_OP;
    }
    if (t == 0) {
        return PDMA_ERR_TENSOR;
    }
    if (d.n_tiles == 0 || t->tile_bytes == 0 || (t->tile_bytes & 63u) != 0 || stage % t->tile_bytes != 0 ||
        t->banks == 0) {
        return PDMA_ERR_SIZE;
    }
    if (d.op == PDMA_LOAD && d.rsvd[1] != 0 &&
        ((d.rsvd[1] & 63u) != 0 || d.rsvd[1] % t->tile_bytes != 0 || d.rsvd[1] > 2u * stage)) {
        return PDMA_ERR_SIZE;  // a burst block is whole tiles, 64-B aligned, and fits the two staging slots
    }
    if (d.tile0 >= t->n_tiles || d.n_tiles > t->n_tiles - d.tile0) {
        return PDMA_ERR_TENSOR;
    }
    if ((d.slab_off & 63u) != 0) {
        return PDMA_ERR_ALIGN;
    }
    const uint64_t bytes = (uint64_t)d.n_tiles * t->tile_bytes;
    if (d.slab >= n_slabs || (uint64_t)d.slab_off + bytes > PDMA_SLAB_BYTES) {
        return PDMA_ERR_SLAB;
    }
    return PDMA_OK;
}
