// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The page-DMA kernel's logic, written against an IO policy so the same code runs on a Tensix data-movement core
// (kernels/page_dma.cpp: NoC + PCIe) and in the host emulator (memcpy on modelled L1 / DRAM / host memory, with
// transfers that land only at the barrier that guarantees them, so a slot reused too early shows up as wrong bytes).
//
// IO policy (all transfers asynchronous; sizes are multiples of 64 B):
//   uint32_t ld(uint32_t l1)                      volatile L1 load (re-reads memory the host writes)
//   void st(uint32_t l1, uint32_t v)              L1 store
//   void host_read(uint64_t src, uint32_t l1, uint32_t n)      PCIe read of pinned host memory into L1
//   void host_write(uint32_t l1, uint64_t dst, uint32_t n)     PCIe write of L1 into pinned host memory
//   void dram_read(uint32_t bank, uint32_t addr, uint32_t l1, uint32_t n)
//   void dram_write(uint32_t l1, uint32_t bank, uint32_t addr, uint32_t n)
//   void read_barrier()                           every issued read has landed in L1
//   void writes_flushed()                         every issued write has left L1 (its source may be reused)
//   void write_barrier()                          every issued write has landed (acked)
//   void end_pcie()                               drop PCIe routing state before the kernel returns
//   void idle()                                   one empty poll (lets the emulator yield)
//   uint32_t clock()                              32-bit cycle counter (device: the RISC-V wall clock; timing only)
//
// Per descriptor the pipeline keeps two staging slots: the transfer INTO slot k + 1 is issued before the transfer OUT
// of slot k, so the PCIe leg of one block overlaps the DRAM leg of the other. A slot is refilled only after
// writes_flushed (its previous outbound writes left L1) and drained only after read_barrier.

#pragma once

#include "page_dma_proto.h"

template <class IO>
struct PdmaKernel {
    IO& io;
    uint32_t win;     // L1 window base
    uint32_t ring = 0;    // ring entries
    uint32_t stage = 0;   // one staging slot, bytes
    uint32_t slot0 = 0;   // L1 address of slot 0 (slot 1 follows)
    uint32_t n_tensors = 0, n_slabs = 0;
    uint32_t done = 0, n_errors = 0, err = PDMA_OK, err_index = 0, last_tag = 0;
    uint32_t bytes = 0;
    // per-descriptor timing (device cycles, published in PdmaDone.rsvd; PDMA_TIMING_MARK says this kernel has it):
    // first = issue of the first inbound transfer -> its read barrier (LOAD: the PCIe read of the first block; SAVE:
    // the DRAM read), total = issue -> the final write barrier (the data landed)
    uint32_t t_first = 0, t_total = 0, n_timed = 0, sum_first = 0, min_first = 0xffffffffu, max_first = 0;
    uint32_t sum_total = 0;
    uint32_t t_reads = 0, n_groups = 0;  // the last descriptor's read phases (burst LOADs; the stage pipeline: 0)

    void timed(uint32_t first, uint32_t total) {
        t_first = first;
        t_total = total;
        n_timed++;
        sum_first += first;
        sum_total += total;
        min_first = first < min_first ? first : min_first;
        max_first = first > max_first ? first : max_first;
    }

    uint32_t ctrl(uint32_t field_off) const { return win + PDMA_CTRL_OFF + field_off; }

    PdmaTensor tensor(uint32_t i) const {
        const uint32_t a = win + PDMA_TENSORS_OFF + 16u * i;
        return PdmaTensor{io.ld(a), io.ld(a + 4), io.ld(a + 8), io.ld(a + 12)};
    }

    uint64_t slab_addr(uint32_t i) const {
        const uint32_t a = win + PDMA_SLABS_OFF + 8u * i;
        return (static_cast<uint64_t>(io.ld(a + 4)) << 32) | io.ld(a);
    }

    // L1 scratch for the host completion record (inside the table pad, 64-B aligned)
    static constexpr uint32_t DONE_SCRATCH_OFF = PDMA_SLABS_OFF + PDMA_MAX_SLABS * 8u;  // 1088

    // the record lands before the kernel goes back to polling (a PCIe round trip, only when drained / every 16)
    void publish(uint32_t done_slab, uint32_t done_off) {
        io.st(ctrl(offsetof(PdmaCtrl, err)), err);
        io.st(ctrl(offsetof(PdmaCtrl, err_index)), err_index);
        io.st(ctrl(offsetof(PdmaCtrl, n_errors)), n_errors);
        io.st(ctrl(offsetof(PdmaCtrl, bytes_lo)), bytes);
        io.st(ctrl(offsetof(PdmaCtrl, done)), done);
        if (done_slab < n_slabs) {
            const uint32_t s = win + DONE_SCRATCH_OFF;
            io.st(s + 0, done);
            io.st(s + 4, last_tag);
            io.st(s + 8, err);
            io.st(s + 12, n_errors);
            io.st(s + 16, t_first);
            io.st(s + 20, t_total);
            io.st(s + 24, n_timed);
            io.st(s + 28, sum_first);
            io.st(s + 32, min_first);
            io.st(s + 36, max_first);
            io.st(s + 40, sum_total);
            io.st(s + 44, io.clock());
            io.st(s + 48, PDMA_TIMING_MARK);
            io.st(s + 52, t_reads);
            io.st(s + 56, n_groups);
            io.host_write(s, slab_addr(done_slab) + done_off, 64);
            io.write_barrier();
        }
    }

    // LOAD: host [src, src + n_tiles * tile_bytes) -> tiles tile0.. of t
    void load(const PdmaTensor& t, uint64_t src, uint32_t tile0, uint32_t n_tiles) {
        const uint32_t total = n_tiles * t.tile_bytes;
        const uint32_t blocks = (total + stage - 1) / stage;
        const uint32_t c0 = io.clock();
        uint32_t first = 0;
        io.host_read(src, slot0, total < stage ? total : stage);
        for (uint32_t k = 0; k < blocks; k++) {
            const uint32_t off = k * stage;
            const uint32_t n = total - off < stage ? total - off : stage;
            const uint32_t slot = slot0 + (k & 1u) * stage;
            io.read_barrier();  // block k is in its slot
            if (k == 0) {
                first = io.clock() - c0;
            }
            if (k + 1 < blocks) {
                const uint32_t off1 = off + stage;
                const uint32_t n1 = total - off1 < stage ? total - off1 : stage;
                io.writes_flushed();  // the other slot's DRAM writes (block k - 1) left L1
                io.host_read(src + off1, slot0 + ((k + 1) & 1u) * stage, n1);
            }
            uint32_t tile = tile0 + off / t.tile_bytes;
            for (uint32_t b = 0; b < n; b += t.tile_bytes, tile++) {
                io.dram_write(slot + b, tile % t.banks, t.base + (tile / t.banks) * t.tile_bytes, t.tile_bytes);
            }
        }
        io.write_barrier();
        timed(first, io.clock() - c0);
        bytes += total;
    }

    // LOAD burst (T2 page-in): blocks of blk bytes, up to k reads in flight into sub-slots of the two staging slots;
    // a group's reads land (one barrier), then its tile writes go out; the next group refills the sub-slots only after
    // those writes left L1
    void load_burst(const PdmaTensor& t, uint64_t src, uint32_t tile0, uint32_t n_tiles, uint32_t k, uint32_t blk) {
        const uint32_t total = n_tiles * t.tile_bytes;
        const uint32_t blocks = (total + blk - 1) / blk;
        const uint32_t room = (2u * stage) / blk;
        const uint32_t slots = k < room ? k : room;
        const uint32_t c0 = io.clock();
        uint32_t first = 0, reads = 0, groups = 0;
        uint32_t tile = tile0;
        for (uint32_t g = 0; g < blocks; g += slots) {
            const uint32_t gn = blocks - g < slots ? blocks - g : slots;
            io.writes_flushed();  // the previous group's tile writes left the sub-slots
            const uint32_t r0 = io.clock();
            for (uint32_t j = 0; j < gn; j++) {
                const uint32_t off = (g + j) * blk;
                const uint32_t n = total - off < blk ? total - off : blk;
                io.host_read(src + off, slot0 + j * blk, n);
            }
            io.read_barrier();
            const uint32_t r1 = io.clock();
            reads += r1 - r0;
            groups++;
            if (g == 0) {
                first = r1 - c0;
            }
            for (uint32_t j = 0; j < gn; j++) {
                const uint32_t off = (g + j) * blk;
                const uint32_t n = total - off < blk ? total - off : blk;
                for (uint32_t b = 0; b < n; b += t.tile_bytes, tile++) {
                    io.dram_write(slot0 + j * blk + b, tile % t.banks, t.base + (tile / t.banks) * t.tile_bytes,
                                  t.tile_bytes);
                }
            }
        }
        io.write_barrier();
        timed(first, io.clock() - c0);
        t_reads = reads;
        n_groups = groups;
        bytes += total;
    }

    // SAVE: tiles tile0.. of t -> host [dst, dst + n_tiles * tile_bytes)
    void save(const PdmaTensor& t, uint64_t dst, uint32_t tile0, uint32_t n_tiles) {
        const uint32_t total = n_tiles * t.tile_bytes;
        const uint32_t blocks = (total + stage - 1) / stage;
        auto fill = [&](uint32_t k) {
            const uint32_t off = k * stage;
            const uint32_t n = total - off < stage ? total - off : stage;
            const uint32_t slot = slot0 + (k & 1u) * stage;
            uint32_t tile = tile0 + off / t.tile_bytes;
            for (uint32_t b = 0; b < n; b += t.tile_bytes, tile++) {
                io.dram_read(tile % t.banks, t.base + (tile / t.banks) * t.tile_bytes, slot + b, t.tile_bytes);
            }
        };
        const uint32_t c0 = io.clock();
        uint32_t first = 0;
        fill(0);
        for (uint32_t k = 0; k < blocks; k++) {
            const uint32_t off = k * stage;
            const uint32_t n = total - off < stage ? total - off : stage;
            io.read_barrier();  // block k is in its slot
            if (k == 0) {
                first = io.clock() - c0;
            }
            if (k + 1 < blocks) {
                io.writes_flushed();  // the other slot's PCIe writes (block k - 1) left L1
                fill(k + 1);
            }
            io.host_write(slot0 + (k & 1u) * stage, dst + off, n);
        }
        io.write_barrier();
        timed(first, io.clock() - c0);
        bytes += total;
    }

    void run() {
        n_tensors = io.ld(ctrl(offsetof(PdmaCtrl, n_tensors)));
        n_slabs = io.ld(ctrl(offsetof(PdmaCtrl, n_slabs)));
        ring = io.ld(ctrl(offsetof(PdmaCtrl, ring)));
        stage = io.ld(ctrl(offsetof(PdmaCtrl, stage)));
        const uint32_t done_slab = io.ld(ctrl(offsetof(PdmaCtrl, done_slab)));
        const uint32_t done_off = io.ld(ctrl(offsetof(PdmaCtrl, done_off)));
        slot0 = win + PDMA_RING_OFF + ring * static_cast<uint32_t>(sizeof(PdmaDesc));
        if (n_tensors > PDMA_MAX_TENSORS) {
            n_tensors = 0;  // every descriptor then fails the tensor check
        }
        if (n_slabs > PDMA_MAX_SLABS) {
            n_slabs = 0;
        }
        uint32_t head = 0, beat = 0;
        io.st(ctrl(offsetof(PdmaCtrl, head)), 0);
        io.st(ctrl(offsetof(PdmaCtrl, done)), 0);
        io.st(ctrl(offsetof(PdmaCtrl, version)), PDMA_VERSION);
        io.st(ctrl(offsetof(PdmaCtrl, magic)), PDMA_MAGIC);  // last: the host's "kernel is up" signal
        for (;;) {
            // the host's tail is a monotonic count; unsigned difference handles wrap
            if (io.ld(ctrl(offsetof(PdmaCtrl, tail))) - head == 0) {
                if ((++beat & 0xffffu) == 0) {
                    io.st(ctrl(offsetof(PdmaCtrl, heartbeat)), beat);
                }
                io.idle();
                continue;
            }
            const uint32_t a = win + PDMA_RING_OFF + (head & (ring - 1u)) * static_cast<uint32_t>(sizeof(PdmaDesc));
            const uint32_t w0 = io.ld(a);
            PdmaDesc d{};
            d.op = static_cast<uint16_t>(w0 & 0xffffu);
            d.tensor = static_cast<uint16_t>(w0 >> 16);
            d.tile0 = io.ld(a + 4);
            d.n_tiles = io.ld(a + 8);
            d.slab = io.ld(a + 12);
            d.slab_off = io.ld(a + 16);
            d.tag = io.ld(a + 20);
            d.rsvd[0] = io.ld(a + 24);
            d.rsvd[1] = io.ld(a + 28);
            head++;
            io.st(ctrl(offsetof(PdmaCtrl, head)), head);
            if (d.op == PDMA_STOP) {
                last_tag = d.tag;
                break;
            }
            // the tensor entry is read from L1 per descriptor (no table copy on the small RISC-V stack)
            const PdmaTensor t = d.tensor < n_tensors ? tensor(d.tensor) : PdmaTensor{};
            const uint32_t e = pdma_check(d, d.tensor < n_tensors ? &t : nullptr, n_slabs, stage);
            if (e != PDMA_OK) {
                if (err == PDMA_OK) {
                    err = e;
                    err_index = head - 1;
                }
                n_errors++;
            } else if (d.op == PDMA_LOAD && d.rsvd[1] != 0) {
                load_burst(t, slab_addr(d.slab) + d.slab_off, d.tile0, d.n_tiles, d.rsvd[0] > 1u ? d.rsvd[0] : 1u,
                           d.rsvd[1]);
            } else if (d.op == PDMA_LOAD) {
                t_reads = n_groups = 0;
                load(t, slab_addr(d.slab) + d.slab_off, d.tile0, d.n_tiles);
            } else if (d.op == PDMA_SAVE) {
                t_reads = n_groups = 0;
                save(t, slab_addr(d.slab) + d.slab_off, d.tile0, d.n_tiles);
            }
            done++;
            last_tag = d.tag;
            // publish when the ring is drained or every 16 descriptors (a record costs one PCIe write)
            if (io.ld(ctrl(offsetof(PdmaCtrl, tail))) - head == 0 || (done & 15u) == 0) {
                publish(done_slab, done_off);
            }
        }
        done++;  // the STOP itself
        publish(done_slab, done_off);
        io.write_barrier();
        io.end_pcie();
    }
};
