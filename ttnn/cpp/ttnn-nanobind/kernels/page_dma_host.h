// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host side of the page-DMA protocol for ONE page-DMA core: window setup, ring producer, completion wait, stop.
// Templated on a Port so the real driver (UMD L1 writes to a service core) and the host emulator share it:
//   void write(uint32_t l1, const void* p, uint32_t n)   device L1 write (bodies before tail: see push)
//   uint32_t read32(uint32_t l1)                          device L1 read
//   void fence()                                          order earlier writes before later ones
// The completion record in host memory (PdmaDone) is read through `done_slot` when configured: a RAM read, so the
// steady-state wait does not touch the chip.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "page_dma_proto.h"

template <class Port>
struct PdmaHostCore {
    Port& port;
    uint32_t win;
    uint32_t ring;
    uint32_t stage;
    std::vector<PdmaTensor> tensors;
    uint32_t n_slabs = 0;
    volatile const PdmaDone* done_slot = nullptr;  // host-memory completion record, or null (then L1 is polled)
    uint32_t tail = 0;                              // descriptors written
    uint32_t head_seen = 0;                         // last head read from L1

    uint32_t ctrl(size_t field) const { return win + PDMA_CTRL_OFF + static_cast<uint32_t>(field); }

    static uint32_t staging_bytes(uint32_t ring, uint32_t stage) {
        return PDMA_RING_OFF + ring * static_cast<uint32_t>(sizeof(PdmaDesc)) + 2u * stage;
    }

    // Before launch: tables + config; magic cleared so wait_up() sees the NEW kernel's signal.
    void init(const std::vector<PdmaTensor>& t, const std::vector<PdmaSlab>& slabs, uint32_t done_slab_,
              uint32_t done_off_, volatile const PdmaDone* done_slot_) {
        if (ring == 0 || (ring & (ring - 1)) != 0) {
            throw std::invalid_argument("ring must be a power of two");
        }
        if (stage == 0 || stage % 64 != 0 || t.size() > PDMA_MAX_TENSORS || slabs.size() > PDMA_MAX_SLABS) {
            throw std::invalid_argument("bad page-DMA config");
        }
        tensors = t;
        n_slabs = static_cast<uint32_t>(slabs.size());
        done_slot = done_slot_;
        PdmaCtrl c{};
        c.n_tensors = static_cast<uint32_t>(t.size());
        c.n_slabs = n_slabs;
        c.ring = ring;
        c.stage = stage;
        c.done_slab = done_slot_ ? done_slab_ : 0xffffffffu;
        c.done_off = done_off_;
        if (!t.empty()) {
            port.write(win + PDMA_TENSORS_OFF, t.data(), static_cast<uint32_t>(t.size() * sizeof(PdmaTensor)));
        }
        if (!slabs.empty()) {
            port.write(win + PDMA_SLABS_OFF, slabs.data(), static_cast<uint32_t>(slabs.size() * sizeof(PdmaSlab)));
        }
        port.write(win + PDMA_CTRL_OFF, &c, sizeof(c));
        port.fence();
        tail = head_seen = 0;
    }

    template <class Clock = std::chrono::steady_clock>
    void wait_up(double timeout_s) {
        const auto t0 = Clock::now();
        while (port.read32(ctrl(offsetof(PdmaCtrl, magic))) != PDMA_MAGIC) {
            if (std::chrono::duration<double>(Clock::now() - t0).count() > timeout_s) {
                throw std::runtime_error("page-DMA kernel did not come up");
            }
            std::this_thread::yield();
        }
    }

    uint32_t check(const PdmaDesc& d) const {
        return pdma_check(d, d.tensor < tensors.size() ? &tensors[d.tensor] : nullptr, n_slabs, stage);
    }

    // Append descriptors; blocks while the ring is full. Bodies are written, then fenced, then tail.
    void push(const PdmaDesc* d, uint32_t n, double timeout_s = 10.0) {
        const auto t0 = std::chrono::steady_clock::now();
        uint32_t i = 0;
        while (i < n) {
            uint32_t room = ring - (tail - head_seen);
            if (room == 0) {
                head_seen = port.read32(ctrl(offsetof(PdmaCtrl, head)));
                room = ring - (tail - head_seen);
                if (room == 0) {
                    if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
                        throw std::runtime_error("page-DMA ring stayed full");
                    }
                    std::this_thread::yield();
                    continue;
                }
            }
            // contiguous run up to the ring end
            const uint32_t slot = tail & (ring - 1);
            uint32_t m = std::min({room, n - i, ring - slot});
            port.write(win + PDMA_RING_OFF + slot * static_cast<uint32_t>(sizeof(PdmaDesc)), d + i,
                       m * static_cast<uint32_t>(sizeof(PdmaDesc)));
            i += m;
            tail += m;
            port.fence();
            port.write(ctrl(offsetof(PdmaCtrl, tail)), &tail, 4);
            port.fence();
        }
    }

    uint32_t done() const {
        if (done_slot) {
            return __atomic_load_n(&done_slot->done, __ATOMIC_ACQUIRE);
        }
        return port.read32(ctrl(offsetof(PdmaCtrl, done)));
    }

    // Wait until `target` descriptors completed (done counts every taken descriptor, skipped ones included).
    void wait_done(uint32_t target, double timeout_s = 10.0) const {
        const auto t0 = std::chrono::steady_clock::now();
        while (static_cast<int32_t>(done() - target) < 0) {
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > timeout_s) {
                throw std::runtime_error("page-DMA completion timed out at " + std::to_string(done()) + " / " +
                                         std::to_string(target));
            }
            std::this_thread::yield();
        }
    }

    uint32_t err() const { return port.read32(ctrl(offsetof(PdmaCtrl, err))); }

    // STOP after everything queued; returns once the kernel published it (it then exits).
    void stop(double timeout_s = 10.0) {
        PdmaDesc s{};
        s.op = PDMA_STOP;
        s.tag = 0xffffffffu;
        push(&s, 1, timeout_s);
        wait_done(tail, timeout_s);
    }
};
