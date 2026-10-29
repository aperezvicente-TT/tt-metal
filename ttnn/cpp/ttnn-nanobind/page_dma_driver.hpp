// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Page-DMA driver: persistent page-DMA kernels (kernels/page_dma.cpp) on claimed SERVICE cores of every chip of a
// mesh, moving pages between a pinned host page pool (1 GiB hugepages; LMCache's CPU tier allocates from it) and
// interleaved DRAM buffers (the paged latent caches).
//
//   - service cores are free dispatch-column cores: outside the worker grid, so the model's programs and traces are
//     untouched; their workload goes through slow dispatch (EnqueueMeshWorkload routes it), so Finish / trace replay
//     never wait on the persistent kernel;
//   - every GiB slab of the pool is pinned once for all chips (PinnedMemory over the whole mesh); the chips address
//     it with raw 64-bit IOVAs;
//   - descriptors are written into each core's L1 ring with UMD writes (no command queue: usable while a trace runs);
//     completions are polled in host memory (PdmaDone records the kernels write over PCIe);
//   - LOAD descriptors go to every chip (the latent cache is replicated), SAVE descriptors to one chip.
//
// Lifetime: stop() (also the destructor) sends STOP to every core, waits for the kernels to exit, releases the cores,
// then unpins. The pool memory must outlive this object (the Python owner keeps the hugetlbfs file named until the
// kernels are stopped).

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <internal/service/service_core_manager.hpp>
#include <tt_stl/assert.hpp>

#include "kernels/page_dma_host.h"

namespace ttnn::dram_image::pdma {

namespace tm = tt::tt_metal;
namespace td = tt::tt_metal::distributed;

constexpr const char* PAGE_DMA_KERNEL = "ttnn/cpp/ttnn-nanobind/kernels/page_dma.cpp";
constexpr uint64_t SLAB = PDMA_SLAB_BYTES;

struct L1Port {
    tm::IDevice* dev;
    tm::CoreCoord core;
    void write(uint32_t a, const void* p, uint32_t n) {
        tm::detail::WriteToDeviceL1(dev, core, a, std::span<const uint8_t>(static_cast<const uint8_t*>(p), n));
    }
    uint32_t read32(uint32_t a) {
        uint32_t v = 0;
        tm::detail::ReadFromDeviceL1(dev, core, a, std::span<uint8_t>(reinterpret_cast<uint8_t*>(&v), 4));
        return v;
    }
    void fence() { std::atomic_thread_fence(std::memory_order_seq_cst); }
};

// pins of a pool a kernel may still write (a failed stop): never unpinned in this process
inline std::vector<std::shared_ptr<tm::experimental::PinnedMemory>>& leaked_pins() {
    static std::vector<std::shared_ptr<tm::experimental::PinnedMemory>> v;
    return v;
}

struct Tensor {
    uint32_t base, n_tiles, tile_bytes;
};

class PageDma {
public:
    PageDma(std::shared_ptr<td::MeshDevice> mesh, uintptr_t pool, uint32_t gib, uint32_t cores_per_chip, uint32_t ring,
            uint32_t stage, const std::vector<Tensor>& tensors, uint64_t done_off, std::pair<uint32_t, uint32_t> save_rc,
            std::mutex& queue_mu)
        : mesh_(std::move(mesh)), pool_(reinterpret_cast<uint8_t*>(pool)), gib_(gib), cores_(cores_per_chip),
          ring_(ring), stage_(stage), done_off_(done_off), save_rc_(save_rc), queue_mu_(queue_mu) {
        TT_FATAL(gib_ >= 1 && gib_ <= PDMA_MAX_SLABS, "pool of {} GiB (1..{})", gib_, PDMA_MAX_SLABS);
        TT_FATAL(cores_ >= 1, "cores_per_chip >= 1");
        TT_FATAL(done_off_ % 64 == 0 && done_off_ / SLAB == (done_off_ + 64ull * cores_ * mesh_->num_devices() - 1) / SLAB &&
                     done_off_ + 64ull * cores_ * mesh_->num_devices() <= gib_ * SLAB,
                 "completion area must be 64-B aligned inside one slab of the pool");  // fmt: skip
        const uint32_t banks = mesh_->num_dram_channels();
        for (const auto& t : tensors) {
            tensors_.push_back(PdmaTensor{t.base, t.n_tiles, t.tile_bytes, banks});
        }
        TT_FATAL(tensors_.size() <= PDMA_MAX_TENSORS, "at most {} tensors", PDMA_MAX_TENSORS);
    }

    PageDma(const PageDma&) = delete;
    PageDma& operator=(const PageDma&) = delete;

    ~PageDma() {
        // a mesh already closed (the owner forgot stop()): no device access is possible any more; the pins stay
        // (leaked) rather than unpinning memory a kernel might still write
        if (started_ && !stopped_ && !mesh_->is_initialized()) {
            for (auto& p : pins_) {
                leaked_pins().push_back(p);
            }
            return;
        }
        try {
            stop(10.0);
        } catch (...) {
        }
    }

    void start(double timeout_s) {
        TT_FATAL(!started_, "PageDma already started");
        started_ = true;
        const td::MeshCoordinateRangeSet all(td::MeshCoordinateRange(mesh_->shape()));
        for (uint32_t s = 0; s < gib_; s++) {
            tm::HostBuffer hb(ttsl::Span<uint8_t>(pool_ + s * SLAB, SLAB), tm::MemoryPin(std::shared_ptr<void>()));
            pins_.push_back(tm::experimental::PinnedMemory::Create(*mesh_, all, hb, true));
        }
        auto& svc = tm::internal::service_core_manager(*mesh_);
        const uint32_t window = PdmaHostCore<L1Port>::staging_bytes(ring_, stage_);
        td::MeshWorkload wl;
        uint32_t chip_index = 0;
        for (const auto& coord : td::MeshCoordinateRange(mesh_->shape())) {
            Chip c;
            c.coord = coord;
            c.dev = mesh_->get_device(coord);
            c.saves = coord[0] == save_rc_.first && coord[1] == save_rc_.second;
            std::vector<PdmaSlab> slabs;
            uint32_t xy = 0;
            for (uint32_t s = 0; s < gib_; s++) {
                auto na = pins_[s]->get_noc_addr(c.dev->id());
                TT_FATAL(na.has_value() && na->device_id == c.dev->id(), "slab {} not addressable from chip {}", s,
                         c.dev->id());  // fmt: skip
                TT_FATAL(s == 0 || na->pcie_xy_enc == xy, "slabs disagree on the PCIe core");
                xy = na->pcie_xy_enc;
                slabs.push_back(PdmaSlab{uint32_t(na->addr & 0xffffffffu), uint32_t(na->addr >> 32)});
            }
            auto claimable = svc.get_claimable_cores(c.dev);
            TT_FATAL(claimable.size() >= cores_, "chip {}: {} claimable service cores < {}", c.dev->id(),
                     claimable.size(), cores_);  // fmt: skip
            c.cores.assign(claimable.begin(), claimable.begin() + cores_);
            svc.claim(c.dev, c.cores);
            c.claimed = true;
            tm::Program prog = tm::CreateProgram();
            std::set<tm::CoreRange> ranges;
            for (const auto& core : c.cores) {
                ranges.insert(tm::CoreRange(core, core));
            }
            auto kernel = tm::CreateKernel(prog, PAGE_DMA_KERNEL, tm::CoreRangeSet(ranges),
                                           tm::DataMovementConfig{.processor = tm::DataMovementProcessor::RISCV_0,
                                                                  .noc = tm::NOC::RISCV_0_default});  // fmt: skip
            for (uint32_t i = 0; i < cores_; i++) {
                const auto core = c.cores[i];
                const uint32_t win = static_cast<uint32_t>(svc.allocate_l1(c.dev, core, window));
                c.ports.push_back(std::make_unique<L1Port>(L1Port{c.dev, core}));
                auto h = std::make_unique<PdmaHostCore<L1Port>>(PdmaHostCore<L1Port>{*c.ports.back(), win, ring_, stage_, {}});
                const uint64_t slot_off = done_off_ + 64ull * (chip_index * cores_ + i);
                h->init(tensors_, slabs, uint32_t(slot_off / SLAB), uint32_t(slot_off % SLAB),
                        reinterpret_cast<volatile const PdmaDone*>(pool_ + slot_off));  // fmt: skip
                std::memset(pool_ + slot_off, 0, 64);
                tm::SetRuntimeArgs(prog, kernel, core, {win, xy});
                c.host.push_back(std::move(h));
            }
            wl.add_program(td::MeshCoordinateRange(coord, coord), std::move(prog));
            chips_.push_back(std::move(c));
            chip_index++;
        }
        {
            std::lock_guard g(queue_mu_);
            td::EnqueueMeshWorkload(mesh_->mesh_command_queue(), wl, false);  // service cores: routed to slow dispatch
        }
        running_ = true;
        for (auto& c : chips_) {
            for (auto& h : c.host) {
                h->wait_up(timeout_s);
            }
        }
    }

    // LOAD descriptors to every chip (replicated caches), round-robin over each chip's cores; `chips` non-empty: only
    // those chips (mesh order) -- the sequence-sharded latent pool keeps a page on its owner chip only.
    void load(const PdmaDesc* d, size_t n, double timeout_s, const std::vector<uint32_t>& chips = {}) {
        submit(d, n, false, timeout_s, chips);
    }
    // SAVE descriptors to the save chip only (replicated caches: one copy is enough); `chips` non-empty: those chips
    // (mesh order) instead -- per-chip state (each chip's own heads / channels) is saved by every chip, each with its
    // own host destinations (one call per chip). Every chip has the slabs mapped (start()), so any chip may save.
    void save(const PdmaDesc* d, size_t n, double timeout_s, const std::vector<uint32_t>& chips = {}) {
        submit(d, n, true, timeout_s, chips);
    }

    // Every descriptor submitted so far has completed on every chip it went to.
    void wait(double timeout_s) {
        for (auto& c : chips_) {
            for (auto& h : c.host) {
                h->wait_done(h->tail, timeout_s);
            }
        }
    }

    // [chip][core] -> {done, tail, err, n_errors, bytes}
    std::vector<std::vector<std::array<uint32_t, 5>>> stats() {
        std::vector<std::vector<std::array<uint32_t, 5>>> out;
        for (auto& c : chips_) {
            out.emplace_back();
            for (auto& h : c.host) {
                out.back().push_back({h->done(), h->tail, h->err(),
                                      h->port.read32(h->ctrl(offsetof(PdmaCtrl, n_errors))),
                                      h->port.read32(h->ctrl(offsetof(PdmaCtrl, bytes_lo)))});  // fmt: skip
            }
        }
        return out;
    }

    void stop(double timeout_s) {
        if (!started_ || stopped_) {
            return;
        }
        stopped_ = true;
        auto& svc = tm::internal::service_core_manager(*mesh_);
        std::string first_error;
        if (running_) {
            for (auto& c : chips_) {
                for (auto& h : c.host) {
                    try {
                        h->stop(timeout_s);
                    } catch (const std::exception& e) {
                        if (first_error.empty()) {
                            first_error = e.what();
                        }
                    }
                }
            }
        }
        if (!first_error.empty()) {
            // a kernel that did not take STOP may still DMA into the pool: keep the cores claimed and the slabs
            // pinned (leaked on purpose) rather than unpin memory a running kernel can write
            for (auto& p : pins_) {
                leaked_pins().push_back(p);
            }
            TT_THROW("page-DMA stop failed ({}); service cores left claimed, pool left pinned", first_error);
        }
        for (auto& c : chips_) {
            if (running_) {
                for (const auto& core : c.cores) {
                    svc.wait_done(c.dev, core);
                }
            }
            if (c.claimed) {
                svc.release(c.dev, c.cores);
                c.claimed = false;
            }
        }
        running_ = false;
        pins_.clear();
    }

    uint32_t cores_per_chip() const { return cores_; }
    uint32_t window_bytes() const { return PdmaHostCore<L1Port>::staging_bytes(ring_, stage_); }

private:
    struct Chip {
        td::MeshCoordinate coord{0, 0};
        tm::IDevice* dev = nullptr;
        bool saves = false, claimed = false;
        std::vector<tm::CoreCoord> cores;
        std::vector<std::unique_ptr<L1Port>> ports;
        std::vector<std::unique_ptr<PdmaHostCore<L1Port>>> host;
    };

    void submit(const PdmaDesc* d, size_t n, bool saves, double timeout_s, const std::vector<uint32_t>& only = {}) {
        TT_FATAL(running_, "PageDma not running");
        for (uint32_t c : only) {
            TT_FATAL(c < chips_.size(), "chip {} outside the mesh ({} chips)", c, chips_.size());
        }
        for (size_t i = 0; i < n; i++) {
            TT_FATAL(d[i].op == (saves ? PDMA_SAVE : PDMA_LOAD), "descriptor {}: op {} in a {} batch", i, d[i].op,
                     saves ? "save" : "load");  // fmt: skip
            const uint32_t e = chips_.front().host.front()->check(d[i]);
            TT_FATAL(e == PDMA_OK, "descriptor {} refused (error {})", i, e);
        }
        std::vector<std::vector<PdmaDesc>> per_core(cores_);
        for (size_t i = 0; i < n; i++) {
            per_core[i % cores_].push_back(d[i]);
        }
        uint32_t ci = 0;
        for (auto& c : chips_) {
            const uint32_t index = ci++;
            if (saves && only.empty() && !c.saves) {
                continue;
            }
            if (!only.empty() && std::find(only.begin(), only.end(), index) == only.end()) {
                continue;
            }
            for (uint32_t k = 0; k < cores_; k++) {
                if (!per_core[k].empty()) {
                    c.host[k]->push(per_core[k].data(), uint32_t(per_core[k].size()), timeout_s);
                }
            }
        }
    }

    std::shared_ptr<td::MeshDevice> mesh_;
    uint8_t* pool_;
    uint32_t gib_, cores_, ring_, stage_;
    uint64_t done_off_;
    std::pair<uint32_t, uint32_t> save_rc_;
    std::mutex& queue_mu_;
    std::vector<PdmaTensor> tensors_;
    std::vector<std::shared_ptr<tm::experimental::PinnedMemory>> pins_;
    std::deque<Chip> chips_;
    bool started_ = false, running_ = false, stopped_ = false;
};

}  // namespace ttnn::dram_image::pdma
