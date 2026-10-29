// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DRAM-image weight loading. read_image / device_block_hashes / ImagePuller work on any 2D mesh: a unit (1x1) submesh
// or a TP parent mesh whose tensors are allocated lock-step (one address, per-chip contents). On a parent mesh every
// call targets ONE mesh coordinate (`coord`), and the per-chip threads share the mesh command queue: their enqueues
// (never their waits) serialize on queue_mu(). Never use 1x1 submeshes of a parent that holds allocations: a submesh
// has its own allocator (its scratch buffers / program binaries would land on the parent's).
//
// Two image layouts:
//
// * region images (read_region / pull_region, the original prototype): the raw content of one DRAM address range
//   [address, address + bytes_per_bank) in every bank, stored in interleaved page order (page j of an interleaved view
//   buffer at `address` with `page_size` maps to bank j % num_banks at address + (j / num_banks) * page_size).
// * segment images (read_image / ImagePuller): a list of segments [lo, hi) (DRAM bank offsets, multiples of 64 B),
//   stored bank-major: for each segment, for each bank b, the bytes [lo, hi) of bank b. Any set of live DRAM buffers
//   (each occupies the same [address, address + aligned_size_per_bank) in every bank) is a list of such segments, so
//   holes between buffers are neither stored nor overwritten.
//
// Loading never pushes from the host: Tensix cores read 1 GB pinned hugepage slots over PCIe and write DRAM, while
// host threads read the next chunks of the file into the free slots. Allocations replayed afterwards at the same
// addresses find their bytes in place.
//
// Checksums: per 1 GiB chunk of image file, sum over the 8-byte words w_i (i = word index in the file) of
// mix64(w_i ^ (i * golden)) mod 2^64 — position-dependent, order-independent, so threads hash their own blocks.

#include "dram_image.hpp"
#include "page_dma_driver.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/pinned_memory.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_event.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"
// process-wide JIT build counters (exported by libtt_metal; header is self-contained)
#include "../../../tt_metal/jit_build/build_cache_telemetry.hpp"

namespace ttnn::dram_image {
namespace {

using namespace tt::tt_metal;
namespace dist = tt::tt_metal::distributed;
using Clock = std::chrono::steady_clock;
constexpr uint64_t GB = 1ull << 30;
constexpr uint32_t BURST = 16384;
constexpr uint32_t ALIGN = 64;
constexpr const char* KERNEL = "tests/tt_metal/tt_metal/perf_microbenchmark/dram_image/kernels/pcie_pull_to_dram.cpp";
constexpr const char* RUNS_KERNEL = "ttnn/cpp/ttnn-nanobind/kernels/dram_image_pull_runs.cpp";
constexpr const char* HASH_KERNEL = "ttnn/cpp/ttnn-nanobind/kernels/dram_image_hash_blocks.cpp";
constexpr uint64_t BLOCK_DEFAULT = 4ull << 20;

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// the per-chip threads of one mesh share its command queue: enqueues (and blocking shard reads) take this lock
std::mutex& queue_mu() {
    static std::mutex m;
    return m;
}

dist::MeshCoordinate coord_of(const dist::MeshDevice& mesh, const std::pair<uint32_t, uint32_t>& rc) {
    TT_FATAL(mesh.shape().dims() == 2, "dram_image: 2D meshes only (got {})", mesh.shape());
    TT_FATAL(
        rc.first < mesh.shape()[0] && rc.second < mesh.shape()[1],
        "coord ({}, {}) outside the mesh {}",
        rc.first,
        rc.second,
        mesh.shape());
    return dist::MeshCoordinate(rc.first, rc.second);
}

// ------------------------------------------------------------------------------------------------ checksum
inline uint64_t mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// bytes [p, p + len) sitting at file offset `off` (both multiples of 8)
uint64_t hash_block(const uint8_t* p, uint64_t len, uint64_t off) {
    const auto* w = reinterpret_cast<const uint64_t*>(p);
    uint64_t idx = off >> 3, s = 0;
    for (uint64_t i = 0, n = len >> 3; i < n; i++) {
        s += mix64(w[i] ^ ((idx + i) * 0x9E3779B97F4A7C15ULL));
    }
    return s;
}

// add hash of [p, p + len) at file offset off into per-chunk sums (splits at chunk boundaries)
void hash_into(std::vector<std::atomic<uint64_t>>& sums, const uint8_t* p, uint64_t len, uint64_t off) {
    while (len) {
        const uint64_t c = off / GB, n = std::min(len, (c + 1) * GB - off);
        sums.at(c).fetch_add(hash_block(p, n, off), std::memory_order_relaxed);
        p += n;
        off += n;
        len -= n;
    }
}

std::vector<uint64_t> to_vec(const std::vector<std::atomic<uint64_t>>& a) {
    std::vector<uint64_t> v;
    for (const auto& x : a) {
        v.push_back(x.load());
    }
    return v;
}

// ------------------------------------------------------------------------------------------------ helpers
std::shared_ptr<dist::MeshBuffer> view(
    dist::MeshDevice& mesh, uint64_t address, uint64_t bytes_per_bank, uint32_t page_size) {
    // non-owning view at `address` on every chip of the mesh (lock-step allocations: same address everywhere)
    const uint32_t banks = mesh.num_dram_channels();
    TT_FATAL(bytes_per_bank % page_size == 0, "bytes_per_bank must be a multiple of page_size");
    dist::DeviceLocalBufferConfig local{.page_size = page_size, .buffer_type = BufferType::DRAM};
    dist::ReplicatedBufferConfig global{.size = bytes_per_bank * banks};
    return dist::MeshBuffer::create(global, local, &mesh, address);
}

uint8_t* map_hugepage_1g(int idx) {
    static std::atomic<int> seq{0};
    std::string path = fmt::format("/dev/hugepages-1G/dram_image_{}_{}_{}", getpid(), idx, seq++);
    int fd = open(path.c_str(), O_CREAT | O_RDWR, 0600);
    TT_FATAL(fd >= 0, "open {} failed", path);
    void* p = mmap(nullptr, GB, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, 0);
    close(fd);
    unlink(path.c_str());
    TT_FATAL(p != MAP_FAILED, "mmap of a 1 GB hugepage failed (free 1G hugepages?)");
    return static_cast<uint8_t*>(p);
}

struct Seg {
    uint64_t lo, hi;
};

std::vector<Seg> check_segments(const std::vector<std::pair<uint64_t, uint64_t>>& segments) {
    std::vector<Seg> s;
    uint64_t prev = 0;
    for (const auto& [lo, hi] : segments) {
        TT_FATAL(lo < hi, "empty segment [{:#x}, {:#x})", lo, hi);
        TT_FATAL(lo % ALIGN == 0 && hi % ALIGN == 0, "segment [{:#x}, {:#x}) not 64 B aligned", lo, hi);
        TT_FATAL(lo >= prev, "segments must be sorted and disjoint");
        TT_FATAL(hi - lo < (1ull << 32), "segment longer than 4 GiB per bank");
        prev = hi;
        s.push_back({lo, hi});
    }
    return s;
}

// one image run: `len` bytes at offset `src` of the pulled byte stream, read from file offset `file` -> bank `bank`,
// bank offset `addr` (a whole-image pull streams the file itself: src == file)
struct Run {
    uint64_t src, file;
    uint32_t bank, addr, len;
};

std::vector<Run> runs_of(const std::vector<Seg>& segs, uint32_t banks, uint64_t& total) {
    std::vector<Run> runs;
    uint64_t off = 0;
    for (const auto& s : segs) {
        for (uint32_t b = 0; b < banks; b++) {
            runs.push_back({off, off, b, uint32_t(s.lo), uint32_t(s.hi - s.lo)});
            off += s.hi - s.lo;
        }
    }
    total = off;
    return runs;
}

// verification / partial-pull blocks: every (segment, bank) run cut into block_bytes pieces (the last one shorter);
// order = segment, bank, block -- the order of read_image's block_hashes and device_block_hashes
std::vector<Run> blocks_of(const std::vector<Seg>& segs, uint32_t banks, uint64_t block_bytes) {
    TT_FATAL(block_bytes % ALIGN == 0 && block_bytes > 0, "block_bytes must be a multiple of 64");
    std::vector<Run> blocks;
    uint64_t off = 0;
    for (const auto& s : segs) {
        const uint64_t n = s.hi - s.lo;
        for (uint32_t b = 0; b < banks; b++) {
            for (uint64_t k = 0; k < n; k += block_bytes) {
                const uint32_t len = uint32_t(std::min(block_bytes, n - k));
                blocks.push_back({off + k, off + k, b, uint32_t(s.lo + k), len});
            }
            off += n;
        }
    }
    return blocks;
}

// host twin of kernels/dram_image_hash_blocks.cpp (32-bit words, j = word index in the block)
uint64_t block_hash(const uint8_t* p, uint64_t len) {
    const auto* w = reinterpret_cast<const uint32_t*>(p);
    uint32_t s1 = 0, s2 = 0;
    for (uint32_t j = 0, n = uint32_t(len >> 2); j < n; j++) {
        const uint32_t v = w[j];
        uint32_t x = v ^ (j * 0x9E3779B9u);
        x = (x << 15) | (x >> 17);
        s1 += x * 0x85EBCA6Bu;
        s2 += ((v + j * 0xC2B2AE35u) ^ (v >> 13)) * 0x27D4EB2Fu;
    }
    return (uint64_t(s2) << 32) | s1;
}

// ------------------------------------------------------------------------------------------------ prototype (region)
// raw region -> file
std::map<std::string, double> read_region(
    const std::shared_ptr<dist::MeshDevice>& mesh,
    uint64_t address,
    uint64_t bytes_per_bank,
    uint32_t page_size,
    const std::string& path) {
    auto t0 = Clock::now();
    auto buf = view(*mesh, address, bytes_per_bank, page_size);
    std::vector<uint8_t> host;
    {
        nb::gil_scoped_release nogil;
        dist::EnqueueReadMeshBuffer(mesh->mesh_command_queue(), host, buf, true);
    }
    double t_read = since(t0);
    FILE* f = fopen(path.c_str(), "wb");
    TT_FATAL(f, "cannot write {}", path);
    TT_FATAL(fwrite(host.data(), 1, host.size(), f) == host.size(), "short write to {}", path);
    fclose(f);
    return {{"bytes", double(host.size())}, {"read_s", t_read}, {"total_s", since(t0)}};
}

// file -> raw region, Tensix cores pulling over PCIe from a ring of pinned 1 GB hugepages
std::map<std::string, double> pull_region(
    const std::shared_ptr<dist::MeshDevice>& mesh,
    uint64_t address,
    uint64_t bytes_per_bank,
    uint32_t page_size,
    const std::string& path,
    uint32_t slots,
    uint32_t num_cores,
    uint32_t read_threads) {
    auto t0 = Clock::now();
    TT_FATAL(GB % page_size == 0 && page_size % BURST == 0, "page_size must divide 1 GB and be a multiple of 16 KB");
    auto buf = view(*mesh, address, bytes_per_bank, page_size);
    const uint64_t total = buf->size();
    const uint32_t n_chunks = (total + GB - 1) / GB;
    const uint32_t chunk_pages = GB / page_size;
    slots = std::max(1u, std::min(slots, n_chunks));
    read_threads = std::max(1u, read_threads);

    const dist::MeshCoordinate coord(0, 0);
    const dist::MeshCoordinateRangeSet here(dist::MeshCoordinateRange(coord, coord));
    IDevice* dev = mesh->get_device(0, 0);
    auto& cq = mesh->mesh_command_queue();

    std::vector<uint8_t*> slot_ptr(slots);
    std::vector<std::shared_ptr<experimental::PinnedMemory>> slot_pin(slots);
    for (uint32_t s = 0; s < slots; s++) {
        slot_ptr[s] = map_hugepage_1g(s);
        HostBuffer hb(ttsl::Span<uint8_t>(slot_ptr[s], GB), MemoryPin(std::shared_ptr<void>()));
        slot_pin[s] = experimental::PinnedMemory::Create(*mesh, here, hb, true);
    }
    double t_setup = since(t0);

    auto grid = dev->compute_with_storage_grid_size();
    num_cores = std::min<uint32_t>(num_cores, grid.x * grid.y);
    std::vector<CoreCoord> cores;
    std::set<CoreRange> ranges;
    for (uint32_t i = 0; i < num_cores; i++) {
        cores.push_back({i % grid.x, i / grid.x});
        ranges.insert(CoreRange(cores.back(), cores.back()));
    }
    const CoreRangeSet core_set(ranges);

    int fd = open(path.c_str(), O_RDONLY);
    TT_FATAL(fd >= 0, "cannot open {}", path);
    double t_file = 0, t_wait = 0;
    std::vector<std::optional<dist::MeshEvent>> done(slots);
    {
        nb::gil_scoped_release nogil;
        for (uint32_t c = 0; c < n_chunks; c++) {
            const uint32_t s = c % slots;
            if (done[s]) {
                auto tw = Clock::now();
                dist::EventSynchronize(*done[s]);  // the slot's previous chunk has been pulled
                t_wait += since(tw);
            }
            const uint64_t off = uint64_t(c) * GB, len = std::min<uint64_t>(GB, total - off);
            auto tf = Clock::now();
            {
                // page-cache copies are CPU bound: split the chunk across threads
                std::vector<std::thread> th;
                std::atomic<bool> ok{true};
                for (uint32_t k = 0; k < read_threads; k++) {
                    th.emplace_back([&, k] {
                        uint64_t lo = len * k / read_threads, hi = len * (k + 1) / read_threads;
                        for (uint64_t got = lo; got < hi;) {
                            ssize_t r = pread(fd, slot_ptr[s] + got, hi - got, off + got);
                            if (r <= 0) {
                                ok = false;
                                return;
                            }
                            got += r;
                        }
                    });
                }
                for (auto& t : th) {
                    t.join();
                }
                TT_FATAL(ok, "short read of {} in chunk {}", path, c);
            }
            t_file += since(tf);

            auto na = slot_pin[s]->get_noc_addr(dev->id());
            TT_FATAL(na.has_value() && na->device_id == dev->id(), "pinned slot not NOC-mapped for this chip");
            Program prog = CreateProgram();
            CreateCircularBuffer(
                prog,
                core_set,
                CircularBufferConfig(2 * page_size, {{0, tt::DataFormat::RawUInt32}}).set_page_size(0, page_size));
            std::vector<uint32_t> ct = {na->pcie_xy_enc, page_size, BURST};
            TensorAccessorArgs(*buf->get_reference_buffer()).append_to(ct);
            auto k = CreateKernel(
                prog,
                KERNEL,
                core_set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = ct});
            const uint32_t pages = len / page_size, page_off = c * chunk_pages;
            for (uint32_t i = 0; i < num_cores; i++) {
                uint32_t mine = i < pages ? (pages - i + num_cores - 1) / num_cores : 0;
                SetRuntimeArgs(
                    prog,
                    k,
                    cores[i],
                    {uint32_t(na->addr & 0xffffffffu),
                     uint32_t(na->addr >> 32),
                     uint32_t(buf->address()),
                     page_off + i,
                     num_cores,
                     mine,
                     0u,
                     page_off});
            }
            dist::MeshWorkload wl;
            wl.add_program(dist::MeshCoordinateRange(coord, coord), std::move(prog));
            dist::EnqueueMeshWorkload(cq, wl, false);
            done[s] = cq.enqueue_record_event_to_host();
        }
        dist::Finish(cq);
    }
    close(fd);
    double t_total = since(t0);
    slot_pin.clear();
    for (auto* p : slot_ptr) {
        munmap(p, GB);
    }
    return {
        {"bytes", double(total)},
        {"chunks", double(n_chunks)},
        {"setup_s", t_setup},
        {"file_read_s", t_file},
        {"wait_s", t_wait},
        {"total_s", t_total},
        {"gb_per_s", total / t_total / 1e9}};
}

// ------------------------------------------------------------------------------------------------ segment images
// DRAM segments -> bank-major image file. Each segment is read in pieces through an interleaved view (page size =
// the largest power-of-two multiple of 64 B, <= 64 KB, that divides the piece) and de-interleaved on the host.
nb::dict read_image(
    const std::shared_ptr<dist::MeshDevice>& mesh,
    const std::vector<std::pair<uint64_t, uint64_t>>& segments,
    const std::string& path,
    uint64_t piece_bytes_per_bank,
    uint32_t threads,
    uint64_t block_bytes,
    const std::pair<uint32_t, uint32_t>& coord_rc) {
    auto t0 = Clock::now();
    const auto coord = coord_of(*mesh, coord_rc);
    const auto segs = check_segments(segments);
    const uint32_t banks = mesh->num_dram_channels();
    uint64_t total = 0;
    runs_of(segs, banks, total);
    const uint32_t n_chunks = (total + GB - 1) / GB;
    std::vector<std::atomic<uint64_t>> sums(n_chunks);
    for (auto& s : sums) {
        s = 0;
    }
    TT_FATAL(block_bytes % 65536 == 0 && block_bytes > 0, "block_bytes must be a multiple of 64 KB");
    piece_bytes_per_bank = std::max<uint64_t>(block_bytes, piece_bytes_per_bank / block_bytes * block_bytes);
    // block hashes, indexed like blocks_of(): pieces start on block boundaries of their segment
    std::vector<uint64_t> bh(blocks_of(segs, banks, block_bytes).size());
    std::vector<uint64_t> seg_block0;  // index of the first block of each segment
    {
        uint64_t i = 0;
        for (const auto& s : segs) {
            seg_block0.push_back(i);
            i += uint64_t(banks) * ((s.hi - s.lo + block_bytes - 1) / block_bytes);
        }
    }
    threads = std::clamp<uint32_t>(threads, 1, banks);
    const bool write = !path.empty();  // empty path: checksums only (verify DRAM against an image)
    int fd = write ? open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644) : -1;
    TT_FATAL(!write || fd >= 0, "cannot write {}", path);
    double t_dev = 0, t_host = 0;
    {
        nb::gil_scoped_release nogil;
        TT_FATAL(!write || ftruncate(fd, off_t(total)) == 0, "ftruncate {} failed", path);
        auto& cq = mesh->mesh_command_queue();
        std::vector<uint8_t> host;
        std::vector<std::vector<uint8_t>> rows(banks);
        uint64_t seg_off = 0;
        for (size_t si = 0; si < segs.size(); si++) {
            const auto& s = segs[si];
            const uint64_t n_seg = s.hi - s.lo;
            const uint64_t nblk = (n_seg + block_bytes - 1) / block_bytes;
            for (uint64_t a = s.lo; a < s.hi; a += piece_bytes_per_bank) {
                const uint64_t n = std::min(piece_bytes_per_bank, s.hi - a);
                const uint32_t page = ALIGN * std::gcd<uint64_t>(n / ALIGN, 1024);
                auto td = Clock::now();
                auto buf = view(*mesh, a, n, page);
                {
                    std::lock_guard g(queue_mu());
                    dist::ReadShard(cq, host, buf, coord, true);
                }
                t_dev += since(td);
                auto th0 = Clock::now();
                TT_FATAL(host.size() == n * banks, "short DRAM read");
                const uint64_t page_rows = n / page;
                std::vector<std::thread> th;
                std::atomic<bool> ok{true};
                for (uint32_t k = 0; k < threads; k++) {
                    th.emplace_back([&, k] {
                        for (uint32_t b = k; b < banks; b += threads) {
                            auto& out = rows[b];
                            out.resize(n);
                            for (uint64_t r = 0; r < page_rows; r++) {
                                std::memcpy(out.data() + r * page, host.data() + (r * banks + b) * page, page);
                            }
                            const uint64_t foff = seg_off + b * n_seg + (a - s.lo);
                            for (uint64_t got = 0; write && got < n;) {
                                ssize_t w = pwrite(fd, out.data() + got, n - got, off_t(foff + got));
                                if (w <= 0) {
                                    ok = false;
                                    return;
                                }
                                got += w;
                            }
                            hash_into(sums, out.data(), n, foff);
                            for (uint64_t x = 0; x < n; x += block_bytes) {
                                const uint64_t k = (a - s.lo + x) / block_bytes;
                                bh[seg_block0[si] + b * nblk + k] =
                                    block_hash(out.data() + x, std::min(block_bytes, n - x));
                            }
                        }
                    });
                }
                for (auto& t : th) {
                    t.join();
                }
                TT_FATAL(ok, "short write to {}", path);
                t_host += since(th0);
            }
            seg_off += n_seg * banks;
        }
        TT_FATAL(!write || fsync(fd) == 0, "fsync {} failed", path);
    }
    if (write) {
        close(fd);
    }
    nb::dict d;
    d["bytes"] = total;
    d["chunks"] = n_chunks;
    d["dram_read_s"] = t_dev;
    d["host_s"] = t_host;
    d["total_s"] = since(t0);
    d["checksums"] = to_vec(sums);
    d["block_bytes"] = block_bytes;
    d["block_hashes"] = bh;
    return d;
}

// on-device block hashes of DRAM segments (kernels/dram_image_hash_blocks.cpp on num_cores Tensix cores): the same
// values read_image computed at build time when DRAM still holds the image. Verifies a device-resident image without
// moving it over PCIe.
nb::dict device_block_hashes(
    const std::shared_ptr<dist::MeshDevice>& mesh,
    const std::vector<std::pair<uint64_t, uint64_t>>& segments,
    uint64_t block_bytes,
    uint32_t num_cores,
    uint32_t max_runs,
    uint32_t stage_bytes) {
    auto t0 = Clock::now();
    TT_FATAL(stage_bytes % ALIGN == 0 && max_runs >= 1 && 2 + 4 * max_runs <= max_runtime_args, "bad stage / max_runs");
    const auto segs = check_segments(segments);
    const uint32_t banks = mesh->num_dram_channels();
    const auto blocks = blocks_of(segs, banks, block_bytes);
    std::vector<uint64_t> out(blocks.size());
    std::vector<std::pair<int, std::vector<uint64_t>>> per_device;  // (device id, hashes) in mesh-coordinate order
    std::vector<std::pair<int, std::pair<uint32_t, uint32_t>>> coords;
    uint64_t launches = 0;
    {
        nb::gil_scoped_release nogil;
        IDevice* dev = mesh->get_device(0, 0);
        auto grid = dev->compute_with_storage_grid_size();
        const uint32_t ncores = std::max(1u, std::min<uint32_t>(num_cores, grid.x * grid.y));
        std::vector<CoreCoord> cores;
        std::set<CoreRange> ranges;
        for (uint32_t i = 0; i < ncores; i++) {
            cores.push_back({i % grid.x, i / grid.x});
            ranges.insert(CoreRange(cores.back(), cores.back()));
        }
        const CoreRangeSet core_set(ranges);
        // result pages (64 B each); allocated and freed here, so the allocator ends in the state it started in
        // top-down, away from the (possibly still unallocated) image segments at the bottom of DRAM
        dist::DeviceLocalBufferConfig local{.page_size = 64, .buffer_type = BufferType::DRAM, .bottom_up = false};
        auto res = dist::MeshBuffer::create(
            dist::ReplicatedBufferConfig{.size = std::max<uint64_t>(1, blocks.size()) * 64}, local, mesh.get());
        Program prog = CreateProgram();
        CreateCircularBuffer(
            prog,
            core_set,
            CircularBufferConfig(2 * stage_bytes + 64, {{0, tt::DataFormat::RawUInt32}}).set_page_size(0, 64));
        auto kernel = CreateKernel(
            prog,
            HASH_KERNEL,
            core_set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = {stage_bytes}});
        const uint32_t n_args = 2 + 4 * max_runs;
        for (const auto& c : cores) {
            SetRuntimeArgs(prog, kernel, c, std::vector<uint32_t>(n_args, 0));
        }
        dist::MeshWorkload wl;
        // the same program on every chip: each hashes its own DRAM (lock-step segments) into its own result pages
        wl.add_program(dist::MeshCoordinateRange(mesh->shape()), std::move(prog));
        Program& p = wl.get_programs().begin()->second;
        auto& cq = mesh->mesh_command_queue();
        std::lock_guard qg(queue_mu());
        // block i -> core i % ncores, launch i / (ncores * max_runs)
        const uint64_t per_launch = uint64_t(ncores) * max_runs;
        for (uint64_t first = 0; first < blocks.size(); first += per_launch) {
            for (uint32_t c = 0; c < ncores; c++) {
                auto& args = GetRuntimeArgs(p, kernel, cores[c]);
                args[0] = uint32_t(res->address());
                uint32_t n = 0;
                for (uint64_t i = first + c; i < std::min<uint64_t>(blocks.size(), first + per_launch); i += ncores) {
                    args[2 + 4 * n] = blocks[i].bank;
                    args[3 + 4 * n] = blocks[i].addr;
                    args[4 + 4 * n] = blocks[i].len;
                    args[5 + 4 * n] = uint32_t(i);
                    n++;
                }
                args[1] = n;
            }
            dist::EnqueueMeshWorkload(cq, wl, false);
            launches++;
        }
        for (const auto& coord : dist::MeshCoordinateRange(mesh->shape())) {
            std::vector<uint64_t> pages;
            dist::ReadShard(cq, pages, res, coord, true);  // 8 uint64 per 64 B page
            std::vector<uint64_t> h(blocks.size());
            for (size_t i = 0; i < blocks.size(); i++) {
                h[i] = pages.at(i * 8);
            }
            const int id = mesh->get_device(coord)->id();
            if (per_device.empty()) {
                out = h;
            }
            per_device.emplace_back(id, std::move(h));
            coords.emplace_back(id, std::make_pair(uint32_t(coord[0]), uint32_t(coord[1])));
        }
    }
    uint64_t bytes = 0;
    for (const auto& b : blocks) {
        bytes += b.len;
    }
    nb::dict d;
    d["block_hashes"] = out;  // the first chip's (a unit mesh: its only chip)
    nb::dict pd, pc;
    for (auto& [id, h] : per_device) {
        pd[nb::int_(id)] = h;
    }
    for (auto& [id, rc] : coords) {
        pc[nb::int_(id)] = rc;
    }
    d["per_device"] = pd;  // device id -> hashes
    d["coords"] = pc;      // device id -> mesh coordinate
    d["blocks"] = uint64_t(blocks.size());
    d["bytes"] = bytes;
    d["launches"] = launches;
    d["seconds"] = since(t0);
    return d;
}

// host-only: block hashes of an image file (offline verification; same values as read_image's)
nb::dict file_block_hashes(
    const std::string& path,
    const std::vector<std::pair<uint64_t, uint64_t>>& segments,
    uint32_t banks,
    uint64_t block_bytes,
    uint32_t threads) {
    auto t0 = Clock::now();
    const auto blocks = blocks_of(check_segments(segments), banks, block_bytes);
    std::vector<uint64_t> out(blocks.size());
    int fd = open(path.c_str(), O_RDONLY);
    TT_FATAL(fd >= 0, "cannot open {}", path);
    {
        nb::gil_scoped_release nogil;
        std::atomic<uint64_t> next{0};
        std::atomic<bool> ok{true};
        std::vector<std::thread> th;
        for (uint32_t k = 0; k < std::max(1u, threads); k++) {
            th.emplace_back([&] {
                std::vector<uint8_t> buf(block_bytes);
                for (uint64_t i; (i = next++) < blocks.size();) {
                    const auto& b = blocks[i];
                    for (uint64_t got = 0; got < b.len;) {
                        ssize_t r = pread(fd, buf.data() + got, b.len - got, off_t(b.file + got));
                        if (r <= 0) {
                            ok = false;
                            return;
                        }
                        got += r;
                    }
                    out[i] = block_hash(buf.data(), b.len);
                }
            });
        }
        for (auto& t : th) {
            t.join();
        }
        TT_FATAL(ok, "short read of {}", path);
    }
    close(fd);
    nb::dict d;
    d["block_hashes"] = out;
    d["seconds"] = since(t0);
    return d;
}

// host-only: per-chunk checksums of an image file (tests and offline verification)
nb::dict checksum_file(const std::string& path, uint32_t threads) {
    auto t0 = Clock::now();
    int fd = open(path.c_str(), O_RDONLY);
    TT_FATAL(fd >= 0, "cannot open {}", path);
    struct stat st{};
    TT_FATAL(fstat(fd, &st) == 0, "stat {}", path);
    const uint64_t total = st.st_size;
    TT_FATAL(total % 8 == 0, "{}: size {} not a multiple of 8", path, total);
    const uint32_t n_chunks = (total + GB - 1) / GB;
    std::vector<std::atomic<uint64_t>> sums(n_chunks);
    for (auto& s : sums) {
        s = 0;
    }
    {
        nb::gil_scoped_release nogil;
        constexpr uint64_t BLK = 8ull << 20;
        const uint64_t n_blk = (total + BLK - 1) / BLK;
        std::atomic<uint64_t> next{0};
        std::atomic<bool> ok{true};
        std::vector<std::thread> th;
        for (uint32_t k = 0; k < std::max(1u, threads); k++) {
            th.emplace_back([&] {
                std::vector<uint8_t> buf(BLK);
                for (uint64_t i; (i = next++) < n_blk;) {
                    const uint64_t off = i * BLK, n = std::min(BLK, total - off);
                    for (uint64_t got = 0; got < n;) {
                        ssize_t r = pread(fd, buf.data() + got, n - got, off_t(off + got));
                        if (r <= 0) {
                            ok = false;
                            return;
                        }
                        got += r;
                    }
                    hash_into(sums, buf.data(), n, off);
                }
            });
        }
        for (auto& t : th) {
            t.join();
        }
        TT_FATAL(ok, "short read of {}", path);
    }
    close(fd);
    nb::dict d;
    d["bytes"] = total;
    d["checksums"] = to_vec(sums);
    d["total_s"] = since(t0);
    return d;
}

// image file -> DRAM segments of one chip. start() pins the slots and starts the host reader (it runs ahead while the
// caller replays allocations); finish() (GIL released) streams the chunks through the slots into DRAM.
class ImagePuller {
public:
    ImagePuller(
        std::shared_ptr<dist::MeshDevice> mesh,
        const std::vector<std::pair<uint64_t, uint64_t>>& segments,
        std::string path,
        uint32_t slots,
        uint32_t num_cores,
        uint32_t read_threads,
        bool direct,
        bool checksum,
        uint32_t stage_bytes,
        uint32_t max_runs,
        uint64_t direct_from,
        const std::vector<uint64_t>& only_blocks,
        uint64_t block_bytes,
        const std::pair<uint32_t, uint32_t>& coord_rc,
        bool direct_partial) :
        mesh_(std::move(mesh)),
        coord_(coord_of(*mesh_, coord_rc)),
        path_(std::move(path)),
        cores_n_(num_cores),
        threads_(std::max(1u, read_threads)),
        direct_from_(direct ? 0 : direct_from),
        checksum_(checksum),
        stage_(stage_bytes),
        max_runs_(max_runs),
        direct_partial_(direct_partial) {
        TT_FATAL(stage_ % BURST == 0 && stage_ >= BURST, "stage_bytes must be a multiple of 16 KB");
        TT_FATAL(max_runs_ >= 1 && 3 + 4 * max_runs_ <= max_runtime_args, "max_runs out of range");
        banks_ = mesh_->num_dram_channels();
        segs_ = check_segments(segments);
        runs_ = runs_of(segs_, banks_, total_);
        file_size_ = total_;
        if (!only_blocks.empty()) {
            // partial pull: only these blocks (blocks_of order), packed back to back in the pulled stream; reads are
            // buffered (block file offsets are 64 B aligned) and the file checksums do not apply (the caller verifies
            // with device_block_hashes)
            const auto all = blocks_of(segs_, banks_, block_bytes);
            runs_.clear();
            uint64_t src = 0;
            for (uint64_t i : only_blocks) {
                TT_FATAL(i < all.size(), "block {} out of range ({} blocks)", i, all.size());
                Run r = all[i];
                r.src = src;
                src += r.len;
                runs_.push_back(r);
            }
            total_ = src;
            partial_ = true;
            checksum_ = false;
            direct_from_ = ~0ull;
        }
        n_chunks_ = (total_ + GB - 1) / GB;
        slots_ = std::max(1u, std::min<uint32_t>(slots, std::max<uint32_t>(n_chunks_, 1)));
        sums_ = std::vector<std::atomic<uint64_t>>(n_chunks_);
        for (auto& s : sums_) {
            s = 0;
        }
    }

    ~ImagePuller() {
        stop_reader();
        slot_pin_.clear();
        for (auto* p : slot_ptr_) {
            munmap(p, GB);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
        if (fd_direct_ >= 0) {
            close(fd_direct_);
        }
    }

    uint64_t total_bytes() const { return total_; }

    void start() {
        TT_FATAL(!started_, "ImagePuller already started");
        started_ = true;
        auto t0 = Clock::now();
        struct stat st{};
        // bytes [0, direct_from) are read through the page cache (they stay resident for the next start), the rest with
        // O_DIRECT (no page-cache copy, no eviction of the resident part)
        fd_ = open(path_.c_str(), O_RDONLY);
        TT_FATAL(fd_ >= 0, "cannot open {} (errno {})", path_, errno);
        if (direct_from_ < total_ || (partial_ && direct_partial_)) {
            fd_direct_ = open(path_.c_str(), O_RDONLY | O_DIRECT);
            TT_FATAL(fd_direct_ >= 0, "cannot open {} with O_DIRECT (errno {})", path_, errno);
        }
        TT_FATAL(
            fstat(fd_, &st) == 0 && uint64_t(st.st_size) == file_size_,
            "{}: size {} != image size {}",
            path_,
            uint64_t(st.st_size),
            file_size_);
        const dist::MeshCoordinateRangeSet here(dist::MeshCoordinateRange(coord_, coord_));
        for (uint32_t s = 0; s < slots_; s++) {
            slot_ptr_.push_back(map_hugepage_1g(s));
            HostBuffer hb(ttsl::Span<uint8_t>(slot_ptr_.back(), GB), MemoryPin(std::shared_ptr<void>()));
            slot_pin_.push_back(experimental::PinnedMemory::Create(*mesh_, here, hb, true));
        }
        state_.assign(slots_, FREE);
        slot_chunk_.assign(slots_, -1);
        setup_s_ = since(t0);
        t_start_ = Clock::now();
        reader_ = std::thread([this] { read_loop(); });
    }

    void finish() {
        TT_FATAL(started_, "ImagePuller::finish before start");
        TT_FATAL(!finished_, "ImagePuller::finish called twice");
        auto t0 = Clock::now();
        IDevice* dev = mesh_->get_device(coord_);
        auto& cq = mesh_->mesh_command_queue();
        const dist::MeshCoordinate& coord = coord_;
        const dist::MeshCoordinateRange here(coord, coord);
        auto grid = dev->compute_with_storage_grid_size();
        const uint32_t ncores = std::max(1u, std::min<uint32_t>(cores_n_, grid.x * grid.y));
        std::vector<CoreCoord> cores;
        std::set<CoreRange> ranges;
        for (uint32_t i = 0; i < ncores; i++) {
            cores.push_back({i % grid.x, i / grid.x});
            ranges.insert(CoreRange(cores.back(), cores.back()));
        }
        const CoreRangeSet core_set(ranges);
        std::vector<dist::MeshEvent> events;
        events.reserve(n_chunks_);
        std::optional<dist::MeshWorkload> wl;
        KernelHandle kernel{};
        const uint32_t n_args = 3 + 4 * max_runs_;
        uint64_t batches = 0;
        double t_wait_read = 0, t_wait_dev = 0;
        size_t r0 = 0;  // first run that may intersect the current chunk
        try {
            for (uint32_t c = 0; c < n_chunks_; c++) {
                const uint32_t s = c % slots_;
                {
                    auto tw = Clock::now();
                    std::unique_lock lk(mu_);
                    cv_.wait(lk, [&] { return failed_ || (state_[s] == READY && slot_chunk_[s] == int64_t(c)); });
                    TT_FATAL(!failed_, "image reader failed: {}", error_);
                    t_wait_read += since(tw);
                }
                auto na = slot_pin_[s]->get_noc_addr(dev->id());
                TT_FATAL(na.has_value() && na->device_id == dev->id(), "pinned slot not addressable from this chip");
                if (!wl) {
                    Program prog = CreateProgram();
                    CreateCircularBuffer(
                        prog,
                        core_set,
                        CircularBufferConfig(2 * stage_, {{0, tt::DataFormat::RawUInt32}}).set_page_size(0, stage_));
                    kernel = CreateKernel(
                        prog,
                        RUNS_KERNEL,
                        core_set,
                        DataMovementConfig{
                            .processor = DataMovementProcessor::RISCV_0,
                            .noc = NOC::RISCV_0_default,
                            .compile_args = {na->pcie_xy_enc, stage_, BURST}});
                    pcie_xy_enc_ = na->pcie_xy_enc;
                    for (const auto& core : cores) {
                        SetRuntimeArgs(prog, kernel, core, std::vector<uint32_t>(n_args, 0));
                    }
                    wl.emplace();
                    wl->add_program(dist::MeshCoordinateRange(coord, coord), std::move(prog));
                }
                TT_FATAL(na->pcie_xy_enc == pcie_xy_enc_, "pinned slots disagree on the PCIe core");
                Program& prog = wl->get_programs().begin()->second;
                // pieces of this chunk, cut into per-core contiguous byte shares
                const uint64_t c_lo = uint64_t(c) * GB, c_hi = std::min(total_, c_lo + GB);
                while (r0 < runs_.size() && runs_[r0].src + runs_[r0].len <= c_lo) {
                    r0++;
                }
                const uint64_t share = (((c_hi - c_lo) + ncores - 1) / ncores + ALIGN - 1) / ALIGN * ALIGN;
                std::vector<std::vector<std::array<uint32_t, 4>>> per_core(ncores);
                for (size_t r = r0; r < runs_.size() && runs_[r].src < c_hi; r++) {
                    const Run& run = runs_[r];
                    uint64_t a = std::max(run.src, c_lo), b = std::min(run.src + run.len, c_hi);
                    while (a < b) {
                        const uint64_t core = std::min<uint64_t>((a - c_lo) / share, ncores - 1);
                        const uint64_t core_end = core == ncores - 1 ? c_hi : c_lo + (core + 1) * share;
                        const uint64_t e = std::min(b, core_end);
                        per_core[core].push_back(
                            {uint32_t(a - c_lo), run.bank, uint32_t(run.addr + (a - run.src)), uint32_t(e - a)});
                        a = e;
                    }
                }
                size_t most = 0;
                for (const auto& v : per_core) {
                    most = std::max(most, v.size());
                }
                for (size_t first = 0; first < most; first += max_runs_) {
                    for (uint32_t i = 0; i < ncores; i++) {
                        auto& args = GetRuntimeArgs(prog, kernel, cores[i]);
                        args[0] = uint32_t(na->addr & 0xffffffffu);
                        args[1] = uint32_t(na->addr >> 32);
                        const size_t lo = std::min(first, per_core[i].size());
                        const size_t hi = std::min(first + max_runs_, per_core[i].size());
                        args[2] = uint32_t(hi - lo);
                        for (size_t j = lo; j < hi; j++) {
                            for (int q = 0; q < 4; q++) {
                                args[3 + 4 * (j - lo) + q] = per_core[i][j][q];
                            }
                        }
                    }
                    {
                        // the chip threads of a mesh share its queue (and the first enqueue compiles the kernel)
                        std::lock_guard g(queue_mu());
                        dist::EnqueueMeshWorkload(cq, *wl, false);
                    }
                    batches++;
                }
                {
                    std::lock_guard g(queue_mu());
                    events.push_back(cq.enqueue_record_event_to_host({}, here));  // this chip only
                }
                // the slot that chunk c + 1 needs holds chunk c + 1 - slots: release it once the device pulled it
                if (c + 1 < n_chunks_ && c + 1 >= slots_) {
                    auto tw = Clock::now();
                    dist::EventSynchronize(events[c + 1 - slots_]);
                    t_wait_dev += since(tw);
                    std::lock_guard lk(mu_);
                    state_[(c + 1) % slots_] = FREE;
                    cv_.notify_all();
                }
            }
            if (!events.empty()) {
                dist::EventSynchronize(events.back());  // this chip's last pull (not the other chips' queues)
            }
        } catch (...) {
            stop_reader();
            try {
                // the device must stop reading the slots before they are unpinned
                if (!events.empty()) {
                    dist::EventSynchronize(events.back());
                }
                std::lock_guard g(queue_mu());
                dist::Finish(cq);
            } catch (...) {
            }
            throw;
        }
        stop_reader();
        finished_ = true;
        stats_ = {
            {"bytes", double(total_)},
            {"chunks", double(n_chunks_)},
            {"batches", double(batches)},
            {"setup_s", setup_s_},
            {"read_s", read_s_},
            {"read_wall_s", read_wall_s_},
            {"finish_s", since(t0)},
            {"wait_read_s", t_wait_read},
            {"wait_device_s", t_wait_dev},
            {"since_start_s", since(t_start_)},
            {"gb_per_s_finish", total_ / std::max(1e-9, since(t0)) / 1e9},
            {"cores", double(ncores)},
            {"slots", double(slots_)},
            {"bytes_cached", double(bytes_cached_)},
            {"bytes_direct", double(bytes_direct_)},
            {"read_cached_s", read_cached_s_},
            {"read_direct_s", read_direct_s_}};
    }

    // statistics and per-GiB checksums of a finished pull (GIL held)
    nb::dict result() const {
        TT_FATAL(finished_, "ImagePuller::result before finish");
        nb::dict d;
        for (const auto& [k, v] : stats_) {
            d[k.c_str()] = v;
        }
        d["checksums"] = to_vec(sums_);
        return d;
    }

private:
    enum State { FREE, FILLING, READY };

    void read_loop() {
        auto t0 = Clock::now();
        for (uint32_t c = 0; c < n_chunks_; c++) {
            const uint32_t s = c % slots_;
            {
                std::unique_lock lk(mu_);
                cv_.wait(lk, [&] { return stop_ || state_[s] == FREE; });
                if (stop_) {
                    return;
                }
                state_[s] = FILLING;
            }
            const uint64_t off = uint64_t(c) * GB, len = std::min<uint64_t>(GB, total_ - off);
            const bool dio = off >= direct_from_;
            const int fd = dio ? fd_direct_ : fd_;
            auto tr = Clock::now();
            std::atomic<bool> ok{true};
            std::vector<std::thread> th;
            // O_DIRECT needs block-aligned lengths: round each thread's range to 1 MiB (the tail read may return short)
            const uint64_t unit = 1ull << 20;
            const uint64_t per = (len + threads_ * unit - 1) / (threads_ * unit) * unit;
            for (uint32_t k = 0; k < threads_; k++) {
                th.emplace_back([&, k] {
                    constexpr uint64_t BLK = 8ull << 20;
                    const uint64_t lo = std::min(len, k * per), hi = std::min(len, (k + 1) * per);
                    for (uint64_t a = lo; a < hi; a += BLK) {
                        const uint64_t n = std::min(BLK, hi - a);
                        if (partial_) {  // stream [off + a, off + a + n) -> the file ranges of its runs
                            if (!read_partial(direct_partial_ ? fd_direct_ : fd, slot_ptr_[s] + a, off + a, n,
                                              direct_partial_)) {
                                ok = false;
                                return;
                            }
                            continue;
                        }
                        const uint64_t want = dio ? (n + 4095) / 4096 * 4096 : n;
                        for (uint64_t got = 0; got < n;) {
                            ssize_t r = pread(fd, slot_ptr_[s] + a + got, want - got, off_t(off + a + got));
                            if (r <= 0) {
                                ok = false;
                                return;
                            }
                            got += r;
                        }
                        if (checksum_) {
                            hash_into(sums_, slot_ptr_[s] + a, n, off + a);
                        }
                    }
                });
            }
            for (auto& t : th) {
                t.join();
            }
            (dio ? read_direct_s_ : read_cached_s_) += since(tr);
            (dio ? bytes_direct_ : bytes_cached_) += len;
            read_s_ += since(tr);
            std::lock_guard lk(mu_);
            if (!ok) {
                failed_ = true;
                error_ = fmt::format("short read of {} in chunk {} (errno {})", path_, c, errno);
                cv_.notify_all();
                return;
            }
            state_[s] = READY;
            slot_chunk_[s] = c;
            cv_.notify_all();
        }
        read_wall_s_ = since(t0);
    }

    bool read_partial(int fd, uint8_t* dst, uint64_t x0, uint64_t n, bool direct = false) const {
        // first run ending after x0
        size_t r =
            std::upper_bound(
                runs_.begin(), runs_.end(), x0, [](uint64_t x, const Run& run) { return x < run.src + run.len; }) -
            runs_.begin();
        for (uint64_t x = x0, x1 = x0 + n; x < x1 && r < runs_.size(); r++) {
            const Run& run = runs_[r];
            const uint64_t e = std::min(x1, run.src + run.len), file = run.file + (x - run.src);
            if (direct) {
                // O_DIRECT: read the 4-KiB-aligned superset of [file, file + len) into an aligned bounce buffer, copy
                // the wanted bytes (block offsets in the file are only 64-B aligned)
                constexpr uint64_t A = 4096;
                const uint64_t lo = file / A * A, want = file + (e - x), hi = (want + A - 1) / A * A;
                uint8_t* b = bounce(hi - lo);
                uint64_t got = 0;
                while (lo + got < want) {
                    ssize_t k = pread(fd, b + got, (hi - lo) - got, off_t(lo + got));
                    if (k <= 0) {
                        return false;
                    }
                    got += k;
                }
                std::memcpy(dst + (x - x0), b + (file - lo), e - x);
                x = e;
                continue;
            }
            for (uint64_t got = 0; got < e - x;) {
                ssize_t k = pread(fd, dst + (x - x0) + got, (e - x) - got, off_t(file + got));
                if (k <= 0) {
                    return false;
                }
                got += k;
            }
            x = e;
        }
        return true;
    }

    // one 4-KiB-aligned bounce buffer per reader thread (grown on demand, kept for the thread's life)
    static uint8_t* bounce(uint64_t bytes) {
        struct Buf {
            uint8_t* p = nullptr;
            uint64_t n = 0;
            ~Buf() { std::free(p); }
        };
        thread_local Buf buf;
        if (buf.n < bytes) {
            std::free(buf.p);
            void* q = nullptr;
            TT_FATAL(posix_memalign(&q, 4096, bytes) == 0, "bounce buffer of {} bytes", bytes);
            buf.p = static_cast<uint8_t*>(q);
            buf.n = bytes;
        }
        return buf.p;
    }

    void stop_reader() {
        {
            std::lock_guard lk(mu_);
            stop_ = true;
            cv_.notify_all();
        }
        if (reader_.joinable()) {
            reader_.join();
        }
    }

    std::shared_ptr<dist::MeshDevice> mesh_;
    dist::MeshCoordinate coord_;
    std::string path_;
    uint32_t cores_n_, threads_;
    uint64_t direct_from_;
    bool checksum_;
    uint32_t stage_, max_runs_;
    bool direct_partial_ = false;
    uint32_t banks_ = 0, n_chunks_ = 0, slots_ = 0;
    uint32_t pcie_xy_enc_ = 0;
    uint64_t total_ = 0, file_size_ = 0;
    bool partial_ = false;
    std::vector<Seg> segs_;
    std::vector<Run> runs_;
    std::vector<std::atomic<uint64_t>> sums_;
    int fd_ = -1, fd_direct_ = -1;
    double read_cached_s_ = 0, read_direct_s_ = 0;
    uint64_t bytes_cached_ = 0, bytes_direct_ = 0;
    bool started_ = false, finished_ = false;
    std::map<std::string, double> stats_;
    std::vector<uint8_t*> slot_ptr_;
    std::vector<std::shared_ptr<experimental::PinnedMemory>> slot_pin_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<State> state_;
    std::vector<int64_t> slot_chunk_;
    bool stop_ = false, failed_ = false;
    std::string error_;
    std::thread reader_;
    double setup_s_ = 0, read_s_ = 0, read_wall_s_ = 0;
    Clock::time_point t_start_;
};

// ------------------------------------------------------------------------------------------------ introspection
nb::dict buffer_info(const ttnn::Tensor& t) {
    TT_FATAL(
        t.storage_type() == StorageType::DEVICE && t.is_allocated(), "buffer_info needs an allocated device tensor");
    Buffer* b = t.buffer();
    nb::dict d;
    d["address"] = uint64_t(b->address());
    d["size"] = uint64_t(b->size());
    d["page_size"] = uint64_t(b->page_size());
    d["aligned_page_size"] = uint64_t(b->aligned_page_size());
    d["aligned_size_per_bank"] = uint64_t(b->aligned_size_per_bank());
    d["num_pages"] = uint64_t(b->num_pages());
    d["dram"] = b->is_dram();
    d["layout"] = fmt::format("{}", b->buffer_layout());
    d["device_id"] = int(t.device()->get_device_ids().at(0));
    d["num_devices"] = int(t.device()->num_devices());
    std::vector<int> ids;
    for (auto id : t.device()->get_device_ids()) {
        ids.push_back(int(id));
    }
    d["device_ids"] = ids;  // a lock-step mesh tensor: the same address on each
    return d;
}

nb::dict jit_stats() {
    auto& tel = BuildCacheTelemetry::inst();
    nb::dict d;
    d["enabled"] = tel.is_enabled();
    d["srcs"] = tel.get_srcs_count();
    d["compiled"] = tel.get_compile_count();
    d["cache_hits"] = tel.get_cache_hit_count();
    d["jit_once_dedup"] = tel.get_jit_once_dedup_count();
    return d;
}

}  // namespace

void py_module(nb::module_& mod) {

    // ---- page-DMA (tiered latent cache): persistent kernels on service cores, pinned host page pool
    using pdma::PageDma;
    auto desc_span = [](const nb::bytes& b) {
        TT_FATAL(b.size() % sizeof(PdmaDesc) == 0, "descriptor buffer of {} B is not a multiple of {}", b.size(),
                 sizeof(PdmaDesc));  // fmt: skip
        return std::make_pair(reinterpret_cast<const PdmaDesc*>(b.c_str()), b.size() / sizeof(PdmaDesc));
    };
    nb::class_<PageDma>(mod, "PageDma")
        .def(
            "__init__",
            [](PageDma* self, std::shared_ptr<dist::MeshDevice> mesh, uintptr_t pool, uint32_t gib, uint32_t cores,
               uint32_t ring, uint32_t stage, const std::vector<std::vector<uint32_t>>& tensors, uint64_t done_off,
               std::pair<uint32_t, uint32_t> save_rc) {
                std::vector<pdma::Tensor> t;
                for (const auto& v : tensors) {
                    TT_FATAL(v.size() == 3, "tensor entries are (base, n_tiles, tile_bytes)");
                    t.push_back({v[0], v[1], v[2]});
                }
                new (self) PageDma(std::move(mesh), pool, gib, cores, ring, stage, t, done_off, save_rc, queue_mu());
            },
            nb::arg("mesh_device"), nb::arg("pool_address"), nb::arg("pool_gib"), nb::arg("cores_per_chip") = 4,
            nb::arg("ring") = 256, nb::arg("stage") = 32768, nb::arg("tensors"), nb::arg("done_offset"),
            nb::arg("save_chip") = std::make_pair(0u, 0u),
            "Page-DMA over a pinned pool of pool_gib x 1 GiB at pool_address; tensors = [(base, n_tiles, tile_bytes)] "
            "(interleaved DRAM buffers descriptors may name); done_offset = pool byte offset of the completion area "
            "(64 B per core per chip).")
        .def("start", [](PageDma& s, double timeout_s) { nb::gil_scoped_release g; s.start(timeout_s); },
             nb::arg("timeout_s") = 10.0, "Pin the pool for every chip, claim service cores, launch the kernels.")
        .def("load", [=](PageDma& s, const nb::bytes& d, double timeout_s, const std::vector<uint32_t>& chips) {
                 auto [p, n] = desc_span(d);
                 nb::gil_scoped_release g;
                 s.load(p, n, timeout_s, chips);
             }, nb::arg("descriptors"), nb::arg("timeout_s") = 10.0, nb::arg("chips") = std::vector<uint32_t>{},
             "Queue LOAD descriptors (PdmaDesc bytes) on every chip, or only on `chips` (mesh order; the sequence-sharded pool).")
        .def("save", [=](PageDma& s, const nb::bytes& d, double timeout_s, const std::vector<uint32_t>& chips) {
                 auto [p, n] = desc_span(d);
                 nb::gil_scoped_release g;
                 s.save(p, n, timeout_s, chips);
             }, nb::arg("descriptors"), nb::arg("timeout_s") = 10.0, nb::arg("chips") = std::vector<uint32_t>{},
             "Queue SAVE descriptors on the save chip, or on `chips` (mesh order; per-chip state, one call per chip).")
        .def("wait", [](PageDma& s, double timeout_s) { nb::gil_scoped_release g; s.wait(timeout_s); },
             nb::arg("timeout_s") = 10.0, "Wait until every queued descriptor completed.")
        .def("stats", [](PageDma& s) {
                 std::vector<std::vector<std::vector<uint32_t>>> out;
                 for (const auto& chip : s.stats()) {
                     out.emplace_back();
                     for (const auto& a : chip) {
                         out.back().push_back(std::vector<uint32_t>(a.begin(), a.end()));
                     }
                 }
                 return out;
             }, "[chip][core] = [done, tail, err, n_errors, bytes].")
        .def("stop", [](PageDma& s, double timeout_s) { nb::gil_scoped_release g; s.stop(timeout_s); },
             nb::arg("timeout_s") = 10.0, "STOP every kernel, wait for them to exit, release the cores, unpin.")
        .def_prop_ro("window_bytes", &PageDma::window_bytes);
    mod.def(
        "read_region",
        &read_region,
        nb::arg("mesh_device"),
        nb::arg("address"),
        nb::arg("bytes_per_bank"),
        nb::arg("page_size"),
        nb::arg("path"),
        "Dump DRAM [address, address + bytes_per_bank) of every bank to `path`, in interleaved page order.");
    mod.def(
        "pull_region",
        &pull_region,
        nb::arg("mesh_device"),
        nb::arg("address"),
        nb::arg("bytes_per_bank"),
        nb::arg("page_size"),
        nb::arg("path"),
        nb::arg("slots") = 2,
        nb::arg("num_cores") = 8,
        nb::arg("read_threads") = 16,
        "Stream an image written by read_region back into DRAM: Tensix cores pull 1 GB pinned hugepage chunks over "
        "PCIe while the host reads the next chunk from `path`.");
    mod.def(
        "read_image",
        &read_image,
        nb::arg("mesh_device"),
        nb::arg("segments"),
        nb::arg("path"),
        nb::arg("piece_bytes_per_bank") = 64ull << 20,
        nb::arg("threads") = 8,
        nb::arg("block_bytes") = BLOCK_DEFAULT,
        nb::arg("coord") = std::pair<uint32_t, uint32_t>{0, 0},
        "Dump DRAM segments [(lo, hi)] (bank offsets, 64 B aligned) of the chip at mesh coordinate `coord` to `path`, "
        "bank-major per segment (path \"\": checksums only). Returns bytes, timings and per-GiB checksums of the image "
        "byte stream.");
    mod.def(
        "checksum_file",
        &checksum_file,
        nb::arg("path"),
        nb::arg("threads") = 16,
        "Per-GiB checksums of a file (the same function read_image / ImagePuller compute).");
    mod.def(
        "device_block_hashes",
        &device_block_hashes,
        nb::arg("mesh_device"),
        nb::arg("segments"),
        nb::arg("block_bytes") = BLOCK_DEFAULT,
        nb::arg("num_cores") = 110,
        nb::arg("max_runs") = 32,
        nb::arg("stage_bytes") = 32768,
        "Hash DRAM segments on every chip of the mesh per block (same values as read_image's block_hashes for the same "
        "bytes): block_hashes = the first chip's, per_device = device id -> hashes, coords = device id -> (row, col).");
    mod.def(
        "file_block_hashes",
        &file_block_hashes,
        nb::arg("path"),
        nb::arg("segments"),
        nb::arg("banks") = 8,
        nb::arg("block_bytes") = BLOCK_DEFAULT,
        nb::arg("threads") = 16,
        "Block hashes of an image file (host).");
    mod.def("buffer_info", &buffer_info, nb::arg("tensor"), "Address / size / per-bank extent of a device tensor.");
    mod.def("jit_stats", &jit_stats, "Process-wide JIT build counters (sources, compiled, cache hits).");
    nb::class_<ImagePuller>(mod, "ImagePuller")
        .def(
            nb::init<
                std::shared_ptr<dist::MeshDevice>,
                const std::vector<std::pair<uint64_t, uint64_t>>&,
                std::string,
                uint32_t,
                uint32_t,
                uint32_t,
                bool,
                bool,
                uint32_t,
                uint32_t,
                uint64_t,
                const std::vector<uint64_t>&,
                uint64_t,
                const std::pair<uint32_t, uint32_t>&,
                bool>(),
            nb::arg("mesh_device"),
            nb::arg("segments"),
            nb::arg("path"),
            nb::arg("slots") = 2,
            nb::arg("num_cores") = 32,
            nb::arg("read_threads") = 16,
            nb::arg("direct") = false,
            nb::arg("checksum") = true,
            nb::arg("stage_bytes") = 65536,
            nb::arg("max_runs") = 32,
            nb::arg("direct_from") = ~0ull,
            nb::arg("only_blocks") = std::vector<uint64_t>{},
            nb::arg("block_bytes") = BLOCK_DEFAULT,
            nb::arg("coord") = std::pair<uint32_t, uint32_t>{0, 0},
            nb::arg("direct_partial") = false)
        .def("start", &ImagePuller::start, "Pin the hugepage slots and start reading the file ahead.")
        .def(
            "finish",
            &ImagePuller::finish,
            nb::call_guard<nb::gil_scoped_release>(),
            "Pull every chunk into DRAM (GIL released; one thread per chip).")
        .def("result", &ImagePuller::result, "Timings and per-GiB checksums of the finished pull.")
        .def_prop_ro("total_bytes", &ImagePuller::total_bytes);
}

}  // namespace ttnn::dram_image
