// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// island_probe: one-shot raw-UMD read of island routers on one chip. It does not open the device as
// tt-metal does (no CHIP_IN_USE, no reset, no power-state change), so it can run while another
// process holds the chip. It reads, it never polls.
//
//   island_probe <pci_device_id> <addr> <bytes> <x,y> [<x,y> ...]
//
// Prints, per core (translated NoC coordinates), the words at addr, then the chip's AICLK.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tt-umd/tt_device/tt_device.hpp"
#include "tt-umd/types/core_coordinates.hpp"

int main(int argc, char** argv) {
    using namespace tt::umd;
    if (argc < 5) {
        std::fprintf(stderr, "usage: island_probe <pci_device_id> <addr> <bytes> <x,y>...\n");
        return 2;
    }
    const int dev = std::atoi(argv[1]);
    const uint64_t addr = std::strtoull(argv[2], nullptr, 0);
    const uint32_t bytes = std::strtoul(argv[3], nullptr, 0);
    auto d = TTDevice::create(dev);
    d->init_tt_device();
    for (int i = 4; i < argc; i++) {
        unsigned x = 0, y = 0;
        if (std::sscanf(argv[i], "%u,%u", &x, &y) != 2) {
            return 2;
        }
        std::vector<uint32_t> v((bytes + 3) / 4, 0);
        d->read_from_device(v.data(), tt_xy_pair(x, y), addr, bytes);
        // A router's status word stays in L1 after its core is put in reset: report the soft-reset
        // register too (a running router has ERISC0/1 out of reset).
        const uint32_t rst = d->get_risc_reset_state(CoreCoord(x, y, tt::CoreType::ETH, tt::CoordSystem::TRANSLATED));
        std::printf("core %u,%u: reset %08x", x, y, rst);
        for (auto w : v) {
            std::printf(" %08x", w);
        }
        std::printf("\n");
    }
    std::printf("aiclk %u\n", d->get_clock());
    return 0;
}
