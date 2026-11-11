// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// island_up: brings up the TT-Fabric routers of every visible chip (a fabric "island"), writes an
// island descriptor (JSON) that client processes connect from, and exits with the routers left
// running (INIT_FABRIC without TERMINATE_FABRIC).
//
//   island_up --output island.json [--payload 8704] [--epoch N]
//
// The descriptor holds, per chip (keyed by ASIC unique id): its fabric node id, the routing table
// its routers run with (routing_l1_info_t, read back from a router's L1), and per router channel the
// worker connection info (fabric_connection_info_t, read back from the device-init connection table
// in Tensix L1), direction and peer. Plus the kernel defines, packet sizes and the router status
// address and READY value.

#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <internal/fabric.hpp>
#include "hostdevcommon/fabric_common.h"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include <llrt/tt_cluster.hpp>
#include "tt_metal/fabric/fabric_context.hpp"
#include "tt_metal/fabric/fabric_builder_context.hpp"

namespace {

std::string hex(const void* p, size_t n) {
    const auto* b = static_cast<const uint8_t*>(p);
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        s += fmt::format("{:02x}", b[i]);
    }
    return s;
}

std::string arg(int argc, char** argv, const std::string& name, const std::string& def) {
    for (int i = 1; i + 1 < argc; i++) {
        if (name == argv[i]) {
            return argv[i + 1];
        }
    }
    return def;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    const std::string out = arg(argc, argv, "--output", "island.json");
    const uint32_t payload = std::stoul(arg(argc, argv, "--payload", "8704"));
    std::random_device rd;
    const uint64_t epoch = std::stoull(arg(argc, argv, "--epoch", std::to_string(uint64_t(rd()) << 32 | rd())));
    setenv("TT_METAL_SLOW_DISPATCH_MODE", "1", 1);

    const auto t0 = std::chrono::steady_clock::now();
    FabricRouterConfig rc;
    rc.max_packet_payload_size_bytes = payload;
    SetFabricConfig(
        FabricConfig::FABRIC_2D,
        FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
        std::nullopt,
        FabricTensixConfig::DISABLED,
        FabricUDMMode::DISABLED,
        FabricManagerMode::INIT_FABRIC,
        rc);
    const auto shape = distributed::SystemMesh::instance().shape();
    auto mesh = distributed::MeshDevice::create(distributed::MeshDeviceConfig(shape));
    const double up_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    auto& ctx = MetalContext::instance();
    const auto& cluster = ctx.get_cluster();
    const auto& hal = ctx.hal();
    const auto& cp = ctx.get_control_plane();
    const auto& bctx = cp.get_fabric_context().get_builder_context();
    const auto [sync_addr, ready] = bctx.get_fabric_router_sync_address_and_status();
    const uint32_t eth_rt = hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::ROUTING_TABLE);
    const uint32_t tx_rt = hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::ROUTING_TABLE);
    const uint32_t tx_conn =
        hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::TENSIX_FABRIC_CONNECTIONS);

    std::ostringstream js;
    js << "{\n  \"format_version\": 1,\n";
    js << fmt::format("  \"epoch\": {},\n  \"island_id\": \"{:016x}\",\n", epoch, epoch * 0x9E3779B97F4A7C15ull);
    js << fmt::format("  \"mesh_shape\": [{}, {}],\n", shape[0], shape[1]);
    js << fmt::format(
        "  \"max_payload\": {},\n  \"packet_header_size\": {},\n",
        get_tt_fabric_max_payload_size_bytes(),
        get_tt_fabric_packet_header_size_bytes());
    js << fmt::format("  \"router_status_addr\": {},\n  \"router_ready\": {},\n", sync_addr, ready);
    js << fmt::format("  \"eth_routing_table_addr\": {},\n  \"tensix_routing_table_addr\": {},\n", eth_rt, tx_rt);
    js << fmt::format("  \"routing_table_size\": {},\n", sizeof(routing_l1_info_t));
    js << "  \"kernel_defines\": {";
    {
        bool first = true;
        for (auto& [k, v] : tt::tt_metal::internal::get_fabric_kernel_defines(FabricApiType::Mesh)) {
            js << (first ? "" : ", ") << fmt::format("\"{}\": \"{}\"", k, v);
            first = false;
        }
        for (auto& [k, v] : cp.get_fabric_kernel_defines()) {
            js << (first ? "" : ", ") << fmt::format("\"{}\": \"{}\"", k, v);
            first = false;
        }
    }
    js << "},\n  \"nodes\": [\n";
    const auto& uids = cluster.get_unique_chip_ids();
    bool first_node = true;
    for (auto* dev : mesh->get_devices()) {
        const tt::ChipId chip = dev->id();
        const auto node = cp.get_fabric_node_id_from_physical_chip_id(chip);
        const CoreCoord w0 =
            cluster.get_virtual_coordinate_from_logical_coordinates(chip, CoreCoord(0, 0), tt::CoreType::WORKER);
        tensix_fabric_connections_l1_info_t conns{};
        cluster.read_core(&conns, sizeof(conns), tt_cxy_pair(chip, w0), tx_conn);
        js << (first_node ? "" : ",\n") << "    {";
        first_node = false;
        js << fmt::format(
            "\"asic_id\": {}, \"physical_chip_id\": {}, \"mesh_id\": {}, \"chip_id\": {}, ",
            uids.at(chip),
            chip,
            *node.mesh_id,
            node.chip_id);
        js << "\"routers\": [";
        bool first_r = true;
        std::string routing_table;
        for (const auto& [chan, dir] : cp.get_active_fabric_eth_channels(node)) {
            const CoreCoord v = cluster.get_virtual_eth_core_from_channel(chip, chan);
            const CoreCoord l = cluster.get_soc_desc(chip).get_eth_core_for_channel(chan, tt::CoordSystem::LOGICAL);
            const auto [peer_chip, peer_core] = cluster.get_connected_ethernet_core({chip, l});
            const auto peer_chan = cluster.get_soc_desc(peer_chip).logical_eth_core_to_chan_map.at(peer_core);
            uint32_t status = 0;
            cluster.read_core(&status, 4, tt_cxy_pair(chip, v), sync_addr);
            std::vector<uint8_t> rt(sizeof(routing_l1_info_t));
            cluster.read_core(rt.data(), rt.size(), tt_cxy_pair(chip, v), eth_rt);
            if (routing_table.empty()) {
                routing_table = hex(rt.data(), rt.size());
            }
            const auto& c = conns.read_only[chan];
            js << (first_r ? "" : ", ")
               << fmt::format(
                      "{{\"eth_chan\": {}, \"direction\": {}, \"noc_x\": {}, \"noc_y\": {}, \"peer_asic_id\": {}, "
                      "\"peer_eth_chan\": {}, \"status\": {}, \"conn_valid\": {}, \"conn\": "
                      "{{\"edm_buffer_base_addr\": {}, "
                      "\"edm_connection_handshake_addr\": {}, \"edm_worker_location_info_addr\": {}, "
                      "\"buffer_index_semaphore_id\": {}, \"buffer_size_bytes\": {}, \"edm_direction\": {}, "
                      "\"edm_noc_x\": {}, \"edm_noc_y\": {}, \"num_buffers_per_channel\": {}, "
                      "\"worker_free_slots_stream_id\": {}}}}}",
                      chan,
                      int(dir),
                      v.x,
                      v.y,
                      uids.at(peer_chip),
                      peer_chan,
                      status,
                      (conns.valid_connections_mask >> chan) & 1,
                      c.edm_buffer_base_addr,
                      c.edm_connection_handshake_addr,
                      c.edm_worker_location_info_addr,
                      c.buffer_index_semaphore_id,
                      c.buffer_size_bytes,
                      int(c.edm_direction),
                      int(c.edm_noc_x),
                      int(c.edm_noc_y),
                      int(c.num_buffers_per_channel),
                      c.worker_free_slots_stream_id);
            first_r = false;
        }
        js << "], \"routing_table\": \"" << routing_table << "\"}";
    }
    js << fmt::format("\n  ],\n  \"bringup_s\": {:.3f}\n}}\n", up_s);
    {
        const std::string tmp = out + ".tmp";
        std::ofstream f(tmp);
        f << js.str();
        f.close();
        std::rename(tmp.c_str(), out.c_str());
    }
    fmt::print(
        "island_up: {} chips, mesh {}x{}, routers up in {:.2f} s, descriptor {}\n",
        mesh->get_devices().size(),
        shape[0],
        shape[1],
        up_s,
        out);
    std::fflush(stdout);
    const auto tc = std::chrono::steady_clock::now();
    mesh->close();
    mesh.reset();
    fmt::print(
        "island_up: mesh closed in {:.2f} s\n",
        std::chrono::duration<double>(std::chrono::steady_clock::now() - tc).count());
    return 0;
}
