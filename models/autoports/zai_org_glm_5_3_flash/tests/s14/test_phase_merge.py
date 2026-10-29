# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host test of tt/phase_merge.py (no device): the composite's source layout and argument bases
(GLM53_DSA_SHARD_LINK_LANES, dsa/shard_w16/levers_design.md section 6)."""

import re

import pytest

import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import phase_merge as PM

CORE = (3, 4)


def _kd(text, ct, rt, defines=(), writer=True, cores=(CORE,)):
    rta = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        rta[x][y] = [v + 100 * i for v in rt]
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    cfg = ttnn.WriterConfigDescriptor() if writer else ttnn.ReaderConfigDescriptor()
    return ttnn.KernelDescriptor(kernel_source=text, source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                                 core_ranges=crs, compile_time_args=list(ct), runtime_args=rta, config=cfg,
                                 defines=list(defines))  # fmt: skip


SEND = """#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
constexpr uint32_t A = get_compile_time_arg_val(PH_CT0 + 0);
void kernel_main() { size_t a = ph_rt0; (void)a; }
"""
READ = """#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
void kernel_main() { (void)get_arg_val<uint32_t>(ph_rt0 + 0); }
"""


def test_composite_layout_and_bases():
    k0 = _kd(SEND, [7, 8, 9], [1, 2, 3, 4], defines=[("FS_CONS", "1"), ("FS_AFTER_SEM", "1")])
    k1 = _kd(SEND, [5], [10, 11], defines=[("FS_SCATTER", "1")])
    k2 = _kd(SEND, [6, 6], [20], defines=[("FAR_LATE_OPEN", "1")], cores=(CORE, (5, 6)))
    out = PM.merge([(k0, {"FS_AFTER_SEM"}), (k1, set()), (k2, set())], CORE, name="t")
    src = str(out.kernel_source)
    # hoisted once, before the first namespace
    first_ns = src.index("namespace ph0 {")
    for inc in ('#include "api/dataflow/dataflow_api.h"', '#include "tt_metal/fabric/hw/inc/packet_header_pool.h"'):
        assert src.index(inc) < first_ns
        assert src[:first_ns].count(inc) == 1
    assert src.index("#define MK_PHASED 1") < first_ns
    # bases: compile-time 0 / 3 / 4, runtime 0 / 4 / 6
    assert re.findall(r"constexpr uint32_t PH_CT0 = (\d+);", src) == ["0", "3", "4"]
    assert re.findall(r"uint32_t ph_rt0 = (\d+);", src) == ["0", "4", "6"]
    assert list(out.compile_time_args) == [7, 8, 9, 5, 6, 6]
    assert list(out.runtime_args[CORE[0]][CORE[1]]) == [1, 2, 3, 4, 10, 11, 20]  # this core's args only
    # defines: per phase, dropped ones never emitted, every one undefined after its phase
    assert "#define FS_AFTER_SEM" not in src
    for d in ("FS_CONS", "FS_SCATTER", "FAR_LATE_OPEN"):
        assert src.count(f"#define {d} 1") == 1 and src.count(f"#undef {d}") == 1
        assert src.index(f"#define {d} 1") < src.index(f"#undef {d}")
    # the phases in order, the header pool reset between fabric phases
    body = src[src.index("void kernel_main() {"):]
    assert body.index("ph0::phase_main();") < body.index("ph1::phase_main();") < body.index("ph2::phase_main();")
    assert body.count("PacketHeaderPool::reset();") == 2
    assert not out.defines


def test_reader_composite_no_reset_and_one_role():
    r0 = _kd(READ, [1], [5], writer=False)
    r1 = _kd(READ, [2], [6], writer=False)
    out = PM.merge([(r0, set()), (r1, set())], CORE)
    assert "PacketHeaderPool::reset" not in str(out.kernel_source)
    w = _kd(SEND, [1], [1])
    with pytest.raises(AssertionError):
        PM.merge([(r0, set()), (w, set())], CORE)  # writer + reader in one composite


def test_probe_phase_refused_and_restrict():
    p = _kd(SEND, [1], [1, 2, 3], defines=[("MK_PROBE_ARG", "1")])
    with pytest.raises(AssertionError):
        PM.merge([(p, set())], CORE)
    two = _kd(SEND, [1], [9], cores=(CORE, (5, 6)))
    rest = PM.restrict(two, {CORE})
    assert PM._cores(rest) == {(5, 6)} and list(rest.runtime_args[5][6]) == [109]
    assert PM.restrict(two, {CORE, (5, 6)}) is None
