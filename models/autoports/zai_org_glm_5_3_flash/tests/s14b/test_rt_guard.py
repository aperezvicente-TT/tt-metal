# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: tt/rt_guard.py refuses a launch that generic_op's program cache would serve with a program built for
fewer runtime args (the 2026-10-19 23:35Z chip-3 hang), and accepts equal arg counts."""
import ttnn
from models.autoports.zai_org_glm_5_3_flash.tt import rt_guard


def _desc(n_args: int, src: str = "models/autoports/zai_org_glm_5_3_flash/tt/kernels/head/slot_op.cpp"):
    core = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])
    rt = ttnn.RuntimeArgs()
    rt[0][0] = list(range(n_args))
    kd = ttnn.KernelDescriptor(kernel_source=src, source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                               core_ranges=core, compile_time_args=[], runtime_args=rt, defines=[("T", "1")],
                               config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0,
                                                                        noc=ttnn.NOC.RISCV_0_default))  # fmt: skip
    return ttnn.ProgramDescriptor(kernels=[kd], semaphores=[], cbs=[])


def test_same_count_ok_other_count_refused(expect_error):
    rt_guard._SEEN.clear()
    rt_guard.checked(_desc(49))
    rt_guard.checked(_desc(49))
    with expect_error(RuntimeError, "rt_guard"):
        rt_guard.checked(_desc(97))


def test_other_signature_independent():
    rt_guard._SEEN.clear()
    rt_guard.checked(_desc(49))
    rt_guard.checked(_desc(97, src="models/autoports/zai_org_glm_5_3_flash/tt/kernels/head/embed_dm0.cpp"))
