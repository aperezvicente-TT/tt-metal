# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 12 x 14 (host-only): the DSA host export's chip-independent unit order under GLM53_DSA_SHARD
(tt/stack_model.py dsa_units_to_unit_order / dsa_unit_order_to_chips; served_1m_memory_plan.md E, shard_tp8_design.md 6.2).

    no-tt-devices python -m pytest --noconftest -q models/autoports/zai_org_glm_5_3_flash/tests/test_dsa_unit_order.py
"""

import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import dsa_unit_order_to_chips, dsa_units_to_unit_order

TPU = 4


def _per_chip(C, local_units, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(C, local_units * TPU, 32, 32, generator=g).to(torch.bfloat16)


@pytest.mark.parametrize("C,U", [(8, 5), (8, 16), (2, 3), (1, 4)])
def test_unit_order_is_chip_independent_and_round_trips(C, U):
    local = -(-U // C)  # local units per chip the export read (ceil(U / C))
    per = _per_chip(C, local)
    uo = dsa_units_to_unit_order(per, C, U, TPU)
    assert tuple(uo.shape) == (1, U * TPU, 32, 32)
    for u in range(U):  # unit u came from chip u % C, local unit u // C
        assert torch.equal(uo[0, u * TPU : (u + 1) * TPU], per[u % C, (u // C) * TPU : (u // C + 1) * TPU])
    back = dsa_unit_order_to_chips(uo, C, local * TPU, TPU)
    for c in range(C):
        for k in range(local):
            u = k * C + c
            want = per[c, k * TPU : (k + 1) * TPU] if u < U else torch.zeros(TPU, 32, 32, dtype=torch.bfloat16)
            assert torch.equal(back[c, k * TPU : (k + 1) * TPU], want)
    # the same blob restores into a cache of another C (a restart with 2 chips, or the replicated pool C = 1)
    for C2 in (1, 2, 4):
        local2 = -(-U // C2)
        b2 = dsa_unit_order_to_chips(uo, C2, local2 * TPU, TPU)
        assert torch.equal(dsa_units_to_unit_order(b2, C2, U, TPU), uo)


def test_all_units_when_none_requested_and_bounds():
    per = _per_chip(4, 3)
    uo = dsa_units_to_unit_order(per, 4, None, TPU)
    assert tuple(uo.shape) == (1, 12 * TPU, 32, 32)
    with pytest.raises(ValueError, match="needs local unit"):
        dsa_units_to_unit_order(per, 4, 13, TPU)
    with pytest.raises(ValueError, match="chip rows"):
        dsa_units_to_unit_order(per, 8, 4, TPU)
    with pytest.raises(ValueError, match="exceeds"):
        dsa_unit_order_to_chips(uo, 4, 2 * TPU, TPU)
