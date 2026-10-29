# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PMK_RS8_QSLICE's k4d3 default applies only where P8 is the TP8 RS8 (14's SHOULD 2026-10-27T14:36:05Z): a DEFAULTED
Q = 4 resolves to 1 on a pair / single chip or with PMK_O_RECTS; an explicit Q > 1 there is still passed through (and
refused by _build_collectives' assert); the TP8 path is unchanged."""

from types import SimpleNamespace

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import k4_defaults as K4D
from models.autoports.zai_org_glm_5_3_flash.tt import pmk_assembly as ASM


def _q(n, o_rects=0):
    return ASM.PmkProgramA._rs8_qslice(SimpleNamespace(n=n, o_rects=o_rects))


@pytest.fixture
def defaulted(monkeypatch):
    monkeypatch.setattr(K4D, "K4_FROM_DEFAULT", frozenset(K4D.K4_FROM_DEFAULT | {"PMK_RS8_QSLICE"}))
    monkeypatch.setenv("PMK_RS8_QSLICE", K4D.K4_DEFAULTS["PMK_RS8_QSLICE"])


@pytest.fixture
def explicit(monkeypatch):
    monkeypatch.setattr(K4D, "K4_FROM_DEFAULT", frozenset(K4D.K4_FROM_DEFAULT - {"PMK_RS8_QSLICE"}))
    monkeypatch.setenv("PMK_RS8_QSLICE", "4")


def test_default_is_4():
    assert K4D.K4_DEFAULTS["PMK_RS8_QSLICE"] == "4"


@pytest.mark.parametrize("n", [1, 2])
def test_defaulted_q_resolves_to_1_below_tp8(defaulted, n):
    assert _q(n) == 1


def test_defaulted_q_resolves_to_1_with_o_rects(defaulted):
    assert _q(8, o_rects=4) == 1


def test_tp8_path_unchanged_defaulted(defaulted):
    assert _q(8) == 4


@pytest.mark.parametrize("n,o_rects", [(2, 0), (1, 0), (8, 4)])
def test_explicit_q_passes_through_for_the_assert(explicit, n, o_rects):
    assert _q(n, o_rects) == 4


def test_tp8_path_unchanged_explicit(explicit):
    assert _q(8) == 4


def test_changed_value_is_not_the_default(defaulted, monkeypatch):
    monkeypatch.setenv("PMK_RS8_QSLICE", "2")
    assert not K4D.from_default("PMK_RS8_QSLICE")
    assert _q(2) == 2


def test_q_off_everywhere(monkeypatch):
    monkeypatch.setenv("PMK_RS8_QSLICE", "0")
    assert _q(2) == 1 and _q(8) == 1
