# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""StackBackend.prefill_group never builds a GroupStack at request time: a group size with no group built at prewarm
returns NotImplemented (the caller runs the chunks one by one) and counts prefill_group_cold (realgate9 2026-10-25:
GROUP_SIZES 4,2 with PREWARM 4@8192 built GroupStack(2) after the captures; the next replay was refused). Host only."""
import types

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend


class _Grp:
    def __init__(self, n):
        self.n = n

    def windows(self, chunk_start, n_valid):
        return (chunk_start,)

    def ready(self, ts, head=True):
        return False


def _backend(groups=None, owner=None):
    be = types.SimpleNamespace(pmk=object(), counters={}, prefill_group_ok=lambda n: n >= 2)
    if groups is not None:
        be._pf_groups = groups
        be._pf_groups_owner = be.pmk if owner is None else owner

    def built_at_request_time(n):
        raise AssertionError(f"GroupStack({n}) built at request time")

    be.prefill_group_obj = built_at_request_time
    return be


def _call(be, n):
    return StackBackend.prefill_group(be, None, kv_cache=None, slot=0, chunk_start=0, n_valid=[2048] * n, logits_rows="none")


@pytest.mark.parametrize("groups", [None, {4: _Grp(4)}])
def test_an_unprewarmed_size_is_cold_and_builds_nothing(groups):
    be = _backend(groups)
    assert _call(be, 2) is NotImplemented
    assert be.counters["prefill_group_cold"] == 1


def test_groups_of_another_chain_are_cold():
    be = _backend({2: _Grp(2)}, owner=object())
    assert _call(be, 2) is NotImplemented
    assert be.counters["prefill_group_cold"] == 1


def test_a_prewarmed_size_reaches_its_readiness_check():
    be = _backend({2: _Grp(2)})
    assert _call(be, 2) is NotImplemented  # built but not captured for these windows: cold, still nothing built
    assert be.counters["prefill_group_cold"] == 1
