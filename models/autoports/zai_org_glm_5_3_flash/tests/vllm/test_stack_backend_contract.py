# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host checks of the Stage 14b StackBackend (tt/stack_model.py) against the serving contract the Stage 9 adapter
relies on, without a device: the slot-op lists are plain Python over the cache's buffer addresses."""

from types import SimpleNamespace

import torch

from models.autoports.zai_org_glm_5_3_flash.tt import stack_model


def test_copy_slot_leaves_latent_pages_to_the_page_table():
    """ModelBackend.copy_slot contract: MLA latent pages are not copied, they follow the page-table row that
    Glm53Generator.remap_slots moves after copy_slot. The stack used to copy blocks along the rows at copy_slot time,
    into the destination's STALE row: a released slot keeps its row while vLLM gives those blocks to another live
    request, so a widths-ON compaction (GLM53_STACK_WIDTHS) overwrote that request's latent cache. A sparse_dsa layer
    must contribute no op to a slot copy or reset."""
    # padded_shape: what the stack's _pages() reads (the old code emitted copies 5 -> 7 and 6 -> 8 for this fixture)
    lat = SimpleNamespace(buffer_address=lambda: 1 << 20, shape=(16, 1, 64, 512), padded_shape=(16, 1, 64, 512))
    # slot 0 = the moving request (blocks 5, 6); slot 1 = a released slot whose stale row lists blocks 7, 8, which
    # vLLM has since given to another live request
    backend = SimpleNamespace(seq=[{"sparse": "dsa", "L": 3}], chunk=4, _pt_host=torch.tensor([[5, 6], [7, 8]]))
    kv = SimpleNamespace(extra={"latent": {0: lat}}, num_blocks=16, blocks_per_seq=2, states=[[None]])
    assert stack_model.StackBackend._state_ops(backend, kv, 1, 0) == []  # copy slot 0 -> 1
    assert stack_model.StackBackend._state_ops(backend, kv, 1, None) == []  # reset slot 1
