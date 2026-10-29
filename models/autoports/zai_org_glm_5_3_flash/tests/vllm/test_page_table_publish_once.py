# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the generator publishes a page table with StackBackend.write(io.page_table) and then
set_page_table_host(kv, same table); on the stack backend both target the same device buffer, so the second write is
skipped when the value is unchanged (cf3g host profile: on the sharded pool each write converts 8 chips' local tables,
14 % of the engine thread), and still happens when the value differs. StackBackend runs as a host stand-in
(StackBackend.__new__ + the attributes write reads); the device copies are stubbed."""

from __future__ import annotations

import types

import torch

from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SM


def stand_in(monkeypatch, shard: bool):
    b = SM.StackBackend.__new__(SM.StackBackend)
    buf = types.SimpleNamespace(name="page_table", t=types.SimpleNamespace(shape=(2, 16)))
    kv = types.SimpleNamespace(extra={"io": types.SimpleNamespace(page_table=buf), "dsa_shard": (8, 100, 2) if shard else None},
                               num_blocks=64)  # fmt: skip
    b._kv, b.tp, b.mesh, b.chunk = kv, 1, None, 32
    writes = []
    b._write_local_tables = lambda buf_, v: writes.append(torch.as_tensor(v).clone())
    monkeypatch.setattr(SM.ttnn, "from_torch", lambda t, **kw: t, raising=False)
    monkeypatch.setattr(SM.ttnn, "copy_host_to_device_tensor", lambda src, dst: writes.append(src.clone()), raising=False)
    return b, kv, buf, writes


def test_sharded_publish_writes_the_local_tables_once(monkeypatch):
    b, kv, buf, writes = stand_in(monkeypatch, shard=True)
    v = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    b.write(buf, v)
    b.set_page_table_host(kv, v)
    assert len(writes) == 1
    v2 = v.clone()
    v2[1, 5] = 61
    b.set_page_table_host(kv, v2)  # a different table: written
    assert len(writes) == 2 and torch.equal(writes[-1], v2) and torch.equal(b._pt_host, v2.to(torch.int64))


def test_plain_publish_writes_once_and_a_change_still_lands(monkeypatch):
    b, kv, buf, writes = stand_in(monkeypatch, shard=False)
    v = torch.arange(32, dtype=torch.int32).reshape(2, 16)
    b.write(buf, v)
    n = len(writes)
    b.set_page_table_host(kv, v)
    assert len(writes) == n
    b.set_page_table_host(kv, v + 1)
    assert len(writes) > n and torch.equal(b._pt_host, (v + 1).to(torch.int64))
