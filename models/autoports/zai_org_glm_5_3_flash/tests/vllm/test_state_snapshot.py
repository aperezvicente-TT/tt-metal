"""Stage 12: the prompt-end KDA checkpoint off the first-token path -- LatentHitDma(state_tensors=) saves a device
snapshot per chip into the pool's state region (PageDma.save(chips=[k])); StateTicket polls the host completion slots,
copies the used bytes out, untilizes; StackBackend.collect_snapshot rebuilds export_kda_state's layout. Host only:
RecordingPageDma executes the SAVEs against a host model of every chip's snapshot buffers."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.autoports.zai_org_glm_5_3_flash.tt import page_pool as PP

CHIPS = 4


class FakeTensor:
    def __init__(self, addr, pages, dtype):
        self.addr, self.padded_shape, self.shape, self.dtype = addr, [pages, 1, 32, 32], [pages, 1, 32, 32], dtype

    def buffer_address(self):
        return self.addr


def tilize_tile(logical: np.ndarray) -> np.ndarray:
    """[..., 32, 32] logical -> device face order bytes"""
    lead = logical.shape[:-2]
    return np.ascontiguousarray(logical.reshape(*lead, 2, 16, 2, 16).swapaxes(-3, -2)).reshape(*lead, 1024)


def _dma(state, devmem, gib=1):
    lat = [FakeTensor(0x100000 * (l + 1), 16 * 32, "DataType.BFLOAT16") for l in range(PP.DSA_LAYERS)]
    impl = PP.RecordingPageDma(None, apply_chips=True)
    dma = PP.LatentHitDma(None, lat, gib=gib, impl=impl, backing="anon", state_tensors=state, chips=CHIPS)

    def apply(desc, chips):
        for d in desc:
            if int(d["op"]) != PP.SAVE:
                continue
            i = int(d["tensor"]) - PP.DSA_LAYERS
            ch = chips[0] if chips else 0
            pb = dma.state[i][2]
            t0, n = int(d["tile0"]), int(d["n_tiles"])
            off = int(d["slab"]) * PP.GIB + int(d["slab_off"])
            dma.pool.array[off : off + n * pb] = devmem[ch][i][t0 : t0 + n].view(np.uint8).reshape(-1)

    impl.apply = apply
    return dma


def test_state_save_round_trips_every_chips_tiles_and_reuses_the_region():
    rng = np.random.default_rng(3)
    state = [FakeTensor(0x9000000, 24, "DataType.FLOAT32"), FakeTensor(0xA000000, 6, "DataType.BFLOAT16"),
             FakeTensor(0xB000000, 40, "DataType.BFLOAT16")]  # fmt: skip
    logical = [[rng.standard_normal((24, 32, 32)).astype(np.float32),
                rng.integers(0, 1 << 16, size=(6, 32, 32), dtype=np.uint16),
                rng.integers(0, 1 << 16, size=(40, 32, 32), dtype=np.uint16)] for _ in range(CHIPS)]  # fmt: skip
    devmem = [[tilize_tile(x) for x in per] for per in logical]
    dma = _dma(state, devmem)
    assert dma.state_ids == [11, 12, 13] and dma.chip_state_bytes == 24 * 4096 + 4096 * 3 + 40 * 2048
    t = dma.save_state([24, 6, 7])  # 7 of the 40 DSA tiles used
    assert t.done() and dma.dma.impl.chips[-CHIPS:] == [[k] for k in range(CHIPS)]  # every chip saves its own
    assert np.array_equal(t.tiles(0, np.float32), np.stack([logical[k][0] for k in range(CHIPS)]))
    assert np.array_equal(t.tiles(1, np.uint16), np.stack([logical[k][1] for k in range(CHIPS)]))
    assert np.array_equal(t.tiles(2, np.uint16), np.stack([logical[k][2][:7] for k in range(CHIPS)]))
    # region reuse: a new save while the previous ticket is not fetched -> the previous bytes are taken out first
    t1 = dma.save_state([24, 6, 1])
    for k in range(CHIPS):
        devmem[k][0][:] = 0  # the device buffers change after the second save
    t2 = dma.save_state([24, 6, 1])
    assert dma.stats["state_region_waits"] == 1 and t1.released
    assert np.array_equal(t1.tiles(0, np.float32), np.stack([logical[k][0] for k in range(CHIPS)]))
    assert not t2.tiles(0, np.float32).any()
    with pytest.raises(ValueError):
        dma.save_state([25, 6, 1])
    dma.close()


def test_ticket_waits_for_the_completion_slots():
    state = [FakeTensor(0x9000000, 4, "DataType.FLOAT32")]
    devmem = [[np.zeros((4, 1024), np.float32)] for _ in range(CHIPS)]
    dma = _dma(state, devmem)
    dma.dma.impl.layout = None  # the kernel has not completed anything yet
    t = dma.save_state([4])
    assert not t.done()
    with pytest.raises(TimeoutError):
        t.fetch(timeout_s=0.01)
    dma.close()


def test_slow_state_save_logs_its_queue_and_per_chip_completion(monkeypatch, capsys):
    """the KDA export timing diagnostic (Stage 14, 2026-10-28): a fetch that waits longer than GLM53_PAGE_DMA_STATE_LOG_S
    logs the descriptors ahead of the save, this save's descriptors and each chip's completion time; a fast one is silent"""
    import threading
    import time

    state = [FakeTensor(0x9000000, 4, "DataType.FLOAT32")]
    devmem = [[np.zeros((4, 1024), np.float32)] for _ in range(CHIPS)]
    dma = _dma(state, devmem)
    monkeypatch.setenv("GLM53_PAGE_DMA_STATE_LOG_S", "0.05")
    t = dma.save_state([4])  # completes at once (the recording kernel)
    t.fetch()
    assert "PAGE_DMA_STATE_WAIT" not in capsys.readouterr().out
    dma.dma.impl.layout = None  # from here the kernel completes nothing by itself
    t = dma.save_state([4])
    assert not t.done() and t.before is not None and t.done0 is not None

    def complete():
        time.sleep(0.15)
        for ch in range(CHIPS):
            for k in range(dma.cores):
                o = dma.dma.done_off + 64 * (ch * dma.cores + k)
                dma.pool.array[o : o + 4] = np.array([t.targets[ch][k]], np.uint32).view(np.uint8)

    th = threading.Thread(target=complete)
    th.start()
    t.fetch(timeout_s=5.0)
    th.join()
    out = capsys.readouterr().out
    assert "PAGE_DMA_STATE_WAIT" in out, out
    line = next(l for l in out.splitlines() if "PAGE_DMA_STATE_WAIT" in l)
    this = [sum(tt - b for tt, b in zip(tr, br)) for tr, br in zip(t.targets, t.before)]
    assert f"this save per chip {this}" in line and "chip done at [" in line and "None" not in line.split("chip done at")[1].split("]")[0]
    dma.close()


def test_collect_snapshot_layout_matches_the_pack_and_dsa_blob():
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend as SB

    g = torch.Generator().manual_seed(1)
    nl, n_s, n_c = 34, 2, 3
    hs = torch.randn(CHIPS, nl * n_s, 32, 32, generator=g)
    hc = torch.randn(CHIPS, 2 * n_c, 32, 32, generator=g).to(torch.bfloat16)
    hd = torch.randn(1, 10, 32, 32, generator=g).to(torch.bfloat16).expand(CHIPS, 10, 32, 32).contiguous()
    plan = dict(layers=list(range(nl)), n_s=n_s, n_c=n_c, n_tokens=300, units=3,
                dsa=[("L40.pools", 6, 0), ("L40.ring", 4, 6)], pages=[nl * n_s, 2 * n_c, 10])  # fmt: skip
    out = SB.collect_snapshot(plan, lambda i: (hs, hc, hd)[i])
    pack, dsa = out[0]["kda"], out[1]["dsa"]
    assert pack["kind"] == SB.KDA_PACK and pack["layers"] == list(range(nl)) and pack["n_s"] == n_s
    assert torch.equal(pack["S"], hs.reshape(CHIPS, nl, n_s, 32, 32))
    # conv: layer j = row j % 32 of the (j // 32)-th set of n_c tiles -> [chips, layers, tiles, 32]
    assert pack["C"].shape == (CHIPS, nl, n_c, 32)
    for j in (0, 5, 31, 32, 33):
        assert torch.equal(pack["C"][:, j], hc[:, (j // 32) * n_c : (j // 32 + 1) * n_c, j % 32, :])
    assert dsa["kind"] == SB.DSA_BLOB and dsa["replicated"] and dsa["units"] == 3 and dsa["n_tokens"] == 300
    assert torch.equal(dsa["tensors"]["L40.pools"], hd[:1, :6]) and torch.equal(dsa["tensors"]["L40.ring"], hd[:1, 6:10])


def test_collect_snapshot_sharded_dsa_is_the_exports_unit_layout(monkeypatch):
    """GLM53_DSA_SHARD (n29h7): the snapshot's DSA rows leave like export_dsa_state's -- unit-paged stores converted to
    the chip-independent global unit order (layout 'unit', one row), a ring kept once only when every chip's copy is
    equal -- never as a per-chip dump that a coincidentally equal set of rows would mark 'replicated'"""
    from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as SMod
    from models.autoports.zai_org_glm_5_3_flash.tt.stack_model import StackBackend as SB

    def to_units(per_chip, C, units, tpu=4):  # = stack_model.dsa_units_to_unit_order of the sharded trees
        out = per_chip.new_zeros(1, units * tpu, *per_chip.shape[2:])
        for u in range(units):
            c, k = u % C, u // C
            out[0, u * tpu : (u + 1) * tpu] = per_chip[c, k * tpu : (k + 1) * tpu]
        return out

    monkeypatch.setattr(SMod, "dsa_units_to_unit_order", to_units, raising=False)
    g = torch.Generator().manual_seed(2)
    C, units = 2, 3
    hs = torch.zeros(C, 2, 32, 32)
    hc = torch.zeros(C, 1, 32, 32).to(torch.bfloat16)
    pools = torch.randn(C, 12, 32, 32, generator=g).to(torch.bfloat16)  # 4 x units local tiles per chip (the plan)
    ring = torch.randn(1, 2, 32, 32, generator=g).to(torch.bfloat16).expand(C, 2, 32, 32).contiguous()  # equal copies
    hd = torch.cat([pools, ring], dim=1)
    plan = dict(layers=[0], n_s=2, n_c=1, n_tokens=300, units=units, shard_c=C,
                dsa=[("L3.pools", 12, 0, True), ("L3.ring", 2, 12, False)], pages=[2, 1, 14])  # fmt: skip
    out = SB.collect_snapshot(plan, lambda i: (hs, hc, hd)[i])
    blob = out[1]["dsa"]
    assert blob["layout"] == "unit" and blob["replicated"] and blob["chips"] == C
    assert torch.equal(blob["tensors"]["L3.pools"], to_units(pools, C, units))
    assert torch.equal(blob["tensors"]["L3.ring"], ring[:1])
    plan_rep = dict(plan, shard_c=None)  # the replicated pool: per-chip rows, deduplicated only when all equal
    blob2 = SB.collect_snapshot(plan_rep, lambda i: (hs, hc, hd)[i])[1]["dsa"]
    assert blob2["layout"] == "chip" and not blob2["replicated"] and blob2["tensors"]["L3.pools"].shape[0] == C
