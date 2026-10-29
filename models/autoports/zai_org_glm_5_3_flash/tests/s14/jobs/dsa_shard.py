# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""serve.py job (Stage 14, the pair / the TP8 2x4 mesh): the SEQUENCE-SHARDED DSA state + split-K decode
(tt/tp_dsa_shard.py TPMlaShard, tt/dsa_shard.py, kernels/mla/dsa_shard.hpp) vs the REPLICATED TPMla(dsa) program
(Stage 14's validated TP DSA decode) on the same prompt / positions, both traced. The replicated case holds every latent
row and pooled key on every chip; the sharded case gives chip r the units u = r mod C (C = 2 on the pair, 8 on the 2x4
mesh: rank r = 4 row + col; local blocks, a local page table, local pooled keys; the ring replicated), built from the
same CPU oracle state (tests/s7b/golden_mla.dsa_state) through ShardMap. synthetic=True (the 1M rung: the float64 model
oracle cannot build that prompt, as tests/s14/jobs/dsa_long.py): random bf16 latent rows and random pooled keys (seed),
the ring zero; the oracle rows are replaced by an emulation on the device's own intermediates (info).

Rows (GLM53_STAGE_EVIDENCE pcc.jsonl / tests.jsonl / perf.jsonl):
  dsa_shard_vs_replicated: per program s1 bitwise across chips; the sharded s1 / o_lat (this chip's heads) vs the
      replicated program (bar 0.9999); the selection threshold key of every sparse token bitwise (the sharded global
      T_g from the program's 10th message vs the replicated program's local threshold: equal thresholds on the same
      scores select the same set unless a tie sits on T, reported); the new token's latent row and pooled key written
      on the owner chip only, bitwise vs the replicated writes; ms / step of both (traced replays);
  dsa_shard_golden: s1 of both programs vs the fp32 oracle (bar 0.995, as dsa_tp).
ARGS: pos (list of positions, <= MK.SHARD_T_MAX = 32 slots; rev 60: t > 4 runs the token lanes, both programs built as
the stack builds them -- TPMla ts (32,), the t-row call on the 32-row FabricPhase, TPMlaShard stream_ts (t,); the
replicated state block-vectorised as RepSynth's), replays (2000), soak_s (0), dsa_kb / rings (the indexer readers' K blocks /
MkMla ring budgets; defaults as dsa_tp), synthetic (False), seed (5), probe (False: rev 59, both programs built with
MkMla(probe=True); the sharded program's per-chip phase marks / spans -- tt/shard_plan.py decode_probe: the local
select, AG rounds 1 / 2, the merge, the q assembly, pass 0 / 1, RS rounds 1 / 2, finalize, the o_proj all-reduce -- go
into its dsa_shard_timing row, the replicated program's MkMla decode_probe into its row; probe flushes cost a little:
time with probe off), env ({GLM53_*}), det (0: N poisoned replays per program -- NaN over every output, replay, each
bitwise vs the first replay; the timing rows' det field), dump (a path prefix: every output per chip -> <dump>_<program>.pt,
the cross-run bitwise gates, e.g. GLM53_DSA_SHARD_LINK_LANES 0 vs 1). The first device run is WATCHED (new kernels)."""

import importlib
import os
import time

import torch

import ttnn

ENV = {k: str(v) for k, v in (ARGS.get("env") or {}).items()}  # noqa: F821
old = {k: os.environ.get(k) for k in ENV}
os.environ.update(ENV)
try:
    from models.autoports.zai_org_glm_5_3_flash.tests.common import int_tensor, pcc
    from models.autoports.zai_org_glm_5_3_flash.tests.s7a.traced_harness import Traced
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import dev_mk_dsa as DSA
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import golden_mla as G
    from models.autoports.zai_org_glm_5_3_flash.tests.s7b import mla_common as MC
    from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DS
    from models.autoports.zai_org_glm_5_3_flash.tt import fabric_stream as FS
    from models.autoports.zai_org_glm_5_3_flash.tt import mk_mla as MK
    from models.autoports.zai_org_glm_5_3_flash.tt import tp_dsa_shard as TPDS
    from models.autoports.zai_org_glm_5_3_flash.tt import tp_mla as TPM

    for m in (MK, TPM, FS, DS, TPDS, MC, G, DSA):
        importlib.reload(m)

    mesh = MESH  # noqa: F821
    pos = [int(v) for v in ARGS.get("pos", [32768 - 1, 4098])]  # noqa: F821
    replays = int(ARGS.get("replays", 2000))  # noqa: F821
    soak_s = float(ARGS.get("soak_s", 0))  # noqa: F821
    assert 1 <= len(pos) <= MK.SHARD_T_MAX, pos
    ndev = int(mesh.get_num_devices())
    assert ndev in (2, 8) and tuple(mesh.shape) in ((1, 2), (2, 1), (2, 4)), ("the pair or the 2x4 mesh", mesh.shape)
    synthetic = bool(ARGS.get("synthetic", False))  # noqa: F821
    seed = int(ARGS.get("seed", 5))  # noqa: F821
    probe = bool(ARGS.get("probe", False))  # noqa: F821
    # dense_window (GLM53_DSA_DENSE_WINDOW's unit form): the sharded program gets the replicated window of every slot;
    # an all-dense launch (every pos + 1 <= 2,051) then runs its one-pass window path (att_dsa_shard_dm0.cpp)
    dense_window = bool(ARGS.get("dense_window", False))  # noqa: F821
    MOCK = os.environ.get("GLM53_MOCK") == "1"
    rows = MC.Rows(DSA.LAYER)
    tr = Traced(mesh)
    per_chip = lambda t: [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]  # noqa: E731
    per_chip_u32 = lambda t: [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]  # noqa: E731
    h0 = lambda t: MK._host0(t).double()  # noqa: E731
    knobs = dict(rev=MK.MLA_REV, pos=pos, tp=ndev, mesh=list(mesh.shape), shard_unit=DS.UNIT, topk=DS.TOPK,
                 synthetic=synthetic, probe=probe)  # fmt: skip

    def soak(tr_, seconds):
        n, t_end = 0, time.time() + seconds
        while time.time() < t_end:
            tr_.replay()
            n += 1
        return n

    L = MC.layer_of(mesh, CTX, DSA.LAYER)  # noqa: F821
    attn = L.attn
    W = dict(
        qa_kva=h0(attn.ds_qa_kva.weight).reshape(-1, attn.ds_qa_kva.n_pad)[:, : attn.ds_qa_kva.n],
        qa_norm=h0(attn.qa_norm).reshape(-1)[: attn.q_lora],
        kva_norm=h0(attn.kva_norm).reshape(-1)[: attn.kv_lora],
        qb=h0(attn.ds_qb.weight).reshape(attn.q_lora, -1)[:, : attn.ds_qb.n],
        w_uk=h0(attn.w_uk).reshape(attn.h, attn.dqk, attn.kv_lora),
        w_uv=h0(attn.w_uv).reshape(attn.h, attn.kv_lora, attn.dv),
        o=h0(attn.ds_o.weight).reshape(attn.h * attn.dv, -1)[:, : attn.ds_o.n],
    )
    ix = MK.DsaIndexer(mesh, L.source, DSA.LAYER)
    t0 = time.time()
    t = len(pos)
    bs = attn.block_size
    cap = -(-(max(pos) + 1) // bs) * bs
    # cap (ARGS, the w1 decode gap row 2026-10-24 03:48:31Z): size the page table / cache / pooled keys for a longer max
    # model length than the prompt (the served 64K vs 1M constructions at the same position)
    cap = max(cap, -(-int(ARGS.get("cap", 0)) // bs) * bs)  # noqa: F821
    tp_g = MK.DsaIndexer.pool_rows(cap)
    if synthetic:  # (dsa_long.py's synthetic state: O(1) latent rows, random pooled keys, the HF decode input rows)
        from models.autoports.zai_org_glm_5_3_flash.tests.common import golden

        gen = torch.Generator().manual_seed(seed + max(pos))
        gd = golden(DSA.LAYER, 2048)
        lat_all = torch.randn(cap, attn.kv_lora, generator=gen).to(torch.bfloat16)
        g_in = L.g.reshape(-1, 16384).double()
        xs = [gd["decode0_streams_in"].double().reshape(-1)] + [g_in[300 + 37 * i] for i in range(1, len(pos))]
        states = []
        for p in pos:
            pg = torch.zeros(tp_g, 128)
            pg[: (p + 1) // 4] = torch.randn((p + 1) // 4, 128, generator=gen)
            states.append((pg, torch.zeros(32, 256)))
        ir = None
    else:
        gd, hn_all, lat_all = DSA._prompt(L, W, attn, max(pos) + 1)
        ir = G.indexer_rows(hn_all, ix.host)
        g_in = L.g.reshape(-1, 16384).double()
        xs = [gd["decode0_streams_in"].double().reshape(-1)] + [g_in[300 + 37 * i] for i in range(1, len(pos))]
        states = [G.dsa_state(p, ir["k"], ir["g"], ix.host["ape"], tp_g) for p in pos]
    prompt_s = round(time.time() - t0, 1)
    rings = tuple(int(v) for v in ARGS.get("rings", DSA.RINGS_SHARD))  # noqa: F821
    dkb = {"dsa_kb": tuple(int(v) for v in ARGS["dsa_kb"])} if ARGS.get("dsa_kb") else {}  # noqa: F821
    knobs.update(rings=list(rings), dsa_kb=list(dkb.get("dsa_kb", (32, 8))))

    class ShardCase:
        """the sharded device state of the same slots: chip r holds units u = r mod C (tt/dsa_shard.py ShardMap); the
        cache / pooled keys block-vectorised (the same content as a per-row fill: ch is zero-initialised, every local
        block belongs to one slot)"""

        def __init__(self, C: int):
            self.sm = sm = DS.ShardMap(C, bs)
            lcap = sm.local_cap(cap)
            self.bps = lcap // bs
            n_l = t * self.bps
            self.tp = sm.local_pool_rows(cap)
            caches, pts, pools = [], [], []
            ring = torch.zeros(t, 1, 32, 256, dtype=torch.float64)
            self.pt_host, self.ref_state = [], []
            for i in range(t):
                ring[i, 0] = states[i][1]
                self.ref_state.append(states[i])
            for r in range(C):
                perm = torch.randperm(n_l, generator=torch.Generator().manual_seed(101 + 7 * r + t))
                pt_l = perm.reshape(t, self.bps).to(torch.int32)
                ch = torch.zeros(n_l, 1, bs, attn.kv_lora, dtype=torch.bfloat16)
                pl = torch.zeros(t, 1, self.tp, 128, dtype=torch.float64)
                for i, p in enumerate(pos):
                    # [lcap, 512] in local order, zero past p (fp32 first: the same rounding as the per-row fill)
                    local = sm.shard_rows(lat_all[:p].float(), r, cap)
                    ch[pt_l[i].long(), 0] = local.reshape(self.bps, bs, attn.kv_lora).to(torch.bfloat16)
                    pl[i, 0] = sm.shard_pools(states[i][0].double(), r, self.tp)
                caches.append(ch)
                pts.append(pt_l)
                pools.append(pl)
                self.pt_host.append(pt_l)
            shard = ttnn.ShardTensorToMesh(mesh, dim=0)
            self.cache = ttnn.from_torch(torch.cat(caches), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard)  # fmt: skip
            del caches
            self.pt = ttnn.from_torch(torch.cat(pts), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=shard)  # fmt: skip
            put = lambda v, mapper: ttnn.from_torch(
                v.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )  # noqa: E731  # fmt: skip
            self.st = dict(pools=put(torch.cat(pools), shard), ring=put(ring, ttnn.ReplicateTensorToMesh(mesh)),
                           tp=self.tp, slots=t, s0=0)  # fmt: skip
            if dense_window:  # the replicated dense window (dsa/dense_window_design.md): rows 0 .. p - 1 of every slot
                WB = MK.DSA_WIN_BLOCKS  # on every chip, the cache's paged layout (slot i = pages i WB .. + WB - 1)
                wch = torch.zeros(t * WB, 1, bs, attn.kv_lora, dtype=torch.bfloat16)
                for i, p in enumerate(pos):
                    rows = torch.zeros(WB * bs, attn.kv_lora, dtype=torch.float32)
                    n_ = min(p, WB * bs)
                    rows[:n_] = lat_all[:n_].float()  # fp32 first: the same rounding as the cache fill
                    wch[i * WB : (i + 1) * WB, 0] = rows.reshape(WB, bs, attn.kv_lora).to(torch.bfloat16)
                self.st["window"] = ttnn.from_torch(wch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
                del wch
            self.cur = int_tensor(pos, mesh)
            self.x = ttnn.from_torch(torch.stack(xs).float().reshape(1, 1, t, -1), dtype=ttnn.float32,
                                     layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            self.t = t

    class RepSynth:
        """synthetic=True (rev 60: and every t > 4 launch): the replicated program's state from the same content
        (DSA.Case's layout, block-vectorised: the per-row fill is ~1M Python steps at 1M, 4M at t = 32 / 131K; the rows
        through fp32 first, as DSA.Case / ShardCase)"""

        def __init__(self):
            self.bps = cap // bs
            n = t * self.bps
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(11 + t))
            self.pt_host = perm.reshape(t, self.bps).to(torch.int32)
            self.pt = ttnn.from_torch(self.pt_host, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
                                      memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            ch = torch.zeros(n, 1, bs, attn.kv_lora, dtype=torch.bfloat16)
            for i, p in enumerate(pos):
                rows = torch.zeros(cap, attn.kv_lora, dtype=torch.bfloat16)
                rows[:p] = lat_all[:p].float()
                ch[self.pt_host[i].long(), 0] = rows.reshape(self.bps, bs, attn.kv_lora)
            self.cache = ttnn.from_torch(ch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                                         memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            del ch
            self.cur = int_tensor(pos, mesh)
            self.x = ttnn.from_torch(torch.stack(xs).float().reshape(1, 1, t, -1), dtype=ttnn.float32,
                                     layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            pools = torch.stack([s_[0] for s_ in states]).reshape(t, 1, tp_g, 128)
            ring = torch.stack([s_[1] for s_ in states]).reshape(t, 1, 32, 256)
            self.st = ix.state(mesh, t, cap, pools=pools, ring=ring)
            self.ref_state = list(states)
            self.t = t

    keys = ("s1", "o_lat", "latent") + (("q_lat", "q_ix", "wsc") if synthetic else ())
    got, us, cases, l1v = {}, {}, {}, {}
    # rev 60: t > 4 as StackBackend builds it (the 32-row FabricPhase serves every native width; the shard streams per t)
    fab_ts = (32,) if t > MK.DSA_T_MAX else (t,)
    if t > MK.DSA_T_MAX and os.environ.get("GLM53_FAB16", "0") == "1" and t <= 16:  # the 16-row phase (twolink_16row_audit.md)
        fab_ts = (16,)
    tp_rep = TPM.TPMla(mesh, L.a_args, attn, kb=DSA.KB_SHARD, rings=rings, ts=fab_ts, dsa=ix, probe=probe, **dkb)
    tp_sh = TPDS.TPMlaShard(mesh, L.a_args, attn, kb=DSA.KB_SHARD, rings=rings, ts=fab_ts, stream_ts=(t,), dsa=ix,
                            shard=tp_rep.shard, probe=probe, **dkb)  # fmt: skip
    knobs.update(lanes={k: list(v) for k, v in tp_sh.lane_core.items()}, chunks=tp_sh.mk.chunks, fabric_ts=list(fab_ts),
                 token_lanes=dict(replicated=tp_rep.mk.att_lanes(t), sharded=tp_sh.mk.att_lanes(t)),
                 ix_lane_cores=[list(v) for v in tp_sh.ix_lane_cores])  # fmt: skip
    n_soak = 0
    for name, prog in (("replicated", tp_rep), ("sharded", tp_sh)):
        if name == "replicated":
            c = RepSynth() if synthetic or t > MK.DSA_T_MAX else DSA.Case(mesh, L, ix, pos, lat_all, ir, xs, dsa=True)
        else:
            c = ShardCase(ndev)
        cases[name] = c
        fn = lambda prog=prog, c=c: {k: v for k, v in prog(c.x, c.cache, c.pt, c.cur, t=c.t, dsa_state=c.st).items()
                                     if k in keys + ("mbox", "hist")}  # fmt: skip
        tr.warm(fn)
        det_n = int(ARGS.get("det", 0)) if tr.enabled else 0  # noqa: F821
        nans = {}
        if det_n:  # the poison sources + ONE compile of their copies, before this program's capture (no first compile
            # after a capture): every output key gets a NaN tensor of its spec, copied over the output before each replay
            o0 = fn()
            for k in keys:
                v = o0[k]
                nans[k] = ttnn.from_torch(torch.full(tuple(v.shape), float("nan")), dtype=v.dtype, layout=v.layout,
                                          device=mesh, memory_config=v.memory_config(),
                                          mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
                ttnn.copy(nans[k], v)
            ttnn.synchronize_device(mesh)
            if ARGS.get("mbox_dead"):  # noqa: F821  (the mailbox sentinel's copy, compiled here too)
                mb0 = o0["mbox"]
                sent = ttnn.from_torch(torch.full(tuple(mb0.shape), 0xDEADBEEF - (1 << 32), dtype=torch.int64).to(torch.int32),
                                       dtype=mb0.dtype, layout=mb0.layout, device=mesh, memory_config=mb0.memory_config(),
                                       mesh_mapper=ttnn.ReplicateTensorToMesh(mesh))  # fmt: skip
                ttnn.copy(sent, mb0)
                nans["__mbox_sentinel"] = sent
            ttnn.synchronize_device(mesh)
            for v in o0.values():
                if v is not None:
                    ttnn.deallocate(v)
            o0 = None
        out = tr.run(fn)
        try:  # the L1 allocator with this program's outputs alive (the census: the tensors' floor vs the CBs)
            mv = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
            l1v[name] = dict(alloc_per_bank=int(mv.total_bytes_allocated_per_bank),
                             largest_free_per_bank=int(mv.largest_contiguous_bytes_free_per_bank),
                             bank_bytes=int(mv.total_bytes_per_bank))  # fmt: skip
        except Exception as e:  # noqa: BLE001
            l1v[name] = dict(error=str(e)[:200])
        got[name] = {k: per_chip(v) for k, v in out.items() if k in keys}
        det = None
        if det_n:  # DET poisoned replays (NaN over every output first), each bitwise vs the first replay's outputs
            same, fin = [], []
            for _ in range(det_n):
                for k in keys:
                    ttnn.copy(nans[k], out[k])
                tr.replay()
                cur = {k: per_chip(out[k]) for k in keys}
                same.append(all(torch.equal(a, b) for k in keys for a, b in zip(cur[k], got[name][k])))
                fin.append(all(bool(torch.isfinite(a).all()) for k in keys for a in cur[k]))
            det = dict(n=det_n, bitwise=all(same), finite=all(fin), bitwise_count=sum(same), finite_count=sum(fin))
            sent = nans.pop("__mbox_sentinel", None)
            if sent is not None:  # mbox_dead: the mailbox words a replay never writes (they keep the sentinel)
                ttnn.copy(sent, out["mbox"])
                tr.replay()
                got[name]["mbox_dead"] = [(m.to(torch.int64) & 0xFFFFFFFF) == 0xDEADBEEF for m in per_chip_u32(out["mbox"])]
                det["mbox_dead_words"] = [int(m.sum()) for m in got[name]["mbox_dead"]]
                ttnn.deallocate(sent)
            for v in nans.values():
                ttnn.deallocate(v)
            print(f"DSA_SHARD_DET {name} " + __import__("json").dumps(det), flush=True)
        got[name]["mbox"] = per_chip_u32(out["mbox"])
        mk_ = prog.mk
        nl_ = mk_.att_lanes(t)
        cl_ = mk_.chunks // nl_
        # the mailbox row of token i's lane root (owner k = lane chunk core 0 of group 0; at t <= 4 every owner holds every
        # token's messages): MkMla.mbox_rows maps owner cores to the mailbox tensor's shard rows
        k_root = lambda i, mk_=mk_, nl_=nl_, cl_=cl_: (int(mk_.place["k"][(i % nl_) * cl_].x),  # noqa: E731
                                                     int(mk_.place["k"][(i % nl_) * cl_].y))
        got[name]["mbox_row"] = [mk_.mbox_rows[k_root(i)] for i in range(t)]
        if name == "sharded":  # the program's T_g probe (ix_dm0.cpp hdr[8 + j] in its shard scratch, per lane core)
            ha = int(out["hist"].buffer_address())
            shs_w = (((ha + max(cl_ * 64, MK.DSA_MAX_CHUNKS * 4) + 63) & ~63) - ha) // 4
            ixc = mk_._ix_cores(t)
            row_of = {xy: r_ for r_, xy in enumerate(mk_.ix_rows)}
            words = int(out["hist"].shape[-1])
            assert shs_w + 8 + MK.shard_tl(t) <= words, (shs_w, words)
            got[name]["tg_probe"] = [
                [int(h.reshape(-1, words)[row_of[(int(ixc[i % nl_].x), int(ixc[i % nl_].y))], shs_w + 8 + i // nl_])
                 for i in range(t)] for h in per_chip_u32(out["hist"])
            ]  # fmt: skip
        got[name]["pools"], got[name]["ring"] = per_chip(c.st["pools"]), per_chip(c.st["ring"])
        if ARGS.get("dump"):  # noqa: F821  (cross-tree bitwise gates: every output of this program, per chip)
            torch.save(got[name], f"{ARGS['dump']}_{name}.pt")  # noqa: F821
        us[name] = MC.timed(tr, replays)
        marks = {}
        if probe:  # the last timed replay's marks (before the soak)
            try:
                if name == "sharded":
                    marks = dict(shard_probe=tp_sh.shard_probe())
                else:
                    pt = ttnn.to_torch(ttnn.get_device_tensors(tp_rep.mk.probe)[0])
                    pr = MK.decode_probe(pt, len(tp_rep.mk.last_readers.assign), chunks=tp_rep.mk.chunks,
                                         groups=tp_rep.mk.groups)  # fmt: skip
                    marks = dict(probe_chip0={k: pr[k] for k in ("owner_dm1", "owner_dm1_max", "ix_dm0", "ix_dm1",
                                                                 "latent_dm1", "reader_a_max") if k in pr})  # fmt: skip
            except Exception as e:  # noqa: BLE001  (a decode problem must not cost the timing row)
                marks = dict(probe_error=str(e)[-300:])
            if marks.get("shard_probe"):
                sp_ = marks["shard_probe"]
                spans = [v["span"] for v in sp_.values() if "span" in v]
                if spans:  # the per-span maximum over the chips (the critical chip)
                    marks["span_max"] = {k: max(s_[k] for s_ in spans if k in s_) for k in spans[0]}
                print("DSA_SHARD_SPANS " + __import__("json").dumps(marks.get("span_max"), default=str), flush=True)
        if name == "sharded" and soak_s > 0 and tr.enabled:
            n_soak = soak(tr, soak_s)
        tr.release()
        out = None
        rows("dsa_shard_timing", t=t, execution=tr.mode, status="info", timing_us={"tp_step": us[name]},
             prompt_s=prompt_s, replays=replays, soak_s=soak_s if name == "sharded" else 0, soak_replays=n_soak,
             program=name, det=det, link_lanes=getattr(prog, "link_lanes", None), **marks, **knobs)  # fmt: skip

    # ---- the comparison
    hr = attn.h // ndev
    knobs["links"] = tp_sh.link_audit  # per chip (audit_links: one connection per (link, direction) at a time)
    knobs["targets"] = tp_sh.target_audit  # per chip (audit_targets: rank / peer words + NoC targets per kernel group)
    knobs["headers"] = tp_sh.header_audit  # per chip (fabric packet headers per lane RISC, <= 12)
    knobs["l1"] = dict(cb=tp_sh.census, view=l1v)  # static CB bytes per core role + the L1 allocator per program
    print("DSA_SHARD_CENSUS " + __import__("json").dumps(dict(mesh=list(mesh.shape), t=t, **knobs["l1"]),
                                                       default=str), flush=True)  # fmt: skip
    r_ = dict(t=t, **knobs)
    s1_rep, s1_sh = got["replicated"]["s1"], got["sharded"]["s1"]
    r_["s1_bitwise_across_chips_replicated"] = all(torch.equal(s1_rep[0], s) for s in s1_rep[1:])
    r_["s1_bitwise_across_chips_sharded"] = all(torch.equal(s1_sh[0], s) for s in s1_sh[1:])
    r_["s1_pcc_sharded_vs_replicated"] = min(pcc(s1_sh[d].reshape(-1), s1_rep[d].reshape(-1)) for d in range(ndev))
    r_["s1_maxabs_sharded_vs_replicated"] = float(max((s1_sh[d] - s1_rep[d]).abs().max() for d in range(ndev)))
    o_p = [pcc(got["sharded"]["o_lat"][d][:, :t, :hr].reshape(-1), got["replicated"]["o_lat"][d][:, :t, :hr].reshape(-1))
           for d in range(ndev)]  # fmt: skip
    r_["o_lat_pcc_sharded_vs_replicated_min"] = min(o_p)
    r_["o_lat_pcc_per_chip_token"] = [
        [
            round(
                pcc(
                    got["sharded"]["o_lat"][d][0, i, :hr].reshape(-1),
                    got["replicated"]["o_lat"][d][0, i, :hr].reshape(-1),
                ),
                6,
            )
            for i in range(t)
        ]
        for d in range(ndev)
    ]  # [chip][token]
    r_["o_lat_pcc_per_chip_token_head_min"] = [
        [
            round(
                min(
                    pcc(got["sharded"]["o_lat"][d][0, i, h], got["replicated"]["o_lat"][d][0, i, h]) for h in range(hr)
                ),
                6,
            )
            for i in range(t)
        ]
        for d in range(ndev)
    ]
    r_["latent_equal"] = all(
        torch.equal(got["sharded"]["latent"][d], got["replicated"]["latent"][d]) for d in range(ndev)
    )
    # the selection thresholds: the replicated program's local T (round 7's prefix, owner row 0) vs the sharded global
    # T_g (the 10th message) per token; sparse tokens only
    T_eq, T_rows = [], []
    # kernels/mla/dsa_shard.hpp GMSG_OFF = MBOX_BYTES (QUOTA_W = 2) in words; rev 60: the shard mailbox holds the
    # lane-local tokens (TL_MAX slots), token i at slot i // L of its lane root's row
    tl = MK.shard_tl(t)
    gmsg_w = (tl * MK.DSA_MSGS * 16 + tl * MK.DSA_MAX_CHUNKS * 4 * 2) // 4
    nl_sh = tp_sh.mk.att_lanes(t)
    for i, p in enumerate(pos):
        if p + 1 <= MK.DSA_DENSE_MAX:
            continue
        rr, rs_ = got["replicated"]["mbox_row"][i], got["sharded"]["mbox_row"][i]
        T_rep = [int(mb[rr, (i * MK.DSA_MSGS + MK.DSA_ROUNDS - 1) * 4]) for mb in got["replicated"]["mbox"]]
        T_sh = [int(mb[rs_, gmsg_w + (i // nl_sh) * 4]) for mb in got["sharded"]["mbox"]]
        T_probe = [v[i] for v in got["sharded"]["tg_probe"]]  # what each program computed
        T_rows.append(dict(slot=i, p=p, T_replicated=T_rep, T_sharded=T_sh, T_coordinator=T_probe))
        T_eq.append(all(v == T_rep[0] for v in T_rep + T_sh + T_probe))
    r_["threshold_bitwise"] = all(T_eq) if T_eq else None
    # the owners read the threshold the program computed (the 10th message landed at GMSG_OFF)
    r_["threshold_probe_eq"] = all(
        row["T_sharded"][d] == row["T_coordinator"][d] for row in T_rows for d in range(ndev)
    )
    r_["thresholds"] = T_rows
    # the index writes: the pooled key p // 4 lands on its owner chip at its local row, bitwise vs the replicated write
    sm = cases["sharded"].sm
    idx_eq = []
    for i, p in enumerate(pos):
        q = p // 4
        o = sm.pool_owner(q)
        a_ = got["sharded"]["pools"][o][i, 0, sm.local_pool(q)]
        b_ = got["replicated"]["pools"][0][i, 0, q]
        idx_eq.append(bool(torch.equal(a_, b_)))
        for d in range(ndev):
            idx_eq.append(
                bool(torch.equal(got["sharded"]["ring"][d][i, 0, p % 4], got["replicated"]["ring"][0][i, 0, p % 4]))
            )
    r_["index_writes_eq"] = all(idx_eq)
    r_["timing_us"] = {"replicated": us["replicated"], "sharded": us["sharded"]}
    r_["dram_gib_per_chip"] = None
    try:
        mv = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        r_["dram_gib_per_chip"] = round(mv.total_bytes_allocated_per_bank * mv.num_banks / 2**30, 3)
    except Exception:  # noqa: BLE001
        pass
    r_["status"] = ("pass" if r_["s1_bitwise_across_chips_sharded"] and r_["s1_pcc_sharded_vs_replicated"] >= 0.9999
                    and min(o_p) >= 0.9999 and r_["threshold_bitwise"] is not False and r_["threshold_probe_eq"]
                    and r_["index_writes_eq"]
                    and r_["latent_equal"] else "fail")  # fmt: skip
    if MOCK:
        r_["status"] = "record"
    rows("dsa_shard_vs_replicated", **r_)

    sc = dict(eps=attn.eps, scale=attn.scale, heads=attn.h, q_lora=attn.q_lora)
    refs = {}
    if synthetic:  # no model oracle at this length: the exact top-512 + attention on the device's own intermediates
        pools_after = got["replicated"]["pools"][0]
        for i, p in enumerate(pos):
            if p + 1 <= MK.DSA_DENSE_MAX:
                continue
            nc = (p + 1) // 4
            qi = got["sharded"]["q_ix"][0][0, i].double()
            w = got["sharded"]["wsc"][0][0, 0, i, :32].double()
            scores = w @ torch.relu(qi @ pools_after[i, 0, :nc].double().T)
            sel = torch.topk(scores, MK.DSA_TOPK).indices.sort().values
            rows_ = torch.cat([(4 * sel[:, None] + torch.arange(4)).reshape(-1), torch.arange(4 * nc, p + 1)])
            kv_all = lat_all[: p + 1].double().clone()
            kv_all[p] = got["sharded"]["latent"][0].reshape(-1, attn.kv_lora)[i].double()
            kv = kv_all[rows_]
            o_vs = []
            for d in range(ndev):
                ql = got["sharded"]["q_lat"][d][0, i, :hr].double()
                o_em = torch.softmax((ql @ kv.T) * attn.scale, dim=-1) @ kv
                o_vs.append(round(pcc(got["sharded"]["o_lat"][d][0, i, :hr].reshape(-1), o_em.reshape(-1).float()), 6))
            rows("dsa_shard_emulation", t=t, slot=i, p=p, program="sharded", status="info",
                 o_lat_pcc_vs_emulation_per_chip=o_vs, o_lat_pcc_vs_emulation_min=min(o_vs), **knobs)  # fmt: skip
    for name in ("replicated", "sharded") if not synthetic else ():
        s1 = got[name]["s1"]
        for i, p in enumerate(pos):
            pl, rg = cases["sharded"].ref_state[i]
            if i not in refs:
                refs[i] = G.dsa_step(xs[i].reshape(1, -1), p, L.a_args, W, ix.host, lat_all[:p], pl, rg, **sc)
            ref = refs[i]
            s1_i = s1[0][i, 0] if t <= 4 else s1[0][0, 0, i].reshape(4, -1)
            a_, b_ = s1_i.reshape(-1), ref["s1"].reshape(-1)[: s1_i.numel()].float()
            o_ref = ref["o_lat"].float().reshape(attn.h, -1)
            o_vs = [
                round(pcc(got[name]["o_lat"][d][0, i, :hr].reshape(-1), o_ref[d * hr : (d + 1) * hr].reshape(-1)), 6)
                for d in range(ndev)
            ]  # this chip's heads vs the oracle
            g_ = dict(t=t, slot=i, p=p, program=name, sparse=p + 1 > MK.DSA_DENSE_MAX, s1_pcc_vs_fp32=pcc(a_, b_),
                      s1_maxabs_vs_fp32=float((a_ - b_).abs().max()), o_lat_pcc_vs_fp32_per_chip=o_vs, **knobs)  # fmt: skip
            g_["status"] = "record" if MOCK else ("pass" if g_["s1_pcc_vs_fp32"] >= DSA.BAR else "fail")
            rows("dsa_shard_golden", **g_)
    rows.flush()
    bad = rows.failed()
    print(f"DSA_SHARD_DONE rows={len(rows.rows)} fail={len(bad)} us={us}", flush=True)
    RESULT = dict(ok=not bad, rows=len(rows.rows), us=us)
    if bad:
        raise RuntimeError(f"dsa_shard: {len(bad)} failing row(s)")
finally:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
