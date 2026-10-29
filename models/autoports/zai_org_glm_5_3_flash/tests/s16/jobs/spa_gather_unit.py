"""Job (serve.py, pair): Stage 16 SP-row sparse attention v1 (M-B2 section 2b) vs lever B-q on the SAME selections.

SP-row (per chip: its own query rows, all 64 heads): the q all-to-all heads -> rows, the union GATHER of the chip's
selected pools from their owners over the fabric into a compact buffer (tt/spa_gather.py), sparse_sdpa on the compact
buffer with remapped ids, the o all-to-all rows -> heads (prefill_tp.MlaPrefill._sparse_sdpa's GLM53_PF_SPARSE_RESHARD=a2a
form, kv = the compact buffer). References: the same a2a path on the full replicated cache with the original ids
(bitwise expected: the gather only relocates rows and the ids keep each row's key order), and lever B-q
(prefill_tp.MlaPrefill._bq_block: the q all-gather, sparse_sdpa_stats over the keys each chip OWNS, the owner merge, the
fp32 head-dim reduce-scatter; not bitwise -- the owner merge renormalises bf16 partials), both against a float64 host
softmax on sampled rows.

Selections: the REAL HF golden top-k of layer 3 at the 32K prompt (goldens/layer_io/L03/32768/topk.pt), the chunk = the
window's last 2,048 rows, chip rank k = r MC + c = rows [k R, (k + 1) R) (SP rows; R = 1,024 on the pair, 256 at TP8). Latents / q: seeded N(0, 1) bf16. The cache
is token-sharded as the served GLM53_DSA_SHARD layout (tt/dsa_shard.py): 128-token units, owner = unit % N, local row =
(unit // N) 128 + t % 128. Per chip: the union of its rows' selected pools (4 tokens) -> compact slots; the request
entries of each owner (src_row, dst_slot, dst_pos); ids remapped to compact rows (4 slot + t % 4); B-q's local ids =
dsa_shard.bq_local_ids (expand_rows SHARD mode). All host-built (static selections; the device-side union + request
exchange is the next piece). Both paths start from the same row-major local latents (local_latents_rm excluded from both).
TP8 (2x4, FABRIC_1D lines per axis): the gather runs in two stages -- axis 0: each owner pushes the pools the OTHER row
needs into its column partner's relay rows (appended to the local store), axis 1: every chip pushes its own and its
relayed pools into its row-mates' compact buffers; the lists are checked by a host simulation of both stages.
Rows: one pcc row (bitwise / max |diff| / pcc per pair of paths, host-reference errors, union sizes, fabric bytes) with
the traced replay medians (gather alone, sparse_sdpa compact / full, the SP-row phase, its full-cache reference, B-q).
ARGS: window (32768), rows (2048), replays (30), k_chunk (sparse default), ref_rows (8), unit (128), split (1: the
components' micro traces -- q / o a2a, B-q's q all-gather, sparse_sdpa_stats, fp32 reduce-scatter), tag, record."""
import hashlib
import os
import sys
import time
import types

import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.dsa_shard import SKIP  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spa_gather import SpaGather  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.spa_union import SpaUnion, walk_bound  # noqa: E402

C = S.C
mesh = MESH  # noqa: F821
args = ARGS  # noqa: F821
T = int(args.get("window", 32768))
CR = int(args.get("rows", 2048))
REPLAYS = int(args.get("replays", 30))
KC = int(args.get("k_chunk", 0) or PF.sparse_kchunk())
tag = args.get("tag", f"spa-gather-T{T}")
N = int(mesh.get_num_devices())
MR, MC = int(mesh.shape[0]), int(mesh.shape[1])  # chip rank k = r MC + c (the SP / shard order)
TWO = MR > 1 and MC > 1  # the 2x4: the gather in two FABRIC_1D stages (axis 0 into a relay region, then axis 1)
DEV = args.get("union", "host") == "device"  # v2: the union, the request exchange and the remap on device (tt/spa_union.py)
R = CR // N
H, D, W = 64, 512, PF.SPARSE_W
UNIT = int(args.get("unit", 128))  # tt/dsa_shard.py UNIT
NREF = int(args.get("ref_rows", 8))
HR = H // N
DRAM = ttnn.DRAM_MEMORY_CONFIG
SCALE = 0.0625

gp = args.get("topk", "/opt/glm53/kit/goldens/layer_io/L03/32768/topk.pt")
if not os.path.exists(gp):
    gp = gp.replace("/opt/glm53/share", "/opt/glm53/home/mpi-shfs")
topk = torch.load(gp, map_location="cpu")
if isinstance(topk, dict) and "rows" in topk:  # the Galaxy asset (glx_assets/L03_topk_32768_ad2s_rows.pt): {start: [CR, W]}
    assert int(topk["T"]) == T and (T - CR) in topk["rows"], ("the topk asset", topk["T"], sorted(topk["rows"]), T, CR)
    topk = topk["rows"][T - CR].to(torch.int64)
else:
    topk = topk["topk"] if isinstance(topk, dict) else topk
    topk = topk.reshape(-1, topk.shape[-1])[:T][T - CR:].to(torch.int64)  # [CR, 2051] the chunk's rows
assert topk.shape[0] == CR, topk.shape

g = torch.Generator().manual_seed(int(args.get("seed", 7)))
# spaprobe (ORDER 2026-10-27 21:13:31Z, the SpaUnion phase probe as black-box arms): SEL replaces the golden rows by a
# synthetic selection of the same shape (512 closed pools x 4 tokens per row, -1 tail) that moves the union / the owner
# mix: "same" = every row the golden row 0's pools (the union ~512: the fixed + barrier term), "uniform" = 512 seeded
# uniform closed pools per row (the largest union: the push slope), "self" / "remote" = uniform over the pools the row's
# chip OWNS / does not own (owner = unit % N: local copies only vs fabric only). "golden" = unchanged.
SEL = str(args.get("sel", "golden"))
# walkbound (program [session 10] 2026-10-28 04:03:56Z): the WARMUP forms. TPPrefill._compile's sparse-key forward
# writes ZERO token ids at chunk_start = t - c (prefill_tp.py _compile), so every query row is the same token and the
# indexer's scores tie: the selections collapse onto one id set for every row ("zero" = the window's first K tokens, the
# lowest-index tie-break; "zero_hi" = the K tokens before the chunk), the unions are identical on every chip and packed
# into a few bitmap words. "empty" = every row SKIP at 0 (empty unions: the walkers' zero-work path); "degenerate" = per
# row i % 4: empty / the single id 0 (the null row's pool) / K copies of one id / the "zero" row. The ids stay inside the
# window: an id past it never reaches the walkers (spa_um.cpp drops it before the bitmap and counts it in DROP).
ZERO_FORMS = ("zero", "zero_hi", "empty", "degenerate")
if SEL in ZERO_FORMS:
    K = min(2048, T - CR, int(topk.shape[1]))
    lo_ids, hi_ids = torch.arange(K), torch.arange(T - CR - K, T - CR)
    new_rows = torch.full_like(topk, -1)
    for i in range(CR):
        f = {"zero": 3, "zero_hi": 4, "empty": 0}.get(SEL, i % 4)
        if f == 1:
            new_rows[i, 0] = 0
        elif f == 2:
            new_rows[i, :K] = (T - CR) // 2
        elif f == 3:
            new_rows[i, :K] = lo_ids
        elif f == 4:
            new_rows[i, :K] = hi_ids
    topk = new_rows
elif SEL != "golden":
    gs = torch.Generator().manual_seed(int(args.get("sel_seed", 11)))
    npool = (T - CR) // 4  # closed pools visible to every row of the chunk
    pool_owner = (torch.arange(npool) * 4 // int(args.get("unit", 128))) % int(mesh.get_num_devices())
    rows_per_chip = CR // int(mesh.get_num_devices())
    base = torch.unique(topk[0][topk[0] >= 0] // 4)[:512]
    new_rows = torch.full_like(topk, -1)
    for i in range(CR):
        chip = i // rows_per_chip
        if SEL == "same":
            pl = base
        else:
            cand = {"uniform": torch.arange(npool), "self": torch.nonzero(pool_owner == chip).reshape(-1),
                    "remote": torch.nonzero(pool_owner != chip).reshape(-1)}[SEL]  # fmt: skip
            pl = torch.sort(cand[torch.randperm(cand.numel(), generator=gs)[:512]]).values
        tok = (pl.reshape(-1, 1) * 4 + torch.arange(4)).reshape(-1)
        new_rows[i, : tok.numel()] = tok
    topk = new_rows
kv = torch.randn(T, D, generator=g).to(torch.bfloat16)
q = torch.randn(N, H, R, D, generator=g).to(torch.bfloat16)

owner = lambda t: (t // UNIT) % N  # noqa: E731
lrow = lambda t: (t // UNIT // N) * UNIT + t % UNIT  # noqa: E731
rc_of = lambda k: (k // MC, k % MC)  # noqa: E731
t_loc = T // N

ids_orig = torch.full((N, 1, R, W), -1, dtype=torch.int64)
ids_new = torch.full((N, 1, R, W), -1, dtype=torch.int64)
unions = []
for c in range(N):
    rows = topk[c * R:(c + 1) * R]
    valid = rows >= 0
    pools = torch.unique(rows[valid] // 4)
    unions.append(pools)
    sl = torch.full((T // 4,), -1, dtype=torch.int64)
    sl[pools] = torch.arange(pools.numel())
    ids_orig[c, 0, :, :rows.shape[1]] = torch.where(valid, rows, torch.full_like(rows, -1))
    remap = sl[(rows.clamp(min=0) // 4)] * 4 + rows.clamp(min=0) % 4
    ids_new[c, 0, :, :rows.shape[1]] = torch.where(valid, remap, torch.full_like(rows, -1))
umax = max(u.numel() for u in unions)
CAP = int(args.get("cap", 0) or (umax + 64 if DEV else umax))  # compact pools per chip (v2: the device flags a larger union)
# entries (src_row, dst_slot, dst_pos) per sender, per stage; dst_pos = the destination's position on the stage's line
ent_a = [[] for _ in range(N)]  # 2x4 stage A (axis 0): owner -> the relay chip (destination row, owner column)
ent_b = [[] for _ in range(N)]  # the line / 2x4 stage B (axis 1): -> the destination's compact buffer
relay = [dict() for _ in range(N)]  # relay chip -> {pool: relay index}
if TWO:
    need = [set() for _ in range(N)]  # relay chip y = (rd, cs): pools owned in column cs (other rows) needed by row rd
    for d in range(N):
        rd, _ = rc_of(d)
        for pool in unions[d].tolist():
            rs, cs = rc_of(owner(4 * pool))
            if rs != rd:
                need[rd * MC + cs].add(pool)
    for y in range(N):
        relay[y] = {pool: k for k, pool in enumerate(sorted(need[y]))}
        ry, _ = rc_of(y)
        for pool, k in relay[y].items():  # stage A: the owner sends it down its column into y's relay rows
            ent_a[owner(4 * pool)].append((lrow(4 * pool), t_loc // 4 + k, ry))
for d in range(N):  # destination d's union, each pool pushed by its owner (same row) or by the relay chip (stage B)
    rd, cd = rc_of(d)
    for k, pool in enumerate(unions[d].tolist()):
        s_ = owner(4 * pool)
        rs, cs = rc_of(s_)
        if not TWO or rs == rd:
            ent_b[s_].append((lrow(4 * pool), k, cd if TWO else d))
        else:
            y = rd * MC + cs
            ent_b[y].append((t_loc + 4 * relay[y][pool], k, cd))
rmax = max([len(r_) for r_ in relay] + [0])
lat = torch.zeros(N, 1, t_loc + 4 * rmax, D, dtype=torch.bfloat16)  # the local store (+ the relay rows on the 2x4)
tok = torch.arange(T)
for c in range(N):
    m = owner(tok) == c
    lat[c, 0, lrow(tok[m])] = kv[m]


def pack(ent, own_pos):
    for s_ in range(N):  # local first, then the peers (the sender's order does not matter for correctness)
        ent[s_].sort(key=lambda e: (e[2] != own_pos(s_), e[2], e[1]))
    lmax = max(1, max(len(e) for e in ent))
    t_ = torch.zeros(N, 1, 1, 3 * lmax, dtype=torch.int64)
    for s_ in range(N):
        if ent[s_]:
            t_[s_, 0, 0, :3 * len(ent[s_])] = torch.tensor(ent[s_], dtype=torch.int64).reshape(-1)
    return t_, [len(e) for e in ent], [sum(1 for e in ent[s_] if e[2] != own_pos(s_)) for s_ in range(N)]


lst_b, n_b, rem_b = pack(ent_b, (lambda k: rc_of(k)[1]) if TWO else (lambda k: k))
lst_a, n_a, rem_a = pack(ent_a, lambda k: rc_of(k)[0]) if TWO else (None, [0] * N, [0] * N)
n_ent = [n_a[k] + n_b[k] for k in range(N)]
remote = [rem_a[k] + rem_b[k] for k in range(N)]
# the lists' host simulation (stage A, then stage B, as the kernels apply them): every chip's compact rows = its union
sim = lat.clone()
cmp_sim = torch.zeros(N, 4 * umax, D, dtype=torch.bfloat16)
for s_ in range(N):
    for src, slot, pos in ent_a[s_]:
        y = pos * MC + rc_of(s_)[1]
        sim[y, 0, 4 * slot:4 * slot + 4] = sim[s_, 0, src:src + 4]
for s_ in range(N):
    for src, slot, pos in ent_b[s_]:
        dst = rc_of(s_)[0] * MC + pos if TWO else pos
        cmp_sim[dst, 4 * slot:4 * slot + 4] = sim[s_, 0, src:src + 4]
sim_ok = [bool(torch.equal(cmp_sim[c][:4 * unions[c].numel()], kv[(unions[c].reshape(-1, 1) * 4 + torch.arange(4)).reshape(-1)]))
          for c in range(N)]  # fmt: skip
assert all(sim_ok), ("the gather lists do not build the unions", sim_ok)
del sim, cmp_sim

# the same q in the model's head-sharded form (chip c: heads [HR c, HR (c + 1)), all rows) for the SP-row phase and B-q
q_g = q.permute(1, 0, 2, 3).reshape(H, CR, D)  # [64, CR, 512]: chip c's SP rows = q[c]
q_h = q_g.reshape(N, HR, CR, D)
# B-q: each chip's OWNED keys as local row ids (dsa_shard.bq_local_ids: same order, SKIP tail, a dummy row 0 when empty)
ids_loc = torch.full((N, 1, CR, W), -1, dtype=torch.int64)
cnt = torch.zeros(N, 1, CR, 16, dtype=torch.float32)  # expand_rows' counts page: word 0 = the keys, word 1 = the valid flag
for r in range(N):
    for i in range(CR):
        v = topk[i][topk[i] >= 0]
        mine = v[owner(v) == r]
        n = int(mine.numel())
        cnt[r, 0, i, 0], cnt[r, 0, i, 1] = float(n), float(n > 0)
        if n:
            ids_loc[r, 0, i, :n] = lrow(mine)
        else:
            ids_loc[r, 0, i, 0] = 0
empty_rows = [int((cnt[r, 0, :, 1] == 0).sum()) for r in range(N)]
assert SKIP == 0xFFFFFFFF  # -1 as int32 -> the uint32 SKIP tail

shard = ttnn.ShardTensorToMesh(mesh, dim=0)
rep = ttnn.ReplicateTensorToMesh(mesh)
put = lambda t, dt, lay, mp: ttnn.from_torch(t, dtype=dt, layout=lay, device=mesh, memory_config=DRAM, mesh_mapper=mp)  # noqa: E731
RM, TL = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT
d_lat = put(lat, ttnn.bfloat16, RM, shard)  # B-q reads rows < t_loc only
d_lstb = put(lst_b.to(torch.int32), ttnn.uint32, RM, shard)
d_lsta = put(lst_a.to(torch.int32), ttnn.uint32, RM, shard) if TWO else None
d_cmp = put(torch.zeros(1, 1, 4 * CAP, D, dtype=torch.bfloat16), ttnn.bfloat16, RM, rep)
d_kv = put(kv.reshape(1, 1, T, D), ttnn.bfloat16, RM, rep)
d_q = put(q, ttnn.bfloat16, RM, shard)  # [1, 64, R, 512] per chip: the micro rows (sparse_sdpa alone)
d_qh = put(q_h, ttnn.bfloat16, TL, shard)  # [1, HR, CR, 512] per chip: the phase input (as the model's q_lat)
d_io = put(ids_orig.to(torch.int32), ttnn.uint32, RM, shard)
d_in = put(ids_new.to(torch.int32), ttnn.uint32, RM, shard)
if DEV:  # the remapped ids come from the device: the host's ids_new is only the reference
    d_in = put(torch.zeros_like(ids_new, dtype=torch.int32), ttnn.uint32, RM, shard)
d_il = put(ids_loc.to(torch.int32), ttnn.uint32, RM, shard)
d_cnt = put(cnt, ttnn.float32, RM, shard)
gat_b = SpaUnion(mesh, d_cmp, R, T, W, cap=CAP) if DEV else SpaGather(mesh, d_cmp, axis=1 if MC > 1 else 0)
gat_a = SpaGather(mesh, d_lat, axis=0) if TWO and not DEV else None  # host lists: into the relay rows of the local store
ckc = PF._ckc_fp32(mesh)

# the production code paths on a minimal context (prefill_tp.MlaPrefill / TPPrefill methods, unchanged)
tpp_f = types.SimpleNamespace(mesh=mesh, ndev=N, sparse_reshard="a2a", sparse_rows=CR)
for name in ("_axes", "gather_dim", "reduce_scatter_dim"):
    setattr(tpp_f, name, types.MethodType(getattr(PF.TPPrefill, name), tpp_f))
mla_f = types.SimpleNamespace(tpp=tpp_f, scale=SCALE, ckc_fp32=ckc)
# k4d7's MlaPrefill._sparse_sdpa calls self._a2a_q / self._a2a (no FabricStream a2a here: the op all-to-all)
mla_f._a2a = types.MethodType(PF.MlaPrefill._a2a, mla_f)
mla_f._a2a_q = types.MethodType(PF.MlaPrefill._a2a_q, mla_f)
mla_f._a2a = types.MethodType(PF.MlaPrefill._a2a, mla_f)

MIB = 2**20
fab = dict(  # bytes each chip SENDS over the fabric per layer (MiB)
    spa_q_a2a=round(HR * (CR - R) * D * 2 / MIB, 1), spa_gather=[round(remote[s] * 4096 / MIB, 1) for s in range(N)],
    spa_o_a2a=round(HR * (CR - R) * D * 2 / MIB, 1), bq_q_allgather_recv=round((N - 1) * HR * CR * D * 2 / MIB, 1),
    bq_stats_allgather=round(2 * CR * H * 4 / MIB, 1), bq_o_reducescatter_fp32=round((N - 1) / N * H * CR * D * 4 / MIB, 1))
fab["spa_total"] = round(fab["spa_q_a2a"] + max(fab["spa_gather"]) + fab["spa_o_a2a"], 1)
fab["bq_total"] = round(fab["bq_q_allgather_recv"] + fab["bq_stats_allgather"] + fab["bq_o_reducescatter_fp32"], 1)
S.log(f"{tag}: mesh {MR}x{MC} T {T} rows {CR} ({R} per chip) unit {UNIT} union pools per chip {[u.numel() for u in unions]} entries "
      f"(stage A {n_a}, stage B {n_b}) remote {remote} relay pools {[len(r_) for r_ in relay]} B-q rows without an owned key "
      f"{empty_rows}; fabric MiB sent per chip {fab}")  # fmt: skip

slots = {k: [None] for k in ("new", "ref", "spa", "spf", "bq")}  # persistent outputs (the compile launch's); traces copy in


def keep(name, o):
    slot = slots[name]
    if slot[0] is None:  # the compile launch: this output becomes the persistent one
        slot[0] = o
        return
    ttnn.copy(o, slot[0])
    ttnn.deallocate(o)


def gather_fn():
    if DEV:
        gat_b(d_lat, d_io, d_in)  # union + exchange + gather + remap (ids_new -> d_in)
        return
    if TWO:
        gat_a(d_lat, d_lsta, n_a)  # the relay rows first (the 1D fabric: one axis per instance)
    gat_b(d_lat, d_lstb, n_b)


def sdpa_cmp():  # the micro: sparse_sdpa alone on the compact buffer (q already in rows)
    keep("new", ttnn.transformer.sparse_sdpa(d_q, d_cmp, d_in, D, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=SCALE,
                                             k_chunk_size=KC, compute_kernel_config=ckc))  # fmt: skip


def sdpa_full():
    keep("ref", ttnn.transformer.sparse_sdpa(d_q, d_kv, d_io, D, kv_format=ttnn.transformer.SparseKVFormat.BF16, scale=SCALE,
                                             k_chunk_size=KC, compute_kernel_config=ckc))  # fmt: skip


def spa_fn():  # the SP-row phase: gather (independent of q) + q a2a + sparse_sdpa(compact) + o a2a
    gather_fn()
    keep("spa", PF.MlaPrefill._sparse_sdpa(mla_f, d_qh, d_in, d_cmp))


def spf_fn():  # the same a2a path on the full replicated cache (the op-level SP-row; needs the replicated cache)
    keep("spf", PF.MlaPrefill._sparse_sdpa(mla_f, d_qh, d_io, d_kv))


def bq_fn():  # lever B-q on the token-sharded cache (the served 1M path)
    keep("bq", PF.MlaPrefill._bq_block(mla_f, d_qh, d_lat, d_il, d_cnt, CR))


PATHS = (("gather", gather_fn), ("sdpa_cmp", sdpa_cmp), ("ref", sdpa_full), ("spa", spa_fn), ("spf", spf_fn), ("bq", bq_fn))
t0 = time.time()
gather_fn()
for _, fn in PATHS[1:]:  # every path's compile launch before any capture
    fn()
SPLIT = bool(int(args.get("split", 1)))
if SPLIT:  # the components alone (persistent inputs staged now; each micro frees its output inside its own trace)
    AX = [a for a in (0, 1) if int(mesh.shape[a]) > 1]  # MlaPrefill._a2a's axes
    d_ot = ttnn.to_layout(slots["new"][0], TL, memory_config=DRAM)  # [1, 64, R, 512] TILE: the o a2a input
    _qa = tpp_f.gather_dim(d_qh, 1)
    d_qrm = ttnn.to_layout(_qa, RM, memory_config=DRAM)  # [1, 64, CR, 512] RM: the stats input
    ttnn.deallocate(_qa)
    d_o32 = ttnn.typecast(ttnn.to_layout(slots["new"][0], TL, memory_config=DRAM), PF.FP32)  # fp32 [1, 64, R, 512]
    d_o32 = ttnn.concat([d_o32] * (CR // R), dim=2)  # fp32 [1, 64, CR, 512]: the RS input

    def _free(*ts):
        for t_ in ts:
            ttnn.deallocate(t_)

    def _chain(x, steps):  # MlaPrefill._a2a / TPPrefill.reduce_scatter_dim without freeing the staged input
        y = x
        for i, f in enumerate(steps):
            z = f(y)
            if i:
                ttnn.deallocate(y)
            y = z
        ttnn.deallocate(y)

    def m_qa2a():
        _chain(d_qh, [lambda x, a=a: ttnn.experimental.all_to_all_async_generic(x, in_dim=1, out_dim=2, cluster_axis=a,
                                                                                memory_config=DRAM) for a in AX])  # fmt: skip

    def m_oa2a():
        _chain(d_ot, [lambda x, a=a: ttnn.experimental.all_to_all_async_generic(x, in_dim=2, out_dim=1, cluster_axis=a,
                                                                                memory_config=DRAM) for a in AX[::-1]])  # fmt: skip

    def m_bqag():
        _free(tpp_f.gather_dim(d_qh, 1))

    def m_bqstats():
        _free(*ttnn.transformer.sparse_sdpa_stats(d_qrm, d_lat, d_il, D, kv_format=ttnn.transformer.SparseKVFormat.BF16,
                                                  scale=SCALE, k_chunk_size=KC, compute_kernel_config=ckc))  # fmt: skip

    def m_bqrs():
        _chain(d_o32, [lambda x, a=a: ttnn.reduce_scatter(x, dim=1, cluster_axis=a, memory_config=DRAM)
                       for a in tpp_f._axes()[::-1]])  # fmt: skip

    MICRO = (("q_a2a", m_qa2a), ("o_a2a", m_oa2a), ("bq_q_allgather", m_bqag), ("bq_stats", m_bqstats), ("bq_rs_fp32", m_bqrs))
    for _, fn in MICRO:
        fn()
    PATHS = PATHS + MICRO
for v in slots.values():  # the output copies compiled before any capture (no compile inside a capture)
    w_ = ttnn.clone(v[0], memory_config=DRAM)
    ttnn.copy(v[0], w_)
    ttnn.deallocate(w_)
ttnn.synchronize_device(mesh)
gcalls = lambda: gat_b.calls  # noqa: E731  # one per gather_fn (gat_a, on the 2x4, the same count)
g0 = gcalls()
pc = lambda: int(mesh.num_program_cache_entries()) if hasattr(mesh, "num_program_cache_entries") else -1  # noqa: E731
pc0 = pc()
S.log(f"{tag}: compile launches in {time.time() - t0:.1f} s; program cache {pc0}")
row = dict(test=f"SP-row attention v1 (q a2a + union gather over the fabric + sparse_sdpa on the compact buffer + o a2a) vs the "
                f"same path on the full cache vs lever B-q (T {T}, {CR} rows, real L3 32K selections, {UNIT}-token shard units)",
           milestone="P3", window=T, rows=CR, tp=N, unit=UNIT, union_pools=[int(u.numel()) for u in unions], entries=n_ent,
           remote_pools=remote, mesh=[MR, MC], stage_a_entries=n_a, relay_pools=[len(r_) for r_ in relay], lists_host_sim=sim_ok,
           bq_empty_rows=empty_rows, fabric_mib=fab, tag=tag, status="mock" if C.MOCK else "info")  # fmt: skip
tids = {}
for name, fn in PATHS:  # the mock cannot capture: it re-runs each path (a program first compiled here = a cache miss in a capture)
    if C.MOCK:
        fn()
        continue
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    try:
        fn()
    finally:
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    tids[name] = tid
assert gcalls() == g0 + 2 and (gat_a is None or gat_a.calls == gat_b.calls), ("eager guard / capture count", gcalls(), g0)
assert pc() == pc0, ("a program compiled inside a capture", pc(), pc0)
for name in ("spa", "sdpa_cmp", "ref", "spf", "bq"):  # spa first: its gather fills the compact buffer sdpa_cmp reads
    if not C.MOCK:
        ttnn.execute_trace(mesh, tids[name], cq_id=0, blocking=True)


def rd(t, shape):  # per chip host copies (zeros of the device shape on the mock: the host analysis still runs)
    if C.MOCK:
        return [torch.zeros(shape) for _ in range(N)]
    return [ttnn.to_torch(t_).float().reshape(shape) for t_ in ttnn.get_device_tensors(t)]


def cmp(a, b):
    return [dict(bitwise=bool(torch.equal(a[c], b[c])), maxabs=float((a[c] - b[c]).abs().max()), pcc=round(C.pcc(a[c], b[c]), 7),
                 finite=bool(torch.isfinite(a[c]).all())) for c in range(N)]  # fmt: skip


o_new, o_ref = rd(slots["new"][0], (H, R, D)), rd(slots["ref"][0], (H, R, D))
o_spa, o_spf, o_bq = (rd(slots[k][0], (HR, CR, D)) for k in ("spa", "spf", "bq"))
cc = rd(d_cmp, (4 * CAP, D))
if DEV:  # the device's remapped ids vs the host's, and the overflow flags
    ids_dev = [ttnn.to_torch(t_).reshape(R, W).to(torch.int64) & 0xFFFFFFFF for t_ in ttnn.get_device_tensors(d_in)] if not C.MOCK \
        else [torch.zeros(R, W, dtype=torch.int64)] * N  # fmt: skip
    ids_ok = [bool(torch.equal(ids_dev[c], ids_new[c, 0] & 0xFFFFFFFF)) for c in range(N)]
    row.update(union="device", cap=CAP, ids_new_exact=ids_ok, union_status=gat_b.status() if not C.MOCK else None)
    # the arms' bitwise key (walkers on vs off, bound on vs off): the compact rows of every chip's union + its ids_new
    hd = hashlib.sha1()
    for c in range(N):
        hd.update(cc[c][:4 * int(unions[c].numel())].contiguous().numpy().tobytes())
        hd.update(ids_dev[c].contiguous().numpy().tobytes())
    row.update(sel=SEL, union_digest=hd.hexdigest()[:20])
    if walk_bound() and not C.MOCK:  # GLM53_SPA_WALK_BOUND: the row lane's expiry record per chip (first expiry + count)
        row.update(walk_bound=walk_bound(), bound_records=gat_b.bound_records())
        row["bound_fired"] = any(r_.get("fired") for r_ in row["bound_records"])
gath_ok = []
for c in range(N):  # the compact buffer's rows vs the host's union rows (the replicated alloc: every chip's own union)
    exp = kv[(unions[c].reshape(-1, 1) * 4 + torch.arange(4)).reshape(-1)].float()
    gath_ok.append(bool(torch.equal(cc[c][:exp.shape[0]], exp)))
# float64 host softmax on sampled rows (all 64 heads): the selected keys, K = V = the latent rows
full = lambda o: torch.cat(o, dim=0)  # noqa: E731  # [64, CR, 512] from the chips' head blocks
idx = torch.linspace(0, CR - 1, NREF).long().tolist() if NREF else []
host = {}
for k, o in (("spa", o_spa), ("bq", o_bq)):
    fo, err, pcs = full(o), [], []
    for i in idx:
        v = topk[i][topk[i] >= 0]
        if v.numel() == 0:  # an empty row (the walkbound forms): no reference softmax
            continue
        kk = kv[v].double()
        pr = torch.softmax((q_g[:, i].double() @ kk.T) * SCALE, dim=-1)
        ref = (pr @ kk).float()
        err.append(float((fo[:, i] - ref).abs().max()))
        pcs.append(C.pcc(fo[:, i], ref))
    host[k] = dict(rows=len(idx), maxabs=round(max(err), 6) if err else None, pcc_min=round(min(pcs), 7) if pcs else None)
row.update(micro_compact_vs_full=cmp(o_new, o_ref), spa_vs_full_cache=cmp(o_spa, o_spf), spa_vs_bq=cmp(o_spa, o_bq),
           host_ref=host, compact_rows_exact=gath_ok)  # fmt: skip
if not C.MOCK and row.get("bound_fired"):  # an expiry leaves the sync generations undefined: no further launch
    S.log(f"{tag}: GLM53_SPA_WALK_BOUND expired: {row['bound_records']} -- no perf replays")
    for t_ in tids.values():
        ttnn.release_trace(mesh, t_)
elif not C.MOCK:
    perf = {k: S.replay_stats(mesh, tids[k], replays=REPLAYS, label=f"{tag} {k}") for k, _ in PATHS}
    us = {k: v.get("median_us") for k, v in perf.items()}
    row.update(us=us, p90_us={k: v.get("p90_us") for k, v in perf.items()}, replays=perf["spa"].get("replays"))
    if SPLIT:  # the glue = the path minus its measured components (layout round trips, clone, merge elementwise ops)
        row.update(glue_us=dict(spa=round(us["spa"] - us["gather"] - us["sdpa_cmp"] - us["q_a2a"] - us["o_a2a"], 1),
                                bq=round(us["bq"] - us["bq_q_allgather"] - us["bq_stats"] - us["bq_rs_fp32"], 1)))  # fmt: skip
    assert gcalls() == g0 + 2, ("eager guard: the gather launched during the replays", gcalls())
    assert pc() == pc0, ("a program compiled during the replays", pc(), pc0)
    if DEV and os.environ.get("GLM53_SPA_STAMPS", "0") == "1":  # the slope unit: phase times of the last replay
        row.update(spa_stamps=gat_b.stamps(), union_per_chip=[int(u.numel()) for u in unions], sel=SEL)
    for t_ in tids.values():
        ttnn.release_trace(mesh, t_)
S.log(f"{tag}: {row}")
if args.get("record", False):
    C.record("pcc", row)
RESULT = [row]
print("SPA_GATHER_UNIT", row, flush=True)
