"""Host side of the tiered latent cache's page-DMA path (Stage 12): the pinned page pool LMCache's CPU tier allocates
from, the device-native page format, and the descriptors the persistent page-DMA kernel (page_dma.cpp) executes.

Pool. N x 1 GiB hugepages from hugetlbfs, mapped as ONE virtual range (LMCache sees a single uint8 tensor) but
physically one extent per GiB: host-8chip runs iommu=pt, so a chip addresses host memory by physical address and only a
hugepage is physically contiguous. A descriptor therefore never crosses a GiB boundary (extents() splits at tile
granularity: every page and chunk size is a multiple of 2,048 B, and so is 1 GiB). The hugetlbfs file stays NAMED
while a page-DMA kernel may run: if the process dies with a SAVE in flight, the chip's writes land in pages the file
still owns instead of in RAM the kernel already gave to someone else. close(devices_stopped=True) unlinks it; after an
unclean death sweep_stale() removes the files of dead processes once the chips were re-initialised (open resets the
cores, which kills a stale kernel).

Format. A pool chunk is LMCache's memory object for 256 tokens of the 11 DSA layers (KV_2LTD, MLA: [1, 11, 256, 544]
bf16 = 3,063,808 B). Inside it, per layer, the 4 pages back to back, each page = the latent page in DEVICE-NATIVE
order (the 32 bf16 tiles of [64, 512] in tile index order, faces 16x16 inside a tile = the bytes of
tiles [32b, 32b + 32) of the device's interleaved latent cache) followed by the 4,096-B pooled-key block (reserved,
zero in v1). LMCache treats the bytes as opaque; the DMA copies them without reformatting. A copy path that goes
through row-major tensors (the dispatched ttnn path) converts with tilize_page / untilize_page, so both paths store
identical bytes and a prefix saved by one is loadable by the other. Only full chunks are stored
(LMCache save_unfull_chunk = False): hits are page aligned anyway (page_contract.md section 5).
"""

from __future__ import annotations

import ctypes
import functools
import mmap
import os
import re
import threading

import numpy as np

GIB = 1 << 30
TILE_BYTES = 2048  # bf16 32 x 32
TOKENS_PER_PAGE = 64
LATENT_DIM = 512
DSA_LAYERS = 11
LATENT_TILES_PER_PAGE = TOKENS_PER_PAGE * LATENT_DIM * 2 // TILE_BYTES  # 32
LATENT_PAGE_BYTES = LATENT_TILES_PER_PAGE * TILE_BYTES  # 65,536
POOLED_PAGE_BYTES = 4096
PAGE_LAYER_BYTES = LATENT_PAGE_BYTES + POOLED_PAGE_BYTES  # 69,632 = 64 tokens x 1,088 B
# the DSA pooled-key store (stack_state "pools", [rows x pps, 1, 32, 128] bf16): one 128-token unit = one [32, 128] page
# = 4 tiles; a 64-token LMCache page's pooled keys = HALF of it (rows 16 h .. 16 h + 15 = faces 2 h, 2 h + 1 of each
# tile = bytes [1,024 h, 1,024 h + 1,024) of each tile): POOLED_PAGE_BYTES, device-native (paged_pooled_keys.md)
POOLED_UNIT_TILES = 4
POOLED_UNIT_BYTES = POOLED_UNIT_TILES * 2048  # 8,192
assert 2 * POOLED_PAGE_BYTES == POOLED_UNIT_BYTES
CHUNK_PAGES = 4
CHUNK_TOKENS = CHUNK_PAGES * TOKENS_PER_PAGE  # 256
CHUNK_LAYER_BYTES = CHUNK_PAGES * PAGE_LAYER_BYTES  # 278,528
CHUNK_BYTES = DSA_LAYERS * CHUNK_LAYER_BYTES  # 3,063,808
assert CHUNK_BYTES == DSA_LAYERS * CHUNK_TOKENS * 1088 and CHUNK_BYTES % TILE_BYTES == 0 and GIB % TILE_BYTES == 0

# page_dma_proto.h PdmaDesc (32 B)
DESC = np.dtype([("op", "<u2"), ("tensor", "<u2"), ("tile0", "<u4"), ("n_tiles", "<u4"), ("slab", "<u4"),
                 ("slab_off", "<u4"), ("tag", "<u4"), ("rsvd", "<u4", (2,))])  # fmt: skip
assert DESC.itemsize == 32
NOP, LOAD, SAVE, STOP = 0, 1, 2, 3

HUGETLBFS = "/dev/hugepages-1G"
TAG = "glm53_pagepool"


class PinnedPagePool:
    """N GiB of host memory for LMCache's CPU tier and the page-DMA kernel. backing="hugetlbfs" (the real pool; the
    C++ driver pins each GiB for every chip) or "anon" (host tests: same layout, not pinnable)."""

    def __init__(self, gib: int, *, backing: str = "hugetlbfs", root: str = HUGETLBFS, tag: str = TAG):
        if gib < 1:
            raise ValueError("gib >= 1")
        self.gib, self.backing, self.path = gib, backing, None
        n = gib * GIB
        if backing == "hugetlbfs":
            self.path = os.path.join(root, f"{tag}_{os.getpid()}")
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.ftruncate(fd, n)
                self.mm = mmap.mmap(fd, n, flags=mmap.MAP_SHARED | getattr(mmap, "MAP_POPULATE", 0),
                                    prot=mmap.PROT_READ | mmap.PROT_WRITE)  # fmt: skip
            except Exception:
                os.close(fd)
                os.unlink(self.path)
                raise
            os.close(fd)
        elif backing == "anon":
            self.mm = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        else:
            raise ValueError(f"backing {backing!r}")
        self.array = np.frombuffer(self.mm, dtype=np.uint8)
        self.base = ctypes.addressof(ctypes.c_char.from_buffer(self.mm))

    def tensor(self, nbytes: int | None = None):
        """Zero-copy torch uint8 view of the first nbytes (LMCache's allocator buffer)."""
        import torch

        t = torch.frombuffer(self.mm, dtype=torch.uint8)
        return t if nbytes is None else t[:nbytes]

    def chunk_capacity(self) -> int:
        return self.gib * GIB // CHUNK_BYTES

    def offset_of(self, ptr: int) -> int:
        off = ptr - self.base
        if not 0 <= off < self.gib * GIB:
            raise ValueError("pointer outside the pool")
        return off

    def extents(self, off: int, n: int):
        """[(slab, slab_off, bytes)] covering pool bytes [off, off + n), split at GiB boundaries."""
        if off < 0 or off + n > self.gib * GIB:
            raise ValueError("range outside the pool")
        out = []
        while n > 0:
            slab, so = divmod(off, GIB)
            m = min(n, GIB - so)
            out.append((slab, so, m))
            off, n = off + m, n - m
        return out

    def close(self, devices_stopped: bool):
        """Unmap; unlink the hugetlbfs file only when no page-DMA kernel can still write into it."""
        try:
            del self.array
            self.mm.close()
        except BufferError:  # a live view (LMCache tensors) still exports the buffer: keep the mapping
            return False
        if self.path and devices_stopped:
            os.unlink(self.path)
            self.path = None
        return True

    @staticmethod
    def sweep_stale(root: str = HUGETLBFS, tag: str = TAG):
        """Remove pool files of processes that no longer exist. Call only after this process opened the chips (the
        open re-initialised every core, so no stale page-DMA kernel writes into those pages any more)."""
        gone = []
        for name in os.listdir(root):
            m = re.fullmatch(re.escape(tag) + r"_(\d+)", name)
            if m and not os.path.exists(f"/proc/{m.group(1)}"):
                os.unlink(os.path.join(root, name))
                gone.append(name)
        return gone


_SHARED: dict = {}


def shared_pool(ring_gib: int, lmcache_gib: int, backing: str | None = None) -> PinnedPagePool:
    """the process's ONE pinned pool when LMCache's CPU tier lives in it (lever (c) step 2): created by whichever asks
    first -- LMCache's allocator (lmcache_external_buffer, at the engine's init) or the model's LatentHitDma(shared=True)
    -- and adopted by the other, so their init order does not matter. ring_gib GiB of ring / state / completion, then
    lmcache_gib GiB of LMCache objects."""
    p = _SHARED.get(os.getpid())
    if p is None:
        mock = os.environ.get("GLM53_MOCK") == "1"
        p = PinnedPagePool(int(ring_gib) + int(lmcache_gib), backing=backing or ("anon" if mock else "hugetlbfs"))
        p.ring_gib, p.lmcache_gib = int(ring_gib), int(lmcache_gib)
        _SHARED[os.getpid()] = p
    elif (p.ring_gib, p.lmcache_gib) != (int(ring_gib), int(lmcache_gib)):
        raise ValueError(f"the shared pool is {p.ring_gib} + {p.lmcache_gib} GiB, asked {ring_gib} + {lmcache_gib}")
    return p


def lmcache_external_buffer(nbytes: int):
    """the LMCache fork's external CPU-buffer hook (extra_config external_cpu_buffer =
    'models.autoports.zai_org_glm_5_3_flash.tt.page_pool:lmcache_external_buffer'): the CPU tier's nbytes buffer = the
    shared pinned pool's LMCache region, above GLM53_LMCACHE_PAGE_DMA_GIB GiB of ring (device-visible: hits page in by
    descriptors at the objects' bytes, LatentHitDma.submit_resident). LMCache's 4,096-B alignment keeps every object
    page tile-aligned."""
    ring = int(os.environ.get("GLM53_LMCACHE_PAGE_DMA_GIB", "1"))
    p = shared_pool(ring, -(-int(nbytes) // GIB))
    return p.tensor()[ring * GIB : ring * GIB + int(nbytes)]


def page_descriptors(op: int, pool: PinnedPagePool, page_off: int, tensor: int, block: int, tag: int = 0):
    """Descriptors moving one latent page (block id `block` of the layer's latent cache = table entry `tensor`) to /
    from pool bytes [page_off, page_off + 65,536): one descriptor, two when the page crosses a GiB boundary."""
    out, tile0 = [], LATENT_TILES_PER_PAGE * block
    for slab, so, n in pool.extents(page_off, LATENT_PAGE_BYTES):
        k = n // TILE_BYTES
        out.append((op, tensor, tile0, k, slab, so, tag, (0, 0)))
        tile0 += k
    return out


def page_desc_array(op: int, pool: PinnedPagePool, off: int, tensor: int, blocks) -> np.ndarray:
    """page_descriptors for pages blocks[i] at pool bytes off + i x 65,536 (tag i), vectorized; a range that crosses a
    GiB boundary falls back to the per-page split"""
    blocks = np.asarray(blocks, dtype=np.int64)
    m = len(blocks)
    slab0, so0 = divmod(off, GIB)
    if m and (off + m * LATENT_PAGE_BYTES - 1) // GIB == slab0:
        d = np.zeros(m, dtype=DESC)
        d["op"], d["tensor"], d["n_tiles"], d["slab"] = op, tensor, LATENT_TILES_PER_PAGE, slab0
        d["tile0"] = LATENT_TILES_PER_PAGE * blocks
        d["slab_off"] = so0 + np.arange(m, dtype=np.int64) * LATENT_PAGE_BYTES
        d["tag"] = np.arange(m)
        return d
    return np.array([r for i, b in enumerate(blocks) for r in page_descriptors(op, pool, off + i * LATENT_PAGE_BYTES,
                                                                            tensor, int(b), tag=i)], dtype=DESC)  # fmt: skip


def page_desc_strided(op: int, pool: PinnedPagePool, off: int, stride: int, tensor: int, blocks, tag0: int = 0) -> np.ndarray:
    """page_descriptors for pages blocks[i] at pool bytes off + i x stride (tag tag0 + i), vectorized: one descriptor
    per page, two for a page that crosses a GiB boundary (page_descriptors' split; offsets must be tile-aligned)"""
    blocks = np.asarray(blocks, dtype=np.int64)
    m = len(blocks)
    offs = off + np.arange(m, dtype=np.int64) * int(stride)
    first, last = offs // GIB, (offs + LATENT_PAGE_BYTES - 1) // GIB
    d = np.zeros(m, dtype=DESC)
    d["op"], d["tensor"], d["n_tiles"] = op, tensor, LATENT_TILES_PER_PAGE
    d["tile0"] = LATENT_TILES_PER_PAGE * blocks
    d["slab"], d["slab_off"] = first, offs - first * GIB
    d["tag"] = tag0 + np.arange(m)
    cross = np.nonzero(first != last)[0]
    if len(cross) == 0:
        return d
    keep = np.ones(m, dtype=bool)
    keep[cross] = False
    split = [r for i in cross for r in page_descriptors(op, pool, int(offs[i]), tensor, int(blocks[i]), tag=tag0 + int(i))]
    return np.concatenate([d[keep], np.array(split, dtype=DESC)])


def page_desc_offsets(op: int, pool: PinnedPagePool, offs, tensor: int, blocks, tags) -> np.ndarray:
    """page_descriptors for pages blocks[i] at pool bytes offs[i] (tag tags[i]), vectorized; a page that crosses a GiB
    boundary takes two descriptors (page_descriptors' split; offsets tile-aligned)"""
    offs, blocks, tags = (np.asarray(v, dtype=np.int64) for v in (offs, blocks, tags))
    first, last = offs // GIB, (offs + LATENT_PAGE_BYTES - 1) // GIB
    d = np.zeros(len(offs), dtype=DESC)
    d["op"], d["tensor"], d["n_tiles"] = op, tensor, LATENT_TILES_PER_PAGE
    d["tile0"] = LATENT_TILES_PER_PAGE * blocks
    d["slab"], d["slab_off"], d["tag"] = first, offs - first * GIB, tags
    cross = np.nonzero(first != last)[0]
    if len(cross) == 0:
        return d
    keep = np.ones(len(offs), dtype=bool)
    keep[cross] = False
    split = [r for i in cross for r in page_descriptors(op, pool, int(offs[i]), tensor, int(blocks[i]), tag=int(tags[i]))]
    return np.concatenate([d[keep], np.array(split, dtype=DESC)])


def tile_object_pages(rows):
    """the device-native PAGE format of a tile-order LMCache object (GLM53_KV_TILE_OBJECTS, lever (c) step 2): rows a
    [layers, 64 m tokens, 1,088] uint8 object, page p of layer l = bytes [p x 69,632, +69,632) of the layer's rows =
    the page's 32 latent tiles (65,536 B, the device's tile order) then its 4,096-B pooled half (device-native) -- the
    same bytes as 64 token rows, so the object's size and token slicing are LMCache's. -> ([L, m, 32, 2,048] latent
    tiles, [L, m, 4,096] pooled halves), views"""
    L, t, w = (int(v) for v in rows.shape)
    if w != PAGE_LAYER_BYTES // TOKENS_PER_PAGE or t % TOKENS_PER_PAGE:
        raise ValueError(f"tile-order object rows {tuple(rows.shape)}: want [layers, 64 m, 1,088]")
    pages = rows.reshape(L, t // TOKENS_PER_PAGE, PAGE_LAYER_BYTES)
    return (pages[..., :LATENT_PAGE_BYTES].reshape(L, -1, LATENT_TILES_PER_PAGE, TILE_BYTES),
            pages[..., LATENT_PAGE_BYTES:])  # fmt: skip


def chunk_descriptors(op: int, pool: PinnedPagePool, chunk_off: int, blocks, latent_tensors, tag: int = 0):
    """All descriptors of one pool chunk: for each DSA layer l (table entry latent_tensors[l]) and page p of the chunk,
    the page of device block blocks[p]. len(blocks) == 4 (full chunks only)."""
    if len(blocks) != CHUNK_PAGES or len(latent_tensors) != DSA_LAYERS:
        raise ValueError("a full chunk: 4 block ids, 11 latent tensors")
    rows = []
    for l, t in enumerate(latent_tensors):
        for p, b in enumerate(blocks):
            rows += page_descriptors(op, pool, chunk_off + l * CHUNK_LAYER_BYTES + p * PAGE_LAYER_BYTES, t, b, tag)
    return np.array(rows, dtype=DESC)


def split_cores(desc: np.ndarray, cores: int):
    """Round-robin over the page-DMA cores (descriptors are independent; each core keeps their order)."""
    return [desc[c::cores] for c in range(cores)]


# ---- device-native page format <-> row-major [64, 512] (for copy paths that move row-major tensors)
def tilize_page(rows: np.ndarray) -> np.ndarray:
    """[64, 512] 2-byte elements -> the 32 tiles of 32 x 32 in tile index order, each tile = faces (0,0) (0,1) (1,0)
    (1,1) of 16 x 16, row-major inside a face (ttnn TILE layout)."""
    a = np.ascontiguousarray(rows).reshape(2, 2, 16, 16, 2, 16)  # [tile_r, face_r, r, tile_c(16), face_c, c]
    # axes: tr, fr, r, tc, fc, c -> tr, tc, fr, fc, r, c
    return a.transpose(0, 3, 1, 4, 2, 5).reshape(-1)


def untilize_page(flat: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(flat).reshape(2, 16, 2, 2, 16, 16)  # tr, tc, fr, fc, r, c
    return a.transpose(0, 2, 4, 1, 3, 5).reshape(TOKENS_PER_PAGE, LATENT_DIM)


DONE_AREA = 64 * 1024  # completion records at the pool's end (64 B per core per chip); LMCache's buffer stops before it


def lmcache_buffer_bytes(pool: PinnedPagePool) -> int:
    """Bytes of the pool handed to LMCache's allocator: whole chunks, the completion area excluded."""
    return (pool.gib * GIB - DONE_AREA) // CHUNK_BYTES * CHUNK_BYTES


def tile_count(t) -> int:
    """Pages of an interleaved bf16 TILE tensor (one 32 x 32 tile = one 2,048-B page)."""
    s = [int(v) for v in t.shape]
    lead = int(np.prod(s[:-2])) if len(s) > 2 else 1
    return lead * (-(-s[-2] // 32)) * (-(-s[-1] // 32))


def pooled_halves(units: np.ndarray) -> np.ndarray:
    """device-native pooled-key unit pages [m, 8,192 B] -> their two 64-token halves [m, 2, 4,096 B] (half h = rows
    16 h .. 16 h + 15 of the [32, 128] page = bytes [1,024 h, + 1,024) of each of its 4 tiles, tile after tile)"""
    u = np.ascontiguousarray(units).view(np.uint8).reshape(-1, POOLED_UNIT_TILES, 2, TILE_BYTES // 2)
    return u.transpose(0, 2, 1, 3).reshape(-1, 2, POOLED_PAGE_BYTES)


def pooled_units(halves: np.ndarray) -> np.ndarray:
    """inverse of pooled_halves: [m, 2, 4,096 B] -> device-native unit pages [m, 8,192 B]"""
    h = np.ascontiguousarray(halves).view(np.uint8).reshape(-1, 2, POOLED_UNIT_TILES, TILE_BYTES // 2)
    return h.transpose(0, 2, 1, 3).reshape(-1, POOLED_UNIT_BYTES)


def block_page_of_half(halves: np.ndarray) -> np.ndarray:
    """GLM53_DSA_PAGED_POOLS (v1 padded layout, paged_pooled_keys.md rev 3): 64-token halves [m, 4,096 B] -> the paged
    store's block pages [m, 8,192 B] (the half in rows 0-15 = the first 1,024 B of each tile; rows 16-31 zero)"""
    h = np.ascontiguousarray(halves).view(np.uint8).reshape(-1, POOLED_UNIT_TILES, 1, TILE_BYTES // 2)
    return np.concatenate([h, np.zeros_like(h)], axis=2).reshape(-1, POOLED_UNIT_BYTES)


def half_of_block_page(pages: np.ndarray) -> np.ndarray:
    """inverse of block_page_of_half: block pages [m, 8,192 B] -> their 64-token halves [m, 4,096 B] (rows 0-15)"""
    p = np.ascontiguousarray(pages).view(np.uint8).reshape(-1, POOLED_UNIT_TILES, 2, TILE_BYTES // 2)
    return np.ascontiguousarray(p[:, :, 0]).reshape(-1, POOLED_PAGE_BYTES)


def _put(dst: np.ndarray, src) -> None:
    """dst[...] = src (strided numpy views of 2-byte elements; src an array or 0): torch's threaded copy when importable
    (tilize_pages' idiom: numpy's strided copy is single-threaded), numpy's otherwise and for a read-only src"""
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # (host tools without torch)
        torch = None
    if torch is None or not (np.isscalar(src) or src.flags.writeable):
        dst[...] = src
    elif np.isscalar(src):
        torch.from_numpy(dst).fill_(src)
    else:
        torch.from_numpy(dst).copy_(torch.from_numpy(src))


def _stage_halves(dest: np.ndarray, units: np.ndarray, j0: int, j1: int) -> None:
    """block pages j0..j1-1 of a row (block j = half j % 2 of unit j // 2) into rows 0-15 of dest [j1 - j0, 4, 2,
    512] int16 (block_page_of_half layout; rows 16-31 left as they are) from units [u, 4, 2, 512] int16 (pooled_halves'
    view)"""
    j, o = j0, 0
    if j % 2 and j < j1:  # an odd start: the second half of unit j // 2
        _put(dest[o, :, 0], units[j // 2, :, 1])
        j, o = j + 1, o + 1
    f = (j1 - j) // 2
    if f:  # whole units: [f, 2 halves, 4 tiles, 512] <- [f, 4 tiles, 2 halves, 512]
        d = dest[o : o + 2 * f].reshape(f, 2, POOLED_UNIT_TILES, 2, TILE_BYTES // 4)
        _put(d[:, :, :, 0], units[j // 2 : j // 2 + f].transpose(0, 2, 1, 3))
        j, o = j + 2 * f, o + 2 * f
    if j < j1:  # an odd end: the first half of unit j // 2
        _put(dest[o, :, 0], units[j // 2, :, 0])


class LatentPageDma:
    """The latent caches' page-DMA: PageDma (ttnn._ttnn.dram_image, C++) over the 11 DSA layers' latent caches
    (replicated [num_blocks, 1, 64, 512] bf16 TILE, DRAM interleaved) and the pinned pool. LMCache chunk objects are
    addressed by their data pointer (a view of the pool); block ids are the device blocks of the chunk's 4 pages.
    impl: a PageDma-like object (tests pass a fake)."""

    def __init__(self, mesh, latent_caches, pool: PinnedPagePool, *, cores: int = 4, ring: int = 256,
                 stage: int = 32768, save_chip=(0, 0), impl=None, extra=(), region_bytes: int | None = None):  # fmt: skip
        if len(latent_caches) != DSA_LAYERS:
            raise ValueError(f"{len(latent_caches)} latent caches, want {DSA_LAYERS}")
        self.pool = pool
        # table entries [address, pages, page bytes]: the 11 latent caches, then `extra` (e.g. the state snapshot
        # buffers of LatentHitDma(state_tensors=): entries 11, 12, ...)
        self.tensors = [[int(t.buffer_address()), tile_count(t), TILE_BYTES] for t in latent_caches]
        self.tensors += [list(e) for e in extra]
        self.ids = list(range(DSA_LAYERS))  # table entry of layer l
        # the completion area at the top of the DMA's own region: the whole pool, or its first region_bytes when the
        # rest is LMCache's CPU tier (LatentHitDma(lmcache_gib=); the C++ driver still pins every GiB of the pool)
        self.done_off = (pool.gib * GIB if region_bytes is None else int(region_bytes)) - DONE_AREA
        if impl is None:
            import ttnn

            impl = ttnn._ttnn.dram_image.PageDma(mesh, pool.base, pool.gib, cores_per_chip=cores, ring=ring, stage=stage,
                                                 tensors=self.tensors, done_offset=self.done_off,
                                                 save_chip=tuple(save_chip))  # fmt: skip
        self.impl = impl
        self.started = False

    def start(self, timeout_s: float = 10.0):
        self.started = True  # first: a start that fails after the launch still needs stop()
        self.impl.start(timeout_s)

    def _desc(self, op, chunks):
        rows = [chunk_descriptors(op, self.pool, self.pool.offset_of(ptr), blocks, self.ids, tag=i)
                for i, (ptr, blocks) in enumerate(chunks)]  # fmt: skip
        return np.concatenate(rows) if rows else np.zeros(0, DESC)

    def load(self, chunks, timeout_s: float = 10.0):
        """chunks = [(object data_ptr, [4 block ids])] -> queued on every chip (call wait() before the step reads them)."""
        d = self._desc(LOAD, chunks)
        self.impl.load(d.tobytes(), timeout_s)
        return len(d)

    def save(self, chunks, timeout_s: float = 10.0):
        """chunks = [(object data_ptr, [4 block ids])] -> queued on the save chip (only after the step that wrote them)."""
        d = self._desc(SAVE, chunks)
        self.impl.save(d.tobytes(), timeout_s)
        return len(d)

    def wait(self, timeout_s: float = 10.0):
        self.impl.wait(timeout_s)

    def stop(self, timeout_s: float = 10.0):
        """STOP the kernels and release the driver - call before the mesh closes (raises if a kernel did not stop:
        then the pool must stay mapped and its file named)."""
        if self.started:
            self.impl.stop(timeout_s)
            self.started = False
        self.impl = None  # the C++ object goes while the mesh is open


def untilize_pages(flat: np.ndarray) -> np.ndarray:
    """inverse of tilize_pages: [n, 32 tiles x 1,024] device order -> [n, 64, 512] row-major 2-byte elements.
    Stage 12 store bandwidth (program 2026-10-24T00:59:38Z): the store read's host untilize was 93 % of read_pages
    (d0 pair, 32K: 0.88 s per 0.36 GB) as numpy's single-threaded transpose copy (0.6-1.4 GB/s); torch's copy runs on
    the intra-op threads (4.0 GB/s into a fresh buffer at 8), the same permutation, as load_many already tilizes."""
    n = flat.shape[0]
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # (host tools without torch)
        a = np.ascontiguousarray(flat).reshape(n, 2, 16, 2, 2, 16, 16)  # n, tr, tc, fr, fc, r, c
        return a.transpose(0, 1, 3, 5, 2, 4, 6).reshape(n, TOKENS_PER_PAGE, LATENT_DIM)
    src = np.ascontiguousarray(flat)
    t = torch.from_numpy(src.view(np.int16)).view(n, 2, 16, 2, 2, 16, 16).permute(0, 1, 3, 5, 2, 4, 6)
    out = torch.empty((n, 2, 2, 16, 16, 2, 16), dtype=torch.int16)  # n, tr, fr, r, tc, fc, c = row-major [n, 64, 512]
    out.copy_(t)
    return out.numpy().view(src.dtype).reshape(n, TOKENS_PER_PAGE, LATENT_DIM)


class TiledPages:
    """one latent layer's pages as read_pages(untilize=False) copied them out of the staging ring: [n, 32 tiles x 1,024]
    uint16 in device order. result() -> torch [n, 64 x 512] int16, row-major: the bytes read_pages(untilize=True)
    returns (the store worker's protocol, as the plugin's pending page reads); nbytes for its in-flight accounting."""

    def __init__(self, tiles: np.ndarray):
        try:  # torch's threaded copy out of the ring (~16x numpy's single-threaded copy on host-8chip)
            import torch  # noqa: PLC0415

            self.tiles = torch.from_numpy(tiles.view(np.int16)).clone().numpy().view(np.uint16)
        except ImportError:  # (host tools without torch)
            self.tiles = np.array(tiles)
        self.nbytes = int(self.tiles.nbytes)

    def result(self):
        import torch  # noqa: PLC0415

        n = self.tiles.shape[0]
        return torch.from_numpy(untilize_pages(self.tiles).view(np.int16).reshape(n, -1))


def tilize_pages(rows: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """tilize_page over n pages at once: [n, 64, 512] 2-byte elements -> [n, 32 tiles x 1,024] in device order. `out`
    (a contiguous [n, 32,768] 2-byte buffer, e.g. the pinned ring) takes the tiles in place. torch's threaded copy when
    importable (Stage 12 load bandwidth: the same permutation as numpy's single-threaded transpose, bitwise)."""
    n = rows.shape[0]
    try:
        import torch  # noqa: PLC0415
    except ImportError:  # (host tools without torch)
        t = np.ascontiguousarray(rows).reshape(n, 2, 2, 16, 16, 2, 16).transpose(0, 1, 4, 2, 5, 3, 6).reshape(n, -1)
        if out is None:
            return t
        out[...] = t.reshape(out.shape)
        return out
    src = np.ascontiguousarray(rows)
    t = torch.from_numpy(src.view(np.int16)).view(n, 2, 2, 16, 16, 2, 16).permute(0, 1, 4, 2, 5, 3, 6)
    if out is None:
        res = torch.empty((n, 2, 16, 2, 2, 16, 16), dtype=torch.int16)  # n, tr, tc, fr, fc, r, c = device order
        res.copy_(t)
        return res.numpy().view(src.dtype).reshape(n, -1)
    torch.from_numpy(out.view(np.int16)).view(n, 2, 16, 2, 2, 16, 16).copy_(t)
    return out


class RecordingPageDma:
    """PageDma stand-in for the mock cluster (its kernels never run) and host tests: keeps every descriptor batch;
    apply (optional) executes LOADs against a host model of the latent caches [layer][tile] (tests)."""

    def __init__(self, pool: PinnedPagePool, apply=None, apply_chips: bool = False):
        self.pool, self.apply, self.apply_chips, self.batches, self.chips, self.started = pool, apply, apply_chips, [], [], False
        self.layout = None  # (chips, cores, done_off): set by LatentHitDma -> the host completion slots are written

    def start(self, timeout_s):
        self.started = True

    def _complete(self, n: int, chips) -> None:
        """every descriptor completes at once: advance each target (chip, core)'s host completion slot like the kernel"""
        if self.layout is None:
            return
        n_chips, cores, done_off = self.layout
        for ch in chips if chips else range(n_chips):
            for k in range(min(cores, n)):
                o = done_off + 64 * (ch * cores + k)
                w = self.pool.array[o : o + 4].view(np.uint32)
                w[0] += (n - k + cores - 1) // cores

    def load(self, desc: bytes, timeout_s, chips=()):
        d = np.frombuffer(desc, dtype=DESC)
        self.batches.append(d)
        self.chips.append(list(chips))
        if self.apply is not None:
            self.apply(d, list(chips)) if self.apply_chips else self.apply(d)
        self._complete(len(d), list(chips))

    def save(self, desc: bytes, timeout_s, chips=()):
        d = np.frombuffer(desc, dtype=DESC)
        self.batches.append(d)
        self.chips.append(list(chips))
        if self.apply is not None:
            self.apply(d, list(chips)) if self.apply_chips else self.apply(d)
        self._complete(len(d), list(chips) or [0])

    def wait(self, timeout_s):
        pass

    def stats(self):
        return dict(descriptors=int(sum(len(b) for b in self.batches)), batches=len(self.batches))

    def stop(self, timeout_s):
        self.started = False


def _pages_of(t) -> int:
    """pages (tiles) of an interleaved TILE tensor, per chip"""
    s = [int(v) for v in t.padded_shape]
    lead = int(np.prod(s[:-2])) if len(s) > 2 else 1
    return lead * (s[-2] // 32) * (s[-1] // 32)


def _page_bytes_of(t) -> int:
    """bytes of one 32 x 32 tile of t's dtype (fp32 4,096; bf16 2,048)"""
    name = str(t.dtype).lower()
    return 4096 if ("float32" in name or "int32" in name or "uint32" in name) else TILE_BYTES


def untilize_tiles(raw: np.ndarray, elem: np.dtype) -> np.ndarray:
    """device tile bytes [..., tile bytes] -> logical [..., 32, 32] (faces (0,0) (0,1) (1,0) (1,1) of 16 x 16,
    row-major inside a face: what ttnn.to_torch of a TILE tensor returns per tile)"""
    a = np.ascontiguousarray(raw).view(elem)
    lead = a.shape[:-1]
    return a.reshape(*lead, 2, 2, 16, 16).swapaxes(-3, -2).reshape(*lead, 32, 32)


class StateTicket:
    """One state save (LatentHitDma.save_state): done() polls the host completion slots (no device access, any
    thread); fetch() waits, copies the state region's bytes out (then the region may be reused) -- once; tiles(i)
    returns state tensor i as logical tiles [chips, pages, 32, 32] of `elem` (np.float32 / a 2-byte dtype)."""

    def __init__(self, dma: "LatentHitDma", layout, targets, before=None, done0=None):
        import threading
        import time

        self.dma, self.layout, self.targets = dma, layout, targets
        self.lock = threading.Lock()
        self.raw = None  # [chips] bytes of the region, after fetch
        self.released = False
        # the KDA export timing diagnostic (Stage 14, program 2026-10-28T08:09:34Z): per (chip, core) ring, the
        # descriptors pushed before this save (before) and completed when it was pushed (done0)
        self.t_save = time.perf_counter()
        self.before, self.done0 = before, done0

    def done(self) -> bool:
        got = self.dma.done_counts()
        return all(g >= t for gr, tr in zip(got, self.targets) for g, t in zip(gr, tr))

    def fetch(self, timeout_s: float = 10.0) -> None:
        import time

        with self.lock:
            if self.raw is not None:
                return
            t0 = time.perf_counter()
            chip_done = [None] * len(self.targets)  # seconds after save_state at which each chip's SAVE completed
            samples, t_next = [], t0  # (s after save_state, this save's descriptors completed per chip) every 0.5 s
            while True:
                got = self.dma.done_counts()
                now = time.perf_counter()
                for ch, (gr, tr) in enumerate(zip(got, self.targets)):
                    if chip_done[ch] is None and all(g >= t for g, t in zip(gr, tr)):
                        chip_done[ch] = round(now - self.t_save, 3)
                if all(c is not None for c in chip_done):
                    break
                if self.before is not None and now >= t_next and len(samples) < 40:
                    samples.append((round(now - self.t_save, 2), [
                        sum(max(0, min(g, t) - b) for g, t, b in zip(gr, tr, br))
                        for gr, tr, br in zip(got, self.targets, self.before)]))  # fmt: skip
                    t_next = now + 0.5
                if now - t0 > timeout_s:
                    raise TimeoutError(f"state save not complete after {timeout_s} s: {got} < {self.targets}")
                time.sleep(0.0005)
            t1 = time.perf_counter()
            d, a = self.dma, self.dma.pool.array
            base = [d.state_off + ch * d.chip_state_bytes for ch in range(d.chips)]
            self.raw = [[a[b + o : b + o + n * pb].copy() for o, n, pb in self.layout] for b in base]  # the used tiles only
            self.released = True
            t2 = time.perf_counter()
            if t1 - t0 > float(os.environ.get("GLM53_PAGE_DMA_STATE_LOG_S", "1.0")):
                ahead = this = None
                if self.before is not None and self.done0 is not None:
                    ahead = [sum(b - g for b, g in zip(br, gr)) for br, gr in zip(self.before, self.done0)]
                    this = [sum(t - b for t, b in zip(tr, br)) for tr, br in zip(self.targets, self.before)]
                print(f"PAGE_DMA_STATE_WAIT fetch waited {t1 - t0:.3f} s ({t1 - self.t_save:.3f} s after save_state); "
                      f"descriptors ahead per chip at save {ahead}; this save per chip {this}; chip done at {chip_done} s "
                      f"after save; progress {samples}; push {getattr(self, 'push_s', 0.0):.3f} s; copy-out {t2 - t1:.3f} s; region waits "
                      f"{d.stats.get('state_region_waits')}", flush=True)  # fmt: skip

    def tiles(self, i: int, elem) -> np.ndarray:
        self.fetch()
        _o, n, pb = self.layout[i]
        return untilize_tiles(np.stack([r[i].reshape(n, pb) for r in self.raw]), np.dtype(elem))


class LatentHitDma:
    """Device-initiated page-in of LMCache hits (t2_pagein_design.md, GLM53_LMCACHE_PAGE_DMA=1): the connector's
    row-major host pages of the 11 DSA layers are tilized into a pinned staging ring and the persistent page-DMA kernel
    LOADs them into the latent caches (one descriptor per page per layer, two across a GiB boundary). submit() returns
    once the descriptors are queued; the DMA runs beside whatever the command queue executes (the previous step's
    trace); wait() before the step that reads the pages. Replaces the per-page from_torch + paged_fill_cache h2d (a
    dispatched program and a staging allocation per page per layer). Built before the first trace capture (the kernel
    start is a program launch); stop() before the mesh closes. impl: PageDma (device), RecordingPageDma (mock)."""

    def __init__(self, mesh, latent_caches, *, gib: int = 1, cores: int = 4, impl=None, backing: str | None = None,
                 state_tensors=(), pool_tensors=(), chips: int | None = None, ring: int = 256, lmcache_gib: int = 0,
                 shared: bool = False):  # fmt: skip
        mock = os.environ.get("GLM53_MOCK") == "1"
        # lmcache_gib (Stage 12 lever (c) step 2, program 2026-10-26T03:59:21Z): the pool's top lmcache_gib GiB are
        # LMCache's CPU tier (lmcache_tensor(): the allocator's external buffer), device-visible like the ring, so a hit
        # whose objects live there pages in by descriptors at their own bytes (submit_resident: no host copy)
        bk = backing or ("anon" if mock or impl is not None else "hugetlbfs")
        # shared (the served model): the process's one pool, which LMCache's allocator may have created first
        self.pool = shared_pool(gib, lmcache_gib, bk) if shared and lmcache_gib else PinnedPagePool(gib + int(lmcache_gib), backing=bk)
        self.lmcache_off, self.lmcache_bytes = gib * GIB, int(lmcache_gib) * GIB
        if impl is None and mock:
            impl = RecordingPageDma(self.pool)
        # state_tensors (Stage 12, StackBackend.state_snapshot_tensors): the persistent device buffers one slot's
        # checkpoint is snapshotted into; table entries 11, 12, ...; save_state() SAVEs them, every chip its own tiles,
        # into the pool's state region (the top of the pool, below the completion area); the hit ring keeps the rest
        self.state = [[int(t.buffer_address()), _pages_of(t), _page_bytes_of(t)] for t in state_tensors]
        # pool_tensors (Stage 12, GLM53_KV_POOLED_PAGES): the 11 DSA layers' pooled-key stores (latent-cache order),
        # table entries after the state buffers; read_pooled / load_pooled move their [32, 128] unit pages
        self.pools = [[int(t.buffer_address()), _pages_of(t), _page_bytes_of(t)] for t in pool_tensors]
        if self.pools and len(self.pools) != DSA_LAYERS:
            raise ValueError(f"{len(self.pools)} pooled-key stores, want {DSA_LAYERS}")
        # ring (Stage 12 lever (c)): descriptors per service core's L1 ring; a submit past cores x ring blocks on ring
        # space while the kernels drain it (the chips then fill one after another) -- a 32K hit is 5,478 descriptors
        self.dma = LatentPageDma(mesh, list(latent_caches), self.pool, cores=cores, impl=impl, ring=int(ring),
                                 extra=self.state + self.pools, region_bytes=gib * GIB)  # fmt: skip
        self.cores = cores
        self.chips = int(chips if chips is not None else (mesh.get_num_devices() if mesh is not None else 1))
        self.state_ids = list(range(DSA_LAYERS, DSA_LAYERS + len(self.state)))
        self.pool_ids = list(range(DSA_LAYERS + len(self.state), DSA_LAYERS + len(self.state) + len(self.pools)))
        self.chip_state_bytes = sum(-(-n * pb // 4096) * 4096 for _a, n, pb in self.state)
        self.state_off = (self.dma.done_off - self.chips * self.chip_state_bytes) // 4096 * 4096
        if self.state_off < LATENT_PAGE_BYTES * DSA_LAYERS:
            raise ValueError(f"the state region ({self.chips} x {self.chip_state_bytes} B) does not fit a {gib} GiB pool")
        self.slots = self.state_off // LATENT_PAGE_BYTES  # staging ring of whole pages (state + completion excluded)
        self.tails = [[0] * cores for _ in range(self.chips)]  # descriptors pushed per (chip, core) = the C++ tails
        if isinstance(self.dma.impl, RecordingPageDma):
            self.dma.impl.pool = self.dma.impl.pool or self.pool
            self.dma.impl.layout = (self.chips, cores, self.dma.done_off)
        self.ticket = None  # the state region's current ticket
        self._after = []  # (thread, fn): after_wait callbacks (e.g. the holds on the objects a submit_resident reads)
        # store path P3 (Stage 12): one transaction at a time -- every public method holds this re-entrant lock, so a
        # store worker thread's SAVEs (save_resident / save_resident_pooled / read_pages) never interleave with the
        # engine thread's page-ins, pooled loads or state saves (they share the ring, the tails and `pending`)
        self._lock = threading.RLock()
        self._tls = threading.local()  # per thread: the lock depth + the after_wait callbacks whose descriptors landed
        self.head, self.pending = 0, 0
        self.stats = dict(submits=0, pages=0, descriptors=0, tilize_s=0.0, submit_s=0.0, wait_s=0.0, state_saves=0,
                          state_bytes=0, state_region_waits=0)  # fmt: skip

    def _account(self, n: int, chips) -> None:
        """mirror the C++ submit: a batch of n descriptors goes round-robin over each target chip's cores"""
        for ch in chips:
            for k in range(min(self.cores, n)):
                self.tails[ch][k] += (n - k + self.cores - 1) // self.cores

    def done_counts(self):
        """[(chip, core) -> descriptors completed] from the host completion slots (plain memory reads: any thread)"""
        a = self.pool.array
        return [[int(a[o : o + 4].view(np.uint32)[0]) for o in
                 (self.dma.done_off + 64 * (ch * self.cores + k) for k in range(self.cores))] for ch in range(self.chips)]  # fmt: skip

    def save_state(self, pages, timeout_s: float = 10.0) -> "StateTicket":
        """SAVE the first pages[i] pages of state tensor i (every chip its own tiles, PageDma.save(chips=[k])) into the
        state region: chip k's bytes at state_off + k x chip_state_bytes, tensor after tensor (4 KiB aligned). The
        snapshot must have landed in the buffers (the caller synchronised the command queue). Returns the ticket; the
        region is reused only after the previous ticket released it (its bytes copied out)."""
        if len(pages) != len(self.state):
            raise ValueError(f"{len(pages)} page counts for {len(self.state)} state tensors")
        if self.ticket is not None and not self.ticket.released:
            self.stats["state_region_waits"] += 1
            self.ticket.fetch(timeout_s)  # the main thread takes the previous bytes out of the region
        layout, off = [], 0
        for (_a, n_all, pb), n in zip(self.state, pages):
            if not 0 <= int(n) <= n_all:
                raise ValueError(f"{n} pages of a {n_all}-page state tensor")
            layout.append((off, int(n), pb))
            off += -(-n_all * pb // 4096) * 4096
        import time  # noqa: PLC0415

        before, done0, t_push = [list(t) for t in self.tails], self.done_counts(), time.perf_counter()  # (diagnostic)
        total = 0
        for ch in range(self.chips):
            base = self.state_off + ch * self.chip_state_bytes
            rows = []
            for tid, (o, n, pb) in zip(self.state_ids, layout):
                tile0 = 0
                for slab, so, nb in self.pool.extents(base + o, n * pb) if n else ():
                    k = nb // pb
                    rows.append((SAVE, tid, tile0, k, slab, so, ch, (0, 0)))
                    tile0 += k
            if rows:
                d = np.array(rows, dtype=DESC)
                self.dma.impl.save(d.tobytes(), timeout_s, chips=[ch])
                self._account(len(d), [ch])
                total += len(d)
        self.stats["state_saves"] += 1
        self.stats["state_bytes"] += self.chips * sum(n * pb for _o, n, pb in layout)
        self.ticket = StateTicket(self, layout, [list(t) for t in self.tails], before=before, done0=done0)
        self.ticket.push_s = time.perf_counter() - t_push  # the descriptor pushes (they block on ring space)
        return self.ticket

    def start(self, timeout_s: float = 10.0):
        self.dma.start(timeout_s)

    def submit(self, pages_by_layer, block_ids, timeout_s: float = 10.0, owners=None) -> int:
        """pages_by_layer: 11 arrays (layer order = the latent caches') [n, 64, 512] 2-byte elements (or anything
        viewable as [n, 65,536] bytes, row-major) for device blocks block_ids [n]. owners None: queued on every chip
        (replicated caches); owners [n] (mesh-order chip of each page; block_ids are then the owner's LOCAL blocks):
        the sequence-sharded pool -- each page's descriptors go to its owner chip only (PageDma.load(chips=)).
        Returns the descriptor count."""
        import time

        n = len(block_ids)
        if len(pages_by_layer) != DSA_LAYERS:
            raise ValueError(f"{len(pages_by_layer)} layers, want {DSA_LAYERS}")
        if n == 0:
            return 0
        if DSA_LAYERS * n > self.slots:
            raise ValueError(f"{n} pages x {DSA_LAYERS} layers exceed the staging ring ({self.slots} pages)")
        if self.pending:
            self.wait(timeout_s)  # the ring is reused per submit: the previous one must have landed
        if owners is not None and len(owners) != n:
            raise ValueError(f"{len(owners)} owners for {n} pages")
        t0 = time.perf_counter()
        base = 0 if self.head + DSA_LAYERS * n > self.slots else self.head
        for l, pages in enumerate(pages_by_layer):  # tilized straight into the ring (threaded; no staging copy)
            off = (base + l * n) * LATENT_PAGE_BYTES
            tilize_pages(np.asarray(pages).view(np.uint16).reshape(n, TOKENS_PER_PAGE, LATENT_DIM),
                         out=self.pool.array[off : off + n * LATENT_PAGE_BYTES].view(np.uint16).reshape(n, -1))
        rows = self._desc_by_owner(LOAD, base, block_ids, owners, None)  # owner (None = every chip) -> descriptors
        t1 = time.perf_counter()
        total = 0
        for o, d in rows.items():
            if o is None:
                self.dma.impl.load(d.tobytes(), timeout_s)
                self._account(len(d), range(self.chips))
            else:
                self.dma.impl.load(d.tobytes(), timeout_s, chips=[o])
                self._account(len(d), [o])
            total += len(d)
        self.head, self.pending = base + DSA_LAYERS * n, total
        s = self.stats
        s["submits"] += 1
        s["pages"] += n
        s["descriptors"] += total
        s["tilize_s"] += t1 - t0
        s["submit_s"] += time.perf_counter() - t1
        return total

    def _push_groups(self, op: int, groups: dict, timeout_s: float, default_chips) -> int:
        """push each owner group's descriptors (op LOAD / SAVE) and count it PER GROUP (program 2026-10-26T16:12:35Z
        S1): self.pending grows by a group the moment its push returns, so a ring-space TimeoutError on a later group
        leaves every queued group counted -- the next wait() (a later submit's first) still waits for their bytes.
        default_chips: the target chips of the owner-less group (a load: every chip; a save: the save chip)."""
        push = self.dma.impl.load if op == LOAD else self.dma.impl.save
        total = 0
        for o, d in groups.items():
            try:
                if o is None:
                    push(d.tobytes(), timeout_s)
                else:
                    push(d.tobytes(), timeout_s, chips=[o])
            except BaseException:
                # Stage 14 re-review (program 16:31:03Z): the C++ push publishes run by run and can throw PART-WAY
                # through a group -- some of its descriptors are in flight. Count the group as pending so the next
                # wait() drains the DMA (impl.wait waits for everything published) before any after_wait release, and
                # mark the per-core tails unknown: that wait resyncs them to the completion slots.
                self.pending += len(d)
                self._tails_stale = True
                raise
            self._account(len(d), default_chips if o is None else [o])
            self.pending += len(d)
            total += len(d)
        return total

    def submit_objects(self, objects, block_ids, timeout_s: float = 10.0, owners=None, layer_order=None) -> int:
        """Stage 12 lever (c) (program 2026-10-26T00:15:39Z): submit() straight from LMCache memory objects, no
        concatenated copy and no separate tilize pass -- each object's latent rows are tilized into their ring slots by
        ONE strided copy per object for all layers (a 32K hit, 498 pages x 11 layers: 33 ms vs 116 ms for the
        torch.cat + tilize_pages it replaces; doc/ttnn/12-fast-start/lever_c/bench_load_variants.py, bitwise).
        objects = [(rows, m)]: rows a [layers, >= 64 m tokens, >= 1,024 B] uint8 view with ANY token-row stride (the
        object's [L, tokens, 1,088 B] rows sliced to the 1,024 latent bytes), holding m full pages of block_ids in
        order; layer_order[j] = the object layer that goes to latent cache j (None: identity). Owners / descriptors /
        the ring rules as submit(). Returns the descriptor count."""
        import time

        import torch

        n = len(block_ids)
        if sum(int(m) for _r, m in objects) != n:
            raise ValueError(f"{sum(int(m) for _r, m in objects)} object pages for {n} blocks")
        if n == 0:
            return 0
        if DSA_LAYERS * n > self.slots:
            raise ValueError(f"{n} pages x {DSA_LAYERS} layers exceed the staging ring ({self.slots} pages)")
        if owners is not None and len(owners) != n:
            raise ValueError(f"{len(owners)} owners for {n} pages")
        order = list(range(DSA_LAYERS)) if layer_order is None else [int(v) for v in layer_order]
        if sorted(order) != list(range(DSA_LAYERS)):
            raise ValueError(f"layer_order {order} is not a permutation of {DSA_LAYERS} layers")
        if self.pending:
            self.wait(timeout_s)  # the ring is reused per submit: the previous one must have landed
        t0 = time.perf_counter()
        base = 0 if self.head + DSA_LAYERS * n > self.slots else self.head
        P = LATENT_PAGE_BYTES // 2  # int16 elements per ring page
        ring = torch.from_numpy(self.pool.array.view(np.int16))
        k = 0
        for rows, m in objects:
            m = int(m)
            if m == 0:
                continue
            if rows.dtype != torch.uint8 or rows.dim() != 3 or rows.shape[0] < DSA_LAYERS:
                raise ValueError(f"object rows {tuple(rows.shape)} {rows.dtype}: want [layers, tokens, bytes] uint8")
            if rows.shape[1] < m * TOKENS_PER_PAGE or rows.shape[2] < LATENT_DIM * 2 or rows.stride(2) != 1:
                raise ValueError(f"object rows {tuple(rows.shape)} stride {rows.stride()} hold no {m} full pages")
            r16 = rows[:, : m * TOKENS_PER_PAGE, : LATENT_DIM * 2].view(torch.int16)  # [L, 64 m, 512], strided rows
            Ls, St = r16.stride(0), r16.stride(1)
            # [layer, page, tr, tc, fr, fc, r, c] (tilize_pages' order): token tr 32 + fr 16 + r, column tc 32 + fc 16 + c
            shape = (2, 16, 2, 2, 16, 16)
            src_st = (32 * St, 32, 16 * St, 16, St, 1)
            dst_st = (16 * 1024, 1024, 512, 256, 16, 1)
            if order == list(range(DSA_LAYERS)):
                dst = ring.as_strided((DSA_LAYERS, m, *shape), (n * P, P, *dst_st), (base + k) * P)
                src = r16.as_strided((DSA_LAYERS, m, *shape), (Ls, TOKENS_PER_PAGE * St, *src_st), r16.storage_offset())
                dst.copy_(src)
            else:
                for j, i in enumerate(order):
                    dst = ring.as_strided((m, *shape), (P, *dst_st), (base + j * n + k) * P)
                    src = r16.as_strided((m, *shape), (TOKENS_PER_PAGE * St, *src_st), r16.storage_offset() + i * Ls)
                    dst.copy_(src)
            k += m
        rows_d = self._desc_by_owner(LOAD, base, block_ids, owners, None)
        t1 = time.perf_counter()
        total = self._push_groups(LOAD, rows_d, timeout_s, range(self.chips))
        self.head = base + DSA_LAYERS * n
        s = self.stats
        s["submits"] += 1
        s["pages"] += n
        s["descriptors"] += total
        s["tilize_s"] += t1 - t0
        s["submit_s"] += time.perf_counter() - t1
        s["object_submits"] = s.get("object_submits", 0) + 1
        return total

    def lmcache_tensor(self):
        """the pool's LMCache region (lmcache_gib > 0) as a torch uint8 view: LMCache's CPU-tier allocator buffer
        (the fork's external-buffer hook), so its objects are device-visible; None without the region"""
        if not self.lmcache_bytes:
            return None
        return self.pool.tensor()[self.lmcache_off : self.lmcache_off + self.lmcache_bytes]

    def holds(self, t) -> bool:
        """t's bytes (any strided view) lie inside the pool's LMCache region"""
        if not self.lmcache_bytes or t.numel() == 0:
            return False
        lo = int(t.data_ptr()) - self.pool.base
        hi = lo + sum((int(n) - 1) * int(st) for n, st in zip(t.shape, t.stride())) * t.element_size() + t.element_size()
        return self.lmcache_off <= lo and hi <= self.lmcache_off + self.lmcache_bytes

    def _resident_groups(self, op: int, objects, block_ids, owners, layer_order) -> tuple:
        """the descriptors moving tile-order objects' whole pages (op LOAD: object -> cache; SAVE: cache -> object) at
        the objects' own pool bytes -> ({owner chip or None: descriptors}, n pages); the checks of submit_resident"""
        n = len(block_ids)
        if sum(int(m) for _r, m in objects) != n:
            raise ValueError(f"{sum(int(m) for _r, m in objects)} object pages for {n} blocks")
        if n == 0:
            return {}, 0
        if owners is not None and len(owners) != n:
            raise ValueError(f"{len(owners)} owners for {n} pages")
        order = list(range(DSA_LAYERS)) if layer_order is None else [int(v) for v in layer_order]
        if sorted(order) != list(range(DSA_LAYERS)):
            raise ValueError(f"layer_order {order} is not a permutation of {DSA_LAYERS} layers")
        blk = np.asarray(block_ids, dtype=np.int64)
        row_bytes = PAGE_LAYER_BYTES // TOKENS_PER_PAGE
        base, lstride = [], []  # per page: its layer-0 pool offset and the object's layer stride
        for rows, m in objects:
            m = int(m)
            if m == 0:
                continue
            if (rows.dim() != 3 or rows.shape[0] < DSA_LAYERS or rows.shape[1] < m * TOKENS_PER_PAGE
                    or rows.shape[2] != row_bytes or rows.stride(1) != row_bytes or rows.stride(2) != 1):  # fmt: skip
                raise ValueError(f"resident object {tuple(rows.shape)} stride {rows.stride()}: want [layers, 64 m, 1,088]")
            if not self.holds(rows[:DSA_LAYERS, : m * TOKENS_PER_PAGE]):
                raise ValueError("resident object outside the pool's LMCache region")
            off0 = int(rows.data_ptr()) - self.pool.base
            if off0 % TILE_BYTES or (int(rows.stride(0)) * rows.element_size()) % TILE_BYTES:
                raise ValueError(f"resident object at pool offset {off0}: not tile-aligned")
            base.append(off0 + np.arange(m, dtype=np.int64) * PAGE_LAYER_BYTES)
            lstride.append(np.full(m, int(rows.stride(0)) * rows.element_size(), dtype=np.int64))
        base, lstride = np.concatenate(base), np.concatenate(lstride)
        tags = np.arange(n, dtype=np.int64)
        parts = [page_desc_offsets(op, self.pool, base + i * lstride, self.dma.ids[j], blk, tags) for j, i in enumerate(order)]
        allp = np.concatenate(parts)
        if owners is None:
            return {None: allp}, n
        own = np.asarray(owners, dtype=np.int64)[allp["tag"]]
        return {int(o): allp[own == o] for o in np.unique(own)}, n

    def submit_resident(self, objects, block_ids, timeout_s: float = 10.0, owners=None, layer_order=None) -> int:
        """Stage 12 lever (c) step 2 (program 2026-10-26T03:56:09Z / 03:59:21Z): ZERO-COPY page-in -- tile-order
        objects (tile_object_pages) that live in the pool's LMCache region (holds) are LOADed by descriptors at their
        own bytes: no host copy, no ring. objects = [(rows, m)]: rows a [layers, >= 64 m tokens, 1,088] uint8 object
        view, contiguous per layer, holding m full pages of block_ids in order; layer_order / owners as
        submit_objects. The DMA reads the objects after this returns: keep them alive (and unmodified) until wait().
        Returns the descriptor count."""
        import time

        if self.pending:
            self.wait(timeout_s)
        t0 = time.perf_counter()
        groups, n = self._resident_groups(LOAD, objects, block_ids, owners, layer_order)
        if n == 0:
            return 0
        t1 = time.perf_counter()
        total = self._push_groups(LOAD, groups, timeout_s, range(self.chips))
        s = self.stats
        s["submits"] += 1
        s["pages"] += n
        s["descriptors"] += total
        s["submit_s"] += time.perf_counter() - t1
        s["desc_s"] = s.get("desc_s", 0.0) + (t1 - t0)
        s["resident_submits"] = s.get("resident_submits", 0) + 1
        return total

    def save_resident(self, objects, block_ids, timeout_s: float = 10.0, owners=None, layer_order=None) -> int:
        """Stage 12 (program 2026-10-26T07:25:07Z, the store at pool speed): ZERO-COPY store -- the store direction
        of submit_resident. Pages block_ids of the 11 latent caches are SAVEd by the page-DMA straight into tile-order
        objects that live in the pool's LMCache region (holds): descriptors at the objects' own bytes, no ring, no
        copy-out, no fill. owners None: from the save chip (the caches are replicated); owners [n]: each page from its
        owner chip (block ids local). objects / layer_order as submit_resident (object layer layer_order[j] <- cache j).
        Call only after the step that wrote the pages finished; the bytes are there after wait(). The objects' 4 KB
        pooled halves are the caller's. Returns the descriptor count."""
        import time

        if self.pending:
            self.wait(timeout_s)
        t0 = time.perf_counter()
        groups, n = self._resident_groups(SAVE, objects, block_ids, owners, layer_order)
        if n == 0:
            return 0
        t1 = time.perf_counter()
        total = self._push_groups(SAVE, groups, timeout_s, [0])
        s = self.stats
        s["resident_saves"] = s.get("resident_saves", 0) + 1
        s["resident_save_pages"] = s.get("resident_save_pages", 0) + n
        s["descriptors"] += total
        s["save_submit_s"] = s.get("save_submit_s", 0.0) + time.perf_counter() - t1
        s["desc_s"] = s.get("desc_s", 0.0) + (t1 - t0)
        return total

    def _desc_by_owner(self, op: int, base: int, block_ids, owners, default) -> dict:
        """op's descriptors of n pages per layer at ring pages base + l n + i (tag i), built vectorized (page_desc_array;
        Stage 12: the per-page loop was 34 % of a 131K store read), grouped by owner chip (owners None: one group keyed
        `default`), layer-major within a group (the per-page loop's order)"""
        n = len(block_ids)
        blk = np.asarray(block_ids, dtype=np.int64)
        allp = np.concatenate([page_desc_array(op, self.pool, (base + l * n) * LATENT_PAGE_BYTES, self.dma.ids[l], blk)
                               for l in range(DSA_LAYERS)])  # fmt: skip
        if owners is None:
            return {default: allp}
        own = np.asarray(owners, dtype=np.int64)[allp["tag"].astype(np.int64)]
        return {int(o): allp[own == o] for o in dict.fromkeys(own.tolist())}

    def _wait_targets(self, targets, timeout_s: float) -> None:
        """until every (chip, core) completed the descriptors pushed when `targets` (a copy of self.tails) was taken"""
        import time

        t0 = time.perf_counter()
        while True:
            got = self.done_counts()
            if all(g >= t for gr, tr in zip(got, targets) for g, t in zip(gr, tr)):
                return
            if time.perf_counter() - t0 > timeout_s:
                raise TimeoutError(f"page-DMA batch not complete after {timeout_s} s: {got} < {targets}")
            time.sleep(0.0002)

    def load_many(self, pages_by_layer, block_ids, timeout_s: float = 30.0, owners=None) -> int:
        """submit() for a hit of ANY size (a 1M-token user = 16,384 pages x 11 layers = 11.5 GB per chip, the staging
        ring ~0.8 GB): batches of half the ring, double-buffered -- batch k+1 is tilized into one half while the DMA of
        batch k drains the other (the host completion slots tell when a half is free). A hit that fits the ring is one
        submit(), unchanged. Returns the descriptor count; wait() before the step that reads the pages."""
        import time

        import torch

        n = len(block_ids)
        if DSA_LAYERS * n <= self.slots:
            return self.submit(pages_by_layer, block_ids, timeout_s=timeout_s, owners=owners)
        if len(pages_by_layer) != DSA_LAYERS:
            raise ValueError(f"{len(pages_by_layer)} layers, want {DSA_LAYERS}")
        if owners is not None and len(owners) != n:
            raise ValueError(f"{len(owners)} owners for {n} pages")
        if self.pending:
            self.wait(timeout_s)
        half = self.slots // 2
        bp = max(1, half // DSA_LAYERS)
        srcs = [np.asarray(p).view(np.int16).reshape(n, TOKENS_PER_PAGE, LATENT_DIM) for p in pages_by_layer]
        busy = [None, None]  # per ring half: the tails after its last batch
        total, t_tilize, t_submit, t_wait = 0, 0.0, 0.0, 0.0
        for i, s0 in enumerate(range(0, n, bp)):
            h, m = i % 2, min(bp, n - s0)
            if busy[h] is not None:
                t = time.perf_counter()
                self._wait_targets(busy[h], timeout_s)
                t_wait += time.perf_counter() - t
            t = time.perf_counter()
            parts = []
            blk = np.asarray(block_ids[s0 : s0 + m], dtype=np.int64)
            for l in range(DSA_LAYERS):
                off = (h * half + l * m) * LATENT_PAGE_BYTES
                dst = torch.from_numpy(self.pool.array[off : off + m * LATENT_PAGE_BYTES].view(np.int16)).view(
                    m, 2, 16, 2, 2, 16, 16)  # [m, tr, tc, fr, fc, r, c] = tilize_pages' order
                dst.copy_(torch.from_numpy(srcs[l][s0 : s0 + m]).view(m, 2, 2, 16, 16, 2, 16).permute(0, 1, 4, 2, 5, 3, 6))
                parts.append(page_desc_array(LOAD, self.pool, off, self.dma.ids[l], blk))
            t1 = time.perf_counter()
            t_tilize += t1 - t
            allp = np.concatenate(parts)
            if owners is None:
                rows = {None: allp}
            else:  # descriptor i of a batch moves page tag i (page_desc_array: tag = the page's index in the batch)
                own = np.asarray(owners[s0 : s0 + m], dtype=np.int64)[allp["tag"].astype(np.int64)]
                rows = {int(o): allp[own == o] for o in np.unique(own)}
            for o, r in rows.items():
                d = r
                if o is None:
                    self.dma.impl.load(d.tobytes(), timeout_s)
                    self._account(len(d), range(self.chips))
                else:
                    self.dma.impl.load(d.tobytes(), timeout_s, chips=[o])
                    self._account(len(d), [o])
                total += len(d)
            busy[h] = [list(tr) for tr in self.tails]
            t_submit += time.perf_counter() - t1
        self.head, self.pending = 0, total
        s = self.stats
        s["submits"] += 1
        s["pages"] += n
        s["descriptors"] += total
        s["tilize_s"] += t_tilize
        s["submit_s"] += t_submit
        s["large_loads"] = s.get("large_loads", 0) + 1
        s["large_batch_wait_s"] = s.get("large_batch_wait_s", 0.0) + t_wait
        return total

    def read_pages(self, block_ids, timeout_s: float = 10.0, owners=None, untilize: bool = True) -> list:
        """the store direction (program 2026-10-23T03:54Z item 2: LMCache's page reads were 0.06 GB/s of SlotOps
        copies + to_torch): pages block_ids [n] of the 11 latent caches SAVEd by the page-DMA into the staging ring
        (owners None: the save chip -- the caches are replicated; owners [n]: each page from its owner chip, block ids
        local), waited for, untilized -> 11 arrays [n, 64, 512] uint16 (row-major; layer order = the latent caches').
        untilize=False: 11 TiledPages instead (the ring's tile-order bytes copied out; result() untilizes on the
        calling thread -- the store worker; Stage 12 W4 F3a: the untilize was 56 % of the engine thread's store read).
        Call only after the step that wrote the pages finished (the DMA reads DRAM, not the command queue)."""
        import time

        n = len(block_ids)
        if n == 0:
            return [np.zeros((0, TOKENS_PER_PAGE, LATENT_DIM), np.uint16) for _ in range(DSA_LAYERS)]
        if DSA_LAYERS * n > self.slots:
            raise ValueError(f"{n} pages x {DSA_LAYERS} layers exceed the staging ring ({self.slots} pages)")
        if owners is not None and len(owners) != n:
            raise ValueError(f"{len(owners)} owners for {n} pages")
        if self.pending:
            self.wait(timeout_s)  # a hit's LOAD still reads the ring
        t0 = time.perf_counter()
        base = 0 if self.head + DSA_LAYERS * n > self.slots else self.head
        rows = self._desc_by_owner(SAVE, base, block_ids, owners, 0)
        for o, d in rows.items():
            if owners is None:
                self.dma.impl.save(d.tobytes(), timeout_s)
            else:
                self.dma.impl.save(d.tobytes(), timeout_s, chips=[o])
            self._account(len(d), [o])
        t1 = time.perf_counter()
        self.dma.wait(timeout_s)
        t2 = time.perf_counter()
        out = []
        for l in range(DSA_LAYERS):
            off = (base + l * n) * LATENT_PAGE_BYTES
            tiles = self.pool.array[off : off + n * LATENT_PAGE_BYTES].view(np.uint16).reshape(n, -1)
            out.append(untilize_pages(tiles) if untilize else TiledPages(tiles))  # a copy: the ring is reused next
        self.head = base + DSA_LAYERS * n
        s = self.stats
        s["reads"] = s.get("reads", 0) + 1
        s["reads_tiled"] = s.get("reads_tiled", 0) + (0 if untilize else 1)
        s["read_pages"] = s.get("read_pages", 0) + n
        s["read_submit_s"] = s.get("read_submit_s", 0.0) + t1 - t0
        s["read_wait_s"] = s.get("read_wait_s", 0.0) + t2 - t1
        s["read_untilize_s"] = s.get("read_untilize_s", 0.0) + time.perf_counter() - t2
        return out

    def _pooled_desc(self, op: int, targets, base: int) -> dict:
        """descriptors moving unit pages targets [(chip or None, page)] of the 11 pooled-key stores to / from ring bytes
        base + (l m + k) x 8,192 (layer l, target k; tag k) -> {chip (None: every chip / the save chip): DESC array},
        layer-major then target order within a chip. Vectorized (Stage 12 zero-copy store: the per-unit loop was 42 ms
        of a 32K store's 105); an 8 KB unit at an 8 KB-aligned ring offset never crosses a GiB boundary (else the split
        reference _pooled_desc_split)."""
        m = len(targets)
        if m == 0:
            return {}
        if base % POOLED_UNIT_BYTES:
            return self._pooled_desc_split(op, targets, base)
        pages = np.fromiter((int(p) for _c, p in targets), dtype=np.int64, count=m)
        chips = [None if c is None else int(c) for c, _p in targets]
        tile0 = POOLED_UNIT_TILES * pages
        caps = np.asarray([int(self.pools[l][1]) for l in range(DSA_LAYERS)], dtype=np.int64)
        bad = (tile0 < 0)[None, :] | (tile0[None, :] > (caps - POOLED_UNIT_TILES)[:, None])
        if bad.any():
            l, k = (int(v[0]) for v in np.nonzero(bad))
            raise ValueError(f"pooled-key page {int(pages[k])} outside the layer-{l} store ({int(caps[l])} tiles)")
        d = np.zeros(DSA_LAYERS * m, dtype=DESC)
        offs = base + np.arange(DSA_LAYERS * m, dtype=np.int64) * POOLED_UNIT_BYTES
        d["op"], d["n_tiles"] = op, POOLED_UNIT_TILES
        d["tensor"] = np.repeat(np.asarray(self.pool_ids, dtype=np.int64), m)
        d["tile0"] = np.tile(tile0, DSA_LAYERS)
        d["slab"], d["slab_off"] = offs // GIB, offs % GIB
        d["tag"] = np.tile(np.arange(m, dtype=np.int64), DSA_LAYERS)
        groups = {}
        for c in dict.fromkeys(chips):
            sel = np.tile(np.asarray([x == c for x in chips], dtype=bool), DSA_LAYERS)
            groups[c] = d[sel]
        return groups

    def _pooled_desc_split(self, op: int, targets, base: int) -> dict:
        """the per-unit reference of _pooled_desc (extents split at GiB boundaries) -> {chip: DESC array}"""
        m, rows = len(targets), {}
        for l in range(DSA_LAYERS):
            for k, (ch, page) in enumerate(targets):
                tile0 = POOLED_UNIT_TILES * int(page)
                if not 0 <= tile0 <= self.pools[l][1] - POOLED_UNIT_TILES:
                    raise ValueError(f"pooled-key page {page} outside the layer-{l} store ({self.pools[l][1]} tiles)")
                for slab, so, nb in self.pool.extents(base + (l * m + k) * POOLED_UNIT_BYTES, POOLED_UNIT_BYTES):
                    n = nb // TILE_BYTES
                    rows.setdefault(None if ch is None else int(ch), []).append(
                        (op, self.pool_ids[l], tile0, n, slab, so, k, (0, 0)))
                    tile0 += n
        return {c: np.array(r, dtype=DESC) for c, r in rows.items()}

    def _pooled_cap(self) -> int:
        """unit pages (x 11 layers) per ring-sized batch"""
        return max(1, self.slots * LATENT_PAGE_BYTES // (DSA_LAYERS * POOLED_UNIT_BYTES))

    def read_pooled(self, targets, timeout_s: float = 10.0) -> list:
        """Stage 12 (GLM53_KV_POOLED_PAGES: the pooled keys ride in the LMCache page): the pooled-key stores' unit pages
        targets [(chip, page)] of the 11 layers (chip = the unit's owner on the sharded store; None = the save chip of a
        replicated store) SAVEd into the ring in ring-sized batches, waited for -> 11 arrays [m, 8,192] uint8,
        device-native (pooled_halves splits the 64-token halves). Call after the steps that wrote them finished."""
        import time

        if not self.pool_ids:
            raise RuntimeError("read_pooled: no pooled-key stores registered (LatentHitDma(pool_tensors=))")
        m = len(targets)
        out = [np.zeros((m, POOLED_UNIT_BYTES), np.uint8) for _ in range(DSA_LAYERS)]
        if m == 0:
            return out
        if self.pending:
            self.wait(timeout_s)  # a LOAD still reads the ring
        t0 = time.perf_counter()
        cap = self._pooled_cap()
        for s0 in range(0, m, cap):
            part = targets[s0 : s0 + cap]
            k = len(part)
            # per owner group, counted in pending (the S1 class): a throw part-way leaves the published SAVEs for the
            # next wait() to drain before anything rewrites the ring
            self._push_groups(SAVE, self._pooled_desc(SAVE, part, 0), timeout_s, [0])
            self.wait(timeout_s)
            for l in range(DSA_LAYERS):
                o = l * k * POOLED_UNIT_BYTES
                out[l][s0 : s0 + k] = self.pool.array[o : o + k * POOLED_UNIT_BYTES].reshape(k, POOLED_UNIT_BYTES)
        self.head = 0
        s = self.stats
        s["pooled_reads"] = s.get("pooled_reads", 0) + 1
        s["pooled_read_units"] = s.get("pooled_read_units", 0) + m
        s["pooled_read_s"] = s.get("pooled_read_s", 0.0) + time.perf_counter() - t0
        return out

    def save_resident_pooled(self, objects, block_ids, pooled_targets, timeout_s: float = 10.0, owners=None,
                             layer_order=None) -> list:
        """Stage 12 store path P2 (program 2026-10-26T23:48:49Z; doc/ttnn/12-fast-start/store_path/README.md): ONE
        device round trip per zero-copy store. save_resident's SAVEs of the latent pages into the objects and
        read_pooled's SAVEs of the pooled-key unit pages pooled_targets [(chip, page)] into the ring are pushed together
        and waited for once (was: save_resident + wait, then read_pooled + its own wait). Returns read_pooled's 11
        arrays [m, 8,192] uint8, device-native. When it returns the objects hold their latent pages (their pooled halves
        stay the caller's) and the arrays hold the pooled pages. Pooled targets past one ring batch follow in
        read_pooled's batches. objects / block_ids / owners / layer_order as save_resident. Call only after the steps
        that wrote the pages finished."""
        import time

        if not self.pool_ids:
            raise RuntimeError("save_resident_pooled: no pooled-key stores registered (LatentHitDma(pool_tensors=))")
        m = len(pooled_targets)
        out = [np.zeros((m, POOLED_UNIT_BYTES), np.uint8) for _ in range(DSA_LAYERS)]
        if self.pending:
            self.wait(timeout_s)  # a LOAD still reads the ring
        t0 = time.perf_counter()
        groups, n = self._resident_groups(SAVE, objects, block_ids, owners, layer_order)
        cap = self._pooled_cap()
        k = min(cap, m)
        t1 = time.perf_counter()
        total = self._push_groups(SAVE, groups, timeout_s, [0]) if n else 0
        if k:
            total += self._push_groups(SAVE, self._pooled_desc(SAVE, list(pooled_targets[:k]), 0), timeout_s, [0])
        t2 = time.perf_counter()
        self.wait(timeout_s)  # the one wait: the objects' latent pages and the first pooled batch
        for l in range(DSA_LAYERS):
            o = l * k * POOLED_UNIT_BYTES
            out[l][:k] = self.pool.array[o : o + k * POOLED_UNIT_BYTES].reshape(k, POOLED_UNIT_BYTES)
        for s0 in range(k, m, cap):  # (more pooled units than one ring batch: read_pooled's loop)
            part = list(pooled_targets[s0 : s0 + cap])
            kk = len(part)
            self._push_groups(SAVE, self._pooled_desc(SAVE, part, 0), timeout_s, [0])
            self.wait(timeout_s)
            for l in range(DSA_LAYERS):
                o = l * kk * POOLED_UNIT_BYTES
                out[l][s0 : s0 + kk] = self.pool.array[o : o + kk * POOLED_UNIT_BYTES].reshape(kk, POOLED_UNIT_BYTES)
        if m:
            self.head = 0
        s = self.stats
        s["resident_saves"] = s.get("resident_saves", 0) + (1 if n else 0)
        s["resident_save_pages"] = s.get("resident_save_pages", 0) + n
        s["resident_pooled_saves"] = s.get("resident_pooled_saves", 0) + 1
        s["pooled_reads"] = s.get("pooled_reads", 0) + (1 if m else 0)
        s["pooled_read_units"] = s.get("pooled_read_units", 0) + m
        s["descriptors"] += total
        s["save_submit_s"] = s.get("save_submit_s", 0.0) + t2 - t1
        s["desc_s"] = s.get("desc_s", 0.0) + (t1 - t0)
        s["resident_pooled_s"] = s.get("resident_pooled_s", 0.0) + time.perf_counter() - t0
        return out

    def load_pooled(self, units_by_layer, targets, timeout_s: float = 10.0) -> int:
        """inverse of read_pooled: 11 arrays [m, 8,192] uint8 device-native unit pages -> targets [(chip, page)] (chip
        None: every chip, a replicated store). Queued like submit(): wait() (or the next submit / load_many) before a
        step reads them; batches past the ring wait for the previous batch. Returns the descriptor count."""
        if not self.pool_ids:
            raise RuntimeError("load_pooled: no pooled-key stores registered (LatentHitDma(pool_tensors=))")
        if len(units_by_layer) != DSA_LAYERS:
            raise ValueError(f"{len(units_by_layer)} layers, want {DSA_LAYERS}")
        m = len(targets)
        if m == 0:
            return 0
        src = [np.ascontiguousarray(a).view(np.uint8).reshape(m, POOLED_UNIT_BYTES) for a in units_by_layer]

        def stage(l, s0, k, dest):
            dest[:] = src[l][s0 : s0 + k].reshape(-1)

        return self._load_pooled(targets, stage, timeout_s)

    def load_pooled_blocks(self, rows, targets, timeout_s: float = 10.0) -> int:
        """GLM53_DSA_PAGED_POOLS: load_pooled of hit rows' PADDED block pages, built in the ring (Stage 12, program
        2026-10-26T17:50:09Z WARM: pooled_halves -> block_page_of_half -> the per-row concat -> the ring copy were four
        passes over a 32K hit's 43 MiB; this is one strided copy + the zero rows). rows [(11 arrays [u, 8,192] uint8
        device-native unit pages, n)]: the row's blocks 0..n-1, block j = half j % 2 of unit j // 2 in rows 0-15 of its
        page, rows 16-31 zero (block_page_of_half(pooled_halves(units).reshape(2 u, -1)[:n]), bitwise) -> targets, one
        per block, rows in order. Returns the descriptor count."""
        if not self.pool_ids:
            raise RuntimeError("load_pooled: no pooled-key stores registered (LatentHitDma(pool_tensors=))")
        src, ends = [], []
        for units, n in rows:
            if len(units) != DSA_LAYERS:
                raise ValueError(f"{len(units)} layers, want {DSA_LAYERS}")
            u = [np.ascontiguousarray(a).view(np.int16).reshape(-1, POOLED_UNIT_TILES, 2, TILE_BYTES // 4)
                 for a in units]  # fmt: skip
            if not 0 <= int(n) <= 2 * len(u[0]):
                raise ValueError(f"{n} blocks from {len(u[0])} unit pages")
            src.append(u)
            ends.append((ends[-1] if ends else 0) + int(n))
        if (ends[-1] if ends else 0) != len(targets):
            raise ValueError(f"{ends[-1] if ends else 0} blocks, {len(targets)} targets")
        begins = [0] + ends[:-1]

        def stage(l, s0, k, dest):
            d = dest.view(np.int16).reshape(k, POOLED_UNIT_TILES, 2, TILE_BYTES // 4)
            _put(d[:, :, 1], 0)  # rows 16-31
            for r, (b0, b1) in enumerate(zip(begins, ends)):
                j0, j1 = max(s0, b0) - b0, min(s0 + k, b1) - b0
                if j0 < j1:
                    _stage_halves(d[b0 + j0 - s0 : b0 + j1 - s0], src[r][l], j0, j1)

        return self._load_pooled(targets, stage, timeout_s)

    def _load_pooled(self, targets, stage, timeout_s: float) -> int:
        """load_pooled's batches: stage(layer, s0, k, ring bytes [k x 8,192]) fills targets s0..s0+k-1 of a layer, then
        the batch's descriptors are pushed. stats pooled_load_{prewait,stage,push}_s split the host time."""
        import time

        m = len(targets)
        if m == 0:
            return 0
        s = self.stats
        t0 = time.perf_counter()
        if self.pending:
            self.wait(timeout_s)
        t_stage = t_push = 0.0
        t1 = time.perf_counter()
        cap, total = self._pooled_cap(), 0
        for s0 in range(0, m, cap):
            if total:
                self.wait(timeout_s)  # the ring is rewritten by the next batch
            part = targets[s0 : s0 + cap]
            k = len(part)
            ta = time.perf_counter()
            for l in range(DSA_LAYERS):
                o = l * k * POOLED_UNIT_BYTES
                stage(l, s0, k, self.pool.array[o : o + k * POOLED_UNIT_BYTES])
            tb = time.perf_counter()
            # per owner group (S1 class, 14's 18:06:16Z SHOULD): a ring-space throw part-way leaves the queued groups,
            # and the throwing one, counted in pending, so the next wait() drains them before the ring is rewritten
            total += self._push_groups(LOAD, self._pooled_desc(LOAD, part, 0), timeout_s, range(self.chips))
            t_stage, t_push = t_stage + tb - ta, t_push + time.perf_counter() - tb
        self.head = self.slots  # the next submit starts at the ring's base (after waiting for these)
        s["pooled_loads"] = s.get("pooled_loads", 0) + 1
        s["pooled_load_units"] = s.get("pooled_load_units", 0) + m
        s["pooled_load_prewait_s"] = s.get("pooled_load_prewait_s", 0.0) + t1 - t0
        s["pooled_load_stage_s"] = s.get("pooled_load_stage_s", 0.0) + t_stage
        s["pooled_load_push_s"] = s.get("pooled_load_push_s", 0.0) + t_push  # descriptors + push (ring-batch waits: neither)
        return total

    def wait(self, timeout_s: float = 10.0):
        import time

        if self.pending:
            t0 = time.perf_counter()
            self.dma.wait(timeout_s)
            self.stats["wait_s"] += time.perf_counter() - t0
            self.pending = 0
            if getattr(self, "_tails_stale", False):  # a push threw part-way: everything published has now landed
                self.tails = [list(r) for r in self.done_counts()]
                self._tails_stale = False
        # everything queued so far has landed: what was held for it is released -- this thread's callbacks only
        # (another thread's run at its own next wait: a store worker's wait never runs the engine thread's LMCache
        # releases), and only once this thread has left the DMA lock (_dma_locked; the lock order below)
        me = threading.get_ident()
        mine = [fn for t, fn in self._after if t == me]
        if mine:
            self._after = [(t, fn) for t, fn in self._after if t != me]
            _ready_of(self).extend(mine)

    def after_wait(self, fn) -> None:
        """fn() runs once the queued descriptors have landed (the next wait() on this thread, also the one a later
        submit does first): submit_resident's caller holds the objects the DMA reads until then, so the page-in runs
        behind other work"""
        self._after.append((threading.get_ident(), fn))

    def close(self, timeout_s: float = 10.0):
        """stop the kernels (mesh still open), then unmap the pool"""
        stopped = True
        try:
            self.wait(timeout_s)
            _ready_of(self).extend(fn for _t, fn in self._after)  # (any thread's: nothing runs after the close)
            self._after = []
            self.dma.stop(timeout_s)
        except Exception:
            stopped = False
            raise
        finally:
            self.pool.close(devices_stopped=stopped)


def _ready_of(dma) -> list:
    """the calling thread's after_wait callbacks whose descriptors have landed (run when it leaves the DMA lock)"""
    tl = dma._tls
    if not hasattr(tl, "ready"):
        tl.ready, tl.depth = [], 0
    return tl.ready


def _dma_locked(fn):
    """LatentHitDma's public methods under its transaction lock (a stand-in built without __init__: the class lock).
    LOCK ORDER (store path P3, 14's 02:13:21Z SHOULD): the DMA lock is the INNERMOST lock. Callers may hold their own
    locks when they call in (the host's _lock, LMCache's, the inner connector's), and nothing runs under the DMA lock
    that takes another lock: the after_wait callbacks (e.g. LMCache's ref_count_down, which takes its allocator lock)
    run here, after the calling thread has left the lock entirely (depth 0, a nested call does not run them). So an
    engine thread waiting with a callback and a store worker inside LMCache's store cannot deadlock."""

    @functools.wraps(fn)
    def run(self, *args, **kwargs):
        ready = _ready_of(self)
        tl = self._tls
        tl.depth += 1
        try:
            with self._lock:
                return fn(self, *args, **kwargs)
        finally:
            tl.depth -= 1
            if tl.depth == 0 and ready:
                todo, ready[:] = list(ready), []
                err = None
                for cb in todo:
                    try:
                        cb()
                    except BaseException as e:  # noqa: BLE001 - run every release, then raise the first failure
                        err = err or e
                if err is not None:
                    raise err

    return run


LatentHitDma._lock = threading.RLock()  # (instances built by __init__ have their own)
LatentHitDma._tls = threading.local()
LatentHitDma.thread_safe = True  # the plugin's store worker may call it (GLM53_KV_STORE_ZC_WORKER)
for _name in ("save_state", "start", "submit", "submit_objects", "submit_resident", "save_resident", "load_many",
              "read_pages", "read_pooled", "save_resident_pooled", "load_pooled", "load_pooled_blocks", "wait",
              "after_wait", "close"):  # fmt: skip
    setattr(LatentHitDma, _name, _dma_locked(getattr(LatentHitDma, _name)))
