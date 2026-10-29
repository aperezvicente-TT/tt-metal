# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LMCache ladder R1 instrumentation of the KV-connector data path, in the EngineCore process (installed by the R1 /
rehearsal backend modules when GLM53_R1_PROBE=1, rows to $GLM53_R1_PROBE_JSONL; a probe failure never stops the
server). Per call: wall seconds, calling-thread CPU seconds and process CPU seconds (the mesh write path also runs on
tt-metal's per-device thread pool), and bytes of

* the plugin's ttnn latent-page path: `_ttnn_read_pages` (d2h: one chip of the replicated latent) and
  `_ttnn_write_pages` (h2d: every chip), the latter split into `stage` (`ttnn.from_torch` with ReplicateTensorToMesh:
  host tile conversion + enqueue of the replica writes), `fill` (`paged_fill_cache` enqueue) and `sync`
  (`synchronize_device`: where the writes and the fill complete); and `tt_copy_kv_blocks` around them (by direction);
* the LMCache TT connector's `to_gpu` / `from_gpu` / `batched_to_gpu` / `batched_from_gpu`;
* the KDA checkpoint (page_contract.md section 5): StackBackend `export_kda_state` / `import_kda_state` (the host
  gather of the slot's 4-row chunk for every KDA layer, and for an import the re-shard write) and the plugin's
  `KDACheckpointStore` `save` / `load` (file bytes).

Rates are end-to-end per call (bytes / wall). Whether the 8 replica writes of a load overlap is inferred from them,
not observed: a host-side probe cannot see per-link PCIe traffic. Names the installed code lacks are skipped (the
`installed` row lists what was wrapped)."""

from __future__ import annotations

import functools
import json
import os
import threading
import time

PROBE_ENV = "GLM53_R1_PROBE"
JSONL_ENV = "GLM53_R1_PROBE_JSONL"
_local = threading.local()
_lock = threading.Lock()
_installed = False


def _emit(row: dict) -> None:
    path = os.environ.get(JSONL_ENV)
    if path:
        with _lock, open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")


def _elem_bytes(tensor) -> int:
    name = str(getattr(tensor, "dtype", "")).upper()
    return 4 if ("FLOAT32" in name or "INT32" in name or "UINT32" in name) else 2  # the latent is bf16


def _nbytes_per_page(tensor) -> int:
    n = _elem_bytes(tensor)
    for v in list(tensor.shape)[1:]:
        n *= int(v)
    return n


def _devices(tensor) -> int:
    try:
        return int(tensor.device().get_num_devices())
    except Exception:
        return 1


def _tensor_bytes(obj) -> int:
    """bytes of the torch tensors in a (nested) checkpoint object"""
    import torch

    if isinstance(obj, torch.Tensor):
        return int(obj.numel()) * int(obj.element_size())
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def _timed(kind: str, fn, describe, sub: bool = False, result=None):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        outer = getattr(_local, "sub", None)
        if sub:
            _local.sub = {}
        t0, c0, p0 = time.perf_counter(), time.thread_time(), time.process_time()
        err, out = None, None
        try:
            out = fn(*a, **k)
            return out
        except BaseException as e:
            err = repr(e)[:200]
            raise
        finally:
            row = {"t": time.time(), "kind": kind, "wall_s": time.perf_counter() - t0, "cpu_s": time.thread_time() - c0,
                   "pcpu_s": time.process_time() - p0, "thread": threading.current_thread().name}  # fmt: skip
            try:
                row.update(describe(a, k))
                if result is not None and err is None:
                    row.update(result(out))
            except Exception as e:  # never let the probe break the data path
                row["describe_error"] = repr(e)[:200]
            if sub:
                row.update({f"{name}_s": v for name, v in _local.sub.items()})
                _local.sub = outer
            if err:
                row["error"] = err
            _emit(row)

    return wrapper


def _sub_timed(name: str, fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        acc = getattr(_local, "sub", None)
        if acc is None:
            return fn(*a, **k)
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            acc[name] = acc.get(name, 0.0) + time.perf_counter() - t0

    return wrapper


def _install_page_path(done: list) -> None:
    from vllm_tt_plugin import kv_connector as kc

    if hasattr(kc, "_ttnn_read_pages"):
        kc._ttnn_read_pages = _timed("d2h_pages", kc._ttnn_read_pages, lambda a, k: {
            "pages": len(a[1]), "bytes": len(a[1]) * _nbytes_per_page(a[0]), "chips_read": 1})  # fmt: skip
        done.append("_ttnn_read_pages")
    if hasattr(kc, "_ttnn_write_pages"):

        def write_desc(a, k):
            n = int(a[2].numel()) * int(a[2].element_size())
            return {"pages": len(a[1]), "bytes": n, "chips": _devices(a[0]), "device_bytes": n * _devices(a[0])}

        kc._ttnn_write_pages = _timed("h2d_pages", kc._ttnn_write_pages, write_desc, sub=True)
        done.append("_ttnn_write_pages")
        import ttnn

        ttnn.from_torch = _sub_timed("stage", ttnn.from_torch)
        ttnn.experimental.paged_fill_cache = _sub_timed("fill", ttnn.experimental.paged_fill_cache)
        ttnn.synchronize_device = _sub_timed("sync", ttnn.synchronize_device)
    if hasattr(kc, "tt_copy_kv_blocks"):

        def copy_desc(a, k):
            d = a[4] if len(a) > 4 else k.get("direction")
            return {"direction": d, "layers": len(a[0]) if a else None, "blocks": len(a[2]) if len(a) > 2 else None}

        kc.tt_copy_kv_blocks = _timed("copy_kv_blocks", kc.tt_copy_kv_blocks, copy_desc)
        done.append("tt_copy_kv_blocks")
    store = getattr(kc, "KDACheckpointStore", None)
    if store is not None:
        def size_of(a, k):
            try:
                p = a[0].path(a[1], a[2])
                return {"tokens": int(a[2]), "file_bytes": os.path.getsize(p) if os.path.exists(p) else None}
            except Exception as e:
                return {"size_error": repr(e)[:120]}

        store.save = _timed("kda_store_save", store.save, lambda a, k: {"tokens": int(a[2])},
                            result=lambda out: {"file_bytes": os.path.getsize(out) if isinstance(out, str) and os.path.exists(out) else None})  # fmt: skip
        store.load = _timed("kda_store_load", store.load, size_of)
        done += ["KDACheckpointStore.save", "KDACheckpointStore.load"]


def _install_lmcache_connector(done: list) -> None:
    from vllm_tt_plugin.lmcache.connector import VLLMPagedMemTTConnector as C

    for name in ("to_gpu", "from_gpu"):
        if hasattr(C, name):
            setattr(C, name, _timed(f"lmcache_{name}", getattr(C, name), lambda a, k: {
                "tokens": int(a[3]) - int(a[2]) if len(a) > 3 else None}))  # fmt: skip
            done.append(name)
    for name in ("batched_to_gpu", "batched_from_gpu"):
        if hasattr(C, name):
            setattr(C, name, _timed(f"lmcache_{name}", getattr(C, name), lambda a, k: {
                "chunks": len(a[1]) if len(a) > 1 else None}))  # fmt: skip
            done.append(name)


def _install_kda(done: list) -> None:
    from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as sm

    B = sm.StackBackend
    if hasattr(B, "export_kda_state"):
        B.export_kda_state = _timed("kda_export", B.export_kda_state, lambda a, k: {"chunk_rows": a[0].chunk},
                                    result=lambda out: {"row_bytes": _tensor_bytes(out)})  # fmt: skip
        done.append("StackBackend.export_kda_state")
    if hasattr(B, "import_kda_state"):
        B.import_kda_state = _timed("kda_import", B.import_kda_state, lambda a, k: {
            "chunk_rows": a[0].chunk, "row_bytes": _tensor_bytes(a[3] if len(a) > 3 else k.get("states"))})  # fmt: skip
        done.append("StackBackend.import_kda_state")


def install() -> list[str]:
    """Wrap what exists; idempotent; never raises (a failure is recorded in the `installed` row)."""
    global _installed
    if _installed:
        return []
    _installed = True
    done, errors = [], {}
    for name, step in (("page_path", _install_page_path), ("lmcache", _install_lmcache_connector),
                       ("kda", _install_kda)):  # fmt: skip
        try:
            step(done)
        except Exception as e:
            errors[name] = repr(e)[:200]
    _emit({"t": time.time(), "kind": "installed", "wrapped": done, "errors": errors, "pid": os.getpid()})
    return done


def install_from_env() -> list[str]:
    return install() if os.environ.get(PROBE_ENV) == "1" else []
