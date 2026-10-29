# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import threading

import pytest
import torch

import ttnn

core = ttnn.operations.core


def _dump(path, torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    ttnn.dump_tensor(path, ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout))
    return path


def _host(tensor):
    return ttnn.to_torch(tensor)


@pytest.fixture
def tensor_file(tmp_path):
    torch.manual_seed(0)
    expected = torch.randn(64, 96).bfloat16()
    return _dump(tmp_path / "w.tensorbin", expected), expected


@pytest.fixture
def descriptor(tensor_file):
    fd = os.open(tensor_file[0], os.O_RDONLY | os.O_CLOEXEC)
    yield fd
    os.close(fd)


def _open_descriptors():
    return set(os.listdir("/proc/self/fd"))


# 1. descriptor-bound loads


def test_fd_load_matches_path_load(tensor_file, descriptor):
    path, expected = tensor_file
    by_fd = ttnn.load_tensor(f"/proc/self/fd/{descriptor}")
    by_path = ttnn.load_tensor(path)
    assert torch.equal(_host(by_fd), expected)
    assert torch.equal(_host(by_fd), _host(by_path))


def test_fd_load_accepts_a_path_object(tensor_file, descriptor):
    loaded = ttnn.load_tensor(core.pathlib.Path(f"/proc/self/fd/{descriptor}"))
    assert torch.equal(_host(loaded), tensor_file[1])


def test_fd_load_reads_the_descriptor_not_the_current_path(tmp_path, tensor_file, descriptor):
    path, expected = tensor_file
    os.rename(path, tmp_path / "moved.tensorbin")
    replacement = torch.full((64, 96), 3.0).bfloat16()
    _dump(path, replacement)
    assert torch.equal(_host(ttnn.load_tensor(path)), replacement)
    assert torch.equal(_host(ttnn.load_tensor(f"/proc/self/fd/{descriptor}")), expected)


def test_fd_load_refuses_a_file_replaced_by_rename(tensor_file, descriptor, expect_error):
    path, _ = tensor_file
    _dump(path, torch.full((64, 96), 3.0).bfloat16())  # the descriptor's file is unlinked by the rename
    with expect_error(RuntimeError, "one live absolute file"):
        ttnn.load_tensor(f"/proc/self/fd/{descriptor}")


def test_fd_load_refuses_an_unlinked_file(tensor_file, descriptor, expect_error):
    path, _ = tensor_file
    os.unlink(path)
    with expect_error(RuntimeError, "one live absolute file"):
        ttnn.load_tensor(f"/proc/self/fd/{descriptor}")


def test_fd_load_keeps_the_callers_descriptor_and_closes_its_own(tensor_file, descriptor):
    before = _open_descriptors()
    ttnn.load_tensor(f"/proc/self/fd/{descriptor}")
    assert _open_descriptors() == before
    os.fstat(descriptor)


def test_fd_load_refuses_a_closed_descriptor(tensor_file, expect_error):
    fd = os.open(tensor_file[0], os.O_RDONLY)
    os.close(fd)
    with expect_error(RuntimeError, "does not name an open descriptor"):
        ttnn.load_tensor(f"/proc/self/fd/{fd}")


def test_fd_load_refuses_a_non_tensorbin_target(tmp_path, tensor_file, expect_error):
    other = tmp_path / "w.bin"
    other.write_bytes(open(tensor_file[0], "rb").read())
    fd = os.open(other, os.O_RDONLY)
    try:
        with expect_error(RuntimeError, "absolute .tensorbin file"):
            ttnn.load_tensor(f"/proc/self/fd/{fd}")
    finally:
        os.close(fd)


def test_fd_load_refuses_a_pipe_and_a_directory(tmp_path, expect_error):
    read_end, write_end = os.pipe()
    directory = os.open(tmp_path, os.O_RDONLY)
    try:
        for fd in (read_end, directory):
            with expect_error(RuntimeError, "must be a regular file"):
                ttnn.load_tensor(f"/proc/self/fd/{fd}")
    finally:
        for fd in (read_end, write_end, directory):
            os.close(fd)


# 2. lexical aliases of descriptor paths


@pytest.mark.parametrize(
    "spelling",
    [
        "/proc/self/fd/0{fd}",
        "/proc/self/fd/{fd}/",
        "/proc/self/fd//{fd}",
        "/proc/self/fd/{fd}/.",
        "/proc/self/fd/+{fd}",
        "/proc/self/fd/ {fd}",
        "/proc/self/fd/0",
    ],
)
def test_noncanonical_descriptor_spellings_are_refused(descriptor, spelling, expect_error):
    with expect_error(RuntimeError, "canonical /proc/self/fd"):
        ttnn.load_tensor(spelling.format(fd=descriptor))


@pytest.mark.parametrize(
    "spelling",
    [
        "/proc/thread-self/fd/{fd}",
        "/proc/{pid}/fd/{fd}",
        "/proc/self/task/{pid}/fd/{fd}",
        "/proc/self/./fd/{fd}",
        "/proc/self/../self/fd/{fd}",
        "//proc/self/fd/{fd}",
    ],
)
def test_other_proc_descriptor_paths_are_refused(descriptor, spelling, expect_error):
    with expect_error(RuntimeError, "must not alias a /proc/self/fd descriptor"):
        ttnn.load_tensor(spelling.format(fd=descriptor, pid=os.getpid()))


def test_symlink_to_a_descriptor_is_refused(tmp_path, descriptor, expect_error):
    link = tmp_path / "alias.tensorbin"
    link.symlink_to(f"/proc/self/fd/{descriptor}")
    with expect_error(RuntimeError, "must not alias a /proc/self/fd descriptor"):
        ttnn.load_tensor(link)


def test_symlinked_directory_into_proc_is_refused(tmp_path, descriptor, expect_error):
    (tmp_path / "proc_self").symlink_to("/proc/self")
    with expect_error(RuntimeError, "must not alias a /proc/self/fd descriptor"):
        ttnn.load_tensor(tmp_path / "proc_self" / "fd" / f"{descriptor}.tensorbin")


def test_relative_symlink_chain_into_proc_is_refused(tmp_path, descriptor, expect_error):
    (tmp_path / "a").symlink_to("b")
    (tmp_path / "b").symlink_to(f"/proc/self/fd/{descriptor}")
    renamed = tmp_path / "c.tensorbin"
    renamed.symlink_to("a")
    with expect_error(RuntimeError, "must not alias a /proc/self/fd descriptor"):
        ttnn.load_tensor(renamed)


def test_symlink_loop_is_bounded(tmp_path, expect_error):
    (tmp_path / "x.tensorbin").symlink_to("y.tensorbin")
    (tmp_path / "y.tensorbin").symlink_to("x.tensorbin")
    with expect_error(RuntimeError, "hop bound"):
        ttnn.load_tensor(tmp_path / "x.tensorbin")


def test_ordinary_symlink_still_loads(tmp_path, tensor_file):
    link = tmp_path / "link.tensorbin"
    link.symlink_to(tensor_file[0])
    assert torch.equal(_host(ttnn.load_tensor(link)), tensor_file[1])


# 3. fail closed on shape / dtype / layout, with cleanup


@pytest.mark.parametrize("route", ["fd", "path"])
def test_matching_expectations_load(tensor_file, descriptor, route):
    name = f"/proc/self/fd/{descriptor}" if route == "fd" else tensor_file[0]
    loaded = ttnn.load_tensor(
        name, expected_shape=(64, 96), expected_dtype=ttnn.bfloat16, expected_layout=ttnn.TILE_LAYOUT
    )
    assert torch.equal(_host(loaded), tensor_file[1])


@pytest.mark.parametrize("route", ["fd", "path"])
@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(expected_shape=(96, 64)), "holds shape"),
        (dict(expected_shape=(1, 64, 96)), "holds shape"),
        (dict(expected_dtype=ttnn.float32), "holds dtype"),
        (dict(expected_layout=ttnn.ROW_MAJOR_LAYOUT), "holds layout"),
    ],
)
def test_mismatch_fails_closed_and_deallocates(
    monkeypatch, tensor_file, descriptor, route, kwargs, message, expect_error
):
    deallocated = []
    real_deallocate = ttnn.deallocate
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: (deallocated.append(t), real_deallocate(t, *a, **k)))
    name = f"/proc/self/fd/{descriptor}" if route == "fd" else tensor_file[0]
    with expect_error(RuntimeError, message):
        ttnn.load_tensor(name, **kwargs)
    assert len(deallocated) == 1


def test_cleanup_failure_reports_both_errors(monkeypatch, descriptor, expect_error):
    def failing_deallocate(*_args, **_kwargs):
        raise ValueError("cleanup failed")

    monkeypatch.setattr(ttnn, "deallocate", failing_deallocate)
    with expect_error(core._LoadTensorCleanupError, "cleanup also failed") as info:
        ttnn.load_tensor(f"/proc/self/fd/{descriptor}", expected_shape=(1, 1))
    assert isinstance(info.value.primary_error, RuntimeError)
    assert "holds shape" in str(info.value.primary_error)
    assert isinstance(info.value.cleanup_error, ValueError)


@pytest.mark.parametrize("route", ["fd", "path"])
def test_file_changed_during_the_read_fails_closed(monkeypatch, tensor_file, descriptor, route, expect_error):
    path, _ = tensor_file
    real_load = ttnn._ttnn.tensor.load_tensor_flatbuffer
    deallocated = []
    real_deallocate = ttnn.deallocate
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: (deallocated.append(t), real_deallocate(t, *a, **k)))

    def load_then_append(name, device):
        tensor = real_load(name, device)
        with open(path, "ab") as f:
            f.write(b"\0" * 64)
        return tensor

    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", load_then_append)
    name = f"/proc/self/fd/{descriptor}" if route == "fd" else path
    with expect_error(RuntimeError, "changed during tensor load"):
        ttnn.load_tensor(name)
    assert len(deallocated) == 1


def test_path_replaced_during_the_read_fails_closed(monkeypatch, tensor_file, expect_error):
    path, _ = tensor_file
    real_load = ttnn._ttnn.tensor.load_tensor_flatbuffer
    host_replacement = ttnn.from_torch(torch.zeros(64, 96).bfloat16(), layout=ttnn.TILE_LAYOUT)

    def load_then_replace(name, device):
        tensor = real_load(name, device)
        ttnn.dump_tensor(path, host_replacement)
        return tensor

    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", load_then_replace)
    with expect_error(RuntimeError, "changed during tensor load"):
        ttnn.load_tensor(path)


def test_descriptor_closed_and_reused_by_another_file_is_not_followed(tmp_path, tensor_file):
    path, expected = tensor_file
    fd = os.open(path, os.O_RDONLY)
    name = f"/proc/self/fd/{fd}"
    seen = []
    real_load = ttnn._ttnn.tensor.load_tensor_flatbuffer

    def load_after_caller_reuses_its_fd(native_name, device):
        os.close(fd)
        other = _dump(tmp_path / "other.tensorbin", torch.ones(64, 96).bfloat16())
        reused = os.open(other, os.O_RDONLY)
        seen.append(reused)
        try:
            return real_load(native_name, device)
        finally:
            os.close(reused)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", load_after_caller_reuses_its_fd)
        loaded = ttnn.load_tensor(name)
    assert torch.equal(_host(loaded), expected)


# 4. tensorbin payload binding and exact types


def test_payload_binding_refuses_extra_bytes_before_the_native_read(monkeypatch, tensor_file, descriptor, expect_error):
    path, _ = tensor_file
    with open(path, "ab") as f:
        f.write(b"\0" * 64)
    calls = []
    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", lambda *a: calls.append(a))
    with expect_error(RuntimeError, "payload is"):
        ttnn.load_tensor(
            f"/proc/self/fd/{descriptor}",
            expected_shape=(64, 96),
            expected_dtype=ttnn.bfloat16,
            expected_layout=ttnn.TILE_LAYOUT,
        )
    assert calls == []


@pytest.mark.parametrize("header_size", [0, 7, 1 << 40])
def test_payload_binding_refuses_a_bad_header_prefix(monkeypatch, tensor_file, header_size, expect_error):
    path, _ = tensor_file
    data = bytearray(open(path, "rb").read())
    data[:8] = header_size.to_bytes(8, "little")
    open(path, "wb").write(bytes(data))
    calls = []
    monkeypatch.setattr(ttnn._ttnn.tensor, "load_tensor_flatbuffer", lambda *a: calls.append(a))
    with expect_error(RuntimeError, "header size"):
        ttnn.load_tensor(path, expected_shape=(64, 96), expected_dtype=ttnn.bfloat16, expected_layout=ttnn.TILE_LAYOUT)
    assert calls == []


def test_payload_binding_refuses_a_truncated_prefix(tmp_path, expect_error):
    path = tmp_path / "short.tensorbin"
    path.write_bytes(b"\x08\0\0")
    with expect_error(RuntimeError, "truncated"):
        ttnn.load_tensor(path, expected_shape=(32, 32), expected_dtype=ttnn.bfloat16, expected_layout=ttnn.TILE_LAYOUT)


@pytest.mark.parametrize(
    "shape,dtype,layout,payload",
    [
        ((64, 96), ttnn.bfloat16, ttnn.TILE_LAYOUT, 6 * 2048),
        ((4096,), ttnn.bfloat16, ttnn.TILE_LAYOUT, 128 * 2048),
        ((64,), ttnn.float32, ttnn.TILE_LAYOUT, 2 * 4096),
        ((4096, 4096), ttnn.bfloat4_b, ttnn.TILE_LAYOUT, 16384 * 576),
        ((4096, 8192), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 32768 * 1088),
        ((64, 256, 512), ttnn.bfloat16, ttnn.TILE_LAYOUT, 64 * 8 * 16 * 2048),
        ((4, 8192), ttnn.bfloat16, ttnn.TILE_LAYOUT, 256 * 2048),
        ((154880, 4096), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 154880 * 4096 * 2),
    ],
)
def test_expected_payload_bytes(shape, dtype, layout, payload):
    assert core._expected_tensor_contract(shape, dtype, layout) == (shape, payload)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("shape", [(32, 64), (33, 65), (2, 64, 32), (100,)])
def test_payload_binding_matches_dumped_files(tmp_path, dtype, shape):
    torch.manual_seed(1)
    path = _dump(tmp_path / "t.tensorbin", torch.randn(shape), dtype=dtype)
    loaded = ttnn.load_tensor(path, expected_shape=shape, expected_dtype=dtype, expected_layout=ttnn.TILE_LAYOUT)
    assert tuple(loaded.shape) == shape


def test_payload_binding_matches_a_row_major_file(tmp_path):
    path = _dump(tmp_path / "r.tensorbin", torch.randn(7, 40).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn.load_tensor(path, expected_shape=(7, 40), expected_dtype=ttnn.bfloat16, expected_layout=ttnn.ROW_MAJOR_LAYOUT)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(expected_shape=(64, True)), "expected_shape"),
        (dict(expected_shape=(64.0, 96)), "expected_shape"),
        (dict(expected_shape=(64, -96)), "expected_shape"),
        (dict(expected_shape=()), "expected_shape"),
        (dict(expected_shape="64,96"), "expected_shape"),
        (dict(expected_dtype="bfloat16"), "expected_dtype"),
        (dict(expected_dtype=torch.bfloat16), "expected_dtype"),
        (dict(expected_layout="TILE"), "expected_layout"),
        (dict(device="device"), "device must be a ttnn.MeshDevice"),
    ],
)
def test_exact_types(descriptor, kwargs, message, expect_error):
    with expect_error(RuntimeError, message):
        ttnn.load_tensor(f"/proc/self/fd/{descriptor}", **kwargs)


def test_bytes_path_is_refused(tensor_file, expect_error):
    with expect_error(RuntimeError, "must be a str"):
        ttnn.load_tensor(os.fsencode(str(tensor_file[0])))


def test_concurrent_fd_loads(tensor_file, descriptor):
    results, errors = [], []

    def worker():
        try:
            results.append(_host(ttnn.load_tensor(f"/proc/self/fd/{descriptor}")))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(torch.equal(r, tensor_file[1]) for r in results)


@pytest.mark.skipif(bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH")), reason="needs real device readback")
def test_fd_load_to_device_matches_path_load(device, tensor_file, descriptor):
    path, expected = tensor_file
    by_fd = ttnn.load_tensor(f"/proc/self/fd/{descriptor}", device=device, expected_shape=(64, 96))
    by_path = ttnn.load_tensor(path, device=device)
    assert torch.equal(ttnn.to_torch(by_fd), expected)
    assert torch.equal(ttnn.to_torch(by_fd), ttnn.to_torch(by_path))
