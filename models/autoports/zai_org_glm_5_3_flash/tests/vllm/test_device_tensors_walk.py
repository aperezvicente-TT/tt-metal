"""Stage 12 (2026-10-27, the A + C startup profile): tp_image.device_tensors -- GLM53_L1_REPORT's object walk, 196.6 s
of a 578-s mock startup -- formats a path only for a tensor it finds and skips leaves before the call. The result must
be the old walk's (a path string per element), in the same order: checked here against a verbatim copy of it, on a
graph with cycles, shared objects, SimpleNamespaces, models.* objects, foreign objects, dict keys of several types,
host / unallocated / device tensors and long leaf lists. Host only (a stand-in ttnn.Tensor)."""

import random
import time
import types

import pytest

from models.autoports.zai_org_glm_5_3_flash.tt import tp_image as TI


class _Tensor:
    def __init__(self, dev, alloc=True):
        self.dev, self.alloc = dev, alloc

    def storage_type(self):
        return "DEVICE" if self.dev else "HOST"

    def is_allocated(self):
        return self.alloc


_FAKE_TTNN = types.SimpleNamespace(Tensor=_Tensor, StorageType=types.SimpleNamespace(DEVICE="DEVICE"))


def _reference(root, prefix="be"):
    """the walk before Stage 12's change, verbatim but for the stand-in ttnn"""
    ttnn = _FAKE_TTNN
    out, seen = [], set()

    def visit(obj, path):
        if isinstance(obj, TI._SKIP) or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, ttnn.Tensor):
            if obj.storage_type() == ttnn.StorageType.DEVICE and obj.is_allocated():
                out.append((path, obj))
            return
        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                visit(v, f"{path}[{k!r}]")
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                visit(v, f"{path}[{i}]")
        elif isinstance(obj, types.SimpleNamespace) or type(obj).__module__.startswith("models."):
            d = getattr(obj, "__dict__", None)
            if d is not None:
                for k, v in list(d.items()):
                    visit(v, f"{path}.{k}")

    visit(root, prefix)
    return out


class _Model:  # a models.* object
    pass


_Model.__module__ = "models.fake.layer"


class _Foreign:  # not walked (not models.*, not a namespace)
    pass


def _graph(seed, n=400):
    rng = random.Random(seed)
    pool = []

    def leaf():
        return rng.choice([1, 2.5, "s", b"b", None, True, int, list(range(rng.randint(0, 50)))])

    def make(depth):
        r = rng.random()
        if depth > 5 or r < 0.25:
            return rng.choice([_Tensor(True), _Tensor(False), _Tensor(True, alloc=False), leaf(), _Foreign()])
        if r < 0.4 and pool:
            return rng.choice(pool)  # shared / cyclic references
        kind = rng.choice(["dict", "list", "tuple", "ns", "model"])
        kids = [make(depth + 1) for _ in range(rng.randint(0, 6))]
        if kind == "dict":
            keys = [rng.choice([f"k{j}", j, (j, "t"), 2.5 * j]) for j in range(len(kids))]
            o = dict(zip(keys, kids))
        elif kind == "list":
            o = list(kids)
        elif kind == "tuple":
            o = tuple(kids)
        else:
            o = types.SimpleNamespace() if kind == "ns" else _Model()
            for j, v in enumerate(kids):
                setattr(o, f"a{j}", v)
        pool.append(o)
        return o

    root = _Model()
    root.children = [make(0) for _ in range(n // 10)]
    root.loop = root  # a cycle through the root
    pool[0:0] = [root]
    if pool[1:]:
        tgt = pool[-1]
        if isinstance(tgt, list):
            tgt.append(root)
    return root


@pytest.mark.parametrize("seed", range(8))
def test_same_tensors_same_paths_same_order(seed, monkeypatch):
    monkeypatch.setattr(TI, "ttnn", _FAKE_TTNN)
    root = _graph(seed)
    want, got = _reference(root), TI.device_tensors(root)
    assert [(p, id(t)) for p, t in got] == [(p, id(t)) for p, t in want]
    assert want, "the graph holds device tensors"
    assert TI.device_tensors(3) == _reference(3) == []


def test_the_walk_is_cheaper_on_long_leaf_lists(monkeypatch):
    """the profile's shape: model objects holding long lists of ints / strings (arg tables, page rows) -- the old walk
    formatted a path string per element"""
    monkeypatch.setattr(TI, "ttnn", _FAKE_TTNN)
    root = _Model()
    root.layers = []
    for i in range(40):
        m = _Model()
        m.args = [list(range(2000)) for _ in range(10)]
        m.names = {f"n{j}": f"v{j}" for j in range(500)}
        m.w = _Tensor(True)
        root.layers.append(m)
    t0 = time.perf_counter()
    want = _reference(root)
    t1 = time.perf_counter()
    got = TI.device_tensors(root)
    t2 = time.perf_counter()
    assert [(p, id(t)) for p, t in got] == [(p, id(t)) for p, t in want] and len(got) == 40
    print(f"old walk {t1 - t0:.3f} s, new {t2 - t1:.3f} s ({(t1 - t0) / max(t2 - t1, 1e-9):.1f}x)")
    assert t2 - t1 < t1 - t0
