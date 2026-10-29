# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py --mgd 2x4 under the mock wrapper, GLM53_MOCK=1): the PMK cache-write phase's host mock -- the
pure-python layout tests (tests/s16/pmk_cache_test.py), then pmk_cache_unit.py on the mock mesh: the three kernels
JIT-build for the mock Blackhole cluster, the descriptors / CBs / runtime args validate, the op-path pieces build; the
readback compares are skipped (garbage under the mock). RESULT lists the built kernel ELFs. ARGS: as pmk_cache_unit."""
import glob
import os
import runpy
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import s16common as S  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
tree = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))))
test = os.path.join(os.path.dirname(here), "pmk_cache_test.py")
out = subprocess.run([sys.executable, "-u", test], capture_output=True, text=True, env={**os.environ, "PMK_TREE": tree})
print(out.stdout[-3000:], out.stderr[-3000:], flush=True)
assert "PMK_CACHE_TEST PASS" in out.stdout, "layout tests failed"
ns = runpy.run_path(
    os.path.join(here, "pmk_cache_unit.py"),
    init_globals={"MESH": MESH, "ARGS": {"t": 2048, **ARGS, "record": False}, "CTX": CTX},  # noqa: F821
    run_name="__job__",
)  # noqa: F821
cache = os.environ.get("TT_METAL_CACHE", "")
elfs = sorted(
    {
        os.path.relpath(e, cache)
        for k in ("cache_lat", "cache_dsa", "cache_state")
        for e in glob.glob(os.path.join(cache, f"**/{k}/**/*.elf"), recursive=True)
    }
)
S.log(f"built kernel ELFs: {elfs}")
RESULT = dict(layout_tests=out.stdout.strip().splitlines()[-5:], unit=ns["RESULT"], elfs=elfs)
