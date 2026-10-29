# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py; outside the tree: runs on the server's tree): Stage 14b M5c stale-input regression (review #11) on the
serving backend: rows 0..L-1 are live decode rows with pending device inputs (token rows, positions, and a sampled
readback row the generator has not read yet: the async decode path), then `prefill_chunk` runs a prompt for ANOTHER slot
(as decode replays of that slot alone). Afterwards every other row's sampled readback, token row and position must be
the values from before (stack_model.prefill_chunk puts them back: int3 5edda88e7e0) and each live row's KDA state must
be bitwise unchanged (its rows were inactive in every prefill replay). ARGS: b (32), layers, native (true), live (8),
slot (8), plen (16), shared_weights, rev, record, tag."""
import os
import sys

import torch

import ttnn

sys.path.insert(0, os.path.join(os.environ["TT_METAL_HOME"], "models/autoports/zai_org_glm_5_3_flash/tests/s14b"))
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402

mesh = MESH  # noqa: F821
args, ctx = ARGS, CTX  # noqa: F821
B = int(args.get("b", 32))
LIVE, SLOT, PLEN = int(args.get("live", 8)), int(args.get("slot", 8)), int(args.get("plen", 16))
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=bool(args.get("native", True)), rev=args.get("rev"),
                 log=lambda *a: print("STALE", *a, flush=True))  # fmt: skip
be = ctx[key]["backend"]
kv = be._kv
io = kv.extra["io"]
sync = lambda: ttnn.synchronize_device(mesh)  # noqa: E731
assert SLOT >= LIVE and SLOT < B, (SLOT, LIVE, B)
be.reset_all(kv)
tok = torch.zeros(B, dtype=torch.int64)
tok[:LIVE] = torch.tensor([1000 + 13 * u for u in range(LIVE)])
pos = torch.full((B,), -1, dtype=torch.int64)
pos[:LIVE] = torch.tensor([40 + 3 * u for u in range(LIVE)])
be.write(io.tokens, tok)
be.write(io.cur_pos, pos)
sync()
for _ in range(3):  # the live rows decode a few steps: real KDA state, real sampled rows
    be.replay(kv.extra["decode_trace"])
sync()
pending = torch.zeros(B, dtype=torch.int64)
pending[:LIVE] = torch.tensor([5000 + u for u in range(LIVE)])  # an async readback not consumed yet
be.write(io.sampled, pending)
sync()
tok_b, pos_b, samp_b = be.read(io.tokens), be.read(io.cur_pos), be.read(io.sampled)
st_b = [be.export_kda_state(kv, u) for u in range(LIVE)] if not SM.MOCK else []
prompt = torch.tensor([2000 + 7 * i for i in range(PLEN)])
be.prefill_chunk(prompt, kv_cache=kv, slot=SLOT, chunk_start=0, n_valid=PLEN, logits_rows="none", io=io)
sync()
tok_a, pos_a, samp_a = be.read(io.tokens), be.read(io.cur_pos), be.read(io.sampled)
st_a = [be.export_kda_state(kv, u) for u in range(LIVE)] if not SM.MOCK else []
others = [u for u in range(B) if u != SLOT]
chk = dict(sampled_restored=bool(torch.equal(samp_a[others], samp_b[others])),
           tokens_restored=bool(torch.equal(tok_a[others], tok_b[others])),
           positions_restored=bool(torch.equal(pos_a, pos_b)),
           live_states_bitwise=all(x["L"] == y["L"] and torch.equal(x["S"], y["S"]) and torch.equal(x["C"], y["C"])
                                    for a, b in zip(st_a, st_b) for x, y in zip(a, b)))  # fmt: skip
ok = all(chk.values()) or SM.MOCK
res = dict(B=B, live=LIVE, slot=SLOT, plen=PLEN, layers=len(ids), mock=SM.MOCK, **chk)
print("STALE RESULT", res, flush=True)
be.reset_all(kv)
sync()
if args.get("record") and not SM.MOCK:
    C.record("tests", dict(test=f"m5c stale inputs: prefill_chunk of slot {SLOT} ({PLEN} tokens) while rows 0..{LIVE - 1} "
                                f"are live with pending sampled readbacks ({len(ids)} layers, B={B}, tp{be.tp}): every "
                                f"other row's sampled / token / position restored, live KDA states bitwise",
                           status="pass" if ok else "fail", milestone="M5c", tag=args.get("tag", "m5c-stale-inputs"),
                           **res))  # fmt: skip
RESULT = dict(ok=ok, **res)
