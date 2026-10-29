# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py, 14b's mesh server): Stage 14b bisect of the served int24svc accuracy failure (program 2026-10-22
05:23Z / 05:26Z). The generator path passes prefill-then-decode TF on the exact plugin prompt (161 ids) at TP8; this job
adds the vLLM adapter's differences one at a time on the resident serving backend with Stage 16's TP prefill attached:
  base  gen.reset() + Glm53Generator.generate (identity row, fresh slot): the harness;
  a     the plugin's page row (its blocks + 0-padding -> vllm_page_rows) via prefill_forward + decode_forward(tokens -1);
  b     a + the adapter's decode: decode_forward(reload_inputs=False, read_from_device=False) + read_decode_output;
  c     a + the plugin's sampling row (temperature 0, top_k = vocab 154880, top_p 1);
  d     a after two earlier requests in the same slot WITHOUT gen.reset (9's order: the 65- then the 31-id prompt, a few
        decode steps each): the slot / cache / KDA / DSA state a reused vLLM slot carries.
Row: the first index where each variant's greedy tokens differ from base (None = identical).
ARGS: b (32), layers ("all"), native (true), rev, prompts (path of {n: {ids, page_table}}), n (161), new (16),
slot (0), knobs, buckets, record (true), tag."""
import datetime as dt
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
S16 = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(os.path.dirname(S16), "s14b"))
sys.path.insert(0, S16)
import m5common as M5  # noqa: E402
import s14common as C  # noqa: E402
import svc_attach  # noqa: E402

import models.autoports.zai_org_glm_5_3_flash.tt.generator as G  # noqa: E402
import models.autoports.zai_org_glm_5_3_flash.tt.stack_model as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tests.common import hf_config  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.generator_vllm import vllm_page_rows  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams  # noqa: E402

mesh, args, ctx = MESH, ARGS, CTX  # noqa: F821
cfg = hf_config()
_L = args.get("layers", "all")
ids = list(range(cfg.num_hidden_layers)) if _L == "all" else [int(v) for v in _L]
B, NATIVE, REV = int(args.get("b", 32)), bool(args.get("native", True)), args.get("rev", "t25")
N, NEW, SLOT = int(args.get("n", 161)), int(args.get("new", 16)), int(args.get("slot", 0))
TP = int(mesh.get_num_devices())
PROMPTS = {(int(k) if k.isdigit() else k): v for k, v in json.load(open(args["prompts"])).items()}
# (d): the requests slot 0 served before (9's dump order; program 05:33Z adds a 512-id one first): (prompt, page row)
PRIOR = [(str(k) if not str(k).isdigit() else int(k), r) for k, r in args.get("prior", [["512g", None], [128, [1, 2]], [128, [3, 2]], [128, [1, 2]]])]
greedy = SamplingParams()
plugin_sp = SamplingParams(temperature=0.0, top_k=154880, top_p=1.0)


def log(*a):
    print(f"S14B-PLUGIN-BISECT {dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}", *a, flush=True)


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if int(x) != int(y):
            return i
    return None


def row_of(n, blocks=None, positions=None):
    """the request's vLLM row: its prompt blocks, + blocks appended as decode grows (vLLM allocates on demand: free
    blocks from the top of the pool here), 0-padded -> vllm_page_rows"""
    b = gen._own_binding()
    pt = torch.zeros((1, int(b.blocks_per_seq)), dtype=torch.int64)
    blocks = list(blocks or [int(v) for v in PROMPTS[n]["page_table"] if int(v) > 0])
    need = -(-int(positions or len(PROMPTS[n]["ids"])) // int(be.block_size))
    free = (v for v in range(int(b.num_blocks) - 1, 0, -1) if v not in blocks)
    while len(blocks) < need:
        blocks.append(next(free))
    pt[0, : len(blocks)] = torch.tensor(blocks)
    return vllm_page_rows(pt).to(torch.int64)


def request(n, new, sp, mode="sync", reset=True, blocks=None):
    """one request of the plugin prompt n in SLOT: prefill_forward + `new` decode_forward steps"""
    b = gen._own_binding()
    if reset:
        gen.reset()
    toks = PROMPTS[n]["ids"]
    pt = row_of(n, blocks, len(toks) + new)
    first = gen.prefill_forward(torch.tensor([toks], dtype=torch.int64), page_table=pt, kv_cache=b.kv_cache,
                                prompt_lens=[len(toks)], empty_slots=[SLOT], sampling_params=[sp])  # fmt: skip
    seq = [int(first[0])]
    for i in range(new):
        if mode == "sync":
            o = gen.decode_forward(torch.tensor([-1]), torch.tensor([len(toks) + i]), page_table=pt, kv_cache=b.kv_cache,
                                   slots=[SLOT], sampling_params=[sp])  # fmt: skip
            seq.append(int(torch.as_tensor(o).reshape(-1)[0]))
        else:  # the adapter: device-resident inputs, the tokens read after the step
            h = gen.decode_forward(None, None, page_table=None, kv_cache=b.kv_cache, reload_inputs=False,
                                   sampling_params=[sp], read_from_device=False)  # fmt: skip
            seq.append(int(torch.as_tensor(gen.read_decode_output(h)).reshape(-1)[0]))
    b.slots[SLOT].active = False  # the request ends (vLLM frees the slot; its pages stay as they are)
    return seq


key = M5.serving(ctx, mesh, SM, G, B=B, ids=ids, cfg=cfg, native=NATIVE, rev=REV, log=log)
att = svc_attach.Attached(ctx, knobs=args.get("knobs"), buckets=str(args.get("buckets", "128,512,2048")))
be, gen = att.be, att.gen
pf = None
res = {}
try:
    pf = att.attach()
    gen.reset()
    base = [int(v) for v in gen.generate(PROMPTS[N]["ids"], NEW + 1, sampling=greedy, stop_on_eos=False)]
    log("base", base)
    # the program's order (05:33Z): (d) slot reuse, (a) page rows, (b) async decode, (c) sampling rows
    def safe(f):
        try:
            return f()
        except Exception as e:  # noqa: BLE001 -- one variant's API mismatch must not hide the others
            return f"error: {e!r}"[:300]

    fresh = safe(lambda: request(N, NEW, greedy))  # 9's page row on a fresh slot (gen.reset)

    def reused():
        gen.reset()
        for k, blk in PRIOR:
            request(k, NEW, greedy, reset=False, blocks=blk)
        return request(N, NEW, greedy, reset=False)

    res["d_reused_slot_vs_fresh"] = safe(reused)
    res["a_page_row_vs_identity"] = fresh
    res["b_async_decode_vs_a"] = safe(lambda: request(N, NEW, greedy, mode="async"))
    res["c_plugin_sampling_vs_a"] = safe(lambda: request(N, NEW, plugin_sp))
    for k, v in res.items():
        log(k, v)
finally:
    try:
        gen.reset()
        gen._ensure_identity_rows(gen._own_binding(), list(range(B)))
    except Exception as e:  # noqa: BLE001
        log("identity rows restore failed:", repr(e))
    pf = None
    log(f"restored: decode-only warm-up {att.restore()} s")
A = res.get("a_page_row_vs_identity")
divs = {}
for k, v in res.items():
    ref_ = base if k == "a_page_row_vs_identity" else A  # (a) against the harness; (d), (b), (c) against (a)
    divs[k] = first_diff(v, ref_) if isinstance(v, list) and isinstance(ref_, list) else (v if not isinstance(v, list) else "no ref")
first = next((k for k in ("d_reused_slot_vs_fresh", "a_page_row_vs_identity", "b_async_decode_vs_a", "c_plugin_sampling_vs_a")
              if isinstance(divs.get(k), int)), None)  # fmt: skip
log("FIRST DIFFERING TOGGLE:", first, divs.get(first) if first else None)
row_ = dict(test=f"served-prefill bisect: the vLLM adapter's differences one at a time vs Glm53Generator.generate on the "
                 f"plugin's {N}-id prompt, greedy {NEW} steps, tp{TP} layers {len(ids)}", milestone="M5c-triage",
            status="info", first_differing_toggle=first, first_divergence=divs, base=base, tokens=res, rev=REV, tag=args.get("tag", f"m5c-plugin-bisect-tp{TP}"))  # fmt: skip
log(json.dumps(divs))
if args.get("record", True) and not C.MOCK:
    C.record("pcc", row_)
RESULT = dict(first=first, divs=divs)
