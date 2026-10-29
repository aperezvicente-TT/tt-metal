# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 6 per-width decode traces (`tt/per_width.py`): one-process device validation (maintenance #3 phase D).

Phases (rows `per_width/*` in $GLM53_OUT_DIR/tests.jsonl, perf rows in perf.jsonl):
  ref1    max_batch-1 generator: AIME24 greedy token-out chain, golden-128 greedy chain, teacher-forced logits-out
          decode after prefill at boundary lengths (the B=1 reference of this process)
  pw      max_batch-32 generator with per-width traces: the same requests alone in slot 0 (width-1 traces) -> tokens
          and logits vs ref1 (bitwise expected: the width-1 traces run the B=1 programs); then 2 / 4 copies of the AIME24
          request in slots 0..w-1 (width-2 / -4 traces): every copy's tokens == slot 0 == ref1
  switch  request A alone (width 1), B admitted to slot 1 (width 2), B released (width 1 again): A's tokens and
          positions == A uninterrupted; capture counters unchanged (no capture at request time)
  perf    token-out ms/step with 1 / 2 / 4 / 8 / 32 live slots (widths 1, 2, 4 per-width; 8 and 32 the full trace)
usage: python run_per_width.py --mesh 2x4 [--widths 1,2,4] [--phases ref1,pw,switch,perf] [--deadline ...]
       (mock: --mock --eager --units "0:L0,L1,L2,L3,L4")
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_full_model as R  # noqa: E402

import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import GREEDY  # noqa: E402

TF_LENS = [95, 97, 128, 1000]


def build(a, mesh, subs):
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator
    from models.autoports.zai_org_glm_5_3_flash.tt.per_width import build_per_width_backend

    t0 = time.perf_counter()
    kw = dict(handoff=a.handoff, max_batch=32)
    if subs is not None:
        kw["submeshes"] = subs
    if a.units:
        kw["units"] = a.units
    else:
        kw["partition"], kw["partition_key"] = a.partition, a.partition_key
    be = build_per_width_backend(str(R.MODEL_DIR), mesh, widths=a.widths, **kw)
    gen = build_generator(str(R.MODEL_DIR), mesh, backend=be, max_batch=1, max_seq_len=a.max_seq_len)
    R.record(
        dict(test="per_width/build", widths=list(a.widths), load_s=round(time.perf_counter() - t0, 1), status="pass")
    )
    return be, gen


def _gen(a, be, gen, max_batch):
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    g = getattr(a, "_g", None) or gen
    if g.max_batch == max_batch:
        g.reset()
        return g
    g.teardown()
    g = Glm53Generator(be, tokenizer=gen.tokenizer, max_batch=max_batch, max_seq_len=a.max_seq_len)
    t0 = time.perf_counter()
    g.warmup(capture=R.TRACE)
    R.record(
        dict(
            test=f"per_width/warmup_b{max_batch}",
            warmup_s=round(time.perf_counter() - t0, 1),
            widths=list(be.m.pw_widths),
            traces=g.trace_report()["bindings"][0],
            dram_per_chip=be.dram_report(),
            status="pass",
        )
    )
    a._g = g
    return g


def _requests(a):
    aime = R._aime_prompt()[0]
    gold = R._golden(2048)[0][:128]
    return {"aime24_chat": aime, "golden128": gold}


def _chain(g, prompt, slot, steps):
    pt = R._pages(g, slot, len(prompt) + steps + 1, "identity")
    return R._greedy_chain(g, prompt, slot, pt, steps)


def _tf_rows(g, slot):
    """Teacher-forced logits-out: prefill golden[:n] + 2 decode steps; the three last-row logits per n."""
    ids, _, _ = R._golden(2048)
    out = {}
    for n in TF_LENS:
        g.reset()
        pt = R._pages(g, slot, n + 2, "identity")
        rows = [R._prefill_rows(g, slot, ids[:n], pt)]
        rows.append(R._decode_row(g, ids[n], n, slot))
        rows.append(R._decode_row(g, ids[n + 1], n + 1, slot))
        out[n] = torch.stack(rows)
    g.reset()
    return out


def phase_ref1(a, be, gen):
    g = _gen(a, be, gen, 1)
    steps = 2 if a.mock else a.steps
    ref = {"chains": {k: _chain(g, p, 0, steps) for k, p in _requests(a).items()}, "tf": _tf_rows(g, 0)}
    a._ref = ref
    return R.record(
        dict(
            test="per_width/ref1",
            chains=ref["chains"],
            tf_top1={n: [int(x) for x in v.argmax(-1)] for n, v in ref["tf"].items()},
            status="info",
        )
    )


def phase_pw(a, be, gen):
    ref = a._ref
    g = _gen(a, be, gen, 32)
    steps = 2 if a.mock else a.steps
    rows = []
    # width 1: alone in slot 0
    chains = {}
    for k, p in _requests(a).items():
        g.reset()
        chains[k] = _chain(g, p, 0, steps)
    tf = _tf_rows(g, 0)
    diff = {n: float((tf[n] - ref["tf"][n]).abs().max()) for n in TF_LENS}
    same_tok = all(chains[k] == ref["chains"][k] for k in chains)
    rows.append(
        R.record(
            dict(
                test="per_width/width1_vs_b1",
                chains_equal=same_tok,
                chains=chains,
                tf_max_abs_logit_diff=diff,
                tf_bitwise=all(v == 0.0 for v in diff.values()),
                replays_by_width=g.trace_report()["bindings"][0].get("decode_replays_by_width"),
                status="pass" if same_tok and all(v == 0.0 for v in diff.values()) else "fail",
            )
        )
    )
    # widths 2 / 4: copies of the AIME24 request in slots 0..w-1 (admitted together, decoded together)
    prompt = _requests(a)["aime24_chat"]
    for w in [x for x in be.m.pw_widths if x > 1]:
        g.reset()
        n = len(prompt)
        firsts = []
        for s in range(w):
            first = g.prefill_forward(
                torch.tensor([prompt]),
                page_table=R._pages(g, s, n + steps + 1, "identity"),
                kv_cache=g._own.kv_cache,
                prompt_lens=[n],
                empty_slots=[s],
                sampling_params=[GREEDY],
            )
            firsts.append(int(first[0]))
        toks = [[t] for t in firsts]
        for i in range(steps):
            out = g.decode_forward(
                torch.full((w, 1), -1, dtype=torch.int64),
                torch.full((w,), n + i, dtype=torch.int64),
                page_table=None,
                kv_cache=g._own.kv_cache,
                slots=list(range(w)),
                sampling_params=[GREEDY] * w,
                enable_trace=R.TRACE,
            )
            for s in range(w):
                toks[s].append(int(out[s]))
        ok = all(t == toks[0] for t in toks) and toks[0] == ref["chains"]["aime24_chat"]
        rows.append(
            R.record(
                dict(
                    test=f"per_width/width{w}_copies",
                    width=w,
                    copies_identical=all(t == toks[0] for t in toks),
                    slot0_equals_b1=toks[0] == ref["chains"]["aime24_chat"],
                    first_divergence_vs_b1=next(
                        (i for i, (x, y) in enumerate(zip(toks[0], ref["chains"]["aime24_chat"])) if x != y), None
                    ),
                    tokens=toks[0],
                    status="pass" if ok else "fail",
                )
            )
        )
    g.reset()
    return rows


def phase_switch(a, be, gen):
    """A alone 8 steps (width 1) -> B joins slot 1 for 8 steps (width 2) -> B leaves, A alone 8 steps (width 1)."""
    g = _gen(a, be, gen, 32)
    reqs = _requests(a)
    pa, pb = reqs["aime24_chat"], reqs["golden128"]
    seg = 2 if a.mock else 8
    kv = g._own.kv_cache
    c0 = dict(g.counters)
    alone = _chain(g, pa, 0, 3 * seg)
    g.reset()
    first = g.prefill_forward(
        torch.tensor([pa]),
        page_table=R._pages(g, 0, len(pa) + 3 * seg + 1, "identity"),
        kv_cache=kv,
        prompt_lens=[len(pa)],
        empty_slots=[0],
        sampling_params=[GREEDY],
    )
    ta, pos = [int(first[0])], len(pa)

    def step(slots, positions):
        out = g.decode_forward(
            torch.full((len(slots), 1), -1, dtype=torch.int64),
            torch.tensor(positions),
            page_table=None,
            kv_cache=kv,
            slots=slots,
            sampling_params=[GREEDY] * len(slots),
            enable_trace=R.TRACE,
        )
        return [int(x) for x in out]

    for _ in range(seg):
        ta.append(step([0], [pos])[0])
        pos += 1
    g.prefill_forward(
        torch.tensor([pb]),
        page_table=R._pages(g, 1, len(pb) + seg + 1, "identity"),
        kv_cache=kv,
        prompt_lens=[len(pb)],
        empty_slots=[1],
        sampling_params=[GREEDY],
    )
    pb_pos = len(pb)
    for _ in range(seg):
        o = step([0, 1], [pos, pb_pos])
        ta.append(o[0])
        pos, pb_pos = pos + 1, pb_pos + 1
    g.release_slot(kv, 1)
    for _ in range(seg):
        ta.append(step([0], [pos])[0])
        pos += 1
    dev_pos = [int(x) for x in be.read(be.io.cur_pos)[:2]]
    caps = {k: g.counters[k] - c0.get(k, 0) for k in ("decode_trace_captures", "sampling_trace_captures")}
    ok = ta == alone and caps["decode_trace_captures"] == 0
    g.reset()
    return R.record(
        dict(
            test="per_width/width_switch",
            tokens_equal_uninterrupted=ta == alone,
            first_divergence=next((i for i, (x, y) in enumerate(zip(ta, alone)) if x != y), None),
            device_positions_after=dev_pos,
            expected_position_slot0=pos,
            captures_during=caps,
            replays_by_width=g.trace_report()["bindings"][0].get("decode_replays_by_width"),
            status="pass" if ok else "fail",
        )
    )


def phase_perf(a, be, gen):
    g = _gen(a, be, gen, 32)
    prompt = _requests(a)["golden128"]
    steps = 2 if a.mock else 64
    kv = g._own.kv_cache
    out = {}
    for live in (1, 2, 4, 8, 32):
        g.reset()
        n = len(prompt)
        for s in range(live):
            g.prefill_forward(
                torch.tensor([prompt]),
                page_table=R._pages(g, s, n + steps + 2, "identity"),
                kv_cache=kv,
                prompt_lens=[n],
                empty_slots=[s],
                sampling_params=[GREEDY],
            )
        per = []
        for r in range(2):
            t0 = time.perf_counter()
            for i in range(steps // 2):
                p = n + r * (steps // 2) + i
                g.decode_forward(
                    torch.full((live, 1), -1, dtype=torch.int64),
                    torch.full((live,), p, dtype=torch.int64),
                    page_table=None,
                    kv_cache=kv,
                    slots=list(range(live)),
                    sampling_params=[GREEDY] * live,
                    enable_trace=R.TRACE,
                )
            per.append((time.perf_counter() - t0) / (steps // 2))
        out[live] = dict(
            width=g.decode_width(g._own, list(range(live))) if hasattr(g, "decode_width") else None,
            token_out_ms=[round(x * 1e3, 3) for x in per],
            token_out_ms_best=round(min(per) * 1e3, 3),
        )
    g.reset()
    return R.record(
        dict(test="per_width/perf_token_out", live_slots=out, tracking=R.TRACKING, status="info"), "perf.jsonl"
    )


PHASES = {"ref1": phase_ref1, "pw": phase_pw, "switch": phase_switch, "perf": phase_perf}
EST_MIN = {"ref1": 4, "pw": 8, "switch": 3, "perf": 6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x4", choices=["2x4", "1x1"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--units", default="")
    ap.add_argument("--partition", default=str(R.WS / "doc/ttnn/04-multichip-decoder/partition.json"))
    ap.add_argument("--partition-key", default="pipeline8_ring_wrap")
    ap.add_argument("--handoff", default="eth_direct", choices=["eth_direct", "eth_fifo", "host"])
    ap.add_argument("--max-seq-len", type=int, default=2051)
    ap.add_argument("--trace-region", type=int, default=800_000_000)
    ap.add_argument("--widths", default="1,2,4")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--phases", default="ref1,pw,switch,perf")
    ap.add_argument("--deadline", default="")
    a = ap.parse_args()
    a.widths = tuple(int(x) for x in a.widths.split(",") if x)
    # an already-queued launcher's phase list can be overridden (and reordered) by $GLM53_WORK/runD.phases
    over = Path(os.environ.get("GLM53_WORK", "/opt/glm53/work-s6")) / "runD.phases"
    if not a.mock and over.exists():
        a.phases = over.read_text().strip()
    R.TRACE = not a.eager
    R.GEN_MAX_BATCH = 32
    if a.mock:
        a.mesh, a.handoff = "1x1", "host"
    R.record(dict(test="per_width/start", argv=sys.argv, phases=a.phases, status="info"))
    end = None
    if a.deadline:
        end = datetime.strptime(a.deadline, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    mesh = be = None
    rc = 0
    try:
        mesh, subs = R.open_mesh(a)
        be, gen = build(a, mesh, subs)
        a._g = gen
        gen.warmup(capture=R.TRACE)
        for ph in [p for p in a.phases.split(",") if p]:
            if end is not None and time.time() + 60 * (EST_MIN.get(ph, 2) + 2) > end:
                R.record(dict(test=f"per_width/phase/{ph}", status="skipped", reason=f"deadline {a.deadline}"))
                continue
            t0 = time.perf_counter()
            try:
                PHASES[ph](a, be, gen)
            except Exception as e:  # noqa: BLE001
                rc = 1
                R.record(
                    dict(
                        test=f"per_width/phase/{ph}",
                        status="error",
                        error=repr(e)[:2000],
                        traceback=traceback.format_exc()[-6000:],
                        wall_s=round(time.perf_counter() - t0, 1),
                    )
                )
                traceback.print_exc()
            be.m.synchronize()
        R.record(dict(test="per_width/trace_report", report=a._g.trace_report(), status="info"))
    except Exception as e:  # noqa: BLE001
        rc = 2
        R.record(
            dict(test="per_width/fatal", status="error", error=repr(e)[:2000], traceback=traceback.format_exc()[-6000:])
        )
        traceback.print_exc()
    finally:
        if mesh is not None:
            from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import close_pipeline_mesh

            try:
                if be is not None:
                    for t in list(be.traces):
                        be.release(t)
                close_pipeline_mesh(mesh, be.m if be is not None else None, fabric=a.mesh != "1x1")
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                rc = rc or 3
    R.record(dict(test="per_width/end", rc=rc, status="info"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
