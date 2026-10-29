# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 6 full-model driver: ONE long-lived device process (DEVICE RULES 2026-10-18: <= 1 mesh open / close per
30 min; the 45-layer load takes minutes) that builds `tt/model.py` + `tt/generator.py` once and runs the requested
phases against the same generator, writing one JSON row per check to $GLM53_OUT_DIR (default $GLM53_E6).

  --mesh 2x4  --plan partition   all 8 chips, Stage 4's pipeline8_ring_wrap (the full model)
  --mesh 1x1  --units "0:L0,L1,L2,L3,L4"   reduced full-model probe on one chip (every layer kind + real terminal)
  --mock                          1x1 mock cluster (compile / shape / allocation check, garbage numerics)

Phases (comma list, in order): mvp, tf, prefill_check, ar, qual, trace, nonaligned, batch, perf, determinism.
Every phase is guarded: an exception is recorded as a failed row and the next phase runs (the process exits cleanly,
closing the mesh, whatever happens).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent
sys.path.insert(0, str(HERE))
WS = Path(os.environ.get("GLM53_WS", "/opt/glm53/kit"))
RUNTIME = WS / "vendor/tenstorrent-skills/plugins/tt-model-bringup/runtime"
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))

import ttnn  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import GREEDY  # noqa: E402

E6 = Path(os.environ.get("GLM53_E6", WS / "doc/ttnn/06-full-model"))
OUT = Path(os.environ.get("GLM53_OUT_DIR", E6))
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
TRACKING = os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
TRACE = True  # False under --eager (mock): the phases below decode without traces
AIME_STEPS = 100


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha():
    try:
        sha = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(MODEL_DIR), "rev-parse", "--short=11", "HEAD"], text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-c", "safe.directory=*", "-C", str(MODEL_DIR), "status", "--porcelain", "--", "tt", "tests"],
            text=True,
        ).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return os.environ.get("GLM53_CODE_SHA", "unknown")


CODE = code_sha()
GEN_MAX_BATCH = None  # max_batch of the run's main generator (--max-batch), set in main()


def record(row, name="tests.jsonl"):
    row = {
        "time_utc": now(),
        "run_id": RUN_ID,
        **row,
        "trace_alloc_tracking": TRACKING,
        "generator_max_batch": GEN_MAX_BATCH,  # phases that build their own generator say so in their rows
        "watcher": os.environ.get("TT_METAL_WATCHER", ""),
        "visible": os.environ.get("TT_VISIBLE_DEVICES"),
        "code": CODE,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("ROW", json.dumps(row)[:600], flush=True)
    return row


# ====================================================================================================== build
def open_mesh(a):
    from models.autoports.zai_org_glm_5_3_flash.tt.multichip_decoder import open_pipeline_mesh

    if a.mesh == "1x1":
        if not a.mock:  # one visible P300 chip = ClusterType CUSTOM: needs the 1x1 mesh graph descriptor (Stage 04a)
            os.environ.setdefault(
                "TT_MESH_GRAPH_DESC_PATH", str(MODEL_DIR / "mesh" / "glm_1x1_mesh_graph_descriptor.textproto")
            )
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=a.trace_region)
        return mesh, {(0, 0): mesh}
    shape = tuple(int(x) for x in a.mesh.split("x"))
    mgd = {"2x4": "glm8_2x4_mesh_graph_descriptor.textproto", "2x2": "glm4_2x2_mesh_graph_descriptor.textproto"}
    path = MODEL_DIR / "mesh" / mgd[a.mesh] if a.mesh in mgd else None
    mesh = open_pipeline_mesh(shape, trace_region_size=a.trace_region, mgd=str(path) if path else None)
    return mesh, None


def build(a, mesh, subs):
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import build_generator
    from models.autoports.zai_org_glm_5_3_flash.tt.model import build_model_backend

    t0 = time.perf_counter()
    kw = dict(handoff=a.handoff, max_batch=32)  # backend slot capacity; the generator allocates a.max_batch
    if subs is not None:
        kw["submeshes"] = subs
    if a.units:
        kw["units"] = a.units
    else:
        kw["partition"] = a.partition
        kw["partition_key"] = a.partition_key
    be = build_model_backend(str(MODEL_DIR), mesh, **kw)
    load_s = time.perf_counter() - t0
    gen = build_generator(str(MODEL_DIR), mesh, backend=be, max_batch=a.max_batch, max_seq_len=a.max_seq_len)
    record(
        dict(
            test="full_model/build",
            mesh=a.mesh,
            plan=[
                {"chip": s.plan.chip, "coord": list(s.plan.coord), "units": [str(u) for u in s.plan.units]}
                for s in be.m.stages
            ],
            hops=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in h.items()} for h in be.m.hops],
            layers=len(be.m.layers),
            load_s=round(load_s, 1),
            terminal_load_s=round(be.term.load_s, 1),
            layer_load_s=getattr(be.loader, "load_s", {}),
            max_batch=a.max_batch,
            max_seq_len=gen.max_seq_len,
            supported_context=be.supported_context,
            status="pass",
        )
    )
    return be, gen


def warmup(be, gen, capture=True):
    t0 = time.perf_counter()
    b = gen.warmup(capture=capture)
    t1 = time.perf_counter()
    rep = gen.trace_report()
    dram = be.dram_report()
    record(
        dict(
            test="full_model/warmup",
            warmup_s=round(t1 - t0, 1),
            traces=rep["bindings"][0],
            counters=rep["counters"],
            dram_per_chip=dram,
            backend_counters=dict(be.counters),
            status="pass",
        )
    )
    return b


# ====================================================================================================== phases
def aime_ref():
    import harness_common as H

    return H.registered_aime24()


def load_ref(path):
    from readiness_check.schema import load_reference

    return load_reference(Path(path))


def phase_mvp(a, be, gen):
    """AIME24 chat prompt -> prefill -> 100 greedy tokens, on-device sampling, traced decode."""
    ref = load_ref(aime_ref())
    e = ref.entries[0]
    prompt = e.prompt_tokens.reshape(-1).tolist()
    tok = gen.tokenizer
    gen.reset()
    c0 = dict(gen.counters)
    t0 = time.perf_counter()
    out = gen.generate(prompt, a.steps, enable_trace=not a.eager, stop_on_eos=False)
    dt = time.perf_counter() - t0
    c1 = dict(gen.counters)
    delta = {k: c1[k] - c0.get(k, 0) for k in c1}
    ref_tokens = e.generated_tokens.reshape(-1).tolist()
    first_div = next((i for i, (x, y) in enumerate(zip(out, ref_tokens)) if x != y), None)
    text = tok.decode([t for t in out if t < 154856]) if tok is not None else None
    match = sum(int(x == y) for x, y in zip(out, ref_tokens))
    row = record(
        dict(
            test="full_model/mvp_aime24_greedy",
            prompt_tokens=len(prompt),
            generated=len(out),
            tokens=out,
            text=text,
            hf_reference_tokens=ref_tokens,
            free_running_match_vs_hf=match,
            first_divergence=first_div,
            wall_s=round(dt, 3),
            host_work=delta,
            eager=a.eager,
            status="pass" if len(out) == a.steps else "fail",
        )
    )
    if a.mvp_file and row["status"] == "pass":
        cmd = " ".join(sys.argv)
        with open(a.mvp_file, "w") as f:
            f.write(
                f"# Stage 6 MVP: first end-to-end greedy tokens on 8 chips ({now()})\n"
                f"code_sha={CODE} branch=glm53/glm53-flash-s6 worktree=/opt/glm53/tt-metal-s6\n"
                f"command=scripts/dev.sh {cmd}\n"
                f"prompt=AIME24 chat reference {aime_ref()} ({len(prompt)} tokens)\n"
                f"generated={len(out)} greedy tokens, traced decode + on-device sampler; host_work={json.dumps(delta)}\n"
                f"free_running_match_vs_hf={match}/{len(out)} first_divergence={first_div}\n"
                f"tokens={out}\n"
                f"----- completion -----\n{text}\n"
            )
        print(f"MVP_TOKENS written: {a.mvp_file}", flush=True)
    return row


def phase_tf(a, be, gen, refs=None, label=""):
    """Plugin teacher forcing (generator.generate(next_input=..., enable_trace=True)) on the AIME24 reference (+ the
    golden 128 / 512 references when requested)."""
    import harness_common as H
    from readiness_check.run_teacher_forcing import run_teacher_forcing

    refs = refs or [aime_ref()] + [
        H.golden_refpt(int(T), Path(os.environ.get("GLM53_WORK", "/opt/glm53/work-s6")) / f"golden_{T}.refpt")
        for T in a.tf_goldens.split(",")
        if T
    ]
    rows = []
    for ref in refs:
        t0 = time.perf_counter()
        c0 = dict(gen.counters)
        per = run_teacher_forcing(
            model_dir=MODEL_DIR, reference_path=Path(ref), mesh_device=None, build_kwargs={"generator": gen}
        )
        c1 = dict(gen.counters)
        agg = H.aggregate(per)
        rows.append(
            record(
                dict(
                    test=f"full_model/teacher_forcing{label}",
                    reference=str(ref),
                    per_entry=per,
                    aggregate=agg,
                    gate=H.gate(agg),
                    host_work={k: c1[k] - c0.get(k, 0) for k in c1},
                    wall_s=round(time.perf_counter() - t0, 1),
                    status="pass" if H.gate(agg)["pass"] else "fail",
                )
            )
        )
    return rows


def phase_prefill_check(a, be, gen, refs=None, label=""):
    import harness_common as H
    from readiness_check.run_prefill_check import run_prefill_check

    refs = refs or [aime_ref()]
    rows = []
    for ref in refs:
        t0 = time.perf_counter()
        per = run_prefill_check(
            model_dir=MODEL_DIR, reference_path=Path(ref), mesh_device=None, build_kwargs={"generator": gen}
        )
        agg = H.aggregate(per)
        rows.append(
            record(
                dict(
                    test=f"full_model/prefill_check{label}",
                    reference=str(ref),
                    per_entry=per,
                    aggregate=agg,
                    gate=H.gate(agg),
                    wall_s=round(time.perf_counter() - t0, 1),
                    status="pass" if H.gate(agg)["pass"] else "fail",
                )
            )
        )
    return rows


def _delta(c0, c1):
    return {k: c1[k] - c0.get(k, 0) for k in c1 if c1[k] - c0.get(k, 0)}


def _aime_prompt():
    e = load_ref(aime_ref()).entries[0]
    return e.prompt_tokens.reshape(-1).tolist(), e.generated_tokens.reshape(-1).tolist()


_HOST_HEAD = {}


def _host_head(be):
    """fp32 host copy of the terminal weights (norm, lm_head) from the tile cache: reference head for the probe."""
    if not _HOST_HEAD:
        from models.autoports.zai_org_glm_5_3_flash.tests.common import CACHE_DIR, MANIFEST
        from models.autoports.zai_org_glm_5_3_flash.tt.optimized_decoder import FdTileCacheSource

        src = FdTileCacheSource(CACHE_DIR, MANIFEST)
        _HOST_HEAD["norm"] = ttnn.to_torch(src._load(src.tensors["norm.weight"])).float().reshape(-1)
        _HOST_HEAD["lm"] = ttnn.to_torch(src._load(src.tensors["lm_head.weight"])).float().reshape(4096, -1)
    return _HOST_HEAD


def _pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    d = a.norm() * b.norm()
    return float((a @ b) / d) if d > 0 else 1.0


def phase_probe(a, be, gen, steps=16):
    """Terminal numerics vs a host fp32 head on the device stack output, and token-out paths compared: traced
    on-device sampling vs eager on-device sampling vs host-sampling compatibility mode (same greedy tokens = the
    device token feedback consumed the right token every step)."""
    prompt, _ = _aime_prompt()
    out = {}
    for name, kw in (("traced", dict(enable_trace=True)), ("eager", dict(enable_trace=False))):
        gen.reset()
        c0 = dict(gen.counters)
        t0 = time.perf_counter()
        out[name] = gen.generate(prompt, steps, stop_on_eos=False, **kw)
        out[name + "_s"] = round(time.perf_counter() - t0, 3)
        out[name + "_work"] = _delta(c0, dict(gen.counters))
        if name == "traced":
            # stack output + logits of the LAST traced step (same step): device head vs host head
            last = be.m.stages[-1].groups[0].out_buf
            st = ttnn.to_torch(last).float().reshape(-1, 16384)[0]
            dev_logits = be.read(be.io.logits)[0, :154880]
            hh = _host_head(be)
            h = st.reshape(4, 4096).mean(0)
            h = h * torch.rsqrt((h * h).mean() + 1e-5) * hh["norm"]
            host_logits = h.to(torch.bfloat16).float() @ hh["lm"]
            out["head_pcc"] = _pcc(dev_logits, host_logits)
            out["head_top1_dev"] = int(dev_logits.argmax())
            out["head_top1_host"] = int(host_logits.argmax())
            out["head_max_abs"] = float((dev_logits - host_logits).abs().max())
            out["positions_after"] = be.m.read_positions()
    gen.reset()
    gen.host_sampling = True
    try:
        c0 = dict(gen.counters)
        out["host_sampling"] = gen.generate(prompt, steps, stop_on_eos=False, enable_trace=True)
        out["host_sampling_work"] = _delta(c0, dict(gen.counters))
    finally:
        gen.host_sampling = False
    ok = out["traced"] == out["eager"] == out["host_sampling"] and out["head_pcc"] > 0.999
    exp = len(prompt) + steps - 1
    pos_ok = all(p[0] == exp for p in out["positions_after"])
    return record(
        dict(
            test="full_model/probe_token_paths",
            prompt_tokens=len(prompt),
            steps=steps,
            **out,
            expected_position_after=exp,
            positions_coherent=pos_ok,
            status="pass" if ok and pos_ok else "fail",
        )
    )


def phase_trace(a, be, gen, steps=24):
    """Split-sampling trace evidence: (1) token feedback: the sampler's output equals the next decode input on device
    and tokens match the host-sampling path; (2) position coherence: every stage's in-trace cur_pos = start + steps
    with zero host position refreshes in the loop; (3) unchanged page table: zero page-table refreshes in the loop;
    (4) changed page table (reversed block order for slot 0, written in place under the live traces) reproduces the
    identity-table tokens; (5) greedy -> sampled -> greedy reuses the captured traces (no recapture)."""
    prompt, _ = _aime_prompt()
    b = gen._own
    caps0 = (gen.counters["decode_trace_captures"], gen.counters["sampling_trace_captures"])
    rows = []
    # (1)-(3): low-level loop with per-step device inspection
    gen.reset()
    first = gen.prefill_forward(
        torch.tensor([prompt]),
        page_table=b.pt_shadow[:1].clone(),
        kv_cache=b.kv_cache,
        prompt_lens=[len(prompt)],
        sampling_params=[GREEDY],
    )
    toks = [int(first[0])]
    feed_ok, c0, c_admit = [], dict(gen.counters), None
    pos = len(prompt)
    for i in range(steps):
        if i == 1:  # step 0 carries the request-boundary position write (admission); steady state from step 1
            c_admit = dict(gen.counters)
        before_in = int(be.read(be.io.tokens)[0])
        t = gen.decode_forward(
            torch.tensor([[-1]]), torch.tensor([pos]), page_table=None, kv_cache=b.kv_cache, sampling_params=[GREEDY]
        )
        nxt_in = int(be.read(be.io.tokens)[0])
        feed_ok.append(dict(step=i, input=before_in, expected_input=toks[-1], sampled=int(t[0]), next_input=nxt_in))
        toks.append(int(t[0]))
        pos += 1
    loop_work = _delta(c_admit, dict(gen.counters))  # steps 1..N-1 (steady state)
    first_step_work = _delta(c0, c_admit)
    positions = be.m.read_positions()
    fb = all(r["input"] == r["expected_input"] and r["next_input"] == r["sampled"] for r in feed_ok)
    pos_ok = all(p[0] == pos for p in positions)
    rows.append(
        record(
            dict(
                test="full_model/trace_token_feedback",
                steps=steps,
                per_step=feed_ok,
                tokens=toks,
                loop_host_work=loop_work,
                feedback_on_device=fb,
                token_refresh_in_loop=loop_work.get("token_refresh", 0),
                status="pass" if fb and loop_work.get("token_refresh", 0) == 0 else "fail",
            )
        )
    )
    rows.append(
        record(
            dict(
                test="full_model/trace_position_coherence",
                positions_per_stage=positions,
                expected=pos,
                pos_refresh_in_loop=loop_work.get("pos_refresh", 0),
                status="pass" if pos_ok and loop_work.get("pos_refresh", 0) == 0 else "fail",
            )
        )
    )
    rows.append(
        record(
            dict(
                test="full_model/trace_page_table_unchanged",
                page_table_refresh_in_loop=loop_work.get("page_table_refresh", 0),
                status="pass" if loop_work.get("page_table_refresh", 0) == 0 else "fail",
            )
        )
    )
    # (4) changed page table: same request with slot 0's blocks in reversed order
    gen.reset()
    rev = torch.flip(b.identity_row(0), dims=[0]).reshape(1, -1)
    c0 = dict(gen.counters)
    alt = gen.generate_batch([prompt], steps + 1, stop_on_eos=False, page_table=rev, slots=[0])[0]
    w = _delta(c0, dict(gen.counters))
    gen.reset()
    rows.append(
        record(
            dict(
                test="full_model/trace_page_table_changed",
                reversed_table_tokens=alt,
                identity_table_tokens=toks,
                page_table_refresh=w.get("page_table_refresh", 0),
                decode_trace_captures=w.get("decode_trace_captures", 0),
                status="pass" if alt == toks and w.get("decode_trace_captures", 0) == 0 else "fail",
            )
        )
    )
    # (5) greedy -> sampled -> greedy
    from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams as SP

    g1 = gen.generate(prompt, 12, stop_on_eos=False)
    sm = gen.generate(prompt, 12, stop_on_eos=False, sampling=SP(temperature=1.0, top_p=0.95))
    g2 = gen.generate(prompt, 12, stop_on_eos=False)
    caps1 = (gen.counters["decode_trace_captures"], gen.counters["sampling_trace_captures"])
    rows.append(
        record(
            dict(
                test="full_model/trace_mode_switch",
                greedy_1=g1,
                sampled=sm,
                greedy_2=g2,
                captures_before=caps0,
                captures_after=caps1,
                trace_report=gen.trace_report()["bindings"],
                status="pass" if g1 == g2 == toks[:12] and caps0 == caps1 else "fail",
            )
        )
    )
    return rows


def sweep_lengths(max_len: int = 2050) -> list[int]:
    """Prompt lengths at every prefill bucket boundary and every 64-token page boundary +-1 (and 1-3, 31-33)."""
    out = {1, 2, 3, 31, 32, 33}
    for b in (64, 128, 256, 512, 1024, 2048):
        out |= {b - 1, b, b + 1}
    for k in range(1, max_len // 64 + 1):
        out |= {64 * k - 1, 64 * k, 64 * k + 1}
    return sorted(n for n in out if 1 <= n <= max_len)


_GOLD_CACHE = {}


def _golden(T: int):
    if T not in _GOLD_CACHE:
        tf = torch.load(WS / "goldens_gpu_fp32" / "teacher_forcing" / f"{T}.pt")
        lg = torch.load(WS / "goldens_gpu_fp32" / "logits" / f"{T}.pt")
        _GOLD_CACHE[T] = (tf["prompt_ids"].reshape(-1).tolist(), lg["prompt_top32_indices"], lg["prompt_top32_values"])
    return _GOLD_CACHE[T]


def tf_decode_sweep(gen, lens, *, label: str, slot: int = 0, pages: str = "identity"):
    """Teacher-forced decode right after a prefill, per length n: prefill golden prompt[:n] into `slot` (logits-out),
    then ONE decode step feeding golden token n at position n; both last-row predictions vs the fp32 golden top-32 at
    n-1 / n (the decode row checks the state the prefill left: KDA recurrent + conv history, MLA pages). pages:
    "identity" = the generator-owned slot rows; "tight" = vLLM-like caller rows of exactly ceil((n + 1) / 64)
    scattered physical blocks (the rest padded)."""
    b = gen._own
    rows, bad = [], []
    for n in lens:
        T = 2048 if n < 2048 else 4096
        ids, I, V = _golden(T)
        gen.reset()
        if pages == "identity":
            pt = b.identity_row(slot).reshape(1, -1)
        else:
            need = -(-(n + 1) // 64)
            pool = b.num_blocks
            blocks = [(slot * 7919 + 37 * j) % pool for j in range(need)]  # scattered, distinct (37 coprime to pool)
            pt = torch.full((1, b.blocks_per_seq), -1, dtype=torch.int64)
            pt[0, :need] = torch.tensor(blocks)
        pl = (
            gen.prefill_forward(
                torch.tensor([ids[:n]]), page_table=pt, kv_cache=b.kv_cache, prompt_lens=[n], empty_slots=[slot]
            )
            .reshape(-1)
            .float()
        )
        dl = (
            gen.decode_forward(
                torch.tensor([[ids[n]]]), torch.tensor([n]), page_table=None, kv_cache=b.kv_cache, slots=[slot]
            )
            .reshape(-1)
            .float()
        )
        p1, d1 = int(pl.argmax()), int(dl.argmax())
        g_p, g_d = I[n - 1][:5].tolist(), I[n][:5].tolist()
        ok = p1 in g_p and d1 in g_d
        r = dict(
            n=n,
            prefill_top1=p1,
            golden_prefill_top5=g_p,
            decode_top1=d1,
            golden_decode_top5=g_d,
            decode_top1_match=d1 == g_d[0],
            prefill_top1_match=p1 == g_p[0],
            decode_margin_golden=round(float(V[n][0] - V[n][1]), 3),
            ok=ok,
        )
        rows.append(r)
        if not ok:
            bad.append(n)
    gen.reset()
    n_top1 = sum(r["decode_top1_match"] for r in rows)
    return record(
        dict(
            test=f"full_model/tf_decode_sweep/{label}",
            slot=slot,
            pages=pages,
            lengths=len(rows),
            decode_top1_matches=n_top1,
            prefill_top1_matches=sum(r["prefill_top1_match"] for r in rows),
            not_in_golden_top5=bad,
            rows=rows,
            status="pass" if not bad else "fail",
        )
    )


def _prefill_rows(gen, slot, ids, pt):
    b = gen._own
    return (
        gen.prefill_forward(
            torch.tensor([ids]), page_table=pt, kv_cache=b.kv_cache, prompt_lens=[len(ids)], empty_slots=[slot]
        )
        .reshape(-1)
        .float()
    )


def _pages(gen, slot, n_pos, mode):
    b = gen._own
    if mode == "identity":
        return b.identity_row(slot).reshape(1, -1)
    need = -(-n_pos // 64)
    pt = torch.full((1, b.blocks_per_seq), -1, dtype=torch.int64)
    pt[0, :need] = torch.tensor([(slot * 7919 + 37 * j) % b.num_blocks for j in range(need)])
    return pt


def _gap_rows(gen, label, slot, mode, contexts, greedy_prefixes, steps=3):
    """Decode-vs-prefill numerics on the SAME context, per n: (i) prefill golden[:n] then ONE decode step fed golden
    token n at position n (decode path) vs (ii) a fresh prefill of golden[:n+1] (prefill path) -> top-1 of each,
    logprob gap of the prefill path's top-1 token, max |logprob diff| over the prefill path's top-20, KL(prefill ||
    decode). Then greedy chains: prefill golden[:n] + `steps` greedy decode steps, each step's token vs a fresh
    prefill of the same prefix (prefill-path greedy)."""
    ids, _, _ = _golden(2048)
    out = {"label": label, "slot": slot, "pages": mode, "gaps": [], "greedy": []}
    for n in contexts:
        gen.reset()
        _prefill_rows(gen, slot, ids[:n], _pages(gen, slot, n + 1, mode))
        d = (
            gen.decode_forward(
                torch.tensor([[ids[n]]]), torch.tensor([n]), page_table=None, kv_cache=gen._own.kv_cache, slots=[slot]
            )
            .reshape(-1)
            .float()
        )
        gen.reset()
        f = _prefill_rows(gen, slot, ids[: n + 1], _pages(gen, slot, n + 1, mode))
        gen.reset()
        ld, lf = torch.log_softmax(d.double(), -1), torch.log_softmax(f.double(), -1)
        top = torch.topk(lf, 20).indices
        out["gaps"].append(
            dict(
                n=n,
                decode_top1=int(d.argmax()),
                prefill_top1=int(f.argmax()),
                top1_agree=int(d.argmax()) == int(f.argmax()),
                logprob_gap_prefill_top1=round(float(lf[top[0]] - ld[top[0]]), 4),
                max_abs_logprob_diff_top20=round(float((lf[top] - ld[top]).abs().max()), 4),
                kl_prefill_decode=round(float((lf.exp() * (lf - ld)).sum()), 5),
                logits_pcc=round(_pcc(d, f), 6),
            )
        )
    for n in greedy_prefixes:
        gen.reset()
        pref = ids[:n]
        first = gen.prefill_forward(
            torch.tensor([pref]),
            page_table=_pages(gen, slot, n + steps + 1, mode),
            kv_cache=gen._own.kv_cache,
            prompt_lens=[n],
            empty_slots=[slot],
            sampling_params=[GREEDY],
        )
        toks = [int(first[0])]
        for k in range(steps):
            t = gen.decode_forward(
                torch.tensor([[-1]]),
                torch.tensor([n + k]),
                page_table=None,
                kv_cache=gen._own.kv_cache,
                slots=[slot],
                sampling_params=[GREEDY],
            )
            toks.append(int(t[0]))
        ctrl = []
        for k in range(1, steps + 1):  # prefill-path greedy of the same prefix (prompt + decode tokens so far)
            gen.reset()
            row = _prefill_rows(gen, slot, pref + toks[:k], _pages(gen, slot, n + k + 1, mode))
            ctrl.append(int(row.argmax()))
        gen.reset()
        out["greedy"].append(
            dict(
                n=n,
                decode_chain=toks,
                prefill_path_next=ctrl,
                first_mismatch=next((k + 1 for k in range(steps) if toks[k + 1] != ctrl[k]), None),
            )
        )
    return out


def phase_gap(a, be, gen):
    """Program 2026-10-19 01:19Z / 01:22Z: decode-vs-prefill logprob gaps and short greedy chains on the standalone
    generator at max_batch 1 (slot 0, identity pages) and max_batch 32 (slot 0 identity, slot 5 vLLM-like tight pages)
    — B=32-specific (masked path / page rows) vs inherent decode-vs-prefill numerics. Rebuilds a max_batch-32 KV
    cache: run it last."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    contexts = [95, 96, 97, 127, 128, 129, 256, 1023]
    prefixes = [95, 96, 97, 127, 129]
    res = [_gap_rows(gen, "b1_slot0_identity", 0, "identity", contexts, prefixes)]
    gen.teardown()
    g = Glm53Generator(be, tokenizer=gen.tokenizer, max_batch=32, max_seq_len=a.max_seq_len)
    t0 = time.perf_counter()
    g.warmup()
    warm = time.perf_counter() - t0
    res.append(_gap_rows(g, "b32_slot0_identity", 0, "identity", contexts, prefixes))
    res.append(_gap_rows(g, "b32_slot5_tight_pages", 5, "tight", contexts, prefixes))
    a._batch_gen = g
    rows = []
    for r in res:
        rows.append(
            record(
                dict(
                    test=f"full_model/decode_vs_prefill_gap/{r['label']}",
                    **r,
                    b32_warmup_s=round(warm, 1),
                    status="info",
                )
            )
        )
    return rows


def phase_tf32(a, be, gen):
    """Readiness on the SERVED configuration: the plugin's teacher forcing (AIME24 + --tf-goldens) and prefill check
    through a max_batch-32 generator (every decode step replays the 32-wide pipeline trace with inactive-row masking),
    in the same process (re-uses the batch phase's generator when present)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    g = getattr(a, "_batch_gen", None)
    if g is None or g.max_batch != 32:
        (g or gen).teardown()
        g = Glm53Generator(be, tokenizer=gen.tokenizer, max_batch=32, max_seq_len=a.max_seq_len)
        g.warmup()
        a._batch_gen = g
    g.reset()
    rows = phase_tf(a, be, g, label="_b32") + phase_prefill_check(a, be, g, label="_b32")
    g.reset()
    return rows


def phase_autocheck(a, be, gen):
    """Re-validation of Stage 4's guarded padded-chunk prefill policy ("auto": one chunk padded to its bucket only when
    it pads < 32 rows and is cheaper; with Stage 4's KDA dead-chunk fix 90a58532246 merged): the boundary sweep under
    "auto" vs the "plan" sweep of the nonaligned phase (same process) — prefill and decode-after-prefill top-1 must be
    identical — plus prefill wall time of lengths the policy pads. The backend's policy is restored to "plan"."""
    plan = getattr(a, "_sweep_plan", None)
    lens = sweep_lengths()
    timing = []
    be.chunk_policy = "auto"
    try:
        sw = tf_decode_sweep(gen, lens, label="b1_slot0_identity_auto")
        ids, _, _ = _golden(2048)
        for n in (250, 500, 1000, 2040, 2047):
            for pol in ("plan", "auto"):
                be.chunk_policy = pol
                gen.reset()
                t0 = time.perf_counter()
                gen.prefill_forward(
                    torch.tensor([ids[:n]]),
                    page_table=gen._own.pt_shadow[:1].clone(),
                    kv_cache=gen._own.kv_cache,
                    prompt_lens=[n],
                )
                timing.append(dict(n=n, policy=pol, prefill_s=round(time.perf_counter() - t0, 3)))
            be.chunk_policy = "auto"
    finally:
        be.chunk_policy = "plan"
        gen.reset()
    same = None
    if plan is not None:
        pr = {r["n"]: r for r in plan["rows"]}
        same = [
            (r["n"], r["prefill_top1"] == pr[r["n"]]["prefill_top1"], r["decode_top1"] == pr[r["n"]]["decode_top1"])
            for r in sw["rows"]
            if r["n"] in pr
        ]
    ok = same is not None and all(p and d for _, p, d in same)
    return record(
        dict(
            test="full_model/chunk_policy_auto_revalidation",
            identical_to_plan=ok,
            differing=[x for x in (same or []) if not (x[1] and x[2])],
            prefill_timing=timing,
            status="pass" if ok else "fail",
        )
    )


def phase_nonaligned(a, be, gen):
    """Prefill of non-aligned logical lengths through the public generator: prefixes of the fp32 golden prompts
    (causal: the last-row logits of a prefix of length n = row n-1 of the full prompt) vs the golden prompt top-32,
    then 1-4 greedy traced decode steps, and one teacher-forced decode step right after the prefill (golden token n at
    position n) vs the golden top-32 at position n (the decode state the prefill leaves)."""
    lens = [int(x) for x in a.nonaligned.split(",") if x]
    gold = WS / "goldens_gpu_fp32"
    rows = []
    for n in lens:
        T = 2048 if n < 2048 else 4096  # the decode check needs golden token n and its top-32 at position n
        tf = torch.load(gold / "teacher_forcing" / f"{T}.pt")
        lg = torch.load(gold / "logits" / f"{T}.pt")
        ids = tf["prompt_ids"].reshape(-1)[:n].tolist()
        top32 = lg["prompt_top32_indices"][n - 1].tolist()
        gen.reset()
        t0 = time.perf_counter()
        logits = gen.prefill_forward(
            torch.tensor([ids]), page_table=gen._own.pt_shadow[:1].clone(), kv_cache=gen._own.kv_cache, prompt_lens=[n]
        )
        dt = time.perf_counter() - t0
        row = logits.reshape(-1).float()
        top1 = int(row.argmax())
        dev_top5 = torch.topk(row, 5).indices.tolist()
        gen.reset()
        toks = gen.generate(ids, max(1, min(4, gen.supported_context - n + 1)), stop_on_eos=False)
        # decode right after the non-aligned prefill (the state it leaves): feed golden token n at position n
        # (logits-out decode, teacher-forced) -> top-1 vs the golden top-32 at position n (a state check the
        # prefill-only comparison cannot see)
        dec_top1, gold_dec = None, None
        if n < tf["prompt_ids"].numel() and n < gen.supported_context:
            gen.reset()
            b = gen._own
            gen.prefill_forward(
                torch.tensor([ids]),
                page_table=b.pt_shadow[:1].clone(),
                kv_cache=b.kv_cache,
                prompt_lens=[n],
                sampling_params=[GREEDY],
            )
            nxt = int(tf["prompt_ids"].reshape(-1)[n])
            dl = gen.decode_forward(torch.tensor([[nxt]]), torch.tensor([n]), page_table=None, kv_cache=b.kv_cache)
            dec_top1 = int(dl.reshape(-1).float().argmax())
            gold_dec = lg["prompt_top32_indices"][n].tolist()
            gen.reset()
        dec_ok = gold_dec is None or dec_top1 in gold_dec[:5]
        rows.append(
            record(
                dict(
                    test=f"full_model/nonaligned_prefill/{n}",
                    decode_after_prefill_top1=dec_top1,
                    decode_after_prefill_golden_top5=None if gold_dec is None else gold_dec[:5],
                    decode_after_prefill_top1_match=None if gold_dec is None else dec_top1 == gold_dec[0],
                    length=n,
                    golden_prompt=T,
                    top1=top1,
                    golden_top1=top32[0],
                    top1_match=top1 == top32[0],
                    top1_in_golden_top5=top1 in top32[:5],
                    golden_top1_in_dev_top5=top32[0] in dev_top5,
                    prefill_s=round(dt, 3),
                    decode_tokens=toks,
                    finite=bool(torch.isfinite(row).all()),
                    status="pass" if (top1 in top32[:5] and torch.isfinite(row).all() and dec_ok) else "fail",
                )
            )
        )
    rows.append(tf_decode_sweep(gen, sweep_lengths(), label="b1_slot0_identity"))
    a._sweep_plan = rows[-1]
    return rows


AR_DIR = MODEL_DIR / "doc" / "full_model" / "autoregressive"


def phase_ar(a, be, gen):
    """Free-running HF vs TT (greedy, stop at the checkpoint stop set): AIME24 chat (100 HF tokens) and the fp32
    golden chat prompts 128 / 512 (256 HF tokens) -> doc/full_model/autoregressive/<label>/ (plugin artifact format +
    check_degenerate_output.py)."""
    import harness_common as H
    import run_autoregressive as AR

    tok = gen.tokenizer
    rows = []
    cases = [("aime24", *_aime_prompt(), str(aime_ref()))]
    for T in (128, 512):
        tf = torch.load(WS / "goldens_gpu_fp32" / "teacher_forcing" / f"{T}.pt")
        cases.append(
            (
                f"golden_{T}",
                tf["prompt_ids"].reshape(-1).tolist(),
                tf["targets"].reshape(-1).tolist(),
                f"goldens_gpu_fp32/teacher_forcing/{T}.pt",
            )
        )
    for label, prompt, targets, src in cases:
        n = len(targets)
        hf = H.truncate_at_stop(targets, gen.stop_token_ids, n)
        gen.reset()
        res = AR.run(
            gen,
            tok,
            prompt,
            hf,
            AR_DIR / label,
            max_new_tokens=n,
            prompt_label=src,
            hf_source="fp32 layer-streamed GPU reference greedy continuation (reference of record)",
        )
        gen.reset()
        rows.append(
            record(
                dict(
                    test=f"full_model/autoregressive/{label}",
                    **res,
                    status="pass" if res["degenerate_check"]["exit_code"] in (0, 1) else "fail",
                )
            )
        )
    return rows


def phase_qual(a, be, gen):
    """$qualitative-check shared suite through the TT generator (chat template), fp32 HF controls at 128 tokens,
    plus a long-budget TT-only pass (answers past the reasoning) when --qual-long > 0."""
    import run_qualitative_suite as Q

    tok = gen.tokenizer
    suite = Q.load_suite()
    ctrl, _ = Q.suite_controls(tok, 128)
    rows = []
    res = Q.run(
        gen,
        tok,
        suite,
        MODEL_DIR / "doc" / "full_model" / "qualitative" / "suite_128",
        max_new_tokens=128,
        controls=ctrl,
        control_source="goldens_gpu_fp32_qualsuite (fp32 HF reference, 128 tokens)",
        label="full model 8-chip, greedy 128",
    )
    rows.append(
        record(
            dict(
                test="full_model/qualitative_suite_128",
                **res,
                status="pass" if res["degenerate_check_exit"] in (0, 1) else "fail",
            )
        )
    )
    if a.qual_long:
        res = Q.run(
            gen,
            tok,
            suite,
            MODEL_DIR / "doc" / "full_model" / "qualitative" / f"suite_{a.qual_long}",
            max_new_tokens=a.qual_long,
            controls=None,
            control_source="none (TT-only long budget)",
            label=f"full model 8-chip, greedy {a.qual_long}",
        )
        rows.append(
            record(
                dict(
                    test=f"full_model/qualitative_suite_{a.qual_long}",
                    **res,
                    status="pass" if res["degenerate_check_exit"] in (0, 1) else "fail",
                )
            )
        )
    return rows


def phase_perf(a, be, gen, reps=3, steps=128):
    """Batch-1 primary profile (prompt 128 / generate 128, the vLLM single-user default): warmed TTFT (prefill +
    first on-device sampled token read), token-out traced decode (decode trace + sampler trace + 1 token read per
    step) and the traced model-only decode (decode trace replays only: no sampler, no readback; the fair comparison
    to the layer-stack lower bound)."""
    tf = torch.load(WS / "goldens_gpu_fp32" / "teacher_forcing" / "128.pt")
    prompt = tf["prompt_ids"].reshape(-1).tolist()
    b = gen._own
    out = {"prompt_tokens": len(prompt), "gen_tokens": steps}
    ttft, tok_out, model_only = [], [], []
    for r in range(reps + 1):
        gen.reset()
        t0 = time.perf_counter()
        first = gen.prefill_forward(
            torch.tensor([prompt]),
            page_table=b.pt_shadow[:1].clone(),
            kv_cache=b.kv_cache,
            prompt_lens=[len(prompt)],
            sampling_params=[GREEDY],
        )
        t1 = time.perf_counter()
        pos = len(prompt)
        c0 = dict(gen.counters)
        t2 = time.perf_counter()
        for i in range(steps - 1):
            gen.decode_forward(
                torch.tensor([[-1]]),
                torch.tensor([pos + i]),
                page_table=None,
                kv_cache=b.kv_cache,
                sampling_params=[GREEDY],
            )
        t3 = time.perf_counter()
        work = _delta(c0, dict(gen.counters))
        if r == 0:
            continue  # warm request (first use of every prefill bucket replay path)
        ttft.append(t1 - t0)
        tok_out.append((t3 - t2) / (steps - 1))
    # model-only traced decode: replay the decode trace back to back (positions advance in-trace)
    dec = b.decode_traces[1] if 1 in b.decode_traces else next(iter(b.decode_traces.values()))
    be.m.synchronize()
    t0 = time.perf_counter()
    n = 64
    for _ in range(n):
        be.replay(dec)
    be.m.synchronize()
    model_only = (time.perf_counter() - t0) / n
    # sampler trace alone (greedy params; the decode trace's last logits stay in io.logits)
    smp = b.sampling_traces["greedy"]
    be.m.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        be.replay(smp)
    be.m.synchronize()
    sampler_only = (time.perf_counter() - t0) / n
    # the chip-0 terminal alone, eager (upper bound: includes the host dispatch of its ops): head = final norm + the
    # 4 DRAM-sharded lm_head chunks (~0.67 GB of bfp8 weights per token), embedding = token row -> 4 fp32 streams
    try:
        hm = be.head_mesh
        tok = ttnn.reshape(b.io.tokens.tensors[0], (1, 32))
        fns = (
            ("head", lambda: be.term.logits_decode(be.dec_streams32)),
            ("embed", lambda: be.term.embed_streams(tok, 1)),
        )
        for name, fn in fns:
            ttnn.deallocate(fn())
            ttnn.synchronize_device(hm)
            t0 = time.perf_counter()
            for _ in range(n):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(hm)
            out[f"terminal_{name}_eager_ms"] = round((time.perf_counter() - t0) / n * 1e3, 3)
    except Exception as e:  # noqa: BLE001  (the perf row must survive a failure of this extra measurement)
        out["terminal_eager_error"] = repr(e)[:500]
    gen.reset()
    out.update(
        ttft_ms=[round(x * 1e3, 2) for x in ttft],
        ttft_ms_median=round(sorted(ttft)[len(ttft) // 2] * 1e3, 2),
        token_out_ms_per_step=[round(x * 1e3, 3) for x in tok_out],
        token_out_ms_median=round(sorted(tok_out)[len(tok_out) // 2] * 1e3, 3),
        token_out_tps_per_user=round(1.0 / sorted(tok_out)[len(tok_out) // 2], 2),
        model_only_decode_ms=round(model_only * 1e3, 3),
        model_only_tps=round(1.0 / model_only, 2),
        sampler_only_ms=round(sampler_only * 1e3, 3),
        stage4_stack_lower_bound_ms={
            "single_chip_sum_b1": 73.225,
            "pipeline_decode_step_b1": 75.08,
            "source": "doc/ttnn/04-multichip-decoder/perf.jsonl run 20261018T214623Z",
        },
        loop_host_work_last_rep=work,
        decode_replay_blocking=False,
        tracking=TRACKING,
    )
    return record(dict(test="full_model/perf_b1_p128_g128", **out, status="pass"), "perf.jsonl")


def phase_determinism(a, be, gen):
    """Repeated identical inputs: prefill logits bitwise-equal across runs; greedy tokens identical."""
    prompt, _ = _aime_prompt()
    b = gen._own
    lg, toks = [], []
    for _ in range(2):
        gen.reset()
        lg.append(
            gen.prefill_forward(
                torch.tensor([prompt]),
                page_table=b.pt_shadow[:1].clone(),
                kv_cache=b.kv_cache,
                prompt_lens=[len(prompt)],
            )
            .reshape(-1)
            .float()
        )
        gen.reset()
        toks.append(gen.generate(prompt, 32, stop_on_eos=False))
    gen.reset()
    same = bool(torch.equal(lg[0], lg[1]))
    return record(
        dict(
            test="full_model/determinism_b1",
            prefill_logits_bitwise_equal=same,
            max_abs_diff=float((lg[0] - lg[1]).abs().max()),
            tokens_equal=toks[0] == toks[1],
            status="pass" if same and toks[0] == toks[1] else "fail",
        )
    )


def phase_tracy(a, be, gen):
    """Device-profiled windows of the (reduced) full model, one signpost pair each, profiler buffers read between
    windows: PERF_PREFILL_128 (prefill of the 128-token golden prompt + on-device first token), PERF_DECODE_TOKEN_OUT
    (decode trace + sampler trace + token read per step) and PERF_DECODE_MODEL (decode trace replays only)."""
    try:
        from tracy import signpost
    except ImportError:  # pragma: no cover
        signpost = lambda *x, **k: None
    flush = lambda: [ttnn.ReadDeviceProfiler(m) for m in {id(st.mesh): st.mesh for st in be.m.stages}.values()]
    tf = torch.load(WS / "goldens_gpu_fp32" / "teacher_forcing" / "128.pt")
    prompt = tf["prompt_ids"].reshape(-1).tolist()
    b = gen._own
    gen.reset()
    gen.generate(prompt, 4, stop_on_eos=False)  # warm request
    gen.reset()
    be.m.synchronize()
    flush()
    signpost("PERF_PREFILL_128")
    first = gen.prefill_forward(
        torch.tensor([prompt]),
        page_table=b.pt_shadow[:1].clone(),
        kv_cache=b.kv_cache,
        prompt_lens=[len(prompt)],
        sampling_params=[GREEDY],
    )
    be.m.synchronize()
    signpost("PERF_PREFILL_128_END")
    flush()
    pos = len(prompt)
    signpost("PERF_DECODE_TOKEN_OUT")
    for i in range(a.tracy_steps):
        gen.decode_forward(
            torch.tensor([[-1]]),
            torch.tensor([pos + i]),
            page_table=None,
            kv_cache=b.kv_cache,
            sampling_params=[GREEDY],
        )
    be.m.synchronize()
    signpost("PERF_DECODE_TOKEN_OUT_END")
    flush()
    dec = next(iter(b.decode_traces.values()))
    signpost("PERF_DECODE_MODEL")
    for _ in range(a.tracy_steps):
        be.replay(dec)
    be.m.synchronize()
    signpost("PERF_DECODE_MODEL_END")
    flush()
    gen.reset()
    return record(
        dict(
            test="full_model/tracy_windows",
            steps=a.tracy_steps,
            first_token=int(first[0]),
            windows=["PERF_PREFILL_128", "PERF_DECODE_TOKEN_OUT", "PERF_DECODE_MODEL"],
            status="pass",
        )
    )


def phase_fallback(a, be, gen):
    """Runtime host-fallback audit (Stage 1 / Stage 4 method: ttnn host-transfer entry points replaced by tripwires +
    a torch call profiler) of the device work the full model runs per token and per prefill chunk: the whole decode
    pipeline step incl. the chip-0 embedding prologue and head epilogue (what the 9 decode traces record), the
    sampler (what the sampler trace records), the prefill terminal (embedding of a 128-token chunk, last-row select +
    head, all-rows head) and the prefill first-token sampler. Host inputs (token ids, selector, positions) are written
    outside the audited windows, as at a request boundary. Positive control first."""
    from models.autoports.zai_org_glm_5_3_flash.tests.test_fallback_audit import Audit

    with Audit() as ctl:
        ttnn.from_torch(torch.zeros(1, 32), layout=ttnn.TILE_LAYOUT, device=be.head_mesh)
    control_ok = bool(ctl.torch_calls) and ctl.tripwires == ["from_torch"]
    prompt, _ = _aime_prompt()
    gen.reset()
    gen.generate(prompt, 2, stop_on_eos=False)  # live request state (positions / tokens valid)
    io, kv = be.io, gen._own.kv_cache
    windows = []
    with Audit() as A:
        be.decode(io, kv, 1)
    be.m.synchronize()
    windows.append(("decode_step_all_stages_with_terminal", A, io.logits.tensors[0]))
    with Audit() as A:
        be._sampler_body(io, "greedy")
    be.m.synchronize()
    windows.append(("sampler_greedy", A, io.sampled.tensors[0]))
    y = be.m.stages[-1].pbufs[128][0]
    with Audit() as A:
        x = be.term.embed_streams(be.ptok[128], 128)
    windows.append(("prefill_embedding_128", A, x))
    ttnn.deallocate(x)
    sel = torch.zeros(1, 1, 32, 128)
    sel[0, 0, 0, 127] = 1.0
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(sel, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), be.psel[128])
    with Audit() as A:
        h = ttnn.matmul(
            be.psel[128],
            y,
            dtype=ttnn.float32,
            compute_kernel_config=be.term.ckc_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lg = be.term.logits_decode(h)
        ttnn.copy(lg, be.plogits)
    windows.append(("prefill_last_row_head_128", A, be.plogits))
    ttnn.deallocate(h)
    ttnn.deallocate(lg)
    with Audit() as A:
        lg = be.term.logits_prefill(y)
    windows.append(("prefill_all_rows_head_128", A, lg))
    ttnn.deallocate(lg)
    with Audit() as A:
        be.psampler(be.plogits, tt_out_tok=be.ptoken)
    windows.append(("prefill_first_token_sampler", A, be.ptoken))
    be.m.synchronize()
    gen.reset()
    rows = []
    for mode, A, outt in windows:
        row = dict(
            test=f"full_model/fallback_audit/{mode}",
            torch_calls=sum(A.torch_calls.values()),
            torch_call_sites=sorted(A.torch_calls)[:10],
            host_transfer_calls=A.tripwires,
            output_on_device=ttnn.is_tensor_storage_on_device(outt),
            positive_control_detected=control_ok,
        )
        row["status"] = (
            "pass"
            if control_ok and not row["torch_calls"] and not row["host_transfer_calls"] and row["output_on_device"]
            else "fail"
        )
        rows.append(record(row, "fallback_audit.jsonl"))
    return rows


def phase_batch(a, be, gen):
    """Batch > 1 through the same backend (one KV cache per pipeline: the B=1 traces are released and a max_batch=B
    generator is built, warmed and captured in-process): mixed-length prompts over fixed slots with per-user budgets
    (finished users become inactive rows), duplicates of one prompt at different batch positions (must be identical),
    slot 0 vs the batch-1 run, a paused slot (not decoding for some steps) that must resume exactly, and the B=32
    token-out throughput. Leaves the B generator in place (run it last)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    B = a.batch_size
    prompt, _ = _aime_prompt()
    gen.reset()
    ref1 = gen.generate(prompt, 16, stop_on_eos=False)
    gen.teardown()
    t0 = time.perf_counter()
    g = Glm53Generator(be, tokenizer=gen.tokenizer, max_batch=B, max_seq_len=a.max_seq_len)
    g.warmup()
    warm_s = time.perf_counter() - t0
    g2048 = torch.load(WS / "goldens_gpu_fp32" / "teacher_forcing" / "2048.pt")["prompt_ids"].reshape(-1).tolist()
    lens = [
        1,
        20,
        33,
        63,
        64,
        65,
        100,
        130,
        257,
        500,
        1000,
        1500,
        2000,
        2047,
        7,
        300,
        777,
        1234,
        31,
        95,
        129,
        1800,
        640,
        1025,
        2,
        3,
        48,
        999,
        1600,
        12,
        450,
    ]
    prompts, mnt, labels = [], [], []
    dup = (0, B // 4, B - 1) if B >= 4 else (0,)
    j = 0
    for slot in range(B):
        if slot in dup:
            prompts.append(prompt)
            labels.append("aime24")
            mnt.append(16)
        else:
            n = lens[j % len(lens)]
            j += 1
            prompts.append(g2048[:n])
            labels.append(f"golden2048[:{n}]")
            mnt.append(16 if slot % 3 else 5)  # every third user finishes early -> inactive row afterwards
        mnt[-1] = min(mnt[-1], a.max_seq_len - len(prompts[-1]) + 1)
    c0 = dict(g.counters)
    t1 = time.perf_counter()
    outs = g.generate_batch(prompts, mnt, stop_on_eos=False)
    gen_s = time.perf_counter() - t1
    work = _delta(c0, dict(g.counters))
    dup_ok = all(outs[d] == outs[dup[0]] for d in dup)
    rows = [
        record(
            dict(
                test=f"full_model/batch{B}_mixed",
                batch=B,
                labels=labels,
                max_new_tokens=mnt,
                outputs=outs,
                duplicates=list(dup),
                duplicates_identical=dup_ok,
                slot0_vs_batch1=ref1,
                slot0_first_divergence_vs_batch1=next(
                    (i for i, (x, y) in enumerate(zip(outs[0], ref1)) if x != y), None
                ),
                warmup_s=round(warm_s, 1),
                generate_s=round(gen_s, 2),
                host_work=work,
                decode_replays_by_width=g.trace_report()["bindings"][0]["decode_replays_by_width"],
                status="pass" if dup_ok and all(len(o) == m for o, m in zip(outs, mnt)) else "fail",
            )
        )
    ]
    # paused slot: slots 5 and 9 decode together, then slot 9 sits out 4 steps (inactive, state untouched), then both
    b = g._own
    pa, pb = g2048[:300], g2048[:777]

    def low_level(pause):
        g.reset()
        first = g.prefill_forward(
            torch.tensor([pa + [0] * (len(pb) - len(pa)), pb]),
            page_table=torch.stack([b.identity_row(5), b.identity_row(9)]),
            kv_cache=b.kv_cache,
            prompt_lens=[len(pa), len(pb)],
            empty_slots=[5, 9],
            sampling_params=[GREEDY, GREEDY],
        )
        seq = {5: [int(first[0])], 9: [int(first[1])]}
        pos = {5: len(pa), 9: len(pb)}
        for step in range(12):
            act = [5] if (pause and 4 <= step < 8) else [5, 9]
            t = g.decode_forward(
                torch.full((len(act), 1), -1),
                torch.tensor([pos[s] for s in act]),
                page_table=None,
                kv_cache=b.kv_cache,
                slots=act,
                sampling_params=[GREEDY] * len(act),
            )
            for r, s in enumerate(act):
                seq[s].append(int(t[r]))
                pos[s] += 1
        return seq

    base, paused = low_level(False), low_level(True)
    ok = paused[9] == base[9][: len(paused[9])] and paused[5] == base[5]
    rows.append(
        record(
            dict(
                test=f"full_model/batch{B}_paused_slot",
                slots=[5, 9],
                uninterrupted=base,
                with_slot9_paused_4_steps=paused,
                status="pass" if ok else "fail",
            )
        )
    )
    # throughput: all B slots decoding together (greedy, token-out)
    g.reset()
    c0 = dict(g.counters)
    t1 = time.perf_counter()
    outs = g.generate_batch([g2048[:128]] * B, 33, stop_on_eos=False)
    dt = time.perf_counter() - t1
    rows.append(
        record(
            dict(
                test=f"full_model/batch{B}_throughput",
                batch=B,
                prompt=128,
                gen=33,
                wall_s=round(dt, 2),
                host_work=_delta(c0, dict(g.counters)),
                identical_rows=all(o == outs[0] for o in outs),
                status="pass" if all(o == outs[0] for o in outs) else "fail",
            ),
            "perf.jsonl",
        )
    )
    g.reset()
    a._batch_gen = g
    if a.batch_sweeps:
        sw = sweep_lengths()
        rows.append(tf_decode_sweep(g, sw, label=f"b{B}_slot0_identity", slot=0))
        rows.append(tf_decode_sweep(g, sw, label=f"b{B}_slot5_tight_pages", slot=5, pages="tight"))
    return rows


def phase_margins(a, be, gen):
    """Teacher-forcing mismatch analysis on the AIME24 reference: TT logits of every continuation position (one
    prefill of prompt + the HF continuation, all rows) vs the fp32 HF reference logits (goldens_gpu_fp32_aime24/
    logits/161.pt): for every position where TT's top-1 differs from HF's, both models' top-1 / runner-up and the
    margins (near-tie or real error). Also the free-running first divergence."""
    prompt, ref = _aime_prompt()
    lg = torch.load(WS / "goldens_gpu_fp32_aime24" / "logits" / "161.pt")
    cont = lg["continuation_logits"].float()  # [100, V]: cont[i] predicted ref[i]
    gen.reset()
    tt = gen.prefill_logits(prompt + ref[:-1]).float()[0, len(prompt) - 1 :]  # [100, V]
    gen.reset()
    tok = gen.tokenizer
    rows = []
    for i in range(len(ref)):
        t_tt, t_hf = int(tt[i].argmax()), int(cont[i].argmax())
        if t_tt == t_hf:
            continue
        tv, ti = torch.topk(tt[i], 3)
        hv, hi = torch.topk(cont[i], 3)
        rows.append(
            dict(
                step=i,
                hf_top1=t_hf,
                tt_top1=t_tt,
                hf_text=[tok.decode([int(x)]) for x in hi.tolist()],
                tt_text=[tok.decode([int(x)]) for x in ti.tolist()],
                hf_top3=hi.tolist(),
                hf_top3_logits=[round(float(x), 4) for x in hv],
                tt_top3=ti.tolist(),
                tt_top3_logits=[round(float(x), 4) for x in tv],
                hf_margin_hf_top1_minus_tt_top1=round(float(cont[i][t_hf] - cont[i][t_tt]), 4),
                tt_margin_tt_top1_minus_hf_top1=round(float(tt[i][t_tt] - tt[i][t_hf]), 4),
                hf_logit_std=round(float(cont[i].std()), 3),
            )
        )
    pcc_rows = [_pcc(tt[i], cont[i]) for i in range(len(ref))]
    return record(
        dict(
            test="full_model/tf_mismatch_margins",
            reference=str(aime_ref()),
            mismatches=rows,
            n_mismatch=len(rows),
            logits_pcc_min=round(min(pcc_rows), 6),
            logits_pcc_mean=round(sum(pcc_rows) / len(pcc_rows), 6),
            status="info",
        )
    )


def phase_pool(a, be, gen):
    """Serving-style caller-owned cache with a SMALL MLA page pool (vLLM block budget: num_blocks = 24 < 4 slots x 40
    blocks): two requests in fixed slots 1 and 3 with caller page-table rows of scattered physical blocks, prefilled
    through `prefill_forward(empty_slots=...)` (first token on device) and decoded through `decode_forward(slots=...)`;
    the same two requests again on different physical blocks must give identical tokens (paging through the bound page
    table, in place under the live traces). Needs its own KV cache: run after the other phases (re-allocates)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    prev = getattr(a, "_batch_gen", None) or gen
    prev.teardown()
    g = Glm53Generator(be, tokenizer=gen.tokenizer, max_batch=4, max_seq_len=a.max_seq_len)
    kv = g.allocate_kv_cache(max_batch=4, max_seq_len=a.max_seq_len, num_blocks=64)
    t0 = time.perf_counter()
    g.warmup(kv)
    warm_s = time.perf_counter() - t0
    b = g._binding(kv)
    prompt, _ = _aime_prompt()
    g2048 = torch.load(WS / "goldens_gpu_fp32" / "teacher_forcing" / "2048.pt")["prompt_ids"].reshape(-1).tolist()
    reqs = [prompt, g2048[:300]]
    steps = 16

    def run(blocks, remap_at=None):
        g.reset()
        pt = torch.full((2, b.blocks_per_seq), -1, dtype=torch.int64)
        for i, bl in enumerate(blocks):
            pt[i, : len(bl)] = torch.tensor(bl)
        width = max(len(r) for r in reqs)
        toks = torch.zeros(2, width, dtype=torch.int64)
        for i, r in enumerate(reqs):
            toks[i, : len(r)] = torch.tensor(r)
        first = g.prefill_forward(
            toks,
            page_table=pt,
            kv_cache=kv,
            prompt_lens=[len(r) for r in reqs],
            empty_slots=[1, 3],
            sampling_params=[GREEDY, GREEDY],
        )
        seq = [[int(first[0])], [int(first[1])]]
        pos = [len(r) for r in reqs]
        slots = [1, 3]
        for step in range(steps - 1):
            if remap_at is not None and step == remap_at:
                g.remap_slots(kv, {3: 2})  # vLLM slot compaction: the request in slot 3 moves to the free slot 2
                slots = [1, 2]
            t = g.decode_forward(
                torch.full((2, 1), -1),
                torch.tensor(pos),
                page_table=None,
                kv_cache=kv,
                slots=slots,
                sampling_params=[GREEDY, GREEDY],
            )
            for i in range(2):
                seq[i].append(int(t[i]))
                pos[i] += 1
        return seq

    c0 = dict(g.counters)
    A = run([[50, 3, 41], [7, 60, 15, 33, 9]])
    B = run([[5, 47, 2], [12, 63, 28, 23, 1]])
    work = _delta(c0, dict(g.counters))
    C = run([[50, 3, 41], [7, 60, 15, 33, 9]], remap_at=7)  # slot 3 -> 2 after 8 tokens
    g.reset()
    ok = A == B == C
    return record(
        dict(
            test="full_model/serving_pool_paging",
            num_blocks=64,
            max_batch=4,
            slots=[1, 3],
            prompt_lens=[len(r) for r in reqs],
            blocks_run_a=[[50, 3, 41], [7, 60, 15, 33, 9]],
            blocks_run_b=[[5, 47, 2], [12, 63, 28, 23, 1]],
            tokens_a=A,
            tokens_b=B,
            tokens_remap_slot3_to_2_after_8=C,
            remap_identical=C == A,
            warmup_s=round(warm_s, 1),
            host_work=work,
            trace_report=g.trace_report()["bindings"],
            status="pass" if ok else "fail",
        )
    )


def phase_rowcheck(a, be, gen):
    """Reviewer 2026-10-19 (item A reference): ONE prefill of the whole golden 2048 prompt (a single 2048-row chunk, no
    tail) with every row's logits vs the HF fp32 golden top-32 of that row: TT top-1's rank in the HF top-32 (-1 =
    outside), the TT rank and probability of the HF top-1, the rows where TT's pick is outside the HF top-32. Then the
    last rows of tail-chunk prefills prefill(n) (n = 95..98, 256, 577, 768) vs the same rows of the whole-prompt prefill
    (same path, different chunking): a chunking defect shows as a difference there, model-level numerics as the same
    divergence from HF in both."""
    ids, I, V = _golden(2048)
    n_all = 160 if a.mock else len(ids)
    gen.reset()
    t0 = time.perf_counter()
    L = gen.prefill_logits(ids[:n_all]).reshape(n_all, -1).float()
    wall = time.perf_counter() - t0
    gen.reset()
    lp = torch.log_softmax(L.double(), -1)
    top1 = L.argmax(-1)
    I32 = I[:n_all]
    ranks = [(I32[r] == int(top1[r])).nonzero().flatten().tolist() for r in range(n_all)]
    ranks = [x[0] if x else -1 for x in ranks]
    hf1 = I32[:, 0].long()
    tt_rank_hf1 = (L > L.gather(1, hf1.view(-1, 1))).sum(-1)
    p_hf1 = lp.gather(1, hf1.view(-1, 1)).exp().flatten()
    p_tt1 = lp.max(-1).values.exp()
    outside = [r for r in range(n_all) if ranks[r] < 0]
    focus = {}
    for r in [x for x in (93, 94, 95, 96, 97) if x < n_all] + outside[:40]:
        focus[r] = dict(
            tt_top5=torch.topk(L[r], 5).indices.tolist(),
            tt_top1_p=round(float(p_tt1[r]), 4),
            hf_top5=I32[r][:5].tolist(),
            hf_top1_tt_rank=int(tt_rank_hf1[r]),
            hf_top1_tt_p=round(float(p_hf1[r]), 4),
        )
    tails = []
    for n in [x for x in (95, 96, 97, 98, 256, 577, 768) if x <= n_all]:
        row = _prefill_rows(gen, 0, ids[:n], gen._own.identity_row(0).reshape(1, -1))
        gen.reset()
        lr = torch.log_softmax(row.double(), -1)
        full = lp[n - 1]
        tails.append(
            dict(
                n=n,
                tail=n % 64,
                top1=int(row.argmax()),
                hf_rank=(
                    I32[n - 1].tolist().index(int(row.argmax())) if int(row.argmax()) in I32[n - 1].tolist() else -1
                ),
                full_prefill_top1=int(top1[n - 1]),
                full_prefill_hf_rank=ranks[n - 1],
                hf_top5=I32[n - 1][:5].tolist(),
                kl_full_vs_tail=round(float((full.exp() * (full - lr)).sum()), 5),
                pcc=round(_pcc(row, L[n - 1]), 6),
            )
        )
    # decode-only path (fp32 KDA gate, decode programs; reviewer: prefill feeds prepare_chunk_recurrence a bf16 gate):
    # prefill ids[:1], then teacher-forced decode of ids[1:m] -> the prediction after ids[:j+1] at every row j < m
    m = 24 if a.mock else 160
    gen.reset()
    pt = gen._own.identity_row(0).reshape(1, -1)
    drows = [_prefill_rows(gen, 0, ids[:1], pt)]
    for j in range(1, m):
        drows.append(_decode_row(gen, ids[j], j, 0))
    gen.reset()
    D = torch.stack(drows)
    ld = torch.log_softmax(D.double(), -1)
    dtop = D.argmax(-1)
    d_ranks = [(I32[r] == int(dtop[r])).nonzero().flatten().tolist() for r in range(m)]
    d_ranks = [x[0] if x else -1 for x in d_ranks]
    decode_only = dict(
        rows=m,
        top1_eq_whole_prefill=sum(int(dtop[r]) == int(top1[r]) for r in range(m)) / m,
        tt_top1_eq_hf_top1=sum(r == 0 for r in d_ranks) / m,
        tt_top1_in_hf_top32=sum(r >= 0 for r in d_ranks) / m,
        prefill_rows_top1_eq_hf_top1=sum(r == 0 for r in ranks[:m]) / m,
        kl_prefill_vs_decode_max=round(float((lp[:m].exp() * (lp[:m] - ld)).sum(-1).max()), 4),
        focus={
            r: dict(
                decode_top5=torch.topk(D[r], 5).indices.tolist(),
                decode_top1_p=round(float(ld[r].max().exp()), 4),
                prefill_top1=int(top1[r]),
                hf_top5=I32[r][:5].tolist(),
                decode_hf_rank=d_ranks[r],
                prefill_hf_rank=ranks[r],
            )
            for r in range(m)
            if r in (93, 94, 95, 96, 97) or d_ranks[r] != ranks[r]
        },
    )
    nr = n_all
    return record(
        dict(
            test="full_model/whole_prompt_rows_vs_hf",
            decode_only=decode_only,
            rows=nr,
            prefill_s=round(wall, 2),
            tt_top1_eq_hf_top1=sum(r == 0 for r in ranks) / nr,
            tt_top1_in_hf_top5=sum(0 <= r < 5 for r in ranks) / nr,
            tt_top1_in_hf_top32=sum(r >= 0 for r in ranks) / nr,
            hf_top1_in_tt_top5=float((tt_rank_hf1 < 5).float().mean()),
            hf_top1_in_tt_top100=float((tt_rank_hf1 < 100).float().mean()),
            outside_hf_top32=outside,
            focus=focus,
            tail_vs_whole_prompt=tails,
            status="info",
        )
    )


def phase_layercurve(a, be, gen):
    """Where the TT residual departs from HF fp32 (reviewer 2026-10-19: per-layer bisection of the document-text
    divergence): ONE whole-prompt prefill of the golden 2048 prompt through the production path (`_run_prefill`, a
    single 2048-row chunk) with the output rows of the layers that have HF fp32 layer_io goldens (L0-L4, L7, L43, L44)
    and of their predecessors kept on host, vs the golden streams_out / streams_in: PCC over all rows, the row-PCC
    median / p1 / min and the rows of interest (96, 255, 767, 1408). Accumulated error (each layer sees TT's own input),
    so a smooth decay = the precision policy compounding, a step at one layer = that layer."""
    T = 2048
    ids, _, _ = _golden(T)
    n = 128 if a.mock else T
    gold = WS / "goldens_gpu_fp32" / "layer_io"
    want = sorted(int(d.name[1:]) for d in gold.iterdir() if d.name.startswith("L") and (d / str(T)).is_dir())
    keep_layers = set(want) | {w - 1 for w in want if w > 0}

    class Keep(dict):
        def setdefault(self, key, default=None):
            if key.endswith(".attn") or int(key[1:]) not in keep_layers:
                return []  # not kept (the unit's rows are still read: the production path is unchanged)
            return super().setdefault(key, default)

    keep = Keep()
    m = be.m
    gen.reset()
    orig = m._stage_prefill_chunk
    m._stage_prefill_chunk = lambda *x, **kw: orig(*x, keep=keep, **kw)
    t0 = time.perf_counter()
    try:
        be._run_prefill(
            torch.tensor(ids[:n]),
            slot=0,
            start=0,
            group=gen._own.kv_cache.group,
            eager=not m.prefill_captured,
            on_output=lambda *x: None,
        )
    finally:
        del m._stage_prefill_chunk
    wall = time.perf_counter() - t0
    gen.reset()

    def row_stats(tt, go):
        rp = torch.tensor([_pcc(tt[r], go[r]) for r in range(tt.shape[0])])
        rp = torch.nan_to_num(rp, nan=-1.0)
        focus = {r: round(float(rp[r]), 5) for r in (96, 255, 767, 1408) if r < tt.shape[0]}
        q = torch.quantile(rp, torch.tensor([0.01, 0.5]))
        return dict(
            pcc=round(_pcc(tt, go), 6),
            row_pcc_min=round(float(rp.min()), 5),
            row_pcc_p1=round(float(q[0]), 5),
            row_pcc_median=round(float(q[1]), 5),
            rows=focus,
            max_abs=round(float((tt - go).abs().max()), 3),
        )

    out = []
    for L in want:
        if f"L{L}" not in keep:  # a reduced (--units) plan
            continue
        go = torch.load(gold / f"L{L:02d}" / str(T) / "streams_out.pt").reshape(T, -1)[:n].float()
        tt = torch.cat(keep[f"L{L}"]).float()[:n]
        r = dict(layer=L, out=row_stats(tt, go))
        if L - 1 in keep_layers and f"L{L - 1}" in keep:
            gi = torch.load(gold / f"L{L:02d}" / str(T) / "streams_in.pt").reshape(T, -1)[:n].float()
            r["in"] = row_stats(torch.cat(keep[f"L{L - 1}"]).float()[:n], gi)
            # this layer's own contribution (residual delta) vs HF's on TT's vs HF's own inputs
            r["delta_pcc"] = round(_pcc(tt - torch.cat(keep[f"L{L - 1}"]).float()[:n], go - gi), 6)
        out.append(r)
        del go, tt
    return record(
        dict(test="full_model/layer_divergence_vs_hf", rows=n, prefill_s=round(wall, 2), layers=out, status="info")
    )


# lengths around every prefill-chunk / page / KDA-tail boundary (tail = n % 64: 31 / 32 / 33 and 63 / 0 / 1 rows)
BOUNDARY = [31, 32, 33, 63, 64, 65, 95, 96, 97, 127, 128, 129, 159, 160, 161, 191, 192, 193, 255, 256, 257]
BOUNDARY += [383, 384, 385, 511, 512, 513, 1023, 1024, 1025]


def _b32_gen(a, be, gen):
    """The max_batch-32 generator of this process (built once; the B=1 generator is torn down first)."""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

    g = getattr(a, "_batch_gen", None)
    if g is None or g.max_batch != 32:
        (g or gen).teardown()
        g = Glm53Generator(be, tokenizer=gen.tokenizer, max_batch=32, max_seq_len=a.max_seq_len)
        g.warmup(capture=TRACE)
        a._batch_gen = g
    g.reset()
    return g


def _decode_row(gen, tok, pos, slot):
    """One teacher-forced logits-out decode step of `slot` (token `tok` at position `pos`)."""
    return (
        gen.decode_forward(
            torch.tensor([[tok]]),
            torch.tensor([pos]),
            page_table=None,
            kv_cache=gen._own.kv_cache,
            slots=[slot],
            enable_trace=TRACE,
        )
        .reshape(-1)
        .float()
    )


def _greedy_chain(gen, ids, slot, pt, n_dec):
    """prefill ids with on-device greedy sampling + n_dec token-out decode steps (token feedback): the tokens."""
    first = gen.prefill_forward(
        torch.tensor([ids]),
        page_table=pt,
        kv_cache=gen._own.kv_cache,
        prompt_lens=[len(ids)],
        empty_slots=[slot],
        sampling_params=[GREEDY],
    )
    toks = [int(first[0])]
    for k in range(n_dec):
        t = gen.decode_forward(
            torch.tensor([[-1]]),
            torch.tensor([len(ids) + k]),
            page_table=None,
            kv_cache=gen._own.kv_cache,
            slots=[slot],
            sampling_params=[GREEDY],
            enable_trace=TRACE,
        )
        toks.append(int(t[0]))
    return toks


def _two_step_rows(gen, label, slot, mode, lens):
    """Program 2026-10-19 03:15Z (A): decode-after-prefill vs the prefill path for TWO decode steps, per n:
    prefill golden[:n], decode golden token n at n (step 1) and golden token n+1 at n+1 (step 2), vs fresh prefills
    of golden[:n+1] / golden[:n+2] (same tokens, same slot / pages). Per step: both top-1, the prefill top-1's
    probability, KL(prefill || decode), the logprob gap of the prefill top-1 and golden top-5 membership. A row is
    `suspect` when a step's top-1 disagrees with a confident prefill path (p >= 0.5) or KL > 0.3 (a near-tie flip
    is not)."""
    ids, I, _ = _golden(2048)
    rows = []
    for n in lens:
        pt = _pages(gen, slot, n + 2, mode)
        gen.reset()
        _prefill_rows(gen, slot, ids[:n], pt)
        d1 = _decode_row(gen, ids[n], n, slot)
        d2 = _decode_row(gen, ids[n + 1], n + 1, slot)
        gen.reset()
        f1 = _prefill_rows(gen, slot, ids[: n + 1], pt)
        gen.reset()
        f2 = _prefill_rows(gen, slot, ids[: n + 2], pt)
        r = dict(n=n, tail=n % 64)
        sus = worse = False
        for k, (d, f) in enumerate(((d1, f1), (d2, f2)), 1):
            lf, ld = torch.log_softmax(f.double(), -1), torch.log_softmax(d.double(), -1)
            top, dt = int(f.argmax()), int(d.argmax())
            p_top = float(lf[top].exp())
            kl = float((lf.exp() * (lf - ld)).sum())
            g5 = I[n + k - 1][:5].tolist()
            g32 = I[n + k - 1].tolist()
            r[f"step{k}"] = dict(
                decode_hf_rank=g32.index(dt) if dt in g32 else -1,  # rank in the HF fp32 top-32 (-1: outside)
                prefill_hf_rank=g32.index(top) if top in g32 else -1,
                hf_top1=g32[0],
                decode_top1=dt,
                prefill_top1=top,
                agree=dt == top,
                prefill_top1_p=round(p_top, 4),
                decode_top1_p=round(float(ld[dt].exp()), 4),
                kl_prefill_decode=round(kl, 5),
                logprob_gap_prefill_top1=round(float(lf[top] - ld[top]), 4),
                decode_in_golden_top5=dt in g5,
                prefill_in_golden_top5=top in g5,
            )
            sus |= (dt != top and p_top >= 0.5) or kl > 0.3
            dr, pr = r[f"step{k}"]["decode_hf_rank"], r[f"step{k}"]["prefill_hf_rank"]
            worse |= (dr < 0 <= pr) or (dr >= 0 and pr >= 0 and dr > pr + 4)
        r["suspect"] = sus  # decode departs from a confident TT prefill path (NOT a correctness verdict: see vs_hf)
        r["decode_worse_vs_hf"] = worse
        rows.append(r)
    gen.reset()
    bad = [r["n"] for r in rows if r["suspect"]]
    worse = [r["n"] for r in rows if r["decode_worse_vs_hf"]]
    return record(
        dict(
            test=f"full_model/decode_after_prefill_2step/{label}",
            slot=slot,
            pages=mode,
            lengths=len(rows),
            step1_top1_agree=sum(r["step1"]["agree"] for r in rows),
            step2_top1_agree=sum(r["step2"]["agree"] for r in rows),
            suspect=bad,
            decode_worse_vs_hf=worse,  # decode pick outside the HF top-32 while the prefill pick is inside, or > 4 ranks worse
            rows=rows,
            status="pass" if not worse else "fail",
        )
    )


def phase_gap2(a, be, gen):
    """Two-step decode-after-prefill vs prefill path at every boundary length, B=1 (slot 0, identity pages)."""
    return _two_step_rows(gen, "b1_slot0_identity", 0, "identity", [95, 97] if a.mock else BOUNDARY)


def phase_gap2_b32(a, be, gen):
    """The same at max_batch 32 (32-wide masked decode trace): slot 0 identity pages and slot 5 vLLM-like tight pages."""
    g = _b32_gen(a, be, gen)
    lens = [95, 97] if a.mock else BOUNDARY
    return [
        _two_step_rows(g, "b32_slot0_identity", 0, "identity", lens),
        _two_step_rows(g, "b32_slot5_tight_pages", 5, "tight", lens),
    ]


def _state_dump(be, gen, slot, n_rows):
    """Host copies of one slot's per-layer request state on every chip: KDA conv rows C [3, 24576] (t-3..t-1) and
    recurrent S [64, 128, 128]; MLA latent rows 0..n_rows-1 through the slot's page-table row."""
    kv = gen._own.kv_cache
    be.m.synchronize()
    out = {}
    for st in be.m.stages:
        g = st.groups[kv.group]
        for l in st.state_layers:
            dec, c = st.decoders[l], g.caches[l]
            if dec.is_kda:
                out[l] = ("kda", ttnn.to_torch(c.kda.C)[0, :, slot, :].float(), ttnn.to_torch(c.kda.S)[slot].float())
            else:
                lat = ttnn.to_torch(c.latent).float()
                row = c.page_table_host[slot]
                nb = -(-n_rows // 64)
                out[l] = ("mla", torch.cat([lat[int(b), 0] for b in row[:nb]])[:n_rows])
    return out


def _rel(x, y):
    return float((x - y).norm() / max(float(y.norm()), 1e-30))


def _state_cmp(x, y):
    """Per layer, test path x vs reference path y. KDA: relative error of each conv row (t-3, t-2, t-1), of the
    one-row-shifted pairs (x[r] vs y[r+1]: x lags y by a token; x[r+1] vs y[r]: x leads) and of S. MLA: relative
    error of the last 4 latent rows and the worst earlier row."""
    out = []
    for l in sorted(y):
        if y[l][0] == "kda":
            (_, cx, sx), (_, cy, sy) = x[l], y[l]
            out.append(
                dict(
                    layer=l,
                    kind="kda",
                    conv_rel=[round(_rel(cx[r], cy[r]), 5) for r in range(3)],
                    conv_lag_rel=[round(_rel(cx[r], cy[r + 1]), 5) for r in range(2)],
                    conv_lead_rel=[round(_rel(cx[r + 1], cy[r]), 5) for r in range(2)],
                    S_rel=round(_rel(sx, sy), 6),
                )
            )
        else:
            lx, ly = x[l][1], y[l][1]
            per = [_rel(lx[i], ly[i]) for i in range(ly.shape[0])]
            out.append(
                dict(
                    layer=l,
                    kind="mla",
                    last4_rel=[round(v, 5) for v in per[-4:]],
                    worst_earlier_rel=round(max(per[:-4], default=0.0), 5),
                    worst_row=int(torch.tensor(per).argmax()),
                )
            )
    return out


def phase_statecheck(a, be, gen):
    """Per-layer request state after the same tokens through the decode path vs the prefill path (B=1, slot 0), on the
    greedy chain's own context seq = golden[:n] + the 2 greedy tokens (so the token-out paths and the teacher-forced
    paths feed identical tokens; n = 97 = the item-A context golden[:97] + [547]): D1 = prefill(n) + 1 decode vs
    P1 = prefill(n+1); D2 = prefill(n) + 2 decodes and E2 = prefill(n+1) + 1 decode vs P2 = prefill(n+2) (teacher-forced,
    logits-out); G1 / G2 = the token-out path (on-device greedy first token + 1 / 2 token-feedback decode steps) vs
    D1 / D2 (a difference = a token-feedback / sampler-trace defect). n = 128 (aligned, no tail) is the baseline of the
    decode-vs-prefill state differences. Layer 0's input is the embedding in every path, so its KDA conv rows / S differ
    only by the prefill-vs-decode arithmetic unless the state carry is wrong.
    Then (reviewer 2026-10-19): the KDA rows of a PARTIAL 32-row chunk past actual_end must not reach S: the state after
    golden[:95] built as prefill(64) + 31 teacher-forced decode steps vs prefill(95) (64 + a 31-of-32-row KDA tail), with
    prefill(64) + 32 decodes vs prefill(96) (64 + an exact 32-row tail) as the no-padding baseline."""
    ids, I, _ = _golden(2048)
    pt = gen._own.identity_row(0).reshape(1, -1)
    rows = []
    for n in (95,) if a.mock else (95, 97, 128):
        logits = {}
        gen.reset()
        greedy = _greedy_chain(gen, ids[:n], 0, pt, 2)
        seq = ids[:n] + greedy[:2]

        def run(name, n_pre, n_dec):
            gen.reset()
            row = _prefill_rows(gen, 0, seq[:n_pre], pt)
            for k in range(n_dec):
                row = _decode_row(gen, seq[n_pre + k], n_pre + k, 0)
            logits[name] = int(row.argmax())
            return _state_dump(be, gen, 0, n_pre + n_dec)

        def run_greedy(name, n_dec):
            gen.reset()
            logits[name] = _greedy_chain(gen, seq[:n], 0, pt, n_dec)
            return _state_dump(be, gen, 0, n + n_dec)

        p1, d1 = run("P1", n + 1, 0), run("D1", n, 1)
        c1 = _state_cmp(d1, p1)
        del p1
        g1 = run_greedy("G1", 1)
        cg1 = _state_cmp(g1, d1)
        del d1, g1
        p2, d2, e2 = run("P2", n + 2, 0), run("D2", n, 2), run("E2", n + 1, 1)
        c2, ce = _state_cmp(d2, p2), _state_cmp(e2, p2)
        del p2, e2
        g2 = run_greedy("G2", 2)
        cg2 = _state_cmp(g2, d2)
        del d2, g2
        rows.append(
            record(
                dict(
                    test=f"full_model/state_decode_vs_prefill/{n}",
                    n=n,
                    tail=n % 64,
                    greedy_chain=greedy,
                    context_tokens=seq[n:],
                    context_is_golden=seq[n:] == ids[n : n + 2],
                    golden_next=ids[n : n + 3],
                    hf_top5_rows=dict(n=I[n][:5].tolist(), n1=I[n + 1][:5].tolist()),  # valid if context_is_golden
                    next_top1=logits,
                    d1_vs_p1=c1,
                    d2_vs_p2=c2,
                    e2_vs_p2=ce,
                    g1_vs_d1=cg1,
                    g2_vs_d2=cg2,
                    status="info",
                )
            )
        )
    leak = {}
    for n in (95, 96):
        gen.reset()
        _prefill_rows(gen, 0, ids[:64], pt)
        for j in range(64, n):
            _decode_row(gen, ids[j], j, 0)
        x = _state_dump(be, gen, 0, n)
        gen.reset()
        _prefill_rows(gen, 0, ids[:n], pt)
        y = _state_dump(be, gen, 0, n)
        leak[n] = _state_cmp(x, y)
        del x, y
    gen.reset()
    rows.append(
        record(
            dict(
                test="full_model/state_partial_chunk_exclusion",
                decode_built_vs_prefill_95=leak[95],
                decode_built_vs_prefill_96_baseline=leak[96],
                status="info",
            )
        )
    )
    return rows


def phase_perf32(a, be, gen):
    """Program 2026-10-19 03:15Z (B): a lone request on the max_batch-32 generator (32-wide masked decode trace,
    31 inactive rows) — token-out ms/step for prompts of different content (Stage 9 measured 125.8 ms AIME24 vs
    164.6 ms random / chat on the served trace before inactive rows were kept out of the MoE expert union)."""
    steps, reps = (2, 1) if a.mock else (64, 2)
    g = _b32_gen(a, be, gen)
    b = g._own
    gold = torch.load(WS / "goldens_gpu_fp32" / "teacher_forcing" / "128.pt")["prompt_ids"].reshape(-1).tolist()
    rnd = torch.randint(0, 150000, (128,), generator=torch.Generator().manual_seed(1234)).tolist()
    prompts = {"aime24_chat": _aime_prompt()[0], "golden128": gold, "random128": rnd}
    out = {}
    for name, prompt in prompts.items():
        per = []
        for r in range(reps + 1):
            g.reset()
            g.prefill_forward(
                torch.tensor([prompt]),
                page_table=b.identity_row(0).reshape(1, -1),
                kv_cache=b.kv_cache,
                prompt_lens=[len(prompt)],
                empty_slots=[0],
                sampling_params=[GREEDY],
            )
            t0 = time.perf_counter()
            for i in range(steps):
                g.decode_forward(
                    torch.tensor([[-1]]),
                    torch.tensor([len(prompt) + i]),
                    page_table=None,
                    kv_cache=b.kv_cache,
                    slots=[0],
                    sampling_params=[GREEDY],
                    enable_trace=TRACE,
                )
            if r:
                per.append((time.perf_counter() - t0) / steps)
        out[name] = dict(
            prompt_tokens=len(prompt),
            token_out_ms=[round(x * 1e3, 3) for x in per],
            token_out_ms_median=round(sorted(per)[len(per) // 2] * 1e3, 3),
        )
    g.reset()
    return record(
        dict(
            test="full_model/perf_b32_lone_request",
            steps=steps,
            prompts=out,
            stage9_before_fix_ms={"aime24": 125.8, "random_chat": 164.6, "one_slot_trace": 75.5},
            tracking=TRACKING,
            status="info",
        ),
        "perf.jsonl",
    )


# minutes per phase on the 8-chip mesh (runs 4 / 5; --deadline skips a phase that would not finish in time)
EST_MIN = dict(nonaligned=8, autocheck=7, batch=5, gap2=4, gap2_b32=8, statecheck=4, pool=4, gap=6, perf32=5, qual=2)
EST_MIN.update(perf=2, tf=2, tf32=3, ar=1, rowcheck=4, layercurve=4)

PHASES = {
    "layercurve": phase_layercurve,
    "rowcheck": phase_rowcheck,
    "gap2": phase_gap2,
    "gap2_b32": phase_gap2_b32,
    "statecheck": phase_statecheck,
    "perf32": phase_perf32,
    "tf32": phase_tf32,
    "autocheck": phase_autocheck,
    "gap": phase_gap,
    "pool": phase_pool,
    "margins": phase_margins,
    "batch": phase_batch,
    "fallback": phase_fallback,
    "tracy": phase_tracy,
    "ar": phase_ar,
    "qual": phase_qual,
    "perf": phase_perf,
    "determinism": phase_determinism,
    "mvp": phase_mvp,
    "tf": phase_tf,
    "prefill_check": phase_prefill_check,
    "probe": phase_probe,
    "trace": phase_trace,
    "nonaligned": phase_nonaligned,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default="2x4", choices=["2x4", "2x2", "1x2", "1x1"])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--units", default="")
    ap.add_argument("--partition", default=str(WS / "doc/ttnn/04-multichip-decoder/partition.json"))
    ap.add_argument("--partition-key", default="pipeline8_ring_wrap")
    ap.add_argument("--handoff", default="eth_direct", choices=["eth_direct", "eth_fifo", "host"])
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=2051)
    ap.add_argument("--trace-region", type=int, default=800_000_000)
    ap.add_argument("--phases", default="mvp,tf")
    ap.add_argument("--mvp-file", default="")
    ap.add_argument("--eager", action="store_true", help="no trace capture (mock: captures are not supported)")
    ap.add_argument("--steps", type=int, default=AIME_STEPS)
    ap.add_argument("--qual-long", type=int, default=0)
    ap.add_argument("--tracy-steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument(
        "--tf-goldens", default="", help="extra teacher-forcing references: fp32 golden prompts, e.g. 128,512"
    )
    ap.add_argument("--nonaligned", default="1,31,63,65,127,129,1000,2047,2048,2049,2051")
    ap.add_argument(
        "--no-batch-sweeps", dest="batch_sweeps", action="store_false", help="batch: skip the 1-step sweeps"
    )
    ap.add_argument(
        "--deadline", default="", help="UTC end of the mesh slot (YYYY-MM-DDTHH:MM:SSZ): later phases are skipped"
    )
    a = ap.parse_args()
    global GEN_MAX_BATCH, TRACE
    GEN_MAX_BATCH = a.max_batch
    TRACE = not a.eager
    # one-shot extra phases for an already-queued run (a chained launcher's command line cannot change): the next run
    # started consumes $GLM53_WORK/EXTRA_PHASES (comma list) and renames it (recorded in full_model/start)
    extra_f = Path(os.environ.get("GLM53_WORK", "/opt/glm53/work-s6")) / "EXTRA_PHASES"
    a.extra_phases = ""
    if not a.mock and extra_f.exists():
        a.extra_phases = extra_f.read_text().strip()
        extra_f.rename(extra_f.with_name(f"EXTRA_PHASES.consumed.{RUN_ID}"))
        a.phases = ",".join(p for p in (a.phases, a.extra_phases) if p)
    if a.mock:
        a.mesh, a.handoff = "1x1", "host"
    record(dict(test="full_model/start", argv=sys.argv, phases=a.phases, extra_phases=a.extra_phases, status="info"))
    mesh = be = None
    rc = 0
    try:
        mesh, subs = open_mesh(a)
        be, gen = build(a, mesh, subs)
        warmup(be, gen, capture=not a.eager)
        end = None
        if a.deadline:
            from datetime import datetime

            end = datetime.strptime(a.deadline, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        for ph in [p for p in a.phases.split(",") if p]:
            if end is not None and time.time() + 60 * (EST_MIN.get(ph, 1) + 2) > end:  # + 2 min: release + close
                record(dict(test=f"full_model/phase/{ph}", status="skipped", reason=f"deadline {a.deadline}"))
                continue
            t0 = time.perf_counter()
            try:
                PHASES[ph](a, be, gen)
            except Exception as e:  # noqa: BLE001
                rc = 1
                record(
                    dict(
                        test=f"full_model/phase/{ph}",
                        status="error",
                        error=repr(e)[:2000],
                        traceback=traceback.format_exc()[-6000:],
                        wall_s=round(time.perf_counter() - t0, 1),
                    )
                )
                traceback.print_exc()
            if be is not None:
                try:
                    be.m.synchronize()
                except Exception:  # noqa: BLE001
                    traceback.print_exc()
        record(
            dict(
                test="full_model/trace_report",
                report=gen.trace_report(),
                backend_counters=dict(be.counters),
                status="info",
            )
        )
    except Exception as e:  # noqa: BLE001
        rc = 2
        record(
            dict(
                test="full_model/fatal", status="error", error=repr(e)[:2000], traceback=traceback.format_exc()[-6000:]
            )
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
    record(dict(test="full_model/end", rc=rc, status="info"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
