# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 16 jobs inside Stage 14b's SHARED MESH SERVICE (q_mesh, 14b #120): attach the batched TP prefill to the
resident serving backend CTX[('m5_serving', B, ids, ndev, native, rev)] and restore it afterwards.

attach(ctx, knobs, buckets) releases the generator's traces, sets GLM53_STACK_PREFILL=tp + the Stage 16 knobs, clears
the binding's preparation flags and runs the generator's own warm-up again (prepare_prefill: TPPrefill + one eager
forward per bucket BEFORE any capture; prepare_decode; capture_decode: the decode trace, then every prefill trace), so
the prefill's persistent buffers precede every capture. restore() releases the traces, drops the TPPrefill, restores
the environment and re-runs the decode-only warm-up. The CALLER must drop its own references to the TPPrefill
(and its layers / EpMoe objects) before restore(), else their device buffers stay allocated. The attach-time eager
forwards overwrite slot 0's KDA state / latent pages and the head rows: run only on an idle service."""
import gc
import os
import time

# every knob a service template sets is pinned here at its code default: attach() writes the job's knobs into
# os.environ, so a knob one job set and the next omits would otherwise leak into the resident server's later jobs
DEFAULT_KNOBS = {"GLM53_PF_MOE": "ep", "GLM53_PF_MOE_V2": "1", "GLM53_PF_MHC": "comp", "GLM53_PF_MHC_POST": "fused",
                 "GLM53_PF_DSA": "0", "GLM53_PF_MLA_FORM": "absorbed", "GLM53_PF_MHA_SDPA": "256,128,HiFi2",
                 "GLM53_PF_KDA_SCAN": "op", "GLM53_PF_MM_NQ": "4", "GLM53_PF_MM_KB": "640", "GLM53_PF_MM_FID": "",
                 "GLM53_PF_MOE_FID": "", "GLM53_PF_SDPA": "", "GLM53_PF_SPARSE_ATTN": "gather",
                 "GLM53_PF_SPARSE_RESHARD": "pad", "GLM53_PF_NOHEAD": "0",
                 # batched admission in the service program: 4 fresh requests per replay (program 2026-10-22 03:34Z; TP8
                 # fit 04:28Z: 4 x 2,016 tokens 1,619 ms vs 2,056 serial, first tokens = serial)
                 "GLM53_PF_BATCH": "4"}  # fmt: skip


class Attached:
    def __init__(self, ctx, knobs=None, buckets="128,512,2048"):
        keys = [k for k in ctx if isinstance(k, tuple) and k and k[0] == "m5_serving"]
        assert keys, "no resident serving backend in CTX (('m5_serving', ...))"
        self.key = keys[0]
        self.be, self.gen = ctx[self.key]["backend"], ctx[self.key]["gen"]
        self.binding = self.gen._own_binding()
        self.knobs = dict(DEFAULT_KNOBS)
        self.knobs.update({str(k): str(v) for k, v in dict(knobs or {}).items()})
        self.buckets = buckets if isinstance(buckets, str) else ",".join(str(int(b)) for b in sorted(set(buckets)))
        env_keys = ["GLM53_STACK_PREFILL", "GLM53_PREFILL_BUCKETS"] + list(self.knobs)
        self.saved_env = {k: os.environ.get(k) for k in env_keys}
        self.warm_s = None

    def _rewarm_release(self):
        self.gen.teardown()
        tpp = getattr(self.be, "tpp", None)
        if tpp is not None:
            tpp.release()
            self.be.tpp = None

    def _set_env(self, tp: bool):
        if tp:
            os.environ.update(self.knobs)
            os.environ["GLM53_STACK_PREFILL"] = "tp"
            os.environ["GLM53_PREFILL_BUCKETS"] = self.buckets
        else:
            for k, v in self.saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _rewarm(self, tp: bool) -> float:
        self._rewarm_release()
        self._set_env(tp)
        self.binding.prepared_prefill = self.binding.prepared_decode = False
        gc.collect()
        t0 = time.time()
        self.gen.warmup()
        return round(time.time() - t0, 1)

    def attach(self):
        import importlib

        import models.autoports.zai_org_glm_5_3_flash.tt.prefill_tp as PF

        self._rewarm_release()
        self._set_env(True)
        importlib.reload(PF)  # knobs read at import (KDA_GROUPS, PF_MOE_CB, ...) follow this job's environment
        self.warm_s = self._rewarm(True)
        assert self.be.tpp is not None, "GLM53_STACK_PREFILL=tp did not attach a TPPrefill"
        return self.be.tpp

    def restore(self) -> float:
        return self._rewarm(False)
