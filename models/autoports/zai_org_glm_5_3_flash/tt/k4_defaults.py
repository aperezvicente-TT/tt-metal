# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""k4 knob DEFAULTS (Stage 16's k4 delivery; the K4 composition ruling 2026-10-27 02:14:40Z).

The PMK merges whose TP8 rows are bitwise and not slower ride k4 DEFAULT ON behind their own knobs. Each entry below
is applied as os.environ.setdefault(name, value) when this module is imported (by the central construction modules,
right after `import os`), so an explicit environment or job setting still wins and the knob still turns it off.
ON (k4b, program RULING 2026-10-27 04:21:04Z: M1 + AD-1A-lite in option B):
    GLM53_PMK_DSA_AD2=dense             AD-2a-dense, the chunk-0 PMK attention core (ad2tp8 PASS); fresh chunks of
                                        c == 2,048 only (prefill_tp.AD2_ROWS; the 512 bucket keeps the op SDPA,
                                        program RULING 02:51:28Z)
    GLM53_PMK_DSA_AD1=lite              AD-1-lite (ad1tp8 PASS, bitwise, time-neutral), now IN PLACE: PMK_AD1_INPLACE
                                        (code default 1) streams the op's resident q_a | kv_a (bf8 DS) and wk | gate
                                        (idx.w_kg) -- no PfStream weight copies (k4's +27.5 MiB / bank, RULING 03:46:31Z)
    GLM53_PMK_P16=1, PMK_B_P16=1        M1, the mHC post P16 in W7 / G (TP8 H10 fixes e9d8144adfa + 120ba370797)
ON (k4d, program RULING 2026-10-27 09:29:42Z: the Galaxy ad2dfid TP8 row):
    GLM53_PMK_ATT_FID_QK=HiFi2,         the AD-2a-dense chunk-0 core's two bf16 products at HiFi2: PMK vs the op SDPA
    GLM53_PMK_ATT_FID_PV=HiFi2          0.99989-0.99996 per chip (bar 0.9998), vs the host fp32 0.9999986 (op
                                        0.99995), 1,362.2 vs the op 1,805.8 us (-24.6 pct; HiFi4 -18.7); the only
                                        PmkAttention user in k4d is prefill_tp._ad2_dense
ON (k4d3, program RULING 2026-10-27 06:50:59Z: 'default ON once the Galaxy k4c rows pass' -- they passed 08:44Z /
    09:00Z, host-8chip k4cg 10:18Z; the flip never landed in k4d / k4d2):
    PMK_RS8_QSLICE=4                    the column-sliced TP8 RS8 (16's a2cfcac3a63; rs8qtp8: Q=4 BITWISE vs Q=0 on every
                                        output of the L4 A15+B16 layer, fused chain 8,447.6 -> 8,235.1 us, -2.5 pct =
                                        ~-6.6 ms per 2,048 chunk over 31 KDA layers). Process-wide: it reaches every
                                        TP8 program A that builds the P8 RS8 -- the W7 sparse_kda layers (rs8qtp8's
                                        form) AND the dense_kda layers' program A (built outside GLM53_PMK_W7_ENV's
                                        scope; the same class and kernels, no device row of its own yet: the k4d3
                                        window's ids bitwise vs k4dwin covers it). G builds no P8 RS8; P15's rs8_15
                                        keeps Q = 1. Not in tp_image.env_signature (a non-GLM53_ knob): restores are
                                        decided by the allocation replay
ON (k4d3, program RULING 2026-10-27 13:49:02Z (2), replacing the knob-only k4da2a window):
    PMK_SPA_A2A_FS=1                    the SPA q / o all-to-all as FabricStreamA2A2D (16's dbcba7d396e / 848343a96d6;
                                        the Galaxy shfs rows: BITWISE 16 / 16 chunks, the sparse 32K key 15,878 -> 15,091
                                        us and 15,877 -> 14,989 us = -787 / -889 us per DSA layer-chunk). Applies only
                                        where MlaPrefill reshards (GLM53_DSA_SHARD) on a 2x4 or pair mesh; else the op
ON (k4d8s, the spaa0rb FOLD: program [session 10] 2026-10-28 00:37:39Z; 16's pmk/merge/spaa0b_d0 + spaa0r_glx):
    GLM53_SPA_WALKERS=2                 SpaUnion: pass 0 (and on the 2x4 the relay pass 1) on the 64 members' BRISC
                                        (spa_mw.cpp), the lane only sends (A0 / A0r)
    GLM53_SPA_A2B=1                     the lane NCRISC as a 2nd sender on fabric link 1 (A2b)
    GLM53_SPA_WALK_HDRB=1               the 64 member headers in one batch
                                        bitwise in every arm (d0 pair + Galaxy 2x4); gather -48 % (pair) / -19.5 % (2x4)
ON (k4d8a, the program-A fold: program [session 10] 2026-10-28 05:11:24Z / 05:11:39Z; 16's k4d8acells Galaxy cell
rows103, the served W7 of L4 at 2,048 rows, TP8, det sha identical in every arm):
    GLM53_PMK_KDA_CARRY_SPREAD=1        L4: P9's conv-carry patch spread over the scan cores (-94 us per KDA layer, bitwise)
    GLM53_FS_C1X_ROWFIRST=1             L1: the C1X x-hat all-gather row first (-36 us, bitwise)
    (GLM53_PMK_A_ZSKIP, L3, stays OFF: bitwise, no gain -- struck 04:13:24Z)
ON (the walkbound default: program [session 10] 2026-10-28 04:28:04Z "a bounded spin that reports beats a silent hang"):
    GLM53_SPA_WALK_BOUND=2700000000     the walker path's waits bounded at ~2 s (1.35 GHz; spa_bound.hpp), REPORT mode:
    GLM53_SPA_WALK_BOUND_MODE=report    an expiry records (row lane sync words 16..23) + counts (word 29) and the wait
                                        resumes (results unchanged); off by itself where it cannot apply (walkers off,
                                        GLM53_SPA_STAMPS on); "bail" = the unit form (16's walkbound d0 rows)
OFF (code in the branch, knob default off):
    GLM53_PMK_DSA_AD1=full / full+qidx  AD-1A / AD-3a (ad3tp8 bitwise but +2.6 % L3 at chunk 0: D2's norm-weight reads)
    GLM53_PMK_PFS_EPOCH=1               the PfStream epoch handshake (stays off until 14's review rules on)
    GLM53_PF_ROW_FENCE=mcast            the F1' row fence (stays off until 14's review rules on)
    GLM53_GRID_X / GLM53_GRID_Y         the grid pin (a Galaxy env; unset by default)
"""

import os

K4_DEFAULTS: dict = {"GLM53_PMK_DSA_AD2": "dense", "GLM53_PMK_DSA_AD1": "lite", "GLM53_PMK_P16": "1", "PMK_B_P16": "1",
                     "GLM53_PMK_ATT_FID_QK": "HiFi2", "GLM53_PMK_ATT_FID_PV": "HiFi2", "PMK_RS8_QSLICE": "4",
                     "PMK_SPA_A2A_FS": "1",
                     "GLM53_SPA_WALKERS": "2", "GLM53_SPA_A2B": "1", "GLM53_SPA_WALK_HDRB": "1",
                     "GLM53_PMK_KDA_CARRY_SPREAD": "1", "GLM53_FS_C1X_ROWFIRST": "1",
                     "GLM53_SPA_WALK_BOUND": "2700000000", "GLM53_SPA_WALK_BOUND_MODE": "report"}

# the keys whose value comes from this module (unset in the environment at import): consumers that apply a default only
# where it can act tell a DEFAULTED value from an explicit one (from_default)
K4_FROM_DEFAULT: frozenset = frozenset(_k for _k in K4_DEFAULTS if _k not in os.environ)

for _k, _v in K4_DEFAULTS.items():
    os.environ.setdefault(_k, _v)


def from_default(name: str) -> bool:
    """True when `name` still holds the K4 default this module applied (unset at import, unchanged since)"""
    return name in K4_FROM_DEFAULT and os.environ.get(name) == K4_DEFAULTS.get(name)
