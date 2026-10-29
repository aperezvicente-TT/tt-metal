# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Job (serve.py): keep the server OPEN after a hung unit until the live triage has read the device (program
2026-10-23 21:39:24Z: the read must happen within the first minute of the timeout, before the close path moves the
cores). No device work. The trigger (work-s14/triage_trigger14.sh) touches HANG_<stg> when the run log shows the device
timeout and TRIAGE_DONE_<stg> when its reads are written; without HANG_<stg> after 8 s this job returns at once.
ARGS: stg, max_s (300)."""
import os
import time

W = "/opt/glm53/work-s14"
stg = str(ARGS["stg"])  # noqa: F821
max_s = float(ARGS.get("max_s", 300))  # noqa: F821
t0 = time.time()
time.sleep(8)
held = False
if os.path.exists(f"{W}/HANG_{stg}"):
    held = True
    while not os.path.exists(f"{W}/TRIAGE_DONE_{stg}") and time.time() - t0 < max_s:
        time.sleep(2)
RESULT = dict(test="hold the server open for the live triage", stg=stg, held=held, wall_s=round(time.time() - t0, 1),
              triage_done=os.path.exists(f"{W}/TRIAGE_DONE_{stg}"))
print(f"HOLD {RESULT}", flush=True)
