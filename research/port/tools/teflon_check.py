#!/usr/bin/env python3
"""teflon_check.py: run a model through a Teflon delegate N times and check EVERY invoke's output (2026-11-13, port
step 6b: shader-core transposes inside the ML command stream).

Checks: each invoke's output against the first (a race between the NN job and a compute grid would make them
differ), against the CPU reference (.npz with "input"/"output", as npu-upscale/npu_detail_test.py), and optionally
byte-for-byte against a saved output (--compare, e.g. the same model with ETNA_ML_SHADER unset: the shader transposes
are exact permutations, so the bytes must be identical).

Only run on the NPU through ~/alta-npu/scripts/npu-attempt, with Jordan's OK for that job. Without -e it runs on the
CPU. Step markers go to $NPU_STEPS_FILE like npu_detail_test.py.
"""
import argparse
import json
import os
import statistics
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-r", "--reference", required=True)
ap.add_argument("-e", "--delegate", help="path to libteflon.so (omit for a CPU run)")
ap.add_argument("-n", "--iterations", type=int, default=10)
ap.add_argument("--save", help="save the first invoke's output to this .npy")
ap.add_argument("--compare", help="a saved .npy the output must equal byte for byte (see --tolerance)")
ap.add_argument("--tolerance", type=int, default=0, help="with --compare: allowed difference per byte (default 0)")
ap.add_argument("--cpu-mean-limit", type=float, default=2.0,
                help="largest mean difference per byte vs the CPU reference that still counts as a match (default 2.0); "
                     "raise it for synthetic models whose NN part is known to differ from the CPU, where --compare is the test")
ap.add_argument("--steps", default=os.environ.get("NPU_STEPS_FILE"))
a = ap.parse_args()

steps_fd = os.open(a.steps, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644) if a.steps else None


def step(msg):
    if steps_fd is None:
        return
    line = "[%12.6f] pid=%d %s\n" % (time.clock_gettime(time.CLOCK_MONOTONIC), os.getpid(), msg)
    os.write(steps_fd, line.encode())
    os.fsync(steps_fd)


ref = np.load(a.reference)
x, want = ref["input"], ref["output"]
step("start: delegate=%s model=%s ETNA_ML_SHADER=%s" % (a.delegate or "cpu", a.model, os.environ.get("ETNA_ML_SHADER")))
delegates = []
if a.delegate:
    step("load_delegate: begin")
    delegates = [load_delegate(a.delegate)]
    step("load_delegate: done")
t0 = time.monotonic()
interp = Interpreter(model_path=a.model, experimental_delegates=delegates, num_threads=2)
interp.allocate_tensors()
setup_ms = (time.monotonic() - t0) * 1000.0
step("allocate_tensors: done (%.1f ms)" % setup_ms)
inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
interp.set_tensor(inp["index"], x[None, ..., None] if x.ndim == 2 else x[None])
saved = np.load(a.compare) if a.compare else None

times, max_vs_first, max_vs_cpu, mean_vs_cpu, vs_saved, vs_saved_tol = [], [], [], [], [], []
first = None
for k in range(a.iterations):
    step("invoke %d/%d: begin" % (k + 1, a.iterations))
    t = time.monotonic()
    interp.invoke()
    times.append((time.monotonic() - t) * 1000.0)
    got = interp.get_tensor(out["index"])[0].copy()
    step("invoke %d/%d: done (%.2f ms)" % (k + 1, a.iterations, times[-1]))
    if first is None:
        first = got
        if a.save:
            np.save(a.save, got)
    d = np.abs(got.astype(np.int32) - want.astype(np.int32))
    max_vs_first.append(int(np.abs(got.astype(np.int32) - first.astype(np.int32)).max()))
    max_vs_cpu.append(int(d.max()))
    mean_vs_cpu.append(round(float(d.mean()), 3))
    if saved is not None:
        dv = np.abs(got.astype(np.int32) - saved.astype(np.int32))
        vs_saved.append(int(np.count_nonzero(dv)))
        vs_saved_tol.append(int(np.count_nonzero(dv > a.tolerance)))
step("finished")

res = {
    "delegate": a.delegate or "cpu",
    "model": a.model,
    "etna_ml_shader": os.environ.get("ETNA_ML_SHADER"),
    "setup_ms": round(setup_ms, 1),
    "invoke_ms": [round(t, 2) for t in times],
    "invoke_ms_median": round(statistics.median(times), 2),
    "invoke_ms_median_after_first": round(statistics.median(times[1:]), 2) if len(times) > 1 else None,
    "all_invokes_identical": all(m == 0 for m in max_vs_first),
    "max_diff_vs_first": max(max_vs_first),
    "diff_vs_cpu_max": max(max_vs_cpu),
    "diff_vs_cpu_mean_worst": max(mean_vs_cpu),
    "bytes_differing_from_saved": vs_saved if saved is not None else None,
    "tolerance": a.tolerance,
    "cpu_mean_limit": a.cpu_mean_limit,
    "bytes_beyond_tolerance": vs_saved_tol if saved is not None else None,
}
res["match"] = (res["all_invokes_identical"] and res["diff_vs_cpu_mean_worst"] <= a.cpu_mean_limit and
                (saved is None or all(v == 0 for v in vs_saved_tol)))
print(json.dumps(res, indent=1))
raise SystemExit(0 if res["match"] else 3)
