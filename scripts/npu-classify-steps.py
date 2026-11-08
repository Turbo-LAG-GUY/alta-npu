#!/usr/bin/env python3
"""npu-classify-steps.py - scripts/npu-classify.py plus crash-proof step markers. Same model and
image handling, same JSON on stdout. It also appends a line to the step file before and after each
stage (delegate load, graph compile, every invoke) and forces it to disk. Each line records the NPU's
runtime-PM state and IRQ count. After a hard freeze, the file shows the last stage that was reached.
The step file is --steps FILE, or $NPU_STEPS_FILE, which scripts/npu-attempt sets.
NPU runs go only through scripts/npu-run (normally via scripts/npu-attempt).
"""
import argparse
import json
import os
import statistics
import time

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    RUNTIME = "tflite_runtime"
except ImportError:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
    RUNTIME = "ai_edge_litert"

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-e", "--delegate", help="path to libteflon.so (omit for a CPU run)")
ap.add_argument("-l", "--labels", help="optional labels file, one label per line")
ap.add_argument("-n", "--iterations", type=int, default=5)
ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--steps", default=os.environ.get("NPU_STEPS_FILE"), help="step-marker file (default: $NPU_STEPS_FILE)")
ap.add_argument("--mark-every", type=int, default=1, metavar="K",
                help="mark invokes 1, 2, every K-th and the last (default 1: every invoke). For long runs, so the "
                     "markers' disk syncs don't put gaps between invokes")
ap.add_argument("--off-seconds", type=float, default=0.0, metavar="S",
                help="before each invoke after the first, wait S seconds and until the NPU has powered down "
                     "(runtime_status 'suspended'), so that every invoke wakes it up (the wake-up test)")
ap.add_argument("--check-each", action="store_true", help="record the top-1 class of every invoke, not just the last")
a = ap.parse_args()
a.mark_every = max(1, a.mark_every)

PM = "/sys/devices/platform/ff100000.npu/power/"
steps_fd = os.open(a.steps, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644) if a.steps else None


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "?"


def npu_irqs():
    try:
        with open("/proc/interrupts") as f:
            for line in f:
                if "ff100000.npu" in line:
                    return str(sum(int(x) for x in line.split()[1:7] if x.isdigit()))
    except OSError:
        pass
    return "?"


def step(msg):
    if steps_fd is None:
        return
    line = "[%12.6f] pid=%d %s | npu=%s active_ms=%s irqs=%s\n" % (
        time.clock_gettime(time.CLOCK_MONOTONIC), os.getpid(), msg,
        read(PM + "runtime_status"), read(PM + "runtime_active_time"), npu_irqs())
    os.write(steps_fd, line.encode())
    os.fsync(steps_fd)


step("start: runtime=%s delegate=%s model=%s" % (RUNTIME, a.delegate or "cpu", a.model))
if a.delegate:
    step("load_delegate: begin (opens the NPU render node)")
    delegates = [load_delegate(a.delegate)]
    step("load_delegate: done")
else:
    delegates = []
t0 = time.monotonic()
step("Interpreter(): begin (the delegate is applied to the graph)")
interp = Interpreter(model_path=a.model, experimental_delegates=delegates, num_threads=a.threads)
step("Interpreter(): done")
step("allocate_tensors: begin (Teflon compiles the graph for the NPU)")
interp.allocate_tensors()
setup_ms = (time.monotonic() - t0) * 1000.0
step("allocate_tensors: done (setup %.1f ms)" % setup_ms)

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
_, h, w, _ = inp["shape"]
x = np.expand_dims(np.asarray(Image.open(a.image).convert("RGB").resize((w, h))), 0)
if inp["dtype"] == np.float32:
    x = (x.astype(np.float32) - 127.5) / 127.5
elif inp["dtype"] == np.int8:
    x = (x.astype(np.int16) - 128).astype(np.int8)
else:
    x = x.astype(inp["dtype"])
interp.set_tensor(inp["index"], x)
step("input set")

times = []
status_before = []  # NPU runtime-PM state just before each invoke ("suspended": the invoke woke it up)
top1_each = []
for k in range(a.iterations):
    if a.off_seconds > 0 and k > 0:
        time.sleep(a.off_seconds)
        t_wait = time.monotonic()
        while read(PM + "runtime_status") != "suspended" and time.monotonic() - t_wait < 5.0:
            time.sleep(0.02)
    mark = k < 2 or (k + 1) % a.mark_every == 0 or k + 1 == a.iterations
    status_before.append(read(PM + "runtime_status"))
    if mark:
        step("invoke %d/%d: begin" % (k + 1, a.iterations))
    t = time.monotonic()
    interp.invoke()
    times.append((time.monotonic() - t) * 1000.0)
    if a.check_each:
        top1_each.append(int(np.argmax(interp.get_tensor(out["index"]))))
    if mark:
        step("invoke %d/%d: done (%.2f ms)" % (k + 1, a.iterations, times[-1]))

y = np.squeeze(interp.get_tensor(out["index"]))
if out["dtype"] != np.float32:
    scale, zero = out["quantization"]
    y = (y.astype(np.float32) - zero) * (scale or 1.0)
top = np.argsort(y)[::-1][:5]
labels = open(a.labels).read().splitlines() if a.labels else None
step("finished: top-1 index %d" % int(top[0]))

res = {
    "runtime": RUNTIME,
    "delegate": a.delegate or "cpu",
    "model": a.model,
    "image": a.image,
    "setup_ms": round(setup_ms, 1),
    "invoke_ms": [round(t, 2) for t in times],
    "invoke_ms_median": round(statistics.median(times), 2),
    "top5": [{"index": int(i), "score": round(float(y[i]), 6),
              "label": labels[i] if labels and i < len(labels) else None} for i in top],
}
if a.off_seconds > 0:
    res["off_seconds"] = a.off_seconds
    res["wakeups_confirmed"] = sum(s == "suspended" for s in status_before)
    res["status_before"] = status_before
if a.check_each:
    res["top1_each"] = top1_each
print(json.dumps(res, indent=1))
