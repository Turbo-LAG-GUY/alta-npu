#!/usr/bin/env python3
"""pipeline_check.py: keep more than one NPU job in flight and measure throughput (hw_job_limit L3, 2026-11-21).

T worker threads, each with its own Interpreter and its own Teflon delegate (each delegate opens its own NPU
device file, so each thread is a separate client to the kernel). The model runtime releases the GIL during
invoke (checked on the CPU: 2 threads did twice the work in the same time), so with T = 2 one job is always
waiting while the other runs: with hw_job_limit 1 it waits in the kernel's scheduler, with 2 or more it waits in
the NPU's own ring. T = 1 is the same work with nothing waiting.

Setup is sequential (interpreter, then one warm-up invoke per thread, which also fills the Teflon disk cache once),
so only invokes overlap. Every invoke's output is compared byte for byte with the first warm-up output (and the
--compare .npy); the full CPU-reference check (.npz with "input"/"output") runs on the warm-up and last outputs only,
because on a 960x540x4 output it costs ~36 ms of CPU per call and would hide the NPU in the throughput.

Only run on the NPU through ~/alta-npu/scripts/npu-attempt, with Jordan's OK. Without -e it runs on the CPU.
Step markers go to $NPU_STEPS_FILE like teflon_check.py.
"""
import argparse
import json
import os
import statistics
import threading
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-r", "--reference", required=True)
ap.add_argument("-e", "--delegate", help="path to libteflon.so (omit for a CPU run)")
ap.add_argument("-t", "--threads", type=int, default=2, help="jobs in flight (worker threads), default 2")
ap.add_argument("-n", "--iterations", type=int, default=40, help="invokes per thread after the warm-up")
ap.add_argument("--save", help="save the first output to this .npy")
ap.add_argument("--compare", help="a saved .npy every output must equal byte for byte")
ap.add_argument("--cpu-mean-limit", type=float, default=2.0)
ap.add_argument("--name", help="set this process's name (comm) before opening the NPU, for hw_job_prio")
ap.add_argument("--seconds", type=float, help="run each thread for this long instead of -n invokes")
ap.add_argument("--ready-file", help="create this file once set up (for prio_pair.sh)")
ap.add_argument("--go-file", help="wait for this file before the timed run (for prio_pair.sh)")
ap.add_argument("--steps", default=os.environ.get("NPU_STEPS_FILE"))
a = ap.parse_args()

if a.name:  # the kernel takes the priority from the group leader's name when the NPU is opened
    import ctypes
    if ctypes.CDLL(None, use_errno=True).prctl(15, a.name.encode()[:15], 0, 0, 0) != 0:  # PR_SET_NAME
        raise SystemExit("prctl(PR_SET_NAME) failed")

steps_fd = os.open(a.steps, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644) if a.steps else None
steps_lock = threading.Lock()


def step(msg):
    if steps_fd is None:
        return
    line = "[%12.6f] pid=%d tid=%d %s\n" % (time.clock_gettime(time.CLOCK_MONOTONIC), os.getpid(),
                                            threading.get_native_id(), msg)
    with steps_lock:
        os.write(steps_fd, line.encode())
        os.fsync(steps_fd)


ref = np.load(a.reference)
x, want = ref["input"], ref["output"]
saved = np.load(a.compare) if a.compare else None
step("start: threads=%d iterations=%d delegate=%s model=%s" % (a.threads, a.iterations, a.delegate or "cpu", a.model))


def make(i):
    step("worker %d: load_delegate + allocate" % i)
    delegates = [load_delegate(a.delegate)] if a.delegate else []
    it = Interpreter(model_path=a.model, experimental_delegates=delegates, num_threads=1)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    shape = tuple(int(d) for d in inp["shape"])
    it.set_tensor(inp["index"], x.reshape(shape) if x.size == int(np.prod(shape)) else x[None])
    step("worker %d: warm-up invoke" % i)
    it.invoke()
    step("worker %d: warm-up done" % i)
    return it, out["index"], it.get_tensor(out["index"])[0].copy()


workers = [make(i) for i in range(a.threads)]
first = workers[0][2]
if a.save:
    np.save(a.save, first)


def check(got):
    d = np.abs(got.astype(np.int32) - want.astype(np.int32))
    return (int(np.abs(got.astype(np.int32) - first.astype(np.int32)).max()), int(d.max()), float(d.mean()),
            None if saved is None else int(np.count_nonzero(got != saved)))


checks = [check(w[2]) for w in workers]           # warm-up outputs too
times = [[] for _ in workers]
counted = [0 for _ in workers]
errors = []
start = threading.Barrier(a.threads + 1)


def run(i):
    it, oi, _ = workers[i]
    try:
        start.wait()
        deadline = time.monotonic() + a.seconds if a.seconds else None
        k = 0
        while (k < a.iterations) if deadline is None else (time.monotonic() < deadline):
            t = time.monotonic()
            it.invoke()
            times[i].append((time.monotonic() - t) * 1000.0)
            got = it.get_tensor(oi)[0]
            same = np.array_equal(got, first)        # equal to the CPU-checked first output
            k += 1
            last = (k == a.iterations) if deadline is None else (time.monotonic() >= deadline)
            if not same or last:
                checks.append(check(got))
            elif saved is not None and not np.array_equal(got, saved):
                checks.append(check(got))
            counted[i] += 1
    except Exception as e:  # report, don't hang the other thread
        errors.append("worker %d: %r" % (i, e))


threads = [threading.Thread(target=run, args=(i,)) for i in range(a.threads)]
for th in threads:
    th.start()
if a.ready_file:
    open(a.ready_file, "w").close()
if a.go_file:
    while not os.path.exists(a.go_file):
        time.sleep(0.01)
step("timed run: begin")
t0 = time.monotonic()
start.wait()
for th in threads:
    th.join()
wall_ms = (time.monotonic() - t0) * 1000.0
step("timed run: done (%.1f ms)" % wall_ms)

total = sum(len(t) for t in times)
allt = [v for t in times for v in t]
res = {
    "delegate": a.delegate or "cpu",
    "model": a.model,
    "threads": a.threads,
    "invokes": total,
    "wall_ms": round(wall_ms, 1),
    "ms_per_invoke_throughput": round(wall_ms / total, 3) if total else None,
    "invokes_per_s": round(total / wall_ms * 1000.0, 2) if total else None,
    "invoke_ms_median": round(statistics.median(allt), 2) if allt else None,
    "invoke_ms_median_per_thread": [round(statistics.median(t), 2) if t else None for t in times],
    "invoke_ms_p90": round(sorted(allt)[int(0.9 * (len(allt) - 1))], 2) if allt else None,
    "invoke_ms_max": round(max(allt), 2) if allt else None,
    "name": a.name,
    "max_diff_vs_first": max(c[0] for c in checks),
    "diff_vs_cpu_max": max(c[1] for c in checks),
    "diff_vs_cpu_mean_worst": round(max(c[2] for c in checks), 3),
    "bytes_differing_from_saved_total": None if saved is None else sum(c[3] for c in checks),
    "outputs_byte_checked": sum(counted),
    "outputs_cpu_checked": len(checks),
    "errors": errors,
}
res["all_invokes_identical"] = res["max_diff_vs_first"] == 0
res["match"] = (not errors and (a.seconds or total == a.threads * a.iterations) and total > 0
                and res["all_invokes_identical"]
                and res["diff_vs_cpu_mean_worst"] <= a.cpu_mean_limit
                and (saved is None or res["bytes_differing_from_saved_total"] == 0))
print(json.dumps(res, indent=1))
raise SystemExit(0 if res["match"] else 3)
