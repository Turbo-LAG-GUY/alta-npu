#!/usr/bin/env python3
"""npu-classify.py - MobileNet-style image classification on the CPU or through the Teflon
delegate (NPU). Prints one JSON object so CPU and NPU runs can be compared.
Works with tflite_runtime (Mesa docs) or ai_edge_litert (its successor, same entry points).

CPU reference : python3 scripts/npu-classify.py -m MODEL.tflite -i IMAGE > results/cpu.json
NPU (always via npu-run):
  scripts/npu-run -n mnv1 -- python3 scripts/npu-classify.py -m MODEL.tflite -i IMAGE -e /path/libteflon.so > results/npu.json
"""
import argparse
import json
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
a = ap.parse_args()

delegates = [load_delegate(a.delegate)] if a.delegate else []
t0 = time.monotonic()
interp = Interpreter(model_path=a.model, experimental_delegates=delegates, num_threads=a.threads)
interp.allocate_tensors()          # Teflon compiles the graph for the NPU here
setup_ms = (time.monotonic() - t0) * 1000.0

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

times = []
for _ in range(a.iterations):
    t = time.monotonic()
    interp.invoke()
    times.append((time.monotonic() - t) * 1000.0)

y = np.squeeze(interp.get_tensor(out["index"]))
if out["dtype"] != np.float32:
    scale, zero = out["quantization"]
    y = (y.astype(np.float32) - zero) * (scale or 1.0)
top = np.argsort(y)[::-1][:5]
labels = open(a.labels).read().splitlines() if a.labels else None

print(json.dumps({
    "runtime": RUNTIME,
    "delegate": a.delegate or "cpu",
    "model": a.model,
    "image": a.image,
    "setup_ms": round(setup_ms, 1),
    "invoke_ms": [round(t, 2) for t in times],
    "invoke_ms_median": round(statistics.median(times), 2),
    "top5": [{"index": int(i), "score": round(float(y[i]), 6),
              "label": labels[i] if labels and i < len(labels) else None} for i in top],
}, indent=1))
