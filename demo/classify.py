#!/usr/bin/env python3
"""Image classification on the Amlogic A311D NPU via Mesa's Teflon TFLite delegate.

Usage:
  .venv/bin/python classify.py                 # run on NPU
  .venv/bin/python classify.py --cpu           # run on CPU for comparison
  .venv/bin/python classify.py -i photo.jpg
"""
import argparse
import os
import time

import numpy as np
from PIL import Image
import ai_edge_litert.interpreter as tflite

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DELEGATE = os.path.expanduser(
    "~/Documents/AI_Projects/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", default=f"{HERE}/models/mobilenet_v1_1.0_224_quant.tflite")
    p.add_argument("-l", "--labels", default=f"{HERE}/models/labels.txt")
    p.add_argument("-i", "--image", default=f"{HERE}/models/grace_hopper.bmp")
    p.add_argument("-d", "--delegate", default=DEFAULT_DELEGATE)
    p.add_argument("--cpu", action="store_true", help="skip the NPU delegate")
    p.add_argument("-n", "--runs", type=int, default=20)
    a = p.parse_args()

    # Keep the NPU's intermediate results in its 1 MiB of on-chip memory instead of RAM. This
    # Teflon build falls back to RAM by itself when the driver's axi_sram switch is off.
    # Run with ETNA_ML_AXI_SRAM=0 to go without it.
    os.environ.setdefault("ETNA_ML_AXI_SRAM", "all")

    delegates = [] if a.cpu else [tflite.load_delegate(a.delegate)]
    interp = tflite.Interpreter(model_path=a.model, experimental_delegates=delegates,
                                num_threads=os.cpu_count())
    interp.allocate_tensors()

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    _, h, w, _ = inp["shape"]
    img = np.asarray(Image.open(a.image).convert("RGB").resize((w, h)), dtype=inp["dtype"])
    interp.set_tensor(inp["index"], img[None, ...])

    interp.invoke()  # warm-up (first NPU run compiles the graph)
    t0 = time.perf_counter()
    for _ in range(a.runs):
        interp.invoke()
    ms = (time.perf_counter() - t0) * 1000 / a.runs

    scores = interp.get_tensor(out["index"])[0].astype(np.float32)
    if out["dtype"] == np.uint8:
        scale, zp = out["quantization"]
        scores = (scores - zp) * scale
    labels = [l.strip() for l in open(a.labels)]
    print(f"Backend: {'CPU' if a.cpu else 'NPU (Teflon)'}  |  {ms:.1f} ms/inference")
    for i in scores.argsort()[::-1][:5]:
        print(f"  {scores[i]:.3f}  {labels[i]}")


if __name__ == "__main__":
    main()
