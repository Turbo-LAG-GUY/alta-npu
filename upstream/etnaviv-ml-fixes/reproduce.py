#!/usr/bin/env python3
"""Check an attached model against its CPU reference; --delegate uses hardware.

Dependencies: numpy and ai_edge_litert (or tflite_runtime).
Without --delegate this runs only on the CPU. On Jordan's board, hardware
invocations require step approval and must use scripts/npu-attempt.
"""
import argparse
import json
from pathlib import Path

import numpy as np
try:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:
    from tflite_runtime.interpreter import Interpreter, load_delegate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model', type=Path)
parser.add_argument('--delegate', help='libteflon.so; omit for CPU reference validation')
parser.add_argument('--iterations', type=int, default=10)
parser.add_argument('--max-diff', type=int, default=1)
parser.add_argument('--save', type=Path, help='save first output as a numpy array')
args = parser.parse_args()
if args.iterations < 1 or args.max_diff < 0:
    parser.error('iterations must be positive and max-diff nonnegative')
ref = np.load(args.model.with_name(args.model.stem + '_reference.npz'))
interpreter = Interpreter(model_path=str(args.model), num_threads=2,
                          experimental_delegates=[load_delegate(args.delegate)] if args.delegate else [])
interpreter.allocate_tensors()
inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
interpreter.set_tensor(inp['index'], ref['input'].reshape(inp['shape']))
first, errors, stable = None, [], True
for _ in range(args.iterations):
    interpreter.invoke()
    got = interpreter.get_tensor(out['index'])[0].copy()
    if first is None:
        first = got
    stable = stable and np.array_equal(got, first)
    delta = np.abs(got.astype(np.int32) - ref['output'].astype(np.int32))
    errors.append(dict(max=int(delta.max()), mean=float(delta.mean())))
if args.save:
    np.save(args.save, first)
passed = stable and max(e['max'] for e in errors) <= args.max_diff
print(json.dumps(dict(model=str(args.model), delegate=args.delegate or 'cpu',
                      iterations=args.iterations, all_invokes_identical=stable,
                      errors=errors, passed=passed), indent=2))
raise SystemExit(0 if passed else 1)
