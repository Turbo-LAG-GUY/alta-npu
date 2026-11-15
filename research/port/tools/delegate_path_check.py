#!/usr/bin/env python3
"""delegate_path_check.py (step 6g, 2026-11-14): which CPU implementation runs an op Teflon leaves on the CPU, when the
Teflon delegate is loaded: XNNPACK (the interpreter's default delegate) or the builtin kernel. They differ for 8-bit
MUL/ADD/SUB, so the shader core must copy the one that actually runs.

  delegate_path_check.py LIBTEFLON          ONLY under the fake NPU (research/port/tools/teflon_offline.sh style:
                                            bwrap --dev /dev + strace -E LD_PRELOAD=shim): the ops here are not
                                            Teflon-supported, so nothing is submitted, but loading the delegate opens
                                            the NPU device. Refuses to run without the shim.
"""
import ctypes
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eltwise_fit as E  # noqa: E402


def main():
    lib = sys.argv[1]
    try:
        ctypes.CDLL(None).npu_etna_shim_active
    except AttributeError:
        sys.exit("refusing: the NPU shim is not preloaded (this would open the real NPU)")
    from ai_edge_litert.interpreter import Interpreter, load_delegate

    rng = np.random.default_rng(1114)
    tmp = tempfile.mkdtemp(prefix="dpc_", dir=os.environ.get("TMPDIR", "/tmp"))
    ga = np.repeat(np.arange(-128, 128), 256).reshape(1, 1, 256, 256).astype(np.int8)
    gb = np.tile(np.arange(-128, 128), 256).reshape(1, 1, 256, 256).astype(np.int8)
    for op in (sys.argv[2].split(",") if len(sys.argv) > 2 else ("mul", "sqdiff")):
        for t in range(6):
            qa, qb, qo = E.rand_q(rng, True), E.rand_q(rng, True), E.rand_q(rng, True, 0.005, 2.0)
            p = os.path.join(tmp, "%s_%d.tflite" % (op, t))
            E.build(p, op, 2, True, [qa, qb], qo, [(1, 1, 256, 256)] * 2, (1, 1, 256, 256))
            xnn = E.run(p, [ga, gb], False)
            bi = E.run(p, [ga, gb], True)
            it = Interpreter(model_path=p, experimental_delegates=[load_delegate(lib)], num_threads=1)
            it.allocate_tensors()
            for d, x in zip(it.get_input_details(), [ga, gb]):
                it.set_tensor(d["index"], x)
            it.invoke()
            got = it.get_tensor(it.get_output_details()[0]["index"]).astype(np.int64)
            disc = xnn != bi
            print("%-6s trial %d: %5d discriminating pairs; with Teflon loaded: = XNNPACK on %s, = builtin on %s" % (
                op, t, int(disc.sum()),
                "all" if (got == xnn).all() else "%d/%d" % (int((got == xnn).sum()), got.size),
                "all" if (got == bi).all() else "%d/%d" % (int((got == bi).sum()), got.size)))


if __name__ == "__main__":
    main()
