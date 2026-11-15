#!/usr/bin/env python3
"""eltwise_table_check.py (step 6g, 2026-11-14, CPU only): checks the lookup tables the driver built for the
shader-core elementwise ops of a model against the TFLite interpreter itself.

Run the model offline first with dumps (research/port/tools/teflon_offline.sh with
TEFLON_OFFLINE_DEBUG=ml_msgs,dump_shaders and ETNA_ML_SHADER=...eltwise...): DIR then holds stdout.txt (the driver's
"shader eltwise" lines) and mesa-table-NNN-000.bin per shader job. For every model op the driver took as eltwise
(same order), this builds the same op alone with the model's quantization, runs every input byte pair through the
interpreter (default delegates = XNNPACK, as beside Teflon) and compares with the table entry by entry; constant
operands stored after the table must equal the model's constants (+128 for int8). RSQRT entries below the input zero
point are skipped (the CPU refuses those inputs).

  eltwise_table_check.py MODEL.tflite DIR
"""
import os
import re
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/home/jordan/Documents/AI_Projects/npu-shader/models")
import eltwise_fit as E  # noqa: E402

OPNAME = {"MUL": "mul", "SQUARED_DIFFERENCE": "sqdiff", "RSQRT": "rsqrt", "ADD": "add", "SUB": "sub"}


def main():
    model, d = sys.argv[1], sys.argv[2]
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=model)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()

    def q(i):
        p = td[i]["quantization_parameters"]
        return (float(p["scales"][0]), int(p["zero_points"][0]))

    def is_const(i):
        try:
            it.get_tensor(i)
            return td[i]["name"] in ("gamma", "beta", "eps") or "const" in td[i]["name"]
        except ValueError:
            return False

    log = open(os.path.join(d, "stdout.txt")).read()
    # shader jobs in instruction order, from the intermediate graph listing
    graph = log[log.index("dumping intermediate graph"):]
    sh_idx = [int(m.group(1)) for m in re.finditer(r"^\s*(\d+) SH ", graph, re.M)]
    lines = re.findall(r"^shader (transpose|detranspose|mean \w+|softmax|avgpool|eltwise \w+): (.*)$", log, re.M)
    if len(lines) != len(sh_idx):
        sys.exit("%d shader lines but %d SH jobs" % (len(lines), len(sh_idx)))
    elt = [(i, kind.split()[1], rest) for i, (kind, rest) in zip(sh_idx, lines) if kind.startswith("eltwise")]

    # model ops the driver took as eltwise: every MUL/SQDIFF/RSQRT, ADD/SUB with a constant input
    cand = []
    for o in ops:
        name = o["op_name"]
        if name not in OPNAME:
            continue
        ins = list(o["inputs"])
        # two-tensor ADD/SUB: on the shader only for int8 (since 2026-11-14: the NN's int8 ADD is broken here)
        if name in ("ADD", "SUB") and not any(is_const(i) for i in ins) and td[list(o["outputs"])[0]]["dtype"] != np.int8:
            continue
        cand.append((OPNAME[name], ins, list(o["outputs"])[0]))
    if len(cand) != len(elt):
        sys.exit("%d eltwise jobs but %d candidate model ops" % (len(elt), len(cand)))

    tmp = tempfile.mkdtemp(prefix="etc_", dir=os.environ.get("TMPDIR", "/tmp"))
    signed = True
    bad_total = 0
    for (idx, op, rest), (mop, ins, out) in zip(elt, cand):
        if op != mop:
            sys.exit("job %d is %s but model op is %s" % (idx, op, mop))
        signed = td[out]["dtype"] == np.int8
        off = 128 if signed else 0
        lo, hi = (-128, 127) if signed else (0, 255)
        dt = np.int8 if signed else np.uint8
        fold = int(re.search(r"fold (-?\d+)", rest).group(1))
        words = int(re.search(r"table (\d+) words", rest).group(1))
        raw = np.fromfile(os.path.join(d, "mesa-table-%03d-000.bin" % idx), np.uint8)
        table = raw[: 4 * words].view(np.uint32)
        qs = [q(i) for i in ins]
        qo = q(out)
        p = os.path.join(tmp, "op%d.tflite" % idx)
        if op == "rsqrt":
            E.build(p, op, 1, signed, qs, qo, [(1, 1, 1, 256)], (1, 1, 1, 256))
            x = np.arange(lo, hi + 1)
            valid = x >= qs[0][1]
            xin = np.where(valid, x, qs[0][1]).astype(dt).reshape(1, 1, 1, 256)
            got = E.run(p, [xin], False).reshape(256)
            want = got + off
            mask = valid
        else:
            E.build(p, op, 2, signed, qs, qo, [(1, 1, 256, 256)] * 2, (1, 1, 256, 256))
            ga = np.repeat(np.arange(lo, hi + 1), 256).reshape(1, 1, 256, 256).astype(dt)
            gb = np.tile(np.arange(lo, hi + 1), 256).reshape(1, 1, 256, 256).astype(dt)
            grid = E.run(p, [ga, gb], False).reshape(256, 256) + off
            if fold < 0:
                want = grid.reshape(65536)
            else:
                c = int(it.get_tensor(ins[fold]).reshape(-1)[0]) + off
                want = grid[c, :] if fold == 0 else grid[:, c]
            mask = np.ones(want.shape, bool)
        bad = int((table[mask] != want[mask]).sum())
        msg = "%d/%d entries differ" % (bad, int(mask.sum())) if bad else "EXACT (%d entries)" % int(mask.sum())
        # constants stored after the table
        cmsg = ""
        pos = 4 * words
        for k, i in enumerate(ins):
            if k == fold or not is_const(i):
                continue
            cval = (it.get_tensor(i).astype(np.int32).reshape(-1) + off).astype(np.uint8)
            stored = raw[pos: pos + cval.size]
            ok = np.array_equal(stored, cval)
            cmsg += ", constant %s %s" % (td[i]["name"], "stored right" if ok else "DIFFERS")
            bad += 0 if ok else 1
            pos += (cval.size + 15) // 16 * 16
        bad_total += bad
        print("job %2d %-6s (%s): table %s%s" % (idx, op, rest.split(",")[1].strip() + "," + rest.split(",")[2], msg, cmsg))
    print("ALL EXACT" if bad_total == 0 else "MISMATCHES: %d" % bad_total)
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
