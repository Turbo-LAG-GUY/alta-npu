#!/usr/bin/env python3
"""Decode an etnaviv command-stream dump (ETNA_MESA_DEBUG=dump_shaders, base_submit_*_cmdstream) into one line per
state write, naming registers from a Mesa state header. CPU only; reads files, touches no device.

usage: cmdstream_decode.py DUMP [--header state_3d.xml.h ...] [--diff OTHER_DUMP]
  --diff: decode both and print a unified diff of the decoded lines (the port vs the 2023 baseline).
"""
import argparse
import difflib
import re
import struct
import sys

DEFAULT_HEADERS = [
    "/home/jordan/Documents/AI_Projects/mesa-compute/src/etnaviv/hw/state_3d.xml.h",
    "/home/jordan/Documents/AI_Projects/mesa-compute/src/etnaviv/hw/state.xml.h",
    "/home/jordan/Documents/AI_Projects/mesa-compute/src/etnaviv/hw/state_blt.xml.h",
]

OPS = {1: "LOAD_STATE", 2: "END", 3: "NOP", 4: "DRAW_2D", 5: "DRAW_PRIMITIVES", 6: "DRAW_INDEXED", 7: "WAIT",
       8: "LINK", 9: "STALL", 10: "CALL", 11: "RETURN", 12: "DRAW_INSTANCED", 13: "CHIP_SELECT"}


def load_names(headers):
    """Map byte address -> register name. Arrays (NAME(i0) with stride) get NAME[i] entries for their range."""
    names = {}
    arrays = []
    for h in headers:
        try:
            text = open(h).read()
        except OSError:
            continue
        for m in re.finditer(r"^#define (VIVS_\w+)\s+(0x[0-9a-fA-F]+)\s*$", text, re.M):
            name, addr = m.group(1), int(m.group(2), 16)
            if re.search(r"_(MASK|SHIFT|__\w+)$", name) or addr < 0x100:
                continue
            names.setdefault(addr, name)
        for m in re.finditer(r"^#define (VIVS_\w+)\(i0\)\s+\((0x[0-9a-fA-F]+) \+ (0x[0-9a-fA-F]+)\*\(i0\)\)", text, re.M):
            arrays.append((m.group(1), int(m.group(2), 16), int(m.group(3), 16)))
        for m in re.finditer(r"^#define (VIVS_\w+)__LEN\s+(0x[0-9a-fA-F]+)", text, re.M):
            for i, a in enumerate(arrays):
                if a[0] == m.group(1) and len(a) == 3:
                    arrays[i] = a + (int(m.group(2), 16),)
    for a in arrays:
        if len(a) == 4:
            n, base, stride, length = a
            for i in range(length):
                names.setdefault(base + stride * i, "%s[%d]" % (n, i))
    return names


def decode(path, names):
    data = open(path, "rb").read()
    words = struct.unpack("<%dI" % (len(data) // 4), data[: len(data) // 4 * 4])
    out = []
    i = 0
    while i < len(words):
        w = words[i]
        op = w >> 27
        if op == 1:
            count = (w >> 16) & 0x3FF or 1024
            off = w & 0xFFFF
            fixp = " FIXP" if w & (1 << 26) else ""
            for k in range(count):
                if i + 1 + k >= len(words):
                    out.append("  (truncated LOAD_STATE)")
                    break
                addr = (off + k) * 4
                out.append("%05X %-40s = 0x%08x%s" % (addr, names.get(addr, "?"), words[i + 1 + k], fixp))
            n = 1 + count
            i += n + (n & 1)
        elif op == 9:
            out.append("STALL 0x%08x" % (words[i + 1] if i + 1 < len(words) else 0))
            i += 2
        elif op == 3:
            out.append("NOP 0x%08x" % (words[i + 1] if i + 1 < len(words) else 0))
            i += 2
        elif w == 0:
            out.append("zero")
            i += 1
        else:
            out.append("%s 0x%08x" % (OPS.get(op, "OP%d" % op), w))
            i += 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--header", action="append")
    ap.add_argument("--diff")
    a = ap.parse_args()
    names = load_names(a.header or DEFAULT_HEADERS)
    lines = decode(a.dump, names)
    if a.diff:
        other = decode(a.diff, names)
        d = list(difflib.unified_diff(other, lines, a.diff, a.dump, lineterm="", n=2))
        print("\n".join(d) if d else "identical after decoding (%d lines)" % len(lines))
        return 1 if d else 0
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
