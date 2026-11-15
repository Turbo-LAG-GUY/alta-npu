#!/usr/bin/env python3
"""tp_decode.py (2026-11-14, CPU only): print the fields of a dumped etnaviv TP config (mesa-tp-NNN-000.bin from
ETNA_MESA_DEBUG=dump_shaders), using the FIELD() list of struct etna_tp_params in etnaviv_ml_tp.c (uint32_t bit
fields, packed little-endian from bit 0 of each 32-bit word, as GCC lays them out on aarch64).

Usage: tp_decode.py CONFIG.bin [OTHER.bin]   (two files: print only the fields that differ)
       an NN config (mesa-nn-*.bin) is decoded with struct etna_nn_params from etnaviv_ml_nn.c instead
"""
import re
import struct
import sys

DRV = "/home/jordan/Documents/AI_Projects/mesa-compute/src/gallium/drivers/etnaviv/"
KIND = {"tp": ("etnaviv_ml_tp.c", "etna_tp_params"), "nn": ("etnaviv_ml_nn.c", "etna_nn_params")}


def layout(kind="tp"):
    src, struct_name = KIND[kind]
    text = open(DRV + src).read()
    body = re.search(r"struct %s \{(.*?)\n\}" % struct_name, text, re.S).group(1)
    fields, bit = [], 0
    for name, bits in re.findall(r"FIELD\((\w+),\s*(\d+)\)", body):
        bits = int(bits)
        if bit % 32 + bits > 32:  # a bit field doesn't straddle two uint32_t words
            bit += 32 - bit % 32
        fields.append((name, bit, bits))
        bit += bits
    return fields


def decode(path, kind=None):
    if kind is None:
        kind = "nn" if "mesa-nn-" in path else "tp"
    data = open(path, "rb").read()
    words = struct.unpack("<%dI" % (len(data) // 4), data[: len(data) // 4 * 4])
    out = {}
    for name, bit, bits in layout(kind):
        w = words[bit // 32]
        out[name] = (w >> (bit % 32)) & ((1 << bits) - 1)
    return out


def main():
    a = decode(sys.argv[1])
    b = decode(sys.argv[2]) if len(sys.argv) > 2 else None
    for k, v in a.items():
        if k.startswith("unused"):
            continue
        if b is None:
            print("%-40s %d (0x%x)" % (k, v, v))
        elif b[k] != v:
            print("%-40s %d -> %d" % (k, v, b[k]))


if __name__ == "__main__":
    main()
