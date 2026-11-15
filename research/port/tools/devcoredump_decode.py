#!/usr/bin/env python3
"""devcoredump_decode.py (2026-11-14, CPU only): lists the objects of an etnaviv devcoredump (format: etnaviv_dump.h
of the installed module's source, build/etnaviv-*/), prints the dumped registers by name and, for the command
buffers, where the front end's DMA address points (the command it was fetching when the GPU hung).

  devcoredump_decode.py DUMP.bin [--cmd]      --cmd: hex dump of each command buffer around the FE address
"""
import struct
import sys

MAGIC = 0x414e5445
TYPES = ["REG", "MMU", "RING", "CMD", "BOMAP", "BO", "END"]
REGS = {  # common.xml.h / state_hi.xml.h addresses of the dumped registers
    0x0000c: "HI_AXI_STATUS", 0x00000: "HI_CLOCK_CONTROL", 0x00004: "HI_IDLE_STATE", 0x00008: "HI_AXI_CONFIG",
    0x00014: "HI_INTR_ENBL", 0x00018: "HI_CHIP_IDENTITY", 0x0001c: "HI_CHIP_FEATURE", 0x00020: "HI_CHIP_MODEL",
    0x00024: "HI_CHIP_REV", 0x00028: "HI_CHIP_DATE", 0x0002c: "HI_CHIP_TIME", 0x00034: "HI_CHIP_MINOR_FEATURE_0",
    0x00038: "HI_CACHE_CONTROL", 0x003a4: "HI_AXI_CONTROL", 0x00100: "PM_POWER_CONTROLS",
    0x00104: "PM_MODULE_CONTROLS", 0x00108: "PM_MODULE_STATUS", 0x0010c: "PM_PULSE_EATER",
    0x00400: "MC_MMU_FE_PAGE_TABLE", 0x00404: "MC_MMU_TX_PAGE_TABLE", 0x00408: "MC_MMU_PE_PAGE_TABLE",
    0x0040c: "MC_MMU_PEZ_PAGE_TABLE", 0x00410: "MC_MMU_RA_PAGE_TABLE", 0x00414: "MC_DEBUG_MEMORY",
    0x00418: "MC_MEMORY_BASE_ADDR_RA", 0x0041c: "MC_MEMORY_BASE_ADDR_FE", 0x00420: "MC_MEMORY_BASE_ADDR_TX",
    0x00424: "MC_MEMORY_BASE_ADDR_PEZ", 0x00428: "MC_MEMORY_BASE_ADDR_PE", 0x0042c: "MC_MEMORY_TIMING_CONTROL",
    0x0040c + 0x28: "MC_BUS_CONFIG", 0x0065c: "FE_DMA_STATUS", 0x00660: "FE_DMA_DEBUG_STATE",
    0x00664: "FE_DMA_ADDRESS", 0x00668: "FE_DMA_LOW", 0x0066c: "FE_DMA_HIGH", 0x00670: "FE_AUTO_FLUSH",
}


def main():
    data = open(sys.argv[1], "rb").read()
    show_cmd = "--cmd" in sys.argv
    objs = []
    off = 0
    while off + 32 <= len(data):
        magic, typ, foff, fsize, iova, d0, d1 = struct.unpack_from("<IIIIQII", data, off)
        if magic != MAGIC:
            break
        objs.append((typ, foff, fsize, iova, d0, d1))
        off += 32
        if typ == 6:
            break
    regs = {}
    for typ, foff, fsize, iova, d0, d1 in objs:
        name = TYPES[typ] if typ < len(TYPES) else str(typ)
        print("%-5s file 0x%06x size %7d iova 0x%08x data %08x %08x" % (name, foff, fsize, iova, d0, d1))
        if typ == 0:
            for i in range(fsize // 8):
                r, v = struct.unpack_from("<II", data, foff + 8 * i)
                regs[r] = v
    print()
    for r, v in regs.items():
        print("  %-28s (0x%05x) = 0x%08x" % (REGS.get(r, "?"), r, v))
    fe = regs.get(0x664)
    if fe is None:
        return
    print("\nFE DMA address 0x%08x (low 0x%08x high 0x%08x)" % (fe, regs.get(0x668, 0), regs.get(0x66c, 0)))
    for typ, foff, fsize, iova, d0, d1 in objs:
        if typ in (2, 3) and iova <= fe < iova + fsize:
            k = fe - iova
            print("  inside %s at offset 0x%x of 0x%x (iova 0x%08x)" % (TYPES[typ], k, fsize, iova))
            if show_cmd:
                lo = max(0, k - 0x80) & ~0xf
                for a in range(lo, min(fsize, k + 0x40), 16):
                    w = struct.unpack_from("<4I", data, foff + a)
                    print("    %s%06x: %s" % ("->" if a <= k < a + 16 else "  ", a, " ".join("%08x" % x for x in w)))


if __name__ == "__main__":
    main()
