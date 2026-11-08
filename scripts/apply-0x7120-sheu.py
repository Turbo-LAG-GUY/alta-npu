#!/usr/bin/env python3
"""apply-0x7120-sheu.py - add the Amlogic A311D NPU (VIPNano-QI, model 0x8000 rev 0x7120)
to etnaviv's SH_EU module-clock-gating workaround in etnaviv_gpu_enable_mlcg(), plus a
one-time kernel-log marker so the running module can be proven to be the patched one.

RECONSTRUCTION: rebuilt from upstream etnaviv (Linux >= 6.11 has the SH_EU block for
0x8000/0x7200, 0x8000/0x8002, 0x9200/0x6304 - not 0x7120). If the original patch from the
earlier work is found on the board, prefer it and record its sha256.

Usage: apply-0x7120-sheu.py path/to/etnaviv_gpu.c
Idempotent. Exits non-zero (and changes nothing) unless exactly one SH_EU block matches.
"""
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "etnaviv_gpu.c")
src = path.read_text()

if "0x8000, 0x7120" in src:
    print(f"{path}: already contains 0x8000/0x7120 - nothing to do")
    sys.exit(0)

# Upstream (>= 6.11) shape, tabs as in the kernel source:
#   \tif (etnaviv_is_model_rev(gpu, 0x8000, 0x7200) ||
#   \t    etnaviv_is_model_rev(gpu, 0x8000, 0x8002) ||
#   \t    etnaviv_is_model_rev(gpu, 0x9200, 0x6304))
#   \t\tpmc |= VIVS_PM_MODULE_CONTROLS_DISABLE_MODULE_CLOCK_GATING_SH_EU;
pat = re.compile(
    r"^(?P<ind>[ \t]+)(?P<last>etnaviv_is_model_rev\(gpu, 0x9200, 0x6304\))\)\n"
    r"(?P<bind>[ \t]+)(?P<body>pmc \|= VIVS_PM_MODULE_CONTROLS_DISABLE_MODULE_CLOCK_GATING_SH_EU;)\n",
    re.M,
)
matches = list(pat.finditer(src))
if len(matches) != 1:
    sys.exit(
        f"{path}: expected exactly 1 SH_EU clock-gating block, found {len(matches)}.\n"
        "Kernel older than 6.11 (no SH_EU block) or the block changed upstream.\n"
        "Make the equivalent change by hand: add 'etnaviv_is_model_rev(gpu, 0x8000, 0x7120)'\n"
        "to the SH_EU condition in etnaviv_gpu_enable_mlcg(), and record the diff."
    )

m = matches[0]
ind, bind = m.group("ind"), m.group("bind")
outer = bind[:-1] if bind.endswith("\t") else bind      # indentation of the 'if' itself
new = (
    f"{ind}{m.group('last')} ||\n"
    f"{ind}etnaviv_is_model_rev(gpu, 0x8000, 0x7120))\n"
    f"{bind}{m.group('body')}\n"
    f"{outer}if (etnaviv_is_model_rev(gpu, 0x8000, 0x7120))\n"
    f"{bind}dev_info_once(gpu->dev, \"alta: SH_EU clock gating disabled (VIPNano-QI rev 0x7120)\\n\");\n"
)
path.write_text(src[: m.start()] + new + src[m.end():])
print(f"{path}: patched - 0x8000/0x7120 added to the SH_EU workaround, dev_info_once marker added")
