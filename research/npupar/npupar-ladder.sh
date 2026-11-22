#!/usr/bin/env bash
# npupar-ladder.sh - NPU ladder for Mesa's ETNA_MESA_DEBUG=npu_parallel mode ("unsafe", off by default since
# Mesa 1277f58d8a4a, 2024-04-25: "doesn't work for all models, with the same happening to the proprietary driver").
# Written 2026-11-21 (CPU only). Jordan's OK 2026-11-21: "You have my OK to run the NPU test".
#
# What the switch changes (offline, fake NPU, kit library mesa-axi-sram; runs/... see research/npupar/NOTES.md):
# only GL_NN_CONFIG (SMALL_BATCH bit 0x10 cleared), PS_NN_INST_ADDR (+op ID in the low bits), PS_UNK10A4 (= op ID)
# and GL_FLUSH_CACHE (0xc23 -> 0x403). Buffer lists identical.
# Hazard found offline: the op ID is `static unsigned i` in etna_ml_subgraph_invoke, never reset, so it keeps
# counting across partitions and invokes of one process, and the TP code uses 0x1f as a marker (the ID field may be
# 5 bits). So every rung keeps IDs <= 30: the upscaler nets have 3 NN ops per invoke, at most 10 invokes per
# process (IDs 1-30, checked offline). MobileNet is NOT in this ladder: its first invoke alone reaches IDs 31-32.
#
# Every rung is one scripts/npu-attempt job; the ladder stops at the first rung that fails (incl. a wrong output:
# teflon_check compares every invoke with the first, the CPU reference, and byte for byte with the default mode's
# output) or sets runs/STOP. No ETNA_ML_AXI_SRAM, so the NPU gets the streams checked offline.
#   1 NPUPAR-U8-BASE    live upscaler net hist2_n8_planar_960x540, default mode, 10 invokes, output saved
#   2 NPUPAR-U8-PAR1    same, npu_parallel, 1 invoke (IDs 1-3)
#   3 NPUPAR-U8-PAR10   same, npu_parallel, 10 invokes (IDs 1-30)
#   4 NPUPAR-U16-BASE   hist2_n16_planar_960x540, default, 10 invokes, output saved
#   5 NPUPAR-U16-PAR10  same, npu_parallel, 10 invokes (IDs 1-30)
#   6 NPUPAR-U8-BASE2   rung 1 again (timing noise check, sandwich)
# Usage: research/npupar/npupar-ladder.sh   (run detached: systemd-run --user --unit=alta-npupar-ladder --collect)
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
CS=$AI/npu-shader
PY=$AI/.venv/bin/python
LIB=$AI/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so
U=$AI/npu-upscale/models
U8=$U/hist2_n8_planar_960x540.tflite;  U8R=$U/hist2_n8_planar_reference_960x540.npz
U16=$U/hist2_n16_planar_960x540.tflite; U16R=$U/hist2_n16_planar_reference_960x540.npz
RD=$K/runs/npupar-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD/cache"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi

rung() { # LABEL DEBUG -- CMD...
   local label=$1 dbg=$2; shift 3
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_ML_AXI_SRAM -u ETNA_CS_BARRIER_REG_BUDGET \
      ETNA_ML_DISK_CACHE_DIR="$RD/cache" ${dbg:+ETNA_MESA_DEBUG=$dbg} \
      scripts/npu-attempt "$label" -- "$@" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc"
   grep -E '"result"|"match"|invoke_ms_median"|all_invokes|bytes_differing' "$RD/$label.stdout" | head -6
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress npupar-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}

scripts/progress npupar-ladder STARTING "research/npupar/npupar-ladder.sh: $RD"
rung NPUPAR-U8-BASE   ""           -- $PY -I $CS/teflon_check.py -m $U8 -r $U8R -e $LIB -n 10 --save $RD/u8_base.npy
rung NPUPAR-U8-PAR1   npu_parallel -- $PY -I $CS/teflon_check.py -m $U8 -r $U8R -e $LIB -n 1 --compare $RD/u8_base.npy
rung NPUPAR-U8-PAR10  npu_parallel -- $PY -I $CS/teflon_check.py -m $U8 -r $U8R -e $LIB -n 10 --compare $RD/u8_base.npy
rung NPUPAR-U16-BASE  ""           -- $PY -I $CS/teflon_check.py -m $U16 -r $U16R -e $LIB -n 10 --save $RD/u16_base.npy
rung NPUPAR-U16-PAR10 npu_parallel -- $PY -I $CS/teflon_check.py -m $U16 -r $U16R -e $LIB -n 10 --compare $RD/u16_base.npy
rung NPUPAR-U8-BASE2  ""           -- $PY -I $CS/teflon_check.py -m $U8 -r $U8R -e $LIB -n 10 --compare $RD/u8_base.npy
rm -rf "$RD/cache"
scripts/progress npupar-ladder DONE "all rungs passed: $RD"
echo "all rungs passed: $RD"
