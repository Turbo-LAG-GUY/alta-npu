#!/usr/bin/env bash
# axi-ladder.sh (2026-11-22, LLM engine): the 1.7B step input in the AXI SRAM, ON THE NPU. Written CPU only.
# NEEDS JORDAN'S OK BEFORE IT RUNS. Message the other NPU session(s) first; Steam closed; quiet CPU.
#
# Why: at 1.7B sizes the step input (2048 ch x 256 tokens = 512 KiB) never fits the 512 KiB VIP SRAM, so each conv
# job re-reads it from DDR once per superblock (4 per job here), and once per job across jobs. The NPU copies it into
# the AXI SRAM first (the CPU has no mapping): channels-last input, one shared driver transpose job
# (ETNA_ML_SHARE_INPUT_TRANSPOSE=1), its output placed in the AXI SRAM (ETNA_ML_AXI_SRAM=all), the convs read it there.
# jb_run 943bf0c2 = mesa-llm + keep-image hook + AXI patch 0002 port + shared transpose hook + a guard for an image of
# exactly 512 KiB (upstream gave an empty kernel area, kernel end 0 < start 0x800; now: image not cached, the layout
# 1-job gateup already ran). Offline: npu-llm/runs/offline-axi-* (llm/l3/axi_offline.sh) must have PASSED.
#
# Packs npu-llm/models/jb/axi-1p7b (synthetic int8, every output checked): A2 / A8 = IC 2048, 2 / 8 jobs x 224 out
# channels, 256 tokens; D4 = IC 8192 (down), 4 jobs x 56, 64 tokens. Per pack, from the known to the new:
#   P0 planar input in DDR (today's path)                                     - layouts that ran before
#   PT P0 + ETNA_ML_V7_TILE_H (8 / 4) + keep-image: the no-AXI alternative     - tile 16x8, kpc 14 (new tile, cap 21)
#   N1 channels-last input, one shared TP transpose into DDR (cost of the copy) - new TP shape 16x16x2048
#   N2 N1 with the copy in the AXI SRAM                                        - first LLM job reading the AXI SRAM
# Stops at the first failure, STOP, an output more than 1 level off, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-axi --collect llm/l3/axi-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JR=$AI/npu-llm/bin/jb_run-943bf0c2
P=$AI/npu-llm/models/jb/axi-1p7b
RD=$K/runs/llm-axi-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JR" | cut -c1-8)" = 943bf0c2 ] || { echo "jb_run is not 943bf0c2"; exit 1; }
[ "$(cat /sys/module/etnaviv/parameters/axi_sram)" = Y ] || { echo "etnaviv axi_sram is off"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
ls -d $AI/npu-llm/runs/offline-axi-* >/dev/null 2>&1 || { echo "no offline-axi run"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_KEEP_IMAGE_CACHE -u ETNA_ML_SHARE_INPUT_TRANSPOSE)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC MAXDIFF BAD
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "${3:-99}" -gt 1 ] || [ "${4:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-axi-ladder FAILED "stopped at $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

one() { # PACK VARIANT
   local pk=$1 v=$2 th=8 f e=() label rc mx bad
   case $pk in D*) th=4 ;; esac
   case $v in
      P0) f=planar ;;
      PT) f=planar e=(ETNA_ML_V7_TILE_H=$th ETNA_ML_KEEP_IMAGE_CACHE=1) ;;
      N1) f=nhwc e=(ETNA_ML_SHARE_INPUT_TRANSPOSE=1) ;;
      N2) f=nhwc e=(ETNA_ML_SHARE_INPUT_TRANSPOSE=1 ETNA_ML_AXI_SRAM=all) ;;
   esac
   label=LLMA-$pk-$v
   "${CLEAN[@]}" "${e[@]}" scripts/npu-attempt "$label" -- "$JR" -n 20 --wait spin "$P/${pk}_$f.jbp" \
      > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_invokes": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   grep -E 'AXI SRAM' "$RD/$label.stdout"
   echo "$label $(grep -oE '"(ms_median_after_first|npu_ms|checked_invokes)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   check "$label" "$rc" "$mx" "$bad"
}

scripts/progress llm-axi-ladder STARTING "llm/l3/axi-ladder.sh: $RD"
for pk in A2 A8 D4; do
   for v in P0 PT N1 N2; do
      one $pk $v
   done
done
scripts/progress llm-axi-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"
