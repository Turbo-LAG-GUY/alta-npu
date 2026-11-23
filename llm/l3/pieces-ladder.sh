#!/usr/bin/env bash
# pieces-ladder.sh (2026-11-22, LLM engine): token pieces through the AXI SRAM, ON THE NPU. Written CPU only.
# NEEDS JORDAN'S OK BEFORE IT RUNS. Message the other NPU session(s) first; Steam closed; quiet CPU.
#
# Why: a 1.7B down step at 256 tokens has a 2 MiB input (8192 ch); it fits neither the 512 KiB VIP SRAM nor the 1 MiB
# AXI SRAM, so every conv job re-reads it from DDR. Cut into token pieces in ONE submit, each piece's input is copied
# once by a transpose job into the AXI SRAM and its conv jobs re-read it there; the next piece reuses the same space.
# DDR per step for this 4-job slice: whole ~1.8 MB weights + 4 x 2 MiB input; 2 pieces ~3.7 MB weights + 2 MiB; 4 pieces
# ~7.3 MB + 2 MiB (weights are read once per piece). Unknown and measured here: how fast the NN core reads the AXI SRAM.
# Library: mesa-llm 4eb662fc (4b666c08 AXI port + one-submit guards, patch 0006 port: refused under npu_no_batching / npu_parallel,
# 0x4000-word bound, pre-flush, split warning); jb_run 21b8ac1f (several graph inputs).
# Offline: npu-llm/runs/offline-pieces-* (llm/l3/pieces_offline.sh) must have PASSED (order T nnnn per piece, every
# piece's copy at the same SRAM address, one submit per invoke, tiles inside the image); one-input packs byte-identical
# to jb_run 943bf0c2 (runs/offline-pieces-regress-*).
#
# Packs npu-llm/models/jb/axi-pieces (synthetic int8, every output checked): IC 8192, 4 jobs x 56 out ch, 256 tokens.
# Rungs, from the known to the new (every one is a new shape on the NPU: no 1.7B-sized job has run yet):
#   W1-P0  whole step, planar input in DDR (today's path, no TP job)
#   W1-N1  whole step, channels-last input, one shared TP transpose 16x16x8192 into DDR
#   P4-N1  4 pieces of 8x8 tokens, 4 TP copies in DDR
#   P4-N2  the same with the copies in the AXI SRAM (512 KiB, shared by the pieces) - first LLM job reading the SRAM
#   P2-N1  2 pieces of 16x8 tokens (H > W: the fixed V7 tiling path), copies in DDR
#   P2-N2  the same in the AXI SRAM (1 MiB = all of it)
# Stops at the first failure, STOP, an output more than 1 level off, a bad invoke, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-pieces --collect llm/l3/pieces-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JR=$AI/npu-llm/bin/jb_run-21b8ac1f
P=$AI/npu-llm/models/jb/axi-pieces
RD=$K/runs/llm-pieces-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JR" | cut -c1-8)" = 21b8ac1f ] || { echo "jb_run is not 21b8ac1f"; exit 1; }
[ "$(cat /sys/module/etnaviv/parameters/axi_sram)" = Y ] || { echo "etnaviv axi_sram is off"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
grep -qx PASS $AI/npu-llm/runs/offline-pieces-2*/VERDICT.txt 2>/dev/null || { echo "no PASSED offline-pieces run"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_KEEP_IMAGE_CACHE -u ETNA_ML_SHARE_INPUT_TRANSPOSE)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC MAXDIFF BAD
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "${3:-99}" -gt 1 ] || [ "${4:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-pieces-ladder FAILED "stopped at $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

one() { # PACK VARIANT
   local pk=$1 v=$2 f e=() label rc mx bad
   case $v in
      P0) f=planar ;;
      N1) f=nhwc e=(ETNA_ML_SHARE_INPUT_TRANSPOSE=1) ;;
      N2) f=nhwc e=(ETNA_ML_SHARE_INPUT_TRANSPOSE=1 ETNA_ML_AXI_SRAM=all) ;;
   esac
   label=LLMP-$pk-$v
   "${CLEAN[@]}" "${e[@]}" scripts/npu-attempt "$label" -- "$JR" -n 20 --wait spin "$P/${pk}_$f.jbp" \
      > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_invokes": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   grep -E 'AXI SRAM' "$RD/$label.stdout"
   echo "$label $(grep -oE '"(ms_median_after_first|npu_ms|checked_invokes)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   check "$label" "$rc" "$mx" "$bad"
}

scripts/progress llm-pieces-ladder STARTING "llm/l3/pieces-ladder.sh: $RD"
for r in W1:P0 W1:N1 P4:N1 P4:N2 P2:N1 P2:N2; do
   one ${r%:*} ${r#*:}
done
scripts/progress llm-pieces-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"
