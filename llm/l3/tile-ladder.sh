#!/usr/bin/env bash
# tile-ladder.sh (2026-11-22, LLM engine): the tile-height rule and no-flush between independent jobs ON THE NPU.
# Written CPU only. NEEDS JORDAN'S OK BEFORE IT RUNS. Message the other NPU sessions first; Steam closed; quiet CPU.
#
# jb_run 674d417f / jb_engine 4002516c = mesa-llm 4eb662fc + ETNA_ML_V7_TILE_FIT (tile height so one input tile is
# <= N bytes; with N = 262144 the configs are byte-identical to ETNA_ML_V7_TILE_H=8 / 4 on the axi-1p7b packs, which
# ran in runs/llm-axi-ladder-20261122-140434) + ETNA_ML_NO_FLUSH_INDEPENDENT (nn_layer_flush = 0 on every job of a step
# but the last: the weight-group jobs read the same input and write separate outputs; new on the NPU).
# Offline: tile_offline.sh on every pack set used here must have PASSED (npu-llm/runs/offline-tile-*).
#
# Synthetic 1.7B-shaped packs (npu-llm/models/jb/axi-1p7b), jb_run, every output checked:
#   S1 A2 tile rule (= the PT layout that ran)        S2 S1 + no flush (2 jobs: the first without flush)
#   S3 A8 tile rule                                    S4 S3 + no flush (8 jobs: 7 without flush)
# 135M real packs, jb_engine all 120 steps, 256 tokens (+ 1 token for no-flush):
#   R1 4 jobs keep-image (baseline, 139.2 ms in keepimg-ladder)      R2 R1 + tile rule (down 16x10: new layout)
#   R3 R2 + no flush                                                   R4 1 token, keep-image + no flush
# 1.7B real packs (npu-llm/models/jb/real-1p7b, 4 jobs), jb_engine --layers 4 (16 steps, ~280 MB of weights; the whole
# model is 1.7 GB of NPU-pinned memory on a 3.7 GB board), 256 tokens:
#   B1 keep-image only (tile 16x16: every 2048-input step has the exactly-512 KiB image -> the guard: not cached)
#   B2 keep-image + tile rule (2048-input steps 16x8, down 16x2)      B3 B2 but down left at 16x16 (TILE_FIT_MAX_IC=2048)
#   B4 B2 + no flush                                                   B5 1 token, keep-image + no flush
# 2026-11-22 15:26: the no-flush rungs are REMOVED: nn_layer_flush = 0 hung the NPU at S2 (MMU fault at address 0, 20
# recovered hangs; incidents/2026-11-22-1526-LLMT-S2-noflush). Remaining: S1 S3, R1 R2, B1 B2 B3 B5.
# Stops at the first failure, STOP, an output more than 1 level off, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-tile --collect llm/l3/tile-ladder.sh [synthetic|135m|1p7b ...]
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JR=$AI/npu-llm/bin/jb_run-674d417f
JE=$AI/npu-llm/bin/jb_engine-4002516c
J=$AI/npu-llm/models/jb
RD=$K/runs/llm-tile-ladder-$(date +%Y%m%d-%H%M%S)
PARTS=${*:-synthetic 135m 1p7b}
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JR" | cut -c1-8)" = 674d417f ] || { echo "jb_run is not 674d417f"; exit 1; }
[ "$(sha256sum "$JE" | cut -c1-8)" = 4002516c ] || { echo "jb_engine is not 4002516c"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
ls -d $AI/npu-llm/runs/offline-tile-* >/dev/null 2>&1 || { echo "no offline-tile run"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_KEEP_IMAGE_CACHE -u ETNA_ML_SHARE_INPUT_TRANSPOSE
       -u ETNA_ML_V7_TILE_FIT -u ETNA_ML_V7_TILE_FIT_MAX_IC -u ETNA_ML_NO_FLUSH_INDEPENDENT)
KEEP=ETNA_ML_KEEP_IMAGE_CACHE=1 FIT=ETNA_ML_V7_TILE_FIT=262144 NOFL=ETNA_ML_NO_FLUSH_INDEPENDENT=1
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC MAXDIFF BAD
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "${3:-99}" -gt 1 ] || [ "${4:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-tile-ladder FAILED "stopped at $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

one() { # LABEL PACK ENV...   (jb_run, 20 invokes)
   local label=LLMT-$1 pack=$2 rc mx bad; shift 2
   "${CLEAN[@]}" "$@" scripts/npu-attempt "$label" -- "$JR" -n 20 --wait spin "$pack" > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_invokes": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   echo "$label $(grep -oE '"(ms_median_after_first|npu_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   check "$label" "$rc" "$mx" "$bad"
}

eng() { # LABEL PACKDIR TOKENS LAYERS ENV...   (jb_engine, 20 rounds, first and last checked)
   local label=LLMT-$1 pd=$2 t=$3 l=$4 rc mx bad; shift 4
   "${CLEAN[@]}" "$@" scripts/npu-attempt "$label" -- "$JE" -n 20 --tokens "$t" --layers "$l" "$pd" \
      > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_packs": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   grep -E '^compiled|^  [a-z]' "$RD/$label.stdout"
   echo "$label $(grep -oE '"(round_ms_median|ms_per_token|readout_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   check "$label" "$rc" "$mx" "$bad"
}

scripts/progress llm-tile-ladder STARTING "llm/l3/tile-ladder.sh ($PARTS): $RD"
for part in $PARTS; do
   case $part in
      synthetic)
         one S1-A2-fit "$J/axi-1p7b/A2_planar.jbp" $KEEP $FIT
         one S3-A8-fit "$J/axi-1p7b/A8_planar.jbp" $KEEP $FIT
         ;;
      135m)
         eng R1-135m-keep "$J/real-135m" 256 30 $KEEP
         eng R2-135m-fit "$J/real-135m" 256 30 $KEEP $FIT
         ;;
      1p7b)
         [ -e "$J/real-1p7b/manifest.json" ] || { echo "no 1.7B packs yet"; exit 1; }
         eng B1-1p7b-keep "$J/real-1p7b" 256 4 $KEEP
         eng B2-1p7b-fit "$J/real-1p7b" 256 4 $KEEP $FIT
         eng B3-1p7b-fit-2048only "$J/real-1p7b" 256 4 $KEEP $FIT ETNA_ML_V7_TILE_FIT_MAX_IC=2048
         eng B5-1p7b-1tok "$J/real-1p7b" 1 4 $KEEP ;;
   esac
done
scripts/progress llm-tile-ladder DONE "all rungs passed ($PARTS): $RD"
echo "ALL PASSED: $RD"
