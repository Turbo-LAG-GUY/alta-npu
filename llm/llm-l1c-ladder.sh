#!/usr/bin/env bash
# llm-l1c-ladder.sh - LLM engine step L1c: (1) cost of one NN job vs tokens M as a 2D image (1x1 ... 16x16),
# (2) CPU + NPU reading memory at the same time. Written 2026-11-19 (CPU only). JORDAN'S OK: 2026-11-19.
# Models: npu-llm/models/sizes (llm/tools/llm_sizes.py): SmolLM2 qkv/o/gateup/down + Qwen3 gateup, 8 sizes, planar.
# Offline (fake NPU, npu-llm/runs/offline-l1c-20261119): all 40 = one NN job, no driver error, tile = image.
# Order: sizes ascending (all shapes at 1x1, then 1x2, ...), stops at the first failure or STOP.
# Part 2 (bandwidth): CPU alone (npu-llm/bin/bw_loop on cpu 2-5), NPU alone (qwen_gateup_1x1_P x2000 on cpu 0-1),
# then both at once; the CPU passes inside the NPU run's window give the shared-memory answer.
# Usage: systemd-run --user --unit=alta-llm-l1c --collect llm/llm-l1c-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
CHECK=$AI/npu-shader/teflon_check.py
LIB=$AI/mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so
M=$AI/npu-llm/models/sizes
BW=$AI/npu-llm/bin/bw_loop
RD=$K/runs/llm-l1c-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$LIB" | cut -c1-8)" = b6c2a28d ] || { echo "upstream lib is not b6c2a28d"; exit 1; }

run() { # NAME ITERATIONS [PREFIX...]
   local n=$1 it=$2 label="LLM1C-$1${TAG:-}"; shift 2
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM \
      scripts/npu-attempt "$label" -- "$@" $PY -I $CHECK -m $M/$n.tflite -r $M/${n}_reference.npz -e $LIB -n $it \
      --save "$RD/$n.npy" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc $(grep -oE '"invoke_ms_median_after_first": [0-9.]+' "$RD/$label.stdout")"
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-l1c-ladder FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
}
scripts/progress llm-l1c-ladder STARTING "llm/llm-l1c-ladder.sh: $RD"
for size in 1x1 1x2 2x2 2x4 4x4 4x8 8x8 16x16; do
   for s in smol_o smol_qkv smol_down smol_gateup qwen_gateup; do
      run ${s}_${size}_P 20
   done
done
# part 2: shared memory bandwidth
$BW 8 > "$RD/bw-cpu-alone.txt"
echo "== CPU alone: $(awk '{s+=$2; n++} END {printf "%.2f GB/s (%d passes)", s/n, n}' "$RD/bw-cpu-alone.txt")"
TAG=-alone run qwen_gateup_1x1_P 2000 taskset -c 0,1
$BW 60 > "$RD/bw-cpu-both.txt" &
bwpid=$!
sleep 2
t0=$(cut -d' ' -f1 /proc/uptime)
TAG=-both run qwen_gateup_1x1_P 2000 taskset -c 0,1
t1=$(cut -d' ' -f1 /proc/uptime)
kill $bwpid 2>/dev/null; wait $bwpid 2>/dev/null
echo "$t0 $t1" > "$RD/npu-window.txt"
echo "== CPU during the NPU run (whole npu-attempt window): $(awk -v a=$t0 -v b=$t1 '$1>a && $1<b {s+=$2; n++} END {if (n) printf "%.2f GB/s (%d passes)", s/n, n}' "$RD/bw-cpu-both.txt")"
scripts/progress llm-l1c-ladder DONE "all 40 size rungs + bandwidth part passed: $RD"
echo "ALL PASSED: $RD"
