#!/usr/bin/env bash
# zero-ladder.sh - zero-copy read-out on the NPU (2026-11-20, LLM engine R3). After patches 10-12 the 256-token
# read-out is ~0.54 ms: one fused pass (copy + int8 conversion) plus the kernel's cache maintenance (~0.4 ms,
# runs/llm-readout-profile-20261120-034535). Our own runtime can skip the pass and read the NPU's bytes in place:
# mesa-llm test-tree hook etna_ml_subgraph_map_output / _unmap_output (llm/l3/mesa-llm-zerocopy-hook.diff, not for
# upstream), jb_run a79b7b5c --readout copy|zero|zero-nofini. zero-nofini also leaves out CPU_FINI (the kernel's
# clean pass): an experiment, correct only while the CPU never writes to the output buffer.
# Fair comparison: --consume reads every int8 value once inside the timed read-out (what the engine's next CPU step
# does anyway); copy reads the converted copy, zero reads the raw bytes and flips them in the register.
# Every invoke is checked (--alt: two inputs alternate), so a stale read in the no-fini experiment shows up.
# Offline (npu-llm/runs/offline-jb-20261120-zerocopy): submit streams identical in all three modes, consumer sums
# identical, zero pairs every CPU_PREP with a CPU_FINI, zero-nofini leaves one prep per invoke (no fini without prep).
# Written 2026-11-20 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS; pause other heavy CPU jobs first.
# Stops at the first failure, STOP, an output more than 1 level off on any checked invoke, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-zero --collect llm/l3/zero-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JB=$AI/npu-llm/bin/jb_run-a79b7b5c
P=$AI/npu-llm/models/jb
RD=$K/runs/llm-zero-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JB" | cut -c1-8)" = a79b7b5c ] || { echo "jb_run is not a79b7b5c"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

jb() { # LABEL MODE N PACK [jb_run options]
   local label=LLMZC-$1 mode=$2 n=$3 pack=$4 rc mx bad w; shift 4
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JB" -n "$n" --wait spin --readout "$mode" --consume \
      --alt "$P/${pack}_alt.jbp" "$@" "$P/$pack.jbp" > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_invokes": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $label rc=$rc max_diff=${mx:-?} bad_invokes=${bad:-?} kernel_warnings=$w $(grep -oE '"(ms_median_after_first|npu_ms|readout_ms|consume_sum|checked_invokes)": -?[0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-99}" -gt 1 ] || [ "${bad:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-zero-ladder FAILED "stopped at $label rc=$rc max_diff=${mx:-?} bad_invokes=${bad:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}
S17=split_17b_gateup_npu6656_g4
scripts/progress llm-zero-ladder STARTING "llm/l3/zero-ladder.sh: $RD"
# one token and the 1.7B split share, main thread on an A53
jb Z1-copy-1x1-a53 copy 200 smol_gateup_1x1 --main-cpu 0
jb Z2-zero-1x1-a53 zero 200 smol_gateup_1x1 --main-cpu 0
jb Z3-nofini-1x1-a53 zero-nofini 200 smol_gateup_1x1 --main-cpu 0
jb Z4-copy-17b-share-a53 copy 200 $S17 --main-cpu 0
jb Z5-zero-17b-share-a53 zero 200 $S17 --main-cpu 0
jb Z6-nofini-17b-share-a53 zero-nofini 200 $S17 --main-cpu 0
# many tokens
jb Z7-copy-8x8 copy 200 smol_gateup_8x8
jb Z8-zero-8x8 zero 200 smol_gateup_8x8
jb Z9-copy-16x16 copy 200 smol_gateup_16x16
jb Z10-zero-16x16 zero 200 smol_gateup_16x16
jb Z11-nofini-16x16 zero-nofini 200 smol_gateup_16x16
# long stale-read checks
jb Z12-zero-16x16-1000 zero 1000 smol_gateup_16x16
jb Z13-nofini-16x16-1000 zero-nofini 1000 smol_gateup_16x16
jb Z14-zero-g16-1000 zero 1000 smol_gateup_1x1_g16
scripts/progress llm-zero-ladder DONE "all 14 rungs passed: $RD"
echo "ALL PASSED: $RD"
