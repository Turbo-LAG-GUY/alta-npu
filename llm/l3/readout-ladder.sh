#!/usr/bin/env bash
# readout-ladder.sh - the Mesa read-out fix on the NPU (2026-11-20). Mesa's etna_ml_subgraph_read_outputs read signed
# int8 outputs one byte at a time straight from the write-combined (uncached) tensor buffer: ~150 ns per byte on an
# A53, ~35 on an A73 (split ladders: 1.04 ms for 6.6 KB). mesa-llm now copies each output out in one bulk read and
# converts it in cached memory (llm/l3/mesa-llm-readout-kpcclamp.diff; lib a26f0852, also clamps the L3 KPC hook to
# the hardware cap 21). Old = jb_run 3fa43e81 (built on lib 678775c5) / npu-llm/bin/libteflon-678775c5.so;
# new = jb_run baca6a17 (lib a26f0852, adds readout_ms) / mesa-llm libteflon a26f0852.
# Same NPU jobs as before: offline (npu-llm/runs/offline-jb-20261120-readout) the 1-token job is unchanged and the 64-
# and 256-token jobs are byte-identical (NN config, weights, submit stream, buffer list) to the L1c jobs that PASSED
# on the NPU 2026-11-19; the 1.7B share is the split-ladder job. Only the CPU-side copy after the job changes.
# Written 2026-11-20 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS; pause other heavy CPU jobs first.
# Stops at the first failure, STOP, or an output more than 1 level off.
# Usage: systemd-run --user --unit=alta-llm-readout --collect llm/l3/readout-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
OLD=$AI/npu-llm/bin/jb_run-3fa43e81
NEW=$AI/npu-llm/bin/jb_run-baca6a17
LIBOLD=$AI/npu-llm/bin/libteflon-678775c5.so
LIBNEW=$AI/mesa-llm/build/src/gallium/targets/teflon/libteflon.so
P=$AI/npu-llm/models/jb
M=$AI/npu-llm/models/sizes
RD=$K/runs/llm-readout-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
for f in "$OLD:3fa43e81" "$NEW:baca6a17" "$LIBOLD:678775c5" "$LIBNEW:a26f0852"; do
   [ "$(sha256sum "${f%:*}" | cut -c1-8)" = "${f##*:}" ] || { echo "${f%:*} is not ${f##*:}"; exit 1; }
done
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)

check() { # LABEL KEY RC
   local label=$1 key=$2 rc=$3 mx
   mx=$(grep -oE "\"$key\": [0-9]+" "$RD/$label.stdout" | grep -oE '[0-9]+$')
   echo "== $label rc=$rc max_diff=${mx:-?} $(grep -oE '"(ms_median_after_first|invoke_ms_median_after_first|read_ms|npu_ms|readout_ms|main_on_a73)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-99}" -gt 1 ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-readout-ladder FAILED "stopped at $label rc=$rc max_diff=${mx:-?}; $RD"
      exit 1
   fi
}
jb() { # LABEL BINARY PACK [jb_run options]
   local label=LLMRO-$1 bin=$2 pack=$3; shift 3
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$bin" -n 200 --wait spin "$@" "$P/$pack.jbp" > "$RD/$label.stdout" 2>&1
   check "$label" diff_vs_expected_max $?
}
teflon() { # LABEL LIB MODEL
   local label=LLMRO-$1
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- $PY -I $AI/npu-shader/teflon_check.py -m "$M/$3.tflite" \
      -r "$M/${3}_reference.npz" -e "$2" -n 200 > "$RD/$label.stdout" 2>&1
   check "$label" diff_vs_cpu_max $?
}
S17=split_17b_gateup_npu6656_g4
scripts/progress llm-readout-ladder STARTING "llm/l3/readout-ladder.sh: $RD"
# one token: 3072 B out (main thread on an A53), the 1.7B split share: 4 outputs, 6656 B (A53)
jb U1-old-1x1-a53 "$OLD" smol_gateup_1x1 --main-cpu 0
jb U2-new-1x1-a53 "$NEW" smol_gateup_1x1 --main-cpu 0
jb U3-old-17b-share-a53 "$OLD" $S17 --main-cpu 0
jb U4-new-17b-share-a53 "$NEW" $S17 --main-cpu 0
# many tokens (prompt chunks): 64 tokens = 196 KB out, 256 tokens = 786 KB out (unpinned)
jb U5-old-8x8 "$OLD" smol_gateup_8x8
jb U6-new-8x8 "$NEW" smol_gateup_8x8
jb U7-old-16x16 "$OLD" smol_gateup_16x16
jb U8-new-16x16 "$NEW" smol_gateup_16x16
# through Teflon (what every signed int8 Teflon model pays), old vs new library
teflon U9-teflon-old-1x1 "$LIBOLD" smol_gateup_1x1_P
teflon U10-teflon-new-1x1 "$LIBNEW" smol_gateup_1x1_P
teflon U11-teflon-old-16x16 "$LIBOLD" smol_gateup_16x16_P
teflon U12-teflon-new-16x16 "$LIBNEW" smol_gateup_16x16_P
scripts/progress llm-readout-ladder DONE "all 12 rungs passed: $RD"
echo "ALL PASSED: $RD"
