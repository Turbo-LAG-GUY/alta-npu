#!/usr/bin/env bash
# fused-ladder.sh - the fused int8 conversion on the NPU (2026-11-20). The read-out profile
# (runs/llm-readout-profile-20261120-034535) showed the scalar -128 loop after the copy at ~0.9 of the 1.56 ms cached
# 256-token read-out. mesa-llm now flips the top bit of each byte eight bytes at a time (flip_sign_bits): straight
# from a cached output in one pass, after the bulk copy for a write-combined one, and on the way in for signed inputs
# (llm/l3/mesa-llm-fused-conversion.diff, lib fa818da9, jb_run a070d63e = jb_run-ad393a6f.c on the new libraries).
# Offline (npu-llm/runs/offline-jb-20261120-fused): NN configs, weights, input buffer content, submit streams, buffer
# lists and the buffer log (cached flags, prep / fini order) identical to the cached build; llm/l3/bench/flip_test.c:
# the helper = the old loops on 17 sizes x 3 alignments. Baseline = the cached build (jb_run ad393a6f, lib 6caca26e).
# --alt alternates two inputs and checks every invoke on both builds.
# Written 2026-11-20 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS; pause other heavy CPU jobs first.
# Stops at the first failure, STOP, an output more than 1 level off on any checked invoke, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-fused --collect llm/l3/fused-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
CA=$AI/npu-llm/bin/jb_run-ad393a6f
FU=$AI/npu-llm/bin/jb_run-a070d63e
LIBCA=$AI/npu-llm/bin/libteflon-6caca26e.so
LIBFU=$AI/npu-llm/bin/libteflon-fa818da9.so
P=$AI/npu-llm/models/jb
M=$AI/npu-llm/models/sizes
RD=$K/runs/llm-fused-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
for f in "$CA:ad393a6f" "$FU:a070d63e" "$LIBCA:6caca26e" "$LIBFU:fa818da9"; do
   [ "$(sha256sum "${f%:*}" | cut -c1-8)" = "${f##*:}" ] || { echo "${f%:*} is not ${f##*:}"; exit 1; }
done
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL KEY RC
   local label=$1 key=$2 rc=$3 mx bad w
   mx=$(grep -oE "\"$key\": [0-9]+" "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_invokes": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $label rc=$rc max_diff=${mx:-?} bad_invokes=${bad:--} kernel_warnings=$w $(grep -oE '"(ms_median_after_first|invoke_ms_median_after_first|invoke_call_ms|npu_ms|readout_ms|checked_invokes)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-99}" -gt 1 ] || [ "${bad:-0}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-fused-ladder FAILED "stopped at $label rc=$rc max_diff=${mx:-?} bad_invokes=${bad:--} kernel_warnings=$w; $RD"
      exit 1
   fi
}
jb() { # LABEL BINARY N PACK [jb_run options]
   local label=LLMFU-$1 bin=$2 n=$3 pack=$4; shift 4
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$bin" -n "$n" --wait spin "$@" "$P/$pack.jbp" > "$RD/$label.stdout" 2>&1
   check "$label" diff_vs_expected_max $?
}
teflon() { # LABEL LIB MODEL
   local label=LLMFU-$1
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- $PY -I $AI/npu-shader/teflon_check.py -m "$M/$3.tflite" \
      -r "$M/${3}_reference.npz" -e "$2" -n 200 > "$RD/$label.stdout" 2>&1
   check "$label" diff_vs_cpu_max $?
}
S17=split_17b_gateup_npu6656_g4
scripts/progress llm-fused-ladder STARTING "llm/l3/fused-ladder.sh: $RD"
# cached build first, then fused, every invoke checked on both
jb F1-cached-1x1-a53 "$CA" 200 smol_gateup_1x1 --main-cpu 0 --alt "$P/smol_gateup_1x1_alt.jbp"
jb F2-fused-1x1-a53 "$FU" 200 smol_gateup_1x1 --main-cpu 0 --alt "$P/smol_gateup_1x1_alt.jbp"
jb F3-cached-17b-share-a53 "$CA" 200 $S17 --main-cpu 0 --alt "$P/${S17}_alt.jbp"
jb F4-fused-17b-share-a53 "$FU" 200 $S17 --main-cpu 0 --alt "$P/${S17}_alt.jbp"
jb F5-cached-8x8 "$CA" 200 smol_gateup_8x8 --alt "$P/smol_gateup_8x8_alt.jbp"
jb F6-fused-8x8 "$FU" 200 smol_gateup_8x8 --alt "$P/smol_gateup_8x8_alt.jbp"
jb F7-cached-16x16 "$CA" 200 smol_gateup_16x16 --alt "$P/smol_gateup_16x16_alt.jbp"
jb F8-fused-16x16 "$FU" 200 smol_gateup_16x16 --alt "$P/smol_gateup_16x16_alt.jbp"
# longer checks on the fused build: 1000 alternating invokes, 256 tokens and the 16-output job
jb F9-fused-16x16-1000 "$FU" 1000 smol_gateup_16x16 --alt "$P/smol_gateup_16x16_alt.jbp"
jb F10-fused-g16-1000 "$FU" 1000 smol_gateup_1x1_g16 --alt "$P/smol_gateup_1x1_g16_alt.jbp"
# Teflon: cached build vs fused
teflon F11-teflon-cached-1x1 "$LIBCA" smol_gateup_1x1_P
teflon F12-teflon-fused-1x1 "$LIBFU" smol_gateup_1x1_P
teflon F13-teflon-cached-16x16 "$LIBCA" smol_gateup_16x16_P
teflon F14-teflon-fused-16x16 "$LIBFU" smol_gateup_16x16_P
scripts/progress llm-fused-ladder DONE "all 14 rungs passed: $RD"
echo "ALL PASSED: $RD"
