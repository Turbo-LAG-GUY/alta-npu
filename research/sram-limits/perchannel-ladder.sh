#!/usr/bin/env bash
# sram-limits/perchannel-ladder.sh (2026-11-22): column strips with one NN job per output channel in the last layer,
# ON THE NPU. Written CPU only. NEEDS JORDAN'S OK BEFORE IT RUNS. Message the other NPU sessions first; Steam closed.
#
# Why: the S3 diagnostic proved the NN unit spaces output planes by out_image_x_size * y_size and ignores the strides
# (s3_layout_check.py, 100% of bytes), so a column strip now writes each output channel with its own z = 1 job.
# Library: mesa-strips libteflon.so.3d1c6999-perchannel (= e962cc17 + the split). Offline PASS
# (research/sram-limits/offline-perchannel-*.VERDICT.txt): write_map_check.py shows every output byte written once
# in the right place (64x64, and both 960x540 models); everything without column strips byte-identical to e962cc17.
#
# P1 hist2_n8_planar_64x64, ETNA_ML_STRIPS=32 + COLUMNS=1: the S3 job (no hang in run 2, wrong output then)
# P2 hist2_n8_planar_960x540 without strips: same-session baseline (proven config, ~14 ms in November)
# P3 hist2_n8_planar_960x540, ETNA_ML_STRIPS=auto + COLUMNS=1: 9 strips of 107 columns, the real target
# P4 hist2_n16_planar_960x540 without strips: baseline (~28 ms in November)
# P5 hist2_n16_planar_960x540, ETNA_ML_STRIPS=auto + COLUMNS=1: 17 strips of 57 columns
# 20 invokes each through npu-attempt, every invoke checked against the CPU (teflon_check.py). Stops at the first
# failure, STOP, mismatch or kernel WARNING. A hang is recovered by the kernel, but npu-run then refuses NPU jobs
# until a reboot: grab the devcoredump (alta-devcd-watch saves it) into the incident folder first.
# Usage: systemd-run --user --unit=alta-sram-perchannel-npu --collect research/sram-limits/perchannel-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
H=$K/research/sram-limits
PY=$AI/.venv/bin/python
CHK=$AI/npu-shader/teflon_check.py
STR=$AI/mesa-strips/build/src/gallium/targets/teflon/libteflon.so.3d1c6999-perchannel
MD=$AI/npu-upscale/models
RD=$K/runs/sram-perchannel-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$STR" | cut -c1-8)" = 3d1c6999 ] || { echo "strips library is not 3d1c6999"; exit 1; }
[ "$(cat /sys/module/etnaviv/parameters/axi_sram)" = Y ] || { echo "etnaviv axi_sram is off"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
ls $H/offline-perchannel-*.VERDICT.txt >/dev/null 2>&1 || { echo "no offline verdict"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM -u ETNA_ML_STRIPS
       -u ETNA_ML_STRIPS_COLUMNS -u ETNA_ML_STRIPS_VERBOSE -u ETNA_ML_DISK_CACHE)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

job() { # LABEL LIBRARY MODEL REFERENCE ENV...
   local label=PERCH-$1 lib=$2 model=$3 ref=$4 rc w m
   shift 4
   "${CLEAN[@]}" ETNA_ML_AXI_SRAM=all ETNA_ML_DISK_CACHE=0 "$@" scripts/npu-attempt "$label" -- \
      "$PY" -I "$CHK" -m "$model" -r "$ref" -e "$lib" -n 20 > "$RD/$label.stdout" 2>&1
   rc=$?
   w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   m=$(grep -oE '"match": (true|false)' "$RD/$label.stdout" | grep -oE '(true|false)$')
   grep -E 'AXI SRAM|strips: [0-9]' "$RD/$label.stdout"
   grep -oE '"(invoke_ms_median_after_first|diff_vs_cpu_max|diff_vs_cpu_mean_worst|all_invokes_identical)": [a-z0-9.]+' \
      "$RD/$label.stdout" | tr '\n' ' '; echo
   echo "== $label rc=$rc match=${m:-?} kernel_warnings=$w"
   if [ "$rc" -ne 0 ] || [ "${m:-}" != true ] || [ -e runs/STOP ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress sram-perchannel-ladder FAILED "stopped at $label rc=$rc match=${m:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

scripts/progress sram-perchannel-ladder STARTING "research/sram-limits/perchannel-ladder.sh: $RD"
COLS=(ETNA_ML_STRIPS_COLUMNS=1 ETNA_ML_STRIPS_VERBOSE=1)
job P1-c64-columns "$STR" "$MD/hist2_n8_planar_64x64.tflite" "$MD/hist2_n8_planar_reference_64x64.npz" ETNA_ML_STRIPS=32 "${COLS[@]}"
job P2-n8-540-whole "$STR" "$MD/hist2_n8_planar_960x540.tflite" "$MD/hist2_n8_planar_reference_960x540.npz"
job P3-n8-540-columns "$STR" "$MD/hist2_n8_planar_960x540.tflite" "$MD/hist2_n8_planar_reference_960x540.npz" ETNA_ML_STRIPS=auto "${COLS[@]}"
job P4-n16-540-whole "$STR" "$MD/hist2_n16_planar_960x540.tflite" "$MD/hist2_n16_planar_reference_960x540.npz"
job P5-n16-540-columns "$STR" "$MD/hist2_n16_planar_960x540.tflite" "$MD/hist2_n16_planar_reference_960x540.npz" ETNA_ML_STRIPS=auto "${COLS[@]}"
scripts/progress sram-perchannel-ladder DONE "all rungs passed: $RD"
