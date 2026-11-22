#!/usr/bin/env bash
# sram-limits/npu-ladder.sh (2026-11-22): the AXI SRAM limit fixes ON THE NPU. Written CPU only.
# NEEDS JORDAN'S OK BEFORE IT RUNS. Message the other NPU session(s) first; Steam closed; quiet CPU.
#
# Run 1 (06:12, a7b117a4/8c1912b6): S1 PASS, S2 FAILED (no hang: the shared tensor came back channels-first; Mesa
# detransposed only outputs nothing inside reads, upstream too). Fixed with a detransposed CPU copy (offline-copyfix-*).
# Libraries (built 2026-11-22, offline verdicts PASS: research/sram-limits/offline-final-*, offline-copyfix-*):
#   kit    mesa-axi-sram libteflon.so.1593e603-sram-fixes2 = kit 41b08ede + patch 0006 (shared outputs stay in RAM,
#          SRAM only when the invoke is one submit, shared tensors returned through a detransposed copy).
#          MobileNet commands byte-identical to 41b08ede offline.
#   strips mesa-strips   libteflon.so.e962cc17-sram-fixes2 = ad2c9c5c + patch 0006 + upstream series 0002 (tile no wider
#          than the image on V7). Column strips stay opt-in (ETNA_ML_STRIPS_COLUMNS=1) until S3 passes.
#
# S1 MobileNet (Mesa-docs copy 1ce11b7e), kit lib, SRAM all: same commands as the proven kit, so a smoke test
# S2 branch_exact_32x32 (new model: conv -> t1 -> conv -> conv, t1 also read by a CPU MAXIMUM), kit lib, SRAM all:
#    first NPU run of an output shared with a CPU layer (old kit aborted here, offline, rc 134). Output 0 = copy of t1
# S3 hist2_n8_planar_64x64, strips lib, ETNA_ML_STRIPS=32 ETNA_ML_STRIPS_COLUMNS=1, SRAM all: the job of incident
#    2026-11-12-1003-STRIPS-C64, now with tiles 32-34 wide instead of 64. If it hangs again, the kernel recovers
#    (it did 20 times that day) but npu-run refuses NPU jobs for the rest of the boot: grab the devcoredump, reboot.
# 20 invokes each through npu-attempt (one job each), every invoke checked against the CPU (teflon_check.py).
# Stops at the first failure, STOP, mismatch or kernel WARNING.
# Usage: systemd-run --user --unit=alta-sram-limits-npu --collect research/sram-limits/npu-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
H=$K/research/sram-limits
PY=$AI/.venv/bin/python
CHK=$AI/npu-shader/teflon_check.py
KIT=$AI/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so.1593e603-sram-fixes2
STR=$AI/mesa-strips/build/src/gallium/targets/teflon/libteflon.so.e962cc17-sram-fixes2
MD=$AI/npu-upscale/models
RD=$K/runs/sram-limits-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$KIT" | cut -c1-8)" = 1593e603 ] || { echo "kit library is not 1593e603"; exit 1; }
[ "$(sha256sum "$STR" | cut -c1-8)" = e962cc17 ] || { echo "strips library is not e962cc17"; exit 1; }
[ "$(sha256sum "$AI/mesa-axi-sram/mobilenet_v1_1.0_224_quant.tflite" | cut -c1-8)" = 1ce11b7e ] || { echo "not the Mesa-docs MobileNet"; exit 1; }
[ "$(cat /sys/module/etnaviv/parameters/axi_sram)" = Y ] || { echo "etnaviv axi_sram is off"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
ls $H/offline-final-*.VERDICT.txt >/dev/null 2>&1 || { echo "no offline verdict"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM -u ETNA_ML_STRIPS
       -u ETNA_ML_STRIPS_COLUMNS -u ETNA_ML_STRIPS_VERBOSE -u ETNA_ML_DISK_CACHE)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

job() { # LABEL LIBRARY MODEL REFERENCE ENV...
   local label=SRAMFIX-$1 lib=$2 model=$3 ref=$4 rc w m
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
      scripts/progress sram-limits-ladder FAILED "stopped at $label rc=$rc match=${m:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

scripts/progress sram-limits-ladder STARTING "research/sram-limits/npu-ladder.sh: $RD"
job S1-mobilenet "$KIT" "$AI/mesa-axi-sram/mobilenet_v1_1.0_224_quant.tflite" "$H/mobilenet_reference_grace_hopper.npz"
job S2-branch "$KIT" "$H/branch_exact_32x32.tflite" "$H/branch_exact_reference_32x32.npz"
job S3-c64-columns "$STR" "$MD/hist2_n8_planar_64x64.tflite" "$MD/hist2_n8_planar_reference_64x64.npz" \
   ETNA_ML_STRIPS=32 ETNA_ML_STRIPS_COLUMNS=1 ETNA_ML_STRIPS_VERBOSE=1
scripts/progress sram-limits-ladder DONE "all rungs passed: $RD"
