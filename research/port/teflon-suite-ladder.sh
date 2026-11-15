#!/usr/bin/env bash
# teflon-suite-ladder.sh - Mesa's own Teflon test suite (test_teflon) on the NPU with the UPSTREAM build
# (mesa-upstream-fixes: main 8a9009e0482 + 8 patches, lib a05298e9). Written 2026-11-15.
# Jordan's OK: "you have my OK to run the NPU when you want" (2026-11-15).
# 1568 tests = the 16 models of Mesa's npu-model-zoo cut into single operations, plus each whole model. Each test
# runs the model on the CPU (TensorFlow Lite 2.13 reference kernels) and through the delegate and compares.
# Offline (fake NPU, runs/cs-offline/teflon-suite + -asan): 777 tests send work to the NPU; no test crashes the
# driver with patch 8; the sanitizer reports one driver overread (ADD input upload, micronet_vww whole) and 40
# overreads in the test program itself (32-bit inputs, CPU side). Most shapes have never run on this board.
# One npu-attempt per family: npu-run stops the job at the first kernel hang line and sets runs/STOP, and this
# script then stops. Test failures (rc 1) are results, not a reason to stop, except in the first family
# (mobilenet_v1, the control: these operations are proven here and must all pass).
# Usage: research/port/teflon-suite-ladder.sh [FAMILY...]   (run detached: systemd-run --user --unit=alta-tfs)
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
B=$AI/mesa-upstream-fixes/build-tests/src/gallium/targets/teflon
LIB=$AI/mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so
RD=$K/runs/teflon-suite-$(date +%Y%m%d-%H%M%S)
FAMS=${*:-mobilenet_v1 mobilenet_v2 posenet_mobilenet_v1 inception_v1 ds_cnn micronet_ad_large micronet_vww wav2letter deeplabv3 ssd_mobilenet_v2 ssdlite_mobiledet ssd_mobilenet_v3 efficientdet_lite0 yolox_nano movenet_lightning movenet_thunder}
# Left out by default (TFS_EXCLUDE, gtest patterns joined with ":"):
#  - teflon/ds_cnn.0014 and ds_cnn.whole: int8 FULLY_CONNECTED 276 -> 12 HUNG the NPU 2026-11-15 07:44
#    (incidents/2026-11-15-0745-TFS-ds_cnn); run it only on purpose, last in a boot
#  - everything in Mesa's own skip list for this NPU (src/etnaviv/ci/etnaviv-vipnano-skips.txt)
SKIPS=$AI/mesa-upstream-fixes/src/etnaviv/ci/etnaviv-vipnano-skips.txt
EXCL=${TFS_EXCLUDE-teflon/ds_cnn.0014:teflon/ds_cnn.whole:$(grep '^teflon/' "$SKIPS" | paste -sd:)}
mkdir -p "$RD"; cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first"; exit 1; fi
[ "$(sha256sum "$LIB" | cut -c1-8)" = a05298e9 ] || { echo "upstream lib is not a05298e9"; exit 1; }
scripts/progress upstream-teflon-suite STARTING "research/port/teflon-suite-ladder.sh: $RD; families: $FAMS; excluded: ${EXCL:-none}"
for fam in $FAMS; do
   label=TFS-$fam
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM -u TEFLON_DEBUG \
      NPU_DEADLINE=900 LD_LIBRARY_PATH=$AI/tflite-2.13/lib TEFLON_TEST_DATA=$B/tests TEFLON_TEST_DELEGATE=$LIB \
      scripts/npu-attempt "$label" -- "$B/test_teflon" --gtest_filter="teflon/$fam.*${EXCL:+-$EXCL}" --gtest_brief=0 \
      > "$RD/$fam.stdout" 2>&1
   rc=$?
   ok=$(grep -c '^\[       OK \]' "$RD/$fam.stdout"); bad=$(grep -c '^\[  FAILED  \] teflon/.* (' "$RD/$fam.stdout")
   echo "== $label rc=$rc passed=$ok failed=$bad"
   if [ -e runs/STOP ] || [ $rc -eq 90 ] || [ $rc -eq 124 ] || [ $rc -eq 77 ] || [ $rc -eq 75 ]; then
      echo "STOPPED at $label (see $RD/$fam.stdout and PROGRESS.md)"
      scripts/progress upstream-teflon-suite FAILED "stopped at $label rc=$rc; $RD"
      exit 1
   fi
   if [ "$fam" = mobilenet_v1 ] && [ $rc -ne 0 ]; then
      echo "STOPPED: the control family did not pass (rc=$rc, failed=$bad)"
      scripts/progress upstream-teflon-suite FAILED "control family mobilenet_v1 rc=$rc passed=$ok failed=$bad; $RD"
      exit 1
   fi
done
scripts/progress upstream-teflon-suite DONE "all families ran without a hang: $RD"
echo "ALL RAN: $RD"
