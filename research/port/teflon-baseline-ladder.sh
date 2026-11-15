#!/usr/bin/env bash
# teflon-baseline-ladder.sh - rerun the tests that FAILED in a suite run (teflon-suite-ladder.sh, series library)
# with the UNPATCHED upstream library (main 8a9009e0482, libteflon.so.base-8a9009e0, sha 385f828f), to see which
# failures the series causes and which are already in upstream. Written 2026-11-15.
# Jordan's OK: "run the baseline comparison after the suite finishes" (2026-11-15).
# Left out: tests in Mesa's expected-failure list for this NPU (src/etnaviv/ci/etnaviv-vipnano-fails.txt) unless
# TFB_ALL=1, and whatever TFB_EXCLUDE names (default: the two ds_cnn FULLY_CONNECTED tests that hung the NPU).
# Risk: the baseline has the bugs the series fixes (W = 1 / H > W tiling hangs, out-of-bounds reads). The tests
# rerun here are ones the series library ran without a hang, but the baseline may behave differently.
# One npu-attempt per family (label TFB-<family>); stops at a hang (rc 90 / STOP), deadline or refusal.
# Usage: research/port/teflon-baseline-ladder.sh SUITE_RUN_DIR...      (run detached: systemd-run --user)
# Result: RD/compare.tsv: test, series result (FAILED), baseline result (OK / FAILED / not run)
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
B=$AI/mesa-upstream-fixes/build-tests/src/gallium/targets/teflon
LIB=$AI/mesa-upstream-fixes/build/src/gallium/targets/teflon/libteflon.so.base-8a9009e0
KNOWN=$AI/mesa-upstream-fixes/src/etnaviv/ci/etnaviv-vipnano-fails.txt
EXCL=${TFB_EXCLUDE-teflon/ds_cnn.0014 teflon/ds_cnn.whole}
RD=$K/runs/teflon-baseline-$(date +%Y%m%d-%H%M%S)
[ $# -ge 1 ] || { echo "usage: $0 SUITE_RUN_DIR..."; exit 1; }
mkdir -p "$RD"; cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first"; exit 1; fi
[ "$(sha256sum "$LIB" | cut -c1-8)" = 385f828f ] || { echo "baseline lib is not 385f828f"; exit 1; }

# failed tests of the given runs, one per line (the latest run of a family wins is not needed: union)
for d in "$@"; do grep -h '^\[  FAILED  \] teflon/.* (' "$d"/*.stdout 2>/dev/null; done |
   sed 's/^\[  FAILED  \] \(teflon\/[^ ]*\) .*/\1/' | sort -u > "$RD/failed-series.txt"
: > "$RD/todo.txt"
while read -r t; do
   if [ -z "${TFB_ALL:-}" ] && grep -q "^$t,Fail" "$KNOWN"; then continue; fi
   case " $EXCL " in *" $t "*) continue;; esac
   echo "$t" >> "$RD/todo.txt"
done < "$RD/failed-series.txt"
echo "failed with the series: $(wc -l < "$RD/failed-series.txt"); to rerun with the baseline: $(wc -l < "$RD/todo.txt")"
scripts/progress upstream-teflon-baseline STARTING "research/port/teflon-baseline-ladder.sh: $RD; $(wc -l < "$RD/todo.txt") tests from: $*"

stopped=
for fam in $(sed 's/^teflon\/\(.*\)\.[^.]*$/\1/' "$RD/todo.txt" | sort -u); do
   filter=$(grep "^teflon/$fam\." "$RD/todo.txt" | paste -sd:)
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM -u TEFLON_DEBUG \
      NPU_DEADLINE=900 LD_LIBRARY_PATH=$AI/tflite-2.13/lib TEFLON_TEST_DATA=$B/tests TEFLON_TEST_DELEGATE=$LIB \
      scripts/npu-attempt "TFB-$fam" -- "$B/test_teflon" --gtest_filter="$filter" --gtest_brief=0 \
      > "$RD/$fam.stdout" 2>&1
   rc=$?
   ok=$(grep -c '^\[       OK \]' "$RD/$fam.stdout"); bad=$(grep -c '^\[  FAILED  \] teflon/.* (' "$RD/$fam.stdout")
   echo "== TFB-$fam rc=$rc passed=$ok failed=$bad"
   if [ -e runs/STOP ] || [ $rc -eq 90 ] || [ $rc -eq 124 ] || [ $rc -eq 77 ] || [ $rc -eq 75 ]; then
      echo "STOPPED at TFB-$fam (see $RD/$fam.stdout and PROGRESS.md)"; stopped="TFB-$fam rc=$rc"; break
   fi
done

while read -r t; do
   if grep -qh "^\[       OK \] $t (" "$RD"/*.stdout 2>/dev/null; then b=OK
   elif grep -qh "^\[  FAILED  \] $t (" "$RD"/*.stdout 2>/dev/null; then b=FAILED
   else b="not run"; fi
   printf '%s\tFAILED\t%s\n' "$t" "$b"
done < "$RD/todo.txt" > "$RD/compare.tsv"
n_ok=$(grep -c "	OK$" "$RD/compare.tsv"); n_bad=$(grep -c "	FAILED$" "$RD/compare.tsv")
echo "baseline passes $n_ok of the series' failures, fails $n_bad too: $RD/compare.tsv"
if [ -n "$stopped" ]; then
   scripts/progress upstream-teflon-baseline FAILED "stopped at $stopped; baseline OK $n_ok, FAILED $n_bad; $RD"; exit 1
fi
scripts/progress upstream-teflon-baseline DONE "baseline OK on $n_ok of the series' failures, FAILED on $n_bad; $RD"
