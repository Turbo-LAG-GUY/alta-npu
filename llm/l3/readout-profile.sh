#!/usr/bin/env bash
# readout-profile.sh (2026-11-20): perf profile of the NPU output read-out (Jordan: "profile the cached read-out with
# perf"). The same 256-token gateup job that passed the cached ladder (14/14) and the readout ladder, 400 invokes,
# cached (jb_run ad393a6f, lib 6caca26e) and write-combined (jb_run baca6a17, lib a26f0852), sampled with cpu-clock
# at 10 kHz (no CPU PMU on this board), user + kernel (root step 230: perf_event_paranoid -1, kptr_restrict 0).
# Through npu-run like every NPU job. Stops at the first failure or STOP.
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PERF=/usr/lib/linux-tools-6.8.0-149/perf
P=$AI/npu-llm/models/jb
RD=$K/runs/llm-readout-profile-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"; cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running"; exit 1; fi
for f in "$AI/npu-llm/bin/jb_run-ad393a6f:ad393a6f" "$AI/npu-llm/bin/jb_run-baca6a17:baca6a17"; do
   [ "$(sha256sum "${f%:*}" | cut -c1-8)" = "${f##*:}" ] || { echo "${f%:*} is not ${f##*:}"; exit 1; }
done
[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -le 0 ] || { echo "kernel sampling not allowed (root step 230)"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)
prof() { # LABEL BINARY
   local label=LLMPF-$1
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- $PERF record -q -e cpu-clock -F 10000 -o "$RD/$1.data" -- \
      "$2" -n 400 --wait spin "$P/smol_gateup_16x16.jbp" > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "== $label rc=$rc $(grep -oE '"(diff_vs_expected_max|ms_median_after_first|npu_ms|readout_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   if [ $rc -ne 0 ] || [ -e runs/STOP ]; then
      scripts/progress llm-readout-profile FAILED "$label rc=$rc; $RD"; exit 1
   fi
   $PERF report -q --stdio -i "$RD/$1.data" --sort dso,sym 2>/dev/null | head -25 > "$RD/$1.report.txt"
}
scripts/progress llm-readout-profile STARTING "llm/l3/readout-profile.sh: $RD"
prof cached "$AI/npu-llm/bin/jb_run-ad393a6f"
prof wc "$AI/npu-llm/bin/jb_run-baca6a17"
scripts/progress llm-readout-profile DONE "$RD"
echo "DONE: $RD"
