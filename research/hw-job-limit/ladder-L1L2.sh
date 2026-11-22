#!/usr/bin/env bash
# ladder-L1L2.sh - first NPU jobs on the job-queue driver (patches 0006 + 0007, root step 250), 2026-11-21.
# Jordan's OK 2026-11-21: "Go for it" (L1, then L2 if it passes).
#
# L1: known-good jobs at hw_job_limit 1 (the boot value): results must match the CPU and the old driver's numbers.
# L2: the same jobs after raising the limit while running, to 2 and then 4: outputs byte-identical to L1's.
#     These programs wait for each job, so the deeper queue is not expected to change timing; this proves the
#     credit callback and runtime changes are safe, not that they are faster (that is L3).
#   L1-UP8   upscaler hist2_n8_planar 960x540, 10 invokes, saves its output    (width ladder W-REAL8: 13.5 ms, diff 1)
#   L1-MN    MobileNet v1 (Mesa docs model), 10 invokes, saves its output
#   L2-UP8-2 / L2-MN-2   at limit 2, --compare to L1
#   L2-UP8-4 / L2-MN-4   at limit 4, --compare to L1
# Every rung: one scripts/npu-attempt job. Before and after it the hw_job_* parameters are recorded; the ladder
# stops at a failed job, STOP, a new hw_job_trips, or a limit that is not the one set. It always puts the limit
# back to 1 at the end.
# Usage: research/hw-job-limit/ladder-L1L2.sh   (run detached: systemd-run --user --unit=alta-hwjl-ladder --collect)
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
CHECK=$AI/npu-shader/teflon_check.py
LIB=$AI/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so
U=$AI/npu-upscale/models
P=/sys/module/etnaviv/parameters
RD=$K/runs/hwjl-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD/cache"
cd "$K" || exit 1

params() { for f in hw_job_limit hw_job_auto hw_job_turns hw_job_trips hw_job_trip_from hw_job_prio_guard_ms; do
             printf '%s=%s ' "$f" "$(cat $P/$f)"; done; printf 'clients=[%s]\n' "$(tr '\n' ';' < $P/hw_job_clients)"; }
finish() { echo 1 > $P/hw_job_limit; echo "limit back to $(cat $P/hw_job_limit)"; }
fail() { echo "STOPPED: $*"; scripts/progress hwjl-ladder FAILED "$*; $RD"; finish; exit 1; }

[ -e runs/STOP ] && { echo "runs/STOP is set: clear it (after review) first"; exit 1; }
[ "$(pgrep -c -i steam)" != 0 ] && { echo "Steam is running: close it first"; exit 1; }
[ -w $P/hw_job_limit ] || { echo "no write access to $P/hw_job_limit (root step 250)"; exit 1; }
[ "$(cat $P/hw_job_trips)" = 0 ] || { echo "hw_job_trips is not 0 this boot: review first"; exit 1; }

set_limit() { echo "$1" > $P/hw_job_limit || fail "could not set hw_job_limit $1"
              [ "$(cat $P/hw_job_limit)" = "$1" ] || fail "hw_job_limit reads $(cat $P/hw_job_limit), not $1"
              echo "== hw_job_limit set to $1"; }

rung() { # LABEL LIMIT MODEL REFERENCE [--save F | --compare F]
   local label=$1 limit=$2 model=$3 ref=$4; shift 4
   echo "$label before: $(params)" >> "$RD/params.log"
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_ML_AXI_SRAM -u ETNA_CS_BARRIER_REG_BUDGET \
      ETNA_ML_DISK_CACHE_DIR="$RD/cache" \
      scripts/npu-attempt "$label" -- $PY -I $CHECK -m "$model" -r "$ref" -e $LIB -n 10 "$@" \
      > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "$label after:  $(params)" >> "$RD/params.log"
   echo "== $label (limit $limit) rc=$rc"
   grep -E 'invoke_ms_median"|all_invokes|diff_vs_cpu_max|"match"|compare' "$RD/$label.stdout" | head -6
   [ $rc -eq 0 ] || fail "$label rc=$rc"
   [ -e runs/STOP ] && fail "$label: runs/STOP appeared"
   [ "$(cat $P/hw_job_trips)" = 0 ] || fail "$label: hw_job_trips went to $(cat $P/hw_job_trips)"
   [ "$(cat $P/hw_job_limit)" = "$limit" ] || fail "$label: hw_job_limit is $(cat $P/hw_job_limit), not $limit"
}

UP=$U/hist2_n8_planar_960x540.tflite;   UPR=$U/hist2_n8_planar_reference_960x540.npz
MN=$K/models/mobilenet_v1_mesa.tflite; MNR=$AI/npu-shader/models/mobilenet_v1_mesa_reference.npz

scripts/progress hwjl-ladder STARTING "research/hw-job-limit/ladder-L1L2.sh: $RD"
set_limit 1
rung L1-UP8   1 "$UP" "$UPR" --save "$RD/up8.npy"
rung L1-MN    1 "$MN" "$MNR" --save "$RD/mn.npy"
set_limit 2
rung L2-UP8-2 2 "$UP" "$UPR" --compare "$RD/up8.npy"
rung L2-MN-2  2 "$MN" "$MNR" --compare "$RD/mn.npy"
set_limit 4
rung L2-UP8-4 4 "$UP" "$UPR" --compare "$RD/up8.npy"
rung L2-MN-4  4 "$MN" "$MNR" --compare "$RD/mn.npy"
finish
rm -rf "$RD/cache"
scripts/progress hwjl-ladder DONE "all 6 rungs passed (limits 1, 2, 4; outputs byte-identical): $RD"
echo "all rungs passed: $RD"
