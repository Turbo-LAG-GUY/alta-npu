#!/usr/bin/env bash
# ladder-L3.sh - does a deeper NPU queue help a program that keeps jobs in flight? (hw_job_limit L3, 2026-11-21)
# Jordan's OK 2026-11-21: "Do option 1" (an offline submit-ahead test program, outside Steam, via npu-run).
#
# pipeline_check.py runs T worker threads, each with its own Teflon delegate (its own NPU device file), invoking
# back to back. With T >= 2 a job is always waiting while another runs: at hw_job_limit 1 it waits in the kernel,
# at 2+ in the NPU's ring. Throughput (wall time / invokes) is the number to compare. Every output is checked against
# the first, the CPU reference and the 1-thread run's saved output.
# Order: least new first. New on this board: two NPU clients in one process interleaving jobs (from L3-2T-L1 on),
# and more than one job in the ring at once (from L3-2T-L2 on). turns off, no priorities (boot defaults).
#   upscaler hist2_n8_planar 960x540, 100 invokes per thread:
#     L3-UP-1T-L1, L3-UP-2T-L1, L3-UP-2T-L2, L3-UP-2T-L4, L3-UP-3T-L4, L3-UP-1T-L4 (control)
#   MobileNet, 200 invokes per thread (shorter jobs, so the gap between jobs weighs more):
#     L3-MN-1T-L1, L3-MN-2T-L1, L3-MN-2T-L4
# Stops at a failed job, STOP, a new hw_job_trips or a wrong limit; puts the limit back to 1 at the end.
# Usage: research/hw-job-limit/ladder-L3.sh   (run detached: systemd-run --user --unit=alta-hwjl-l3 --collect)
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
CHECK=$K/research/hw-job-limit/pipeline_check.py
LIB=$AI/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so
P=/sys/module/etnaviv/parameters
RD=$K/runs/hwjl-l3-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD/cache"
cd "$K" || exit 1

params() { for f in hw_job_limit hw_job_turns hw_job_trips hw_job_prio_guard_ms; do printf '%s=%s ' "$f" "$(cat $P/$f)"; done
           printf 'clients=[%s]\n' "$(tr '\n' ';' < $P/hw_job_clients)"; }
finish() { echo 1 > $P/hw_job_limit; echo "limit back to $(cat $P/hw_job_limit)"; }
fail() { echo "STOPPED: $*"; scripts/progress hwjl-l3 FAILED "$*; $RD"; finish; exit 1; }

[ -e runs/STOP ] && { echo "runs/STOP is set: clear it (after review) first"; exit 1; }
[ "$(pgrep -c -i steam)" != 0 ] && { echo "Steam is running: close it first"; exit 1; }
[ -w $P/hw_job_limit ] || { echo "no write access to $P/hw_job_limit (root step 250)"; exit 1; }
[ "$(cat $P/hw_job_trips)" = 0 ] || { echo "hw_job_trips is not 0 this boot: review first"; exit 1; }
[ "$(cat $P/hw_job_turns)" = N ] || { echo "hw_job_turns is on: this ladder expects the default (off)"; exit 1; }

set_limit() { echo "$1" > $P/hw_job_limit || fail "could not set hw_job_limit $1"
              [ "$(cat $P/hw_job_limit)" = "$1" ] || fail "hw_job_limit reads $(cat $P/hw_job_limit), not $1"; }

summary() { # LABEL
   python3 - "$RD/$1.stdout" "$1" <<'EOF' >> "$RD/summary.tsv"
import json, sys
t = open(sys.argv[1]).read()
try:
    j = json.loads(t[t.index("{"):t.rindex("}") + 1])
    print("\t".join(str(v) for v in (sys.argv[2], j["threads"], j["invokes"], j["wall_ms"], j["ms_per_invoke_throughput"],
                                     j["invokes_per_s"], j["invoke_ms_median"], j["diff_vs_cpu_max"],
                                     j["bytes_differing_from_saved_total"], j["match"])))
except Exception as e:
    print(sys.argv[2], "no result:", e)
EOF
}

rung() { # LABEL LIMIT THREADS N MODEL REFERENCE [--save F | --compare F]
   local label=$1 limit=$2 threads=$3 n=$4 model=$5 ref=$6; shift 6
   set_limit "$limit"
   echo "$label before: $(params)" >> "$RD/params.log"
   env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_ML_AXI_SRAM -u ETNA_CS_BARRIER_REG_BUDGET \
      ETNA_ML_DISK_CACHE_DIR="$RD/cache" \
      scripts/npu-attempt "$label" -- $PY -I $CHECK -m "$model" -r "$ref" -e $LIB -t "$threads" -n "$n" "$@" \
      > "$RD/$label.stdout" 2>&1
   local rc=$?
   echo "$label after:  $(params)" >> "$RD/params.log"
   summary "$label"
   echo "== $label (limit $limit, $threads threads) rc=$rc: $(tail -1 "$RD/summary.tsv" | cut -f4-10)"
   [ $rc -eq 0 ] || fail "$label rc=$rc"
   [ -e runs/STOP ] && fail "$label: runs/STOP appeared"
   [ "$(cat $P/hw_job_trips)" = 0 ] || fail "$label: hw_job_trips went to $(cat $P/hw_job_trips)"
   [ "$(cat $P/hw_job_limit)" = "$limit" ] || fail "$label: hw_job_limit is $(cat $P/hw_job_limit), not $limit"
}

U=$AI/npu-upscale/models
UP=$U/hist2_n8_planar_960x540.tflite; UPR=$U/hist2_n8_planar_reference_960x540.npz
MN=$K/models/mobilenet_v1_mesa.tflite; MNR=$AI/npu-shader/models/mobilenet_v1_mesa_reference.npz
printf 'label\tthreads\tinvokes\twall_ms\tms_per_invoke\tinvokes_per_s\tinvoke_median_ms\tdiff_cpu\tbytes_vs_saved\tmatch\n' > "$RD/summary.tsv"

scripts/progress hwjl-l3 STARTING "research/hw-job-limit/ladder-L3.sh: $RD"
rung L3-UP-1T-L1 1 1 100 "$UP" "$UPR" --save "$RD/up.npy"
rung L3-UP-2T-L1 1 2 100 "$UP" "$UPR" --compare "$RD/up.npy"
rung L3-UP-2T-L2 2 2 100 "$UP" "$UPR" --compare "$RD/up.npy"
rung L3-UP-2T-L4 4 2 100 "$UP" "$UPR" --compare "$RD/up.npy"
rung L3-UP-3T-L4 4 3 100 "$UP" "$UPR" --compare "$RD/up.npy"
rung L3-UP-1T-L4 4 1 100 "$UP" "$UPR" --compare "$RD/up.npy"
rung L3-MN-1T-L1 1 1 200 "$MN" "$MNR" --save "$RD/mn.npy"
rung L3-MN-2T-L1 1 2 200 "$MN" "$MNR" --compare "$RD/mn.npy"
rung L3-MN-2T-L4 4 2 200 "$MN" "$MNR" --compare "$RD/mn.npy"
finish
rm -rf "$RD/cache"
scripts/progress hwjl-l3 DONE "all 9 rungs passed: $RD (summary.tsv)"
echo "all rungs passed: $RD"
column -t -s $'\t' "$RD/summary.tsv"
