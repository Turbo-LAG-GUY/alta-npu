#!/usr/bin/env bash
# ladder-L4.sh - upscaler L3 rerun with the fixed checker, then program priority (hw_job_limit L4), 2026-11-21.
# Jordan's OK 2026-11-21: "Run the next steps for me".
#   A  upscaler hist2_n8_planar, 100 invokes/thread: L4-UP-1T-L1 (save), L4-UP-2T-L1, L4-UP-2T-L4
#   B  limit 4, MobileNet, two programs at once for 5 s (prio_pair.sh, one npu-attempt job):
#      hwjl_hi 1 thread, hwjl_lo 3 threads.
#      L4-MN-SAVE (1 thread, saves the reference output), L4-NOPRIO, L4-PRIO (hi=high, lo=low, guard 100),
#      L4-PRIO-G0 (guard 0), L4-TURNS (no priority, turns on)
# Stops at a failure, STOP or a trip. Always restores: limit 1, hw_job_prio empty, guard 100, turns off.
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
H=$K/research/hw-job-limit
LIB=$AI/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so
P=/sys/module/etnaviv/parameters
RD=$K/runs/hwjl-l4-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD/cache"
cd "$K" || exit 1

restore() { echo 1 > $P/hw_job_limit; echo "" > $P/hw_job_prio; echo 100 > $P/hw_job_prio_guard_ms; echo N > $P/hw_job_turns
            echo "restored: limit $(cat $P/hw_job_limit) prio [$(cat $P/hw_job_prio)] guard $(cat $P/hw_job_prio_guard_ms) turns $(cat $P/hw_job_turns)"; }
fail() { echo "STOPPED: $*"; scripts/progress hwjl-l4 FAILED "$*; $RD"; restore; exit 1; }
[ -e runs/STOP ] && { echo "runs/STOP is set"; exit 1; }
[ "$(pgrep -c -i steam)" != 0 ] && { echo "Steam is running: close it first"; exit 1; }
[ "$(cat $P/hw_job_trips)" = 0 ] || { echo "hw_job_trips is not 0 this boot"; exit 1; }

after() { # LABEL RC
   [ "$2" -eq 0 ] || fail "$1 rc=$2"
   [ -e runs/STOP ] && fail "$1: runs/STOP appeared"
   [ "$(cat $P/hw_job_trips)" = 0 ] || fail "$1: hw_job_trips went to $(cat $P/hw_job_trips)"
}
nenv() { env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_ML_AXI_SRAM -u ETNA_CS_BARRIER_REG_BUDGET "$@"; }
pick() { python3 -c "
import json,sys
t=open(sys.argv[1]).read(); j=json.loads(t[t.index('{'):t.rindex('}')+1])
print(*(f'{k}={j.get(k)}' for k in sys.argv[2:]))" "$@"; }

one() { # LABEL LIMIT THREADS N MODEL REF ARGS...
   local l=$1 lim=$2 t=$3 n=$4 m=$5 r=$6; shift 6
   echo $lim > $P/hw_job_limit
   nenv ETNA_ML_DISK_CACHE_DIR="$RD/cache" scripts/npu-attempt "$l" -- $PY -I $H/pipeline_check.py -m "$m" -r "$r" \
      -e $LIB -t $t -n $n "$@" > "$RD/$l.stdout" 2>&1; local rc=$?
   echo "== $l limit=$lim threads=$t: $(pick "$RD/$l.stdout" ms_per_invoke_throughput invoke_ms_median bytes_differing_from_saved_total match 2>&1)"
   after "$l" $rc
}
pair() { # LABEL PRIO GUARD TURNS
   local l=$1
   echo 4 > $P/hw_job_limit; echo "$2" > $P/hw_job_prio; echo "$3" > $P/hw_job_prio_guard_ms; echo "$4" > $P/hw_job_turns
   mkdir -p "$RD/$l"
   nenv LIB=$LIB scripts/npu-attempt "$l" -- $H/prio_pair.sh "$RD/$l" 5 "$MN" "$MNR" 1 3 --compare "$RD/mn.npy" \
      > "$RD/$l.stdout" 2>&1; local rc=$?
   echo "== $l prio=[$2] guard=$3 turns=$4"
   for w in hi lo; do echo "   $w: $(pick "$RD/$l/$w.json" invokes invokes_per_s invoke_ms_median invoke_ms_p90 invoke_ms_max bytes_differing_from_saved_total match 2>&1)"; done
   echo "   clients during: $(tr '\n' ';' < "$RD/$l/clients-during.txt")"
   after "$l" $rc
}

U=$AI/npu-upscale/models
UP=$U/hist2_n8_planar_960x540.tflite; UPR=$U/hist2_n8_planar_reference_960x540.npz
MN=$K/models/mobilenet_v1_mesa.tflite; MNR=$AI/npu-shader/models/mobilenet_v1_mesa_reference.npz

scripts/progress hwjl-l4 STARTING "research/hw-job-limit/ladder-L4.sh: $RD"
one L4-UP-1T-L1 1 1 100 "$UP" "$UPR" --save "$RD/up.npy"
one L4-UP-2T-L1 1 2 100 "$UP" "$UPR" --compare "$RD/up.npy"
one L4-UP-2T-L4 4 2 100 "$UP" "$UPR" --compare "$RD/up.npy"
one L4-MN-SAVE 1 1 20 "$MN" "$MNR" --save "$RD/mn.npy"
pair L4-NOPRIO   ""                          100 N
pair L4-PRIO     "hwjl_hi=high,hwjl_lo=low"  100 N
pair L4-PRIO-G0  "hwjl_hi=high,hwjl_lo=low"  0   N
pair L4-TURNS    ""                          100 Y
restore
rm -rf "$RD/cache"
scripts/progress hwjl-l4 DONE "all rungs passed: $RD (see runs/hwjl-l4.out)"
echo "all rungs passed: $RD"
