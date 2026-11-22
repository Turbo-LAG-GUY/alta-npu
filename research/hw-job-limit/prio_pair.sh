#!/usr/bin/env bash
# prio_pair.sh - two NPU programs at once, for the priority test (hw_job_limit L4, 2026-11-21).
# Run as ONE npu-attempt job (npu-run allows one job at a time; this job is both programs).
# Usage: prio_pair.sh OUTDIR SECONDS MODEL REFERENCE HI_THREADS LO_THREADS [extra pipeline_check args]
#   starts "hwjl_lo" (LO_THREADS threads), waits until it is set up, then "hwjl_hi" (HI_THREADS), waits again,
#   samples /sys/module/etnaviv/parameters/hw_job_clients, then lets both run SECONDS at the same time.
#   Separate Teflon disk caches per program. Exit 0 only if both pass.
set -u
OUT=$1 SECS=$2 MODEL=$3 REF=$4 HI=$5 LO=$6; shift 6
PY=/home/jordan/Documents/AI_Projects/.venv/bin/python
CHECK=/home/jordan/alta-npu/research/hw-job-limit/pipeline_check.py
LIB=${LIB:-/home/jordan/Documents/AI_Projects/mesa-axi-sram/build/src/gallium/targets/teflon/libteflon.so}
E=(-e "$LIB"); [ "$LIB" = none ] && E=()   # LIB=none: CPU run, for testing the script itself
mkdir -p "$OUT/cache-hi" "$OUT/cache-lo"
rm -f "$OUT/go" "$OUT/ready-hi" "$OUT/ready-lo"

waitfor() { # FILE PID
   local i=0
   while [ ! -e "$1" ]; do
      kill -0 "$2" 2>/dev/null || { echo "prio_pair: process $2 ended before it was ready"; return 1; }
      i=$((i + 1)); [ $i -gt 6000 ] && { echo "prio_pair: $1 not ready after 60 s"; return 1; }
      sleep 0.01
   done
}

ETNA_ML_DISK_CACHE_DIR="$OUT/cache-lo" $PY -I $CHECK -m "$MODEL" -r "$REF" "${E[@]}" -t "$LO" --seconds "$SECS" \
   --name hwjl_lo --ready-file "$OUT/ready-lo" --go-file "$OUT/go" "$@" > "$OUT/lo.json" 2> "$OUT/lo.err" &
LOP=$!
waitfor "$OUT/ready-lo" $LOP || { kill $LOP 2>/dev/null; exit 1; }
ETNA_ML_DISK_CACHE_DIR="$OUT/cache-hi" $PY -I $CHECK -m "$MODEL" -r "$REF" "${E[@]}" -t "$HI" --seconds "$SECS" \
   --name hwjl_hi --ready-file "$OUT/ready-hi" --go-file "$OUT/go" "$@" > "$OUT/hi.json" 2> "$OUT/hi.err" &
HIP=$!
waitfor "$OUT/ready-hi" $HIP || { kill $LOP $HIP 2>/dev/null; touch "$OUT/go"; exit 1; }
cat /sys/module/etnaviv/parameters/hw_job_clients > "$OUT/clients-before.txt" 2>/dev/null
touch "$OUT/go"
sleep 1
cat /sys/module/etnaviv/parameters/hw_job_clients > "$OUT/clients-during.txt" 2>/dev/null
wait $HIP; rh=$?
wait $LOP; rl=$?
echo "prio_pair: hi rc=$rh lo rc=$rl"
rm -rf "$OUT/cache-hi" "$OUT/cache-lo"
[ $rh -eq 0 ] && [ $rl -eq 0 ]
