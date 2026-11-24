#!/usr/bin/env bash
# wait-mem-then-1p7b.sh (2026-11-23, Jordan: "I'll free memory first"): wait (up to WAIT_S, default 3600 s) until
# MemAvailable >= MEM_MIN_MB (default 2300), then run llm/l3/prefill2-ladder.sh with SKIP_135M=1 (the 1.7B rungs only;
# the ladder checks memory again before every 1.7B rung). Progress in PROGRESS.md (llm-prefill2-wait).
# Usage: systemd-run --user --unit=alta-llm-1p7b-wait --collect llm/l3/wait-mem-then-1p7b.sh
set -u
K=/home/jordan/alta-npu
need=${MEM_MIN_MB:-2300} wait_s=${WAIT_S:-3600} t0=$SECONDS last=0
cd "$K" || exit 1
scripts/progress llm-prefill2-wait STARTING "waiting up to $wait_s s for MemAvailable >= $need MB, then prefill2-ladder.sh SKIP_135M=1"
while :; do
   avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
   if [ "$avail" -ge "$need" ]; then
      scripts/progress llm-prefill2-wait NOTE "MemAvailable $avail MB after $((SECONDS - t0)) s: starting the ladder"
      SKIP_135M=1 exec llm/l3/prefill2-ladder.sh
   fi
   if [ $((SECONDS - t0)) -ge "$wait_s" ]; then
      scripts/progress llm-prefill2-wait FAILED "MemAvailable still $avail MB after $wait_s s: ladder not started"
      exit 1
   fi
   if [ $((SECONDS - last)) -ge 300 ]; then
      scripts/progress llm-prefill2-wait NOTE "MemAvailable $avail MB (need $need), waiting"
      last=$SECONDS
   fi
   sleep 10
done
