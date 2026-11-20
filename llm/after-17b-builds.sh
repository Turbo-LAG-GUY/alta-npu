#!/usr/bin/env bash
# after-17b-builds.sh (2026-11-19): when the 1.7B L2 check is done, stop the L2 chain before SmolLM3 (rerun later),
# then build mesa-gpu-dot (G1) and mesa-llm (L3), one at a time, -j2. CPU only, nothing touches the NPU or GPU.
set -u
K=/home/jordan/alta-npu; AI=/home/jordan/Documents/AI_Projects
until grep -q 'llm-l2-big | DONE | SmolLM2-1.7B' $K/PROGRESS.md; do sleep 20; done
systemctl --user stop alta-llm-l2big
$K/scripts/progress llm-l2-big NOTE "chain stopped after 1.7B (SmolLM3 later, after the G1/L3 builds)"
for t in mesa-gpu-dot mesa-llm; do
   $K/scripts/progress build-$t STARTING "ninja -C $AI/$t/build -j2, log $AI/$t-build.log"
   s=$(date +%s)
   nice -n 5 ninja -C $AI/$t/build -j2 > $AI/$t-build.log 2>&1
   rc=$?
   $K/scripts/progress build-$t $([ $rc = 0 ] && echo DONE || echo FAILED) "rc=$rc in $(( ($(date +%s) - s) / 60 )) min: $(tail -n 1 $AI/$t-build.log | cut -c1-150)"
done
