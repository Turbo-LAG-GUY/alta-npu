#!/usr/bin/env bash
# prefill2-ladder.sh (2026-11-23, LLM engine): jb_prefill on the NPU, round 2 (Jordan's OK, 2026-11-23: "You have the OK
# for the NPU run"): the 135M re-timing with the dynamic-scheduling build, then the FIRST WHOLE 1.7B PROMPT CHUNK in C on
# the NPU (24 layers, 96 steps, 1.7 GB of weights pinned for the NPU; real-1p7b-split4 = 4 jobs, down 4 pieces x 4
# range-grouped jobs; the gated tile rule + keep-image, as in rsplit-ladder S2).
# jb_prefill 91ee17b9 = cb935e3f (prefill-ladder 4/4 PASS) + schedule(dynamic) + the aux file mapped, not copied + load
# progress lines (flushed, with MemAvailable). Incident incidents/2026-11-23-1537-LLMP2-prefill2-swap-stall: run 1's R1
# stalled the board in swap for 1 h 47 min (stopped by Jordan); run 2's R1 hit the deadline (STOP). Hence the memory
# guard below (MEM_MIN_MB, default 2300) before every 1.7B rung.
# Run 1 (2026-11-23 17:26, runs/llm-prefill2-ladder-20261123-172638): Q1-Q3 PASS, R1 hit npu-run's 600 s deadline
# while still loading / compiling the 1.7B packs under memory pressure (no hang: trips 0, no etnaviv lines) -> STOP.
# This version: the 1.7B rungs get NPU_DEADLINE=1500, and Q1-Q3 are skipped with SKIP_135M=1. Before running it:
# Jordan reviews + clears runs/STOP, and closes what he can (other Claude sessions, the remote desktop): the 1.7B
# run pins ~1.8 GB of NPU buffers on a 3.7 GB board. Checked first: fake NPU
# 121 submits rc 0 (135M); 1.7B --ref vs fp32: KL .152, top-1 87.5%, ppl 2.451 vs 2.154 (= the simulator's layout).
#   Q1-3 135M, last token only (timing)       R1 1.7B --all: scored vs fp32 and vs --ref      R2-3 1.7B last token only
# Stops at a failure, STOP or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-prefill2 --collect llm/l3/prefill2-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JP=$AI/npu-llm/bin/jb_prefill-91ee17b9
J=$AI/npu-llm/models/jb
X=$AI/npu-llm/runs/x86
RD=$K/runs/llm-prefill2-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JP" | cut -c1-8)" = 91ee17b9 ] || { echo "jb_prefill is not 91ee17b9"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_SHARE_INPUT_TRANSPOSE -u ETNA_ML_V7_TILE_FIT_MAX_IC
       -u ETNA_ML_V7_TILE_FIT_ALWAYS -u ETNA_ML_NO_FLUSH_INDEPENDENT
       ETNA_ML_KEEP_IMAGE_CACHE=1 ETNA_ML_V7_TILE_FIT=262144 OMP_NUM_THREADS=4 GOMP_CPU_AFFINITY=2-5
       OMP_WAIT_POLICY=active)
A135=("$J/real-135m" "$J/aux-135m/model.jba" "$J/aux-135m/gpl3.jbp0")
A17=("$J/real-1p7b-split4" "$J/aux-1p7b/model.jba" "$J/aux-1p7b/gpl3.jbp0")
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-prefill2-ladder FAILED "stopped at $1 rc=$2 kernel_warnings=$w; $RD"
      exit 1
   fi
}

mem_guard() { # LABEL: refuse a 1.7B rung without the memory it needs (~1.8 GB pinned NPU buffers + ~0.3 GB process)
   local avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo) need=${MEM_MIN_MB:-2300}
   echo "== $1: MemAvailable $avail MB (need $need)"
   if [ "$avail" -lt "$need" ]; then
      echo "NOT STARTED: $1 needs $need MB available, $avail MB now (close desktop sessions / other programs first)"
      scripts/progress llm-prefill2-ladder FAILED "$1 not started: MemAvailable $avail MB < $need MB; $RD"
      exit 1
   fi
}

run() { # LABEL DEADLINE ARGS...
   local label=$1 dl=$2; shift 2
   "${CLEAN[@]}" NPU_DEADLINE=$dl scripts/npu-attempt "$label" -- "$JP" --npu "$@" > "$RD/$label.stdout" 2>&1
   check "$label" $?
   grep -E '^loaded|^chunk' "$RD/$label.stdout" | cut -c1-260     # (the loading lines: progress, see the stdout)
}

scripts/progress llm-prefill2-ladder STARTING "llm/l3/prefill2-ladder.sh: $RD"
[ "${SKIP_135M:-0}" = 1 ] || for r in 1 2 3; do run LLMP2-Q$r-135m-last 120 "${A135[@]}"; done
mem_guard LLMP2-R1-1p7b-all
run LLMP2-R1-1p7b-all 1500 --all --out "$RD/r1-logits.f32" "${A17[@]}"
$AI/.venv/bin/python -I $K/llm/l3/glue/score_prefill.py "$RD/r1-logits.f32" "$X/fp32-1p7b-gpl3.npy" \
   "$X/fp32-1p7b-gpl3-ids.npy" "$AI/npu-llm/runs/prefill-ref-1p7b/logits.f32" | tee "$RD/r1-score.txt"
for r in 2 3; do mem_guard LLMP2-R$r-1p7b-last; run LLMP2-R$r-1p7b-last 1500 "${A17[@]}"; done
scripts/progress llm-prefill2-ladder DONE "all rungs passed: $RD ($(head -n 1 "$RD/r1-score.txt"))"
echo "ALL PASSED: $RD"
