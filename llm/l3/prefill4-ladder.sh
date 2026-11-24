#!/usr/bin/env bash
# prefill4-ladder.sh (2026-11-24, LLM engine, Jordan's OK: "for 1 you have the OK to run the NPU job"): the C-glue
# changes of 2026-11-24 on the NPU - timing in steady state and the first generation after an NPU prompt chunk.
# jb_prefill 944b41e6 = 36703f9e (prefill3-ladder 5/5 PASS) + chunk() / --repeat, page-cache hints (FADV_DONTNEED on
# the packs, WILLNEED on the aux file), jbg_head_bf16, the FFN / scatter tuning (prompt path bit-identical: 135M --ref
# --all = 91ee17b9's logits; old vs new glue identical at 1.7B shapes) + decode (--generate: CPU, int8 weights kept,
# int16 activations, KV cache from the chunk; fake-NPU submits byte-identical).
#   P1 135M --all --repeat 2: logits must equal prefill3 E2's (same NPU work, same glue bits)
#   P2 135M --repeat 3 --generate 64: steady-state chunk timing + 64 greedy tokens after the NPU chunk
#   P3 1.7B --all --repeat 2: logits must equal prefill3 R1's
#   P4 1.7B --repeat 3: steady-state timing
# Memory: no guard by default (Jordan, 2026-11-24: "don't worry about available ram"; MEM_MIN_MB=N to set one); the
# MemAvailable before each rung is printed. Stops at a failure, a logits mismatch, STOP or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-prefill4 --collect llm/l3/prefill4-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JP=$AI/npu-llm/bin/jb_prefill-944b41e6
J=$AI/npu-llm/models/jb
P3=$K/runs/llm-prefill3-ladder-20261124-021642
RD=$K/runs/llm-prefill4-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JP" | cut -c1-8)" = 944b41e6 ] || { echo "jb_prefill is not 944b41e6"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_SHARE_INPUT_TRANSPOSE -u ETNA_ML_V7_TILE_FIT_MAX_IC
       -u ETNA_ML_V7_TILE_FIT_ALWAYS -u ETNA_ML_NO_FLUSH_INDEPENDENT -u JB_DECODE_Q16
       ETNA_ML_KEEP_IMAGE_CACHE=1 ETNA_ML_V7_TILE_FIT=262144 ETNA_ML_EXACT_BO=1 OMP_NUM_THREADS=4
       GOMP_CPU_AFFINITY=2-5 OMP_WAIT_POLICY=active)
A135=("$J/real-135m" "$J/aux-135m/model.jba" "$J/aux-135m/gpl3.jbp0")
A17=("$J/real-1p7b-split4" "$J/aux-1p7b/model.jba" "$J/aux-1p7b/gpl3.jbp0")
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

fail() { # LABEL WHY
   echo "STOPPED at $1: $2 (see $RD and PROGRESS.md)"
   scripts/progress llm-prefill4-ladder FAILED "stopped at $1: $2; $RD"
   exit 1
}

check() { # LABEL RC
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "$w" -ne 0 ]; then
      fail "$1" "rc=$2 kernel_warnings=$w"
   fi
}

mem() { # LABEL
   local avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo) need=${MEM_MIN_MB:-0}
   echo "== $1: MemAvailable $avail MB"
   [ "$avail" -ge "$need" ] || fail "$1" "not started: MemAvailable $avail MB < $need MB"
}

run() { # LABEL DEADLINE ARGS...
   local label=$1 dl=$2; shift 2
   mem "$label"
   "${CLEAN[@]}" NPU_DEADLINE=$dl scripts/npu-attempt "$label" -- "$JP" --npu "$@" > "$RD/$label.stdout" 2>&1
   check "$label" $?
   grep -E '^loaded|^chunk|^generated|^decode' "$RD/$label.stdout" | cut -c1-300
}

same() { # LABEL A B
   cmp -s "$2" "$3" || fail "$1" "logits differ from $(basename "$(dirname "$3")")/$(basename "$3")"
   echo "== $1: logits bit-identical to $(basename "$3") of prefill3"
}

scripts/progress llm-prefill4-ladder STARTING "llm/l3/prefill4-ladder.sh: $RD"
run LLMP4-P1-135m-all 240 --all --repeat 2 --out "$RD/p1-logits.f32" "${A135[@]}"
same LLMP4-P1-135m-all "$RD/p1-logits.f32" "$P3/e2-logits.f32"
run LLMP4-P2-135m-gen 240 --repeat 3 --generate 64 "${A135[@]}"
run LLMP4-P3-1p7b-all 1500 --all --repeat 2 --out "$RD/p3-logits.f32" "${A17[@]}"
same LLMP4-P3-1p7b-all "$RD/p3-logits.f32" "$P3/r1-logits.f32"
run LLMP4-P4-1p7b-last 1500 --repeat 3 "${A17[@]}"
scripts/progress llm-prefill4-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"
