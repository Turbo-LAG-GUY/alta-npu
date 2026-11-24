#!/usr/bin/env bash
# prefill5-ladder.sh (2026-11-24, LLM engine; design (a), Jordan: "go for (a)"): decode on the CPU straight from the
# NPU's weight buffers. NEW DRIVER BEHAVIOUR: mesa-llm ETNA_ML_CACHED_COEFFS=1 allocates the coefficient BOs CPU-cached
# (they were write-combined; written once between cpu_prep / cpu_fini, where the kernel cleans the caches; the NPU only
# reads them). C1 / C3 prove the NPU still computes the same bits from cached buffers.
# jb_prefill d17bb7cc = 944b41e6 (prefill4 4/4 PASS) + etna_ml_llm_coefs() (mesa-llm: the layout recorded at compile time)
# + jbg_gemv_coef_q16 (int8 x int16 GEMV on the NPU's V7 1x1 layout) + exact integer dots in both int16 GEMVs.
# Checked on the fake NPU: --coef-check 135M 480/480 and 1.7B 672/672 jobs decode to the packs' weights byte for byte;
# 135M teacher-forced decode from the buffers = from kept weights, bit for bit; 1.7B decode from the buffers 325 ms per
# token, teacher-forced vs fp32 KL .028, top-1 95.7%, ppl 2.145 (fp32 2.154; the NPU prefill: KL .154, ppl 2.440).
#   C1 135M --all, cached coefficient BOs: logits must equal prefill3 E2's
#   C2 135M --coef-check --generate 64 --teacher, decode from the NPU's buffers (JB_DECODE_COEF=1): the teacher
#      logits must equal a CPU-only --ref teacher run's (kept weights) bit for bit
#   C3 1.7B --all, cached coefficient BOs: logits must equal prefill3 R1's
#   C4 1.7B --coef-check --generate 64, decode from the NPU's buffers
# Memory: no guard (MEM_MIN_MB=N to set one). Stops at a failure, a mismatch, STOP or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-prefill5 --collect llm/l3/prefill5-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JP=$AI/npu-llm/bin/jb_prefill-d17bb7cc
J=$AI/npu-llm/models/jb
P3=$K/runs/llm-prefill3-ladder-20261124-021642
RD=$K/runs/llm-prefill5-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JP" | cut -c1-8)" = d17bb7cc ] || { echo "jb_prefill is not d17bb7cc"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_SHARE_INPUT_TRANSPOSE -u ETNA_ML_V7_TILE_FIT_MAX_IC
       -u ETNA_ML_V7_TILE_FIT_ALWAYS -u ETNA_ML_NO_FLUSH_INDEPENDENT -u JB_DECODE_Q16 -u JB_DECODE_COEF
       ETNA_ML_KEEP_IMAGE_CACHE=1 ETNA_ML_V7_TILE_FIT=262144 ETNA_ML_EXACT_BO=1 ETNA_ML_CACHED_COEFFS=1 OMP_NUM_THREADS=4
       GOMP_CPU_AFFINITY=2-5 OMP_WAIT_POLICY=active)
A135=("$J/real-135m" "$J/aux-135m/model.jba" "$J/aux-135m/gpl3.jbp0")
A17=("$J/real-1p7b-split4" "$J/aux-1p7b/model.jba" "$J/aux-1p7b/gpl3.jbp0")
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

fail() { # LABEL WHY
   echo "STOPPED at $1: $2 (see $RD and PROGRESS.md)"
   scripts/progress llm-prefill5-ladder FAILED "stopped at $1: $2; $RD"
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
   "${CLEAN[@]}" JB_DECODE_COEF=${JB_DECODE_COEF:-0} NPU_DEADLINE=$dl scripts/npu-attempt "$label" -- "$JP" --npu "$@" > "$RD/$label.stdout" 2>&1
   check "$label" $?
   grep -E '^loaded|^chunk|^generated|^decode' "$RD/$label.stdout" | cut -c1-300
}

same() { # LABEL A B
   cmp -s "$2" "$3" || fail "$1" "logits differ from $(basename "$(dirname "$3")")/$(basename "$3")"
   echo "== $1: logits bit-identical to $(basename "$3") of prefill3"
}

scripts/progress llm-prefill5-ladder STARTING "llm/l3/prefill5-ladder.sh: $RD"
run LLMP5-C1-135m-all 240 --all --out "$RD/c1-logits.f32" "${A135[@]}"
same LLMP5-C1-135m-all "$RD/c1-logits.f32" "$P3/e2-logits.f32"
# the reference: the same teacher-forced decode from kept weights, CPU only (--ref: no NPU)
"${CLEAN[@]}" "$JP" --ref --teacher "$RD/ref-teacher.f32" "${A135[@]}" > "$RD/ref-teacher.stdout" 2>&1 ||
   fail LLMP5-C2-135m-gen "the CPU reference teacher run failed"
JB_DECODE_COEF=1 run LLMP5-C2-135m-gen 300 --coef-check --generate 64 --teacher "$RD/c2-teacher.f32" "${A135[@]}"
grep -q 'coef-check: .* all identical to the packs' "$RD/LLMP5-C2-135m-gen.stdout" || fail LLMP5-C2-135m-gen "coef-check"
cmp -s "$RD/c2-teacher.f32" "$RD/ref-teacher.f32" || fail LLMP5-C2-135m-gen "decode from the NPU's buffers != from kept weights"
echo "== C2: teacher-forced decode from the NPU's buffers bit-identical to the kept-weights CPU reference"
run LLMP5-C3-1p7b-all 1500 --all --out "$RD/c3-logits.f32" "${A17[@]}"
same LLMP5-C3-1p7b-all "$RD/c3-logits.f32" "$P3/r1-logits.f32"
JB_DECODE_COEF=1 run LLMP5-C4-1p7b-gen 1500 --coef-check --generate 64 "${A17[@]}"
grep -q 'coef-check: .* all identical to the packs' "$RD/LLMP5-C4-1p7b-gen.stdout" || fail LLMP5-C4-1p7b-gen "coef-check"
scripts/progress llm-prefill5-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"
