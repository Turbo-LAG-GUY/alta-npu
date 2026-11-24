#!/usr/bin/env bash
# prefill3-ladder.sh (2026-11-23, LLM engine): the first whole 1.7B prompt chunk in C on the NPU, round 3, with a smaller
# memory footprint (incident incidents/2026-11-23-1537-LLMP2-prefill2-swap-stall: the 1.7B run needs more than the board
# had free). Measured on the fake NPU (shim BO lists + Mesa's calculate_weight_bo_size, which reproduces every recorded
# size): the 1.7B steps pin 2127 MiB of NPU buffers, 299 MiB of it the libdrm BO cache rounding coefficient BOs up to
# the next quarter power of two (gateup jobs: 8.03 MiB of weights in 10 MiB BOs), plus ~192 MiB of per-pack read-back
# buffers in the process: ~2.5 GB in all, more than the ~2.26 GB available with one desktop session.
#   jb_prefill 36703f9e = 91ee17b9 + one set of read-back buffers per step kind (-184 MiB; 135M --ref logits and all
#     121 fake-NPU submits byte-identical to 91ee17b9), linked against mesa-llm with:
#   ETNA_ML_EXACT_BO=1 (mesa-llm test tree, new): coefficient BOs of their exact size (-299 MiB pinned on 1.7B).
#     Without the variable every fake-NPU submit is byte-identical; with it only BO addresses change (480 stream
#     words on 135M, every one an address inside a BO). NEW DRIVER BEHAVIOUR: E1/E2 prove it on the NPU first.
#   E1 135M --all, bucket BOs        E2 135M --all, exact BOs: logits must be bit-identical to E1
#   R1 1.7B --all, exact BOs: scored vs fp32 and vs --ref       R2-3 1.7B last token only (timing)
# Memory guard before each 1.7B rung: MEM_MIN_MB (default 2200 = 1828 MiB pinned + ~150 MiB process and compile
# transients + ~200 MiB of the mapped aux file, of which only the touched part is resident).
# Stops at a failure, a logits mismatch, STOP or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-prefill3 --collect llm/l3/prefill3-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JP=$AI/npu-llm/bin/jb_prefill-36703f9e
J=$AI/npu-llm/models/jb
X=$AI/npu-llm/runs/x86
RD=$K/runs/llm-prefill3-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JP" | cut -c1-8)" = 36703f9e ] || { echo "jb_prefill is not 36703f9e"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_SHARE_INPUT_TRANSPOSE -u ETNA_ML_V7_TILE_FIT_MAX_IC
       -u ETNA_ML_V7_TILE_FIT_ALWAYS -u ETNA_ML_NO_FLUSH_INDEPENDENT -u ETNA_ML_EXACT_BO
       ETNA_ML_KEEP_IMAGE_CACHE=1 ETNA_ML_V7_TILE_FIT=262144 OMP_NUM_THREADS=4 GOMP_CPU_AFFINITY=2-5
       OMP_WAIT_POLICY=active)
A135=("$J/real-135m" "$J/aux-135m/model.jba" "$J/aux-135m/gpl3.jbp0")
A17=("$J/real-1p7b-split4" "$J/aux-1p7b/model.jba" "$J/aux-1p7b/gpl3.jbp0")
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

fail() { # LABEL WHY
   echo "STOPPED at $1: $2 (see $RD and PROGRESS.md)"
   scripts/progress llm-prefill3-ladder FAILED "stopped at $1: $2; $RD"
   exit 1
}

check() { # LABEL RC
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "$w" -ne 0 ]; then
      fail "$1" "rc=$2 kernel_warnings=$w"
   fi
}

mem_guard() { # LABEL
   local avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo) need=${MEM_MIN_MB:-2200}
   echo "== $1: MemAvailable $avail MB (need $need)"
   [ "$avail" -ge "$need" ] || fail "$1" "not started: MemAvailable $avail MB < $need MB"
}

run() { # LABEL DEADLINE EXACT(0/1) ARGS...
   local label=$1 dl=$2 ex=$3; shift 3
   local e=(); [ "$ex" = 1 ] && e=(ETNA_ML_EXACT_BO=1)
   "${CLEAN[@]}" "${e[@]}" NPU_DEADLINE=$dl scripts/npu-attempt "$label" -- "$JP" --npu "$@" > "$RD/$label.stdout" 2>&1
   check "$label" $?
   grep -E '^loaded|^chunk' "$RD/$label.stdout" | cut -c1-260
}

scripts/progress llm-prefill3-ladder STARTING "llm/l3/prefill3-ladder.sh: $RD"
run LLMP3-E1-135m-bucket 180 0 --all --out "$RD/e1-logits.f32" "${A135[@]}"
run LLMP3-E2-135m-exact 180 1 --all --out "$RD/e2-logits.f32" "${A135[@]}"
cmp -s "$RD/e1-logits.f32" "$RD/e2-logits.f32" || fail LLMP3-E2-135m-exact "logits differ from E1 (bucket BOs)"
echo "== E2 logits bit-identical to E1"
scripts/progress llm-prefill3-ladder NOTE "E1/E2 PASS: 135M on the NPU with exact-size coefficient BOs, logits bit-identical"
mem_guard LLMP3-R1-1p7b-all
run LLMP3-R1-1p7b-all 1500 1 --all --out "$RD/r1-logits.f32" "${A17[@]}"
$AI/.venv/bin/python -I $K/llm/l3/glue/score_prefill.py "$RD/r1-logits.f32" "$X/fp32-1p7b-gpl3.npy" \
   "$X/fp32-1p7b-gpl3-ids.npy" "$AI/npu-llm/runs/prefill-ref-1p7b/logits.f32" | tee "$RD/r1-score.txt"
for r in 2 3; do mem_guard LLMP3-R$r-1p7b-last; run LLMP3-R$r-1p7b-last 1500 1 "${A17[@]}"; done
scripts/progress llm-prefill3-ladder DONE "all rungs passed: $RD ($(head -n 1 "$RD/r1-score.txt"))"
echo "ALL PASSED: $RD"
