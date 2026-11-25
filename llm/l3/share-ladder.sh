#!/usr/bin/env bash
# share-ladder.sh (2026-11-25, LLM engine, 1.7B memory fixes; Jordan: "you have my OK and all the time you need").
# NEW DRIVER BEHAVIOUR: mesa-llm ETNA_ML_SHARE_TENSORS=1 - the k-th tensor of every subgraph reuses one pooled buffer of
# its size (the engine runs its steps one after another and reads each step's outputs before the next), so the 96
# subgraphs of the 1.7B share one set of input / output tensors instead of 24. Reused buffers are not zeroed again.
# Fake NPU first (scratchpad share/): 135M 121 submits identical except 476 words, all addresses inside the submit's
# BOs; distinct BOs 167.4 -> 104.4 MiB. 1.7B: 97 submits, 668 differing words, all addresses; 1825.6 -> 1603.3 MiB.
# jb_prefill 4bfd0dc8 = 21e4e4e9 (--serve) + JB_HEAD8 at startup (the int8 head for chunk + decode, the bf16 table's
# pages dropped). Its default chunk path is d17bb7cc's (135M --ref logits bit-identical).
#   S1 135M --all, shared tensors: logits must equal prefill5 C1's
#   S2 135M --generate 64, shared tensors + decode from the NPU's buffers + fp16 cache + int8 head (timing, text)
#   S3 1.7B --all, shared tensors: logits must equal prefill5 C3's; MemAvailable after the load vs C3's
#   S4 1.7B --generate 64, as S2 (prefill5 C4: 340.5 ms per token with float cache + bf16 head)
# Memory: no guard (MEM_MIN_MB=N to set one). Stops at a failure, a mismatch, STOP or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-share --collect llm/l3/share-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JP=$AI/npu-llm/bin/jb_prefill-4bfd0dc8
J=$AI/npu-llm/models/jb
P5=$K/runs/llm-prefill5-ladder-20261124-034056
RD=$K/runs/llm-share-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JP" | cut -c1-8)" = 4bfd0dc8 ] || { echo "jb_prefill is not 4bfd0dc8"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
for f in c1-logits.f32 c3-logits.f32; do [ -s "$P5/$f" ] || { echo "missing $P5/$f"; exit 1; }; done
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_SHARE_INPUT_TRANSPOSE -u ETNA_ML_V7_TILE_FIT_MAX_IC
       -u ETNA_ML_V7_TILE_FIT_ALWAYS -u ETNA_ML_NO_FLUSH_INDEPENDENT -u JB_DECODE_Q16 -u JB_DECODE_COEF -u JB_KV16
       -u JB_HEAD8
       ETNA_ML_KEEP_IMAGE_CACHE=1 ETNA_ML_V7_TILE_FIT=262144 ETNA_ML_EXACT_BO=1 ETNA_ML_CACHED_COEFFS=1
       ETNA_ML_SHARE_TENSORS=1 OMP_NUM_THREADS=4 GOMP_CPU_AFFINITY=2-5 OMP_WAIT_POLICY=active)
GEN=(JB_DECODE_COEF=1 JB_KV16=1 JB_HEAD8=1)
A135=("$J/real-135m" "$J/aux-135m/model.jba" "$J/aux-135m/gpl3.jbp0")
A17=("$J/real-1p7b-split4" "$J/aux-1p7b/model.jba" "$J/aux-1p7b/gpl3.jbp0")
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

fail() { # LABEL WHY
   echo "STOPPED at $1: $2 (see $RD and PROGRESS.md)"
   scripts/progress llm-share-ladder FAILED "stopped at $1: $2; $RD"
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

run() { # LABEL DEADLINE [VAR=VALUE ...] -- ARGS...
   local label=$1 dl=$2 extra=(); shift 2
   while [ "$1" != -- ]; do extra+=("$1"); shift; done
   shift
   mem "$label"
   "${CLEAN[@]}" "${extra[@]}" NPU_DEADLINE=$dl scripts/npu-attempt "$label" -- "$JP" --npu "$@" > "$RD/$label.stdout" 2>&1
   check "$label" $?
   grep -E '^int8 head|^loaded|^chunk|^generated|^decode' "$RD/$label.stdout" | cut -c1-300
   grep -E 'loading layer' "$RD/$label.stdout" | tail -1
}

same() { # LABEL A B
   cmp -s "$2" "$3" || fail "$1" "logits differ from $(basename "$(dirname "$3")")/$(basename "$3")"
   echo "== $1: logits bit-identical to $(basename "$3") of prefill5"
}

scripts/progress llm-share-ladder STARTING "llm/l3/share-ladder.sh: $RD"
run LLMS-S1-135m-all 240 -- --all --out "$RD/s1-logits.f32" "${A135[@]}"
same LLMS-S1-135m-all "$RD/s1-logits.f32" "$P5/c1-logits.f32"
run LLMS-S2-135m-gen 300 "${GEN[@]}" -- --generate 64 "${A135[@]}"
run LLMS-S3-1p7b-all 1500 -- --all --out "$RD/s3-logits.f32" "${A17[@]}"
same LLMS-S3-1p7b-all "$RD/s3-logits.f32" "$P5/c3-logits.f32"
echo "== prefill5 C3 for comparison: $(grep 'loading layer' "$P5/LLMP5-C3-1p7b-all.stdout" | tail -1)"
run LLMS-S4-1p7b-gen 1500 "${GEN[@]}" -- --generate 64 "${A17[@]}"
(cd "$RD" && sha256sum ./*.f32 > logits.sha256)
scripts/progress llm-share-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"
