#!/usr/bin/env bash
# prefill-ladder.sh (2026-11-23, LLM engine): the first prompt chunk with the WHOLE loop in C on the NPU
# (jb_prefill: llm/l3/glue/jb_glue.c compiled in, the steps through jb_engine's compile / invoke / read-out code).
# NEEDS JORDAN'S OK BEFORE IT RUNS. Message the other NPU sessions first; Steam closed.
#
# jb_prefill cb935e3f (mesa-llm e69bdfa4 + 612e677d libs; the no-flush hook is NOT enabled). 135M packs
# npu-llm/models/jb/real-135m (the 4-job packs that ran many times, all 120 steps compiled in one process as in
# jb_engine), aux + GPL-3 prompt from glue/export_aux.py (npu-llm/models/jb/aux-135m). Checked first:
#   --ref (CPU int8 reference of every step) vs the Python C-glue path: KL 0.009, top-1 94.5%, ppl 15.74 / 15.82
#   (the rounding-tie cascade; the format's own noise is ~0.02); fake NPU: 121 submits, rc 0.
#   P1 jb_prefill --npu --all: logits of all 256 tokens -> compared with the Python reference (compare_prefill.py)
#   P2 jb_prefill --npu: the last token's logits only = the real prompt cost, 3 runs for the timing
# Stops at a failure, STOP or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-prefill --collect llm/l3/prefill-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JP=$AI/npu-llm/bin/jb_prefill-cb935e3f
J=$AI/npu-llm/models/jb
RD=$K/runs/llm-prefill-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1
if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JP" | cut -c1-8)" = cb935e3f ] || { echo "jb_prefill is not cb935e3f"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H -u ETNA_ML_SHARE_INPUT_TRANSPOSE -u ETNA_ML_V7_TILE_FIT_MAX_IC
       -u ETNA_ML_V7_TILE_FIT_ALWAYS -u ETNA_ML_NO_FLUSH_INDEPENDENT
       ETNA_ML_KEEP_IMAGE_CACHE=1 ETNA_ML_V7_TILE_FIT=262144 OMP_NUM_THREADS=4 GOMP_CPU_AFFINITY=2-5
       OMP_WAIT_POLICY=active)
ARGS=("$J/real-135m" "$J/aux-135m/model.jba" "$J/aux-135m/gpl3.jbp0")
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-prefill-ladder FAILED "stopped at $1 rc=$2 kernel_warnings=$w; $RD"
      exit 1
   fi
}

scripts/progress llm-prefill-ladder STARTING "llm/l3/prefill-ladder.sh: $RD"
label=LLMP-P1-135m-all
"${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JP" --npu --all --dump "$RD" --out "$RD/p1-logits.f32" "${ARGS[@]}" \
   > "$RD/$label.stdout" 2>&1
check "$label" $?
grep -E '^chunk' "$RD/$label.stdout"
$AI/.venv/bin/python -I $K/llm/l3/glue/compare_prefill.py "$J/aux-135m/gpl3.ref" "$J/aux-135m/gpl3.jbp0" "$RD" \
   "$RD/p1-logits.f32" | tee "$RD/p1-compare.txt" | tail -1
for r in 1 2 3; do
   label=LLMP-P2-135m-last-$r
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JP" --npu "${ARGS[@]}" > "$RD/$label.stdout" 2>&1
   check "$label" $?
   grep -E '^chunk' "$RD/$label.stdout"
done
rm -f "$RD"/x.*.f32
scripts/progress llm-prefill-ladder DONE "all rungs passed: $RD ($(tail -n 1 "$RD/p1-compare.txt"))"
echo "ALL PASSED: $RD"
