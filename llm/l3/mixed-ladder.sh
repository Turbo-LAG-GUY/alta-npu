#!/usr/bin/env bash
# mixed-ladder.sh (2026-11-21, LLM engine R2): the mixed jobs-per-step packs (qkv 2, o 1, gateup 4, down 2: each job as
# wide as fits the 512 KiB VIP SRAM; from wide-ladder.sh, runs/llm-mixed-ladder-20261121-211302, estimated ~114 ms per
# chunk vs 137.8 for 4 jobs). Packs npu-llm/models/jb/real-135m-mixed, offline PASS (offline-jb-20261121-real135m-mixed:
# 0 problems, kpc max 20). 135M quality of the mix = 4 jobs (l2h-mixed log). Rungs: M1 mixed 256 tokens, M2 4 jobs 256
# (same-session baseline), M3 mixed 1 token, X2 one chunk end to end on the mixed packs. Written CPU only.
# (Header below kept from wide-ladder.sh.)
# NEEDS JORDAN'S OK BEFORE IT RUNS. Message session ai-projects-60 first (it also does NPU work).
#
# W rungs: do wider jobs pay? jb_engine (frozen 3b2807b6 = f85c58fa + --serve; the rounds mode is unchanged) runs all
# 120 layer steps of SmolLM2-135M back to back, 20 rounds, every output checked in the first and last round, with
# 4 / 2 / 1 weight-group jobs per step (llm/l3/real_packs.py GROUPS; quality cost from llm/ref/l2h_groups.py:
# KL GPL-3 .023 / .033 / .053). The 4-job rungs repeat runs/llm-engine-ladder-20261121-153843 (140 ms per 256-token
# chunk, 32.7 ms per token) on today's driver state (job-queue build f317fc8f, hw_job_limit 4 at runtime).
# Offline first: npu-llm/runs/offline-jb-20261121-real135m-g1 / -g2 (every pack, fake NPU) must have passed.
#
# X1: one whole 257-token prompt chunk end to end (llm/l3/chunk_glue.py --backend npu): the CPU glue in numpy and every
# linear step on the NPU through jb_engine --serve (4-job packs). Passes when every step on its recorded input is
# within 1 level of its pack and the run ends cleanly; its quality is then compared with the ref run
# (npu-llm/runs/chunk-glue-ref-135m*.json / -logits.npz) and the noise band (chunk-glue-ref-135m-jitter.json).
#
# Stops at the first failure, STOP, an output more than 1 level from the reference, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-mixed --collect llm/l3/mixed-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JE=$AI/npu-llm/bin/jb_engine-3b2807b6
PY=$AI/.venv/bin/python
MD=$AI/npu-llm/models/SmolLM2-135M-Instruct
J=$AI/npu-llm/models/jb
RD=$K/runs/llm-mixed-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JE" | cut -c1-8)" = 3b2807b6 ] || { echo "jb_engine is not 3b2807b6"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
[ -d "$J/real-135m-mixed" ] || { echo "no packs $J/real-135m-mixed"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL RC MAXDIFF BAD
   local w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w"
   if [ "$2" -ne 0 ] || [ -e runs/STOP ] || [ "${3:-99}" -gt 1 ] || [ "${4:-99}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-mixed-ladder FAILED "stopped at $1 rc=$2 max_diff=${3:-?} bad=${4:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}

run() { # LABEL PACKDIR TOKENS
   local label=LLMW-$1 rc mx bad
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JE" -n 20 --tokens "$3" "$2" > "$RD/$label.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_packs": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   grep -E '^compiled|^  [a-z]' "$RD/$label.stdout"
   grep -oE '"(round_ms_median|ms_per_token|compile_ms|readout_ms)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' '; echo
   check "$label" "$rc" "$mx" "$bad"
}

scripts/progress llm-mixed-ladder STARTING "llm/l3/mixed-ladder.sh: $RD"
run M1-mixed-256 "$J/real-135m-mixed" 256
run M2-4jobs-256 "$J/real-135m" 256
run M3-mixed-1 "$J/real-135m-mixed" 1

label=LLMW-X2-chunk-end-to-end-mixed
NPU_DEADLINE=400 "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$PY" -I "$K/llm/l3/chunk_glue.py" --backend npu \
   --engine "$JE" --json "$RD/x2-npu.json" --save-logits "$RD/x2-npu-logits.npz" "$MD" "$J/real-135m-mixed" \
   > "$RD/$label.stdout" 2>&1
rc=$?
grep -E '^steps|^gpl3|^readme|engine:' "$RD/$label.stdout"
mx=$(grep -oE 'max [0-9]+ levels from the packs' "$RD/$label.stdout" | grep -oE '[0-9]+')
bad=$(grep -oE '[0-9]+ steps > 1' "$RD/$label.stdout" | grep -oE '^[0-9]+')
check "$label" "$rc" "$mx" "$bad"

scripts/progress llm-mixed-ladder DONE "all rungs passed: $RD"
echo "ALL PASSED: $RD"
