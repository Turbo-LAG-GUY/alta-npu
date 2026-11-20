#!/usr/bin/env bash
# real-ladder.sh - the REAL-WEIGHT packs of SmolLM2-135M on the NPU (2026-11-20, LLM engine R2): every layer step of
# the model in the 8-bit format the simulator found nearly lossless (rotation folded into the weights, 4 weight-group
# jobs per step with per-job output scales; llm/l3/real_packs.py), at 1 token and at 256 tokens (16 x 16), with the
# step's real 8-bit input from a GPL-3 forward pass. 240 packs, one NPU submit each.
# Offline first (npu-llm/runs/offline-jb-20261120-real135m): all 240 run with one submit per invoke and no driver
# complaint; stock tiling (hooks unset), kernels per core <= 20 (the hardware cap is 21), tiles 1x1 / 16x16.
# The 1x1 conv shapes per job: qkv 576 -> 240, o 576 -> 144, gateup 576 -> 768, down 1536 -> 144.
# Phase A = layer 0 (8 packs), then phase B = the other 232. jb_run a79b7b5c (mesa-llm with patches 10-12 and the
# test-tree hooks, all inactive here), 20 invokes per pack, polling wait.
# Written 2026-11-20 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS.
# Stops at the first failure, STOP, an output more than 1 level from the reference, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-real --collect llm/l3/real-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
JB=$AI/npu-llm/bin/jb_run-a79b7b5c
PD=$AI/npu-llm/models/jb/real-135m
RD=$K/runs/llm-real-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
[ "$(sha256sum "$JB" | cut -c1-8)" = a79b7b5c ] || { echo "jb_run is not a79b7b5c"; exit 1; }
[ "$(ls "$PD"/L*.jbp | wc -l)" = 240 ] || { echo "expected 240 packs in $PD"; exit 1; }
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

run() { # PACK_NAME
   local label=LLMRW-$1 rc mx w
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$JB" -n 20 --wait spin "$PD/$1.jbp" > "$RD/$1.stdout" 2>&1
   rc=$?
   mx=$(grep -oE '"diff_vs_expected_max": [0-9]+' "$RD/$1.stdout" | grep -oE '[0-9]+$')
   w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $1 rc=$rc max_diff=${mx:-?} kernel_warnings=$w $(grep -oE '"(ms_median_after_first|npu_ms|readout_ms|compile_ms)": [0-9.]+' "$RD/$1.stdout" | tr '\n' ' ')"
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-99}" -gt 1 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $1 (see $RD/$1.stdout and PROGRESS.md)"
      scripts/progress llm-real-ladder FAILED "stopped at $1 rc=$rc max_diff=${mx:-?} kernel_warnings=$w; $RD"
      exit 1
   fi
}
scripts/progress llm-real-ladder STARTING "llm/l3/real-ladder.sh: $RD"
for f in "$PD"/L00_*.jbp; do run "$(basename "$f" .jbp)"; done
scripts/progress llm-real-ladder NOTE "phase A (layer 0, 8 packs) passed: $RD"
for f in "$PD"/L*.jbp; do n=$(basename "$f" .jbp); case "$n" in L00_*) continue ;; esac; run "$n"; done
python3 - "$RD" <<'PY'
import glob, json, os, re, sys
rd = sys.argv[1]
tot = {}
for f in glob.glob(os.path.join(rd, "L*.stdout")):
    s = open(f).read()
    j = json.loads(re.findall(r"^\{.*\}$", s, re.M)[-1])
    t = int(re.search(r"_t(\d+)\.stdout$", f).group(1))
    a = tot.setdefault(t, {"packs": 0, "ms": 0.0, "npu_ms": 0.0, "readout_ms": 0.0, "max_diff": 0})
    a["packs"] += 1; a["ms"] += j["ms_median_after_first"]; a["npu_ms"] += j["npu_ms"]; a["readout_ms"] += j["readout_ms"]
    a["max_diff"] = max(a["max_diff"], j["diff_vs_expected_max"])
json.dump(tot, open(os.path.join(rd, "totals.json"), "w"), indent=1)
for t, a in sorted(tot.items()):
    print("tokens %3d: %d packs, all layer steps %.1f ms (NPU %.1f, read-out %.1f) = %.3f ms per token, max diff %d" % (
        t, a["packs"], a["ms"], a["npu_ms"], a["readout_ms"], a["ms"] / t, a["max_diff"]))
PY
scripts/progress llm-real-ladder DONE "all 240 packs passed: $RD"
echo "ALL PASSED: $RD"
