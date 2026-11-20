#!/usr/bin/env bash
# cached-ladder.sh - cached output buffers on the NPU (2026-11-20). After the read-out fix the CPU still read each
# output from a write-combined (uncached) buffer at ~0.3 GB/s (A73). mesa-llm now allocates graph outputs (and the
# buffers they alias through a bypass or split) cached on arm64, like etnaviv's CPU-read textures; the kernel's
# CPU_PREP invalidates them before each read, CPU_FINI cleans them (llm/l3/mesa-llm-cached-outputs.diff, lib 6caca26e,
# jb_run ad393a6f). Stale-read check: --alt alternates two inputs (same weights, input values permuted, 96-98% of
# the output bytes differ) and checks EVERY invoke; a missing invalidation would show the previous invoke's output.
# WC baseline = jb_run baca6a17 (lib a26f0852: bulk copy, write-combined) and, for Teflon, the upstream 10-patch
# library 9574fdd1 (same read-out code, no test hooks). Offline (npu-llm/runs/offline-jb-20261120-cached): NN
# configs, weights, submit streams and buffer lists unchanged; exactly the graph outputs are cached (fake-NPU buffer
# log); no CPU_FINI without a CPU_PREP.
# Written 2026-11-20 (CPU only). NEEDS JORDAN'S OK BEFORE IT RUNS; pause other heavy CPU jobs first.
# Stops at the first failure, STOP, an output more than 1 level off on any checked invoke, or a kernel WARNING.
# Usage: systemd-run --user --unit=alta-llm-cached --collect llm/l3/cached-ladder.sh
set -u
K=/home/jordan/alta-npu
AI=/home/jordan/Documents/AI_Projects
PY=$AI/.venv/bin/python
WC=$AI/npu-llm/bin/jb_run-baca6a17
CA=$AI/npu-llm/bin/jb_run-ad393a6f
LIBWC=$AI/npu-llm/bin/libteflon-9574fdd1.so
LIBCA=$AI/npu-llm/bin/libteflon-6caca26e.so
P=$AI/npu-llm/models/jb
M=$AI/npu-llm/models/sizes
RD=$K/runs/llm-cached-ladder-$(date +%Y%m%d-%H%M%S)
mkdir -p "$RD"
cd "$K" || exit 1

if [ -e runs/STOP ]; then echo "runs/STOP is set: clear it (after review) first"; exit 1; fi
if [ "$(pgrep -c -i steam)" != 0 ]; then echo "Steam is running: close it first (it can land on the NPU)"; exit 1; fi
for f in "$WC:baca6a17" "$CA:ad393a6f" "$LIBWC:9574fdd1" "$LIBCA:6caca26e"; do
   [ "$(sha256sum "${f%:*}" | cut -c1-8)" = "${f##*:}" ] || { echo "${f%:*} is not ${f##*:}"; exit 1; }
done
[ -z "${LD_PRELOAD:-}" ] || { echo "LD_PRELOAD is set"; exit 1; }
CLEAN=(env -u ETNA_MESA_DEBUG -u ETNA_ML_SHADER -u ETNA_CS_BARRIER_REG_BUDGET -u ETNA_ML_AXI_SRAM
       -u ETNA_ML_V7_KPC_CAP -u ETNA_ML_V7_TILE_H)
W0=$(journalctl -k -b --no-pager | grep -c 'WARNING')

check() { # LABEL KEY RC
   local label=$1 key=$2 rc=$3 mx bad w
   mx=$(grep -oE "\"$key\": [0-9]+" "$RD/$label.stdout" | grep -oE '[0-9]+$')
   bad=$(grep -oE '"bad_invokes": [0-9]+' "$RD/$label.stdout" | grep -oE '[0-9]+$')
   w=$(( $(journalctl -k -b --no-pager | grep -c 'WARNING') - W0 ))
   echo "== $label rc=$rc max_diff=${mx:-?} bad_invokes=${bad:--} kernel_warnings=$w $(grep -oE '"(ms_median_after_first|invoke_ms_median_after_first|npu_ms|readout_ms|checked_invokes)": [0-9.]+' "$RD/$label.stdout" | tr '\n' ' ')"
   if [ "$rc" -ne 0 ] || [ -e runs/STOP ] || [ "${mx:-99}" -gt 1 ] || [ "${bad:-0}" -ne 0 ] || [ "$w" -ne 0 ]; then
      echo "STOPPED at $label (see $RD/$label.stdout and PROGRESS.md)"
      scripts/progress llm-cached-ladder FAILED "stopped at $label rc=$rc max_diff=${mx:-?} bad_invokes=${bad:--} kernel_warnings=$w; $RD"
      exit 1
   fi
}
jb() { # LABEL BINARY N PACK [jb_run options]
   local label=LLMCA-$1 bin=$2 n=$3 pack=$4; shift 4
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- "$bin" -n "$n" --wait spin "$@" "$P/$pack.jbp" > "$RD/$label.stdout" 2>&1
   check "$label" diff_vs_expected_max $?
}
teflon() { # LABEL LIB MODEL
   local label=LLMCA-$1
   "${CLEAN[@]}" scripts/npu-attempt "$label" -- $PY -I $AI/npu-shader/teflon_check.py -m "$M/$3.tflite" \
      -r "$M/${3}_reference.npz" -e "$2" -n 200 > "$RD/$label.stdout" 2>&1
   check "$label" diff_vs_cpu_max $?
}
S17=split_17b_gateup_npu6656_g4
scripts/progress llm-cached-ladder STARTING "llm/l3/cached-ladder.sh: $RD"
# one token (3 KB out) and the 1.7B split share (4 outputs, 6.6 KB), main thread on an A53; WC first, then cached
jb K1-wc-1x1-a53 "$WC" 200 smol_gateup_1x1 --main-cpu 0
jb K2-cached-1x1-a53 "$CA" 200 smol_gateup_1x1 --main-cpu 0 --alt "$P/smol_gateup_1x1_alt.jbp"
jb K3-wc-17b-share-a53 "$WC" 200 $S17 --main-cpu 0
jb K4-cached-17b-share-a53 "$CA" 200 $S17 --main-cpu 0 --alt "$P/${S17}_alt.jbp"
# many tokens: 64 tokens = 196 KB, 256 tokens = 786 KB
jb K5-wc-8x8 "$WC" 200 smol_gateup_8x8
jb K6-cached-8x8 "$CA" 200 smol_gateup_8x8 --alt "$P/smol_gateup_8x8_alt.jbp"
jb K7-wc-16x16 "$WC" 200 smol_gateup_16x16
jb K8-cached-16x16 "$CA" 200 smol_gateup_16x16 --alt "$P/smol_gateup_16x16_alt.jbp"
# longer stale-read checks: 1000 alternating invokes of the 256-token job, 16 outputs of the 16-group job
jb K9-cached-16x16-1000 "$CA" 1000 smol_gateup_16x16 --alt "$P/smol_gateup_16x16_alt.jbp"
jb K10-cached-g16-1000 "$CA" 1000 smol_gateup_1x1_g16 --alt "$P/smol_gateup_1x1_g16_alt.jbp"
# Teflon: write-combined (upstream 10-patch library) vs cached
teflon K11-teflon-wc-1x1 "$LIBWC" smol_gateup_1x1_P
teflon K12-teflon-cached-1x1 "$LIBCA" smol_gateup_1x1_P
teflon K13-teflon-wc-16x16 "$LIBWC" smol_gateup_16x16_P
teflon K14-teflon-cached-16x16 "$LIBCA" smol_gateup_16x16_P
scripts/progress llm-cached-ladder DONE "all 14 rungs passed: $RD"
echo "ALL PASSED: $RD"
