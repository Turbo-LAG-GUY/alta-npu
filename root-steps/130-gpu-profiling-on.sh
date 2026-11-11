#!/usr/bin/env bash
# 130 - turn on the GPU's per-program busy-time counters until the next boot, for the GPU icon.
# Why: the panfrost driver records how long each program's GPU jobs run only while its "profiling" switch is 1, and
# it boots as 0. Without it, npu-status/gpu_status.py can show clock speed, job rate and memory, but no busy %.
# What it does: writes 1 to the GPU's profiling file. Nothing is written to disk: the next boot is back to 0.
# Risk: low. This is the kernel's documented switch (Documentation/gpu/panfrost.rst), off by default "because of
# power saving concerns": it runs the GPU's cycle counter while jobs are on the GPU. It does not touch the NPU.
# Run:    sudo bash root-steps/130-gpu-profiling-on.sh
# Revert: reboot, or echo 0 | sudo tee /sys/bus/platform/devices/ffe40000.gpu/profiling
# Verify: cat /sys/bus/platform/devices/ffe40000.gpu/profiling   (expect: 1)
#         The GPU icon then shows a percentage while the GPU works, and its menu lists each program's share.
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
D=/sys/bus/platform/devices/ffe40000.gpu
P="$D/profiling"
die() { "$here/scripts/progress" 130 FAILED "$*"; echo "130: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 130 FAILED "root step stopped at line $LINENO"' ERR
[ "$(id -u)" -eq 0 ] || { echo "130: run it with sudo" >&2; exit 1; }
[ "$(basename "$(readlink -f "$D/driver")")" = panfrost ] || die "ffe40000.gpu is not bound to panfrost"
[ -w "$P" ] || die "$P is missing or not writable"
if [ "$(cat "$P")" = 1 ]; then
  "$here/scripts/progress" 130 NOTE "GPU profiling was already on"
  exit 0
fi
"$here/scripts/progress" 130 STARTING "turn on GPU job profiling until the next boot (profiling 0 -> 1)"
echo 1 > "$P"
[ "$(cat "$P")" = 1 ] || die "profiling did not read back as 1"
"$here/scripts/progress" 130 DONE "GPU profiling on until the next boot. Undo: reboot, or echo 0 > $P"
echo "Done: the GPU's busy time is now counted, until the next boot."
