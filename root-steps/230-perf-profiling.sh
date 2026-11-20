#!/usr/bin/env bash
# 230 - perf for profiling (2026-11-20, LLM engine: where the NPU output read-out spends its time).
# Why: perf was not installed, and an unprivileged user may not sample kernel code (perf_event_paranoid = 2) or see
# kernel symbols (kptr_restrict = 1), so the kernel's cache maintenance (CPU_PREP / CPU_FINI of cached BOs) cannot be
# told apart from the copy. No CPU PMU device is registered on this board (the device tree has no PMU node; not
# touched), so perf can only sample on a timer (cpu-clock); the meson_ddr_bw DDR bandwidth counters are there.
# What it does:
#   1. apt install linux-tools-common linux-tools-6.8.0-149-generic (4 packages incl. hwdata, ~14 MB; simulated first:
#      no kernel image, headers or initramfs). Ubuntu's /usr/bin/perf wrapper wants tools for the running kernel
#      (6.12.73, a custom build, none exist), so call the binary directly: /usr/lib/linux-tools-6.8.0-149/perf
#      (perf's interface to the kernel is stable; a 6.8 perf works on 6.12).
#   2. Until the next boot only: kernel.perf_event_paranoid = -1, kernel.kptr_restrict = 0 (kernel sampling and
#      symbols for the user). Nothing is written to /etc/sysctl.d.
# Risk: low. Any local user can profile the whole system until the next boot (single-user board).
# Run:    sudo bash ~/alta-npu/root-steps/230-perf-profiling.sh
# Revert: sudo sysctl -w kernel.perf_event_paranoid=2 kernel.kptr_restrict=1   (or reboot)
#         sudo apt remove linux-tools-6.8.0-149-generic linux-tools-6.8.0-149 linux-tools-common hwdata
set -euo pipefail
apt-get install -y --no-install-recommends linux-tools-common linux-tools-6.8.0-149-generic
sysctl -w kernel.perf_event_paranoid=-1 kernel.kptr_restrict=0
P=/usr/lib/linux-tools-6.8.0-149/perf
[ -x "$P" ] || P=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | head -1)
echo "perf binary: $P"; "$P" --version
