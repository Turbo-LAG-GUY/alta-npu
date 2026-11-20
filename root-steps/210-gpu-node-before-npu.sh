#!/usr/bin/env bash
# 210 - SUPERSEDED BY 211 (this order put desktop apps on the NPU). make the Mali (panfrost) load before the NPU (etnaviv), so the Mali always gets the first render node.
# Why: etnaviv and panfrost are loaded in parallel at boot, so which one gets /dev/dri/renderD128 is a race
# (boots -4 and -2 and this boot 45e4c1fe: etnaviv first; boot -3: panfrost first). Steam's interface
# (steamwebhelper, Chromium) opens the FIRST render node for VA-API video decoding before its sandbox starts,
# and keeps it open. On boots where the NPU wins the race, Steam holds the NPU's node (seen 2026-11-17:
# steamwebhelper fd 25 -> renderD128 = etnaviv, opened without O_CLOEXEC unlike every Mesa open, libva loaded,
# "Decoding: (null)" in webhelper_gpu.txt). Nothing is submitted, but it breaks "no NPU users while Steam is
# open" and is the first thing to fix before NPU work with Steam games. It also ends the node-number swaps
# between boots (see gpu-compute-speed notes: panvk and no-npu-guard).
# What it does: writes /etc/modprobe.d/alta-drm-node-order.conf with
#     softdep etnaviv pre: panfrost
# so loading etnaviv first loads panfrost. Nothing is loaded, unloaded or restarted now; it takes effect at the
# next boot. initramfs is not touched (etnaviv and panfrost are not in it, only their .conf copies are).
# Not guaranteed: if panfrost's probe is deferred (clocks/power not ready), etnaviv can still win; check below.
# Risk: low. Worst case the order stays random as today. NPU driver options (alta-etnaviv-*.conf) unchanged.
# Run:    sudo bash ~/alta-npu/root-steps/210-gpu-node-before-npu.sh          (--check: show only, no root)
# Revert: sudo rm /etc/modprobe.d/alta-drm-node-order.conf   (takes effect at the next boot)
# Verify after the next boot:
#     journalctl -k -b | grep -E 'Initialized (panfrost|etnaviv)'   -> panfrost on minor 1, etnaviv on minor 2
#     ls -l /dev/dri/by-path | grep render                           -> ffe40000.gpu-render -> renderD128
#     with Steam open: no steamwebhelper has renderD129 (now the NPU) open
set -euo pipefail
conf=/etc/modprobe.d/alta-drm-node-order.conf
line='softdep etnaviv pre: panfrost'

echo "Current boot order:"
journalctl -k -b --no-pager 2>/dev/null | grep -oE 'Initialized (etnaviv|panfrost) [0-9.]+ for [^ ]+ on minor [0-9]+' | sed 's/^/  /' || true
ls -l /dev/dri/by-path 2>/dev/null | awk '/render/ {print "  " $9 " -> " $11}'

if [ "${1:-}" = --check ]; then
  echo "--check: would write $conf with: $line"
  [ -f "$conf" ] && { echo "Already present:"; cat "$conf"; }
  exit 0
fi
[ "$(id -u)" = 0 ] || { echo "Run with sudo (or --check)."; exit 1; }

printf '# Alta root step 210 (%s): Mali gets the first render node, NPU the second.\n%s\n' "$(date -I)" "$line" > "$conf"
chmod 644 "$conf"
echo "Wrote $conf:"; cat "$conf"
echo "Check that modprobe sees it:"
modprobe --showconfig | grep -E '^softdep etnaviv' || { echo "WARNING: modprobe does not show the softdep"; exit 1; }
echo "Done. Takes effect at the next boot; nothing was loaded or unloaded now."
