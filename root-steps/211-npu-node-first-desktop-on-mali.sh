#!/usr/bin/env bash
# 211 - UNDO 210's order: load the NPU (etnaviv) BEFORE the Mali (panfrost), so desktop apps always pick the Mali.
# Why: 210 (softdep etnaviv pre: panfrost) made panfrost renderD128 and etnaviv renderD129, and on boot 7b415337
# GNOME apps (gnome-control-center, gnome-system-monitor, ...) opened the NPU's render node 3 times each, i.e.
# they draw on the NPU. Cause: Mesa's render-only path for the meson display (kmsro) takes the FIRST device that
# libdrm's drmGetDevices2 returns, and that follows /dev/dri's directory order, which lists the NEWEST node first
# (`ls -U /dev/dri`: card2 renderD129 card1 renderD128 card0). So the driver that loads LAST is the one desktop
# apps use. Before 210 that was a race that the Mali usually won by loading last; 210 made the NPU always last.
# What it does: rewrites /etc/modprobe.d/alta-drm-node-order.conf (from 210) to
#     softdep panfrost pre: etnaviv
# so etnaviv always loads first (renderD128, listed last) and panfrost second (renderD129, listed first, chosen).
# Trade-off: Steam's interface (Chromium VA-API) opens /dev/dri/renderD128 = the NPU and keeps it open, as it did
# on most boots before 210. That is an open handle with no jobs; it is solved separately, not by node order.
# Nothing is loaded, unloaded or restarted now; takes effect at the next boot. initramfs is not touched.
# Not guaranteed: if etnaviv's probe is deferred, panfrost can still load first; check below after the reboot.
# Run:    sudo bash ~/alta-npu/root-steps/211-npu-node-first-desktop-on-mali.sh      (--check: show only)
# Revert: sudo rm /etc/modprobe.d/alta-drm-node-order.conf   (back to the boot race, next boot)
# Verify after the next boot:
#     journalctl -k -b | grep -E 'Initialized (panfrost|etnaviv)'  -> etnaviv minor 1, panfrost minor 2
#     ls -U /dev/dri                                                -> renderD129 (Mali) listed before renderD128
#     open GNOME Settings, then: ls -l /proc/$(pgrep -n gnome-control)/fd | grep dri  -> no renderD128 (NPU)
set -euo pipefail
conf=/etc/modprobe.d/alta-drm-node-order.conf
line='softdep panfrost pre: etnaviv'

echo "This boot:"
journalctl -k -b --no-pager 2>/dev/null | grep -oE 'Initialized (etnaviv|panfrost) [0-9.]+ for [^ ]+ on minor [0-9]+' | sed 's/^/  /' || true
echo "  /dev/dri scan order: $(ls -U /dev/dri | tr '\n' ' ')"
[ -f "$conf" ] && { echo "Current $conf:"; sed 's/^/  /' "$conf"; }

if [ "${1:-}" = --check ]; then echo "--check: would write $conf with: $line"; exit 0; fi
[ "$(id -u)" = 0 ] || { echo "Run with sudo (or --check)."; exit 1; }

printf '# Alta root step 211 (%s): NPU loads first (renderD128), Mali last (renderD129) so desktop GL picks the Mali.\n# Replaces step 210 (softdep etnaviv pre: panfrost), which put desktop apps on the NPU.\n%s\n' "$(date -I)" "$line" > "$conf"
chmod 644 "$conf"
echo "Wrote $conf:"; sed 's/^/  /' "$conf"
modprobe --showconfig | grep -E '^softdep panfrost' || { echo "WARNING: modprobe does not show the softdep"; exit 1; }
modprobe --showconfig | grep -E '^softdep etnaviv' && { echo "WARNING: an etnaviv softdep is still configured somewhere"; exit 1; }
echo "Done. Reboot to apply; until then, close GNOME Settings / System Monitor and avoid starting new apps."
