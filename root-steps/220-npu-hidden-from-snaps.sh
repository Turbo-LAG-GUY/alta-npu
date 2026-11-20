#!/usr/bin/env bash
# 220 - stop snap apps (Steam, Firefox, Chromium, ...) from opening the NPU's device nodes. The Mali stays open to them.
# Why: Steam's interface (steamwebhelper = Chromium) opens /dev/dri/renderD128 for VA-API video decoding before its
# sandbox starts and keeps it. Since step 211 renderD128 is ALWAYS the NPU (etnaviv), so with Steam open the NPU
# always has a Steam user (seen boot 9b62251c: steamwebhelper fd 25 -> renderD128, flags 0400002, no CLOEXEC, i.e.
# not Mesa). No jobs are sent, but it breaks "no NPU users while Steam is open", and it is the blocker for any NPU
# work next to Steam games. The browser snaps (Firefox, Chromium) carry the same VA-API code.
# How snaps get devices: each snap may only open devices that udev tagged for it (snap_<snap>_<app>, from
# /etc/udev/rules.d/70-snap.*.rules; the "opengl" interface tags every renderD* and card*). snap-confine builds the
# snap's device allowlist from those tags when the app starts.
# What it does: writes /etc/udev/rules.d/99-alta-npu-no-snaps.rules, which REMOVES every snap tag from the NPU's
# card and render node only (KERNELS=="etnaviv", same match as the existing 99-etnaviv-npu.rules). Tags are taken
# from the current 70-snap.*.rules, so a snap installed later is not covered: run this step again after
# installing a new snap that uses graphics.
# Takes effect at the next boot (no udevadm trigger, per the project rules). Snaps then get "permission denied" on
# the NPU node and skip it; Mesa, Chromium and libva all handle that. Your own NPU tools are not snaps and keep
# working (you stay in the render group).
# Risk: low. Worst case a snap that really wanted the NPU fails to; none should.
# Run:    sudo bash ~/alta-npu/root-steps/220-npu-hidden-from-snaps.sh        (--check: print the rule, no root)
# Revert: sudo rm /etc/udev/rules.d/99-alta-npu-no-snaps.rules   (next boot)
# Verify after the next boot, with Steam open:
#     udevadm info /dev/dri/by-path/platform-etnaviv-render | grep CURRENT_TAGS   -> no snap_ tags
#     no steamwebhelper/firefox process has the etnaviv render node open (ask Claude, or the NPU tray indicator)
set -euo pipefail
out=/etc/udev/rules.d/99-alta-npu-no-snaps.rules

tags=$(grep -hoE 'KERNEL=="(renderD|card)[^"]*".*TAG\+="snap_[^"]+"' /etc/udev/rules.d/70-snap.*.rules 2>/dev/null \
       | grep -oE 'snap_[^"]+' | sort -u)
[ -n "$tags" ] || { echo "No snap DRM tags found in /etc/udev/rules.d/70-snap.*.rules; nothing to do."; exit 1; }
drop=$(for t in $tags; do printf ', TAG-="%s"' "$t"; done)

rule=$(cat <<EOF
# Alta root step 220 ($(date -I)): no snap may open the NPU (etnaviv) nodes. Regenerate after installing a snap.
SUBSYSTEM=="drm", KERNEL=="card*", KERNELS=="etnaviv"${drop}
SUBSYSTEM=="drm", KERNEL=="renderD*", KERNELS=="etnaviv"${drop}
EOF
)

echo "Snap tags on the NPU's render node now:"
udevadm info /dev/dri/by-path/platform-etnaviv-render 2>/dev/null | grep CURRENT_TAGS | tr ':' '\n' | grep '^snap_' | sed 's/^/  /' || true

if [ "${1:-}" = --check ]; then printf '\n--check: would write %s:\n%s\n' "$out" "$rule"; exit 0; fi
[ "$(id -u)" = 0 ] || { echo "Run with sudo (or --check)."; exit 1; }

printf '%s\n' "$rule" > "$out"
chmod 644 "$out"
echo "Wrote $out"
udevadm verify "$out" 2>/dev/null || echo "(udevadm verify not available or reported issues; check the file)"
echo "Done. Reboot to apply (nothing was re-triggered now)."
