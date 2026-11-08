#!/usr/bin/env bash
# 110 - install the rebuilt etnaviv.ko: the 0x7120 SH_EU fix it already had, plus a reset at every NPU power-up,
# three upstream reset/power fixes, and keep_powered=1, which keeps the NPU powered from boot, as Amlogic's own
# driver does.
# Why: the board froze at a program's first NPU job in 3 of 15 starts, each time right after the NPU had powered
# back up, and never in ~2,040 later jobs. The NPU is power-cycled after every 200 ms idle (pm_genpd_summary showed
# NNA "off-0") and comes back with no reset (research/2026-11-07-first-job-freeze.md).
# Source: ~/kernel-patch/lc-kernel-src HEAD 63bcbfd8cac6 + patches/etnaviv-reset-keepon/0001-0004:
#   0001 c182a2ae1388 force flush on power register ops (TI)     0002 1006d1f7c03d check all modules idle after reset
#   0003 4720915ab482 check MMUv2 disabled after reset            0004 alta: reset on power-up + keep_powered=1
# What it does: copies the module to /lib/modules/<kver>/updates/alta/etnaviv.ko, which modprobe prefers
# (/etc/depmod.d/ubuntu.conf: "search updates ubuntu built-in"), and runs depmod. The April build in
# kernel/drivers/gpu/drm/etnaviv/ stays untouched (still held and diverted by 075) as the fallback. Nothing is
# loaded, unloaded or rebooted, and the initramfs holds no etnaviv.ko, so it isn't touched. It takes effect at
# the next boot, which Jordan does.
# Run:    sudo bash root-steps/110-install-etnaviv-reset-keepon.sh
# Revert: sudo rm /lib/modules/6.12.73-g6b2d05eda916/updates/alta/etnaviv.ko
#         sudo depmod -a 6.12.73-g6b2d05eda916            then reboot: the April build (5717091b...) loads again.
#         To keep the new build but let the NPU power down when idle instead:
#         echo auto | sudo tee /sys/devices/platform/ff100000.npu/power/control   (until the next boot)
# Verify after the reboot:
#   modinfo -n etnaviv                    -> /lib/modules/6.12.73-g6b2d05eda916/updates/alta/etnaviv.ko
#   journalctl -k -b -g 'alta: reset'     -> etnaviv-gpu ff100000.npu: alta: reset on every power-up, kept powered (keep_powered=1)
#   cat /sys/devices/platform/ff100000.npu/power/{control,runtime_status}    -> on, active
#   sudo grep NNA /sys/kernel/debug/pm_genpd/pm_genpd_summary                 -> NNA on
# Note: 075's final check (modprobe loads the kernel/ path) is no longer true after this step. 075 is done; don't
# rerun it.
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
KVER=6.12.73-g6b2d05eda916
B="$here/build/etnaviv-6.12.73-g6b2d05eda916-20261107-230436-reset-keepon"
NEW=1b277d6c003922d5c32a94079cab5d05154e2a67a933467d6fea9f974ec85889   # the reviewed build (BUILDINFO.txt)
OLD=5717091bbe05739ebabc4604bf6eac3321648cec9f061a541cc69257f7a7a2f0   # the April build, the fallback
DST="/lib/modules/$KVER/updates/alta/etnaviv.ko"
APRIL="/lib/modules/$KVER/kernel/drivers/gpu/drm/etnaviv/etnaviv.ko"
die() { "$here/scripts/progress" 110 FAILED "$*"; echo "110: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 110 FAILED "root step stopped at line $LINENO"' ERR
[ "$(id -u)" -eq 0 ] || { echo "110: run it with sudo" >&2; exit 1; }
[ "$(uname -r)" = "$KVER" ] || die "running $(uname -r), not $KVER"
[ "$(sha256sum < "$B/etnaviv.ko" | cut -d' ' -f1)" = "$NEW" ] || die "$B/etnaviv.ko is not the reviewed build"
grep -qx "kernel: $KVER" "$B/BUILDINFO.txt" || die "BUILDINFO.txt is not for kernel $KVER"
[ "$(modinfo -F vermagic "$B/etnaviv.ko")" = "$(modinfo -k "$KVER" -F vermagic panfrost)" ] || die "vermagic differs from panfrost's"
[ "$(sha256sum < "$APRIL" | cut -d' ' -f1)" = "$OLD" ] || die "$APRIL is not the April build, so the fallback would be wrong"
n="$(lsinitramfs "/boot/initrd.img-$KVER" | grep -c 'etnaviv\.ko' || true)"
[ "$n" = 0 ] || die "the initramfs contains etnaviv.ko; this step does not handle that"
"$here/scripts/progress" 110 STARTING "install the reset-on-power-up + keep_powered etnaviv build (sha256 ${NEW:0:12}) as $DST; the April build stays as the fallback"

install -D -m 0644 "$B/etnaviv.ko" "$DST"
depmod -a "$KVER"

[ "$(modinfo -k "$KVER" -n etnaviv)" = "$DST" ] || die "after depmod, modprobe would not load $DST (search order?)"
[ "$(sha256sum < "$DST" | cut -d' ' -f1)" = "$NEW" ] || die "the installed copy differs from the reviewed build"
[ "$(sha256sum < "$APRIL" | cut -d' ' -f1)" = "$OLD" ] || die "the April build changed"
"$here/scripts/progress" 110 DONE "installed $DST (sha256 ${NEW:0:12}); modprobe picks it from the next boot. Revert: rm $DST; depmod -a $KVER; reboot"
echo "Done. It takes effect at the next boot. Reboot when you're ready."
