#!/usr/bin/env bash
# 140 - install the etnaviv build that can give the NPU buffers in its 1 MiB AXI SRAM.
# Why: Amlogic's own driver uses this SRAM (physical 0xff000000) to keep NPU traffic off DDR. Mainline etnaviv and
# Teflon never touch it. Jordan asked to use it (2026-11-08).
# Source: build/src-lc-63bcbfd8cac6 + patches/etnaviv-reset-keepon/0001-0004 (the build loaded today, root step 110)
#         + patches/etnaviv-axi-sram/0005, which adds a private GEM flag and the axi_sram module parameter.
# What it does: copies the module to /lib/modules/<kver>/updates/alta/etnaviv.ko and runs depmod. Nothing is loaded,
# unloaded or rebooted. It takes effect at the next boot, which Jordan does.
# Risk: with axi_sram off, which is how it boots, the only difference from the current driver is that the ioctl
# accepts one more flag and refuses it with ENODEV. This build has not run on the board yet.
# Run:    sudo bash root-steps/140-install-etnaviv-axi-sram.sh     then reboot
# Revert: sudo bash root-steps/110-install-etnaviv-reset-keepon.sh  then reboot (puts back the current build, 1b277d6c...)
# Verify after the reboot:
#   modinfo -n etnaviv                                 -> /lib/modules/6.12.73-g6b2d05eda916/updates/alta/etnaviv.ko
#   cat /sys/module/etnaviv/parameters/axi_sram        -> N
#   journalctl -k -b -g 'alta: reset'                  -> the same marker line as today
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
KVER=6.12.73-g6b2d05eda916
B="$here/build/etnaviv-6.12.73-g6b2d05eda916-20261108-182004-axi-sram"
NEW=e5a72dd9514b76b594557962d410891b03a60dd95b4851a1b06d9c3fc9361bcd   # the reviewed build
CUR=1b277d6c003922d5c32a94079cab5d05154e2a67a933467d6fea9f974ec85889   # the build it replaces (root step 110)
DST="/lib/modules/$KVER/updates/alta/etnaviv.ko"
die() { "$here/scripts/progress" 140 FAILED "$*"; echo "140: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 140 FAILED "root step stopped at line $LINENO"' ERR
[ "$(id -u)" -eq 0 ] || { echo "140: run it with sudo" >&2; exit 1; }
[ "$(uname -r)" = "$KVER" ] || die "running $(uname -r), not $KVER"
[ "$(sha256sum < "$B/etnaviv.ko" | cut -d' ' -f1)" = "$NEW" ] || die "$B/etnaviv.ko is not the reviewed build"
grep -qx "kernel: $KVER" "$B/BUILDINFO.txt" || die "BUILDINFO.txt is not for kernel $KVER"
[ "$(modinfo -F vermagic "$B/etnaviv.ko")" = "$(modinfo -k "$KVER" -F vermagic panfrost)" ] || die "vermagic differs from panfrost's"
[ "$(sha256sum < "$DST" | cut -d' ' -f1)" = "$CUR" ] || die "$DST is not the expected current build"
n="$(lsinitramfs "/boot/initrd.img-$KVER" | grep -c 'etnaviv\.ko' || true)"
[ "$n" = 0 ] || die "the initramfs contains etnaviv.ko; this step does not handle that"
"$here/scripts/progress" 140 STARTING "install the AXI SRAM etnaviv build (sha256 ${NEW:0:12}) as $DST, replacing ${CUR:0:12}"

install -D -m 0644 "$B/etnaviv.ko" "$DST"
depmod -a "$KVER"

[ "$(modinfo -k "$KVER" -n etnaviv)" = "$DST" ] || die "after depmod, modprobe would not load $DST"
[ "$(sha256sum < "$DST" | cut -d' ' -f1)" = "$NEW" ] || die "the installed copy differs from the reviewed build"
"$here/scripts/progress" 140 DONE "installed $DST (sha256 ${NEW:0:12}); modprobe picks it from the next boot, with axi_sram off. Revert: root step 110, reboot"
echo "Done. It takes effect at the next boot. Reboot when you're ready."
