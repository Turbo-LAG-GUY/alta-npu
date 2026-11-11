#!/usr/bin/env bash
# 160 - allow NPU buffers in the A311D's 1 MiB AXI SRAM at every boot, and turn the switch on now.
# Why: Jordan asked on 2026-11-08 to keep the SRAM on across reboots, after the five test jobs passed (RESULTS.md,
# SRAM-one and SRAM-all). Root step 150's switch resets at every boot.
# What it does: installs /etc/modprobe.d/alta-etnaviv-axi-sram.conf (options etnaviv axi_sram=1) and writes 1 to
# /sys/module/etnaviv/parameters/axi_sram for this boot. etnaviv loads from the root filesystem, not the initramfs
# (see root step 080), so /etc/modprobe.d is enough and the initramfs stays as it is.
# Effect: the switch only allows these buffers; a program still has to ask (Teflon test build: ETNA_ML_AXI_SRAM). Nothing
# that starts at login asks, so if the SRAM ever froze the board, the freeze couldn't repeat itself at boot.
# Run:    sudo bash root-steps/160-npu-axi-sram-every-boot.sh     (no reboot needed)
# Revert: sudo rm /etc/modprobe.d/alta-etnaviv-axi-sram.conf, then reboot. Also do this before going back to the
#         1b277d6c driver with root step 110, which doesn't know this option (it would only log a warning at boot).
# Verify after a reboot: cat /sys/module/etnaviv/parameters/axi_sram   (expect: Y)
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
P=/sys/module/etnaviv/parameters/axi_sram
CONF=/etc/modprobe.d/alta-etnaviv-axi-sram.conf
die() { "$here/scripts/progress" 160 FAILED "$*"; echo "160: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 160 FAILED "root step stopped at line $LINENO"' ERR
[ "$(id -u)" -eq 0 ] || { echo "160: run it with sudo" >&2; exit 1; }
[ -e "$P" ] || die "no $P: the loaded etnaviv is not the root step 140 build"
[ "$(cat /sys/module/etnaviv/parameters/keep_powered)" = Y ] || die "keep_powered is not on"
wd="$(systemctl show -p RuntimeWatchdogUSec --value)"
[ -n "$wd" ] && [ "$wd" != 0 ] || die "the hardware watchdog is not armed (root step 090); it is the safety net for the SRAM"
[ ! -e "$here/runs/STOP" ] || die "the STOP latch is set: $(head -1 "$here/runs/STOP")"
"$here/scripts/progress" 160 STARTING "install $CONF (axi_sram=1 at every boot) and turn the switch on for this boot"
install -m 0644 "$here/system/alta-etnaviv-axi-sram.conf" "$CONF"
echo 1 > "$P"
[ "$(cat "$P")" = Y ] || die "axi_sram did not read back as Y"
load="$(modprobe --show-depends etnaviv)"     # read-only: prints what modprobe would load at the next boot
case "$load" in
  *axi_sram=1*) ;;
  *) die "modprobe does not pick up $CONF" ;;
esac
"$here/scripts/progress" 160 DONE "axi_sram on now and at every boot via $CONF. Undo: rm $CONF, reboot"
echo "Done: the AXI SRAM is allowed now and at every boot. No reboot needed."
