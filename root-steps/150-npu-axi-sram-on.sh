#!/usr/bin/env bash
# 150 - allow NPU buffers in the A311D's 1 MiB AXI SRAM, until the next boot.
# Why: the build from root step 140 refuses these buffers unless its axi_sram switch is on, and the switch is off at
# every boot, so a freeze during testing can't leave the feature enabled.
# What it does: writes 1 to /sys/module/etnaviv/parameters/axi_sram. Nothing is written to disk.
# Risk: this step alone does nothing visible. A program must also ask for the SRAM (Teflon: ETNA_ML_AXI_SRAM set). The
# first NPU job that uses it is the real test and has never been done with this driver. If the SRAM can't be reached
# as expected, the board can freeze; the hardware watchdog (root step 090) reboots it in about 80 s, with the switch off.
# Run:    sudo bash root-steps/150-npu-axi-sram-on.sh
# Revert: reboot, or echo 0 | sudo tee /sys/module/etnaviv/parameters/axi_sram
# Verify: cat /sys/module/etnaviv/parameters/axi_sram   (expect: Y)
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
P=/sys/module/etnaviv/parameters/axi_sram
die() { "$here/scripts/progress" 150 FAILED "$*"; echo "150: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 150 FAILED "root step stopped at line $LINENO"' ERR
[ "$(id -u)" -eq 0 ] || { echo "150: run it with sudo" >&2; exit 1; }
[ -e "$P" ] || die "no $P: the loaded etnaviv is not the root step 140 build (did you reboot after 140?)"
[ "$(cat /sys/module/etnaviv/parameters/keep_powered)" = Y ] || die "keep_powered is not on"
wd="$(systemctl show -p RuntimeWatchdogUSec --value)"
[ -n "$wd" ] && [ "$wd" != 0 ] || die "the hardware watchdog is not armed (root step 090); it is the safety net for the test"
[ ! -e "$here/runs/STOP" ] || die "the STOP latch is set: $(head -1 "$here/runs/STOP")"
if [ "$(cat "$P")" = Y ]; then
  "$here/scripts/progress" 150 NOTE "axi_sram was already on"
  exit 0
fi
"$here/scripts/progress" 150 STARTING "allow NPU buffers in the AXI SRAM until the next boot (axi_sram N -> Y)"
echo 1 > "$P"
[ "$(cat "$P")" = Y ] || die "axi_sram did not read back as Y"
"$here/scripts/progress" 150 DONE "axi_sram on until the next boot. Undo: reboot, or echo 0 > $P"
echo "Done: the AXI SRAM can be used until the next boot. No NPU job has run yet."
