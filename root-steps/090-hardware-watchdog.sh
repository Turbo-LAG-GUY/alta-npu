#!/usr/bin/env bash
# 090 - arm the SoC's hardware watchdog through systemd, so a total freeze reboots the board by itself
# about a minute later instead of needing an unplug. Today nothing pets /dev/watchdog0 (the Meson GXBB
# watchdog; systemd has RuntimeWatchdogSec=off). systemd reads the setting at boot.
# Side effect: if PID 1 itself is stuck for 60 s (for example, extreme swap thrashing), the board resets too.
# Run: sudo bash root-steps/090-hardware-watchdog.sh, then reboot (the reboot after 080 is enough).
# Revert: sudo rm /etc/systemd/system.conf.d/alta-watchdog.conf, then reboot.
# Verify after the reboot: systemctl show -p RuntimeWatchdogUSec     (expect RuntimeWatchdogUSec=1min)
#                          journalctl -b -g 'hardware watchdog' | head
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
die() { "$here/scripts/progress" 090 FAILED "$*"; echo "090: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 090 FAILED "root step stopped at line $LINENO"' ERR
"$here/scripts/progress" 090 STARTING "install /etc/systemd/system.conf.d/alta-watchdog.conf (RuntimeWatchdogSec=60s)"
[ -c /dev/watchdog0 ] || die "no /dev/watchdog0"
grep -qx 'Meson GXBB Watchdog' /sys/class/watchdog/watchdog0/identity || die "watchdog0 is not the Meson GXBB watchdog"
if systemctl is-active --quiet watchdog.service wd_keepalive.service; then
  die "a watchdog daemon is already running - use it or systemd, not both"
fi
install -d -m 0755 /etc/systemd/system.conf.d
install -m 0644 "$here/system/alta-watchdog.conf" /etc/systemd/system.conf.d/alta-watchdog.conf
"$here/scripts/progress" 090 DONE "installed; it applies after the reboot"
echo "Reboot to apply."
