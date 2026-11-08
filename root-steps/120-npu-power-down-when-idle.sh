#!/usr/bin/env bash
# 120 - let the NPU power down when idle again, until the next boot, to test the new driver's reset at power-up.
# Why: the build from root step 110 keeps the NPU powered from boot (keep_powered=1), and that passed T3 x2, T4 and
# T5. The same build also has the "power reset trick": when the NPU powers down, the driver holds its reset, and
# on the next job it powers back up with the reset still held, waits 10 ms, releases it, waits 20 ms and soft-resets
# it before use, as Amlogic's own driver does. That path has never run yet. On the old driver, 3 of 15 power-ups at
# a program start froze the board; the wake-up test (scripts/npu-wake) checks the new path with ~50 power-ups.
# What it does: writes "auto" to the NPU's power/control, so the driver powers it down 200 ms after its last job.
# Nothing is written to disk: the next boot is back to "kept powered", including after a watchdog reset.
# Risk: this step only powers the NPU down, like the old driver did after every job. The power-ups that matter
# come with the wake-up test, and if one freezes the board, the hardware watchdog (root step 090) reboots it in
# ~80 s, back to kept powered.
# Run:    sudo bash root-steps/120-npu-power-down-when-idle.sh
# Revert: reboot (preferred: back to kept powered, with the reset at boot), or
#         echo on | sudo tee /sys/devices/platform/ff100000.npu/power/control   (wakes the NPU through the new path)
# Verify: cat /sys/devices/platform/ff100000.npu/power/{control,runtime_status}   (expect: auto, suspended)
#         The tray indicator then shows a percentage again instead of "kept powered on".
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
D=/sys/devices/platform/ff100000.npu
P="$D/power"
die() { "$here/scripts/progress" 120 FAILED "$*"; echo "120: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 120 FAILED "root step stopped at line $LINENO"' ERR
[ "$(id -u)" -eq 0 ] || { echo "120: run it with sudo" >&2; exit 1; }
[ "$(basename "$(readlink -f "$D/driver")")" = etnaviv-gpu ] || die "ff100000.npu is not bound to etnaviv"
[ -e /sys/module/etnaviv/parameters/keep_powered ] || die "the loaded etnaviv is not the root step 110 build (no keep_powered parameter)"
n="$(journalctl -k -b 0 -q --no-pager | grep -c 'alta: reset on every power-up' || true)"
[ "$n" -ge 1 ] || die "no 'alta: reset on every power-up' line in this boot's kernel log"
wd="$(systemctl show -p RuntimeWatchdogUSec --value)"
[ -n "$wd" ] && [ "$wd" != 0 ] || die "the hardware watchdog is not armed (root step 090); it is the safety net for the test"
if [ "$(cat "$P/control")" = auto ]; then
  "$here/scripts/progress" 120 NOTE "already auto (runtime_status=$(cat "$P/runtime_status"))"
  exit 0
fi
"$here/scripts/progress" 120 STARTING "let the NPU power down when idle (power/control on -> auto) until the next boot; before: runtime_status=$(cat "$P/runtime_status") suspended_ms=$(cat "$P/runtime_suspended_time")"
echo auto > "$P/control"
sleep 3   # the driver powers the NPU down 200 ms after the write
st="$(cat "$P/runtime_status")"
[ "$st" = suspended ] || die "control=auto but runtime_status=$st after 3 s"
nna="$(grep -E '^NNA ' /sys/kernel/debug/pm_genpd/pm_genpd_summary 2>/dev/null | awk '{print $2}' || true)"
"$here/scripts/progress" 120 DONE "the NPU powers down when idle again (control=auto, runtime_status=$st, NNA domain ${nna:-unknown}) until the next boot. Undo: reboot, or echo on > $P/control"
echo "Done: the NPU now powers down when idle, until the next boot. NNA domain: ${nna:-unknown}"
