#!/usr/bin/env bash
# 080 - handoff section 12.3 step 2: run ONE NPU hardware job at a time (etnaviv hw_job_limit=1;
# the default is 4), to test whether overlapping NPU jobs cause the hard freezes (incident
# 2026-11-07-1405-T2b). etnaviv loads from the root filesystem, not the initramfs (it loaded at
# 11.7 s, after root was mounted at 5.3 s), so /etc/modprobe.d is enough and the initramfs stays as is.
# The parameter is read-only at runtime, so it takes effect at the next boot.
# Run: sudo bash root-steps/080-etnaviv-hw-job-limit.sh, then reboot.
# Revert: sudo rm /etc/modprobe.d/alta-etnaviv-debug.conf, then reboot.
# Verify after the reboot: cat /sys/module/etnaviv/parameters/hw_job_limit    (expect 1)
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
trap '"$here/scripts/progress" 080 FAILED "root step stopped at line $LINENO"' ERR
"$here/scripts/progress" 080 STARTING "install /etc/modprobe.d/alta-etnaviv-debug.conf (hw_job_limit=1)"
[ -d /sys/module/etnaviv/parameters ] && grep -q . /sys/module/etnaviv/parameters/hw_job_limit
install -m 0644 "$here/system/alta-etnaviv-debug.conf" /etc/modprobe.d/alta-etnaviv-debug.conf
"$here/scripts/progress" 080 DONE "installed; it applies after the reboot"
echo "Reboot to apply."
