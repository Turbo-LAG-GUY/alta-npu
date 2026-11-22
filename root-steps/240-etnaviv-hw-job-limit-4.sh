#!/usr/bin/env bash
# 240 - let the NPU take up to 4 jobs at a time again (etnaviv hw_job_limit=4, the driver's default), 2026-11-21.
# Why: root step 080 set hw_job_limit=1 on 2026-11-07 to test whether overlapping NPU jobs caused the hard freezes.
# They did not: the freezes went on with 1 (memory alta-freeze-findings: "hw_job_limit=1 is no fix") and ended with
# the reset / keep-powered driver (root steps 110 / 140). With 1 the kernel's scheduler gives the NPU the next job
# only after the previous one has finished, so queued submits (two programs, or one program that submits ahead) wait
# in the kernel instead of in the NPU's queue. Jordan asked for 4 (2026-11-21).
# What it does: replaces /etc/modprobe.d/alta-etnaviv-debug.conf (hw_job_limit=1) with
# ~/alta-npu/system/alta-etnaviv-hw-job-limit-4.conf (hw_job_limit=4). Nothing else: the parameter is read-only at
# runtime and etnaviv loads from the root filesystem (not the initramfs, see 080), so it applies at the next boot.
# Risk: every NPU run since 2026-11-07, including the 53 clean starts of the reset build, had hw_job_limit=1, so 4 is
# untested with the reset build. Our own programs wait for each job before the next, so they hand the kernel one job
# at a time anyway; only queued submits change. After the reboot: verify, then a check ladder (Jordan's OK) before
# other NPU work.
# Run:    sudo bash ~/alta-npu/root-steps/240-etnaviv-hw-job-limit-4.sh, then reboot.
# Revert: sudo install -m 0644 ~/alta-npu/system/alta-etnaviv-debug.conf /etc/modprobe.d/alta-etnaviv-debug.conf
#         (hw_job_limit=1 again), then reboot.
# Verify after the reboot: cat /sys/module/etnaviv/parameters/hw_job_limit    (expect 4)
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
trap '"$here/scripts/progress" 240 FAILED "root step stopped at line $LINENO"' ERR
"$here/scripts/progress" 240 STARTING "replace /etc/modprobe.d/alta-etnaviv-debug.conf: hw_job_limit 1 -> 4"
[ -d /sys/module/etnaviv/parameters ] && grep -q . /sys/module/etnaviv/parameters/hw_job_limit
grep -q 'hw_job_limit=1' /etc/modprobe.d/alta-etnaviv-debug.conf
install -m 0644 "$here/system/alta-etnaviv-hw-job-limit-4.conf" /etc/modprobe.d/alta-etnaviv-debug.conf
grep -q 'hw_job_limit=4' /etc/modprobe.d/alta-etnaviv-debug.conf
"$here/scripts/progress" 240 DONE "installed (hw_job_limit=4); it applies after the reboot"
echo "Reboot to apply."
