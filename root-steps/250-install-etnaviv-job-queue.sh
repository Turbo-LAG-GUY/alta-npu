#!/usr/bin/env bash
# 250 - install the etnaviv build with the adjustable job queue and program priority, and let the user change it.
# Why: Jordan wants hw_job_limit dynamic (2026-11-21): changeable while running, dropping to 1 by itself after a
# recovered hang (with a warning and a choice to put it back or reboot), program priority by name, and a window
# (npu-status/npu_control.py, "Job queue…" in the NPU indicator's menu) instead of the terminal.
# Source: build/src-lc-63bcbfd8cac6 + patches 0001-0005 (the build loaded today, root step 140)
#         + patches/etnaviv-job-limit/0006 (hw_job_limit writable 1-4, hw_job_auto, hw_job_trips, hw_job_trip_from,
#           hw_job_turns) + 0007 (hw_job_prio by process name, hw_job_clients, hw_job_prio_guard_ms).
#         Research notes: research/hw-job-limit/PLAN.md.
# What it does:
#   1. copies the module to /lib/modules/<kver>/updates/alta/etnaviv.ko and runs depmod (as 140 did);
#   2. writes /usr/local/sbin/alta-npu-queue-perms and /etc/udev/rules.d/71-alta-npu-queue.rules: when etnaviv
#      loads (and again when its render node appears), the five writable hw_job_* parameter files get group
#      <your user's group> and mode 0664, so the window and the indicator can change them without sudo.
#      Nothing else becomes writable; the NPU device nodes are untouched.
#   Nothing is loaded, unloaded or rebooted. /etc/modprobe.d/alta-etnaviv-debug.conf (hw_job_limit=1) stays, so the
#   board boots with one job at a time exactly as now; the indicator then applies the limit saved in the window
#   (once per boot, not while runs/STOP exists or after a trip). Root step 240 is not needed with this one.
# Risk: this build has not run on the board yet. At the boot defaults (limit 1, turns off, no priority table) the
# scheduler does the same as today; what is new is the per-job credit callback (12 / limit credits from a pool of
# 12) and three cheap checks per job. The NPU ladder in PLAN.md (Jordan's OK) comes before raising the limit.
# Run:    sudo bash ~/alta-npu/root-steps/250-install-etnaviv-job-queue.sh            then reboot
# Revert: sudo bash ~/alta-npu/root-steps/250-install-etnaviv-job-queue.sh --revert   then reboot
#         (puts back the 140 build e5a72dd9... and removes the udev rule and helper)
# Verify after the reboot:
#   modinfo -n etnaviv                                 -> /lib/modules/6.12.73-g6b2d05eda916/updates/alta/etnaviv.ko
#   journalctl -k -b -g 'alta: hw_job_limit'           -> hw_job_limit 1 (writable, 1-4), hw_job_auto on, ...
#   ls -l /sys/module/etnaviv/parameters/hw_job_limit  -> -rw-rw-r-- root <your group>
#   cat /sys/module/etnaviv/parameters/hw_job_clients  -> the programs that have the NPU open
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
KVER=6.12.73-g6b2d05eda916
B="$here/build/etnaviv-6.12.73-g6b2d05eda916-20261121-163831-job-prio"
OLD_B="$here/build/etnaviv-6.12.73-g6b2d05eda916-20261108-182004-axi-sram"
NEW=f317fc8f4d6ec5b887de213aa539801961f7ccfdf007df730880ff330134e88f   # 0001-0007, the reviewed build
CUR=e5a72dd9514b76b594557962d410891b03a60dd95b4851a1b06d9c3fc9361bcd   # the build it replaces (root step 140)
DST="/lib/modules/$KVER/updates/alta/etnaviv.ko"
HELPER=/usr/local/sbin/alta-npu-queue-perms
RULE=/etc/udev/rules.d/71-alta-npu-queue.rules
die() { "$here/scripts/progress" 250 FAILED "$*"; echo "250: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 250 FAILED "root step stopped at line $LINENO"' ERR
[ "$(id -u)" -eq 0 ] || { echo "250: run it with sudo" >&2; exit 1; }
[ "$(uname -r)" = "$KVER" ] || die "running $(uname -r), not $KVER"
sha() { sha256sum < "$1" | cut -d' ' -f1; }

if [ "${1:-}" = "--revert" ]; then
  [ "$(sha "$OLD_B/etnaviv.ko")" = "$CUR" ] || die "$OLD_B/etnaviv.ko is not the 140 build"
  "$here/scripts/progress" 250 STARTING "revert: put back the 140 build (${CUR:0:12}) as $DST, remove $RULE and $HELPER"
  install -D -m 0644 "$OLD_B/etnaviv.ko" "$DST"
  depmod -a "$KVER"
  rm -f "$RULE" "$HELPER"
  udevadm control --reload
  [ "$(sha "$DST")" = "$CUR" ] || die "the installed copy is not the 140 build"
  "$here/scripts/progress" 250 DONE "reverted to the 140 build (${CUR:0:12}); udev rule and helper removed; takes effect at the next boot"
  echo "Reverted. It takes effect at the next boot."
  exit 0
fi

USER_NAME="${SUDO_USER:-}"
[ -n "$USER_NAME" ] && [ "$USER_NAME" != root ] || die "run it with sudo from your own account (SUDO_USER is empty or root)"
GROUP="$(id -gn "$USER_NAME")"
[ "$(sha "$B/etnaviv.ko")" = "$NEW" ] || die "$B/etnaviv.ko is not the reviewed build"
grep -qx "kernel: $KVER" "$B/BUILDINFO.txt" || die "BUILDINFO.txt is not for kernel $KVER"
[ "$(modinfo -F vermagic "$B/etnaviv.ko")" = "$(modinfo -k "$KVER" -F vermagic panfrost)" ] || die "vermagic differs from panfrost's"
[ "$(sha "$DST")" = "$CUR" ] || die "$DST is not the expected current build (140)"
n="$(lsinitramfs "/boot/initrd.img-$KVER" | grep -c 'etnaviv\.ko' || true)"
[ "$n" = 0 ] || die "the initramfs contains etnaviv.ko; this step does not handle that"
"$here/scripts/progress" 250 STARTING "install the job-queue etnaviv build (sha256 ${NEW:0:12}) as $DST, replacing ${CUR:0:12}; udev write access to hw_job_* for group $GROUP"

install -D -m 0644 "$B/etnaviv.ko" "$DST"
depmod -a "$KVER"

cat > "$HELPER" <<EOF
#!/bin/sh
# alta-npu root step 250: let group $GROUP change the NPU job-queue settings of the alta etnaviv driver.
# Run by $RULE when etnaviv loads and when its render node appears. Revert: root step 250 --revert.
cd /sys/module/etnaviv/parameters 2>/dev/null || exit 0
for p in hw_job_limit hw_job_auto hw_job_turns hw_job_prio_guard_ms hw_job_prio; do
  [ -e "\$p" ] || continue
  chgrp $GROUP "\$p" && chmod 0664 "\$p"
done
exit 0
EOF
chmod 0755 "$HELPER"

cat > "$RULE" <<EOF
# alta-npu root step 250: group $GROUP may change the NPU job-queue settings (hw_job_*), see $HELPER.
# Both events: the module's own, and the render node's, in case etnaviv loaded before udevd was listening.
ACTION=="add", SUBSYSTEM=="module", KERNEL=="etnaviv", RUN+="$HELPER"
ACTION=="add", SUBSYSTEM=="drm", KERNEL=="renderD*", KERNELS=="etnaviv", RUN+="$HELPER"
EOF
chmod 0644 "$RULE"
udevadm control --reload

[ "$(modinfo -k "$KVER" -n etnaviv)" = "$DST" ] || die "after depmod, modprobe would not load $DST"
[ "$(sha "$DST")" = "$NEW" ] || die "the installed copy differs from the reviewed build"
"$here/scripts/progress" 250 DONE "installed $DST (sha256 ${NEW:0:12}), $RULE, $HELPER (group $GROUP); modprobe picks it from the next boot, at hw_job_limit=1. Revert: 250 --revert, reboot"
echo "Done. It takes effect at the next boot. Reboot when you're ready."
