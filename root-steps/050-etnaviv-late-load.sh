#!/usr/bin/env bash
# 050 - ONLY IF NEEDED (handoff section 8.4): load etnaviv late from a service instead of udev coldplug.
# Run: sudo bash root-steps/050-etnaviv-late-load.sh
# Revert: systemctl disable alta-npu-load.service; rm /etc/systemd/system/alta-npu-load.service \
#         /etc/modprobe.d/alta-etnaviv.conf; systemctl daemon-reload; update-initramfs -u
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
install -m 0644 "$here/system/alta-etnaviv.conf" /etc/modprobe.d/alta-etnaviv.conf
install -m 0644 "$here/system/alta-npu-load.service" /etc/systemd/system/alta-npu-load.service
systemctl daemon-reload
systemctl enable alta-npu-load.service
if command -v update-initramfs >/dev/null; then update-initramfs -u -k "$(uname -r)"; fi
echo "Reboot to apply."
