#!/usr/bin/env bash
# 040 - keep GNOME/mutter away from the NPU's DRM nodes.
# Run: sudo bash root-steps/040-udev-mutter-ignore.sh
# Revert: rm /etc/udev/rules.d/61-alta-npu-mutter-ignore.rules && udevadm control --reload
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
install -m 0644 "$here/system/61-alta-npu-mutter-ignore.rules" /etc/udev/rules.d/61-alta-npu-mutter-ignore.rules
udevadm verify /etc/udev/rules.d/61-alta-npu-mutter-ignore.rules 2>/dev/null || true   # systemd >= 255 only
udevadm control --reload
echo "Takes effect when the NPU device is next added (next boot). Do NOT run 'udevadm trigger' with a desktop up."
