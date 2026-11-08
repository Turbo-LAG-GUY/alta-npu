#!/usr/bin/env bash
# 030 - boot to text mode for NPU work (no GNOME; frees roughly 1 GB and removes the display-stack failure modes).
# Run: sudo bash root-steps/030-headless-boot.sh     Revert: sudo systemctl set-default graphical.target
set -euxo pipefail
systemctl set-default multi-user.target
echo "Takes effect at the next boot. One-off desktop: sudo systemctl start display-manager"
