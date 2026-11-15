#!/usr/bin/env bash
# 200 - install the hang-dump watcher as a boot service, so NPU hang dumps are saved without anyone's password.
# Why: after an NPU hang the kernel keeps the dump (/sys/class/devcoredump/devcd*/data) for about 5 minutes and
# only root can read it. Two dumps were lost because nobody with root was at the board (2026-11-14, 2026-11-15).
# What it does:
#   1. copies scripts/devcd-watch to /usr/local/sbin/alta-devcd-watch (root-owned, so the root service does not
#      run a file a user can change)
#   2. writes /etc/systemd/system/alta-devcd-watch.service and enables + starts it (starts at every boot)
# The service checks once a second for a dump, copies it to ~/alta-npu/incidents/devcoredumps/ (owned by jordan),
# adds a line to index.log there, and keeps only the newest 20 dumps (KEEP below; about 40 KiB each).
# It only reads the dump; it does not touch etnaviv, the NPU, udev or Mesa. Cost: one directory look per second.
# Risk: low. No reboot.
# Run:    sudo bash ~/alta-npu/root-steps/200-devcd-watch-service.sh        (KEEP=10 sudo -E bash ... for 10)
# Revert: sudo systemctl disable --now alta-devcd-watch; sudo rm /etc/systemd/system/alta-devcd-watch.service
#         /usr/local/sbin/alta-devcd-watch; sudo systemctl daemon-reload
# Verify: ends with "200 done"; systemctl is-active alta-devcd-watch; journalctl -u alta-devcd-watch -n 5
# After changing scripts/devcd-watch: run this step again.
set -euo pipefail
KEEP=${KEEP:-20}
SRC=/home/jordan/alta-npu/scripts/devcd-watch
[ "$(id -u)" = 0 ] || { echo "run with sudo: sudo bash $0"; exit 1; }
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
bash -n "$SRC"
systemctl stop alta-devcd-watch-once 2>/dev/null || true
install -o root -g root -m 0755 "$SRC" /usr/local/sbin/alta-devcd-watch
cat > /etc/systemd/system/alta-devcd-watch.service <<UNIT
[Unit]
Description=Alta NPU: copy etnaviv hang dumps (devcoredump) to the kit, keep the newest $KEEP

[Service]
Environment=DEVCD_KEEP=$KEEP
ExecStart=/usr/local/sbin/alta-devcd-watch
Restart=on-failure
RestartSec=5
Nice=5

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable alta-devcd-watch >/dev/null 2>&1
systemctl restart alta-devcd-watch
sleep 1
systemctl is-active alta-devcd-watch
systemctl is-enabled alta-devcd-watch
echo "200 done"
