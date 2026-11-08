#!/usr/bin/env bash
# 010 - keep kernel logs across lockups; let the user (and npu-run) read the kernel log and open
#       /dev/dri/renderD* over SSH (SSH sessions get no seat ACLs, so group "render" is needed).
# Run: sudo bash root-steps/010-journal-and-log-access.sh
# Revert: gpasswd -d "$SUDO_USER" systemd-journal; gpasswd -d "$SUDO_USER" render  (keep the persistent journal)
set -euxo pipefail
mkdir -p /var/log/journal
systemd-tmpfiles --create --prefix /var/log/journal
systemctl restart systemd-journald
journalctl --flush || true
usermod -aG systemd-journal "${SUDO_USER:?run via sudo}"
getent group render >/dev/null && usermod -aG render "$SUDO_USER"
for s in armbian-ramlog log2ram; do
  if systemctl is-active --quiet "$s"; then
    echo "NOTE: $s keeps /var/log in RAM - logs from a hard lockup can still be lost. See handoff section 6.1."
  fi
done
echo "Log out and back in (new SSH session) for the group change to apply."
