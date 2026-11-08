#!/usr/bin/env bash
# 020 - memory safety net: compressed swap in RAM (zram) + an early OOM killer that prefers compilers.
# Skips whatever already exists (Armbian ships its own zram; Ubuntu desktop ships systemd-oomd).
# Run: sudo bash root-steps/020-memory-safety-net.sh
# Revert: apt-get purge zram-tools earlyoom  (restores nothing else; backups are *.alta-bak)
set -euxo pipefail
if ls -d /sys/block/zram* >/dev/null 2>&1; then
  echo "zram already present - not adding another"
else
  apt-get install -y zram-tools
  [ -f /etc/default/zramswap ] && cp -n /etc/default/zramswap /etc/default/zramswap.alta-bak
  printf 'ALGO=zstd\nPERCENT=50\nPRIORITY=100\n' > /etc/default/zramswap
  systemctl restart zramswap
fi
if systemctl is-active --quiet systemd-oomd; then
  echo "systemd-oomd is active - not adding earlyoom (one OOM daemon is enough)"
else
  apt-get install -y earlyoom
  [ -f /etc/default/earlyoom ] && cp -n /etc/default/earlyoom /etc/default/earlyoom.alta-bak
  cat > /etc/default/earlyoom <<'CONF'
# No spaces inside the regexes, so no inner quotes are needed when systemd splits $EARLYOOM_ARGS.
EARLYOOM_ARGS="-m 6 -s 10 -r 3600 --prefer ^(cc1|cc1plus|ld|ld.bfd|ld.gold|ninja|make|python3)$ --avoid ^(sshd|systemd|systemd-journal|tmux|gnome-shell|Xwayland)$"
CONF
  systemctl enable --now earlyoom
  systemctl restart earlyoom
fi
swapon --show || true
