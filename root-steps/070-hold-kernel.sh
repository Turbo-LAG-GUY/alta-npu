#!/usr/bin/env bash
# 070 - freeze kernel upgrades while the patched module is in use: an upgrade boots a new kernel that
# has only the STOCK etnaviv, which silently brings the hangs back.
# Run: sudo bash root-steps/070-hold-kernel.sh      Revert: sudo apt-mark unhold <the packages it lists>
set -euxo pipefail
mapfile -t pk < <(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' 2>/dev/null | awk '
  $1 == "ii" && ($2 ~ /^linux-(generic|image-generic|headers-generic|image-arm64|headers-arm64)(-hwe-[0-9.]+)?$/ ||
                 $2 ~ /^linux-(image|dtb|headers)-(current|edge|legacy|vendor)-meson64$/) {print $2}')
if [ "${#pk[@]}" -eq 0 ]; then echo "no kernel meta-packages found - list them with: dpkg -l 'linux-*'"; exit 1; fi
apt-mark hold "${pk[@]}"
apt-mark showhold
