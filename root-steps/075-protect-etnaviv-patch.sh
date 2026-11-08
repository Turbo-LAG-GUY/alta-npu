#!/usr/bin/env bash
# 075 - keep the patched etnaviv.ko (0x7120 SH_EU clock-gating fix) through package updates.
# On this board the kit's 070 holds nothing: the kernel comes from Libre Computer packages, not Ubuntu's.
#  1. Holds the Libre Computer kernel packages, so apt (and unattended-upgrades) never upgrades or removes
#     them. Either would bring back the stock driver, or a kernel without the fix.
#  2. Diverts the module file with dpkg-divert. Even a reinstall of the kernel package then writes the stock
#     etnaviv.ko to etnaviv.ko.distrib and leaves the patched one where the kernel loads it from.
# It doesn't load, rebuild or reboot anything; the running system is unchanged. No reboot is needed.
# Run:    sudo bash root-steps/075-protect-etnaviv-patch.sh
# Revert: sudo apt-mark unhold linux-image-6.12.73-g6b2d05eda916 linux-headers-6.12.73-g6b2d05eda916 \
#              linux-image-lc-lts-arm64 linux-headers-lc-lts-arm64
#         sudo dpkg-divert --local --no-rename --remove /lib/modules/6.12.73-g6b2d05eda916/kernel/drivers/gpu/drm/etnaviv/etnaviv.ko
#         sudo rm /lib/modules/6.12.73-g6b2d05eda916/kernel/drivers/gpu/drm/etnaviv/etnaviv.ko.distrib
# Verify: apt-mark showhold;  dpkg-divert --truename <module path>   (expect ...etnaviv.ko.distrib)
# A newer kernel means rebuilding the patch for it first (handoff section 8.6), then unholding.
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
KVER=6.12.73-g6b2d05eda916
F="/lib/modules/$KVER/kernel/drivers/gpu/drm/etnaviv/etnaviv.ko"
PATCHED=5717091bbe05739ebabc4604bf6eac3321648cec9f061a541cc69257f7a7a2f0   # ~/kernel-patch/lc-kernel-src build
STOCK_MD5=522a0a5267508f0fec61be9b9c49a0d8                                  # the package's own etnaviv.ko
die() { "$here/scripts/progress" 075 FAILED "$*"; echo "075: $*" >&2; exit 1; }
trap '"$here/scripts/progress" 075 FAILED "root step stopped at line $LINENO"' ERR
[ "$(id -u)" -eq 0 ] || { echo "075: run it with sudo" >&2; exit 1; }
[ "$(uname -r)" = "$KVER" ] || die "running $(uname -r), not the patched kernel $KVER"
[ "$(sha256sum < "$F" | cut -d' ' -f1)" = "$PATCHED" ] || die "$F is not the patched build - nothing to protect; restore it first"
[ "$(md5sum < "$F.orig" | cut -d' ' -f1)" = "$STOCK_MD5" ] || die "$F.orig is not the package's stock module"
grep -q "with Linux $KVER\"" /etc/default/grub || die "GRUB_DEFAULT does not pin $KVER"
"$here/scripts/progress" 075 STARTING "hold the Libre Computer kernel packages + dpkg-divert the patched etnaviv.ko"

mapfile -t pk < <(dpkg-query -W -f='${db:Status-Abbrev} ${Package}\n' "linux-image-$KVER" "linux-headers-$KVER" \
                  linux-image-lc-lts-arm64 linux-headers-lc-lts-arm64 2>/dev/null | awk '$1 == "ii" {print $2}')
[ "${#pk[@]}" -gt 0 ] || die "none of the Libre Computer kernel packages is installed"
apt-mark hold "${pk[@]}"

if [ "$(dpkg-divert --truename "$F")" != "$F.distrib" ]; then
  dpkg-divert --local --no-rename --divert "$F.distrib" --add "$F"
fi
install -m 0644 "$F.orig" "$F.distrib"   # the diverted name holds the package's version, as dpkg expects

# Checks: nothing about the running module changed.
[ "$(dpkg-divert --truename "$F")" = "$F.distrib" ] || die "diversion not registered"
[ "$(sha256sum < "$F" | cut -d' ' -f1)" = "$PATCHED" ] || die "the patched module changed during the step"
[ "$(modinfo -k "$KVER" -n etnaviv)" = "$F" ] || die "modprobe would no longer load $F"
held="$(apt-mark showhold | tr '\n' ' ')"
"$here/scripts/progress" 075 DONE "held: $held; $F diverted to .distrib (stock copy); patched sha256 ${PATCHED:0:12} still in place"
echo "Done: held $held and diverted $F."
