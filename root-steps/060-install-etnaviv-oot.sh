#!/usr/bin/env bash
# 060 - install a patched etnaviv.ko built by scripts/build-etnaviv-oot.sh for the RUNNING kernel.
# Run: sudo bash root-steps/060-install-etnaviv-oot.sh build/etnaviv-<kver>-<stamp>
# Revert: rm "/lib/modules/$(uname -r)/updates/alta/etnaviv.ko"; depmod -a; (update-initramfs -u if it was used)
set -euxo pipefail
B="${1:?usage: $0 BUILD_DIR}"
KVER="$(uname -r)"
test -f "$B/etnaviv.ko"
grep -qx "kernel: $KVER" "$B/BUILDINFO.txt"
install -D -m 0644 "$B/etnaviv.ko" "/lib/modules/$KVER/updates/alta/etnaviv.ko"
depmod -a "$KVER"
test "$(modinfo -n etnaviv)" = "/lib/modules/$KVER/updates/alta/etnaviv.ko"
if lsinitramfs "/boot/initrd.img-$KVER" 2>/dev/null | grep -q 'etnaviv'; then update-initramfs -u -k "$KVER"; fi
echo "Installed. Reboot (never rmmod a live etnaviv), then run the post-install checks."
