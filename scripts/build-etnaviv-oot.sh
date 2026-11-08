#!/usr/bin/env bash
# build-etnaviv-oot.sh - build a patched etnaviv.ko for the RUNNING kernel, out of tree.
# Installs NOTHING. At the end it prints the root-only install commands for Jordan to run.
#
# Usage: scripts/build-etnaviv-oot.sh /path/to/kernel-source [scripts/apply-0x7120-sheu.py | fix.patch ...]
#   The source tree must be the same major.minor as `uname -r`; the distro's own source
#   package for this exact kernel is best. Patches (*.patch/*.diff) are applied with -p5 by default
#   (paths like a/drivers/gpu/drm/etnaviv/etnaviv_gpu.c; set PATCH_P=1 for a/etnaviv_gpu.c);
#   *.py patchers are called with the path of the copied etnaviv_gpu.c.
set -euo pipefail

SRC="${1:?usage: $0 /path/to/kernel-source [patcher.py|fix.patch ...]}"; shift
KVER="$(uname -r)"
HDR="/lib/modules/$KVER/build"
WORK="${ALTA_WORK:-$HOME/alta-npu/build/etnaviv-$KVER-$(date +%Y%m%d-%H%M%S)}"
JOBS="${JOBS:-2}"

die() { echo "build-etnaviv-oot: ERROR: $*" >&2; exit 1; }
[ -f "$HDR/Makefile" ] || die "kernel headers for $KVER not found at $HDR (install the headers package for this kernel)"
[ -d "$SRC/drivers/gpu/drm/etnaviv" ] || die "$SRC/drivers/gpu/drm/etnaviv not found"
[ "$JOBS" -le 2 ] || die "JOBS=$JOBS refused (max 2 on this 4 GB board)"

# 0. Built-in etnaviv cannot be replaced by a module.
CFG="/boot/config-$KVER"; [ -r "$CFG" ] || CFG="$HDR/.config"
if grep -q '^CONFIG_DRM_ETNAVIV=y' "$CFG" 2>/dev/null; then
  die "etnaviv is built into this kernel (CONFIG_DRM_ETNAVIV=y); an out-of-tree module cannot replace it - this needs a kernel rebuild"
fi

# 1. Use the compiler the kernel was built with. A mismatch shows up as e.g.
#    "gcc: error: unrecognized command-line option '-fmin-function-alignment=8'" (GCC 13 vs a GCC 14 kernel).
#    kbuild's "compiler differs" warning is harmless when only the binary NAME differs (gcc vs gcc-14).
command -v modinfo >/dev/null || die "modinfo not found (package kmod)"
CCTEXT="$(sed -n 's/^CONFIG_CC_VERSION_TEXT="\(.*\)"$/\1/p' "$HDR/.config" 2>/dev/null || true)"
cands=()
if [ -n "$CCTEXT" ]; then cands+=("$(printf '%s' "$CCTEXT" | awk '{print $1}')"); else CCTEXT="$(cat /proc/version)"; fi
GCCMAJ="$(printf '%s\n' "$CCTEXT" | sed -nE 's/.*gcc[^,]*\) ([0-9]+)\.[0-9]+(\.[0-9]+)?.*/\1/p' | head -1)"
[ -n "$GCCMAJ" ] || die "could not find a GCC version in: $CCTEXT (clang-built kernel? then build with LLVM=1 instead)"
cands+=("gcc-$GCCMAJ" "aarch64-linux-gnu-gcc-$GCCMAJ" gcc aarch64-linux-gnu-gcc)
CC=""
for cand in "${cands[@]}"; do
  command -v "$cand" >/dev/null 2>&1 || continue
  if [ "$("$cand" -dumpversion 2>/dev/null | cut -d. -f1)" = "$GCCMAJ" ]; then CC="$cand"; break; fi
done
[ -n "$CC" ] || die "kernel was built with GCC $GCCMAJ but no gcc-$GCCMAJ is installed (root step: apt install gcc-$GCCMAJ)"
echo "kernel compiler : $CCTEXT"
echo "using           : CC=$CC ($("$CC" -dumpfullversion 2>/dev/null))"

# 2. Source tree must match the running kernel's major.minor.
# (parse the top-level Makefile: `make kernelversion` fails in a sparse checkout)
SRCVER="$(awk -F' *= *' '/^VERSION *=/{v=$2} /^PATCHLEVEL *=/{p=$2} /^SUBLEVEL *=/{s=$2} END{if (v != "") print v "." p "." s}' "$SRC/Makefile" 2>/dev/null || true)"
KMM="$(printf '%s' "$KVER" | grep -oE '^[0-9]+\.[0-9]+' || true)"
SMM="$(printf '%s' "$SRCVER" | grep -oE '^[0-9]+\.[0-9]+' || true)"
echo "running kernel  : $KVER ; source tree: ${SRCVER:-unknown}"
if [ "$KMM" != "$SMM" ] && [ "${ALLOW_VERSION_MISMATCH:-0}" != 1 ]; then
  die "source $SRCVER does not match running kernel $KVER (major.minor); use the matching source"
fi

# 3. Copy, record, patch.
mkdir -p "$WORK/orig"
cp -a "$SRC/drivers/gpu/drm/etnaviv/." "$WORK/"
cp -a "$SRC/drivers/gpu/drm/etnaviv/." "$WORK/orig/"
( cd "$WORK" && sha256sum ./*.c ./*.h > SHA256SUMS.orig )
: > "$WORK/PATCHES.sha256"
for p in "$@"; do
  case "$p" in
    *.py)          python3 "$p" "$WORK/etnaviv_gpu.c" ;;
    *.patch|*.diff)
      patch -d "$WORK" -p"${PATCH_P:-5}" --forward --dry-run < "$p" >/dev/null || die "patch does not apply cleanly: $p (strip level PATCH_P=${PATCH_P:-5})"
      patch -d "$WORK" -p"${PATCH_P:-5}" --forward < "$p" ;;
    *) die "unknown patch type: $p" ;;
  esac
  sha256sum "$p" >> "$WORK/PATCHES.sha256"
done
( cd "$WORK" && sha256sum ./*.c ./*.h > SHA256SUMS.patched )
( cd "$WORK" && diff -u orig/etnaviv_gpu.c etnaviv_gpu.c > ALTA-etnaviv_gpu.diff ) || true
if cmp -s "$WORK/SHA256SUMS.orig" "$WORK/SHA256SUMS.patched"; then
  echo "WARNING: no source file changed - you are building a STOCK etnaviv module"
fi

# 4. Build (serial-ish, -j2).
make -C "$HDR" M="$WORK" CC="$CC" CONFIG_DRM_ETNAVIV=m -j"$JOBS" modules 2>&1 | tee "$WORK/build.log"
KO="$WORK/etnaviv.ko"
[ -f "$KO" ] || die "build produced no etnaviv.ko (see $WORK/build.log)"

# 5. Sanity: vermagic must match a module that already loads on this kernel.
VM_NEW="$(modinfo -F vermagic "$KO")"
VM_REF="$(modinfo -F vermagic panfrost 2>/dev/null || modinfo -F vermagic meson_drm 2>/dev/null || true)"
echo "vermagic (new)  : $VM_NEW"
echo "vermagic (ref)  : ${VM_REF:-(no loadable reference module found)}"
if [ -n "$VM_REF" ] && [ "$VM_NEW" != "$VM_REF" ]; then die "vermagic mismatch - this module would be rejected"; fi
grep -q 'alta: SH_EU' "$WORK/etnaviv_gpu.c" && echo "patch marker    : present in etnaviv_gpu.c (will appear in the kernel log on load)"

{
  echo "date: $(date -Is)"
  echo "kernel: $KVER"
  echo "headers: $(readlink -f "$HDR")"
  echo "kernel compiler: $CCTEXT"
  echo "CC: $CC $("$CC" -dumpfullversion 2>/dev/null)"
  echo "source: $(readlink -f "$SRC") (kernelversion ${SRCVER:-unknown})"
  echo "patches (sha256):"; cat "$WORK/PATCHES.sha256"
  echo "module sha256: $(sha256sum "$KO" | cut -d' ' -f1)"
  echo "module srcversion: $(modinfo -F srcversion "$KO" 2>/dev/null)"
  echo "vermagic: $VM_NEW"
} > "$WORK/BUILDINFO.txt"

cat <<EOF

Built   : $KO
Record  : $WORK/BUILDINFO.txt   (diff: $WORK/ALTA-etnaviv_gpu.diff)

Install - Jordan runs these himself (or as a reviewed root-steps/ script):
  sudo install -D -m 0644 "$KO" "/lib/modules/$KVER/updates/alta/etnaviv.ko"
  sudo depmod -a "$KVER"
  modinfo -n etnaviv     # must print /lib/modules/$KVER/updates/alta/etnaviv.ko
  # only if audit section 6 showed etnaviv inside the initramfs:
  # sudo update-initramfs -u -k "$KVER"
Then REBOOT (never rmmod/modprobe -r a live etnaviv) and run the post-install checks (handoff section 8.5).
EOF
