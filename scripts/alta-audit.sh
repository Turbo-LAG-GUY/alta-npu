#!/usr/bin/env bash
# alta-audit.sh - READ-ONLY inventory of the Alta (AML-A311D-CC) NPU / GPU / memory state.
# Changes nothing. Root-only reads are attempted with `sudo -n` and skipped if a password is needed.
# Usage: bash scripts/alta-audit.sh [OUTFILE]
set -u
OUT="${1:-$HOME/alta-npu/audit/audit-$(date +%Y%m%d-%H%M%S).txt}"
mkdir -p "$(dirname "$OUT")"
exec > >(tee "$OUT") 2>&1

KVER="$(uname -r)"
HDR="/lib/modules/$KVER/build"
sec()  { printf '\n===== %s =====\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
asroot() {
  if [ "$(id -u)" -eq 0 ]; then "$@"
  elif sudo -n true 2>/dev/null; then sudo -n "$@"
  else echo "(skipped - needs root: $*)"; return 1; fi
}
klog() {  # kernel log, with or without journal group membership
  if [ -n "$(journalctl -k -n 1 -q --no-pager 2>/dev/null)" ]; then
    journalctl -k --no-pager -q "$@" 2>/dev/null
  else
    asroot journalctl -k --no-pager -q "$@"
  fi
}

sec "1. System"
date -Is
uname -a
grep -E '^(PRETTY_NAME|VERSION_ID|ID)=' /etc/os-release 2>/dev/null
printf 'DT model: %s\n' "$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)"
printf 'DT compatible: %s\n' "$(tr '\0' ' ' < /proc/device-tree/compatible 2>/dev/null)"
printf 'CPUs: %s\n' "$(nproc)"
uptime

sec "2. Kernel build compiler (out-of-tree modules must use the same one)"
cat /proc/version
if [ -e "$HDR/Makefile" ]; then
  echo "headers: $HDR -> $(readlink -f "$HDR")"
  grep -E '^CONFIG_CC_VERSION_TEXT=' "$HDR/.config" 2>/dev/null || echo "no .config inside the headers"
else
  echo "headers: MISSING ($HDR) - the headers package for $KVER is not installed"
fi
for c in gcc gcc-12 gcc-13 gcc-14 gcc-15 aarch64-linux-gnu-gcc-14; do
  have "$c" && printf '  %-26s %s\n' "$c" "$("$c" -dumpfullversion 2>/dev/null)"
done
echo "installed kernels: $(ls /lib/modules 2>/dev/null | tr '\n' ' ')"
echo "--- kernel package history (apt; did a kernel upgrade happen after the fix last worked?)"
find /var/log/apt -maxdepth 1 -name 'history.log*' -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2- \
  | while IFS= read -r f; do zcat -f "$f"; done \
  | awk '/^Start-Date:/ {d=$0} /^(Install|Upgrade|Remove):.*linux-(image|dtb)/ {print "  " d; print "    " substr($0, 1, 240)}' \
  | tail -24

sec "3. Kernel config (etnaviv / panfrost / signing)"
CFG=""
for f in "/boot/config-$KVER" "$HDR/.config"; do [ -r "$f" ] && { CFG="$f"; break; }; done
CFGRE='^(# )?CONFIG_(DRM_ETNAVIV|DRM_ETNAVIV_THERMAL|DRM_PANFROST|DRM_SCHED|DEV_COREDUMP|MODULE_SIG|MODULE_SIG_FORCE|MODULE_SRCVERSION_ALL|ZRAM|PSI)( |=)'
if [ -n "$CFG" ]; then
  echo "config: $CFG"; grep -E "$CFGRE" "$CFG"
elif [ -r /proc/config.gz ]; then
  zcat /proc/config.gz | grep -E "$CFGRE"
else
  echo "no kernel config found"
fi

sec "4. Device tree: NPU node"
n=0
while IFS= read -r d; do
  n=$((n+1))
  st="$(tr -d '\0' < "$d/status" 2>/dev/null)"; [ -n "$st" ] || st="(no status property = enabled)"
  printf '%s\n  compatible: %s\n  status: %s\n' "$d" "$(tr '\0' ' ' < "$d/compatible" 2>/dev/null)" "$st"
  [ -r "$d/assigned-clock-rates" ] && printf '  assigned-clock-rates: %s\n' \
    "$(od -An -tu4 --endian=big "$d/assigned-clock-rates" 2>/dev/null | xargs)"
  [ -e "$d/power-domains" ] && echo "  power-domains: present"
done < <(find /proc/device-tree/ -maxdepth 4 -type d -name 'npu@*' 2>/dev/null)
[ "$n" -gt 0 ] || echo "no npu@* node in the live device tree -> etnaviv cannot bind to the NPU"

sec "5. Kernel modules"
{ have lsmod && lsmod | grep -E '^(etnaviv|panfrost|gpu_sched|meson_drm|drm_shmem_helper) '; } || echo "(none of etnaviv/panfrost/gpu_sched/meson_drm loaded)"
for m in etnaviv panfrost; do
  echo "--- $m"
  echo "  modprobe would load: $(modinfo -n "$m" 2>/dev/null || echo 'NOT FOUND')"
  for k in vermagic intree srcversion signer; do
    v="$(modinfo -F "$k" "$m" 2>/dev/null | head -1)"; printf '  %-11s %s\n' "$k:" "${v:-(none)}"
  done
  if [ -d "/sys/module/$m" ]; then
    printf '  loaded: yes  taint=%s  srcversion=%s\n' \
      "$(cat "/sys/module/$m/taint" 2>/dev/null)" "$(cat "/sys/module/$m/srcversion" 2>/dev/null)"
  else
    echo "  loaded: no"
  fi
done
echo "--- every etnaviv.ko* for EVERY installed kernel (a patched module left behind for an older kernel = fix silently lost after an upgrade)"
find /lib/modules -name 'etnaviv.ko*' -printf '  %p  %s bytes  %TY-%Tm-%Td %TH:%TM\n' 2>/dev/null
find "/lib/modules/$KVER" -name 'etnaviv.ko*' -exec sha256sum {} + 2>/dev/null | sed 's/^/  sha256 /'
grep -h -E '^(search|override)' /etc/depmod.d/*.conf /usr/lib/depmod.d/*.conf /lib/depmod.d/*.conf 2>/dev/null | sed 's/^/  depmod.d: /'

sec "6. Is etnaviv inside the initramfs? (initramfs udev does not apply our rules)"
INITRD=""
for f in "/boot/initrd.img-$KVER" "/boot/initramfs-$KVER.img" /boot/initrd.img; do [ -e "$f" ] && { INITRD="$f"; break; }; done
if [ -z "$INITRD" ]; then
  echo "no initramfs image found in /boot"; ls -la /boot 2>/dev/null
elif ! have lsinitramfs; then
  echo "found $INITRD but lsinitramfs is not installed"
else
  lst="$(lsinitramfs "$INITRD" 2>/dev/null)"
  [ -n "$lst" ] || lst="$(asroot lsinitramfs "$INITRD" 2>/dev/null)"
  if [ -z "$lst" ]; then echo "could not list $INITRD (needs root?)"
  else printf '%s\n' "$lst" | grep -E 'etnaviv|modprobe\.d/|rules\.d/.*(mutter|etnaviv|npu)' || echo "etnaviv not in $INITRD"; fi
fi

sec "7. modprobe / modules-load policy"
grep -rHn -E 'etnaviv|panfrost' /etc/modprobe.d /usr/lib/modprobe.d /lib/modprobe.d /etc/modules \
  /etc/modules-load.d /usr/lib/modules-load.d 2>/dev/null || echo "(nothing mentions etnaviv/panfrost)"
systemctl list-unit-files 2>/dev/null | grep -iE 'etnaviv|npu' || true

sec "8. udev rules (etnaviv / mutter / npu)"
grep -rHn -E 'etnaviv|mutter-device-ignore|alta-npu' /etc/udev/rules.d /usr/lib/udev/rules.d /lib/udev/rules.d 2>/dev/null \
  || echo "(none)"

sec "9. DRM nodes, drivers, udev tags"
ls -l /dev/dri/ /dev/dri/by-path/ 2>/dev/null
for node in /dev/dri/card* /dev/dri/renderD*; do
  [ -e "$node" ] || continue
  b="$(basename "$node")"
  drv="$(basename "$(readlink -f "/sys/class/drm/$b/device/driver" 2>/dev/null)")"
  dev="$(basename "$(readlink -f "/sys/class/drm/$b/device" 2>/dev/null)")"
  echo "$node  device=$dev  driver=$drv"
  udevadm info --query=property --name="$node" 2>/dev/null | grep -E '^(ID_PATH|TAGS|CURRENT_TAGS)=' | sed 's/^/    /'
done
echo "my groups: $(id -nG)"

sec "10. Processes holding DRM nodes"
if ! have fuser; then
  echo "fuser not installed (package psmisc)"
elif [ "$(id -u)" -eq 0 ] || sudo -n true 2>/dev/null; then
  asroot fuser -v /dev/dri/* 2>&1 | head -40
else
  fuser -v /dev/dri/* 2>&1 | head -40; echo "(non-root view: only your own processes)"
fi

sec "11. Userspace: Mesa, Teflon, Python runtime, desktop"
have dpkg-query && dpkg-query -W -f='  ${Package} ${Version}\n' mesa-libgallium libgl1-mesa-dri libegl-mesa0 \
  mesa-vulkan-drivers mesa-teflon-delegate libgtk-4-1 gnome-shell mutter-common 2>/dev/null
for f in /usr/lib/teflon/libteflon.so /usr/lib/aarch64-linux-gnu/libteflon.so /usr/local/lib/libteflon.so \
         /usr/local/lib/aarch64-linux-gnu/libteflon.so; do
  [ -e "$f" ] && echo "  teflon: $f"
done
find "$HOME" -maxdepth 7 -name 'libteflon.so' -not -path '*/.cache/*' 2>/dev/null | sed 's/^/  teflon (home): /'
python3 -c 'import sys; print("  python", sys.version.split()[0])' 2>/dev/null
for py in python3 "$HOME/alta-npu/venv/bin/python"; do
  for mod in tflite_runtime ai_edge_litert; do
    [ -x "$(command -v "$py")" ] && "$py" -c "import $mod" 2>/dev/null && echo "  $py: $mod importable"
  done
done
printf '  display-manager: %s   default target: %s\n' \
  "$(systemctl is-active display-manager 2>/dev/null)" "$(systemctl get-default 2>/dev/null)"
loginctl list-sessions --no-legend 2>/dev/null | sed 's/^/  session: /'

sec "12. Vulkan/GL override variables (the panvk override must NOT be global)"
grep -rHn -E 'PAN_I_WANT_A_BROKEN_VULKAN_DRIVER|GSK_RENDERER|MESA_LOADER_DRIVER_OVERRIDE|VK_ICD_FILENAMES|VK_DRIVER_FILES|ETNA_MESA_DEBUG|TEFLON_DEBUG' \
  /etc/environment /etc/environment.d /etc/profile /etc/profile.d /etc/X11/Xsession.d /etc/systemd/user.conf.d \
  "$HOME/.profile" "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.zprofile" "$HOME/.pam_environment" \
  "$HOME/.config/environment.d" 2>/dev/null || echo "  (not set in any global or login file - good)"
env | grep -E '^(PAN_I_WANT|GSK_RENDERER|MESA_|VK_|ETNA_|TEFLON_)' | sed 's/^/  current env: /'
systemctl --user show-environment 2>/dev/null | grep -E '^(PAN_I_WANT|GSK_RENDERER|VK_)' | sed 's/^/  user manager env: /'

sec "13. Memory, swap, OOM protection"
free -h
swapon --show 2>/dev/null || cat /proc/swaps
echo "  PSI memory: $(tr '\n' ' ' < /proc/pressure/memory 2>/dev/null)"
for s in earlyoom systemd-oomd zramswap zram-config nohang; do
  printf '  %-14s %s\n' "$s" "$(systemctl is-active "$s" 2>/dev/null)"
done
ls -d /sys/block/zram* 2>/dev/null | sed 's/^/  /' || echo "  no zram devices"
sysctl vm.swappiness vm.min_free_kbytes vm.overcommit_memory 2>/dev/null | sed 's/^/  /'

sec "14. Thermal, CPU clocks"
for z in /sys/class/thermal/thermal_zone*; do
  [ -r "$z/temp" ] && printf '  %s %-16s %s C\n' "$(basename "$z")" "$(cat "$z/type")" "$(( $(cat "$z/temp") / 1000 ))"
done
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  [ -r "$p/scaling_cur_freq" ] && printf '  %s cur=%s max=%s gov=%s\n' "$(basename "$p")" \
    "$(cat "$p/scaling_cur_freq")" "$(cat "$p/scaling_max_freq")" "$(cat "$p/scaling_governor")"
done

sec "15. Journal persistence and recent kernel trouble"
if [ -d /var/log/journal ]; then echo "  persistent journal: yes"; else echo "  persistent journal: NO - logs from a boot that locked up are lost"; fi
journalctl --list-boots --no-pager 2>/dev/null | tail -6 | sed 's/^/  /'
PAT='etnaviv|panfrost|ff100000\.npu|out of memory|oom-kill|oom_reaper|hung_task|soft lockup|hard lockup|rcu.*stall|mmu fault|recover hung|Internal error|Oops|BUG:|thermal'
echo "--- previous boot (-1): matching kernel lines (last 60)"
klog -b -1 | grep -iE "$PAT" | tail -60
echo "--- previous boot (-1): its final 25 kernel lines (what happened right before the lockup / reboot)"
klog -b -1 | tail -25
echo "--- this boot: matching kernel lines (last 60)"
klog -b 0 | grep -iE "$PAT" | tail -60

sec "16. Artifacts from the earlier NPU work"
for d in "$HOME" /usr/src /opt /usr/local/src; do
  [ -d "$d" ] || continue
  find "$d" -maxdepth 5 \( -name '*.patch' -o -name '*.diff' -o -name 'etnaviv*.ko*' -o -iname '*teflon*' \
       -o -iname '*etnaviv*' -o \( -iname '*npu*' -not -iname '*input*' \) \) -not -path '*/.cache/*' -not -path '*/node_modules/*' \
       -not -path '*/.git/*' 2>/dev/null | head -80 | sed 's/^/  /'
done
echo "--- patch/diff files that touch clock gating, 0x7120 or mutter tags"
find "$HOME" /usr/src /usr/local/src -maxdepth 6 \( -name '*.patch' -o -name '*.diff' \) -size -2M -print0 2>/dev/null \
  | xargs -0 -r grep -l -E '0x7120|CLOCK_GATING|enable_mlcg|mutter-device-ignore' 2>/dev/null | head -20 | sed 's/^/  /'
echo "--- shell history hints"
grep -h -E 'etnaviv|teflon|mutter-device|0x7120|PAN_I_WANT|modprobe|insmod|M=|CC=gcc-1[0-9]' \
  "$HOME/.bash_history" "$HOME/.zsh_history" 2>/dev/null | tail -40 | sed 's/^/  /'

sec "17. Claude Code"
if have claude; then echo "  claude: $(claude --version 2>/dev/null)"; else echo "  claude not on PATH"; fi
for f in "$HOME/.claude/settings.json" "$PWD/.claude/settings.json" "$PWD/.claude/settings.local.json" "$PWD/CLAUDE.md"; do
  [ -f "$f" ] && echo "  present: $f"
done
cgp="$(cut -d: -f3 /proc/self/cgroup 2>/dev/null | head -1)"
echo "  my cgroup: $cgp"
[ -r "/sys/fs/cgroup$cgp/memory.max" ] && echo "  cgroup memory.max=$(cat "/sys/fs/cgroup$cgp/memory.max") memory.high=$(cat "/sys/fs/cgroup$cgp/memory.high" 2>/dev/null)"
echo
echo "Audit written to: $OUT"
