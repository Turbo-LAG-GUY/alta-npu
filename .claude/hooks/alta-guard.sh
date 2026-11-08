#!/usr/bin/env bash
# .claude/hooks/alta-guard.sh - PreToolUse hook (matcher "Bash") for Claude Code on the Alta.
# Exit 0 = allow. Exit 2 = block; stderr is shown to Claude. Fails CLOSED if input can't be parsed.
# This is a safety net for the rules in ALTA_NPU_HANDOFF.md section 1, not a sandbox.
input="$(cat)"
cmd="$(printf '%s' "$input" | python3 -c 'import json,sys
print(json.load(sys.stdin).get("tool_input",{}).get("command",""))' 2>/dev/null)" || {
  echo "alta-guard: could not parse hook input (is python3 installed?) - blocking to be safe" >&2; exit 2; }
[ -n "$cmd" ] || exit 0

c="$(printf '%s' "$cmd" | tr '\n' ';')"
block() {
  printf 'alta-guard BLOCKED: %s\n  command: %s\n  Rules: ALTA_NPU_HANDOFF.md section 1. If it is really needed, write it to root-steps/ for Jordan to run.\n' "$1" "$cmd" >&2
  exit 2
}
# A word in command position: line start, after ; & | ( ` $( or a quote, optionally after sudo/env and VAR=val.
pre='(^|[;&|(`"]|\$\()[[:space:]]*(sudo[[:space:]]+(-[[:alnum:]-]+[[:space:]]+)*)?(env[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]*[[:space:]]+)*'
post='([[:space:]]|$|[;&|)])'
is_cmd() { local re="${pre}($1)${post}"; [[ $c =~ $re ]]; }

# R2 - no manual module load/unload, no replayed udev events
is_cmd 'rmmod|insmod' && block "manual module load/unload (rmmod/insmod)"
if is_cmd 'modprobe' && [[ $c =~ modprobe[^\;\&\|]*[[:space:]](-[[:alpha:]]*r[[:alpha:]]*|--remove[[:alnum:]-]*)([[:space:]]|$) ]]; then
  block "modprobe -r (module removal)"
fi
[[ $c =~ udevadm[[:space:]]+trigger ]] && block "udevadm trigger (re-plugs DRM devices under a running compositor)"

# R6 - power state, targets, boot chain, raw devices, system paths
is_cmd 'reboot|poweroff|halt|shutdown|kexec' && block "reboot/shutdown"
[[ $c =~ systemctl[[:space:]]+([^\;\&\|]*[[:space:]])?(reboot|poweroff|halt|kexec|soft-reboot|suspend|hibernate|isolate|set-default|rescue|emergency)([[:space:]]|$) ]] \
  && block "systemctl power or boot-target change"
is_cmd 'update-initramfs|mkinitramfs|dracut|update-grub|grub-install|flash-kernel|u-boot-update|depmod' \
  && block "initramfs / bootloader / depmod change (root step for Jordan)"
[[ $c =~ (^|[[:space:]])of=/dev/ ]] && block "dd to a raw device"
if { is_cmd 'cp|mv|install|ln|rm|tee|rsync|truncate' || { is_cmd 'sed' && [[ $c =~ [[:space:]]-i ]]; }; } && \
   [[ $c =~ [[:space:]=](/boot|/lib/modules|/usr/lib/modules|/etc/udev|/etc/modprobe\.d|/etc/environment|/etc/systemd|/etc/default)(/|[[:space:]]|$) ]]; then
  block "writing a system path (/boot, /lib/modules, /etc/udev, /etc/modprobe.d, /etc/environment, /etc/systemd, /etc/default)"
fi
[[ $c =~ PAN_I_WANT_A_BROKEN_VULKAN_DRIVER ]] && \
  [[ $c =~ (/etc/environment|environment\.d|\.profile|\.bashrc|\.zshrc|\.pam_environment|set-environment|export[[:space:]]+PAN_I_WANT) ]] \
  && block "making the panvk override global"

# R3 - build parallelism on a 4 GB board
if is_cmd 'make|ninja|cmake|meson'; then
  maxj="$(printf '%s' "$c" | grep -oE '(-j|--jobs[= ]|--parallel[= ])[[:space:]]*[0-9]+' | grep -oE '[0-9]+$' | sort -n | tail -1)"
  if [ -n "$maxj" ] && [ "$maxj" -gt 2 ]; then
    block "build with $maxj parallel jobs (max 2 on this board)"
  elif [[ $c =~ (-j|--jobs|--parallel)[=[:space:]]*[\$\`] ]]; then
    block "computed job count such as -j\$(nproc) (use -j2)"
  elif bare='[[:space:]]-j([[:space:]]*$|[[:space:]]*[;&|]|[[:space:]]+[^0-9[:space:]])'; [[ $c =~ $bare ]]; then
    block "make -j without a number (unlimited jobs)"
  elif [ -z "$maxj" ] && { is_cmd 'ninja' || [[ $c =~ meson[[:space:]]+compile ]] || [[ $c =~ cmake[[:space:]]+--build ]]; }; then
    block "ninja / meson compile / cmake --build without an explicit -j2 (ninja defaults to all cores)"
  fi
fi

# R4 - NPU inference only through scripts/npu-run
if ! [[ $c =~ npu-run ]] && [[ $c =~ (python[0-9.]*|benchmark_model|label_image) ]] && \
   [[ $c =~ (teflon|external_delegate|TEFLON_DEBUG|ETNA_MESA_DEBUG) ]]; then
  block "NPU inference outside scripts/npu-run (no lock, deadline or kernel-log check)"
fi

# R7 - no whole-filesystem scans
[[ $c =~ (^|[\;\&\|[:space:]])find[[:space:]]+/([[:space:]]|$) ]] && block "find over / (scope it to a directory)"
[[ $c =~ (grep[[:space:]]+-[[:alpha:]]*[rR]|rg[[:space:]])[^\;\&\|]*[[:space:]](/|~|\$HOME|/home/[^/[:space:]]+)/?([[:space:]]|$) ]] \
  && block "recursive search over / or the whole home directory"
exit 0
