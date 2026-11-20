#!/usr/bin/env bash
# run_gpu.sh (2026-11-19, G1): run vk_dot on the Mali under the NPU guard, like npu-upscale/gpu-speed/run.sh.
# Usage: run_gpu.sh system|patched check|bench ARGS...   (patched = mesa-gpu-dot/build's panvk)
set -u
here=$(cd "$(dirname "$0")" && pwd)
G=/home/jordan/Documents/AI_Projects/npu-upscale/gpu-speed/build/no-npu-guard.so
case "$1" in
   system) ICD=/usr/share/vulkan/icd.d/panfrost_icd.json ;;
   patched) ICD=$(ls /home/jordan/Documents/AI_Projects/mesa-gpu-dot/build/src/panfrost/vulkan/panfrost_devenv_icd.*.json | head -n 1) ;;
   *) echo "system|patched"; exit 1 ;;
esac
shift
kcount() { journalctl -k -b --no-pager 2>/dev/null | grep -ciE 'panfrost.*(fault|timeout|reset|error)|gpu sched timeout'; }
before=$(kcount)
env LD_PRELOAD="$G" PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1 VK_DRIVER_FILES="$ICD" VK_LOADER_LAYERS_DISABLE='~all~' \
   NODEVICE_SELECT=1 "$here/vk_dot" "$@"
rc=$?
sleep 1; after=$(kcount)
echo "[exit $rc, new kernel GPU error lines: $((after - before))]"
[ "$after" -gt "$before" ] && journalctl -k -b --no-pager | grep -iE 'panfrost|gpu sched' | tail -n 10
exit $rc
