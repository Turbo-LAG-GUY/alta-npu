# Alta (AML-A311D-CC) NPU — findings summary

Compiled 2026-11-07. This is the short version. The full handoff (`ALTA_NPU_HANDOFF.md`: operating rules, procedures and an embedded, tested script kit) was delivered in chat and is meant to live at `~/alta-npu/` on the board. It was too large to store here as a single doc.

## Board identity
- Amlogic A311D (4× A73 @ 2.2 GHz + 2× A53 @ 1.8 GHz), Mali-G52 MP4, 4 GB LPDDR4, 5 V / 3 A over USB-C.
- NPU: VeriSilicon VIPNano-QI.7120, driven by etnaviv (kernel) + Mesa Teflon (TFLite delegate). The DT node is `npu@ff100000` (`vivante,gc`); etnaviv reports it as model GC8000, revision 7120.
- etnaviv creates a virtual `etnaviv` platform device, so its DRM nodes are `/dev/dri/by-path/platform-etnaviv-{card,render}` (`ID_PATH=platform-etnaviv`).

## Earlier fixes and their status (checked 2026-11-07)
| Fix | Status |
|---|---|
| etnaviv (not galcore) drives the NPU | Still the path; Mesa lists the Alta as Teflon test hardware |
| etnaviv rebuilt out of tree with GCC 14 (must match the kernel's compiler) | GCC 13 against a GCC 14 kernel fails on `-fmin-function-alignment=8` |
| Clock-gating patch for rev 0x7120 | Upstream's SH_EU workaround (6.11+) covers 0x7200, 0x8002 and 0x6304 but not 0x7120 (still true in Linux master), so the patch is still needed |
| udev rule so GNOME ignores the NPU | `TAG+="mutter-device-ignore"` on the `platform-etnaviv` nodes; Mesa 25.1's kmsro change also helps |
| panvk Vulkan on Mali-G52 via env var | Bifrost v7 still needs `PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1`; keep it per command, never global |

## Why lockups can come back
1. A kernel upgrade drops the patched module, which lives under the old kernel's `updates/`. On top of that, the Alta DTS enables the NPU by default from the 6.17 cycle on.
2. GNOME regressions: the rule went missing, or etnaviv loads from the initramfs.
3. Memory: Claude Code's documented minimum is 4 GB, which is all the board has; GNOME plus parallel builds thrash.
4. A global panvk override, combined with GTK ≥ 4.16 defaulting to Vulkan, puts terminals on a non-conformant driver (hypothesis).
5. A hung NPU reset can wedge the SoC. Stop at the first error and reboot.

## Operating approach (from the handoff)
- Run headless, start Claude Code over SSH + tmux inside a memory-capped systemd scope, build at `-j2` at most, and keep zram plus earlyoom/oomd on.
- Run every NPU job through a wrapper that takes a lock, enforces a deadline, checks the kernel log and sets a STOP latch on any error.
- The patched module prints a marker line (`alta: SH_EU clock gating disabled ...`), so it is provable that it's the one loaded.
- Root changes go into reviewed `root-steps/` scripts. A PreToolUse hook blocks rmmod, `modprobe -r`, `udevadm trigger`, reboot, `-j>2` and the like.

## Still to settle on the board
- Which kernel and image were running when the fix last worked, and is there a kernel upgrade in the apt history since then?
- Where is the original 0x7120 patch, and did it change only SH_EU gating?
- Was the panvk override made global?
