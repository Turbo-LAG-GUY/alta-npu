# Alta (AML-A311D-CC) NPU — findings, fixes and operating procedure

*Handoff for a local Claude Code session running on the board. Compiled 2026-11-07 for Jordan. Everything marked [UPSTREAM] was checked against the sources in Appendix C on that date.*

**Purpose:** make NPU work on the Libre Computer Alta repeatable and controlled, and stop Claude Code sessions from locking up the board. This file is both the reference and the rulebook for the session, and it carries every script it needs (Appendix A).

---

## 0. Read me first

### 0.1 How to use this file

Layout on the Alta (the project directory stays small; big source trees live in `~/src`):

```text
~/alta-npu/
  ALTA_NPU_HANDOFF.md          this file
  CLAUDE.md                    imports this file into every Claude Code session
  .claude/settings.json        env limits, permission rules, hook registration
  .claude/hooks/alta-guard.sh  blocks the dangerous commands listed in section 1
  scripts/                     audit, launcher, OOT build, patcher, npu-run, classifier, extractor
  root-steps/                  scripts that need sudo; Jordan reviews and runs them
  system/                      udev rule, modprobe snippet, systemd unit (installed by root-steps)
  templates/                   CHANGELOG.md / RESULTS.md skeletons
  audit/ build/ runs/ results/ models/ incidents/   created as you go
~/src/                         kernel / Mesa source trees (outside the project on purpose)
```

One-time bootstrap, done by Jordan in a plain SSH shell **before** the first Claude Code session, so the guard rails exist from the start. `jordan` and `alta` stand for the actual user and hostname:

```bash
# from the workstation
scp ALTA_NPU_HANDOFF.md jordan@alta:~/alta-npu/      # create ~/alta-npu first if needed
ssh jordan@alta
cd ~/alta-npu
# unpack the kit from Appendix A (this is the same code as scripts/extract-kit.py)
python3 - ALTA_NPU_HANDOFF.md . <<'PY'
import os, pathlib, re, sys
args = [a for a in sys.argv[1:] if a != "--force"]; force = "--force" in sys.argv
src = pathlib.Path(args[0]); dst = pathlib.Path(args[1] if len(args) > 1 else ".")
F = "`" * 3
pat = re.compile(r"^<!-- file: (\S+)(?: mode=([0-7]{3,4}))? -->\n" + F + r"[^\n]*\n(.*?)\n" + F + r"[ \t]*$", re.S | re.M)
n = 0
for rel, mode, body in pat.findall(src.read_text()):
    if rel.startswith("/") or ".." in pathlib.PurePosixPath(rel).parts: sys.exit(f"unsafe path {rel}")
    p = dst / rel
    if p.exists() and not force: print("skip (exists):", p); continue
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(body + "\n")
    if mode: os.chmod(p, int(mode, 8))
    n += 1; print("wrote", p)
print(n, "file(s) written")
PY
cp -n templates/CHANGELOG.md templates/RESULTS.md .
mkdir -p ~/src audit build runs results models incidents
```

Then: run the root steps Jordan decides on (section 6), reboot, and start every Claude Code session like this (section 6.3):

```bash
ssh jordan@alta
tmux new -As alta
~/alta-npu/scripts/alta-claude
```

`CLAUDE.md` imports this whole file, so every session starts with it in context. That costs roughly 25k tokens per session; if it's too heavy, replace the import line with an instruction to read sections 1, 5 and 10 at session start. Claude Code snapshots hooks at startup: after any change to `.claude/settings.json` or the hook, restart the session.

### 0.2 Provenance tags

| Tag | Meaning |
|---|---|
| **[RECORD]** | Established in our earlier work on this board. Only a summary of it survived: the original patch file, udev rule and exact commands were not carried over, so audit section 16 hunts for them on the board. Originals found there beat the reconstructions in this file. |
| **[UPSTREAM]** | Checked against upstream code or docs (Linux master, Mesa docs and release notes, mutter, Claude Code docs) on 2026-11-07. |
| **[RECON]** | Rebuilt from upstream to reproduce a [RECORD] fix. Verify on the board. |
| **[HYPOTHESIS]** | Plausible cause. Confirm or rule out with the evidence named next to it. |

### 0.3 TL;DR

We did fix the lockups before [RECORD], with three things: a **patched etnaviv module built out of tree with GCC 14**, a **clock-gating patch for NPU revision 0x7120**, and a **udev rule that keeps GNOME from enumerating the NPU as a display device**. Each of these can stop applying without any visible error:

1. **A kernel upgrade drops the patched module.** It lives under `/lib/modules/<old-version>/updates`, so the new kernel loads the stock etnaviv, which has no 0x7120 fix. On top of that, the Alta's device tree enables the NPU node by default from the Linux 6.17 cycle on, so etnaviv now binds at boot even where it used to stay idle [UPSTREAM]. → sections 8.1, 8.6
2. **The GNOME side regresses.** An OS reinstall or upgrade can drop the udev rule, or etnaviv can load from the initramfs, where the rule does not apply. → section 8.4
3. **Running Claude Code on the board's own desktop.** Claude Code's documented minimum is 4 GB of RAM [UPSTREAM], which is all the Alta has, and GNOME plus a parallel build can exhaust it [HYPOTHESIS]. GTK ≥ 4.16 renders with Vulkan by default on Wayland, so a global panvk override pushes the terminal onto a non-conformant driver [UPSTREAM + HYPOTHESIS]. → sections 4, 6.3, 11

Fix order: read-only audit → hardening (journal, memory net, headless over SSH, guard rails) → confirm or rebuild the patched etnaviv for the *running* kernel → udev rule → NPU tests, only through `npu-run`.

---

## 1. Operating rules for Claude Code on this board

These are binding for every session. `.claude/hooks/alta-guard.sh` mechanically blocks the worst violations of R2, R3, R4, R6, R7, R8 and R10. It is a safety net, not a sandbox: follow the rules even where the hook can't see the violation.

- **R1 — Audit first.** Start every session with `bash scripts/alta-audit.sh`, summarize what it found for Jordan, and agree on the next step before changing anything. Read-only commands are always fine.
- **R2 — Never touch live drivers.** No `rmmod`, `modprobe -r`, `insmod`, and no `udevadm trigger`. A changed kernel module or udev rule takes effect at the next boot, and Jordan does the reboot.
- **R3 — One heavy job at a time, at `-j2` or lower:** `make -j2`, `ninja -j2`, `meson compile -C build -j 2`, `cmake --build . -j 2`. A bare `make` is fine because `MAKEFLAGS=-j2` is set. Never run a build and an NPU job at the same time. Keep source trees in `~/src` and scope every search to a directory.
- **R4 — Every NPU inference goes through `scripts/npu-run`.** It adds the lock, the deadline and the kernel-log check. Planned repetitions are fine, as separate `npu-run` calls that stop at the first failure. Retry loops are not.
- **R5 — Stop at the first kernel error.** If `npu-run` sets `runs/STOP`, stop all NPU work, collect the evidence in section 12.2 and report it. Only Jordan clears the latch, after a reboot.
- **R6 — Jordan does all root work.** Anything that changes `/boot`, the DTB or overlays, the bootloader, the initramfs, kernel packages, `/lib/modules`, `/etc/udev`, `/etc/modprobe.d`, `/etc/environment` or `/etc/systemd` goes into `root-steps/NNN-name.sh`. Each one starts with a header saying what it does, why, how to verify it and how to revert it. Jordan reviews it and runs it with sudo. Claude Code's Bash tool generally can't answer an interactive sudo password prompt anyway.
- **R7 — No recursive scans of `/` or the whole home directory.**
- **R8 — `PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1` is set per command only,** never in `/etc/environment`, shell profiles or `environment.d`.
- **R9 — Never install a Mesa build over the system Mesa** (`ninja install`, `meson install`). Point Teflon at the `libteflon.so` inside the build tree.
- **R10 — Never reboot, power off, suspend or change the systemd default target.**
- **R11 — Keep records.** Every system change gets a row in `CHANGELOG.md`, and every NPU test gets a row in `RESULTS.md` pointing to its `runs/` directory.
- **R12 — The board beats this file.** Where the audit contradicts something written here, report the discrepancy. Don't bend the system to match the document.

---

## 2. Hardware and identity facts

| Item | Value | Tag |
|---|---|---|
| Board | Libre Computer AML-A311D-CC "Alta" (early name "Big Cottonwood"), Raspberry Pi 3B-style form factor | [UPSTREAM] |
| SoC | Amlogic A311D: 4× Cortex-A73 up to 2.2 GHz + 2× Cortex-A53 up to 1.8 GHz | [UPSTREAM] |
| GPU | Arm Mali-G52 MP4 (Bifrost v7). GL/GLES via panfrost; Vulkan via panvk, which is non-conformant on v7 | [UPSTREAM] |
| NPU | VeriSilicon Vivante VIPNano-QI.7120, marketed as 5 TOPS. Kernel driver etnaviv, userspace Mesa Teflon (TensorFlow Lite external delegate) | [UPSTREAM] |
| RAM | 4 GB LPDDR4 (a 2 GB variant also exists) | [UPSTREAM] |
| I/O | 4× USB 3.0 A, USB-C (power + USB 2.0), Gigabit Ethernet, HDMI, 40-pin header, 22-pin MIPI CSI and DSI, eMMC connector, no Wi-Fi | [UPSTREAM] |
| Power | 5 V / 3 A over USB-C; the board ships with a heatsink | [UPSTREAM] |
| Upstream DTS | `arch/arm64/boot/dts/amlogic/meson-g12b-a311d-libretech-cc.dts` (contains `&npu { status = "okay"; };` in Linux master) | [UPSTREAM] |
| NPU DT node | `npu@ff100000`, compatible `"vivante,gc"`, interrupt GIC SPI 147, clocks NNA core + AXI (`"core"`, `"bus"`), both assigned 800 MHz, reset `RESET_NNA`, power domain `PWRC_G12A_NNA_ID` (added in `meson-g12b.dtsi`). The common dtsi sets it `disabled` and the board DTS enables it | [UPSTREAM] |
| etnaviv hwdb identity | model 0x8000 (printed as GC8000), revision 0x7120, product_id 0x45080009, customer_id 0x88, nn_core_count 8 | [UPSTREAM] |
| Expected probe lines | `etnaviv-gpu ff100000.npu: model: GC8000, revision: 7120` and `etnaviv-gpu ff100000.npu: etnaviv has been instantiated on a NPU, for which the UAPI is still experimental` | [UPSTREAM] |
| DRM nodes | etnaviv registers a virtual platform device named `etnaviv`, giving `/dev/dri/by-path/platform-etnaviv-card` and `platform-etnaviv-render` (udev `ID_PATH=platform-etnaviv`) | [UPSTREAM] |
| etnaviv module parameters | `job_hang_limit` (default 0), `hw_job_limit` (default 4), `force_flop_reset` (default 0) | [UPSTREAM] |
| Job timeout | 500 ms per job, with a forward-progress check before a hang is declared | [UPSTREAM] |

---

## 3. What the earlier work established, and where each item stands now

| # | Finding or fix [RECORD] | Why it mattered | Status in November 2026 [UPSTREAM] | Section |
|---|---|---|---|---|
| F1 | The NPU is driven by **etnaviv** (open stack), not the vendor galcore driver | Lets Mesa Teflon run TFLite models | Still the path. Mesa's Teflon docs list the Alta as tested hardware | 8, 9 |
| F2 | etnaviv built **out of tree with GCC 14** | Ships a patched module for the running kernel; the kernel was built by GCC 14, so the module has to be too | GCC 13 against a GCC 14 kernel fails with `unrecognized command-line option '-fmin-function-alignment=8'` | 8.3 |
| F3 | **Clock-gating patch for revision 0x7120** | Avoids NPU hangs | Upstream's SH_EU clock-gating workaround (in since 6.11) covers 0x8000/0x7200, 0x8000/0x8002 and 0x9200/0x6304, **but not 0x7120** (checked in Linux master). The patch is still needed | 8.3, App. B |
| F4 | **udev rule so GNOME doesn't enumerate the NPU** | GNOME treated the NPU's DRM device as a GPU, leaving a failed or frozen session | Still relevant with Mesa older than 25.1, or whenever the tag goes missing. mutter's tag for this is `mutter-device-ignore` | 8.4 |
| F5 | **Mali-G52 Vulkan via panvk**, with environment-variable workarounds | Vulkan experiments | On Bifrost v7, panvk still only appears with `PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1`. Since Mesa 25.0 it is exposed by default only on v10 GPUs | 11 |

Upstream changes since then that bear on the lockups [UPSTREAM]:

- **Linux 6.17 enables the NPU on the Alta by default.** Tomeu Vizoso's patch "Enable the npu node for Alta and VIM3" went into the Amlogic `v6.17/arm64-dt` branch. On older kernels the NPU appeared only through a DT edit or overlay.
- **Mesa 25.1 made kmsro look for a graphics-capable screen as the renderonly device,** and surfaceless EGL now only picks drivers that expose graphics. Both reduce the risk of the desktop picking the NPU as its GPU.
- **GTK 4.16 made Vulkan the default GSK renderer on Wayland.**
- **etnaviv's reset path still writes `VIVS_PM_POWER_CONTROLS = 0`** ("disable clock gating") on every reset attempt in Linux master. A June 2025 proposal by Lucas Stach to stop doing this is not merged. His reasoning: toggling clock gating during the reset sequence made resets less reliable and could leave parts of the GPU in an indeterminate state. That matters if hang recovery fails (section 12.3).

---

## 4. Why the board "locks up": failure modes, evidence, prevention

**L1 — The display stack grabs the NPU** [RECORD F4 + UPSTREAM]. When etnaviv binds, it creates a DRM card and a render node with no outputs. Before Mesa 25.1, Mesa's renderonly matching would accept any render node it could open a screen on, so the meson display controller could be paired with the NPU's etnaviv node instead of panfrost, and GNOME ended up "rendering" on a device that can't render. The same SoC on the Khadas VIM3 showed exactly this: a GNOME Wayland session failed with the NPU enabled, and `mutter-device-ignore` matched on `DEVNAME` alone did not fix it. The working recipe there was ignore tags matched by `ID_PATH`, plus reloading etnaviv after boot so udev applied the tags, plus starting gdm only after that. Hot-loading or unloading etnaviv under a running GNOME session also generates DRM hotplug events that mutter has to handle [HYPOTHESIS: this is one way a session could freeze].
*Evidence:* screen frozen but SSH responsive; `journalctl -b -o cat _COMM=gnome-shell` mentions the etnaviv card or render node; `udevadm info` shows no `mutter-device-ignore` tag on the `platform-etnaviv` nodes.
*Prevention:* the udev rule (A.10, installed by root step 040); keep etnaviv out of the initramfs; late loading if needed (section 8.4, root step 050); Mesa ≥ 25.1; or run headless.

**L2 — An NPU hang, then a reset that doesn't take, then a wedged SoC** [RECORD F3 + UPSTREAM]. etnaviv's scheduler flags a job after 500 ms without forward progress and resets the core, logging `recover hung GPU!`. The reset sequence toggles clock gating, which etnaviv's maintainer has argued makes resets less reliable (that proposal is unmerged; section 3). A failed reset, or a stall at the bus level, can freeze the whole SoC: no SSH, no ping.
*Evidence:* the previous boot's kernel log ends with etnaviv lines (`recover hung GPU`, `GPU failed to reset`, `MMU fault`, `GPU not yet idle`).
*Prevention:* the patched module with the 0x7120 fix, proven by its kernel-log marker; NPU jobs serialized under a deadline through `npu-run`; stop at the first error; reboot after any hang rather than carrying on with a "recovered" core.

**L3 — Memory exhaustion** [UPSTREAM + HYPOTHESIS]. Claude Code's documented requirement is "4 GB+ RAM", and the Alta has exactly 4 GB, shared with the kernel, GNOME (roughly 1 GB) and everything Claude starts. ninja defaults to CPU count + 2 jobs (8 on this board), and each compiler process for Mesa or the kernel can take hundreds of MB. The result is swap thrash on microSD that looks exactly like a lockup.
*Evidence:* OOM lines in the kernel log; high `full` values in `/proc/pressure/memory`; earlyoom or oomd log lines; SSH that connects but crawls.
*Prevention:* run headless; zram plus earlyoom or systemd-oomd (root step 020); `-j2`; the memory-capped scope from `scripts/alta-claude`; background tasks disabled (`CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1`) so builds can't stack up.

**L4 — The Mali Vulkan path** [RECORD F5 + HYPOTHESIS]. If `PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1` is global (some forum guides put it in `/etc/environment`), every Vulkan-capable program sees panvk. GTK ≥ 4.16 picks Vulkan by default on Wayland, so GTK4 terminals such as GNOME Console or Ptyxis render through a non-conformant driver, and Claude Code's full-screen terminal UI redraws constantly.
*Evidence:* panfrost job timeouts or faults in the kernel log; desktop frozen while SSH still works.
*Prevention:* set the variable per command only; set `GSK_RENDERER=ngl` if a global override can't be avoided; or run Claude Code over SSH, so the terminal renders on the workstation.

**L5 — Power or heat.** The board needs a 5 V / 3 A supply and its heatsink. Under-voltage on these boards typically leaves no clear log line (unlike a Raspberry Pi's firmware warning); it shows up as random reboots or freezes under load.
*Evidence:* temperatures in audit section 14; reboots whose logs look clean.
*Prevention:* a proper PSU, heatsink or fan, `-j2`.

**L6 — A kernel upgrade silently undoes the fix** [HYPOTHESIS, check this first]. See TL;DR item 1.
*Evidence:* audit section 5 shows `etnaviv.ko` under an older kernel's `updates/` but not the running one's; no `alta: SH_EU` marker in the kernel log; `/sys/module/etnaviv/taint` empty; a kernel upgrade in the apt history (audit section 2) after the last good date.
*Prevention:* `npu-run` refuses to run on a stock module; hold kernel packages (root step 070); rebuild after every kernel update (section 8.6).

### Triage

- **Screen frozen, but SSH works and is quick** → L1 or L4 (display stack). First evidence:
  ```bash
  journalctl -b -o short-monotonic _COMM=gnome-shell --no-pager | tail -80
  journalctl -k -b --no-pager | grep -iE 'panfrost|etnaviv|drm' | tail -40
  ```
- **SSH connects but every command crawls, storage LED busy** → L3 (memory thrash). First evidence:
  ```bash
  cat /proc/pressure/memory; free -m
  journalctl -k -b --no-pager | grep -iE 'oom|out of memory'      # once it recovers
  ```
- **Nothing responds, no ping, a power cycle is needed, and the previous boot's log stops abruptly (often right after etnaviv lines)** → L2 (NPU or SoC wedge), or L5. First evidence, after the power cycle:
  ```bash
  journalctl -k -b -1 --no-pager | tail -50
  ```
- **Random reboot under load** → L5. Check `journalctl -b -1 -n 50 --no-pager`, the temperatures in audit section 14, and the PSU.
- **Lockups came back after an update** → L6, then L2. Check audit sections 2 and 5 (installed kernels, apt history, the marker, `/sys/module/etnaviv/taint`).

---

## 5. Phase 0 — the read-only audit

```bash
cd ~/alta-npu && bash scripts/alta-audit.sh        # writes audit/audit-<timestamp>.txt
```

It changes nothing. Reads that need root are attempted with `sudo -n` and skipped if a password would be required. How to read it:

| Audit section shows | Meaning and next step |
|---|---|
| 2: kernel compiler is GCC N, and `gcc-N` is missing | The OOT build will pick the wrong compiler. Root step for Jordan: `apt install gcc-N` |
| 2: apt history has a `linux-image` / `linux-dtb` upgrade after the last known-good date | Strong signal for L6 |
| 3: `CONFIG_DRM_ETNAVIV=m` | An OOT replacement module works |
| 3: `CONFIG_DRM_ETNAVIV=y` | etnaviv is built in and cannot be replaced by a module; needs a kernel rebuild (Jordan's decision) |
| 3: `# CONFIG_DRM_ETNAVIV is not set` | The OOT build provides the module (needs `DRM_SCHED`, which panfrost already requires) |
| 4: `npu@ff100000` status `okay` or no status | NPU enabled in the DT |
| 4: status `disabled`, or no node | Section 7 |
| 5: "modprobe would load" points to `.../updates/alta/etnaviv.ko`, taint contains `O`, and section 15 shows the `alta: SH_EU` marker | Patched module active: F2 and F3 OK |
| 5: an `etnaviv.ko` exists only under an *older* kernel's `updates/` | L6 confirmed → section 8 |
| 6: etnaviv inside the initramfs | The early-boot udev stage won't apply our rule → root step 050 |
| 8–9: rule present and `TAGS`/`CURRENT_TAGS` on the `platform-etnaviv` nodes contain `mutter-device-ignore` | F4 OK |
| 11: Mesa older than 25.1 and GNOME in use | Also consider root step 050 (late load) |
| 12: panvk override set globally | L4 risk. Jordan removes it, or adds `GSK_RENDERER=ngl` |
| 13: no zram and no OOM daemon | Root step 020 |
| 15: "persistent journal: NO" | Root step 010. Without it, a hard lockup leaves no evidence |
| 15: previous boot ends with etnaviv errors / OOM lines | Evidence for L2 / L3 |
| 16: original patch, rule, build dir or shell history found | Prefer the originals; record their sha256 in `CHANGELOG.md` |
| 17: cgroup `memory.max` is `max` | Claude Code was not started through `scripts/alta-claude`. Restart it that way |

---

## 6. Phase 1 — baseline hardening

Jordan runs these with `sudo bash root-steps/<script>` after reading them. Each is idempotent and has its revert in the header.

### 6.1 Persistent journal and log access — `root-steps/010`

This makes `/var/log/journal` persistent, so the kernel log of a boot that locked up survives the power cycle. It also adds Jordan to `systemd-journal` (which `npu-run` needs to read the kernel log) and to `render` (SSH sessions get no seat ACLs on `/dev/dri/renderD*`). If `armbian-ramlog` or `log2ram` is active, `/var/log` lives in RAM and is only synced periodically, so lockup logs can still be lost. For forensics Jordan can disable it, which costs extra SD-card writes; on Armbian that is `ENABLED=false` in `/etc/default/armbian-ramlog`, then reboot.

### 6.2 Memory safety net — `root-steps/020` plus the launcher

- **zram** with zstd at 50 % of RAM, skipped if Armbian's zram already exists.
- **earlyoom**, which prefers to kill compilers, linkers and Python and avoids sshd, systemd, tmux and the compositor. It is skipped if `systemd-oomd` is already active (Ubuntu desktop), since one OOM daemon is enough.
- **`scripts/alta-claude`** puts Claude Code and everything it spawns into a systemd user scope with `MemoryHigh=2200M`, `MemoryMax=2800M`, `MemorySwapMax=1G` and `TasksMax=512` (override with `ALTA_MEM_HIGH` and friends). A runaway build gets throttled, then OOM-killed inside the scope, while sshd and the rest of the system keep running. Depending on the systemd version an OOM kill may stop the whole scope, Claude Code included; that is still far better than a frozen board.

### 6.3 Headless operation over SSH and tmux — `root-steps/030` (recommended)

Boot to `multi-user.target` for NPU work. This frees roughly 1 GB and removes L1 and L4 entirely. Run Claude Code from the workstation: `ssh jordan@alta` → `tmux new -As alta` → `~/alta-npu/scripts/alta-claude`. The terminal UI then renders on the workstation, and the session survives SSH drops. For a one-off desktop: `sudo systemctl start display-manager`. To go back to graphical boot: `sudo systemctl set-default graphical.target`.

### 6.4 Claude Code configuration — `.claude/settings.json` and the hook

- `env`: `BASH_DEFAULT_TIMEOUT_MS=300000` (5 min), `BASH_MAX_TIMEOUT_MS=3600000` (60 min), `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS=1` (no backgrounded or auto-backgrounded commands, so nothing piles up), `MAKEFLAGS=-j2`.
- `permissions`: deny rules for module removal and `udevadm trigger`; ask rules for `sudo`, `modprobe`, state-changing `systemctl`, `apt` and `pip install`.
- `hooks.PreToolUse` (matcher `Bash`) runs `.claude/hooks/alta-guard.sh`. Exit code 2 blocks the command and shows the reason to Claude. The hook fails closed if it can't parse its input. Its behavior was checked against 72 allow and block cases (A.3).

Verify inside a fresh session: `/permissions` and `/hooks` list the rules; asking Claude to run `make -j8` gets blocked; and this prints `2936012800`, which is 2800M:

```bash
cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max
```

### 6.5 Freeze kernel upgrades while the patched module is in use — `root-steps/070`

This holds the kernel meta-packages: Ubuntu `linux-generic*` / `-hwe-*`, Debian `linux-image-arm64`, Armbian `linux-image|dtb|headers-current-meson64`, and so on. Unhold them deliberately, then follow section 8.6.

---

## 7. Device tree: is the NPU described?

```bash
d=/proc/device-tree/npu@ff100000
ls -d "$d" && { tr -d '\0' 2>/dev/null < "$d/status" || echo "(no status = enabled)"; }; echo
```

- **`okay`, or no status property** → fine.
- **`disabled`** → kernel older than 6.17, or a vendor DT. Options, all Jordan's decision and never done from a Claude session: move to a kernel ≥ 6.17, whose Alta DTS enables the node, or apply an overlay. The overlay source would be the following; how to load it depends on the boot chain (U-Boot `fdtoverlays`, Libre Computer's overlay tooling, and so on):

```dts
/dts-v1/;
/plugin/;
&{/npu@ff100000} { status = "okay"; };
```

- **No node at all** → the DT predates NPU support, which landed for kernel 6.7 according to Mesa's Teflon docs. A newer kernel and DTB are needed.

---

## 8. Kernel side: etnaviv

### 8.1 Decide what to build

- The audit already shows the patched module for the **running** kernel (the `alta: SH_EU` marker, or an out-of-tree taint backed by a recorded build) → nothing to build; go to 8.5.
- Otherwise you need three things: headers for the running kernel (`/lib/modules/$(uname -r)/build`), source matching that kernel, and the GCC major version the kernel was built with.

### 8.2 Get source matching the running kernel

The etnaviv code must come from the same kernel version, because it uses internal DRM APIs that change between releases.

- **Debian/Ubuntu kernels:** the distro source package is best: `apt-get source linux` (needs `deb-src`), or the `linux-source-X.Y` package (tarball in `/usr/src`).
- **Upstream stable / Armbian-style kernels:** use a sparse, shallow checkout of the matching stable tag. It is about 13 MB, with only the etnaviv directory and the top-level files checked out. `gregkh/linux` on GitHub mirrors the kernel.org stable tree. Tested here with v6.12.43:

```bash
V=v6.12.43    # match the base version of `uname -r`
git clone --depth 1 --filter=blob:none --sparse --branch "$V" https://github.com/gregkh/linux.git ~/src/linux-$V
git -C ~/src/linux-$V sparse-checkout set drivers/gpu/drm/etnaviv
```

### 8.3 Build the patched module (F2 + F3) [RECON]

```bash
cd ~/alta-npu
scripts/build-etnaviv-oot.sh ~/src/linux-v6.12.43 scripts/apply-0x7120-sheu.py
# if the ORIGINAL patch turned up in audit section 16, use it instead (and record its sha256):
#   scripts/build-etnaviv-oot.sh ~/src/linux-v6.12.43 /path/to/original.patch     # PATCH_P=1 if its paths are a/etnaviv_gpu.c
```

What the script does (A.6):

1. Refuses if etnaviv is built in, if headers are missing, or if `JOBS` is above 2.
2. Reads the kernel's compiler from `CONFIG_CC_VERSION_TEXT` (falling back to `/proc/version`) and uses that exact binary name first, then `gcc-N`. kbuild's "compiler differs" warning is harmless when only the binary name differs and the version is the same.
3. Checks the source's `VERSION.PATCHLEVEL` against `uname -r`.
4. Copies `drivers/gpu/drm/etnaviv` to `build/etnaviv-<kver>-<stamp>/`, records sha256 sums, applies the patch, writes `ALTA-etnaviv_gpu.diff`, and builds with `CONFIG_DRM_ETNAVIV=m -j2`.
5. Compares vermagic with panfrost's, writes `BUILDINFO.txt`, and prints the install commands. It installs nothing itself.

The 0x7120 change (Appendix B) adds `etnaviv_is_model_rev(gpu, 0x8000, 0x7120)` to the SH_EU condition in `etnaviv_gpu_enable_mlcg()`, plus a `dev_info_once()` marker line. That marker is how `npu-run` and you can prove the patched module is the one running. Tested before handoff: the patcher applies cleanly to Linux master and to v6.12.43, and a v6.12.43 build (x86 test config) compiled, with the marker string present in the resulting `.ko`. On kernels before 6.11 there is no SH_EU block; the patcher refuses and changes nothing.

### 8.4 GNOME and the load order (F4)

- **Always, if GNOME may run on the board:** root step 040 installs `system/61-alta-npu-mutter-ignore.rules`, which tags the etnaviv card and render nodes `mutter-device-ignore`, matched by `ID_PATH=platform-etnaviv`, by `DEVPATH` and by `DRIVERS=="etnaviv"`. This is the mechanism mutter's own `61-mutter.rules` uses to ignore vkms.
- **Only if needed** (Mesa older than 25.1 with GNOME, tags missing after boot, or etnaviv inside the initramfs): root step 050 blacklists etnaviv autoloading and loads it from `alta-npu-load.service`, after udev is up and before the display manager. This mirrors the working VIM3 recipe. An explicit `modprobe etnaviv` still works with the blacklist in place.
- **Headless:** none of this affects NPU work, but keep the rule for when the desktop comes back.

### 8.5 Install and post-install checks

Install with `sudo bash root-steps/060-install-etnaviv-oot.sh build/etnaviv-<kver>-<stamp>`, then **reboot**. After the reboot:

```bash
modinfo -n etnaviv                                   # /lib/modules/<kver>/updates/alta/etnaviv.ko
cat /sys/module/etnaviv/taint                        # contains O (out of tree), usually E (unsigned) too
journalctl -k -b -g 'etnaviv|ff100000.npu' --no-pager
#  etnaviv-gpu ff100000.npu: model: GC8000, revision: 7120
#  etnaviv-gpu ff100000.npu: etnaviv has been instantiated on a NPU, for which the UAPI is still experimental
#  etnaviv-gpu ff100000.npu: alta: SH_EU clock gating disabled (VIPNano-QI rev 0x7120)
#  [drm] Initialized etnaviv ...
ls -l /dev/dri/by-path/ | grep etnaviv
udevadm info -q property -n /dev/dri/by-path/platform-etnaviv-card | grep -E '^(ID_PATH|TAGS|CURRENT_TAGS)='
#  ID_PATH=platform-etnaviv    TAGS / CURRENT_TAGS contain :mutter-device-ignore:   (once root step 040 is in place)
journalctl -b -o cat _COMM=gnome-shell | grep -i 'Ignoring DRM device'   # desktop boots only
```

Then log the module sha256 (from `BUILDINFO.txt`) in `CHANGELOG.md`.

### 8.6 After every kernel update

This is a fixed loop, and until it is done `npu-run` refuses to run because the marker is missing:

1. Install headers for the new kernel and the matching `gcc-N` (audit section 2).
2. Check out the matching source (8.2).
3. Rebuild (8.3).
4. Install with root step 060 and reboot.
5. Run the checks in 8.5.

DKMS is deliberately not used: some DKMS setups don't pass `CC=gcc-N`, which is the exact failure F2 is about.

---

## 9. Userspace: Teflon and the TFLite runtime

### 9.1 Getting `libteflon.so`

- **Distro package** `mesa-teflon-delegate`, where available (check `apt-cache policy mesa-teflon-delegate`). On Debian the library is `/usr/lib/teflon/libteflon.so`. Teflon first shipped in Mesa 24.1; later releases added operations and models. Tomeu Vizoso's 2025 XDC update listed ReLU, Reshape, Abs, Logistic, Subtract and Transpose, plus YOLOX-based models.
- **Isolated source build** that is never installed (R9). Ask Jordan first; at `-j2` this takes a long time, so he may prefer to run it in his own tmux window:

```bash
cd ~/src && git clone --depth 1 https://gitlab.freedesktop.org/mesa/mesa.git && cd mesa
meson setup build -Dgallium-drivers=etnaviv -Dvulkan-drivers= -Dteflon=true -Dbuildtype=release
meson compile -C build -j 2
ls -l build/src/gallium/targets/teflon/libteflon.so
```

Build dependencies per Mesa's docs: `apt-get build-dep mesa` (needs `deb-src`) plus `git` and `cmake`. Teflon only works with the etnaviv or rocket Gallium drivers, and etnaviv is the one the Alta needs.

### 9.2 Python runtime

```bash
python3 -m venv ~/alta-npu/venv
~/alta-npu/venv/bin/pip install numpy pillow ai-edge-litert
```

Mesa's docs use Python 3.10 with `tflite-runtime==2.13.0`. Its wheels don't cover newer Python versions, which is why `ai-edge-litert` is used here: it is the successor and keeps the same `Interpreter` / `load_delegate` entry points. `scripts/npu-classify.py` accepts either.

### 9.3 Test assets, pinned by sha256

```bash
cd ~/alta-npu/models
B=https://raw.githubusercontent.com/google-coral/test_data/master
curl -fLO "$B/mobilenet_v1_1.0_224_quant.tflite"; curl -fLO "$B/grace_hopper.bmp"; curl -fLO "$B/imagenet_labels.txt"
sha256sum -c - <<'EOF'
2d0ebfa2e75ae93c709e9ecc27570c9c0e49b7780733a840f82589fc1272fc3d  mobilenet_v1_1.0_224_quant.tflite
c43444bf3b1b1e3080551ca194bd231da4b67f20fc91454502915a6b806b614f  grace_hopper.bmp
b8aaeb5630c62626a7d124a05c0e14230a7ffdd2136a31d0190a4fb71c1d82f1  imagenet_labels.txt
EOF
```

These hashes were captured on 2026-11-07. Reference CPU result for this exact model, image and script (x86, LiteRT 2.3.0): top-1 index 653 "military uniform", score 0.906, then Windsor tie, mortarboard, stage, academic gown. Mesa's own example run on the NPU also reports "military uniform" first (0.87, with Mesa's script). Mesa's docs give about 15 ms per MobileNet V1 inference on the Alta, and the Mesa 26.0 example log shows about 7 ms. The first run also compiles the graph, which takes seconds.

---

## 10. Test protocol (repeatable)

The verdict rule is the same for every test. **PASS** means: npu-run exit 0, no kernel messages, CPU and NPU top-1 identical, and at least 4 of the top 5 shared. Record every test as a row in `RESULTS.md`.

Put the delegate path in a small env file once. Claude Code's separate Bash calls shouldn't be relied on to keep shell variables, so every test command below sources this file first:

```bash
cd ~/alta-npu
echo 'export TEFLON=/usr/lib/teflon/libteflon.so' > .alta-env      # or ~/src/mesa/build/src/gallium/targets/teflon/libteflon.so
echo 'export M=models/mobilenet_v1_1.0_224_quant.tflite I=models/grace_hopper.bmp L=models/imagenet_labels.txt' >> .alta-env
```

**T0, preconditions:** audit sections 5, 9, 13 and 15 are clean; the marker is present; there is no `runs/STOP`; the board is headless; `MemAvailable` is at least 1 GB.

**T1, CPU reference** (pass: top-1 = 653):

```bash
. ./.alta-env && venv/bin/python scripts/npu-classify.py -m "$M" -i "$I" -l "$L" -n 20 > results/T1-cpu.json
```

**T2, first NPU run** (pass: PASS verdict, and `runs/*-T2/stderr.txt` shows Teflon loaded the etnaviv driver):

```bash
. ./.alta-env && TEFLON_DEBUG=verbose scripts/npu-run -t 120 -n T2 -- \
  venv/bin/python scripts/npu-classify.py -m "$M" -i "$I" -l "$L" -e "$TEFLON" -n 5 > results/T2-npu.json
```

**T3, repeatability:** 20 separate processes; it stops at the first failure (pass: 20/20 PASS):

```bash
. ./.alta-env && for i in $(seq -w 1 20); do
  scripts/npu-run -t 120 -n "T3-$i" -- venv/bin/python scripts/npu-classify.py -m "$M" -i "$I" -e "$TEFLON" -n 5 \
    > "results/T3-$i.json" || break
done
```

**T4, soak** (only after T3, and only with Jordan's go-ahead; record temperatures before and after):

```bash
. ./.alta-env && scripts/npu-run -t 900 -n T4 -- venv/bin/python scripts/npu-classify.py -m "$M" -i "$I" -e "$TEFLON" -n 2000 > results/T4.json
```

**T5, desktop coexistence** (only if GNOME will run alongside the NPU): boot graphical, check the tags (8.5), then run T2 from an SSH session while the desktop is up. Pass: PASS verdict, and the desktop stays responsive.

Compare a CPU and an NPU result with:

```bash
python3 - results/T1-cpu.json results/T2-npu.json <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
ta = [t["index"] for t in a["top5"]]; tb = [t["index"] for t in b["top5"]]
ok = ta[0] == tb[0] and len(set(ta) & set(tb)) >= 4
print(f"CPU top1 {ta[0]} | NPU top1 {tb[0]} | top5 overlap {len(set(ta) & set(tb))}/5 | "
      f"CPU median {a['invoke_ms_median']} ms | NPU median {b['invoke_ms_median']} ms -> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
```

Debug variables: `TEFLON_DEBUG=verbose`, plus `ETNA_MESA_DEBUG=ml_msgs` (the name in the Mesa 26.0 docs; older docs use `ml_dbgs`). Only set them through `npu-run`.

---

## 11. Mali-G52 notes (panfrost and panvk)

- **GL/GLES:** panfrost. Mesa's docs call it conformant OpenGL ES on Mali-G52.
- **Vulkan:** panvk on Bifrost v7 is non-conformant and stays hidden unless `PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1` is set. Without it, the driver refuses to load with a "not a conformant vulkan implementation" warning. Since Mesa 25.0, panvk is enumerated by default only on v10 GPUs (Mali-G610 and G310).
- Use it per command, e.g. `PAN_I_WANT_A_BROKEN_VULKAN_DRIVER=1 vulkaninfo --summary`. Never make it global: GTK ≥ 4.16 defaults to Vulkan on Wayland. If a global override is unavoidable, also set `GSK_RENDERER=ngl` for the session.
- The earlier "environment variable workarounds" [RECORD] are located by audit section 12.

---

## 12. Troubleshooting and recovery

### 12.1 Symptom → cause → action

| Symptom | Cause | Action |
|---|---|---|
| `unrecognized command-line option '-fmin-function-alignment=8'`, or "compiler differs" with different *versions* | Wrong GCC for this kernel | Install the `gcc-N` that `/proc/version` names; the build script picks it |
| `Invalid module format` or `disagrees about version of symbol` when loading | Source or headers don't match the running kernel | Rebuild from the exact source; compare against `BUILDINFO.txt` |
| `module verification failed: signature and/or required key missing - tainting kernel` | Unsigned OOT module | Expected (taint E). Only fatal with `CONFIG_MODULE_SIG_FORCE=y` |
| etnaviv registration fails because the driver already exists | etnaviv is built in (`=y`) | Kernel rebuild needed (Jordan's decision) |
| No `npu@` node, or status `disabled` | Device tree | Section 7 |
| Inference runs at CPU speed and Teflon never reports loading the etnaviv driver | Delegate not loaded | Check the `-e` path, `runs/*/stderr.txt`, render-node permissions |
| `Permission denied` on `/dev/dri/renderD*` over SSH | Not in group `render` | Root step 010 |
| GNOME black screen or failed session after the NPU is enabled | L1 | Root step 040, then 050 if needed; Mesa ≥ 25.1 |
| `npu-run: REFUSED: stock in-tree etnaviv is loaded` | L6 | Section 8.6 |
| `npu-run` set the STOP latch | L2 (or a deadline) | 12.2, then reboot |

### 12.2 Collect evidence after a hang (before anything else)

If the board still answers right after a hang, Jordan grabs etnaviv's devcoredump first. Devcoredumps are removed automatically about 5 minutes after the hang:

```bash
mkdir -p ~/alta-npu/incidents/$(date +%F-%H%M) && cd "$_"
ls /sys/class/devcoredump/ && sudo cat /sys/class/devcoredump/devcd*/data > etnaviv-devcoredump.bin
journalctl -k -b 0 --no-pager > kernel-thisboot.txt
```

After a power cycle (or after Jordan's reboot), from the same incident directory:

```bash
journalctl -k -b -1 --no-pager > kernel-prevboot.txt
journalctl -b -1 --no-pager -n 400 > system-prevboot-tail.txt
cp -r ~/alta-npu/runs/<run named in runs/STOP> .
```

Jordan reviews the incident, and only then runs `rm ~/alta-npu/runs/STOP`.

### 12.3 If hangs continue with the patched module

Run these as separate, recorded experiments, in this order:

1. Re-prove the marker is in this boot's log, and that the module sha256 matches `BUILDINFO.txt`.
2. Serialize hardware jobs: `options etnaviv hw_job_limit=1` in `/etc/modprobe.d/alta-etnaviv-debug.conf` (root step, then reboot).
3. Add Lucas Stach's unmerged reset patch, which stops touching the clock-gating register during reset (section 3), as a second patch in the OOT build.
4. Try a newer Teflon from an isolated Mesa build (9.1) to separate userspace problems from kernel ones.
5. If the original 0x7120 patch from the earlier work turns up and differs from the reconstruction (for example, if it disables other clock-gating bits), test it as its own variant.

### 12.4 Reverting

Every root step names its revert in its header. Full rollback of the NPU stack:

1. Remove `/lib/modules/$(uname -r)/updates/alta/etnaviv.ko`, run `depmod -a`, and `update-initramfs -u` if 060 used it.
2. Remove `/etc/udev/rules.d/61-alta-npu-mutter-ignore.rules`.
3. Remove `/etc/modprobe.d/alta-etnaviv.conf` and `alta-npu-load.service`.
4. `systemctl set-default graphical.target`.
5. `apt-mark unhold` the held kernel packages.

---

## 13. Open questions to settle on the board (Claude asks Jordan, with the audit results in hand)

1. Which image and kernel are running now, and which were running when the fix last worked? (Audit sections 1–2 and the apt kernel history.)
2. Where is the original 0x7120 patch, and did it touch only SH_EU or other clock-gating bits too? (Audit section 16, shell history.)
3. Does GNOME need to run during NPU work, or can the board stay headless?
4. Was the panvk override made global, and where? (Audit section 12.)
5. How is Claude Code started today: from a terminal on the Alta's desktop, or over SSH?

---

## Appendix A — Kit files

Every block below is preceded by an invisible `file:` marker, which is what the extractor in 0.1 (and `scripts/extract-kit.py`) uses to write it out. Existing files are kept unless `--force` is given. These are the exact files that were linted (shellcheck, py_compile, `systemd-analyze verify`) and tested before this handoff was written.

### A.1 `CLAUDE.md`

<!-- file: CLAUDE.md -->
```markdown
# Alta (AML-A311D-CC) NPU project

@ALTA_NPU_HANDOFF.md

## Non-negotiables (full text: handoff section 1)
- First action in a new session: `bash scripts/alta-audit.sh`, then report the findings before changing anything.
- Never `rmmod`, `modprobe -r`, `insmod`, `udevadm trigger`, reboot, or touch /boot, /lib/modules, initramfs, DT.
- Root work = write a script in `root-steps/` (what, why, revert) for Jordan to review and run with sudo.
- Builds: at most `-j2`, one heavy job at a time. Never `ninja install` Mesa over the system Mesa.
- NPU jobs only through `scripts/npu-run`. If it sets `runs/STOP`, stop all NPU work and report.
- Log every system change in CHANGELOG.md and every test in RESULTS.md.
```

### A.2 `.claude/settings.json`

<!-- file: .claude/settings.json -->
```json
{
  "env": {
    "BASH_DEFAULT_TIMEOUT_MS": "300000",
    "BASH_MAX_TIMEOUT_MS": "3600000",
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "MAKEFLAGS": "-j2"
  },
  "permissions": {
    "deny": [
      "Bash(rmmod *)",
      "Bash(sudo rmmod *)",
      "Bash(modprobe -r *)",
      "Bash(sudo modprobe -r *)",
      "Bash(udevadm trigger *)",
      "Bash(sudo udevadm trigger *)"
    ],
    "ask": [
      "Bash(sudo *)",
      "Bash(modprobe *)",
      "Bash(systemctl start *)",
      "Bash(systemctl stop *)",
      "Bash(systemctl restart *)",
      "Bash(systemctl enable *)",
      "Bash(systemctl disable *)",
      "Bash(apt *)",
      "Bash(apt-get *)",
      "Bash(pip install *)",
      "Bash(python3 -m pip install *)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command", "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/alta-guard.sh" }
        ]
      }
    ]
  }
}
```

### A.3 `.claude/hooks/alta-guard.sh` (PreToolUse guard; 72 allow/block test cases passed)

<!-- file: .claude/hooks/alta-guard.sh mode=755 -->
```bash
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
```

### A.4 `scripts/alta-audit.sh` (read-only)

<!-- file: scripts/alta-audit.sh mode=755 -->
```bash
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
```

### A.5 `scripts/alta-claude` (memory-capped launcher)

<!-- file: scripts/alta-claude mode=755 -->
```bash
#!/usr/bin/env bash
# alta-claude - start Claude Code inside a memory-capped systemd user scope.
# Everything Claude Code spawns (builds, tests, NPU jobs) inherits the cap, so a runaway job is
# OOM-killed inside the scope instead of freezing the whole 4 GB board.
# Run it from tmux in an SSH session:  ssh alta  ->  tmux new -As alta  ->  ~/alta-npu/scripts/alta-claude
set -eu
: "${ALTA_MEM_HIGH:=2200M}"   # above this the scope is throttled and reclaimed
: "${ALTA_MEM_MAX:=2800M}"    # hard cap: the kernel OOM-kills inside this scope only
: "${ALTA_SWAP_MAX:=1G}"
: "${ALTA_TASKS_MAX:=512}"
export MAKEFLAGS="${MAKEFLAGS:--j2}"
command -v systemd-run >/dev/null || { echo "alta-claude: systemd-run not found" >&2; exit 1; }
command -v claude >/dev/null || { echo "alta-claude: claude not on PATH" >&2; exit 1; }
cd "${ALTA_PROJECT_DIR:-$HOME/alta-npu}"
exec systemd-run --user --scope --quiet --unit="alta-claude-$(date +%H%M%S)" \
  -p MemoryHigh="$ALTA_MEM_HIGH" -p MemoryMax="$ALTA_MEM_MAX" \
  -p MemorySwapMax="$ALTA_SWAP_MAX" -p TasksMax="$ALTA_TASKS_MAX" \
  claude "$@"
```

### A.6 `scripts/build-etnaviv-oot.sh` (tested end to end against a Linux v6.12.43 tree, x86 test config)

<!-- file: scripts/build-etnaviv-oot.sh mode=755 -->
```bash
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
```

### A.7 `scripts/apply-0x7120-sheu.py` [RECON]

<!-- file: scripts/apply-0x7120-sheu.py mode=755 -->
```python
#!/usr/bin/env python3
"""apply-0x7120-sheu.py - add the Amlogic A311D NPU (VIPNano-QI, model 0x8000 rev 0x7120)
to etnaviv's SH_EU module-clock-gating workaround in etnaviv_gpu_enable_mlcg(), plus a
one-time kernel-log marker so the running module can be proven to be the patched one.

RECONSTRUCTION: rebuilt from upstream etnaviv (Linux >= 6.11 has the SH_EU block for
0x8000/0x7200, 0x8000/0x8002, 0x9200/0x6304 - not 0x7120). If the original patch from the
earlier work is found on the board, prefer it and record its sha256.

Usage: apply-0x7120-sheu.py path/to/etnaviv_gpu.c
Idempotent. Exits non-zero (and changes nothing) unless exactly one SH_EU block matches.
"""
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "etnaviv_gpu.c")
src = path.read_text()

if "0x8000, 0x7120" in src:
    print(f"{path}: already contains 0x8000/0x7120 - nothing to do")
    sys.exit(0)

# Upstream (>= 6.11) shape, tabs as in the kernel source:
#   \tif (etnaviv_is_model_rev(gpu, 0x8000, 0x7200) ||
#   \t    etnaviv_is_model_rev(gpu, 0x8000, 0x8002) ||
#   \t    etnaviv_is_model_rev(gpu, 0x9200, 0x6304))
#   \t\tpmc |= VIVS_PM_MODULE_CONTROLS_DISABLE_MODULE_CLOCK_GATING_SH_EU;
pat = re.compile(
    r"^(?P<ind>[ \t]+)(?P<last>etnaviv_is_model_rev\(gpu, 0x9200, 0x6304\))\)\n"
    r"(?P<bind>[ \t]+)(?P<body>pmc \|= VIVS_PM_MODULE_CONTROLS_DISABLE_MODULE_CLOCK_GATING_SH_EU;)\n",
    re.M,
)
matches = list(pat.finditer(src))
if len(matches) != 1:
    sys.exit(
        f"{path}: expected exactly 1 SH_EU clock-gating block, found {len(matches)}.\n"
        "Kernel older than 6.11 (no SH_EU block) or the block changed upstream.\n"
        "Make the equivalent change by hand: add 'etnaviv_is_model_rev(gpu, 0x8000, 0x7120)'\n"
        "to the SH_EU condition in etnaviv_gpu_enable_mlcg(), and record the diff."
    )

m = matches[0]
ind, bind = m.group("ind"), m.group("bind")
outer = bind[:-1] if bind.endswith("\t") else bind      # indentation of the 'if' itself
new = (
    f"{ind}{m.group('last')} ||\n"
    f"{ind}etnaviv_is_model_rev(gpu, 0x8000, 0x7120))\n"
    f"{bind}{m.group('body')}\n"
    f"{outer}if (etnaviv_is_model_rev(gpu, 0x8000, 0x7120))\n"
    f"{bind}dev_info_once(gpu->dev, \"alta: SH_EU clock gating disabled (VIPNano-QI rev 0x7120)\\n\");\n"
)
path.write_text(src[: m.start()] + new + src[m.end():])
print(f"{path}: patched - 0x8000/0x7120 added to the SH_EU workaround, dev_info_once marker added")
```

### A.8 `scripts/npu-run` (lock, deadline, kernel-log verdict, STOP latch; all paths exercised against a simulated kernel log)

<!-- file: scripts/npu-run mode=755 -->
```bash
#!/usr/bin/env bash
# npu-run - run exactly ONE NPU job: global lock, hard deadline, kernel-log check, run record.
# Usage: scripts/npu-run [-t SECONDS] [-n LABEL] -- command [args...]
# Exit : the command's code | 124 deadline hit | 75 another job running | 77 refused | 90 kernel errors
# Any kernel error or deadline sets the STOP latch ($NPU_RUN_DIR/STOP); every later run is refused
# until Jordan reboots, reviews the named run, and deletes the latch by hand.
set -u
TIMEOUT=120
LABEL=job
while [ $# -gt 0 ]; do
  case "$1" in
    -t) TIMEOUT="$2"; shift 2 ;;
    -n) LABEL="$2"; shift 2 ;;
    --) shift; break ;;
    *)  break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: npu-run [-t SECONDS] [-n LABEL] -- cmd ..." >&2; exit 2; }

ROOT="${NPU_RUN_DIR:-$HOME/alta-npu/runs}"
mkdir -p "$ROOT"
refuse() { echo "npu-run: REFUSED: $*" >&2; exit 77; }
klog()   { journalctl -k --no-pager -q "$@" 2>/dev/null; }

# 0. Stop latch.
[ -e "$ROOT/STOP" ] && refuse "STOP latch present ($(head -1 "$ROOT/STOP")). Reboot, review that run, then: rm $ROOT/STOP"

# 1. Kernel log must be readable or hangs cannot be detected.
[ -n "$(klog -n 1)" ] || refuse "cannot read the kernel log (add the user to group systemd-journal or adm, log in again)"

# 2. The patched etnaviv must be the one that is loaded.
[ -d /sys/module/etnaviv ] || refuse "etnaviv is not loaded"
TAINT="$(cat /sys/module/etnaviv/taint 2>/dev/null)"
if klog -b 0 | grep -q 'alta: SH_EU clock gating disabled'; then
  PATCHSTATE="patched (marker in kernel log)"
elif [ "${TAINT#*O}" != "$TAINT" ]; then
  PATCHSTATE="out-of-tree module, no marker (taint=$TAINT) - confirm it is the 0x7120 build"
else
  refuse "stock in-tree etnaviv is loaded (no 'alta: SH_EU' marker, taint='$TAINT'): the 0x7120 clock-gating fix is NOT active"
fi

# 3. Do not start on a board that is already in trouble or short of memory.
klog -b 0 | grep -qiE 'recover hung gpu|failed to reset|mmu fault' && refuse "this boot already logged an etnaviv hang or MMU fault - reboot first"
AVAIL_MB="$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo)"
[ "$AVAIL_MB" -ge "${NPU_MIN_AVAIL_MB:-600}" ] || refuse "only ${AVAIL_MB} MB available (< ${NPU_MIN_AVAIL_MB:-600} MB) - stop builds first"

# 4. One NPU job at a time.
LOCK="${XDG_RUNTIME_DIR:-/tmp}/alta-npu.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "npu-run: another NPU job holds $LOCK" >&2; exit 75; }

RUN="$ROOT/$(date +%Y%m%d-%H%M%S)-$LABEL"
mkdir -p "$RUN"
START="$(date +%s)"
{
  echo "label: $LABEL"
  echo "cmd: $*"
  echo "cwd: $PWD"
  echo "start: $(date -Is)"
  echo "kernel: $(uname -r)"
  echo "etnaviv: $PATCHSTATE; srcversion=$(cat /sys/module/etnaviv/srcversion 2>/dev/null)"
  echo "mem_available_mb: $AVAIL_MB"
  echo "deadline_s: $TIMEOUT"
  env | grep -E '^(TEFLON_|ETNA_|MESA_)' || true
} > "$RUN/meta.txt"

# 5. Run with our own deadline (survives a child stuck in D state, unlike plain `timeout`).
"$@" > "$RUN/stdout.txt" 2> "$RUN/stderr.txt" &
PID=$!
RC=""
while kill -0 "$PID" 2>/dev/null; do
  if [ $(( $(date +%s) - START )) -ge "$TIMEOUT" ]; then
    pkill -TERM -P "$PID" 2>/dev/null; kill -TERM "$PID" 2>/dev/null; sleep 5
    pkill -KILL -P "$PID" 2>/dev/null; kill -KILL "$PID" 2>/dev/null; sleep 2
    if kill -0 "$PID" 2>/dev/null; then
      echo "pid $PID unkillable (state $(awk '{print $3}' "/proc/$PID/stat" 2>/dev/null)) - NPU likely wedged" >> "$RUN/meta.txt"
    fi
    RC=124
    break
  fi
  sleep 0.5
done
if [ -z "$RC" ]; then wait "$PID"; RC=$?; fi
sleep 1  # let late kernel messages land
echo "end: $(date -Is)  rc: $RC  wall_s: $(( $(date +%s) - START ))" >> "$RUN/meta.txt"
klog --since "@$((START - 1))" -o short-monotonic > "$RUN/kernel.txt"

# 6. Verdict. During a healthy run etnaviv logs nothing at all.
BAD='recover hung gpu|failed to reset|mmu fault|not yet idle|not idle|timed? ?out|out of memory|oom-kill|soft lockup|hung_task|rcu.*stall|Internal error|Oops|BUG:'
if grep -iE 'etnaviv|ff100000\.npu' "$RUN/kernel.txt" | grep -viE 'alta: SH_EU' | grep -q . \
   || grep -qiE "$BAD" "$RUN/kernel.txt"; then
  printf '%s %s (kernel messages)\n' "$(date -Is)" "$RUN" > "$ROOT/STOP"
  echo "npu-run: KERNEL MESSAGES DURING THE RUN -> STOP latch set. Do not retry. Evidence: $RUN/kernel.txt" >&2
  head -20 "$RUN/kernel.txt" >&2
  exit 90
fi
if [ "$RC" -eq 124 ]; then
  printf '%s %s (deadline)\n' "$(date -Is)" "$RUN" > "$ROOT/STOP"
  echo "npu-run: DEADLINE (${TIMEOUT}s) hit -> STOP latch set. Evidence: $RUN" >&2
fi
cat "$RUN/stdout.txt"
[ -s "$RUN/stderr.txt" ] && tail -20 "$RUN/stderr.txt" >&2
echo "npu-run: rc=$RC record=$RUN" >&2
exit "$RC"
```

### A.9 `scripts/npu-classify.py` (CPU path verified with the pinned model in 9.3)

<!-- file: scripts/npu-classify.py mode=755 -->
```python
#!/usr/bin/env python3
"""npu-classify.py - MobileNet-style image classification on the CPU or through the Teflon
delegate (NPU). Prints one JSON object so CPU and NPU runs can be compared.
Works with tflite_runtime (Mesa docs) or ai_edge_litert (its successor, same entry points).

CPU reference : python3 scripts/npu-classify.py -m MODEL.tflite -i IMAGE > results/cpu.json
NPU (always via npu-run):
  scripts/npu-run -n mnv1 -- python3 scripts/npu-classify.py -m MODEL.tflite -i IMAGE -e /path/libteflon.so > results/npu.json
"""
import argparse
import json
import statistics
import time

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
    RUNTIME = "tflite_runtime"
except ImportError:
    from ai_edge_litert.interpreter import Interpreter, load_delegate
    RUNTIME = "ai_edge_litert"

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-e", "--delegate", help="path to libteflon.so (omit for a CPU run)")
ap.add_argument("-l", "--labels", help="optional labels file, one label per line")
ap.add_argument("-n", "--iterations", type=int, default=5)
ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()

delegates = [load_delegate(a.delegate)] if a.delegate else []
t0 = time.monotonic()
interp = Interpreter(model_path=a.model, experimental_delegates=delegates, num_threads=a.threads)
interp.allocate_tensors()          # Teflon compiles the graph for the NPU here
setup_ms = (time.monotonic() - t0) * 1000.0

inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
_, h, w, _ = inp["shape"]
x = np.expand_dims(np.asarray(Image.open(a.image).convert("RGB").resize((w, h))), 0)
if inp["dtype"] == np.float32:
    x = (x.astype(np.float32) - 127.5) / 127.5
elif inp["dtype"] == np.int8:
    x = (x.astype(np.int16) - 128).astype(np.int8)
else:
    x = x.astype(inp["dtype"])
interp.set_tensor(inp["index"], x)

times = []
for _ in range(a.iterations):
    t = time.monotonic()
    interp.invoke()
    times.append((time.monotonic() - t) * 1000.0)

y = np.squeeze(interp.get_tensor(out["index"]))
if out["dtype"] != np.float32:
    scale, zero = out["quantization"]
    y = (y.astype(np.float32) - zero) * (scale or 1.0)
top = np.argsort(y)[::-1][:5]
labels = open(a.labels).read().splitlines() if a.labels else None

print(json.dumps({
    "runtime": RUNTIME,
    "delegate": a.delegate or "cpu",
    "model": a.model,
    "image": a.image,
    "setup_ms": round(setup_ms, 1),
    "invoke_ms": [round(t, 2) for t in times],
    "invoke_ms_median": round(statistics.median(times), 2),
    "top5": [{"index": int(i), "score": round(float(y[i]), 6),
              "label": labels[i] if labels and i < len(labels) else None} for i in top],
}, indent=1))
```

### A.10 `system/61-alta-npu-mutter-ignore.rules`

<!-- file: system/61-alta-npu-mutter-ignore.rules -->
```text
# /etc/udev/rules.d/61-alta-npu-mutter-ignore.rules
# The A311D NPU (VeriSilicon VIPNano-QI.7120) is driven by etnaviv and shows up as a DRM card plus a
# render node with no display outputs. Tag both so GNOME's compositor (mutter) never treats it as a GPU.
# Same mechanism mutter's own 61-mutter.rules uses for vkms. Named 61-* so it runs after 60-drm.rules
# (which imports ID_PATH). Three equivalent matches for robustness; tags are a set, duplicates are harmless.
SUBSYSTEM=="drm", KERNEL=="card*|renderD*", ENV{ID_PATH}=="platform-etnaviv", TAG+="mutter-device-ignore"
SUBSYSTEM=="drm", KERNEL=="card*|renderD*", ENV{DEVPATH}=="/devices/platform/etnaviv/*", TAG+="mutter-device-ignore"
SUBSYSTEM=="drm", KERNEL=="card*|renderD*", DRIVERS=="etnaviv", TAG+="mutter-device-ignore"
```

### A.11 `system/alta-etnaviv.conf` (only with root step 050)

<!-- file: system/alta-etnaviv.conf -->
```text
# /etc/modprobe.d/alta-etnaviv.conf  (only used with root-steps/050)
# Stop udev coldplug from auto-loading the NPU driver early; alta-npu-load.service loads it later,
# after our udev rule is active and after panfrost has taken the first render node.
# An explicit `modprobe etnaviv` still works with this blacklist in place.
blacklist etnaviv
```

### A.12 `system/alta-npu-load.service` (only with root step 050)

<!-- file: system/alta-npu-load.service -->
```ini
# /etc/systemd/system/alta-npu-load.service  (only used with root-steps/050)
[Unit]
Description=Load etnaviv (A311D NPU) after udev rules are active and before the display manager
After=systemd-udevd.service systemd-udev-trigger.service systemd-modules-load.service
Before=display-manager.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/sbin/modprobe etnaviv
ExecStartPost=-/usr/bin/udevadm settle --timeout=10

[Install]
WantedBy=multi-user.target
```

### A.13 Root steps (Jordan runs: `sudo bash root-steps/<file>`)

<!-- file: root-steps/010-journal-and-log-access.sh mode=755 -->
```bash
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
```

<!-- file: root-steps/020-memory-safety-net.sh mode=755 -->
```bash
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
```

<!-- file: root-steps/030-headless-boot.sh mode=755 -->
```bash
#!/usr/bin/env bash
# 030 - boot to text mode for NPU work (no GNOME; frees roughly 1 GB and removes the display-stack failure modes).
# Run: sudo bash root-steps/030-headless-boot.sh     Revert: sudo systemctl set-default graphical.target
set -euxo pipefail
systemctl set-default multi-user.target
echo "Takes effect at the next boot. One-off desktop: sudo systemctl start display-manager"
```

<!-- file: root-steps/040-udev-mutter-ignore.sh mode=755 -->
```bash
#!/usr/bin/env bash
# 040 - keep GNOME/mutter away from the NPU's DRM nodes.
# Run: sudo bash root-steps/040-udev-mutter-ignore.sh
# Revert: rm /etc/udev/rules.d/61-alta-npu-mutter-ignore.rules && udevadm control --reload
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
install -m 0644 "$here/system/61-alta-npu-mutter-ignore.rules" /etc/udev/rules.d/61-alta-npu-mutter-ignore.rules
udevadm verify /etc/udev/rules.d/61-alta-npu-mutter-ignore.rules 2>/dev/null || true   # systemd >= 255 only
udevadm control --reload
echo "Takes effect when the NPU device is next added (next boot). Do NOT run 'udevadm trigger' with a desktop up."
```

<!-- file: root-steps/050-etnaviv-late-load.sh mode=755 -->
```bash
#!/usr/bin/env bash
# 050 - ONLY IF NEEDED (handoff section 8.4): load etnaviv late from a service instead of udev coldplug.
# Run: sudo bash root-steps/050-etnaviv-late-load.sh
# Revert: systemctl disable alta-npu-load.service; rm /etc/systemd/system/alta-npu-load.service \
#         /etc/modprobe.d/alta-etnaviv.conf; systemctl daemon-reload; update-initramfs -u
set -euxo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
install -m 0644 "$here/system/alta-etnaviv.conf" /etc/modprobe.d/alta-etnaviv.conf
install -m 0644 "$here/system/alta-npu-load.service" /etc/systemd/system/alta-npu-load.service
systemctl daemon-reload
systemctl enable alta-npu-load.service
if command -v update-initramfs >/dev/null; then update-initramfs -u -k "$(uname -r)"; fi
echo "Reboot to apply."
```

<!-- file: root-steps/060-install-etnaviv-oot.sh mode=755 -->
```bash
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
```

<!-- file: root-steps/070-hold-kernel.sh mode=755 -->
```bash
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
```

### A.14 Templates

<!-- file: templates/CHANGELOG.md -->
```markdown
# CHANGELOG - system changes on the Alta (newest first)

| Date | Step / script | What changed | Why | Verify | Revert |
|---|---|---|---|---|---|
| YYYY-MM-DD | root-steps/0NN-... | | | | |
```

<!-- file: templates/RESULTS.md -->
```markdown
# RESULTS - NPU test runs (newest first)

| Date | Test | Kernel | etnaviv (marker / taint / srcversion) | Mesa / Teflon path | Model sha256 (12) | CPU top1 | NPU top1 | Median invoke ms | Kernel msgs | Verdict | Run dir |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | T2 | | | | | | | | none | PASS | runs/... |
```

### A.15 `scripts/extract-kit.py` (same logic as the bootstrap in 0.1, for re-extracting later)

<!-- file: scripts/extract-kit.py mode=755 -->
```python
#!/usr/bin/env python3
"""extract-kit.py - write every code block in the handoff that is preceded by an HTML comment
line of the form  file: RELATIVE/PATH [mode=NNN]  to disk under DEST. Existing files are kept unless --force.
Usage: python3 scripts/extract-kit.py ALTA_NPU_HANDOFF.md [DEST] [--force]"""
import os
import pathlib
import re
import sys

args = [a for a in sys.argv[1:] if a != "--force"]
force = "--force" in sys.argv
src = pathlib.Path(args[0] if args else "ALTA_NPU_HANDOFF.md")
dst = pathlib.Path(args[1] if len(args) > 1 else ".")
FENCE = "`" * 3
pat = re.compile(r"^<!-- file: (\S+)(?: mode=([0-7]{3,4}))? -->\n" + FENCE + r"[^\n]*\n(.*?)\n" + FENCE + r"[ \t]*$",
                 re.S | re.M)
n = 0
for rel, mode, body in pat.findall(src.read_text()):
    if rel.startswith("/") or ".." in pathlib.PurePosixPath(rel).parts:
        sys.exit(f"refusing unsafe path: {rel}")
    p = dst / rel
    if p.exists() and not force:
        print(f"skip (exists): {p}")
        continue
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body + "\n")
    if mode:
        os.chmod(p, int(mode, 8))
    n += 1
    print(f"wrote {p}")
print(f"{n} file(s) written to {dst.resolve()}")
```

---

## Appendix B — The 0x7120 change as a diff (for review)

This was generated by running `apply-0x7120-sheu.py` on Linux master's `etnaviv_gpu.c`; on v6.12.43 the same hunk lands at line 657. Use the script, not this text: copying a diff out of rendered markdown can turn its tabs into spaces.

```diff
--- a/drivers/gpu/drm/etnaviv/etnaviv_gpu.c
+++ b/drivers/gpu/drm/etnaviv/etnaviv_gpu.c
@@ -700,8 +700,11 @@
 	/* Disable SH_EU clock gating on affected core revisions. */
 	if (etnaviv_is_model_rev(gpu, 0x8000, 0x7200) ||
 	    etnaviv_is_model_rev(gpu, 0x8000, 0x8002) ||
-	    etnaviv_is_model_rev(gpu, 0x9200, 0x6304))
+	    etnaviv_is_model_rev(gpu, 0x9200, 0x6304) ||
+	    etnaviv_is_model_rev(gpu, 0x8000, 0x7120))
 		pmc |= VIVS_PM_MODULE_CONTROLS_DISABLE_MODULE_CLOCK_GATING_SH_EU;
+	if (etnaviv_is_model_rev(gpu, 0x8000, 0x7120))
+		dev_info_once(gpu->dev, "alta: SH_EU clock gating disabled (VIPNano-QI rev 0x7120)\n");
 
 	pmc |= VIVS_PM_MODULE_CONTROLS_DISABLE_MODULE_CLOCK_GATING_RA_HZ;
 	pmc |= VIVS_PM_MODULE_CONTROLS_DISABLE_MODULE_CLOCK_GATING_RA_EZ;
```

---

## Appendix C — Sources

Hardware and DT
- CNX Software, Alta launch and specs: https://www.cnx-software.com/2023/11/09/libre-computer-aml-a311d-cc-alta-sbc-features-amlogic-a311d-ai-processor/
- Bret.dk Alta review (heatsink, no Wi-Fi, Libre Computer images): https://bret.dk/libre-computer-alta-review-big-cottonwood/
- TechInfoDepot Alta page (5 V/3 A, ports): https://techinfodepot.shoutwiki.com/wiki/Libre_Computer_Alta
- U-Boot board doc for AML-A311D-CC: https://concept-docs.readthedocs.io/en/latest/_sources/board/amlogic/aml-a311d-cc.rst.txt
- "arm64: dts: amlogic: Enable the npu node for Alta and VIM3" (applied to v6.17/arm64-dt): https://git.kernel.org/amlogic/c/587c1c00f75565567d1f26a333a3392f7a21c28c
- Original A311D NPU series (DT node, power domain, hwdb): https://lists.infradead.org/pipermail/linux-arm-kernel/2022-December/794254.html
- Linux master sources checked 2026-11-07: `drivers/gpu/drm/etnaviv/{etnaviv_gpu.c,etnaviv_hwdb.c,etnaviv_sched.c,etnaviv_drv.c,etnaviv_flop_reset.c,Makefile}`, `arch/arm64/boot/dts/amlogic/{meson-g12-common.dtsi,meson-g12b.dtsi,meson-g12b-a311d-libretech-cc.dts}` via https://github.com/torvalds/linux

etnaviv clock gating and reset
- SH_EU clock-gating workaround (0x7200 / 0x8002 / 0x6304): https://lkml.org/lkml/2024/1/25/449
- etnaviv changes for Linux 6.11, including the NPU clock-gating fix: https://www.phoronix.com/news/Etnaviv-Driver-Linux-6.11
- "drm/etnaviv: don't touch clock gating register during reset" (June 2025, unmerged in master): https://lists.freedesktop.org/archives/etnaviv/2025-June/005267.html

GNOME / Mesa interaction
- VIM3 GNOME failure with the NPU, and the working recipe: https://lists.infradead.org/pipermail/linux-amlogic/2024-March/020488.html and https://lists.infradead.org/pipermail/linux-amlogic/2023-January/014717.html
- Render-node / renderonly discussion ("Create an accel device node if compute-only"): https://lkml.iu.edu/2406.3/03933.html
- Mesa 25.1.0 release notes (kmsro graphics-capable renderonly; surfaceless graphics-only): https://docs.mesa3d.org/relnotes/25.1.0.html
- mutter `mutter-device-ignore` on init and hotplug: https://lists.gnome.org/archives/commits-list/2021-June/msg02190.html ; mutter `data/61-mutter.rules` (vkms ignore rules) via https://github.com/GNOME/mutter

Teflon / TFLite
- Mesa Teflon docs: https://docs.mesa3d.org/teflon.html (and the Debian copies of the 25.3 / 26.0 versions on sources.debian.org)
- Teflon upstreamed for Mesa 24.1: https://www.cnx-software.com/2024/01/25/vivante-npu-amlogic-a311d-open-source-driver-mesa/
- Debian `mesa-teflon-delegate` file list: https://packages.debian.org/hu/bookworm-backports/arm64/mesa-teflon-delegate/filelist
- XDC 2025 NPU update (new Etnaviv ops, YOLOX): https://indico.freedesktop.org/event/10/contributions/404/attachments/245/333/XDC_2025_NPU_TV.pdf
- ai-edge-litert as successor to tflite-runtime (same entry points): https://git.laurivan.com/Mirrors/blakeblackshear.frigate/src/branch/vulkan/frigate/detectors/detector_utils.py
- Test assets: https://github.com/google-coral/test_data

Mali / GTK
- panvk conformance gate message: https://forums.puri.sm/t/broken-vulkan-driver-on-librem-5/25325
- Mesa 25.0 panvk default only on v10: https://test.www.collabora.com/news-and-blog/news-and-events/mesa-25-panvk-moves-towards-production-quality.html
- Panfrost GLES conformance on G52: https://nxr.netbsd.org/raw/xsrc/external/mit/MesaLib/dist/docs/drivers/panfrost.rst
- GTK 4.16 Vulkan default on Wayland: https://download.gnome.org/sources/gtk/4.16/gtk-4.16.0.news

Toolchain
- GCC 14 kernel vs GCC 13 userspace (`-fmin-function-alignment=8`): https://forum.armbian.com/profile/211944-li20034/content/

Claude Code
- System requirements (4 GB+ RAM, ARM64): https://code.claude.com/docs/en/setup
- Environment variables (`BASH_*_TIMEOUT_MS`, `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS`): https://code.claude.com/docs/pt/env-vars
- Hooks (exit code 2 blocks a PreToolUse call): https://docs.anthropic.com/en/docs/claude-code/hooks
- CLAUDE.md imports: https://code.claude.com/docs/en/memory
- Permission rule syntax: https://www.developersdigest.tech/blog/claude-code-permissions-settings-guide
