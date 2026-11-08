# Alta NPU

Getting the NPU in the **Libre Computer AML-A311D-CC "Alta"** (Amlogic A311D) to run TensorFlow Lite models reliably on
a fully open-source stack (the etnaviv kernel driver and Mesa's Teflon delegate), and finding and fixing the hard
lockups that got in the way.

| | |
|---|---|
| **Status** | Working. MobileNet V1 classifies an image on the NPU in **about 8 ms**, against about 39 ms on the CPU, with the same answer as the CPU every time. |
| **The bug** | The board froze solid at a program's first NPU job, in 3 of 15 program starts. |
| **The cause** | Linux switched the NPU's power off whenever it sat idle, and back on with no reset. |
| **The fix** | Keep the NPU powered, as Amlogic's own driver does, and reset it properly on any power-up ([patches/etnaviv-reset-keepon](patches/etnaviv-reset-keepon)). |
| **The evidence** | 0 freezes in 53 program starts plus a 2,000-job soak, against 3 in 15 before (Fisher's exact test, p = 0.009). |
| **Upstream** | An RFC patch for mainline Linux is drafted in [upstream/](upstream/etnaviv-reset-runtime-pm) and not sent yet. |

## Hardware and software

- **Board:** Libre Computer AML-A311D-CC "Alta": Amlogic A311D (4× Cortex-A73 at 2.2 GHz + 2× Cortex-A53 at 1.8 GHz),
  Mali-G52 MP4 GPU, 4 GB LPDDR4.
- **NPU:** VeriSilicon VIPNano-QI, which etnaviv reports as GC8000 revision 0x7120. It sits in its own power
  domain, NNA, and runs at 800 MHz (already its fastest clock source).
- **OS:** Ubuntu 24.04 with Libre Computer's 6.12.73 kernel (`6.12.73-g6b2d05eda916`) and a GNOME desktop.
- **Inference stack:** etnaviv (kernel) → Mesa's Teflon TensorFlow Lite delegate (Mesa main at `c5423d3e273`,
  26.1.0-devel) → LiteRT (`ai_edge_litert`) in Python.
- **Test job:** quantized MobileNet V1 (224×224) classifying `grace_hopper.bmp`. The right answer is class 653,
  "military uniform".

## What went wrong

Earlier work in April 2026 got models running on the NPU, with a patched etnaviv that includes a clock-gating fix for
this core revision. But the board still hard-froze during NPU work. In November it was worked through methodically,
with a test kit that makes every NPU job crash-safe ([How the work was done](#how-the-work-was-done)).

1. **The freezes had a pattern.** There were three hard freezes ([T2b](incidents/2026-11-07-1405-T2b),
   [T3 run 11](incidents/2026-11-07-1812-T3-11), [WAKE-01](incidents/2026-11-07-2104-WAKE-01)). Each one came at a
   program's first NPU job, with the NPU powered down just before. There were none in about 2,000 jobs sent while
   it was already powered. Nothing ever reached the kernel log, and the board needed a reboot each time (by the
   hardware watchdog, once that was set up).
2. **The NPU really was being power-cycled.** etnaviv suspends the NPU 200 ms after its last job, and the power-domain
   driver then cuts the NNA domain (`pm_genpd_summary` showed `NNA off-0`).
3. **It came back up with no reset.** The NNA domain has no reset of its own, and etnaviv releases the NPU's reset
   line only once, at driver start-up. On the next job it just reprograms a few registers, so the NPU starts from
   whatever state its logic happened to power up in.
4. **Amlogic's own driver never does that.** Its galcore driver
   ([khadas/android_vendor_amlogic_npu](https://github.com/khadas/android_vendor_amlogic_npu)) never turns the NPU's
   power off in normal use. When its reset path does cycle the power, it holds the reset throughout, waits 10 ms after
   power-up, releases it, and waits another 20 ms.

The leading theory for why this hangs the whole chip: this NPU runs its MMU in secure mode, and etnaviv skips
loading the NPU's page tables if the MMU already reads as switched on. After a power-up with no reset, that reading
is down to chance, and a wrong one would send the NPU to garbage memory addresses. It is only a theory: the lockup
itself was never captured. Full analysis: [research/2026-11-07-first-job-freeze.md](research/2026-11-07-first-job-freeze.md).

## The fix

| | Stock etnaviv | Patched: default | Patched: power-down mode |
|---|---|---|---|
| **NPU idle for 200 ms** | Clocks off, power cut | Stays powered | Clocks off, **reset held**, power cut |
| **Next job** | Power on, clocks on, set up registers | Runs straight away | Power on (still in reset), clocks on, wait 10 ms, release the reset, wait 20 ms, **soft reset**, set up registers |
| **Cost** | | Nothing per job | About 33 ms per wake-up, 30 ms of it the reset waits |

The patched module is the April base plus four patches:

- **[patches/base-6.12.73-lc](patches/base-6.12.73-lc)**, from April 2026: the SH_EU clock-gating fix for revision
  0x7120, and a backport of mainline's optional reset support (6bde14ba5f7e).
- **[patches/etnaviv-reset-keepon](patches/etnaviv-reset-keepon)**, the fix:
  - `0001`: mainline c182a2ae1388. Power-register writes are read back, so each one lands before the next step.
  - `0002`: mainline 1006d1f7c03d. After a soft reset, check that every module is idle.
  - `0003`: mainline 4720915ab482. After a soft reset, check that the MMU is off.
  - `0004`: the Alta change. It adds the `keep_powered` option (on by default) and holds the reset across any
    power-down. On power-up it releases the reset with Amlogic's waits and soft-resets the NPU. It also prints a
    boot-log marker (`alta: reset on every power-up, ...`) that proves the right module is loaded.

[root-steps/110](root-steps/110-install-etnaviv-reset-keepon.sh) installs it under `/lib/modules/.../updates/alta/`
and leaves the previous module in place as a fallback. Build record and checksums:
[BUILDINFO.txt](build/etnaviv-6.12.73-g6b2d05eda916-20261107-230436-reset-keepon/BUILDINFO.txt).

## Results

In every test, each classification is checked against a CPU run: same top-1 class, and at least 4 of the top 5
shared. A test also requires no kernel messages, and a clean 5-minute watch afterwards. Every run is in
[RESULTS.md](RESULTS.md), with its log directory under [runs/](runs).

| Test | What it does | Old driver | Patched driver |
|---|---|---|---|
| **T3** | 20 separate program starts | FAIL: froze at run 11 | **PASS 20/20, twice**, plus 13/13 in a run cut short by a power cut |
| **T4** | 2,000 classifications in one program | PASS, 7.62 ms median | **PASS**, 8.02 ms median |
| **T5** | NPU jobs with the GNOME desktop up | not run | **PASS** (the desktop stayed responsive) |
| **WAKE** | 50 classifications, each after ≥ 10 s powered down | FAIL: froze at the first | **PASS 50/50** (power-down mode) |
| **T3, power-down mode** | 20 program starts, each from a powered-down NPU | not run | **PASS 20/20** |

- **The statistics:** the old driver froze in 3 of 15 program starts.
  - Kept powered, the patched driver has 53 clean starts: p = 0.009 (Fisher's exact test, one-sided).
  - In power-down mode, it has 70 clean power-ups: p = 0.005, or p = 0.064 counting only the 21 that were program
    starts.
- **Speed:** about 7-8 ms per classification on the NPU, against 39 ms on the CPU (XNNPACK, 2 threads), so about 5×
  faster. A program's first job also compiles the graph, which takes about 1.2 s.
- **The choice:** keep the NPU powered, as Amlogic does. Power-down mode works but is shelved, and the clock was left
  at 800 MHz. Both decisions, and how to bring power-down mode back, are in
  [research/2026-11-08-npu-power-and-clock-options.md](research/2026-11-08-npu-power-and-clock-options.md).

## How the work was done

A board with 4 GB of RAM and a habit of locking up calls for care, so all of this followed a rulebook,
[ALTA_NPU_HANDOFF.md](ALTA_NPU_HANDOFF.md) (rules R1-R12, test protocol T0-T5):

- **One guarded NPU job at a time.** Every NPU job runs through [scripts/npu-run](scripts/npu-run): a global lock, a
  hard deadline and a kernel-log check afterwards. Any error sets a STOP latch that refuses further NPU work until
  a person has looked.
- **Evidence that survives a freeze.** [scripts/npu-attempt](scripts/npu-attempt) logs a STARTING line to
  [PROGRESS.md](PROGRESS.md) and forces it to disk. It also keeps a live copy of the kernel log
  ([kmsg-catch](scripts/kmsg-catch)) and samples the NPU every second ([npu-observe](scripts/npu-observe)).
- **Long tests run detached**, as systemd user units ([npu-detached](scripts/npu-detached)), so they can't die with the
  terminal session that started them.
- **A hardware watchdog** ([root step 090](root-steps/090-hardware-watchdog.sh)) reboots the board about a minute
  after a total freeze.
- **No direct root changes.** Each one is a script in [root-steps/](root-steps), saying what it does, why, and how to
  verify and revert it. Jordan reviewed each one and ran it with sudo.
- **Guard rails for the AI assistant.** It may not reboot, load or unload modules, or replay udev events. A hook,
  [.claude/hooks/alta-guard.sh](.claude/hooks/alta-guard.sh), blocks the worst of these.
- **Everything is written down.** System changes go in [CHANGELOG.md](CHANGELOG.md), tests in
  [RESULTS.md](RESULTS.md), and every step in [PROGRESS.md](PROGRESS.md). Freezes get their own notes in
  [incidents/](incidents).

## Repository layout

| Path | What it is |
|---|---|
| [ALTA_NPU_HANDOFF.md](ALTA_NPU_HANDOFF.md) | The rulebook: rules, procedures, test protocol, and the original kit in Appendix A |
| [Alta-NPU-findings-summary.md](Alta-NPU-findings-summary.md) | The short version of the handoff |
| [research/](research) | Root-cause analysis; power and clock options |
| [patches/](patches) | Kernel patches (base and fix), plus a one-line Mesa change ([patches/mesa](patches/mesa)) |
| [upstream/](upstream/etnaviv-reset-runtime-pm) | The RFC patch for mainline Linux, and how to send it |
| [scripts/](scripts) | The test kit: npu-run, npu-attempt, the T3/T4/WAKE test drivers, audit, out-of-tree build, progress log |
| [root-steps/](root-steps) | Every root change, as a reviewed script |
| [system/](system) | Config files the root steps install (udev rule, modprobe options, watchdog, service) |
| [runs/](runs), [results/](results) | Per-job records (command, output, kernel log, step markers, samples) and JSON results |
| [incidents/](incidents) | The five incidents, with notes and captures |
| [build/](build) | The record of the installed build, and its combined diff |
| [indicator/](indicator) | `npu_status.py`: a GNOME top-bar indicator and terminal view of what the NPU is doing |
| [demo/](demo) | `classify.py`: the first standalone NPU demo |
| [CLAUDE.md](CLAUDE.md), [.claude/](.claude) | Claude Code's project instructions, permissions and guard hook |
| [PROGRESS.md](PROGRESS.md), [RESULTS.md](RESULTS.md), [CHANGELOG.md](CHANGELOG.md), [templates/](templates) | The records |

## Using it on your own Alta

The kit was written for one board: Libre Computer's `6.12.73-g6b2d05eda916` kernel, with a few paths under
`/home/jordan` (`.alta-env`, `scripts/npu-detached`, `scripts/npu-t3-detached`). Read the rulebook and each script
before running it. The root steps change the system.

1. **Kernel module.** Take Libre Computer's kernel source at `6b2d05eda916`
   ([libretech-linux](https://github.com/libre-computer-project/libretech-linux)) and apply
   `patches/base-6.12.73-lc`. Then build etnaviv out of tree with the fix:
   `scripts/build-etnaviv-oot.sh <source tree> patches/etnaviv-reset-keepon/*.patch`. This needs GCC 14, to match
   the kernel, and builds at `-j2`. The script installs nothing.
   - **Install** with `sudo bash root-steps/060-install-etnaviv-oot.sh build/<build dir>`, then reboot.
     [root-steps/110](root-steps/110-install-etnaviv-reset-keepon.sh) is the same install, pinned to this board's
     build and checksum.
   - **Check** that the boot log says `alta: reset on every power-up, kept powered (keep_powered=1)`.
2. **Teflon.** Build Mesa with etnaviv and Teflon as in
   [Mesa's Teflon docs](https://docs.mesa3d.org/teflon.html), and don't install it over the system Mesa. Point
   `TEFLON` in `.alta-env` at `build/src/gallium/targets/teflon/libteflon.so`.
3. **Python.** Make a venv with `ai_edge_litert`, `numpy` and `pillow`, and point `PY` in `.alta-env` at it.
4. **Test assets** in `models/` (not included; see the table below).
5. **First NPU run:**
   ```bash
   . ./.alta-env && scripts/npu-run -t 120 -n T2 -- "$PY" scripts/npu-classify.py \
       -m models/mobilenet_v1_mesa.tflite -i "$I" -l "$L" -e "$TEFLON" -n 5
   ```
   After that come the test drivers: `scripts/npu-t3` (20 program starts), `scripts/npu-t4` (soak) and
   `scripts/npu-wake` (wake-ups).

| Test asset | sha256 | Where from |
|---|---|---|
| `models/mobilenet_v1_mesa.tflite` | `1ce11b7e1895e31bf14ba33f57845d16e169277058eaa85c99edecaea418666a` | The MobileNet V1 Jordan used with Mesa's `classification.py` in April; where it was downloaded from wasn't recorded. Every test from T1b on uses it. The TensorFlow archive linked from Mesa's docs contains a different build (`ecc3a67c47c5…`), untested here. |
| `models/mobilenet_v1_1.0_224_quant.tflite` | `2d0ebfa2e75ae93c709e9ecc27570c9c0e49b7780733a840f82589fc1272fc3d` | [google-coral/test_data](https://github.com/google-coral/test_data). With this Teflon build it MMU-faults the NPU ([incident T2](incidents/2026-11-07-1201-T2)), so don't use it. |
| `models/grace_hopper.bmp` | `c43444bf3b1b1e3080551ca194bd231da4b67f20fc91454502915a6b806b614f` | google-coral/test_data |
| `models/imagenet_labels.txt` | `b8aaeb5630c62626a7d124a05c0e14230a7ffdd2136a31d0190a4fb71c1d82f1` | google-coral/test_data |

## Upstream

- **The fix:** [upstream/etnaviv-reset-runtime-pm](upstream/etnaviv-reset-runtime-pm) is an RFC patch,
  "drm/etnaviv: keep the core in reset across runtime power-down", against v7.4-rc1. Mainline has the same gap on
  every A311D board. It is drafted but not sent: Jordan will review it, sign it off and send it.
- **Already in mainline:** patches 0001-0003 and the optional reset support.
- **Not submitted:** the SH_EU clock-gating fix for revision 0x7120.

## What's not in this repo

- **`models/`:** third-party files. The sources and checksums are in the table above.
- **`audit/`:** full machine inventories from `scripts/alta-audit.sh`.
- **Whole-boot logs:** kernel logs and journals from the incidents. Each incident keeps its notes, summaries and
  per-job captures.
- **Large or generated files:** tray-indicator recordings (`*indicator-watch.txt`), build outputs (`etnaviv.ko`,
  object files) and the copied kernel sources.
- **The Mesa checkout** (Mesa main at `c5423d3e273`, plus the one-line change in [patches/mesa](patches/mesa)) and the
  Python venv.
- **Copies, not originals:** `indicator/npu_status.py` and `demo/classify.py` are copies. The copies that actually
  run live in `~/Documents/AI_Projects`.

## Credits

- **etnaviv:** Lucas Stach, Christian Gmeiner, Russell King and the other etnaviv developers. Tomeu Vizoso brought up
  this NPU in etnaviv and wrote Teflon. Philippe Lecointre wrote the optional reset support, and Randolph Sapp the
  power-register fix.
- **Amlogic's galcore driver** was the reference for the reset sequence.
- **This work:** Jordan ([Turbo-LAG-GUY](https://github.com/Turbo-LAG-GUY)) with Claude Code (Anthropic's Claude Opus
  5.5) running on the board. Claude wrote most of the code and documents here, including the analysis, the patches,
  the scripts and the notes, and ran the tests through the kit. Jordan approved every NPU test, ran every root step
  and made the calls.

## License

The kernel patches (`patches/base-6.12.73-lc`, `patches/etnaviv-reset-keepon`, `upstream/`) modify the Linux kernel
and are GPL-2.0, like the code they change. The rest of this repository has no license yet.
