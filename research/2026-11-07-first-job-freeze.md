# Research 2026-11-07: why the NPU freezes the board at a program's first job

## The symptom (from our runs)
3 freezes in 15 program starts (T2b, T3 run 11, WAKE-01), every one at the program's first NPU job, with the NPU
powered down just before. 0 freezes in ~2,040 later jobs (T4: 2,000 in a row). Nothing reaches the kernel log;
the hardware watchdog resets the board. Details: incidents/2026-11-07-1405-T2b, -1812-T3-11, -2104-WAKE-01.

## What happens at every NPU power-up on this kernel (6.12.73 Libre Computer tree, same as mainline 7.4-rc1)
1. genpd powers the NNA domain on (drivers/pmdomain/amlogic/meson-ee-pwrc.c, meson_ee_pwrc_on): clear the sleep bits,
   udelay(20), memories on, udelay(20), then reset assert / iso off / reset deassert. But the NNA domain is a TOP_PD
   with **no reset attached**, so those reset calls do nothing. Then clocks.
2. etnaviv_gpu_rpm_resume -> etnaviv_gpu_clk_enable -> etnaviv_gpu_hw_resume -> etnaviv_gpu_hw_init: clock gating
   (incl. our SH_EU bit), AXI cache config, pulse eater, interrupts. **No reset**: no reset-line pulse, and no
   etnaviv_hw_reset (soft reset). The NPU's reset line (DT: resets = <&reset RESET_NNA>) is deasserted once at
   driver init and never pulsed again. Soft resets happen only at init and on hang recovery.
3. First submit -> start_fe_idleloop: MMU restore, FE start, the job runs.
So after each power-off, the first job starts on an NPU whose internal state came back from a power cycle without
a reset. The driver powers the domain down 200 ms after every job (autosuspend), so this happens on every
"burst", including the first job after boot (the NPU is powered down right after probe).

## What Amlogic's own driver does (galcore, khadas/android_vendor_amlogic_npu, platform/amlogic)
- Platform ops are only adjustParam, getPower and reset: **no setPower**, so it never cuts the NNA domain during
  normal use. getPower powers it on once at probe (sleep off, iso off, mem on, set clocks).
- `_Reset` power-cycles with the reset **held asserted**: RESET_LEVEL2 bit 12 low, mem off, iso on, sleep on,
  mdelay(100), sleep off, iso off, mem on, **mdelay(10)**, release reset, **mdelay(20)**.
- So the vendor never does what upstream does: power on without a reset, with only ~20 us of settling.

## Upstream fixes since 6.12, and which ones this kernel lacks
| commit | what | in 6.12.73 tree? |
|---|---|---|
| a042beac6e6f (Tomeu Vizoso, 2025-10) | fix flush sequence logic (missed MMU flushes on address-space switch) | yes (stable backport) |
| 4f8dbadef085 (Lucas Stach, 2024-10) | flush shader L1 cache after user commandstream | yes (stable backport) |
| c182a2ae1388 (Randolph Sapp, TI, 2025-10) | force flush (read back) on power register ops; races found experimentally | **no** |
| 1006d1f7c03d, 4720915ab482 (Lucas Stach, 2026-05) | after reset, check all modules idle and MMUv2 disabled | **no** |
| 85ba57ad88cf + 4 (Gert Wollny, 2025-11) | PPU flop reset at FE start (etnaviv_buffer_init); only for 0x8000 r0x6205, `force_flop_reset=1` for others with PIPE_3D (0x7120 qualifies) | **no** |
| (none) | SH_EU clock gating for 0x8000 r0x7120 | ours only (mainline has 0x7200/0x8002/0x9200) |
| (none) | reset of the NPU on runtime resume | **nowhere, not even mainline** |
Also checked: the NNA memory power-down masks (GENMASK(31,0) x2, the 2023 "8th NN core hangs" fix) are present.
The pwrc changes since 6.12 are only `const`.

## Hypothesis
The freezes come from powering the NNA domain back on without a reset (and with very short settling) before the
first job. That would also explain why freezes sit only at first jobs and come intermittently.
Precondition still to confirm: the NNA domain really powers off while the NPU is idle
(`sudo cat /sys/kernel/debug/pm_genpd/pm_genpd_summary`; look for NNA "off").

## Candidate fix (not written yet)
In the out-of-tree etnaviv: on runtime resume, before hw_init, pulse the NPU reset like the vendor (assert, wait,
release, wait ~20 ms) and do a soft reset (etnaviv_hw_reset). Optionally add the upstream c182a2ae1388 and the two
reset-check commits. Build with the kit, install as a root step, reboot, then validate with the wake-up test /
T3 (many power-ups). Cost: ~30 ms more per wake-up.
The vendor-style alternative is not power-cycling the domain at all (keeping the NPU powered). Jordan has ruled that
out for now.

## Sources
- Flop reset: https://www.phoronix.com/news/Etnaviv-PPU-Flop-Reset , https://lwn.net/Articles/1047239/ ,
  https://lkml.org/lkml/2025/11/10/1137 , https://lkml.org/lkml/2026/1/6/1141
- Optional reset support: https://lists.freedesktop.org/archives/etnaviv/2024-December/005079.html
- SH_EU on VIPNano-Si+: https://lists.freedesktop.org/archives/etnaviv/2024-January/004684.html
- NNA power domain: https://lkml.iu.edu/hypermail/linux/kernel/2212.0/01864.html ; mask fix:
  https://lkml.iu.edu/hypermail/linux/kernel/2310.2/00205.html
- Mainline sources (7.4-rc1) and commits: https://github.com/torvalds/linux (drivers/gpu/drm/etnaviv,
  drivers/pmdomain/amlogic/meson-ee-pwrc.c); copies in the Claude scratchpad (lost on reboot)
- Vendor driver: https://github.com/khadas/android_vendor_amlogic_npu (branch khadas-vim3-pie,
  hal/os/linux/kernel/platform/amlogic/gc_hal_kernel_platform_amlogic.c)
- Background: https://blog.tomeuvizoso.net/2023/06/etnaviv-npu-update-2-diving-into.html

## Update 2026-11-07 ~23:10: precondition confirmed, fix built (not installed)
- Jordan ran `sudo cat /sys/kernel/debug/pm_genpd/pm_genpd_summary`: `NNA off-0` with `ff100000.npu suspended`.
  The domain really is cut while idle, so every first job runs on an NPU that powered up without a reset.
- One more concrete way this can hang the bus: this NPU runs the MMU in secure mode (minor_features7 BIT_SECURITY
  and minor_features10 SECURITY_AHB are set, so sec_mode = ETNA_SEC_KERNEL). `etnaviv_iommuv2_restore_sec` skips
  all MMU programming when `VIVS_MMUv2_SEC_CONTROL` already reads "enabled" ("the state is still there"). After an
  unreset power-up that bit is whatever the flop came up as. If it reads 1, the FE starts fetching through page-table
  pointers that were never written.
- Jordan then chose to keep the NPU powered, as Amlogic's driver does, and asked to include the upstream patches.
- Built (sha256 1b277d6c...): lc-kernel-src HEAD + patches/etnaviv-reset-keepon/0001-0004:
  0001 c182a2ae1388 (read back power-register writes), 0002 1006d1f7c03d (after reset, all modules idle),
  0003 4720915ab482 (after reset, MMUv2 disabled; context adapted, 6.12 lacks ca0593a29e9e),
  0004 alta: keep_powered=1 (pm_runtime_forbid after init, so the only power-up is the one at init, with the reset
  held); if it does power-cycle (power/control=auto), assert the reset before power-off, and on resume release it
  (10 ms / 20 ms like Amlogic) and soft-reset before hw_init; marker "alta: reset on every power-up, ..." at init.
- Kernel logs from 43 boots: no "not yet idle", "timed out waiting for idle" or "failed to reset" ever; the 5 hang
  recoveries (soft reset) all succeeded. So the stricter reset checks should pass on this NPU.
- Not included: the PPU flop reset (85ba57ad88cf + 4). It only applies to 0x7120 when forced, and it is untested on
  this core.
- Install: root-steps/110 (Jordan), then reboot. Validation: T3 (20 separate program starts, each with no power-up
  now). At the old rate of 3 freezes in 15 starts, 20 clean starts would happen by luck only ~1% of the time (0.8^20).

## Result 2026-11-07 23:43: T3 PASS 20/20 on the new build
Root step 110 installed the build (Jordan, 23:25), then Jordan rebooted (boot 211472be). The NPU has stayed powered
since init (58 ms suspended in total), so none of the 20 program starts powered it up. All 20 runs passed (top-1 653,
5/5 shared, median 7.09-7.74 ms). No etnaviv kernel lines appeared after init, and the 5-minute post-watch was clean.
The old build froze in 3 of 15 program starts, each one at a power-up. If the old 20% per-start rate were exact, a
lucky 20/20 would have a 1.2% chance. Because that rate was measured on only 15 starts, Fisher's exact test (3/15 vs
0/20, one-sided) gives p = 0.07. Strong evidence, not proof. One more clean T3 brings it to p = 0.017.
Not tested: the reset-on-resume path. It only runs with power/control=auto.

## Result 2026-11-08 00:10: second complete T3 PASS 20/20
T3 #2 was cut off at run 14 by an accidental power cut (per Jordan). The cut came during run 14's graph compile,
before any of that run's NPU jobs, and runs 01-13 had passed. T3 #3 (boot 7111797a) then passed 20/20 with a clean
post-watch and no etnaviv kernel lines. Peak temps were cpu 43 C and ddr 43 C. The new build now has 53 clean program
starts in a row (20 + 13 + 20), against 3 freezes in 15 on the old build: Fisher's exact test, one-sided, p = 0.009
(0.017 counting the two complete T3s only). Keeping the NPU powered with a proper reset at init evidently removes the
first-job freezes. Which part of the fix matters can't be separated, since no power-ups happen now. The
reset-on-resume path (power/control=auto) remains untested.

## 2026-11-08 00:24: T4 PASS on the new build
2000 invokes in one process, median 8.02 ms (old build 7.62 ms; the patches do not touch the per-job path), p99 12.66 ms,
no kernel messages, clean post-watch, peak temps cpu 43 C / ddr 45 C.

## 2026-11-08: T5 PASS, ladder complete
Jordan: the desktop stayed responsive through all the runs. On the new build T0, T3 (x2), T4 and T5 all pass.

## Result 2026-11-08 01:28: the reset at power-up works (wake-up test PASS 50/50)
Root step 120 (boot a31f2917) set power/control=auto, so the NPU powered down when idle again. The wake-up test then
woke it from power-down 50 times (1 new program + 49 in-program). Every wake-up went through the new path: reset
held across the power cycle, released 10 ms after power-up, 20 ms wait, soft reset, init. All 50 classifications
were correct and no kernel messages appeared. A wake-up costs 41.3 ms (median) against ~8 ms for a classification
on a powered NPU, so the reset waits add ~30 ms. The old build froze at the first of these power-ups (WAKE-01), and
in 3 of 15 power-ups at program starts. Fisher's exact test (3/15 vs 0/50, one-sided) gives p = 0.010. So both
modes now pass: kept powered (53 clean program starts, T4, T5) and power down when idle (50 clean power-ups).
Power-ups at program starts in auto mode were exercised only once here; a T3 in auto mode would test them directly.

## Result 2026-11-08 01:45: T3 in power-down mode PASS 20/20
All 20 program starts began with the NPU powered down (status before invoke 1 = suspended, 20/20), so each was a
program-start power-up through the new reset path, the exact case that froze the old build (3 of 15). The first
invoke took 38.5-45.2 ms (median 41.4), later invokes 7.3-8.4 ms. No kernel messages, clean post-watch. Power-down
mode total: 70 clean power-ups (21 at program starts, 49 in-program). Fisher's exact test (one-sided) gives
p = 0.005 counting all power-ups against the old 3/15, and p = 0.064 counting program starts only.
