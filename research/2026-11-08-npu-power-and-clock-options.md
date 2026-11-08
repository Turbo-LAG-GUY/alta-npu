# NPU power and clock options: what we tested, what we chose (2026-11-08)

**Decision (Jordan, 2026-11-08):** keep the NPU powered from boot, as Amlogic's own driver does. This is the default
of the driver installed by root step 110 (`keep_powered=1`), so nothing needs to change. Power-down-when-idle works
too, but it is set aside. This file records how to bring it back.

## 1. Power down when idle (tested, shelved)

### What it is
The driver from root step 110 can also let the NPU power off when it's idle, as the old driver did 200 ms after the
last job. The difference is how the NPU comes back:
- **Old driver:** power returned with no reset. That froze the board in 3 of 15 power-ups at a program start.
- **New driver:** the reset is held while the power is off. At the next job it powers up still in reset, waits
  10 ms, releases it, waits 20 ms, then soft-resets before use, following Amlogic's _Reset. See
  patches/etnaviv-reset-keepon/0004 and research/2026-11-07-first-job-freeze.md.

### Test results (boot a31f2917, all with no kernel messages and a clean 5-minute watch)
| Test | Power-ups | Result |
|---|---|---|
| Wake-up test, one program (runs/20261108-011958-WAKE) | 50 (1 program start + 49 within the program) | PASS 50/50, all correct, 41.3 ms per wake-up |
| T3 in power-down mode (runs/20261108-013850-T3) | 20 program starts, each from fully off | PASS 20/20, first job 38.5-45.2 ms, then 7.3-8.4 ms |
That is 70 clean power-ups in all, against the old driver's 3 freezes in 15. Fisher's exact test (one-sided) gives
p = 0.005 counting all 70, and p = 0.064 counting the 21 program starts only.

### Trade-offs (why it is shelved)
- Each wake-up adds ~33 ms to the first job after a pause longer than 0.2 s. About 30 ms of that is the reset
  waits.
- The power it saves is unknown. The board has no power meter, and idle temperatures look the same in both modes
  (~32-37 C).
- Kept powered has more evidence: 53 clean program starts, T4 soak and T5 desktop, and it is Amlogic's way.
- Not tested in this mode: long runs over hours, and the T4 soak.

### How to bring it back
- **For one boot:** `sudo bash ~/alta-npu/root-steps/120-npu-power-down-when-idle.sh`. Any reboot undoes it.
- **Permanently (no root step written yet):** add `options etnaviv keep_powered=0` to a file in /etc/modprobe.d/
  (as a root step), then reboot. The boot log line then reads "alta: reset on every power-up, runtime PM
  (keep_powered=0)", and power/control reads "auto".
- **Optional:** raise the idle delay before power-down, for example to 5 s, so a burst of jobs pays the wake-up only
  once: `echo 5000 | sudo tee /sys/devices/platform/ff100000.npu/power/autosuspend_delay_ms` (per boot). Not
  tested.
- **Before relying on it:** run T4 and a longer soak in this mode.

## 2. NPU clock frequency (looked at, left alone)
- **Speed now:** both NPU clocks (core and bus) run at 800 MHz, set by the device tree (`assigned-clock-rates`).
  Nothing changes them at runtime: the NPU's cooling device (cooling_device2) sits at state 0, and no thermal zone
  uses it.
- **Amlogic's driver (galcore, khadas-vim3-pie):** it sets both clocks to `MAX_NANOQ_FREQ` = 800 MHz once at
  power-up and never changes them. Its automatic speed scaling is compiled out (`gcdDVFS 0`). It keeps a manual
  throttle (`gcdENABLE_FSCALE_VAL_ADJUST 1`): the core's built-in clock divider, 1-64 of 64, which defaults to
  64 (full speed).
- **Faster isn't available.** The NPU clock's sources are xtal, gp0_pll, hifi_pll, fclk_div2p5 (800 MHz),
  fclk_div3, div4, div5 and div7 (drivers/clk/meson/g12a.c, nna_clk_parent_data). 800 MHz is the fastest fixed
  source. The PLLs serve other hardware, and the NPU's voltage is fixed (no regulator in the device tree).
- **Slower is possible but not worth it:**
  - etnaviv's only knob is cooling_device2/cur_state (0-6, halving the core clock each step). It is applied with
    clk_set_rate, because this core reports DYNAMIC_FREQUENCY_SCALING.
  - The NPU clock is a single mux/div/gate chain, with no glitch-free double branch like the Mali GPU's
    mali_0/mali_1. A rate change on a running NPU can glitch.
  - With the voltage fixed, a lower clock saves little. An idle NPU already gates most of its clocks.
  - etnaviv has no devfreq, so nothing would scale it automatically.
  - Upstream quirk: after a soft reset, etnaviv_hw_reset also applies the internal divider (FSCALE) from freq_scale.
    On this core that combines with the clk_set_rate, so states above 0 would run slower than intended after a
    reset.
- **If we ever revisit:** the safe route is the core's built-in divider (FSCALE), which Amlogic's driver uses and
  which is designed for runtime changes. That needs a small etnaviv change, since etnaviv uses the external clock
  on this core.
