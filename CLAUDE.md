# Alta (AML-A311D-CC) NPU project

@ALTA_NPU_HANDOFF.md

## Non-negotiables (full text: handoff section 1)
- First action in a new session: `bash scripts/alta-audit.sh`, then report the findings before changing anything.
- Never `rmmod`, `modprobe -r`, `insmod`, `udevadm trigger`, reboot, or touch /boot, /lib/modules, initramfs, DT.
- Root work = write a script in `root-steps/` (what, why, revert) for Jordan to review and run with sudo.
- Builds: at most `-j2`, one heavy job at a time. Never `ninja install` Mesa over the system Mesa.
- NPU jobs only through `scripts/npu-run`. If it sets `runs/STOP`, stop all NPU work and report.
- Log every system change in CHANGELOG.md and every test in RESULTS.md.
