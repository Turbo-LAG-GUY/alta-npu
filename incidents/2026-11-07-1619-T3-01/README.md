# Incident 2026-11-07 16:19 - T3-01 interrupted (NOT a freeze)

- Test: T3 (20 separate NPU runs) started 16:19:31 (boot b11450ee, uptime 1593.6 s) as a background job of the
  Claude Code session, with hw_job_limit=1 and the watchdog armed. Tray indicator plus `--watch` recorder running.
- T3-01 step markers (fsync'd): delegate load, Interpreter() (graph compile 1.25 s) and allocate_tensors all done;
  last marker `invoke 1/5: begin` at 1600.97 s, NPU still suspended. No `done` marker.
- Every writer of the job stopped together: npu-observe (last sample 1600.32 s), kmsg-catch (no end line, so it was
  killed rather than exiting) and the classifier. The kernel logged NOTHING: no etnaviv message, no hung task,
  no OOM, no fault.
- The system kept running (cron, WiFi, PackageKit ...). At 1828-1865 s Jordan ran, on pts/0: sudo rm
  /etc/apt/sources.list.d/coral-edgetpu.list; apt update; apt upgrade -y; sudo reboot. The reboot was a normal
  shutdown (snapd etc. stopped). The watchdog never fired.
- Clock: NTP was turned on at 1457 s (polkit via pkttyagent, timesyncd restart). It synced over IPv4 to "Thu
  2026-11-05 15:09:39 EST", 2 days wrong. Between 1552 s (a pkexec session) and 1593 s the clock was set back to
  Nov 7 ~16:19. Between 1600.32 s and 1629.5 s it silently stepped back to the Nov 5 time. This boot (c40e6c74)
  synced over IPv6 to the correct Sat 2026-11-07.
- Reading: the T3 job (and the NPU process inside invoke 1) were killed when the Claude Code session ended; the
  background-task notice confirms it "didn't finish before the previous session ended". That happened together
  with the ~2-day clock step. Whether invoke 1 itself completed is unknown. Verdict: INTERRUPTED / inconclusive.
- Lessons: run long NPU tests detached from the Claude Code session (systemd-run --user), and don't rely on
  the wall clock in the tools (npu-observe fixed to use uptime; npu-run's deadline and its
  `journalctl --since @epoch` check are wall-clock based - proposed fix: uptime + journal cursor).
