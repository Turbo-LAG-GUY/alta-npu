# Incident 2026-11-07 14:05 - T2b hard freeze

- Test: T2b, one npu-run job: kit npu-classify.py, Mesa-docs MobileNet v1 (sha256 1ce11b7e1895...), grace_hopper.bmp,
  -n 5, TEFLON_DEBUG=verbose. Kernel 6.12.73-g6b2d05eda916, patched etnaviv (sha256 5717091b...), Mesa 26.1.0-devel c5423d3e273.
- Board: fresh boot (13:56), STOP cleared by Jordan, T0 clean (0 etnaviv errors, NPU suspended, 37-40 C).
  Indicator (tray) and npu-observe were running.
- Job start 14:05:05 (uptime ~537.5 s). Last npu-observe sample 14:05:06 (up=538.56): npu=suspended, 0 NPU IRQs,
  CmaFree 20 kB (0-88 kB since 14:04:51). stderr.txt ends mid-way through Teflon's "compiling graph" tensor
  table, so compile had just begun at the last sync.
- Freeze: between ~538.6 s and ~541.7 s uptime (no further samples), i.e. 1-4 s into the job, around the end of
  graph compile / first NPU submit. Journal for boot -1 ends at 533.6 s with nothing relevant; no kernel message survived.
  No shutdown sequence logged: Jordan power-cycled; new boot 14:06:43.
- npu-run could not set STOP (machine dead), so it was set afterwards by Claude.

Context: the only other runs today that made the NPU really compute (01:17, Mesa-docs model, ~360 inferences,
completed) was also followed by a hard freeze ~1 min later. The 12:01 coral-model run MMU-faulted
on every invoke (NPU barely ran) and did not freeze. CmaFree was 0-2220 kB during that run.
Jordan's own earlier successful run used the same venv, delegate and model file via Mesa's classification.py
(test.bmp, default threads, no TEFLON_DEBUG) - not hardware-relevant differences.
