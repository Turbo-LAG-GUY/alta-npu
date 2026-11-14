# Incident 2026-11-13 05:35 - SHADER-HELLO (first program on the NPU shader core)

- Job: npu-attempt SHADER-HELLO = ~/Documents/AI_Projects/npu-shader/hello -n 256 -w 64 (Jordan's OK: "Start step 1 then try step 2").
  4-instruction copy kernel built only from Mesa-verified ISA words (round trip through Mesa's decoder passed); one submit;
  compute dispatch modelled on the 2023 etnaviv-opencl branch (refs/remotes/cl/tomeu-etnaviv-opencl in ~/Documents/AI_Projects/mesa).
- Result: NPU MMU fault (page not present) during the job, then "recover hung GPU!"; npu-run rc=90, STOP latch set.
- Board: stayed up; NPU recovered and is active. No other jobs were running; Steam closed.
- Evidence: run/ (kernel.txt, stdout, meta), capture/ (kmsg, observe, steps).
- Devcoredump: /sys/class/devcoredump/devcd1 existed after the hang (root-only, expires ~5 min after creation).
- Next: reboot before any NPU work (npu-run refuses this boot); Jordan clears runs/STOP after review.
  Debug offline first (CPU only): compare the stream with what the 2023 branch emits for the same kernel.

## Offline comparison (2026-11-13, boot d5fe88d6, CPU only)
- Command-stream contents match the 2023 etnaviv-opencl code (context reset for OpenCL, shader upload, uniforms at
  SH_HALTI5_UNIFORMS(0), dispatch states). The global ID comes straight from the hardware register (no lowering).
- One difference: the 2023 code writes buffer addresses directly (softpin); hello uses libdrm relocations, which this
  kernel applies at any position (submit_reloc), so that should be equivalent.
- No cause found by reading. Next: isolate on the hardware with a smaller verified kernel:
  hello --kernel store1 -n 1 -w 1 (one thread stores its global ID to out[0]; Mesa test vector line 164).
