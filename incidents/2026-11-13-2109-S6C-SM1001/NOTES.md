# Incident 2026-11-13 21:09 - S6C-SM1001 (first softmax kernel on the NPU shader core)

- Job: npu-attempt S6C-SM1001 = ~/Documents/AI_Projects/npu-shader/cs_run --kernel softmax --len 1001 --rows 1
  --repeat 10 (Jordan's blanket OK for the step 6c ladder: "I give you the OK for all steps ... full autonomy till I get
  back"). Boot d5fe88d6. Kernel = the driver's own builder, etna_ml_shader_build_softmax (mesa-compute
  etnaviv_ml_shader.c): one workgroup of 64 threads, 16 uint8 values per thread kept in registers, two shared-memory
  tree reductions (max, sum: 16 barriers in all), exp2/rcp/f2u. 336 instructions, num_temps 15 (driver programs 16),
  thread allocation 16 (= 64 / 4), value order 0x1 (workgroup ID, local ID). Disassembly: softmax1001.disasm.txt.
- Result: "recover hung GPU!" once per launch (10 launches, ~0.54 s each until recovery), no MMU fault; output buffer
  untouched (still 0xdeadbeef). npu-run rc=90, STOP latch set. Ladder stopped: S6C-SM4x1000 not run.
- Board: stayed up. Steam closed, no other NPU users.
- Evidence: run/ (kernel.txt, stdout.txt, meta.txt), capture/ (kmsg, observe, steps), SM1001.stdout.
- Devcoredump: /sys/class/devcoredump/devcd1 existed (data is root-only, mode 0600); Claude could not read it and it
  expired (~5 min) before Jordan was back. Not saved.

## What passed just before, same boot, same driver build
- S6C-IDSGL: same ID inputs (workgroup, local; value order 0x1), 8 instructions - PASS. So the IDs are right.
- S6C-RED64: shared-memory tree sum, 64 threads, 7 barriers, thread allocation 16, num_temps 5 (+1) - PASS (also
  proves the new workgroup clamp in the shared-slot address).
- S6C-FOPS: exp2, exp, rcp, div, fmax/fmin, f2u, floor - PASS (~1 ulp).
- S5-REV64 / S5-RED64 earlier: 64-thread workgroups with barriers, few temps - PASS.

## Leading hypothesis (not proven): register file capacity vs a barrier
With barriers, all 64 threads of a workgroup must be resident at once (CL_THREAD_ALLOCATION 16). The passing barrier
kernels used 4-6 temps per thread (64 x 6 = 384 vec4 registers); this one uses 16 (64 x 16 = 1024). If the core's
register file cannot hold 64 threads x 16 temps, some threads can't be scheduled while the others wait at the first
barrier: a deadlock, which matches a hang with no fault and nothing written. Other differences (more barriers, 336
instructions vs 261 for the passing fma kernel) seem less likely.

## Next (needs a reboot first; npu-run refuses NPU jobs this boot; Jordan clears runs/STOP after review)
1. Offline (CPU only, done in this session): a low-register softmax (values recomputed instead of kept: ~5-6 temps)
   and a register-pressure probe (the reduce kernel with K extra values live across its barriers) so the limit can
   be mapped: workgroup 64 with K = 4, 8, 12 -> passes until temps x 64 exceeds the capacity.
2. On the NPU after reboot, with Jordan's OK: probe ladder from low to high pressure (stop at the first hang), then
   the low-register softmax.
3. Driver: once the limit is known, refuse (BUG) a grid with barriers whose workgroup size x temps exceeds it, so no
   kernel can deadlock the core this way again.

## CPU follow-up the same evening (boot d5fe88d6, no NPU job)
- Why 16 temps: NIR CSE merged the per-element index/validity values of the kernel's passes and the reduction load
  addresses (identical in both reductions), so they stayed live across every barrier; the 16 values per thread did too.
- Softmax kernel v4 (mesa-compute etnaviv_ml_shader.c): three passes that reload the row, validity known at compile
  time except for the last word, reductions = one store + one barrier + 16 vec4 loads, each reduction in its own
  shared region. MobileNet's row (1001): 398 instructions, 9 temps programmed (64 x 9 = 576 registers).
- Guard: etna_compute_emit_grid refuses a barrier grid with workgroup size x temps above
  etna_compute_barrier_reg_budget() (default 384 = 64 x 6, the largest proven; ETNA_CS_BARRIER_REG_BUDGET raises it
  for a deliberate test). Teflon only claims a softmax whose kernel fits the budget.
- The shared-memory slot address now clamps the workgroup index to the grid (S6C-RED64 passed with it).
- Probe kernels regpK (cs_run --kernel regp4 ... regp12 = 7 ... 15 temps programmed) to map the limit.
- Ladder for after the reboot: research/port/step6c-ladder.sh (stops at the first problem; --map adds the probes
  above 576 that may hang).
