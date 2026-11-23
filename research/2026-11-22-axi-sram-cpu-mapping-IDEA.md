# Idea (parked 2026-11-22): let the CPU write the NPU's AXI SRAM directly

Status: **not built, not tested.** Parked by Jordan to come back to later. Nothing here has touched the kernel or the NPU.

## The idea
Today only the NPU can put data into the 1 MiB AXI SRAM (physical 0xff000000): the kernel patch
`patches/etnaviv-axi-sram/0005-drm-etnaviv-alta-NPU-AXI-SRAM-buffers.patch` maps it into the NPU's MMU only, and its
`mmap` / `vmap` hooks return `-ENXIO`. So to get an LLM step's input into the AXI SRAM, the NPU has to copy it there
first with a transpose (TP) job (`ETNA_ML_SHARE_INPUT_TRANSPOSE=1` + `ETNA_ML_AXI_SRAM=all`, mesa-llm 4b666c08).

The idea: give the CPU a mapping of the same buffer, so the CPU glue writes the step input straight into the AXI SRAM
and the NPU's convs read it from there with no copy job.

## What we know (measured 2026-11-22)
- There is **no general-purpose DMA engine** in this board's device tree (only the CMA pool), so "CPU DMA" means the
  CPU itself writing through a mapping (memcpy).
- The device tree has **no node for 0xff000000**. The NPU's registers are at 0xff100000, right next to it, so the
  SRAM is probably part of the NPU block. Amlogic's galcore driver (VIM3 platform file) maps it for the NPU only.
- Whether the CPU can reach 0xff000000 at all is **unknown**. If it can't (not routed, or secure-only), the first
  access is an external abort: usually a kernel oops / panic or a hung board.
- NPU ladder runs/llm-axi-ladder-20261122-140434 (8 jobs, 2048 -> 8 x 224, 256 tokens, NPU ms):

  | input path | ms |
  |---|---|
  | NPU TP copy into the AXI SRAM, convs read it there (N2) | 8.87 |
  | same, copy made free by a CPU write (estimate: N2 - ~1.9 ms copy) | ~6.97 |
  | input left in DDR (P0) | 6.57 |
  | tile 16x8 + keep-image, input cached in the 512 KiB VIP SRAM (PT) | **2.81** |

  The convs read the AXI SRAM no faster than DDR (the NN core's input fetch runs ~3 GB/s either way), so even a free
  copy would not beat leaving the input in DDR, and the VIP SRAM route is 2.3x faster than both.
- Writing a 512 KiB input to DDR from the CPU already costs only ~0.15 ms, so the CPU-side saving is nil too.

## Why it might still be worth it later
- **DDR bandwidth for the CPU.** If the CPU glue (rewritten in C/NEON) and the NPU run at the same time and fight for
  DDR, inputs and in-between tensors that never touch DDR could help the CPU side. Unmeasured.
- **A different NPU access pattern.** If a later finding (circular buffers, weight streaming, a different tiling)
  makes the NN core read external memory faster, the AXI SRAM's own speed could start to matter.
- **Results back to the CPU without DDR**, if the CPU could also read it (read-out is ~30 ms of a 104 ms chunk on the
  135M), but only if CPU reads from that SRAM are fast, which is also unknown.

## What it would take
1. Kernel patch (a successor to 0005): implement `etnaviv_gem_axi_sram_mmap` with `remap_pfn_range` on
   `ALTA_AXI_SRAM_PHYS >> PAGE_SHIFT`, `pgprot_writecombine` (or `pgprot_noncached` first, the most conservative), and
   keep it behind the existing `axi_sram` module parameter plus a new one (e.g. `axi_sram_cpu`, default off).
   Build with the headers' gcc-14 (see memory: alta-npu-driver-verified); Jordan installs it as a root step.
2. First test = **one read** of one word through the mapping, with a reboot ready and nothing else running
   (devcoredump watcher on; incident folder ready). Then one write + read-back. Then a small memcpy.
3. Only then: Mesa side - let a graph input live in the AXI BO (patch 0002 excludes graph inputs today because the
   CPU touches them), map it for the CPU, and measure a pack against P0 / PT.
4. Measure CPU write and read speed into it, and whether the CPU glue gets faster while the NPU runs (the actual
   reason to do this).

## Risks
- A CPU access the interconnect rejects can take the whole board down (no recovery without a power cycle).
- The SRAM is one physical 1 MiB that every NPU program maps (see the one-submit guards, mesa-llm 4eb662fc): a CPU
  mapping that lives across submits breaks the "only valid inside one submit" rule unless the engine owns the NPU
  exclusively.

## Pointers
- Kernel patch today: patches/etnaviv-axi-sram/0005-drm-etnaviv-alta-NPU-AXI-SRAM-buffers.patch
- Mesa AXI placement: patches/mesa/0002-etnaviv-ml-alta-AXI-SRAM-tensors.diff, mesa-llm 4b666c08 / 4eb662fc
- Measurements: runs/llm-axi-ladder-20261122-140434, llm/DESIGN.md ("Resume here 2026-11-22")
- Memory: npu-axi-sram.md
