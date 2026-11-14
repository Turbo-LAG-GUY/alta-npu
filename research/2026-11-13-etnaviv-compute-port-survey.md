# Porting etnaviv compute (NPU shader core) to current Mesa — survey, 2026-11-13

Source: Tomeu Vizoso's `etnaviv-opencl` branch (tomeu/mesa, newest commit 2023-06-26), fetched shallowly into
`~/Documents/AI_Projects/mesa` as `refs/remotes/cl/tomeu-etnaviv-opencl`; unpacked and built (etnaviv only) in
`~/Documents/AI_Projects/mesa-cl2023`. Proven on this board the same day: copy kernel correct at 1/64/256 threads,
float multiply-add kernel 6.14 GFLOPS fp32 (RESULTS.md CL23-*, FMA-*; harness `npu-shader/cl2023_copy`).
Target: the Mesa 26.1-devel tree that Teflon/the kit library come from (`mesa-axi-sram`).

## Size

125 commits on top of upstream (106 Tomeu Vizoso, 17 Italo Nicola, 1 Christian Gmeiner, 1 Karol Herbst),
67 files, about +3200/-300 lines. Most of it is WIP/FIXUP history, so the port should re-implement the branch's
*final* code for the needed paths as a clean patch series, not cherry-pick commits.

| Group | Commits | Needed for the proven kernels? |
|---|---|---|
| Dispatch + compiler core (compute API, launch_grid, cmdstream state, kernel args, thread IDs, global load/store, temps, integer types) | 45 | **yes** |
| Small fixups (input-arg patching, WC buffers, NOP padding, fround_even) | ~5 of 12 "other" | yes |
| Shared/private/scratch memory, barriers | 9 | later (softmax/layernorm reductions need shared memory + barriers) |
| Integer / 64-bit / vector-width coverage | 13 | partly, later |
| Images | 9 | no |
| Atomics | 5 | no |
| rusticl / OpenCL C front end | 5 | no (no LLVM/libclc/bindgen here) |
| CI, test lists | 11 | no |
| Debug hacks, dumps, scripts | 16 | no |

Where the lines are: etnaviv_compiler_nir.c 755, etnaviv_uniforms.c 291, etnaviv_context.c 272, etnaviv_emit.c 239,
etnaviv_compute.c 186 (new), etnaviv_screen.c 112, etnaviv_shader.c 103, plus small NIR passes.

## What current Mesa already has

- Nothing of the compute work: no `launch_grid`, no compute/kernel stage in the etnaviv compiler (0 references),
  no `etnaviv_compute.c`, no `etna_nir_lower_global`, no `nir_lower_kernel_io_to_scalar`.
- NIR has `load_global_ir3`/`store_global_ir3` (for freedreno), but `load_global`/`store_global` have no
  DEST_TYPE/SRC_TYPE (the branch added those; its compiler asserts without them).
- The register database already has all `VIVS_CL_*` dispatch states; the ISA (isaspec) has load/store/atomics/barrier.
- Since 2023 the etnaviv compiler moved to isaspec (`ISA_OPC_*`, new `struct etna_inst`), NIR renamed `nir_ssa_def`
  to `nir_def`, and gallium moved screen caps into structs. All three need adapting, not copying.

## Port plan

1. **Tree:** a new isolated copy of `mesa-axi-sram` (same Mesa as Teflon, so Teflon integration can follow in the
   same tree); the kit library is never rebuilt. Copy procedure as in the AXI SRAM notes (cp -a, retarget the
   regenerate lines, touch build.ninja; check deps with `ninja -t deps`). Builds at -j2.
2. **Driver side:** `etnaviv_compute.c` (compute state, global bindings), `launch_grid`, compute-only context reset,
   `etna_emit_compute_state` (dispatch states as on the branch, which match the vendor-derived kernel flop-reset
   sequence), kernel arguments as uniforms through the command stream, compute caps in the screen.
3. **Compiler side:** kernel/compute stage I/O (thread-ID registers + VALUE_ORDER), kernel input → uniform,
   global load/store emission (u32 type from bit size instead of new NIR type indices, as the branch's
   "Use U32 instead of F32 for stores and loads" fixup did), store scratch register and temp count, integer fixes
   only as the kernels need them.
4. **Test:** port `cl2023_copy` to the new API; CPU compile checks first, then the same NPU ladder as today
   (compile-only, copy 1/64/256, multiply-add), each NPU step with an explicit OK. The 2023 build stays as the
   reference: its compiled-shader and command-stream dumps (runs/cl2023-180102, runs/fma-180621) are the
   byte-level baseline to diff the port against.
5. **Later:** shared memory + barriers (needed for softmax/layernorm reductions), then Teflon placing those ops
   on the shader core inside the same NPU job.

Estimate: steps 1–4 several days of work; step 5 more on top.
