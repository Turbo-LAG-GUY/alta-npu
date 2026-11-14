/* cs_kernels.h: test kernels for the etnaviv compute port (current Mesa NIR API, nir_def), shared by the
 * offline compiler cs_compile.c and the device harness. Same kernels as cl2023_copy.c (2023 branch API):
 *   copy: out[i] = in[i]                       (uint32, one per thread)
 *   fma:  out[i] = iters x (x * 0.999 + 0.001)  (float4 per thread)
 * Kernel arguments (kernel input buffer, 8 bytes): offset 0 = out address, offset 4 = in address (32-bit GPU VAs).
 *
 * Step 5 (shared memory + barriers), kernel input 16 bytes: 0 = out, 4 = in, 8 = A, 12 = B (uint32):
 *   ids:     out[8i..8i+7] = global.x, local.x, group.x, group.y, group.z, groups.x, groups.y, groups.z
 *            (reads the IDs in the order the shared kernels do: group first, then global, then local; every
 *            address comes from global.x only, so wrong IDs give wrong values, not stray addresses)
 *   reverse: shared[l] = in[i]; barrier; out[i] = shared[A + B*l]   (A = wg-1, B = -1 mirrors the workgroup;
 *            A = 0, B = 1 reads the thread's own slot); --no-barrier builds it without the barrier
 *   reduce:  tree sum of the workgroup in shared memory, one barrier per level, no branches (select instead);
 *            every thread writes its workgroup's sum: out[i] = sum of in[] over i's workgroup (uint32)
 * Shared memory per workgroup: CS_SHARED_BYTES (reverse uses 4 bytes per thread, reduce 4 bytes per thread plus
 * the half-workgroup it reads past the end at each level, so every index stays inside the slice).
 */
#ifndef CS_KERNELS_H
#define CS_KERNELS_H

#include "compiler/nir/nir_builder.h"

static nir_shader *
cs_build_copy(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "copy");
   b.shader->info.workgroup_size_variable = true;

   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *out_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 8);
   nir_def *in_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 4), .base = 0, .range = 8);
   nir_def *off = nir_ishl_imm(&b, gid, 2);

   nir_def *v = nir_load_global(&b, 1, 32, nir_iadd(&b, in_addr, off), .align_mul = 4);
   nir_store_global(&b, v, nir_iadd(&b, out_addr, off), .write_mask = 0x1, .align_mul = 4);

   return b.shader;
}

static nir_shader *
cs_build_fma(const nir_shader_compiler_options *opts, unsigned iters)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "fma");
   b.shader->info.workgroup_size_variable = true;

   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *out_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 8);
   nir_def *in_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 4), .base = 0, .range = 8);
   nir_def *off = nir_ishl_imm(&b, gid, 4);

   nir_def *x = nir_load_global(&b, 4, 32, nir_iadd(&b, in_addr, off), .align_mul = 16);
   nir_def *a = nir_imm_float(&b, 0.999f), *c = nir_imm_float(&b, 0.001f);
   for (unsigned i = 0; i < iters; i++)
      x = nir_ffma(&b, x, a, c);
   nir_store_global(&b, x, nir_iadd(&b, out_addr, off), .write_mask = 0xf, .align_mul = 16);

   return b.shader;
}

/* Step 6 (transposes between channels-last NHWC and channel planes NCHW, uint8 tensors; 2D grids):
 *   ids2d:   2D grid; thread (x, y) writes 8 words at out + (min(y, ny-1) * A + min(x, A-1)) * 32 (A = threads in x;
 *            kernel input: out, -, A, A-1, ny-1):
 *            global.x, global.y, local.x, local.y, group.x, group.y, groups.x, groups.y
 *   t4x4:    4x4 byte-block transpose; thread (x, y) reads 4 words at in + x*RX + y*RY + i*RS (i = 0..3) and writes
 *            4 words at out + x*WX + y*WY + k*WS, byte i of word k = byte k of word i. x = group of 4 pixels,
 *            y = group of 4 channels. NHWC -> NCHW: RX = 4C, RY = 4, RS = C, WX = 4, WY = 4HW, WS = HW (needs
 *            C % 4 == 0, HW % 4 == 0); NCHW -> NHWC: RX = 4, RY = 4HW, RS = HW, WX = 4C, WY = 4, WS = C.
 *            Kernel input 32 bytes: out, in, RX, RY, RS, WX, WY, WS. t4x4b: same result, butterfly shuffle;
 *            t4x4copy: same loads/stores without the shuffle (memory floor, output is not a transpose).
 *   tgather: NHWC -> NCHW for any C (HW % 4 == 0): thread (x, c) writes the word at out + c*HW + 4x, gathering byte
 *            (4x+i)*C + c of the input for i = 0..3 (aligned word load + shift). Kernel input: out, in, C, HW.
 */
/* softmax (step 6c): the driver's own kernel builder (mesa-compute etnaviv_ml_shader.c), so the harness tests the
 * exact code Teflon runs. Rows of n uint8, factor = beta * input_scale * log2(e), 64 threads per row. */
struct nir_shader *
etna_ml_shader_build_softmax(const struct nir_shader_compiler_options *opts, unsigned n, unsigned rows,
                             float factor, unsigned threads);

/* ids_gl: the ID order of the softmax kernel (workgroup first, then local; value order 0x1). Thread writes
 * group.x, local.x at out + min(group.x * A + local.x, B) * 8 (A = workgroup size, B = threads - 1). Input: out, -, A, B. */
static nir_shader *
cs_build_ids_gl(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "ids_gl");
   b.shader->info.workgroup_size_variable = true;

   nir_def *group = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *out_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 16);
   nir_def *wgs = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 8), .base = 0, .range = 16);
   nir_def *last = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 12), .base = 0, .range = 16);

   nir_def *idx = nir_umin(&b, nir_iadd(&b, nir_imul(&b, group, wgs), lid), last);
   nir_store_global(&b, nir_vec2(&b, group, lid), nir_iadd(&b, out_addr, nir_ishl_imm(&b, idx, 3)),
                    .write_mask = 0x3, .align_mul = 8);

   return b.shader;
}

#define CS_SHARED_BYTES 2048 /* 2 x 256 threads x 4 bytes */

static void
cs_kernel_args(nir_builder *b, nir_def **out_addr, nir_def **in_addr, nir_def **a, nir_def **bb)
{
   *out_addr = nir_load_kernel_input(b, 1, 32, nir_imm_int(b, 0), .base = 0, .range = 16);
   *in_addr = nir_load_kernel_input(b, 1, 32, nir_imm_int(b, 4), .base = 0, .range = 16);
   if (a)
      *a = nir_load_kernel_input(b, 1, 32, nir_imm_int(b, 8), .base = 0, .range = 16);
   if (bb)
      *bb = nir_load_kernel_input(b, 1, 32, nir_imm_int(b, 12), .base = 0, .range = 16);
}

static void
cs_workgroup_barrier(nir_builder *b)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
}

static nir_shader *
cs_build_ids(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "ids");
   b.shader->info.workgroup_size_variable = true;

   nir_def *group = nir_load_workgroup_id(&b);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *groups = nir_load_num_workgroups(&b);
   nir_def *out_addr, *in_addr;
   cs_kernel_args(&b, &out_addr, &in_addr, NULL, NULL);

   nir_def *addr = nir_iadd(&b, out_addr, nir_ishl_imm(&b, gid, 5));
   nir_def *v0 = nir_vec4(&b, gid, lid, nir_channel(&b, group, 0), nir_channel(&b, group, 1));
   nir_def *v1 = nir_vec4(&b, nir_channel(&b, group, 2), nir_channel(&b, groups, 0), nir_channel(&b, groups, 1),
                          nir_channel(&b, groups, 2));
   nir_store_global(&b, v0, addr, .write_mask = 0xf, .align_mul = 16);
   nir_store_global(&b, v1, nir_iadd_imm(&b, addr, 16), .write_mask = 0xf, .align_mul = 16);

   return b.shader;
}

static nir_shader *
cs_build_reverse(const nir_shader_compiler_options *opts, bool barrier)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, barrier ? "reverse" : "reverse_nobarrier");
   b.shader->info.workgroup_size_variable = true;
   b.shader->info.shared_size = CS_SHARED_BYTES;

   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *out_addr, *in_addr, *a, *bb;
   cs_kernel_args(&b, &out_addr, &in_addr, &a, &bb);
   nir_def *off = nir_ishl_imm(&b, gid, 2);

   nir_def *v = nir_load_global(&b, 1, 32, nir_iadd(&b, in_addr, off), .align_mul = 4);
   nir_store_shared(&b, v, nir_ishl_imm(&b, lid, 2), .base = 0, .write_mask = 0x1, .align_mul = 4);
   if (barrier)
      cs_workgroup_barrier(&b);
   nir_def *idx = nir_iadd(&b, a, nir_imul(&b, bb, lid));
   nir_def *w = nir_load_shared(&b, 1, 32, nir_ishl_imm(&b, idx, 2), .base = 0, .align_mul = 4);
   nir_store_global(&b, w, nir_iadd(&b, out_addr, off), .write_mask = 0x1, .align_mul = 4);

   return b.shader;
}

/* regp (register budget probe, after incident 2026-11-13-2109-S6C-SM1001): reverse (mirror, one barrier) plus k4 vec4
 * values loaded before the barrier and added after it, so they are live across it: k4 sets the temps. Thread i loads
 * extras at in + n*4 + (i*k4 + v)*16 (A = n*4 passed at offset 8); out[i] = in[mirror(i)] + sum of its extras' 4*k4
 * words. Kernel input: out, in, A, wg-1. */
static nir_shader *
cs_build_regp(const nir_shader_compiler_options *opts, unsigned k4)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "regp");
   b.shader->info.workgroup_size_variable = true;
   b.shader->info.shared_size = CS_SHARED_BYTES;

   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *out_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 16);
   nir_def *in_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 4), .base = 0, .range = 16);
   nir_def *extra_base = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 8), .base = 0, .range = 16);
   nir_def *wgm1 = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 12), .base = 0, .range = 16);
   nir_def *off = nir_ishl_imm(&b, gid, 2);

   nir_def *ex[k4 ? k4 : 1];
   nir_def *ebase = nir_iadd(&b, in_addr, nir_iadd(&b, extra_base, nir_imul_imm(&b, gid, k4 * 16)));
   for (unsigned i = 0; i < k4; i++)
      ex[i] = nir_load_global(&b, 4, 32, nir_iadd_imm(&b, ebase, i * 16), .align_mul = 16);

   nir_def *v = nir_load_global(&b, 1, 32, nir_iadd(&b, in_addr, off), .align_mul = 4);
   nir_store_shared(&b, v, nir_ishl_imm(&b, lid, 2), .base = 0, .write_mask = 0x1, .align_mul = 4);
   cs_workgroup_barrier(&b);
   nir_def *w = nir_load_shared(&b, 1, 32, nir_ishl_imm(&b, nir_isub(&b, wgm1, lid), 2), .base = 0, .align_mul = 4);

   for (unsigned i = 0; i < k4; i++)
      w = nir_iadd(&b, w, nir_iadd(&b, nir_iadd(&b, nir_channel(&b, ex[i], 0), nir_channel(&b, ex[i], 1)),
                                    nir_iadd(&b, nir_channel(&b, ex[i], 2), nir_channel(&b, ex[i], 3))));
   nir_store_global(&b, w, nir_iadd(&b, out_addr, off), .write_mask = 0x1, .align_mul = 4);

   return b.shader;
}

static nir_shader *
cs_build_reduce(const nir_shader_compiler_options *opts, unsigned wg)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "reduce");
   b.shader->info.workgroup_size_variable = true;
   b.shader->info.shared_size = CS_SHARED_BYTES;

   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *out_addr, *in_addr;
   cs_kernel_args(&b, &out_addr, &in_addr, NULL, NULL);
   nir_def *off = nir_ishl_imm(&b, gid, 2);
   nir_def *lofs = nir_ishl_imm(&b, lid, 2);

   nir_def *v = nir_load_global(&b, 1, 32, nir_iadd(&b, in_addr, off), .align_mul = 4);
   nir_store_shared(&b, v, lofs, .base = 0, .write_mask = 0x1, .align_mul = 4);
   cs_workgroup_barrier(&b);
   for (unsigned s = wg / 2; s >= 1; s /= 2) {
      nir_def *x = nir_load_shared(&b, 1, 32, lofs, .base = 0, .align_mul = 4);
      nir_def *y = nir_load_shared(&b, 1, 32, lofs, .base = s * 4, .align_mul = 4);
      nir_def *r = nir_bcsel(&b, nir_ult_imm(&b, lid, s), nir_iadd(&b, x, y), x);
      nir_store_shared(&b, r, lofs, .base = 0, .write_mask = 0x1, .align_mul = 4);
      cs_workgroup_barrier(&b);
   }
   nir_def *sum = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .align_mul = 4);
   nir_store_global(&b, sum, nir_iadd(&b, out_addr, off), .write_mask = 0x1, .align_mul = 4);

   return b.shader;
}

/* fops (step 6c, float ops a softmax needs): thread i takes a = (float)(in[i] & 0xff), x = a/32 - 4, and writes 8 words
 * at out + 32i: exp2(x), exp(x), 1/(a+1), a/(a+3), max(x, 0.25), (uint)(a*0.75 + 0.5), floor(a*0.3), min(1/(a+1), 0.5).
 * Kernel input: out, in. */
static nir_shader *
cs_build_fops(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "fops");
   b.shader->info.workgroup_size_variable = true;

   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *out_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 8);
   nir_def *in_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 4), .base = 0, .range = 8);

   nir_def *w = nir_load_global(&b, 1, 32, nir_iadd(&b, in_addr, nir_ishl_imm(&b, gid, 2)), .align_mul = 4);
   nir_def *a = nir_u2f32(&b, nir_iand_imm(&b, w, 0xff));
   nir_def *x = nir_fadd_imm(&b, nir_fmul_imm(&b, a, 1.0 / 32.0), -4.0);
   nir_def *r = nir_frcp(&b, nir_fadd_imm(&b, a, 1.0));

   nir_def *v0 = nir_vec4(&b, nir_fexp2(&b, x), nir_fexp2(&b, nir_fmul_imm(&b, x, M_LOG2E)), r,
                          nir_fdiv(&b, a, nir_fadd_imm(&b, a, 3.0)));
   nir_def *v1 = nir_vec4(&b, nir_fmax(&b, x, nir_imm_float(&b, 0.25)),
                          nir_f2u32(&b, nir_fadd_imm(&b, nir_fmul_imm(&b, a, 0.75), 0.5)),
                          nir_ffloor(&b, nir_fmul_imm(&b, a, 0.3)), nir_fmin(&b, r, nir_imm_float(&b, 0.5)));

   nir_def *addr = nir_iadd(&b, out_addr, nir_ishl_imm(&b, gid, 5));
   nir_store_global(&b, v0, addr, .write_mask = 0xf, .align_mul = 16);
   nir_store_global(&b, v1, nir_iadd_imm(&b, addr, 16), .write_mask = 0xf, .align_mul = 16);

   return b.shader;
}

static nir_def *
cs_arg(nir_builder *b, unsigned index, unsigned range)
{
   return nir_load_kernel_input(b, 1, 32, nir_imm_int(b, index * 4), .base = 0, .range = range);
}

static nir_shader *
cs_build_ids2d(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "ids2d");
   b.shader->info.workgroup_size_variable = true;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *lid = nir_load_local_invocation_id(&b);
   nir_def *group = nir_load_workgroup_id(&b);
   nir_def *groups = nir_load_num_workgroups(&b);
   nir_def *out_addr = cs_arg(&b, 0, 32), *gx = cs_arg(&b, 2, 32);
   nir_def *max_x = cs_arg(&b, 3, 32), *max_y = cs_arg(&b, 4, 32);

   /* the address uses the IDs clamped to the grid, so wrong IDs give wrong values, never stray addresses */
   nir_def *x = nir_umin(&b, nir_channel(&b, gid, 0), max_x), *y = nir_umin(&b, nir_channel(&b, gid, 1), max_y);
   nir_def *linear = nir_iadd(&b, nir_imul(&b, y, gx), x);
   nir_def *addr = nir_iadd(&b, out_addr, nir_ishl_imm(&b, linear, 5));
   nir_def *v0 = nir_vec4(&b, nir_channel(&b, gid, 0), nir_channel(&b, gid, 1), nir_channel(&b, lid, 0),
                          nir_channel(&b, lid, 1));
   nir_def *v1 = nir_vec4(&b, nir_channel(&b, group, 0), nir_channel(&b, group, 1), nir_channel(&b, groups, 0),
                          nir_channel(&b, groups, 1));
   nir_store_global(&b, v0, addr, .write_mask = 0xf, .align_mul = 16);
   nir_store_global(&b, v1, nir_iadd_imm(&b, addr, 16), .write_mask = 0xf, .align_mul = 16);

   return b.shader;
}

/* mode 0: per-byte shuffle; 1: butterfly (2x2 byte swaps within word pairs, then 2x2 halfword swaps, 28 ops);
 * 2: no shuffle (word k = word k: same loads and stores, i.e. the memory floor; output is NOT a transpose) */
static nir_shader *
cs_build_t4x4(const nir_shader_compiler_options *opts, unsigned mode)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts,
                                                  mode == 1 ? "t4x4b" : mode == 2 ? "t4x4copy" : "t4x4");
   b.shader->info.workgroup_size_variable = true;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0), *y = nir_channel(&b, gid, 1);
   nir_def *out_addr = cs_arg(&b, 0, 32), *in_addr = cs_arg(&b, 1, 32);
   nir_def *rx = cs_arg(&b, 2, 32), *ry = cs_arg(&b, 3, 32), *rs = cs_arg(&b, 4, 32);
   nir_def *wx = cs_arg(&b, 5, 32), *wy = cs_arg(&b, 6, 32), *ws = cs_arg(&b, 7, 32);

   nir_def *raddr = nir_iadd(&b, in_addr, nir_iadd(&b, nir_imul(&b, x, rx), nir_imul(&b, y, ry)));
   nir_def *waddr = nir_iadd(&b, out_addr, nir_iadd(&b, nir_imul(&b, x, wx), nir_imul(&b, y, wy)));
   nir_def *w[4];
   for (unsigned i = 0; i < 4; i++) {
      w[i] = nir_load_global(&b, 1, 32, raddr, .align_mul = 4);
      if (i < 3)
         raddr = nir_iadd(&b, raddr, rs);
   }
   nir_def *outw[4];
   if (mode == 1) {
      /* rows a, b, c, d = w0..w3: a' = [a0 b0 a2 b2], b' = [a1 b1 a3 b3] (bytes, low first), same for c, d */
      nir_def *t[4];
      for (unsigned p = 0; p < 4; p += 2) {
         nir_def *lo = w[p], *hi = w[p + 1];
         t[p] = nir_ior(&b, nir_iand_imm(&b, lo, 0x00ff00ff), nir_ishl_imm(&b, nir_iand_imm(&b, hi, 0x00ff00ff), 8));
         t[p + 1] = nir_ior(&b, nir_iand_imm(&b, nir_ushr_imm(&b, lo, 8), 0x00ff00ff), nir_iand_imm(&b, hi, 0xff00ff00));
      }
      /* then halfwords: o0 = [a0 b0 c0 d0] etc. */
      outw[0] = nir_ior(&b, nir_iand_imm(&b, t[0], 0xffff), nir_ishl_imm(&b, t[2], 16));
      outw[1] = nir_ior(&b, nir_iand_imm(&b, t[1], 0xffff), nir_ishl_imm(&b, t[3], 16));
      outw[2] = nir_ior(&b, nir_ushr_imm(&b, t[0], 16), nir_iand_imm(&b, t[2], 0xffff0000));
      outw[3] = nir_ior(&b, nir_ushr_imm(&b, t[1], 16), nir_iand_imm(&b, t[3], 0xffff0000));
   } else {
      for (unsigned k = 0; k < 4; k++) {
         nir_def *o = NULL;
         for (unsigned i = 0; i < 4 && mode == 0; i++) {
            nir_def *byte = nir_iand_imm(&b, k ? nir_ushr_imm(&b, w[i], 8 * k) : w[i], 0xff);
            if (i)
               byte = nir_ishl_imm(&b, byte, 8 * i);
            o = o ? nir_ior(&b, o, byte) : byte;
         }
         outw[k] = mode == 2 ? w[k] : o;
      }
   }
   for (unsigned k = 0; k < 4; k++) {
      nir_store_global(&b, outw[k], waddr, .write_mask = 0x1, .align_mul = 4);
      if (k < 3)
         waddr = nir_iadd(&b, waddr, ws);
   }

   return b.shader;
}

static nir_shader *
cs_build_tgather(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "tgather");
   b.shader->info.workgroup_size_variable = true;

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0), *c = nir_channel(&b, gid, 1);
   nir_def *out_addr = cs_arg(&b, 0, 16), *in_addr = cs_arg(&b, 1, 16);
   nir_def *chans = cs_arg(&b, 2, 16), *plane = cs_arg(&b, 3, 16);

   /* byte index of (pixel 4x, channel c) in the input, then + C per pixel */
   nir_def *byte_idx = nir_iadd(&b, nir_imul(&b, nir_ishl_imm(&b, x, 2), chans), c);
   nir_def *o = NULL;
   for (unsigned i = 0; i < 4; i++) {
      nir_def *word = nir_load_global(&b, 1, 32, nir_iadd(&b, in_addr, nir_iand_imm(&b, byte_idx, ~3u)),
                                      .align_mul = 4);
      nir_def *shift = nir_ishl_imm(&b, nir_iand_imm(&b, byte_idx, 3), 3);
      nir_def *byte = nir_iand_imm(&b, nir_ushr(&b, word, shift), 0xff);
      if (i)
         byte = nir_ishl_imm(&b, byte, 8 * i);
      o = o ? nir_ior(&b, o, byte) : byte;
      if (i < 3)
         byte_idx = nir_iadd(&b, byte_idx, chans);
   }
   nir_def *waddr = nir_iadd(&b, out_addr, nir_iadd(&b, nir_imul(&b, c, plane), nir_ishl_imm(&b, x, 2)));
   nir_store_global(&b, o, waddr, .write_mask = 0x1, .align_mul = 4);

   return b.shader;
}

#endif
