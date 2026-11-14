/*
 * SPDX-License-Identifier: MIT
 */

/*
 * ML operations run on the NPU's shader core, as compute grids inside the ML command stream (alta port, step 6).
 *
 * So far: the implicit layout transposes Teflon adds at the ends of a graph (TFLite tensors are channels-last,
 * NHWC; the NN cores want channel planes, NCHW). On the TP unit they ran at ~78 MB/s for 256-channel tensors; the
 * shader-core kernel below (4x4 byte blocks, butterfly shuffle) measured 14.7 ms in / 13.2 ms out at 112x112x256
 * (RESULTS.md S6-T4B-IN/OUT), about 3x faster. On narrow tensors (4-5 channels) the TP is faster, so only tensors
 * with enough channels are moved.
 *
 * Opt-in: ETNA_ML_SHADER=transpose. ETNA_ML_SHADER_MIN_CHANNELS (default 16) sets the narrowest tensor moved.
 *
 * Softmax (ETNA_ML_SHADER=softmax, can be combined: "transpose,softmax"): TFLite SOFTMAX over rows [N, 1, 1, C] of
 * 8-bit values, one 64-thread workgroup per row (etna_ml_shader_build_softmax); a SQUEEZE in front of it is taken as a
 * reshape so both can join the NPU graph (MobileNet ends CONV -> SQUEEZE -> SOFTMAX). Only kernels that fit the
 * register budget of barrier kernels (etna_compute_barrier_reg_budget) are used; others stay on the CPU.
 *
 * Global average pool (ETNA_ML_SHADER=avgpool): TFLite AVERAGE_POOL_2D over the whole input (MobileNet's 7x7x1024 ->
 * 1x1x1024), one thread per 4 channel planes (etna_ml_shader_build_avgpool), no barriers. Without it Teflon splits
 * such a graph into two NPU submits with the pool on the CPU between them.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/nir/nir_builder.h"

#include "etnaviv_compiler.h"
#include "etnaviv_context.h"
#include "etnaviv_debug.h"
#include "etnaviv_emit.h"
#include "etnaviv_ml.h"
#include "etnaviv_ml_shader.h"
#include "etnaviv_resource.h"
#include "etnaviv_screen.h"
#include "etnaviv_shader.h"

#include "hw/common.xml.h"
#include "hw/state.xml.h"

#include "util/u_inlines.h"

/* kernel input of the transpose kernel: out, in, RX, RY, RS, WX, WY, WS (see etna_ml_shader_build_transpose) */
#define TRANSPOSE_ARGS 8

static bool
shader_option(const char *name)
{
   static const char *env = NULL;
   static bool read = false;

   if (!read) {
      env = getenv("ETNA_ML_SHADER");
      read = true;
   }

   return env && strstr(env, name);
}

static unsigned
min_channels(void)
{
   const char *env = getenv("ETNA_ML_SHADER_MIN_CHANNELS");
   return env ? (unsigned)atoi(env) : 16;
}

/* Workgroup width for a grid of n threads in x: the largest power of two up to 64 that divides n. */
static unsigned
block_width(unsigned n)
{
   unsigned w = 64;

   while (w > 1 && n % w)
      w /= 2;

   return w;
}

bool
etna_ml_shader_wants_transpose(const struct etna_ml_subgraph *subgraph, const struct etna_operation *operation)
{
   const unsigned c = operation->input_channels;
   const unsigned hw = operation->input_width * operation->input_height;

   if (!shader_option("transpose") || !subgraph->screen->base.caps.compute ||
       DBG_ENABLED(ETNA_DBG_NPU_PARALLEL))
      return false;

   if (operation->type != ETNA_JOB_TYPE_TP ||
       (operation->tp_type != ETNA_ML_TP_TRANSPOSE && operation->tp_type != ETNA_ML_TP_DETRANSPOSE))
      return false;

   if (!c || !hw || c % 4 || hw % 4 || c < min_channels())
      return false;

   /* the grid: x = groups of 4 pixels, y = groups of 4 channels, within the 65535 workgroups per dimension */
   const unsigned gx = hw / 4;
   return gx / block_width(gx) <= 65535 && c / 4 <= 65535;
}

/*
 * 4x4 byte-block transpose, uint8. Thread (x, y) reads 4 words at in + x*RX + y*RY + i*RS (i = 0..3) and writes 4
 * words at out + x*WX + y*WY + k*WS, where byte i of word k = byte k of word i. The shuffle is a butterfly: 2x2 byte
 * swaps within word pairs, then 2x2 halfword swaps (28 integer ops). Same kernel as npu-shader/cs_kernels.h t4x4b,
 * proven byte-exact on the NPU.
 */
static nir_shader *
etna_ml_shader_build_transpose(const nir_shader_compiler_options *opts)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "etna_ml_transpose4x4");
   b.shader->info.workgroup_size_variable = true;

   nir_def *arg[TRANSPOSE_ARGS];
   for (unsigned i = 0; i < TRANSPOSE_ARGS; i++)
      arg[i] = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, i * 4), .base = 0, .range = TRANSPOSE_ARGS * 4);

   nir_def *gid = nir_load_global_invocation_id(&b, 32);
   nir_def *x = nir_channel(&b, gid, 0), *y = nir_channel(&b, gid, 1);

   nir_def *raddr = nir_iadd(&b, arg[1], nir_iadd(&b, nir_imul(&b, x, arg[2]), nir_imul(&b, y, arg[3])));
   nir_def *waddr = nir_iadd(&b, arg[0], nir_iadd(&b, nir_imul(&b, x, arg[5]), nir_imul(&b, y, arg[6])));

   nir_def *w[4];
   for (unsigned i = 0; i < 4; i++) {
      w[i] = nir_load_global(&b, 1, 32, raddr, .align_mul = 4);
      if (i < 3)
         raddr = nir_iadd(&b, raddr, arg[4]);
   }

   /* rows a, b, c, d = w0..w3: a' = [a0 b0 a2 b2], b' = [a1 b1 a3 b3] (bytes, low first), same for c, d */
   nir_def *t[4];
   for (unsigned p = 0; p < 4; p += 2) {
      nir_def *lo = w[p], *hi = w[p + 1];
      t[p] = nir_ior(&b, nir_iand_imm(&b, lo, 0x00ff00ff), nir_ishl_imm(&b, nir_iand_imm(&b, hi, 0x00ff00ff), 8));
      t[p + 1] = nir_ior(&b, nir_iand_imm(&b, nir_ushr_imm(&b, lo, 8), 0x00ff00ff), nir_iand_imm(&b, hi, 0xff00ff00));
   }

   /* then halfwords: out0 = [a0 b0 c0 d0], out1 = [a1 b1 c1 d1], ... */
   nir_def *o[4] = {
      nir_ior(&b, nir_iand_imm(&b, t[0], 0xffff), nir_ishl_imm(&b, t[2], 16)),
      nir_ior(&b, nir_iand_imm(&b, t[1], 0xffff), nir_ishl_imm(&b, t[3], 16)),
      nir_ior(&b, nir_ushr_imm(&b, t[0], 16), nir_iand_imm(&b, t[2], 0xffff0000)),
      nir_ior(&b, nir_ushr_imm(&b, t[1], 16), nir_iand_imm(&b, t[3], 0xffff0000)),
   };

   for (unsigned k = 0; k < 4; k++) {
      nir_store_global(&b, o[k], waddr, .write_mask = 0x1, .align_mul = 4);
      if (k < 3)
         waddr = nir_iadd(&b, waddr, arg[7]);
   }

   return b.shader;
}

/*
 * Softmax over rows of n uint8 values with TFLite's uint8 semantics: p = exp(beta * scale * (x - max)) / sum, output
 * q = round(256 p) clamped to 255 (output scale 1/256, zero point 0; the input zero point cancels out). factor is
 * beta * input_scale * log2(e). One workgroup of `threads` (4..ETNA_ML_SOFTMAX_THREADS, multiple of 4) per row (workgroup x = row; rows
 * after the first need n % 4 == 0 so they start on a word), each thread owning words lid + 64k of the row. Branch-free:
 * a thread past the row end redoes the last word (same values, so its duplicate store writes the same bytes) and is
 * left out of the max and the sum. The max and the sum are reductions through shared memory (softmax_reduce).
 * Three passes over the row (max, sum of exp, output), each reloading the words, so few registers stay live.
 * Kernel input: out, in. Writes whole words: up to 3 bytes past the row end of the last row are overwritten. rows (the
 * grid size) is baked in only to clamp the row index.
 */
/* Workgroup reduction (max of uint32 or sum of float): every thread stores its value, and after one barrier every
 * thread reads all of them back (vec4 loads) and combines them itself. Each reduction has its own region of shared
 * memory (region 0, 1, ...): nothing is overwritten while others still read it, and the load addresses differ between
 * reductions, so NIR CSE can't keep one reduction's addresses alive until the next. Few stores on purpose too: the
 * compiler keeps a store's source registers live to the end of the shader (a hazard rule from the 2023 compiler).
 * Registers are what limits a barrier kernel (incident 2026-11-13-2109-S6C-SM1001). */
static nir_def *
softmax_reduce(nir_builder *b, nir_def *lid, nir_def *value, bool fsum, unsigned region, unsigned threads)
{
   const unsigned base = region * threads * 4;

   nir_store_shared(b, value, nir_ishl_imm(b, lid, 2), .base = base, .write_mask = 0x1, .align_mul = 4);
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);

   nir_def *acc = NULL;
   for (unsigned t = 0; t < threads; t += 4) {
      nir_def *v = nir_load_shared(b, 4, 32, nir_imm_int(b, 0), .base = base + t * 4, .align_mul = 16);
      acc = !acc ? v : fsum ? nir_fadd(b, acc, v) : nir_umax(b, acc, v);
   }
   nir_def *x = nir_channel(b, acc, 0), *y = nir_channel(b, acc, 1);
   nir_def *z = nir_channel(b, acc, 2), *w = nir_channel(b, acc, 3);
   return fsum ? nir_fadd(b, nir_fadd(b, x, y), nir_fadd(b, z, w)) : nir_umax(b, nir_umax(b, x, y), nir_umax(b, z, w));
}

nir_shader *
etna_ml_shader_build_softmax(const nir_shader_compiler_options *opts, unsigned n, unsigned rows, float factor,
                             unsigned threads)
{
   const unsigned nw = DIV_ROUND_UP(n, 4), wpt = DIV_ROUND_UP(nw, threads);
   assert(threads >= 4 && threads <= ETNA_ML_SOFTMAX_THREADS && threads % 4 == 0);
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "etna_ml_softmax");
   b.shader->info.workgroup_size_variable = true;
   /* one slot per thread for each of the two reductions */
   b.shader->info.shared_size = threads * 4 * 2;

   nir_def *out_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 8);
   nir_def *in_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 4), .base = 0, .range = 8);
   /* IDs clamped to the grid, so a wrong ID can only give wrong values, never a stray address */
   nir_def *row = nir_umin(&b, nir_channel(&b, nir_load_workgroup_id(&b), 0), nir_imm_int(&b, rows - 1));
   nir_def *lid = nir_umin(&b, nir_channel(&b, nir_load_local_invocation_id(&b), 0), nir_imm_int(&b, threads - 1));
   nir_def *row_off = nir_imul_imm(&b, row, n);
   nir_def *in_row = nir_iadd(&b, in_addr, row_off), *out_row = nir_iadd(&b, out_addr, row_off);
   nir_def *sum_scale = NULL;

   /* Each pass reloads the row's words instead of keeping them in registers across the barriers: with barriers all
    * 64 threads must be resident at once, and a version that kept 16 values per thread (16 temps) hung the core
    * (incident 2026-11-13-2109-S6C-SM1001; register file capacity is the leading suspect). */
   /* Which words can be partial or duplicated is known at compile time: only the row's last word (nw - 1) holds bytes
    * past the row end (r valid ones), and only threads with j >= nw redo it. So only words k whose index range
    * reaches nw - 1 need runtime flags; NIR CSE keeps such flags live across all passes, so keep them few. */
   const unsigned r = n - 4 * (nw - 1);
   nir_def *m = nir_imm_int(&b, 0);
   for (unsigned pass = 0; pass < 3; pass++) {
      nir_def *sum = nir_imm_float(&b, 0.0f);
      nir_def *row_max = pass ? nir_u2f32(&b, m) : NULL;
      nir_def *scale = pass == 2 ? sum_scale : NULL;

      for (unsigned k = 0; k < wpt; k++) {
         const bool may_be_last = k * threads + threads - 1 >= nw - 1;
         nir_def *j = nir_iadd_imm(&b, lid, k * threads);
         nir_def *jc = may_be_last ? nir_umin(&b, j, nir_imm_int(&b, nw - 1)) : j;
         nir_def *word = nir_load_global(&b, 1, 32, nir_iadd(&b, in_row, nir_ishl_imm(&b, jc, 2)), .align_mul = 4);
         nir_def *is_last = may_be_last && r < 4 ? nir_ieq_imm(&b, jc, nw - 1) : NULL;
         nir_def *dup = may_be_last ? nir_uge_imm(&b, j, nw) : NULL;
         nir_def *out_word = NULL;

         for (unsigned i = 0; i < 4; i++) {
            nir_def *v = nir_iand_imm(&b, i ? nir_ushr_imm(&b, word, 8 * i) : word, 0xff);
            /* a byte past the row end: 0 in the max (values are >= 0 anyway), left out of the sum */
            nir_def *past_end = is_last && i >= r ? is_last : NULL;
            if (past_end)
               v = nir_bcsel(&b, past_end, nir_imm_int(&b, 0), v);

            if (pass == 0) {
               m = nir_umax(&b, m, v);
               continue;
            }

            nir_def *e = nir_fexp2(&b, nir_fmul_imm(&b, nir_fsub(&b, nir_u2f32(&b, v), row_max), factor));
            if (pass == 1) {
               nir_def *skip = past_end && dup ? nir_ior(&b, past_end, dup) : past_end ? past_end : dup;
               sum = nir_fadd(&b, sum, skip ? nir_bcsel(&b, skip, nir_imm_float(&b, 0.0f), e) : e);
               continue;
            }

            nir_def *q = nir_fmin(&b, nir_fadd_imm(&b, nir_fmul(&b, e, scale), 0.5), nir_imm_float(&b, 255.0f));
            nir_def *byte = nir_f2u32(&b, q);
            if (i)
               byte = nir_ishl_imm(&b, byte, 8 * i);
            out_word = out_word ? nir_ior(&b, out_word, byte) : byte;
         }

         if (pass == 2)
            nir_store_global(&b, out_word, nir_iadd(&b, out_row, nir_ishl_imm(&b, jc, 2)), .write_mask = 0x1,
                             .align_mul = 4);
      }

      if (pass == 0)
         m = softmax_reduce(&b, lid, m, false, 0, threads);
      else if (pass == 1)
         sum_scale = nir_fmul_imm(&b, nir_frcp(&b, softmax_reduce(&b, lid, sum, true, 1, threads)), 256.0);
   }

   return b.shader;
}

/* (n * m) >> s == n / d for every n in 0..n_max, with n_max * m < 2^32: the average pool's division as an integer
 * multiply and shift. Checked here for every n the kernel can see; false if no such m, s exist. */
static bool
udiv_magic(unsigned d, unsigned n_max, uint32_t *m, unsigned *s)
{
   for (unsigned sh = 0; sh < 32; sh++) {
      const uint64_t mm = DIV_ROUND_UP((uint64_t)1 << sh, d);
      if ((uint64_t)n_max * mm >= (1ull << 32))
         return false;

      bool exact = true;
      for (uint64_t n = 0; n <= n_max && exact; n++)
         exact = ((n * mm) >> sh) == n / d;
      if (exact) {
         *m = (uint32_t)mm;
         *s = sh;
         return true;
      }
   }
   return false;
}

/*
 * Global average pool over channel planes (NCHW, the NN cores' output layout) of hw uint8 values each, with TFLite's
 * uint8 rounding: out[c] = (sum of plane c + hw / 2) / hw (input and output share their quantization, so nothing is
 * requantized). One thread per 4 channels: 4 consecutive planes are 4 * hw contiguous bytes, i.e. hw whole words
 * starting on a word, and which channel each byte belongs to is known at compile time, so the loop is unrolled and
 * branch-free with no shared memory or barriers. A word whose 4 bytes are all in one channel is added two bytes at a
 * time into 16-bit lanes ((w & 0x00ff00ff) + ((w >> 8) & 0x00ff00ff)); bytes of words that straddle two channels go
 * into the low lane one by one. A lane sums at most hw bytes of its channel, so hw <= 256 can't overflow it. The
 * division is a multiply and shift (udiv_magic). Thread g writes channels 4g..4g+3 as one word, so the output is
 * exactly channels bytes. Kernel input: out, in.
 */
nir_shader *
etna_ml_shader_build_avgpool(const nir_shader_compiler_options *opts, unsigned hw, unsigned channels)
{
   const unsigned groups = channels / 4;
   uint32_t m;
   unsigned s;

   if (!hw || hw > ETNA_ML_AVGPOOL_MAX_HW || !channels || channels % 4 ||
       !udiv_magic(hw, 255 * hw + hw / 2, &m, &s))
      return NULL;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "etna_ml_avgpool");
   b.shader->info.workgroup_size_variable = true;

   nir_def *out_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = 8);
   nir_def *in_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 4), .base = 0, .range = 8);
   /* the ID clamped to the grid, so a wrong ID can only give wrong values, never a stray address */
   nir_def *g = nir_umin(&b, nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0), nir_imm_int(&b, groups - 1));
   nir_def *src = nir_iadd(&b, in_addr, nir_imul_imm(&b, g, 4 * hw));

   nir_def *acc[4] = { NULL };
   for (unsigned k = 0; k < hw; k++) {
      nir_def *word = nir_load_global(&b, 1, 32, k ? nir_iadd_imm(&b, src, 4 * k) : src, .align_mul = 4);
      const unsigned first = 4 * k / hw, last = (4 * k + 3) / hw;

      if (first == last) {
         nir_def *pairs = nir_iadd(&b, nir_iand_imm(&b, word, 0x00ff00ff),
                                   nir_iand_imm(&b, nir_ushr_imm(&b, word, 8), 0x00ff00ff));
         acc[first] = acc[first] ? nir_iadd(&b, acc[first], pairs) : pairs;
         continue;
      }

      for (unsigned i = 0; i < 4; i++) {
         const unsigned c = (4 * k + i) / hw;
         nir_def *v = nir_iand_imm(&b, i ? nir_ushr_imm(&b, word, 8 * i) : word, 0xff);
         acc[c] = acc[c] ? nir_iadd(&b, acc[c], v) : v;
      }
   }

   /* every channel has hw >= 1 bytes, so every acc[c] is set */
   nir_def *out_word = NULL;
   for (unsigned c = 0; c < 4; c++) {
      nir_def *sum = nir_iadd(&b, nir_iand_imm(&b, acc[c], 0xffff), nir_ushr_imm(&b, acc[c], 16));
      nir_def *q = nir_ushr_imm(&b, nir_imul_imm(&b, nir_iadd_imm(&b, sum, hw / 2), m), s);
      if (c)
         q = nir_ishl_imm(&b, q, 8 * c);
      out_word = out_word ? nir_ior(&b, out_word, q) : q;
   }
   nir_store_global(&b, out_word, nir_iadd(&b, out_addr, nir_ishl_imm(&b, g, 2)), .write_mask = 0x1,
                    .align_mul = 4);

   return b.shader;
}

/* The transpose kernel, compiled once per screen (subgraphs are created from one thread at a time). */
static struct etna_shader_variant *
transpose_kernel(struct etna_screen *screen)
{
   struct etna_ml_device *dev = &screen->ml_device;

   if (!dev->transpose_kernel) {
      const nir_shader_compiler_options *opts = etna_compiler_get_options(screen->compiler);
      dev->transpose_kernel = etna_compute_shader_create(screen, etna_ml_shader_build_transpose(opts), NULL);
      if (!dev->transpose_kernel)
         return NULL;
   }

   return etna_compute_shader_get_variant(dev->transpose_kernel, NULL);
}

static bool
compile_transpose(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
                  struct etna_vip_instruction *instruction)
{
   struct pipe_resource *input = etna_ml_get_resource(subgraph, operation->input_tensors[0]);
   struct pipe_resource *output = etna_ml_get_resource(subgraph, operation->output_tensors[0]);
   const unsigned in_offset = etna_ml_get_offset(subgraph, operation->input_tensors[0]);
   const unsigned out_offset = etna_ml_get_offset(subgraph, operation->output_tensors[0]);
   const unsigned c = operation->input_channels;
   const unsigned hw = operation->input_width * operation->input_height;

   /* word accesses: both tensors must start on a 4-byte boundary, and only plain RAM (not the AXI SRAM) */
   if (!input || !output || in_offset % 4 || out_offset % 4 ||
       input == subgraph->axi_sram || output == subgraph->axi_sram)
      return false;

   struct etna_shader_variant *v = transpose_kernel(subgraph->screen);
   if (!v)
      return false;

   memset(instruction, 0, sizeof(*instruction));
   instruction->type = ETNA_JOB_TYPE_SHADER;
   instruction->tp_type = operation->tp_type;
   instruction->shader = v;
   pipe_resource_reference(&instruction->input, input);
   pipe_resource_reference(&instruction->output, output);
   instruction->input_offset = in_offset;
   instruction->output_offset = out_offset;

   const uint32_t in_va = etna_bo_gpu_va(etna_buffer_resource(input)->bo) + in_offset;
   const uint32_t out_va = etna_bo_gpu_va(etna_buffer_resource(output)->bo) + out_offset;
   uint32_t *a = instruction->shader_args;

   a[0] = out_va;
   a[1] = in_va;
   if (operation->tp_type == ETNA_ML_TP_TRANSPOSE) {
      /* NHWC -> NCHW: read 4 pixels x 4 channels, write 4 channel planes x 4 pixels */
      a[2] = 4 * c; a[3] = 4; a[4] = c;
      a[5] = 4; a[6] = 4 * hw; a[7] = hw;
   } else {
      /* NCHW -> NHWC */
      a[2] = 4; a[3] = 4 * hw; a[4] = hw;
      a[5] = 4 * c; a[6] = 4; a[7] = c;
   }

   const unsigned gx = hw / 4;
   instruction->work_dim = 2;
   instruction->block[0] = block_width(gx);
   instruction->block[1] = instruction->block[2] = 1;
   instruction->grid[0] = gx / instruction->block[0];
   instruction->grid[1] = c / 4;
   instruction->grid[2] = 1;

   ML_DBG("shader %s: %ux%u channels %u, grid %ux%u of %ux1, in 0x%x out 0x%x\n",
          operation->tp_type == ETNA_ML_TP_TRANSPOSE ? "transpose" : "detranspose", operation->input_width,
          operation->input_height, c, instruction->grid[0], instruction->grid[1], instruction->block[0], in_va,
          out_va);

   return true;
}

bool
etna_ml_shader_softmax_enabled(const struct etna_screen *screen)
{
   return shader_option("softmax") && screen->base.caps.compute && !DBG_ENABLED(ETNA_DBG_NPU_PARALLEL);
}

/* Shapes and quantization the softmax kernel handles: 8-bit tensors, per-tensor quantization, output scale 1/256
 * with zero point 0 (uint8) or -128 (int8, stored +128 like every signed tensor), rows [N, 1, 1, C] or 2-D (the
 * spatial dimensions 1, so the NHWC/NCHW layout question doesn't arise). */
static bool
softmax_params(const struct pipe_ml_operation *poperation, unsigned *n, unsigned *rows, float *factor)
{
   const struct pipe_tensor *in = poperation->input_tensors[0], *out = poperation->output_tensors[0];

   if (poperation->input_count != 1 || poperation->output_count != 1 || in->data)
      return false;
   if (in->type_size != 1 || out->type_size != 1 || in->scales || out->scales)
      return false;
   if (fabsf(out->scale - 1.0f / 256.0f) > 1e-7f || out->zero_point + (out->is_signed ? 128 : 0) != 0)
      return false;
   if (in->dims[1] != 1 || in->dims[2] != 1 || memcmp(in->dims, out->dims, sizeof(in->dims)))
      return false;

   *n = in->dims[3];
   *rows = in->dims[0];
   *factor = poperation->softmax.beta * in->scale * (float)M_LOG2E;

   return *n >= 1 && *rows >= 1 && *rows <= 65535 && (*rows == 1 || *n % 4 == 0) && *factor > 0.0f;
}

/* The softmax kernel for a row of n with the most threads per workgroup (64, 32, 16) whose registers fit the
 * budget of barrier kernels (64 threads x 9 temps hung the core, incident 2026-11-13-2143-S6C-REGP6). Returns the
 * shader (caller owns it) with *variant and *threads set, or NULL: then the operation stays on the CPU. */
static struct etna_shader *
softmax_kernel(struct etna_screen *screen, unsigned n, unsigned rows, float factor,
               struct etna_shader_variant **variant, unsigned *threads)
{
   const nir_shader_compiler_options *opts = etna_compiler_get_options(screen->compiler);

   for (unsigned t = ETNA_ML_SOFTMAX_THREADS; t >= 16; t /= 2) {
      struct etna_shader *shader =
         etna_compute_shader_create(screen, etna_ml_shader_build_softmax(opts, n, rows, factor, t), NULL);
      struct etna_shader_variant *v = shader ? etna_compute_shader_get_variant(shader, NULL) : NULL;

      if (v && etna_compute_barrier_regs(v, t) <= etna_compute_barrier_reg_budget()) {
         *variant = v;
         *threads = t;
         return shader;
      }

      ML_DBG("softmax kernel with %u threads needs %u registers per workgroup, budget %u\n", t,
             v ? etna_compute_barrier_regs(v, t) : 0, etna_compute_barrier_reg_budget());
      if (shader)
         etna_compute_shader_destroy(shader);
   }

   ML_DBG("softmax kernel fits no workgroup size: left to the CPU\n");
   return NULL;
}

bool
etna_ml_shader_softmax_supported(struct etna_screen *screen, const struct pipe_ml_operation *poperation)
{
   unsigned n, rows;
   float factor;

   if (!etna_ml_shader_softmax_enabled(screen) || !softmax_params(poperation, &n, &rows, &factor))
      return false;

   /* compile it now: if it can't run (registers), the operation stays on the CPU instead of failing later */
   struct etna_shader_variant *v;
   unsigned threads;
   struct etna_shader *shader = softmax_kernel(screen, n, rows, factor, &v, &threads);
   if (!shader)
      return false;

   etna_compute_shader_destroy(shader);
   return true;
}

void
etna_ml_lower_softmax(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,
                      struct etna_operation *operation)
{
   ASSERTED bool ok = softmax_params(poperation, &operation->softmax_len, &operation->softmax_rows,
                                     &operation->softmax_factor);
   assert(ok);

   operation->type = ETNA_JOB_TYPE_SHADER;
   operation->shader_op = ETNA_ML_SHADER_SOFTMAX;
   operation->input_count = 1;
   operation->output_count = 1;
   operation->input_tensor_sizes[0] = operation->output_tensor_sizes[0] =
      operation->softmax_len * operation->softmax_rows;
   operation->input_channels = operation->output_channels = operation->softmax_len;
   operation->input_width = operation->input_height = 1;
   operation->output_width = operation->output_height = 1;
}

static bool
compile_softmax(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
                struct etna_vip_instruction *instruction)
{
   struct pipe_resource *input = etna_ml_get_resource(subgraph, operation->input_tensors[0]);
   struct pipe_resource *output = etna_ml_get_resource(subgraph, operation->output_tensors[0]);
   const unsigned in_offset = etna_ml_get_offset(subgraph, operation->input_tensors[0]);
   const unsigned out_offset = etna_ml_get_offset(subgraph, operation->output_tensors[0]);
   const unsigned n = operation->softmax_len, rows = operation->softmax_rows;

   if (!input || !output || in_offset % 4 || out_offset % 4 ||
       input == subgraph->axi_sram || output == subgraph->axi_sram)
      return false;

   /* the kernel writes whole words: up to 3 bytes past the last row, which must still be in the buffer */
   struct etna_bo *out_bo = etna_buffer_resource(output)->bo;
   if (etna_bo_size(out_bo) < out_offset + align(n * rows, 4))
      return false;

   struct etna_shader_variant *v;
   unsigned threads;
   struct etna_shader *shader = softmax_kernel(subgraph->screen, n, rows, operation->softmax_factor, &v, &threads);
   if (!shader)
      return false;

   memset(instruction, 0, sizeof(*instruction));
   instruction->type = ETNA_JOB_TYPE_SHADER;
   instruction->owned_shader = shader;
   instruction->shader = v;
   pipe_resource_reference(&instruction->input, input);
   pipe_resource_reference(&instruction->output, output);
   instruction->input_offset = in_offset;
   instruction->output_offset = out_offset;
   instruction->shader_args[0] = etna_bo_gpu_va(out_bo) + out_offset;
   instruction->shader_args[1] = etna_bo_gpu_va(etna_buffer_resource(input)->bo) + in_offset;
   instruction->work_dim = 1;
   instruction->block[0] = threads;
   instruction->block[1] = instruction->block[2] = 1;
   instruction->grid[0] = rows;
   instruction->grid[1] = instruction->grid[2] = 1;

   ML_DBG("shader softmax: %u rows of %u, factor %f, %u threads, %u instructions, %u temps\n", rows, n,
          operation->softmax_factor, threads, v->code_size / 4, v->num_temps + 1);
   return true;
}

static bool
avgpool_enabled(const struct etna_screen *screen)
{
   return shader_option("avgpool") && screen->base.caps.compute && !DBG_ENABLED(ETNA_DBG_NPU_PARALLEL);
}

/* The average pools the kernel handles: global ones (the filter covers the whole input, no padding, so the output
 * is 1x1), batch 1, 8-bit unsigned tensors with the same per-tensor quantization in and out (the result is then the
 * integer mean, TFLite's (sum + hw/2) / hw), no fused activation, channels % 4 == 0, at most ETNA_ML_AVGPOOL_MAX_HW
 * values per channel. MobileNet's 7x7x1024 pool is one; it used to split the graph into two NPU submits. */
static bool
avgpool_params(const struct pipe_ml_operation *poperation, unsigned *hw, unsigned *channels)
{
   const struct pipe_tensor *in = poperation->input_tensors[0], *out = poperation->output_tensors[0];

   if (poperation->type != PIPE_ML_OPERATION_TYPE_POOLING || poperation->pooling.type != PIPE_ML_POOLING_TYPE_AVG ||
       poperation->pooling.fused_activation)
      return false;
   if (poperation->input_count != 1 || poperation->output_count != 1 || in->data)
      return false;
   if (in->type_size != 1 || out->type_size != 1 || in->is_signed || out->is_signed || in->scales ||
       out->scales || in->zero_points || out->zero_points)
      return false;
   if (in->scale != out->scale || in->zero_point != out->zero_point)
      return false;
   if (in->dims[0] != 1 || out->dims[0] != 1 || out->dims[1] != 1 || out->dims[2] != 1 ||
       out->dims[3] != in->dims[3])
      return false;
   if (poperation->pooling.filter_height != in->dims[1] || poperation->pooling.filter_width != in->dims[2] ||
       poperation->pooling.padding_top || poperation->pooling.padding_bottom || poperation->pooling.padding_left ||
       poperation->pooling.padding_right)
      return false;

   *hw = in->dims[1] * in->dims[2];
   *channels = in->dims[3];
   const unsigned groups = *channels / 4;

   return *hw >= 1 && *hw <= ETNA_ML_AVGPOOL_MAX_HW && *channels >= 4 && *channels % 4 == 0 &&
          groups / block_width(groups) <= 65535;
}

/* The kernel for this pool (caller owns the shader), or NULL */
static struct etna_shader *
avgpool_kernel(struct etna_screen *screen, unsigned hw, unsigned channels, struct etna_shader_variant **variant)
{
   const nir_shader_compiler_options *opts = etna_compiler_get_options(screen->compiler);
   nir_shader *nir = etna_ml_shader_build_avgpool(opts, hw, channels);
   if (!nir)
      return NULL;

   struct etna_shader *shader = etna_compute_shader_create(screen, nir, NULL);
   struct etna_shader_variant *v = shader ? etna_compute_shader_get_variant(shader, NULL) : NULL;
   if (!v) {
      ML_DBG("average pool kernel %u x %u channels did not compile: left to the CPU\n", hw, channels);
      if (shader)
         etna_compute_shader_destroy(shader);
      return NULL;
   }

   *variant = v;
   return shader;
}

bool
etna_ml_shader_avgpool_supported(struct etna_screen *screen, const struct pipe_ml_operation *poperation)
{
   unsigned hw, channels;

   if (!avgpool_enabled(screen) || !avgpool_params(poperation, &hw, &channels))
      return false;

   /* compile it now: if it can't, the operation stays on the CPU instead of failing later */
   struct etna_shader_variant *v;
   struct etna_shader *shader = avgpool_kernel(screen, hw, channels, &v);
   if (!shader)
      return false;

   etna_compute_shader_destroy(shader);
   return true;
}

void
etna_ml_lower_avgpool(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,
                      struct etna_operation *operation)
{
   const struct pipe_tensor *in = poperation->input_tensors[0];
   unsigned hw, channels;
   ASSERTED bool ok = avgpool_params(poperation, &hw, &channels);
   assert(ok);

   operation->type = ETNA_JOB_TYPE_SHADER;
   operation->shader_op = ETNA_ML_SHADER_AVGPOOL;
   operation->input_count = 1;
   operation->output_count = 1;
   operation->input_width = in->dims[1];
   operation->input_height = in->dims[2];
   operation->input_channels = operation->output_channels = channels;
   operation->output_width = operation->output_height = 1;
   operation->input_tensor_sizes[0] = hw * channels;
   operation->output_tensor_sizes[0] = channels;
}

static bool
compile_avgpool(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
                struct etna_vip_instruction *instruction)
{
   struct pipe_resource *input = etna_ml_get_resource(subgraph, operation->input_tensors[0]);
   struct pipe_resource *output = etna_ml_get_resource(subgraph, operation->output_tensors[0]);
   const unsigned in_offset = etna_ml_get_offset(subgraph, operation->input_tensors[0]);
   const unsigned out_offset = etna_ml_get_offset(subgraph, operation->output_tensors[0]);
   const unsigned hw = operation->input_width * operation->input_height, channels = operation->input_channels;

   if (!input || !output || in_offset % 4 || out_offset % 4 ||
       input == subgraph->axi_sram || output == subgraph->axi_sram)
      return false;

   struct etna_bo *in_bo = etna_buffer_resource(input)->bo, *out_bo = etna_buffer_resource(output)->bo;
   if (etna_bo_size(in_bo) < in_offset + hw * channels || etna_bo_size(out_bo) < out_offset + channels)
      return false;

   struct etna_shader_variant *v;
   struct etna_shader *shader = avgpool_kernel(subgraph->screen, hw, channels, &v);
   if (!shader)
      return false;

   const unsigned groups = channels / 4;
   memset(instruction, 0, sizeof(*instruction));
   instruction->type = ETNA_JOB_TYPE_SHADER;
   instruction->owned_shader = shader;
   instruction->shader = v;
   pipe_resource_reference(&instruction->input, input);
   pipe_resource_reference(&instruction->output, output);
   instruction->input_offset = in_offset;
   instruction->output_offset = out_offset;
   instruction->shader_args[0] = etna_bo_gpu_va(out_bo) + out_offset;
   instruction->shader_args[1] = etna_bo_gpu_va(in_bo) + in_offset;
   instruction->work_dim = 1;
   instruction->block[0] = block_width(groups);
   instruction->block[1] = instruction->block[2] = 1;
   instruction->grid[0] = groups / instruction->block[0];
   instruction->grid[1] = instruction->grid[2] = 1;

   ML_DBG("shader avgpool: %ux%u channels %u, grid %u of %u, %u instructions, %u temps\n", operation->input_width,
          operation->input_height, channels, instruction->grid[0], instruction->block[0], v->code_size / 4,
          v->num_temps + 1);
   return true;
}

bool
etna_ml_compile_operation_shader(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
                                 struct etna_vip_instruction *instruction)
{
   switch (operation->shader_op) {
   case ETNA_ML_SHADER_TRANSPOSE:
      return compile_transpose(subgraph, operation, instruction);
   case ETNA_ML_SHADER_SOFTMAX:
      return compile_softmax(subgraph, operation, instruction);
   case ETNA_ML_SHADER_AVGPOOL:
      return compile_avgpool(subgraph, operation, instruction);
   }
   return false;
}

void
etna_ml_shader_instruction_fini(struct etna_vip_instruction *instruction)
{
   if (instruction->owned_shader) {
      etna_compute_shader_destroy(instruction->owned_shader);
      instruction->owned_shader = NULL;
   }
}

/* Caches the NN and TP units write through, flushed before a grid reads their output and after it, so the next
 * NN/TP job sees what the grid wrote. Same bits as close_batch in etnaviv_ml.c (non-parallel mode).
 */
static void
flush_and_stall(struct etna_cmd_stream *stream)
{
   const uint32_t cache = VIVS_GL_FLUSH_CACHE_DEPTH | VIVS_GL_FLUSH_CACHE_COLOR | VIVS_GL_FLUSH_CACHE_UNK10 |
                          VIVS_GL_FLUSH_CACHE_UNK11 | VIVS_GL_FLUSH_CACHE_SHADER_L1;

   etna_set_state(stream, VIVS_GL_FLUSH_CACHE, cache);
   etna_stall(stream, SYNC_RECIPIENT_FE, SYNC_RECIPIENT_PE);
}

void
etna_ml_emit_operation_shader(struct pipe_context *pctx, struct etna_ml_subgraph *subgraph,
                              struct etna_vip_instruction *operation, unsigned idx)
{
   struct etna_context *ctx = etna_context(pctx);
   const struct pipe_grid_info info = {
      .work_dim = operation->work_dim,
      .block = { operation->block[0], operation->block[1], operation->block[2] },
      .grid = { operation->grid[0], operation->grid[1], operation->grid[2] },
   };
   struct pipe_resource *buffers[2] = { operation->input, operation->output };

   flush_and_stall(ctx->stream);
   if (!etna_compute_emit_grid(ctx, operation->shader, &info, operation->shader_args,
                               sizeof(operation->shader_args), buffers, 2, false))
      BUG("ML shader operation %u could not be emitted", idx);
   flush_and_stall(ctx->stream);
}

void
etna_ml_shader_screen_fini(struct etna_screen *screen)
{
   if (screen->ml_device.transpose_kernel) {
      etna_compute_shader_destroy(screen->ml_device.transpose_kernel);
      screen->ml_device.transpose_kernel = NULL;
   }
}
