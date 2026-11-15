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
static nir_def *mean_requant_nir(nir_builder *b, nir_def *sum, nir_def *table_addr,
                                 const struct etna_ml_mean_requant *rq);

/* The pool kernel below; with rq (MEAN over the spatial dimensions) each channel's sum goes through MEAN's
 * requantization table instead of the division, and the kernel input has a third address (the table). */
static nir_shader *
build_pool(const nir_shader_compiler_options *opts, unsigned hw, unsigned channels,
           const struct etna_ml_mean_requant *rq)
{
   const unsigned groups = channels / 4;
   const unsigned range = rq ? 12 : 8;
   uint32_t m = 1;
   unsigned s = 0;

   if (!hw || hw > ETNA_ML_AVGPOOL_MAX_HW || !channels || channels % 4 ||
       (!rq && !udiv_magic(hw, 255 * hw + hw / 2, &m, &s)))
      return NULL;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, rq ? "etna_ml_mean_spatial" : "etna_ml_avgpool");
   b.shader->info.workgroup_size_variable = true;

   nir_def *out_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 0), .base = 0, .range = range);
   nir_def *in_addr = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 4), .base = 0, .range = range);
   nir_def *table_addr = rq ? nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, 8), .base = 0, .range = range) : NULL;
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
      nir_def *q = rq ? mean_requant_nir(&b, sum, table_addr, rq)
                      : nir_ushr_imm(&b, nir_imul_imm(&b, nir_iadd_imm(&b, sum, hw / 2), m), s);
      if (c)
         q = nir_ishl_imm(&b, q, 8 * c);
      out_word = out_word ? nir_ior(&b, out_word, q) : q;
   }
   nir_store_global(&b, out_word, nir_iadd(&b, out_addr, nir_ishl_imm(&b, g, 2)), .write_mask = 0x1,
                    .align_mul = 4);

   return b.shader;
}

nir_shader *
etna_ml_shader_build_avgpool(const nir_shader_compiler_options *opts, unsigned hw, unsigned channels)
{
   return build_pool(opts, hw, channels, NULL);
}

/*
 * MEAN (step 6e). The CPU path beside Teflon is XNNPACK (its leftover ops run there, not on TFLite's builtin
 * kernels), and its quantized mean is, with S the int32 sum of the n values in the tensors' own domain:
 *    s_eff = fl(fl(in_scale / out_scale) * fl(1 / n)),  zp' = fma(-(n * in_zp), s_eff, out_zp),
 *    q = clamp(nearbyint(fma(S, s_eff, zp')))
 * (XNNPACK's reduce-nd.c + reference convert; the two multiply-adds are fused in the aarch64 build: fitted exact on
 * 205200 outputs, research/port/tools/mean_fit_xnnpack.py).
 */
#define ETNA_ML_MEAN_MAX_N 65535

int
etna_ml_mean_xnnpack(int64_t s, unsigned n, float in_scale, int in_zp, float out_scale, int out_zp, bool is_signed)
{
   const float io = in_scale / out_scale;
   const float inv_n = 1.0f / (float)n;
   const float s_eff = io * inv_n;
   const float zp = fmaf(-(float)((int32_t)n * in_zp), s_eff, (float)out_zp);
   const float v = nearbyintf(fmaf((float)s, s_eff, zp));
   const int lo = is_signed ? -128 : 0, hi = is_signed ? 127 : 255;

   return v < lo ? lo : v > hi ? hi : (int)v;
}

/* Builds and checks the table and the estimate over every sum of n stored bytes (etnaviv stores int8 +128, so S is
 * unsigned for both types and the stored output is q + 128 for int8). */
bool
etna_ml_mean_requant(unsigned n, float in_scale, int in_zp, float out_scale, int out_zp, bool is_signed,
                     struct etna_ml_mean_requant *rq)
{
   const int lo = is_signed ? -128 : 0;
   const uint32_t s_max = 255 * n;

   if (!n || n > ETNA_ML_MEAN_MAX_N || !(in_scale > 0.0f) || !(out_scale > 0.0f))
      return false;

   /* the table: stored output as a step function of S, which must not decrease */
   unsigned k = 1;
   int prev = 0;
   rq->table[0] = 0;
   for (uint32_t s = 0; s <= s_max; s++) {
      const int q = etna_ml_mean_xnnpack((int64_t)s + (int64_t)n * lo, n, in_scale, in_zp, out_scale, out_zp,
                                         is_signed) - lo;
      if (q < prev)
         return false;
      prev = q;
      while (k <= 255 && (int)k <= q)
         rq->table[k++] = s;
   }
   for (; k <= 256; k++)
      rq->table[k] = UINT32_MAX;

   /* the estimate: the straight line through the steps in the stored domain (S = sum of stored bytes = the own-domain
    * sum - n * lo, output stored = q - lo), stored ~ S * slope + zp, rounded: (S * m + b) >> sh in int32 */
   const double slope = (double)(in_scale / out_scale) / n;
   const double zp = (double)out_zp - (double)n * in_zp * slope - lo + (double)n * lo * slope;
   unsigned sh = 30;
   for (;; sh--) {
      const double m = floor(slope * (double)(1u << sh) + 0.5);
      const double b = floor((zp + 0.5) * (double)(1u << sh));
      if (m >= 1.0 && (double)s_max * m + fabs(b) < 2147483647.0) {
         rq->m = (int32_t)m;
         rq->b = (int32_t)b;
         rq->sh = sh;
         break;
      }
      if (sh == 0)
         return false;
   }

   /* the kernel's correction is exact only if the estimate is never more than one step off */
   for (uint32_t s = 0; s <= s_max; s++) {
      int64_t e = (int64_t)s * rq->m + rq->b;
      int q0 = (int)(e >> rq->sh); /* arithmetic shift, as the kernel's ishr */
      q0 = q0 < 0 ? 0 : q0 > 255 ? 255 : q0;
      const int q = q0 + (s >= rq->table[q0 + 1]) - (s < rq->table[q0]);
      if (q < 0 || q > 255 || q != etna_ml_mean_xnnpack((int64_t)s + (int64_t)n * lo, n, in_scale, in_zp,
                                                         out_scale, out_zp, is_signed) - lo)
         return false;
   }
   return true;
}

/* The stored output byte for one sum (see struct etna_ml_mean_requant): the estimate, then two table loads. */
static nir_def *
mean_requant_nir(nir_builder *b, nir_def *sum, nir_def *table_addr, const struct etna_ml_mean_requant *rq)
{
   nir_def *e = nir_iadd_imm(b, nir_imul_imm(b, sum, rq->m), rq->b);
   nir_def *q0 = nir_imin(b, nir_imax(b, nir_ishr_imm(b, e, rq->sh), nir_imm_int(b, 0)), nir_imm_int(b, 255));
   nir_def *t = nir_iadd(b, table_addr, nir_ishl_imm(b, q0, 2));
   nir_def *t_lo = nir_load_global(b, 1, 32, t, .align_mul = 4);
   nir_def *t_hi = nir_load_global(b, 1, 32, nir_iadd_imm(b, t, 4), .align_mul = 4);
   nir_def *up = nir_bcsel(b, nir_uge(b, sum, t_hi), nir_imm_int(b, 1), nir_imm_int(b, 0));
   nir_def *down = nir_bcsel(b, nir_ult(b, sum, t_lo), nir_imm_int(b, 1), nir_imm_int(b, 0));
   return nir_isub(b, nir_iadd(b, q0, up), down);
}

nir_shader *
etna_ml_shader_build_mean_spatial(const nir_shader_compiler_options *opts, unsigned hw, unsigned channels,
                                  const struct etna_ml_mean_requant *rq)
{
   return rq ? build_pool(opts, hw, channels, rq) : NULL;
}

/*
 * MEAN over the channels (the reduction in a layernorm) of NCHW planes, positions = H*W bytes each, positions % 4 ==
 * 0: thread t owns positions 4t..4t+3, i.e. word t of every plane, and loops over the planes (one word load, four
 * byte sums per channel; the first kernel that loops, proven by S6E-LOOP*). The trip count and the plane length
 * are kernel arguments, so the loop isn't unrolled; the count is capped inside the kernel. Each sum is
 * requantized through the table. Kernel input: out, in, table, channels, plane bytes, last thread index.
 */
nir_shader *
etna_ml_shader_build_mean_channels(const nir_shader_compiler_options *opts, const struct etna_ml_mean_requant *rq)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_KERNEL, opts, "etna_ml_mean_channels");
   b.shader->info.workgroup_size_variable = true;

   nir_def *arg[6];
   for (unsigned i = 0; i < 6; i++)
      arg[i] = nir_load_kernel_input(&b, 1, 32, nir_imm_int(&b, i * 4), .base = 0, .range = 24);
   nir_def *channels = nir_umin(&b, arg[3], nir_imm_int(&b, ETNA_ML_MEAN_MAX_N));
   /* the ID clamped to the grid, so a wrong ID can only give wrong values, never a stray address */
   nir_def *t = nir_umin(&b, nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0), arg[5]);

   nir_variable *sum_v[4], *k_v, *addr_v;
   for (unsigned i = 0; i < 4; i++) {
      sum_v[i] = nir_local_variable_create(b.impl, glsl_uint_type(), "sum");
      nir_store_var(&b, sum_v[i], nir_imm_int(&b, 0), 0x1);
   }
   k_v = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
   addr_v = nir_local_variable_create(b.impl, glsl_uint_type(), "addr");
   nir_store_var(&b, k_v, nir_imm_int(&b, 0), 0x1);
   nir_store_var(&b, addr_v, nir_iadd(&b, arg[1], nir_ishl_imm(&b, t, 2)), 0x1);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, k_v);
      nir_push_if(&b, nir_uge(&b, k, channels));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_def *addr = nir_load_var(&b, addr_v);
      nir_def *w = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
      for (unsigned i = 0; i < 4; i++) {
         nir_def *v = nir_iand_imm(&b, i ? nir_ushr_imm(&b, w, 8 * i) : w, 0xff);
         nir_store_var(&b, sum_v[i], nir_iadd(&b, nir_load_var(&b, sum_v[i]), v), 0x1);
      }
      nir_store_var(&b, addr_v, nir_iadd(&b, addr, arg[4]), 0x1);
      nir_store_var(&b, k_v, nir_iadd_imm(&b, k, 1), 0x1);
   }
   nir_pop_loop(&b, loop);

   nir_def *out_word = NULL;
   for (unsigned i = 0; i < 4; i++) {
      nir_def *q = mean_requant_nir(&b, nir_load_var(&b, sum_v[i]), arg[2], rq);
      if (i)
         q = nir_ishl_imm(&b, q, 8 * i);
      out_word = out_word ? nir_ior(&b, out_word, q) : q;
   }
   nir_store_global(&b, out_word, nir_iadd(&b, arg[0], nir_ishl_imm(&b, t, 2)), .write_mask = 0x1, .align_mul = 4);

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

static bool
mean_enabled(const struct etna_screen *screen)
{
   return shader_option("mean") && screen->base.caps.compute && !DBG_ENABLED(ETNA_DBG_NPU_PARALLEL);
}

/* The MEANs the kernels handle: 8-bit per-tensor quantization (same signedness in and out), batch 1, over the
 * channels (axes 0x8: positions = H*W, a multiple of 4, up to ETNA_ML_MEAN_MAX_N channels) or over the spatial
 * dimensions (axes 0x6: at most ETNA_ML_AVGPOOL_MAX_HW values per channel, channels % 4 == 0). keep_dims doesn't
 * matter: the output holds the same bytes either way. */
static bool
mean_params(const struct pipe_ml_operation *poperation, unsigned *n, unsigned *threads)
{
   const struct pipe_tensor *in = poperation->input_tensors[0], *out = poperation->output_tensors[0];

   if (poperation->type != PIPE_ML_OPERATION_TYPE_MEAN || poperation->output_count != 1 || in->data)
      return false;
   if (in->type_size != 1 || out->type_size != 1 || in->is_signed != out->is_signed || in->scales || out->scales ||
       in->zero_points || out->zero_points || !(in->scale > 0.0f) || !(out->scale > 0.0f))
      return false;
   if (in->dims[0] != 1)
      return false;

   const unsigned hw = in->dims[1] * in->dims[2], c = in->dims[3];
   const unsigned out_elems = out->dims[0] * out->dims[1] * out->dims[2] * out->dims[3];
   unsigned groups;

   if (poperation->reduce.axes == 0x8) {
      if (!hw || hw % 4 || !c || c > ETNA_ML_MEAN_MAX_N || out_elems != hw)
         return false;
      *n = c;
      groups = hw / 4;
   } else if (poperation->reduce.axes == 0x6) {
      if (!hw || hw > ETNA_ML_AVGPOOL_MAX_HW || c < 4 || c % 4 || out_elems != c)
         return false;
      *n = hw;
      groups = c / 4;
   } else {
      return false;
   }

   *threads = groups;
   return groups / block_width(groups) <= 65535;
}

/* The kernel for this MEAN (caller owns the shader), with its requantization in *rq, or NULL */
static struct etna_shader *
mean_kernel(struct etna_screen *screen, unsigned axes, unsigned hw,
            unsigned channels, float in_scale, int in_zp, float out_scale, int out_zp, bool is_signed,
            struct etna_ml_mean_requant *rq, struct etna_shader_variant **variant)
{
   const unsigned n = axes == 0x8 ? channels : hw;

   if (!etna_ml_mean_requant(n, in_scale, in_zp, out_scale, out_zp, is_signed, rq)) {
      ML_DBG("mean: no exact requantization for n %u, scales %g -> %g: left to the CPU\n", n, in_scale, out_scale);
      return NULL;
   }

   const nir_shader_compiler_options *opts = etna_compiler_get_options(screen->compiler);
   nir_shader *nir = axes == 0x8 ? etna_ml_shader_build_mean_channels(opts, rq)
                                 : etna_ml_shader_build_mean_spatial(opts, hw, channels, rq);
   if (!nir)
      return NULL;

   struct etna_shader *shader = etna_compute_shader_create(screen, nir, NULL);
   struct etna_shader_variant *v = shader ? etna_compute_shader_get_variant(shader, NULL) : NULL;
   if (!v) {
      ML_DBG("mean kernel did not compile: left to the CPU\n");
      if (shader)
         etna_compute_shader_destroy(shader);
      return NULL;
   }

   *variant = v;
   return shader;
}

bool
etna_ml_shader_mean_supported(struct etna_screen *screen, const struct pipe_ml_operation *poperation)
{
   const struct pipe_tensor *in = poperation->input_tensors[0], *out = poperation->output_tensors[0];
   unsigned n, threads;

   if (!mean_enabled(screen) || !mean_params(poperation, &n, &threads))
      return false;

   /* build the table and compile now: if either fails, the operation stays on the CPU instead of failing later */
   struct etna_ml_mean_requant *rq = malloc(sizeof(*rq));
   struct etna_shader_variant *v;
   struct etna_shader *shader = rq ? mean_kernel(screen, poperation->reduce.axes,
                                                 in->dims[1] * in->dims[2], in->dims[3], in->scale, in->zero_point,
                                                 out->scale, out->zero_point, in->is_signed, rq, &v) : NULL;
   free(rq);
   if (!shader)
      return false;

   etna_compute_shader_destroy(shader);
   return true;
}

void
etna_ml_lower_mean(struct etna_ml_subgraph *subgraph, const struct pipe_ml_operation *poperation,
                   struct etna_operation *operation)
{
   const struct pipe_tensor *in = poperation->input_tensors[0], *out = poperation->output_tensors[0];
   unsigned n, threads;
   ASSERTED bool ok = mean_params(poperation, &n, &threads);
   assert(ok);

   const unsigned hw = in->dims[1] * in->dims[2], c = in->dims[3];
   operation->type = ETNA_JOB_TYPE_SHADER;
   operation->shader_op = ETNA_ML_SHADER_MEAN;
   operation->input_count = 1;
   operation->output_count = 1;
   operation->input_width = in->dims[1];
   operation->input_height = in->dims[2];
   operation->input_channels = c;
   operation->input_tensor_sizes[0] = hw * c;
   operation->output_tensor_sizes[0] = poperation->reduce.axes == 0x8 ? hw : c;
   operation->output_width = poperation->reduce.axes == 0x8 ? in->dims[1] : 1;
   operation->output_height = poperation->reduce.axes == 0x8 ? in->dims[2] : 1;
   operation->output_channels = poperation->reduce.axes == 0x8 ? 1 : c;
   operation->mean_axes = poperation->reduce.axes;
   operation->mean_in_scale = in->scale;
   operation->mean_out_scale = out->scale;
   operation->mean_in_zp = in->zero_point;
   operation->mean_out_zp = out->zero_point;
   operation->mean_signed = in->is_signed;
}

static bool
compile_mean(struct etna_ml_subgraph *subgraph, const struct etna_operation *operation,
             struct etna_vip_instruction *instruction)
{
   struct pipe_resource *input = etna_ml_get_resource(subgraph, operation->input_tensors[0]);
   struct pipe_resource *output = etna_ml_get_resource(subgraph, operation->output_tensors[0]);
   const unsigned in_offset = etna_ml_get_offset(subgraph, operation->input_tensors[0]);
   const unsigned out_offset = etna_ml_get_offset(subgraph, operation->output_tensors[0]);
   const unsigned hw = operation->input_width * operation->input_height, c = operation->input_channels;
   const bool channel_mean = operation->mean_axes == 0x8;

   if (!input || !output || in_offset % 4 || out_offset % 4 ||
       input == subgraph->axi_sram || output == subgraph->axi_sram)
      return false;

   struct etna_bo *in_bo = etna_buffer_resource(input)->bo, *out_bo = etna_buffer_resource(output)->bo;
   if (etna_bo_size(in_bo) < in_offset + hw * c ||
       etna_bo_size(out_bo) < out_offset + operation->output_tensor_sizes[0])
      return false;

   struct etna_ml_mean_requant *rq = malloc(sizeof(*rq));
   struct etna_shader_variant *v;
   struct etna_shader *shader = rq ? mean_kernel(subgraph->screen, operation->mean_axes, hw, c,
                                                 operation->mean_in_scale, operation->mean_in_zp,
                                                 operation->mean_out_scale, operation->mean_out_zp,
                                                 operation->mean_signed, rq, &v) : NULL;
   if (!shader) {
      free(rq);
      return false;
   }

   /* the thresholds, read by the kernel as a third buffer */
   struct pipe_resource *table = etna_ml_create_resource(&subgraph->screen->base, sizeof(rq->table));
   if (!table) {
      etna_compute_shader_destroy(shader);
      free(rq);
      return false;
   }
   struct etna_bo *table_bo = etna_buffer_resource(table)->bo;
   etna_bo_cpu_prep(table_bo, DRM_ETNA_PREP_WRITE);
   memcpy(etna_bo_map(table_bo), rq->table, sizeof(rq->table));
   etna_bo_cpu_fini(table_bo);

   const unsigned groups = channel_mean ? hw / 4 : c / 4;
   memset(instruction, 0, sizeof(*instruction));
   instruction->type = ETNA_JOB_TYPE_SHADER;
   instruction->owned_shader = shader;
   instruction->shader = v;
   instruction->table = table; /* the reference from etna_ml_create_resource */
   pipe_resource_reference(&instruction->input, input);
   pipe_resource_reference(&instruction->output, output);
   instruction->input_offset = in_offset;
   instruction->output_offset = out_offset;
   instruction->shader_args[0] = etna_bo_gpu_va(out_bo) + out_offset;
   instruction->shader_args[1] = etna_bo_gpu_va(in_bo) + in_offset;
   instruction->shader_args[2] = etna_bo_gpu_va(table_bo);
   if (channel_mean) {
      instruction->shader_args[3] = c;      /* trip count */
      instruction->shader_args[4] = hw;     /* plane bytes */
      instruction->shader_args[5] = groups - 1;
   }
   instruction->work_dim = 1;
   instruction->block[0] = block_width(groups);
   instruction->block[1] = instruction->block[2] = 1;
   instruction->grid[0] = groups / instruction->block[0];
   instruction->grid[1] = instruction->grid[2] = 1;

   ML_DBG("shader mean %s: %ux%u channels %u, n %u, estimate (S*%d%+d)>>%u, grid %u of %u, %u instructions, %u temps\n",
          channel_mean ? "channels" : "spatial", operation->input_width, operation->input_height, c,
          channel_mean ? c : hw, rq->m, rq->b, rq->sh, instruction->grid[0], instruction->block[0], v->code_size / 4,
          v->num_temps + 1);
   free(rq);
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
   case ETNA_ML_SHADER_MEAN:
      return compile_mean(subgraph, operation, instruction);
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
   pipe_resource_reference(&instruction->table, NULL);
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
   struct pipe_resource *buffers[3] = { operation->input, operation->output, operation->table };

   flush_and_stall(ctx->stream);
   if (!etna_compute_emit_grid(ctx, operation->shader, &info, operation->shader_args,
                               sizeof(operation->shader_args), buffers, operation->table ? 3 : 2, false))
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
