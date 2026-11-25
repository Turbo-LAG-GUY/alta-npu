/*
 * jb_glue.c (2026-11-23, LLM engine): the CPU side of a prompt chunk in C (NEON + OpenMP), called from
 * llm/l3/chunk_glue.py --glue c through ctypes. Everything between the NPU's linear steps: RMSNorm, quantize +
 * transpose into the NPU's planar int8 input, dequantize + scatter the jobs' outputs back to model row order (adding
 * the partials of a split step), RoPE, causal attention (grouped KV heads), SiLU * up, the randomized block Hadamard
 * on the down input, the residual's randomized Hadamard (power-of-2 widths) and the position-0 int8 GEMV straight
 * from the packs' int8 weights (no fp32 copy of the weights: 6.8 GB for the 1.7B).
 * float32 throughout like the numpy reference (llm/ref/smollm.py); reductions that numpy does in float64 (RMSNorm's
 * mean square) are float64 here too. Rounding to levels = rintf (round half to even, as np.round).
 * Build: llm/l3/glue/build.sh  ->  npu-llm/bin/libjbglue.so
 */
#include <arm_neon.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

static double
now_s(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec + ts.tv_nsec * 1e-9;
}

#define EXPORT __attribute__((visibility("default")))

static inline float
dot_f32(const float *a, const float *b, int n)
{
   float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
   int i = 0;
   for (; i + 8 <= n; i += 8) {
      s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
      s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
   }
   float s = vaddvq_f32(vaddq_f32(s0, s1));
   for (; i < n; i++)
      s += a[i] * b[i];
   return s;
}

/* exp(x) for x <= 0 (softmax after subtracting the max), 4 lanes: 2^n * p(r), r in [-ln2/2, ln2/2], degree-6
 * polynomial (Cephes expf coefficients), relative error ~1.2e-7; x < -87 -> 0 */
static inline float32x4_t
exp_neg_f32(float32x4_t x)
{
   const float32x4_t lo = vdupq_n_f32(-87.3f), log2e = vdupq_n_f32(1.44269504088896341f);
   const float32x4_t c1 = vdupq_n_f32(0.693359375f), c2 = vdupq_n_f32(-2.12194440e-4f);
   float32x4_t z = vmaxq_f32(x, lo);
   float32x4_t n = vrndnq_f32(vmulq_f32(z, log2e));
   float32x4_t r = vsubq_f32(vsubq_f32(z, vmulq_f32(n, c1)), vmulq_f32(n, c2));
   float32x4_t p = vdupq_n_f32(1.9875691500e-4f);
   p = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), p, r);
   p = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), p, r);
   p = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), p, r);
   p = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), p, r);
   p = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), p, r);
   float32x4_t r2 = vmulq_f32(r, r);
   p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), p, r2);
   int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
   float32x4_t y = vmulq_f32(p, vreinterpretq_f32_s32(e));
   return vbslq_f32(vcltq_f32(x, lo), vdupq_n_f32(0), y);
}

/* exp(x) for any x, 4 lanes (clamped to [-87.3, 88.7]: 0 / ~3e38 at the ends), same polynomial as exp_neg_f32 */
static inline float32x4_t
exp_f32(float32x4_t x)
{
   x = vminq_f32(x, vdupq_n_f32(88.7f));
   return exp_neg_f32(x);
}

/* out[t] = x[t] / sqrt(mean(x[t]^2) + eps) (* w if w); the mean square in float64 like numpy, then a float32
 * multiply by the reciprocal (numpy divides in float64: this can differ in the last bit, rarely) */
EXPORT void
jbg_rmsnorm(const float *x, const float *w, float *out, int T, int D, float eps)
{
#pragma omp parallel for schedule(dynamic)
   for (int t = 0; t < T; t++) {
      const float *r = x + (size_t)t * D;
      float64x2_t a0 = vdupq_n_f64(0), a1 = a0;
      int i = 0;
      for (; i + 4 <= D; i += 4) {
         float32x4_t v = vld1q_f32(r + i);
         float64x2_t lo = vcvt_f64_f32(vget_low_f32(v)), hi = vcvt_high_f64_f32(v);
         a0 = vfmaq_f64(a0, lo, lo);
         a1 = vfmaq_f64(a1, hi, hi);
      }
      double s = vaddvq_f64(vaddq_f64(a0, a1));
      for (; i < D; i++)
         s += (double)r[i] * r[i];
      float inv = (float)(1.0 / sqrt(s / D + eps));
      float32x4_t iv = vdupq_n_f32(inv);
      float *o = out + (size_t)t * D;
      i = 0;
      if (w) {
         for (; i + 4 <= D; i += 4)
            vst1q_f32(o + i, vmulq_f32(vmulq_f32(vld1q_f32(r + i), iv), vld1q_f32(w + i)));
         for (; i < D; i++)
            o[i] = r[i] * inv * w[i];
      } else {
         for (; i + 4 <= D; i += 4)
            vst1q_f32(o + i, vmulq_f32(vld1q_f32(r + i), iv));
         for (; i < D; i++)
            o[i] = r[i] * inv;
      }
   }
}

/* x [T, IC] float -> out [IC, T] int8 levels clip(rint(x / s) + zp) (the NPU's planar input). Exact like numpy
 * (vector divide, round half to even). Per block of 16 tokens x 16 channels: quantize 16 token rows, transpose the
 * 16 x 16 bytes in registers, store 16 channel rows. */
static inline void tr16x16(uint8x16_t r[16]);
static inline void fwht_row(float *r, int n, int b, const float *sg);
EXPORT void
jbg_quant_planar(const float *x, int T, int IC, float s, int zp, int8_t *out)
{
   float32x4_t sv = vdupq_n_f32(s), zv = vdupq_n_f32((float)zp), lo = vdupq_n_f32(-128.f), hi = vdupq_n_f32(127.f);
#pragma omp parallel for schedule(dynamic) collapse(2)
   for (int c0 = 0; c0 < IC; c0 += 16)
      for (int t0 = 0; t0 < T; t0 += 16) {
         if (c0 + 16 <= IC && t0 + 16 <= T) {
            uint8x16_t r[16];
            for (int i = 0; i < 16; i++) {
               const float *p = x + (size_t)(t0 + i) * IC + c0;
               int32x4_t q[4];
               for (int k = 0; k < 4; k++) {
                  float32x4_t v = vaddq_f32(vrndnq_f32(vdivq_f32(vld1q_f32(p + 4 * k), sv)), zv);
                  q[k] = vcvtq_s32_f32(vminq_f32(vmaxq_f32(v, lo), hi));
               }
               int16x8_t h0 = vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1]));
               int16x8_t h1 = vcombine_s16(vmovn_s32(q[2]), vmovn_s32(q[3]));
               r[i] = vreinterpretq_u8_s8(vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
            }
            tr16x16(r);
            for (int k = 0; k < 16; k++)
               vst1q_u8((uint8_t *)out + (size_t)(c0 + k) * T + t0, r[k]);
         } else {
            int c1 = c0 + 16 < IC ? c0 + 16 : IC, t1 = t0 + 16 < T ? t0 + 16 : T;
            for (int t = t0; t < t1; t++)
               for (int c = c0; c < c1; c++) {
                  float v = rintf(x[(size_t)t * IC + c] / s) + (float)zp;
                  v = v < -128.f ? -128.f : (v > 127.f ? 127.f : v);
                  out[(size_t)c * T + t] = (int8_t)v;
               }
         }
      }
}

/* y[t, rows[j]] (+)= (lv[t * ts + j * js] - zp) * s for t < T, j < n (a job's 8-bit outputs back to model order).
 * Tiles of 16 tokens x 64 rows: read the tile once, then write row by row (16 output rows stay in L2). */
EXPORT void
jbg_dequant_scatter(const int8_t *lv, long ts, long js, int T, int n, const int32_t *rows, float s, int zp,
                    float *y, int ldy, int accumulate)
{
   float fz = (float)zp;
#pragma omp parallel for schedule(dynamic) collapse(2)
   for (int t0 = 0; t0 < T; t0 += 16)
      for (int j0 = 0; j0 < n; j0 += 64) {
         int t1 = t0 + 16 < T ? t0 + 16 : T, j1 = j0 + 64 < n ? j0 + 64 : n;
         float tile[16][64];
         for (int j = j0; j < j1; j++) {
            const int8_t *l = lv + j * js;
            for (int t = t0; t < t1; t++)
               tile[t - t0][j - j0] = ((float)l[t * ts] - fz) * s;
         }
         for (int t = t0; t < t1; t++) {
            float *o = y + (size_t)t * ldy;
            const float *tr = tile[t - t0];
            if (accumulate)
               for (int j = j0; j < j1; j++)
                  o[rows[j]] += tr[j - j0];
            else
               for (int j = j0; j < j1; j++)
                  o[rows[j]] = tr[j - j0];
         }
      }
}

/* 16 x 16 byte transpose in registers (trn on 8 / 16 / 32 / 64-bit lanes) */
static inline void
tr16x16(uint8x16_t r[16])
{
   uint8x16_t a[16];
   for (int i = 0; i < 8; i++) {
      a[2 * i] = vtrn1q_u8(r[2 * i], r[2 * i + 1]);
      a[2 * i + 1] = vtrn2q_u8(r[2 * i], r[2 * i + 1]);
   }
   uint16x8_t b[16];
   for (int i = 0; i < 4; i++)
      for (int j = 0; j < 2; j++) {
         int p = 4 * i + j, q = p + 2;
         b[p] = vtrn1q_u16(vreinterpretq_u16_u8(a[p]), vreinterpretq_u16_u8(a[q]));
         b[q] = vtrn2q_u16(vreinterpretq_u16_u8(a[p]), vreinterpretq_u16_u8(a[q]));
      }
   uint32x4_t c[16];
   for (int i = 0; i < 2; i++)
      for (int j = 0; j < 4; j++) {
         int p = 8 * i + j, q = p + 4;
         c[p] = vtrn1q_u32(vreinterpretq_u32_u16(b[p]), vreinterpretq_u32_u16(b[q]));
         c[q] = vtrn2q_u32(vreinterpretq_u32_u16(b[p]), vreinterpretq_u32_u16(b[q]));
      }
   for (int j = 0; j < 8; j++) {
      r[j] = vreinterpretq_u8_u64(vtrn1q_u64(vreinterpretq_u64_u32(c[j]), vreinterpretq_u64_u32(c[j + 8])));
      r[j + 8] = vreinterpretq_u8_u64(vtrn2q_u64(vreinterpretq_u64_u32(c[j]), vreinterpretq_u64_u32(c[j + 8])));
   }
}

/* All jobs of one step at once (2026-11-23): y [T, ldy] (+)= sum over jobs j of (lv_j[t * ts_j + r * js_j] - zp_j) *
 * s_j at rows rows_j[r] (a split step's pieces add up). accumulate = 0: y is overwritten; 1: added to (the residual).
 * Per block of 16 tokens: every job's block is first brought to token-major bytes (planar jobs, ts 1 = the NPU's
 * layout: 16 x 16 NEON transposes), then each token row of y is filled from all jobs while it sits in L1. */
EXPORT void
jbg_scatter_jobs(int nj, const int8_t *const *lv, const long *ts, const long *js, const int *n,
                 const int32_t *const *rows, const float *s, const int *zp, float *y, int T, int ldy, int accumulate)
{
   long off[257];
   long tot = 0;
   for (int j = 0; j < nj && j < 256; j++) {
      off[j] = tot;
      tot += (n[j] + 15) & ~15;
   }
#pragma omp parallel
   {
      int8_t *tm = malloc((size_t)16 * tot);              /* per job: [16 tokens][n_j] bytes */
#pragma omp for schedule(dynamic)
      for (int t0 = 0; t0 < T; t0 += 16) {
         int nt = T - t0 < 16 ? T - t0 : 16;
         const int8_t *src[256];
         long sst[256];
         for (int j = 0; j < nj; j++) {
            const int8_t *l = lv[j];
            int nn = n[j];
            int8_t *dst = tm + 16 * off[j];
            if (ts[j] == 1 && nt == 16) {
               int k = 0;
               for (; k + 16 <= nn; k += 16) {
                  uint8x16_t r[16];
                  for (int i = 0; i < 16; i++)
                     r[i] = vld1q_u8((const uint8_t *)l + (size_t)(k + i) * js[j] + t0);
                  tr16x16(r);
                  for (int i = 0; i < 16; i++)
                     vst1q_u8((uint8_t *)dst + (size_t)i * nn + k, r[i]);
               }
               for (; k < nn; k++)
                  for (int i = 0; i < 16; i++)
                     dst[(size_t)i * nn + k] = l[(size_t)k * js[j] + t0 + i];
               src[j] = dst;
               sst[j] = nn;
            } else if (js[j] == 1) {
               src[j] = l + (size_t)t0 * ts[j];
               sst[j] = ts[j];
            } else {
               for (int i = 0; i < nt; i++)
                  for (int k = 0; k < nn; k++)
                     dst[(size_t)i * nn + k] = l[(t0 + i) * ts[j] + (size_t)k * js[j]];
               src[j] = dst;
               sst[j] = nn;
            }
         }
         for (int i = 0; i < nt; i++) {
            float *o = y + (size_t)(t0 + i) * ldy;
            if (!accumulate)
               memset(o, 0, ldy * sizeof(float));
            for (int j = 0; j < nj; j++) {
               const int8_t *row = src[j] + (size_t)i * sst[j];
               const int32_t *r = rows[j];
               float sj = s[j], zj = (float)zp[j];
               float32x4_t sv = vdupq_n_f32(sj), zv = vdupq_n_f32(zj);
               int nn = n[j], k = 0;
               for (; k + 8 <= nn; k += 8) {
                  /* (level - zp) * s 8 at a time, then the adds in order: a separate multiply and add, as numpy
                   * and jb_prefill (built -std=c11: no FMA contraction) do; the .so is now built -ffp-contract=off
                   * too (before 2026-11-24 GCC fused this loop's multiply-add in the .so only, so the two builds
                   * differed in the last bit of accumulated partial sums) */
                  int16x8_t w = vmovl_s8(vld1_s8(row + k));
                  float f[8];
                  vst1q_f32(f, vmulq_f32(vsubq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), zv), sv));
                  vst1q_f32(f + 4, vmulq_f32(vsubq_f32(vcvtq_f32_s32(vmovl_high_s16(w)), zv), sv));
                  for (int l = 0; l < 8; l++)
                     o[r[k + l]] += f[l];
               }
               for (; k < nn; k++)
                  o[r[k]] += ((float)row[k] - zj) * sj;
            }
         }
      }
      free(tm);
   }
}

/* ---- fused passes (2026-11-23): no float array between the NPU steps but the residual ---- */

/* the block of 16 tokens t0.. of every job as token-major int8 rows: src[j] + i * sst[j] = token t0 + i, its n[j]
 * outputs. Planar jobs (ts 1) are transposed into tm (16 x 16 NEON blocks); token-major ones are used in place. */
static void
block_rows(int nj, const int8_t *const *lv, const long *ts, const long *js, const int *n, int t0, int nt,
           int8_t *tm, const long *off, const int8_t **src, long *sst)
{
   for (int j = 0; j < nj; j++) {
      const int8_t *l = lv[j];
      int nn = n[j];
      int8_t *dst = tm + 16 * off[j];
      if (ts[j] == 1 && nt == 16) {
         int k = 0;
         for (; k + 16 <= nn; k += 16) {
            uint8x16_t r[16];
            for (int i = 0; i < 16; i++)
               r[i] = vld1q_u8((const uint8_t *)l + (size_t)(k + i) * js[j] + t0);
            tr16x16(r);
            for (int i = 0; i < 16; i++)
               vst1q_u8((uint8_t *)dst + (size_t)i * nn + k, r[i]);
         }
         for (; k < nn; k++)
            for (int i = 0; i < 16; i++)
               dst[(size_t)i * nn + k] = l[(size_t)k * js[j] + t0 + i];
         src[j] = dst;
         sst[j] = nn;
      } else if (js[j] == 1) {
         src[j] = l + (size_t)t0 * ts[j];
         sst[j] = ts[j];
      } else {
         for (int i = 0; i < nt; i++)
            for (int k = 0; k < nn; k++)
               dst[(size_t)i * nn + k] = l[(t0 + i) * ts[j] + (size_t)k * js[j]];
         src[j] = dst;
         sst[j] = nn;
      }
   }
}

static long
block_offsets(int nj, const int *n, long *off)
{
   long tot = 0;
   for (int j = 0; j < nj; j++) {
      off[j] = tot;
      tot += (n[j] + 15) & ~15;
   }
   return tot;
}

/* do the jobs' rows cover 0..oc-1 exactly once? (every unsplit step; a split step's partials overlap) */
static int
rows_partition(int nj, const int *n, const int32_t *const *rows, int oc)
{
   long tot = 0;
   for (int j = 0; j < nj; j++)
      tot += n[j];
   if (tot != oc)
      return 0;
   uint8_t *seen = calloc(oc, 1);
   int ok = 1;
   for (int j = 0; j < nj && ok; j++)
      for (int k = 0; k < n[j]; k++) {
         int r = rows[j][k];
         if (r < 0 || r >= oc || seen[r]) {
            ok = 0;
            break;
         }
         seen[r] = 1;
      }
   free(seen);
   return ok;
}

/* one token row of a step's output in model order: o[rows] = sum of the jobs' dequantized outputs. part (the rows
 * are a partition, rows_partition): each row is stored once, (level - zp) * s converted 8 at a time; the same bits as
 * zeroing + adding (0 + v = v; v is never -0: the scales are positive) (2026-11-24) */
static inline void
row_from_jobs(int nj, const int8_t *const *src, const long *sst, int i, const int *n, const int32_t *const *rows,
              const float *s, const int *zp, float *o, int oc, int part)
{
   if (part) {
      for (int j = 0; j < nj; j++) {
         const int8_t *row = src[j] + (size_t)i * sst[j];
         const int32_t *r = rows[j];
         float sj = s[j], zj = (float)zp[j];
         float32x4_t sv = vdupq_n_f32(sj), zv = vdupq_n_f32(zj);
         int k = 0, nn = n[j];
         for (; k + 8 <= nn; k += 8) {
            int16x8_t w = vmovl_s8(vld1_s8(row + k));
            float32x4_t f0 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(w))), zv), sv);
            float32x4_t f1 = vmulq_f32(vsubq_f32(vcvtq_f32_s32(vmovl_high_s16(w)), zv), sv);
            o[r[k]] = vgetq_lane_f32(f0, 0);
            o[r[k + 1]] = vgetq_lane_f32(f0, 1);
            o[r[k + 2]] = vgetq_lane_f32(f0, 2);
            o[r[k + 3]] = vgetq_lane_f32(f0, 3);
            o[r[k + 4]] = vgetq_lane_f32(f1, 0);
            o[r[k + 5]] = vgetq_lane_f32(f1, 1);
            o[r[k + 6]] = vgetq_lane_f32(f1, 2);
            o[r[k + 7]] = vgetq_lane_f32(f1, 3);
         }
         for (; k < nn; k++)
            o[r[k]] = ((float)row[k] - zj) * sj;
      }
      return;
   }
   memset(o, 0, oc * sizeof(float));
   for (int j = 0; j < nj; j++) {
      const int8_t *row = src[j] + (size_t)i * sst[j];
      const int32_t *r = rows[j];
      float sj = s[j], zj = (float)zp[j];
      for (int k = 0; k < n[j]; k++)
         o[r[k]] += ((float)row[k] - zj) * sj;
   }
}

/* quantize the 16 x D block rows[0..15] (float, token-major) into the planar output out [D, T] at token t0 */
static void
quant_block(const float *rows, int ld, int nt, int D, float s, int zp, int8_t *out, int T, int t0)
{
   /* exact division like numpy: multiplying by 1 / s was no faster (memory-bound) and moved ~3 levels per 10^6 */
   float32x4_t sv = vdupq_n_f32(s), zv = vdupq_n_f32((float)zp), lo = vdupq_n_f32(-128.f), hi = vdupq_n_f32(127.f);
   for (int c0 = 0; c0 < D; c0 += 16) {
      if (nt == 16 && c0 + 16 <= D) {
         uint8x16_t r[16];
         for (int i = 0; i < 16; i++) {
            const float *p = rows + (size_t)i * ld + c0;
            int32x4_t q[4];
            for (int k = 0; k < 4; k++) {
               float32x4_t v = vaddq_f32(vrndnq_f32(vdivq_f32(vld1q_f32(p + 4 * k), sv)), zv);
               q[k] = vcvtq_s32_f32(vminq_f32(vmaxq_f32(v, lo), hi));
            }
            int16x8_t h0 = vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1]));
            int16x8_t h1 = vcombine_s16(vmovn_s32(q[2]), vmovn_s32(q[3]));
            r[i] = vreinterpretq_u8_s8(vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
         }
         tr16x16(r);
         for (int k = 0; k < 16; k++)
            vst1q_u8((uint8_t *)out + (size_t)(c0 + k) * T + t0, r[k]);
      } else {
         int c1 = c0 + 16 < D ? c0 + 16 : D;
         for (int i = 0; i < nt; i++)
            for (int c = c0; c < c1; c++) {
               float v = rintf(rows[(size_t)i * ld + c] / s) + (float)zp;
               v = v < -128.f ? -128.f : (v > 127.f ? 127.f : v);
               out[(size_t)c * T + t0 + i] = (int8_t)v;
            }
      }
   }
}

/* one token row h [D] (float) -> int8 levels q [D]: the same arithmetic as quant_block (exact division, rint, + zp,
 * clip), so bit-identical to it; used where the row is still in L1 (2026-11-24) */
static inline void
quant_row(const float *h, int D, float s, int zp, int8_t *q)
{
   float32x4_t sv = vdupq_n_f32(s), zv = vdupq_n_f32((float)zp), lo = vdupq_n_f32(-128.f), hi = vdupq_n_f32(127.f);
   int c = 0;
   for (; c + 16 <= D; c += 16) {
      int32x4_t v[4];
      for (int k = 0; k < 4; k++) {
         float32x4_t f = vaddq_f32(vrndnq_f32(vdivq_f32(vld1q_f32(h + c + 4 * k), sv)), zv);
         v[k] = vcvtq_s32_f32(vminq_f32(vmaxq_f32(f, lo), hi));
      }
      int16x8_t h0 = vcombine_s16(vmovn_s32(v[0]), vmovn_s32(v[1]));
      int16x8_t h1 = vcombine_s16(vmovn_s32(v[2]), vmovn_s32(v[3]));
      vst1q_s8(q + c, vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
   }
   for (; c < D; c++) {
      float v = rintf(h[c] / s) + (float)zp;
      v = v < -128.f ? -128.f : (v > 127.f ? 127.f : v);
      q[c] = (int8_t)v;
   }
}

/* the int8 token-major tile q [nt, D] -> the planar output out [D, T] at tokens t0.. (16 x 16 NEON transposes) */
static void
tile_to_planar(const int8_t *q, int nt, int D, int8_t *out, int T, int t0)
{
   int c0 = 0;
   if (nt == 16)
      for (; c0 + 16 <= D; c0 += 16) {
         uint8x16_t r[16];
         for (int i = 0; i < 16; i++)
            r[i] = vld1q_u8((const uint8_t *)q + (size_t)i * D + c0);
         tr16x16(r);
         for (int k = 0; k < 16; k++)
            vst1q_u8((uint8_t *)out + (size_t)(c0 + k) * T + t0, r[k]);
      }
   for (int c = c0; c < D; c++)
      for (int i = 0; i < nt; i++)
         out[(size_t)c * T + t0 + i] = q[(size_t)i * D + c];
}

static inline void
rms_row(const float *r, float *o, int D, float eps)
{
   float64x2_t a0 = vdupq_n_f64(0), a1 = a0;
   int i = 0;
   for (; i + 4 <= D; i += 4) {
      float32x4_t v = vld1q_f32(r + i);
      float64x2_t lo = vcvt_f64_f32(vget_low_f32(v)), hi = vcvt_high_f64_f32(v);
      a0 = vfmaq_f64(a0, lo, lo);
      a1 = vfmaq_f64(a1, hi, hi);
   }
   double s = vaddvq_f64(vaddq_f64(a0, a1));
   for (; i < D; i++)
      s += (double)r[i] * r[i];
   float inv = (float)(1.0 / sqrt(s / D + eps));
   float32x4_t iv = vdupq_n_f32(inv);
   i = 0;
   for (; i + 4 <= D; i += 4)
      vst1q_f32(o + i, vmulq_f32(vld1q_f32(r + i), iv));
   for (; i < D; i++)
      o[i] = r[i] * inv;
}

/* one row: blockwise normalized FWHT(r * sg) over blocks of b (b a power of 2, >= 4), bit-exact with the radix-2
 * numpy version (the same sums in the same order): stages h = 1, 2 in registers (NEON shuffles) with the signs, then
 * two stages per pass (radix 4), a last radix-2 stage if the count is odd, then the normalization */
static inline void
fwht_row(float *r, int n, int b, const float *sg)
{
   const float32x4_t pm = {1.f, -1.f, 1.f, -1.f}, ppmm = {1.f, 1.f, -1.f, -1.f};
   for (int i = 0; i + 4 <= n; i += 4) {
      float32x4_t x = vmulq_f32(vld1q_f32(r + i), vld1q_f32(sg + i));
      float32x4_t z = vfmaq_f32(vtrn1q_f32(x, x), vtrn2q_f32(x, x), pm);        /* [a+b, a-b, c+d, c-d] */
      float32x4_t lo = vcombine_f32(vget_low_f32(z), vget_low_f32(z));
      float32x4_t hi = vcombine_f32(vget_high_f32(z), vget_high_f32(z));
      vst1q_f32(r + i, vfmaq_f32(lo, hi, ppmm));                              /* [z0+z2, z1+z3, z0-z2, z1-z3] */
   }
   float norm = (float)(1.0 / sqrt((double)b));
   for (int o = 0; o < n; o += b) {
      float *y = r + o;
      int h = 4;
      for (; 4 * h <= b; h *= 4)                        /* stages h and 2h together */
         for (int i = 0; i < b; i += 4 * h)
            for (int j = i; j < i + h; j += 4) {
               float32x4_t a0 = vld1q_f32(y + j), a1 = vld1q_f32(y + j + h);
               float32x4_t a2 = vld1q_f32(y + j + 2 * h), a3 = vld1q_f32(y + j + 3 * h);
               float32x4_t s01 = vaddq_f32(a0, a1), d01 = vsubq_f32(a0, a1);
               float32x4_t s23 = vaddq_f32(a2, a3), d23 = vsubq_f32(a2, a3);
               vst1q_f32(y + j, vaddq_f32(s01, s23));
               vst1q_f32(y + j + h, vaddq_f32(d01, d23));
               vst1q_f32(y + j + 2 * h, vsubq_f32(s01, s23));
               vst1q_f32(y + j + 3 * h, vsubq_f32(d01, d23));
            }
      for (; h < b; h *= 2)                             /* an odd stage left */
         for (int i = 0; i < b; i += 2 * h)
            for (int j = i; j < i + h; j += 4) {
               float32x4_t a0 = vld1q_f32(y + j), a1 = vld1q_f32(y + j + h);
               vst1q_f32(y + j, vaddq_f32(a0, a1));
               vst1q_f32(y + j + h, vsubq_f32(a0, a1));
            }
      float32x4_t nv = vdupq_n_f32(norm);
      for (int i = 0; i + 4 <= b; i += 4)
         vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), nv));
   }
}

/* RMSNorm (no weight: folded) of rows 1..T of x [T+1, D] straight into the NPU's planar int8 input out [D, T]
 * (s, zp: the step's input scale), and row 0 normalized in float into a0 (position 0: the CPU's). */
EXPORT void
jbg_norm_quant(const float *x, int T, int D, float eps, float s, int zp, int8_t *out, float *a0)
{
   rms_row(x, a0, D, eps);
#pragma omp parallel
   {
      float *blk = malloc((size_t)16 * D * sizeof(float));
#pragma omp for schedule(dynamic)
      for (int t0 = 0; t0 < T; t0 += 16) {
         int nt = T - t0 < 16 ? T - t0 : 16;
         for (int i = 0; i < nt; i++)
            rms_row(x + (size_t)(1 + t0 + i) * D, blk + (size_t)i * D, D, eps);
         quant_block(blk, D, nt, D, s, zp, out, T, t0);
      }
      free(blk);
   }
}

/* The whole FFN glue of tokens 1..T in one pass: the gateup jobs' 8-bit outputs -> gate | up rows in model order ->
 * h = SiLU(gate) * up -> blockwise FWHT(h * signs) -> quantized into down's planar int8 input out [I, T]. Nothing
 * but the int8 output is written. (Position 0 comes from the CPU's GEMV: done separately.) */
EXPORT void
jbg_ffn_fused(int nj, const int8_t *const *lv, const long *ts, const long *js, const int *n,
              const int32_t *const *rows, const float *s, const int *zp, int T, int I, const float *signs, int b,
              float sx, int zx, int8_t *out)
{
   long off[257];
   long tot = block_offsets(nj, n, off);
   int prof = getenv("JBG_PROF") != NULL;
   int part = rows_partition(nj, n, rows, 2 * I);
   double ph[5] = {0, 0, 0, 0, 0}, w0 = now_s();
#pragma omp parallel
   {
      double my[5] = {0, 0, 0, 0, 0}, tq = 0, t2;
      int8_t *tm = malloc((size_t)16 * tot);
      float *gu = malloc((size_t)2 * I * sizeof(float));
      float *h = malloc((size_t)I * sizeof(float));
      int8_t *qt = malloc((size_t)16 * I);          /* the tile's int8 rows (2026-11-24: was a 16 x I float tile,
                                                     * 512 KB per thread for the 1.7B: 4 of them overflowed the A73s'
                                                     * 1 MB L2, so SiLU's output went to DRAM and back) */
      const int8_t *src[256];
      long sst[256];
      float32x4_t one = vdupq_n_f32(1.0f);
#pragma omp for schedule(dynamic)
      for (int t0 = 0; t0 < T; t0 += 16) {
         int nt = T - t0 < 16 ? T - t0 : 16;
         if (prof) tq = now_s();
         block_rows(nj, lv, ts, js, n, t0, nt, tm, off, src, sst);
         if (prof) { t2 = now_s(); my[0] += t2 - tq; tq = t2; }
         for (int i = 0; i < nt; i++) {
            row_from_jobs(nj, src, sst, i, n, rows, s, zp, gu, 2 * I, part);
            if (prof) { t2 = now_s(); my[1] += t2 - tq; tq = t2; }
            int c = 0;
            for (; c + 8 <= I; c += 8) {                     /* 2 independent exp chains in flight */
               float32x4_t g0 = vld1q_f32(gu + c), g1 = vld1q_f32(gu + c + 4);
               float32x4_t e0 = exp_f32(vnegq_f32(g0)), e1 = exp_f32(vnegq_f32(g1));
               float32x4_t u0 = vld1q_f32(gu + I + c), u1 = vld1q_f32(gu + I + c + 4);
               vst1q_f32(h + c, vmulq_f32(vdivq_f32(g0, vaddq_f32(one, e0)), u0));
               vst1q_f32(h + c + 4, vmulq_f32(vdivq_f32(g1, vaddq_f32(one, e1)), u1));
            }
            for (; c + 4 <= I; c += 4) {
               float32x4_t g = vld1q_f32(gu + c), u = vld1q_f32(gu + I + c);
               vst1q_f32(h + c, vmulq_f32(vdivq_f32(g, vaddq_f32(one, exp_f32(vnegq_f32(g)))), u));
            }
            for (; c < I; c++)
               h[c] = gu[c] / (1.0f + expf(-gu[c])) * gu[I + c];
            if (prof) { t2 = now_s(); my[2] += t2 - tq; tq = t2; }
            fwht_row(h, I, b, signs);
            if (prof) { t2 = now_s(); my[3] += t2 - tq; tq = t2; }
            quant_row(h, I, sx, zx, qt + (size_t)i * I);
            if (prof) { t2 = now_s(); my[4] += t2 - tq; tq = t2; }
         }
         tile_to_planar(qt, nt, I, out, T, t0);
         if (prof) { t2 = now_s(); my[4] += t2 - tq; tq = t2; }
      }
      if (prof) {
#pragma omp critical
         for (int q_ = 0; q_ < 5; q_++)
            ph[q_] += my[q_];
      }
      free(tm);
      free(gu);
      free(h);
      free(qt);
   }
   if (prof)
      fprintf(stderr, "jbg_ffn_fused: thread-summed transpose %.2f, scatter %.2f, silu %.2f, fwht %.2f, quant %.2f ms; "
              "wall %.2f ms\n", ph[0] * 1e3, ph[1] * 1e3, ph[2] * 1e3, ph[3] * 1e3, ph[4] * 1e3, (now_s() - w0) * 1e3);
}

/* qkv of tokens 1..T in one pass: the jobs' outputs -> q / k / v rows in model order, RoPE on q and k (positions
 * pos0+1 ..), written to q [T+1, H, hd], k / v [T+1, KV, hd] at rows 1..T (row 0: the CPU's GEMV, separately). */
EXPORT void
jbg_qkv_rope(int nj, const int8_t *const *lv, const long *ts, const long *js, const int *n,
             const int32_t *const *rows, const float *s, const int *zp, int T, int H, int KV, int hd, int pos0,
             const double *inv_freq, int rope, float *q, float *k, float *v)
{
   long off[257];
   long tot = block_offsets(nj, n, off);
   int oc = (H + 2 * KV) * hd, h2 = hd / 2;
   int part = rows_partition(nj, n, rows, oc);
#pragma omp parallel
   {
      int8_t *tm = malloc((size_t)16 * tot);
      float *row = malloc((size_t)oc * sizeof(float));
      const int8_t *src[256];
      long sst[256];
#pragma omp for schedule(dynamic)
      for (int t0 = 0; t0 < T; t0 += 16) {
         int nt = T - t0 < 16 ? T - t0 : 16;
         block_rows(nj, lv, ts, js, n, t0, nt, tm, off, src, sst);
         for (int i = 0; i < nt; i++) {
            int t = 1 + t0 + i;
            row_from_jobs(nj, src, sst, i, n, rows, s, zp, row, oc, part);
            double cs[256], sn[256];
            if (rope)
               for (int f = 0; f < h2; f++) {
                  double a = (double)(pos0 + t) * inv_freq[f];
                  cs[f] = cos(a);
                  sn[f] = sin(a);
               }
            for (int hh = 0; hh < H + KV; hh++) {
               const float *src_h = row + (size_t)hh * hd;
               float *dst = hh < H ? q + ((size_t)t * H + hh) * hd : k + ((size_t)t * KV + hh - H) * hd;
               if (rope)
                  for (int f = 0; f < h2; f++) {
                     double a = src_h[f], c = src_h[f + h2];
                     dst[f] = (float)(a * cs[f] - c * sn[f]);
                     dst[f + h2] = (float)(c * cs[f] + a * sn[f]);
                  }
               else
                  memcpy(dst, src_h, hd * sizeof(float));
            }
            memcpy(v + (size_t)t * KV * hd, row + (size_t)(H + KV) * hd, (size_t)KV * hd * sizeof(float));
         }
      }
      free(tm);
      free(row);
   }
}

/* one row's dot: the int8 weights widened to float, 4 accumulators over 16-wide steps, their sum, the scalar tail */
#define GEMV_ROW_DECL(i) float32x4_t s0##i = vdupq_n_f32(0), s1##i = s0##i, s2##i = s0##i, s3##i = s0##i
#define GEMV_ROW_STEP(i, wr)                                                                                     \
   do {                                                                                                          \
      int8x16_t b_ = vld1q_s8((wr) + c);                                                                         \
      int16x8_t lo_ = vmovl_s8(vget_low_s8(b_)), hi_ = vmovl_s8(vget_high_s8(b_));                               \
      s0##i = vfmaq_f32(s0##i, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo_))), x0);                                 \
      s1##i = vfmaq_f32(s1##i, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo_))), x1);                                \
      s2##i = vfmaq_f32(s2##i, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi_))), x2);                                 \
      s3##i = vfmaq_f32(s3##i, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi_))), x3);                                \
   } while (0)

static inline float
gemv_tail(float32x4_t s0, float32x4_t s1, float32x4_t s2, float32x4_t s3, const int8_t *wr, const float *x, int c,
          int ic)
{
   float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
   for (; c < ic; c++)
      acc += (float)wr[c] * x[c];
   return acc;
}

/* All jobs of one step for one token (position 0, decode): y[rows_j[r]] (+)= sw_j * w_j[r, :] . x[col_j ..) (y zeroed
 * unless accumulate). One parallel region; rows 4 at a time sharing each load of x, in chunks of 16 rows; a barrier
 * between jobs (the pieces of a split step can share output rows). Every row's sum is formed exactly as before
 * (2026-11-24: was a parallel region per job with one row per dynamic dispatch: 3 GB/s on the 135M's decode). */
EXPORT void
jbg_gemv_jobs(int nj, const float *x, const int8_t *const *w, const int *n, const int *ic, const int *col,
              const float *sw, const int32_t *const *rows, float *y, int oc, int accumulate)
{
   if (!accumulate)
      memset(y, 0, oc * sizeof(float));
   int maxn = 0;
   for (int j = 0; j < nj; j++)
      maxn = n[j] > maxn ? n[j] : maxn;
   /* each row's dot into dots[r] (16 rows = one cache line per chunk), then y[rows[r]] += dots[r] * s by one thread
    * after the job's barrier: threads writing y[rows[r]] directly shared y's cache lines (false sharing) */
   float *dots = aligned_alloc(64, (((size_t)maxn + 15) & ~(size_t)15) * sizeof(float));
#pragma omp parallel
   for (int j = 0; j < nj; j++) {
      const int8_t *wj = w[j];
      const float *xj = x + col[j];
      int icj = ic[j], nn = n[j];
      const int32_t *rj = rows[j];
      float s = sw[j];
#pragma omp for schedule(dynamic, 4)
      for (int rb = 0; rb < (nn + 3) / 4; rb++) {
         int r = 4 * rb;
         if (r + 4 > nn) {                               /* the last 1-3 rows */
            for (; r < nn; r++) {
               const int8_t *wr = wj + (size_t)r * icj;
               GEMV_ROW_DECL(0);
               int c = 0;
               for (; c + 16 <= icj; c += 16) {
                  float32x4_t x0 = vld1q_f32(xj + c), x1 = vld1q_f32(xj + c + 4);
                  float32x4_t x2 = vld1q_f32(xj + c + 8), x3 = vld1q_f32(xj + c + 12);
                  GEMV_ROW_STEP(0, wr);
               }
               dots[r] = gemv_tail(s00, s10, s20, s30, wr, xj, c, icj);
            }
            continue;
         }
         const int8_t *w0 = wj + (size_t)r * icj, *w1 = w0 + icj, *w2 = w1 + icj, *w3 = w2 + icj;
         GEMV_ROW_DECL(0);
         GEMV_ROW_DECL(1);
         GEMV_ROW_DECL(2);
         GEMV_ROW_DECL(3);
         int c = 0;
         for (; c + 16 <= icj; c += 16) {
            float32x4_t x0 = vld1q_f32(xj + c), x1 = vld1q_f32(xj + c + 4);
            float32x4_t x2 = vld1q_f32(xj + c + 8), x3 = vld1q_f32(xj + c + 12);
            GEMV_ROW_STEP(0, w0);
            GEMV_ROW_STEP(1, w1);
            GEMV_ROW_STEP(2, w2);
            GEMV_ROW_STEP(3, w3);
         }
         dots[r] = gemv_tail(s00, s10, s20, s30, w0, xj, c, icj);
         dots[r + 1] = gemv_tail(s01, s11, s21, s31, w1, xj, c, icj);
         dots[r + 2] = gemv_tail(s02, s12, s22, s32, w2, xj, c, icj);
         dots[r + 3] = gemv_tail(s03, s13, s23, s33, w3, xj, c, icj);
      }
#pragma omp single
      for (int r = 0; r < nn; r++)
         y[rj[r]] += dots[r] * s;
   }
   free(dots);
}

/* decode's GEMV (2026-11-24): like jbg_gemv_jobs, but x is first quantized to int16 with one scale for the call
 * (max |x| -> 32767: 128x finer than the NPU path's 8-bit activations) and the rows are int8 x int16 -> int32
 * (SMLAL: 2 widenings + 4 multiply-accumulates per 16 weights instead of 14 float operations; the A73 has no int8 dot
 * product). Each lane takes one product per 16 weights; blocks of 4096 weights (<= 256 products of |w| <= 128 and
 * |x| <= 32767 per lane: < 2^31) go to an int64 sum, so every dot is the exact integer, rounded to float once (the
 * same bits whatever the weight layout: jbg_gemv_coef_q16 reads the NPU's). xq: scratch of >= max(col + ic) int16s. */
#define Q16_BLOCK 4096
#define Q16_ROW_DECL(i) int32x4_t a0##i = vdupq_n_s32(0), a1##i = a0##i, a2##i = a0##i, a3##i = a0##i
#define Q16_ROW_STEP(i, wr)                                                                                       \
   do {                                                                                                          \
      int8x16_t b_ = vld1q_s8((wr) + c);                                                                         \
      int16x8_t lo_ = vmovl_s8(vget_low_s8(b_)), hi_ = vmovl_s8(vget_high_s8(b_));                               \
      a0##i = vmlal_s16(a0##i, vget_low_s16(lo_), vget_low_s16(xa));                                             \
      a1##i = vmlal_high_s16(a1##i, lo_, xa);                                                                    \
      a2##i = vmlal_s16(a2##i, vget_low_s16(hi_), vget_low_s16(xb));                                             \
      a3##i = vmlal_high_s16(a3##i, hi_, xb);                                                                    \
   } while (0)
#define Q16_ROW_FLUSH(i)                                                                                         \
   do {                                                                                                          \
      f##i += vaddlvq_s32(a0##i) + vaddlvq_s32(a1##i) + vaddlvq_s32(a2##i) + vaddlvq_s32(a3##i);                  \
      a0##i = a1##i = a2##i = a3##i = vdupq_n_s32(0);                                                            \
   } while (0)

static inline float
q16_row1(const int8_t *wr, const int16_t *xq, int ic)
{
   Q16_ROW_DECL(0);
   int64_t f0 = 0;
   int c = 0;
   while (c + 16 <= ic) {
      int end = c + Q16_BLOCK < ic ? c + Q16_BLOCK : ic;
      for (; c + 16 <= end; c += 16) {
         int16x8_t xa = vld1q_s16(xq + c), xb = vld1q_s16(xq + c + 8);
         Q16_ROW_STEP(0, wr);
      }
      Q16_ROW_FLUSH(0);
   }
   for (; c < ic; c++)
      f0 += (int32_t)wr[c] * xq[c];
   return (float)f0;                                    /* the exact integer dot, rounded once */
}

EXPORT void
jbg_gemv_jobs_q16(int nj, const float *x, const int8_t *const *w, const int *n, const int *ic, const int *col,
                  const float *sw, const int32_t *const *rows, float *y, int oc, int accumulate, int16_t *xq)
{
   int nx = 0;
   for (int j = 0; j < nj; j++)
      nx = col[j] + ic[j] > nx ? col[j] + ic[j] : nx;
   float mx = 0;
   for (int i = 0; i < nx; i++)
      mx = fabsf(x[i]) > mx ? fabsf(x[i]) : mx;
   float sx = mx > 0 ? mx / 32767.0f : 1.0f, inv = 1.0f / sx;
   for (int i = 0; i < nx; i++)
      xq[i] = (int16_t)lrintf(x[i] * inv);
   if (!accumulate)
      memset(y, 0, oc * sizeof(float));
   int maxn = 0;
   for (int j = 0; j < nj; j++)
      maxn = n[j] > maxn ? n[j] : maxn;
   float *dots = aligned_alloc(64, (((size_t)maxn + 15) & ~(size_t)15) * sizeof(float));
#pragma omp parallel
   for (int j = 0; j < nj; j++) {
      const int8_t *wj = w[j];
      const int16_t *xj = xq + col[j];
      int icj = ic[j], nn = n[j];
      const int32_t *rj = rows[j];
      float s = sw[j] * sx;
#pragma omp for schedule(dynamic, 4)
      for (int rb = 0; rb < (nn + 3) / 4; rb++) {
         int r = 4 * rb;
         if (r + 4 > nn) {
            for (; r < nn; r++)
               dots[r] = q16_row1(wj + (size_t)r * icj, xj, icj);
            continue;
         }
         const int8_t *w0 = wj + (size_t)r * icj, *w1 = w0 + icj, *w2 = w1 + icj, *w3 = w2 + icj;
         Q16_ROW_DECL(0);
         Q16_ROW_DECL(1);
         Q16_ROW_DECL(2);
         Q16_ROW_DECL(3);
         int64_t f0 = 0, f1 = 0, f2 = 0, f3 = 0;
         int c = 0;
         while (c + 16 <= icj) {
            int end = c + Q16_BLOCK < icj ? c + Q16_BLOCK : icj;
            for (; c + 16 <= end; c += 16) {
               int16x8_t xa = vld1q_s16(xj + c), xb = vld1q_s16(xj + c + 8);
               Q16_ROW_STEP(0, w0);
               Q16_ROW_STEP(1, w1);
               Q16_ROW_STEP(2, w2);
               Q16_ROW_STEP(3, w3);
            }
            Q16_ROW_FLUSH(0);
            Q16_ROW_FLUSH(1);
            Q16_ROW_FLUSH(2);
            Q16_ROW_FLUSH(3);
         }
         int64_t acc[4] = {f0, f1, f2, f3};
         const int8_t *wr4[4] = {w0, w1, w2, w3};
         for (int i = 0; i < 4; i++) {
            for (int cc = c; cc < icj; cc++)
               acc[i] += (int32_t)wr4[i][cc] * xj[cc];
            dots[r + i] = (float)acc[i];                  /* the exact integer dot, rounded once */
         }
      }
#pragma omp single
      for (int r = 0; r < nn; r++)
         y[rj[r]] += dots[r] * s;
   }
   free(dots);
}

/* x [n] -> int16 with one scale (max |x| -> 32767); returns the scale */
EXPORT float
jbg_quant16(const float *x, int n, int16_t *xq)
{
   float mx = 0;
   for (int i = 0; i < n; i++)
      mx = fabsf(x[i]) > mx ? fabsf(x[i]) : mx;
   float sx = mx > 0 ? mx / 32767.0f : 1.0f, inv = 1.0f / sx;
   for (int i = 0; i < n; i++)
      xq[i] = (int16_t)lrintf(x[i] * inv);
   return sx;
}

/* One job's weights as the NPU holds them (a V7 1x1 coefficient buffer, layout in mesa-llm etnaviv_ml_llm.h). */
struct jbg_coef {
   const uint8_t *map;
   int cores, header_size, kpc, superblocks, ic, oc, stride, zp;
};

/* dots of one (core, superblock) unit: the exact integer sum_c (u - zp) * xq[c] of each of its K kernels, through
 * sum u * x - zp * sum x (u: the stored byte); out[k] = (float)dot of the unit's kernel k. p: the superblock's first
 * byte; returns the byte after it. Middle blocks of 6: 8 kernels per step (vld3q_u16 splits the 48 bytes into byte
 * pairs: lane k = kernel k), int32 lanes flushed to int64 every 32 blocks (6 products of <= 255 * 32767 per block). */
static const uint8_t *
coef_unit(const uint8_t *p, int K, int ic, int stride, int zp, const int16_t *xq, long sumx, int64_t *acc)
{
   int nb = (ic + stride - 1) / stride;
   for (int k = 0; k < K; k++)
      acc[k] = 0;
   for (int b = 0; b < nb; b++) {
      int c0 = b * stride, w = ic - c0 < stride ? ic - c0 : stride;
      int first = b == 0, last = b == nb - 1;
      if (!first && !last && stride == 6) {
         int b1 = b;                                    /* a run of middle blocks: blocks b .. b1 - 1 */
         while (b1 < nb - 1)
            b1++;
         for (int g = 0; g < K; g += 8) {
            const uint8_t *q = p + 6 * g;
            int32x4_t lo = vdupq_n_s32(0), hi = lo;
            int64_t s64[8] = {0};
            for (int bb = b; bb < b1; bb++, q += 6 * K) {
               uint16x8x3_t v = vld3q_u16((const uint16_t *)q);
               const int16_t *x = xq + bb * 6;
               int16x8_t w0 = vreinterpretq_s16_u16(vandq_u16(v.val[0], vdupq_n_u16(0xff)));
               int16x8_t w1 = vreinterpretq_s16_u16(vshrq_n_u16(v.val[0], 8));
               int16x8_t w2 = vreinterpretq_s16_u16(vandq_u16(v.val[1], vdupq_n_u16(0xff)));
               int16x8_t w3 = vreinterpretq_s16_u16(vshrq_n_u16(v.val[1], 8));
               int16x8_t w4 = vreinterpretq_s16_u16(vandq_u16(v.val[2], vdupq_n_u16(0xff)));
               int16x8_t w5 = vreinterpretq_s16_u16(vshrq_n_u16(v.val[2], 8));
               lo = vmlal_n_s16(lo, vget_low_s16(w0), x[0]);
               hi = vmlal_high_n_s16(hi, w0, x[0]);
               lo = vmlal_n_s16(lo, vget_low_s16(w1), x[1]);
               hi = vmlal_high_n_s16(hi, w1, x[1]);
               lo = vmlal_n_s16(lo, vget_low_s16(w2), x[2]);
               hi = vmlal_high_n_s16(hi, w2, x[2]);
               lo = vmlal_n_s16(lo, vget_low_s16(w3), x[3]);
               hi = vmlal_high_n_s16(hi, w3, x[3]);
               lo = vmlal_n_s16(lo, vget_low_s16(w4), x[4]);
               hi = vmlal_high_n_s16(hi, w4, x[4]);
               lo = vmlal_n_s16(lo, vget_low_s16(w5), x[5]);
               hi = vmlal_high_n_s16(hi, w5, x[5]);
               if (((bb - b) & 31) == 31) {
                  int32_t t[8];
                  vst1q_s32(t, lo);
                  vst1q_s32(t + 4, hi);
                  for (int i = 0; i < 8; i++)
                     s64[i] += t[i];
                  lo = hi = vdupq_n_s32(0);
               }
            }
            int32_t t[8];
            vst1q_s32(t, lo);
            vst1q_s32(t + 4, hi);
            for (int i = 0; i < 8 && g + i < K; i++)
               acc[g + i] += s64[i] + t[i];
         }
         p += (size_t)(b1 - b) * 6 * K;
         b = b1 - 1;
         continue;
      }
      for (int k = 0; k < K; k++) {                    /* first / last block (bias, offset), or stride 1: bytes */
         int64_t a = 0;
         for (int i = 0; i < w; i++) {
            a += (int32_t)*p++ * xq[c0 + i];
            if (first && i == 0)
               p += 4;                                  /* the bias */
         }
         if (last)
            p += 4;                                     /* the output offset */
         acc[k] += a;
      }
   }
   for (int k = 0; k < K; k++)
      acc[k] -= (int64_t)zp * sumx;
   return p;
}

/* all jobs of one step for one token from the NPU's coefficient buffers: as jbg_gemv_jobs_q16 (the same int16 x and
 * exact integer dots, so the same bits), y[rows_j[r]] (+)= sw_j * sx * dot. Units (core, superblock) in parallel. */
EXPORT void
jbg_gemv_coef_q16(int nj, const float *x, const struct jbg_coef *cf, const int *col, const float *sw,
                  const int32_t *const *rows, float *y, int oc, int accumulate, int16_t *xq)
{
   int nx = 0, maxo = 0;
   for (int j = 0; j < nj; j++) {
      nx = col[j] + cf[j].ic > nx ? col[j] + cf[j].ic : nx;
      int pad = cf[j].kpc * cf[j].cores;
      maxo = pad > maxo ? pad : maxo;
   }
   float sx = jbg_quant16(x, nx, xq);
   if (!accumulate)
      memset(y, 0, oc * sizeof(float));
   float *dots = aligned_alloc(64, (((size_t)maxo + 15) & ~(size_t)15) * sizeof(float));
#pragma omp parallel
   for (int j = 0; j < nj; j++) {
      const struct jbg_coef *c = &cf[j];
      const int16_t *xj = xq + col[j];
      int sb = c->superblocks, kis0 = (c->kpc + sb - 1) / sb;
      long sumx = 0;
      for (int i = 0; i < c->ic; i++)
         sumx += xj[i];
#pragma omp for schedule(dynamic) collapse(2)
      for (int core = 0; core < c->cores; core++)
         for (int u = 0; u < sb; u++) {
            const uint32_t *hdr = (const uint32_t *)c->map;
            const uint8_t *p = c->map + c->header_size;
            for (int cc = 0; cc < core; cc++)
               p += hdr[cc];
            p += 3;                                     /* zrl_bits (8 bits), kernels_per_core (16 bits) */
            for (int s2 = 0; s2 < u; s2++) {            /* skip the earlier superblocks of this core */
               int kis = s2 == sb - 1 ? c->kpc - kis0 * (sb - 1) : kis0;
               p += (size_t)kis * (c->ic + 8);
            }
            int K = u == sb - 1 ? c->kpc - kis0 * (sb - 1) : kis0;
            int64_t acc[64];
            coef_unit(p, K, c->ic, c->stride, c->zp, xj, sumx, acc);
            for (int k = 0; k < K; k++) {
               int o = core * K + k + u * kis0 * c->cores;
               dots[o] = (float)acc[k];
            }
         }
#pragma omp single
      for (int r = 0; r < c->oc; r++)
         y[rows[j][r]] += dots[r] * (sw[j] * sx);
   }
   free(dots);
}

/* the int8 weights [oc, ic] of a coefficient buffer, decoded (scalar: a check that the layout is understood) */
EXPORT void
jbg_coef_decode(const struct jbg_coef *c, int8_t *w)
{
   const uint32_t *hdr = (const uint32_t *)c->map;
   const uint8_t *p = c->map + c->header_size;
   int sb = c->superblocks, kis0 = (c->kpc + sb - 1) / sb, nb = (c->ic + c->stride - 1) / c->stride;
   for (int core = 0; core < c->cores; core++) {
      const uint8_t *q = p + 3;
      for (int u = 0; u < sb; u++) {
         int K = u == sb - 1 ? c->kpc - kis0 * (sb - 1) : kis0;
         for (int b = 0; b < nb; b++) {
            int c0 = b * c->stride, wn = c->ic - c0 < c->stride ? c->ic - c0 : c->stride;
            for (int k = 0; k < K; k++) {
               int o = core * K + k + u * kis0 * c->cores;
               for (int i = 0; i < wn; i++) {
                  int v = *q++ - c->zp;
                  if (o < c->oc)
                     w[(size_t)o * c->ic + c0 + i] = (int8_t)v;
                  if (b == 0 && i == 0)
                     q += 4;
               }
               if (b == nb - 1)
                  q += 4;
            }
         }
      }
      p += hdr[core];
   }
}

/* attention of one new token (decode): q [H, hd] at the last position over the cache k / v [S, KV, hd] (read in
 * place) -> out [H * hd]; head h reads kv head h / (H / KV). One head per thread at a time. (2026-11-24) */
EXPORT void
jbg_attn_decode(const float *q, const float *k, const float *v, float *out, int S, int H, int KV, int hd)
{
   int g = H / KV;
   float scale = 1.0f / sqrtf((float)hd);
#pragma omp parallel
   {
      float *sc = malloc(((size_t)S + 4) * sizeof(float));
#pragma omp for schedule(dynamic)
      for (int h = 0; h < H; h++) {
         const float *qh = q + (size_t)h * hd;
         int kh = h / g;
         float mx = -INFINITY;
         for (int t = 0; t < S; t++) {
            float d = dot_f32(qh, k + ((size_t)t * KV + kh) * hd, hd) * scale;
            sc[t] = d;
            mx = d > mx ? d : mx;
         }
         for (int t = S; t < ((S + 3) & ~3); t++)
            sc[t] = -INFINITY;
         float32x4_t m4 = vdupq_n_f32(mx), sum4 = vdupq_n_f32(0);
         for (int t = 0; t < S; t += 4) {
            float32x4_t e = exp_neg_f32(vsubq_f32(vld1q_f32(sc + t), m4));
            vst1q_f32(sc + t, e);
            sum4 = vaddq_f32(sum4, e);
         }
         float inv = 1.0f / vaddvq_f32(sum4);
         float *o = out + (size_t)h * hd;
         int d0 = 0;
         for (; d0 + 16 <= hd; d0 += 16) {               /* 16 dims per pass: 4 independent accumulators */
            float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
            const float *vt = v + (size_t)kh * hd + d0;
            for (int t = 0; t < S; t++, vt += (size_t)KV * hd) {
               float p = sc[t];
               a0 = vfmaq_n_f32(a0, vld1q_f32(vt), p);
               a1 = vfmaq_n_f32(a1, vld1q_f32(vt + 4), p);
               a2 = vfmaq_n_f32(a2, vld1q_f32(vt + 8), p);
               a3 = vfmaq_n_f32(a3, vld1q_f32(vt + 12), p);
            }
            vst1q_f32(o + d0, vmulq_n_f32(a0, inv));
            vst1q_f32(o + d0 + 4, vmulq_n_f32(a1, inv));
            vst1q_f32(o + d0 + 8, vmulq_n_f32(a2, inv));
            vst1q_f32(o + d0 + 12, vmulq_n_f32(a3, inv));
         }
         for (; d0 < hd; d0 += 4) {
            float32x4_t a = vdupq_n_f32(0);
            for (int t = 0; t < S; t++)
               a = vfmaq_n_f32(a, vld1q_f32(v + ((size_t)t * KV + kh) * hd + d0), sc[t]);
            vst1q_f32(o + d0, vmulq_n_f32(a, inv));
         }
      }
      free(sc);
   }
}

/* decode KV cache, head-major (2026-11-24): layer base [KV][cap][hd], element float (f16 = 0) or IEEE half (f16 = 1,
 * stored as uint16; the A73 converts in hardware, arithmetic stays float32). Each head's keys / values are contiguous,
 * so the decode attention streams them (the [t][KV][hd] layout strided by KV * hd and ran at ~2.8 GB/s). */
static inline float32x4_t
kv_ld4(const void *base, size_t i, int f16)
{
   if (f16)
      return vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16((const uint16_t *)base + i)));
   return vld1q_f32((const float *)base + i);
}

/* rows src [n, KV, hd] (positions pos0 .. pos0+n-1) into the cache layer base dst */
EXPORT void
jbg_kv_store(void *dst, const float *src, int pos0, int n, int KV, int hd, int cap, int f16)
{
   for (int t = 0; t < n; t++)
      for (int h = 0; h < KV; h++) {
         const float *s = src + ((size_t)t * KV + h) * hd;
         size_t o = ((size_t)h * cap + pos0 + t) * hd;
         if (!f16) {
            memcpy((float *)dst + o, s, (size_t)hd * sizeof(float));
            continue;
         }
         uint16_t *d = (uint16_t *)dst + o;
         int i = 0;
         for (; i + 4 <= hd; i += 4)
            vst1_u16(d + i, vreinterpret_u16_f16(vcvt_f16_f32(vld1q_f32(s + i))));
         for (; i < hd; i++) {
            __fp16 hv = (__fp16)s[i];
            memcpy(d + i, &hv, 2);
         }
      }
}

static inline __attribute__((always_inline)) void
attn_head_hm(const float *qh, const void *kb, const void *vb, float *o, float *sc, int S, int hd, float scale,
             int f16)
{
   float mx = -INFINITY;
   int t = 0;
   if (hd % 8 == 0) {
      for (; t + 4 <= S; t += 4) {                  /* 4 keys at a time; each key's sum in dot_f32's order */
         float32x4_t a0 = vdupq_n_f32(0), b0 = a0, a1 = a0, b1 = a0, a2 = a0, b2 = a0, a3 = a0, b3 = a0;
         size_t k0 = (size_t)t * hd;
         for (int i = 0; i < hd; i += 8) {
            float32x4_t qa = vld1q_f32(qh + i), qb = vld1q_f32(qh + i + 4);
            a0 = vfmaq_f32(a0, qa, kv_ld4(kb, k0 + i, f16));
            b0 = vfmaq_f32(b0, qb, kv_ld4(kb, k0 + i + 4, f16));
            a1 = vfmaq_f32(a1, qa, kv_ld4(kb, k0 + hd + i, f16));
            b1 = vfmaq_f32(b1, qb, kv_ld4(kb, k0 + hd + i + 4, f16));
            a2 = vfmaq_f32(a2, qa, kv_ld4(kb, k0 + 2 * hd + i, f16));
            b2 = vfmaq_f32(b2, qb, kv_ld4(kb, k0 + 2 * hd + i + 4, f16));
            a3 = vfmaq_f32(a3, qa, kv_ld4(kb, k0 + 3 * hd + i, f16));
            b3 = vfmaq_f32(b3, qb, kv_ld4(kb, k0 + 3 * hd + i + 4, f16));
         }
         float d0 = vaddvq_f32(vaddq_f32(a0, b0)) * scale, d1 = vaddvq_f32(vaddq_f32(a1, b1)) * scale;
         float d2 = vaddvq_f32(vaddq_f32(a2, b2)) * scale, d3 = vaddvq_f32(vaddq_f32(a3, b3)) * scale;
         sc[t] = d0; sc[t + 1] = d1; sc[t + 2] = d2; sc[t + 3] = d3;
         mx = d0 > mx ? d0 : mx; mx = d1 > mx ? d1 : mx; mx = d2 > mx ? d2 : mx; mx = d3 > mx ? d3 : mx;
      }
   }
   for (; t < S; t++) {
      float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
      size_t k0 = (size_t)t * hd;
      int i = 0;
      for (; i + 8 <= hd; i += 8) {
         s0 = vfmaq_f32(s0, vld1q_f32(qh + i), kv_ld4(kb, k0 + i, f16));
         s1 = vfmaq_f32(s1, vld1q_f32(qh + i + 4), kv_ld4(kb, k0 + i + 4, f16));
      }
      float d = vaddvq_f32(vaddq_f32(s0, s1));
      for (; i < hd; i++)
         d += qh[i] * (f16 ? (float)((const __fp16 *)kb)[k0 + i] : ((const float *)kb)[k0 + i]);
      d *= scale;
      sc[t] = d;
      mx = d > mx ? d : mx;
   }
   for (t = S; t < ((S + 3) & ~3); t++)
      sc[t] = -INFINITY;
   float32x4_t m4 = vdupq_n_f32(mx), sum4 = vdupq_n_f32(0);
   for (t = 0; t < S; t += 4) {
      float32x4_t e = exp_neg_f32(vsubq_f32(vld1q_f32(sc + t), m4));
      vst1q_f32(sc + t, e);
      sum4 = vaddq_f32(sum4, e);
   }
   float inv = 1.0f / vaddvq_f32(sum4);
   int d0 = 0;
   for (; d0 + 16 <= hd; d0 += 16) {
      float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
      size_t vi = d0;
      for (t = 0; t < S; t++, vi += hd) {
         float p = sc[t];
         a0 = vfmaq_n_f32(a0, kv_ld4(vb, vi, f16), p);
         a1 = vfmaq_n_f32(a1, kv_ld4(vb, vi + 4, f16), p);
         a2 = vfmaq_n_f32(a2, kv_ld4(vb, vi + 8, f16), p);
         a3 = vfmaq_n_f32(a3, kv_ld4(vb, vi + 12, f16), p);
      }
      vst1q_f32(o + d0, vmulq_n_f32(a0, inv));
      vst1q_f32(o + d0 + 4, vmulq_n_f32(a1, inv));
      vst1q_f32(o + d0 + 8, vmulq_n_f32(a2, inv));
      vst1q_f32(o + d0 + 12, vmulq_n_f32(a3, inv));
   }
   for (; d0 < hd; d0 += 4) {
      float32x4_t a = vdupq_n_f32(0);
      for (t = 0; t < S; t++)
         a = vfmaq_n_f32(a, kv_ld4(vb, (size_t)t * hd + d0, f16), sc[t]);
      vst1q_f32(o + d0, vmulq_n_f32(a, inv));
   }
}

/* jbg_attn_decode on the head-major cache: k / v = this layer's base [KV][cap][hd] (f16: halves). With f16 = 0 the
 * result is bit-identical to jbg_attn_decode on the same values. hd % 4 == 0. */
EXPORT void
jbg_attn_decode_hm(const float *q, const void *k, const void *v, float *out, int S, int H, int KV, int hd, int cap,
                   int f16)
{
   int g = H / KV;
   float scale = 1.0f / sqrtf((float)hd);
   size_t es = f16 ? 2 : 4;
#pragma omp parallel
   {
      float *sc = malloc(((size_t)S + 4) * sizeof(float));
#pragma omp for schedule(dynamic)
      for (int h = 0; h < H; h++) {
         int kh = h / g;
         const char *kb = (const char *)k + (size_t)kh * cap * hd * es, *vb = (const char *)v + (size_t)kh * cap * hd * es;
         if (f16)
            attn_head_hm(q + (size_t)h * hd, kb, vb, out + (size_t)h * hd, sc, S, hd, scale, 1);
         else
            attn_head_hm(q + (size_t)h * hd, kb, vb, out + (size_t)h * hd, sc, S, hd, scale, 0);
      }
      free(sc);
   }
}

/* position 0 on the CPU: y[rows[j]] (+)= sw * sum_c w[j, c] * x[c] for one job's int8 weights [n, ic] */
EXPORT void
jbg_gemv_i8(const float *x, const int8_t *w, int n, int ic, float sw, const int32_t *rows, float *y, int accumulate)
{
#pragma omp parallel for schedule(dynamic)
   for (int j = 0; j < n; j++) {
      const int8_t *r = w + (size_t)j * ic;
      float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0), s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
      int c = 0;
      for (; c + 16 <= ic; c += 16) {
         int8x16_t b = vld1q_s8(r + c);
         int16x8_t lo = vmovl_s8(vget_low_s8(b)), hi = vmovl_s8(vget_high_s8(b));
         s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(x + c));
         s1 = vfmaq_f32(s1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(x + c + 4));
         s2 = vfmaq_f32(s2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(x + c + 8));
         s3 = vfmaq_f32(s3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(x + c + 12));
      }
      float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
      for (; c < ic; c++)
         s += (float)r[c] * x[c];
      if (accumulate)
         y[rows[j]] += s * sw;
      else
         y[rows[j]] = s * sw;
   }
}

/* RoPE in place, non-interleaved (rotate_half): x [T, heads, hd], positions pos0 .. pos0+T-1 */
EXPORT void
jbg_rope(float *x, int T, int heads, int hd, int pos0, const double *inv_freq)
{
   int h2 = hd / 2;
#pragma omp parallel for schedule(dynamic)
   for (int t = 0; t < T; t++) {
      double cs[256], sn[256];                       /* numpy: float64 cos / sin, products in float64 */
      for (int i = 0; i < h2; i++) {
         double f = (double)(pos0 + t) * inv_freq[i];
         cs[i] = cos(f);
         sn[i] = sin(f);
      }
      for (int h = 0; h < heads; h++) {
         float *v = x + ((size_t)t * heads + h) * hd;
         for (int i = 0; i < h2; i++) {
            double a = v[i], b = v[i + h2];
            v[i] = (float)(a * cs[i] - b * sn[i]);
            v[i + h2] = (float)(b * cs[i] + a * sn[i]);
         }
      }
   }
}

/* causal attention, query positions q0 .. q0+T-1 over keys 0 .. S-1 (S = q0 + T):
 * q [T, H, hd], k / v [S, KV, hd] -> out [T, H * hd]; head h reads kv head h / (H / KV).
 * Per (head, block of 4 queries): scores from the transposed keys (4 queries x 4 keys per FMA step), softmax with a
 * NEON exp, then P V in chunks of 16 dims (4 queries x 4 vectors of accumulators). hd <= 256, S <= 8192. */
#define QB 4
EXPORT void
jbg_attention(const float *q, const float *k, const float *v, float *out, int T, int S, int q0, int H, int KV, int hd)
{
   int g = H / KV;
   float scale = 1.0f / sqrtf((float)hd);
   int Sp = (S + 3) & ~3;                               /* keys padded to a multiple of 4 */
   int prof = getenv("JBG_PROF") != NULL;
   double tk0 = now_s(), ph[3] = {0, 0, 0};
   float *kt = malloc((size_t)KV * hd * Sp * sizeof(float));   /* kt[kh][d][s] (keys transposed) */
   float *vh = malloc((size_t)KV * Sp * hd * sizeof(float));   /* vh[kh][s][d] (values, head-major, contiguous) */
   int nsb = Sp / 16 + (Sp % 16 != 0);
#pragma omp parallel for schedule(dynamic) collapse(2)
   for (int kh = 0; kh < KV; kh++)
      for (int b = 0; b < nsb; b++) {                   /* 16 keys: contiguous row reads, a [hd][16] tile */
         int s0 = b * 16, s1 = s0 + 16 < Sp ? s0 + 16 : Sp;
         float tile[256][16];
         for (int s = s0; s < s1; s++) {
            if (s < S) {
               const float *kr = k + ((size_t)s * KV + kh) * hd;
               for (int d = 0; d < hd; d++)
                  tile[d][s - s0] = kr[d];
               memcpy(vh + ((size_t)kh * Sp + s) * hd, v + ((size_t)s * KV + kh) * hd, hd * sizeof(float));
            } else {
               for (int d = 0; d < hd; d++)
                  tile[d][s - s0] = 0;
               memset(vh + ((size_t)kh * Sp + s) * hd, 0, hd * sizeof(float));
            }
         }
         for (int d = 0; d < hd; d++)
            memcpy(kt + ((size_t)kh * hd + d) * Sp + s0, tile[d], (s1 - s0) * sizeof(float));
      }
   int nqb = (T + QB - 1) / QB;
   double tk1 = now_s();
#pragma omp parallel
   {
      double my[3] = {0, 0, 0}, tq;
      float *sc = malloc((size_t)QB * Sp * sizeof(float));
      float *qt = malloc((size_t)4 * hd * sizeof(float));
      float *pt = malloc((size_t)4 * Sp * sizeof(float));     /* softmax weights transposed: pt[s][i] */
#pragma omp for schedule(dynamic, 2) collapse(2)
      for (int h = 0; h < H; h++)
         for (int b = 0; b < nqb; b++) {
            int t0 = b * QB, nq = T - t0 < QB ? T - t0 : QB;
            int kh = h / g;
            int last_blk = q0 + t0 + nq - 1;              /* the block's last key */
            int ns = (last_blk + 1 + 3) & ~3;
            const float *qv[QB];
            for (int i = 0; i < QB; i++)
               qv[i] = q + ((size_t)(t0 + (i < nq ? i : 0)) * H + h) * hd;
            const float *K = kt + (size_t)kh * hd * Sp;
            const float *q0p = qv[0], *q1p = qv[1], *q2p = qv[2], *q3p = qv[3];
            tq = prof ? now_s() : 0;
            for (int d = 0; d < hd; d++) {                  /* queries transposed: qt[d][i] */
               qt[4 * d] = q0p[d];
               qt[4 * d + 1] = q1p[d];
               qt[4 * d + 2] = q2p[d];
               qt[4 * d + 3] = q3p[d];
            }
            int s = 0;
            for (; s + 16 <= ns; s += 16) {                 /* 4 queries x 16 keys, 16 named accumulators */
               float32x4_t a00 = vdupq_n_f32(0), a01 = a00, a02 = a00, a03 = a00;
               float32x4_t a10 = a00, a11 = a00, a12 = a00, a13 = a00;
               float32x4_t a20 = a00, a21 = a00, a22 = a00, a23 = a00;
               float32x4_t a30 = a00, a31 = a00, a32 = a00, a33 = a00;
               const float *kr = K + s;
               for (int d = 0; d < hd; d++, kr += Sp) {
                  float32x4_t k0 = vld1q_f32(kr), k1 = vld1q_f32(kr + 4), k2 = vld1q_f32(kr + 8), k3 = vld1q_f32(kr + 12);
                  float32x4_t qq = vld1q_f32(qt + 4 * d);      /* the 4 queries' d-th values: one load */
                  a00 = vfmaq_laneq_f32(a00, k0, qq, 0); a01 = vfmaq_laneq_f32(a01, k1, qq, 0);
                  a02 = vfmaq_laneq_f32(a02, k2, qq, 0); a03 = vfmaq_laneq_f32(a03, k3, qq, 0);
                  a10 = vfmaq_laneq_f32(a10, k0, qq, 1); a11 = vfmaq_laneq_f32(a11, k1, qq, 1);
                  a12 = vfmaq_laneq_f32(a12, k2, qq, 1); a13 = vfmaq_laneq_f32(a13, k3, qq, 1);
                  a20 = vfmaq_laneq_f32(a20, k0, qq, 2); a21 = vfmaq_laneq_f32(a21, k1, qq, 2);
                  a22 = vfmaq_laneq_f32(a22, k2, qq, 2); a23 = vfmaq_laneq_f32(a23, k3, qq, 2);
                  a30 = vfmaq_laneq_f32(a30, k0, qq, 3); a31 = vfmaq_laneq_f32(a31, k1, qq, 3);
                  a32 = vfmaq_laneq_f32(a32, k2, qq, 3); a33 = vfmaq_laneq_f32(a33, k3, qq, 3);
               }
               float *o0 = sc + s, *o1 = sc + Sp + s, *o2 = sc + 2 * Sp + s, *o3 = sc + 3 * Sp + s;
               vst1q_f32(o0, vmulq_n_f32(a00, scale)); vst1q_f32(o0 + 4, vmulq_n_f32(a01, scale));
               vst1q_f32(o0 + 8, vmulq_n_f32(a02, scale)); vst1q_f32(o0 + 12, vmulq_n_f32(a03, scale));
               vst1q_f32(o1, vmulq_n_f32(a10, scale)); vst1q_f32(o1 + 4, vmulq_n_f32(a11, scale));
               vst1q_f32(o1 + 8, vmulq_n_f32(a12, scale)); vst1q_f32(o1 + 12, vmulq_n_f32(a13, scale));
               vst1q_f32(o2, vmulq_n_f32(a20, scale)); vst1q_f32(o2 + 4, vmulq_n_f32(a21, scale));
               vst1q_f32(o2 + 8, vmulq_n_f32(a22, scale)); vst1q_f32(o2 + 12, vmulq_n_f32(a23, scale));
               vst1q_f32(o3, vmulq_n_f32(a30, scale)); vst1q_f32(o3 + 4, vmulq_n_f32(a31, scale));
               vst1q_f32(o3 + 8, vmulq_n_f32(a32, scale)); vst1q_f32(o3 + 12, vmulq_n_f32(a33, scale));
            }
            for (; s < ns; s += 4) {
               float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
               for (int d = 0; d < hd; d++) {
                  float32x4_t kk = vld1q_f32(K + (size_t)d * Sp + s);
                  a0 = vfmaq_n_f32(a0, kk, q0p[d]);
                  a1 = vfmaq_n_f32(a1, kk, q1p[d]);
                  a2 = vfmaq_n_f32(a2, kk, q2p[d]);
                  a3 = vfmaq_n_f32(a3, kk, q3p[d]);
               }
               vst1q_f32(sc + 0 * Sp + s, vmulq_n_f32(a0, scale));
               vst1q_f32(sc + 1 * Sp + s, vmulq_n_f32(a1, scale));
               vst1q_f32(sc + 2 * Sp + s, vmulq_n_f32(a2, scale));
               vst1q_f32(sc + 3 * Sp + s, vmulq_n_f32(a3, scale));
            }
            if (prof) {
               double t2 = now_s();
               my[0] += t2 - tq;
               tq = t2;
            }
            float inv[QB];
            int nk[QB];
            for (int i = 0; i < nq; i++) {
               float *r = sc + (size_t)i * Sp;
               int n_ = q0 + t0 + i + 1;                  /* causal: keys 0 .. q0+t */
               nk[i] = n_;
               float mx = -INFINITY;
               for (int s = 0; s < n_; s++)
                  mx = r[s] > mx ? r[s] : mx;
               for (int s = n_; s < ns; s++)
                  r[s] = -INFINITY;
               float32x4_t m4 = vdupq_n_f32(mx), sum4 = vdupq_n_f32(0);
               for (int s = 0; s < ns; s += 4) {
                  float32x4_t e = exp_neg_f32(vsubq_f32(vld1q_f32(r + s), m4));
                  vst1q_f32(r + s, e);
                  sum4 = vaddq_f32(sum4, e);
               }
               inv[i] = 1.0f / vaddvq_f32(sum4);
            }
            for (int i = nq; i < QB; i++) {                 /* rows past the end of the chunk: zero weights */
               nk[i] = 0;
               inv[i] = 0;
               memset(sc + (size_t)i * Sp, 0, ns * sizeof(float));
            }
            int nmax = nk[0];
            for (int i = 1; i < nq; i++)
               nmax = nk[i] > nmax ? nk[i] : nmax;
            const float *V = vh + (size_t)kh * Sp * hd;       /* this head's values, rows of hd contiguous */
            size_t vs = (size_t)hd;
            if (prof) {
               double t2 = now_s();
               my[1] += t2 - tq;
               tq = t2;
            }
            const float *p0 = sc, *p1 = sc + Sp, *p2 = sc + 2 * Sp, *p3 = sc + 3 * Sp;
            for (int s2 = 0; s2 < nmax; s2++) {              /* weights transposed: one load per key below */
               pt[4 * s2] = p0[s2];
               pt[4 * s2 + 1] = p1[s2];
               pt[4 * s2 + 2] = p2[s2];
               pt[4 * s2 + 3] = p3[s2];
            }
            for (int d0 = 0; d0 < hd; d0 += 16) {           /* 4 queries x 16 dims, 16 named accumulators */
               float32x4_t c00 = vdupq_n_f32(0), c01 = c00, c02 = c00, c03 = c00;
               float32x4_t c10 = c00, c11 = c00, c12 = c00, c13 = c00;
               float32x4_t c20 = c00, c21 = c00, c22 = c00, c23 = c00;
               float32x4_t c30 = c00, c31 = c00, c32 = c00, c33 = c00;
               const float *vr = V + d0;
               for (int s2 = 0; s2 < nmax; s2++, vr += vs) {
                  float32x4_t v0 = vld1q_f32(vr), v1 = vld1q_f32(vr + 4), v2 = vld1q_f32(vr + 8), v3 = vld1q_f32(vr + 12);
                  float32x4_t ww = vld1q_f32(pt + 4 * s2);
                  c00 = vfmaq_laneq_f32(c00, v0, ww, 0); c01 = vfmaq_laneq_f32(c01, v1, ww, 0);
                  c02 = vfmaq_laneq_f32(c02, v2, ww, 0); c03 = vfmaq_laneq_f32(c03, v3, ww, 0);
                  c10 = vfmaq_laneq_f32(c10, v0, ww, 1); c11 = vfmaq_laneq_f32(c11, v1, ww, 1);
                  c12 = vfmaq_laneq_f32(c12, v2, ww, 1); c13 = vfmaq_laneq_f32(c13, v3, ww, 1);
                  c20 = vfmaq_laneq_f32(c20, v0, ww, 2); c21 = vfmaq_laneq_f32(c21, v1, ww, 2);
                  c22 = vfmaq_laneq_f32(c22, v2, ww, 2); c23 = vfmaq_laneq_f32(c23, v3, ww, 2);
                  c30 = vfmaq_laneq_f32(c30, v0, ww, 3); c31 = vfmaq_laneq_f32(c31, v1, ww, 3);
                  c32 = vfmaq_laneq_f32(c32, v2, ww, 3); c33 = vfmaq_laneq_f32(c33, v3, ww, 3);
               }
               float32x4_t cc[QB][4] = {{c00, c01, c02, c03}, {c10, c11, c12, c13}, {c20, c21, c22, c23},
                                        {c30, c31, c32, c33}};
               for (int i = 0; i < nq; i++) {
                  float *o = out + (size_t)(t0 + i) * H * hd + (size_t)h * hd + d0;
                  for (int c = 0; c < 4; c++)
                     vst1q_f32(o + 4 * c, vmulq_n_f32(cc[i][c], inv[i]));
               }
            }
            if (prof)
               my[2] += now_s() - tq;
         }
      if (prof) {
#pragma omp critical
         {
            ph[0] += my[0];
            ph[1] += my[1];
            ph[2] += my[2];
         }
      }
      free(sc);
      free(qt);
      free(pt);
   }
   free(kt);
   free(vh);
   if (prof)
      fprintf(stderr, "jbg_attention: K transpose %.2f ms; thread-summed: scores %.2f, softmax %.2f, PV %.2f ms; "
              "wall %.2f ms\n", (tk1 - tk0) * 1e3, ph[0] * 1e3, ph[1] * 1e3, ph[2] * 1e3, (now_s() - tk0) * 1e3);
}

/* h[i] = gate[i] / (1 + exp(-gate[i])) * up[i]; gate / up are column blocks of gu [T, 2 * I] (NEON exp) */
EXPORT void
jbg_silu_mul(const float *gu, int T, int I, float *h)
{
#pragma omp parallel for schedule(dynamic)
   for (int t = 0; t < T; t++) {
      const float *g = gu + (size_t)t * 2 * I, *u = g + I;
      float *o = h + (size_t)t * I;
      int i = 0;
      float32x4_t one = vdupq_n_f32(1.0f);
      for (; i + 4 <= I; i += 4) {
         float32x4_t gv = vld1q_f32(g + i);
         float32x4_t e = exp_f32(vnegq_f32(gv));
         vst1q_f32(o + i, vmulq_f32(vdivq_f32(gv, vaddq_f32(one, e)), vld1q_f32(u + i)));
      }
      for (; i < I; i++)
         o[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
   }
}

/* in place: x[t] = blockwise normalized FWHT(x[t] * s) over blocks of b (a power of 2) along the last axis of [T, n]
 * (l2g_outliers BlockH.right with signs s [n]; s NULL = no signs, i.e. plain fwht); post_signs: multiply by s AFTER
 * the transform instead (Orth.back: fwht(A) * s) */
EXPORT void
jbg_fwht(float *x, int T, int n, int b, const float *s, int post_signs)
{
   float norm = (float)(1.0 / sqrt((double)b));          /* = np.float32(1 / np.sqrt(b)) */
   if (b >= 4 && s && !post_signs) {                   /* the radix-4 row kernel (bit-exact) */
#pragma omp parallel for schedule(dynamic)
      for (int t = 0; t < T; t++)
         fwht_row(x + (size_t)t * n, n, b, s);
      return;
   }
#pragma omp parallel for schedule(dynamic)
   for (int t = 0; t < T; t++) {
      float *r = x + (size_t)t * n;
      if (s && !post_signs)
         for (int i = 0; i < n; i++)
            r[i] *= s[i];
      for (int o = 0; o < n; o += b) {
         float *y = r + o;
         for (int h = 1; h < b && h < 4; h *= 2)          /* the first two stages scalar */
            for (int i = 0; i < b; i += 2 * h)
               for (int j = i; j < i + h; j++) {
                  float a = y[j], c = y[j + h];
                  y[j] = a + c;
                  y[j + h] = a - c;
               }
         for (int h = 4; h < b; h *= 2)                   /* then 4 lanes at a time */
            for (int i = 0; i < b; i += 2 * h)
               for (int j = i; j < i + h; j += 4) {
                  float32x4_t a = vld1q_f32(y + j), c = vld1q_f32(y + j + h);
                  vst1q_f32(y + j, vaddq_f32(a, c));
                  vst1q_f32(y + j + h, vsubq_f32(a, c));
               }
         float32x4_t nv = vdupq_n_f32(norm);
         int i = 0;
         for (; i + 4 <= b; i += 4)
            vst1q_f32(y + i, vmulq_f32(vld1q_f32(y + i), nv));
         for (; i < b; i++)
            y[i] *= norm;
      }
      if (s && post_signs)
         for (int i = 0; i < n; i++)
            r[i] *= s[i];
   }
}

/* ---- the tied head (2026-11-24) ---- */

static inline void
bf16x8(const uint16_t *p, float32x4_t *lo, float32x4_t *hi)
{
   uint16x8_t u = vld1q_u16(p);
   *lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(u), 16));
   *hi = vreinterpretq_f32_u32(vshll_high_n_u16(u, 16));
}

static inline float
bf16f(uint16_t h)
{
   uint32_t u = (uint32_t)h << 16;
   float f;
   memcpy(&f, &u, 4);
   return f;
}

/* NT tokens (1 or 2) x 4 vocab rows: the same per-pair summation as dot_f32 (lanes 0-3 and 4-7 of each 8-wide step in
 * two accumulators, vaddvq of their sum, then the scalar tail in order), so the logits are bit-identical to it */
static inline void
head_4xn(const float *const *x, int NT, const uint16_t *e, int D, float *const *o)
{
   float32x4_t a0[2][4], a1[2][4];
   for (int t = 0; t < 2; t++)
      for (int r = 0; r < 4; r++)
         a0[t][r] = a1[t][r] = vdupq_n_f32(0);
   int d = 0;
   for (; d + 8 <= D; d += 8) {
      float32x4_t el[4], eh[4];
      for (int r = 0; r < 4; r++)
         bf16x8(e + (size_t)r * D + d, &el[r], &eh[r]);
      for (int t = 0; t < NT; t++) {
         float32x4_t xl = vld1q_f32(x[t] + d), xh = vld1q_f32(x[t] + d + 4);
         for (int r = 0; r < 4; r++) {
            a0[t][r] = vfmaq_f32(a0[t][r], xl, el[r]);
            a1[t][r] = vfmaq_f32(a1[t][r], xh, eh[r]);
         }
      }
   }
   for (int t = 0; t < NT; t++)
      for (int r = 0; r < 4; r++) {
         float s = vaddvq_f32(vaddq_f32(a0[t][r], a1[t][r]));
         for (int k = d; k < D; k++)
            s += x[t][k] * bf16f(e[(size_t)r * D + k]);
         o[t][r] = s;
      }
}

/* out[i * V + v] = x[i] . E[v] for i < nl, v < V; E [V, D] bf16 (the raw uint16 words). The bf16 rows are widened
 * in registers (never stored), 4 rows x 2 tokens at a time; tokens in blocks of 32 (fp32 rows stay in cache while the
 * table streams by: the 1.7B's 201 MB table is read ceil(nl / 32) times). Bit-identical to dot_f32 on widened rows. */
EXPORT void
jbg_head_bf16(const float *x, int nl, int D, const uint16_t *E, int V, float *out)
{
   const int TB = 32;
   for (int tb = 0; tb < nl; tb += TB) {
      int tn = nl - tb < TB ? nl - tb : TB;
#pragma omp parallel for schedule(dynamic, 64)
      for (int vb = 0; vb < V; vb += 4) {
         const uint16_t *e = E + (size_t)vb * D;
         if (vb + 4 > V) {                        /* the last few rows (V % 4) */
            float *row = malloc((size_t)D * sizeof(float));
            for (int v = vb; v < V; v++) {
               for (int k = 0; k < D; k++)
                  row[k] = bf16f(E[(size_t)v * D + k]);
               for (int t = tb; t < tb + tn; t++)
                  out[(size_t)t * V + v] = dot_f32(x + (size_t)t * D, row, D);
            }
            free(row);
            continue;
         }
         for (int t = tb; t < tb + tn; t += 2) {
            int NT = t + 1 < tb + tn ? 2 : 1;
            const float *xs[2] = {x + (size_t)t * D, x + (size_t)(t + NT - 1) * D};
            float o[2][4];
            float *os[2] = {o[0], o[1]};
            head_4xn(xs, NT, e, D, os);
            for (int i = 0; i < NT; i++)
               memcpy(out + (size_t)(t + i) * V + vb, o[i], 4 * sizeof(float));
         }
      }
   }
}

/* int8 head (decode, 2026-11-24): E8 [V, D] = rint(E / sv[v]), sv[v] = max |E[v]| / 127 (per row, from the bf16
 * table) -- half the bytes per token of jbg_head_bf16 (1.7B: 101 vs 201 MB) */
EXPORT void
jbg_head_i8_build(const uint16_t *E, int V, int D, int8_t *E8, float *sv)
{
#pragma omp parallel for schedule(dynamic, 64)
   for (int v = 0; v < V; v++) {
      const uint16_t *e = E + (size_t)v * D;
      float mx = 0;
      for (int k = 0; k < D; k++)
         mx = fabsf(bf16f(e[k])) > mx ? fabsf(bf16f(e[k])) : mx;
      float s = mx > 0 ? mx / 127.0f : 1.0f, inv = 1.0f / s;
      for (int k = 0; k < D; k++)
         E8[(size_t)v * D + k] = (int8_t)lrintf(bf16f(e[k]) * inv);
      sv[v] = s;
   }
}

/* out[v] = x . E[v] from the int8 table: x to int16 (one scale), exact integer dots (as jbg_gemv_jobs_q16), each
 * scaled by sv[v] * sx. xq: scratch of D int16s. D % 16 == 0. */
EXPORT void
jbg_head_i8(const float *x, int D, const int8_t *E8, const float *sv, int V, float *out, int16_t *xq)
{
   float sx = jbg_quant16(x, D, xq);
#pragma omp parallel for schedule(dynamic, 16)
   for (int vb = 0; vb < V; vb += 4) {
      if (vb + 4 > V) {
         for (int v = vb; v < V; v++)
            out[v] = q16_row1(E8 + (size_t)v * D, xq, D) * (sv[v] * sx);
         continue;
      }
      const int8_t *w0 = E8 + (size_t)vb * D, *w1 = w0 + D, *w2 = w1 + D, *w3 = w2 + D;
      Q16_ROW_DECL(0);
      Q16_ROW_DECL(1);
      Q16_ROW_DECL(2);
      Q16_ROW_DECL(3);
      int64_t f0 = 0, f1 = 0, f2 = 0, f3 = 0;
      int c = 0;
      while (c + 16 <= D) {
         int end = c + Q16_BLOCK < D ? c + Q16_BLOCK : D;
         for (; c + 16 <= end; c += 16) {
            int16x8_t xa = vld1q_s16(xq + c), xb = vld1q_s16(xq + c + 8);
            Q16_ROW_STEP(0, w0);
            Q16_ROW_STEP(1, w1);
            Q16_ROW_STEP(2, w2);
            Q16_ROW_STEP(3, w3);
         }
         Q16_ROW_FLUSH(0);
         Q16_ROW_FLUSH(1);
         Q16_ROW_FLUSH(2);
         Q16_ROW_FLUSH(3);
      }
      out[vb] = (float)f0 * (sv[vb] * sx);
      out[vb + 1] = (float)f1 * (sv[vb + 1] * sx);
      out[vb + 2] = (float)f2 * (sv[vb + 2] * sx);
      out[vb + 3] = (float)f3 * (sv[vb + 3] * sx);
   }
}

EXPORT int
jbg_threads(void)
{
   return omp_get_max_threads();
}
