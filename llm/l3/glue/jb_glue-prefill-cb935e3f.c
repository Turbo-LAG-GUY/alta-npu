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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static) collapse(2)
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
#pragma omp parallel for schedule(static) collapse(2)
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
#pragma omp for schedule(static)
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
               int nn = n[j];
               for (int k = 0; k < nn; k++)
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

/* one token row of a step's output in model order: o[rows] = sum of the jobs' dequantized outputs */
static inline void
row_from_jobs(int nj, const int8_t *const *src, const long *sst, int i, const int *n, const int32_t *const *rows,
              const float *s, const int *zp, float *o, int oc)
{
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
#pragma omp for schedule(static)
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
   double ph[5] = {0, 0, 0, 0, 0}, w0 = now_s();
#pragma omp parallel
   {
      double my[5] = {0, 0, 0, 0, 0}, tq = 0, t2;
      int8_t *tm = malloc((size_t)16 * tot);
      float *gu = malloc((size_t)2 * I * sizeof(float));
      float *hb = malloc((size_t)16 * I * sizeof(float));
      const int8_t *src[256];
      long sst[256];
      float32x4_t one = vdupq_n_f32(1.0f);
#pragma omp for schedule(static)
      for (int t0 = 0; t0 < T; t0 += 16) {
         int nt = T - t0 < 16 ? T - t0 : 16;
         if (prof) tq = now_s();
         block_rows(nj, lv, ts, js, n, t0, nt, tm, off, src, sst);
         if (prof) { t2 = now_s(); my[0] += t2 - tq; tq = t2; }
         for (int i = 0; i < nt; i++) {
            row_from_jobs(nj, src, sst, i, n, rows, s, zp, gu, 2 * I);
            if (prof) { t2 = now_s(); my[1] += t2 - tq; tq = t2; }
            float *h = hb + (size_t)i * I;
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
         }
         quant_block(hb, I, nt, I, sx, zx, out, T, t0);
         if (prof) { t2 = now_s(); my[4] += t2 - tq; tq = t2; }
      }
      if (prof) {
#pragma omp critical
         for (int q_ = 0; q_ < 5; q_++)
            ph[q_] += my[q_];
      }
      free(tm);
      free(gu);
      free(hb);
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
#pragma omp parallel
   {
      int8_t *tm = malloc((size_t)16 * tot);
      float *row = malloc((size_t)oc * sizeof(float));
      const int8_t *src[256];
      long sst[256];
#pragma omp for schedule(static)
      for (int t0 = 0; t0 < T; t0 += 16) {
         int nt = T - t0 < 16 ? T - t0 : 16;
         block_rows(nj, lv, ts, js, n, t0, nt, tm, off, src, sst);
         for (int i = 0; i < nt; i++) {
            int t = 1 + t0 + i;
            row_from_jobs(nj, src, sst, i, n, rows, s, zp, row, oc);
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

/* All jobs of one step, position 0: y[rows_j[r]] = sum over jobs of sw_j * w_j[r, :] . x[col_j .. col_j + ic_j)
 * (y zeroed here). One parallel region over every job's rows. */
EXPORT void
jbg_gemv_jobs(int nj, const float *x, const int8_t *const *w, const int *n, const int *ic, const int *col,
              const float *sw, const int32_t *const *rows, float *y, int oc, int accumulate)
{
   if (!accumulate)
      memset(y, 0, oc * sizeof(float));
   long total = 0;
   long start[256];
   for (int j = 0; j < nj && j < 256; j++) {
      start[j] = total;
      total += n[j];
   }
   /* rows of different pieces of a split step can be the same output row: per-thread partial sums would be needed
    * for parallel accumulation; instead parallelise within a job, jobs in order (a job's rows are distinct) */
   for (int j = 0; j < nj; j++) {
      const int8_t *wj = w[j];
      const float *xj = x + col[j];
      int icj = ic[j];
      const int32_t *rj = rows[j];
      float s = sw[j];
#pragma omp parallel for schedule(static)
      for (int r = 0; r < n[j]; r++) {
         const int8_t *wr = wj + (size_t)r * icj;
         float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0), s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
         int c = 0;
         for (; c + 16 <= icj; c += 16) {
            int8x16_t b = vld1q_s8(wr + c);
            int16x8_t lo = vmovl_s8(vget_low_s8(b)), hi = vmovl_s8(vget_high_s8(b));
            s0 = vfmaq_f32(s0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), vld1q_f32(xj + c));
            s1 = vfmaq_f32(s1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))), vld1q_f32(xj + c + 4));
            s2 = vfmaq_f32(s2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), vld1q_f32(xj + c + 8));
            s3 = vfmaq_f32(s3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))), vld1q_f32(xj + c + 12));
         }
         float acc = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
         for (; c < icj; c++)
            acc += (float)wr[c] * xj[c];
         y[rj[r]] += acc * s;
      }
   }
   (void)start;
}

/* position 0 on the CPU: y[rows[j]] (+)= sw * sum_c w[j, c] * x[c] for one job's int8 weights [n, ic] */
EXPORT void
jbg_gemv_i8(const float *x, const int8_t *w, int n, int ic, float sw, const int32_t *rows, float *y, int accumulate)
{
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static) collapse(2)
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
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static)
      for (int t = 0; t < T; t++)
         fwht_row(x + (size_t)t * n, n, b, s);
      return;
   }
#pragma omp parallel for schedule(static)
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

EXPORT int
jbg_threads(void)
{
   return omp_get_max_threads();
}
