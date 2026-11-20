/*
 * jb_run.c (2026-11-19, LLM engine L3b) - NN jobs through the Gallium ML API directly: no TFLite, no Teflon delegate.
 *
 * Reads a job pack (llm/l3/jb_pack.py: tensors + 1x1-conv operations of one subgraph = one NPU submit, a test input
 * and the expected outputs), compiles it with the driver's own ml_subgraph_create (Mesa's NN config / weight
 * compiler, so the jobs are the ones Teflon would build), invokes it N times and reports per-phase times and the
 * largest output difference.
 *
 *   jb_run [--offline] [--ctx persist|per-invoke] [-n N] [--batch K] [--wait mesa|block|spin]
 *          [--cpu-rows R [--cpu-ic IC]] PACK.jbp
 *   jb_run --cpu-only --cpu-rows R --cpu-ic IC [-n N]          (no device opened at all)
 *
 * --offline     the fake NPU (LD_PRELOAD=.../npu-shader/shim/libnpu_etna_shim.so, run inside bwrap with an empty /dev);
 *               refused without the shim, and a device run is refused with it (same rule as cs_run).
 * --ctx         persist = one pipe_context for every invoke (the engine); per-invoke = create + destroy a context
 *               around every invoke, as Teflon's partition_invoke does.
 * --batch K     invoke the subgraph K times before one read (K copies of its jobs in one submit).
 * --wait        mesa  = read_output flushes and waits (Teflon's way, the default);
 *               block = flush, then a blocking fence wait, then read_output (no second submit: an empty flush is a
 *                       no-op in etna_cmd_stream_flush);
 *               spin  = flush, then poll the fence without sleeping (does the CPU's wake-up cost show in the 0.16 ms?).
 * --cpu-rows R  the CPU + NPU split: 4 threads on the A73s (cpus 2-5, spinning between steps) compute an int8 matrix x
 *               vector of R rows x IC (random weights, NEON) between the flush and the wait, i.e. while the NPU runs;
 *               reported: npu_ms (flush -> fence), cpu_ms, both together, and the CPU's GB/s. Needs --wait block|spin.
 * --cpu-only    the CPU part alone (no NPU, no device): its speed without the NPU streaming next to it.
 *
 * Device runs ONLY through scripts/npu-run with Jordan's OK for that step.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <arm_neon.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>

#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "etnaviv/drm/etnaviv_drm_public.h"

#define MAX_T 256
#define MAX_OPS 128

struct jb_tensor {
   int32_t index;
   uint32_t dims[4];
   float scale;
   int32_t zp;
   uint8_t is_signed, is_constant, type_size, is_output;
   uint32_t n_bytes;
   uint8_t *data;
};

static struct jb_tensor T[MAX_T];
static struct pipe_tensor PT[MAX_T];
static struct pipe_ml_operation OPS[MAX_OPS];
static struct pipe_tensor *OP_IN[MAX_OPS][3], *OP_OUT[MAX_OPS][1];
static unsigned n_t, n_ops;
static uint8_t *input;
static uint32_t input_bytes;
static uint8_t *expected[MAX_T];
static uint32_t expected_bytes[MAX_T];

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void
rd(FILE *f, void *p, size_t n)
{
   if (n && fread(p, 1, n, f) != n) {
      fprintf(stderr, "pack truncated\n");
      exit(2);
   }
}

static void
load_pack(const char *path)
{
   FILE *f = fopen(path, "rb");
   char magic[4];
   uint32_t hdr[2];
   if (!f) {
      perror(path);
      exit(2);
   }
   rd(f, magic, 4);
   rd(f, hdr, 8);
   if (memcmp(magic, "JBP1", 4) || hdr[0] > MAX_T || hdr[1] > MAX_OPS) {
      fprintf(stderr, "%s: not a JBP1 pack (or too big)\n", path);
      exit(2);
   }
   n_t = hdr[0];
   n_ops = hdr[1];
   for (unsigned i = 0; i < n_t; i++) {
      struct jb_tensor *t = &T[i];
      rd(f, &t->index, 4);
      rd(f, t->dims, 16);
      rd(f, &t->scale, 4);
      rd(f, &t->zp, 4);
      rd(f, &t->is_signed, 4); /* the four u8 flags */
      rd(f, &t->n_bytes, 4);
      if ((unsigned)t->index != i) {
         fprintf(stderr, "tensor %u has index %d\n", i, t->index);
         exit(2);
      }
      t->data = t->n_bytes ? malloc(t->n_bytes) : NULL;
      rd(f, t->data, t->n_bytes);
   }
   for (unsigned i = 0; i < n_ops; i++) {
      uint32_t type;
      int32_t in, w, b, out;
      rd(f, &type, 4);
      rd(f, &in, 4);
      rd(f, &w, 4);
      rd(f, &b, 4);
      rd(f, &out, 4);
      struct pipe_ml_operation *op = &OPS[i];
      if (type == 2 && in >= 0 && out >= 0 && (unsigned)in < n_t && (unsigned)out < n_t) {
         /* explicit TRANSPOSE at a graph end: the driver makes it a bypass and marks the tensor channels-first */
         op->type = PIPE_ML_OPERATION_TYPE_TRANSPOSE;
         OP_IN[i][0] = &PT[in];
         OP_OUT[i][0] = &PT[out];
         op->input_tensors = OP_IN[i];
         op->input_count = 1;
         op->output_tensors = OP_OUT[i];
         op->output_count = 1;
         for (unsigned k = 0; k < 4; k++)
            op->transpose.perm[k] = ((uint32_t)w >> (8 * k)) & 0xff;
         continue;
      }
      if (type != 1 || in < 0 || w < 0 || out < 0 || (unsigned)in >= n_t || (unsigned)w >= n_t ||
          (unsigned)out >= n_t || (b >= 0 && (unsigned)b >= n_t)) {
         fprintf(stderr, "bad op %u\n", i);
         exit(2);
      }
      op->type = PIPE_ML_OPERATION_TYPE_CONVOLUTION;
      OP_IN[i][0] = &PT[in];
      OP_IN[i][1] = &PT[w];
      OP_IN[i][2] = b >= 0 ? &PT[b] : NULL;
      OP_OUT[i][0] = &PT[out];
      op->input_tensors = OP_IN[i];
      op->input_count = b >= 0 ? 3 : 2;
      op->output_tensors = OP_OUT[i];
      op->output_count = 1;
      op->conv.weight_tensor = &PT[w];
      op->conv.bias_tensor = b >= 0 ? &PT[b] : NULL;
      op->conv.stride_x = op->conv.stride_y = 1;
      op->conv.dilation_width_factor = op->conv.dilation_height_factor = 1;
      op->conv.pointwise = T[w].dims[1] == 1 && T[w].dims[2] == 1;
      op->conv.depthwise = false;
      op->conv.relu = false;
      op->conv.activation_min = T[out].is_signed ? -128 : 0;
      op->conv.activation_max = T[out].is_signed ? 127 : 255;
   }
   rd(f, &input_bytes, 4);
   input = malloc(input_bytes);
   rd(f, input, input_bytes);
   for (unsigned i = 0; i < n_t; i++) {
      if (!T[i].is_output)
         continue;
      rd(f, &expected_bytes[i], 4);
      expected[i] = malloc(expected_bytes[i]);
      rd(f, expected[i], expected_bytes[i]);
   }
   fclose(f);

   for (unsigned i = 0; i < n_t; i++) {
      struct pipe_tensor *p = &PT[i];
      memset(p, 0, sizeof(*p));
      p->data = T[i].data; /* constants: pointed to, not copied (the engine passes memory-mapped weights) */
      p->index = i;
      memcpy(p->dims, T[i].dims, sizeof(p->dims));
      p->rank = 4;
      p->scale = T[i].scale;
      p->zero_point = T[i].zp;
      p->is_signed = T[i].is_signed;
      p->is_constant = T[i].is_constant;
      p->type_size = T[i].type_size;
   }
}

static bool
is_etnaviv(int fd)
{
   drmVersionPtr v = drmGetVersion(fd);
   bool etna = v && !strcmp(v->name, "etnaviv");
   drmFreeVersion(v);
   return etna;
}

static int
open_etnaviv(bool offline)
{
   const bool shim = dlsym(RTLD_DEFAULT, "npu_etna_shim_active") != NULL;
   if (offline != shim) {
      fprintf(stderr, offline ? "--offline needs LD_PRELOAD=.../shim/libnpu_etna_shim.so; nothing opened\n"
                              : "the NPU shim is loaded: use --offline (or unset LD_PRELOAD); nothing opened\n");
      return -1;
   }
   if (offline) {
      int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
      if (fd >= 0 && is_etnaviv(fd)) {
         printf("offline: fake NPU (shim), submits are recorded, not executed\n");
         return fd;
      }
      if (fd >= 0)
         close(fd);
      return -1;
   }
   for (int minor = 128; minor < 192; minor++) {
      char path[64];
      snprintf(path, sizeof(path), "/dev/dri/renderD%d", minor);
      int fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd < 0)
         continue;
      if (is_etnaviv(fd)) {
         printf("%s: etnaviv\n", path);
         return fd;
      }
      close(fd);
   }
   return -1;
}

static int
cmp_d(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

static double
median(double *v, unsigned n)
{
   if (!n)
      return 0;
   qsort(v, n, sizeof(*v), cmp_d);
   return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ---- CPU share of a split step: int8 GEMV on 4 A73 threads ---- */
#define CPU_THREADS 4
static int8_t *cw, *cx;
static int32_t *cy;
static unsigned c_rows, c_ic, c_copies; /* copies of the matrix, used in turn: >= 64 MB in all, so no step finds its
                                          weights in the A73 L2 (1 MB), as in a real token where every layer is new */
static atomic_uint c_gen, c_done;
static volatile bool c_quit;

static void
gemv4(const int8_t *x, const int8_t *w, unsigned ic, int32_t *y)
{
   int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
   for (unsigned k = 0; k < ic; k += 16) {
      int8x16_t xv = vld1q_s8(x + k);
      int8x16_t w0 = vld1q_s8(w + k), w1 = vld1q_s8(w + ic + k), w2 = vld1q_s8(w + 2 * ic + k),
                w3 = vld1q_s8(w + 3 * ic + k);
      /* |x*w| <= 128*127, two of them per int16 lane: no overflow */
      a0 = vpadalq_s16(a0, vmlal_high_s8(vmull_s8(vget_low_s8(xv), vget_low_s8(w0)), xv, w0));
      a1 = vpadalq_s16(a1, vmlal_high_s8(vmull_s8(vget_low_s8(xv), vget_low_s8(w1)), xv, w1));
      a2 = vpadalq_s16(a2, vmlal_high_s8(vmull_s8(vget_low_s8(xv), vget_low_s8(w2)), xv, w2));
      a3 = vpadalq_s16(a3, vmlal_high_s8(vmull_s8(vget_low_s8(xv), vget_low_s8(w3)), xv, w3));
   }
   y[0] = vaddvq_s32(a0);
   y[1] = vaddvq_s32(a1);
   y[2] = vaddvq_s32(a2);
   y[3] = vaddvq_s32(a3);
}

static void *
cpu_worker(void *arg)
{
   unsigned id = (unsigned)(uintptr_t)arg, seen = 0;
   cpu_set_t set;
   CPU_ZERO(&set);
   CPU_SET(2 + id, &set);
   pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
   unsigned per = (c_rows / 4 + CPU_THREADS - 1) / CPU_THREADS * 4;
   unsigned r0 = id * per, r1 = r0 + per > c_rows ? c_rows : r0 + per;
   for (;;) {
      unsigned g;
      while ((g = atomic_load_explicit(&c_gen, memory_order_acquire)) == seen && !c_quit)
         __asm__ volatile("yield");
      if (c_quit)
         return NULL;
      seen = g;
      const int8_t *wb = cw + (size_t)(g % c_copies) * c_rows * c_ic;
      for (unsigned r = r0; r < r1; r += 4)
         gemv4(cx, wb + (size_t)r * c_ic, c_ic, cy + r);
      atomic_fetch_add_explicit(&c_done, 1, memory_order_release);
   }
}

static pthread_t c_thr[CPU_THREADS];

static bool
cpu_setup(unsigned rows, unsigned ic)
{
   if (!rows)
      return true;
   if (rows % 4 || ic % 16 || !ic) {
      fprintf(stderr, "--cpu-rows must be a multiple of 4 and --cpu-ic of 16\n");
      return false;
   }
   c_rows = rows;
   c_ic = ic;
   c_copies = (unsigned)((64u << 20) / ((size_t)rows * ic)) + 1;
   cw = aligned_alloc(64, ((size_t)rows * ic * c_copies + 63) / 64 * 64);
   cx = aligned_alloc(64, (ic + 63) / 64 * 64);
   cy = calloc(rows, sizeof(*cy));
   uint32_t v = 12345;
   for (size_t i = 0; i < (size_t)rows * ic; i++)
      cw[i] = (int8_t)((v = v * 1103515245u + 12345u) >> 24);
   for (unsigned c = 1; c < c_copies; c++)
      memcpy(cw + (size_t)c * rows * ic, cw, (size_t)rows * ic);
   for (unsigned i = 0; i < ic; i++)
      cx[i] = (int8_t)((v = v * 1103515245u + 12345u) >> 24);
   for (uintptr_t t = 0; t < CPU_THREADS; t++)
      pthread_create(&c_thr[t], NULL, cpu_worker, (void *)t);
   return true;
}

static void
cpu_start(void)
{
   atomic_store_explicit(&c_done, 0, memory_order_relaxed);
   atomic_fetch_add_explicit(&c_gen, 1, memory_order_release);
}

static void
cpu_join(void)
{
   while (atomic_load_explicit(&c_done, memory_order_acquire) < CPU_THREADS)
      __asm__ volatile("yield");
}

static bool
cpu_check(void)
{
   for (unsigned r = 0; r < c_rows; r += c_rows / 7 + 1) {
      int32_t ref = 0;
      for (unsigned k = 0; k < c_ic; k++)
         ref += cx[k] * cw[(size_t)r * c_ic + k];
      if (ref != cy[r]) {
         fprintf(stderr, "cpu gemv row %u: %d != %d\n", r, cy[r], ref);
         return false;
      }
   }
   return true;
}

static void
cpu_stop(void)
{
   if (!c_rows)
      return;
   c_quit = true;
   for (unsigned t = 0; t < CPU_THREADS; t++)
      pthread_join(c_thr[t], NULL);
}

int
main(int argc, char **argv)
{
   bool offline = false, per_invoke = false, cpu_only = false;
   unsigned n = 20, batch = 1, cpu_rows = 0, cpu_ic = 0;
   enum { WAIT_MESA, WAIT_BLOCK, WAIT_SPIN } wait = WAIT_MESA;
   const char *pack = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--offline"))
         offline = true;
      else if (!strcmp(argv[i], "--cpu-only"))
         cpu_only = true;
      else if (!strcmp(argv[i], "--cpu-rows") && i + 1 < argc)
         cpu_rows = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--cpu-ic") && i + 1 < argc)
         cpu_ic = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--wait") && i + 1 < argc) {
         i++;
         wait = !strcmp(argv[i], "spin") ? WAIT_SPIN : !strcmp(argv[i], "block") ? WAIT_BLOCK : WAIT_MESA;
      }
      else if (!strcmp(argv[i], "--ctx") && i + 1 < argc)
         per_invoke = !strcmp(argv[++i], "per-invoke");
      else if (!strcmp(argv[i], "-n") && i + 1 < argc)
         n = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--batch") && i + 1 < argc)
         batch = atoi(argv[++i]);
      else
         pack = argv[i];
   }
   if (cpu_only) {
      if (!cpu_rows || !cpu_ic || n < 1 || n > 100000 || !cpu_setup(cpu_rows, cpu_ic))
         return 2;
      double *tc = calloc(n, sizeof(double));
      for (unsigned it = 0; it < n; it++) {
         double a = now_ms();
         cpu_start();
         cpu_join();
         tc[it] = now_ms() - a;
      }
      bool ok = cpu_check();
      unsigned m = n > 1 ? n - 1 : 1;
      double med = median(tc + (n > 1), m);
      printf("{\"cpu_only\": true, \"cpu_rows\": %u, \"cpu_ic\": %u, \"invokes\": %u, \"cpu_ms\": %.4f, "
             "\"cpu_gbs\": %.3f, \"cpu_check\": %s}\n", cpu_rows, cpu_ic, n, med,
             (double)cpu_rows * cpu_ic / 1e6 / med, ok ? "true" : "false");
      cpu_stop();
      return ok ? 0 : 4;
   }
   if (!pack || n < 1 || n > 100000 || batch < 1 || batch > 64 || (cpu_rows && wait == WAIT_MESA)) {
      fprintf(stderr, "usage: jb_run [--offline] [--ctx persist|per-invoke] [-n N] [--batch K] [--wait mesa|block|spin] "
                      "[--cpu-rows R [--cpu-ic IC]] PACK.jbp   (--cpu-rows needs --wait block|spin)\n");
      return 2;
   }
   load_pack(pack);

   /* graph inputs = non-constant tensors no op writes; outputs = flagged tensors */
   unsigned in_idx[MAX_T], out_idx[MAX_T], n_in = 0, n_out = 0;
   bool written[MAX_T] = {0};
   for (unsigned i = 0; i < n_ops; i++)
      written[OP_OUT[i][0]->index] = true;
   for (unsigned i = 0; i < n_t; i++) {
      if (!T[i].is_constant && !written[i])
         in_idx[n_in++] = i;
      if (T[i].is_output)
         out_idx[n_out++] = i;
   }
   if (n_in != 1) {
      fprintf(stderr, "expected one graph input, found %u\n", n_in);
      return 2;
   }
   if (!cpu_setup(cpu_rows, cpu_ic ? cpu_ic : T[in_idx[0]].dims[1] * T[in_idx[0]].dims[2] * T[in_idx[0]].dims[3]))
      return 2;

   int fd = open_etnaviv(offline);
   if (fd < 0) {
      fprintf(stderr, "no etnaviv node\n");
      return 2;
   }
   double t0 = now_ms();
   struct pipe_screen *ps = etna_drm_screen_create(fd);
   struct pipe_ml_device *dev = ps && ps->get_ml_device ? ps->get_ml_device(ps) : NULL;
   if (!dev) {
      fprintf(stderr, "no ML device\n");
      return 2;
   }
   double t1 = now_ms();
   for (unsigned i = 0; i < n_t; i++)
      PT[i].is_external_output = T[i].is_output;
   struct pipe_ml_subgraph *sg = dev->ml_subgraph_create(dev, OPS, n_ops);
   double t2 = now_ms();
   if (!sg) {
      fprintf(stderr, "ml_subgraph_create failed\n");
      return 3;
   }
   for (unsigned i = 0; i < n_t; i++) /* the weights now live in the driver's buffers only */
      if (T[i].is_constant) {
         free(T[i].data);
         T[i].data = PT[i].data = NULL;
      }
   printf("screen %.2f ms, compile %.2f ms (%u ops, %u outputs)\n", t1 - t0, t2 - t1, n_ops, n_out);

   void *outs[MAX_T];
   bool out_signed[MAX_T], in_signed = T[in_idx[0]].is_signed;
   for (unsigned i = 0; i < n_out; i++) {
      outs[i] = calloc(1, expected_bytes[out_idx[i]]);
      out_signed[i] = T[out_idx[i]].is_signed;
   }
   double *t_ctx = calloc(n, sizeof(double)), *t_inv = calloc(n, sizeof(double)),
          *t_read = calloc(n, sizeof(double)), *t_all = calloc(n, sizeof(double)),
          *t_npu = calloc(n, sizeof(double)), *t_cpu = calloc(n, sizeof(double));
   unsigned spins = 0;
   struct pipe_context *pc = per_invoke ? NULL : ps->context_create(ps, NULL, PIPE_CONTEXT_COMPUTE_ONLY);
   int maxdiff = 0;
   for (unsigned it = 0; it < n; it++) {
      double a = now_ms();
      if (per_invoke)
         pc = ps->context_create(ps, NULL, PIPE_CONTEXT_COMPUTE_ONLY);
      if (!pc) {
         fprintf(stderr, "context_create failed\n");
         return 3;
      }
      double b = now_ms();
      void *in_buf = input;
      for (unsigned k = 0; k < batch; k++)
         pc->ml_subgraph_invoke(pc, sg, 1, in_idx, &in_buf, &in_signed);
      double c = now_ms(), cpu_end = c, npu_end = c;
      if (wait != WAIT_MESA) {
         struct pipe_fence_handle *fence = NULL;
         pc->flush(pc, &fence, 0);
         double f0 = now_ms();
         if (cpu_rows)
            cpu_start();
         bool done = false;
         if (wait == WAIT_SPIN) {
            while (!(done = ps->fence_finish(ps, NULL, fence, 0)) && now_ms() - f0 < 3000)
               spins++;
         } else {
            done = ps->fence_finish(ps, NULL, fence, 3000000000ull);
         }
         npu_end = now_ms();
         if (cpu_rows) {
            cpu_join();
            cpu_end = now_ms();
            t_cpu[it] = cpu_end - f0;
         }
         ps->fence_reference(ps, &fence, NULL);
         if (!done) {
            fprintf(stderr, "fence wait timed out (3 s)\n");
            return 3;
         }
         t_npu[it] = npu_end - c;
      }
      pc->ml_subgraph_read_output(pc, sg, n_out, out_idx, outs, out_signed);
      double d = now_ms();
      if (per_invoke) {
         pc->destroy(pc);
         pc = NULL;
      }
      double e = now_ms();
      t_ctx[it] = (b - a) + (e - d);
      t_inv[it] = c - b;
      t_read[it] = d - c;
      t_all[it] = e - a;
      if (it == 0 || it == n - 1)
         for (unsigned i = 0; i < n_out; i++)
            for (unsigned k = 0; k < expected_bytes[out_idx[i]]; k++) {
               int x = ((int8_t *)outs[i])[k], y = ((int8_t *)expected[out_idx[i]])[k];
               if (!out_signed[i])
                  x = ((uint8_t *)outs[i])[k], y = ((uint8_t *)expected[out_idx[i]])[k];
               if (abs(x - y) > maxdiff)
                  maxdiff = abs(x - y);
            }
   }
   unsigned m = n > 1 ? n - 1 : 1;
   double first = t_all[0];
   bool cpu_ok = !cpu_rows || cpu_check();
   double cpu_med = cpu_rows ? median(t_cpu + (n > 1), m) : 0;
   printf("{\"pack\": \"%s\", \"offline\": %s, \"ctx\": \"%s\", \"wait\": \"%s\", \"invokes\": %u, "
          "\"batch\": %u, \"ops\": %u, \"compile_ms\": %.3f, \"first_ms\": %.3f, \"ms_median_after_first\": %.4f, "
          "\"ctx_ms\": %.4f, \"invoke_call_ms\": %.4f, \"read_ms\": %.4f, \"npu_ms\": %.4f, \"spins_per_wait\": %.0f, "
          "\"cpu_rows\": %u, \"cpu_ms\": %.4f, \"cpu_gbs\": %.3f, \"cpu_check\": %s, \"diff_vs_expected_max\": %d}\n",
          pack, offline ? "true" : "false", per_invoke ? "per-invoke" : "persist",
          wait == WAIT_SPIN ? "spin" : wait == WAIT_BLOCK ? "block" : "mesa", n, batch, n_ops, t2 - t1, first,
          median(t_all + (n > 1), m), median(t_ctx + (n > 1), m), median(t_inv + (n > 1), m),
          median(t_read + (n > 1), m), median(t_npu + (n > 1), m), (double)spins / n, cpu_rows, cpu_med,
          cpu_rows ? (double)c_rows * c_ic / 1e6 / cpu_med : 0.0, cpu_ok ? "true" : "false", maxdiff);
   cpu_stop();
   if (pc)
      pc->destroy(pc);
   dev->ml_subgraph_destroy(dev, sg);
   ps->destroy(ps);
   close(fd);
   return 0;
}
