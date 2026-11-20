/*
 * jb_run.c (2026-11-19, LLM engine L3b) - NN jobs through the Gallium ML API directly: no TFLite, no Teflon delegate.
 *
 * Reads a job pack (llm/l3/jb_pack.py: tensors + 1x1-conv operations of one subgraph = one NPU submit, a test input
 * and the expected outputs), compiles it with the driver's own ml_subgraph_create (Mesa's NN config / weight
 * compiler, so the jobs are the ones Teflon would build), invokes it N times and reports per-phase times and the
 * largest output difference.
 *
 *   jb_run [--offline] [--ctx persist|per-invoke] [-n N] [--batch K] PACK.jbp
 *
 * --offline     the fake NPU (LD_PRELOAD=.../npu-shader/shim/libnpu_etna_shim.so, run inside bwrap with an empty /dev);
 *               refused without the shim, and a device run is refused with it (same rule as cs_run).
 * --ctx         persist = one pipe_context for every invoke (the engine); per-invoke = create + destroy a context
 *               around every invoke, as Teflon's partition_invoke does.
 * --batch K     invoke the subgraph K times before one read (K copies of its jobs in one submit).
 *
 * Device runs ONLY through scripts/npu-run with Jordan's OK for that step.
 */
#include <dlfcn.h>
#include <fcntl.h>
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

int
main(int argc, char **argv)
{
   bool offline = false, per_invoke = false;
   unsigned n = 20, batch = 1;
   const char *pack = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--offline"))
         offline = true;
      else if (!strcmp(argv[i], "--ctx") && i + 1 < argc)
         per_invoke = !strcmp(argv[++i], "per-invoke");
      else if (!strcmp(argv[i], "-n") && i + 1 < argc)
         n = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--batch") && i + 1 < argc)
         batch = atoi(argv[++i]);
      else
         pack = argv[i];
   }
   if (!pack || n < 1 || n > 100000 || batch < 1 || batch > 64) {
      fprintf(stderr, "usage: jb_run [--offline] [--ctx persist|per-invoke] [-n N] [--batch K] PACK.jbp\n");
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
          *t_read = calloc(n, sizeof(double)), *t_all = calloc(n, sizeof(double));
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
      double c = now_ms();
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
   printf("{\"pack\": \"%s\", \"offline\": %s, \"ctx\": \"%s\", \"invokes\": %u, \"batch\": %u, \"ops\": %u, "
          "\"compile_ms\": %.3f, \"first_ms\": %.3f, \"ms_median_after_first\": %.4f, \"ctx_ms\": %.4f, "
          "\"invoke_call_ms\": %.4f, \"read_ms\": %.4f, \"diff_vs_expected_max\": %d}\n",
          pack, offline ? "true" : "false", per_invoke ? "per-invoke" : "persist", n, batch, n_ops, t2 - t1, first,
          median(t_all + (n > 1), m), median(t_ctx + (n > 1), m), median(t_inv + (n > 1), m),
          median(t_read + (n > 1), m), maxdiff);
   if (pc)
      pc->destroy(pc);
   dev->ml_subgraph_destroy(dev, sg);
   ps->destroy(ps);
   close(fd);
   return 0;
}
