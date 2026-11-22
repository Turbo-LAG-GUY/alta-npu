/*
 * jb_engine.c (2026-11-21, LLM engine R2 step 1) - every layer step of a model compiled in ONE process and run back to
 * back in one context: the NPU side of the engine, without the CPU work between the steps yet.
 *
 * Reads the packs of llm/l3/real_packs.py (PACK_DIR/L<layer>_<step>_t<tokens>.jbp, steps qkv o gateup down), compiles
 * each with the driver's ml_subgraph_create (frees its weight copy right after: the NPU buffers are the only copy that
 * stays), then runs N rounds; one round = every step once in model order, each with its own recorded input, polled
 * fence wait, Gallium read-out. Checks every output against the pack's expected levels in the first and last round.
 * Reports compile time, peak memory, the time per round (= all layer steps of one chunk) and per step kind.
 *
 *   jb_engine [--offline] [-n ROUNDS] [--tokens T] [--layers L] PACK_DIR
 *
 * --offline: the fake NPU (LD_PRELOAD=.../npu-shader/shim/libnpu_etna_shim.so, inside bwrap with an empty /dev).
 * Device runs ONLY through scripts/npu-run with Jordan's OK for that step.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <dlfcn.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include <xf86drm.h>

#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "etnaviv/drm/etnaviv_drm_public.h"

#define MAX_T 64
#define MAX_OPS 32
#define MAX_PACKS 512
static const char *STEPS[] = {"qkv", "o", "gateup", "down"};

struct jb_tensor {
   int32_t index;
   uint32_t dims[4];
   float scale;
   int32_t zp;
   uint8_t is_signed, is_constant, type_size, is_output;
   uint32_t n_bytes;
   uint8_t *data;
};

struct pack {
   char name[64];
   int step;
   unsigned n_t, n_ops, n_out, in_idx, out_idx[MAX_T];
   struct jb_tensor T[MAX_T];
   struct pipe_tensor PT[MAX_T];
   struct pipe_ml_operation OPS[MAX_OPS];
   struct pipe_tensor *op_in[MAX_OPS][3], *op_out[MAX_OPS][1];
   uint8_t *input, *expected[MAX_T];
   uint32_t input_bytes, expected_bytes[MAX_T];
   bool in_signed, out_signed[MAX_T];
   void *outs[MAX_T];
   struct pipe_ml_subgraph *sg;
   double compile_ms, *t_all, *t_npu, *t_ro;
   uint64_t weight_bytes;
   int maxdiff;
};

static double
now_ms(void)
{
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static int
cmp_d(const void *a, const void *b)
{
   double x = *(const double *)a, y = *(const double *)b;
   return x < y ? -1 : x > y;
}

static double
median(const double *v, unsigned n)
{
   if (!n)
      return 0;
   double *c = malloc(n * sizeof(*c));
   memcpy(c, v, n * sizeof(*c));
   qsort(c, n, sizeof(*c), cmp_d);
   double m = n & 1 ? c[n / 2] : 0.5 * (c[n / 2 - 1] + c[n / 2]);
   free(c);
   return m;
}

static bool
rd(FILE *f, void *p, size_t n)
{
   return fread(p, 1, n, f) == n;
}

/* JBP1 (llm/l3/jb_pack.py): tensors, ops (1 = 1x1 conv, 2 = TRANSPOSE), the input, the expected outputs */
static bool
load_pack(struct pack *k, const char *path)
{
   FILE *f = fopen(path, "rb");
   char magic[4];
   uint32_t hdr[2];
   if (!f || !rd(f, magic, 4) || !rd(f, hdr, 8) || memcmp(magic, "JBP1", 4) || hdr[0] > MAX_T || hdr[1] > MAX_OPS) {
      fprintf(stderr, "%s: not a JBP1 pack (or too big)\n", path);
      if (f)
         fclose(f);
      return false;
   }
   k->n_t = hdr[0];
   k->n_ops = hdr[1];
   for (unsigned i = 0; i < k->n_t; i++) {
      struct jb_tensor *t = &k->T[i];
      if (!rd(f, &t->index, 4) || !rd(f, t->dims, 16) || !rd(f, &t->scale, 4) || !rd(f, &t->zp, 4) ||
          !rd(f, &t->is_signed, 4) || !rd(f, &t->n_bytes, 4) || (unsigned)t->index != i)
         goto bad;
      t->data = t->n_bytes ? malloc(t->n_bytes) : NULL;
      if (t->n_bytes && !rd(f, t->data, t->n_bytes))
         goto bad;
      if (t->is_constant)
         k->weight_bytes += t->n_bytes;
   }
   for (unsigned i = 0; i < k->n_t; i++) {
      struct pipe_tensor *p = &k->PT[i];
      memset(p, 0, sizeof(*p));
      p->data = k->T[i].data;
      p->index = i;
      memcpy(p->dims, k->T[i].dims, sizeof(p->dims));
      p->rank = 4;
      p->scale = k->T[i].scale;
      p->zero_point = k->T[i].zp;
      p->is_signed = k->T[i].is_signed;
      p->is_constant = k->T[i].is_constant;
      p->type_size = k->T[i].type_size;
      p->is_external_output = k->T[i].is_output;
   }
   for (unsigned i = 0; i < k->n_ops; i++) {
      uint32_t type;
      int32_t in, w, b, out;
      if (!rd(f, &type, 4) || !rd(f, &in, 4) || !rd(f, &w, 4) || !rd(f, &b, 4) || !rd(f, &out, 4) || in < 0 ||
          out < 0 || (unsigned)in >= k->n_t || (unsigned)out >= k->n_t)
         goto bad;
      struct pipe_ml_operation *op = &k->OPS[i];
      memset(op, 0, sizeof(*op));
      k->op_in[i][0] = &k->PT[in];
      k->op_out[i][0] = &k->PT[out];
      op->input_tensors = k->op_in[i];
      op->output_tensors = k->op_out[i];
      op->output_count = 1;
      if (type == 2) {
         op->type = PIPE_ML_OPERATION_TYPE_TRANSPOSE;
         op->input_count = 1;
         for (unsigned j = 0; j < 4; j++)
            op->transpose.perm[j] = ((uint32_t)w >> (8 * j)) & 0xff;
         continue;
      }
      if (type != 1 || w < 0 || (unsigned)w >= k->n_t || (b >= 0 && (unsigned)b >= k->n_t))
         goto bad;
      op->type = PIPE_ML_OPERATION_TYPE_CONVOLUTION;
      k->op_in[i][1] = &k->PT[w];
      k->op_in[i][2] = b >= 0 ? &k->PT[b] : NULL;
      op->input_count = b >= 0 ? 3 : 2;
      op->conv.weight_tensor = &k->PT[w];
      op->conv.bias_tensor = b >= 0 ? &k->PT[b] : NULL;
      op->conv.stride_x = op->conv.stride_y = 1;
      op->conv.dilation_width_factor = op->conv.dilation_height_factor = 1;
      op->conv.pointwise = k->T[w].dims[1] == 1 && k->T[w].dims[2] == 1;
      op->conv.activation_min = k->T[out].is_signed ? -128 : 0;
      op->conv.activation_max = k->T[out].is_signed ? 127 : 255;
   }
   if (!rd(f, &k->input_bytes, 4))
      goto bad;
   k->input = malloc(k->input_bytes);
   if (!rd(f, k->input, k->input_bytes))
      goto bad;
   bool written[MAX_T] = {0};
   unsigned n_in = 0;
   for (unsigned i = 0; i < k->n_ops; i++)
      written[k->op_out[i][0]->index] = true;
   for (unsigned i = 0; i < k->n_t; i++) {
      if (!k->T[i].is_constant && !written[i]) {
         k->in_idx = i;
         n_in++;
      }
      if (!k->T[i].is_output)
         continue;
      if (!rd(f, &k->expected_bytes[i], 4))
         goto bad;
      k->expected[i] = malloc(k->expected_bytes[i]);
      if (!rd(f, k->expected[i], k->expected_bytes[i]))
         goto bad;
      k->out_signed[k->n_out] = k->T[i].is_signed;
      k->outs[k->n_out] = calloc(1, k->expected_bytes[i]);
      k->out_idx[k->n_out++] = i;
   }
   fclose(f);
   k->in_signed = k->T[k->in_idx].is_signed;
   if (n_in != 1) {
      fprintf(stderr, "%s: expected one graph input, found %u\n", path, n_in);
      return false;
   }
   return true;
bad:
   fprintf(stderr, "%s: bad or short pack\n", path);
   fclose(f);
   return false;
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
   for (int minor = 128; minor < 192; minor++) {
      char path[64];
      snprintf(path, sizeof(path), "/dev/dri/renderD%d", minor);
      int fd = open(path, O_RDWR | O_CLOEXEC);
      if (fd < 0)
         continue;
      if (is_etnaviv(fd)) {
         printf("%s: etnaviv%s\n", path, offline ? " (fake NPU, submits are recorded, not executed)" : "");
         return fd;
      }
      close(fd);
   }
   return -1;
}

/* resident memory of this process in MB, by kind (anonymous / file / shared memory: the GEM buffers it maps) */
static void
rss(double *anon, double *file, double *shmem)
{
   FILE *f = fopen("/proc/self/status", "r");
   char line[256];
   *anon = *file = *shmem = 0;
   while (f && fgets(line, sizeof(line), f)) {
      long kb;
      if (sscanf(line, "RssAnon: %ld", &kb) == 1)
         *anon = kb / 1024.0;
      else if (sscanf(line, "RssFile: %ld", &kb) == 1)
         *file = kb / 1024.0;
      else if (sscanf(line, "RssShmem: %ld", &kb) == 1)
         *shmem = kb / 1024.0;
   }
   if (f)
      fclose(f);
}

int
main(int argc, char **argv)
{
   bool offline = false;
   unsigned rounds = 10, tokens = 256, max_layers = 1000;
   const char *dir = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--offline"))
         offline = true;
      else if (!strcmp(argv[i], "-n") && i + 1 < argc)
         rounds = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--tokens") && i + 1 < argc)
         tokens = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--layers") && i + 1 < argc)
         max_layers = atoi(argv[++i]);
      else
         dir = argv[i];
   }
   if (!dir || rounds < 1 || rounds > 10000) {
      fprintf(stderr, "usage: jb_engine [--offline] [-n ROUNDS] [--tokens T] [--layers L] PACK_DIR\n");
      return 2;
   }
   static struct pack packs[MAX_PACKS];
   unsigned np = 0, layers = 0;
   for (unsigned l = 0; l < max_layers; l++) {
      char path[512];
      snprintf(path, sizeof(path), "%.400s/L%02u_%s_t%u.jbp", dir, l, STEPS[0], tokens);
      if (access(path, R_OK))
         break;
      layers++;
      for (unsigned s = 0; s < 4 && np < MAX_PACKS; s++) {
         struct pack *k = &packs[np];
         snprintf(k->name, sizeof(k->name), "L%02u_%s_t%u", l, STEPS[s], tokens);
         snprintf(path, sizeof(path), "%.400s/%.60s.jbp", dir, k->name);
         k->step = s;
         np++;
      }
   }
   if (!np) {
      fprintf(stderr, "no packs L00_qkv_t%u.jbp ... in %s\n", tokens, dir);
      return 2;
   }

   int fd = open_etnaviv(offline);
   if (fd < 0) {
      fprintf(stderr, "no etnaviv node\n");
      return 2;
   }
   struct pipe_screen *ps = etna_drm_screen_create(fd);
   struct pipe_ml_device *dev = ps && ps->get_ml_device ? ps->get_ml_device(ps) : NULL;
   if (!dev) {
      fprintf(stderr, "no ML device\n");
      return 2;
   }

   /* compile every step, one after the other; each step's weight copy is freed right after its compile */
   double c0 = now_ms(), anon, file, shmem;
   uint64_t weights = 0, test_bytes = 0;
   double *t_compile = calloc(np, sizeof(double));
   for (unsigned p = 0; p < np; p++) {
      struct pack *k = &packs[p];
      char path[512];
      snprintf(path, sizeof(path), "%.400s/%.60s.jbp", dir, k->name);
      if (!load_pack(k, path))
         return 2;
      double a = now_ms();
      k->sg = dev->ml_subgraph_create(dev, k->OPS, k->n_ops);
      k->compile_ms = t_compile[p] = now_ms() - a;
      if (!k->sg) {
         fprintf(stderr, "%s: ml_subgraph_create failed\n", k->name);
         return 3;
      }
      for (unsigned i = 0; i < k->n_t; i++)
         if (k->T[i].is_constant) {
            free(k->T[i].data);
            k->T[i].data = k->PT[i].data = NULL;
         }
      weights += k->weight_bytes;
      test_bytes += k->input_bytes;
      for (unsigned o = 0; o < k->n_out; o++)
         test_bytes += 2 * k->expected_bytes[k->out_idx[o]];
      k->t_all = calloc(rounds, sizeof(double));
      k->t_npu = calloc(rounds, sizeof(double));
      k->t_ro = calloc(rounds, sizeof(double));
   }
   double compile_total = now_ms() - c0, cmax = 0;
   for (unsigned p = 0; p < np; p++)
      if (t_compile[p] > cmax)
         cmax = t_compile[p];
   rss(&anon, &file, &shmem);
   struct rusage ru;
   getrusage(RUSAGE_SELF, &ru);
   printf("compiled %u steps (%u layers, %u tokens) in %.0f ms (median %.1f, max %.1f per step); weights %.1f MB "
          "(freed after compile); resident now anon %.1f + file %.1f + shmem %.1f MB, peak %.1f MB; test data %.1f MB\n",
          np, layers, tokens, compile_total, median(t_compile, np), cmax, weights / 1e6, anon, file, shmem,
          ru.ru_maxrss / 1024.0, test_bytes / 1e6);
   fflush(stdout);

   struct pipe_context *pc = ps->context_create(ps, NULL, PIPE_CONTEXT_COMPUTE_ONLY);
   if (!pc) {
      fprintf(stderr, "context_create failed\n");
      return 3;
   }
   double *t_round = calloc(rounds, sizeof(double));
   int maxdiff = 0;
   unsigned bad_packs = 0;
   for (unsigned r = 0; r < rounds; r++) {
      double r0 = now_ms();
      for (unsigned p = 0; p < np; p++) {
         struct pack *k = &packs[p];
         double a = now_ms();
         void *in_buf = k->input;
         pc->ml_subgraph_invoke(pc, k->sg, 1, &k->in_idx, &in_buf, &k->in_signed);
         struct pipe_fence_handle *fence = NULL;
         pc->flush(pc, &fence, 0);
         double f0 = now_ms();
         bool done;
         while (!(done = ps->fence_finish(ps, NULL, fence, 0)) && now_ms() - f0 < 3000)
            ;
         ps->fence_reference(ps, &fence, NULL);
         if (!done) {
            fprintf(stderr, "%s: fence wait timed out (3 s)\n", k->name);
            return 3;
         }
         double b = now_ms();
         pc->ml_subgraph_read_output(pc, k->sg, k->n_out, k->out_idx, k->outs, k->out_signed);
         double c = now_ms();
         k->t_all[r] = c - a;
         k->t_npu[r] = b - a;
         k->t_ro[r] = c - b;
      }
      t_round[r] = now_ms() - r0;
      if (r != 0 && r != rounds - 1)
         continue;
      for (unsigned p = 0; p < np; p++) { /* the check, outside the timed round */
         struct pack *k = &packs[p];
         int worst = 0;
         for (unsigned o = 0; o < k->n_out; o++) {
            unsigned ti = k->out_idx[o];
            for (unsigned j = 0; j < k->expected_bytes[ti]; j++) {
               int x = ((int8_t *)k->outs[o])[j], y = ((int8_t *)k->expected[ti])[j];
               if (!k->out_signed[o])
                  x = ((uint8_t *)k->outs[o])[j], y = k->expected[ti][j];
               if (abs(x - y) > worst)
                  worst = abs(x - y);
            }
         }
         if (worst > k->maxdiff)
            k->maxdiff = worst;
      }
   }
   double step_ms[4] = {0}, step_npu[4] = {0}, step_ro[4] = {0};
   unsigned m = rounds > 1 ? rounds - 1 : 1, skip = rounds > 1;
   for (unsigned p = 0; p < np; p++) {
      struct pack *k = &packs[p];
      step_ms[k->step] += median(k->t_all + skip, m);
      step_npu[k->step] += median(k->t_npu + skip, m);
      step_ro[k->step] += median(k->t_ro + skip, m);
      if (k->maxdiff > maxdiff)
         maxdiff = k->maxdiff;
      if (k->maxdiff > 1) {
         bad_packs++;
         fprintf(stderr, "%s: max diff %d\n", k->name, k->maxdiff);
      }
   }
   double round_med = median(t_round + skip, m);
   rss(&anon, &file, &shmem);
   getrusage(RUSAGE_SELF, &ru);
   for (unsigned s = 0; s < 4; s++)
      printf("  %-7s %u steps: %.2f ms per round (NPU incl. submit %.2f, read-out %.2f)\n", STEPS[s], layers,
             step_ms[s], step_npu[s], step_ro[s]);
   printf("{\"offline\": %s, \"steps\": %u, \"layers\": %u, \"tokens\": %u, \"rounds\": %u, \"compile_ms\": %.1f, "
          "\"compile_ms_max\": %.2f, \"weights_mb\": %.2f, \"peak_rss_mb\": %.1f, \"rss_anon_mb\": %.1f, "
          "\"rss_file_mb\": %.1f, \"rss_shmem_mb\": %.1f, \"test_data_mb\": %.1f, \"round_ms_median\": %.3f, "
          "\"ms_per_token\": %.4f, \"qkv_ms\": %.3f, \"o_ms\": %.3f, \"gateup_ms\": %.3f, \"down_ms\": %.3f, "
          "\"readout_ms\": %.3f, \"bad_packs\": %u, \"diff_vs_expected_max\": %d}\n",
          offline ? "true" : "false", np, layers, tokens, rounds, compile_total, cmax, weights / 1e6,
          ru.ru_maxrss / 1024.0, anon, file, shmem, test_bytes / 1e6, round_med, round_med / tokens, step_ms[0],
          step_ms[1], step_ms[2], step_ms[3], step_ro[0] + step_ro[1] + step_ro[2] + step_ro[3], bad_packs, maxdiff);
   pc->destroy(pc);
   for (unsigned p = 0; p < np; p++)
      dev->ml_subgraph_destroy(dev, packs[p].sg);
   ps->destroy(ps);
   close(fd);
   return bad_packs ? 4 : 0;
}
