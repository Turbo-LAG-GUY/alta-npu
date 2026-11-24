/*
 * jb_prefill.c (2026-11-23, LLM engine): one prompt chunk end to end in C, no Python between the steps.
 * Tokens 1..T through every layer: RMSNorm -> planar int8 -> qkv step -> q / k / v + RoPE (position 0's K / V from the
 * .jbp0 file: position 0 only attends to itself, so it is computed once, on the CPU, elsewhere) -> causal attention ->
 * o step -> residual -> RMSNorm -> gateup step -> SiLU * up, block FWHT, down's planar int8 -> down step -> residual;
 * then the final norm and the tied head. The CPU glue is llm/l3/glue/jb_glue.c (compiled in); the steps go to
 *   --npu  the NPU through the packs (jb_engine's compile / invoke / read-out code, included below), or
 *   --ref  a CPU int8 reference of every job (the packs' own arithmetic: exact, for checking without the NPU).
 * Inputs: PACK_DIR (real_packs.py packs, 256 tokens), AUX (glue/export_aux.py model.jba) and PROMPT (<text>.jbp0: the
 * 257 token ids and position 0's K / V per layer). Output: logits of tokens 1..T (--all) or of the last one.
 * Usage: jb_prefill [--npu|--ref] [--offline] [--all] [--repeat N] [--dump DIR] [--out LOGITS] PACK_DIR AUX PROMPT
 *   --repeat N: the chunk N times in one process (steady-state timing; every repeat's logits must equal the first's)
 * Build: llm/l3/build_jb_run.sh jb_prefill (adds -fopenmp for the glue)
 */
#include <sys/stat.h>
#define main jb_engine_main
#include "jb_engine.c"
#undef main
#include "glue/jb_glue.c"

/* ---- JBA1 records (glue/export_aux.py) ---- */
struct rec {
   char name[48];
   uint32_t dtype, ndim, dims[4];
   uint64_t nbytes;
   void *data;
};

struct jba {
   unsigned n;
   struct rec *r;
};

/* the file is mapped, not copied: the records point into the mapping (page cache the kernel can drop and re-read:
 * the 1.7B's 200 MB of bf16 embeddings stay out of the process's anonymous memory, next to 1.7 GB of NPU buffers) */
static bool
read_jba(const char *path, struct jba *a)
{
   int fd = open(path, O_RDONLY | O_CLOEXEC);
   struct stat st;
   if (fd < 0 || fstat(fd, &st)) {
      fprintf(stderr, "%s: cannot open\n", path);
      return false;
   }
   uint8_t *m = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
   close(fd);
   if (m != MAP_FAILED)
      madvise(m, st.st_size, MADV_WILLNEED);      /* read it in now, while the packs load and compile (2026-11-24:
                                                   * the 1.7B head paged its 201 MB of embeddings in from the SD
                                                   * card: 1.3-5.8 s) */
   if (m == MAP_FAILED || st.st_size < 4 || memcmp(m, "JBA1", 4)) {
      fprintf(stderr, "%s: not a JBA1 file\n", path);
      return false;
   }
   unsigned cap = 64;
   a->n = 0;
   a->r = malloc(cap * sizeof(*a->r));
   size_t o = 4;
   while (o + 80 <= (size_t)st.st_size) {
      struct rec r;
      memcpy(r.name, m + o, 48);
      memcpy(&r.dtype, m + o + 48, 4);
      memcpy(&r.ndim, m + o + 52, 4);
      memcpy(r.dims, m + o + 56, 16);
      memcpy(&r.nbytes, m + o + 72, 8);
      o += 80;
      if (o + r.nbytes > (size_t)st.st_size) {
         fprintf(stderr, "%s: short record %.48s\n", path, r.name);
         return false;
      }
      r.data = m + o;                             /* 8-byte aligned? records are copied where alignment matters */
      o += r.nbytes;
      if (a->n == cap)
         a->r = realloc(a->r, (cap *= 2) * sizeof(*a->r));
      a->r[a->n++] = r;
   }
   return true;
}

static struct rec *
find(struct jba *a, const char *name)
{
   for (unsigned i = 0; i < a->n; i++)
      if (!strncmp(a->r[i].name, name, 48))
         return &a->r[i];
   return NULL;
}

static void *
need(struct jba *a, const char *name, uint32_t dtype)
{
   struct rec *r = find(a, name);
   if (!r || r->dtype != dtype) {
      fprintf(stderr, "aux: record %s missing or wrong type\n", name);
      exit(2);
   }
   if (((uintptr_t)r->data & 15) && r->nbytes < (64u << 20)) {   /* unaligned small record: an aligned copy, once */
      void *c = aligned_alloc(16, (r->nbytes + 15) & ~(size_t)15);
      memcpy(c, r->data, r->nbytes);
      r->data = c;
   }
   return r->data;
}

/* ---- one step: its pack, its jobs' row maps, scales ---- */
struct step {
   struct pack *k;
   int nj, ic;
   float sx;
   int zx;
   const int32_t *rows[64];
   int n[64], col[64], wic[64];   /* job rows, its first input channel (split pieces), its input width */
   float sw[64], sy[64];
   int zy[64];
   const int8_t *w[64];           /* the job's int8 weights (ref backend) */
   const int8_t *lv[64];          /* the job's outputs: planar [n, T] */
   long ts[64], js[64];
   double t_npu, t_ro;
};

static void
step_setup(struct step *S, struct pack *k, struct jba *aux, int layer, int T)
{
   memset(S, 0, sizeof(*S));
   S->k = k;
   struct jb_tensor *in = &k->T[k->in_idx];
   S->ic = in->dims[1];                           /* planar input [1, C, H, W] */
   S->sx = in->scale;
   S->zx = in->zp;
   int split_first = -1, split_n = 0;
   for (unsigned i = 0; i < k->n_ops; i++)
      if (k->OPS[i].type == PIPE_ML_OPERATION_TYPE_SPLIT) {
         split_first = k->OPS[i].output_tensors[0]->index;
         split_n = k->OPS[i].output_count;
      }
   int piece_c = split_n ? S->ic / split_n : S->ic;
   for (unsigned i = 0; i < k->n_ops; i++) {
      struct pipe_ml_operation *op = &k->OPS[i];
      if (op->type != PIPE_ML_OPERATION_TYPE_CONVOLUTION)
         continue;
      int j = S->nj++;
      int ii = op->input_tensors[0]->index, wi = op->input_tensors[1]->index, oi = op->output_tensors[0]->index;
      S->col[j] = split_n && ii >= split_first && ii < split_first + split_n ? (ii - split_first) * piece_c : 0;
      S->n[j] = k->T[wi].dims[0];
      S->wic[j] = k->T[wi].dims[3];                /* weight tensor [n, 1, 1, ic] */
      S->sw[j] = k->T[wi].scale;
      S->w[j] = (const int8_t *)k->T[wi].data;
      S->sy[j] = k->T[oi].scale;
      S->zy[j] = k->T[oi].zp;
      char name[64];
      snprintf(name, sizeof(name), "rows.%d.%s.%d", layer, STEPS[k->step], j);
      S->rows[j] = need(aux, name, 2);
      S->lv[j] = k->outs[j];                      /* outputs in job order (tensor order = job order) */
      S->ts[j] = 1;
      S->js[j] = T;
   }
   (void)piece_c;
}

/* CPU int8 reference of one step: every job's planar output [n, T] from the planar input [ic, T], the packs'
 * arithmetic exactly (int32 accumulation, round(acc * sx * sw / sy) + zy in double, clip) */
static void
ref_step(struct step *S, const int8_t *in, int T)
{
   int16_t *x16 = malloc((size_t)S->ic * T * sizeof(int16_t));
   for (size_t i = 0; i < (size_t)S->ic * T; i++)
      x16[i] = (int16_t)(in[i] - S->zx);
   for (int j = 0; j < S->nj; j++) {
      int n = S->n[j], ic = S->wic[j];
      double scale = (double)S->sx * (double)S->sw[j] / (double)S->sy[j];
      int8_t *out = (int8_t *)S->lv[j];
#pragma omp parallel for schedule(dynamic)
      for (int r = 0; r < n; r++) {
         int32_t acc[1024];
         for (int t = 0; t < T; t++)
            acc[t] = 0;
         const int8_t *wr = S->w[j] + (size_t)r * ic;
         for (int c = 0; c < ic; c++) {
            int32_t wv = wr[c];
            const int16_t *xr = x16 + (size_t)(S->col[j] + c) * T;
            for (int t = 0; t < T; t++)
               acc[t] += wv * xr[t];
         }
         for (int t = 0; t < T; t++) {
            double v = rint((double)acc[t] * scale) + S->zy[j];
            out[(size_t)r * T + t] = (int8_t)(v < -128 ? -128 : v > 127 ? 127 : v);
         }
      }
   }
   free(x16);
}

struct ctx {
   bool ref;
   struct pipe_screen *ps;
   struct pipe_context *pc;
   int T;
};

static int
run_step(struct ctx *c, struct step *S, int8_t *in)
{
   double a = now_ms();
   if (c->ref) {
      ref_step(S, in, c->T);
      S->t_npu += now_ms() - a;
      return 0;
   }
   struct pack *k = S->k;
   void *in_buf = in;
   c->pc->ml_subgraph_invoke(c->pc, k->sg, 1, &k->in_idx, &in_buf, &k->in_signed);
   struct pipe_fence_handle *fence = NULL;
   c->pc->flush(c->pc, &fence, 0);
   double f0 = now_ms();
   bool done;
   while (!(done = c->ps->fence_finish(c->ps, NULL, fence, 0)) && now_ms() - f0 < 3000)
      ;
   c->ps->fence_reference(c->ps, &fence, NULL);
   if (!done) {
      fprintf(stderr, "%s: fence wait timed out (3 s)\n", k->name);
      return 3;
   }
   double b = now_ms();
   c->pc->ml_subgraph_read_output(c->pc, k->sg, k->n_out, k->out_idx, k->outs, k->out_signed);
   S->t_npu += b - a;
   S->t_ro += now_ms() - b;
   return 0;
}

static void
write_f32(const char *path, const float *a, size_t n)
{
   FILE *f = fopen(path, "wb");
   if (!f || fwrite(a, sizeof(float), n, f) != n)
      fprintf(stderr, "%s: write failed\n", path);
   if (f)
      fclose(f);
}

/* ---- one prompt chunk: tokens 1..T through every layer, then the final norm + tied head (2026-11-24: a function, so
 * one process can run several chunks: --repeat, and later the serving / decode loop) ---- */
struct model {
   int L, D, H, KV, hd, I, V, qpow2, bdown, T;
   float eps;
   const uint16_t *embed;
   const float *Q, *Hs, *normw;
   const double *inv_freq;
   const int32_t *rope;
   struct step *steps;
};

struct bufs {
   float *x, *a0, *q, *kk, *vv, *att, *xf, *lg;   /* xf [nl, D], lg [nl, V] */
   int8_t *in;
};

/* tokens ids[1..T] (position 0's K / V from the prompt file pr); logits of the last nl tokens into b->lg; tp[8] +=
 * embed, norm+quant, qkv+rope, attention, quantize, scatter, ffn, head (ms); returns 0 or a step's error */
static int
chunk(struct ctx *c, const struct model *m, struct bufs *b, struct jba *pr, const int32_t *ids, int nl,
      const char *dump, double tp[8])
{
   const int L = m->L, D = m->D, H = m->H, KV = m->KV, hd = m->hd, I = m->I, V = m->V, T = m->T;
   const int qpow2 = m->qpow2, bdown = m->bdown;
   const float eps = m->eps, *Q = m->Q, *Hs = m->Hs, *normw = m->normw;
   const uint16_t *embed = m->embed;
   const double *inv_freq = m->inv_freq;
   const int32_t *rope = m->rope;
   struct step *steps = m->steps;
   float *x = b->x, *a0 = b->a0, *q = b->q, *kk = b->kk, *vv = b->vv, *att = b->att;
   int8_t *in = b->in;
   double t0 = now_ms(), t;

   /* embeddings of tokens 1..T (bf16 -> fp32), the residual rotation Q */
   for (int i = 0; i < T; i++) {
      float *r = x + (size_t)(1 + i) * D;
      const uint16_t *e = embed + (size_t)ids[1 + i] * D;
      for (int d = 0; d < D; d++) {
         uint32_t u = (uint32_t)e[d] << 16;
         memcpy(&r[d], &u, 4);
      }
   }
   if (qpow2) {
      jbg_fwht(x + D, T, D, D, Q, 0);
   } else {
#pragma omp parallel
      {
         float *tmp = malloc((size_t)D * sizeof(float));
#pragma omp for schedule(dynamic)
         for (int i = 0; i < T; i++) {            /* x @ Q (dense): y[j] = sum_i x[i] Q[i][j] */
            float *r = x + (size_t)(1 + i) * D;
            for (int j = 0; j < D; j++)
               tmp[j] = 0;
            for (int ii = 0; ii < D; ii++) {
               float xi = r[ii];
               const float *qr = Q + (size_t)ii * D;
               for (int j = 0; j < D; j++)
                  tmp[j] += xi * qr[j];
            }
            memcpy(r, tmp, D * sizeof(float));
         }
         free(tmp);
      }
   }
   tp[0] += now_ms() - t0;

   for (int l = 0; l < L; l++) {
      struct step *Sq = &steps[4 * l], *So = &steps[4 * l + 1], *Sg = &steps[4 * l + 2], *Sd = &steps[4 * l + 3];
      char nm[32];
      t = now_ms();
      jbg_norm_quant(x, T, D, eps, Sq->sx, Sq->zx, in, a0);
      tp[1] += now_ms() - t;
      if (run_step(c, Sq, in))
         return 3;
      t = now_ms();
      jbg_qkv_rope(Sq->nj, Sq->lv, Sq->ts, Sq->js, Sq->n, Sq->rows, Sq->sy, Sq->zy, T, H, KV, hd, 0, inv_freq,
                   rope[l], q, kk, vv);
      snprintf(nm, sizeof(nm), "k0.%d", l);
      memcpy(kk, need(pr, nm, 0), (size_t)KV * hd * sizeof(float));
      snprintf(nm, sizeof(nm), "v0.%d", l);
      memcpy(vv, need(pr, nm, 0), (size_t)KV * hd * sizeof(float));
      tp[2] += now_ms() - t;
      t = now_ms();
      jbg_attention(q + (size_t)H * hd, kk, vv, att, T, T + 1, 1, H, KV, hd);   /* queries 1..T over keys 0..T */
      tp[3] += now_ms() - t;
      t = now_ms();
      jbg_quant_planar(att, T, H * hd, So->sx, So->zx, in);
      tp[4] += now_ms() - t;
      if (run_step(c, So, in))
         return 3;
      t = now_ms();
      jbg_scatter_jobs(So->nj, So->lv, So->ts, So->js, So->n, So->rows, So->sy, So->zy, x + D, T, D, 1);
      tp[5] += now_ms() - t;
      t = now_ms();
      jbg_norm_quant(x, T, D, eps, Sg->sx, Sg->zx, in, a0);
      tp[1] += now_ms() - t;
      if (run_step(c, Sg, in))
         return 3;
      t = now_ms();
      jbg_ffn_fused(Sg->nj, Sg->lv, Sg->ts, Sg->js, Sg->n, Sg->rows, Sg->sy, Sg->zy, T, I, Hs, bdown, Sd->sx, Sd->zx, in);
      tp[6] += now_ms() - t;
      if (run_step(c, Sd, in))
         return 3;
      t = now_ms();
      jbg_scatter_jobs(Sd->nj, Sd->lv, Sd->ts, Sd->js, Sd->n, Sd->rows, Sd->sy, Sd->zy, x + D, T, D, 1);
      tp[5] += now_ms() - t;
      if (dump) {
         char path[512];
         snprintf(path, sizeof(path), "%.400s/x.%d.f32", dump, l);
         write_f32(path, x, (size_t)(T + 1) * D);
      }
   }

   /* final: Q back, RMSNorm with the weight, the tied head (bf16 embeddings) for the last token or all of them */
   t = now_ms();
   float *xf = b->xf;
   memcpy(xf, x + (size_t)(T + 1 - nl) * D, (size_t)nl * D * sizeof(float));
   if (qpow2) {
      jbg_fwht(xf, nl, D, D, Q, 1);
   } else {
      float *tmp = malloc((size_t)D * sizeof(float));
      for (int i = 0; i < nl; i++) {              /* x @ Q.T: y[j] = sum_i x[i] Q[j][i] */
         float *r = xf + (size_t)i * D;
         for (int j = 0; j < D; j++)
            tmp[j] = dot_f32(r, Q + (size_t)j * D, D);
         memcpy(r, tmp, D * sizeof(float));
      }
      free(tmp);
   }
   jbg_rmsnorm(xf, normw, xf, nl, D, eps);
   float *lg = b->lg;
   jbg_head_bf16(xf, nl, D, embed, V, lg);       /* bf16 rows widened in registers (2026-11-24: was a scalar
                                                   * widening loop, 0.9 GB/s); bit-identical to dot_f32 */
   tp[7] += now_ms() - t;
   return 0;
}

int
main(int argc, char **argv)
{
   bool ref = true, offline = false, all = false;
   int repeat = 1;
   const char *pd = NULL, *auxp = NULL, *promptp = NULL, *outp = NULL, *dump = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--npu"))
         ref = false;
      else if (!strcmp(argv[i], "--ref"))
         ref = true;
      else if (!strcmp(argv[i], "--offline"))
         offline = true;
      else if (!strcmp(argv[i], "--all"))
         all = true;
      else if (!strcmp(argv[i], "--repeat") && i + 1 < argc)
         repeat = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 1;
      else if (!strcmp(argv[i], "--out") && i + 1 < argc)
         outp = argv[++i];
      else if (!strcmp(argv[i], "--dump") && i + 1 < argc)
         dump = argv[++i];
      else if (!pd)
         pd = argv[i];
      else if (!auxp)
         auxp = argv[i];
      else
         promptp = argv[i];
   }
   if (!pd || !auxp || !promptp) {
      fprintf(stderr, "usage: jb_prefill [--npu|--ref] [--offline] [--all] [--repeat N] [--dump DIR] [--out LOGITS] "
                      "PACK_DIR AUX PROMPT\n");
      return 2;
   }
   double t_start = now_ms();
   struct jba aux, pr;
   if (!read_jba(auxp, &aux) || !read_jba(promptp, &pr))
      return 2;
   const int32_t *cfg = need(&aux, "cfg", 2);
   int L = cfg[0], D = cfg[1], H = cfg[2], KV = cfg[3], hd = cfg[4], I = cfg[5], V = cfg[6], qpow2 = cfg[7];
   int bdown = cfg[8], T = cfg[9];
   float eps = *(float *)need(&aux, "eps", 0);
   const uint16_t *embed = need(&aux, "embed", 3);
   const float *Q = need(&aux, "Q", 0), *Hs = need(&aux, "Hs", 0), *normw = need(&aux, "norm", 0);
   const double *inv_freq = need(&aux, "inv_freq", 1);
   const int32_t *rope = need(&aux, "rope", 2);
   struct rec *idsr = find(&pr, "ids");
   if (!idsr || (int)idsr->dims[0] != T + 1) {
      fprintf(stderr, "prompt: need %d token ids\n", T + 1);
      return 2;
   }
   const int32_t *ids = idsr->data;
   printf("jb_prefill: %d layers, D %d, H %d / KV %d x %d, I %d, vocab %d, %d tokens after position 0, backend %s\n",
          L, D, H, KV, hd, I, V, T, ref ? "ref (CPU int8)" : offline ? "npu (fake)" : "npu");

   /* packs: load every step; NPU: compile and free the weights; ref: keep the weights */
   static struct pack packs[MAX_PACKS];
   static struct step steps[MAX_PACKS];
   struct ctx c = {.ref = ref, .T = T};
   struct pipe_ml_device *dev = NULL;
   if (!ref) {
      int fd = open_etnaviv(offline);
      c.ps = fd < 0 ? NULL : etna_drm_screen_create(fd);
      dev = c.ps && c.ps->get_ml_device ? c.ps->get_ml_device(c.ps) : NULL;
      if (!dev) {
         fprintf(stderr, "no NPU ML device\n");
         return 2;
      }
   }
   double tc = now_ms();
   for (int l = 0; l < L; l++) {
      if (l % 2 == 0) {                          /* progress, flushed: a swap stall shows up as a stopped counter
                                                  * (incidents/2026-11-23-1537-LLMP2-prefill2-swap-stall) */
         long avail = -1;
         FILE *mf = fopen("/proc/meminfo", "r");
         char ln[128];
         while (mf && fgets(ln, sizeof(ln), mf))
            if (sscanf(ln, "MemAvailable: %ld", &avail) == 1)
               break;
         if (mf)
            fclose(mf);
         printf("loading layer %d / %d (%.0f ms, MemAvailable %ld MB)\n", l, L, now_ms() - tc, avail / 1024);
         fflush(stdout);
      }
      for (int s = 0; s < 4; s++) {
         struct pack *k = &packs[l * 4 + s];
         char path[512];
         snprintf(k->name, sizeof(k->name), "L%02d_%s_t%d", l, STEPS[s], T);
         snprintf(path, sizeof(path), "%.400s/%.60s.jbp", pd, k->name);
         k->step = s;
         if (!load_pack(k, path))
            return 2;
         if (l > 0) {                             /* one set of read-back buffers per step kind, not per layer: the
                                                   * steps run one after another and each output is consumed before
                                                   * the next layer's same step (1.7B: -184 MiB; incident
                                                   * 2026-11-23-1537-LLMP2-prefill2-swap-stall) */
            struct pack *k0 = &packs[s];
            bool same = k->n_out == k0->n_out;
            for (unsigned o = 0; same && o < k->n_out; o++)
               same = k->out_idx[o] == k0->out_idx[o] &&
                      k->expected_bytes[k->out_idx[o]] == k0->expected_bytes[k0->out_idx[o]];
            for (unsigned o = 0; same && o < k->n_out; o++) {
               free(k->outs[o]);
               k->outs[o] = k0->outs[o];
            }
         }
         if (!ref) {
            k->sg = dev->ml_subgraph_create(dev, k->OPS, k->n_ops);
            if (!k->sg) {
               fprintf(stderr, "%s: ml_subgraph_create failed\n", k->name);
               return 3;
            }
         }
         step_setup(&steps[l * 4 + s], k, &aux, l, T);
         free(k->input);                          /* the pack's test input / expected outputs: not needed here */
         k->input = NULL;
         for (unsigned o = 0; o < k->n_out; o++) {
            free(k->expected[k->out_idx[o]]);
            k->expected[k->out_idx[o]] = NULL;
         }
         if (!ref)
            for (unsigned i = 0; i < k->n_t; i++)
               if (k->T[i].is_constant) {
                  free(k->T[i].data);
                  k->T[i].data = k->PT[i].data = NULL;
               }
      }
   }
   printf("loaded%s %d steps in %.0f ms\n", ref ? "" : " + compiled", 4 * L, now_ms() - tc);
   fflush(stdout);
   if (!ref) {
      c.pc = c.ps->context_create(c.ps, NULL, PIPE_CONTEXT_COMPUTE_ONLY);
      if (!c.pc) {
         fprintf(stderr, "context_create failed\n");
         return 3;
      }
   }

   /* buffers: x [T+1, D] (row 0 unused: position 0 lives in the .jbp0 file), q / k / v [T+1, heads, hd] */
   size_t imax = (size_t)(I > D ? I : D) * T;
   int nl = all ? T : 1;
   struct bufs b = {
      .x = calloc((size_t)(T + 1) * D, sizeof(float)),
      .a0 = malloc((size_t)(I > D ? I : D) * sizeof(float)),
      .q = malloc((size_t)(T + 1) * H * hd * sizeof(float)),
      .kk = malloc((size_t)(T + 1) * KV * hd * sizeof(float)),
      .vv = malloc((size_t)(T + 1) * KV * hd * sizeof(float)),
      .att = malloc((size_t)T * H * hd * sizeof(float)),
      .xf = malloc((size_t)nl * D * sizeof(float)),
      .lg = malloc((size_t)nl * V * sizeof(float)),
      .in = malloc(imax),
   };
   struct model mdl = {
      .L = L, .D = D, .H = H, .KV = KV, .hd = hd, .I = I, .V = V, .qpow2 = qpow2, .bdown = bdown, .T = T, .eps = eps,
      .embed = embed, .Q = Q, .Hs = Hs, .normw = normw, .inv_freq = inv_freq, .rope = rope, .steps = steps,
   };
   /* --repeat N: the same chunk N times in this process (steady state: the first pays page-ins of the mapped
    * embeddings and first touches of the NPU buffers); every repeat's logits must equal the first's */
   float *lg0 = repeat > 1 ? malloc((size_t)nl * V * sizeof(float)) : NULL;
   for (int rep = 0; rep < repeat; rep++) {
      double tp[8] = {0};
      for (int p = 0; p < 4 * L; p++)
         steps[p].t_npu = steps[p].t_ro = 0;
      double t0 = now_ms();
      int rc = chunk(&c, &mdl, &b, &pr, ids, nl, rep == 0 ? dump : NULL, tp);
      if (rc)
         return rc;
      double wall = now_ms() - t0;
      double npu = 0, ro = 0;
      for (int p = 0; p < 4 * L; p++) {
         npu += steps[p].t_npu;
         ro += steps[p].t_ro;
      }
      const float *lg = b.lg;
      int best = 0;
      for (int v = 1; v < V; v++)
         if (lg[(size_t)(nl - 1) * V + v] > lg[(size_t)(nl - 1) * V + best])
            best = v;
      char same[64] = "", tag[24] = "";
      if (repeat > 1)
         snprintf(tag, sizeof(tag), " %d/%d", rep + 1, repeat);
      if (lg0 && rep == 0)
         memcpy(lg0, lg, (size_t)nl * V * sizeof(float));
      else if (lg0)
         snprintf(same, sizeof(same), "; logits %s repeat 0",
                  memcmp(lg0, lg, (size_t)nl * V * sizeof(float)) ? "DIFFER FROM" : "identical to");
      printf("chunk%s: %.1f ms = steps %.1f (%s; read-out %.1f) + glue: embed %.1f, norm+quant %.1f, qkv+rope %.1f, "
             "attention %.1f, quantize %.1f, scatter %.1f, ffn %.1f, head %.1f ms; last token's argmax %d%s\n",
             tag, wall, npu, ref ? "CPU reference" : "NPU incl. submit", ro, tp[0], tp[1],
             tp[2], tp[3], tp[4], tp[5], tp[6], tp[7], best, same);
      printf("{\"rep\": %d, \"wall_ms\": %.1f, \"steps_ms\": %.1f, \"readout_ms\": %.1f, \"glue_ms\": %.1f, "
             "\"head_ms\": %.1f, \"argmax\": %d, \"same_as_rep0\": %s, \"total_ms\": %.1f}\n", rep, wall, npu, ro,
             wall - npu, tp[7], best, !lg0 || rep == 0 || !memcmp(lg0, lg, (size_t)nl * V * sizeof(float)) ? "true" :
             "false", now_ms() - t_start);
      fflush(stdout);
      if (lg0 && rep > 0 && memcmp(lg0, lg, (size_t)nl * V * sizeof(float)))
         return 4;
   }
   if (outp)
      write_f32(outp, b.lg, (size_t)nl * V);
   (void)jb_engine_main;
   return 0;
}
