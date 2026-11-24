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
 *   --generate N: then N tokens greedily, one at a time on the CPU (decode_step: int8 weights kept in RAM, float
 *     activations quantized to int16 per GEMV (JB_DECODE_Q16=0: float), KV cache filled by the chunk); --teacher LOGITS: a check of the decode path - positions 0..T of the
 *     prompt through decode_step one by one (its own cache), logits of 1..T written to LOGITS (compare with --all),
 *     position 0's K / V compared with the prompt file's
 *   JB_DECODE_COEF=1 (NPU backend): decode reads the int8 weights straight from the NPU's coefficient buffers (mesa-llm
 *     etna_ml_llm_coefs(); ETNA_ML_CACHED_COEFFS=1 makes them CPU-cached), so they are not kept a second time; the
 *     same bits as the kept-weights int16 decode. --coef-check: decode every job's buffer and compare it with the
 *     pack's weights, each pack right after its compile (no extra memory)
 * Build: llm/l3/build_jb_run.sh jb_prefill (adds -fopenmp for the glue)
 */
#include <sys/stat.h>
#define main jb_engine_main
#include "jb_engine.c"
#undef main
#include "glue/jb_glue.c"
#include "etnaviv_ml_llm.h"                        /* mesa-llm test tree: etna_ml_llm_coefs() (the NPU's weights) */

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
   struct jbg_coef cf[64];        /* the job's weights as the NPU holds them (NPU backend; decode, --coef-check) */
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
   float *kc, *vc;                                /* KV cache [L][cap][KV * hd] (decode; NULL: none) */
   int cap;
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
      if (b->kc) {                                /* keys / values 0..T of this layer into the decode cache */
         memcpy(b->kc + (size_t)l * b->cap * KV * hd, kk, (size_t)(T + 1) * KV * hd * sizeof(float));
         memcpy(b->vc + (size_t)l * b->cap * KV * hd, vv, (size_t)(T + 1) * KV * hd * sizeof(float));
      }
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

/* ---- decode (2026-11-24): one token on the CPU, the float path of export_aux.py's position 0 (int8 weights x the
 * job's weight scale, float activations: no 8-bit activations), with the KV cache; the weights must be kept ---- */
struct dbufs {
   float *x, *a, *y, *q, *att, *h, *xf;
};

static struct dbufs
dbufs_new(const struct model *m)
{
   int D = m->D, I = m->I, oc = (m->H + 2 * m->KV) * m->hd;
   int big = 2 * I > oc ? 2 * I : oc;
   struct dbufs d = {
      .x = malloc(D * sizeof(float)), .a = malloc((I > D ? I : D) * sizeof(float)),
      .y = malloc(big * sizeof(float)), .q = malloc(m->H * m->hd * sizeof(float)),
      .att = malloc(m->H * m->hd * sizeof(float)), .h = malloc(I * sizeof(float)), .xf = malloc(D * sizeof(float)),
   };
   return d;
}

static double gemv_ms;                             /* decode: time in the GEMVs */
static int16_t *gemv_xq;                           /* decode, JB_DECODE_Q16=1: int16 activations (jbg_gemv_jobs_q16) */
static int gemv_coef;                              /* decode, JB_DECODE_COEF=1: from the NPU's coefficient buffers */

static void
gemv_step(const struct step *S, const float *x, float *y, int oc, int accumulate)
{
   double t = now_ms();
   if (gemv_coef)
      jbg_gemv_coef_q16(S->nj, x, S->cf, S->col, S->sw, S->rows, y, oc, accumulate, gemv_xq);
   else if (gemv_xq)
      jbg_gemv_jobs_q16(S->nj, x, S->w, S->n, S->wic, S->col, S->sw, S->rows, y, oc, accumulate, gemv_xq);
   else
      jbg_gemv_jobs(S->nj, x, S->w, S->n, S->wic, S->col, S->sw, S->rows, y, oc, accumulate);
   gemv_ms += now_ms() - t;
}

/* token id at position pos: appends its K / V to the cache (rows pos), logits into lg [V] (NULL: skip the head) */
static void
decode_step(const struct model *m, struct bufs *b, struct dbufs *d, int pos, int id, float *lg, double tp[4])
{
   const int L = m->L, D = m->D, H = m->H, KV = m->KV, hd = m->hd, I = m->I, V = m->V;
   const struct step *steps = m->steps;
   size_t kvw = (size_t)KV * hd;
   double t = now_ms(), t1;
   const uint16_t *e = m->embed + (size_t)id * D;
   for (int i = 0; i < D; i++) {
      uint32_t u = (uint32_t)e[i] << 16;
      memcpy(&d->x[i], &u, 4);
   }
   if (m->qpow2) {
      jbg_fwht(d->x, 1, D, D, m->Q, 0);
   } else {
      for (int j = 0; j < D; j++)
         d->a[j] = 0;
      for (int ii = 0; ii < D; ii++)
         for (int j = 0; j < D; j++)
            d->a[j] += d->x[ii] * m->Q[(size_t)ii * D + j];
      memcpy(d->x, d->a, D * sizeof(float));
   }
   for (int l = 0; l < L; l++) {
      const struct step *Sq = &steps[4 * l], *So = &steps[4 * l + 1], *Sg = &steps[4 * l + 2], *Sd = &steps[4 * l + 3];
      float *kc = b->kc + (size_t)l * b->cap * kvw, *vc = b->vc + (size_t)l * b->cap * kvw;
      jbg_rmsnorm(d->x, NULL, d->a, 1, D, m->eps);
      gemv_step(Sq, d->a, d->y, (H + 2 * KV) * hd, 0);
      memcpy(d->q, d->y, (size_t)H * hd * sizeof(float));
      memcpy(kc + pos * kvw, d->y + (size_t)H * hd, kvw * sizeof(float));
      memcpy(vc + pos * kvw, d->y + (size_t)(H + KV) * hd, kvw * sizeof(float));
      if (m->rope[l]) {
         jbg_rope(d->q, 1, H, hd, pos, m->inv_freq);
         jbg_rope(kc + pos * kvw, 1, KV, hd, pos, m->inv_freq);
      }
      t1 = now_ms(); tp[0] += t1 - t; t = t1;
      jbg_attn_decode(d->q, kc, vc, d->att, pos + 1, H, KV, hd);
      t1 = now_ms(); tp[1] += t1 - t; t = t1;
      gemv_step(So, d->att, d->x, D, 1);
      jbg_rmsnorm(d->x, NULL, d->a, 1, D, m->eps);
      gemv_step(Sg, d->a, d->y, 2 * I, 0);
      jbg_silu_mul(d->y, 1, I, d->h);
      jbg_fwht(d->h, 1, I, m->bdown, m->Hs, 0);
      gemv_step(Sd, d->h, d->x, D, 1);
      t1 = now_ms(); tp[0] += t1 - t; t = t1;
   }
   if (!lg)
      return;
   memcpy(d->xf, d->x, D * sizeof(float));
   if (m->qpow2) {
      jbg_fwht(d->xf, 1, D, D, m->Q, 1);
   } else {
      for (int j = 0; j < D; j++)
         d->a[j] = dot_f32(d->xf, m->Q + (size_t)j * D, D);
      memcpy(d->xf, d->a, D * sizeof(float));
   }
   jbg_rmsnorm(d->xf, m->normw, d->xf, 1, D, m->eps);
   jbg_head_bf16(d->xf, 1, D, m->embed, V, lg);
   tp[2] += now_ms() - t;
}

static int
argmax_f32(const float *a, int n)
{
   int best = 0;
   for (int i = 1; i < n; i++)
      if (a[i] > a[best])
         best = i;
   return best;
}

int
main(int argc, char **argv)
{
   bool ref = true, offline = false, all = false;
   int repeat = 1, generate = 0, coef_check = 0;
   const char *teacher = NULL;
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
      else if (!strcmp(argv[i], "--generate") && i + 1 < argc)
         generate = atoi(argv[++i]) > 0 ? atoi(argv[i]) : 0;
      else if (!strcmp(argv[i], "--coef-check"))
         coef_check = 1;
      else if (!strcmp(argv[i], "--teacher") && i + 1 < argc)
         teacher = argv[++i];
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
      fprintf(stderr, "usage: jb_prefill [--npu|--ref] [--offline] [--all] [--repeat N] [--generate N] [--teacher LOGITS] "
                      "[--dump DIR] [--out LOGITS] PACK_DIR AUX PROMPT\n");
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
   int decode_coef = !ref && getenv("JB_DECODE_COEF") && atoi(getenv("JB_DECODE_COEF")) > 0;
   long ck_jobs = 0, ck_bad = 0, ck_bytes = 0;     /* --coef-check, per pack right after its compile */
   double ck_ms = 0;
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
         if (!ref) {                              /* each job's coefficient buffer (job j = the j-th NN job) */
            struct step *S = &steps[l * 4 + s];
            for (int j = 0; j < S->nj; j++) {
               struct etna_ml_llm_coefs e;
               if (etna_ml_llm_coefs(k->sg, j, &e) || !e.pointwise || e.zrl_bits || (int)e.output_channels != S->n[j] ||
                   (int)e.input_channels != S->wic[j]) {
                  if (decode_coef || coef_check) {
                     fprintf(stderr, "%s job %d: no usable coefficient buffer\n", k->name, j);
                     return 3;
                  }
                  continue;
               }
               S->cf[j] = (struct jbg_coef){e.map, (int)e.cores, (int)e.header_size, (int)e.kernels_per_core,
                                            (int)e.superblocks, (int)e.input_channels, (int)e.output_channels,
                                            (int)e.stride, (int)e.weight_zero_point};
               int kis0 = ((int)e.kernels_per_core + (int)e.superblocks - 1) / (int)e.superblocks;
               if (kis0 > 64) {
                  fprintf(stderr, "%s job %d: %d kernels per superblock (> 64)\n", k->name, j, kis0);
                  return 3;
               }
               if (coef_check) {                  /* decoded buffer = the pack's weights (still here)? */
                  double tq = now_ms();
                  size_t nb = (size_t)S->n[j] * S->wic[j];
                  int8_t *w = malloc(nb);
                  jbg_coef_decode(&S->cf[j], w);
                  int diff = memcmp(w, S->w[j], nb) != 0;
                  if (diff && ck_bad < 5)
                     printf("coef-check: %s job %d DIFFERS (%d x %d, stride %d, superblocks %d, cores %d)\n", k->name,
                            j, S->n[j], S->wic[j], S->cf[j].stride, S->cf[j].superblocks, S->cf[j].cores);
                  ck_bad += diff;
                  ck_jobs++;
                  ck_bytes += nb;
                  free(w);
                  ck_ms += now_ms() - tq;
               }
            }
         }
         free(k->input);                          /* the pack's test input / expected outputs: not needed here */
         k->input = NULL;
         for (unsigned o = 0; o < k->n_out; o++) {
            free(k->expected[k->out_idx[o]]);
            k->expected[k->out_idx[o]] = NULL;
         }
         if (!ref && ((!generate && !teacher) || decode_coef))   /* decode from kept weights: keep them */
            for (unsigned i = 0; i < k->n_t; i++)
               if (k->T[i].is_constant) {
                  free(k->T[i].data);
                  k->T[i].data = k->PT[i].data = NULL;
               }
      }
   }
   printf("loaded%s %d steps in %.0f ms\n", ref ? "" : " + compiled", 4 * L, now_ms() - tc);
   fflush(stdout);
   if (coef_check && !ref) {
      printf("coef-check: %ld jobs, %.1f MB of weights decoded from the NPU's buffers in %.0f ms: %s\n", ck_jobs,
             ck_bytes / 1e6, ck_ms, ck_bad ? "MISMATCH" : "all identical to the packs");
      fflush(stdout);
      if (ck_bad)
         return 5;
   }
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
   /* decode's GEMVs: int16 activations by default (135M: 56.9 -> 39.8 ms per token; teacher-forced logits vs float
    * activations KL 3e-8, top-1 100%, ppl 15.482 vs 15.481); JB_DECODE_Q16=0: float activations */
   if ((generate || teacher) && !(getenv("JB_DECODE_Q16") && !strcmp(getenv("JB_DECODE_Q16"), "0")))
      gemv_xq = malloc((size_t)(I > D ? I : D) * sizeof(int16_t));
   if ((generate || teacher) && decode_coef) {
      gemv_coef = 1;
      if (!gemv_xq)
         gemv_xq = malloc((size_t)(I > D ? I : D) * sizeof(int16_t));
   }
   if (generate || teacher) {
      b.cap = T + 1 + generate;
      b.kc = malloc((size_t)L * b.cap * KV * hd * sizeof(float));
      b.vc = malloc((size_t)L * b.cap * KV * hd * sizeof(float));
   }
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
   if (generate) {
      struct dbufs d = dbufs_new(&mdl);
      float *lgd = malloc((size_t)V * sizeof(float));
      int next = argmax_f32(b.lg + (size_t)(nl - 1) * V, V);
      double tp[4] = {0}, g0 = now_ms();
      gemv_ms = 0;
      printf("generated:");
      for (int i = 0; i < generate; i++) {
         printf(" %d", next);
         fflush(stdout);
         decode_step(&mdl, &b, &d, T + 1 + i, next, lgd, tp);
         next = argmax_f32(lgd, V);
      }
      double gms = now_ms() - g0;
      printf("\ndecode: %d tokens in %.1f ms = %.1f ms per token (%.2f tok/s): linear+glue %.1f (GEMVs %.1f), "
             "attention %.1f, head %.1f ms per token\n", generate, gms, gms / generate, 1e3 * generate / gms,
             tp[0] / generate, gemv_ms / generate, tp[1] / generate, tp[2] / generate);
      printf("{\"generate\": %d, \"decode_ms_per_token\": %.2f, \"next\": %d}\n", generate, gms / generate, next);
   }
   if (teacher) {
      struct dbufs d = dbufs_new(&mdl);
      float *lt = malloc((size_t)T * V * sizeof(float));
      double tp[4] = {0}, g0 = now_ms();
      decode_step(&mdl, &b, &d, 0, ids[0], NULL, tp);
      double ek = 0, ev = 0, mk = 0, mv = 0;
      const float *k0 = need(&pr, "k0.0", 0), *v0 = need(&pr, "v0.0", 0);
      for (int i = 0; i < KV * hd; i++) {
         ek = fmax(ek, fabs(b.kc[i] - k0[i]));
         ev = fmax(ev, fabs(b.vc[i] - v0[i]));
         mk = fmax(mk, fabs(k0[i]));
         mv = fmax(mv, fabs(v0[i]));
      }
      printf("teacher: position 0 K / V of layer 0 vs the prompt file: max |diff| %.3g / %.3g (rel %.2e / %.2e)\n",
             ek, ev, ek / mk, ev / mv);
      for (int p = 1; p <= T; p++)
         decode_step(&mdl, &b, &d, p, ids[p], lt + (size_t)(p - 1) * V, tp);
      double gms = now_ms() - g0;
      write_f32(teacher, lt, (size_t)T * V);
      printf("teacher: positions 0..%d decoded in %.1f ms (%.1f ms per token), logits of 1..%d -> %s\n", T, gms,
             gms / (T + 1), T, teacher);
   }
   (void)jb_engine_main;
   return 0;
}
