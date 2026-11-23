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
 * Usage: jb_prefill [--npu|--ref] [--offline] [--all] [--dump DIR] [--out LOGITS] PACK_DIR AUX PROMPT
 * Build: llm/l3/build_jb_run.sh jb_prefill (adds -fopenmp for the glue)
 */
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

static bool
read_jba(const char *path, struct jba *a)
{
   FILE *f = fopen(path, "rb");
   char magic[4];
   if (!f || fread(magic, 1, 4, f) != 4 || memcmp(magic, "JBA1", 4)) {
      fprintf(stderr, "%s: not a JBA1 file\n", path);
      return false;
   }
   unsigned cap = 64;
   a->n = 0;
   a->r = malloc(cap * sizeof(*a->r));
   for (;;) {
      struct rec r;
      if (fread(r.name, 1, 48, f) != 48)
         break;
      if (fread(&r.dtype, 4, 1, f) != 1 || fread(&r.ndim, 4, 1, f) != 1 || fread(r.dims, 4, 4, f) != 4 ||
          fread(&r.nbytes, 8, 1, f) != 1) {
         fprintf(stderr, "%s: short record header\n", path);
         return false;
      }
      r.data = malloc(r.nbytes ? r.nbytes : 1);
      if (fread(r.data, 1, r.nbytes, f) != r.nbytes) {
         fprintf(stderr, "%s: short record %s\n", path, r.name);
         return false;
      }
      if (a->n == cap)
         a->r = realloc(a->r, (cap *= 2) * sizeof(*a->r));
      a->r[a->n++] = r;
   }
   fclose(f);
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
#pragma omp parallel for schedule(static)
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

int
main(int argc, char **argv)
{
   bool ref = true, offline = false, all = false;
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
      fprintf(stderr, "usage: jb_prefill [--npu|--ref] [--offline] [--all] [--dump DIR] [--out LOGITS] PACK_DIR AUX "
                      "PROMPT\n");
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
   for (int l = 0; l < L; l++)
      for (int s = 0; s < 4; s++) {
         struct pack *k = &packs[l * 4 + s];
         char path[512];
         snprintf(k->name, sizeof(k->name), "L%02d_%s_t%d", l, STEPS[s], T);
         snprintf(path, sizeof(path), "%.400s/%.60s.jbp", pd, k->name);
         k->step = s;
         if (!load_pack(k, path))
            return 2;
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
   float *x = calloc((size_t)(T + 1) * D, sizeof(float));
   float *a0 = malloc((size_t)(I > D ? I : D) * sizeof(float));
   int8_t *in = malloc(imax);
   float *q = malloc((size_t)(T + 1) * H * hd * sizeof(float));
   float *kk = malloc((size_t)(T + 1) * KV * hd * sizeof(float));
   float *vv = malloc((size_t)(T + 1) * KV * hd * sizeof(float));
   float *att = malloc((size_t)T * H * hd * sizeof(float));
   double tp[8] = {0};                            /* embed, norm+quant, qkv+rope, attention, quantize, scatter, ffn, head */
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
#pragma omp for schedule(static)
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
      if (run_step(&c, Sq, in))
         return 3;
      t = now_ms();
      jbg_qkv_rope(Sq->nj, Sq->lv, Sq->ts, Sq->js, Sq->n, Sq->rows, Sq->sy, Sq->zy, T, H, KV, hd, 0, inv_freq,
                   rope[l], q, kk, vv);
      snprintf(nm, sizeof(nm), "k0.%d", l);
      memcpy(kk, need(&pr, nm, 0), (size_t)KV * hd * sizeof(float));
      snprintf(nm, sizeof(nm), "v0.%d", l);
      memcpy(vv, need(&pr, nm, 0), (size_t)KV * hd * sizeof(float));
      tp[2] += now_ms() - t;
      t = now_ms();
      jbg_attention(q + (size_t)H * hd, kk, vv, att, T, T + 1, 1, H, KV, hd);   /* queries 1..T over keys 0..T */
      tp[3] += now_ms() - t;
      t = now_ms();
      jbg_quant_planar(att, T, H * hd, So->sx, So->zx, in);
      tp[4] += now_ms() - t;
      if (run_step(&c, So, in))
         return 3;
      t = now_ms();
      jbg_scatter_jobs(So->nj, So->lv, So->ts, So->js, So->n, So->rows, So->sy, So->zy, x + D, T, D, 1);
      tp[5] += now_ms() - t;
      t = now_ms();
      jbg_norm_quant(x, T, D, eps, Sg->sx, Sg->zx, in, a0);
      tp[1] += now_ms() - t;
      if (run_step(&c, Sg, in))
         return 3;
      t = now_ms();
      jbg_ffn_fused(Sg->nj, Sg->lv, Sg->ts, Sg->js, Sg->n, Sg->rows, Sg->sy, Sg->zy, T, I, Hs, bdown, Sd->sx, Sd->zx, in);
      tp[6] += now_ms() - t;
      if (run_step(&c, Sd, in))
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
   int nl = all ? T : 1;
   float *xf = malloc((size_t)nl * D * sizeof(float));
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
   float *lg = malloc((size_t)nl * V * sizeof(float));
#pragma omp parallel
   {
      float *e = malloc((size_t)D * sizeof(float));
#pragma omp for schedule(static)
      for (int v = 0; v < V; v++) {
         const uint16_t *er = embed + (size_t)v * D;
         for (int d = 0; d < D; d++) {
            uint32_t u = (uint32_t)er[d] << 16;
            memcpy(&e[d], &u, 4);
         }
         for (int i = 0; i < nl; i++)
            lg[(size_t)i * V + v] = dot_f32(xf + (size_t)i * D, e, D);
      }
      free(e);
   }
   tp[7] += now_ms() - t;
   double wall = now_ms() - t0;
   double npu = 0, ro = 0;
   for (int p = 0; p < 4 * L; p++) {
      npu += steps[p].t_npu;
      ro += steps[p].t_ro;
   }
   int best = 0;
   for (int v = 1; v < V; v++)
      if (lg[(size_t)(nl - 1) * V + v] > lg[(size_t)(nl - 1) * V + best])
         best = v;
   printf("chunk: %.1f ms = steps %.1f (%s; read-out %.1f) + glue: embed %.1f, norm+quant %.1f, qkv+rope %.1f, "
          "attention %.1f, quantize %.1f, scatter %.1f, ffn %.1f, head %.1f ms; last token's argmax %d\n",
          wall, npu, ref ? "CPU reference" : "NPU incl. submit", ro, tp[0], tp[1], tp[2], tp[3], tp[4], tp[5],
          tp[6], tp[7], best);
   if (outp)
      write_f32(outp, lg, (size_t)nl * V);
   printf("{\"wall_ms\": %.1f, \"steps_ms\": %.1f, \"readout_ms\": %.1f, \"glue_ms\": %.1f, \"argmax\": %d, "
          "\"total_ms\": %.1f}\n", wall, npu, ro, wall - npu, best, now_ms() - t_start);
   (void)jb_engine_main;
   return 0;
}
