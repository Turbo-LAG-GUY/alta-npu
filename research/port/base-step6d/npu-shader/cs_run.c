/*
 * cs_run: device harness for the etnaviv compute port (~/Documents/AI_Projects/mesa-compute), 2026-11-13.
 * Runs the test kernels of cs_kernels.h (copy, fma) on the A311D NPU's shader core through the port's own
 * compiler and launch_grid, as cl2023_copy did through the 2023 branch.
 *
 * usage: cs_run [--offline] [--kernel K] [-n N] [-w WG] [--fma ITERS] [--self] [--repeat R] [--compile-only]
 *               [--out DIR]
 *   --kernel K      copy (default), fma, ids, reverse, reverse-nobarrier, reduce (see cs_kernels.h; step 5 kernels use
 *                   shared memory and/or barriers); --self: reverse reads the thread's own slot instead of the mirror
 *   --offline       run against the fake NPU of shim/libnpu_etna_shim.so (LD_PRELOAD): everything up to the submit
 *                   runs on the CPU, the shim records the submits, nothing executes; refused without the shim
 *   --fma ITERS     float4 kernel with ITERS unrolled multiply-adds per thread (8 flops each); timed over R launches
 *   --compile-only  open the device, compile the kernel, launch nothing
 *   --out DIR       work in DIR: ETNA_MESA_DEBUG=dump_shaders writes the shader and command-stream dumps there
 * Without --offline it opens the NPU: only through ~/alta-npu/scripts/npu-attempt, with Jordan's OK.
 */
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/stat.h>
#include <xf86drm.h>

#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/u_inlines.h"
#include "etnaviv/drm/etnaviv_drm_public.h"

#include "cs_kernels.h"

static int steps_fd = -1;
static void
step(const char *msg)
{
   if (steps_fd < 0)
      return;
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC, &ts);
   char line[256];
   int n = snprintf(line, sizeof(line), "[%12.6f] pid=%d %s\n", ts.tv_sec + ts.tv_nsec / 1e9, getpid(), msg);
   if (write(steps_fd, line, n) < 0) { }
   fsync(steps_fd);
}

static bool
is_etnaviv(int fd)
{
   drmVersionPtr v = drmGetVersion(fd);
   bool etna = v && !strcmp(v->name, "etnaviv");
   if (etna)
      printf("DRM %s %d.%d\n", v->name, v->version_major, v->version_minor);
   drmFreeVersion(v);
   return etna;
}

static int
open_etnaviv(bool offline)
{
   /* The shim exports this symbol; its presence decides which kind of node may be opened. */
   const bool shim = dlsym(RTLD_DEFAULT, "npu_etna_shim_active") != NULL;

   if (offline != shim) {
      fprintf(stderr, offline ? "--offline needs LD_PRELOAD=.../shim/libnpu_etna_shim.so; nothing opened\n"
                              : "the NPU shim is loaded: use --offline (or unset LD_PRELOAD); nothing opened\n");
      return -1;
   }

   if (offline) {
      /* the shim takes over renderD128 (open returns /dev/null); no other node is opened */
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

int
main(int argc, char **argv)
{
   unsigned n = 1, wg = 1, fma_iters = 0, repeat = 1, ny = 1, wy = 1, H = 0, W = 0, C = 0, len = 1001, rows = 1;
   float factor = 0.16609922f * 1.4426950409f; /* MobileNet's softmax: beta 1, input scale 0.16609922 */
   bool compile_only = false, offline = false, self = false, dir_out = false;
   const char *kernel = NULL;
   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "-n") && i + 1 < argc)
         n = strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "-w") && i + 1 < argc)
         wg = strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--fma") && i + 1 < argc)
         fma_iters = strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--kernel") && i + 1 < argc)
         kernel = argv[++i];
      else if (!strcmp(argv[i], "--self"))
         self = true;
      else if (!strcmp(argv[i], "--len") && i + 1 < argc)
         len = strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--rows") && i + 1 < argc)
         rows = strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--factor") && i + 1 < argc)
         factor = strtof(argv[++i], NULL);
      else if (!strcmp(argv[i], "--ny") && i + 1 < argc)
         ny = strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--wy") && i + 1 < argc)
         wy = strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--hwc") && i + 3 < argc) {
         H = strtoul(argv[++i], NULL, 0);
         W = strtoul(argv[++i], NULL, 0);
         C = strtoul(argv[++i], NULL, 0);
      } else if (!strcmp(argv[i], "--dir") && i + 1 < argc) {
         const char *d = argv[++i];
         if (strcmp(d, "in") && strcmp(d, "out")) {
            fprintf(stderr, "--dir is in (NHWC -> NCHW) or out (NCHW -> NHWC)\n");
            return 2;
         }
         dir_out = !strcmp(d, "out");
      }
      else if (!strcmp(argv[i], "--repeat") && i + 1 < argc)
         repeat = strtoul(argv[++i], NULL, 0);
      else if (!strcmp(argv[i], "--compile-only"))
         compile_only = true;
      else if (!strcmp(argv[i], "--offline"))
         offline = true;
      else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
         const char *dir = argv[++i];
         mkdir(dir, 0755);
         if (chdir(dir)) {
            perror(dir);
            return 2;
         }
      } else {
         fprintf(stderr, "usage: %s [--offline] [--kernel K] [-n N] [-w WG] [--ny NY] [--wy WY] [--hwc H W C] "
                         "[--dir in|out] [--fma ITERS] [--self] [--repeat R] [--compile-only] [--out DIR]\n", argv[0]);
         return 2;
      }
   }
   if (!kernel)
      kernel = fma_iters ? "fma" : "copy";
   enum { K_COPY, K_FMA, K_IDS, K_REVERSE, K_REVERSE_NB, K_REDUCE, K_IDS2D, K_T4X4, K_TGATHER, K_T4X4B, K_T4X4COPY, K_FOPS, K_SOFTMAX, K_IDS_GL, K_REGP, K_AVGPOOL, K_LOOPSUM } kid;
   unsigned k4 = 0;
   if (!strcmp(kernel, "copy"))
      kid = K_COPY;
   else if (!strcmp(kernel, "fma"))
      kid = K_FMA;
   else if (!strcmp(kernel, "ids"))
      kid = K_IDS;
   else if (!strcmp(kernel, "reverse"))
      kid = K_REVERSE;
   else if (!strcmp(kernel, "reverse-nobarrier"))
      kid = K_REVERSE_NB;
   else if (!strcmp(kernel, "reduce"))
      kid = K_REDUCE;
   else if (!strcmp(kernel, "ids2d"))
      kid = K_IDS2D;
   else if (!strcmp(kernel, "t4x4"))
      kid = K_T4X4;
   else if (!strcmp(kernel, "tgather"))
      kid = K_TGATHER;
   else if (!strcmp(kernel, "t4x4b"))
      kid = K_T4X4B;
   else if (!strcmp(kernel, "t4x4copy"))
      kid = K_T4X4COPY;
   else if (!strcmp(kernel, "fops"))
      kid = K_FOPS;
   else if (!strcmp(kernel, "softmax"))
      kid = K_SOFTMAX;
   else if (!strcmp(kernel, "ids_gl"))
      kid = K_IDS_GL;
   else if (!strcmp(kernel, "avgpool"))
      kid = K_AVGPOOL;
   else if (!strcmp(kernel, "loopsum"))
      kid = K_LOOPSUM; /* -n threads, --len trip count */
   else if (!strncmp(kernel, "regp", 4)) {
      kid = K_REGP; /* regpK: K vec4 values live across the barrier */
      k4 = atoi(kernel + 4);
   }
   else {
      fprintf(stderr, "unknown kernel %s\n", kernel);
      return 2;
   }
   if (kid == K_FMA && !fma_iters)
      fma_iters = 16;
   if (kid != K_FMA)
      fma_iters = 0;
   if (kid == K_REDUCE && (wg & (wg - 1))) {
      fprintf(stderr, "reduce needs a power-of-two workgroup size\n");
      return 2;
   }
   const bool t4 = kid == K_T4X4 || kid == K_T4X4B || kid == K_T4X4COPY;
   const bool transpose = t4 || kid == K_TGATHER, grid2d = transpose || kid == K_IDS2D;
   const unsigned HW = H * W;
   if (transpose) {
      /* threads: x = groups of 4 pixels, y = groups of 4 channels (t4x4) or channels (tgather) */
      if (!HW || !C || HW % 4 || (t4 && C % 4) || (kid == K_TGATHER && dir_out)) {
         fprintf(stderr, "transpose: --hwc H W C with H*W %% 4 == 0 (t4x4: C %% 4 == 0; tgather: --dir in only)\n");
         return 2;
      }
      n = HW / 4;
      ny = t4 ? C / 4 : C;
   }
   if (!grid2d)
      ny = wy = 1;
   if (kid == K_SOFTMAX) {
      if (!len || !rows || (rows > 1 && len % 4)) {
         fprintf(stderr, "softmax: --len N --rows R (R > 1 needs N %% 4 == 0)\n");
         return 2;
      }
      if (wg < 4 || wg > 64 || wg % 4) {
         fprintf(stderr, "softmax: -w THREADS, 4..64 and a multiple of 4\n");
         return 2;
      }
      n = rows * wg;
   }
   if (kid == K_AVGPOOL) {
      /* one thread per 4 channel planes of H*W bytes */
      if (!HW || HW > 256 || !C || C % 4) {
         fprintf(stderr, "avgpool: --hwc H W C with H*W 1..256 and C %% 4 == 0\n");
         return 2;
      }
      n = C / 4;
   }
   if (!wg || !wy || wg * wy > 256 || !n || n % wg || !ny || ny % wy || !repeat) {
      fprintf(stderr, "threads (%u x %u) must be nonzero multiples of the workgroup (%u x %u, at most 256 threads), "
                      "repeat >= 1\n", n, ny, wg, wy);
      return 2;
   }
   const char *sf = getenv("NPU_STEPS_FILE");
   if (sf)
      steps_fd = open(sf, O_WRONLY | O_CREAT | O_APPEND, 0644);

   step("open: begin");
   int fd = open_etnaviv(offline);
   if (fd < 0) {
      fprintf(stderr, "no etnaviv render node\n");
      return 2;
   }
   struct pipe_screen *ps = etna_drm_screen_create(fd);
   if (!ps) {
      fprintf(stderr, "etna_drm_screen_create failed\n");
      return 2;
   }
   printf("screen: %s, compute %s\n", ps->get_name(ps), ps->caps.compute ? "yes" : "no");
   struct pipe_context *pc = ps->context_create(ps, NULL, PIPE_CONTEXT_COMPUTE_ONLY);
   if (!pc || !pc->launch_grid) {
      fprintf(stderr, pc ? "context has no compute (launch_grid)\n" : "context_create failed\n");
      return 2;
   }
   step("context: done");

   const nir_shader_compiler_options *opts = ps->nir_options[MESA_SHADER_COMPUTE];
   if (!opts) {
      fprintf(stderr, "screen has no compute compiler options\n");
      return 2;
   }
   nir_shader *nir = kid == K_FMA ? cs_build_fma(opts, fma_iters) :
                     kid == K_IDS ? cs_build_ids(opts) :
                     kid == K_REVERSE ? cs_build_reverse(opts, true) :
                     kid == K_REVERSE_NB ? cs_build_reverse(opts, false) :
                     kid == K_REDUCE ? cs_build_reduce(opts, wg) :
                     kid == K_IDS2D ? cs_build_ids2d(opts) :
                     kid == K_T4X4 ? cs_build_t4x4(opts, 0) :
                     kid == K_T4X4B ? cs_build_t4x4(opts, 1) :
                     kid == K_T4X4COPY ? cs_build_t4x4(opts, 2) :
                     kid == K_TGATHER ? cs_build_tgather(opts) :
                     kid == K_FOPS ? cs_build_fops(opts) :
                     kid == K_SOFTMAX ? etna_ml_shader_build_softmax(opts, len, rows, factor, wg) :
                     kid == K_AVGPOOL ? etna_ml_shader_build_avgpool(opts, HW, C) :
                     kid == K_LOOPSUM ? cs_build_loopsum(opts) :
                     kid == K_IDS_GL ? cs_build_ids_gl(opts) :
                     kid == K_REGP ? cs_build_regp(opts, k4) : cs_build_copy(opts);
   if (!nir) {
      fprintf(stderr, "kernel builder refused these parameters\n");
      return 2;
   }
   if (!fma_iters)
      nir_print_shader(nir, stdout);
   printf("kernel: %s, shared %u bytes per workgroup\n", kernel, nir->info.shared_size);
   struct pipe_compute_state cs = {
      .ir_type = PIPE_SHADER_IR_NIR,
      .prog = nir,
      .static_shared_mem = 0,
   };
   void *cso = pc->create_compute_state(pc, &cs);
   if (!cso) {
      fprintf(stderr, "create_compute_state failed\n");
      return 2;
   }
   step("compute state: created");
   if (compile_only) {
      printf("compile only: nothing launched\n");
      pc->delete_compute_state(pc, cso);
      pc->destroy(pc);
      ps->destroy(ps);
      return 0;
   }
   pc->bind_compute_state(pc, cso);

   unsigned elem = fma_iters ? 16 : (kid == K_IDS || kid == K_IDS2D || kid == K_FOPS) ? 32 : kid == K_IDS_GL ? 8 : 4;
   size_t bytes = transpose ? (size_t)HW * C : (size_t)n * ny * elem; /* input buffer: same size */
   if (kid == K_SOFTMAX)
      bytes = ((size_t)rows * len + 3) / 4 * 4; /* whole words: the kernel writes up to 3 bytes past the end */
   if (kid == K_REGP)
      bytes = (size_t)n * 4 + (size_t)n * k4 * 16; /* out uses the first n words; in holds the extras after them */
   if (kid == K_LOOPSUM)
      bytes = (size_t)n * 4 * len; /* in: len rows of n words; out: the first n words, the rest must stay 0xdeadbeef */
   if (kid == K_AVGPOOL)
      bytes = (size_t)HW * C; /* in: C planes of HW bytes; out: the first C bytes, the rest must stay 0xdeadbeef */
   uint32_t *in = malloc(bytes), *out = malloc(bytes), *want = malloc(bytes);
   float *fin = (float *)in, *fout = (float *)out;
   uint8_t *in8 = (uint8_t *)in, *want8 = (uint8_t *)want;
   for (unsigned i = 0; i < bytes / 4; i++) {
      if (fma_iters)
         fin[i] = (float)((i * 2654435761u) % 1000u) / 1000.0f;
      else
         in[i] = i * 7 + 1;
      out[i] = 0xdeadbeef;
   }
   if (kid == K_SOFTMAX) {
      /* logits like MobileNet's: a spread of values plus a few peaks (byte past the end stays 0xef) */
      for (size_t i = 0; i < (size_t)rows * len; i++)
         in8[i] = (uint8_t)((i * 37u + 11u) % 97u + 40u);
      for (unsigned r = 0; r < rows; r++) {
         in8[(size_t)r * len + (5 + 13 * r) % len] = 230;
         in8[(size_t)r * len + (500 + 7 * r) % len] = 215;
         in8[(size_t)r * len + (len - 1)] = 200;
      }
   }
   if (kid == K_AVGPOOL) {
      /* channels 0-3 all 255 (the largest lane sums), 4-7 all 0, the rest pseudo-random bytes */
      uint32_t x = 0x9e3779b9u;
      for (size_t i = 0; i < bytes; i++) {
         const size_t c = i / HW;
         x ^= x << 13;
         x ^= x >> 17;
         x ^= x << 5;
         in8[i] = c < 4 ? 255 : c < 8 ? 0 : (uint8_t)(x >> 24);
      }
   }
   if (transpose) {
      /* a byte pattern that changes with every position (period far above any stride used) */
      for (size_t i = 0; i < bytes; i++)
         in8[i] = (uint8_t)(i * 131u + (i >> 8) * 7u + (i >> 16) * 3u + 1u);
   }
   struct pipe_resource *res[2] = {
      pipe_buffer_create(ps, PIPE_BIND_GLOBAL, PIPE_USAGE_DEFAULT, bytes < 4096 ? 4096 : bytes),
      pipe_buffer_create(ps, PIPE_BIND_GLOBAL, PIPE_USAGE_DEFAULT, bytes < 4096 ? 4096 : bytes),
   };
   if (!res[0] || !res[1]) {
      fprintf(stderr, "buffer allocation failed\n");
      return 2;
   }
   pipe_buffer_write(pc, res[0], 0, bytes, out);
   pipe_buffer_write(pc, res[1], 0, bytes, in);
   uint32_t addr[2] = { 0, 0 }; /* offsets into the buffers on entry, GPU addresses on return */
   uint32_t *handles[2] = { &addr[0], &addr[1] };
   pc->set_global_binding(pc, 0, 2, res, handles);
   printf("GPU addresses: out 0x%x in 0x%x (%zu bytes each)\n", addr[0], addr[1], bytes);
   /* kernel input buffer = constant buffer 0 (offset 0 = out address, 4 = in address, 8 = A, 12 = B); read by
    * launch_grid. reverse: shared index = A + B * local id (mirror: A = wg - 1, B = -1; --self: A = 0, B = 1) */
   uint32_t args[8] = { addr[0], addr[1], self ? 0 : wg - 1, self ? 1 : 0xffffffffu, 0, 0, 0, 0 };
   if (kid == K_IDS_GL) {
      args[2] = wg;
      args[3] = n - 1;
   }
   if (kid == K_REGP) {
      args[2] = n * 4;
      args[3] = wg - 1;
   }
   if (kid == K_LOOPSUM) {
      args[2] = len;   /* trip count */
      args[3] = n * 4; /* bytes between a thread's words */
   }
   if (kid == K_IDS2D) {
      args[2] = n; /* threads in x */
      args[3] = n - 1;
      args[4] = ny - 1;
   }
   if (t4 && !dir_out) { /* NHWC -> NCHW: RX RY RS WX WY WS */
      const uint32_t a[6] = { 4 * C, 4, C, 4, 4 * HW, HW };
      memcpy(&args[2], a, sizeof(a));
   }
   if (t4 && dir_out) { /* NCHW -> NHWC */
      const uint32_t a[6] = { 4, 4 * HW, HW, 4 * C, 4, C };
      memcpy(&args[2], a, sizeof(a));
   }
   if (kid == K_TGATHER) {
      args[2] = C;
      args[3] = HW;
   }
   pc->set_constant_buffer(pc, MESA_SHADER_COMPUTE, 0,
                           &(struct pipe_constant_buffer){ .user_buffer = args, .buffer_size = sizeof(args) });

   struct pipe_grid_info g = { .work_dim = grid2d ? 2 : 1, .block = { wg, wy, 1 }, .grid = { n / wg, ny / wy, 1 } };
   printf("grid: %u dimension(s), %u x %u workgroups of %u x %u threads\n", g.work_dim, g.grid[0], g.grid[1], wg, wy);
   double ms = 0, first_ms = 0, *times = calloc(repeat, sizeof(double));
   for (unsigned r = 0; r < repeat; r++) {
      step(r == 0 ? "launch 1: begin" : "launch: begin");
      struct timespec t0, t1;
      clock_gettime(CLOCK_MONOTONIC, &t0);
      pc->launch_grid(pc, &g);
      struct pipe_fence_handle *fence = NULL;
      pc->flush(pc, &fence, 0);
      bool done = fence && ps->fence_finish(ps, NULL, fence, 3000000000ull);
      ps->fence_reference(ps, &fence, NULL);
      clock_gettime(CLOCK_MONOTONIC, &t1);
      times[r] = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;
      if (r == 0)
         first_ms = times[r];
      if (!done) {
         step("launch: fence wait FAILED");
         printf("{\"result\": \"TIMEOUT\", \"launch\": %u, \"ms\": %.2f}\n", r + 1, times[r]);
         return 4;
      }
   }
   step("launches: done");
   /* median of the launches after the first (the first one also uploads the shader) */
   unsigned m = repeat > 1 ? repeat - 1 : 1;
   double *t = repeat > 1 ? times + 1 : times;
   for (unsigned i = 0; i < m; i++)
      for (unsigned j = i + 1; j < m; j++)
         if (t[j] < t[i]) {
            double x = t[i];
            t[i] = t[j];
            t[j] = x;
         }
   ms = t[m / 2];

   /* expected output of the integer kernels */
   if (kid == K_T4X4COPY) {
      /* no shuffle: thread (x, y) writes its input word k (at in + x*RX + y*RY + k*RS) to out + x*WX + y*WY + k*WS */
      const uint32_t *a = &args[2];
      for (unsigned y = 0; y < ny; y++)
         for (unsigned x = 0; x < n; x++)
            for (unsigned k = 0; k < 4; k++)
               want[((size_t)x * a[3] + (size_t)y * a[4] + (size_t)k * a[5]) / 4] =
                  in[((size_t)x * a[0] + (size_t)y * a[1] + (size_t)k * a[2]) / 4];
   } else if (transpose) {
      for (unsigned p = 0; p < HW; p++)
         for (unsigned c = 0; c < C; c++) {
            if (dir_out)
               want8[(size_t)p * C + c] = in8[(size_t)c * HW + p];
            else
               want8[(size_t)c * HW + p] = in8[(size_t)p * C + c];
         }
   }
   if (kid == K_IDS2D) {
      for (unsigned y = 0; y < ny; y++)
         for (unsigned x = 0; x < n; x++) {
            const uint32_t w[8] = { x, y, x % wg, y % wy, x / wg, y / wy, n / wg, ny / wy };
            memcpy(&want[8 * ((size_t)y * n + x)], w, sizeof(w));
         }
   }
   for (unsigned i = 0; i < n && !grid2d; i++) {
      const unsigned g = i / wg, l = i % wg;
      switch (kid) {
      case K_IDS: {
         const uint32_t w[8] = { i, l, g, 0, 0, n / wg, 1, 1 };
         memcpy(&want[8 * i], w, sizeof(w));
         break;
      }
      case K_REVERSE:
      case K_REVERSE_NB:
         want[i] = in[g * wg + (self ? l : wg - 1 - l)];
         break;
      case K_REDUCE: {
         uint32_t sum = 0;
         for (unsigned j = 0; j < wg; j++)
            sum += in[g * wg + j];
         want[i] = sum;
         break;
      }
      case K_LOOPSUM: {
         uint32_t sum = 0;
         for (unsigned k = 0; k < len && k < CS_LOOP_MAX; k++)
            sum += in[i + (size_t)k * n];
         want[i] = sum;
         break;
      }
      case K_IDS_GL:
         want[2 * i] = g;
         want[2 * i + 1] = l;
         break;
      case K_REGP: {
         uint32_t w = in[g * wg + (wg - 1 - l)];
         for (unsigned e = 0; e < 4 * k4; e++)
            w += in[n + (size_t)i * k4 * 4 + e];
         want[i] = w;
         break;
      }
      default:
         want[i] = in[i];
         break;
      }
   }

   if (kid == K_LOOPSUM)
      for (size_t i = n; i < bytes / 4; i++)
         want[i] = 0xdeadbeef; /* only the first n words are written */
   pipe_buffer_read(pc, res[0], 0, bytes, out);
   if (kid == K_AVGPOOL) {
      const uint8_t *out8 = (const uint8_t *)out;
      const uint32_t fill = 0xdeadbeef;
      unsigned exact = 0, wrong = 0, stray = 0, untouched = 0;
      for (unsigned c = 0; c < C; c++) {
         unsigned sum = 0;
         for (unsigned p = 0; p < HW; p++)
            sum += in8[(size_t)c * HW + p];
         const unsigned want_q = (sum + HW / 2) / HW;
         untouched += out8[c] == ((const uint8_t *)&fill)[c % 4];
         if (out8[c] == want_q) {
            exact++;
         } else {
            if (wrong < 8)
               printf("channel %u: got %u want %u (sum %u)\n", c, out8[c], want_q, sum);
            wrong++;
         }
      }
      for (size_t i = C; i < bytes; i++)
         stray += out8[i] != ((const uint8_t *)&fill)[i % 4];
      printf("avgpool: %ux%u x %u channels: %u exact, %u wrong, %u stray bytes past the output; first: %u %u %u %u\n",
             H, W, C, exact, wrong, stray, out8[0], out8[4], out8[8], out8[9]);
      const char *ares = offline ? "OFFLINE" : (wrong || stray ? "MISMATCH" : "PASS");
      printf("{\"result\": \"%s\", \"kernel\": \"avgpool\", \"h\": %u, \"w\": %u, \"c\": %u, \"exact\": %u, "
             "\"wrong\": %u, \"stray\": %u, \"untouched\": %u, \"first_launch_ms\": %.3f, \"median_ms\": %.3f, "
             "\"launches\": %u}\n", ares, H, W, C, exact, wrong, stray, untouched, first_ms, ms, repeat);
      step("finished");
      pc->set_global_binding(pc, 0, 2, NULL, NULL);
      pipe_resource_reference(&res[0], NULL);
      pipe_resource_reference(&res[1], NULL);
      pc->delete_compute_state(pc, cso);
      pc->destroy(pc);
      ps->destroy(ps);
      return offline ? 0 : (wrong || stray ? 3 : 0);
   }
   if (kid == K_SOFTMAX) {
      const uint8_t *out8 = (const uint8_t *)out;
      unsigned exact = 0, off1 = 0, worse = 0, worst = 0, untouched8 = 0;
      for (unsigned r = 0; r < rows; r++) {
         const uint8_t *x = in8 + (size_t)r * len;
         unsigned mx = 0;
         for (unsigned i = 0; i < len; i++)
            mx = x[i] > mx ? x[i] : mx;
         double sum = 0;
         for (unsigned i = 0; i < len; i++)
            sum += exp2(((double)x[i] - mx) * factor);
         for (unsigned i = 0; i < len; i++) {
            double q = floor(exp2(((double)x[i] - mx) * factor) / sum * 256.0 + 0.5);
            unsigned want_q = q > 255 ? 255 : (unsigned)q;
            unsigned got_q = out8[(size_t)r * len + i];
            unsigned d = got_q > want_q ? got_q - want_q : want_q - got_q;
            untouched8 += got_q == 0xef;
            exact += d == 0;
            off1 += d == 1;
            worse += d > 1;
            worst = d > worst ? d : worst;
         }
      }
      printf("softmax: %u rows of %u, factor %.7g: %u exact, %u off by 1, %u worse (worst %u); row 0 peaks: %u %u %u\n",
             rows, len, factor, exact, off1, worse, worst, out8[5], out8[500 % len], out8[len - 1]);
      const char *sres = offline ? "OFFLINE" : (worst > 1 ? "MISMATCH" : "PASS");
      printf("{\"result\": \"%s\", \"kernel\": \"softmax\", \"rows\": %u, \"len\": %u, \"exact\": %u, \"off_by_1\": %u, "
             "\"worse\": %u, \"worst\": %u, \"bytes_0xef\": %u, \"first_launch_ms\": %.3f, \"median_ms\": %.3f, "
             "\"launches\": %u}\n", sres, rows, len, exact, off1, worse, worst, untouched8, first_ms, ms, repeat);
      step("finished");
      pc->set_global_binding(pc, 0, 2, NULL, NULL);
      pipe_resource_reference(&res[0], NULL);
      pipe_resource_reference(&res[1], NULL);
      pc->delete_compute_state(pc, cso);
      pc->destroy(pc);
      ps->destroy(ps);
      return offline ? 0 : (worst > 1 ? 3 : 0);
   }
   if (kid == K_FOPS) {
      /* per result: worst relative error (f2u, floor, max, min: must be exact) */
      static const char *names[8] = { "exp2", "exp", "rcp", "div", "fmax", "f2u", "floor", "fmin" };
      double worst[8] = { 0 };
      unsigned fbad = 0, funtouched = 0;
      for (unsigned i = 0; i < n; i++) {
         const float a = (float)(in[i] & 0xff), x = a / 32.0f - 4.0f, r = 1.0f / (a + 1.0f);
         const float ref[8] = { exp2f(x), expf(x), r, a / (a + 3.0f), fmaxf(x, 0.25f), 0, floorf(a * 0.3f),
                                fminf(r, 0.5f) };
         const uint32_t ref_u = (uint32_t)(a * 0.75f + 0.5f);
         for (unsigned k = 0; k < 8; k++) {
            const uint32_t gotw = out[8 * i + k];
            funtouched += gotw == 0xdeadbeef;
            double err;
            if (k == 5)
               err = gotw == ref_u ? 0.0 : 1.0;
            else {
               float g;
               memcpy(&g, &gotw, 4);
               err = fabs((double)g - ref[k]) / fmax(fabs(ref[k]), 1e-30);
               if (ref[k] == 0.0f)
                  err = fabs((double)g);
            }
            if (err > worst[k])
               worst[k] = err;
         }
      }
      const double tol[8] = { 1e-4, 1e-4, 1e-4, 1e-4, 0, 0, 0, 1e-4 };
      printf("fops worst relative error:");
      for (unsigned k = 0; k < 8; k++) {
         printf(" %s %.3g", names[k], worst[k]);
         fbad += worst[k] > tol[k];
      }
      printf("\nthread 1:");
      for (unsigned k = 0; k < 8; k++) {
         float g;
         memcpy(&g, &out[8 + k], 4);
         k == 5 ? printf(" %u", out[8 + k]) : printf(" %.6g", g);
      }
      printf("\n");
      const char *fres = offline ? (funtouched == bytes / 4 ? "OFFLINE" : "OFFLINE-UNEXPECTED")
                                 : (fbad ? "MISMATCH" : "PASS");
      printf("{\"result\": \"%s\", \"kernel\": \"fops\", \"threads\": %u, \"results_out_of_tolerance\": %u, "
             "\"untouched\": %u, \"worst\": [%.3g, %.3g, %.3g, %.3g, %.3g, %.3g, %.3g, %.3g], \"ms\": %.3f}\n",
             fres, n, fbad, funtouched, worst[0], worst[1], worst[2], worst[3], worst[4], worst[5], worst[6],
             worst[7], first_ms);
      step("finished");
      pc->set_global_binding(pc, 0, 2, NULL, NULL);
      pipe_resource_reference(&res[0], NULL);
      pipe_resource_reference(&res[1], NULL);
      pc->delete_compute_state(pc, cso);
      pc->destroy(pc);
      ps->destroy(ps);
      if (offline)
         return funtouched == bytes / 4 ? 0 : 5;
      return fbad ? 3 : 0;
   }
   unsigned bad = 0, untouched = 0;
   unsigned field_bad[8] = { 0 };
   int first = -1;
   double worst = 0;
   const unsigned checked_words = kid == K_REGP ? n : bytes / 4; /* regp writes the first n words only */
   for (unsigned i = 0; i < checked_words; i++) {
      untouched += out[i] == 0xdeadbeef;
      bool ok;
      if (fma_iters) {
         float x = fin[i];
         for (unsigned k = 0; k < fma_iters; k++)
            x = fmaf(x, 0.999f, 0.001f);
         double d = fabs((double)fout[i] - x);
         if (d > worst)
            worst = d;
         ok = d <= 1e-4;
      } else {
         ok = out[i] == want[i];
         if (kid == K_IDS && !ok)
            field_bad[i % 8]++;
      }
      if (!ok) {
         bad++;
         if (first < 0)
            first = i;
      }
   }
   if (kid == K_IDS && !offline) {
      static const char *names[8] = { "global.x", "local.x", "group.x", "group.y", "group.z",
                                      "groups.x", "groups.y", "groups.z" };
      printf("ids: wrong values per field:");
      for (unsigned f = 0; f < 8; f++)
         printf(" %s %u", names[f], field_bad[f]);
      printf("\nthread 0..1:");
      for (unsigned i = 0; i < 16 && i < bytes / 4; i++)
         printf(" %u", out[i]);
      printf("\n");
   }
   printf("out[0..3]:");
   for (unsigned i = 0; i < 4 && i < bytes / 4; i++)
      fma_iters ? printf(" %.6f", fout[i]) : printf(" 0x%x", out[i]);
   printf("\n");
   /* offline, nothing executed: the output buffer is expected to be untouched */
   const char *result = offline ? (untouched == checked_words ? "OFFLINE" : "OFFLINE-UNEXPECTED")
                                : (bad ? "MISMATCH" : "PASS");
   if (fma_iters) {
      double flops = (double)n * fma_iters * 8.0;
      printf("{\"result\": \"%s\", \"kernel\": \"fma\", \"threads\": %u, \"workgroup\": %u, \"fma_per_thread\": %u, "
             "\"mismatches\": %u, \"untouched\": %u, \"worst_abs_error\": %.3g, \"first_launch_ms\": %.3f, "
             "\"median_ms\": %.3f, \"launches\": %u, \"gflops\": %.3f}\n",
             result, n, wg, fma_iters, bad, untouched, worst, first_ms, ms, repeat, flops / (ms * 1e6));
   } else
      printf("{\"result\": \"%s\", \"kernel\": \"%s%s%s\", \"threads\": \"%u x %u\", \"workgroup\": \"%u x %u\", "
             "\"hwc\": \"%ux%ux%u\", \"mismatched_words\": %u, \"untouched\": %u, \"first_mismatch\": %d, "
             "\"got\": %u, \"want\": %u, \"launch_to_done_ms\": %.3f, \"median_ms\": %.3f, \"launches\": %u, "
             "\"mb_per_s\": %.1f}\n",
             result, kernel, self ? " (self)" : "", transpose ? (dir_out ? " out" : " in") : "", n, ny, wg, wy,
             H, W, C, bad, untouched, first, first >= 0 ? out[first] : 0, first >= 0 ? want[first] : 0, first_ms,
             ms, repeat, transpose ? bytes / (ms * 1000.0) : 0.0);
   step("finished");
   pc->set_global_binding(pc, 0, 2, NULL, NULL);
   pipe_resource_reference(&res[0], NULL);
   pipe_resource_reference(&res[1], NULL);
   pc->delete_compute_state(pc, cso);
   pc->destroy(pc);
   ps->destroy(ps);
   if (offline)
      return untouched == checked_words ? 0 : 5;
   return bad ? 3 : 0;
}
