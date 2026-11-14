/* cs_compile: offline compiler check for the etnaviv compute port (mesa-compute tree), 2026-11-13.
 *
 * Compiles the test kernels of cs_kernels.h with the port tree's etnaviv compiler for the A311D NPU's shader
 * core WITHOUT opening any device: the core description comes from Mesa's hardware database (the same entry
 * the driver matches on this board: GC8000 rev 0x7120, product 0x45080009, customer 0x88, eco 0) and the
 * compiler-relevant screen specs are derived as etnaviv_screen.c does. CPU only, nothing is submitted.
 *
 *   cs_compile [--kernel copy|fma|ids|reverse|reverse-nobarrier|reduce|ids2d|t4x4|tgather] [--fma N] [-w WG] [--out FILE.bin] [--dump]
 *   (--fma N alone = --kernel fma; -w is the workgroup size the reduce kernel is unrolled for, default 64)
 *
 * Prints the variant's uniforms, temps and work-item ID order; --out writes the shader binary (compare with
 * research/port/shader_diff.sh); --dump also prints NIR and Mesa's disassembly (ETNA_DBG_DUMP_SHADERS).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "etnaviv_compiler.h"
#include "etnaviv_context.h"
#include "etnaviv_debug.h"
#include "etnaviv_internal.h"
#include "etnaviv_shader.h"
#include "hwdb/etna_hwdb.h"
#include "compiler/glsl_types.h"

#include "cs_kernels.h"

static const char *
contents_name(enum etna_uniform_contents c)
{
   switch (c) {
   case ETNA_UNIFORM_UNUSED: return "unused";
   case ETNA_UNIFORM_CONSTANT: return "constant";
   case ETNA_UNIFORM_UNIFORM: return "uniform";
   case ETNA_UNIFORM_KERNEL_ARG: return "kernel_arg";
   case ETNA_UNIFORM_GROUP_COUNT: return "group_count";
   case ETNA_UNIFORM_SHARED_ADDR: return "shared_addr";
   default: return "other";
   }
}

/* compiler-relevant parts of etna_get_specs() in etnaviv_screen.c, for a HALTI5 core */
static void
fill_specs(const struct etna_core_info *info, struct etna_specs *specs)
{
   memset(specs, 0, sizeof(*specs));
   specs->vs_need_z_div = info->model < 0x1000 && info->model != 0x880;
   specs->has_new_transcendentals = etna_core_has_feature(info, ETNA_FEATURE_HAS_FAST_TRANSCENDENTALS);
   specs->has_no_oneconst_limit = etna_core_has_feature(info, ETNA_FEATURE_SH_NO_ONECONST_LIMIT);
   specs->max_instructions = 0; /* halti5: shaders always load from memory */
   specs->has_icache = true;
   specs->num_rts = 8;
   specs->max_vs_uniforms = specs->max_ps_uniforms = MIN2(512, info->gpu.num_constants - 64);
   specs->vertex_sampler_offset = 16;
   specs->fragment_sampler_count = 16;
   specs->vertex_sampler_count = 16;
   specs->has_unified_uniforms = true;
   specs->max_cs_uniforms = MIN2(specs->max_ps_uniforms, 256);
   specs->max_vs_outputs = 32;
}

int
main(int argc, char **argv)
{
   unsigned fma_iters = 0, wg = 64, len = 1001;
   float factor = 0.16609922f * 1.4426950409f;
   const char *out = NULL, *kernel = NULL;
   bool dump = false;

   for (int i = 1; i < argc; i++) {
      if (!strcmp(argv[i], "--fma") && i + 1 < argc)
         fma_iters = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--kernel") && i + 1 < argc)
         kernel = argv[++i];
      else if (!strcmp(argv[i], "-w") && i + 1 < argc)
         wg = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--len") && i + 1 < argc)
         len = atoi(argv[++i]);
      else if (!strcmp(argv[i], "--factor") && i + 1 < argc)
         factor = atof(argv[++i]);
      else if (!strcmp(argv[i], "--out") && i + 1 < argc)
         out = argv[++i];
      else if (!strcmp(argv[i], "--dump"))
         dump = true;
      else {
         fprintf(stderr, "usage: %s [--kernel copy|fma|ids|reverse|reverse-nobarrier|reduce|ids2d|t4x4|tgather] [--fma N] [-w WG] "
                         "[--out FILE.bin] [--dump]\n", argv[0]);
         return 2;
      }
   }
   if (!kernel)
      kernel = fma_iters ? "fma" : "copy";
   if (!strcmp(kernel, "fma") && !fma_iters)
      fma_iters = 16;
   if (!wg || wg > 256 || (wg & (wg - 1))) {
      fprintf(stderr, "-w must be a power of two, 1..256\n");
      return 2;
   }

   /* before etna_compiler_create: no disk cache reads or writes */
   etna_mesa_debug = ETNA_DBG_NOCACHE | (dump ? ETNA_DBG_DUMP_SHADERS : 0);

   struct etna_core_info info = {
      .model = 0x8000, .revision = 0x7120, .product_id = 0x45080009, .eco_id = 0, .customer_id = 0x88,
   };
   if (!etna_query_feature_db(&info)) {
      fprintf(stderr, "no hwdb entry for GC8000 rev 0x7120\n");
      return 1;
   }
   if (!etna_core_has_feature(&info, ETNA_FEATURE_HALTI5)) {
      fprintf(stderr, "hwdb entry is not HALTI5\n");
      return 1;
   }
   info.halti = 5; /* as determine_halti() */

   struct etna_specs specs;
   fill_specs(&info, &specs);
   printf("core: GC%x rev 0x%x halti %d, compute_only %d, temps %u, constants %u, oneconst limit %s\n",
          info.model, info.revision, info.halti, etna_core_has_feature(&info, ETNA_FEATURE_COMPUTE_ONLY),
          info.gpu.max_registers, info.gpu.num_constants, specs.has_no_oneconst_limit ? "no" : "yes");

   glsl_type_singleton_init_or_ref();

   struct etna_compiler *compiler = etna_compiler_create("offline GC8000 rev 7120", &info);
   if (!compiler) {
      fprintf(stderr, "compiler creation failed\n");
      return 1;
   }

   const nir_shader_compiler_options *opts = etna_compiler_get_options(compiler);
   nir_shader *nir;
   if (!strcmp(kernel, "copy"))
      nir = cs_build_copy(opts);
   else if (!strcmp(kernel, "fma"))
      nir = cs_build_fma(opts, fma_iters);
   else if (!strcmp(kernel, "ids"))
      nir = cs_build_ids(opts);
   else if (!strcmp(kernel, "reverse"))
      nir = cs_build_reverse(opts, true);
   else if (!strcmp(kernel, "reverse-nobarrier"))
      nir = cs_build_reverse(opts, false);
   else if (!strcmp(kernel, "reduce"))
      nir = cs_build_reduce(opts, wg);
   else if (!strcmp(kernel, "ids2d"))
      nir = cs_build_ids2d(opts);
   else if (!strcmp(kernel, "t4x4"))
      nir = cs_build_t4x4(opts, 0);
   else if (!strcmp(kernel, "fops"))
      nir = cs_build_fops(opts);
   else if (!strcmp(kernel, "softmax"))
      nir = etna_ml_shader_build_softmax(opts, len, 1, factor, wg);
   else if (!strncmp(kernel, "regp", 4))
      nir = cs_build_regp(opts, atoi(kernel + 4));
   else if (!strcmp(kernel, "ids_gl"))
      nir = cs_build_ids_gl(opts);
   else if (!strcmp(kernel, "t4x4b"))
      nir = cs_build_t4x4(opts, 1);
   else if (!strcmp(kernel, "t4x4copy"))
      nir = cs_build_t4x4(opts, 2);
   else if (!strcmp(kernel, "tgather"))
      nir = cs_build_tgather(opts);
   else {
      fprintf(stderr, "unknown kernel %s\n", kernel);
      return 2;
   }
   if (dump)
      nir_print_shader(nir, stdout);

   struct etna_shader shader = {
      .id = 1,
      .nir = nir,
      .info = &info,
      .specs = &specs,
      .compiler = compiler,
   };
   util_queue_fence_init(&shader.ready);

   struct etna_shader_key key = {};
   struct etna_shader_variant *v = etna_shader_variant(&shader, &key, NULL, false);
   if (!v) {
      fprintf(stderr, "compile FAILED\n");
      return 1;
   }

   printf("kernel: %s, %u instructions, num_temps %u (driver programs %u), uniforms %u, shared %u bytes per "
          "workgroup, barriers %s\n", kernel, v->code_size / 4, v->num_temps, v->num_temps + 1, v->uniforms.count,
          v->shared_size, v->has_barriers ? "yes" : "no");
   for (unsigned i = 0; i < 3; i++)
      printf("  input t%u: %s\n", i,
             v->workitem_funcs[i] ? nir_intrinsic_infos[v->workitem_funcs[i]].name : "-");
   for (unsigned i = 0; i < v->uniforms.count; i++) {
      uint32_t d = v->uniforms.data[i];
      printf("  u%u.%c: %-11s", i / 4, "xyzw"[i % 4], contents_name(v->uniforms.contents[i]));
      if (v->uniforms.contents[i] == ETNA_UNIFORM_KERNEL_ARG)
         printf(" offset %u size %u\n", ETNA_KERNEL_ARG_OFFSET(d), ETNA_KERNEL_ARG_SIZE(d));
      else
         printf(" 0x%08x\n", d);
   }

   if (out) {
      FILE *f = fopen(out, "wb");
      if (!f || fwrite(v->code, 4, v->code_size, f) != v->code_size) {
         perror(out);
         return 1;
      }
      fclose(f);
      printf("wrote %s\n", out);
   }

   return 0;
}
