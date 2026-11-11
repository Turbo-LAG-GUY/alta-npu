/*
 * teflon-cache-trim: applies the NPU weight cache's size limit while no NPU program
 * is running, so the cache still shrinks if the disk fills up between runs. It uses
 * the same policy code as Teflon. Run by the teflon-cache-trim.timer user unit.
 *
 * Build: gcc -O2 -Wall -D_GNU_SOURCE \
 *          -I ~/Documents/AI_Projects/mesa-axi-sram/src/gallium/drivers/etnaviv \
 *          -o ~/.local/bin/teflon-cache-trim teflon-cache-trim.c
 * Use:   teflon-cache-trim            trim, and report only if something was evicted
 *        teflon-cache-trim --status   trim, and always report
 */

#include "etnaviv_ml_disk_cache.h"

int
main(int argc, char **argv)
{
   char dir[PATH_MAX];
   struct npu_cache_stats stats = {0};
   bool status = argc > 1 && strcmp(argv[1], "--status") == 0;

   if (!npu_cache_default_dir(dir, sizeof(dir))) {
      fprintf(stderr, "teflon-cache-trim: no cache directory (HOME is not set)\n");
      return 1;
   }

   if (access(dir, F_OK) != 0) {
      if (status)
         printf("%s: no cache yet\n", dir);
      return 0;
   }

   if (!npu_cache_trim(dir, 0, &stats)) {
      fprintf(stderr, "teflon-cache-trim: can't measure the disk holding %s\n", dir);
      return 1;
   }

   if (status || stats.evicted)
      printf("%s: holding %.1f MiB in %u entries, limit %.0f MiB, %.1f GiB free, evicted %u\n",
             dir, stats.used / 1048576.0, stats.entries, stats.budget / 1048576.0,
             stats.avail / 1073741824.0, stats.evicted);
   if (stats.low_space)
      fprintf(stderr, "teflon-cache-trim: WARNING: disk space is low: %.1f GiB free, under 10 GiB. "
                      "The NPU weight cache won't grow until space is freed\n",
              stats.avail / 1073741824.0);

   return 0;
}
