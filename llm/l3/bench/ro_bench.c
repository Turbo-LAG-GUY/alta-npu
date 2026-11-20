/* ro_bench.c (2026-11-20, CPU only, no device): the user-space part of reading an int8 NPU output back.
 * src = the output buffer (made cold before every run, as after the kernel's invalidate), dst = the caller's buffer.
 * Variants: Mesa's (memcpy + byte loop out[k] -= 128, built -O2 like mesa-llm), memcpy only, byte loop only,
 * memcpy + 64-bit XOR (k -= 128 == k ^ 0x80 per byte), and one fused pass (64-bit load, XOR, store).
 * Usage: ro_bench SIZE CPU [RUNS] */
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
static int cmp(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static uint8_t *evict;
static void cold(void) { for (size_t i = 0; i < (8u << 20); i += 64) evict[i]++; }   /* 8 MB > L2 of both clusters */

__attribute__((noinline)) static void v_mesa(uint8_t *out, const uint8_t *src, unsigned size)
{ memcpy(out, src, size); for (unsigned k = 0; k < size; k++) out[k] -= 128; }
__attribute__((noinline)) static void v_memcpy(uint8_t *out, const uint8_t *src, unsigned size) { memcpy(out, src, size); }
__attribute__((noinline)) static void v_loop(uint8_t *out, const uint8_t *src, unsigned size)
{ (void)src; for (unsigned k = 0; k < size; k++) out[k] -= 128; }
__attribute__((noinline)) static void v_xor64(uint8_t *out, const uint8_t *src, unsigned size)
{
   memcpy(out, src, size);
   unsigned k = 0;
   for (; k + 8 <= size; k += 8) { uint64_t w; memcpy(&w, out + k, 8); w ^= 0x8080808080808080ull; memcpy(out + k, &w, 8); }
   for (; k < size; k++) out[k] ^= 0x80;
}
__attribute__((noinline)) static void v_fused(uint8_t *out, const uint8_t *src, unsigned size)
{
   unsigned k = 0;
   for (; k + 16 <= size; k += 16) {
      uint64_t a, b;
      memcpy(&a, src + k, 8); memcpy(&b, src + k + 8, 8);
      a ^= 0x8080808080808080ull; b ^= 0x8080808080808080ull;
      memcpy(out + k, &a, 8); memcpy(out + k + 8, &b, 8);
   }
   for (; k < size; k++) out[k] = src[k] ^ 0x80;
}

/* the helper as it would go into Mesa: one 64-bit word per iteration */
__attribute__((noinline)) static void v_fused8(uint8_t *out, const uint8_t *src, unsigned size)
{
   unsigned k = 0;
   for (; k + 8 <= size; k += 8) { uint64_t v; memcpy(&v, src + k, 8); v ^= 0x8080808080808080ull; memcpy(out + k, &v, 8); }
   for (; k < size; k++) out[k] = src[k] ^ 0x80;
}

int main(int argc, char **argv)
{
   unsigned size = atoi(argv[1]), cpu = atoi(argv[2]), runs = argc > 3 ? atoi(argv[3]) : 41;
   cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpu, &s); sched_setaffinity(0, sizeof(s), &s);
   uint8_t *src = aligned_alloc(4096, size + 4096), *out = aligned_alloc(4096, size + 4096), *ref = malloc(size);
   evict = malloc(8u << 20);
   for (unsigned i = 0; i < size; i++) src[i] = (uint8_t)(i * 2654435761u >> 13);
   for (unsigned i = 0; i < size; i++) ref[i] = src[i] - 128;
   struct { const char *name; void (*f)(uint8_t *, const uint8_t *, unsigned); int check; } v[] = {
      {"mesa (memcpy + byte loop)", v_mesa, 1}, {"memcpy only", v_memcpy, 0}, {"byte loop only (hot)", v_loop, 0},
      {"memcpy + 64-bit xor", v_xor64, 1}, {"fused 64-bit xor", v_fused, 1}, {"fused, one word per step", v_fused8, 1}};
   double *t = malloc(runs * sizeof(double));
   printf("size %u cpu %u\n", size, cpu);
   for (unsigned j = 0; j < sizeof(v) / sizeof(v[0]); j++) {
      for (unsigned r = 0; r < runs; r++) {
         if (v[j].f == v_loop) memcpy(out, src, size); else cold();
         double a = now(); v[j].f(out, src, size); t[r] = now() - a;
      }
      qsort(t, runs, sizeof(double), cmp);
      int ok = !v[j].check || !memcmp(out, ref, size);
      printf("  %-28s %8.4f ms  %6.2f GB/s  %s\n", v[j].name, t[runs / 2], size / 1e6 / t[runs / 2], ok ? "ok" : "WRONG");
   }
   return 0;
}
