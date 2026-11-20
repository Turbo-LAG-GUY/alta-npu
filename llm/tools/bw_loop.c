// bw_loop.c (2026-11-19, CPU only): memory read bandwidth on the four Cortex-A73 cores (cpu 2-5) for N seconds.
// One line per pass over a 256 MiB buffer: "<monotonic s> <GB/s>". For LLM step L1c: does the CPU keep its
// bandwidth while the NPU streams weights (shared DRAM), and does the NPU keep its speed?
// Build: gcc -O3 -o bw_loop bw_loop.c -lpthread      Usage: bw_loop SECONDS
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define SZ (256u << 20)
#define NT 4
static uint8_t *buf;
static volatile uint64_t sink[NT];
static pthread_barrier_t start_b, end_b;
static volatile int stop;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void *rd(void *a) {
   long i = (long)a; cpu_set_t s; CPU_ZERO(&s); CPU_SET(2 + i, &s); sched_setaffinity(0, sizeof s, &s);
   size_t chunk = SZ / NT; const uint8_t *p = buf + i * chunk;
   for (;;) {
      pthread_barrier_wait(&start_b);
      if (stop) return 0;
      uint64x2_t acc = vdupq_n_u64(0);
      for (size_t o = 0; o < chunk; o += 64) {
         uint8x16x4_t v = vld1q_u8_x4(p + o);
         acc = vaddq_u64(acc, vreinterpretq_u64_u8(veorq_u8(veorq_u8(v.val[0], v.val[1]), veorq_u8(v.val[2], v.val[3]))));
      }
      sink[i] = vgetq_lane_u64(acc, 0);
      pthread_barrier_wait(&end_b);
   }
}
int main(int argc, char **argv) {
   double secs = argc > 1 ? atof(argv[1]) : 10;
   buf = aligned_alloc(64, SZ); memset(buf, 1, SZ);
   pthread_barrier_init(&start_b, 0, NT + 1); pthread_barrier_init(&end_b, 0, NT + 1);
   pthread_t t[NT];
   for (long i = 0; i < NT; i++) pthread_create(&t[i], 0, rd, (void *)i);
   double t_end = now() + secs;
   while (now() < t_end) {
      double t0 = now();
      pthread_barrier_wait(&start_b); pthread_barrier_wait(&end_b);
      double t1 = now();
      printf("%.3f %.2f\n", t1, SZ / (t1 - t0) / 1e9); fflush(stdout);
   }
   stop = 1; pthread_barrier_wait(&start_b);
   for (int i = 0; i < NT; i++) pthread_join(t[i], 0);
   return 0;
}
