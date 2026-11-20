#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void
flip_sign_bits(uint8_t *dst, const uint8_t *src, unsigned size)
{
   unsigned k = 0;

   for (; k + 8 <= size; k += 8) {
      uint64_t v;
      memcpy(&v, src + k, sizeof(v));
      v ^= 0x8080808080808080ull;
      memcpy(dst + k, &v, sizeof(v));
   }

   for (; k < size; k++)
      dst[k] = src[k] ^ 0x80;
}
int main(void)
{
   unsigned sizes[] = {0, 1, 7, 8, 9, 15, 16, 17, 31, 33, 1536, 3072, 6656, 6659, 196608, 786432, 786437};
   uint8_t *s = malloc(800000), *a = malloc(800016), *b = malloc(800016), *c = malloc(800016);
   int bad = 0;
   srand(1);
   for (unsigned j = 0; j < sizeof(sizes) / sizeof(sizes[0]); j++)
      for (unsigned off = 0; off < 3; off++) {                 /* unaligned source / destination too */
         unsigned n = sizes[j];
         for (unsigned i = 0; i < n; i++) s[i] = rand();
         for (unsigned i = 0; i < n; i++) a[off + i] = s[i] - 128;              /* old output loop */
         memset(b, 0x55, n + 16); flip_sign_bits(b + off, s, n);                 /* new, cached path */
         memset(c, 0x55, n + 16); memcpy(c + off, s, n); flip_sign_bits(c + off, c + off, n);  /* new, WC path */
         if (memcmp(a + off, b + off, n) || memcmp(a + off, c + off, n) || b[off + n] != 0x55 || c[off + n] != 0x55)
            { printf("MISMATCH size %u off %u\n", n, off); bad++; }
         for (unsigned i = 0; i < n; i++) a[i] = s[i] + 128;                      /* old input loop */
         flip_sign_bits(b, s, n);
         if (memcmp(a, b, n)) { printf("INPUT MISMATCH size %u\n", n); bad++; }
      }
   printf("%s: %u sizes x 3 offsets, output (cached + WC paths) and input conversions vs the old loops\n", bad ? "FAIL" : "PASS", (unsigned)(sizeof(sizes) / sizeof(sizes[0])));
   return bad;
}
