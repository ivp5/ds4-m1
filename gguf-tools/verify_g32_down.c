#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
enum { NE = 384, ROWS = 5120, NBLK = 9, NG = 72, QB = 84, RB = NBLK * QB, MID = 2304 };
static const uint64_t EB = (uint64_t)ROWS * RB, MB = 3502080ull, HDR = 16384, QOFF = 0, SOFF = 2949120ull, DOFF = 3317760ull;
static void g32sim(uint8_t *blk) {   /* Reference requantization. */
    uint8_t *sc = blk, *qs = blk + 16; _Float16 dh, mh; memcpy(&dh, blk + 80, 2); memcpy(&mh, blk + 82, 2);
    const float d = (float)dh, dmin = (float)mh;
    for (int p = 0; p < 8; p++) {
        const int n = p >> 2, sh = 2 * (p & 3); uint8_t *q = qs + 32 * n;
        float v[32]; int is0 = 2 * p, is1 = 2 * p + 1;
        for (int l = 0; l < 16; l++) { v[l] = d * (float)(sc[is0] & 15) * (float)((q[l] >> sh) & 3) - dmin * (float)(sc[is0] >> 4);
                                       v[16 + l] = d * (float)(sc[is1] & 15) * (float)((q[l + 16] >> sh) & 3) - dmin * (float)(sc[is1] >> 4); }
        float lo = v[0], hi = v[0]; for (int i = 1; i < 32; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
        int cs[3], cm[3]; cs[0] = sc[is0] & 15; cm[0] = sc[is0] >> 4; cs[1] = sc[is1] & 15; cm[1] = sc[is1] >> 4;
        { int mm = dmin > 0.0f ? (int)lrintf((lo < 0.0f ? -lo : 0.0f) / dmin) : 0; if (mm > 15) mm = 15; if (mm < 0) mm = 0;
          int ss = d > 0.0f ? (int)lrintf((hi + dmin * (float)mm) / (3.0f * d)) : 0; if (ss > 15) ss = 15; if (ss < 0) ss = 0; cs[2] = ss; cm[2] = mm; }
        int best = 0; float be = 3.4e38f; uint8_t bq[32];
        for (int c = 0; c < 3; c++) {
            const float dl = d * (float)cs[c], ml = dmin * (float)cm[c]; float err = 0.0f; uint8_t qq[32];
            for (int i = 0; i < 32; i++) { int k = dl > 0.0f ? (int)lrintf((v[i] + ml) / dl) : 0; if (k < 0) k = 0; if (k > 3) k = 3; qq[i] = (uint8_t)k;
                                           const float r = dl * (float)k - ml - v[i]; err += r * r; }
            if (err < be) { be = err; best = c; memcpy(bq, qq, 32); }
        }
        sc[is0] = sc[is1] = (uint8_t)(cs[best] | (cm[best] << 4));
        const uint8_t mk = (uint8_t)~(3u << sh);
        for (int l = 0; l < 16; l++) { q[l] = (uint8_t)((q[l] & mk) | (bq[l] << sh)); q[l + 16] = (uint8_t)((q[l + 16] & mk) | (bq[16 + l] << sh)); }
    }
}
static float h2f(uint16_t h) { _Float16 v; memcpy(&v, &h, 2); return (float)v; }
static long check(const uint8_t *src, const uint8_t *side, int flip) {   /* every 37th row of one expert; flip = 1: the canary */
    long bad = 0; const uint32_t *Q = (const uint32_t *)(side + QOFF); const uint8_t *S = side + SOFF; const uint16_t *DD = (const uint16_t *)(side + DOFF);
    for (int r = 0; r < ROWS; r += 37) for (int ib = 0; ib < NBLK; ib++) {
        uint8_t blk[QB]; memcpy(blk, src + (uint64_t)r * RB + (uint64_t)ib * QB, QB); g32sim(blk);
        uint16_t dh, mh; memcpy(&dh, blk + 80, 2); memcpy(&mh, blk + 82, 2); const float d = h2f(dh), dm = h2f(mh);
        const float d2 = h2f(DD[((uint64_t)ib * ROWS + r) * 2]), dm2 = h2f(DD[((uint64_t)ib * ROWS + r) * 2 + 1]);
        for (int k = 0; k < 256; k++) {
            const int n = k >> 7, j = (k >> 5) & 3, l = k & 31, is = 8 * n + 2 * j + (l >> 4);   /* q2_K: input 128 n + 32 j + l */
            const int c = (blk[16 + 32 * n + l] >> (2 * j)) & 3;
            const float vq = d * (float)(blk[is] & 15) * (float)c - dm * (float)(blk[is] >> 4);
            const int g = ib * 8 + (k >> 5), i = k & 31;
            uint32_t w0 = Q[((uint64_t)g * ROWS + r) * 2], w1 = Q[((uint64_t)g * ROWS + r) * 2 + 1];
            if (flip && r == 0 && ib == 0 && k == 0) w0 ^= 1u;
            const int c2 = (int)((w0 >> i) & 1u) | (int)(((w1 >> i) & 1u) << 1); const uint8_t s = S[(uint64_t)g * ROWS + r];
            const float vs = d2 * (float)(s & 15) * (float)c2 - dm2 * (float)(s >> 4);
            if (vs != vq) bad++;
        }
    }
    return bad;
}
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: verify_g32_down <model.gguf> <offsets.txt> <g32_down.bin>\n"); return 2; }
    uint64_t off[40]; FILE *f = fopen(argv[2], "r"); if (!f) { perror("offsets"); return 1; }
    for (int L = 0; L < 40; L++) if (fscanf(f, "%llu", (unsigned long long *)&off[L]) != 1) return 1;
    fclose(f);
    const int fm = open(argv[1], O_RDONLY), fs = open(argv[3], O_RDONLY); if (fm < 0 || fs < 0) { perror("open"); return 1; }
    char h[8]; if (pread(fs, h, 4, 0) != 4 || memcmp(h, "G32D", 4)) { fprintf(stderr, "not a G32D file\n"); return 1; }
    uint8_t *src = malloc(EB), *side = malloc(MB); const int ex[6] = {0, 77, 155, 233, 311, 383}; long total = 0, canary = -1;
    for (int L = 0; L < 40; L++) {
        long bad = 0;
        for (int x = 0; x < 6; x++) {
            const int e = ex[x];
            if (pread(fm, src, EB, (off_t)(off[L] + (uint64_t)e * EB)) != (ssize_t)EB) { fprintf(stderr, "short model read\n"); return 1; }
            if (pread(fs, side, MB, (off_t)(HDR + ((uint64_t)L * NE + e) * MB)) != (ssize_t)MB) { fprintf(stderr, "short side read L %d e %d\n", L, e); return 1; }
            if (canary < 0) canary = check(src, side, 1);
            bad += check(src, side, 0);
        }
        total += bad; printf("layer %2d: %ld mismatches\n", L, bad);
    }
    printf("CANARY (one plane bit flipped): %ld mismatches (must be > 0); TOTAL mismatches %ld\n", canary, total);
    free(src); free(side); close(fm); close(fs);
    return total == 0 && canary > 0 ? 0 : 1;
}
