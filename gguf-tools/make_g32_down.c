#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
enum { NE = 384, ROWS = 5120, NBLK = 9, NG = 72, QB = 84, RB = NBLK * QB };
static const uint64_t EB = (uint64_t)ROWS * RB, MB = (uint64_t)NG * ROWS * 8 + (uint64_t)NG * ROWS + (uint64_t)NBLK * ROWS * 4, HDR = 16384;
static uint64_t g_off[40]; static int g_in = -1, g_out = -1; static int g_layer; static int g_next; static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static void g32sim(uint8_t *blk) {   /* Requantize one Q2_K block into eight shared-scale groups. */
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
static void encode(const uint8_t *src, uint8_t *dst) {   /* one expert: q2_K rows -> column-major G32 planes */
    uint32_t *Q = (uint32_t *)dst; uint8_t *S = dst + (uint64_t)NG * ROWS * 8; uint16_t *DD = (uint16_t *)(dst + (uint64_t)NG * ROWS * 9);
    for (int r = 0; r < ROWS; r++) for (int ib = 0; ib < NBLK; ib++) {
        uint8_t blk[QB]; memcpy(blk, src + (uint64_t)r * RB + (uint64_t)ib * QB, QB); g32sim(blk);
        memcpy(&DD[((uint64_t)ib * ROWS + r) * 2], blk + 80, 2); memcpy(&DD[((uint64_t)ib * ROWS + r) * 2 + 1], blk + 82, 2);
        for (int p = 0; p < 8; p++) {
            const int sh = 2 * (p & 3), g = ib * 8 + p; const uint8_t *q = blk + 16 + 32 * (p >> 2); uint32_t w0 = 0, w1 = 0;
            for (int i = 0; i < 32; i++) { const int c = (q[i] >> sh) & 3; w0 |= (uint32_t)(c & 1) << i; w1 |= (uint32_t)(c >> 1) << i; }
            Q[((uint64_t)g * ROWS + r) * 2] = w0; Q[((uint64_t)g * ROWS + r) * 2 + 1] = w1; S[(uint64_t)g * ROWS + r] = blk[2 * p];
        }
    }
}
static void *worker(void *arg) {
    (void)arg; uint8_t *src = malloc(EB), *dst = malloc(MB);
    if (!src || !dst) { fprintf(stderr,"allocation failed\n"); exit(1); }
    for (;;) {
        pthread_mutex_lock(&g_mu); const int e = g_next < NE ? g_next++ : -1; pthread_mutex_unlock(&g_mu);
        if (e < 0) break;
        uint64_t got = 0; const uint64_t so = g_off[g_layer] + (uint64_t)e * EB;
        while (got < EB) { const ssize_t k = pread(g_in, src + got, EB - got, (off_t)(so + got)); if (k <= 0) { perror("pread"); exit(1); } got += (uint64_t)k; }
        encode(src, dst);
        uint64_t put = 0; const uint64_t dof = HDR + ((uint64_t)g_layer * NE + (uint64_t)e) * MB;
        while (put < MB) { const ssize_t k = pwrite(g_out, dst + put, MB - put, (off_t)(dof + put)); if (k <= 0) { perror("pwrite"); exit(1); } put += (uint64_t)k; }
    }
    free(src); free(dst); return NULL;
}
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: make_g32_down <model.gguf> <offsets.txt> <out.bin> [threads]\n"); return 2; }
    const int nt = argc > 4 ? atoi(argv[4]) : 8;
    FILE *f = fopen(argv[2], "r"); if (!f) { perror("offsets"); return 1; }
    for (int L = 0; L < 40; L++) if (fscanf(f, "%llu", (unsigned long long *)&g_off[L]) != 1) { fprintf(stderr, "offsets.txt: need 40\n"); return 1; }
    fclose(f);
    g_in = open(argv[1], O_RDONLY); g_out = open(argv[3], O_RDWR | O_CREAT | O_EXCL, 0644);
    if (g_in < 0 || g_out < 0) { perror("open"); return 1; }
#ifdef F_NOCACHE
    (void)fcntl(g_in, F_NOCACHE, 1);
#endif
    uint8_t h[HDR]; memset(h, 0, sizeof h); memcpy(h, "G32D", 4);
    const uint32_t u32[5] = {1u, 40u, 384u, 5120u, 2304u}; const uint64_t u64[6] = {MB, (uint64_t)NE * MB, HDR, 0, (uint64_t)NG * ROWS * 8, (uint64_t)NG * ROWS * 9};
    memcpy(h + 4, u32, sizeof u32); memcpy(h + 24, u64, sizeof u64);
    if (pwrite(g_out, h, sizeof h, 0) != (ssize_t)sizeof h) { perror("header"); return 1; }
    const time_t t0 = time(NULL);
    for (g_layer = 0; g_layer < 40; g_layer++) {
        g_next = 0; pthread_t th[32]; const int n = nt < 1 ? 1 : nt > 32 ? 32 : nt;
        for (int i = 0; i < n; i++) if (pthread_create(&th[i], NULL, worker, NULL)) { fprintf(stderr,"pthread_create failed\n"); exit(1); }
        for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
        printf("layer %d done (%ld s)\n", g_layer, (long)(time(NULL) - t0)); fflush(stdout);
    }
    fsync(g_out); close(g_out); close(g_in);
    printf("done.\n"); return 0;
}
