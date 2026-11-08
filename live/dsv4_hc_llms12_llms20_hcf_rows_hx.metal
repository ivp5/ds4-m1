struct ds4_metal_args_dsv4_hc_split_sinkhorn {
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_weighted_sum {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
};


struct ds4_metal_args_dsv4_output_hc_weights4 {
    float post_scale;
    float eps;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    float    eps;
};

struct ds4_metal_args_dsv4_hc_split_weighted_sum_norm {
    int64_t  n_embd;
    int32_t  n_hc;
    int32_t  sinkhorn_iters;
    int64_t  n_rows;
    int64_t  mix_hc;
    uint64_t nb_mix1;
    uint64_t nb_split1;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb_norm1;
    float    eps;
    float    norm_eps;
};

struct ds4_metal_args_dsv4_hc_expand {
    int64_t  n_embd;
    int64_t  n_hc;
    int64_t  n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_add0;
    uint64_t nb_add1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    int32_t  has_add;
};

// Numerically stable sigmoid for the standalone split/sinkhorn path. The naive
// form 1/(1+exp(-z)) overflows for large negative z (exp(-z) blows up);
// replacing it with the 0.5*(tanh(z/2)+1) identity keeps the value bounded in
// [0, 1] across the entire float range. Gated by DS4_METAL_HC_STABLE so we can
// A/B vs the historical form on M5 Max where the faster ALU is more likely to
// push HC mixer inputs into the unstable regime.
//
// Do not automatically use these helpers in the fused HC decode kernels below:
// routing the fused vector sites through the tanh form produced non-finite
// logits on M5 Max, while the historical inline exp form remains finite and is
// the decode throughput baseline.
#ifdef DS4_METAL_HC_STABLE
static inline float  ds4_hc_sigmoid(float  z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 0.5f * tanh(0.5f * z) + 0.5f; }
// 2 * sigmoid(z) == 1 + tanh(z/2).
static inline float  ds4_hc_twice_sigmoid(float  z) { return 1.0f + tanh(0.5f * z); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 1.0f + tanh(0.5f * z); }
#else
static inline float  ds4_hc_sigmoid(float  z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_sigmoid(float4 z)  { return 1.0f / (1.0f + exp(-z)); }
static inline float  ds4_hc_twice_sigmoid(float  z) { return 2.0f / (1.0f + exp(-z)); }
static inline float4 ds4_hc_twice_sigmoid(float4 z) { return 2.0f / (1.0f + exp(-z)); }
#endif

// Splits an HC mixer row into pre weights, post gates, and the HC-to-HC
// combination matrix. The 4-channel path is specialized because DS4 Flash uses
// HC=4 in normal inference, while the scalar fallback keeps diagnostics usable.
kernel void kernel_dsv4_hc_split_sinkhorn(
        constant ds4_metal_args_dsv4_hc_split_sinkhorn & args,
        device  const float * mixes,
        device  const float * scale,
        device  const float * base,
        device        float * dst,
        uint tid [[thread_position_in_grid]]) {
    if ((int64_t) tid >= args.n_rows) {
        return;
    }

    constexpr int HC_MAX = 16;
    const int HC = args.n_hc;
    if (HC <= 0 || HC > HC_MAX) {
        return;
    }

    device const float * mix = mixes + ((int64_t) tid)*args.mix_hc;
    device       float * out = dst    + ((int64_t) tid)*args.mix_hc;

    const float epsv       = args.eps;
    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    if (HC == 4) {
        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        *((device float4 *) out) = ds4_hc_sigmoid(pre_z) + epsv;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = ds4_hc_twice_sigmoid(post_z);

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
        return;
    }

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = ds4_hc_sigmoid(z) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = ds4_hc_twice_sigmoid(z);
    }

    float c[HC_MAX*HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const int off = 2*HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = max(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            const float v = exp(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc*HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc*HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc*HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc*HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc*HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC*HC; ++i) {
        out[2*HC + i] = c[i];
    }
}

// Decode-side fusion of HC split and pre-weighted HC reduction. One threadgroup
// handles one token row: lane 0 computes the HC=4 mixer split once, stores the
// post/comb data for the following HC expand, and all lanes reuse the pre
// weights from threadgroup memory to produce the embedding row.
kernel void kernel_dsv4_hc_split_weighted_sum(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        threadgroup   float * pre_shmem [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint ntg [[threads_per_threadgroup]]) {
    if ((int64_t) row >= args.n_rows || args.n_hc != 4) {
        return;
    }

    device const float * mix = (device const float *) (mixes + (uint64_t)row*args.nb_mix1);
    device       float * out = (device       float *) (split + (uint64_t)row*args.nb_split1);

    if (tid == 0) {
        const float epsv       = args.eps;
        const float pre_scale  = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *) mix) * pre_scale +
            *((device const float4 *) base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *) out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *) (mix  + 4)) * post_scale +
            *((device const float4 *) (base + 4));
        *((device float4 *) (out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *) (mix  +  8)) * comb_scale +
            *((device const float4 *) (base +  8));
        float4 r1 =
            *((device const float4 *) (mix  + 12)) * comb_scale +
            *((device const float4 *) (base + 12));
        float4 r2 =
            *((device const float4 *) (mix  + 16)) * comb_scale +
            *((device const float4 *) (base + 16));
        float4 r3 =
            *((device const float4 *) (mix  + 20)) * comb_scale +
            *((device const float4 *) (base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *) (out +  8)) = r0;
        *((device float4 *) (out + 12)) = r1;
        *((device float4 *) (out + 16)) = r2;
        *((device float4 *) (out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int64_t d = tid; d < args.n_embd; d += ntg) {
        float acc = 0.0f;
        acc += *((device const float *) (x + d*args.nb_x0 + 0*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[0];
        acc += *((device const float *) (x + d*args.nb_x0 + 1*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[1];
        acc += *((device const float *) (x + d*args.nb_x0 + 2*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[2];
        acc += *((device const float *) (x + d*args.nb_x0 + 3*args.nb_x1 + (uint64_t)row*args.nb_x2)) * pre_shmem[3];
        *((device float *) (dst + d*args.nb0 + (uint64_t)row*args.nb1)) = acc;
    }
}

// Decode HC-pre plus the following RMSNorm.  DS4 uses HC=4 here.  The normal
// release path computes HC coefficients, collapses four residual streams into
// the model row, then immediately launches a weighted RMSNorm over the row.
// This kernel keeps the HC split math identical to
// kernel_dsv4_hc_split_weighted_sum, stores the HC-pre row for diagnostics, and
// reuses the just-collapsed values from threadgroup memory for the RMSNorm
// reduction.
static __attribute__((always_inline)) inline void ds4_hc_comb_weights4_exact(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device volatile const float *mix,
        device const float *scale,
        device const float *base,
        device float *out) {
    const float epsv = args.eps;
    const float comb_scale = scale[2];

    float4 r0 =
        *((device volatile const float4 *)(mix + 8)) * comb_scale +
        *((device const float4 *)(base + 8));
    float4 r1 =
        *((device volatile const float4 *)(mix + 12)) * comb_scale +
        *((device const float4 *)(base + 12));
    float4 r2 =
        *((device volatile const float4 *)(mix + 16)) * comb_scale +
        *((device const float4 *)(base + 16));
    float4 r3 =
        *((device volatile const float4 *)(mix + 20)) * comb_scale +
        *((device const float4 *)(base + 20));

    const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
    const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
    const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
    const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

    r0 = exp(r0 - m0);
    r1 = exp(r1 - m1);
    r2 = exp(r2 - m2);
    r3 = exp(r3 - m3);

    r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
    r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
    r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
    r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

    float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
    r0 *= col_inv;
    r1 *= col_inv;
    r2 *= col_inv;
    r3 *= col_inv;

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
        r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
        r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
        r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

        col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;
    }

    *((device float4 *)(out + 8)) = r0;
    *((device float4 *)(out + 12)) = r1;
    *((device float4 *)(out + 16)) = r2;
    *((device float4 *)(out + 20)) = r3;
}

kernel void kernel_dsv4_hc_split_weighted_sum_norm4(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & args,
        device  const char  * mixes,
        device  const float * scale,
        device  const float * base,
        device  const char  * x,
        device        char  * split,
        device        char  * dst,
        device  const char  * norm_weight,
        device        char  * norm_dst,
        threadgroup   float * shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    if ((int64_t)row >= args.n_rows || args.n_hc != 4 || (args.n_embd & 3) != 0) {
        return;
    }

    const uint n_embd = uint(args.n_embd);
    const uint n4 = n_embd >> 2;
    threadgroup float4 *row_shmem = (threadgroup float4 *)shared;
    threadgroup float *pre_shmem = shared + n_embd;
    threadgroup float *sum_shmem = pre_shmem + 4;

    device const float *mix = (device const float *)(mixes + (uint64_t)row * args.nb_mix1);
    device float *out = (device float *)(split + (uint64_t)row * args.nb_split1);

    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    if (tid == 0) {
        const float epsv = args.eps;
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float4 pre_z =
            *((device const float4 *)mix) * pre_scale +
            *((device const float4 *)base);
        const float4 pre = 1.0f / (1.0f + exp(-pre_z)) + epsv;
        *((device float4 *)out) = pre;
        pre_shmem[0] = pre.x;
        pre_shmem[1] = pre.y;
        pre_shmem[2] = pre.z;
        pre_shmem[3] = pre.w;

        const float4 post_z =
            *((device const float4 *)(mix + 4)) * post_scale +
            *((device const float4 *)(base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));

        float4 r0 =
            *((device const float4 *)(mix + 8)) * comb_scale +
            *((device const float4 *)(base + 8));
        float4 r1 =
            *((device const float4 *)(mix + 12)) * comb_scale +
            *((device const float4 *)(base + 12));
        float4 r2 =
            *((device const float4 *)(mix + 16)) * comb_scale +
            *((device const float4 *)(base + 16));
        float4 r3 =
            *((device const float4 *)(mix + 20)) * comb_scale +
            *((device const float4 *)(base + 20));

        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));

        r0 = exp(r0 - m0);
        r1 = exp(r1 - m1);
        r2 = exp(r2 - m2);
        r3 = exp(r3 - m3);

        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;

        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv;
        r1 *= col_inv;
        r2 *= col_inv;
        r3 *= col_inv;

        for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);

            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv;
            r1 *= col_inv;
            r2 *= col_inv;
            r3 *= col_inv;
        }

        *((device float4 *)(out + 8)) = r0;
        *((device float4 *)(out + 12)) = r1;
        *((device float4 *)(out + 16)) = r2;
        *((device float4 *)(out + 20)) = r3;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        device const float4 *x0 = (device const float4 *)(x + 0 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x1 = (device const float4 *)(x + 1 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x2 = (device const float4 *)(x + 2 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        device const float4 *x3 = (device const float4 *)(x + 3 * args.nb_x1 + (uint64_t)row * args.nb_x2);
        // Preserve the standalone HC collapse's explicit accumulation order.
        float4 v = 0.0f;
        v += x0[i] * pre_shmem[0];
        v += x1[i] * pre_shmem[1];
        v += x2[i] * pre_shmem[2];
        v += x3[i] * pre_shmem[3];
        row_shmem[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);
    // Batched prefill must match kernel_rms_norm_fuse_impl so enabling this
    // fusion does not change its scale by an ULP. Keep the established
    // single-row decode result unchanged; that path historically used rsqrt.
    const float norm_arg = sumf / float(n_embd) + args.norm_eps;
    const float norm_scale = args.n_rows > 1 ? 1.0f / sqrt(norm_arg) : rsqrt(norm_arg);

    device float4 *dst4 = (device float4 *)(dst + (uint64_t)row * args.nb1);
    device const float4 *w4 = (device const float4 *)norm_weight;
    device float4 *norm4 = (device float4 *)(norm_dst + (uint64_t)row * args.nb_norm1);
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = row_shmem[i];
        dst4[i] = v;
        norm4[i] = (v * norm_scale) * w4[i];
    }
}

// Expands an embedding-sized block back into HC channels after attention/FFN.
// The post gate scales the current block, while the Sinkhorn combination matrix
// mixes residual HC channels from the previous state.
kernel void kernel_dsv4_hc_expand(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d      = ((int64_t) gid) % args.n_embd;
    const int64_t tmp    = ((int64_t) gid) / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t      = tmp / args.n_hc;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }
    const float post_v  = *((device const float *) (post      + dst_hc*args.nb_post0 + t*args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((device const float *) (comb     + dst_hc*args.nb_comb0 + src_hc*args.nb_comb1 + t*args.nb_comb2));
        const float res_v  = *((device const float *) (residual + d*args.nb_res0 + src_hc*args.nb_res1 + t*args.nb_res2));
        acc += comb_v * res_v;
    }

    *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
}

// HC=4 specialization of the post/expand step. One thread computes all four
// destination HC streams for one token/dimension, reusing the same block output
// and residual HC values while preserving the per-stream accumulation order.
kernel void kernel_dsv4_hc_expand4(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    if (args.n_hc != 4) {
        return;
    }

    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float block_v = *((device const float *) (block_out + d*args.nb_block0 + t*args.nb_block1));
    if (args.has_add) {
        block_v += *((device const float *) (block_add + d*args.nb_add0 + t*args.nb_add1));
    }

    const float r0 = *((device const float *) (residual + d*args.nb_res0 + 0*args.nb_res1 + t*args.nb_res2));
    const float r1 = *((device const float *) (residual + d*args.nb_res0 + 1*args.nb_res1 + t*args.nb_res2));
    const float r2 = *((device const float *) (residual + d*args.nb_res0 + 2*args.nb_res1 + t*args.nb_res2));
    const float r3 = *((device const float *) (residual + d*args.nb_res0 + 3*args.nb_res1 + t*args.nb_res2));

    for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float acc = block_v * *((device const float *) (post + dst_hc*args.nb_post0 + t*args.nb_post1));

        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 0*args.nb_comb1 + t*args.nb_comb2)) * r0;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 1*args.nb_comb1 + t*args.nb_comb2)) * r1;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 2*args.nb_comb1 + t*args.nb_comb2)) * r2;
        acc += *((device const float *) (comb + dst_hc*args.nb_comb0 + 3*args.nb_comb1 + t*args.nb_comb2)) * r3;

        *((device float *) (dst + d*args.nb0 + dst_hc*args.nb1 + t*args.nb2)) = acc;
    }
}

// Decode-time FFN tail fusion:
//
//     shared_out = shared_mid @ Wshared_down
//     after_ffn_hc = HCPost(routed_out + shared_out, residual_hc, split)
//
// The Q8_0 dot reduction is intentionally copied from the normal matvec shape
// so the shared expert result is bit-identical.  The only specialization is
// that DS4 decode has one token and HC=4, so the thread that finishes each
// shared-down output row can immediately expand it into the four HC streams.
kernel void kernel_dsv4_shared_down_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * shared_mid,
        device        char * shared_out,
        device  const char * routed_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(shared_mid);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float shared_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(shared_out + (uint64_t)d * sizeof(float))) = shared_v;

            float block_v = *((device const float *)(routed_out + (uint64_t)d * hc.nb_block0));
            block_v += shared_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

// Decode-time attention output tail fusion:
//
//     attn_out = attn_low @ Wob
//     after_attn_hc = HCPost(attn_out, residual_hc, split)
//
// This is the no-add sibling of the shared-down/FFN fusion above.  It preserves
// the exact Q8_0 matvec reduction, stores `attn_out` for diagnostics, and then
// writes the four HC streams for the same embedding dimension.
kernel void kernel_dsv4_q8_hc_expand4_q8_0(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * input,
        device        char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(input);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float block_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = block_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc * hc.nb_post0));

                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 0 * hc.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 1 * hc.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 2 * hc.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc * hc.nb_comb0 + 3 * hc.nb_comb1)) * r3;

                *((device float *)(dst + (uint64_t)d * hc.nb0 + dst_hc * hc.nb1)) = acc;
            }
        }
    }
}

kernel void kernel_dsv4_q8_hc_expand4_q8_0_vec_hc(
        constant ds4_metal_args_mul_mv        & mv,
        constant ds4_metal_args_dsv4_hc_expand & hc,
        device  const char * weight,
        device  const char * input,
        device        char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device        char * dst,
        threadgroup   char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    if (hc.n_hc != 4 || hc.n_tokens != 1) {
        return;
    }

    const short NSG = FC_mul_mv_nsg;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NQ = 8;
    constexpr short NR0 = N_R0_Q8_0;

    const int nb = mv.ne00 / QK8_0;
    const int row0 = tgpig.x * NR0;

    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;

    device const float *y = (device const float *)(input);
    device const float *yb = y + ib0 * QK8_0 + il * NQ;

    device const block_q8_0 *ax[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const uint64_t off0 = (uint64_t)(row0 + row) * mv.nb01;
        ax[row] = (device const block_q8_0 *)(weight + off0);
    }

    float sumf[NR0] = { 0.0f };
    float yl[NQ];

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        FOR_UNROLL(short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        FOR_UNROLL(short row = 0; row < NR0; ++row) {
            device const int8_t *qs = ax[row][ib].qs + il * NQ;

            float sumq = 0.0f;
            FOR_UNROLL(short i = 0; i < NQ; ++i) {
                sumq += qs[i] * yl[i];
            }

            sumf[row] += sumq * ax[row][ib].d;
        }

        yb += NSG * NQ * QK8_0;
    }

    threadgroup float *shmem_f32[NR0];
    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *)shmem + NW * row;
        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }
        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    FOR_UNROLL(short row = 0; row < NR0; ++row) {
        const int d = row0 + row;
        if (d >= mv.ne01) {
            continue;
        }

        const float block_v = simd_sum(shmem_f32[row][tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            *((device float *)(block_out + (uint64_t)d * sizeof(float))) = block_v;

            const float r0 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 0 * hc.nb_res1));
            const float r1 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 1 * hc.nb_res1));
            const float r2 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 2 * hc.nb_res1));
            const float r3 = *((device const float *)(residual + (uint64_t)d * hc.nb_res0 + 3 * hc.nb_res1));

            const float4 post4 = *((device const float4 *)post);
            const float4 comb0 = *((device const float4 *)(comb + 0 * hc.nb_comb1));
            const float4 comb1 = *((device const float4 *)(comb + 1 * hc.nb_comb1));
            const float4 comb2 = *((device const float4 *)(comb + 2 * hc.nb_comb1));
            const float4 comb3 = *((device const float4 *)(comb + 3 * hc.nb_comb1));
            float4 acc = block_v * post4;
            acc += comb0 * r0;
            acc += comb1 * r1;
            acc += comb2 * r2;
            acc += comb3 * r3;
            FOR_UNROLL (short dst_hc = 0; dst_hc < 4; ++dst_hc) {
                *((device float *)(dst + (uint64_t)d * hc.nb0 +
                                   (uint64_t)dst_hc * hc.nb1)) = acc[dst_hc];
            }
        }
    }
}

// Reduces HC channels to a normal embedding row with the learned pre weights.
// This is the input adapter before the attention block and before the FFN block.
kernel void kernel_dsv4_hc_weighted_sum(
        constant ds4_metal_args_dsv4_hc_weighted_sum & args,
        device  const char * x,
        device  const char * weights,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n_elem = args.n_embd * args.n_tokens;
    if ((int64_t) gid >= n_elem) {
        return;
    }

    const int64_t d = ((int64_t) gid) % args.n_embd;
    const int64_t t = ((int64_t) gid) / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((device const float *) (x       + d*args.nb_x0 + h*args.nb_x1 + t*args.nb_x2));
        const float wv = *((device const float *) (weights + h*args.nb_w0 + t*args.nb_w1));
        acc += xv * wv;
    }

    *((device float *) (dst + d*args.nb0 + t*args.nb1)) = acc;
}

// The one-row HC=4 output head historically materializes four device-F32
// stages across separate launches. Collapse those launches into one tiny
// two-thread group while preserving the scalar/vector lane mapping and every
// global rounding boundary.
kernel void kernel_dsv4_output_hc_weights4(
        constant ds4_metal_args_dsv4_output_hc_weights4 & args,
        device  const float * pre,
        device  const float * hc_scale,
        device  const float * hc_base,
        device        float * dst,
        ushort tid [[thread_position_in_threadgroup]]) {
    device volatile float *stage = (device volatile float *)dst;

    for (uint i = tid; i < 4; i += 2) {
        stage[i] = pre[i] * hc_scale[0];
    }
    threadgroup_barrier(mem_flags::mem_device);

    for (uint i = tid; i < 4; i += 2) {
        stage[i] = stage[i] + hc_base[i];
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        const float4 x = *((device volatile float4 *)stage);
        *((device volatile float4 *)stage) = 1 / (1 + exp(-x));
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        const float4 x = *((device volatile float4 *)stage);
        *((device volatile float4 *)stage) =
            args.post_scale * x + args.eps;
    }
}


struct ds4_metal_args_hc_norm_mix {
    int32_t n;
    int32_t out_dim;
    float   eps;
};

// Fused unweighted RMSNorm + F16 HC-mix projection for DS4 decode HC-pre.
// The standalone decode path runs kernel_rms_norm_f32_4 over the flattened
// 4*embd HC row (1024 threads, one threadgroup) and then
// kernel_mul_mv_f16_f32_4 (nsg=8, nr0=2) over the normalized row.  Both
// stages are reproduced bit-exactly in one dispatch: every threadgroup
// redundantly recomputes the norm partials with the original 1024-thread
// mapping (each real lane covers one virtual thread of each 256-thread
// slice, preserving every simd_sum tree), and the matvec keeps the original
// per-row accumulation order with y = x*scale computed on the fly, which
// rounds identically to the materialized normalized row.  The host wrapper
// gates this to n == 16384 && out_dim == 24, where the virtual-thread count
// is exactly 1024 and the mv tail loop is empty.
kernel void kernel_dsv4_hc_rms_norm_mix_f16(
        constant ds4_metal_args_hc_norm_mix & args,
        device const char  * x,
        device const char  * weight,
        device       char  * dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 8;   // ds4_gpu_make_plain_mv_dispatch(16384)
    constexpr short NW  = N_SIMDWIDTH;
    constexpr short NR0 = 2;   // plain mv nr0
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;
    constexpr uint  VTHREADS = 1024u;                 // rms norm threads at n == 16384
    constexpr short VSLICES  = VTHREADS/(NSG*NW);     // virtual 256-thread slices

    const uint n  = (uint)args.n;
    const uint n4 = n >> 2;

    device const float4 *x4 = (device const float4 *)x;

    threadgroup float *norm_shmem = (threadgroup float *)shmem;        // NW slots
    threadgroup float *mv_shmem   = (threadgroup float *)shmem + NW;   // NW*NR0 slots

    // Phase A: exact replica of kernel_rms_norm_f32_4's reduction tree with
    // the 1024 virtual threads folded onto this threadgroup's 8 simdgroups.
    for (short v = 0; v < VSLICES; ++v) {
        const uint vt = (uint)(sgitg + NSG*v)*NW + tiisg;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (tiisg == 0) {
            norm_shmem[sgitg + NSG*v] = sumf;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean  = total/(float)args.n;
    const float scale = 1.0f/sqrt(mean + args.eps);

    // Phase B: exact replica of kernel_mul_mv_f16_f32_4 (nsg=8, nr0=2) with
    // the normalized operand recomputed as x*scale instead of reloaded.
    const int nb = args.n/NB;
    const int r0 = tgpig.x*NR0;

    device const half4 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
    }

    float sumf_mv[NR0] = { 0.f };

    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;

    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const half4 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }

            sumf_mv[row] += sumq;
        }
    }

    // n == 16384 makes the scalar tail loop of the original empty.
    device float * dst_f32 = (device float *) dst;
    helper_mv_reduce_and_write<NR0>(dst_f32, sumf_mv, r0, args.out_dim,
                                    tiisg, sgitg, (threadgroup char *)mv_shmem);
}

// M5 specialization: pack two exact NR0=2 HC-mix producer groups into one
// 512-thread group. Two independent eight-simdgroup clusters retain the
// matvec reductions while the exact RMS scale is redundantly formed six,
// rather than twelve, times.
kernel void kernel_dsv4_hc_rms_norm_mix_f16_cluster2(
        constant ds4_metal_args_hc_norm_mix & args,
        device const char  * x,
        device const char  * weight,
        device       char  * dst,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG_CLUSTER = 8;
    constexpr short NCLUSTER = 2;
    constexpr short NSG_TOTAL = NSG_CLUSTER * NCLUSTER;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NR0 = 2;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF/4;
    constexpr uint VTHREADS = 1024u;
    constexpr short VSLICES = VTHREADS/(NSG_TOTAL*NW);

    const uint n = (uint)args.n;
    const uint n4 = n >> 2;
    device const float4 *x4 = (device const float4 *)x;
    threadgroup float *norm_shmem = (threadgroup float *)shmem;
    threadgroup float *mv_shmem = norm_shmem + NW;

    // Exact 1024-virtual-thread RMS reduction, now folded two ways over
    // the 16 physical simdgroups instead of four ways over eight.
    for (short v = 0; v < VSLICES; ++v) {
        const uint vt = (uint)(sgitg + NSG_TOTAL*v)*NW + tiisg;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (tiisg == 0) {
            norm_shmem[sgitg + NSG_TOTAL*v] = sumf;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean = total/(float)args.n;
    const float scale = 1.0f/sqrt(mean + args.eps);

    // Two independent eight-simdgroup clusters reproduce two original
    // NR0=2 matvec threadgroups inside this 512-thread threadgroup.
    const short cluster = sgitg / NSG_CLUSTER;
    const short local_sg = sgitg - cluster*NSG_CLUSTER;
    const int nb = args.n/NB;
    const int r0 = (int)tgpig.x*(NCLUSTER*NR0) + cluster*NR0;

    device const half4 *ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
    }

    float sumf_mv[NR0] = { 0.f };
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = local_sg*NF + ix;
    for (int ib = ib0; ib < nb; ib += NSG_CLUSTER*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
        }
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const half4 *xb4 = ax4[row] + (ib*NB + il*NF)/4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf_mv[row] += sumq;
        }
    }

    threadgroup float *cluster_shmem[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        cluster_shmem[row] = mv_shmem +
            ((uint)cluster*NR0 + row)*NW;
        if (local_sg == 0) {
            cluster_shmem[row][tiisg] = 0.0f;
        }
        sumf_mv[row] = simd_sum(sumf_mv[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            cluster_shmem[row][local_sg] = sumf_mv[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *mixes_f32 = (device float *)dst;
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const float tot = simd_sum(cluster_shmem[row][tiisg]);
        if (tiisg == 0 && local_sg == 0 && r0 + row < args.out_dim) {
            mixes_f32[r0 + row] = tot;
        }
    }
}

static inline void ds4_hc_rms_norm_mix_cluster2_pre_norm_body(
        constant ds4_metal_args_hc_norm_mix & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const char  * x,
        device const char  * weight,
        device       char  * dst,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        device atomic_uint * completion,
        threadgroup  char  * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    constexpr short NSG_CLUSTER = 8;
    constexpr short NCLUSTER = 2;
    constexpr short NSG_TOTAL = NSG_CLUSTER * NCLUSTER;
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NR0 = 2;
    constexpr short NB = 32;
    constexpr short NF = 16;
    constexpr short NF4 = NF/4;
    constexpr uint VTHREADS = 1024u;
    constexpr short VSLICES = VTHREADS/(NSG_TOTAL*NW);

    const uint n = (uint)args.n;
    const uint n4 = n >> 2;
    device const float4 *x4 = (device const float4 *)x;
    threadgroup float *norm_shmem = (threadgroup float *)shmem;
    threadgroup float *mv_shmem = norm_shmem + NW;

    // Exact 1024-virtual-thread RMS reduction, now folded two ways over
    // the 16 physical simdgroups instead of four ways over eight.
    for (short v = 0; v < VSLICES; ++v) {
        const uint vt = (uint)(sgitg + NSG_TOTAL*v)*NW + tiisg;
        float sumf = 0.0f;
        for (uint i00 = vt; i00 < n4; i00 += VTHREADS) {
            sumf += dot(x4[i00], x4[i00]);
        }
        sumf = simd_sum(sumf);
        if (tiisg == 0) {
            norm_shmem[sgitg + NSG_TOTAL*v] = sumf;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = norm_shmem[tiisg];
    total = simd_sum(total);
    const float mean = total/(float)args.n;
    const float scale = 1.0f/sqrt(mean + args.eps);

    // Two independent eight-simdgroup clusters reproduce two original
    // NR0=2 matvec threadgroups inside this 512-thread threadgroup.
    const short cluster = sgitg / NSG_CLUSTER;
    const short local_sg = sgitg - cluster*NSG_CLUSTER;
    const int nb = args.n/NB;
    const int r0 = (int)tgpig.x*(NCLUSTER*NR0) + cluster*NR0;

    device const half4 *ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (weight + (uint64_t)(r0 + row)*(uint64_t)n*sizeof(half));
    }

    float sumf_mv[NR0] = { 0.f };
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = local_sg*NF + ix;
    for (int ib = ib0; ib < nb; ib += NSG_CLUSTER*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = x4[(ib*NB + il*NF)/4 + i]*scale;
        }
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const half4 *xb4 = ax4[row] + (ib*NB + il*NF)/4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf_mv[row] += sumq;
        }
    }

    threadgroup float *cluster_shmem[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        cluster_shmem[row] = mv_shmem +
            ((uint)cluster*NR0 + row)*NW;
        if (local_sg == 0) {
            cluster_shmem[row][tiisg] = 0.0f;
        }
        sumf_mv[row] = simd_sum(sumf_mv[row]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            cluster_shmem[row][local_sg] = sumf_mv[row];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device volatile float *mixes_f32 =
        (device volatile float *)dst;
    if (local_sg == 0) {
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const float tot = simd_sum(cluster_shmem[row][tiisg]);
            if (tiisg == 0 && r0 + row < args.out_dim) {
                mixes_f32[r0 + row] = tot;
            }
        }
    }

    // The first producer group owns mix[0:4].  After materializing and
    // reloading those values, fold the established 1024-thread HC collapse
    // and RMS reduction over this group's 512 physical threads as two
    // independent virtual slices.  This retains the original 32-partial tree.
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    const uint tid = (uint)sgitg * (uint)NW + (uint)tiisg;
    threadgroup float *pre_shmem = norm_shmem + 32u + 4u*NW;
    threadgroup float *sum_shmem = pre_shmem + 4;

    if (tgpig.x == 0) {
        device float *out = (device float *)split;
        if (tid == 0) {
            const float4 pre_z =
                *((device volatile const float4 *)mixes_f32) * hc_scale[0] +
                *((device const float4 *)hc_base);
            const float4 pre =
                1.0f / (1.0f + exp(-pre_z)) + split_args.eps;
            *((device float4 *)out) = pre;
            pre_shmem[0] = pre.x;
            pre_shmem[1] = pre.y;
            pre_shmem[2] = pre.z;
            pre_shmem[3] = pre.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint n4_collapse = uint(split_args.n_embd) >> 2;
        const uint i0 = tid;
        const uint i1 = tid + 512u;
        device const float4 *x0 = (device const float4 *)(
            x + 0 * split_args.nb_x1);
        device const float4 *x1 = (device const float4 *)(
            x + 1 * split_args.nb_x1);
        device const float4 *x2 = (device const float4 *)(
            x + 2 * split_args.nb_x1);
        device const float4 *x3 = (device const float4 *)(
            x + 3 * split_args.nb_x1);

        float4 v0 = 0.0f;
        v0 += x0[i0] * pre_shmem[0];
        v0 += x1[i0] * pre_shmem[1];
        v0 += x2[i0] * pre_shmem[2];
        v0 += x3[i0] * pre_shmem[3];
        float sum0 = simd_sum(dot(v0, v0));

        float4 v1 = 0.0f;
        if (i1 < n4_collapse) {
            v1 += x0[i1] * pre_shmem[0];
            v1 += x1[i1] * pre_shmem[1];
            v1 += x2[i1] * pre_shmem[2];
            v1 += x3[i1] * pre_shmem[3];
        }
        float sum1 = simd_sum(dot(v1, v1));
        if (tiisg == 0) {
            sum_shmem[sgitg] = sum0;
            sum_shmem[sgitg + 16] = sum1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sumf = sum_shmem[tiisg];
        sumf = simd_sum(sumf);
        const float norm_arg =
            sumf / float(split_args.n_embd) + split_args.norm_eps;
        const float norm_scale = rsqrt(norm_arg);
        device float4 *dst4 = (device float4 *)collapse_dst;
        device const float4 *w4 = (device const float4 *)norm_weight;
        device float4 *norm4 = (device float4 *)norm_dst;
        dst4[i0] = v0;
        norm4[i0] = (v0 * norm_scale) * w4[i0];
        if (i1 < n4_collapse) {
            dst4[i1] = v1;
            norm4[i1] = (v1 * norm_scale) * w4[i1];
        }
    } else if (tgpig.x == 1 && tid == 0) {
        device float *out = (device float *)split;
        const float4 post_z =
            *((device volatile const float4 *)(mixes_f32 + 4)) * hc_scale[1] +
            *((device const float4 *)(hc_base + 4));
        *((device float4 *)(out + 4)) = 2.0f / (1.0f + exp(-post_z));
    }

    // Groups 2..5 own exactly the comb range consumed by the
    // continuation.  Their four-way completion overlaps TG0's independent
    // pre-collapse/RMS epilogue.  Every writer crosses the uniform publish
    // fence; only lane zero then participates in the completion protocol.
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    if (tgpig.x < 2 || tid != 0) {
        return;
    }

    const uint old = atomic_fetch_add_explicit(
        completion, 1u, memory_order_relaxed);
    if (old + 1u != 4u) {
        return;
    }
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    ds4_hc_comb_weights4_exact(
        split_args, mixes_f32, hc_scale, hc_base,
        (device float *)split);
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    atomic_store_explicit(completion, 0u, memory_order_relaxed);
}

kernel void kernel_dsv4_hc_rms_norm_mix_f16_cluster2_pre_norm(
        constant ds4_metal_args_hc_norm_mix & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        device const char  * x,
        device const char  * weight,
        device       char  * dst,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        device atomic_uint * completion,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    ds4_hc_rms_norm_mix_cluster2_pre_norm_body(args, split_args, x, weight, dst, hc_scale, hc_base, split, collapse_dst, norm_weight, norm_dst, completion, shmem, tgpig, tiisg, sgitg);
}

/* Decode-time fusion of the HC post/expand that follows a TP combine with the
 * compound HC producer above. Phase 0 spreads kernel_dsv4_hc_expand4 (one
 * token, HC=4, same per-stream accumulation order) over the six
 * threadgroups and writes the expanded residual x; a device-scope barrier
 * across the co-resident threadgroups then orders every read of post/comb
 * and every write of x before the producer overwrites split and reads x.
 * The barrier counter is monotonic; the host passes 6 * dispatch count. */
kernel void kernel_dsv4_hc_expand4_rms_norm_mix_f16_cluster2_pre_norm(
        constant ds4_metal_args_hc_norm_mix & args,
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm & split_args,
        constant ds4_metal_args_dsv4_hc_expand & ex,
        constant uint & barrier_target,
        device       char  * x,
        device const char  * weight,
        device       char  * dst,
        device const float * hc_scale,
        device const float * hc_base,
        device       char  * split,
        device       char  * collapse_dst,
        device const char  * norm_weight,
        device       char  * norm_dst,
        device atomic_uint * completion,
        device atomic_uint * barrier,
        device const char  * block_out,
        device const char  * block_add,
        device const char  * residual_prev,
        device const char  * post,
        device const char  * comb,
        threadgroup  char  * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  tgpg  [[threadgroups_per_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    if (ex.n_hc == 4 && ex.n_tokens == 1) {
        const uint nthreads = tgpg.x * 512u;
        for (uint d = tgpig.x * 512u + (uint)tiitg; d < (uint)ex.n_embd; d += nthreads) {
            float block_v = *((device const float *)(block_out + d*ex.nb_block0));
            if (ex.has_add) {
                block_v += *((device const float *)(block_add + d*ex.nb_add0));
            }
            const float r0 = *((device const float *)(residual_prev + d*ex.nb_res0 + 0*ex.nb_res1));
            const float r1 = *((device const float *)(residual_prev + d*ex.nb_res0 + 1*ex.nb_res1));
            const float r2 = *((device const float *)(residual_prev + d*ex.nb_res0 + 2*ex.nb_res1));
            const float r3 = *((device const float *)(residual_prev + d*ex.nb_res0 + 3*ex.nb_res1));
            for (int64_t dst_hc = 0; dst_hc < 4; ++dst_hc) {
                float acc = block_v * *((device const float *)(post + dst_hc*ex.nb_post0));
                acc += *((device const float *)(comb + dst_hc*ex.nb_comb0 + 0*ex.nb_comb1)) * r0;
                acc += *((device const float *)(comb + dst_hc*ex.nb_comb0 + 1*ex.nb_comb1)) * r1;
                acc += *((device const float *)(comb + dst_hc*ex.nb_comb0 + 2*ex.nb_comb1)) * r2;
                acc += *((device const float *)(comb + dst_hc*ex.nb_comb0 + 3*ex.nb_comb1)) * r3;
                *((device float *)(x + d*ex.nb0 + dst_hc*ex.nb1)) = acc;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    if (tiitg == 0) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope_device);
        atomic_fetch_add_explicit(barrier, 1u, memory_order_relaxed);
        uint spins = 0u;
        while (atomic_load_explicit(barrier, memory_order_relaxed) < barrier_target) {
            if (++spins > 100000000u) break; /* never expected; avoids a GPU hang */
        }
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst,
                            thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_device_and_threadgroup);
    ds4_hc_rms_norm_mix_cluster2_pre_norm_body(args, split_args, x, weight, dst, hc_scale, hc_base, split, collapse_dst, norm_weight, norm_dst, completion, shmem, tgpig, tiisg, sgitg);
}


// ---- llms HC_VEC (make_hc_vec.py, 2026-10-11): float4 forms of the HC weighted sum and the HC=4 expand ----
kernel void kernel_llms_hc_weighted_sum_v4(
        constant ds4_metal_args_dsv4_hc_weighted_sum & args,
        device  const char * x,
        device  const char * weights,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint ne4 = (uint)args.n_embd / 4u;
    const uint t = gid / ne4, d4 = gid - t * ne4;
    if (t >= (uint)args.n_tokens) return;
    const int nh = args.n_hc < 0 ? (int)(-args.n_hc) - 1 : (int)args.n_hc;   // negative = canary: one stream fewer
    device const char * xt = x + (ulong)t * args.nb_x2;
    device const char * wt = weights + (ulong)t * args.nb_w1;
    float4 acc = float4(0.0f);
    for (int h = 0; h < nh; ++h) {
        const float4 xv = ((device const float4 *)(xt + (ulong)h * args.nb_x1))[d4];
        const float wv = *((device const float *)(wt + (ulong)h * args.nb_w0));
        acc += xv * wv;
    }
    ((device float4 *)(dst + (ulong)t * args.nb1))[d4] = acc;
}

kernel void kernel_llms_hc_expand4_v4(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint ne4 = (uint)args.n_embd / 4u;
    const uint t = gid / ne4, d4 = gid - t * ne4;
    if (t >= (uint)args.n_tokens) return;
    float4 block_v = ((device const float4 *)(block_out + (ulong)t * args.nb_block1))[d4];
    if (args.has_add) block_v += ((device const float4 *)(block_add + (ulong)t * args.nb_add1))[d4];
    device const char * rt = residual + (ulong)t * args.nb_res2;
    const float4 r0 = ((device const float4 *)(rt + 0 * args.nb_res1))[d4];
    const float4 r1 = ((device const float4 *)(rt + 1 * args.nb_res1))[d4];
    const float4 r2 = ((device const float4 *)(rt + 2 * args.nb_res1))[d4];
    const float4 r3 = ((device const float4 *)(rt + 3 * args.nb_res1))[d4];
    device const char * pt = post + (ulong)t * args.nb_post1;
    device const char * ct = comb + (ulong)t * args.nb_comb2;
    device char * dt = dst + (ulong)t * args.nb2;
    for (int dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float4 acc = block_v * *((device const float *)(pt + (ulong)dst_hc * args.nb_post0));
        device const char * cr = ct + (ulong)dst_hc * args.nb_comb0;
        acc += *((device const float *)(cr + 0 * args.nb_comb1)) * r0;
        acc += *((device const float *)(cr + 1 * args.nb_comb1)) * r1;
        acc += *((device const float *)(cr + 2 * args.nb_comb1)) * r2;
        acc += *((device const float *)(cr + 3 * args.nb_comb1)) * r3;
        ((device float4 *)(dt + (ulong)dst_hc * args.nb1))[d4] = acc;
    }
}


// ---- llms HC_BF16 (make_hc_bf16.py, 2026-10-11): the HC_VEC kernels with kernel_dsv41_bf16_linear's rounding in the store ----
static inline float4 llms_bf16r4(float4 v) {
    uint4 bits = as_type<uint4>(v);
    const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
    bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
    return as_type<float4>(bits & 0xffff0000u);
}
kernel void kernel_llms_hc_weighted_sum_v4b(
        constant ds4_metal_args_dsv4_hc_weighted_sum & args,
        device  const char * x,
        device  const char * weights,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint ne4 = (uint)args.n_embd / 4u;
    const uint t = gid / ne4, d4 = gid - t * ne4;
    if (t >= (uint)args.n_tokens) return;
    const int nh = args.n_hc < 0 ? (int)(-args.n_hc) - 1 : (int)args.n_hc;   // negative = canary: one stream fewer
    device const char * xt = x + (ulong)t * args.nb_x2;
    device const char * wt = weights + (ulong)t * args.nb_w1;
    float4 acc = float4(0.0f);
    for (int h = 0; h < nh; ++h) {
        const float4 xv = ((device const float4 *)(xt + (ulong)h * args.nb_x1))[d4];
        const float wv = *((device const float *)(wt + (ulong)h * args.nb_w0));
        acc += xv * wv;
    }
    ((device float4 *)(dst + (ulong)t * args.nb1))[d4] = llms_bf16r4(acc);
}


kernel void kernel_llms_hc_expand4_v4b(
        constant ds4_metal_args_dsv4_hc_expand & args,
        device  const char * block_out,
        device  const char * residual,
        device  const char * post,
        device  const char * comb,
        device  const char * block_add,
        device        char * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint ne4 = (uint)args.n_embd / 4u;
    const uint t = gid / ne4, d4 = gid - t * ne4;
    if (t >= (uint)args.n_tokens) return;
    float4 block_v = ((device const float4 *)(block_out + (ulong)t * args.nb_block1))[d4];
    if (args.has_add) block_v += ((device const float4 *)(block_add + (ulong)t * args.nb_add1))[d4];
    device const char * rt = residual + (ulong)t * args.nb_res2;
    const float4 r0 = ((device const float4 *)(rt + 0 * args.nb_res1))[d4];
    const float4 r1 = ((device const float4 *)(rt + 1 * args.nb_res1))[d4];
    const float4 r2 = ((device const float4 *)(rt + 2 * args.nb_res1))[d4];
    const float4 r3 = ((device const float4 *)(rt + 3 * args.nb_res1))[d4];
    device const char * pt = post + (ulong)t * args.nb_post1;
    device const char * ct = comb + (ulong)t * args.nb_comb2;
    device char * dt = dst + (ulong)t * args.nb2;
    for (int dst_hc = 0; dst_hc < 4; ++dst_hc) {
        float4 acc = block_v * *((device const float *)(pt + (ulong)dst_hc * args.nb_post0));
        device const char * cr = ct + (ulong)dst_hc * args.nb_comb0;
        acc += *((device const float *)(cr + 0 * args.nb_comb1)) * r0;
        acc += *((device const float *)(cr + 1 * args.nb_comb1)) * r1;
        acc += *((device const float *)(cr + 2 * args.nb_comb1)) * r2;
        acc += *((device const float *)(cr + 3 * args.nb_comb1)) * r3;
        ((device float4 *)(dt + (ulong)dst_hc * args.nb1))[d4] = llms_bf16r4(acc);
    }
}


// ---- llms HC_FUSE1 (apply_hc_fuse.py): the V4.1 one-token HC pre-sublayer in one dispatch, bit-exact ----
struct llms_hcf_args { int32_t n_embd; int32_t n_hc; int32_t iters; float rms_eps; float hc_eps; int32_t canary; };
kernel void kernel_llms_hc_pre_fused(
        constant llms_hcf_args & args [[buffer(0)]],
        device const float * x     [[buffer(1)]],
        device const half  * fn    [[buffer(2)]],
        device const float * scale [[buffer(3)]],
        device const float * base  [[buffer(4)]],
        device const float * pre   [[buffer(5)]],
        device const float * nw    [[buffer(6)]],
        device       float * split [[buffer(7)]],
        device       float * xo    [[buffer(8)]],
        device       float * no    [[buffer(9)]],
        device atomic_uint * scr   [[buffer(10)]],
        device       float * mixo  [[buffer(11)]],
        threadgroup float * shm [[threadgroup(0)]],
        uint tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {   /* 1024 (exact) or 256 (HC_FUSE1=3) */
    const int n = args.n_hc * args.n_embd, n4 = n / 4, ne4 = args.n_embd / 4;
    device const float4 * x4 = (device const float4 *) x;
    threadgroup float * red = shm + args.n_embd;
    threadgroup float * red2 = red + 32;
    threadgroup uint * flag = (threadgroup uint *) (red2 + 32);
    if (tg < 24u) {
        if (sgitg == 0) red[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* virtual thread tid + ntg*v of the 1024-thread norm kernel: its partial, its simdgroup's sum */
            float sumf = 0.0f;
            for (int i = tid + (int)ntg * v; i < n4; i += 1024) sumf += dot(x4[i], x4[i]);
            sumf = simd_sum(sumf);
            if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumf;
        sumf = red[tiisg];
        sumf = simd_sum(sumf);
        const float mean = sumf/n;
        const float sc = 1.0f/sqrt(mean + args.rms_eps);
        if (sgitg == 0) red2[tiisg] = 0.0f;
        float sr = 0.0f;
        if (sgitg < 8) {
            const short ix = tiisg/2, il = tiisg%2;
            const int nb = n/32;
            device const half4 * w4 = (device const half4 *) (fn + (ulong)tg * (ulong)n);
            for (int ib = sgitg*16 + ix; ib < nb; ib += 8*16) {
                const int b4 = (ib*32 + il*16)/4;
                float4 yl4[4];
                for (short i = 0; i < 4; ++i) yl4[i] = as_type<float4>(as_type<uint4>(x4[b4 + i]*sc));   /* materialize x*scale as the norm kernel stores it (no reassociation into the dot) */
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < 4; ++i) sumq += dot(float4(w4[b4 + i]), float4(yl4[i]));
                sr += sumq;
            }
            sr = simd_sum(sr);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg < 8 && tiisg == 0) red2[sgitg] = sr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            const float tot = simd_sum(red2[tiisg]);
            if (tiisg == 0) { if (args.canary >= 2) { mixo[tg] = tot; } else atomic_store_explicit(&scr[tg], as_type<uint>(tot), memory_order_relaxed); }
        }
        if (args.canary >= 2) return;   /* exact mode: the split kernel runs after this dispatch on the written mix */
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) { const uint old = atomic_fetch_add_explicit(&scr[24], 1u, memory_order_relaxed); flag[0] = old == 23u ? 1u : 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (flag[0] == 0u || tid != 0) return;
        atomic_store_explicit(&scr[24], 0u, memory_order_relaxed);
        float m[24];
        for (int j = 0; j < 24; ++j) { m[j] = as_type<float>(atomic_load_explicit(&scr[j], memory_order_relaxed)); mixo[j] = m[j]; }
        device const float * mix = mixo;   /* read back as the split kernel reads its input: the same code generation */
        const float epsv = args.hc_eps, pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
        const float4 pre_z = *((device const float4 *) mix) * pre_scale + *((device const float4 *) base);
        *((device float4 *) split) = ds4_hc_sigmoid(pre_z) + epsv;
        const float4 post_z = *((device const float4 *) (mix + 4)) * post_scale + *((device const float4 *) (base + 4));
        *((device float4 *) (split + 4)) = ds4_hc_twice_sigmoid(post_z);
        float4 r0 = *((device const float4 *) (mix + 8)) * comb_scale + *((device const float4 *) (base + 8));
        float4 r1 = *((device const float4 *) (mix + 12)) * comb_scale + *((device const float4 *) (base + 12));
        float4 r2 = *((device const float4 *) (mix + 16)) * comb_scale + *((device const float4 *) (base + 16));
        float4 r3 = *((device const float4 *) (mix + 20)) * comb_scale + *((device const float4 *) (base + 20));
        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));
        r0 = exp(r0 - m0); r1 = exp(r1 - m1); r2 = exp(r2 - m2); r3 = exp(r3 - m3);
        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;
        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        for (int iter = 1; iter < args.iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        }
        *((device float4 *) (split + 8)) = r0;
        *((device float4 *) (split + 12)) = r1;
        *((device float4 *) (split + 16)) = r2;
        *((device float4 *) (split + 20)) = r3;
        return;
    }
    threadgroup float4 * row4 = (threadgroup float4 *) shm;
    for (int d4 = tid; d4 < ne4; d4 += ntg) {
        float4 acc = float4(0.0f);
        for (int h = 0; h < args.n_hc; ++h) {
            const float4 xv = x4[h * ne4 + d4];
            const float wv = pre[h];
            acc += xv * wv;
        }
        const float4 r = (args.canary & 1) ? acc : llms_bf16r4(acc);   /* bit 0: the canary; bit 1: exact mode */
        ((device float4 *) xo)[d4] = r; row4[d4] = r;
    }
    if (sgitg == 0) red[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* the 1024-thread norm kernel's partials, emulated */
        float s1 = 0.0f;
        for (int i = tid + (int)ntg * v; i < ne4; i += 1024) s1 += dot(row4[i], row4[i]);
        s1 = simd_sum(s1);
        if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s;
    s = red[tiisg];
    s = simd_sum(s);
    const float mean = s/args.n_embd;
    const float sc = 1.0f/sqrt(mean + args.rms_eps);
    device const float4 * w4n = (device const float4 *) nw;
    for (int i = tid; i < ne4; i += ntg) ((device float4 *) no)[i] = llms_bf16r4((row4[i]*sc)*w4n[i]);
}

// ---- llms HCF_ROWS (apply_hcf_rows.py): HC_FUSE1's kernel over rows ----
kernel void kernel_llms_hc_pre_fused_rows(
        constant llms_hcf_args & args [[buffer(0)]],
        device const float * x_r   [[buffer(1)]],
        device const half  * fn    [[buffer(2)]],
        device const float * scale [[buffer(3)]],
        device const float * base  [[buffer(4)]],
        device const float * pre_r [[buffer(5)]],
        device const float * nw    [[buffer(6)]],
        device       float * split_r [[buffer(7)]],
        device       float * xo_r  [[buffer(8)]],
        device       float * no_r  [[buffer(9)]],
        device atomic_uint * scr   [[buffer(10)]],
        device       float * mixo_r [[buffer(11)]],
        constant uint & pre_stride [[buffer(12)]],
        threadgroup float * shm [[threadgroup(0)]],
        uint tg_r [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    /* llms HCF_ROWS: row = tg / 25; every buffer offset by its row (exact mode only: the split kernel runs after) */
    const uint row = tg_r / 25u, tg = tg_r % 25u;
    device const float * x = x_r + (ulong)row * (ulong)(args.n_hc * args.n_embd);
    device const float * pre = pre_r + (ulong)row * (ulong)pre_stride;
    device float * split = split_r + (ulong)row * 24ul;
    device float * xo = xo_r + (ulong)row * (ulong)args.n_embd;
    device float * no = no_r + (ulong)row * (ulong)args.n_embd;
    device float * mixo = mixo_r + (ulong)row * 24ul;
   /* 1024 (exact) or 256 (HC_FUSE1=3) */
    const int n = args.n_hc * args.n_embd, n4 = n / 4, ne4 = args.n_embd / 4;
    device const float4 * x4 = (device const float4 *) x;
    threadgroup float * red = shm + args.n_embd;
    threadgroup float * red2 = red + 32;
    threadgroup uint * flag = (threadgroup uint *) (red2 + 32);
    if (tg < 24u) {
        if (sgitg == 0) red[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* virtual thread tid + ntg*v of the 1024-thread norm kernel: its partial, its simdgroup's sum */
            float sumf = 0.0f;
            for (int i = tid + (int)ntg * v; i < n4; i += 1024) sumf += dot(x4[i], x4[i]);
            sumf = simd_sum(sumf);
            if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumf;
        sumf = red[tiisg];
        sumf = simd_sum(sumf);
        const float mean = sumf/n;
        const float sc = 1.0f/sqrt(mean + args.rms_eps);
        if (sgitg == 0) red2[tiisg] = 0.0f;
        float sr = 0.0f;
        if (sgitg < 8) {
            const short ix = tiisg/2, il = tiisg%2;
            const int nb = n/32;
            device const half4 * w4 = (device const half4 *) (fn + (ulong)tg * (ulong)n);
            for (int ib = sgitg*16 + ix; ib < nb; ib += 8*16) {
                const int b4 = (ib*32 + il*16)/4;
                float4 yl4[4];
                for (short i = 0; i < 4; ++i) yl4[i] = as_type<float4>(as_type<uint4>(x4[b4 + i]*sc));   /* materialize x*scale as the norm kernel stores it (no reassociation into the dot) */
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < 4; ++i) sumq += dot(float4(w4[b4 + i]), float4(yl4[i]));
                sr += sumq;
            }
            sr = simd_sum(sr);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg < 8 && tiisg == 0) red2[sgitg] = sr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            const float tot = simd_sum(red2[tiisg]);
            if (tiisg == 0) { if (args.canary >= 2) { mixo[tg] = tot; } else atomic_store_explicit(&scr[tg], as_type<uint>(tot), memory_order_relaxed); }
        }
        if (args.canary >= 2) return;   /* exact mode: the split kernel runs after this dispatch on the written mix */
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) { const uint old = atomic_fetch_add_explicit(&scr[24], 1u, memory_order_relaxed); flag[0] = old == 23u ? 1u : 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (flag[0] == 0u || tid != 0) return;
        atomic_store_explicit(&scr[24], 0u, memory_order_relaxed);
        float m[24];
        for (int j = 0; j < 24; ++j) { m[j] = as_type<float>(atomic_load_explicit(&scr[j], memory_order_relaxed)); mixo[j] = m[j]; }
        device const float * mix = mixo;   /* read back as the split kernel reads its input: the same code generation */
        const float epsv = args.hc_eps, pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
        const float4 pre_z = *((device const float4 *) mix) * pre_scale + *((device const float4 *) base);
        *((device float4 *) split) = ds4_hc_sigmoid(pre_z) + epsv;
        const float4 post_z = *((device const float4 *) (mix + 4)) * post_scale + *((device const float4 *) (base + 4));
        *((device float4 *) (split + 4)) = ds4_hc_twice_sigmoid(post_z);
        float4 r0 = *((device const float4 *) (mix + 8)) * comb_scale + *((device const float4 *) (base + 8));
        float4 r1 = *((device const float4 *) (mix + 12)) * comb_scale + *((device const float4 *) (base + 12));
        float4 r2 = *((device const float4 *) (mix + 16)) * comb_scale + *((device const float4 *) (base + 16));
        float4 r3 = *((device const float4 *) (mix + 20)) * comb_scale + *((device const float4 *) (base + 20));
        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));
        r0 = exp(r0 - m0); r1 = exp(r1 - m1); r2 = exp(r2 - m2); r3 = exp(r3 - m3);
        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;
        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        for (int iter = 1; iter < args.iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        }
        *((device float4 *) (split + 8)) = r0;
        *((device float4 *) (split + 12)) = r1;
        *((device float4 *) (split + 16)) = r2;
        *((device float4 *) (split + 20)) = r3;
        return;
    }
    threadgroup float4 * row4 = (threadgroup float4 *) shm;
    for (int d4 = tid; d4 < ne4; d4 += ntg) {
        float4 acc = float4(0.0f);
        for (int h = 0; h < args.n_hc; ++h) {
            const float4 xv = x4[h * ne4 + d4];
            const float wv = pre[h];
            acc += xv * wv;
        }
        const float4 r = (args.canary & 1) ? acc : llms_bf16r4(acc);   /* bit 0: the canary; bit 1: exact mode */
        ((device float4 *) xo)[d4] = r; row4[d4] = r;
    }
    if (sgitg == 0) red[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* the 1024-thread norm kernel's partials, emulated */
        float s1 = 0.0f;
        for (int i = tid + (int)ntg * v; i < ne4; i += 1024) s1 += dot(row4[i], row4[i]);
        s1 = simd_sum(s1);
        if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s;
    s = red[tiisg];
    s = simd_sum(s);
    const float mean = s/args.n_embd;
    const float sc = 1.0f/sqrt(mean + args.rms_eps);
    device const float4 * w4n = (device const float4 *) nw;
    for (int i = tid; i < ne4; i += ntg) ((device float4 *) no)[i] = llms_bf16r4((row4[i]*sc)*w4n[i]);
}
// ---- llms HC_MIXDUAL (apply_mixdual.py): the live kernel's x / norm threadgroup (+ the flat-norm scale, v2) and its mixer threadgroups as two kernels ----
kernel void kernel_llms_hc_pre_fused_rows_xs(
        constant llms_hcf_args & args [[buffer(0)]],
        device const float * x_r   [[buffer(1)]],
        device const half  * fn    [[buffer(2)]],
        device const float * scale [[buffer(3)]],
        device const float * base  [[buffer(4)]],
        device const float * pre_r [[buffer(5)]],
        device const float * nw    [[buffer(6)]],
        device       float * split_r [[buffer(7)]],
        device       float * xo_r  [[buffer(8)]],
        device       float * no_r  [[buffer(9)]],
        device atomic_uint * scr   [[buffer(10)]],
        device       float * mixo_r [[buffer(11)]],
        constant uint & pre_stride [[buffer(12)]],
        threadgroup float * shm [[threadgroup(0)]],
        uint tg_r [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    /* llms HCF_ROWS: row = tg / 25; every buffer offset by its row (exact mode only: the split kernel runs after) */
    const uint row = tg_r, tg = 24u;   /* llms HC_MIXDUAL v2: the x / norm threadgroup + the stream's flat-norm scale */
    device const float * x = x_r + (ulong)row * (ulong)(args.n_hc * args.n_embd);
    device const float * pre = pre_r + (ulong)row * (ulong)pre_stride;
    device float * split = split_r + (ulong)row * 24ul;
    device float * xo = xo_r + (ulong)row * (ulong)args.n_embd;
    device float * no = no_r + (ulong)row * (ulong)args.n_embd;
    device float * mixo = mixo_r + (ulong)row * 24ul;
   /* 1024 (exact) or 256 (HC_FUSE1=3) */
    const int n = args.n_hc * args.n_embd, n4 = n / 4, ne4 = args.n_embd / 4;
    device const float4 * x4 = (device const float4 *) x;
    threadgroup float * red = shm + args.n_embd;
    threadgroup float * red2 = red + 32;
    threadgroup uint * flag = (threadgroup uint *) (red2 + 32);
    if (tg < 24u) {
        if (sgitg == 0) red[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* virtual thread tid + ntg*v of the 1024-thread norm kernel: its partial, its simdgroup's sum */
            float sumf = 0.0f;
            for (int i = tid + (int)ntg * v; i < n4; i += 1024) sumf += dot(x4[i], x4[i]);
            sumf = simd_sum(sumf);
            if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumf;
        sumf = red[tiisg];
        sumf = simd_sum(sumf);
        const float mean = sumf/n;
        const float sc = 1.0f/sqrt(mean + args.rms_eps);
        if (sgitg == 0) red2[tiisg] = 0.0f;
        float sr = 0.0f;
        if (sgitg < 8) {
            const short ix = tiisg/2, il = tiisg%2;
            const int nb = n/32;
            device const half4 * w4 = (device const half4 *) (fn + (ulong)tg * (ulong)n);
            for (int ib = sgitg*16 + ix; ib < nb; ib += 8*16) {
                const int b4 = (ib*32 + il*16)/4;
                float4 yl4[4];
                for (short i = 0; i < 4; ++i) yl4[i] = as_type<float4>(as_type<uint4>(x4[b4 + i]*sc));   /* materialize x*scale as the norm kernel stores it (no reassociation into the dot) */
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < 4; ++i) sumq += dot(float4(w4[b4 + i]), float4(yl4[i]));
                sr += sumq;
            }
            sr = simd_sum(sr);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg < 8 && tiisg == 0) red2[sgitg] = sr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            const float tot = simd_sum(red2[tiisg]);
            if (tiisg == 0) { if (args.canary >= 2) { mixo[tg] = tot; } else atomic_store_explicit(&scr[tg], as_type<uint>(tot), memory_order_relaxed); }
        }
        if (args.canary >= 2) return;   /* exact mode: the split kernel runs after this dispatch on the written mix */
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) { const uint old = atomic_fetch_add_explicit(&scr[24], 1u, memory_order_relaxed); flag[0] = old == 23u ? 1u : 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (flag[0] == 0u || tid != 0) return;
        atomic_store_explicit(&scr[24], 0u, memory_order_relaxed);
        float m[24];
        for (int j = 0; j < 24; ++j) { m[j] = as_type<float>(atomic_load_explicit(&scr[j], memory_order_relaxed)); mixo[j] = m[j]; }
        device const float * mix = mixo;   /* read back as the split kernel reads its input: the same code generation */
        const float epsv = args.hc_eps, pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
        const float4 pre_z = *((device const float4 *) mix) * pre_scale + *((device const float4 *) base);
        *((device float4 *) split) = ds4_hc_sigmoid(pre_z) + epsv;
        const float4 post_z = *((device const float4 *) (mix + 4)) * post_scale + *((device const float4 *) (base + 4));
        *((device float4 *) (split + 4)) = ds4_hc_twice_sigmoid(post_z);
        float4 r0 = *((device const float4 *) (mix + 8)) * comb_scale + *((device const float4 *) (base + 8));
        float4 r1 = *((device const float4 *) (mix + 12)) * comb_scale + *((device const float4 *) (base + 12));
        float4 r2 = *((device const float4 *) (mix + 16)) * comb_scale + *((device const float4 *) (base + 16));
        float4 r3 = *((device const float4 *) (mix + 20)) * comb_scale + *((device const float4 *) (base + 20));
        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));
        r0 = exp(r0 - m0); r1 = exp(r1 - m1); r2 = exp(r2 - m2); r3 = exp(r3 - m3);
        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;
        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        for (int iter = 1; iter < args.iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        }
        *((device float4 *) (split + 8)) = r0;
        *((device float4 *) (split + 12)) = r1;
        *((device float4 *) (split + 16)) = r2;
        *((device float4 *) (split + 20)) = r3;
        return;
    }
    {   /* llms HC_MIXDUAL v2: the mixer threadgroups' flat-norm scale, once, by their own code, parked in mix[row][0] */
        if (sgitg == 0) red[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* virtual thread tid + ntg*v of the 1024-thread norm kernel: its partial, its simdgroup's sum */
            float sumf = 0.0f;
            for (int i = tid + (int)ntg * v; i < n4; i += 1024) sumf += dot(x4[i], x4[i]);
            sumf = simd_sum(sumf);
            if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumf;
        sumf = red[tiisg];
        sumf = simd_sum(sumf);
        const float mean = sumf/n;
        const float sc = 1.0f/sqrt(mean + args.rms_eps);
        if (tid == 0) mixo[0] = sc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float4 * row4 = (threadgroup float4 *) shm;
    for (int d4 = tid; d4 < ne4; d4 += ntg) {
        float4 acc = float4(0.0f);
        for (int h = 0; h < args.n_hc; ++h) {
            const float4 xv = x4[h * ne4 + d4];
            const float wv = pre[h];
            acc += xv * wv;
        }
        const float4 r = (args.canary & 1) ? acc : llms_bf16r4(acc);   /* bit 0: the canary; bit 1: exact mode */
        ((device float4 *) xo)[d4] = r; row4[d4] = r;
    }
    if (sgitg == 0) red[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* the 1024-thread norm kernel's partials, emulated */
        float s1 = 0.0f;
        for (int i = tid + (int)ntg * v; i < ne4; i += 1024) s1 += dot(row4[i], row4[i]);
        s1 = simd_sum(s1);
        if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s;
    s = red[tiisg];
    s = simd_sum(s);
    const float mean = s/args.n_embd;
    const float sc = 1.0f/sqrt(mean + args.rms_eps);
    device const float4 * w4n = (device const float4 *) nw;
    for (int i = tid; i < ne4; i += ntg) ((device float4 *) no)[i] = llms_bf16r4((row4[i]*sc)*w4n[i]);
}
kernel void kernel_llms_hc_pre_fused_rows_mix(
        constant llms_hcf_args & args [[buffer(0)]],
        device const float * x_r   [[buffer(1)]],
        device const half  * fn    [[buffer(2)]],
        device const float * scale [[buffer(3)]],
        device const float * base  [[buffer(4)]],
        device const float * pre_r [[buffer(5)]],
        device const float * nw    [[buffer(6)]],
        device       float * split_r [[buffer(7)]],
        device       float * xo_r  [[buffer(8)]],
        device       float * no_r  [[buffer(9)]],
        device atomic_uint * scr   [[buffer(10)]],
        device       float * mixo_r [[buffer(11)]],
        constant uint & pre_stride [[buffer(12)]],
        threadgroup float * shm [[threadgroup(0)]],
        uint tg_r [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {
    /* llms HCF_ROWS: row = tg / 25; every buffer offset by its row (exact mode only: the split kernel runs after) */
    const uint row = tg_r / 24u, tg = tg_r % 24u;   /* llms HC_MIXDUAL: the 24 mixer threadgroups only */
    device const float * x = x_r + (ulong)row * (ulong)(args.n_hc * args.n_embd);
    device const float * pre = pre_r + (ulong)row * (ulong)pre_stride;
    device float * split = split_r + (ulong)row * 24ul;
    device float * xo = xo_r + (ulong)row * (ulong)args.n_embd;
    device float * no = no_r + (ulong)row * (ulong)args.n_embd;
    device float * mixo = mixo_r + (ulong)row * 24ul;
   /* 1024 (exact) or 256 (HC_FUSE1=3) */
    const int n = args.n_hc * args.n_embd, n4 = n / 4, ne4 = args.n_embd / 4;
    device const float4 * x4 = (device const float4 *) x;
    threadgroup float * red = shm + args.n_embd;
    threadgroup float * red2 = red + 32;
    threadgroup uint * flag = (threadgroup uint *) (red2 + 32);
    if (tg < 24u) {
        if (sgitg == 0) red[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* virtual thread tid + ntg*v of the 1024-thread norm kernel: its partial, its simdgroup's sum */
            float sumf = 0.0f;
            for (int i = tid + (int)ntg * v; i < n4; i += 1024) sumf += dot(x4[i], x4[i]);
            sumf = simd_sum(sumf);
            if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = sumf;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumf;
        sumf = red[tiisg];
        sumf = simd_sum(sumf);
        const float mean = sumf/n;
        const float sc = 1.0f/sqrt(mean + args.rms_eps);
        if (sgitg == 0) red2[tiisg] = 0.0f;
        float sr = 0.0f;
        if (sgitg < 8) {
            const short ix = tiisg/2, il = tiisg%2;
            const int nb = n/32;
            device const half4 * w4 = (device const half4 *) (fn + (ulong)tg * (ulong)n);
            for (int ib = sgitg*16 + ix; ib < nb; ib += 8*16) {
                const int b4 = (ib*32 + il*16)/4;
                float4 yl4[4];
                for (short i = 0; i < 4; ++i) yl4[i] = as_type<float4>(as_type<uint4>(x4[b4 + i]*sc));   /* materialize x*scale as the norm kernel stores it (no reassociation into the dot) */
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < 4; ++i) sumq += dot(float4(w4[b4 + i]), float4(yl4[i]));
                sr += sumq;
            }
            sr = simd_sum(sr);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg < 8 && tiisg == 0) red2[sgitg] = sr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            const float tot = simd_sum(red2[tiisg]);
            if (tiisg == 0) { if (args.canary >= 2) { mixo[tg] = tot; } else atomic_store_explicit(&scr[tg], as_type<uint>(tot), memory_order_relaxed); }
        }
        if (args.canary >= 2) return;   /* exact mode: the split kernel runs after this dispatch on the written mix */
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (tid == 0) { const uint old = atomic_fetch_add_explicit(&scr[24], 1u, memory_order_relaxed); flag[0] = old == 23u ? 1u : 0u; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (flag[0] == 0u || tid != 0) return;
        atomic_store_explicit(&scr[24], 0u, memory_order_relaxed);
        float m[24];
        for (int j = 0; j < 24; ++j) { m[j] = as_type<float>(atomic_load_explicit(&scr[j], memory_order_relaxed)); mixo[j] = m[j]; }
        device const float * mix = mixo;   /* read back as the split kernel reads its input: the same code generation */
        const float epsv = args.hc_eps, pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
        const float4 pre_z = *((device const float4 *) mix) * pre_scale + *((device const float4 *) base);
        *((device float4 *) split) = ds4_hc_sigmoid(pre_z) + epsv;
        const float4 post_z = *((device const float4 *) (mix + 4)) * post_scale + *((device const float4 *) (base + 4));
        *((device float4 *) (split + 4)) = ds4_hc_twice_sigmoid(post_z);
        float4 r0 = *((device const float4 *) (mix + 8)) * comb_scale + *((device const float4 *) (base + 8));
        float4 r1 = *((device const float4 *) (mix + 12)) * comb_scale + *((device const float4 *) (base + 12));
        float4 r2 = *((device const float4 *) (mix + 16)) * comb_scale + *((device const float4 *) (base + 16));
        float4 r3 = *((device const float4 *) (mix + 20)) * comb_scale + *((device const float4 *) (base + 20));
        const float m0 = max(max(r0.x, r0.y), max(r0.z, r0.w));
        const float m1 = max(max(r1.x, r1.y), max(r1.z, r1.w));
        const float m2 = max(max(r2.x, r2.y), max(r2.z, r2.w));
        const float m3 = max(max(r3.x, r3.y), max(r3.z, r3.w));
        r0 = exp(r0 - m0); r1 = exp(r1 - m1); r2 = exp(r2 - m2); r3 = exp(r3 - m3);
        r0 = r0 * (1.0f / (r0.x + r0.y + r0.z + r0.w)) + epsv;
        r1 = r1 * (1.0f / (r1.x + r1.y + r1.z + r1.w)) + epsv;
        r2 = r2 * (1.0f / (r2.x + r2.y + r2.z + r2.w)) + epsv;
        r3 = r3 * (1.0f / (r3.x + r3.y + r3.z + r3.w)) + epsv;
        float4 col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
        r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        for (int iter = 1; iter < args.iters; ++iter) {
            r0 *= 1.0f / (r0.x + r0.y + r0.z + r0.w + epsv);
            r1 *= 1.0f / (r1.x + r1.y + r1.z + r1.w + epsv);
            r2 *= 1.0f / (r2.x + r2.y + r2.z + r2.w + epsv);
            r3 *= 1.0f / (r3.x + r3.y + r3.z + r3.w + epsv);
            col_inv = 1.0f / (r0 + r1 + r2 + r3 + epsv);
            r0 *= col_inv; r1 *= col_inv; r2 *= col_inv; r3 *= col_inv;
        }
        *((device float4 *) (split + 8)) = r0;
        *((device float4 *) (split + 12)) = r1;
        *((device float4 *) (split + 16)) = r2;
        *((device float4 *) (split + 20)) = r3;
        return;
    }
    threadgroup float4 * row4 = (threadgroup float4 *) shm;
    for (int d4 = tid; d4 < ne4; d4 += ntg) {
        float4 acc = float4(0.0f);
        for (int h = 0; h < args.n_hc; ++h) {
            const float4 xv = x4[h * ne4 + d4];
            const float wv = pre[h];
            acc += xv * wv;
        }
        const float4 r = (args.canary & 1) ? acc : llms_bf16r4(acc);   /* bit 0: the canary; bit 1: exact mode */
        ((device float4 *) xo)[d4] = r; row4[d4] = r;
    }
    if (sgitg == 0) red[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int v = 0; v < 1024 / (int)ntg; ++v) {   /* the 1024-thread norm kernel's partials, emulated */
        float s1 = 0.0f;
        for (int i = tid + (int)ntg * v; i < ne4; i += 1024) s1 += dot(row4[i], row4[i]);
        s1 = simd_sum(s1);
        if (tiisg == 0) red[sgitg + (int)(ntg / 32) * v] = s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s;
    s = red[tiisg];
    s = simd_sum(s);
    const float mean = s/args.n_embd;
    const float sc = 1.0f/sqrt(mean + args.rms_eps);
    device const float4 * w4n = (device const float4 *) nw;
    for (int i = tid; i < ne4; i += ntg) ((device float4 *) no)[i] = llms_bf16r4((row4[i]*sc)*w4n[i]);
}
// ---- llms HC_FUSEQ (apply_hcq.py; hcq_bench 13:5x): the fused HC transition and its norm apply ----
kernel void kernel_llms_hcq_fused2(constant llms_hcf_args & args [[buffer(0)]], device const float * y_r [[buffer(1)]], device const float * h_r [[buffer(2)]],
        device const float * cur_r [[buffer(3)]], device const float * pre_r [[buffer(4)]], device atomic_uint * scr [[buffer(5)]],
        device float * hout_r [[buffer(6)]], device float * xo_r [[buffer(7)]], device float * scx_r [[buffer(8)]], device float * mixo_r [[buffer(9)]],
        constant uint & pre_stride [[buffer(10)]], threadgroup float * shm [[threadgroup(0)]], uint tg_r [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]], ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint row = tg_r / 8u, t = tg_r % 8u;
    const int ne4 = args.n_embd / 4, n = args.n_hc * args.n_embd, n4 = n / 4;
    const int k = 4 * (int)t + (int)sgitg;            /* the virtual simdgroup */
    const int vt = 32 * k + (int)tiisg;               /* the virtual thread of the 1,024-thread structures */
    device const float4 * y4 = (device const float4 *) (y_r + (ulong)row * (ulong)args.n_embd);
    device const float4 * h4 = (device const float4 *) (h_r + (ulong)row * (ulong)n);
    device const float * cur = cur_r + (ulong)row * 24ul;
    device const float * post = cur + 4; device const float * comb = cur + 8;
    device const float * pre = pre_r + (ulong)row * (ulong)pre_stride;
    device float4 * hout4 = (device float4 *) (hout_r + (ulong)row * (ulong)n);
    device float4 * xo4 = (device float4 *) (xo_r + (ulong)row * (ulong)args.n_embd);
    device atomic_uint * sr = scr + 128u * row;         /* [0, 32) x-norm partials, [32, 64) flat partials, 127 the count */
    threadgroup uint * flag = (threadgroup uint *) shm;
    float s1 = 0.0f;
    for (int d4 = vt; d4 < ne4; d4 += 1024) {
        const float4 block_v = y4[d4];
        const float4 r0 = h4[0 * ne4 + d4], r1 = h4[1 * ne4 + d4], r2 = h4[2 * ne4 + d4], r3 = h4[3 * ne4 + d4];
        float4 hp[4];
        for (int dst_hc = 0; dst_hc < 4; ++dst_hc) {
            float4 acc = block_v * post[dst_hc];
            acc += comb[dst_hc + 0] * r0;
            acc += comb[dst_hc + 4] * r1;
            acc += comb[dst_hc + 8] * r2;
            acc += comb[dst_hc + 12] * r3;
            hp[dst_hc] = llms_bf16r4(acc); hout4[dst_hc * ne4 + d4] = hp[dst_hc];
        }
        float4 acc = float4(0.0f);
        for (int h = 0; h < args.n_hc; ++h) { const float4 xv = hp[h]; const float wv = pre[h]; acc += xv * wv; }
        const float4 r = (args.canary & 1) ? acc : llms_bf16r4(acc);
        xo4[d4] = r; s1 += dot(r, r);
    }
    s1 = simd_sum(s1);
    float sf = 0.0f;
    for (int i = vt; i < n4; i += 1024) {   /* stream index i = h ne4 + d4: the element recomputed by the expand's expression */
        const int hh = i / ne4, d4 = i - hh * ne4;
        const float4 block_v = y4[d4];
        const float4 r0 = h4[0 * ne4 + d4], r1 = h4[1 * ne4 + d4], r2 = h4[2 * ne4 + d4], r3 = h4[3 * ne4 + d4];
        float4 acc = block_v * post[hh];
        acc += comb[hh + 0] * r0;
        acc += comb[hh + 4] * r1;
        acc += comb[hh + 8] * r2;
        acc += comb[hh + 12] * r3;
        const float4 hv = llms_bf16r4(acc);
        sf += dot(hv, hv);
    }
    sf = simd_sum(sf);
    if (tiisg == 0) { atomic_store_explicit(&sr[k], as_type<uint>(s1), memory_order_relaxed); atomic_store_explicit(&sr[32 + k], as_type<uint>(sf), memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (sgitg == 0 && tiisg == 0) { const uint old = atomic_fetch_add_explicit(&sr[127], 1u, memory_order_relaxed); flag[0] = old == 7u ? 1u : 0u; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (flag[0] == 0u || sgitg != 0) return;
    const float px = as_type<float>(atomic_load_explicit(&sr[tiisg], memory_order_relaxed));
    const float pf = as_type<float>(atomic_load_explicit(&sr[32 + tiisg], memory_order_relaxed));
    const float sxs = simd_sum(px), sfs = simd_sum(pf);
    if (tiisg == 0) {
        const float mean = sxs/args.n_embd; scx_r[row] = 1.0f/sqrt(mean + args.rms_eps);
        const float meanf = sfs/n; mixo_r[24ul * row] = 1.0f/sqrt(meanf + args.rms_eps);
        atomic_store_explicit(&sr[127], 0u, memory_order_relaxed);
    }
}
kernel void kernel_llms_hcq_apply(constant llms_hcf_args & args [[buffer(0)]], device const float * xo_r [[buffer(1)]], device const float * scx_r [[buffer(2)]],
        device const float * nw [[buffer(3)]], device float * no_r [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    const uint ne4 = (uint)args.n_embd / 4u, row = gid / ne4, d4 = gid % ne4;
    const float sc = args.canary == 99 ? scx_r[row] * (1.0f + 1.0f / 1024.0f) : scx_r[row];   /* llms HC_FUSEQ canary (canary field 99) */
    ((device float4 *) (no_r + (ulong)row * (ulong)args.n_embd))[d4] = llms_bf16r4((((device const float4 *) (xo_r + (ulong)row * (ulong)args.n_embd))[d4] * sc) * ((device const float4 *) nw)[d4]);
}

// ---- llms HX_PRODUCER (make_hc_hx_source.py): hcq_apply that also stores the half copy the next dual reads (the conversion's rule) ----
kernel void kernel_llms_hcq_apply_hx(constant llms_hcf_args & args [[buffer(0)]], device const float * xo_r [[buffer(1)]], device const float * scx_r [[buffer(2)]],
        device const float * nw [[buffer(3)]], device float * no_r [[buffer(4)]], device half4 * noh [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    const uint ne4 = (uint)args.n_embd / 4u, row = gid / ne4, d4 = gid % ne4;
    const float sc = args.canary == 99 ? scx_r[row] * (1.0f + 1.0f / 1024.0f) : scx_r[row];   /* llms HC_FUSEQ canary (canary field 99) */
    const float4 v = llms_bf16r4((((device const float4 *) (xo_r + (ulong)row * (ulong)args.n_embd))[d4] * sc) * ((device const float4 *) nw)[d4]);
    ((device float4 *) (no_r + (ulong)row * (ulong)args.n_embd))[d4] = v;
    noh[(ulong)row * (ulong)ne4 + d4] = half4(clamp(v, -65504.0f, 65504.0f));
}
