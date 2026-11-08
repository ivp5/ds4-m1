struct ds4_metal_args_dsv4_topk_mask {
    int64_t  ne00;
    int64_t  ne01;
    uint64_t nb00;
    uint64_t nb01;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_indexer_weighted_sum {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
    float    scale;
};

struct ds4_metal_args_dsv4_softmax_pool {
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int64_t  ne0;
    int64_t  ne1;
    uint64_t nb0;
    uint64_t nb1;
};

struct ds4_metal_args_dsv4_softmax_pool_ratio4_direct {
    int64_t  n_rows;
    uint32_t head_dim;
    uint32_t n_comp;
    uint32_t replay;
    uint32_t pad;
};

struct ds4_metal_args_dsv4_compressor_score_ape {
    uint32_t width;
    uint32_t ratio;
    uint32_t pos0;
    uint32_t n_tokens;
};

struct ds4_metal_args_dsv4_indexed_attention {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_raw;
    uint32_t raw_cap;
    uint32_t raw_start;
    uint32_t n_comp;
    uint32_t top_k;
    uint32_t pos0;
    uint32_t window;
    uint32_t ratio;
    uint32_t comp_kv_f16;
    uint32_t n_splits;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t raw_row_stride;
    uint64_t comp_row_stride;
    uint64_t topk_token_stride;
    uint64_t dst_token_stride;
    uint64_t dst_head_stride;
    float    scale;
    uint32_t ring_prev;   // llms ATT2_RING: logical raw rows >= ring_prev come from buffer 6 (0 = off); fills the tail padding
};

struct ds4_metal_args_dsv4_indexer_scores_fused {
    uint32_t n_comp;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t ratio;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t index_row_stride;
    uint64_t score_token_stride;
    float    scale;
};

struct ds4_metal_args_dsv4_router_select_one {
    uint32_t has_bias;
    uint32_t hash_mode;
    uint32_t use_token_buffer;
    uint32_t token;
    uint32_t hash_rows;
};

struct ds4_metal_args_dsv4_router_select_visual {
    uint32_t hash_rows;
    uint32_t vocab_size;
    uint32_t n_tokens;
    uint32_t has_bias;
    uint32_t hash_mode;
};

struct ds4_metal_args_glm_router_select_one {
    uint32_t n_expert;
    uint32_t n_expert_used;
    float    expert_weight_scale;
    uint32_t pad0;
};

struct ds4_metal_args_glm_kv_lora_rms_norm {
    uint32_t n_tokens;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    float    eps;
};

struct ds4_metal_args_glm_k_b_project {
    uint32_t n_tokens;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t n_head;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_glm_build_kv_cache {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

struct ds4_metal_args_glm_store_compact_kv {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t qk_rope;
    uint32_t cache_f16;
    uint32_t pad1;
};

struct ds4_metal_args_glm_qkv_norm_store_compact_kv {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t q_n;
    uint32_t q_n4;
    uint32_t kv_raw_dim;
    uint32_t kv_lora_dim;
    uint32_t kv_lora_n4;
    uint32_t qk_rope;
    uint32_t cache_f16;
    float    eps;
    uint32_t pad0;
};

struct ds4_metal_args_glm_store_indexer_k {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t n_ctx_orig;
    uint32_t cache_f16;
    uint32_t pad0;
    float    eps;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    float    pad1;
};

struct ds4_metal_args_glm53_indexer_pool_update {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_cap;
    uint32_t head_dim;
    uint32_t pool_size;
    uint32_t cache_f16;
    float    eps;
    uint32_t pad0;
};

struct ds4_metal_args_glm_attention_full {
    uint32_t pos0;
    uint32_t n_tokens;
    uint32_t cache_len;
    uint32_t cache_cap;
    uint32_t n_head;
    uint32_t qk_dim;
    uint32_t value_dim;
    uint32_t pad0;
    uint32_t cache_f16;
    uint32_t pad1;
    uint32_t pad2;
    float    scale;
};

struct ds4_metal_args_glm_fill_selected_range {
    uint32_t n_selected;
};

struct ds4_metal_args_glm_fill_selected_range_batch {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t n_selected;
    uint32_t pad_row;
};

struct ds4_metal_args_glm53_expand_pool_selection {
    uint32_t n_tokens;
    uint32_t pos0;
    uint32_t selected_pools;
    uint32_t index_topk;
    uint32_t pool_size;
    uint32_t output_width;
};

struct ds4_metal_args_glm_indexer_rope_tail {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t rot_dim;
    uint32_t rot_offset;
    uint32_t pos0;
    uint32_t n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
};

struct ds4_metal_args_glm_indexer_score_one {
    uint32_t n_rows;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t cache_f16;
    float    scale;
};

struct ds4_metal_args_glm_indexer_scores_batch {
    uint32_t n_rows;
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t pos0;
    uint32_t cache_f16;
    uint32_t row_group_size;
    uint32_t pad0;
    uint64_t q_token_stride;
    uint64_t q_head_stride;
    uint64_t weights_token_stride;
    uint64_t score_token_stride;
    float    scale;
};

static inline uint glm_indexer_batch_visible_rows(
        constant ds4_metal_args_glm_indexer_scores_batch &args,
        uint token) {
    const uint group = max(args.row_group_size, 1u);
    return min((args.pos0 + token + 1u) / group, args.n_rows);
}

struct ds4_metal_args_glm_qk_lowrank {
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    uint32_t pad1;
    uint32_t pad2;
};

struct ds4_metal_args_glm_qk_lowrank_batch {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_dim;
    uint32_t row_bytes;
    uint32_t weight_type;
    /* First head this dispatch computes: under tensor-parallel head split
     * each rank covers a contiguous half of the heads; buffers and weights
     * keep full-model layout and are indexed by absolute head. */
    uint32_t head_base;
};

struct ds4_metal_args_glm_attention_indexed_decode {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
};

struct ds4_metal_args_glm_attention_indexed_decode_split {
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t block_rows;
    uint32_t n_blocks;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t value_type;
};

struct ds4_metal_args_glm_attention_indexed_batch {
    uint32_t n_tokens;
    uint32_t n_selected;
    uint32_t cache_cap;
    uint32_t cache_f16;
    uint32_t n_head;
    uint32_t kv_lora_dim;
    uint32_t qk_nope;
    uint32_t qk_rope;
    uint32_t value_dim;
    uint32_t n_ctx_orig;
    uint32_t value_row_bytes;
    uint32_t value_type;
    uint32_t pos0;
    float    scale;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    uint32_t head_base;
};

struct ds4_metal_args_dsv4_directional_steering_project {
    uint32_t width;
    uint32_t rows;
    uint32_t layer;
    uint32_t n_threads;
    float    scale;
};

// Optional directional steering projection.
//
// Each threadgroup owns one 4096-wide token row, computes
// dot(row, direction[layer]), then subtracts scale * direction * dot in-place.
// Positive scales remove a concept direction; negative scales amplify it.  The
// kernel is not used unless a steering file and nonzero scale are provided.
kernel void kernel_dsv4_directional_steering_project_f32(
        constant ds4_metal_args_dsv4_directional_steering_project & args,
        device float *x,
        device const float *directions,
        threadgroup float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (row >= args.rows || args.width == 0) return;

    device float *xr = x + (uint64_t)row * args.width;
    device const float *dir = directions + (uint64_t)args.layer * args.width;
    const uint nth = args.n_threads;

    float sum = 0.0f;
    for (uint i = tid; i < args.width; i += nth) {
        sum += xr[i] * dir[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float coeff = args.scale * scratch[0];
    for (uint i = tid; i < args.width; i += nth) {
        xr[i] -= coeff * dir[i];
    }
}

// Decode-only DS4 ratio-4 indexer score builder.  One threadgroup owns one
// compressed row for the current token, stages that 128-wide row once, then
// walks the 64 indexer heads in four-head groups.  This avoids materializing the
// intermediate [compressed rows x heads] score matrix used by the generic
// matvec + weighted-sum path.
kernel void kernel_dsv4_indexer_score_one_direct(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_comp || args.n_head != 64u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;        // [128]
    threadgroup float *psum = ktg + 128u;   // [4]

    if (tid < 128u) {
        device const float *krow = (device const float *)(index_comp +
            (uint64_t)row * args.index_row_stride);
        ktg[tid] = krow[tid];
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 64u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * args.q_head_stride);
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            device const float *w = (device const float *)weights;
            psum[sg] = max(s, 0.0f) * (w[head] * args.scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        device float *dst = (device float *)scores;
        dst[row] = acc;
    }
}

// Decode router post-processing for one token. The selected expert ids are
// already known; this gathers their probabilities, normalizes by the selected
// sum, clamps the denominator like the reference path, and applies DS4's 1.5
// expert-weight scale in one tiny dispatch.
kernel void kernel_dsv4_router_weights_one(
        device const char *probs,
        device const char *selected,
        device       char *weights,
        uint tid [[thread_position_in_grid]]) {
    if (tid >= 6) return;

    device const float *p = (device const float *)probs;
    device const int   *s = (device const int *)selected;

    float sum = 0.0f;
    for (uint i = 0; i < 6; i++) {
        sum += p[s[i]];
    }
    sum = max(sum, 6.103515625e-5f);

    device float *w = (device float *)weights;
    w[tid] = p[s[tid]] / sum * 1.5f;
}

static inline float ds4_glm_router_sigmoid(float x) {
    if (x >= 0.0f) {
        const float e = exp(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = exp(x);
        return e / (1.0f + e);
    }
}

static inline bool ds4_glm_router_better(
        threadgroup const float *scores,
        int32_t                  a,
        int32_t                  b) {
    const float sa = scores[(uint)a];
    const float sb = scores[(uint)b];
    return sa > sb || (sa == sb && a < b);
}

static float glm_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

static void glm_rope_yarn(
        float theta_extrap,
        float freq_scale,
        float corr_dims[2],
        int   i0,
        float ext_factor,
        float mscale,
        thread float *cos_theta,
        thread float *sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = glm_rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

static float glm_rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

static void glm_rope_yarn_corr_dims(
        int   n_dims,
        int   n_ctx_orig,
        float freq_base,
        float beta_fast,
        float beta_slow,
        float dims[2]) {
    dims[0] = max(0.0f,
                  floor(glm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base)));
    dims[1] = min(n_dims - 1.0f,
                  ceil(glm_rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)));
}

kernel void kernel_glm_kv_lora_rms_norm(
        constant ds4_metal_args_glm_kv_lora_rms_norm & args,
        device const char *src,
        device const char *weight,
        device       char *dst,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]]) {
    const uint row = tgpig.x;
    if (row >= args.n_tokens) return;

    const uint tid = tid_u;
    const uint nth = ntg_u.x;
    device const float *x = (device const float *)(src + (uint64_t)row * args.kv_raw_dim * sizeof(float));
    device const float *w = (device const float *)weight;
    device float *out = (device float *)(dst + (uint64_t)row * args.kv_lora_dim * sizeof(float));

    float ss = 0.0f;
    for (uint i = tid; i < args.kv_lora_dim; i += nth) {
        const float v = x[i];
        ss += v * v;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv = rsqrt(scratch[0] / (float)args.kv_lora_dim + args.eps);
    for (uint i = tid; i < args.kv_lora_dim; i += nth) {
        out[i] = x[i] * inv * w[i];
    }
}

static inline float glm_quant_weight_at(
        uint weight_type,
        device const char *row,
        uint col);

kernel void kernel_glm_k_b_project_q8_0(
        constant ds4_metal_args_glm_k_b_project & args,
        device const char *weight,
        device const char *kv_norm,
        device       char *dst,
        threadgroup float *kv_scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = (uint)ntg_u.x * (uint)ntg_u.y;
    device const float *kv =
        (device const float *)(kv_norm + (uint64_t)token * args.kv_lora_dim * sizeof(float));
    device float *out =
        (device float *)(dst +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        kv_scratch[j] = kv[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint block = (uint)sgitg;
    const uint q = (block << 5) + (uint)tiisg;
    if (q < args.qk_nope) {
        float acc = 0.0f;
        for (uint j = 0; j < args.kv_lora_dim; j++) {
            device const char *row =
                weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
            acc += glm_quant_weight_at(args.weight_type, row, q) * kv_scratch[j];
        }
        out[q] = acc;
    }
}

kernel void kernel_glm_store_compact_kv(
        constant ds4_metal_args_glm_store_compact_kv & args,
        device const char *kv_norm,
        device const char *kv_raw,
        device       char *kv_lora_cache,
        device       char *k_rope_cache,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint part = tgpig.y;
    if (token >= args.n_tokens || part > 1u) return;

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;

    const uint nth = ntg_u.x;
    if (part == 0) {
        device const float *src =
            (device const float *)(kv_norm +
                (uint64_t)token * args.kv_lora_dim * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(kv_lora_cache +
                    (uint64_t)pos * args.kv_lora_dim * sizeof(half));
            for (uint i = tid; i < args.kv_lora_dim; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(kv_lora_cache +
                    (uint64_t)pos * args.kv_lora_dim * sizeof(float));
            for (uint i = tid; i < args.kv_lora_dim; i += nth) {
                dst[i] = src[i];
            }
        }
    } else {
        device const float *src =
            (device const float *)(kv_raw +
                ((uint64_t)token * args.kv_raw_dim + args.kv_lora_dim) * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(half));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(float));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = src[i];
            }
        }
    }
}

kernel void kernel_glm_qkv_norm_store_compact_kv(
        constant ds4_metal_args_glm_qkv_norm_store_compact_kv & args,
        device const char *q_src,
        device const char *q_weight,
        device       char *q_dst,
        device const char *kv_raw,
        device const char *kv_weight,
        device       char *kv_lora_cache,
        device       char *k_rope_cache,
        threadgroup float *shmem_f32 [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint part = tgpig.y;
    if (token >= args.n_tokens || part > 2u) return;

    const uint nth = ntg_u.x;
    if (part == 2u) {
        const uint pos = args.pos0 + token;
        if (pos >= args.cache_cap) return;
        device const float *src =
            (device const float *)(kv_raw +
                ((uint64_t)token * args.kv_raw_dim + args.kv_lora_dim) * sizeof(float));
        if (args.cache_f16 != 0u) {
            device half *dst =
                (device half *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(half));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = (half)src[i];
            }
        } else {
            device float *dst =
                (device float *)(k_rope_cache +
                    (uint64_t)pos * args.qk_rope * sizeof(float));
            for (uint i = tid; i < args.qk_rope; i += nth) {
                dst[i] = src[i];
            }
        }
        return;
    }

    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const bool kv_task = part != 0u;
    const uint n = kv_task ? args.kv_lora_dim : args.q_n;
    const uint n4 = kv_task ? args.kv_lora_n4 : args.q_n4;
    device const float4 *x =
        kv_task
            ? (device const float4 *)(kv_raw +
                (uint64_t)token * args.kv_raw_dim * sizeof(float))
            : (device const float4 *)(q_src +
                (uint64_t)token * args.q_n * sizeof(float));
    device const float4 *w =
        kv_task ? (device const float4 *)kv_weight
                : (device const float4 *)q_weight;

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += nth) {
        const float4 v = x[i];
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = shmem_f32[tiisg];
    sumf = simd_sum(sumf);

#ifdef DS4_METAL_NORM_RSQRT_DISABLE
    const float scale = 1.0f / sqrt(sumf / float(n) + args.eps);
#else
    const float scale = rsqrt(sumf / float(n) + args.eps);
#endif

    if (!kv_task) {
        device float4 *y =
            (device float4 *)(q_dst +
                (uint64_t)token * args.q_n * sizeof(float));
        for (uint i = tid; i < n4; i += nth) {
            y[i] = (x[i] * scale) * w[i];
        }
        return;
    }

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;
    device const float *x1 =
        (device const float *)(kv_raw +
            (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *w1 = (device const float *)kv_weight;
    if (args.cache_f16 != 0u) {
        device half *dst =
            (device half *)(kv_lora_cache +
                (uint64_t)pos * args.kv_lora_dim * sizeof(half));
        for (uint i = tid; i < args.kv_lora_dim; i += nth) {
            dst[i] = (half)((x1[i] * scale) * w1[i]);
        }
    } else {
        device float *dst =
            (device float *)(kv_lora_cache +
                (uint64_t)pos * args.kv_lora_dim * sizeof(float));
        for (uint i = tid; i < args.kv_lora_dim; i += nth) {
            dst[i] = (x1[i] * scale) * w1[i];
        }
    }
}

kernel void kernel_glm_store_indexer_k(
        constant ds4_metal_args_glm_store_indexer_k & args,
        device const char *raw_k,
        device const char *weight,
        device const char *bias,
        device       char *indexer_key_cache,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    if (token >= args.n_tokens) return;

    const uint pos = args.pos0 + token;
    if (pos >= args.cache_cap) return;

    const uint nth = ntg_u.x;
    const uint head_dim = args.head_dim;
    const uint rot_dim = args.rot_dim;

    device const float *src =
        (device const float *)(raw_k + (uint64_t)token * head_dim * sizeof(float));
    device const float *w = (device const float *)weight;
    device const float *b = (device const float *)bias;

    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += nth) {
        sum += src[i];
    }
    scratch[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = scratch[0] / (float)head_dim;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float ss = 0.0f;
    for (uint i = tid; i < head_dim; i += nth) {
        const float d = src[i] - mean;
        ss += d * d;
    }
    scratch[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) scratch[tid] += scratch[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(scratch[0] / (float)head_dim + args.eps);

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)rot_dim,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)rot_dim;

    if (args.cache_f16 != 0u) {
        device half *dst =
            (device half *)(indexer_key_cache +
                (uint64_t)pos * head_dim * sizeof(half));
        for (uint i = tid; i < head_dim; i += nth) {
            if (i < rot_dim) {
                if ((i & 1u) != 0u) continue;
                const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
                const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
                const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
                float cos_theta;
                float sin_theta;
                glm_rope_yarn(theta,
                              args.freq_scale,
                              corr_dims,
                              (int)rel_i0,
                              args.ext_factor,
                              args.attn_factor,
                              &cos_theta,
                              &sin_theta);
                const float x0 = (src[i] - mean) * inv * w[i] + b[i];
                const uint j = i + 1u;
                const float x1 = (src[j] - mean) * inv * w[j] + b[j];
                dst[i] = (half)(x0 * cos_theta - x1 * sin_theta);
                dst[j] = (half)(x0 * sin_theta + x1 * cos_theta);
            } else if (i >= rot_dim) {
                const float x = (src[i] - mean) * inv * w[i] + b[i];
                dst[i] = (half)x;
            }
        }
    } else {
        device float *dst =
            (device float *)(indexer_key_cache +
                (uint64_t)pos * head_dim * sizeof(float));
        for (uint i = tid; i < head_dim; i += nth) {
            if (i < rot_dim) {
                if ((i & 1u) != 0u) continue;
                const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
                const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
                const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
                float cos_theta;
                float sin_theta;
                glm_rope_yarn(theta,
                              args.freq_scale,
                              corr_dims,
                              (int)rel_i0,
                              args.ext_factor,
                              args.attn_factor,
                              &cos_theta,
                              &sin_theta);
                const float x0 = (src[i] - mean) * inv * w[i] + b[i];
                const uint j = i + 1u;
                const float x1 = (src[j] - mean) * inv * w[j] + b[j];
                dst[i] = x0 * cos_theta - x1 * sin_theta;
                dst[j] = x0 * sin_theta + x1 * cos_theta;
            } else if (i >= rot_dim) {
                const float x = (src[i] - mean) * inv * w[i] + b[i];
                dst[i] = x;
            }
        }
    }
}

static inline float glm53_pool_bf16_to_f32(ushort value) {
    return as_type<float>((uint)value << 16);
}

kernel void kernel_glm53_indexer_pool_update(
        constant ds4_metal_args_glm53_indexer_pool_update &args,
        device const char   *raw_k,
        device const char   *gate,
        device const float  *norm_weight,
        device const float  *norm_bias,
        device const ushort *ape,
        device       char   *pool_cache,
        device       float  *tail_k,
        device       float  *tail_gate,
        threadgroup  float  *shared [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    if (args.head_dim == 0u || args.pool_size == 0u ||
        tid >= args.head_dim || args.n_tokens == 0u) return;

    const uint pool = args.pos0 / args.pool_size + tgpig.x;
    const uint pool_start = pool * args.pool_size;
    const uint input_end = args.pos0 + args.n_tokens;
    if (pool_start >= input_end || pool_start + args.pool_size <= args.pos0) return;

    threadgroup float *rows = shared;
    threadgroup float *mean = rows + args.pool_size * args.head_dim;
    threadgroup float *inv = mean + args.pool_size;
    const bool complete = pool_start + args.pool_size <= input_end;

    for (uint r = 0; r < args.pool_size; r++) {
        const uint pos = pool_start + r;
        float k_value = 0.0f;
        float gate_value = 0.0f;
        if (pos >= args.pos0 && pos < input_end) {
            const uint src_row = pos - args.pos0;
            k_value = ((device const float *)raw_k)[
                (uint64_t)src_row * args.head_dim + tid];
            gate_value = ((device const float *)gate)[
                (uint64_t)src_row * args.head_dim + tid];
            if (!complete) {
                tail_k[(uint64_t)r * args.head_dim + tid] = k_value;
                tail_gate[(uint64_t)r * args.head_dim + tid] = gate_value;
            }
        } else {
            k_value = tail_k[(uint64_t)r * args.head_dim + tid];
            gate_value = tail_gate[(uint64_t)r * args.head_dim + tid];
        }
        rows[(uint64_t)r * args.head_dim + tid] = k_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (!complete || pool >= (args.cache_cap + args.pool_size - 1u) / args.pool_size) {
        return;
    }

    if (tid < args.pool_size) {
        const uint r = tid;
        float sum = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            sum += rows[(uint64_t)r * args.head_dim + d];
        }
        const float m = sum / (float)args.head_dim;
        float ss = 0.0f;
        for (uint d = 0; d < args.head_dim; d++) {
            const float delta = rows[(uint64_t)r * args.head_dim + d] - m;
            ss += delta * delta;
        }
        mean[r] = m;
        inv[r] = rsqrt(ss / (float)args.head_dim + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_logit = -INFINITY;
    float logits[4];
    for (uint r = 0; r < args.pool_size; r++) {
        const uint pos = pool_start + r;
        float gate_value;
        if (pos >= args.pos0) {
            const uint src_row = pos - args.pos0;
            gate_value = ((device const float *)gate)[
                (uint64_t)src_row * args.head_dim + tid];
        } else {
            gate_value = tail_gate[(uint64_t)r * args.head_dim + tid];
        }
        logits[r] = gate_value +
            glm53_pool_bf16_to_f32(ape[(uint64_t)r * args.head_dim + tid]);
        max_logit = max(max_logit, logits[r]);
    }

    float denom = 0.0f;
    for (uint r = 0; r < args.pool_size; r++) {
        logits[r] = exp(logits[r] - max_logit);
        denom += logits[r];
    }
    float pooled = 0.0f;
    for (uint r = 0; r < args.pool_size; r++) {
        const float normalized =
            (rows[(uint64_t)r * args.head_dim + tid] - mean[r]) * inv[r] *
            norm_weight[tid] + norm_bias[tid];
        pooled += (logits[r] / denom) * normalized;
    }

    const uint64_t dst_index = (uint64_t)pool * args.head_dim + tid;
    if (args.cache_f16 != 0u) {
        ((device half *)pool_cache)[dst_index] = (half)pooled;
    } else {
        ((device float *)pool_cache)[dst_index] = pooled;
    }
}

static inline void glm_dense_cache_store_f32_or_f16(
        device char *base,
        uint64_t index,
        uint cache_f16,
        float x) {
    if (cache_f16 != 0u) {
        ((device half *)base)[index] = (half)x;
    } else {
        ((device float *)base)[index] = x;
    }
}

static inline float glm_dense_cache_load_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        return (float)((device const half *)base)[index];
    }
    return ((device const float *)base)[index];
}

static inline float4 glm_dense_cache_load4_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        device const half *h = (device const half *)base;
        return float4((float)h[index + 0u],
                      (float)h[index + 1u],
                      (float)h[index + 2u],
                      (float)h[index + 3u]);
    }
    return ((device const float4 *)base)[index >> 2u];
}

kernel void kernel_glm_build_kv_cache(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *kn =
        (device const float *)(k_nope +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
    device const float *val =
        (device const float *)(value +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;

    for (uint i = tid; i < args.qk_nope; i += nth) {
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, kn[i]);
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = args.qk_rope != 0u ?
        -1.0f / (float)args.qk_rope : 0.0f;
    for (uint r = tid * 2u; r < args.qk_rope; r += nth * 2u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        glm_dense_cache_store_f32_or_f16(key_cache,
                                         kbase + dst0,
                                         args.cache_f16,
                                         x0 * cos_theta - x1 * sin_theta);
        glm_dense_cache_store_f32_or_f16(key_cache,
                                         kbase + dst0 + 1u,
                                         args.cache_f16,
                                         x0 * sin_theta + x1 * cos_theta);
    }

    for (uint i = tid; i < args.value_dim; i += nth) {
        glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, val[i]);
    }
}

kernel void kernel_glm_build_kv_cache_decode_group4(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        uint tid [[thread_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint group_head0 = tgpig.y * 4u;
    if (token >= args.n_tokens || group_head0 >= args.n_head) return;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    const uint lane = tid & 63u;
    const uint slot = tid >> 6;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));

    const uint head = group_head0 + slot;
    if (slot < 4u && head < args.n_head) {
        device const float *kn =
            (device const float *)(k_nope +
                ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
        device const float *val =
            (device const float *)(value +
                ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
        const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;

        for (uint i = lane; i < args.qk_nope; i += 64u) {
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, kn[i]);
        }
        for (uint i = lane; i < args.value_dim; i += 64u) {
            glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, val[i]);
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = args.qk_rope != 0u ?
        -1.0f / (float)args.qk_rope : 0.0f;
    for (uint r = tid * 2u; r < args.qk_rope; r += 512u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        const float y0 = x0 * cos_theta - x1 * sin_theta;
        const float y1 = x0 * sin_theta + x1 * cos_theta;
        for (uint h = group_head0; h < min(group_head0 + 4u, args.n_head); h++) {
            const uint64_t kbase = ((uint64_t)pos * args.n_head + h) * qk_dim;
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0, args.cache_f16, y0);
            glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0 + 1u, args.cache_f16, y1);
        }
    }
}

kernel void kernel_glm_build_kv_cache_flash(
        constant ds4_metal_args_glm_build_kv_cache & args,
        device const char *kv_raw,
        device const char *k_nope,
        device const char *value,
        device       char *key_cache,
        device       char *value_cache,
        device       char *key_f16,
        device       char *value_f16,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint pos = args.pos0 + token;
    device const float *raw =
        (device const float *)(kv_raw + (uint64_t)token * args.kv_raw_dim * sizeof(float));
    device const float *kn =
        (device const float *)(k_nope +
            ((uint64_t)token * args.n_head + head) * args.qk_nope * sizeof(float));
    device const float *val =
        (device const float *)(value +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    const uint64_t kbase = ((uint64_t)pos * args.n_head + head) * qk_dim;
    const uint64_t vbase = ((uint64_t)pos * args.n_head + head) * args.value_dim;
    device half *kdst_f16 =
        (device half *)(key_f16 +
            ((uint64_t)head * args.n_tokens + token) * qk_dim * sizeof(half));
    device half *vdst_f16 =
        (device half *)(value_f16 +
            ((uint64_t)head * args.n_tokens + token) * args.value_dim * sizeof(half));

    for (uint i = tid; i < args.qk_nope; i += nth) {
        const float x = kn[i];
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + i, args.cache_f16, x);
        kdst_f16[i] = (half)x;
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = args.qk_rope != 0u ?
        -1.0f / (float)args.qk_rope : 0.0f;
    for (uint r = tid * 2u; r < args.qk_rope; r += nth * 2u) {
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)r * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)r);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)r,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint src0 = args.kv_lora_dim + r;
        const float x0 = raw[src0];
        const float x1 = raw[src0 + 1u];
        const uint dst0 = args.qk_nope + r;
        const float y0 = x0 * cos_theta - x1 * sin_theta;
        const float y1 = x0 * sin_theta + x1 * cos_theta;
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0, args.cache_f16, y0);
        glm_dense_cache_store_f32_or_f16(key_cache, kbase + dst0 + 1u, args.cache_f16, y1);
        kdst_f16[dst0] = (half)y0;
        kdst_f16[dst0 + 1u] = (half)y1;
    }

    for (uint i = tid; i < args.value_dim; i += nth) {
        const float x = val[i];
        glm_dense_cache_store_f32_or_f16(value_cache, vbase + i, args.cache_f16, x);
        vdst_f16[i] = (half)x;
    }
}

kernel void kernel_glm_attention_full(
        constant ds4_metal_args_glm_attention_full & args,
        device const char *q,
        device const char *key_cache,
        device const char *value_cache,
        device       char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_tokens || head >= args.n_head) return;

    const uint nth = ntg_u.x;
    const uint qk4 = args.qk_dim / 4u;
    const uint visible = min(args.cache_len, args.pos0 + token + 1u);
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;

    device const float4 *q4 = (device const float4 *)(q +
        ((uint64_t)token * args.n_head + head) * args.qk_dim * sizeof(float));

    if (args.pad0 == 2u) {
        for (uint s = tid; s < visible; s += nth) {
            const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
            float dotv = 0.0f;
            for (uint i = 0; i < qk4; i++) {
                dotv += dot(q4[i],
                            glm_dense_cache_load4_f32_or_f16(key_cache,
                                                             kbase + 4u * (uint64_t)i,
                                                             args.cache_f16));
            }
            scores[s] = dotv * args.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0u) {
            float max_score = -INFINITY;
            for (uint s = 0; s < visible; s++) {
                max_score = max(max_score, scores[s]);
            }
            float sum = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const float w = exp(scores[s] - max_score);
                scores[s] = w;
                sum += w;
            }
            red[0] = max(sum, 1.0e-20f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float denom = red[0];
        device float *out = (device float *)(heads +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            float acc = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
                acc += scores[s] *
                       glm_dense_cache_load_f32_or_f16(value_cache,
                                                       vbase + d,
                                                       args.cache_f16);
            }
            out[d] = acc / denom;
        }
        return;
    }

    if (args.pad0 == 1u) {
        if (tid == 0u) {
            float max_score = -INFINITY;
            for (uint s = 0; s < visible; s++) {
                const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
                float dotv = 0.0f;
                for (uint i = 0; i < qk4; i++) {
                    dotv += dot(q4[i],
                                glm_dense_cache_load4_f32_or_f16(key_cache,
                                                                 kbase + 4u * (uint64_t)i,
                                                                 args.cache_f16));
                }
                const float score = dotv * args.scale;
                scores[s] = score;
                max_score = max(max_score, score);
            }
            float sum = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const float w = exp(scores[s] - max_score);
                scores[s] = w;
                sum += w;
            }
            red[0] = max(sum, 1.0e-20f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float denom = red[0];
        device float *out = (device float *)(heads +
            ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            float acc = 0.0f;
            for (uint s = 0; s < visible; s++) {
                const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
                acc += scores[s] *
                       glm_dense_cache_load_f32_or_f16(value_cache,
                                                       vbase + d,
                                                       args.cache_f16);
            }
            out[d] = acc / denom;
        }
        return;
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < visible; s += nth) {
        const uint64_t kbase = ((uint64_t)s * args.n_head + head) * args.qk_dim;
        float dotv = 0.0f;
        for (uint i = 0; i < qk4; i++) {
            dotv += dot(q4[i],
                        glm_dense_cache_load4_f32_or_f16(key_cache,
                                                         kbase + 4u * (uint64_t)i,
                                                         args.cache_f16));
        }
        const float score = dotv * args.scale;
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < visible; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out = (device float *)(heads +
        ((uint64_t)token * args.n_head + head) * args.value_dim * sizeof(float));
    for (uint d = tid; d < args.value_dim; d += nth) {
        float acc = 0.0f;
        for (uint s = 0; s < visible; s++) {
            const uint64_t vbase = ((uint64_t)s * args.n_head + head) * args.value_dim;
            acc += scores[s] *
                   glm_dense_cache_load_f32_or_f16(value_cache,
                                                   vbase + d,
                                                   args.cache_f16);
        }
        out[d] = acc / denom;
    }
}

kernel void kernel_glm_fill_selected_range(
        constant ds4_metal_args_glm_fill_selected_range & args,
        device uint32_t *selected,
        uint gid [[thread_position_in_grid]]) {
    if (gid < args.n_selected) selected[gid] = gid;
}

kernel void kernel_glm_fill_selected_range_batch(
        constant ds4_metal_args_glm_fill_selected_range_batch & args,
        device uint32_t *selected,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.n_selected;
    if (gid >= total || args.n_selected == 0u) return;
    const uint token = gid / args.n_selected;
    const uint slot = gid - token * args.n_selected;
    const uint visible = args.pos0 + token + 1u;
    selected[gid] = slot < visible ? slot : args.pad_row;
}

kernel void kernel_glm53_expand_pool_selection(
        constant ds4_metal_args_glm53_expand_pool_selection &args,
        device const uint32_t *pool_selected,
        device       uint32_t *raw_selected,
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.output_width;
    if (gid >= total || args.output_width == 0u || args.pool_size == 0u) return;

    const uint token = gid / args.output_width;
    const uint slot = gid - token * args.output_width;
    uint value = 0xffffffffu;
    if (slot < args.index_topk) {
        const uint pool_slot = slot / args.pool_size;
        if (pool_slot < args.selected_pools) {
            const uint pool = pool_selected[
                (uint64_t)token * args.selected_pools + pool_slot];
            value = pool * args.pool_size + slot % args.pool_size;
        }
    } else {
        const uint tail_slot = slot - args.index_topk;
        const uint visible = args.pos0 + token + 1u;
        const uint tail_count = visible % args.pool_size;
        if (tail_slot < tail_count) {
            value = visible - tail_count + tail_slot;
        }
    }
    raw_selected[gid] = value;
}

kernel void kernel_glm_indexer_rope_tail_f32(
        constant ds4_metal_args_glm_indexer_rope_tail & args,
        device char *x,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    if (args.rot_dim == 0u || args.rot_offset > args.head_dim ||
        args.rot_dim > args.head_dim - args.rot_offset || (args.rot_dim & 1u) != 0u) return;

    const uint nth = ntg_u.x;
    const uint pos = args.pos0 + token;
    device float *row =
        (device float *)(x +
            ((uint64_t)token * args.n_head + head) * args.head_dim * sizeof(float));
    row += args.rot_offset;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.rot_dim,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }
    const float theta_base = (float)pos;
    const float inv_ndims = -1.0f / (float)args.rot_dim;
    for (uint i = tid * 2u; i < args.rot_dim; i += nth * 2u) {
        const uint rel_i0 = i;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
        const float theta = theta_base * exp2(inv_ndims * (float)rel_i0 * log2(args.freq_base));
#else
        const float theta = theta_base * pow(args.freq_base, inv_ndims * (float)rel_i0);
#endif
        float cos_theta;
        float sin_theta;
        glm_rope_yarn(theta,
                      args.freq_scale,
                      corr_dims,
                      (int)rel_i0,
                      args.ext_factor,
                      args.attn_factor,
                      &cos_theta,
                      &sin_theta);
        const uint j = i + 1u;
        const float x0 = row[i];
        const float x1 = row[j];
        row[i] = x0 * cos_theta - x1 * sin_theta;
        row[j] = x0 * sin_theta + x1 * cos_theta;
    }
}

static inline float glm_cache_load_f32_or_f16(
        device const char *base,
        uint64_t index,
        uint cache_f16) {
    if (cache_f16 != 0u) {
        return (float)((device const half *)base)[index];
    }
    return ((device const float *)base)[index];
}

static inline float glm_cache_load_f16_only(
        device const char *base,
        uint64_t index) {
    return (float)((device const half *)base)[index];
}

static inline float2 glm_cache_load_rotated_rope_pair(
        device const char *base,
        uint64_t           rope_base,
        uint               r,
        uint               row,
        uint               qk_rope,
        uint               cache_f16,
        float              freq_base,
        float              freq_scale,
        float              ext_factor,
        float              attn_factor,
        float              corr0,
        float              corr1) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = theta_base * exp2(inv_ndims * (float)r * log2(freq_base));
#else
    const float theta = theta_base * pow(freq_base, inv_ndims * (float)r);
#endif
    float corr_dims[2] = {corr0, corr1};
    float cos_theta;
    float sin_theta;
    glm_rope_yarn(theta,
                  freq_scale,
                  corr_dims,
                  (int)r,
                  ext_factor,
                  attn_factor,
                  &cos_theta,
                  &sin_theta);
    const float x0 = glm_cache_load_f32_or_f16(base, rope_base + r, cache_f16);
    const float x1 = glm_cache_load_f32_or_f16(base, rope_base + r + 1u, cache_f16);
    return float2(x0 * cos_theta - x1 * sin_theta,
                  x0 * sin_theta + x1 * cos_theta);
}

static inline float2 glm_cache_load_rotated_rope_pair_f16_only(
        device const char *base,
        uint64_t           rope_base,
        uint               r,
        uint               row,
        uint               qk_rope,
        float              freq_base,
        float              freq_scale,
        float              ext_factor,
        float              attn_factor,
        float              corr0,
        float              corr1) {
    const float theta_base = (float)row;
    const float inv_ndims = -1.0f / (float)qk_rope;
#ifdef DS4_METAL_ROPE_EXP2_LOG2
    const float theta = theta_base * exp2(inv_ndims * (float)r * log2(freq_base));
#else
    const float theta = theta_base * pow(freq_base, inv_ndims * (float)r);
#endif
    float corr_dims[2] = {corr0, corr1};
    float cos_theta;
    float sin_theta;
    glm_rope_yarn(theta,
                  freq_scale,
                  corr_dims,
                  (int)r,
                  ext_factor,
                  attn_factor,
                  &cos_theta,
                  &sin_theta);
    const float x0 = glm_cache_load_f16_only(base, rope_base + r);
    const float x1 = glm_cache_load_f16_only(base, rope_base + r + 1u);
    return float2(x0 * cos_theta - x1 * sin_theta,
                  x0 * sin_theta + x1 * cos_theta);
}

static inline float glm_q8_0_weight_at(
        device const char *row,
        uint col) {
    const uint block = col >> 5;
    const uint qi = col & 31u;
    device const char *block_base = row + (uint64_t)block * 34u;
    const float d = (float)(*((device const half *)block_base));
    device const int8_t *qs = (device const int8_t *)(block_base + 2u);
    return d * (float)qs[qi];
}

static inline float glm_q8_0_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    const uint n_blocks = (n_cols + 31u) >> 5;
    for (uint block = 0; block < n_blocks; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        const uint count = min(32u, n_cols - base);
        for (uint qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static inline float glm_q8_0_dot_row_tg_f32_512(
        device const char *row,
        threadgroup const float *x) {
    float acc = 0.0f;
    for (uint block = 0; block < 16u; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static inline float glm_q8_0_dot_row_tg_f32_fast(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    if (n_cols == 512u) {
        return glm_q8_0_dot_row_tg_f32_512(row, x);
    }
    return glm_q8_0_dot_row_tg_f32(row, x, n_cols);
}

static inline float glm_q8_0_dot_row_dev_f32(
        device const char *row,
        device const float *x,
        uint n_cols) {
    float acc = 0.0f;
    const uint n_blocks = (n_cols + 31u) >> 5;
    for (uint block = 0; block < n_blocks; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        const uint count = min(32u, n_cols - base);
        for (uint qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

#define DS4_METAL_GGUF_Q4_0 2u
#define DS4_METAL_GGUF_Q8_0 8u
#define DS4_METAL_GGUF_Q4_K 12u

static inline uchar2 glm_q4_K_scale_min(int j, int k, device const uchar *q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0x0f) | ((q[j - 4 + k] & 0xc0) >> 2)),
                          uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

static inline float glm_q4_0_weight_at(device const char *row, uint col) {
    const uint block = col >> 5;
    const uint qi = col & 31u;
    device const char *block_base = row + (uint64_t)block * 18u;
    const float d = (float)(*((device const half *)block_base));
    device const uchar *qs = (device const uchar *)(block_base + 2u);
    /* ggml Q4_0: elems 0..15 = low nibbles of qs[0..15], 16..31 = high. */
    const uchar packed = qs[qi & 15u];
    const uchar q = (qi < 16u) ? (packed & 0x0f) : (packed >> 4);
    return d * ((float)q - 8.0f);
}

static inline float glm_q4_K_weight_at(device const char *row, uint col) {
    const uint block = col >> 8u;
    const uint idx = col & 255u;
    device const char *block_base = row + (uint64_t)block * 144u;
    const float d = (float)(*((device const half *)(block_base + 0u)));
    const float dmin = (float)(*((device const half *)(block_base + 2u)));
    device const uchar *scales = (device const uchar *)(block_base + 4u);
    device const uchar *qs = (device const uchar *)(block_base + 16u);
    const uint group = idx >> 5u;
    const uint l = idx & 31u;
    const uchar2 sm = glm_q4_K_scale_min((int)group, 0, scales);
    const uint byte_off = (group >> 1u) * 32u + l;
    const uint shift = (group & 1u) * 4u;
    const uint q = ((uint)qs[byte_off] >> shift) & 0x0fu;
    return d * (float)sm.x * (float)q - dmin * (float)sm.y;
}

static inline float glm_quant_weight_at(
        uint weight_type,
        device const char *row,
        uint col) {
    if (weight_type == DS4_METAL_GGUF_Q4_0) return glm_q4_0_weight_at(row, col);
    if (weight_type == DS4_METAL_GGUF_Q4_K) return glm_q4_K_weight_at(row, col);
    return glm_q8_0_weight_at(row, col);
}

static inline float glm_q4_0_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_q4_0_weight_at(row, col) * x[col];
    }
    return acc;
}

static inline float glm_q4_K_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_q4_K_weight_at(row, col) * x[col];
    }
    return acc;
}

static inline float glm_quant_dot_row_tg_f32(
        uint weight_type,
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    if (weight_type == DS4_METAL_GGUF_Q4_0) return glm_q4_0_dot_row_tg_f32(row, x, n_cols);
    if (weight_type == DS4_METAL_GGUF_Q4_K) return glm_q4_K_dot_row_tg_f32(row, x, n_cols);
    return glm_q8_0_dot_row_tg_f32_fast(row, x, n_cols);
}

/* Per-lane Q4_K row dot: lane l covers elements (g*32 + l) of every
 * 32-group so the 144-byte superblocks are read with coalesced per-lane
 * bytes; callers simd_sum the result. x lives in threadgroup memory. */
static inline float glm_q4_K_dot_row_lane_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols,
        ushort lane) {
    float acc = 0.0f;
    const uint nblocks = n_cols >> 8u;
    for (uint b = 0; b < nblocks; b++) {
        device const char *block_base = row + (uint64_t)b * 144u;
        const float d = (float)(*((device const half *)(block_base + 0u)));
        const float dmin = (float)(*((device const half *)(block_base + 2u)));
        device const uchar *scales = (device const uchar *)(block_base + 4u);
        device const uchar *qs = (device const uchar *)(block_base + 16u);
        threadgroup const float *xb = x + (b << 8u);
        FOR_UNROLL (uint g = 0; g < 8u; g++) {
            const uchar2 sm = glm_q4_K_scale_min((int)g, 0, scales);
            const uint byte_off = (g >> 1u) * 32u + lane;
            const uint shift = (g & 1u) * 4u;
            const uint q = ((uint)qs[byte_off] >> shift) & 0x0fu;
            const float xv = xb[(g << 5u) + lane];
            acc += (d * (float)sm.x * (float)q - dmin * (float)sm.y) * xv;
        }
    }
    return acc;
}

static inline float glm_quant_dot_row_dev_f32(
        uint weight_type,
        device const char *row,
        device const float *x,
        uint n_cols) {
    if (weight_type == DS4_METAL_GGUF_Q8_0) return glm_q8_0_dot_row_dev_f32(row, x, n_cols);
    float acc = 0.0f;
    for (uint col = 0; col < n_cols; col++) {
        acc += glm_quant_weight_at(weight_type, row, col) * x[col];
    }
    return acc;
}

kernel void kernel_glm_indexer_score_one(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint row = tgpig.x;
    if (row >= args.n_rows) return;
    const uint nth = ntg_u.x;
    float score = 0.0f;
    for (uint h = 0; h < args.n_head; h++) {
        float partial = 0.0f;
        device const float *qh =
            (device const float *)(q + (uint64_t)h * args.head_dim * sizeof(float));
        for (uint d = tid; d < args.head_dim; d += nth) {
            const float k = glm_cache_load_f32_or_f16(indexer_key_cache,
                                                      (uint64_t)row * args.head_dim + d,
                                                      args.cache_f16);
            partial += qh[d] * k;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            score += max(scratch[0] * args.scale, 0.0f) * weights[h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) scores[row] = score;
}

kernel void kernel_glm_indexer_score_one_direct(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= args.n_rows || args.n_head != 32u || args.head_dim != 128u) {
        return;
    }

    threadgroup float *ktg = shared;
    threadgroup float *psum = ktg + 128u;

    if (tid < 128u) {
        ktg[tid] = glm_cache_load_f32_or_f16(indexer_key_cache,
                                             (uint64_t)row * 128u + tid,
                                             args.cache_f16);
    }

    float acc = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head0 = 0; head0 < 32u; head0 += 4u) {
        const uint head = head0 + (uint)sg;
        device const float4 *q4 = (device const float4 *)(q +
            (uint64_t)head * 128u * sizeof(float));
        threadgroup const float4 *k4 = (threadgroup const float4 *)ktg;

        float s = dot(q4[lane], k4[lane]);
        s = simd_sum(s);
        if (lane == 0) {
            psum[sg] = max(s * args.scale, 0.0f) * weights[head];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            acc += psum[0];
            acc += psum[1];
            acc += psum[2];
            acc += psum[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        scores[row] = acc;
    }
}

kernel void kernel_glm_indexer_scores_batch(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint row = tgpig.x;
    const uint token = tgpig.y;
    if (row >= args.n_rows || token >= args.n_tokens) return;

    device float *dst = (device float *)(scores +
        (uint64_t)token * args.score_token_stride) + row;
    const uint visible = glm_indexer_batch_visible_rows(args, token);
    if (row >= visible) {
        if (tid == 0) *dst = -INFINITY;
        return;
    }

    const uint nth = ntg_u.x;
    float score = 0.0f;
    for (uint h = 0; h < args.n_head; h++) {
        float partial = 0.0f;
        device const float *qh = (device const float *)(q +
            (uint64_t)token * args.q_token_stride +
            (uint64_t)h     * args.q_head_stride);
        for (uint d = tid; d < args.head_dim; d += nth) {
            const float k = glm_cache_load_f32_or_f16(indexer_key_cache,
                                                      (uint64_t)row * args.head_dim + d,
                                                      args.cache_f16);
            partial += qh[d] * k;
        }
        scratch[tid] = partial;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) scratch[tid] += scratch[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token * args.weights_token_stride);
            score += max(scratch[0] * args.scale, 0.0f) * w[h];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) *dst = score;
}

kernel void kernel_glm_indexer_scores_tiled_f32(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint row_base = tgpig.x * TN;
    const uint token_base = tgpig.y * TM;

    threadgroup float *qtg = shared;
    threadgroup float *ktg = qtg + TM*D;
    threadgroup float *dot = ktg + TN*D;

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        float v = 0.0f;
        if (row < args.n_rows) {
            v = glm_cache_load_f32_or_f16(indexer_key_cache,
                                          (uint64_t)row * args.head_dim + d,
                                          args.cache_f16);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint tr = i / D;
            const uint d = i - tr*D;
            const uint token = token_base + tr;
            float v = 0.0f;
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = qrow[d];
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && row0 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[token_row0*TN + col0];
            acc0 += max(s * args.scale, 0.0f) * w[head];
        }
        if (token1 < args.n_tokens && row1 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[token_row1*TN + col1];
            acc1 += max(s * args.scale, 0.0f) * w[head];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_glm_indexer_scores_tiled(
        constant ds4_metal_args_glm_indexer_scores_batch & args,
        device const char *q,
        device const char *weights,
        device const char *indexer_key_cache,
        device char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint row_base = tgpig.x * TN;
    const uint token_base = tgpig.y * TM;

    threadgroup half *qtg = (threadgroup half *)shared;
    threadgroup half *ktg = qtg + TM*D;
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D);

    const uint last_token = min(token_base + TM, args.n_tokens);
    const uint max_visible = last_token > token_base ?
        glm_indexer_batch_visible_rows(args, last_token - 1u) : 0u;

    if (row_base >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint tr = i / TN;
            const uint rc = i - tr*TN;
            const uint token = token_base + tr;
            const uint row = row_base + rc;
            if (token < args.n_tokens && row < args.n_rows) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + row;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint rc = i / D;
        const uint d = i - rc*D;
        const uint row = row_base + rc;
        half v = half(0.0f);
        if (row < args.n_rows) {
            v = half(glm_cache_load_f32_or_f16(indexer_key_cache,
                                               (uint64_t)row * args.head_dim + d,
                                               args.cache_f16));
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint token_row0 = cell0 >> 3;
    const uint token_row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = token_base + token_row0;
    const uint token1 = token_base + token_row1;
    const uint row0 = row_base + col0;
    const uint row1 = row_base + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint tr = i / D;
            const uint d = i - tr*D;
            const uint token = token_base + tr;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && row0 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[token_row0*TN + col0];
            acc0 += max(s * args.scale, 0.0f) * w[head];
        }
        if (token1 < args.n_tokens && row1 < args.n_rows) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[token_row1*TN + col1];
            acc1 += max(s * args.scale, 0.0f) * w[head];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && row0 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token0);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + row0;
        *dst = row0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && row1 < args.n_rows) {
        const uint visible = glm_indexer_batch_visible_rows(args, token1);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + row1;
        *dst = row1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_glm_qk_lowrank_q8_0(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head) return;
    const uint nth = ntg_u.x;
    device const float *qh =
        (device const float *)(q + (uint64_t)head * args.qk_dim * sizeof(float));
    device float *out =
        (device float *)(qk_low + (uint64_t)head * args.kv_lora_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
        out[j] = glm_quant_dot_row_dev_f32(args.weight_type, row, qh, args.qk_nope);
    }
}

kernel void kernel_glm_qk_lowrank_q8_0_glm52(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint n_head = 64u;
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 192u;
    constexpr uint qk_dim = 256u;
    constexpr uint row_bytes = 204u;

    const uint head = tgpig.x;
    if (head >= n_head ||
        args.n_head != n_head ||
        args.kv_lora_dim != kv_lora_dim ||
        args.qk_nope != qk_nope ||
        args.qk_dim != qk_dim ||
        args.row_bytes != row_bytes ||
        args.weight_type != DS4_METAL_GGUF_Q8_0) {
        return;
    }
    const uint nth = ntg_u.x;
    device const float *qh =
        (device const float *)(q + (uint64_t)head * qk_dim * sizeof(float));
    for (uint d = tid; d < qk_nope; d += nth) {
        x[d] = qh[d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(qk_low + (uint64_t)head * kv_lora_dim * sizeof(float));
    for (uint j = tid; j < kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc = 0.0f;
        for (uint block = 0; block < 6u; block++) {
            device const char *block_base = row + (uint64_t)block * 34u;
            const float d = (float)(*((device const half *)block_base));
            device const int8_t *qs = (device const int8_t *)(block_base + 2u);
            const uint base = block << 5;
            FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
                const uint col = base + qi;
                acc += d * (float)qs[qi] * x[col];
            }
        }
        out[j] = acc;
    }
}

// Coalesced GLM 5.2 decode qk-low: one simdgroup per pair of output rows,
// lanes split the 192-wide dot so the 204-byte Q8 rows are read with
// consecutive per-lane bytes. The thread-per-row variant above issues
// strided scalar byte loads from only 64 threadgroups and measures ~7.5x
// off the weight-bandwidth floor.
kernel void kernel_glm_qk_lowrank_q8_0_glm52_sg(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 192u;
    constexpr uint qk_dim = 256u;
    constexpr uint NR = 2u;

    const uint head = tgpig.x;
    const uint wt = args.weight_type;
    if (head >= args.n_head ||
        (args.n_head != 32u && args.n_head != 64u) ||
        args.kv_lora_dim != kv_lora_dim ||
        args.qk_nope != qk_nope ||
        args.qk_dim != qk_dim ||
        !((wt == DS4_METAL_GGUF_Q8_0 && args.row_bytes == 204u) ||
          (wt == DS4_METAL_GGUF_Q4_0 && args.row_bytes == 108u))) {
        return;
    }
    const uint row_bytes = args.row_bytes;

    const uint nsg = ntg_u.y;
    const uint row0 = (tgpig.y * nsg + (uint)sgitg) * NR;
    if (row0 >= kv_lora_dim) return;

    device const float *qh =
        (device const float *)(q + (uint64_t)head * qk_dim * sizeof(float));
    float qv[6];
    FOR_UNROLL (uint b = 0; b < 6u; b++) {
        qv[b] = qh[(b << 5) + tiisg];
    }

    device float *out =
        (device float *)(qk_low + (uint64_t)head * kv_lora_dim * sizeof(float));
    for (uint r = 0; r < NR; r++) {
        const uint j = row0 + r;
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc = 0.0f;
        if (wt == DS4_METAL_GGUF_Q8_0) {
            FOR_UNROLL (uint b = 0; b < 6u; b++) {
                device const char *block_base = row + (uint64_t)b * 34u;
                const float d = (float)(*((device const half *)block_base));
                device const int8_t *qs = (device const int8_t *)(block_base + 2u);
                acc += d * (float)qs[tiisg] * qv[b];
            }
        } else {
            /* Q4_0: 18B blocks; elems 0..15 = low nibbles, 16..31 = high. */
            FOR_UNROLL (uint b = 0; b < 6u; b++) {
                device const char *block_base = row + (uint64_t)b * 18u;
                const float d = (float)(*((device const half *)block_base));
                device const uint8_t *qs = (device const uint8_t *)(block_base + 2u);
                const uint byte = qs[tiisg & 15u];
                const float v = (float)((tiisg < 16u) ? (byte & 0xFu) : (byte >> 4)) - 8.0f;
                acc += d * v * qv[b];
            }
        }
        const float sum = simd_sum(acc);
        if (tiisg == 0) {
            out[j] = sum;
        }
    }
}

kernel void kernel_glm_qk_lowrank_q8_0_batch(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x + args.head_base;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_dim;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head * qk_dim * sizeof(float));
    device float *out =
        (device float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.kv_lora_dim + j) * args.row_bytes;
        out[j] = glm_quant_dot_row_dev_f32(args.weight_type, row, qh, args.qk_nope);
    }
}

kernel void kernel_glm_qk_lowrank_q8_0_batch_glm52_t4(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *q,
        device char *qk_low,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint n_head = 64u;
    constexpr uint kv_lora_dim = 512u;
    constexpr uint qk_nope = 192u;
    constexpr uint qk_dim = 256u;
    constexpr uint tile_tokens = 4u;
    constexpr uint row_bytes = 204u;

    const uint head = tgpig.x + args.head_base;
    const uint token0 = tgpig.y * tile_tokens;
    const uint nth = ntg_u.x;
    const uint64_t q_token_stride = (uint64_t)n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)n_head * kv_lora_dim * sizeof(float);

    for (uint t = 0; t < tile_tokens; t++) {
        const uint token = token0 + t;
        threadgroup float *xt = x + t * qk_nope;
        if (token < args.n_tokens) {
            device const float *qh =
                (device const float *)(q +
                    (uint64_t)token * q_token_stride +
                    (uint64_t)head * qk_dim * sizeof(float));
            for (uint d = tid; d < qk_nope; d += nth) {
                xt[d] = qh[d];
            }
        } else {
            for (uint d = tid; d < qk_nope; d += nth) {
                xt[d] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < kv_lora_dim; j += nth) {
        device const char *row =
            weight + ((uint64_t)head * kv_lora_dim + j) * row_bytes;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        float acc2 = 0.0f;
        float acc3 = 0.0f;
        for (uint block = 0; block < 6u; block++) {
            device const char *block_base = row + (uint64_t)block * 34u;
            const float d = (float)(*((device const half *)block_base));
            device const int8_t *qs = (device const int8_t *)(block_base + 2u);
            const uint base = block << 5;
            FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
                const uint col = base + qi;
                const float wq = d * (float)qs[qi];
                acc0 += wq * x[col];
                acc1 += wq * x[qk_nope + col];
                acc2 += wq * x[2u * qk_nope + col];
                acc3 += wq * x[3u * qk_nope + col];
            }
        }

        if (token0 < args.n_tokens) {
            device float *out0 =
                (device float *)(qk_low +
                    (uint64_t)token0 * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out0[j] = acc0;
        }
        if (token0 + 1u < args.n_tokens) {
            device float *out1 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 1u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out1[j] = acc1;
        }
        if (token0 + 2u < args.n_tokens) {
            device float *out2 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 2u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out2[j] = acc2;
        }
        if (token0 + 3u < args.n_tokens) {
            device float *out3 =
                (device float *)(qk_low +
                    (uint64_t)(token0 + 3u) * low_token_stride +
                    (uint64_t)head * kv_lora_dim * sizeof(float));
            out3[j] = acc3;
        }
    }
}

kernel void kernel_glm_value_project_q8_0(
        constant ds4_metal_args_glm_qk_lowrank & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head) return;
    const uint nth = ntg_u.x;
    device const float *src =
        (device const float *)(lora + (uint64_t)head * args.kv_lora_dim * sizeof(float));
    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        x[j] = src[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.qk_dim * sizeof(float));
    for (uint d = tid; d < args.qk_dim; d += nth) {
        device const char *row =
            weight + ((uint64_t)head * args.qk_dim + d) * args.row_bytes;
        out[d] = glm_quant_dot_row_tg_f32(args.weight_type, row, x, args.kv_lora_dim);
    }
}

kernel void kernel_glm_value_project_q8_0_batch_heads(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup float *x [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x + args.head_base;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens) return;
    const uint nth = ntg_u.x;
    const uint value_dim = args.qk_dim;
    const uint64_t lora_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride =
        (uint64_t)args.n_head * value_dim * sizeof(float);
    device const float *src =
        (device const float *)(lora +
            (uint64_t)token * lora_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));
    device float *out =
        (device float *)(heads +
            (uint64_t)token * heads_token_stride +
            (uint64_t)head * value_dim * sizeof(float));

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        x[j] = src[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < value_dim; d += nth) {
        device const char *row =
            weight + ((uint64_t)head * value_dim + d) * args.row_bytes;
        out[d] = glm_quant_dot_row_tg_f32(args.weight_type, row, x, args.kv_lora_dim);
    }
}

kernel void kernel_glm_value_project_q8_0_batch_heads_mma(
        constant ds4_metal_args_glm_qk_lowrank_batch & args,
        device const char *weight,
        device const char *lora,
        device char *heads,
        threadgroup char *shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint NR0 = 64u;
    constexpr uint NR1 = 32u;
    constexpr uint NK = 32u;
    constexpr uint NL0 = 2u;
    constexpr uint NL1 = 4u;

    const uint token0 = tgpig.x * NR1;
    const uint value0 = tgpig.y * NR0;
    const uint head = tgpig.z + args.head_base;
    if (head >= args.n_head || token0 >= args.n_tokens || value0 >= args.qk_dim) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)shmem;
    threadgroup half *sb = (threadgroup half *)(shmem + 4096u);

    const uint nr0 = min(NR0, args.qk_dim - value0);
    const uint nr1 = min(NR1, args.n_tokens - token0);

    const uint lr0 = min((uint)tid / NL0, nr0 - 1u);
    const uint lr1 = min((uint)tid / NL1, nr1 - 1u);
    const uint il0 = (uint)tid & 1u;
    const uint iy = 8u * ((uint)tid & (NL1 - 1u));

    const uint64_t lora_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride =
        (uint64_t)args.n_head * args.qk_dim * sizeof(float);
    const uint64_t head_lora_base =
        (uint64_t)head * args.kv_lora_dim * sizeof(float);
    const uint64_t head_out_base =
        (uint64_t)head * args.qk_dim * sizeof(float);

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (uint i = 0; i < 8u; i++) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.kv_lora_dim; loop_k += NK) {
        const uint value = value0 + lr0;
        const uint block = loop_k >> 5;
        device const char *row =
            weight + ((uint64_t)head * args.qk_dim + value) * args.row_bytes;
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < 16u; i++) {
            const uint k = loop_k + 16u * il0 + i;
            const uint sx = 2u * il0 + i / 8u;
            const uint sy = ((uint)tid / NL0) / 8u;
            const uint lx = ((uint)tid / NL0) & 7u;
            const uint ly = i & 7u;
            const uint ib = 8u * sx + sy;
            const half v = (value < args.qk_dim && k < args.kv_lora_dim) ?
                half(d * (float)qs[16u * il0 + i]) :
                half(0.0f);
            *(sa + 64u * ib + 8u * ly + lx) = v;
        }

        const uint token = token0 + lr1;
        device const float *y =
            (device const float *)(lora +
                (uint64_t)token * lora_token_stride +
                head_lora_base +
                (uint64_t)loop_k * sizeof(float) +
                (uint64_t)iy * sizeof(float));
        for (uint i = 0; i < 8u; i++) {
            const uint k = loop_k + iy + i;
            const uint sx = ((uint)tid) & (NL1 - 1u);
            const uint sy = ((uint)tid / NL1) / 8u;
            const uint lx = i;
            const uint ly = ((uint)tid / NL1) & 7u;
            const uint ib = 4u * sx + sy;
            const half v = (token < args.n_tokens && k < args.kv_lora_dim) ?
                half(y[i]) :
                half(0.0f);
            *(sb + 64u * ib + 8u * ly + lx) = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half *lsma = sa + 4u * 64u * ((uint)sg & 1u);
        threadgroup const half *lsmb = sb + 2u * 64u * ((uint)sg >> 1);

        for (uint ik = 0; ik < NK / 8u; ik++) {
            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 4u; i++) {
                simdgroup_load(ma[i], lsma + 64u * i, 8u, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 2u; i++) {
                simdgroup_load(mb[i], lsmb + 64u * i, 8u, 0, false);
            }

            simdgroup_barrier(mem_flags::mem_none);

            for (uint i = 0; i < 8u; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4u], ma[i & 3u], mc[i]);
            }

            lsma += 8u * 64u;
            lsmb += 4u * 64u;
        }
    }

    if (nr0 == NR0 && nr1 == NR1) {
        device float *dst =
            (device float *)(heads +
                (uint64_t)(token0 + 16u * ((uint)sg >> 1)) * heads_token_stride +
                head_out_base +
                (uint64_t)(value0 + 32u * ((uint)sg & 1u)) * sizeof(float));
        for (uint i = 0; i < 8u; i++) {
            simdgroup_store(mc[i],
                            dst + 8u * (i & 3u) + 8u * (heads_token_stride / sizeof(float)) * (i / 4u),
                            heads_token_stride / sizeof(float),
                            0,
                            false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float *tmp = (threadgroup float *)shmem;
        for (uint i = 0; i < 8u; i++) {
            simdgroup_store(mc[i],
                            tmp + 32u * ((uint)sg & 1u) +
                                  16u * ((uint)sg >> 1) * NR0 +
                                  8u * (i & 3u) + 8u * NR0 * (i / 4u),
                            NR0,
                            0,
                            false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            for (uint t = tid; t < nr1; t += 128u) {
                device float *dst =
                    (device float *)(heads +
                        (uint64_t)(token0 + t) * heads_token_stride +
                        head_out_base +
                        (uint64_t)value0 * sizeof(float));
                threadgroup const float *src = tmp + t * NR0;
                for (uint v = 0; v < nr0; v++) {
                    dst[v] = src[v];
                }
            }
        }
    }
}

template <bool assume_valid_rows, bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_decode_split_group8_partial_impl(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *partial_lora,
        device char *partial_ms,
        threadgroup half4 *scratch [[threadgroup(0)]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group;
    const uint block = tgpig.y;
    if (args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        args.qk_rope != 64u ||
        args.block_rows == 0u ||
        block >= args.n_blocks) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint block_start = block * args.block_rows;
    const uint block_end = min(args.n_selected, block_start + args.block_rows);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q + (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = block_start; base < block_end; base += stage_rows) {
        const uint rows = min(stage_rows, block_end - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                const float2 y0 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                const float2 y1 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r + 2u,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
            } else {
                rope_shared[off] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        device float4 *out4 =
            (device float4 *)(partial_lora +
                ((uint64_t)block * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0;
        out4[lane + 32u] = o1;
        out4[lane + 64u] = o2;
        out4[lane + 96u] = o3;
        if (lane == 0u) {
            device float *ms =
                (device float *)(partial_ms +
                    ((uint64_t)block * args.n_head + head) * 2u * sizeof(float));
            ms[0] = M;
            ms[1] = S;
        }
    }
}

typedef decltype(kernel_glm_attention_indexed_decode_split_group8_partial_impl<false, false>)
        glm_attention_indexed_decode_split_group8_partial_t;

template [[host_name("kernel_glm_attention_indexed_decode_split_group8_partial")]]
kernel glm_attention_indexed_decode_split_group8_partial_t
kernel_glm_attention_indexed_decode_split_group8_partial_impl<false, false>;

template [[host_name("kernel_glm_attention_indexed_decode_split_group8_partial_valid_fullheads")]]
kernel glm_attention_indexed_decode_split_group8_partial_t
kernel_glm_attention_indexed_decode_split_group8_partial_impl<true, true>;

template<uint FIXED_BLOCKS>
static void kernel_glm_attention_indexed_decode_split_group8_reduce_impl(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint n_blocks = FIXED_BLOCKS != 0u ? FIXED_BLOCKS : args.n_blocks;
    if (head >= args.n_head ||
        args.n_selected == 0u ||
        args.kv_lora_dim != 512u ||
        n_blocks == 0u ||
        n_blocks > 64u ||
        (FIXED_BLOCKS != 0u && args.n_blocks != FIXED_BLOCKS)) {
        return;
    }

    const uint nth = ntg_u.x;
    threadgroup float *red = scratch;
    threadgroup float *block_scale = scratch + 256u;
    threadgroup float *lora_sum = scratch + 320u;

    float local_m = -FLT_MAX / 2.0f;
    if (tid < n_blocks) {
        device const float *ms =
            (device const float *)(partial_ms +
                ((uint64_t)tid * args.n_head + head) * 2u * sizeof(float));
        local_m = ms[1] > 0.0f ? ms[0] : -FLT_MAX / 2.0f;
    }
    red[tid] = local_m;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_denom = 0.0f;
    if (tid < n_blocks) {
        device const float *ms =
            (device const float *)(partial_ms +
                ((uint64_t)tid * args.n_head + head) * 2u * sizeof(float));
        const float s = ms[1];
        const float e = s > 0.0f ? exp(ms[0] - max_m) : 0.0f;
        block_scale[tid] = e;
        local_denom = s * e;
    }
    red[tid] = local_denom;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint b = 0u; b < n_blocks; b++) {
            device const float *src =
                (device const float *)(partial_lora +
                    ((uint64_t)b * args.n_head + head) *
                        args.kv_lora_dim * sizeof(float));
            acc += src[j] * block_scale[b];
        }
        lora_sum[j] = acc / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_decode_split_group8_reduce(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    kernel_glm_attention_indexed_decode_split_group8_reduce_impl<0>(
            args, partial_lora, partial_ms, value_weight, heads, scratch,
            tid, ntg_u, tgpig);
}

kernel void kernel_glm_attention_indexed_decode_split_group8_reduce16(
        constant ds4_metal_args_glm_attention_indexed_decode_split & args,
        device const char *partial_lora,
        device const char *partial_ms,
        device const char *value_weight,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    kernel_glm_attention_indexed_decode_split_group8_reduce_impl<16>(
            args, partial_lora, partial_ms, value_weight, heads, scratch,
            tid, ntg_u, tgpig);
}

kernel void kernel_glm_attention_indexed_decode(
        constant ds4_metal_args_glm_attention_indexed_decode & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    if (head >= args.n_head || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;
    threadgroup float *lora_sum = scores + args.n_selected;

    device const float *qh =
        (device const float *)(q + (uint64_t)head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low + (uint64_t)head * args.kv_lora_dim * sizeof(float));

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    if (args.cache_f16 != 0u) {
        float local_max = -INFINITY;
        for (uint s = tid; s < args.n_selected; s += nth) {
            const uint row = selected[s];
            float score = -INFINITY;
            if (row < args.cache_cap) {
                float dotv = 0.0f;
                const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
                uint j = 0;
                for (; j + 3u < args.kv_lora_dim; j += 4u) {
                    device const half4 *kv4 =
                        (device const half4 *)((device const half *)kv_lora_cache + lora_base + j);
                    device const float4 *low4 =
                        (device const float4 *)(low + j);
                    const float4 kv = (float4)(*kv4);
                    const float4 qv = *low4;
                    dotv += qv.x * kv.x + qv.y * kv.y +
                            qv.z * kv.z + qv.w * kv.w;
                }
                if (j < args.kv_lora_dim) {
                    for (; j < args.kv_lora_dim; j++) {
                        const float kv = glm_cache_load_f16_only(kv_lora_cache,
                                                                 lora_base + j);
                        dotv += low[j] * kv;
                    }
                }
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                for (uint r = 0; r < args.qk_rope; r += 2u) {
                    const float2 y = glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                                                rope_base,
                                                                                r,
                                                                                row,
                                                                                args.qk_rope,
                                                                                args.freq_base,
                                                                                args.freq_scale,
                                                                                args.ext_factor,
                                                                                args.attn_factor,
                                                                                corr_dims[0],
                                                                                corr_dims[1]);
                    dotv += qh[args.qk_nope + r] * y.x +
                            qh[args.qk_nope + r + 1u] * y.y;
                }
                score = dotv * args.scale;
            }
            scores[s] = score;
            local_max = max(local_max, score);
        }
        red[tid] = local_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] = max(red[tid], red[tid + step]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float max_score = red[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float local_sum = 0.0f;
        for (uint s = tid; s < args.n_selected; s += nth) {
            const float w = exp(scores[s] - max_score);
            scores[s] = w;
            local_sum += w;
        }
        red[tid] = local_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = nth >> 1; step > 0; step >>= 1) {
            if (tid < step) red[tid] += red[tid + step];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float denom = max(red[0], 1.0e-20f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j0 = tid * 2u; j0 < args.kv_lora_dim; j0 += nth * 2u) {
            const uint j1 = j0 + 1u;
            const bool use_j1 = j1 < args.kv_lora_dim;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (uint s = 0; s < args.n_selected; s++) {
                const uint row = selected[s];
                if (row < args.cache_cap) {
                    const uint64_t row_base = (uint64_t)row * args.kv_lora_dim;
                    const float w = scores[s];
                    if (use_j1) {
                        device const half2 *kv2 =
                            (device const half2 *)((device const half *)kv_lora_cache + row_base + j0);
                        const float2 kv = (float2)(*kv2);
                        acc0 += w * kv.x;
                        acc1 += w * kv.y;
                    } else {
                        const float kv0 = glm_cache_load_f16_only(kv_lora_cache,
                                                                  row_base + j0);
                        acc0 += w * kv0;
                    }
                }
            }
            lora_sum[j0] = acc0 / denom;
            if (use_j1) lora_sum[j1] = acc1 / denom;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        device float *out =
            (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
        return;
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = selected[s];
        float score = -INFINITY;
        if (row < args.cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dotv += low[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dotv += qh[args.qk_nope + r] * y.x +
                        qh[args.qk_nope + r + 1u] * y.y;
            }
            score = dotv * args.scale;
        }
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j0 = tid; j0 < args.kv_lora_dim; j0 += nth * 2u) {
        const uint j1 = j0 + nth;
        const bool use_j1 = j1 < args.kv_lora_dim;
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = selected[s];
            if (row < args.cache_cap) {
                const uint64_t row_base = (uint64_t)row * args.kv_lora_dim;
                const float w = scores[s];
                const float kv0 = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                            row_base + j0,
                                                            args.cache_f16);
                acc0 += w * kv0;
                if (use_j1) {
                    const float kv1 = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                                row_base + j1,
                                                                args.cache_f16);
                    acc1 += w * kv1;
                }
            }
        }
        lora_sum[j0] = acc0 / denom;
        if (use_j1) lora_sum[j1] = acc1 / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads + (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint head = tgpig.x;
    const uint token = tgpig.y;
    if (head >= args.n_head || token >= args.n_tokens || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);
    threadgroup float *red = scratch;
    threadgroup float *scores = scratch + 256u;
    threadgroup float *lora_sum = scores + args.n_selected;

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float local_max = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = token_selected[s];
        float score = -INFINITY;
        if (row < args.cache_cap) {
            float dotv = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dotv += low[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dotv += qh[args.qk_nope + r] * y.x +
                        qh[args.qk_nope + r + 1u] * y.y;
            }
            score = dotv * args.scale;
        }
        scores[s] = score;
        local_max = max(local_max, score);
    }
    red[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] = max(red[tid], red[tid + step]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w = exp(scores[s] - max_score);
        scores[s] = w;
        local_sum += w;
    }
    red[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) red[tid] += red[tid + step];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom = max(red[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = token_selected[s];
            if (row < args.cache_cap) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           (uint64_t)row * args.kv_lora_dim + j,
                                                           args.cache_f16);
                acc += scores[s] * kv;
            }
        }
        lora_sum[j] = acc / denom;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out =
        (device float *)(heads +
            (uint64_t)token * heads_token_stride +
            (uint64_t)head * args.value_dim * sizeof(float));
    if (args.value_type == DS4_METAL_GGUF_Q4_K &&
        (args.kv_lora_dim & 255u) == 0u) {
        /* Lane-split Q4_K value project: one simdgroup per output row with
         * coalesced per-lane superblock reads; the per-thread scalar
         * fallback below walks the 144-byte rows one element at a time. */
        const uint vp_sg = tid >> 5u;
        const uint vp_lane = tid & 31u;
        const uint vp_nsg = nth >> 5u;
        for (uint d = vp_sg; d < args.value_dim; d += vp_nsg) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            const float part = glm_q4_K_dot_row_lane_f32(row, lora_sum,
                                                         args.kv_lora_dim,
                                                         (ushort)vp_lane);
            const float sum = simd_sum(part);
            if (vp_lane == 0u) {
                out[d] = sum;
            }
        }
    } else {
        for (uint d = tid; d < args.value_dim; d += nth) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, lora_sum, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch_group2(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort3 ntg_u [[threads_per_threadgroup]],
        uint3 tgpig [[threadgroup_position_in_grid]]) {
    const uint token = tgpig.y;
    if (token >= args.n_tokens || args.n_selected == 0u) return;
    const uint nth = ntg_u.x;
    const uint head0 = tgpig.x * 2u;
    const uint head1 = head0 + 1u;
    const bool valid0 = head0 < args.n_head;
    const bool valid1 = head1 < args.n_head;
    if (!valid0 && !valid1) return;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    threadgroup float *red0 = scratch;
    threadgroup float *red1 = red0 + 256u;
    threadgroup float *scores0 = red1 + 256u;
    threadgroup float *scores1 = scores0 + args.n_selected;
    threadgroup float *lora0 = scores1 + args.n_selected;
    threadgroup float *lora1 = lora0 + args.kv_lora_dim;

    device const float *qh0 =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)head0 * qk_dim * sizeof(float));
    device const float *qh1 =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)(valid1 ? head1 : head0) * qk_dim * sizeof(float));
    device const float *low0 =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)head0 * args.kv_lora_dim * sizeof(float));
    device const float *low1 =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)(valid1 ? head1 : head0) * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const uint row = token_selected[s];
        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (row < args.cache_cap) {
            float dot0 = 0.0f;
            float dot1 = 0.0f;
            const uint64_t lora_base = (uint64_t)row * args.kv_lora_dim;
            for (uint j = 0; j < args.kv_lora_dim; j++) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           lora_base + j,
                                                           args.cache_f16);
                dot0 += low0[j] * kv;
                if (valid1) dot1 += low1[j] * kv;
            }
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            for (uint r = 0; r < args.qk_rope; r += 2u) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   rope_base,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                dot0 += qh0[args.qk_nope + r] * y.x +
                        qh0[args.qk_nope + r + 1u] * y.y;
                if (valid1) {
                    dot1 += qh1[args.qk_nope + r] * y.x +
                            qh1[args.qk_nope + r + 1u] * y.y;
                }
            }
            score0 = dot0 * args.scale;
            if (valid1) score1 = dot1 * args.scale;
        }
        scores0[s] = score0;
        scores1[s] = score1;
        local_max0 = max(local_max0, score0);
        local_max1 = max(local_max1, score1);
    }
    red0[tid] = local_max0;
    red1[tid] = local_max1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red0[tid] = max(red0[tid], red0[tid + step]);
            red1[tid] = max(red1[tid], red1[tid + step]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float max_score0 = red0[0];
    const float max_score1 = red1[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum0 = 0.0f;
    float local_sum1 = 0.0f;
    for (uint s = tid; s < args.n_selected; s += nth) {
        const float w0 = (max_score0 > -INFINITY) ? exp(scores0[s] - max_score0) : 0.0f;
        const float w1 = (valid1 && max_score1 > -INFINITY) ? exp(scores1[s] - max_score1) : 0.0f;
        scores0[s] = w0;
        scores1[s] = w1;
        local_sum0 += w0;
        local_sum1 += w1;
    }
    red0[tid] = local_sum0;
    red1[tid] = local_sum1;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint step = nth >> 1; step > 0; step >>= 1) {
        if (tid < step) {
            red0[tid] += red0[tid + step];
            red1[tid] += red1[tid + step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float denom0 = max(red0[0], 1.0e-20f);
    const float denom1 = max(red1[0], 1.0e-20f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = tid; j < args.kv_lora_dim; j += nth) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint s = 0; s < args.n_selected; s++) {
            const uint row = token_selected[s];
            if (row < args.cache_cap) {
                const float kv = glm_cache_load_f32_or_f16(kv_lora_cache,
                                                           (uint64_t)row * args.kv_lora_dim + j,
                                                           args.cache_f16);
                acc0 += scores0[s] * kv;
                if (valid1) acc1 += scores1[s] * kv;
            }
        }
        lora0[j] = acc0 / denom0;
        if (valid1) lora1[j] = acc1 / denom1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint d = tid; d < args.value_dim; d += nth) {
        device float *out0 =
            (device float *)(heads +
                (uint64_t)token * heads_token_stride +
                (uint64_t)head0 * args.value_dim * sizeof(float));
        device const char *row0 =
            value_weight + ((uint64_t)head0 * args.value_dim + d) * args.value_row_bytes;
        out0[d] = glm_quant_dot_row_tg_f32(args.value_type, row0, lora0, args.kv_lora_dim);

        if (valid1) {
            device float *out1 =
                (device float *)(heads +
                    (uint64_t)token * heads_token_stride +
                    (uint64_t)head1 * args.value_dim * sizeof(float));
            device const char *row1 =
                value_weight + ((uint64_t)head1 * args.value_dim + d) * args.value_row_bytes;
            out1[d] = glm_quant_dot_row_tg_f32(args.value_type, row1, lora1, args.kv_lora_dim);
        }
    }
}

template <bool assume_valid_rows, bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const uint32_t *selected,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.cache_f16 == 0u ||
        args.kv_lora_dim != 512u ||
        (args.qk_rope != 0u && args.qk_rope != 64u)) {
        return;
    }

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.qk_rope != 0u && args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                kv_shared[off] = half4(half(0.0f));
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            if (valid_row) {
                const uint64_t rope_base = (uint64_t)row * args.qk_rope;
                const float2 y0 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                const float2 y1 =
                    glm_cache_load_rotated_rope_pair_f16_only(k_rope_cache,
                                                              rope_base,
                                                              r + 2u,
                                                              row,
                                                              args.qk_rope,
                                                              args.freq_base,
                                                              args.freq_scale,
                                                              args.ext_factor,
                                                              args.attn_factor,
                                                              corr_dims[0],
                                                              corr_dims[1]);
                rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
            } else {
                rope_shared[off] = float4(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = assume_valid_rows || row < args.cache_cap;
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score =
                (valid_head && valid_row) ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head && valid_row) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_impl<false, false>)
        glm_attention_indexed_batch_lora_group8_vec_t;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<false, false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_valid")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<true, false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_valid_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_t
kernel_glm_attention_indexed_batch_lora_group8_vec_impl<true, true>;

template <bool assume_valid_heads>
kernel void kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device char *lora_out,
        threadgroup half4 *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 16u;
    const uint token = tgpig.y;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    if (token >= args.n_tokens ||
        args.n_selected == 0u ||
        args.kv_lora_dim != 512u ||
        (args.qk_rope != 0u && args.qk_rope != 64u)) {
        return;
    }

    const uint visible = min(args.n_selected, args.pos0 + token + 1u);
    if (visible == 0u) return;

    const bool valid_head = assume_valid_heads || head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;
    const uint kv_vecs = args.kv_lora_dim >> 2;
    const uint rope_vecs = args.qk_rope >> 2;
    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride =
        (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);

    threadgroup half4 *kv_shared = scratch;
    threadgroup float4 *rope_shared =
        (threadgroup float4 *)(kv_shared + stage_rows * kv_vecs);

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float4 *low4 =
        (device const float4 *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));

    float4 low0 = 0.0f;
    float4 low1 = 0.0f;
    float4 low2 = 0.0f;
    float4 low3 = 0.0f;
    float4 qrope = 0.0f;
    if (valid_head) {
        low0 = low4[lane + 0u];
        low1 = low4[lane + 32u];
        low2 = low4[lane + 64u];
        low3 = low4[lane + 96u];
        if (lane < rope_vecs) {
            qrope = *((device const float4 *)(qh + args.qk_nope + lane * 4u));
        }
    }

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.qk_rope != 0u && args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    float M = -FLT_MAX / 2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    for (uint base = 0u; base < visible; base += stage_rows) {
        const uint rows = min(stage_rows, visible - base);
        for (uint off = tid; off < rows * kv_vecs; off += 256u) {
            const uint rr = off / kv_vecs;
            const uint vv = off - rr * kv_vecs;
            const uint row = base + rr;
            if (args.cache_f16 != 0u) {
                device const half4 *src =
                    (device const half4 *)((device const half *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = src[vv];
            } else {
                device const float4 *src =
                    (device const float4 *)((device const float *)kv_lora_cache +
                        (uint64_t)row * args.kv_lora_dim);
                kv_shared[off] = (half4)src[vv];
            }
        }
        for (uint off = tid; off < rows * rope_vecs; off += 256u) {
            const uint rr = off / rope_vecs;
            const uint vv = off - rr * rope_vecs;
            const uint r = vv * 4u;
            const uint row = base + rr;
            const uint64_t rope_base = (uint64_t)row * args.qk_rope;
            const float2 y0 =
                glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                 rope_base,
                                                 r,
                                                 row,
                                                 args.qk_rope,
                                                 args.cache_f16,
                                                 args.freq_base,
                                                 args.freq_scale,
                                                 args.ext_factor,
                                                 args.attn_factor,
                                                 corr_dims[0],
                                                 corr_dims[1]);
            const float2 y1 =
                glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                 rope_base,
                                                 r + 2u,
                                                 row,
                                                 args.qk_rope,
                                                 args.cache_f16,
                                                 args.freq_base,
                                                 args.freq_scale,
                                                 args.ext_factor,
                                                 args.attn_factor,
                                                 corr_dims[0],
                                                 corr_dims[1]);
            rope_shared[off] = float4(y0.x, y0.y, y1.x, y1.y);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            threadgroup const half4 *kv_row = kv_shared + rr * kv_vecs;
            threadgroup const float4 *rope_row = rope_shared + rr * rope_vecs;
            float partial = 0.0f;
            if (valid_head) {
                partial += dot(low0, (float4)kv_row[lane + 0u]);
                partial += dot(low1, (float4)kv_row[lane + 32u]);
                partial += dot(low2, (float4)kv_row[lane + 64u]);
                partial += dot(low3, (float4)kv_row[lane + 96u]);
                if (lane < rope_vecs) {
                    partial += dot(qrope, rope_row[lane]);
                }
            }
            const float sum = simd_sum(partial);
            const float score = valid_head ? sum * args.scale : -FLT_MAX / 2.0f;
            if (valid_head) {
                const float new_m = max(M, score);
                const float old_scale = exp(M - new_m);
                const float row_scale = exp(score - new_m);
                o0 = o0 * old_scale + (float4)kv_row[lane + 0u] * row_scale;
                o1 = o1 * old_scale + (float4)kv_row[lane + 32u] * row_scale;
                o2 = o2 * old_scale + (float4)kv_row[lane + 64u] * row_scale;
                o3 = o3 * old_scale + (float4)kv_row[lane + 96u] * row_scale;
                S = S * old_scale + row_scale;
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_head) {
        const float inv_s = S > 0.0f ? 1.0f / S : 0.0f;
        device float4 *out4 =
            (device float4 *)(lora_out +
                ((uint64_t)token * args.n_head + head) *
                    args.kv_lora_dim * sizeof(float));
        out4[lane + 0u] = o0 * inv_s;
        out4[lane + 32u] = o1 * inv_s;
        out4[lane + 64u] = o2 * inv_s;
        out4[lane + 96u] = o3 * inv_s;
    }
}

typedef decltype(kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<false>)
        glm_attention_indexed_batch_lora_group8_vec_causal_t;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_causal")]]
kernel glm_attention_indexed_batch_lora_group8_vec_causal_t
kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<false>;

template [[host_name("kernel_glm_attention_indexed_batch_lora_group8_vec_causal_fullheads")]]
kernel glm_attention_indexed_batch_lora_group8_vec_causal_t
kernel_glm_attention_indexed_batch_lora_group8_vec_causal_impl<true>;

kernel void kernel_glm_attention_indexed_batch_group8(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup float *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.y;
    if (token >= args.n_tokens || args.n_selected == 0u) return;

    constexpr uint group_heads = 8u;
    constexpr uint stage_rows = 8u;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    const bool valid_head = head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    threadgroup half *kv_shared = (threadgroup half *)scratch;
    threadgroup half *rope_shared = kv_shared + stage_rows * args.kv_lora_dim;
    threadgroup float *lora_sums =
        (threadgroup float *)(rope_shared + stage_rows * args.qk_rope);
    threadgroup float *head_lora = lora_sums + head_in_group * args.kv_lora_dim;

    device const float *qh =
        (device const float *)(q +
            (uint64_t)token * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *low =
        (device const float *)(qk_low +
            (uint64_t)token * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *token_selected =
        selected + (uint64_t)token * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    constexpr uint max_low_cache = 16u;
    constexpr uint max_qrope_cache = 4u;
    const bool use_low_cache = args.kv_lora_dim <= max_low_cache * 32u;
    const bool use_qrope_cache = args.qk_rope <= max_qrope_cache * 32u;
    half low_cache[max_low_cache];
    half qrope_cache[max_qrope_cache];
    for (uint k = 0u; k < max_low_cache; k++) {
        const uint j = lane + k * 32u;
        low_cache[k] = (valid_head && use_low_cache && j < args.kv_lora_dim) ?
            (half)low[j] : (half)0.0f;
    }
    for (uint k = 0u; k < max_qrope_cache; k++) {
        const uint r = lane + k * 32u;
        qrope_cache[k] = (valid_head && use_qrope_cache && r < args.qk_rope) ?
            (half)qh[args.qk_nope + r] : (half)0.0f;
    }

    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora[j] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;
    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        const uint kv_count = rows * args.kv_lora_dim;
        const uint rope_pairs = args.qk_rope >> 1;
        const uint rope_count = rows * rope_pairs;

        for (uint idx = tid; idx < kv_count; idx += 256u) {
            const uint rr = idx / args.kv_lora_dim;
            const uint j = idx - rr * args.kv_lora_dim;
            const uint row = token_selected[base + rr];
            kv_shared[idx] = (row < args.cache_cap)
                ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                  (uint64_t)row * args.kv_lora_dim + j,
                                                  args.cache_f16)
                : (half)0.0f;
        }
        for (uint idx = tid; idx < rope_count; idx += 256u) {
            const uint rr = idx / rope_pairs;
            const uint pair = idx - rr * rope_pairs;
            const uint r = pair * 2u;
            const uint row = token_selected[base + rr];
            threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
            if (row < args.cache_cap) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   (uint64_t)row * args.qk_rope,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                rope_row[r] = (half)y.x;
                rope_row[r + 1u] = (half)y.y;
            } else {
                rope_row[r] = (half)0.0f;
                rope_row[r + 1u] = (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = token_selected[base + rr];
            const bool valid_row = row < args.cache_cap;
            float partial = 0.0f;
            if (valid_head && valid_row) {
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;
                if (use_low_cache) {
                    for (uint k = 0u; k < max_low_cache; k++) {
                        const uint j = lane + k * 32u;
                        if (j < args.kv_lora_dim) {
                            partial += (float)(low_cache[k] * kv_row[j]);
                        }
                    }
                } else {
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        partial += low[j] * (float)kv_row[j];
                    }
                }
                if (use_qrope_cache) {
                    for (uint k = 0u; k < max_qrope_cache; k++) {
                        const uint r = lane + k * 32u;
                        if (r < args.qk_rope) {
                            partial += (float)(qrope_cache[k] * rope_row[r]);
                        }
                    }
                } else {
                    for (uint r = lane; r < args.qk_rope; r += 32u) {
                        partial += qh[args.qk_nope + r] * (float)rope_row[r];
                    }
                }
            }

            const float sum = simd_sum(partial);
            const float score = (valid_head && valid_row) ? sum * args.scale : -INFINITY;
            if (valid_head && valid_row) {
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                const float old_m = M;
                const float new_m = max(M, score);
                const float old_scale = (old_m == -INFINITY) ? 0.0f : exp(old_m - new_m);
                const float row_scale = exp(score - new_m);
                S = S * old_scale + row_scale;
                for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                    head_lora[j] = head_lora[j] * old_scale + row_scale * (float)kv_row[j];
                }
                M = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_s = (valid_head && S > 0.0f) ? 1.0f / S : 0.0f;
    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora[j] *= inv_s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_head) {
        if (args.value_type == 1u) {
            const uint64_t offset =
                (uint64_t)token *
                    ((uint64_t)args.n_head * args.kv_lora_dim * sizeof(float)) +
                (uint64_t)head * args.kv_lora_dim * sizeof(float);
            device float *out =
                (device float *)(heads + offset);
            for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                out[j] = head_lora[j];
            }
            return;
        }
        device float *out =
            (device float *)(heads +
                (uint64_t)token * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = lane; d < args.value_dim; d += 32u) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora, args.kv_lora_dim);
        }
    }
}

kernel void kernel_glm_attention_indexed_batch_q2_group4(
        constant ds4_metal_args_glm_attention_indexed_batch & args,
        device const char *q,
        device const char *qk_low,
        device const char *kv_lora_cache,
        device const char *k_rope_cache,
        device const char *value_weight,
        device const uint32_t *selected,
        device char *heads,
        threadgroup uint *scratch [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort sg_u [[simdgroup_index_in_threadgroup]]) {
    const uint token0 = tgpig.y * 2u;
    if (token0 >= args.n_tokens || args.n_selected == 0u) return;

    constexpr uint group_heads = 4u;
    constexpr uint stage_rows = 4u;
    constexpr uint group_threads = 128u;
    const uint token1 = token0 + 1u;
    const bool valid1 = token1 < args.n_tokens;
    const uint tid = (uint)tid_u;
    const uint lane = (uint)lane_u;
    const uint head_in_group = (uint)sg_u;
    const uint head = tgpig.x * group_heads + head_in_group + args.head_base;
    const bool valid_head = head < args.n_head;
    const uint safe_head = valid_head ? head : 0u;

    const uint qk_dim = args.qk_nope + args.qk_rope;
    const uint64_t q_token_stride = (uint64_t)args.n_head * qk_dim * sizeof(float);
    const uint64_t low_token_stride = (uint64_t)args.n_head * args.kv_lora_dim * sizeof(float);
    const uint64_t heads_token_stride = (uint64_t)args.n_head * args.value_dim * sizeof(float);

    const uint bit_words = (args.cache_cap + 31u) >> 5;
    threadgroup atomic_uint *member_bits = (threadgroup atomic_uint *)scratch;
    threadgroup half *kv_shared = (threadgroup half *)(scratch + bit_words);
    threadgroup half *rope_shared = kv_shared + stage_rows * args.kv_lora_dim;
    threadgroup float *lora_sums =
        (threadgroup float *)(rope_shared + stage_rows * args.qk_rope);
    threadgroup float *head_lora0 = lora_sums + head_in_group * args.kv_lora_dim;
    threadgroup float *head_lora1 =
        lora_sums + (group_heads + head_in_group) * args.kv_lora_dim;

    const uint safe_token1 = valid1 ? token1 : token0;
    device const float *qh0 =
        (device const float *)(q +
            (uint64_t)token0 * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *qh1 =
        (device const float *)(q +
            (uint64_t)safe_token1 * q_token_stride +
            (uint64_t)safe_head * qk_dim * sizeof(float));
    device const float *low0 =
        (device const float *)(qk_low +
            (uint64_t)token0 * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const float *low1 =
        (device const float *)(qk_low +
            (uint64_t)safe_token1 * low_token_stride +
            (uint64_t)safe_head * args.kv_lora_dim * sizeof(float));
    device const uint32_t *selected0 = selected + (uint64_t)token0 * args.n_selected;
    device const uint32_t *selected1 = selected + (uint64_t)safe_token1 * args.n_selected;

    float corr_dims[2] = {0.0f, 0.0f};
    if (args.ext_factor != 0.0f) {
        glm_rope_yarn_corr_dims((int)args.qk_rope,
                                (int)args.n_ctx_orig,
                                args.freq_base,
                                args.beta_fast,
                                args.beta_slow,
                                corr_dims);
    }

    constexpr uint max_low_cache = 16u;
    constexpr uint max_qrope_cache = 4u;
    const bool use_low_cache = args.kv_lora_dim <= max_low_cache * 32u;
    const bool use_qrope_cache = args.qk_rope <= max_qrope_cache * 32u;
    half low_cache0[max_low_cache];
    half low_cache1[max_low_cache];
    half qrope_cache0[max_qrope_cache];
    half qrope_cache1[max_qrope_cache];
    for (uint k = 0u; k < max_low_cache; k++) {
        const uint j = lane + k * 32u;
        low_cache0[k] = (valid_head && use_low_cache && j < args.kv_lora_dim) ?
            (half)low0[j] : (half)0.0f;
        low_cache1[k] = (valid_head && valid1 && use_low_cache && j < args.kv_lora_dim) ?
            (half)low1[j] : (half)0.0f;
    }
    for (uint k = 0u; k < max_qrope_cache; k++) {
        const uint r = lane + k * 32u;
        qrope_cache0[k] = (valid_head && use_qrope_cache && r < args.qk_rope) ?
            (half)qh0[args.qk_nope + r] : (half)0.0f;
        qrope_cache1[k] = (valid_head && valid1 && use_qrope_cache && r < args.qk_rope) ?
            (half)qh1[args.qk_nope + r] : (half)0.0f;
    }

    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora0[j] = 0.0f;
        if (valid1) head_lora1[j] = 0.0f;
    }
    for (uint i = tid; i < bit_words; i += group_threads) {
        atomic_store_explicit(member_bits + i, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = tid; s < args.n_selected; s += group_threads) {
        const uint row = selected0[s];
        if (row < args.cache_cap) {
            const uint mask = 1u << (row & 31u);
            atomic_fetch_or_explicit(member_bits + (row >> 5), mask, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M0 = -INFINITY;
    float S0 = 0.0f;
    float M1 = -INFINITY;
    float S1 = 0.0f;

    if (valid1) {
        for (uint base = 0u; base < args.n_selected; base += stage_rows) {
            const uint rows = min(stage_rows, args.n_selected - base);
            const uint kv_count = rows * args.kv_lora_dim;
            const uint rope_pairs = args.qk_rope >> 1;
            const uint rope_count = rows * rope_pairs;

            for (uint idx = tid; idx < kv_count; idx += 256u) {
                const uint rr = idx / args.kv_lora_dim;
                const uint j = idx - rr * args.kv_lora_dim;
                const uint row = selected1[base + rr];
                kv_shared[idx] = (row < args.cache_cap)
                    ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                      (uint64_t)row * args.kv_lora_dim + j,
                                                      args.cache_f16)
                    : (half)0.0f;
            }
            for (uint idx = tid; idx < rope_count; idx += 256u) {
                const uint rr = idx / rope_pairs;
                const uint pair = idx - rr * rope_pairs;
                const uint r = pair * 2u;
                const uint row = selected1[base + rr];
                threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
                if (row < args.cache_cap) {
                    const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                       (uint64_t)row * args.qk_rope,
                                                                       r,
                                                                       row,
                                                                       args.qk_rope,
                                                                       args.cache_f16,
                                                                       args.freq_base,
                                                                       args.freq_scale,
                                                                       args.ext_factor,
                                                                       args.attn_factor,
                                                                       corr_dims[0],
                                                                       corr_dims[1]);
                    rope_row[r] = (half)y.x;
                    rope_row[r + 1u] = (half)y.y;
                } else {
                    rope_row[r] = (half)0.0f;
                    rope_row[r + 1u] = (half)0.0f;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint rr = 0u; rr < rows; rr++) {
                const uint row = selected1[base + rr];
                const bool valid_row = row < args.cache_cap;
                const bool in_token0 = valid_row &&
                    ((atomic_load_explicit(member_bits + (row >> 5),
                                           memory_order_relaxed) &
                      (1u << (row & 31u))) != 0u);
                threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
                threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;

                float partial0 = 0.0f;
                float partial1 = 0.0f;
                if (valid_head && valid_row) {
                    if (use_low_cache) {
                        for (uint k = 0u; k < max_low_cache; k++) {
                            const uint j = lane + k * 32u;
                            if (j < args.kv_lora_dim) {
                                const half kv = kv_row[j];
                                if (in_token0) partial0 += (float)(low_cache0[k] * kv);
                                partial1 += (float)(low_cache1[k] * kv);
                            }
                        }
                    } else {
                        for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                            const float kv = (float)kv_row[j];
                            if (in_token0) partial0 += low0[j] * kv;
                            partial1 += low1[j] * kv;
                        }
                    }
                    if (use_qrope_cache) {
                        for (uint k = 0u; k < max_qrope_cache; k++) {
                            const uint r = lane + k * 32u;
                            if (r < args.qk_rope) {
                                const half kv = rope_row[r];
                                if (in_token0) partial0 += (float)(qrope_cache0[k] * kv);
                                partial1 += (float)(qrope_cache1[k] * kv);
                            }
                        }
                    } else {
                        for (uint r = lane; r < args.qk_rope; r += 32u) {
                            const float kv = (float)rope_row[r];
                            if (in_token0) partial0 += qh0[args.qk_nope + r] * kv;
                            partial1 += qh1[args.qk_nope + r] * kv;
                        }
                    }
                }

                const float sum0 = simd_sum(partial0);
                const float sum1 = simd_sum(partial1);
                const float score0 = (valid_head && in_token0) ? sum0 * args.scale : -INFINITY;
                const float score1 = (valid_head && valid_row) ? sum1 * args.scale : -INFINITY;
                if (valid_head && in_token0) {
                    const float new_m = max(M0, score0);
                    const float old_scale = (M0 == -INFINITY) ? 0.0f : exp(M0 - new_m);
                    const float row_scale = exp(score0 - new_m);
                    S0 = S0 * old_scale + row_scale;
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        head_lora0[j] = head_lora0[j] * old_scale + row_scale * (float)kv_row[j];
                    }
                    M0 = new_m;
                }
                if (valid_head && valid_row) {
                    const float new_m = max(M1, score1);
                    const float old_scale = (M1 == -INFINITY) ? 0.0f : exp(M1 - new_m);
                    const float row_scale = exp(score1 - new_m);
                    S1 = S1 * old_scale + row_scale;
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        head_lora1[j] = head_lora1[j] * old_scale + row_scale * (float)kv_row[j];
                    }
                    M1 = new_m;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint i = tid; i < bit_words; i += group_threads) {
            atomic_store_explicit(member_bits + i, 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = tid; s < args.n_selected; s += group_threads) {
            const uint row = selected1[s];
            if (row < args.cache_cap) {
                const uint mask = 1u << (row & 31u);
                atomic_fetch_or_explicit(member_bits + (row >> 5), mask, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint base = 0u; base < args.n_selected; base += stage_rows) {
        const uint rows = min(stage_rows, args.n_selected - base);
        const uint kv_count = rows * args.kv_lora_dim;
        const uint rope_pairs = args.qk_rope >> 1;
        const uint rope_count = rows * rope_pairs;

        for (uint idx = tid; idx < kv_count; idx += 256u) {
            const uint rr = idx / args.kv_lora_dim;
            const uint j = idx - rr * args.kv_lora_dim;
            const uint row = selected0[base + rr];
            kv_shared[idx] = (row < args.cache_cap)
                ? (half)glm_cache_load_f32_or_f16(kv_lora_cache,
                                                  (uint64_t)row * args.kv_lora_dim + j,
                                                  args.cache_f16)
                : (half)0.0f;
        }
        for (uint idx = tid; idx < rope_count; idx += 256u) {
            const uint rr = idx / rope_pairs;
            const uint pair = idx - rr * rope_pairs;
            const uint r = pair * 2u;
            const uint row = selected0[base + rr];
            threadgroup half *rope_row = rope_shared + rr * args.qk_rope;
            if (row < args.cache_cap) {
                const float2 y = glm_cache_load_rotated_rope_pair(k_rope_cache,
                                                                   (uint64_t)row * args.qk_rope,
                                                                   r,
                                                                   row,
                                                                   args.qk_rope,
                                                                   args.cache_f16,
                                                                   args.freq_base,
                                                                   args.freq_scale,
                                                                   args.ext_factor,
                                                                   args.attn_factor,
                                                                   corr_dims[0],
                                                                   corr_dims[1]);
                rope_row[r] = (half)y.x;
                rope_row[r + 1u] = (half)y.y;
            } else {
                rope_row[r] = (half)0.0f;
                rope_row[r + 1u] = (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint rr = 0u; rr < rows; rr++) {
            const uint row = selected0[base + rr];
            const bool valid_row = row < args.cache_cap;
            const bool in_token1 = valid1 && valid_row &&
                ((atomic_load_explicit(member_bits + (row >> 5),
                                       memory_order_relaxed) &
                  (1u << (row & 31u))) != 0u);
            const bool take0 = valid_row && !in_token1;
            threadgroup const half *kv_row = kv_shared + rr * args.kv_lora_dim;
            threadgroup const half *rope_row = rope_shared + rr * args.qk_rope;

            float partial0 = 0.0f;
            if (valid_head && take0) {
                if (use_low_cache) {
                    for (uint k = 0u; k < max_low_cache; k++) {
                        const uint j = lane + k * 32u;
                        if (j < args.kv_lora_dim) {
                            partial0 += (float)(low_cache0[k] * kv_row[j]);
                        }
                    }
                } else {
                    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                        partial0 += low0[j] * (float)kv_row[j];
                    }
                }
                if (use_qrope_cache) {
                    for (uint k = 0u; k < max_qrope_cache; k++) {
                        const uint r = lane + k * 32u;
                        if (r < args.qk_rope) {
                            partial0 += (float)(qrope_cache0[k] * rope_row[r]);
                        }
                    }
                } else {
                    for (uint r = lane; r < args.qk_rope; r += 32u) {
                        partial0 += qh0[args.qk_nope + r] * (float)rope_row[r];
                    }
                }
            }

            const float sum0 = simd_sum(partial0);
            const float score0 = (valid_head && take0) ? sum0 * args.scale : -INFINITY;
            if (valid_head && take0) {
                const float new_m = max(M0, score0);
                const float old_scale = (M0 == -INFINITY) ? 0.0f : exp(M0 - new_m);
                const float row_scale = exp(score0 - new_m);
                S0 = S0 * old_scale + row_scale;
                for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
                    head_lora0[j] = head_lora0[j] * old_scale + row_scale * (float)kv_row[j];
                }
                M0 = new_m;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_s0 = (valid_head && S0 > 0.0f) ? 1.0f / S0 : 0.0f;
    const float inv_s1 = (valid_head && valid1 && S1 > 0.0f) ? 1.0f / S1 : 0.0f;
    for (uint j = lane; j < args.kv_lora_dim; j += 32u) {
        head_lora0[j] *= inv_s0;
        if (valid1) head_lora1[j] *= inv_s1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (valid_head) {
        device float *out0 =
            (device float *)(heads +
                (uint64_t)token0 * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        device float *out1 =
            (device float *)(heads +
                (uint64_t)safe_token1 * heads_token_stride +
                (uint64_t)head * args.value_dim * sizeof(float));
        for (uint d = lane; d < args.value_dim; d += 32u) {
            device const char *row =
                value_weight + ((uint64_t)head * args.value_dim + d) * args.value_row_bytes;
            out0[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora0, args.kv_lora_dim);
            if (valid1) {
                out1[d] = glm_quant_dot_row_tg_f32(args.value_type, row, head_lora1, args.kv_lora_dim);
            }
        }
    }
}

// GLM-5.2 decode router for one token. Selection uses sigmoid(logit)+bias,
// while route weights are normalized from the unbiased sigmoid probabilities.
kernel void kernel_glm_router_select_one(
        constant ds4_metal_args_glm_router_select_one & args,
        device const float *logits,
        device const float *bias,
        device int32_t *selected,
        device float *weights,
        device float *probs,
        threadgroup float *scratch [[threadgroup(0)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint sort_width = args.n_expert > 256u ? 512u : 256u;
    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + sort_width);
    device const float *token_logits = logits + (uint64_t)token * args.n_expert;
    device int32_t *token_selected = selected + (uint64_t)token * args.n_expert_used;
    device float *token_weights = weights + (uint64_t)token * args.n_expert_used;
    device float *token_probs = probs + (uint64_t)token * args.n_expert;

    const uint n_expert = min(args.n_expert, 512u);
    const bool active = tid < n_expert;
    const float p = active ? ds4_glm_router_sigmoid(token_logits[tid]) : 0.0f;
    if (active) token_probs[tid] = p;
    sel_scores[tid] = active ? p + bias[tid] : -INFINITY;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= sort_width; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint other = tid ^ j;
            if (other > tid) {
                const int32_t a = idx[tid];
                const int32_t b = idx[other];
                const bool descending = (tid & k) == 0;
                const bool swap = descending
                    ? ds4_glm_router_better(sel_scores, b, a)
                    : ds4_glm_router_better(sel_scores, a, b);
                if (swap) {
                    idx[tid] = b;
                    idx[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint k_used = min(args.n_expert_used, n_expert);
    if (tid < k_used) {
        token_selected[tid] = idx[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < k_used) {
        float sum = 0.0f;
        for (uint i = 0; i < k_used; i++) {
            sum += token_probs[(uint)token_selected[i]];
        }
        sum = max(sum, 6.103515625e-5f);
        token_weights[tid] = token_probs[(uint)token_selected[tid]] / sum * args.expert_weight_scale;
    }
}

// Batched Flash-router weight finalization after selection is already known.
// Six active lanes deliberately match kernel_sum_rows_f32_f32's reduction
// topology. The denominator and divided weights cross threadgroup storage
// boundaries so division cannot be reassociated with the final scale.
kernel void kernel_dsv4_router_weights_batch(
        constant float &scale,
        device const float *probs,
        device const int32_t *selected,
        device float *weights,
        threadgroup volatile float *scratch [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (tid >= 6) return;

    threadgroup volatile float *sum_scratch = scratch;
    threadgroup volatile float *denom_scratch = scratch + 32;
    threadgroup volatile float *div_scratch = scratch + 33;
    const uint out_index = row * 6u + (uint)tid;
    const int32_t expert = selected[out_index];
    const float p = probs[row * 256u + (uint)expert];

    // Keep this sequence identical to kernel_sum_rows_f32_f32 for width 6.
    if (sgitg == 0) {
        sum_scratch[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    sumf += p;
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sum_scratch[sgitg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sum_scratch[tiisg];
    sumf = simd_sum(sumf);

    if (tid == 0) {
        denom_scratch[0] = clamp(sumf, 6.103515625e-5f, INFINITY);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    div_scratch[tid] = p / denom_scratch[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    weights[out_index] = div_scratch[tid] * scale;
}

// Decode router selection for one token after the existing
// sqrt(softplus(logit)) probability kernel has run. Bias affects only top-k
// selection. Route-weight normalization deliberately stays in the old one-token
// kernel: even tiny denominator-order changes here are amplified by 43 MoE
// layers, so this kernel only replaces the selection work.
kernel void kernel_dsv4_router_finalize_one(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256) return;

    threadgroup float *sel_scores = scratch;
    threadgroup int32_t *idx = (threadgroup int32_t *)(scratch + 256);
    const float p = probs[tid];
    sel_scores[tid] = args.has_bias ? p + bias[tid] : p;
    idx[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.hash_mode) {
        if (tid == 0) {
            const uint token = args.use_token_buffer ? (uint)tokens[0] : args.token;
            const uint row = min(token, args.hash_rows - 1u);
            device const int32_t *src = hash + row * 6u;
            for (uint i = 0; i < 6; i++) {
                selected[i] = src[i];
            }
        }
    } else {
        for (uint k = 2; k <= 256; k <<= 1) {
            for (uint j = k >> 1; j > 0; j >>= 1) {
                const uint other = tid ^ j;
                if (other > tid) {
                    if ((tid & k) == 0) {
                        if (sel_scores[(uint)idx[tid]] < sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    } else {
                        if (sel_scores[(uint)idx[tid]] > sel_scores[(uint)idx[other]]) {
                            const int32_t tmp = idx[tid];
                            idx[tid] = idx[other];
                            idx[other] = tmp;
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (tid < 6) {
            selected[tid] = idx[tid];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

/* Vision-Exp uses score-based visual routing even in the first three layers,
 * where ordinary text rows use a token-id hash table. One threadgroup owns a
 * row so the image/text decision is uniform across all barriers. */
kernel void kernel_dsv4_router_select_visual_batch(
        constant ds4_metal_args_dsv4_router_select_visual & args,
        device const float *probs,
        device const float *bias,
        device const float *visual_bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]],
        uint row [[threadgroup_position_in_grid]]) {
    if (tid >= 256u || row >= args.n_tokens) return;

    const int32_t token = tokens[row];
    const bool image = token >= 0 && (uint32_t)token >= args.vocab_size;
    device int32_t *out = selected + (uint64_t)row * 6u;
    if (args.hash_mode && !image) {
        const uint hash_row = token >= 0 && (uint32_t)token < args.hash_rows
            ? (uint32_t)token : 0u;
        if (tid < 6u) out[tid] = hash[(uint64_t)hash_row * 6u + tid];
        return;
    }

    threadgroup float *scores = scratch;
    threadgroup int32_t *indices = (threadgroup int32_t *)(scratch + 256u);
    const float p = probs[(uint64_t)row * 256u + tid];
    scores[tid] = p + (image ? visual_bias[tid]
                             : (args.has_bias ? bias[tid] : 0.0f));
    indices[tid] = (int32_t)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= 256u; k <<= 1u) {
        for (uint j = k >> 1u; j > 0u; j >>= 1u) {
            const uint other = tid ^ j;
            if (other > tid) {
                const bool descending = (tid & k) == 0u;
                const int32_t a = indices[tid];
                const int32_t b = indices[other];
                const float sa = scores[(uint)a];
                const float sb = scores[(uint)b];
                if ((descending && sa < sb) || (!descending && sa > sb)) {
                    indices[tid] = b;
                    indices[other] = a;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < 6u) out[tid] = indices[tid];
}

// M3 decode specialization for the non-hash one-token router. Scores and ids
// stay in registers. Intra-SIMD bitonic stages use shuffle-xor; the six stages
// that cross 32-lane SIMD groups exchange through alternating threadgroup
// banks. The next bank's publish barrier proves every prior-bank read finished;
// by the time a bank is reused two cross stages later, no reader can remain.
kernel void kernel_dsv4_router_finalize_one_simd(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
}

// M3 decode specialization that extends the register/TG SIMD selection above
// through the existing six-value serial weight normalization. The selected ids
// cross the same device-memory boundary as the standalone weight kernel;
// volatile TG stores pin its left-fold and scaled-reciprocal rounding points.
kernel void kernel_dsv4_router_finalize_weights_one_simd(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        device const int32_t *s = selected;
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; i++) {
            norm_scratch[0] = norm_scratch[0] + probs[s[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        device const int32_t *s = selected;
        weights[tid] = probs[s[tid]] * norm_scratch[1];
    }
}

// M3 decode specialization that materializes the probability
// transform in device memory before running the exact SIMD selection and
// weight normalization above. The volatile reload after the device barrier
// pins the same float store/load boundary as the standalone transform dispatch.
kernel void kernel_dsv4_router_transform_finalize_weights_one_simd(
        constant ds4_metal_args_dsv4_router_select_one & args,
        device const float *logits,
        device float *probs,
        device const float *bias,
        device const int32_t *hash,
        device const int32_t *tokens,
        device int32_t *selected,
        device float *weights,
        threadgroup float *scratch [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= 256 || args.hash_mode) return;

    if (tid < 64) {
        device const float4 *s = (device const float4 *)logits;
        device float4 *d = (device float4 *)probs;
        const float4 x = s[tid];
        const float4 ex = exp(x);
        const float4 em = min(ex, 0.03125f);
        const float4 poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float4 sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        d[tid] = sqrt(sp);
    }
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float *reloaded_probs =
        (device volatile const float *)probs;

    (void)hash;
    (void)tokens;
    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = reloaded_probs[tid];
    float score = args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;

    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;

            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }

    if (tid < 6) {
        selected[tid] = idx;
    }
    threadgroup_barrier(mem_flags::mem_device);

    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        device const int32_t *s = selected;
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; i++) {
            norm_scratch[0] =
                norm_scratch[0] + reloaded_probs[s[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        device const int32_t *s = selected;
        weights[tid] = reloaded_probs[s[tid]] * norm_scratch[1];
    }
}

kernel void kernel_dsv4_router_project_select_fused(
        constant ds4_metal_args_mul_mv & args,
        constant ds4_metal_args_dsv4_router_select_one & select_args,
        device const char * src0_router,
        device const char * src1,
        device float * logits,
        device float * probs,
        device const float * bias,
        device int32_t * selected,
        device float * weights,
        device atomic_uint * completion,
        threadgroup char * shmem_raw [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short NSG = 8;
    constexpr short NR0 = 2;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;
    constexpr short NW  = N_SIMDWIDTH;
    const uint tid = tpitg.x;
    const int nb = args.ne00/NB;
    const int r0 = tgpig.x*NR0;
    device const float4 *y4 = (device const float4 *)src1;
    device const half4 *ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        ax4[row] = (device const half4 *)
            (src0_router + (uint64_t)(r0 + row)*args.nb01);
    }
    float sumf[NR0] = {0.f};
    const short ix = tiisg/(NW/NF);
    const short il = tiisg%(NW/NF);
    const int ib0 = sgitg*NF + ix;
    device const float4 *yb4 = y4 + (ib0*NB + il*NF)/4;
    for (int ib = ib0; ib < nb; ib += NSG*NF) {
        float4 yl4[NF4];
        FOR_UNROLL (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }
        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const half4 *xb4 = ax4[row] + (ib*NB + il*NF)/4;
            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), yl4[i]);
            }
            sumf[row] += sumq;
        }
        yb4 += NSG*NF*NW/4;
    }
    helper_mv_reduce_and_write<NR0>(logits, sumf, r0, args.ne01,
                                    tiisg, sgitg, shmem_raw);

    threadgroup float *scratch = (threadgroup float *)shmem_raw;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    if (tid == 0) {
        const uint old = atomic_fetch_add_explicit(
            completion, 1u, memory_order_relaxed);
        scratch[0] = old == 127u ? 1.0f : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (scratch[0] == 0.0f) return;
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);

    if (tid < 64) {
        device volatile const float4 *s =
            (device volatile const float4 *)logits;
        device float4 *d = (device float4 *)probs;
        const float4 xv = s[tid];
        const float4 ex = exp(xv);
        const float4 em = min(ex, 0.03125f);
        const float4 poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float4 sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), xv, xv > 20.0f);
        d[tid] = sqrt(sp);
    }
    threadgroup_barrier(mem_flags::mem_device);
    device volatile const float *reloaded_probs =
        (device volatile const float *)probs;

    threadgroup float *score0_tg = scratch;
    threadgroup int32_t *idx0_tg =
        (threadgroup int32_t *)(scratch + 256);
    threadgroup float *score1_tg = scratch + 512;
    threadgroup int32_t *idx1_tg =
        (threadgroup int32_t *)(scratch + 768);
    const float p = reloaded_probs[tid];
    float score = select_args.has_bias ? p + bias[tid] : p;
    int32_t idx = (int32_t)tid;
    uint cross_stage = 0;
    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            float peer_score;
            int32_t peer_idx;
            bool take_peer;
            const bool lower = (tid & j) == 0;
            const bool descending = (tid & k) == 0;
            if (j < 32) {
                peer_score = simd_shuffle_xor(score, (ushort)j);
                peer_idx = simd_shuffle_xor(idx, (ushort)j);
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
            } else {
                threadgroup float *score_tg =
                    (cross_stage & 1u) != 0u ? score1_tg : score0_tg;
                threadgroup int32_t *idx_tg =
                    (cross_stage & 1u) != 0u ? idx1_tg : idx0_tg;
                score_tg[tid] = score;
                idx_tg[tid] = idx;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                const uint other = tid ^ j;
                peer_score = score_tg[other];
                peer_idx = idx_tg[other];
                take_peer = descending
                    ? (lower ? score < peer_score : score > peer_score)
                    : (lower ? score > peer_score : score < peer_score);
                if (take_peer) {
                    score = peer_score;
                    idx = peer_idx;
                }
                cross_stage++;
            }
        }
    }
    if (tid < 6) selected[tid] = idx;
    threadgroup_barrier(mem_flags::mem_device);
    threadgroup volatile float *norm_scratch =
        (threadgroup volatile float *)scratch;
    if (tid == 0) {
        norm_scratch[0] = 0.0f;
        for (uint i = 0; i < 6; ++i) {
            norm_scratch[0] = norm_scratch[0] + reloaded_probs[selected[i]];
        }
        norm_scratch[0] = max(norm_scratch[0], 6.103515625e-5f);
        norm_scratch[1] = 1.5f / norm_scratch[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 6) {
        weights[tid] = reloaded_probs[selected[tid]] * norm_scratch[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_thread_fence(mem_flags::mem_device,
                        memory_order_seq_cst,
                        thread_scope_device);
    if (tid == 0) {
        atomic_store_explicit(completion, 0u, memory_order_relaxed);
    }
}


// Fills the dense compressed-attention mask with -inf. The selected top-k rows
// are enabled by kernel_dsv4_topk_mask_scatter in a second ordered dispatch.
kernel void kernel_dsv4_topk_mask(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    (void)topk;
    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = -INFINITY;
}

// Enables the selected compressed rows in the dense mask. This replaces the
// old O(n_comp * n_tokens * top_k) membership test with O(top_k * n_tokens)
// writes while preserving exactly the same 0/-inf mask consumed by attention.
kernel void kernel_dsv4_topk_mask_scatter(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * topk,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne00 * args.ne01;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ik = gid % args.ne00;
    const int64_t it = gid / args.ne00;
    const int32_t idx = *((device const int32_t *) (topk + ik*args.nb00 + it*args.nb01));
    if (idx >= 0 && (int64_t)idx < args.ne0) {
        *((device float *) (dst + (int64_t)idx*args.nb0 + it*args.nb1)) = 0.0f;
    }
}

// Sorts each token's selected compressed rows by row id. The indexer selects by
// score, but attention scans compressed K/V in cache order in the dense graph.
// Sorting preserves that order while still letting the indexed attention kernel
// touch only the selected rows.
kernel void kernel_dsv4_sort_i32_rows_asc(
        constant ds4_metal_args_dsv4_topk_mask & args,
        device const char * src,
        device       char * dst,
        threadgroup int32_t * row_tmp [[threadgroup(0)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint n_threads [[threads_per_threadgroup]]) {
    const uint top_k = (uint)args.ne00;
    if (row >= (uint)args.ne01 || tid >= n_threads) {
        return;
    }

    for (uint i = tid; i < top_k; i += n_threads) {
        row_tmp[i] = *((device const int32_t *) (src + (uint64_t)i*args.nb00 + (uint64_t)row*args.nb01));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2; k <= top_k; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = tid; i < top_k; i += n_threads) {
                const uint other = i ^ j;
                if (other > i && other < top_k) {
                    const int32_t a = row_tmp[i];
                    const int32_t b = row_tmp[other];
                    const bool up = (i & k) == 0;
                    if ((up && a > b) || (!up && a < b)) {
                        row_tmp[i] = b;
                        row_tmp[other] = a;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint i = tid; i < top_k; i += n_threads) {
        *((device int32_t *) (dst + (uint64_t)i*args.nb00 + (uint64_t)row*args.nb01)) = row_tmp[i];
    }
}

static inline void dsv4_attend_f32_row_as_f16(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    device const float4 *kv4 = (device const float4 *)(kv + (uint64_t)row * row_stride);
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16(
        threadgroup const float4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = (half4)kv4[lane +  0];
    const half4 k1 = (half4)kv4[lane + 32];
    const half4 k2 = (half4)kv4[lane + 64];
    const half4 k3 = (half4)kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_f32_row_as_f16_at(
        threadgroup const float4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_f32_row_as_f16(kv4 + row_in_tg * 128u,
                                      q0, q1, q2, q3,
                                      scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
}

static inline void dsv4_attend_shared_h4_row(
        threadgroup const half4 *kv4,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const half4 k0 = kv4[lane +  0];
    const half4 k1 = kv4[lane + 32];
    const half4 k2 = kv4[lane + 64];
    const half4 k3 = kv4[lane + 96];

    float score = dot((float4)q0, (float4)k0) +
                  dot((float4)q1, (float4)k1) +
                  dot((float4)q2, (float4)k2) +
                  dot((float4)q3, (float4)k3);
    score = simd_sum(score) * scale;

    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;

    o0 += (float4)k0 * row_scale;
    o1 += (float4)k1 * row_scale;
    o2 += (float4)k2 * row_scale;
    o3 += (float4)k3 * row_scale;
    M = new_m;
}

static inline void dsv4_attend_shared_h4_row_at(
        threadgroup const half4 *kv4,
        uint row_in_tg,
        half4 q0,
        half4 q1,
        half4 q2,
        half4 q3,
        float scale,
        ushort lane,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    dsv4_attend_shared_h4_row(kv4 + row_in_tg * 128u,
                              q0, q1, q2, q3,
                              scale,
                              lane,
                              M, S,
                              o0, o1, o2, o3);
}

static inline half4 dsv4_load_cache_h4(
        device const char *kv,
        uint64_t row_stride,
        uint row,
        uint col,
        bool f16_rows) {
    device const char *base = kv + (uint64_t)row * row_stride;
    if (f16_rows) {
        return ((device const half4 *)base)[col];
    }
    return (half4)((device const float4 *)base)[col];
}

static inline void dsv4_attend_sink(
        float score,
        thread float &M,
        thread float &S,
        thread float4 &o0,
        thread float4 &o1,
        thread float4 &o2,
        thread float4 &o3) {
    const float old_m = M;
    const float new_m = max(M, score);
    const float old_scale = exp(old_m - new_m);
    const float row_scale = exp(score - new_m);

    S = S * old_scale + row_scale;
    o0 *= old_scale;
    o1 *= old_scale;
    o2 *= old_scale;
    o3 *= old_scale;
    M = new_m;
}

// DS4 ratio-4 indexed mixed attention. It replaces the dense top-k mask path:
// the threadgroup covers one token and eight heads. Top-k rows and local raw
// rows are the same for all heads of a token, so K/V is staged once in
// threadgroup memory and reused by the eight simdgroups. It keeps the DS4 F16
// attention rounding by casting Q/K/V to half before the dot/value update.
kernel void kernel_dsv4_indexed_mixed_attention_heads8(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            const uint logical = pos - first_raw_pos;
            const uint row = (args.raw_start + logical) % args.raw_cap;
            device const float4 *src = (device const float4 *)(raw_kv +
                (uint64_t)row * args.raw_row_stride);
            if (tid < 128) kv_shared[tid] = (half4)src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_h4_row(kv_shared,
                                      q0, q1, q2, q3,
                                      args.scale,
                                      lane,
                                      M, S,
                                      o0, o1, o2, o3);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    for (uint i = 0; i < args.top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) {
            continue;
        }
        if ((uint)idx >= visible) {
            break;
        }
        if (tid < 128) {
            kv_shared[tid] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                (uint)idx,
                                                tid,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dsv4_attend_shared_h4_row(kv_shared,
                                  q0, q1, q2, q3,
                                  args.scale,
                                  lane,
                                  M, S,
                                  o0, o1, o2, o3);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

// Each simdgroup owns two heads and updates both from one staged K/V row.
// This doubles row reuse without increasing the 256-thread workgroup.
kernel void kernel_dsv4_indexed_mixed_attention_heads16_dual(
        constant ds4_metal_args_dsv4_indexed_attention &args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head0 = tgpig.y*16u + (uint)sg;
    const uint head1 = head0 + 8u;
    if (token >= args.n_tokens || head0 >= args.n_head) return;

    device const float4 *qa = (device const float4 *)(q +
        (uint64_t)token*args.q_token_stride +
        (uint64_t)head0*args.q_head_stride);
    half4 qa0 = (half4)qa[lane + 0];
    half4 qa1 = (half4)qa[lane + 32];
    half4 qa2 = (half4)qa[lane + 64];
    half4 qa3 = (half4)qa[lane + 96];
    half4 qb0 = half4(0.0h), qb1 = half4(0.0h);
    half4 qb2 = half4(0.0h), qb3 = half4(0.0h);
    if (head1 < args.n_head) {
        device const float4 *qb = (device const float4 *)(q +
            (uint64_t)token*args.q_token_stride +
            (uint64_t)head1*args.q_head_stride);
        qb0 = (half4)qb[lane + 0];
        qb1 = (half4)qb[lane + 32];
        qb2 = (half4)qb[lane + 64];
        qb3 = (half4)qb[lane + 96];
    }

    float Ma = -FLT_MAX/2.0f, Sa = 0.0f;
    float Mb = -FLT_MAX/2.0f, Sb = 0.0f;
    float4 ao0 = 0.0f, ao1 = 0.0f, ao2 = 0.0f, ao3 = 0.0f;
    float4 bo0 = 0.0f, bo1 = 0.0f, bo2 = 0.0f, bo3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    const uint first = max(first_raw_pos, window_first);
    const uint last = min(qpos, raw_last_pos);
    if (first <= last) {
        for (uint pos = first; pos <= last; pos++) {
            const uint logical = pos - first_raw_pos;
            const uint row = (args.raw_start + logical)%args.raw_cap;
            device const float4 *src = (device const float4 *)(raw_kv +
                (uint64_t)row*args.raw_row_stride);
            if (tid < 128) kv_shared[tid] = (half4)src[tid];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dsv4_attend_shared_h4_row(kv_shared, qa0, qa1, qa2, qa3,
                args.scale, lane, Ma, Sa, ao0, ao1, ao2, ao3);
            if (head1 < args.n_head) {
                dsv4_attend_shared_h4_row(kv_shared, qb0, qb1, qb2, qb3,
                    args.scale, lane, Mb, Sb, bo0, bo1, bo2, bo3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint visible = min((qpos + 1u)/args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token*args.topk_token_stride);
    for (uint i = 0; i < args.top_k; i++) {
        const int32_t idx = row_topk[i];
        if (idx < 0) continue;
        if ((uint)idx >= visible) break;
        if (tid < 128) {
            kv_shared[tid] = dsv4_load_cache_h4(comp_kv,
                args.comp_row_stride, (uint)idx, tid, args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dsv4_attend_shared_h4_row(kv_shared, qa0, qa1, qa2, qa3,
            args.scale, lane, Ma, Sa, ao0, ao1, ao2, ao3);
        if (head1 < args.n_head) {
            dsv4_attend_shared_h4_row(kv_shared, qb0, qb1, qb2, qb3,
                args.scale, lane, Mb, Sb, bo0, bo1, bo2, bo3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head0],
        Ma, Sa, ao0, ao1, ao2, ao3);
    const float ia = Sa == 0.0f ? 0.0f : 1.0f/Sa;
    device float4 *da = (device float4 *)(dst +
        (uint64_t)token*args.dst_token_stride +
        (uint64_t)head0*args.dst_head_stride);
    da[lane + 0] = ao0*ia; da[lane + 32] = ao1*ia;
    da[lane + 64] = ao2*ia; da[lane + 96] = ao3*ia;
    if (head1 < args.n_head) {
        dsv4_attend_sink(((device const float *)sinks)[head1],
            Mb, Sb, bo0, bo1, bo2, bo3);
        const float ib = Sb == 0.0f ? 0.0f : 1.0f/Sb;
        device float4 *db = (device float4 *)(dst +
            (uint64_t)token*args.dst_token_stride +
            (uint64_t)head1*args.dst_head_stride);
        db[lane + 0] = bo0*ib; db[lane + 32] = bo1*ib;
        db[lane + 64] = bo2*ib; db[lane + 96] = bo3*ib;
    }
}

// Decode specialization of kernel_dsv4_indexed_mixed_attention_heads8.
// Generation attends one token at a time, so the ratio-4 indexed path spends a
// visible amount of time repeatedly staging the same K/V row for the eight
// heads in a group. This variant stages sixteen selected rows at once and then
// consumes them sequentially, preserving the row order and online softmax math
// while cutting threadgroup barriers in the long top-k scan.
kernel void kernel_dsv4_indexed_mixed_attention_heads8_rb16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device const char *sinks,
        device       char *dst,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    if (token >= args.n_tokens || head >= args.n_head) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    uint first = max(first_raw_pos, window_first);
    uint last = min(qpos, raw_last_pos);

    if (first <= last) {
        for (uint pos0 = first; pos0 <= last; pos0 += 16u) {
            const uint n_rows = min(16u, last - pos0 + 1u);
            for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
                const uint r = off >> 7;
                const uint c = off & 127u;
                const uint logical = pos0 + r - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                kv_shared[off] = (half4)src[c];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint r = 0; r < n_rows; r++) {
                dsv4_attend_shared_h4_row_at(kv_shared,
                                             r,
                                             q0, q1, q2, q3,
                                             args.scale,
                                             lane,
                                             M, S,
                                             o0, o1, o2, o3);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    uint visible = (qpos + 1u) / args.ratio;
    visible = min(visible, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);
    bool stop = false;
    for (uint i = 0; i < args.top_k && !stop; i += 16u) {
        uint rows[16];
        uint n_rows = 0;
        for (uint j = 0; j < 16u && i + j < args.top_k; j++) {
            const int32_t idx = row_topk[i + j];
            if (idx < 0) {
                continue;
            }
            if ((uint)idx >= visible) {
                stop = true;
                break;
            }
            rows[n_rows++] = (uint)idx;
        }
        if (n_rows == 0) {
            continue;
        }
        for (uint off = (uint)tid; off < n_rows * 128u; off += 256u) {
            const uint r = off >> 7;
            const uint c = off & 127u;
            kv_shared[off] = dsv4_load_cache_h4(comp_kv,
                                                args.comp_row_stride,
                                                rows[r],
                                                c,
                                                args.comp_kv_f16 != 0u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            dsv4_attend_shared_h4_row_at(kv_shared,
                                         r,
                                         q0, q1, q2, q3,
                                         args.scale,
                                         lane,
                                         M, S,
                                         o0, o1, o2, o3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    dsv4_attend_sink(((device const float *)sinks)[head], M, S, o0, o1, o2, o3);

    const float inv_s = S == 0.0f ? 0.0f : 1.0f/S;
    device float4 *dst4 = (device float4 *)(dst +
        (uint64_t)token * args.dst_token_stride +
        (uint64_t)head  * args.dst_head_stride);
    dst4[lane +  0] = o0 * inv_s;
    dst4[lane + 32] = o1 * inv_s;
    dst4[lane + 64] = o2 * inv_s;
    dst4[lane + 96] = o3 * inv_s;
}

// Long-context decode specialization of the indexed mixed-attention path.
//
// The ordinary heads8 kernel reuses each K/V row across eight heads, but only
// launches one threadgroup per head group. Long-context decode therefore has
// too little parallel work while each group scans its raw and selected rows.
// This kernel retains the same eight-head reuse while splitting that row
// sequence across args.n_splits workgroups. A second kernel merges the online
// softmax partials and applies the attention sink.
kernel void kernel_dsv4_indexed_mixed_attention_heads8_split(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q,
        device const char *raw_kv,
        device const char *comp_kv,
        device const char *topk,
        device       char *tmp,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_block = 16u;
    constexpr uint vecs_per_row = 128u;

    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    const uint split = tgpig.z;
    const uint n_splits = args.n_splits;
    if (token >= args.n_tokens || head >= args.n_head ||
        n_splits < 2u || n_splits > 31u || split >= n_splits) {
        return;
    }

    device const float4 *q4 = (device const float4 *)(q +
        (uint64_t)token * args.q_token_stride +
        (uint64_t)head  * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];

    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f;
    float4 o1 = 0.0f;
    float4 o2 = 0.0f;
    float4 o3 = 0.0f;

    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ?
        qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ?
        raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split =
        (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk +
        (uint64_t)token * args.topk_token_stride);

    for (uint seq0 = split_first; seq0 < split_last;
         seq0 += rows_per_block) {
        const uint n_rows = min(rows_per_block, split_last - seq0);
        for (uint off = (uint)tid;
             off < n_rows * vecs_per_row;
             off += 256u) {
            const uint r = off / vecs_per_row;
            const uint c = off - r * vecs_per_row;
            const uint seq = seq0 + r;
            half4 value = half4(0.0h);
            if (seq < raw_count) {
                const uint pos = raw_first + seq;
                const uint logical = pos - first_raw_pos;
                const uint row = (args.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv +
                    (uint64_t)row * args.raw_row_stride);
                value = (half4)src[c];
            } else {
                const int32_t idx = row_topk[seq - raw_count];
                if (idx >= 0 && (uint)idx < visible) {
                    value = dsv4_load_cache_h4(comp_kv,
                                               args.comp_row_stride,
                                               (uint)idx,
                                               c,
                                               args.comp_kv_f16 != 0u);
                }
            }
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            const uint seq = seq0 + r;
            bool valid = true;
            if (seq >= raw_count) {
                const int32_t idx = row_topk[seq - raw_count];
                valid = idx >= 0 && (uint)idx < visible;
            }
            if (valid) {
                dsv4_attend_shared_h4_row_at(kv_shared,
                                             r,
                                             q0, q1, q2, q3,
                                             args.scale,
                                             lane,
                                             M, S,
                                             o0, o1, o2, o3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device float4 *partials = (device float4 *)tmp;
    partials[(row * vecs_per_row + lane +  0u) * n_splits + split] = o0;
    partials[(row * vecs_per_row + lane + 32u) * n_splits + split] = o1;
    partials[(row * vecs_per_row + lane + 64u) * n_splits + split] = o2;
    partials[(row * vecs_per_row + lane + 96u) * n_splits + split] = o3;

    if (lane == 0u) {
        device float *stats = (device float *)(partials +
            n_rows * vecs_per_row * n_splits);
        const uint64_t stat = (row * n_splits + split) * 2u;
        stats[stat + 0u] = S;
        stats[stat + 1u] = M;
    }
}

kernel void kernel_dsv4_indexed_mixed_attention_heads8_split_reduce(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *tmp,
        device const char *sinks,
        device       char *dst,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr uint vecs_per_row = 128u;
    const uint n_splits = args.n_splits;
    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = tgpig;
    if (row >= n_rows || n_splits < 2u || n_splits > 31u) {
        return;
    }

    device const float4 *partials = (device const float4 *)tmp;
    device const float *stats = (device const float *)(partials +
        n_rows * vecs_per_row * n_splits);
    float part_sum = 0.0f;
    float part_max = -FLT_MAX/2.0f;
    if ((uint)lane < n_splits) {
        const uint64_t stat = (row * n_splits + (uint)lane) * 2u;
        part_sum = stats[stat + 0u];
        part_max = stats[stat + 1u];
    } else if ((uint)lane == n_splits) {
        const uint head = (uint)(row % args.n_head);
        part_sum = 1.0f;
        part_max = ((device const float *)sinks)[head];
    }

    const float global_max = simd_max(part_max);
    const float part_scale = part_sum > 0.0f ?
        exp(part_max - global_max) : 0.0f;
    const float total_sum = simd_sum(part_sum * part_scale);
    const float inv_sum = total_sum > 0.0f ? 1.0f / total_sum : 0.0f;

    device float4 *out = (device float4 *)dst + row * vecs_per_row;
    for (uint i = (uint)sg; i < vecs_per_row; i += 4u) {
        float4 value = float4(0.0f);
        if ((uint)lane < n_splits) {
            value = partials[(row * vecs_per_row + i) * n_splits +
                             (uint)lane] * part_scale;
        }
        value = simd_sum(value);
        if (lane == 0u) {
            out[i] = value * inv_sum;
        }
    }
}


kernel void kernel_astra_indexed_reduce_t32(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *tmp,
        device const char *sinks,
        device       char *dst,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr uint vecs_per_row = 128u;
    const uint n_splits = args.n_splits;
    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = tgpig / 4u;
    const uint vbase = (tgpig % 4u) * 32u;
    if (row >= n_rows || n_splits < 2u || n_splits > 31u) {
        return;
    }

    device const float4 *partials = (device const float4 *)tmp;
    device const float *stats = (device const float *)(partials +
        n_rows * vecs_per_row * n_splits);
    float part_sum = 0.0f;
    float part_max = -FLT_MAX/2.0f;
    if ((uint)lane < n_splits) {
        const uint64_t stat = (row * n_splits + (uint)lane) * 2u;
        part_sum = stats[stat + 0u];
        part_max = stats[stat + 1u];
    } else if ((uint)lane == n_splits) {
        const uint head = (uint)(row % args.n_head);
        part_sum = 1.0f;
        part_max = ((device const float *)sinks)[head];
    }

    const float global_max = simd_max(part_max);
    const float part_scale = part_sum > 0.0f ?
        exp(part_max - global_max) : 0.0f;
    const float total_sum = simd_sum(part_sum * part_scale);
    const float inv_sum = total_sum > 0.0f ? 1.0f / total_sum : 0.0f;

    device float4 *out = (device float4 *)dst + row * vecs_per_row;
    for (uint i = vbase + (uint)sg; i < vbase + 32u; i += 4u) {
        float4 value = float4(0.0f);
        if ((uint)lane < n_splits) {
            value = partials[(row * vecs_per_row + i) * n_splits +
                             (uint)lane] * part_scale;
        }
        value = simd_sum(value);
        if (lane == 0u) {
            out[i] = value * inv_sum;
        }
    }
}

static inline float dsv4_indexer_dot128_shared_q(
        float4 c0,
        float4 c1,
        float4 c2,
        float4 c3,
        threadgroup const float4 *q4,
        ushort lane) {
    float sum = 0.0f;
    if (lane < 8) {
        const ushort ib = lane >> 1;
        const ushort il = lane & 1;
        const ushort base = ib*8 + il*4;
        sum += dot(c0, q4[base + 0]);
        sum += dot(c1, q4[base + 1]);
        sum += dot(c2, q4[base + 2]);
        sum += dot(c3, q4[base + 3]);
    }
    return simd_sum(sum);
}

// Tiled prefill score builder for the sparse-compressed attention indexer.
//
// The kernel covers an 8-token by 32-compressed-row rectangle: K is copied into
// threadgroup memory once, then reused for all 64 indexer heads, while simdgroup
// matrix multiply computes each 8x8 score subtile.
//
// It still writes the exact score matrix consumed by top-k:
//
//     score[t,c] = sum_h relu(dot(Q[t,h], K[c])) * W[t,h] * scale
//
// Causal masking is applied on store so invisible compressed rows become -inf.
kernel void kernel_dsv4_indexer_scores_tiled_f32(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup float *qtg = shared;             // [8][128]
    threadgroup float *ktg = qtg + TM*D;         // [32][128]
    threadgroup float *dot = ktg + TN*D;         // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        float v = 0.0f;
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = row[d];
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            float v = 0.0f;
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = qrow[d];
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_float8x8 mq;
            simdgroup_float8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

kernel void kernel_dsv4_indexer_scores_tiled(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup float *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint TM = 8;
    constexpr uint TN = 32;
    constexpr uint TS = 8;
    constexpr uint D  = 128;

    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    // Q/K are staged as half but the dot accumulator and final score remain
    // float. This is the one intentional precision tradeoff in the indexer:
    // the indexer only ranks compressed rows for top-k selection, and long
    // context profiling shows this score matrix dominates the prefill slope.
    threadgroup half *qtg = (threadgroup half *)shared; // [8][128]
    threadgroup half *ktg = qtg + TM*D;                 // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [8][32]

    const uint last_token = min(t0 + TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += 128) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    // Stage compressed index rows once. Edge columns are zeroed so the matrix
    // loads below can stay regular; guarded stores discard them.
    for (uint i = tid; i < TN*D; i += 128) {
        const uint cc = i / D;
        const uint d = i - cc*D;
        const uint comp = c0 + cc;
        half v = half(0.0f);
        if (comp < args.n_comp) {
            device const float *row = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
            v = half(row[d]);
        }
        ktg[i] = v;
    }

    const uint cell0 = lane;
    const uint cell1 = lane + 32u;
    const uint row0 = cell0 >> 3;
    const uint row1 = cell1 >> 3;
    const uint sub0 = cell0 & 7u;
    const uint sub1 = cell1 & 7u;
    const uint col0 = (uint)sg * TS + sub0;
    const uint col1 = (uint)sg * TS + sub1;
    const uint token0 = t0 + row0;
    const uint token1 = t0 + row1;
    const uint comp0 = c0 + col0;
    const uint comp1 = c0 + col1;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint head = 0; head < args.n_head; head++) {
        // Stage Q for the eight-token tile. Each 8x8 matrix load below reads a
        // contiguous depth block from this layout.
        for (uint i = tid; i < TM*D; i += 128) {
            const uint r = i / D;
            const uint d = i - r*D;
            const uint token = t0 + r;
            half v = half(0.0f);
            if (token < args.n_tokens) {
                device const float *qrow = (device const float *)(q +
                    (uint64_t)token * args.q_token_stride +
                    (uint64_t)head  * args.q_head_stride);
                v = half(qrow[d]);
            }
            qtg[i] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 mdot = make_filled_simdgroup_matrix<float, 8>(0.0f);
        for (uint db = 0; db < D/TS; db++) {
            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk;
            simdgroup_load(mq, qtg + db*TS, D, 0, false);
            simdgroup_load(mk, ktg + ((uint)sg * TS) * D + db*TS, D, 0, true);
            simdgroup_multiply_accumulate(mdot, mq, mk, mdot);
        }

        simdgroup_store(mdot, dot + (uint)sg * TS, TN, 0, false);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (token0 < args.n_tokens && comp0 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token0 * args.weights_token_stride);
            const float s = dot[row0*TN + col0];
            acc0 += max(s, 0.0f) * (w[head] * args.scale);
        }
        if (token1 < args.n_tokens && comp1 < args.n_comp) {
            device const float *w = (device const float *)(weights +
                (uint64_t)token1 * args.weights_token_stride);
            const float s = dot[row1*TN + col1];
            acc1 += max(s, 0.0f) * (w[head] * args.scale);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (token0 < args.n_tokens && comp0 < args.n_comp) {
        const uint visible = min((args.pos0 + token0 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token0 * args.score_token_stride) + comp0;
        *dst = comp0 < visible ? acc0 : -INFINITY;
    }
    if (token1 < args.n_tokens && comp1 < args.n_comp) {
        const uint visible = min((args.pos0 + token1 + 1u) / args.ratio, args.n_comp);
        device float *dst = (device float *)(scores +
            (uint64_t)token1 * args.score_token_stride) + comp1;
        *dst = comp1 < visible ? acc1 : -INFINITY;
    }
}

#ifdef DS4_METAL_HAS_TENSOR
// Retained full-512 prefill indexer score path.  This is the part of sparse
// compressed attention that maps cleanly to TensorOps: a regular token by
// compressed-row dot tile.  The kernel intentionally leaves top-k selection and
// indexed attention semantics unchanged; all 512 selected rows remain available
// to the later attention kernel.
//
// Each matmul processes a pair of heads (TQ = 2 x TM q rows): the per-element
// dot is still a 128-deep reduction in 32-wide k-steps, so scores are
// bit-identical to single-head tiles while the run count halves.  The q tile
// is double-buffered, so the next k-step's stage overlaps the current
// cooperative matmul and each pair needs 5 barriers instead of 10.  q and k
// staging use one float4/half4 per lane (each thread covers one row of 8/32
// consecutive elements), which is the same half(float) conversion per element
// as the scalar form.
kernel void kernel_dsv4_indexer_scores_nax(
        constant ds4_metal_args_dsv4_indexer_scores_fused & args,
        device const char *q,
        device const char *weights,
        device const char *index_comp,
        device       char *scores,
        threadgroup half *shared [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]]) {
    constexpr int TM = 16;
    constexpr int TQ = 32;
    constexpr int TN = 32;
    constexpr int NK = 32;
    constexpr int D  = 128;
    constexpr int NUM_THREADS = 128;

    // The 16-token x 32-row tile was the winning NAX shape in local sweeps.  A
    // wider 64-row compressed tile increased setup/cache pressure and was
    // slower despite doing more work per dispatch.
    const uint c0 = tgpig.x * TN;
    const uint t0 = tgpig.y * TM;

    threadgroup half  *qtg = shared;               // 2 x [TQ][NK]
    threadgroup half  *ktg = qtg + 2*TQ*NK;        // [32][128]
    threadgroup float *dot = (threadgroup float *)(ktg + TN*D); // [TQ][TN], column-major

    const uint last_token = min(t0 + (uint)TM, args.n_tokens);
    const uint max_visible = last_token > t0 ?
        min((args.pos0 + last_token) / args.ratio, args.n_comp) : 0u;

    if (c0 >= max_visible) {
        for (uint i = tid; i < TM*TN; i += NUM_THREADS) {
            const uint r = i / TN;
            const uint cc = i - r*TN;
            const uint token = t0 + r;
            const uint comp = c0 + cc;
            if (token < args.n_tokens && comp < args.n_comp) {
                device float *dst = (device float *)(scores +
                    (uint64_t)token * args.score_token_stride) + comp;
                *dst = -INFINITY;
            }
        }
        return;
    }

    {
        // One compressed row per 4 threads, 32 consecutive floats per thread.
        const uint cc = tid / 4;
        const uint comp = c0 + cc;
        device const float *krow = nullptr;
        if (comp < args.n_comp) {
            krow = (device const float *)(index_comp +
                (uint64_t)comp * args.index_row_stride);
        }
        const uint d0 = (tid % 4) * 32;
        FOR_UNROLL (uint j = 0; j < 8; j++) {
            const float4 kv = krow ? *(device const float4 *)(krow + d0 + 4*j)
                                   : float4(0.0f);
            *(threadgroup half4 *)(ktg + cc*D + d0 + 4*j) = half4(kv);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc[4];
    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        acc[j] = 0.0f;
    }

    auto tq0 = tensor(qtg,          dextents<int32_t, 2>(NK, TQ));
    auto tq1 = tensor(qtg + TQ*NK,  dextents<int32_t, 2>(NK, TQ));
    auto tk = tensor(ktg, dextents<int32_t, 2>(D, TN));
    auto td = tensor(dot, dextents<int32_t, 2>(TQ, TN), array<int, 2>({1, TQ}));

    matmul2d<
        matmul2d_descriptor(TN, TQ, NK, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    // One q row per 4 threads, 8 consecutive floats per thread.  Row r covers
    // head (r / TM) of the pair and token row (r % TM).
    const uint q_r = tid / 4;
    const uint q_k4 = (tid % 4) * 8;
    const uint q_hl = q_r / TM;
    const uint q_tr = q_r % TM;
    const uint q_token = t0 + q_tr;
    device const char *q_row_base = nullptr;
    if (q_token < args.n_tokens) {
        q_row_base = q + (uint64_t)q_token * args.q_token_stride;
    }

    auto stage_q = [&](const uint head0, const uint loop_k, threadgroup half *buf) {
        const uint head = head0 + q_hl;
        half4 v0 = half4(0.0f);
        half4 v1 = half4(0.0f);
        if (q_row_base && head < args.n_head) {
            device const float4 *src4 = (device const float4 *)
                (q_row_base + (uint64_t)head * args.q_head_stride +
                 (uint64_t)(loop_k + q_k4) * sizeof(float));
            v0 = half4(src4[0]);
            v1 = half4(src4[1]);
        }
        *(threadgroup half4 *)(buf + q_r*NK + q_k4)     = v0;
        *(threadgroup half4 *)(buf + q_r*NK + q_k4 + 4) = v1;
    };

    for (uint head0 = 0; head0 < args.n_head; head0 += 2) {
        auto ct = mm.template get_destination_cooperative_tensor<decltype(tk), decltype(tq0), float>();
        #pragma unroll
        for (uint16_t i = 0; i < ct.get_capacity(); i++) {
            if (ct.is_valid_element(i)) {
                ct[i] = 0.0f;
            }
        }

        stage_q(head0, 0, qtg);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint qsel = 0;
        FOR_UNROLL (uint i = 0; i < 4; i++) {
            auto mk = tk.slice(i*NK, 0);
            auto mq = (qsel ? tq1 : tq0).slice(0, 0);
            mm.run(mk, mq, ct);
            if (i < 3) {
                qsel ^= 1u;
                stage_q(head0, (i + 1)*NK, qsel ? qtg + TQ*NK : qtg);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }

        ct.store(td);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            const uint linear = (uint)tid + j*NUM_THREADS;
            if (linear < TM*TN) {
                const uint r = linear / TN;
                const uint cc = linear - r*TN;
                const uint token = t0 + r;
                if (token < args.n_tokens) {
                    device const float *w = (device const float *)(weights +
                        (uint64_t)token * args.weights_token_stride);
                    acc[j] += max(dot[cc*TQ + r], 0.0f) * (w[head0] * args.scale);
                    if (head0 + 1 < args.n_head) {
                        acc[j] += max(dot[cc*TQ + TM + r], 0.0f) * (w[head0 + 1] * args.scale);
                    }
                }
            }
        }
        // No barrier here: the next pair's q stage and these dot reads touch
        // different buffers, and the next q-stage barrier separates the next
        // ct.store from these reads.
    }

    #pragma unroll
    for (uint j = 0; j < 4; j++) {
        const uint linear = (uint)tid + j*NUM_THREADS;
        if (linear >= TM*TN) {
            continue;
        }
        const uint r = linear / TN;
        const uint cc = linear - r*TN;
        const uint token = t0 + r;
        const uint comp = c0 + cc;
        if (token < args.n_tokens && comp < args.n_comp) {
            const uint visible = min((args.pos0 + token + 1u) / args.ratio, args.n_comp);
            device float *dst = (device float *)(scores +
                (uint64_t)token * args.score_token_stride) + comp;
            *dst = comp < visible ? acc[j] : -INFINITY;
        }
    }
}
#endif

// Collapses per-head indexer scores into one score per compressed row using the
// learned head weights. Negative head scores are clipped exactly as DS4 expects.
kernel void kernel_dsv4_indexer_weighted_sum(
        constant ds4_metal_args_dsv4_indexer_weighted_sum & args,
        device const char * scores,
        device const char * weights,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t ic = gid % args.ne0;
    const int64_t it = gid / args.ne0;

    float acc = 0.0f;
    for (int64_t ih = 0; ih < args.ne02; ++ih) {
        const float s = *((device const float *) (scores  + ic*args.nb00 + it*args.nb01 + ih*args.nb02));
        const float w = *((device const float *) (weights + ih*args.nb10 + it*args.nb11));
        acc += max(s, 0.0f) * (w * args.scale);
    }

    *((device float *) (dst + ic*args.nb0 + it*args.nb1)) = acc;
}

// Adds the periodic compressor APE directly to projected scores. The legacy
// path materializes one repeated APE segment per period and then performs this
// same single F32 add; these kernels remove only that intermediate copy graph.
kernel void kernel_dsv4_compressor_score_ape_f32(
        constant ds4_metal_args_dsv4_compressor_score_ape & args,
        device const float *score,
        device const float *ape,
        device       float *dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t)args.n_tokens * args.width;
    if ((uint64_t)gid >= total) return;

    const uint token = gid / args.width;
    const uint col = gid - token*args.width;
    const uint ape_row = (uint)(((uint64_t)args.pos0 + token) % args.ratio);
    dst[gid] = score[gid] + ape[(uint64_t)ape_row*args.width + col];
}

kernel void kernel_dsv4_compressor_score_ape_f16(
        constant ds4_metal_args_dsv4_compressor_score_ape & args,
        device const float *score,
        device const half  *ape,
        device       float *dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t total = (uint64_t)args.n_tokens * args.width;
    if ((uint64_t)gid >= total) return;

    const uint token = gid / args.width;
    const uint col = gid - token*args.width;
    const uint ape_row = (uint)(((uint64_t)args.pos0 + token) % args.ratio);
    dst[gid] = score[gid] + float(ape[(uint64_t)ape_row*args.width + col]);
}

// Fused softmax-weighted pooling of compressed KV rows. It is used when several
// compressor rows are present; the one-row case deliberately follows the
// unfused softmax/mul/sum graph in Objective-C to keep identical reductions.
kernel void kernel_dsv4_softmax_pool(
        constant ds4_metal_args_dsv4_softmax_pool & args,
        device const char * kv,
        device const char * score,
        device       char * dst,
        uint gid [[thread_position_in_grid]]) {
    const int64_t n = args.ne0 * args.ne1;
    if ((int64_t) gid >= n) {
        return;
    }

    const int64_t id = gid % args.ne0;
    const int64_t ic = gid / args.ne0;

    float max_s = -INFINITY;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        max_s = max(max_s, s);
    }

    float sum = 0.0f;
    float acc = 0.0f;
    for (int64_t ir = 0; ir < args.ne00; ++ir) {
        const float s = *((device const float *) (score + ir*args.nb10 + id*args.nb11 + ic*args.nb12));
        const float w = exp(s - max_s);
        const float v = *((device const float *) (kv + ir*args.nb00 + id*args.nb01 + ic*args.nb02));
        sum += w;
        acc += v*w;
    }

    *((device float *) (dst + id*args.nb0 + ic*args.nb1)) = acc/sum;
}



// Tensor-parallel keep-alive: a few threadgroups of FMAs dispatched
// back-to-back on a side queue while TP decode runs.  The per-layer gate
// stalls make the real workload look idle to the GPU power manager, which
// otherwise halves the clocks within a second (~2x decode regression);
// this holds them up for negligible bandwidth and a few watts.
kernel void kernel_dsv4_tp_keepalive(
        device float * out,
        constant uint & iters,
        uint tid [[thread_position_in_grid]]) {
    float a = out[tid];
    const float b = 1.000001f;
    for (uint i = 0; i < iters; i++) {
        a = fma(a, b, 0.000001f);
        a = fma(a, b, -0.000001f);
    }
    out[tid] = a;
}

// Tensor-parallel gate flag: publishes a sequence number to a slab slot the
// CPU service thread spin-reads, replacing the much slower shared-event
// signal for the GPU->CPU direction.  Ordering against the partial-output
// kernels comes from the buffer hazard on the shared slab.
kernel void kernel_dsv4_tp_flag_set(
        device atomic_uint & flag,
        constant uint & value,
        uint tid [[thread_position_in_grid]]) {
    if (tid == 0) {
        atomic_store_explicit(&flag, value, memory_order_relaxed);
    }
}

// Poll-gate flag with a payload checksum: the CPU sees the flag word as soon
// as its cache line is written back at command-buffer completion, which can
// precede the rest of the partial's lines. Publishing an integer checksum of
// the partial lets the service thread verify the payload in memory before it
// posts the RDMA send, instead of waiting for the (much later) completion
// status. Integer sums are order-independent, so the value is exact.
kernel void kernel_dsv4_tp_flag_set_checked(
        device atomic_uint & flag,
        device atomic_uint & check,
        constant uint & value,
        device const uint * payload,
        constant uint & words,
        threadgroup uint * shmem [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    uint sum = 0u;
    for (uint i = tid; i < words; i += 256u) sum += payload[i];
    sum = simd_sum(sum);
    if (tiisg == 0) shmem[sgitg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint total = 0u;
        for (uint s = 0; s < 8u; s++) total += shmem[s];
        atomic_store_explicit(&check, total ^ (value * 0x9E3779B9u), memory_order_relaxed);
        atomic_store_explicit(&flag, value, memory_order_relaxed);
    }
}

// Fused local FFN sum + checked poll-gate flag.  out = a + b for n words
// (the rank's FFN partial in its slab slot); every threadgroup adds the
// integer sum of the words it stored to a device accumulator and the
// last-arriving threadgroup publishes the checksum and the flag.  Integer
// sums are order independent, so the checksum equals the one
// kernel_dsv4_tp_flag_set_checked would compute over the same payload, and
// the gate loses one kernel.  ctl[0] counts arrivals, ctl[1] accumulates;
// the last arriver resets both for the next use of the slot.
kernel void kernel_dsv4_add2_f32_tp_flag_checked(
        constant uint & n,
        device const float * a,
        device const float * b,
        device float * out,
        device atomic_uint & flag,
        device atomic_uint & check,
        constant uint & value,
        device atomic_uint * ctl,
        constant uint & ntg,
        threadgroup uint * shmem [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint tgid [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const uint i = tgid * 256u + tid;
    uint w = 0u;
    if (i < n) {
        const float v = a[i] + b[i];
        out[i] = v;
        w = as_type<uint>(v);
    }
    w = simd_sum(w);
    if (tiisg == 0) shmem[sgitg] = w;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s = 0u;
        for (uint k = 0; k < 8u; k++) s += shmem[k];
        atomic_fetch_add_explicit(&ctl[1], s, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        const uint old = atomic_fetch_add_explicit(&ctl[0], 1u, memory_order_relaxed);
        if (old + 1u == ntg) {
            const uint total = atomic_exchange_explicit(&ctl[1], 0u, memory_order_relaxed);
            atomic_store_explicit(&ctl[0], 0u, memory_order_relaxed);
            atomic_store_explicit(&check, total ^ (value * 0x9E3779B9u), memory_order_relaxed);
            atomic_store_explicit(&flag, value, memory_order_relaxed);
        }
    }
}

// Tensor-parallel poll gate: waits for the service thread's release of gate
// `value` without parking the command buffer on a shared event, which costs
// tens of microseconds of GPU idle per gate on Apple silicon. A running
// kernel never observes an external write to a cache line it already read
// (the L2 copy stays stale until the command buffer ends), so every probe
// reads a line this command buffer has not touched: the region is consumed
// front to back, 32 lines per simdgroup probe, with growing pauses between
// rounds. The gate is always the first work of its command buffer, so the
// whole region is fresh. 8192 lines cover roughly 600 ms before `status`
// reports a timeout, which the service thread reports at the next gate.
kernel void kernel_dsv4_tp_poll_release(
        device const uint * region,
        constant uint & value,
        constant uint & nlines,
        device uint * status,
        uint tid [[thread_index_in_threadgroup]]) {
    const uint rounds = nlines / 32u;
    float spin = 1.0f;
    for (uint r = 0; r < rounds; r++) {
        const uint line = r * 32u + tid;
        const uint x = region[line * 32u];
        const uint hit = (x == value) ? line : 0xffffffffu;
        const uint first = simd_min(hit);
        if (first != 0xffffffffu) {
            if (tid == 0) status[0] = first;
            return;
        }
        uint pause = 0u;
        if (r >= 160u) pause = 375000u;      /* ~5 ms   x 96 rounds */
        else if (r >= 96u) pause = 150000u;  /* ~2 ms   x 64 rounds */
        else if (r >= 64u) pause = 20000u;   /* ~270 us x 32 rounds */
        else if (r >= 32u) pause = 1200u;    /* ~16 us  x 32 rounds */
        else if (r >= 16u) pause = 150u;     /* ~2 us   x 16 rounds */
        for (uint i = 0; i < pause; i++) {
            spin = fma(spin, 1.000001f, 0.000001f);
            spin = fma(spin, 1.000001f, -0.000001f);
        }
    }
    if (tid == 0) status[0] = 0xffffffffu;
    if (spin == 0.0f) status[1] = 0u; /* keeps the pause loop alive */
}

// Ratio-4 compressor pooling without materializing the [n_comp, 8, head_dim]
// KV and score packs. The row mapping and both reduction loops deliberately
// match kernel_dsv4_softmax_pool so the arithmetic order is unchanged.
kernel void kernel_dsv4_softmax_pool_ratio4_direct(
        constant ds4_metal_args_dsv4_softmax_pool_ratio4_direct & args,
        device const float * kv,
        device const float * score,
        device const float * state_kv,
        device const float * state_score,
        device       float * dst,
        uint gid [[thread_position_in_grid]]) {
    const uint64_t n = (uint64_t)args.head_dim * args.n_comp;
    if ((uint64_t)gid >= n || args.head_dim == 0u) {
        return;
    }

    const uint64_t id = gid % args.head_dim;
    const uint64_t ic = gid / args.head_dim;
    const uint64_t input_row_stride = 2ull * args.head_dim;

    float max_s = -INFINITY;
    float sum = 0.0f;
    float acc = 0.0f;
    if (ic != 0u) {
        const int64_t token_base = (int64_t)ic * 4 - 4;
        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            const uint64_t token = (uint64_t)(token_base + ir);
            const uint64_t src = token * input_row_stride +
                                 ((uint64_t)ir >> 2u) * args.head_dim + id;
            const float s = score[src];
            max_s = max(max_s, s);
        }

        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            const uint64_t token = (uint64_t)(token_base + ir);
            const uint64_t src = token * input_row_stride +
                                 ((uint64_t)ir >> 2u) * args.head_dim + id;
            const float s = score[src];
            const float w = exp(s - max_s);
            const float v = kv[src];
            sum += w;
            acc += v*w;
        }
    } else {
        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            float s;
            if (ir >= 4) {
                const uint64_t src = (uint64_t)(ir - 4) * input_row_stride +
                                     args.head_dim + id;
                s = score[src];
            } else if (args.replay != 0u) {
                s = state_score[(uint64_t)ir * input_row_stride + id];
            } else {
                s = -INFINITY;
            }
            max_s = max(max_s, s);
        }

        for (int64_t ir = 0; ir < args.n_rows; ++ir) {
            float s;
            float v;
            if (ir >= 4) {
                const uint64_t src = (uint64_t)(ir - 4) * input_row_stride +
                                     args.head_dim + id;
                s = score[src];
                v = kv[src];
            } else if (args.replay != 0u) {
                const uint64_t src = (uint64_t)ir * input_row_stride + id;
                s = state_score[src];
                v = state_kv[src];
            } else {
                s = -INFINITY;
                v = 0.0f;
            }
            const float w = exp(s - max_s);
            sum += w;
            acc += v*w;
        }
    }

    dst[ic * args.head_dim + id] = acc/sum;
}

// llms LLMS_DS4_IDX_SCORE_SG (apply_ds4_idx_score_sg.py): indexer scores, R compressed rows per simdgroup, no threadgroup memory,
// no barrier; same per-lane dots, same simd_sum, same relu x scale x weight, heads added in the engine's order, strictly.
template<short R, short NSGX>
void kernel_glm_indexer_score_one_sg_impl(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        uint tg,
        ushort lane,
        ushort sg) {
    const uint row0 = (tg * (uint)NSGX + (uint)sg) * (uint)R;
    if (row0 >= args.n_rows || args.n_head != 32u || args.head_dim != 128u || args.cache_f16 != 0u) {
        return;
    }
    float4 k[R];
    FOR_UNROLL (short r = 0; r < R; ++r) {
        const uint row = min(row0 + (uint)r, args.n_rows - 1u);   // reads clamped, writes guarded
        k[r] = ((device const float4 *)indexer_key_cache)[(uint64_t)row * 32u + lane];
    }
    float acc[R];
    FOR_UNROLL (short r = 0; r < R; ++r) acc[r] = 0.0f;
    device const float4 *q4 = (device const float4 *)q;
    for (uint head0 = 0; head0 < 32u; head0 += 4u) {
        float p[R][4];
        FOR_UNROLL (short j = 0; j < 4; ++j) {
            const float4 qv = q4[(head0 + (uint)j) * 32u + lane];
            const float w = weights[head0 + (uint)j];
            FOR_UNROLL (short r = 0; r < R; ++r) {
                float s = dot(qv, k[r]);
                s = simd_sum(s);
                p[r][j] = max(s * args.scale, 0.0f) * w;
            }
        }
        FOR_UNROLL (short r = 0; r < R; ++r) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            acc[r] = acc[r] + p[r][0];
            acc[r] = acc[r] + p[r][1];
            acc[r] = acc[r] + p[r][2];
            acc[r] = acc[r] + p[r][3];
        }
    }
    if (lane == 0) {
        FOR_UNROLL (short r = 0; r < R; ++r) {
            if (row0 + (uint)r < args.n_rows) scores[row0 + (uint)r] = acc[r];
        }
    }
}
kernel void kernel_glm_indexer_score_one_sg(
        constant ds4_metal_args_glm_indexer_score_one & args,
        device const char *q,
        device const float *weights,
        device const char *indexer_key_cache,
        device float *scores,
        uint tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    kernel_glm_indexer_score_one_sg_impl<4, 4>(args, q, weights, indexer_key_cache, scores, tg, lane, sg);   // 4 rows per simdgroup, 4 simdgroups
}

// ---- appended from dsv4_misc_llms.metal (rows attention, LLMS_DS4_V41_ROWS_ATTN) ----

// ---- llms LLMS_DS4_V41_ROWS_ATTN (2026-10-08, user critique 'step B'): one dispatch for the decode attention of every row of a
// batched step. Copy of kernel_dsv4_indexed_mixed_attention_heads8_split / _split_reduce with every per-session quantity taken from a
// per-row record (GPU addresses of q, raw window ring, compressed cache, top-k row, output heads; position, ring start, n_raw, n_comp).
struct llms_v41_row {
    ulong q, raw, comp, topk, dst;
    uint pos, raw_start, n_raw, n_comp;
};

kernel void kernel_llms_v41_rows_attn_split(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        constant llms_v41_row *rows,
        device       char *tmp,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_block = 16u;
    constexpr uint vecs_per_row = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    const uint split = tgpig.z;
    const uint n_splits = args.n_splits;
    if (token >= args.n_tokens || head >= args.n_head || n_splits < 2u || n_splits > 31u || split >= n_splits) return;
    const llms_v41_row R = rows[token];
    device const char *q = reinterpret_cast<device const char *>(R.q);
    device const char *raw_kv = reinterpret_cast<device const char *>(R.raw);
    device const char *comp_kv = reinterpret_cast<device const char *>(R.comp);
    device const int32_t *row_topk = reinterpret_cast<device const int32_t *>(R.topk);

    device const float4 *q4 = (device const float4 *)(q + (uint64_t)head * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];
    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const uint qpos = R.pos;
    const uint first_raw_pos = qpos + 1u - R.n_raw;
    const uint raw_last_pos = qpos;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, R.n_comp);

    for (uint seq0 = split_first; seq0 < split_last; seq0 += rows_per_block) {
        const uint n_rows = min(rows_per_block, split_last - seq0);
        for (uint off = (uint)tid; off < n_rows * vecs_per_row; off += 256u) {
            const uint r = off / vecs_per_row;
            const uint c = off - r * vecs_per_row;
            const uint seq = seq0 + r;
            half4 value = half4(0.0h);
            if (seq < raw_count) {
                const uint pos = raw_first + seq;
                const uint logical = pos - first_raw_pos;
                const uint row = (R.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv + (uint64_t)row * args.raw_row_stride);
                value = (half4)src[c];
            } else {
                const int32_t idx = row_topk[seq - raw_count];
                if (idx >= 0 && (uint)idx < visible) {
                    value = dsv4_load_cache_h4(comp_kv, args.comp_row_stride, (uint)idx, c, args.comp_kv_f16 != 0u);
                }
            }
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            const uint seq = seq0 + r;
            bool valid = true;
            if (seq >= raw_count) {
                const int32_t idx = row_topk[seq - raw_count];
                valid = idx >= 0 && (uint)idx < visible;
            }
            if (valid) {
                dsv4_attend_shared_h4_row_at(kv_shared, r, q0, q1, q2, q3, args.scale, lane, M, S, o0, o1, o2, o3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device float4 *partials = (device float4 *)tmp;
    partials[(row * vecs_per_row + lane +  0u) * n_splits + split] = o0;
    partials[(row * vecs_per_row + lane + 32u) * n_splits + split] = o1;
    partials[(row * vecs_per_row + lane + 64u) * n_splits + split] = o2;
    partials[(row * vecs_per_row + lane + 96u) * n_splits + split] = o3;
    if (lane == 0u) {
        device float *stats = (device float *)(partials + n_rows * vecs_per_row * n_splits);
        const uint64_t stat = (row * n_splits + split) * 2u;
        stats[stat + 0u] = S;
        stats[stat + 1u] = M;
    }
}

kernel void kernel_llms_v41_rows_attn_reduce(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        constant llms_v41_row *rows,
        device const char *tmp,
        device const char *sinks,
        uint tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]]) {
    constexpr uint vecs_per_row = 128u;
    const uint n_splits = args.n_splits;
    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = tgpig;
    if (row >= n_rows || n_splits < 2u || n_splits > 31u) return;
    const uint token = (uint)(row / args.n_head);
    const uint head = (uint)(row % args.n_head);
    device const float4 *partials = (device const float4 *)tmp;
    device const float *stats = (device const float *)(partials + n_rows * vecs_per_row * n_splits);
    float part_sum = 0.0f;
    float part_max = -FLT_MAX/2.0f;
    if ((uint)lane < n_splits) {
        const uint64_t stat = (row * n_splits + (uint)lane) * 2u;
        part_sum = stats[stat + 0u];
        part_max = stats[stat + 1u];
    } else if ((uint)lane == n_splits) {
        part_sum = 1.0f;
        part_max = ((device const float *)sinks)[head];
    }
    const float global_max = simd_max(part_max);
    const float part_scale = part_sum > 0.0f ? exp(part_max - global_max) : 0.0f;
    const float total_sum = simd_sum(part_sum * part_scale);
    const float inv_sum = total_sum > 0.0f ? 1.0f / total_sum : 0.0f;
    device float4 *out = reinterpret_cast<device float4 *>(rows[token].dst + (ulong)head * args.dst_head_stride);
    for (uint i = (uint)sg; i < vecs_per_row; i += 4u) {
        float4 value = float4(0.0f);
        if ((uint)lane < n_splits) {
            value = partials[(row * vecs_per_row + i) * n_splits + (uint)lane] * part_scale;
        }
        value = simd_sum(value);
        if (lane == 0u) {
            out[i] = value * inv_sum;
        }
    }
}

/* llms ROWS_ROPE (make_misc_rope_rows.py): kernel_dsv41_rope for several rows at their own positions and addresses */
struct llms_rope_row { ulong x; uint pos; uint pad; };
struct llms_rope_rows_args { uint width; uint heads; uint rows; uint inverse; uint bf16_first; uint pad[3]; float frequencies[32]; };
kernel void kernel_llms_dsv41_rope_rows(
        constant llms_rope_rows_args &args [[buffer(0)]],
        constant llms_rope_row *rows [[buffer(1)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    device float *x = (device float *)rows[group.y].x;
    const float theta = float(rows[group.y].pos) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong h0 = (ulong)group.x * args.width;
    if (args.bf16_first) for (uint e = lane; e < args.width - 64u; e += 32u) x[h0 + e] = dsv41_bf16(x[h0 + e]);
    const ulong i = h0 + args.width - 64u + 2u * lane;
    float re = x[i], im = x[i + 1u];
    if (args.bf16_first) { re = dsv41_bf16(re); im = dsv41_bf16(im); }
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

// llms ROWS_ATTN_HG (make_misc_attn_hg.py): the split kernel with 16 heads per threadgroup (512 threads); the KV rows staged 4 times per token instead of 8
kernel void kernel_llms_v41_rows_attn_split_hg16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        constant llms_v41_row *rows,
        device       char *tmp,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_block = 16u;
    constexpr uint vecs_per_row = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y * 16u + (uint)sg;   /* llms ROWS_ATTN_HG: 16 heads per threadgroup */
    const uint split = tgpig.z;
    const uint n_splits = args.n_splits;
    if (token >= args.n_tokens || head >= args.n_head || n_splits < 2u || n_splits > 31u || split >= n_splits) return;
    const llms_v41_row R = rows[token];
    device const char *q = reinterpret_cast<device const char *>(R.q);
    device const char *raw_kv = reinterpret_cast<device const char *>(R.raw);
    device const char *comp_kv = reinterpret_cast<device const char *>(R.comp);
    device const int32_t *row_topk = reinterpret_cast<device const int32_t *>(R.topk);

    device const float4 *q4 = (device const float4 *)(q + (uint64_t)head * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];
    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const uint qpos = R.pos;
    const uint first_raw_pos = qpos + 1u - R.n_raw;
    const uint raw_last_pos = qpos;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, R.n_comp);

    for (uint seq0 = split_first; seq0 < split_last; seq0 += rows_per_block) {
        const uint n_rows = min(rows_per_block, split_last - seq0);
        for (uint off = (uint)tid; off < n_rows * vecs_per_row; off += 512u) {
            const uint r = off / vecs_per_row;
            const uint c = off - r * vecs_per_row;
            const uint seq = seq0 + r;
            half4 value = half4(0.0h);
            if (seq < raw_count) {
                const uint pos = raw_first + seq;
                const uint logical = pos - first_raw_pos;
                const uint row = (R.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv + (uint64_t)row * args.raw_row_stride);
                value = (half4)src[c];
            } else {
                const int32_t idx = row_topk[seq - raw_count];
                if (idx >= 0 && (uint)idx < visible) {
                    value = dsv4_load_cache_h4(comp_kv, args.comp_row_stride, (uint)idx, c, args.comp_kv_f16 != 0u);
                }
            }
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            const uint seq = seq0 + r;
            bool valid = true;
            if (seq >= raw_count) {
                const int32_t idx = row_topk[seq - raw_count];
                valid = idx >= 0 && (uint)idx < visible;
            }
            if (valid) {
                dsv4_attend_shared_h4_row_at(kv_shared, r, q0, q1, q2, q3, args.scale, lane, M, S, o0, o1, o2, o3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device float4 *partials = (device float4 *)tmp;
    partials[(row * vecs_per_row + lane +  0u) * n_splits + split] = o0;
    partials[(row * vecs_per_row + lane + 32u) * n_splits + split] = o1;
    partials[(row * vecs_per_row + lane + 64u) * n_splits + split] = o2;
    partials[(row * vecs_per_row + lane + 96u) * n_splits + split] = o3;
    if (lane == 0u) {
        device float *stats = (device float *)(partials + n_rows * vecs_per_row * n_splits);
        const uint64_t stat = (row * n_splits + split) * 2u;
        stats[stat + 0u] = S;
        stats[stat + 1u] = M;
    }
}


// llms ROWS_ATTN_HG (make_misc_attn_hg.py): the split kernel with 32 heads per threadgroup (1024 threads); the KV rows staged 2 times per token instead of 8
kernel void kernel_llms_v41_rows_attn_split_hg32(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        constant llms_v41_row *rows,
        device       char *tmp,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_block = 16u;
    constexpr uint vecs_per_row = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y * 32u + (uint)sg;   /* llms ROWS_ATTN_HG: 32 heads per threadgroup */
    const uint split = tgpig.z;
    const uint n_splits = args.n_splits;
    if (token >= args.n_tokens || head >= args.n_head || n_splits < 2u || n_splits > 31u || split >= n_splits) return;
    const llms_v41_row R = rows[token];
    device const char *q = reinterpret_cast<device const char *>(R.q);
    device const char *raw_kv = reinterpret_cast<device const char *>(R.raw);
    device const char *comp_kv = reinterpret_cast<device const char *>(R.comp);
    device const int32_t *row_topk = reinterpret_cast<device const int32_t *>(R.topk);

    device const float4 *q4 = (device const float4 *)(q + (uint64_t)head * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];
    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const uint qpos = R.pos;
    const uint first_raw_pos = qpos + 1u - R.n_raw;
    const uint raw_last_pos = qpos;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, R.n_comp);

    for (uint seq0 = split_first; seq0 < split_last; seq0 += rows_per_block) {
        const uint n_rows = min(rows_per_block, split_last - seq0);
        for (uint off = (uint)tid; off < n_rows * vecs_per_row; off += 1024u) {
            const uint r = off / vecs_per_row;
            const uint c = off - r * vecs_per_row;
            const uint seq = seq0 + r;
            half4 value = half4(0.0h);
            if (seq < raw_count) {
                const uint pos = raw_first + seq;
                const uint logical = pos - first_raw_pos;
                const uint row = (R.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv + (uint64_t)row * args.raw_row_stride);
                value = (half4)src[c];
            } else {
                const int32_t idx = row_topk[seq - raw_count];
                if (idx >= 0 && (uint)idx < visible) {
                    value = dsv4_load_cache_h4(comp_kv, args.comp_row_stride, (uint)idx, c, args.comp_kv_f16 != 0u);
                }
            }
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            const uint seq = seq0 + r;
            bool valid = true;
            if (seq >= raw_count) {
                const int32_t idx = row_topk[seq - raw_count];
                valid = idx >= 0 && (uint)idx < visible;
            }
            if (valid) {
                dsv4_attend_shared_h4_row_at(kv_shared, r, q0, q1, q2, q3, args.scale, lane, M, S, o0, o1, o2, o3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device float4 *partials = (device float4 *)tmp;
    partials[(row * vecs_per_row + lane +  0u) * n_splits + split] = o0;
    partials[(row * vecs_per_row + lane + 32u) * n_splits + split] = o1;
    partials[(row * vecs_per_row + lane + 64u) * n_splits + split] = o2;
    partials[(row * vecs_per_row + lane + 96u) * n_splits + split] = o3;
    if (lane == 0u) {
        device float *stats = (device float *)(partials + n_rows * vecs_per_row * n_splits);
        const uint64_t stat = (row * n_splits + split) * 2u;
        stats[stat + 0u] = S;
        stats[stat + 1u] = M;
    }
}


// ---- llms IDX_SCORE_SGM (make_misc_idx_tiles.py): the indexer score on simdgroup-matrix tiles ----
template<short NRT>
void kernel_llms_indexer_score_sgm_impl(constant ds4_metal_args_glm_indexer_score_one & args, device const char *q, device const float *weights,
                                        device const char *indexer_key_cache, device float *scores, threadgroup float *shmem, uint tg, ushort lane, ushort sg) {
    constexpr uint NR = NRT * 8;
    if (args.n_head != 32u || args.head_dim != 128u || args.cache_f16 != 0u || args.n_rows < NR) return;
    const uint r0 = min(tg * NR, args.n_rows - NR);
    device const float *K = (device const float *)indexer_key_cache + (uint64_t)r0 * 128u;
    device const float *Q = (device const float *)q + (uint)sg * 8u * 128u;
    simdgroup_float8x8 C[NRT];
    FOR_UNROLL (short t = 0; t < NRT; t++) C[t] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (short kt = 0; kt < 16; kt++) {
        simdgroup_float8x8 B; simdgroup_load(B, Q + kt * 8, 128, ulong2(0, 0), true);
        FOR_UNROLL (short t = 0; t < NRT; t++) { simdgroup_float8x8 A; simdgroup_load(A, K + t * 8 * 128 + kt * 8, 128); simdgroup_multiply_accumulate(C[t], A, B, C[t]); }
    }
    FOR_UNROLL (short t = 0; t < NRT; t++) simdgroup_store(C[t], shmem + (t * 8) * 32 + sg * 8, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        for (uint row = lane; row < NR; row += 32u) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            float acc = 0.0f;
            for (uint h = 0; h < 32u; h++) acc = acc + max(shmem[row * 32u + h] * args.scale, 0.0f) * weights[h];
            scores[r0 + row] = acc;
        }
    }
}
#define LLMS_IDX_SGM(NAME, NRT) \
kernel void NAME(constant ds4_metal_args_glm_indexer_score_one & args, device const char *q, device const float *weights, \
                 device const char *indexer_key_cache, device float *scores, threadgroup float *shmem [[threadgroup(0)]], \
                 uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) { \
    kernel_llms_indexer_score_sgm_impl<NRT>(args, q, weights, indexer_key_cache, scores, shmem, tg, lane, sg); }
LLMS_IDX_SGM(kernel_llms_indexer_score_sgm_t2, 2)
LLMS_IDX_SGM(kernel_llms_indexer_score_sgm_t4, 4)
LLMS_IDX_SGM(kernel_llms_indexer_score_sgm_t8, 8)


// ---- llms ROWS_WIN (apply_ds4_rows_win.py): one dispatch writes every row's KV row into its own raw window slot ----
struct llms_copy_row { ulong src, dst; };
kernel void kernel_llms_rows_copy(
        constant uint & nfloat4 [[buffer(0)]],
        constant llms_copy_row * rows [[buffer(1)]],
        uint tg [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort nt [[threads_per_threadgroup]]) {
    device const float4 * s = reinterpret_cast<device const float4 *>(rows[tg].src);
    device float4 * d = reinterpret_cast<device float4 *>(rows[tg].dst);
    for (uint i = tid; i < nfloat4; i += nt) d[i] = s[i];
}


// ---- llms IDX_HALF (make_misc_idx_half.py, 2026-10-11): the SGM indexer score on a HALF key cache ----
template<short NRT>
void kernel_llms_indexer_score_sgm_h_impl(constant ds4_metal_args_glm_indexer_score_one & args, device const char *q, device const float *weights,
                                          device const char *indexer_key_cache, device float *scores, threadgroup float *shmem, uint tg, uint ntg, ushort lane, ushort sg) {
    constexpr uint NR = NRT * 8;
    if (args.n_head != 32u || args.head_dim != 128u || args.cache_f16 == 0u || args.n_rows < NR) return;
    threadgroup half *qs = (threadgroup half *)(shmem + NR * 32u);          // the query: 32 heads x 128, after the C tiles; staged ONCE
    for (uint i = (uint)sg * 32u + lane; i < 32u * 128u; i += 128u) qs[i] = (half)(((device const float *)q)[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const half *Q = qs + (uint)sg * 8u * 128u;
    const uint nblk = (args.n_rows + NR - 1u) / NR;
    for (uint blk = tg; blk < nblk; blk += ntg) {                          // persistent: a threadgroup walks several row blocks
        const uint r0 = min(blk * NR, args.n_rows - NR);
        device const half *K = (device const half *)indexer_key_cache + (uint64_t)r0 * 128u;
        simdgroup_float8x8 C[NRT];
        FOR_UNROLL (short t = 0; t < NRT; t++) C[t] = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (short kt = 0; kt < 16; kt++) {
            simdgroup_half8x8 B; simdgroup_load(B, Q + kt * 8, 128, ulong2(0, 0), true);
            FOR_UNROLL (short t = 0; t < NRT; t++) { simdgroup_half8x8 A; simdgroup_load(A, K + t * 8 * 128 + kt * 8, 128); simdgroup_multiply_accumulate(C[t], A, B, C[t]); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short t = 0; t < NRT; t++) simdgroup_store(C[t], shmem + (t * 8) * 32 + sg * 8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (uint row = lane; row < NR; row += 32u) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
                float acc = 0.0f;
                for (uint h = 0; h < 32u; h++) acc = acc + max(shmem[row * 32u + h] * args.scale, 0.0f) * weights[h];
                scores[r0 + row] = acc;
            }
        }
    }
}
#define LLMS_IDX_SGM_H(NAME, NRT) \
kernel void NAME(constant ds4_metal_args_glm_indexer_score_one & args, device const char *q, device const float *weights, \
                 device const char *indexer_key_cache, device float *scores, threadgroup float *shmem [[threadgroup(0)]], \
                 uint tg [[threadgroup_position_in_grid]], uint ntg [[threadgroups_per_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) { \
    kernel_llms_indexer_score_sgm_h_impl<NRT>(args, q, weights, indexer_key_cache, scores, shmem, tg, ntg, lane, sg); }
LLMS_IDX_SGM_H(kernel_llms_indexer_score_sgm_h_t2, 2)
LLMS_IDX_SGM_H(kernel_llms_indexer_score_sgm_h_t4, 4)
LLMS_IDX_SGM_H(kernel_llms_indexer_score_sgm_h_t8, 8)


// ---- llms KV_FP4 (make_misc_kv_fp4.py, 2026-10-11): the compressed KV cache in V4.1's FP4 format, 288 bytes per 512-value row ----
static inline int llms_e4m3fn_index(float x) {   // the magnitude index dsv4_e4m3fn_dequant picks (x > 0 here)
    const float ax = min(abs(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) { const int mid = (lo + hi + 1) >> 1; if (dsv4_e4m3fn_value(mid) <= ax) lo = mid; else hi = mid - 1; }
    int best = lo;
    if (best < 126) {
        const float best_diff = abs(ax - dsv4_e4m3fn_value(best));
        const float next_diff = abs(ax - dsv4_e4m3fn_value(best + 1));
        if (next_diff < best_diff || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)) best = best + 1;
    }
    return best;
}
static inline int llms_e2m1fn_index(float x) {   // the magnitude index dsv4_e2m1fn_dequant picks
    const float ax = min(abs(x), 6.0f);
    int best = 0; float best_diff = abs(ax - dsv4_e2m1fn_values[0]);
    for (int i = 1; i < 8; i++) { const float diff = abs(ax - dsv4_e2m1fn_values[i]);
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) { best = i; best_diff = diff; } }
    return best;
}
static inline float llms_fp4_code_value(uint code) {   // sign | magnitude index; code 8 = -0.0
    const float m = dsv4_e2m1fn_values[code & 7u];
    return (code & 8u) ? -m : m;
}
static inline half4 llms_fp4_load_h4(device const char *comp, uint row, uint c) {   // dims 4c..4c+3 of a packed row
    device const uchar *rp = (device const uchar *)comp + (ulong)row * 288u;
    const uint nib = (uint)((device const ushort *)rp)[c];
    const float sc = dsv4_e4m3fn_value((int)rp[256u + (c >> 2)]);
    return half4((half)(llms_fp4_code_value(nib & 15u) * sc), (half)(llms_fp4_code_value((nib >> 4) & 15u) * sc),
                 (half)(llms_fp4_code_value((nib >> 8) & 15u) * sc), (half)(llms_fp4_code_value((nib >> 12) & 15u) * sc));
}
struct llms_fp4_pack_args { uint rows; uint canary; ulong dst_row0; };   // dst_row0: the first packed row written; canary: block's first code sign-flipped
kernel void kernel_llms_dsv41_quantize_fp4_pack(constant llms_fp4_pack_args & a, device float *x, device uchar *dst,
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint block = 16u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < 512u;
    const ulong index = (ulong)group.y * 512u + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    const int sidx = llms_e4m3fn_index(max(amax, 0.01171875f) / 6.0f);
    const float scale = dsv4_e4m3fn_value(sidx);                       // == dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
    const int cidx = llms_e2m1fn_index(abs(value) / scale);
    const float result = copysign(dsv4_e2m1fn_values[cidx], value) * scale;   // == mode 3's result
    if (valid) x[index] = dsv41_bf16(result);
    const uint code = ((uint)cidx | (signbit(result) ? 8u : 0u)) ^ (a.canary && lane == 0u ? 8u : 0u);
    const uint upper = simd_shuffle_down(code, (ushort)1);
    device uchar *row = dst + (a.dst_row0 + (ulong)group.y) * 288u;
    if (valid && (lane & 1u) == 0u) row[column >> 1] = (uchar)(code | (upper << 4));
    if (lane == 0u) row[256u + group.x] = (uchar)sidx;
}
// llms C2H_SEL (apply_c2h_sel.py): the selected rows only, in slot order; id' = slot if accepted else -1
struct llms_fp4_psel_args { uint n_sel; uint visible; uint canary; };
kernel void kernel_llms_fp4_pack_selected(constant llms_fp4_psel_args & a, device const char *src, device const int *sel, device half4 *dst,
        device int *ids, uint gid [[thread_position_in_grid]]) {
    const uint s = gid >> 7, c = gid & 127u;
    if (s >= a.n_sel) return;
    const int id = sel[s];
    const bool ok = id >= 0 && (uint)id < a.visible && !(a.canary && s == a.n_sel - 1u);   // canary: the last selected slot dropped
    if (ok) dst[(ulong)s * 128u + c] = llms_fp4_load_h4(src, (uint)id, c);
    if (c == 0u) ids[s] = ok ? (int)s : -1;
}
kernel void kernel_llms_fp4_rows_to_half(constant uint & n_rows, device const char *src, device half4 *dst, uint gid [[thread_position_in_grid]]) {
    const uint row = gid >> 7, c = gid & 127u;
    if (row >= n_rows) return;
    dst[(ulong)row * 128u + c] = llms_fp4_load_h4(src, row, c);
}

kernel void kernel_llms_v41_rows_attn_split_fp4(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        constant llms_v41_row *rows,
        device       char *tmp,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    constexpr uint rows_per_block = 16u;
    constexpr uint vecs_per_row = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y * 8u + (uint)sg;
    const uint split = tgpig.z;
    const uint n_splits = args.n_splits;
    if (token >= args.n_tokens || head >= args.n_head || n_splits < 2u || n_splits > 31u || split >= n_splits) return;
    const llms_v41_row R = rows[token];
    device const char *q = reinterpret_cast<device const char *>(R.q);
    device const char *raw_kv = reinterpret_cast<device const char *>(R.raw);
    device const char *comp_kv = reinterpret_cast<device const char *>(R.comp);
    device const int32_t *row_topk = reinterpret_cast<device const int32_t *>(R.topk);

    device const float4 *q4 = (device const float4 *)(q + (uint64_t)head * args.q_head_stride);
    const half4 q0 = (half4)q4[lane +  0];
    const half4 q1 = (half4)q4[lane + 32];
    const half4 q2 = (half4)q4[lane + 64];
    const half4 q3 = (half4)q4[lane + 96];
    float M = -FLT_MAX/2.0f;
    float S = 0.0f;
    float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const uint qpos = R.pos;
    const uint first_raw_pos = qpos + 1u - R.n_raw;
    const uint raw_last_pos = qpos;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, R.n_comp);

    for (uint seq0 = split_first; seq0 < split_last; seq0 += rows_per_block) {
        const uint n_rows = min(rows_per_block, split_last - seq0);
        for (uint off = (uint)tid; off < n_rows * vecs_per_row; off += 256u) {
            const uint r = off / vecs_per_row;
            const uint c = off - r * vecs_per_row;
            const uint seq = seq0 + r;
            half4 value = half4(0.0h);
            if (seq < raw_count) {
                const uint pos = raw_first + seq;
                const uint logical = pos - first_raw_pos;
                const uint row = (R.raw_start + logical) % args.raw_cap;
                device const float4 *src = (device const float4 *)(raw_kv + (uint64_t)row * args.raw_row_stride);
                value = (half4)src[c];
            } else {
                const int32_t idx = row_topk[seq - raw_count];
                if (idx >= 0 && (uint)idx < visible) {
                    value = llms_fp4_load_h4(comp_kv, (uint)idx, c);   // llms KV_FP4
                }
            }
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint r = 0; r < n_rows; r++) {
            const uint seq = seq0 + r;
            bool valid = true;
            if (seq >= raw_count) {
                const int32_t idx = row_topk[seq - raw_count];
                valid = idx >= 0 && (uint)idx < visible;
            }
            if (valid) {
                dsv4_attend_shared_h4_row_at(kv_shared, r, q0, q1, q2, q3, args.scale, lane, M, S, o0, o1, o2, o3);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    const uint64_t row = (uint64_t)token * args.n_head + head;
    device float4 *partials = (device float4 *)tmp;
    partials[(row * vecs_per_row + lane +  0u) * n_splits + split] = o0;
    partials[(row * vecs_per_row + lane + 32u) * n_splits + split] = o1;
    partials[(row * vecs_per_row + lane + 64u) * n_splits + split] = o2;
    partials[(row * vecs_per_row + lane + 96u) * n_splits + split] = o3;
    if (lane == 0u) {
        device float *stats = (device float *)(partials + n_rows * vecs_per_row * n_splits);
        const uint64_t stat = (row * n_splits + split) * 2u;
        stats[stat + 0u] = S;
        stats[stat + 1u] = M;
    }
}



// ---- llms TOPK_RADIX (make_misc_topk_radix.py, 2026-10-11): top-k by radix select, one threadgroup of 1024 threads per row ----
struct llms_topk_args { uint n, k, rows, canary; ulong score_stride, sel_stride; uint mode; };   /* mode: llms TK_IDXORD */
static inline uint llms_fkey(float f) { const uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
kernel void kernel_llms_topk_radix(constant llms_topk_args & a, device const float *scores, device int *selected,
                                   threadgroup uint *tg [[threadgroup(0)]], uint row [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint *hist = tg;                          // 2048 bins
    threadgroup uint *ctl = tg + 2048;                    // [0] digit, [1] need, [2] count >, [3] count ==
    threadgroup uint *skey = tg + 2048 + 16;              // 1024 keys of the collected (sorted later)
    threadgroup int  *sidx = (threadgroup int *)(tg + 2048 + 16 + 1024);
    if (row >= a.rows) return;
    device const float *s = scores + (ulong)row * a.score_stride;
    const uint n = a.n, k = a.k;
    uint prefix = 0u, pmask = 0u, need = k;
    const bool all = k >= n;   /* llms TOPK_FAST: every element selected */
    for(uint pass=0u;pass<(all?0u:3u);pass++) {
        const uint shift=pass==0u?21u:pass==1u?10u:0u;
        const uint nb=pass==2u?1024u:2048u;
        // Two replicas with16wordpadding. Last live index2064+2047=4111.
        for(uint i=tid;i<nb;i+=1024u){tg[i]=0u;tg[2064u+i]=0u;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint lane=tid&31u,sg=tid>>5;
        threadgroup uint *local_hist=tg+(sg&1u)*2064u;
        for(uint i=tid;i<n;i+=1024u) {
            const uint key=llms_fkey(s[i]);
            if((key&pmask)==prefix)
                atomic_fetch_add_explicit((threadgroup atomic_uint*)&local_hist[(key>>shift)&(nb-1u)],1u,memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint per=nb/1024u,b_hi=nb-1u-tid*per,b_lo=per==2u?b_hi-1u:b_hi;
        const uint c_hi=tg[b_hi]+tg[2064u+b_hi];
        const uint c_lo=per==2u?(tg[b_lo]+tg[2064u+b_lo]):0u,loc=c_hi+c_lo;
        const uint incl=simd_prefix_inclusive_sum(loc);
        // Every histogram count is now in registers; all readers must finish
        // before the same storage becomes scan/selection metadata.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup uint *rs=tg;
        threadgroup uint *rc=tg+96u;
        if(lane==31u)rs[sg]=incl;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(sg==0u){const uint v=rs[lane],vi=simd_prefix_inclusive_sum(v);rs[32u+lane]=vi-v;if(lane==31u)rs[64u]=vi;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint above=rs[32u+sg]+incl-loc;
        if(above<need&&above+c_hi>=need){rc[0]=b_hi;rc[1]=need-above;rc[2]=c_hi;}
        else if(per==2u&&above+c_hi<need&&above+loc>=need){rc[0]=b_lo;rc[1]=need-(above+c_hi);rc[2]=c_lo;}
        if(tid==0u&&rs[64u]<need){rc[0]=0u;rc[1]=need-rs[64u];rc[2]=0u;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix|=rc[0]<<shift;pmask|=(nb-1u)<<shift;need=rc[1];
        const bool saturated=rc[2]==need && prefix>0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(pass<2u && saturated){prefix-=1u;need=0u;break;}
    }
    // Strict cutoff: key > prefix wins; need lowest-ID ties at prefix also win.
    // A saturated bucket uses prefix=lower_bound-1 and need=0.
    if (a.mode == 1u && n <= 62720u) {
        device int *out = selected + (ulong)row*a.sel_stride;
        if (all) { for(uint i=tid;i<k;i+=1024u) out[i]=i<n ? (int)i ^ (a.canary && i==0u ? 1:0) : -1; return; }
        const uint lane=tid&31u,sg=tid>>5;
        threadgroup uint *masks=tg+192u;
        for(uint phase=0u;phase<2u;phase++) {
            const uint extent=phase==0u?min(n,1024u):n;
            const uint span=((extent+1023u)/1024u)*32u;
            const uint begin=sg*span,end=min(begin+span,extent);
            uint eq_count=0u,gt_count=0u;
            for(uint i0=begin;i0<end;i0+=32u) {
                const uint i=i0+lane;
                const uint key=i<end?llms_fkey(s[i]):0u;
                const bool eq=i<end && key==prefix, gt=i<end && key>prefix;
                eq_count+=uint(eq);gt_count+=uint(gt);
                const uint eb=(uint)((uint64_t)simd_ballot(eq));
                const uint gb=(uint)((uint64_t)simd_ballot(gt));
                if(lane==0u){masks[2u*(i0>>5)]=eb;masks[2u*(i0>>5)+1u]=gb;}
            }
            const uint eq_total=simd_sum(eq_count),gt_total=simd_sum(gt_count);
            if(lane==0u){hist[sg]=eq_total;hist[32u+sg]=gt_total;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(sg==0u) {
                const uint eq=hist[lane],gt=hist[32u+lane];
                const uint eq_inc=simd_prefix_inclusive_sum(eq),gt_inc=simd_prefix_inclusive_sum(gt);
                const uint tie_before=eq_inc-eq;
                const uint selected_count=gt+min(eq,need>tie_before?need-tie_before:0u);
                hist[64u+lane]=tie_before;hist[96u+lane]=simd_prefix_exclusive_sum(selected_count);
                hist[128u+lane]=selected_count;
                if(lane==31u) hist[160]=(gt_inc==k-need && eq_inc>=need)?1u:0u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(phase==0u && hist[160]==0u) continue;
            const uint count=hist[128u+sg];
            if(count==0u) return;
            const uint tie_before=hist[64u+sg],lowmask=(1u<<lane)-1u;
            uint tbase=0u,obase=hist[96u+sg];const uint finish=obase+count;
            for(uint i0=begin;i0<end;i0+=32u) {
                const uint eb=masks[2u*(i0>>5)],gb=masks[2u*(i0>>5)+1u];
                const uint epre=popcount(eb&lowmask);
                const uint remain=need>tie_before+tbase?need-tie_before-tbase:0u;
                const bool sel=((gb>>lane)&1u) || (((eb>>lane)&1u) && epre<remain);
                const uint pos=obase+popcount(gb&lowmask)+min(epre,remain);
                if(sel) out[pos]=(int)(i0+lane) ^ (a.canary && pos==0u ? 1:0);
                tbase+=popcount(eb);obase+=popcount(gb)+min(popcount(eb),remain);
                if(obase>=finish) break;
            }
            return;
        }
    }
    if (a.mode == 1u) {
        device int *out = selected + (ulong)row*a.sel_stride;
        if (all) { for(uint i=tid;i<k;i+=1024u) out[i]=i<n ? (int)i ^ (a.canary && i==0u ? 1:0) : -1; return; }
        const uint lane=tid&31u,sg=tid>>5;
        for(uint phase=0u;phase<2u;phase++) {
            const uint extent=phase==0u?min(n,1024u):n;
            const uint span=((extent+1023u)/1024u)*32u;
            const uint begin=sg*span,end=min(begin+span,extent);
            uint eq_count=0u,gt_count=0u;
            for(uint i=begin+lane;i<end;i+=32u) {
                const uint key=llms_fkey(s[i]);eq_count+=(key==prefix);gt_count+=(key>prefix);
            }
            const uint eq_total=simd_sum(eq_count),gt_total=simd_sum(gt_count);
            if(lane==0u){hist[sg]=eq_total;hist[32u+sg]=gt_total;}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(sg==0u) {
                const uint eq=hist[lane],gt=hist[32u+lane];
                const uint eq_inc=simd_prefix_inclusive_sum(eq),gt_inc=simd_prefix_inclusive_sum(gt);
                const uint tie_before=eq_inc-eq;
                const uint selected_count=gt+min(eq,need>tie_before?need-tie_before:0u);
                hist[64u+lane]=tie_before;hist[96u+lane]=simd_prefix_exclusive_sum(selected_count);
                hist[128u+lane]=selected_count;
                if(lane==31u) ctl[4]=(gt_inc==k-need && eq_inc>=need)?1u:0u;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(phase==0u && ctl[4]==0u) continue;
            const uint count=hist[128u+sg];
            if(count==0u) return;
            const uint tie_before=hist[64u+sg];
            uint tbase=0u,obase=hist[96u+sg];const uint finish=obase+count;
            for(uint i0=begin;i0<end;i0+=32u) {
                const uint i=i0+lane,key=i<end?llms_fkey(s[i]):0u;
                const uint eq=(i<end && key==prefix)?1u:0u;
                const uint epre=simd_prefix_exclusive_sum(eq);
                const uint sel=(i<end && (key>prefix || (eq && tie_before+tbase+epre<need)))?1u:0u;
                const uint pos=obase+simd_prefix_exclusive_sum(sel);
                if(sel) out[pos]=(int)i ^ (a.canary && pos==0u ? 1:0);
                tbase+=simd_sum(eq);obase+=simd_sum(sel);
                if(obase>=finish) break;
            }
            return;
        }
    }
    if (tid == 0) { ctl[2] = 0u; ctl[3] = 0u; }
    for (uint i = tid; i < 1024u; i += 1024u) { skey[i] = 0u; sidx[i] = 0x7fffffff; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_gt = k - need;
    if (all) { for (uint i = tid; i < n; i += 1024u) { skey[i] = llms_fkey(s[i]); sidx[i] = (int)i; } }   /* llms TOPK_FAST */
    else
    for (uint i = tid; i < n; i += 1024u) {
        const uint key = llms_fkey(s[i]);
        if (key > prefix) { const uint slot = atomic_fetch_add_explicit((threadgroup atomic_uint *)&ctl[2], 1u, memory_order_relaxed); skey[slot] = key; sidx[slot] = (int)i; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!all) {   /* llms TKDET (make_tkdet.py): cutoff ties in INDEX order, lowest first */
        threadgroup uint *sgt = hist;   /* the histogram is free after the passes */
        const uint lane = tid & 31u, sg = tid >> 5;
        uint tbase = 0u;
        for (uint i0 = 0u; i0 < n; i0 += 1024u) {
            const uint i = i0 + tid;
            const uint key = i < n ? llms_fkey(s[i]) : 0u;
            const uint t = (i < n && key == prefix) ? 1u : 0u;
            const uint pre = simd_prefix_exclusive_sum(t), tot = simd_sum(t);
            if (lane == 0u) sgt[sg] = tot;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint off = 0u, chunk = 0u;
            for (uint w = 0u; w < 32u; w++) { const uint v = sgt[w]; if (w < sg) off += v; chunk += v; }
            const uint e = tbase + off + pre;
            if (t && e < need) { skey[n_gt + e] = key; sidx[n_gt + e] = (int)i; }
            tbase += chunk;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tbase >= need) break;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // ties at the threshold were taken in arrival order: restore index order among them by sorting everything (key desc, index asc)
    uint m = 1u; while (m < k) m <<= 1;
    for (uint size = 2u; size <= m; size <<= 1) {
        for (uint stride = size >> 1; stride > 0u; stride >>= 1) {
            for (uint i = tid; i < m; i += 1024u) {
                const uint j = i ^ stride;
                if (j > i) {
                    const bool desc = (i & size) == 0u;   // descending blocks first: final order = key desc
                    const uint ki = skey[i], kj = skey[j]; const int ii = sidx[i], ij = sidx[j];
                    const bool i_first = ki > kj || (ki == kj && ii < ij);   // the order we want: key desc, index asc
                    if (desc ? !i_first : i_first) { skey[i] = kj; skey[j] = ki; sidx[i] = ij; sidx[j] = ii; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    device int *out = selected + (ulong)row * a.sel_stride;
    for (uint i = tid; i < k; i += 1024u) out[i] = sidx[i] == 0x7fffffff ? -1 : sidx[i] ^ (a.canary && i == 0u ? 1 : 0);
}

#include <metal_simdgroup_matrix>


// ---- llms ATTN_GEMM (make_misc_attn_gemm.py, 2026-10-11): the rows attention as gather + Q.K^T + softmax + P.V on simdgroup matrices ----
struct llms_agemm_args { uint n_rows, n_head, top_k, nk, raw_cap, window, ratio, comp_fmt; ulong raw_row_stride, comp_row_stride; float scale; uint canary; };
// grid: x = 64 threads per key row (one dim block of 8 each), y = key slot (nk) + q head slots (n_head), z = row
kernel void kernel_llms_agemm_gather(constant llms_agemm_args &a, constant llms_v41_row *rows, device half *kt, device half *qt, device float *kmask,
                                     uint3 gid [[thread_position_in_grid]]) {
    const uint db = gid.x, j = gid.y, r = gid.z;
    if (db >= 64u || r >= a.n_rows) return;
    const llms_v41_row R = rows[r];
    const uint nkb = a.nk / 8u;
    if (j >= a.nk) {
        const uint h = j - a.nk; if (h >= a.n_head) return;
        device const float4 *src = (device const float4 *)R.q + (ulong)h * 128u + db * 2u;
        const float4 x0 = src[0], x1 = src[1];
        device half4 *dst = (device half4 *)(qt + ((((ulong)r * (a.n_head / 8u) + h / 8u) * 64u + db) * 64u + (h & 7u) * 8u));
        dst[0] = (half4)x0; dst[1] = (half4)x1;
        return;
    }
    const uint qpos = R.pos, first_raw_pos = qpos + 1u - R.n_raw;
    const uint window_first = (a.window != 0u && qpos + 1u > a.window) ? qpos + 1u - a.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first), raw_last = qpos;
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint visible = min((qpos + 1u) / a.ratio, R.n_comp);
    half4 v0 = half4(0.0h), v1 = half4(0.0h); bool ok = false;
    if (j < raw_count) {
        const uint pos = raw_first + j, logical = pos - first_raw_pos, row = (R.raw_start + logical) % a.raw_cap;
        device const float4 *p = (device const float4 *)((device const char *)R.raw + (ulong)row * a.raw_row_stride) + db * 2u;
        v0 = (half4)p[0]; v1 = (half4)p[1]; ok = true;
    } else if (j - raw_count < a.top_k && !(a.canary != 0u && j - raw_count + 8u >= a.top_k)) {
        const int idx = ((device const int *)R.topk)[j - raw_count];
        if (idx >= 0 && (uint)idx < visible) {
            device const char *comp = (device const char *)R.comp;
            if (a.comp_fmt == 2u) { v0 = llms_fp4_load_h4(comp, (uint)idx, db * 2u); v1 = llms_fp4_load_h4(comp, (uint)idx, db * 2u + 1u); }
            else {
                device const char *rp = comp + (ulong)(uint)idx * a.comp_row_stride;
                if (a.comp_fmt == 1u) { device const half4 *p = (device const half4 *)rp + db * 2u; v0 = p[0]; v1 = p[1]; }
                else { device const float4 *p = (device const float4 *)rp + db * 2u; v0 = (half4)p[0]; v1 = (half4)p[1]; }
            }
            ok = true;
        }
    }
    device half4 *dst = (device half4 *)(kt + ((((ulong)r * nkb + j / 8u) * 64u + db) * 64u + (j & 7u) * 8u));
    dst[0] = v0; dst[1] = v1;
    if (db == 0u) kmask[(ulong)r * a.nk + j] = ok ? 0.0f : 1.0f;
}
// grid: threadgroups (nk / 32, n_head / 32, rows) of one simdgroup
kernel void kernel_llms_agemm_qk(constant llms_agemm_args &a, device const half *kt, device const half *qt, device float *S,
                                 uint3 tg [[threadgroup_position_in_grid]]) {
    const uint kb0 = tg.x * 4u, hb0 = tg.y * 4u, r = tg.z;
    const uint nkb = a.nk / 8u, nhb = a.n_head / 8u;
    if (kb0 >= nkb || hb0 >= nhb || r >= a.n_rows) return;
    device const half *q = qt + ((ulong)r * nhb + hb0) * 4096u;
    device const half *k = kt + ((ulong)r * nkb + kb0) * 4096u;
    simdgroup_float8x8 c[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint db = 0; db < 64u; db++) {
        simdgroup_half8x8 A[4], B[4];
        for (int i = 0; i < 4; i++) simdgroup_load(A[i], q + ((ulong)i * 64u + db) * 64u, 8);
        for (int j = 0; j < 4; j++) simdgroup_load(B[j], k + ((ulong)j * 64u + db) * 64u, 8, ulong2(0, 0), true);
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(c[i][j], A[i], B[j], c[i][j]);
    }
    device float *s = S + ((ulong)r * a.n_head + hb0 * 8u) * a.nk + kb0 * 8u;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_store(c[i][j], s + (ulong)(8 * i) * a.nk + 8 * j, a.nk);
}
// one simdgroup per (row, head); nk <= 768
kernel void kernel_llms_agemm_softmax(constant llms_agemm_args &a, device const float *S, device const float *kmask, device const float *sinks,
                                      device half *pt, uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg / a.n_head, h = tg % a.n_head;
    if (r >= a.n_rows) return;
    device const float *s = S + ((ulong)r * a.n_head + h) * a.nk;
    device const float *mk = kmask + (ulong)r * a.nk;
    float x[24];
    float m = -FLT_MAX;
    const uint n = (a.nk + 31u) / 32u;
    for (uint i = 0; i < 24u; i++) { x[i] = -FLT_MAX; if (i < n) { const uint k = i * 32u + lane; if (k < a.nk && mk[k] == 0.0f) { x[i] = s[k] * a.scale; m = max(m, x[i]); } } }
    const float sink = sinks[h];
    m = max(simd_max(m), sink);
    float sum = 0.0f;
    for (uint i = 0; i < 24u; i++) { if (i < n) { x[i] = x[i] == -FLT_MAX ? 0.0f : exp(x[i] - m); sum += x[i]; } }
    sum = simd_sum(sum) + exp(sink - m);
    const float inv = 1.0f / sum;
    const uint nkb = a.nk / 8u;
    device half *p = pt + ((ulong)r * (a.n_head / 8u) + h / 8u) * nkb * 64u + (h & 7u) * 8u;
    for (uint i = 0; i < 24u; i++) { if (i < n) { const uint k = i * 32u + lane; if (k < a.nk) p[(k / 8u) * 64u + (k & 7u)] = (half)(x[i] * inv); } }
}
// grid: threadgroups (512 / 32, n_head / 32, rows) of one simdgroup
kernel void kernel_llms_agemm_pv(constant llms_agemm_args &a, constant llms_v41_row *rows, device const half *kt, device const half *pt,
                                 uint3 tg [[threadgroup_position_in_grid]]) {
    const uint db0 = tg.x * 4u, hb0 = tg.y * 4u, r = tg.z;
    const uint nkb = a.nk / 8u, nhb = a.n_head / 8u;
    if (db0 >= 64u || hb0 >= nhb || r >= a.n_rows) return;
    device const half *p = pt + ((ulong)r * nhb + hb0) * nkb * 64u;
    device const half *v = kt + (ulong)r * nkb * 4096u + (ulong)db0 * 64u;
    simdgroup_float8x8 c[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kb = 0; kb < nkb; kb++) {
        simdgroup_half8x8 A[4], B[4];
        for (int i = 0; i < 4; i++) simdgroup_load(A[i], p + ((ulong)i * nkb + kb) * 64u, 8);
        for (int j = 0; j < 4; j++) simdgroup_load(B[j], v + ((ulong)kb * 64u + j) * 64u, 8);
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(c[i][j], A[i], B[j], c[i][j]);
    }
    device float *o = (device float *)rows[r].dst + (ulong)hb0 * 8u * 512u + db0 * 8u;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_store(c[i][j], o + (ulong)(8 * i) * 512u + 8 * j, 512);
}


// ---- llms ATTN_GEMM v2 (make_misc_attn_gemm2.py, 2026-10-11): float P, P(float) . V(half) -> float; strong canary ----
kernel void kernel_llms_agemm2_gather(constant llms_agemm_args &a, constant llms_v41_row *rows, device half *kt, device half *qt, device float *kmask,
                                     uint3 gid [[thread_position_in_grid]]) {
    const uint db = gid.x, j = gid.y, r = gid.z;
    if (db >= 64u || r >= a.n_rows) return;
    const llms_v41_row R = rows[r];
    const uint nkb = a.nk / 8u;
    if (j >= a.nk) {
        const uint h = j - a.nk; if (h >= a.n_head) return;
        device const float4 *src = (device const float4 *)R.q + (ulong)h * 128u + db * 2u;
        const float4 x0 = src[0], x1 = src[1];
        device half4 *dst = (device half4 *)(qt + ((((ulong)r * (a.n_head / 8u) + h / 8u) * 64u + db) * 64u + (h & 7u) * 8u));
        dst[0] = (half4)x0; dst[1] = (half4)x1;
        return;
    }
    const uint qpos = R.pos, first_raw_pos = qpos + 1u - R.n_raw;
    const uint window_first = (a.window != 0u && qpos + 1u > a.window) ? qpos + 1u - a.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first), raw_last = qpos;
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint visible = min((qpos + 1u) / a.ratio, R.n_comp);
    half4 v0 = half4(0.0h), v1 = half4(0.0h); bool ok = false;
    if (j < raw_count && !(a.canary == 2u && j + 1u == raw_count)) {   // canary 2: the newest raw key masked
        const uint pos = raw_first + j, logical = pos - first_raw_pos, row = (R.raw_start + logical) % a.raw_cap;
        device const float4 *p = (device const float4 *)((device const char *)R.raw + (ulong)row * a.raw_row_stride) + db * 2u;
        v0 = (half4)p[0]; v1 = (half4)p[1]; ok = true;
    } else if (j >= raw_count && j - raw_count < a.top_k && !(a.canary == 1u && j - raw_count + 8u >= a.top_k)) {
        const int idx = ((device const int *)R.topk)[j - raw_count];
        if (idx >= 0 && (uint)idx < visible) {
            device const char *comp = (device const char *)R.comp;
            if (a.comp_fmt == 2u) { v0 = llms_fp4_load_h4(comp, (uint)idx, db * 2u); v1 = llms_fp4_load_h4(comp, (uint)idx, db * 2u + 1u); }
            else {
                device const char *rp = comp + (ulong)(uint)idx * a.comp_row_stride;
                if (a.comp_fmt == 1u) { device const half4 *p = (device const half4 *)rp + db * 2u; v0 = p[0]; v1 = p[1]; }
                else { device const float4 *p = (device const float4 *)rp + db * 2u; v0 = (half4)p[0]; v1 = (half4)p[1]; }
            }
            ok = true;
        }
    }
    device half4 *dst = (device half4 *)(kt + ((((ulong)r * nkb + j / 8u) * 64u + db) * 64u + (j & 7u) * 8u));
    dst[0] = v0; dst[1] = v1;
    if (db == 0u) kmask[(ulong)r * a.nk + j] = ok ? 0.0f : 1.0f;
}

kernel void kernel_llms_agemm2_softmax(constant llms_agemm_args &a, device const float *S, device const float *kmask, device const float *sinks,
                                       device float *pt, uint tg [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint r = tg / a.n_head, h = tg % a.n_head;
    if (r >= a.n_rows) return;
    device const float *s = S + ((ulong)r * a.n_head + h) * a.nk;
    device const float *mk = kmask + (ulong)r * a.nk;
    float x[24];
    float m = -FLT_MAX;
    const uint n = (a.nk + 31u) / 32u;
    for (uint i = 0; i < 24u; i++) { x[i] = -FLT_MAX; if (i < n) { const uint k = i * 32u + lane; if (k < a.nk && mk[k] == 0.0f) { x[i] = s[k] * a.scale; m = max(m, x[i]); } } }
    const float sink = sinks[h];
    m = max(simd_max(m), sink);
    float sum = 0.0f;
    for (uint i = 0; i < 24u; i++) { if (i < n) { x[i] = x[i] == -FLT_MAX ? 0.0f : exp(x[i] - m); sum += x[i]; } }
    sum = simd_sum(sum) + exp(sink - m);
    const float inv = 1.0f / sum;
    const uint nkb = a.nk / 8u;
    device float *p = pt + ((ulong)r * (a.n_head / 8u) + h / 8u) * nkb * 64u + (h & 7u) * 8u;
    for (uint i = 0; i < 24u; i++) { if (i < n) { const uint k = i * 32u + lane; if (k < a.nk) p[(k / 8u) * 64u + (k & 7u)] = x[i] * inv; } }
}
kernel void kernel_llms_agemm2_pv(constant llms_agemm_args &a, constant llms_v41_row *rows, device const half *kt, device const float *pt,
                                  uint3 tg [[threadgroup_position_in_grid]]) {
    const uint db0 = tg.x * 4u, hb0 = tg.y * 4u, r = tg.z;
    const uint nkb = a.nk / 8u, nhb = a.n_head / 8u;
    if (db0 >= 64u || hb0 >= nhb || r >= a.n_rows) return;
    device const float *p = pt + ((ulong)r * nhb + hb0) * nkb * 64u;
    device const half *v = kt + (ulong)r * nkb * 4096u + (ulong)db0 * 64u;
    simdgroup_float8x8 c[4][4];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint kb = 0; kb < nkb; kb++) {
        simdgroup_float8x8 A[4]; simdgroup_half8x8 B[4];
        for (int i = 0; i < 4; i++) simdgroup_load(A[i], p + ((ulong)i * nkb + kb) * 64u, 8);
        for (int j = 0; j < 4; j++) simdgroup_load(B[j], v + ((ulong)kb * 64u + j) * 64u, 8);
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(c[i][j], A[i], B[j], c[i][j]);
    }
    device float *o = (device float *)rows[r].dst + (ulong)hb0 * 8u * 512u + db0 * 8u;
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) simdgroup_store(c[i][j], o + (ulong)(8 * i) * 512u + 8 * j, 512);
}


// ---- llms MQA_ATTN (apply_mqa_attn.py): the V4.1 one-token decode attention with all heads sharing each KV row ----
// V4.1's attention reads ONE 512-wide KV row per position for all 64 heads (K = V; flash_attn_ext_vec binds the same buffer twice).
// The stock vec kernel runs a threadgroup per (head, KV split): every head re-reads every row (~41 MB of cache traffic per layer for
// 0.7 MB of rows), after staging copies, a pad and a mask fill. Here: llms_mqa_partial -- a threadgroup per (chunk of 16 rows, group
// of 16 heads); the rows are read ONCE per threadgroup straight from the raw window ring (f32) and the selected compressed rows (f32
// or f16), rounded to half as the staging did, kept in threadgroup memory and used by all 16 heads (2 per simdgroup); per (chunk, head)
// the partial max m, sum l and the unnormalized 512-wide output -> part. llms_mqa_reduce -- a threadgroup per head: the chunks and the
// head's sink logit (it adds exp(sink - M) to the denominator only) -> heads (f32), the input of the BF16 pass + inverse rope.
// q rounds to half before the dot (the f16 vec kernel's Q type); scores, softmax and the output accumulate in float.
struct llms_mqa_args { uint n_raw, raw_cap, raw_start, n_comp, n_head, comp_f16, n_chunk, pad0; float scale; };
#define LLMS_MQA_R 16u
#define LLMS_MQA_PS 516u   /* floats per (chunk, head) partial: m, l, 2 pad, 512 outputs (16-byte aligned) */
kernel void llms_mqa_partial(constant llms_mqa_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const float * raw [[buffer(2)]],
        device const char * comp [[buffer(3)]], device float * part [[buffer(4)]], threadgroup half4 * kvs [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint N = a.n_raw + a.n_comp, j0 = tg.x * LLMS_MQA_R, j1 = min(N, j0 + LLMS_MQA_R), nr = j1 > j0 ? j1 - j0 : 0u;
    for (uint idx = tid; idx < nr * 128u; idx += 256u) {
        const uint r = idx / 128u, e4 = idx % 128u, j = j0 + r;
        float4 v;
        if (j < a.n_raw) v = ((device const float4 *)(raw + (uint64_t)((a.raw_start + j) % a.raw_cap) * 512u))[e4];
        else if (a.comp_f16) v = float4(((device const half4 *)(comp + (uint64_t)(j - a.n_raw) * 1024u))[e4]);
        else v = ((device const float4 *)(comp + (uint64_t)(j - a.n_raw) * 2048u))[e4];
        kvs[r * 128u + e4] = half4(v);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint hh = 0; hh < 2u; hh++) {
        const uint h = tg.y * 16u + sg * 2u + hh;
        if (h >= a.n_head) break;
        float4 qv[4];
        for (uint k = 0; k < 4u; k++) qv[k] = float4(half4(((device const float4 *)(q + (uint64_t)h * 512u))[lane + 32u * k]));
        float s[LLMS_MQA_R]; float m = -INFINITY;
        for (uint r = 0; r < nr; r++) {
            float d = 0.f;
            for (uint k = 0; k < 4u; k++) d += dot(qv[k], float4(kvs[r * 128u + lane + 32u * k]));
            d = simd_sum(d) * a.scale; s[r] = d; m = max(m, d);
        }
        float l = 0.f; float4 o[4] = { float4(0.f), float4(0.f), float4(0.f), float4(0.f) };
        for (uint r = 0; r < nr; r++) {
            const float p = exp(s[r] - m); l += p;
            for (uint k = 0; k < 4u; k++) o[k] += p * float4(kvs[r * 128u + lane + 32u * k]);
        }
        device float * pp = part + ((uint64_t)tg.x * a.n_head + h) * LLMS_MQA_PS;
        if (lane == 0) { pp[0] = nr ? m : -INFINITY; pp[1] = l; }
        for (uint k = 0; k < 4u; k++) ((device float4 *)(pp + 4))[lane + 32u * k] = o[k];
    }
}
kernel void llms_mqa_reduce(constant llms_mqa_args & a [[buffer(0)]], device const float * part [[buffer(1)]], device const float * sinks [[buffer(2)]],
        device float * heads [[buffer(3)]], uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    /* 128 threads per head: thread t owns the float4 of dims 4t..4t+3 */
    const float sk = sinks[h];
    float M = sk;
    for (uint c = 0; c < a.n_chunk; c++) M = max(M, part[((uint64_t)c * a.n_head + h) * LLMS_MQA_PS]);
    float L = exp(sk - M); float4 O = float4(0.f);
    for (uint c = 0; c < a.n_chunk; c++) {
        device const float * pp = part + ((uint64_t)c * a.n_head + h) * LLMS_MQA_PS;
        const float w = exp(pp[0] - M);
        L += pp[1] * w;
        O += w * ((device const float4 *)(pp + 4))[tid];
    }
    ((device float4 *)(heads + (uint64_t)h * 512u))[tid] = O / L;
}


// ---- llms MQA4 : llms_mqa_partial with fixed-size (unrolled, masked) 16-row loops and the compressed rows read as
// converted half rows by index (no gather pass); the same (m, l, o) partials for llms_mqa_reduce ----
struct llms_mqa4_args { uint n_raw, raw_cap, raw_start, n_comp, n_head, has_sel, n_chunk, pad0; float scale; };
kernel void llms_mqa4_partial(constant llms_mqa4_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const float * raw [[buffer(2)]],
        device const half * comp [[buffer(3)]], device const int * sel [[buffer(4)]], device float * part [[buffer(5)]], threadgroup half4 * kvs [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint N = a.n_raw + a.n_comp, j0 = tg.x * 16u, j1 = min(N, j0 + 16u), nr = j1 > j0 ? j1 - j0 : 0u;
    for (uint idx = tid; idx < 16u * 128u; idx += 256u) {
        const uint r = idx >> 7, e4 = idx & 127u, j = j0 + min(r, nr - 1u);
        half4 v;
        if (j < a.n_raw) v = half4(((device const float4 *)(raw + (uint64_t)((a.raw_start + j) % a.raw_cap) * 512u))[e4]);
        else { const uint cj = a.has_sel ? (uint)sel[j - a.n_raw] : j - a.n_raw; v = ((device const half4 *)(comp + (uint64_t)cj * 512u))[e4]; }
        kvs[idx] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint hh = 0; hh < 2u; hh++) {
        const uint h = tg.y * 16u + sg * 2u + hh;
        float4 qv[4];
        #pragma unroll
        for (uint k = 0; k < 4u; k++) qv[k] = float4(half4(((device const float4 *)(q + (uint64_t)h * 512u))[lane + 32u * k]));
        float s[16];
        #pragma unroll
        for (uint r = 0; r < 16u; r++) {
            float d = 0.f;
            #pragma unroll
            for (uint k = 0; k < 4u; k++) d += dot(qv[k], float4(kvs[r * 128u + lane + 32u * k]));
            s[r] = d;
        }
        float m = -INFINITY;
        #pragma unroll
        for (uint r = 0; r < 16u; r++) { s[r] = simd_sum(s[r]) * a.scale; if (r < nr) m = max(m, s[r]); }
        float l = 0.f; float4 o[4] = { float4(0.f), float4(0.f), float4(0.f), float4(0.f) };
        #pragma unroll
        for (uint r = 0; r < 16u; r++) {
            const float p = r < nr ? exp(s[r] - m) : 0.f; l += p;
            #pragma unroll
            for (uint k = 0; k < 4u; k++) o[k] += p * float4(kvs[r * 128u + lane + 32u * k]);
        }
        device float * pp = part + ((uint64_t)tg.x * a.n_head + h) * 516u;
        if (lane == 0) { pp[0] = nr ? m : -INFINITY; pp[1] = l; }
        #pragma unroll
        for (uint k = 0; k < 4u; k++) ((device float4 *)(pp + 4))[lane + 32u * k] = o[k];
    }
}

kernel void llms_mqa4_reduce(constant llms_mqa4_args & a [[buffer(0)]], device const float * part [[buffer(1)]], device const float * sinks [[buffer(2)]],
        device float * heads [[buffer(3)]], uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const float sk = sinks[h];
    const uint nc = a.n_chunk;
    float M = sk;
    for (uint c = 0; c < nc; c++) M = max(M, part[((uint64_t)c * a.n_head + h) * 516u]);
    float L = exp(sk - M); float4 O = float4(0.f);
    uint c = 0;
    for (; c + 4u <= nc; c += 4u) {
        device const float * p0 = part + ((uint64_t)(c + 0u) * a.n_head + h) * 516u; device const float * p1 = part + ((uint64_t)(c + 1u) * a.n_head + h) * 516u;
        device const float * p2 = part + ((uint64_t)(c + 2u) * a.n_head + h) * 516u; device const float * p3 = part + ((uint64_t)(c + 3u) * a.n_head + h) * 516u;
        const float4 o0 = ((device const float4 *)(p0 + 4))[tid], o1 = ((device const float4 *)(p1 + 4))[tid], o2 = ((device const float4 *)(p2 + 4))[tid], o3 = ((device const float4 *)(p3 + 4))[tid];
        const float w0 = exp(p0[0] - M), w1 = exp(p1[0] - M), w2 = exp(p2[0] - M), w3 = exp(p3[0] - M);
        L += p0[1] * w0; O += w0 * o0; L += p1[1] * w1; O += w1 * o1; L += p2[1] * w2; O += w2 * o2; L += p3[1] * w3; O += w3 * o3;
    }
    for (; c < nc; c++) {
        device const float * pp = part + ((uint64_t)c * a.n_head + h) * 516u;
        const float w = exp(pp[0] - M);
        L += pp[1] * w;
        O += w * ((device const float4 *)(pp + 4))[tid];
    }
    ((device float4 *)(heads + (uint64_t)h * 512u))[tid] = O / L;
}

kernel void kernel_astra_mqa_reduce_group(constant llms_mqa4_args & a [[buffer(0)]], device const float * part [[buffer(1)]], device const float * sinks [[buffer(2)]],
        device float * heads [[buffer(3)]], uint h [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float groupmax[4], maximum, weights[128], denominator;
    const uint nc=a.n_chunk, lane=tid&31u, sg=tid>>5;
    const float sk=sinks[h];
    float own=tid<nc?part[((ulong)tid*a.n_head+h)*516u]:-INFINITY;
    const float gm=simd_max(own);
    if(lane==0u)groupmax[sg]=gm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0u){float m=sk;for(uint j=0;j<4;j++)m=max(m,groupmax[j]);maximum=m;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid<nc)weights[tid]=exp(own-maximum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0u){
        float l=exp(sk-maximum);uint c=0;
        for(;c+4u<=nc;c+=4u){
            l+=part[((ulong)(c+0u)*a.n_head+h)*516u+1u]*weights[c+0u];
            l+=part[((ulong)(c+1u)*a.n_head+h)*516u+1u]*weights[c+1u];
            l+=part[((ulong)(c+2u)*a.n_head+h)*516u+1u]*weights[c+2u];
            l+=part[((ulong)(c+3u)*a.n_head+h)*516u+1u]*weights[c+3u];
        }
        for(;c<nc;c++)l+=part[((ulong)c*a.n_head+h)*516u+1u]*weights[c];
        denominator=l;
    }
    float4 O=float4(0.f);uint c=0;
    for(;c+4u<=nc;c+=4u){
        device const float *p0=part+((ulong)(c+0u)*a.n_head+h)*516u;
        device const float *p1=part+((ulong)(c+1u)*a.n_head+h)*516u;
        device const float *p2=part+((ulong)(c+2u)*a.n_head+h)*516u;
        device const float *p3=part+((ulong)(c+3u)*a.n_head+h)*516u;
        const float4 o0=((device const float4*)(p0+4))[tid],o1=((device const float4*)(p1+4))[tid];
        const float4 o2=((device const float4*)(p2+4))[tid],o3=((device const float4*)(p3+4))[tid];
        O+=weights[c+0u]*o0;O+=weights[c+1u]*o1;O+=weights[c+2u]*o2;O+=weights[c+3u]*o3;
    }
    for(;c<nc;c++){device const float *pp=part+((ulong)c*a.n_head+h)*516u;O+=weights[c]*((device const float4*)(pp+4))[tid];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ((device float4*)(heads+(ulong)h*512u))[tid]=O/denominator;
}


// mqa7 (2026-10-15 09:0x): llms_mqa4_partial with QK and PV on 8x8 simdgroup matrices; everything else as mqa4 (16-key chunks,
// 16 heads per threadgroup, 256 threads, the same (m, l, o[512]) partial layout and the same llms_mqa4_reduce). Q is rounded to half
// exactly as mqa4 does (float4(half4(q))), K/V staged as half; products are exact in fp32, the summation ORDER differs (MMA tiles) --
// not bit-identical; judged against the double reference beside mqa4's own error (mqa_bench.m). Threadgroup memory: K 16 KB + Q 16 KB.
kernel void llms_mqa7_partial(constant llms_mqa4_args & a [[buffer(0)]], device const float * q [[buffer(1)]], device const float * raw [[buffer(2)]],
        device const half * comp [[buffer(3)]], device const int * sel [[buffer(4)]], device float * part [[buffer(5)]], threadgroup half4 * kvs [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint N = a.n_raw + a.n_comp, j0 = tg.x * 16u, j1 = min(N, j0 + 16u), nr = j1 > j0 ? j1 - j0 : 0u;
    threadgroup half4 * qs = kvs + 16u * 128u;                 // 16 heads x 512 dims as half
    for (uint idx = tid; idx < 16u * 128u; idx += 256u) {
        const uint r = idx >> 7, e4 = idx & 127u, j = j0 + min(r, nr - 1u);
        half4 v;
        if (j < a.n_raw) v = half4(((device const float4 *)(raw + (uint64_t)((a.raw_start + j) % a.raw_cap) * 512u))[e4]);
        else { const uint cj = a.has_sel ? (uint)sel[j - a.n_raw] : j - a.n_raw; v = ((device const half4 *)(comp + (uint64_t)cj * 512u))[e4]; }
        kvs[idx] = v;
        const uint h = tg.y * 16u + r;
        qs[idx] = half4(((device const float4 *)(q + (uint64_t)h * 512u))[e4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // S[16 heads x 16 keys] = Qh[16 x 512] * K^T[512 x 16]: sg -> tile (sg & 3) = (head block, key block), k half (sg >> 2)
    const uint tb = sg & 3u, hb = tb >> 1, kb = tb & 1u, kh = sg >> 2;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8>(0.f);
    threadgroup const half * qh = (threadgroup const half *)qs;
    threadgroup const half * kh8 = (threadgroup const half *)kvs;
    for (uint s = 0; s < 32u; s++) {
        const uint d0 = kh * 256u + s * 8u;
        simdgroup_half8x8 A, B;
        simdgroup_load(A, qh + (hb * 8u) * 512u + d0, 512);                 // rows = heads, cols = dims
        simdgroup_load(B, kh8 + (kb * 8u) * 512u + d0, 512, ulong2(0, 0), true);   // K^T: rows = dims, cols = keys
        simdgroup_multiply_accumulate(acc, A, B, acc);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);              // Q no longer needed: its 16 KB hold the S halves (2 x 16 x 16 floats)
    threadgroup float * sh = (threadgroup float *)qs;
    simdgroup_store(acc, sh + kh * 256u + (hb * 8u) * 16u + kb * 8u, 16);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // softmax per head over the chunk's 16 keys: thread tid -> head tid / 16, key tid % 16 (16-lane groups inside a simdgroup)
    threadgroup float * ph = sh + 512u;                           // P[16 heads x 16 keys]
    {
        const uint hh = tid >> 4, kk = tid & 15u;
        float s = (sh[hh * 16u + kk] + sh[256u + hh * 16u + kk]) * a.scale;
        const bool ok = kk < nr;
        float m = ok ? s : -INFINITY;
        m = max(m, simd_shuffle_xor(m, (ushort)1)); m = max(m, simd_shuffle_xor(m, (ushort)2)); m = max(m, simd_shuffle_xor(m, (ushort)4)); m = max(m, simd_shuffle_xor(m, (ushort)8));
        const float p = ok ? exp(s - m) : 0.f;
        float l = p;
        l += simd_shuffle_xor(l, (ushort)1); l += simd_shuffle_xor(l, (ushort)2); l += simd_shuffle_xor(l, (ushort)4); l += simd_shuffle_xor(l, (ushort)8);
        ph[hh * 16u + kk] = p;
        if (kk == 0u) {
            device float * pp = part + ((uint64_t)tg.x * a.n_head + tg.y * 16u + hh) * 516u;
            pp[0] = nr ? m : -INFINITY; pp[1] = l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // O[16 heads x 512] = P[16 x 16] * V[16 keys x 512]: 2 x 64 tiles, sg -> head block (sg & 1), 32 dim blocks (sg >> 1) * 16 .. + 16
    {
        const uint hb2 = sg & 1u, db0 = (sg >> 1) * 16u;
        simdgroup_float8x8 P0, P1;
        simdgroup_load(P0, ph + (hb2 * 8u) * 16u + 0u, 16);
        simdgroup_load(P1, ph + (hb2 * 8u) * 16u + 8u, 16);
        for (uint db = 0; db < 16u; db++) {
            const uint d = (db0 + db) * 8u;
            simdgroup_half8x8 V0, V1;
            simdgroup_load(V0, kh8 + 0u * 512u + d, 512);             // rows = keys 0..7, cols = dims
            simdgroup_load(V1, kh8 + 8u * 512u + d, 512);             // keys 8..15
            simdgroup_float8x8 O = make_filled_simdgroup_matrix<float, 8>(0.f);
            simdgroup_multiply_accumulate(O, P0, V0, O);
            simdgroup_multiply_accumulate(O, P1, V1, O);
            device float * dst = part + ((uint64_t)tg.x * a.n_head + tg.y * 16u + hb2 * 8u) * 516u + 4u + d;
            simdgroup_store(O, dst, 516);
        }
    }
}


// ---- llms IDX_COMPACT:
struct astra_candidate_args { uint visible; float scale; };
kernel void astra_candidate_tail(constant astra_candidate_args &a [[buffer(0)]],
    device const half *keys [[buffer(1)]], device half *tail [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= 1024u || a.visible == 0u) return;
    const uint row = ((a.visible - 1u) / 8u) * 8u + i / 128u;
    tail[i] = row < a.visible ? keys[(ulong)row * 128u + i % 128u] : half(0);
}
kernel void astra_candidate_score_t4(constant astra_candidate_args &a [[buffer(0)]],
    device const float *q [[buffer(1)]], device const float *weights [[buffer(2)]],
    device const half *keys [[buffer(3)]], device float *scores [[buffer(4)]],
    device const uint *blocks [[buffer(5)]], device const half *tail [[buffer(6)]],
    threadgroup float *shmem [[threadgroup(0)]], uint tg [[threadgroup_position_in_grid]],
    uint ntg [[threadgroups_per_grid]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (a.visible <= 16384u) return; // dense identity path belongs to caller
    threadgroup half *qs = (threadgroup half *)(shmem + 32u * 32u);
    for (uint i = (uint)sg * 32u + lane; i < 32u * 128u; i += 128u) qs[i] = half(q[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const half *Q = qs + (uint)sg * 8u * 128u;
    for (uint blk = tg; blk < 512u; blk += ntg) {
        simdgroup_float8x8 C[4];
#pragma clang loop unroll(full)
        for (short t = 0; t < 4; t++) C[t] = make_filled_simdgroup_matrix<float,8>(0.f);
        for (short kt = 0; kt < 16; kt++) {
            simdgroup_half8x8 B; simdgroup_load(B,Q + kt * 8,128,ulong2(0,0),true);
#pragma clang loop unroll(full)
            for (short t = 0; t < 4; t++) {
                const ulong base = (ulong)blocks[blk * 4u + (uint)t] * 8u;
                device const half *K = base + 8u <= a.visible ? keys + base * 128u : tail;
                simdgroup_half8x8 A; simdgroup_load(A,K + kt * 8,128);
                simdgroup_multiply_accumulate(C[t],A,B,C[t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (short t = 0; t < 4; t++) simdgroup_store(C[t],shmem + (t * 8) * 32 + sg * 8,32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
            const uint slot = blk * 32u + lane;
            const ulong original = (ulong)blocks[slot / 8u] * 8u + slot % 8u;
            float acc = 0.0f;
            for (uint h = 0; h < 32u; h++) acc = acc + max(shmem[lane * 32u + h] * a.scale,0.0f) * weights[h];
            scores[slot] = original < a.visible ? acc : -INFINITY;
        }
    }
}
kernel void astra_candidate_remap(device const uint *blocks [[buffer(0)]],
    device const int *compact [[buffer(1)]], device int *original [[buffer(2)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= 512u) return;
    const int c = compact[i];
    original[i] = c >= 0 && c < 16384 ? int(blocks[uint(c)/8u]*8u+uint(c)%8u) : -1;
}

kernel void astra_mask_list(device const float *mask [[buffer(0)]],
    device uint *list [[buffer(1)]], device uint *count [[buffer(2)]],
    constant uint &nb [[buffer(3)]], ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint starts[32];
    const uint span=((nb+1023u)/1024u)*32u;
    const uint lo=uint(sg)*span, hi=min(lo+span,nb);
    uint local=0u;
    for(uint i=lo+lane;i<hi;i+=32u) local+=uint(mask[i]==0.0f);
    const uint total=simd_sum(local);
    if(lane==0u) starts[sg]=total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0u) {
        const uint v=starts[lane];
        const uint prefix=simd_prefix_exclusive_sum(v);
        starts[lane]=prefix;
        if(lane==31u) count[0]=prefix+v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint offset=starts[sg];
    for(uint i0=lo;i0<hi;i0+=32u) {
        const uint i=i0+lane;
        const uint active=(i<hi && mask[i]==0.0f)?1u:0u;
        const uint rank=offset+simd_prefix_exclusive_sum(active);
        // Concrete guard: an invalid mask with >2048 entries may not corrupt memory.
        // count exposes that invalid input; the fixture checker must reject it.
        if(active && rank<2048u) list[rank]=i;
        offset+=simd_sum(active);
    }
}

// ---- llms IDX_COMPACT canary and check kernels (idxc_edit.py) ----
kernel void llms_idxc_remap_canary(device const uint *blocks [[buffer(0)]], device const int *compact [[buffer(1)]], device int *original [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= 512u) return; const int c = compact[i];
    original[i] = c >= 0 && c < 16384 ? int(blocks[uint(c) / 8u] * 8u) : -1;   // the position within the block dropped: wrong IDs by construction
}
kernel void llms_idxc_compare(device const int *full [[buffer(0)]], device const int *comp [[buffer(1)]], device atomic_uint *ctr [[buffer(2)]], constant uint &rows [[buffer(3)]],
                              uint row [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= rows || tid >= 512) return;
    const int c = comp[row * 512u + tid]; bool found = false;
    for (uint j = 0; j < 512u; j++) found = found || (full[row * 512u + j] == c);
    if (!found) atomic_fetch_add_explicit(&ctr[1], 1u, memory_order_relaxed);
    if (tid == 0) atomic_fetch_add_explicit(&ctr[0], 1u, memory_order_relaxed);
}

// ---- llms ATT2 (apply_att2.py; make_att2b.py r16qi) ----
kernel void kernel_llms_att2_r16qi(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q, device const char *raw_kv, device const char *comp_kv, device const char *topk, device char *tmp,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x, head0 = tgpig.y * 8u, split = tgpig.z, n_splits = args.n_splits;
    if (token >= args.n_tokens || head0 >= args.n_head || n_splits < 2u || n_splits > 31u || split >= n_splits) return;
    threadgroup half  * kh = (threadgroup half *) kv_shared;             // K chunk 16 x 512 half; before the loop: Q 8 x 512 half
    threadgroup float * sp = (threadgroup float *) (kh + 16u * 512u);   // partial scores: 8 sg x 8 heads x 16 keys
    threadgroup float * pp = sp + 1024u;                          // P: 8 heads x 16 keys
    threadgroup float * dg = pp + 128u;                              // diag(alpha): 8 x 8
    threadgroup int * six = (threadgroup int *) (dg + 64u);              // IDX: the split's row codes (<= 320)
    for (uint i = tid; i < 8u * 128u; i += 256u) {
        const uint h = i >> 7, c = i & 127u;
        device const float4 *q4 = (device const float4 *)(q + (uint64_t)token * args.q_token_stride + (uint64_t)(head0 + h) * args.q_head_stride);
        kv_shared[i] = (half4) q4[c];
    }
    if (tid < 64u) dg[tid] = 0.0f;
    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk + (uint64_t)token * args.topk_token_stride);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d_sg = (uint)sg * 64u;
    simdgroup_half8x8 A[8];
    for (uint k = 0; k < 8u; k++) simdgroup_load(A[k], kh + d_sg + k * 8u, 512);
    simdgroup_float8x8 O[8];
    for (uint n = 0; n < 8u; n++) O[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
    float m_run = -FLT_MAX / 2.0f, l_run = 0.0f;
    const uint hh = (uint)tid >> 4, kk = (uint)tid & 15u;
    for (uint i = tid; i < split_last - split_first; i += 256u) {
        const uint seq = split_first + i; int code;
        if (seq < raw_count) { const uint pos = raw_first + seq, logical = pos - first_raw_pos; code = -2 - (int)((args.raw_start + logical) % args.raw_cap); }
        else { const int32_t idx = row_topk[seq - raw_count]; code = (idx >= 0 && (uint)idx < visible) ? idx : -1; }
        six[i] = code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                       // every simdgroup holds its Q slice: the area becomes K
    for (uint seq0 = split_first; seq0 < split_last; seq0 += 16u) {
        for (uint off = tid; off < 16u * 128u; off += 256u) {
            const uint r = off >> 7, c = off & 127u, seq = seq0 + r;
            half4 value = half4(0.0h);
            const int code = seq < split_last ? six[seq - split_first] : -1;
            if (code >= 0) value = dsv4_load_cache_h4(comp_kv, args.comp_row_stride, (uint)code, c, args.comp_kv_f16 != 0u);
            else if (code <= -2) value = (half4) ((device const float4 *)(raw_kv + (uint64_t)(-2 - code) * args.raw_row_stride))[c];
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8>(0.f), s1 = make_filled_simdgroup_matrix<float, 8>(0.f);
            for (uint k = 0; k < 8u; k++) {
                simdgroup_half8x8 B0; simdgroup_load(B0, kh + 0u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s0, A[k], B0, s0);
                simdgroup_half8x8 B1; simdgroup_load(B1, kh + 8u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s1, A[k], B1, s1);
            }
            simdgroup_store(s0, sp + (uint)sg * 128u + 0u, 16);
            simdgroup_store(s1, sp + (uint)sg * 128u + 8u, 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128u) {
            float s = 0.0f;
            for (uint j = 0; j < 8u; j++) s += sp[j * 128u + hh * 16u + kk];
            s *= args.scale;
            const uint seq = seq0 + kk;
            const bool ok = seq < split_last && six[seq - split_first] != -1;
            float mc = ok ? s : -INFINITY;
            mc = max(mc, simd_shuffle_xor(mc, (ushort)1)); mc = max(mc, simd_shuffle_xor(mc, (ushort)2)); mc = max(mc, simd_shuffle_xor(mc, (ushort)4)); mc = max(mc, simd_shuffle_xor(mc, (ushort)8));
            const float m_new = max(m_run, mc);
            const float alpha = exp(m_run - m_new);
            const float p = ok ? exp(s - m_new) : 0.0f;
            float lc = p;
            lc += simd_shuffle_xor(lc, (ushort)1); lc += simd_shuffle_xor(lc, (ushort)2); lc += simd_shuffle_xor(lc, (ushort)4); lc += simd_shuffle_xor(lc, (ushort)8);
            l_run = l_run * alpha + lc; m_run = m_new;
            pp[hh * 16u + kk] = p;
#if LLMS_ATT2_RS
            if (kk == 0u) dg[hh] = LLMS_ATT2_RS == 2 ? alpha * alpha : alpha;   /* llms ATT2_RS : the row scale as a vector; 2 = canary */
#else
            if (kk == 0u) dg[hh * 8u + hh] = alpha;
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
#if LLMS_ATT2_RS
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 0, ulong2(0, 0), true);   /* llms ATT2_RS: Dm(i, j) = dg[i] */
#else
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 8);
#endif
            simdgroup_float8x8 P0; simdgroup_load(P0, pp + 0u, 16);
            simdgroup_float8x8 P1; simdgroup_load(P1, pp + 8u, 16);
            for (uint n = 0; n < 8u; n++) {
#if LLMS_ATT2_RS
                simdgroup_float8x8 t = O[n]; t.thread_elements() *= Dm.thread_elements();   /* llms ATT2_RS: row i scaled by alpha_i, no MMA */
#else
                simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O[n]);
#endif
                simdgroup_half8x8 V0; simdgroup_load(V0, kh + 0u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(t, P0, V0, t);
                simdgroup_half8x8 V1; simdgroup_load(V1, kh + 8u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(O[n], P1, V1, t);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float4 *partials = (device float4 *)tmp;
    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    threadgroup float * of = (threadgroup float *) kv_shared;              // O staged in 8 KB pieces of 8 heads x 512 dims
    const uint PD = 512u;                                   // dims per piece: the K area holds 8 x PD floats
    for (uint d0 = 0; d0 < 512u; d0 += PD) {
        for (uint n = 0; n < 8u; n++) { const uint d = d_sg + n * 8u; if (d >= d0 && d < d0 + PD) simdgroup_store(O[n], of + (d - d0), PD); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8u * (PD / 4u); i += 256u) {
            const uint h = i / (PD / 4u), v = i % (PD / 4u);
            const uint64_t row = (uint64_t)token * args.n_head + head0 + h;
            partials[(row * 128u + d0 / 4u + v) * n_splits + split] = ((threadgroup float4 *) of)[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 128u && kk == 0u) {
        device float *stats = (device float *)(partials + n_rows * 128u * n_splits);
        const uint64_t row = (uint64_t)token * args.n_head + head0 + hh;
        stats[(row * n_splits + split) * 2u + 0u] = l_run;
        stats[(row * n_splits + split) * 2u + 1u] = m_run;
    }
}

// reduce_v2.metal (2026-10-18 07:3x; appended to lib_att2b.metal by bench_reduce.py): the split attention's reduce with ONE THREAD PER OUTPUT
// float4 -- the live reduce (kernel_dsv4_indexed_mixed_attention_heads8_split_reduce) takes ~21 us for 1 MB of
// partials (latency: 128 threadgroups x 128-step simd_sum loops). Same partial layout, same sink rule, same arithmetic per element:
// out = sum_s e^(M_s - M*) o_s / (sum_s S_s e^(M_s - M*) + e^(sink - M*)), M* = max(M_s, sink); each thread recomputes its row's split scales.
kernel void kernel_llms_att_reduce_v2(constant ds4_metal_args_dsv4_indexed_attention & args, device const char *tmp, device const char *sinks, device char *dst,
        uint2 gid [[thread_position_in_grid]]) {
    const uint v = gid.x, row = gid.y, ns = args.n_splits; const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    if (v >= 128u || row >= n_rows || ns < 2u || ns > 31u) return;
    device const float4 *partials = (device const float4 *)tmp; device const float *stats = (device const float *)(partials + n_rows * 128u * ns);
    const float sink = ((device const float *)sinks)[row % args.n_head];
    float gmax = sink;
    for (uint s = 0; s < ns; s++) gmax = max(gmax, stats[(row * ns + s) * 2u + 1u]);
    float den = exp(sink - gmax); float4 acc = float4(0.f);
    for (uint s = 0; s < ns; s++) {
        const float S = stats[(row * ns + s) * 2u + 0u], M = stats[(row * ns + s) * 2u + 1u];
        const float sc = S > 0.0f ? exp(M - gmax) : 0.0f;
        den += S * sc; acc += partials[(row * 128u + v) * ns + s] * sc;
    }
    ((device float4 *)dst)[row * 128u + v] = den > 0.0f ? acc / den : float4(0.f);
}

kernel void kernel_llms_att2_r16qj(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q, device const char *raw_kv, device const char *comp_kv, device const char *topk, device char *tmp,
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.x, head0 = tgpig.y * 8u, split = tgpig.z, n_splits = args.n_splits;
    if (token >= args.n_tokens || head0 >= args.n_head || n_splits < 2u || n_splits > 31u || split >= n_splits) return;
    threadgroup half  * kh = (threadgroup half *) kv_shared;             // K chunk 16 x 512 half; before the loop: Q 8 x 512 half
    threadgroup float * sp = (threadgroup float *) (kh + 16u * 512u);   // partial scores: 8 sg x 8 heads x 16 keys
    threadgroup float * pp = sp + 1024u;                          // P: 8 heads x 16 keys
    threadgroup float * dg = pp + 128u;                              // diag(alpha): 8 x 8
    threadgroup int * six = (threadgroup int *) (dg + 64u);              // IDX: the split's row codes (<= 320)
    for (uint i = tid; i < 8u * 128u; i += 256u) {
        const uint h = i >> 7, c = i & 127u;
        device const float4 *q4 = (device const float4 *)(q + (uint64_t)token * args.q_token_stride + (uint64_t)(head0 + h) * args.q_head_stride);
        kv_shared[i] = (half4) q4[c];
    }
    if (tid < 64u) dg[tid] = 0.0f;
    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk + (uint64_t)token * args.topk_token_stride);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d_sg = (uint)sg * 64u;
    simdgroup_half8x8 A[8];
    for (uint k = 0; k < 8u; k++) simdgroup_load(A[k], kh + d_sg + k * 8u, 512);
    simdgroup_float8x8 O[8];
    for (uint n = 0; n < 8u; n++) O[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
    float m_run = -FLT_MAX / 2.0f, l_run = 0.0f;
    const uint hh = (uint)tid >> 4, kk = (uint)tid & 15u;
    for (uint i = tid; i < split_last - split_first; i += 256u) {
        const uint seq = split_first + i; int code;
        if (seq < raw_count) { const uint pos = raw_first + seq, logical = pos - first_raw_pos; code = -2 - (int)((args.raw_start + logical) % args.raw_cap); }
        else { const int32_t idx = row_topk[seq - raw_count]; code = (idx >= 0 && (uint)idx < visible) ? idx : -1; }
        six[i] = code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                       // every simdgroup holds its Q slice: the area becomes K
    for (uint seq0 = split_first; seq0 < split_last; seq0 += 16u) {
        for (uint off = tid; off < 16u * 128u; off += 256u) {
            const uint r = off >> 7, c = off & 127u, seq = seq0 + r;
            half4 value = half4(0.0h);
            const int code = seq < split_last ? six[seq - split_first] : -1;
            if (code >= 0) value = dsv4_load_cache_h4(comp_kv, args.comp_row_stride, (uint)code, c, args.comp_kv_f16 != 0u);
            else if (code <= -2) value = (half4) ((device const float4 *)(raw_kv + (uint64_t)(-2 - code) * args.raw_row_stride))[c];
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8>(0.f), s1 = make_filled_simdgroup_matrix<float, 8>(0.f);
            for (uint k = 0; k < 8u; k++) {
                simdgroup_half8x8 B0; simdgroup_load(B0, kh + 0u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s0, A[k], B0, s0);
                simdgroup_half8x8 B1; simdgroup_load(B1, kh + 8u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s1, A[k], B1, s1);
            }
            simdgroup_store(s0, sp + (uint)sg * 128u + 0u, 16);
            simdgroup_store(s1, sp + (uint)sg * 128u + 8u, 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128u) {
            float s = 0.0f;
            for (uint j = 0; j < 8u; j++) s += sp[j * 128u + hh * 16u + kk];
            s *= args.scale;
            const uint seq = seq0 + kk;
            const bool ok = seq < split_last && six[seq - split_first] != -1;
            float mc = ok ? s : -INFINITY;
            mc = max(mc, simd_shuffle_xor(mc, (ushort)1)); mc = max(mc, simd_shuffle_xor(mc, (ushort)2)); mc = max(mc, simd_shuffle_xor(mc, (ushort)4)); mc = max(mc, simd_shuffle_xor(mc, (ushort)8));
            const float m_new = max(m_run, mc);
            const float alpha = exp(m_run - m_new);
            const float p = ok ? exp(s - m_new) : 0.0f;
            float lc = p;
            lc += simd_shuffle_xor(lc, (ushort)1); lc += simd_shuffle_xor(lc, (ushort)2); lc += simd_shuffle_xor(lc, (ushort)4); lc += simd_shuffle_xor(lc, (ushort)8);
            l_run = l_run * alpha + lc; m_run = m_new;
            pp[hh * 16u + kk] = p;
#if LLMS_ATT2_RS
            if (kk == 0u) dg[hh] = LLMS_ATT2_RS == 2 ? alpha * alpha : alpha;   /* llms ATT2_RS : the row scale as a vector; 2 = canary */
#else
            if (kk == 0u) dg[hh * 8u + hh] = alpha;
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
#if LLMS_ATT2_RS
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 0, ulong2(0, 0), true);   /* llms ATT2_RS: Dm(i, j) = dg[i] */
#else
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 8);
#endif
            simdgroup_float8x8 P0; simdgroup_load(P0, pp + 0u, 16);
            simdgroup_float8x8 P1; simdgroup_load(P1, pp + 8u, 16);
            for (uint n = 0; n < 8u; n++) {
#if LLMS_ATT2_RS
                simdgroup_float8x8 t = O[n]; t.thread_elements() *= Dm.thread_elements();   /* llms ATT2_RS: row i scaled by alpha_i, no MMA */
#else
                simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O[n]);
#endif
                simdgroup_half8x8 V0; simdgroup_load(V0, kh + 0u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(t, P0, V0, t);
                simdgroup_half8x8 V1; simdgroup_load(V1, kh + 8u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(O[n], P1, V1, t);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float4 *partials = (device float4 *)tmp;
    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    threadgroup float * of = (threadgroup float *) kv_shared;              // O staged in 8 KB pieces of 8 heads x 512 dims
    const uint PD = 512u;                                   // dims per piece: the K area holds 8 x PD floats
    for (uint d0 = 0; d0 < 512u; d0 += PD) {
        for (uint n = 0; n < 8u; n++) { const uint d = d_sg + n * 8u; if (d >= d0 && d < d0 + PD) simdgroup_store(O[n], of + (d - d0), PD); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8u * (PD / 4u); i += 256u) {
            const uint h = i / (PD / 4u), v = i % (PD / 4u);
            const uint64_t row = (uint64_t)token * args.n_head + head0 + h;
            partials[((uint64_t)split * n_rows + row) * 128u + d0 / 4u + v] = ((threadgroup float4 *) of)[i];   // j: split-major partials, contiguous per row
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 128u && kk == 0u) {
        device float *stats = (device float *)(partials + n_rows * 128u * n_splits);
        const uint64_t row = (uint64_t)token * args.n_head + head0 + hh;
        stats[(row * n_splits + split) * 2u + 0u] = l_run;
        stats[(row * n_splits + split) * 2u + 1u] = m_run;
    }
}

// reduce_v3 (07:3x): kernel_llms_att_reduce_v2 for the SPLIT-MAJOR partials of kernel_llms_att2_r16qj ([split][row][128] float4; stats unchanged)
kernel void kernel_llms_att_reduce_v3(constant ds4_metal_args_dsv4_indexed_attention & args, device const char *tmp, device const char *sinks, device char *dst,
        uint2 gid [[thread_position_in_grid]]) {
    const uint v = gid.x, row = gid.y, ns = args.n_splits; const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    if (v >= 128u || row >= n_rows || ns < 2u || ns > 31u) return;
    device const float4 *partials = (device const float4 *)tmp; device const float *stats = (device const float *)(partials + n_rows * 128u * ns);
    const float sink = ((device const float *)sinks)[row % args.n_head];
    float gmax = sink;
    for (uint s = 0; s < ns; s++) gmax = max(gmax, stats[(row * ns + s) * 2u + 1u]);
    float den = exp(sink - gmax); float4 acc = float4(0.f);
    for (uint s = 0; s < ns; s++) {
        const float S = stats[(row * ns + s) * 2u + 0u], M = stats[(row * ns + s) * 2u + 1u];
        const float sc = S > 0.0f ? exp(M - gmax) : 0.0f;
        den += S * sc; acc += partials[((uint64_t)s * n_rows + row) * 128u + v] * sc;
    }
    ((device float4 *)dst)[row * 128u + v] = den > 0.0f ? acc / den : float4(0.f);
}

// ---- llms ATT2_HFIRST (apply_a2h.py): the split attention launched heads-fastest ----
kernel void kernel_llms_att2_r16qjh(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q, device const char *raw_kv, device const char *comp_kv, device const char *topk, device char *tmp,
        device const char *new_kv,   /* llms ATT2_RING: buffer 6, the step's new kv rows (a dummy when ring_prev is 0) */
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.y, head0 = tgpig.x * 8u, split = tgpig.z, n_splits = args.n_splits;   /* llms ATT2_HFIRST: heads fastest */
    if (token >= args.n_tokens || head0 >= args.n_head || n_splits < 2u || n_splits > 31u || split >= n_splits) return;
    threadgroup half  * kh = (threadgroup half *) kv_shared;             // K chunk 16 x 512 half; before the loop: Q 8 x 512 half
    threadgroup float * sp = (threadgroup float *) (kh + 16u * 512u);   // partial scores: 8 sg x 8 heads x 16 keys
    threadgroup float * pp = sp + 1024u;                          // P: 8 heads x 16 keys
    threadgroup float * dg = pp + 128u;                              // diag(alpha): 8 x 8
    threadgroup int * six = (threadgroup int *) (dg + 64u);              // IDX: the split's row codes (<= 320)
    for (uint i = tid; i < 8u * 128u; i += 256u) {
        const uint h = i >> 7, c = i & 127u;
        device const float4 *q4 = (device const float4 *)(q + (uint64_t)token * args.q_token_stride + (uint64_t)(head0 + h) * args.q_head_stride);
        kv_shared[i] = (half4) q4[c];
    }
    if (tid < 64u) dg[tid] = 0.0f;
    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk + (uint64_t)token * args.topk_token_stride);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d_sg = (uint)sg * 64u;
    simdgroup_half8x8 A[8];
    for (uint k = 0; k < 8u; k++) simdgroup_load(A[k], kh + d_sg + k * 8u, 512);
    simdgroup_float8x8 O[8];
    for (uint n = 0; n < 8u; n++) O[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
    float m_run = -FLT_MAX / 2.0f, l_run = 0.0f;
    const uint hh = (uint)tid >> 4, kk = (uint)tid & 15u;
    for (uint i = tid; i < split_last - split_first; i += 256u) {
        const uint seq = split_first + i; int code;
        if (seq < raw_count) { const uint pos = raw_first + seq, logical = pos - first_raw_pos;
            code = (args.ring_prev != 0u && logical >= args.ring_prev) ? (-200 - (int)(logical - args.ring_prev))   /* llms ATT2_RING: a new row */
                                                                       : (-2 - (int)((args.raw_start + logical) % args.raw_cap)); }
        else { const int32_t idx = row_topk[seq - raw_count]; code = (idx >= 0 && (uint)idx < visible) ? idx : -1; }
        six[i] = code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                       // every simdgroup holds its Q slice: the area becomes K
    for (uint seq0 = split_first; seq0 < split_last; seq0 += 16u) {
        for (uint off = tid; off < 16u * 128u; off += 256u) {
            const uint r = off >> 7, c = off & 127u, seq = seq0 + r;
            half4 value = half4(0.0h);
            const int code = seq < split_last ? six[seq - split_first] : -1;
            if (code >= 0) value = dsv4_load_cache_h4(comp_kv, args.comp_row_stride, (uint)code, c, args.comp_kv_f16 != 0u);
            else if (args.ring_prev != 0u && code <= -200) value = (half4) ((device const float4 *)(new_kv + (uint64_t)(-200 - code) * args.raw_row_stride))[c];   /* llms ATT2_RING */
            else if (code <= -2) value = (half4) ((device const float4 *)(raw_kv + (uint64_t)(-2 - code) * args.raw_row_stride))[c];
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8>(0.f), s1 = make_filled_simdgroup_matrix<float, 8>(0.f);
            for (uint k = 0; k < 8u; k++) {
                simdgroup_half8x8 B0; simdgroup_load(B0, kh + 0u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s0, A[k], B0, s0);
                simdgroup_half8x8 B1; simdgroup_load(B1, kh + 8u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s1, A[k], B1, s1);
            }
            simdgroup_store(s0, sp + (uint)sg * 128u + 0u, 16);
            simdgroup_store(s1, sp + (uint)sg * 128u + 8u, 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 128u) {
            float s = 0.0f;
            for (uint j = 0; j < 8u; j++) s += sp[j * 128u + hh * 16u + kk];
            s *= args.scale;
            const uint seq = seq0 + kk;
            const bool ok = seq < split_last && six[seq - split_first] != -1;
            float mc = ok ? s : -INFINITY;
            mc = max(mc, simd_shuffle_xor(mc, (ushort)1)); mc = max(mc, simd_shuffle_xor(mc, (ushort)2)); mc = max(mc, simd_shuffle_xor(mc, (ushort)4)); mc = max(mc, simd_shuffle_xor(mc, (ushort)8));
            const float m_new = max(m_run, mc);
            const float alpha = exp(m_run - m_new);
            const float p = ok ? exp(s - m_new) : 0.0f;
            float lc = p;
            lc += simd_shuffle_xor(lc, (ushort)1); lc += simd_shuffle_xor(lc, (ushort)2); lc += simd_shuffle_xor(lc, (ushort)4); lc += simd_shuffle_xor(lc, (ushort)8);
            l_run = l_run * alpha + lc; m_run = m_new;
            pp[hh * 16u + kk] = p;
#if LLMS_ATT2_RS
            if (kk == 0u) dg[hh] = LLMS_ATT2_RS == 2 ? alpha * alpha : alpha;   /* llms ATT2_RS : the row scale as a vector; 2 = canary */
#else
            if (kk == 0u) dg[hh * 8u + hh] = alpha;
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
#if LLMS_ATT2_RS
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 0, ulong2(0, 0), true);   /* llms ATT2_RS: Dm(i, j) = dg[i] */
#else
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 8);
#endif
            simdgroup_float8x8 P0; simdgroup_load(P0, pp + 0u, 16);
            simdgroup_float8x8 P1; simdgroup_load(P1, pp + 8u, 16);
            for (uint n = 0; n < 8u; n++) {
#if LLMS_ATT2_RS
                simdgroup_float8x8 t = O[n]; t.thread_elements() *= Dm.thread_elements();   /* llms ATT2_RS: row i scaled by alpha_i, no MMA */
#else
                simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O[n]);
#endif
                simdgroup_half8x8 V0; simdgroup_load(V0, kh + 0u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(t, P0, V0, t);
                simdgroup_half8x8 V1; simdgroup_load(V1, kh + 8u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(O[n], P1, V1, t);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float4 *partials = (device float4 *)tmp;
    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    threadgroup float * of = (threadgroup float *) kv_shared;              // O staged in 8 KB pieces of 8 heads x 512 dims
    const uint PD = 512u;                                   // dims per piece: the K area holds 8 x PD floats
    for (uint d0 = 0; d0 < 512u; d0 += PD) {
        for (uint n = 0; n < 8u; n++) { const uint d = d_sg + n * 8u; if (d >= d0 && d < d0 + PD) simdgroup_store(O[n], of + (d - d0), PD); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8u * (PD / 4u); i += 256u) {
            const uint h = i / (PD / 4u), v = i % (PD / 4u);
            const uint64_t row = (uint64_t)token * args.n_head + head0 + h;
            partials[((uint64_t)split * n_rows + row) * 128u + d0 / 4u + v] = ((threadgroup float4 *) of)[i];   // j: split-major partials, contiguous per row
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 128u && kk == 0u) {
        device float *stats = (device float *)(partials + n_rows * 128u * n_splits);
        const uint64_t row = (uint64_t)token * args.n_head + head0 + hh;
        stats[(row * n_splits + split) * 2u + 0u] = l_run;
        stats[(row * n_splits + split) * 2u + 1u] = m_run;
    }
}

// ---- llms ATT2_H16 (make_att2h16_source.py): kernel_llms_att2_r16qjh with 16 heads per threadgroup ----
kernel void kernel_llms_att2_r16qjh16(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q, device const char *raw_kv, device const char *comp_kv, device const char *topk, device char *tmp,
        device const char *new_kv,   /* llms ATT2_RING: buffer 6, the step's new kv rows (a dummy when ring_prev is 0) */
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.y, head0 = tgpig.x * 16u, split = tgpig.z, n_splits = args.n_splits;   /* llms ATT2_HFIRST: heads fastest */
    if (token >= args.n_tokens || head0 >= args.n_head || n_splits < 2u || n_splits > 31u || split >= n_splits) return;
    threadgroup half  * kh = (threadgroup half *) kv_shared;             // K chunk 16 x 512 half; before the loop: Q 8 x 512 half
    threadgroup float * sp = (threadgroup float *) (kh + 16u * 512u);   // partial scores: 8 sg x 8 heads x 16 keys
    threadgroup float * pp = sp + 2048u;                          // P: 8 heads x 16 keys
    threadgroup float * dg = pp + 256u;                              // diag(alpha): 8 x 8
    threadgroup int * six = (threadgroup int *) (dg + 128u);              // IDX: the split's row codes (<= 320)
    for (uint i = tid; i < 16u * 128u; i += 256u) {
        const uint h = i >> 7, c = i & 127u;
        device const float4 *q4 = (device const float4 *)(q + (uint64_t)token * args.q_token_stride + (uint64_t)(head0 + h) * args.q_head_stride);
        kv_shared[i] = (half4) q4[c];
    }
    if (tid < 128u) dg[tid] = 0.0f;
    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk + (uint64_t)token * args.topk_token_stride);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d_sg = (uint)sg * 64u;
    simdgroup_half8x8 A[8];
    for (uint k = 0; k < 8u; k++) simdgroup_load(A[k], kh + d_sg + k * 8u, 512);
    simdgroup_half8x8 A2[8];   /* llms ATT2_H16: heads 8-15 */
    for (uint k = 0; k < 8u; k++) simdgroup_load(A2[k], kh + 8u * 512u + d_sg + k * 8u, 512);
    simdgroup_float8x8 O[8];
    for (uint n = 0; n < 8u; n++) O[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
    simdgroup_float8x8 O2[8];   /* llms ATT2_H16 */
    for (uint n = 0; n < 8u; n++) O2[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
    float m_run = -FLT_MAX / 2.0f, l_run = 0.0f;
    const uint hh = (uint)tid >> 4, kk = (uint)tid & 15u;
    for (uint i = tid; i < split_last - split_first; i += 256u) {
        const uint seq = split_first + i; int code;
        if (seq < raw_count) { const uint pos = raw_first + seq, logical = pos - first_raw_pos;
            code = (args.ring_prev != 0u && logical >= args.ring_prev) ? (-200 - (int)(logical - args.ring_prev))   /* llms ATT2_RING: a new row */
                                                                       : (-2 - (int)((args.raw_start + logical) % args.raw_cap)); }
        else { const int32_t idx = row_topk[seq - raw_count]; code = (idx >= 0 && (uint)idx < visible) ? idx : -1; }
        six[i] = code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                       // every simdgroup holds its Q slice: the area becomes K
    for (uint seq0 = split_first; seq0 < split_last; seq0 += 16u) {
        for (uint off = tid; off < 16u * 128u; off += 256u) {
            const uint r = off >> 7, c = off & 127u, seq = seq0 + r;
            half4 value = half4(0.0h);
            const int code = seq < split_last ? six[seq - split_first] : -1;
            if (code >= 0) value = dsv4_load_cache_h4(comp_kv, args.comp_row_stride, (uint)code, c, args.comp_kv_f16 != 0u);
            else if (args.ring_prev != 0u && code <= -200) value = (half4) ((device const float4 *)(new_kv + (uint64_t)(-200 - code) * args.raw_row_stride))[c];   /* llms ATT2_RING */
            else if (code <= -2) value = (half4) ((device const float4 *)(raw_kv + (uint64_t)(-2 - code) * args.raw_row_stride))[c];
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8>(0.f), s1 = make_filled_simdgroup_matrix<float, 8>(0.f);
            simdgroup_float8x8 s2 = make_filled_simdgroup_matrix<float, 8>(0.f), s3 = make_filled_simdgroup_matrix<float, 8>(0.f);   /* llms ATT2_H16 */
            for (uint k = 0; k < 8u; k++) {
                simdgroup_half8x8 B0; simdgroup_load(B0, kh + 0u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s0, A[k], B0, s0); simdgroup_multiply_accumulate(s2, A2[k], B0, s2);
                simdgroup_half8x8 B1; simdgroup_load(B1, kh + 8u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s1, A[k], B1, s1); simdgroup_multiply_accumulate(s3, A2[k], B1, s3);
            }
            simdgroup_store(s0, sp + (uint)sg * 256u + 0u, 16);
            simdgroup_store(s1, sp + (uint)sg * 256u + 8u, 16);
            simdgroup_store(s2, sp + (uint)sg * 256u + 128u, 16);
            simdgroup_store(s3, sp + (uint)sg * 256u + 136u, 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   /* llms ATT2_H16: all 256 threads, heads 0-15 */
            float s = 0.0f;
            for (uint j = 0; j < 8u; j++) s += sp[j * 256u + hh * 16u + kk];
            s *= args.scale;
            const uint seq = seq0 + kk;
            const bool ok = seq < split_last && six[seq - split_first] != -1;
            float mc = ok ? s : -INFINITY;
            mc = max(mc, simd_shuffle_xor(mc, (ushort)1)); mc = max(mc, simd_shuffle_xor(mc, (ushort)2)); mc = max(mc, simd_shuffle_xor(mc, (ushort)4)); mc = max(mc, simd_shuffle_xor(mc, (ushort)8));
            const float m_new = max(m_run, mc);
            const float alpha = exp(m_run - m_new);
            const float p = ok ? exp(s - m_new) : 0.0f;
            float lc = p;
            lc += simd_shuffle_xor(lc, (ushort)1); lc += simd_shuffle_xor(lc, (ushort)2); lc += simd_shuffle_xor(lc, (ushort)4); lc += simd_shuffle_xor(lc, (ushort)8);
            l_run = l_run * alpha + lc; m_run = m_new;
            pp[hh * 16u + kk] = p;
#if LLMS_ATT2_RS
            if (kk == 0u) dg[hh] = LLMS_ATT2_RS == 2 ? alpha * alpha : alpha;   /* llms ATT2_RS : the row scale as a vector; 2 = canary */
#else
            if (kk == 0u) dg[(hh >> 3) * 64u + (hh & 7u) * 8u + (hh & 7u)] = alpha;   /* llms ATT2_H16: two diagonal blocks */
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
#if LLMS_ATT2_RS
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 0, ulong2(0, 0), true);   /* llms ATT2_RS: Dm(i, j) = dg[i] */
#else
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 8);
#endif
            simdgroup_float8x8 P0; simdgroup_load(P0, pp + 0u, 16);
            simdgroup_float8x8 P1; simdgroup_load(P1, pp + 8u, 16);
            for (uint n = 0; n < 8u; n++) {
#if LLMS_ATT2_RS
                simdgroup_float8x8 t = O[n]; t.thread_elements() *= Dm.thread_elements();   /* llms ATT2_RS: row i scaled by alpha_i, no MMA */
#else
                simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O[n]);
#endif
                simdgroup_half8x8 V0; simdgroup_load(V0, kh + 0u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(t, P0, V0, t);
                simdgroup_half8x8 V1; simdgroup_load(V1, kh + 8u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(O[n], P1, V1, t);
            }
        }
        {
#if LLMS_ATT2_RS
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg + 8u, 0, ulong2(0, 0), true);   /* llms ATT2_RS: Dm(i, j) = dg[i] */
#else
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg + 64u, 8);
#endif
            simdgroup_float8x8 P0; simdgroup_load(P0, pp + 128u, 16);
            simdgroup_float8x8 P1; simdgroup_load(P1, pp + 136u, 16);
            for (uint n = 0; n < 8u; n++) {
#if LLMS_ATT2_RS
                simdgroup_float8x8 t = O2[n]; t.thread_elements() *= Dm.thread_elements();   /* llms ATT2_RS: row i scaled by alpha_i, no MMA */
#else
                simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O2[n]);
#endif
                simdgroup_half8x8 V0; simdgroup_load(V0, kh + 0u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(t, P0, V0, t);
                simdgroup_half8x8 V1; simdgroup_load(V1, kh + 8u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(O2[n], P1, V1, t);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float4 *partials = (device float4 *)tmp;
    const uint64_t n_rows = (uint64_t)args.n_tokens * args.n_head;
    threadgroup float * of = (threadgroup float *) kv_shared;              // O staged in 8 KB pieces of 8 heads x 512 dims
    const uint PD = 512u;                                   // dims per piece: the K area holds 8 x PD floats
    for (uint d0 = 0; d0 < 512u; d0 += PD) {
        for (uint n = 0; n < 8u; n++) { const uint d = d_sg + n * 8u; if (d >= d0 && d < d0 + PD) simdgroup_store(O[n], of + (d - d0), PD); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8u * (PD / 4u); i += 256u) {
            const uint h = i / (PD / 4u), v = i % (PD / 4u);
            const uint64_t row = (uint64_t)token * args.n_head + head0 + h;
            partials[((uint64_t)split * n_rows + row) * 128u + d0 / 4u + v] = ((threadgroup float4 *) of)[i];   // j: split-major partials, contiguous per row
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint d0 = 0; d0 < 512u; d0 += PD) {
        for (uint n = 0; n < 8u; n++) { const uint d = d_sg + n * 8u; if (d >= d0 && d < d0 + PD) simdgroup_store(O2[n], of + (d - d0), PD); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8u * (PD / 4u); i += 256u) {
            const uint h = i / (PD / 4u), v = i % (PD / 4u);
            const uint64_t row = (uint64_t)token * args.n_head + head0 + 8u + h;
            partials[((uint64_t)split * n_rows + row) * 128u + d0 / 4u + v] = ((threadgroup float4 *) of)[i];   // j: split-major partials, contiguous per row
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (kk == 0u) {   /* llms ATT2_H16: 16 heads' stats */
        device float *stats = (device float *)(partials + n_rows * 128u * n_splits);
        const uint64_t row = (uint64_t)token * args.n_head + head0 + hh;
        stats[(row * n_splits + split) * 2u + 0u] = l_run;
        stats[(row * n_splits + split) * 2u + 1u] = m_run;
    }
}

// ---- llms ATT2_S1 (make_att2s1_source.py): kernel_llms_att2_r16qjh16 at one split, finalized in-kernel ----
kernel void kernel_llms_att2_r16qjh16s1(
        constant ds4_metal_args_dsv4_indexed_attention & args,
        device const char *q, device const char *raw_kv, device const char *comp_kv, device const char *topk, device char *tmp,
        device const char *new_kv,   /* llms ATT2_RING: buffer 6, the step's new kv rows (a dummy when ring_prev is 0) */
        device const float *sinks [[buffer(7)]],   /* llms ATT2_S1: the heads' sink logits */
        threadgroup half4 *kv_shared [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint token = tgpig.y, head0 = tgpig.x * 16u, split = tgpig.z, n_splits = args.n_splits;   /* llms ATT2_HFIRST: heads fastest */
    if (token >= args.n_tokens || head0 >= args.n_head || n_splits != 1u || split >= n_splits) return;
    threadgroup half  * kh = (threadgroup half *) kv_shared;             // K chunk 16 x 512 half; before the loop: Q 8 x 512 half
    threadgroup float * sp = (threadgroup float *) (kh + 16u * 512u);   // partial scores: 8 sg x 8 heads x 16 keys
    threadgroup float * pp = sp + 2048u;                          // P: 8 heads x 16 keys
    threadgroup float * dg = pp + 256u;                              // diag(alpha): 8 x 8
    threadgroup int * six = (threadgroup int *) (dg + 128u);              // IDX: the split's row codes (<= 320)
    for (uint i = tid; i < 16u * 128u; i += 256u) {
        const uint h = i >> 7, c = i & 127u;
        device const float4 *q4 = (device const float4 *)(q + (uint64_t)token * args.q_token_stride + (uint64_t)(head0 + h) * args.q_head_stride);
        kv_shared[i] = (half4) q4[c];
    }
    if (tid < 128u) dg[tid] = 0.0f;
    const uint qpos = args.pos0 + token;
    const uint last_pos = args.pos0 + args.n_tokens - 1u;
    const uint first_raw_pos = last_pos + 1u - args.n_raw;
    const uint raw_last_pos = first_raw_pos + args.n_raw - 1u;
    const uint window_first = (args.window != 0u && qpos + 1u > args.window) ? qpos + 1u - args.window : 0u;
    const uint raw_first = max(first_raw_pos, window_first);
    const uint raw_last = min(qpos, raw_last_pos);
    const uint raw_count = raw_first <= raw_last ? raw_last - raw_first + 1u : 0u;
    const uint total_rows = raw_count + args.top_k;
    const uint rows_per_split = (total_rows + n_splits - 1u) / n_splits;
    const uint split_first = min(split * rows_per_split, total_rows);
    const uint split_last = min(split_first + rows_per_split, total_rows);
    const uint visible = min((qpos + 1u) / args.ratio, args.n_comp);
    device const int32_t *row_topk = (device const int32_t *)(topk + (uint64_t)token * args.topk_token_stride);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint d_sg = (uint)sg * 64u;
    simdgroup_half8x8 A[8];
    for (uint k = 0; k < 8u; k++) simdgroup_load(A[k], kh + d_sg + k * 8u, 512);
    simdgroup_half8x8 A2[8];   /* llms ATT2_H16: heads 8-15 */
    for (uint k = 0; k < 8u; k++) simdgroup_load(A2[k], kh + 8u * 512u + d_sg + k * 8u, 512);
    simdgroup_float8x8 O[8];
    for (uint n = 0; n < 8u; n++) O[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
    simdgroup_float8x8 O2[8];   /* llms ATT2_H16 */
    for (uint n = 0; n < 8u; n++) O2[n] = make_filled_simdgroup_matrix<float, 8>(0.f);
    float m_run = -FLT_MAX / 2.0f, l_run = 0.0f;
    const uint hh = (uint)tid >> 4, kk = (uint)tid & 15u;
    for (uint i = tid; i < split_last - split_first; i += 256u) {
        const uint seq = split_first + i; int code;
        if (seq < raw_count) { const uint pos = raw_first + seq, logical = pos - first_raw_pos;
            code = (args.ring_prev != 0u && logical >= args.ring_prev) ? (-200 - (int)(logical - args.ring_prev))   /* llms ATT2_RING: a new row */
                                                                       : (-2 - (int)((args.raw_start + logical) % args.raw_cap)); }
        else { const int32_t idx = row_topk[seq - raw_count]; code = (idx >= 0 && (uint)idx < visible) ? idx : -1; }
        six[i] = code;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);                       // every simdgroup holds its Q slice: the area becomes K
    for (uint seq0 = split_first; seq0 < split_last; seq0 += 16u) {
        for (uint off = tid; off < 16u * 128u; off += 256u) {
            const uint r = off >> 7, c = off & 127u, seq = seq0 + r;
            half4 value = half4(0.0h);
            const int code = seq < split_last ? six[seq - split_first] : -1;
            if (code >= 0) value = dsv4_load_cache_h4(comp_kv, args.comp_row_stride, (uint)code, c, args.comp_kv_f16 != 0u);
            else if (args.ring_prev != 0u && code <= -200) value = (half4) ((device const float4 *)(new_kv + (uint64_t)(-200 - code) * args.raw_row_stride))[c];   /* llms ATT2_RING */
            else if (code <= -2) value = (half4) ((device const float4 *)(raw_kv + (uint64_t)(-2 - code) * args.raw_row_stride))[c];
            kv_shared[off] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8>(0.f), s1 = make_filled_simdgroup_matrix<float, 8>(0.f);
            simdgroup_float8x8 s2 = make_filled_simdgroup_matrix<float, 8>(0.f), s3 = make_filled_simdgroup_matrix<float, 8>(0.f);   /* llms ATT2_H16 */
            for (uint k = 0; k < 8u; k++) {
                simdgroup_half8x8 B0; simdgroup_load(B0, kh + 0u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s0, A[k], B0, s0); simdgroup_multiply_accumulate(s2, A2[k], B0, s2);
                simdgroup_half8x8 B1; simdgroup_load(B1, kh + 8u * 512u + d_sg + k * 8u, 512, ulong2(0, 0), true); simdgroup_multiply_accumulate(s1, A[k], B1, s1); simdgroup_multiply_accumulate(s3, A2[k], B1, s3);
            }
            simdgroup_store(s0, sp + (uint)sg * 256u + 0u, 16);
            simdgroup_store(s1, sp + (uint)sg * 256u + 8u, 16);
            simdgroup_store(s2, sp + (uint)sg * 256u + 128u, 16);
            simdgroup_store(s3, sp + (uint)sg * 256u + 136u, 16);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {   /* llms ATT2_H16: all 256 threads, heads 0-15 */
            float s = 0.0f;
            for (uint j = 0; j < 8u; j++) s += sp[j * 256u + hh * 16u + kk];
            s *= args.scale;
            const uint seq = seq0 + kk;
            const bool ok = seq < split_last && six[seq - split_first] != -1;
            float mc = ok ? s : -INFINITY;
            mc = max(mc, simd_shuffle_xor(mc, (ushort)1)); mc = max(mc, simd_shuffle_xor(mc, (ushort)2)); mc = max(mc, simd_shuffle_xor(mc, (ushort)4)); mc = max(mc, simd_shuffle_xor(mc, (ushort)8));
            const float m_new = max(m_run, mc);
            const float alpha = exp(m_run - m_new);
            const float p = ok ? exp(s - m_new) : 0.0f;
            float lc = p;
            lc += simd_shuffle_xor(lc, (ushort)1); lc += simd_shuffle_xor(lc, (ushort)2); lc += simd_shuffle_xor(lc, (ushort)4); lc += simd_shuffle_xor(lc, (ushort)8);
            l_run = l_run * alpha + lc; m_run = m_new;
            pp[hh * 16u + kk] = p;
#if LLMS_ATT2_RS
            if (kk == 0u) dg[hh] = LLMS_ATT2_RS == 2 ? alpha * alpha : alpha;   /* llms ATT2_RS : the row scale as a vector; 2 = canary */
#else
            if (kk == 0u) dg[(hh >> 3) * 64u + (hh & 7u) * 8u + (hh & 7u)] = alpha;   /* llms ATT2_H16: two diagonal blocks */
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
#if LLMS_ATT2_RS
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 0, ulong2(0, 0), true);   /* llms ATT2_RS: Dm(i, j) = dg[i] */
#else
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg, 8);
#endif
            simdgroup_float8x8 P0; simdgroup_load(P0, pp + 0u, 16);
            simdgroup_float8x8 P1; simdgroup_load(P1, pp + 8u, 16);
            for (uint n = 0; n < 8u; n++) {
#if LLMS_ATT2_RS
                simdgroup_float8x8 t = O[n]; t.thread_elements() *= Dm.thread_elements();   /* llms ATT2_RS: row i scaled by alpha_i, no MMA */
#else
                simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O[n]);
#endif
                simdgroup_half8x8 V0; simdgroup_load(V0, kh + 0u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(t, P0, V0, t);
                simdgroup_half8x8 V1; simdgroup_load(V1, kh + 8u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(O[n], P1, V1, t);
            }
        }
        {
#if LLMS_ATT2_RS
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg + 8u, 0, ulong2(0, 0), true);   /* llms ATT2_RS: Dm(i, j) = dg[i] */
#else
            simdgroup_float8x8 Dm; simdgroup_load(Dm, dg + 64u, 8);
#endif
            simdgroup_float8x8 P0; simdgroup_load(P0, pp + 128u, 16);
            simdgroup_float8x8 P1; simdgroup_load(P1, pp + 136u, 16);
            for (uint n = 0; n < 8u; n++) {
#if LLMS_ATT2_RS
                simdgroup_float8x8 t = O2[n]; t.thread_elements() *= Dm.thread_elements();   /* llms ATT2_RS: row i scaled by alpha_i, no MMA */
#else
                simdgroup_float8x8 t; simdgroup_multiply(t, Dm, O2[n]);
#endif
                simdgroup_half8x8 V0; simdgroup_load(V0, kh + 0u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(t, P0, V0, t);
                simdgroup_half8x8 V1; simdgroup_load(V1, kh + 8u * 512u + d_sg + n * 8u, 512);
                simdgroup_multiply_accumulate(O2[n], P1, V1, t);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    /* llms ATT2_S1: one split -- finalize in-kernel with kernel_llms_att_reduce_v3's arithmetic, no partials */
    if (kk == 0u) {
        const float sink = sinks[head0 + hh];
        const float gmax = max(sink, m_run);
        const float sc = l_run > 0.0f ? exp(m_run - gmax) : 0.0f;
        const float den = exp(sink - gmax) + l_run * sc;
        dg[hh] = den > 0.0f ? sc / den : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float4 *out4 = (device float4 *)tmp;
    threadgroup float * of = (threadgroup float *) kv_shared;              // O staged in 16 KB pieces of 8 heads x 512 dims
    for (uint n = 0; n < 8u; n++) simdgroup_store(O[n], of + d_sg + n * 8u, 512u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 8u * 128u; i += 256u) {
        const uint h = i >> 7, v = i & 127u;
        const uint64_t row = (uint64_t)token * args.n_head + head0 + h;
        out4[row * 128u + v] = ((threadgroup float4 *) of)[i] * dg[h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n = 0; n < 8u; n++) simdgroup_store(O2[n], of + d_sg + n * 8u, 512u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 8u * 128u; i += 256u) {
        const uint h = i >> 7, v = i & 127u;
        const uint64_t row = (uint64_t)token * args.n_head + head0 + 8u + h;
        out4[row * 128u + v] = ((threadgroup float4 *) of)[i] * dg[8u + h];
    }
}

// ---- upstream antirez/ds4 5379c1c (PR #1135, e5b19e2), verbatim ----
// Copy the selected compressed-key mask while retaining causal visibility.
// Image bidirectionality applies to raw keys only, never compressed keys.
struct ds4_metal_args_causal_comp_mask {
    uint n_comp, n_tokens, pos0, ratio, n_keys;
};

kernel void kernel_dsv4_causal_comp_mask(
        constant ds4_metal_args_causal_comp_mask &args,
        device const float *selected_mask,
        device half *dst,
        uint2 gid [[thread_position_in_grid]]) {
    const uint col = gid.x, row = gid.y;
    if (col >= args.n_comp || row >= args.n_tokens) return;
    const ulong visible = (ulong(args.pos0) + ulong(row) + 1ul) / args.ratio;
    dst[ulong(row) * args.n_keys + col] = ulong(col) < visible
        ? half(selected_mask[ulong(row) * args.n_comp + col]) : half(-INFINITY);
}


// ---- llms SP_IDXB (make_idxr_source.py, 2026-10-22): the short prefill's per-row indexer scores in ONE dispatch,
// grid (key blocks, rows) --; each row its own argument record. ----
struct llms_idx_rows_batch { uint rows; uint score_stride; };
kernel void kernel_llms_indexer_score_rows_h_t4(constant ds4_metal_args_glm_indexer_score_one *a [[buffer(0)]], device const char *q [[buffer(1)]],
        device const float *w [[buffer(2)]], device const char *k [[buffer(3)]], device float *s [[buffer(4)]], constant llms_idx_rows_batch &b [[buffer(5)]],
        threadgroup float *sm [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ntg [[threadgroups_per_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    if (tg.y >= b.rows) return;
    kernel_llms_indexer_score_sgm_h_impl<4>(a[tg.y], q + (ulong)tg.y * 32u * 128u * 4u, w + (ulong)tg.y * 32u, k, s + (ulong)tg.y * b.score_stride, sm, tg.x, ntg.x, lane, sg);
}

template<short NRT>
void masked_impl(constant ds4_metal_args_glm_indexer_score_one & args, device const char *q, device const float *weights,
                                          device const char *indexer_key_cache, device float *scores, device const float *mask, threadgroup float *shmem, uint tg, uint ntg, ushort lane, ushort sg) {
    constexpr uint NR = NRT * 8;
    if (args.n_head != 32u || args.head_dim != 128u || args.cache_f16 == 0u || args.n_rows < NR) return;
    threadgroup half *qs = (threadgroup half *)(shmem + NR * 32u);          // the query: 32 heads x 128, after the C tiles; staged ONCE
    for (uint i = (uint)sg * 32u + lane; i < 32u * 128u; i += 128u) qs[i] = (half)(((device const float *)q)[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const half *Q = qs + (uint)sg * 8u * 128u;
    const uint nblk = (args.n_rows + NR - 1u) / NR;
    for (uint blk = tg; blk < nblk; blk += ntg) {                          // persistent: a threadgroup walks several row blocks
        const uint r0 = min(blk * NR, args.n_rows - NR);
        bool active = false;
        for(uint b=r0/8u;b<=(r0+NR-1u)/8u;b++) active = active || mask[b]==0.0f;
        if(!active) { if(sg==0) for(uint row=lane;row<NR;row+=32u) scores[r0+row]=-INFINITY; continue; }
        device const half *K = (device const half *)indexer_key_cache + (uint64_t)r0 * 128u;
        simdgroup_float8x8 C[NRT];
        FOR_UNROLL (short t = 0; t < NRT; t++) C[t] = make_filled_simdgroup_matrix<float, 8>(0.f);
        for (short kt = 0; kt < 16; kt++) {
            simdgroup_half8x8 B; simdgroup_load(B, Q + kt * 8, 128, ulong2(0, 0), true);
            FOR_UNROLL (short t = 0; t < NRT; t++) { simdgroup_half8x8 A; simdgroup_load(A, K + t * 8 * 128 + kt * 8, 128); simdgroup_multiply_accumulate(C[t], A, B, C[t]); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short t = 0; t < NRT; t++) simdgroup_store(C[t], shmem + (t * 8) * 32 + sg * 8, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (uint row = lane; row < NR; row += 32u) {
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
                float acc = 0.0f;
                for (uint h = 0; h < 32u; h++) acc = acc + max(shmem[row * 32u + h] * args.scale, 0.0f) * weights[h];
                scores[r0 + row] = mask[(r0+row)/8u]==0.0f ? acc : -INFINITY;
            }
        }
    }
}
kernel void masked_rows(constant ds4_metal_args_glm_indexer_score_one *a [[buffer(0)]], device const char *q [[buffer(1)]], device const float *w [[buffer(2)]],device const char *k [[buffer(3)]], device float *s [[buffer(4)]], constant llms_idx_rows_batch &b [[buffer(5)]], device const float *m [[buffer(6)]],constant uint &ms [[buffer(7)]],threadgroup float *sm [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]], uint3 ntg [[threadgroups_per_grid]], ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) { if(tg.y>=b.rows)return; masked_impl<4>(a[tg.y],q+(ulong)tg.y*4096u*4u,w+(ulong)tg.y*32u,k,s+(ulong)tg.y*b.score_stride,m+(ulong)tg.y*ms,sm,tg.x,ntg.x,lane,sg); }
