#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

#include "common/stride_utils.h"

// -----------------------------------------------------------------------------
// RoPE (float32 only for now)
// -----------------------------------------------------------------------------

#define DEFINE_ROPE_KERNEL(NAME, VALUE_T, READ_FN, WRITE_FN)                                                           \
    kernel void NAME(                                                                                                  \
        device const VALUE_T *x [[buffer(0)]], device const float *positions [[buffer(1)]],                            \
        device VALUE_T *out [[buffer(2)]], constant uint &seq_len [[buffer(3)]], constant uint &dim [[buffer(4)]],     \
        constant uint &total_seqs [[buffer(5)]], constant float *freqs [[buffer(6)]],                                  \
        constant float &attn_scale [[buffer(7)]], constant uint &rope_type [[buffer(8)]],                              \
        uint gid [[thread_position_in_grid]]                                                                           \
    ) {                                                                                                                \
        uint total_tokens = total_seqs * seq_len;                                                                      \
        if (gid >= total_tokens) {                                                                                     \
            return;                                                                                                    \
        }                                                                                                              \
        uint seq_index = gid / seq_len;                                                                                \
        uint token_index = gid % seq_len;                                                                              \
        float pos = positions[gid];                                                                                    \
        uint token_base = (seq_index * seq_len + token_index) * dim;                                                   \
        uint half_dim = dim >> 1;                                                                                      \
        bool is_neox = rope_type != 0u;                                                                                \
        for (uint i = 0; i < half_dim; ++i) {                                                                          \
            uint even_index = is_neox ? (token_base + i) : (token_base + i * 2u);                                      \
            uint odd_index = is_neox ? (token_base + half_dim + i) : (even_index + 1u);                                \
            float freq = pos * freqs[i];                                                                               \
            float cos_freq = cos(freq) * attn_scale;                                                                   \
            float sin_freq = sin(freq) * attn_scale;                                                                   \
            float x0 = READ_FN(x[even_index]);                                                                         \
            float x1 = READ_FN(x[odd_index]);                                                                          \
            float out_even = x0 * cos_freq - x1 * sin_freq;                                                            \
            float out_odd = x0 * sin_freq + x1 * cos_freq;                                                             \
            out[even_index] = WRITE_FN(out_even);                                                                      \
            out[odd_index] = WRITE_FN(out_odd);                                                                        \
        }                                                                                                              \
    }

DEFINE_ROPE_KERNEL(rope_f32, float, read_float, write_float)
DEFINE_ROPE_KERNEL(rope_f16, half, read_half, write_half)
DEFINE_ROPE_KERNEL(rope_bf16, ushort, read_bf16, write_bf16)
#if MARMOT_ENABLE_FP8
DEFINE_ROPE_KERNEL(rope_fp8_e4m3, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_ROPE_KERNEL(rope_fp8_e5m2, uchar, read_fp8_e5m2, write_fp8_e5m2)
#endif

#undef DEFINE_ROPE_KERNEL

#define DEFINE_ROPE_KERNEL_STRIDED(NAME, VALUE_T, READ_FN, WRITE_FN)                                                   \
    kernel void NAME(                                                                                                  \
        device const VALUE_T *x [[buffer(0)]], device const float *positions [[buffer(1)]],                            \
        device VALUE_T *out [[buffer(2)]], constant uint &seq_len [[buffer(3)]], constant uint &dim [[buffer(4)]],     \
        constant uint &total_seqs [[buffer(5)]], constant float *freqs [[buffer(6)]],                                  \
        constant float &attn_scale [[buffer(7)]], constant uint &rope_type [[buffer(8)]],                              \
        constant uint *shape [[buffer(9)]], constant size_t *x_strides [[buffer(10)]],                                 \
        constant size_t *out_strides [[buffer(11)]], constant uint &ndim [[buffer(12)]],                               \
        uint gid [[thread_position_in_grid]]                                                                           \
    ) {                                                                                                                \
        uint total_tokens = total_seqs * seq_len;                                                                      \
        if (gid >= total_tokens || ndim < 2) {                                                                         \
            return;                                                                                                    \
        }                                                                                                              \
        float pos = positions[gid];                                                                                    \
        size_t token = (size_t)gid;                                                                                    \
        size_t x_base = elem_to_loc<size_t>(token, shape, x_strides, ndim - 1u);                                       \
        size_t out_base = elem_to_loc<size_t>(token, shape, out_strides, ndim - 1u);                                   \
        size_t x_inner = x_strides[ndim - 1u];                                                                         \
        size_t out_inner = out_strides[ndim - 1u];                                                                     \
        uint half_dim = dim >> 1;                                                                                      \
        bool is_neox = rope_type != 0u;                                                                                \
        for (uint i = 0; i < half_dim; ++i) {                                                                          \
            uint even_offset = is_neox ? i : (i * 2u);                                                                 \
            uint odd_offset = is_neox ? (half_dim + i) : (even_offset + 1u);                                           \
            size_t x_even = x_base + ((size_t)even_offset * x_inner);                                                  \
            size_t x_odd = x_base + ((size_t)odd_offset * x_inner);                                                    \
            size_t out_even = out_base + ((size_t)even_offset * out_inner);                                            \
            size_t out_odd = out_base + ((size_t)odd_offset * out_inner);                                              \
            float freq = pos * freqs[i];                                                                               \
            float cos_freq = cos(freq) * attn_scale;                                                                   \
            float sin_freq = sin(freq) * attn_scale;                                                                   \
            float x0 = READ_FN(x[x_even]);                                                                             \
            float x1 = READ_FN(x[x_odd]);                                                                              \
            float out_even_val = x0 * cos_freq - x1 * sin_freq;                                                        \
            float out_odd_val = x0 * sin_freq + x1 * cos_freq;                                                         \
            out[out_even] = WRITE_FN(out_even_val);                                                                    \
            out[out_odd] = WRITE_FN(out_odd_val);                                                                      \
        }                                                                                                              \
    }

DEFINE_ROPE_KERNEL_STRIDED(rope_f32_g, float, read_float, write_float)
DEFINE_ROPE_KERNEL_STRIDED(rope_f16_g, half, read_half, write_half)
DEFINE_ROPE_KERNEL_STRIDED(rope_bf16_g, ushort, read_bf16, write_bf16)
#if MARMOT_ENABLE_FP8
DEFINE_ROPE_KERNEL_STRIDED(rope_fp8_e4m3_g, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_ROPE_KERNEL_STRIDED(rope_fp8_e5m2_g, uchar, read_fp8_e5m2, write_fp8_e5m2)
#endif

#undef DEFINE_ROPE_KERNEL_STRIDED

struct RopeQKUniforms {
    uint rows;
    uint dim;
    uint head_dim;
    uint apply_q;
    uint apply_k;
    uint rope_type;
    float attn_scale;
};

#define DEFINE_ROPE_QK_KERNEL(NAME, VALUE_T, READ_FN, WRITE_FN)                                                        \
    kernel void NAME(                                                                                                  \
        device VALUE_T *q [[buffer(0)]], device VALUE_T *k [[buffer(1)]], device const float *positions [[buffer(2)]], \
        device const float *freqs [[buffer(3)]], constant RopeQKUniforms &uniforms [[buffer(4)]],                      \
        uint gid [[thread_position_in_grid]]                                                                           \
    ) {                                                                                                                \
        uint head_dim = uniforms.head_dim;                                                                             \
        if (head_dim == 0u || head_dim > uniforms.dim || (uniforms.dim % head_dim) != 0u || (head_dim & 1u) != 0u) {   \
            head_dim = uniforms.dim;                                                                                   \
        }                                                                                                              \
        uint pairs = head_dim >> 1;                                                                                    \
        if (pairs == 0u) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        uint heads = uniforms.dim / head_dim;                                                                          \
        uint pairs_per_row = heads * pairs;                                                                            \
        uint total = uniforms.rows * pairs_per_row;                                                                    \
        if (gid >= total) {                                                                                            \
            return;                                                                                                    \
        }                                                                                                              \
        uint row = gid / pairs_per_row;                                                                                \
        uint pair_index = gid % pairs_per_row;                                                                         \
        uint head = pair_index / pairs;                                                                                \
        uint pair = pair_index % pairs;                                                                                \
        uint half_dim = head_dim >> 1;                                                                                 \
        float pos = positions[row];                                                                                    \
        float freq = freqs[pair];                                                                                      \
        float angle = pos * freq;                                                                                      \
        float cos_freq = cos(angle) * uniforms.attn_scale;                                                             \
        float sin_freq = sin(angle) * uniforms.attn_scale;                                                             \
        bool is_neox = uniforms.rope_type != 0u;                                                                       \
        uint head_base = head * head_dim;                                                                              \
        uint even_index = row * uniforms.dim + head_base + (is_neox ? pair : (pair * 2u));                             \
        uint odd_index = row * uniforms.dim + head_base + (is_neox ? (pair + half_dim) : (pair * 2u + 1u));            \
        if (uniforms.apply_q != 0u) {                                                                                  \
            float even_val = READ_FN(q[even_index]);                                                                   \
            float odd_val = READ_FN(q[odd_index]);                                                                     \
            float out_even = even_val * cos_freq - odd_val * sin_freq;                                                 \
            float out_odd = even_val * sin_freq + odd_val * cos_freq;                                                  \
            q[even_index] = WRITE_FN(out_even);                                                                        \
            q[odd_index] = WRITE_FN(out_odd);                                                                          \
        }                                                                                                              \
        if (uniforms.apply_k != 0u) {                                                                                  \
            float even_val = READ_FN(k[even_index]);                                                                   \
            float odd_val = READ_FN(k[odd_index]);                                                                     \
            float out_even = even_val * cos_freq - odd_val * sin_freq;                                                 \
            float out_odd = even_val * sin_freq + odd_val * cos_freq;                                                  \
            k[even_index] = WRITE_FN(out_even);                                                                        \
            k[odd_index] = WRITE_FN(out_odd);                                                                          \
        }                                                                                                              \
    }

DEFINE_ROPE_QK_KERNEL(rope_qk_f32, float, read_float, write_float)
DEFINE_ROPE_QK_KERNEL(rope_qk_f16, half, read_half, write_half)
DEFINE_ROPE_QK_KERNEL(rope_qk_bf16, ushort, read_bf16, write_bf16)

#undef DEFINE_ROPE_QK_KERNEL

// -----------------------------------------------------------------------------
// Paged attention (packed tokens + paged KV)
// -----------------------------------------------------------------------------

struct PagedAttentionUniforms {
    uint token_count;
    uint token_offset;
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint block_size;
    uint block_shift;
    uint layer_idx;
    uint gqa_group_size;
    uint max_blocks_per_seq;
    uint meta_stride0;
    uint meta_stride1;
    uint block_stride0;
    uint block_stride1;
    float scale;
};

struct PagedAttentionStrides {
    ulong q_stride0;
    ulong q_stride1;
    ulong q_stride2;
    ulong k_stride0;
    ulong k_stride1;
    ulong k_stride2;
    ulong v_stride0;
    ulong v_stride1;
    ulong v_stride2;
    ulong out_stride0;
    ulong out_stride1;
    ulong out_stride2;
    ulong kv_stride0;
    ulong kv_stride1;
    ulong kv_stride2;
    ulong kv_stride3;
    ulong kv_stride4;
};

#define DEFINE_PAGED_KV_SCATTER(NAME, VALUE_T)                                                                         \
    kernel void NAME(                                                                                                  \
        device const uint *token_meta [[buffer(0)]], device const VALUE_T *k_new [[buffer(1)]],                        \
        device const VALUE_T *v_new [[buffer(2)]], device VALUE_T *kv_k [[buffer(3)]],                                 \
        device VALUE_T *kv_v [[buffer(4)]], constant PagedAttentionUniforms &u [[buffer(5)]],                          \
        constant PagedAttentionStrides &s [[buffer(6)]], uint gid [[thread_position_in_grid]]                          \
    ) {                                                                                                                \
        if (u.token_count == 0u || u.num_kv_heads == 0u || u.head_dim == 0u) {                                         \
            return;                                                                                                    \
        }                                                                                                              \
        const uint total = u.token_count * u.num_kv_heads * u.head_dim;                                                \
        if (gid >= total) {                                                                                            \
            return;                                                                                                    \
        }                                                                                                              \
        const uint d = gid % u.head_dim;                                                                               \
        uint tmp = gid / u.head_dim;                                                                                   \
        const uint kv_head = tmp % u.num_kv_heads;                                                                     \
        const uint t = tmp / u.num_kv_heads;                                                                           \
        const uint token_index = t + u.token_offset;                                                                   \
        const size_t meta_base = (size_t)token_index * (size_t)u.meta_stride0;                                         \
        const uint kv_slot = token_meta[meta_base + (size_t)2u * u.meta_stride1];                                      \
        const uint block_id = kv_slot >> u.block_shift;                                                                \
        const uint offset = kv_slot & (u.block_size - 1u);                                                             \
        const size_t k_idx =                                                                                           \
            (size_t)token_index * s.k_stride0 + (size_t)kv_head * s.k_stride1 + (size_t)d * s.k_stride2;               \
        const size_t v_idx =                                                                                           \
            (size_t)token_index * s.v_stride0 + (size_t)kv_head * s.v_stride1 + (size_t)d * s.v_stride2;               \
        const size_t kv_idx = (size_t)block_id * s.kv_stride0 + (size_t)u.layer_idx * s.kv_stride1 +                   \
            (size_t)kv_head * s.kv_stride2 + (size_t)offset * s.kv_stride3 + (size_t)d * s.kv_stride4;                 \
        kv_k[kv_idx] = k_new[k_idx];                                                                                   \
        kv_v[kv_idx] = v_new[v_idx];                                                                                   \
    }

DEFINE_PAGED_KV_SCATTER(paged_kv_scatter_f32, float)
DEFINE_PAGED_KV_SCATTER(paged_kv_scatter_f16, half)
DEFINE_PAGED_KV_SCATTER(paged_kv_scatter_bf16, ushort)

#undef DEFINE_PAGED_KV_SCATTER

#define DEFINE_PAGED_KV_SCATTER_MIXED(NAME, IN_T, KV_T, READ_IN, WRITE_KV)                                             \
    kernel void NAME(                                                                                                  \
        device const uint *token_meta [[buffer(0)]], device const IN_T *k_new [[buffer(1)]],                           \
        device const IN_T *v_new [[buffer(2)]], device KV_T *kv_k [[buffer(3)]], device KV_T *kv_v [[buffer(4)]],      \
        constant PagedAttentionUniforms &u [[buffer(5)]], constant PagedAttentionStrides &s [[buffer(6)]],             \
        uint gid [[thread_position_in_grid]]                                                                           \
    ) {                                                                                                                \
        if (u.token_count == 0u || u.num_kv_heads == 0u || u.head_dim == 0u) {                                         \
            return;                                                                                                    \
        }                                                                                                              \
        const uint total = u.token_count * u.num_kv_heads * u.head_dim;                                                \
        if (gid >= total) {                                                                                            \
            return;                                                                                                    \
        }                                                                                                              \
        const uint d = gid % u.head_dim;                                                                               \
        uint tmp = gid / u.head_dim;                                                                                   \
        const uint kv_head = tmp % u.num_kv_heads;                                                                     \
        const uint t = tmp / u.num_kv_heads;                                                                           \
        const uint token_index = t + u.token_offset;                                                                   \
        const size_t meta_base = (size_t)token_index * (size_t)u.meta_stride0;                                         \
        const uint kv_slot = token_meta[meta_base + (size_t)2u * u.meta_stride1];                                      \
        const uint block_id = kv_slot >> u.block_shift;                                                                \
        const uint offset = kv_slot & (u.block_size - 1u);                                                             \
        const size_t k_idx =                                                                                           \
            (size_t)token_index * s.k_stride0 + (size_t)kv_head * s.k_stride1 + (size_t)d * s.k_stride2;               \
        const size_t v_idx =                                                                                           \
            (size_t)token_index * s.v_stride0 + (size_t)kv_head * s.v_stride1 + (size_t)d * s.v_stride2;               \
        const size_t kv_idx = (size_t)block_id * s.kv_stride0 + (size_t)u.layer_idx * s.kv_stride1 +                   \
            (size_t)kv_head * s.kv_stride2 + (size_t)offset * s.kv_stride3 + (size_t)d * s.kv_stride4;                 \
        kv_k[kv_idx] = WRITE_KV(READ_IN(k_new[k_idx]));                                                                \
        kv_v[kv_idx] = WRITE_KV(READ_IN(v_new[v_idx]));                                                                \
    }

DEFINE_PAGED_KV_SCATTER_MIXED(paged_kv_scatter_f32_kv_f16, float, half, read_float, write_half)
DEFINE_PAGED_KV_SCATTER_MIXED(paged_kv_scatter_f32_kv_bf16, float, ushort, read_float, write_bf16)
DEFINE_PAGED_KV_SCATTER_MIXED(paged_kv_scatter_f16_kv_f32, half, float, read_half, write_float)
DEFINE_PAGED_KV_SCATTER_MIXED(paged_kv_scatter_f16_kv_bf16, half, ushort, read_half, write_bf16)
DEFINE_PAGED_KV_SCATTER_MIXED(paged_kv_scatter_bf16_kv_f32, ushort, float, read_bf16, write_float)
DEFINE_PAGED_KV_SCATTER_MIXED(paged_kv_scatter_bf16_kv_f16, ushort, half, read_bf16, write_half)

#undef DEFINE_PAGED_KV_SCATTER_MIXED

#define PAGED_ATTN_SIMD_WIDTH 32
#define PAGED_ATTN_MAX_DIM 256
#define PAGED_ATTN_MAX_CHUNKS (PAGED_ATTN_MAX_DIM / PAGED_ATTN_SIMD_WIDTH)

#define DEFINE_PAGED_ATTENTION_KERNEL(NAME, VALUE_T, READ_FN, WRITE_FN, SIMDGROUPS)                                    \
    kernel void NAME(                                                                                                  \
        device const uint *token_meta [[buffer(0)]], device const uint *block_table [[buffer(1)]],                     \
        device const VALUE_T *q [[buffer(2)]], device const VALUE_T *kv_k [[buffer(3)]],                               \
        device const VALUE_T *kv_v [[buffer(4)]], device VALUE_T *out [[buffer(5)]],                                   \
        constant PagedAttentionUniforms &u [[buffer(6)]], constant PagedAttentionStrides &s [[buffer(7)]],             \
        uint3 gid [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],                      \
        uint simd_group [[simdgroup_index_in_threadgroup]]                                                             \
    ) {                                                                                                                \
        const uint token = gid.x;                                                                                      \
        const uint q_head = gid.y;                                                                                     \
        if (token >= u.token_count || q_head >= u.num_q_heads || u.head_dim == 0u ||                                   \
            u.head_dim > PAGED_ATTN_MAX_DIM) {                                                                         \
            return;                                                                                                    \
        }                                                                                                              \
        if (simd_group >= SIMDGROUPS) {                                                                                \
            return;                                                                                                    \
        }                                                                                                              \
        const uint token_index = token + u.token_offset;                                                               \
        const size_t meta_base = (size_t)token_index * (size_t)u.meta_stride0;                                         \
        const uint seq_slot = token_meta[meta_base + (size_t)0u * u.meta_stride1];                                     \
        const uint pos = token_meta[meta_base + (size_t)1u * u.meta_stride1];                                          \
        const uint kv_len = pos + 1u;                                                                                  \
        const uint gqa_group = (u.gqa_group_size == 0u) ? 1u : u.gqa_group_size;                                       \
        const uint kv_head = (gqa_group == 1u) ? q_head : (q_head / gqa_group);                                        \
        const size_t q_base = (size_t)token_index * s.q_stride0 + (size_t)q_head * s.q_stride1;                        \
        const size_t out_base = (size_t)token_index * s.out_stride0 + (size_t)q_head * s.out_stride1;                  \
        const uint chunks = (u.head_dim + PAGED_ATTN_SIMD_WIDTH - 1u) / PAGED_ATTN_SIMD_WIDTH;                         \
        float q_local[PAGED_ATTN_MAX_CHUNKS];                                                                          \
        float o_acc[PAGED_ATTN_MAX_CHUNKS];                                                                            \
        for (uint c = 0; c < chunks; ++c) {                                                                            \
            uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                            \
            q_local[c] = d < u.head_dim ? (READ_FN(q[q_base + (size_t)d * s.q_stride2]) * u.scale) : 0.0f;             \
            o_acc[c] = 0.0f;                                                                                           \
        }                                                                                                              \
        float max_score = -INFINITY;                                                                                   \
        float sum_exp = 0.0f;                                                                                          \
        const uint block_mask = u.block_size - 1u;                                                                     \
        for (uint p = simd_group; p < kv_len; p += SIMDGROUPS) {                                                       \
            const uint logical_block = p >> u.block_shift;                                                             \
            if (logical_block >= u.max_blocks_per_seq) {                                                               \
                continue;                                                                                              \
            }                                                                                                          \
            const size_t block_index =                                                                                 \
                (size_t)seq_slot * (size_t)u.block_stride0 + (size_t)logical_block * (size_t)u.block_stride1;          \
            const uint block_id = block_table[block_index];                                                            \
            if (block_id == 0xFFFFFFFFu) {                                                                             \
                continue;                                                                                              \
            }                                                                                                          \
            const uint offset = p & block_mask;                                                                        \
            const size_t kv_base = (size_t)block_id * s.kv_stride0 + (size_t)u.layer_idx * s.kv_stride1 +              \
                (size_t)kv_head * s.kv_stride2 + (size_t)offset * s.kv_stride3;                                        \
            float partial = 0.0f;                                                                                      \
            for (uint c = 0; c < chunks; ++c) {                                                                        \
                uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                        \
                if (d < u.head_dim) {                                                                                  \
                    partial += q_local[c] * READ_FN(kv_k[kv_base + (size_t)d * s.kv_stride4]);                         \
                }                                                                                                      \
            }                                                                                                          \
            float logit = simd_sum(partial);                                                                           \
            float next_max = max(max_score, logit);                                                                    \
            float exp_prev = exp(max_score - next_max);                                                                \
            float exp_logit = exp(logit - next_max);                                                                   \
            sum_exp = sum_exp * exp_prev + exp_logit;                                                                  \
            for (uint c = 0; c < chunks; ++c) {                                                                        \
                uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                        \
                if (d < u.head_dim) {                                                                                  \
                    float v_val = READ_FN(kv_v[kv_base + (size_t)d * s.kv_stride4]);                                   \
                    o_acc[c] = o_acc[c] * exp_prev + exp_logit * v_val;                                                \
                }                                                                                                      \
            }                                                                                                          \
            max_score = next_max;                                                                                      \
        }                                                                                                              \
        threadgroup float max_scores[SIMDGROUPS];                                                                      \
        threadgroup float sum_scores[SIMDGROUPS];                                                                      \
        threadgroup float outputs[SIMDGROUPS * PAGED_ATTN_SIMD_WIDTH];                                                 \
        if (simd_lane == 0u) {                                                                                         \
            max_scores[simd_group] = max_score;                                                                        \
            sum_scores[simd_group] = sum_exp;                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        float lane_max = (simd_lane < SIMDGROUPS) ? max_scores[simd_lane] : -INFINITY;                                 \
        float global_max = simd_max(lane_max);                                                                         \
        if (global_max == -INFINITY) {                                                                                 \
            if (simd_group == 0u) {                                                                                    \
                for (uint c = 0; c < chunks; ++c) {                                                                    \
                    uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                    \
                    if (d < u.head_dim) {                                                                              \
                        out[out_base + (size_t)d * s.out_stride2] = WRITE_FN(0.0f);                                    \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
        float lane_sum = (simd_lane < SIMDGROUPS) ? sum_scores[simd_lane] * exp(lane_max - global_max) : 0.0f;         \
        float global_sum = simd_sum(lane_sum);                                                                         \
        float inv_sum = global_sum > 0.0f ? (1.0f / global_sum) : 0.0f;                                                \
        float corr = exp(max_score - global_max);                                                                      \
        for (uint c = 0; c < chunks; ++c) {                                                                            \
            uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                            \
            outputs[simd_group * PAGED_ATTN_SIMD_WIDTH + simd_lane] = (d < u.head_dim) ? (o_acc[c] * corr) : 0.0f;     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (simd_group == 0u && d < u.head_dim) {                                                                  \
                float out_val = 0.0f;                                                                                  \
                for (uint g = 0; g < SIMDGROUPS; ++g) {                                                                \
                    out_val += outputs[g * PAGED_ATTN_SIMD_WIDTH + simd_lane];                                         \
                }                                                                                                      \
                out[out_base + (size_t)d * s.out_stride2] = WRITE_FN(out_val * inv_sum);                               \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
    }

DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_sg4_f32, float, read_float, write_float, 4)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_sg8_f32, float, read_float, write_float, 8)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_sg4_f16, half, read_half, write_half, 4)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_sg8_f16, half, read_half, write_half, 8)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_sg4_bf16, ushort, read_bf16, write_bf16, 4)
DEFINE_PAGED_ATTENTION_KERNEL(paged_attention_sg8_bf16, ushort, read_bf16, write_bf16, 8)

#define DEFINE_PAGED_ATTENTION_KERNEL_MIXED(NAME, Q_T, KV_T, READ_Q, READ_KV, WRITE_OUT, SIMDGROUPS)                   \
    kernel void NAME(                                                                                                  \
        device const uint *token_meta [[buffer(0)]], device const uint *block_table [[buffer(1)]],                     \
        device const Q_T *q [[buffer(2)]], device const KV_T *kv_k [[buffer(3)]],                                      \
        device const KV_T *kv_v [[buffer(4)]], device Q_T *out [[buffer(5)]],                                          \
        constant PagedAttentionUniforms &u [[buffer(6)]], constant PagedAttentionStrides &s [[buffer(7)]],             \
        uint3 gid [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],                      \
        uint simd_group [[simdgroup_index_in_threadgroup]]                                                             \
    ) {                                                                                                                \
        const uint token = gid.x;                                                                                      \
        const uint q_head = gid.y;                                                                                     \
        if (token >= u.token_count || q_head >= u.num_q_heads || u.head_dim == 0u ||                                   \
            u.head_dim > PAGED_ATTN_MAX_DIM) {                                                                         \
            return;                                                                                                    \
        }                                                                                                              \
        if (simd_group >= SIMDGROUPS) {                                                                                \
            return;                                                                                                    \
        }                                                                                                              \
        const uint token_index = token + u.token_offset;                                                               \
        const size_t meta_base = (size_t)token_index * (size_t)u.meta_stride0;                                         \
        const uint seq_slot = token_meta[meta_base + (size_t)0u * u.meta_stride1];                                     \
        const uint pos = token_meta[meta_base + (size_t)1u * u.meta_stride1];                                          \
        const uint kv_len = pos + 1u;                                                                                  \
        const uint gqa_group = (u.gqa_group_size == 0u) ? 1u : u.gqa_group_size;                                       \
        const uint kv_head = (gqa_group == 1u) ? q_head : (q_head / gqa_group);                                        \
        const size_t q_base = (size_t)token_index * s.q_stride0 + (size_t)q_head * s.q_stride1;                        \
        const size_t out_base = (size_t)token_index * s.out_stride0 + (size_t)q_head * s.out_stride1;                  \
        const uint chunks = (u.head_dim + PAGED_ATTN_SIMD_WIDTH - 1u) / PAGED_ATTN_SIMD_WIDTH;                         \
        float q_local[PAGED_ATTN_MAX_CHUNKS];                                                                          \
        float o_acc[PAGED_ATTN_MAX_CHUNKS];                                                                            \
        for (uint c = 0; c < chunks; ++c) {                                                                            \
            uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                            \
            q_local[c] = d < u.head_dim ? (READ_Q(q[q_base + (size_t)d * s.q_stride2]) * u.scale) : 0.0f;              \
            o_acc[c] = 0.0f;                                                                                           \
        }                                                                                                              \
        float max_score = -INFINITY;                                                                                   \
        float sum_exp = 0.0f;                                                                                          \
        const uint block_mask = u.block_size - 1u;                                                                     \
        for (uint p = simd_group; p < kv_len; p += SIMDGROUPS) {                                                       \
            const uint logical_block = p >> u.block_shift;                                                             \
            if (logical_block >= u.max_blocks_per_seq) {                                                               \
                continue;                                                                                              \
            }                                                                                                          \
            const size_t block_index =                                                                                 \
                (size_t)seq_slot * (size_t)u.block_stride0 + (size_t)logical_block * (size_t)u.block_stride1;          \
            const uint block_id = block_table[block_index];                                                            \
            if (block_id == 0xFFFFFFFFu) {                                                                             \
                continue;                                                                                              \
            }                                                                                                          \
            const uint offset = p & block_mask;                                                                        \
            const size_t kv_base = (size_t)block_id * s.kv_stride0 + (size_t)u.layer_idx * s.kv_stride1 +              \
                (size_t)kv_head * s.kv_stride2 + (size_t)offset * s.kv_stride3;                                        \
            float partial = 0.0f;                                                                                      \
            for (uint c = 0; c < chunks; ++c) {                                                                        \
                uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                        \
                if (d < u.head_dim) {                                                                                  \
                    partial += q_local[c] * READ_KV(kv_k[kv_base + (size_t)d * s.kv_stride4]);                         \
                }                                                                                                      \
            }                                                                                                          \
            float logit = simd_sum(partial);                                                                           \
            float next_max = max(max_score, logit);                                                                    \
            float exp_prev = exp(max_score - next_max);                                                                \
            float exp_logit = exp(logit - next_max);                                                                   \
            sum_exp = sum_exp * exp_prev + exp_logit;                                                                  \
            for (uint c = 0; c < chunks; ++c) {                                                                        \
                uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                        \
                if (d < u.head_dim) {                                                                                  \
                    float v_val = READ_KV(kv_v[kv_base + (size_t)d * s.kv_stride4]);                                   \
                    o_acc[c] = o_acc[c] * exp_prev + exp_logit * v_val;                                                \
                }                                                                                                      \
            }                                                                                                          \
            max_score = next_max;                                                                                      \
        }                                                                                                              \
        threadgroup float max_scores[SIMDGROUPS];                                                                      \
        threadgroup float sum_scores[SIMDGROUPS];                                                                      \
        threadgroup float outputs[SIMDGROUPS * PAGED_ATTN_SIMD_WIDTH];                                                 \
        if (simd_lane == 0u) {                                                                                         \
            max_scores[simd_group] = max_score;                                                                        \
            sum_scores[simd_group] = sum_exp;                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        float lane_max = (simd_lane < SIMDGROUPS) ? max_scores[simd_lane] : -INFINITY;                                 \
        float global_max = simd_max(lane_max);                                                                         \
        if (global_max == -INFINITY) {                                                                                 \
            if (simd_group == 0u) {                                                                                    \
                for (uint c = 0; c < chunks; ++c) {                                                                    \
                    uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                    \
                    if (d < u.head_dim) {                                                                              \
                        out[out_base + (size_t)d * s.out_stride2] = WRITE_OUT(0.0f);                                   \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
        float lane_sum = (simd_lane < SIMDGROUPS) ? sum_scores[simd_lane] * exp(lane_max - global_max) : 0.0f;         \
        float global_sum = simd_sum(lane_sum);                                                                         \
        float inv_sum = global_sum > 0.0f ? (1.0f / global_sum) : 0.0f;                                                \
        float corr = exp(max_score - global_max);                                                                      \
        for (uint c = 0; c < chunks; ++c) {                                                                            \
            uint d = simd_lane + c * PAGED_ATTN_SIMD_WIDTH;                                                            \
            outputs[simd_group * PAGED_ATTN_SIMD_WIDTH + simd_lane] = (d < u.head_dim) ? (o_acc[c] * corr) : 0.0f;     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (simd_group == 0u && d < u.head_dim) {                                                                  \
                float out_val = 0.0f;                                                                                  \
                for (uint g = 0; g < SIMDGROUPS; ++g) {                                                                \
                    out_val += outputs[g * PAGED_ATTN_SIMD_WIDTH + simd_lane];                                         \
                }                                                                                                      \
                out[out_base + (size_t)d * s.out_stride2] = WRITE_OUT(out_val * inv_sum);                              \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
    }

DEFINE_PAGED_ATTENTION_KERNEL_MIXED(paged_attention_sg4_f32_kv_f16, float, half, read_float, read_half, write_float, 4)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(paged_attention_sg8_f32_kv_f16, float, half, read_float, read_half, write_float, 8)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(
    paged_attention_sg4_f32_kv_bf16, float, ushort, read_float, read_bf16, write_float, 4
)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(
    paged_attention_sg8_f32_kv_bf16, float, ushort, read_float, read_bf16, write_float, 8
)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(paged_attention_sg4_f16_kv_f32, half, float, read_half, read_float, write_half, 4)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(paged_attention_sg8_f16_kv_f32, half, float, read_half, read_float, write_half, 8)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(paged_attention_sg4_f16_kv_bf16, half, ushort, read_half, read_bf16, write_half, 4)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(paged_attention_sg8_f16_kv_bf16, half, ushort, read_half, read_bf16, write_half, 8)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(
    paged_attention_sg4_bf16_kv_f32, ushort, float, read_bf16, read_float, write_bf16, 4
)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(
    paged_attention_sg8_bf16_kv_f32, ushort, float, read_bf16, read_float, write_bf16, 8
)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(paged_attention_sg4_bf16_kv_f16, ushort, half, read_bf16, read_half, write_bf16, 4)
DEFINE_PAGED_ATTENTION_KERNEL_MIXED(paged_attention_sg8_bf16_kv_f16, ushort, half, read_bf16, read_half, write_bf16, 8)

#undef DEFINE_PAGED_ATTENTION_KERNEL_MIXED

#undef DEFINE_PAGED_ATTENTION_KERNEL
#undef PAGED_ATTN_MAX_CHUNKS
#undef PAGED_ATTN_MAX_DIM
#undef PAGED_ATTN_SIMD_WIDTH

// -----------------------------------------------------------------------------
// Flash Attention (simdgroup_matrix-based, paged KV cache)
// -----------------------------------------------------------------------------

constant uint FLASH_ATTN_NSG [[function_constant(0)]];

constexpr constant ushort MAX_Q_ROWS = 8;

template <
    ushort DK, ushort Q_ROWS, ushort C_PER_SG, typename KV_T, float (*READ_KV)(KV_T), typename Q_T,
    float (*READ_Q)(Q_T), typename OUT_T, OUT_T (*WRITE_OUT)(float)>
void paged_flash_attention_impl(
    device const uint *token_meta, device const uint *block_table, device const Q_T *q, device const KV_T *kv_k,
    device const KV_T *kv_v, device OUT_T *out, constant PagedAttentionUniforms &u, constant PagedAttentionStrides &s,
    uint3 tgid [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]], threadgroup half *shared_mem
) {
    const uint q_block_idx = tgid.x;
    const uint q_head = tgid.y;

    if (q_head >= u.num_q_heads)
        return;

    const uint q_start = q_block_idx * Q_ROWS;
    if (q_start >= u.token_count)
        return;

    const uint block_m = min((uint)Q_ROWS, u.token_count - q_start);
    const uint gqa_group = (u.gqa_group_size == 0u) ? 1u : u.gqa_group_size;
    const uint kv_head = (gqa_group == 1u) ? q_head : (q_head / gqa_group);

    // --- Threadgroup memory layout ---
    // sq:       Q_ROWS * DK halfs (query tile, shared across simdgroups)
    // ss:       NSG * Q_ROWS * C_PER_SG floats (attention scores, per-simdgroup)
    // sk:       NSG * C_PER_SG * DK halfs (KV scratch, per-simdgroup, reused for K and V)
    // sm/sl:    NSG * Q_ROWS floats each (per-simdgroup running max/sum)
    // row_pos:  Q_ROWS uints
    threadgroup half *sq = shared_mem;
    threadgroup float *ss_all = (threadgroup float *)(sq + Q_ROWS * DK);
    threadgroup half *sk_all = (threadgroup half *)(ss_all + FLASH_ATTN_NSG * Q_ROWS * C_PER_SG);
    threadgroup float *sm = (threadgroup float *)(sk_all + FLASH_ATTN_NSG * C_PER_SG * DK);
    threadgroup float *sl = sm + FLASH_ATTN_NSG * Q_ROWS;

    // Per-simdgroup pointers
    threadgroup float *ss = ss_all + sgitg * Q_ROWS * C_PER_SG;
    threadgroup half *sk = sk_all + sgitg * C_PER_SG * DK;

    // Shared metadata
    threadgroup uint *row_pos_shared = (threadgroup uint *)(sl + FLASH_ATTN_NSG * Q_ROWS);
    threadgroup uint *seq_slot_ptr = row_pos_shared + Q_ROWS;
    threadgroup uint *kv_len_max_ptr = seq_slot_ptr + 1;

    // Per-thread output accumulator in registers
    // Each thread handles dimensions tiisg, tiisg+32, tiisg+64, tiisg+96 (for DK=128)
    constexpr ushort D_PER_THREAD = (DK + 31) / 32;
    float o_reg[MAX_Q_ROWS * D_PER_THREAD];
    float m_reg[MAX_Q_ROWS];
    float l_reg[MAX_Q_ROWS];
    for (ushort r = 0; r < Q_ROWS; ++r) {
        m_reg[r] = -INFINITY;
        l_reg[r] = 0.0f;
        for (ushort di = 0; di < D_PER_THREAD; ++di) {
            o_reg[r * D_PER_THREAD + di] = 0.0f;
        }
    }

    // --- Phase 0: Load metadata ---
    if (sgitg == 0 && tiisg < Q_ROWS) {
        if (tiisg < block_m) {
            const uint token_index = u.token_offset + q_start + tiisg;
            const ulong meta_base = (ulong)token_index * (ulong)u.meta_stride0;
            row_pos_shared[tiisg] = token_meta[meta_base + (ulong)1u * u.meta_stride1];
            if (tiisg == 0u) {
                *seq_slot_ptr = token_meta[meta_base + (ulong)0u * u.meta_stride1];
            }
        } else {
            row_pos_shared[tiisg] = 0u;
            if (tiisg == 0u) {
                *seq_slot_ptr = 0u;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0 && tiisg == 0) {
        uint max_pos = 0u;
        for (uint r = 0; r < block_m; ++r) {
            max_pos = max(max_pos, row_pos_shared[r]);
        }
        *kv_len_max_ptr = max_pos + 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint seq_slot = *seq_slot_ptr;
    const uint kv_len = *kv_len_max_ptr;
    const uint block_mask = u.block_size - 1u;

    // --- Phase 1: Load Q into shared memory as half with scale ---
    for (uint r = sgitg; r < block_m; r += FLASH_ATTN_NSG) {
        const uint token_index = u.token_offset + q_start + r;
        const ulong q_base = (ulong)token_index * s.q_stride0 + (ulong)q_head * s.q_stride1;
        for (uint d = tiisg; d < DK; d += 32) {
            float qv = READ_Q(q[q_base + (ulong)d * s.q_stride2]) * u.scale;
            sq[r * DK + d] = (half)qv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Phase 2: KV block loop ---
    const uint total_c = C_PER_SG * FLASH_ATTN_NSG;
    const uint num_kv_iters = (kv_len + total_c - 1u) / total_c;

    for (uint kv_iter = 0; kv_iter < num_kv_iters; ++kv_iter) {
        const uint kv_base_pos = kv_iter * total_c + sgitg * C_PER_SG;
        const uint c_count = min((uint)C_PER_SG, kv_len > kv_base_pos ? kv_len - kv_base_pos : 0u);

        // Load K into sk (per-simdgroup buffer)
        for (uint ci = 0; ci < c_count; ++ci) {
            const uint global_kv_pos = kv_base_pos + ci;
            const uint logical_block = global_kv_pos >> u.block_shift;
            uint block_id = 0xFFFFFFFFu;
            if (logical_block < u.max_blocks_per_seq) {
                const ulong block_index =
                    (ulong)seq_slot * (ulong)u.block_stride0 + (ulong)logical_block * (ulong)u.block_stride1;
                block_id = block_table[block_index];
            }
            if (block_id != 0xFFFFFFFFu) {
                const uint offset = global_kv_pos & block_mask;
                const ulong kv_addr = (ulong)block_id * s.kv_stride0 + (ulong)u.layer_idx * s.kv_stride1 +
                    (ulong)kv_head * s.kv_stride2 + (ulong)offset * s.kv_stride3;
                for (uint d = tiisg; d < DK; d += 32) {
                    sk[ci * DK + d] = (half)READ_KV(kv_k[kv_addr + (ulong)d * s.kv_stride4]);
                }
            } else {
                for (uint d = tiisg; d < DK; d += 32) {
                    sk[ci * DK + d] = 0.0h;
                }
            }
        }
        for (uint ci = c_count; ci < C_PER_SG; ++ci) {
            for (uint d = tiisg; d < DK; d += 32) {
                sk[ci * DK + d] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // QK^T via simdgroup_matrix for each 8-row chunk of C
        for (uint cc = 0; cc < C_PER_SG; cc += 8) {
            simdgroup_float8x8 mqk(0);

            for (ushort dk = 0; dk < DK; dk += 8) {
                simdgroup_half8x8 mq;
                simdgroup_half8x8 mk;
                simdgroup_load(mq, sq + dk, DK);
                simdgroup_load(mk, sk + cc * DK + dk, DK, 0, true);
                simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
            }

            simdgroup_store(mqk, ss + cc, C_PER_SG);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Online softmax + rescale output registers
        for (uint r = 0; r < Q_ROWS && r < block_m; ++r) {
            float row_max_prev = m_reg[r];
            float row_sum_prev = l_reg[r];

            float local_max = -INFINITY;
            for (uint c = tiisg; c < c_count; c += 32) {
                uint global_kv_pos = kv_base_pos + c;
                float score = ss[r * C_PER_SG + c];
                if (global_kv_pos <= row_pos_shared[r]) {
                    local_max = max(local_max, score);
                }
            }
            float block_max_val = simd_max(local_max);

            float new_max = max(row_max_prev, block_max_val);
            float scale_prev = (row_max_prev == -INFINITY) ? 0.0f : exp(row_max_prev - new_max);
            float new_sum = row_sum_prev * scale_prev;

            float local_exp_sum = 0.0f;
            for (uint c = tiisg; c < c_count; c += 32) {
                uint global_kv_pos = kv_base_pos + c;
                float score = ss[r * C_PER_SG + c];
                float w = 0.0f;
                if (global_kv_pos <= row_pos_shared[r] && new_max != -INFINITY) {
                    w = exp(score - new_max);
                }
                ss[r * C_PER_SG + c] = w;
                local_exp_sum += w;
            }
            new_sum += simd_sum(local_exp_sum);

            // Rescale previous output accumulator in registers
            for (ushort di = 0; di < D_PER_THREAD; ++di) {
                o_reg[r * D_PER_THREAD + di] *= scale_prev;
            }

            m_reg[r] = new_max;
            l_reg[r] = new_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Load V into sk (reuse K scratch)
        for (uint ci = 0; ci < c_count; ++ci) {
            const uint global_kv_pos = kv_base_pos + ci;
            const uint logical_block = global_kv_pos >> u.block_shift;
            uint block_id = 0xFFFFFFFFu;
            if (logical_block < u.max_blocks_per_seq) {
                const ulong block_index =
                    (ulong)seq_slot * (ulong)u.block_stride0 + (ulong)logical_block * (ulong)u.block_stride1;
                block_id = block_table[block_index];
            }
            if (block_id != 0xFFFFFFFFu) {
                const uint offset = global_kv_pos & block_mask;
                const ulong kv_addr = (ulong)block_id * s.kv_stride0 + (ulong)u.layer_idx * s.kv_stride1 +
                    (ulong)kv_head * s.kv_stride2 + (ulong)offset * s.kv_stride3;
                for (uint d = tiisg; d < DK; d += 32) {
                    sk[ci * DK + d] = (half)READ_KV(kv_v[kv_addr + (ulong)d * s.kv_stride4]);
                }
            } else {
                for (uint d = tiisg; d < DK; d += 32) {
                    sk[ci * DK + d] = 0.0h;
                }
            }
        }
        for (uint ci = c_count; ci < C_PER_SG; ++ci) {
            for (uint d = tiisg; d < DK; d += 32) {
                sk[ci * DK + d] = 0.0h;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S × V: per-thread scalar accumulation into registers
        for (uint r = 0; r < Q_ROWS && r < block_m; ++r) {
            for (uint c = 0; c < c_count; ++c) {
                float w = ss[r * C_PER_SG + c];
                for (ushort di = 0; di < D_PER_THREAD; ++di) {
                    uint d = tiisg + di * 32;
                    o_reg[r * D_PER_THREAD + di] += w * (float)sk[c * DK + d];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // --- Phase 3: Write output ---
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (FLASH_ATTN_NSG == 1) {
        if (sgitg == 0) {
            for (uint r = 0; r < block_m; ++r) {
                const uint token_index = u.token_offset + q_start + r;
                const ulong out_base = (ulong)token_index * s.out_stride0 + (ulong)q_head * s.out_stride1;
                float inv_sum = l_reg[r] > 0.0f ? (1.0f / l_reg[r]) : 0.0f;
                for (ushort di = 0; di < D_PER_THREAD; ++di) {
                    uint d = tiisg + di * 32;
                    out[out_base + (ulong)d * s.out_stride2] = WRITE_OUT(o_reg[r * D_PER_THREAD + di] * inv_sum);
                }
            }
        }
    } else {
        // Cross-simdgroup reduction using threadgroup memory
        // Reuse sk area (large enough: C_PER_SG * DK * 2 >= Q_ROWS * DK * 4 when C_PER_SG >= Q_ROWS*2)
        // Actually, use ss area + sk area as a flat float buffer for reduction
        // We need NSG * Q_ROWS * DK floats, but we have limited tg memory.
        // Instead, reduce one row at a time using ss (Q_ROWS * C_PER_SG floats >= DK floats when C_PER_SG >= DK/Q_ROWS)

        // Write per-simdgroup max/sum to shared memory
        if (tiisg < Q_ROWS) {
            sm[sgitg * Q_ROWS + tiisg] = m_reg[tiisg];
            sl[sgitg * Q_ROWS + tiisg] = l_reg[tiisg];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // For each row, each simdgroup writes its partial output to a shared buffer,
        // then sg0 reduces and writes final output
        // Reuse the ss buffer (Q_ROWS * C_PER_SG floats = 8*32 = 256 floats)
        // We need DK floats per simdgroup = 128 or 64 floats
        // Process one simdgroup at a time
        threadgroup float *reduce_buf = ss_all; // Reuse ss_all for reduction

        for (uint r = 0; r < block_m; ++r) {
            // First, compute the global max for this row
            float global_max = -INFINITY;
            if (sgitg == 0) {
                for (uint sg = 0; sg < FLASH_ATTN_NSG; ++sg) {
                    global_max = max(global_max, sm[sg * Q_ROWS + r]);
                }
            }
            global_max = simd_broadcast_first(global_max);
            // Broadcast from sg0 - all threads in sg0 have it, others need it via shared mem
            if (sgitg == 0 && tiisg == 0) {
                reduce_buf[0] = global_max;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            global_max = reduce_buf[0];
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each simdgroup writes its corrected output for this row
            float sg_max = m_reg[r];
            float corr = (sg_max == -INFINITY) ? 0.0f : exp(sg_max - global_max);

            // Write corrected partial output to reduce_buf at sgitg offset
            for (ushort di = 0; di < D_PER_THREAD; ++di) {
                uint d = tiisg + di * 32;
                reduce_buf[sgitg * DK + d] = o_reg[r * D_PER_THREAD + di] * corr;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // sg0 reduces across all simdgroups and writes output
            if (sgitg == 0) {
                float global_sum = 0.0f;
                for (uint sg = 0; sg < FLASH_ATTN_NSG; ++sg) {
                    float sg_m = sm[sg * Q_ROWS + r];
                    float sg_c = (sg_m == -INFINITY) ? 0.0f : exp(sg_m - global_max);
                    global_sum += sl[sg * Q_ROWS + r] * sg_c;
                }
                float inv_sum = global_sum > 0.0f ? (1.0f / global_sum) : 0.0f;

                const uint token_index = u.token_offset + q_start + r;
                const ulong out_base = (ulong)token_index * s.out_stride0 + (ulong)q_head * s.out_stride1;

                for (ushort di = 0; di < D_PER_THREAD; ++di) {
                    uint d = tiisg + di * 32;
                    float val = 0.0f;
                    for (uint sg = 0; sg < FLASH_ATTN_NSG; ++sg) {
                        val += reduce_buf[sg * DK + d];
                    }
                    out[out_base + (ulong)d * s.out_stride2] = WRITE_OUT(val * inv_sum);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

// --- Kernel entry points ---

#define DEFINE_PAGED_FLASH_ATTENTION_KERNEL(                                                                           \
    NAME, DK_VAL, Q_ROWS_VAL, C_VAL, KV_T, READ_KV_FN, Q_T, READ_Q_FN, OUT_T, WRITE_OUT_FN                             \
)                                                                                                                      \
    kernel void NAME(                                                                                                  \
        device const uint *token_meta [[buffer(0)]], device const uint *block_table [[buffer(1)]],                     \
        device const Q_T *q [[buffer(2)]], device const KV_T *kv_k [[buffer(3)]],                                      \
        device const KV_T *kv_v [[buffer(4)]], device OUT_T *out [[buffer(5)]],                                        \
        constant PagedAttentionUniforms &u [[buffer(6)]], constant PagedAttentionStrides &s [[buffer(7)]],             \
        uint3 tgid [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],                       \
        ushort sgitg [[simdgroup_index_in_threadgroup]], threadgroup half *shared_mem [[threadgroup(0)]]               \
    ) {                                                                                                                \
        paged_flash_attention_impl<DK_VAL, Q_ROWS_VAL, C_VAL, KV_T, READ_KV_FN, Q_T, READ_Q_FN, OUT_T, WRITE_OUT_FN>(  \
            token_meta, block_table, q, kv_k, kv_v, out, u, s, tgid, tiisg, sgitg, shared_mem                          \
        );                                                                                                             \
    }

// f32 Q/KV/out, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_128, 128, 8, 32, float, read_float, float, read_float, float, write_float
)

// f16 Q/KV/out, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_128, 128, 8, 32, half, read_half, half, read_half, half, write_half
)

// bf16 Q/KV/out, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_128, 128, 8, 32, ushort, read_bf16, ushort, read_bf16, ushort, write_bf16
)

// f32 Q/out, f16 KV, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_kv_f16_128, 128, 8, 32, half, read_half, float, read_float, float, write_float
)

// f32 Q/out, bf16 KV, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_kv_bf16_128, 128, 8, 32, ushort, read_bf16, float, read_float, float, write_float
)

// f16 Q/out, f32 KV, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_kv_f32_128, 128, 8, 32, float, read_float, half, read_half, half, write_half
)

// f16 Q/out, bf16 KV, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_kv_bf16_128, 128, 8, 32, ushort, read_bf16, half, read_half, half, write_half
)

// bf16 Q/out, f32 KV, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_kv_f32_128, 128, 8, 32, float, read_float, ushort, read_bf16, ushort, write_bf16
)

// bf16 Q/out, f16 KV, DK=128
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_kv_f16_128, 128, 8, 32, half, read_half, ushort, read_bf16, ushort, write_bf16
)

// DK=128, C=64 variants (used with NSG=1 for larger KV sequences)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_128_c64, 128, 8, 64, float, read_float, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_128_c64, 128, 8, 64, half, read_half, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_128_c64, 128, 8, 64, ushort, read_bf16, ushort, read_bf16, ushort, write_bf16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_kv_f16_128_c64, 128, 8, 64, half, read_half, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_kv_bf16_128_c64, 128, 8, 64, ushort, read_bf16, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_kv_f32_128_c64, 128, 8, 64, float, read_float, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_kv_bf16_128_c64, 128, 8, 64, ushort, read_bf16, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_kv_f32_128_c64, 128, 8, 64, float, read_float, ushort, read_bf16, ushort, write_bf16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_kv_f16_128_c64, 128, 8, 64, half, read_half, ushort, read_bf16, ushort, write_bf16
)

// DK=64, C=64 variants
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_64_c64, 64, 8, 64, float, read_float, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_64_c64, 64, 8, 64, half, read_half, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_64_c64, 64, 8, 64, ushort, read_bf16, ushort, read_bf16, ushort, write_bf16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_kv_f16_64_c64, 64, 8, 64, half, read_half, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_kv_bf16_64_c64, 64, 8, 64, ushort, read_bf16, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_kv_f32_64_c64, 64, 8, 64, float, read_float, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_kv_bf16_64_c64, 64, 8, 64, ushort, read_bf16, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_kv_f32_64_c64, 64, 8, 64, float, read_float, ushort, read_bf16, ushort, write_bf16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_kv_f16_64_c64, 64, 8, 64, half, read_half, ushort, read_bf16, ushort, write_bf16
)

// DK=64 variants
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_64, 64, 8, 32, float, read_float, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_64, 64, 8, 32, half, read_half, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_64, 64, 8, 32, ushort, read_bf16, ushort, read_bf16, ushort, write_bf16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_kv_f16_64, 64, 8, 32, half, read_half, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_kv_bf16_64, 64, 8, 32, ushort, read_bf16, float, read_float, float, write_float
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_kv_f32_64, 64, 8, 32, float, read_float, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_kv_bf16_64, 64, 8, 32, ushort, read_bf16, half, read_half, half, write_half
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_kv_f32_64, 64, 8, 32, float, read_float, ushort, read_bf16, ushort, write_bf16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_kv_f16_64, 64, 8, 32, half, read_half, ushort, read_bf16, ushort, write_bf16
)

#undef DEFINE_PAGED_FLASH_ATTENTION_KERNEL
