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
// Flash Attention (tiled, memory-efficient scaled dot-product attention)
// -----------------------------------------------------------------------------

#define FLASH_ATTN_BLOCK_M_F32 8
#define FLASH_ATTN_BLOCK_N_F32 32
#define FLASH_ATTN_MAX_DIM_F32 128
#define FLASH_ATTN_BLOCK_M_F32_NARROW 8
#define FLASH_ATTN_BLOCK_N_F32_NARROW 16
#define FLASH_ATTN_BLOCK_M_F32_WIDE 4
#define FLASH_ATTN_BLOCK_N_F32_WIDE 16
#define FLASH_ATTN_MAX_DIM_F32_WIDE 256
#define FLASH_ATTN_BLOCK_M_F32_WIDE_NARROW 4
#define FLASH_ATTN_BLOCK_N_F32_WIDE_NARROW 8
#define FLASH_ATTN_BLOCK_M_F16 16
#define FLASH_ATTN_BLOCK_N_F16 32
#define FLASH_ATTN_MAX_DIM_F16 128
#define FLASH_ATTN_BLOCK_M_F16_NARROW 16
#define FLASH_ATTN_BLOCK_N_F16_NARROW 16
#define FLASH_ATTN_BLOCK_M_F16_WIDE 8
#define FLASH_ATTN_BLOCK_N_F16_WIDE 32
#define FLASH_ATTN_MAX_DIM_F16_WIDE 256
#define FLASH_ATTN_BLOCK_M_F16_WIDE_NARROW 8
#define FLASH_ATTN_BLOCK_N_F16_WIDE_NARROW 16
#define FLASH_ATTN_BLOCK_M_BF16 16
#define FLASH_ATTN_BLOCK_N_BF16 32
#define FLASH_ATTN_MAX_DIM_BF16 128
#define FLASH_ATTN_BLOCK_M_BF16_NARROW 16
#define FLASH_ATTN_BLOCK_N_BF16_NARROW 16
#define FLASH_ATTN_BLOCK_M_BF16_WIDE 8
#define FLASH_ATTN_BLOCK_N_BF16_WIDE 32
#define FLASH_ATTN_MAX_DIM_BF16_WIDE 256
#define FLASH_ATTN_BLOCK_M_BF16_WIDE_NARROW 8
#define FLASH_ATTN_BLOCK_N_BF16_WIDE_NARROW 16
#if MARMOT_ENABLE_FP8
#define FLASH_ATTN_BLOCK_M_FP8 16
#define FLASH_ATTN_BLOCK_N_FP8 32
#define FLASH_ATTN_MAX_DIM_FP8 128
#define FLASH_ATTN_BLOCK_M_FP8_WIDE 8
#define FLASH_ATTN_BLOCK_N_FP8_WIDE 32
#define FLASH_ATTN_MAX_DIM_FP8_WIDE 256
#endif

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

#define DEFINE_PAGED_FLASH_ATTENTION_KERNEL(NAME, VALUE_T, READ_FN, WRITE_FN, TILE_T, MAX_DIM, BLOCK_M, BLOCK_N)       \
    kernel void NAME(                                                                                                  \
        device const uint *token_meta [[buffer(0)]], device const uint *block_table [[buffer(1)]],                     \
        device const VALUE_T *q [[buffer(2)]], device const VALUE_T *kv_k [[buffer(3)]],                               \
        device const VALUE_T *kv_v [[buffer(4)]], device VALUE_T *out [[buffer(5)]],                                   \
        constant PagedAttentionUniforms &u [[buffer(6)]], constant PagedAttentionStrides &s [[buffer(7)]],             \
        uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]],                      \
        uint3 tgsize [[threads_per_threadgroup]]                                                                       \
    ) {                                                                                                                \
        const uint q_block_idx = gid.x;                                                                                \
        const uint q_head = gid.y;                                                                                     \
        if (q_head >= u.num_q_heads || u.head_dim == 0u || u.head_dim > MAX_DIM) {                                     \
            return;                                                                                                    \
        }                                                                                                              \
        const uint q_start = q_block_idx * BLOCK_M;                                                                    \
        if (q_start >= u.token_count) {                                                                                \
            return;                                                                                                    \
        }                                                                                                              \
        const uint block_m = min((uint)BLOCK_M, u.token_count - q_start);                                              \
        const uint tid_k = tid.x;                                                                                      \
        const uint tid_q = tid.y;                                                                                      \
        const uint gqa_group = (u.gqa_group_size == 0u) ? 1u : u.gqa_group_size;                                       \
        const uint kv_head = (gqa_group == 1u) ? q_head : (q_head / gqa_group);                                        \
        threadgroup TILE_T q_tile[BLOCK_M * MAX_DIM];                                                                  \
        threadgroup TILE_T kv_tile[BLOCK_N * MAX_DIM];                                                                 \
        threadgroup float s_tile[BLOCK_M * BLOCK_N];                                                                   \
        threadgroup float r_tile[BLOCK_M * BLOCK_N];                                                                   \
        threadgroup float o_tile[BLOCK_M * MAX_DIM];                                                                   \
        threadgroup float m_tile[BLOCK_M];                                                                             \
        threadgroup float l_tile[BLOCK_M];                                                                             \
        threadgroup float block_max[BLOCK_M];                                                                          \
        threadgroup float block_sum[BLOCK_M];                                                                          \
        threadgroup float exp_prev_tile[BLOCK_M];                                                                      \
        threadgroup float exp_block_tile[BLOCK_M];                                                                     \
        threadgroup uint row_pos[BLOCK_M];                                                                             \
        threadgroup uint seq_slot_shared;                                                                              \
        threadgroup uint kv_len_max;                                                                                   \
        threadgroup uint kv_valid[BLOCK_N];                                                                            \
        threadgroup ulong kv_row_base[BLOCK_N];                                                                        \
        const uint linear_tid = tid_q * tgsize.x + tid_k;                                                              \
        const uint total_threads = tgsize.x * tgsize.y;                                                                \
        const uint my_q_row = tid_q;                                                                                   \
        if (tid_k == 0 && tid_q < BLOCK_M) {                                                                           \
            if (tid_q < block_m) {                                                                                     \
                const uint token_index = u.token_offset + q_start + tid_q;                                             \
                const size_t meta_base = (size_t)token_index * (size_t)u.meta_stride0;                                 \
                row_pos[tid_q] = token_meta[meta_base + (size_t)1u * u.meta_stride1];                                  \
                if (tid_q == 0u) {                                                                                     \
                    seq_slot_shared = token_meta[meta_base + (size_t)0u * u.meta_stride1];                             \
                }                                                                                                      \
            } else {                                                                                                   \
                row_pos[tid_q] = 0u;                                                                                   \
                if (tid_q == 0u) {                                                                                     \
                    seq_slot_shared = 0u;                                                                              \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        if (tid_k == 0 && tid_q == 0) {                                                                                \
            uint max_pos = 0u;                                                                                         \
            for (uint r = 0; r < block_m; ++r) {                                                                       \
                max_pos = max(max_pos, row_pos[r]);                                                                    \
            }                                                                                                          \
            kv_len_max = max_pos + 1u;                                                                                 \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        if (my_q_row < block_m) {                                                                                      \
            for (uint d = tid_k; d < u.head_dim; d += tgsize.x) {                                                      \
                o_tile[my_q_row * MAX_DIM + d] = 0.0f;                                                                 \
            }                                                                                                          \
            if (tid_k == 0) {                                                                                          \
                m_tile[my_q_row] = -INFINITY;                                                                          \
                l_tile[my_q_row] = 0.0f;                                                                               \
            }                                                                                                          \
        }                                                                                                              \
        if (my_q_row < block_m) {                                                                                      \
            const uint token_index = u.token_offset + q_start + my_q_row;                                              \
            const size_t q_base = (size_t)token_index * s.q_stride0 + (size_t)q_head * s.q_stride1;                    \
            for (uint d = tid_k; d < u.head_dim; d += tgsize.x) {                                                      \
                q_tile[my_q_row * MAX_DIM + d] = q[q_base + (size_t)d * s.q_stride2];                                  \
            }                                                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        const uint block_mask = u.block_size - 1u;                                                                     \
        const uint num_k_blocks = (kv_len_max + BLOCK_N - 1u) / BLOCK_N;                                               \
        for (uint k_block = 0; k_block < num_k_blocks; ++k_block) {                                                    \
            const uint k_start = k_block * BLOCK_N;                                                                    \
            const uint k_end = min(k_start + BLOCK_N, kv_len_max);                                                     \
            const uint block_n = k_end - k_start;                                                                      \
            if (block_n == 0) {                                                                                        \
                continue;                                                                                              \
            }                                                                                                          \
            for (uint idx = linear_tid; idx < block_n; idx += total_threads) {                                         \
                const uint global_k = k_start + idx;                                                                   \
                const uint logical_block = global_k >> u.block_shift;                                                  \
                if (logical_block >= u.max_blocks_per_seq) {                                                           \
                    kv_valid[idx] = 0u;                                                                                \
                    kv_row_base[idx] = 0u;                                                                             \
                } else {                                                                                               \
                    const size_t block_index = (size_t)seq_slot_shared * (size_t)u.block_stride0 +                     \
                        (size_t)logical_block * (size_t)u.block_stride1;                                               \
                    const uint block_id = block_table[block_index];                                                    \
                    if (block_id == 0xFFFFFFFFu) {                                                                     \
                        kv_valid[idx] = 0u;                                                                            \
                        kv_row_base[idx] = 0u;                                                                         \
                    } else {                                                                                           \
                        const uint offset = global_k & block_mask;                                                     \
                        kv_valid[idx] = 1u;                                                                            \
                        kv_row_base[idx] = (ulong)block_id * s.kv_stride0 + (ulong)u.layer_idx * s.kv_stride1 +        \
                            (ulong)kv_head * s.kv_stride2 + (ulong)offset * s.kv_stride3;                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            for (uint idx = linear_tid; idx < block_n * u.head_dim; idx += total_threads) {                            \
                const uint k_local_row = idx / u.head_dim;                                                             \
                const uint d = idx % u.head_dim;                                                                       \
                if (k_local_row < block_n && d < u.head_dim) {                                                         \
                    if (kv_valid[k_local_row] != 0u) {                                                                 \
                        kv_tile[k_local_row * MAX_DIM + d] = kv_k[kv_row_base[k_local_row] + (ulong)d * s.kv_stride4]; \
                    } else {                                                                                           \
                        kv_tile[k_local_row * MAX_DIM + d] = (TILE_T)0;                                                \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (my_q_row < block_m && tid_k < BLOCK_N) {                                                               \
                const uint global_k_col = k_start + tid_k;                                                             \
                float dot = -INFINITY;                                                                                 \
                if (tid_k < block_n && kv_valid[tid_k] != 0u && global_k_col <= row_pos[my_q_row]) {                   \
                    dot = 0.0f;                                                                                        \
                    for (uint d = 0; d < u.head_dim; ++d) {                                                            \
                        dot += READ_FN(q_tile[my_q_row * MAX_DIM + d]) * READ_FN(kv_tile[tid_k * MAX_DIM + d]);        \
                    }                                                                                                  \
                    dot *= u.scale;                                                                                    \
                }                                                                                                      \
                s_tile[my_q_row * BLOCK_N + tid_k] = dot;                                                              \
                r_tile[my_q_row * BLOCK_N + tid_k] = dot;                                                              \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            for (uint stride = BLOCK_N / 2; stride > 0; stride >>= 1) {                                                \
                if (my_q_row < block_m && tid_k < stride) {                                                            \
                    const uint base = my_q_row * BLOCK_N + tid_k;                                                      \
                    r_tile[base] = max(r_tile[base], r_tile[base + stride]);                                           \
                }                                                                                                      \
                threadgroup_barrier(mem_flags::mem_threadgroup);                                                       \
            }                                                                                                          \
            if (my_q_row < block_m && tid_k == 0) {                                                                    \
                block_max[my_q_row] = r_tile[my_q_row * BLOCK_N];                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (my_q_row < block_m && tid_k < BLOCK_N) {                                                               \
                float m_block = block_max[my_q_row];                                                                   \
                float weight = (m_block == -INFINITY) ? 0.0f : exp(s_tile[my_q_row * BLOCK_N + tid_k] - m_block);      \
                s_tile[my_q_row * BLOCK_N + tid_k] = weight;                                                           \
                r_tile[my_q_row * BLOCK_N + tid_k] = weight;                                                           \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            for (uint stride = BLOCK_N / 2; stride > 0; stride >>= 1) {                                                \
                if (my_q_row < block_m && tid_k < stride) {                                                            \
                    const uint base = my_q_row * BLOCK_N + tid_k;                                                      \
                    r_tile[base] += r_tile[base + stride];                                                             \
                }                                                                                                      \
                threadgroup_barrier(mem_flags::mem_threadgroup);                                                       \
            }                                                                                                          \
            if (my_q_row < block_m && tid_k == 0) {                                                                    \
                block_sum[my_q_row] = r_tile[my_q_row * BLOCK_N];                                                      \
                float m_prev = m_tile[my_q_row];                                                                       \
                float l_prev = l_tile[my_q_row];                                                                       \
                float m_block = block_max[my_q_row];                                                                   \
                float exp_prev = 1.0f;                                                                                 \
                float exp_block = 0.0f;                                                                                \
                float m_new = m_prev;                                                                                  \
                float l_new = l_prev;                                                                                  \
                if (m_block != -INFINITY) {                                                                            \
                    m_new = max(m_prev, m_block);                                                                      \
                    exp_prev = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);                                     \
                    exp_block = exp(m_block - m_new);                                                                  \
                    l_new = l_prev * exp_prev + block_sum[my_q_row] * exp_block;                                       \
                }                                                                                                      \
                m_tile[my_q_row] = m_new;                                                                              \
                l_tile[my_q_row] = l_new;                                                                              \
                exp_prev_tile[my_q_row] = exp_prev;                                                                    \
                exp_block_tile[my_q_row] = exp_block;                                                                  \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            for (uint idx = linear_tid; idx < block_n * u.head_dim; idx += total_threads) {                            \
                const uint v_local_row = idx / u.head_dim;                                                             \
                const uint d = idx % u.head_dim;                                                                       \
                if (v_local_row < block_n && d < u.head_dim) {                                                         \
                    if (kv_valid[v_local_row] != 0u) {                                                                 \
                        kv_tile[v_local_row * MAX_DIM + d] = kv_v[kv_row_base[v_local_row] + (ulong)d * s.kv_stride4]; \
                    } else {                                                                                           \
                        kv_tile[v_local_row * MAX_DIM + d] = (TILE_T)0;                                                \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (my_q_row < block_m) {                                                                                  \
                float exp_prev = exp_prev_tile[my_q_row];                                                              \
                float exp_block = exp_block_tile[my_q_row];                                                            \
                for (uint d = tid_k; d < u.head_dim; d += tgsize.x) {                                                  \
                    float acc = 0.0f;                                                                                  \
                    for (uint j = 0; j < block_n; ++j) {                                                               \
                        float weight = s_tile[my_q_row * BLOCK_N + j];                                                 \
                        if (weight != 0.0f) {                                                                          \
                            acc += weight * READ_FN(kv_tile[j * MAX_DIM + d]);                                         \
                        }                                                                                              \
                    }                                                                                                  \
                    const uint out_idx = my_q_row * MAX_DIM + d;                                                       \
                    o_tile[out_idx] = o_tile[out_idx] * exp_prev + acc * exp_block;                                    \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
        if (my_q_row < block_m) {                                                                                      \
            const uint token_index = u.token_offset + q_start + my_q_row;                                              \
            const size_t out_base = (size_t)token_index * s.out_stride0 + (size_t)q_head * s.out_stride1;              \
            float inv_l = (l_tile[my_q_row] > 0.0f) ? (1.0f / l_tile[my_q_row]) : 0.0f;                                \
            for (uint d = tid_k; d < u.head_dim; d += tgsize.x) {                                                      \
                out[out_base + (size_t)d * s.out_stride2] = WRITE_FN(o_tile[my_q_row * MAX_DIM + d] * inv_l);          \
            }                                                                                                          \
        }                                                                                                              \
    }

#define DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(                                                                     \
    NAME, Q_T, KV_T, READ_Q, READ_KV, WRITE_OUT, MAX_DIM, BLOCK_M, BLOCK_N                                             \
)                                                                                                                      \
    kernel void NAME(                                                                                                  \
        device const uint *token_meta [[buffer(0)]], device const uint *block_table [[buffer(1)]],                     \
        device const Q_T *q [[buffer(2)]], device const KV_T *kv_k [[buffer(3)]],                                      \
        device const KV_T *kv_v [[buffer(4)]], device Q_T *out [[buffer(5)]],                                          \
        constant PagedAttentionUniforms &u [[buffer(6)]], constant PagedAttentionStrides &s [[buffer(7)]],             \
        uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]],                      \
        uint3 tgsize [[threads_per_threadgroup]]                                                                       \
    ) {                                                                                                                \
        const uint q_block_idx = gid.x;                                                                                \
        const uint q_head = gid.y;                                                                                     \
        if (q_head >= u.num_q_heads || u.head_dim == 0u || u.head_dim > MAX_DIM) {                                     \
            return;                                                                                                    \
        }                                                                                                              \
        const uint q_start = q_block_idx * BLOCK_M;                                                                    \
        if (q_start >= u.token_count) {                                                                                \
            return;                                                                                                    \
        }                                                                                                              \
        const uint block_m = min((uint)BLOCK_M, u.token_count - q_start);                                              \
        const uint tid_k = tid.x;                                                                                      \
        const uint tid_q = tid.y;                                                                                      \
        const uint gqa_group = (u.gqa_group_size == 0u) ? 1u : u.gqa_group_size;                                       \
        const uint kv_head = (gqa_group == 1u) ? q_head : (q_head / gqa_group);                                        \
        threadgroup Q_T q_tile[BLOCK_M * MAX_DIM];                                                                     \
        threadgroup KV_T kv_tile[BLOCK_N * MAX_DIM];                                                                   \
        threadgroup float s_tile[BLOCK_M * BLOCK_N];                                                                   \
        threadgroup float r_tile[BLOCK_M * BLOCK_N];                                                                   \
        threadgroup float o_tile[BLOCK_M * MAX_DIM];                                                                   \
        threadgroup float m_tile[BLOCK_M];                                                                             \
        threadgroup float l_tile[BLOCK_M];                                                                             \
        threadgroup float block_max[BLOCK_M];                                                                          \
        threadgroup float block_sum[BLOCK_M];                                                                          \
        threadgroup float exp_prev_tile[BLOCK_M];                                                                      \
        threadgroup float exp_block_tile[BLOCK_M];                                                                     \
        threadgroup uint row_pos[BLOCK_M];                                                                             \
        threadgroup uint seq_slot_shared;                                                                              \
        threadgroup uint kv_len_max;                                                                                   \
        threadgroup uint kv_valid[BLOCK_N];                                                                            \
        threadgroup ulong kv_row_base[BLOCK_N];                                                                        \
        const uint linear_tid = tid_q * tgsize.x + tid_k;                                                              \
        const uint total_threads = tgsize.x * tgsize.y;                                                                \
        const uint my_q_row = tid_q;                                                                                   \
        if (tid_k == 0 && tid_q < BLOCK_M) {                                                                           \
            if (tid_q < block_m) {                                                                                     \
                const uint token_index = u.token_offset + q_start + tid_q;                                             \
                const size_t meta_base = (size_t)token_index * (size_t)u.meta_stride0;                                 \
                row_pos[tid_q] = token_meta[meta_base + (size_t)1u * u.meta_stride1];                                  \
                if (tid_q == 0u) {                                                                                     \
                    seq_slot_shared = token_meta[meta_base + (size_t)0u * u.meta_stride1];                             \
                }                                                                                                      \
            } else {                                                                                                   \
                row_pos[tid_q] = 0u;                                                                                   \
                if (tid_q == 0u) {                                                                                     \
                    seq_slot_shared = 0u;                                                                              \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        if (tid_k == 0 && tid_q == 0) {                                                                                \
            uint max_pos = 0u;                                                                                         \
            for (uint r = 0; r < block_m; ++r) {                                                                       \
                max_pos = max(max_pos, row_pos[r]);                                                                    \
            }                                                                                                          \
            kv_len_max = max_pos + 1u;                                                                                 \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        if (my_q_row < block_m) {                                                                                      \
            for (uint d = tid_k; d < u.head_dim; d += tgsize.x) {                                                      \
                o_tile[my_q_row * MAX_DIM + d] = 0.0f;                                                                 \
            }                                                                                                          \
            if (tid_k == 0) {                                                                                          \
                m_tile[my_q_row] = -INFINITY;                                                                          \
                l_tile[my_q_row] = 0.0f;                                                                               \
            }                                                                                                          \
        }                                                                                                              \
        if (my_q_row < block_m) {                                                                                      \
            const uint token_index = u.token_offset + q_start + my_q_row;                                              \
            const size_t q_base = (size_t)token_index * s.q_stride0 + (size_t)q_head * s.q_stride1;                    \
            for (uint d = tid_k; d < u.head_dim; d += tgsize.x) {                                                      \
                q_tile[my_q_row * MAX_DIM + d] = q[q_base + (size_t)d * s.q_stride2];                                  \
            }                                                                                                          \
        }                                                                                                              \
        threadgroup_barrier(mem_flags::mem_threadgroup);                                                               \
        const uint block_mask = u.block_size - 1u;                                                                     \
        const uint num_k_blocks = (kv_len_max + BLOCK_N - 1u) / BLOCK_N;                                               \
        for (uint k_block = 0; k_block < num_k_blocks; ++k_block) {                                                    \
            const uint k_start = k_block * BLOCK_N;                                                                    \
            const uint k_end = min(k_start + BLOCK_N, kv_len_max);                                                     \
            const uint block_n = k_end - k_start;                                                                      \
            if (block_n == 0) {                                                                                        \
                continue;                                                                                              \
            }                                                                                                          \
            for (uint idx = linear_tid; idx < block_n; idx += total_threads) {                                         \
                const uint global_k = k_start + idx;                                                                   \
                const uint logical_block = global_k >> u.block_shift;                                                  \
                if (logical_block >= u.max_blocks_per_seq) {                                                           \
                    kv_valid[idx] = 0u;                                                                                \
                    kv_row_base[idx] = 0u;                                                                             \
                } else {                                                                                               \
                    const size_t block_index = (size_t)seq_slot_shared * (size_t)u.block_stride0 +                     \
                        (size_t)logical_block * (size_t)u.block_stride1;                                               \
                    const uint block_id = block_table[block_index];                                                    \
                    if (block_id == 0xFFFFFFFFu) {                                                                     \
                        kv_valid[idx] = 0u;                                                                            \
                        kv_row_base[idx] = 0u;                                                                         \
                    } else {                                                                                           \
                        const uint offset = global_k & block_mask;                                                     \
                        kv_valid[idx] = 1u;                                                                            \
                        kv_row_base[idx] = (ulong)block_id * s.kv_stride0 + (ulong)u.layer_idx * s.kv_stride1 +        \
                            (ulong)kv_head * s.kv_stride2 + (ulong)offset * s.kv_stride3;                              \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            for (uint idx = linear_tid; idx < block_n * u.head_dim; idx += total_threads) {                            \
                const uint k_local_row = idx / u.head_dim;                                                             \
                const uint d = idx % u.head_dim;                                                                       \
                if (k_local_row < block_n && d < u.head_dim) {                                                         \
                    if (kv_valid[k_local_row] != 0u) {                                                                 \
                        kv_tile[k_local_row * MAX_DIM + d] = kv_k[kv_row_base[k_local_row] + (ulong)d * s.kv_stride4]; \
                    } else {                                                                                           \
                        kv_tile[k_local_row * MAX_DIM + d] = (KV_T)0;                                                  \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (my_q_row < block_m && tid_k < BLOCK_N) {                                                               \
                const uint global_k_col = k_start + tid_k;                                                             \
                float dot = -INFINITY;                                                                                 \
                if (tid_k < block_n && kv_valid[tid_k] != 0u && global_k_col <= row_pos[my_q_row]) {                   \
                    dot = 0.0f;                                                                                        \
                    for (uint d = 0; d < u.head_dim; ++d) {                                                            \
                        dot += READ_Q(q_tile[my_q_row * MAX_DIM + d]) * READ_KV(kv_tile[tid_k * MAX_DIM + d]);         \
                    }                                                                                                  \
                    dot *= u.scale;                                                                                    \
                }                                                                                                      \
                s_tile[my_q_row * BLOCK_N + tid_k] = dot;                                                              \
                r_tile[my_q_row * BLOCK_N + tid_k] = dot;                                                              \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            for (uint stride = BLOCK_N / 2; stride > 0; stride >>= 1) {                                                \
                if (my_q_row < block_m && tid_k < stride) {                                                            \
                    const uint base = my_q_row * BLOCK_N + tid_k;                                                      \
                    r_tile[base] = max(r_tile[base], r_tile[base + stride]);                                           \
                }                                                                                                      \
                threadgroup_barrier(mem_flags::mem_threadgroup);                                                       \
            }                                                                                                          \
            if (my_q_row < block_m && tid_k == 0) {                                                                    \
                block_max[my_q_row] = r_tile[my_q_row * BLOCK_N];                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (my_q_row < block_m && tid_k < BLOCK_N) {                                                               \
                float m_block = block_max[my_q_row];                                                                   \
                float weight = (m_block == -INFINITY) ? 0.0f : exp(s_tile[my_q_row * BLOCK_N + tid_k] - m_block);      \
                s_tile[my_q_row * BLOCK_N + tid_k] = weight;                                                           \
                r_tile[my_q_row * BLOCK_N + tid_k] = weight;                                                           \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            for (uint stride = BLOCK_N / 2; stride > 0; stride >>= 1) {                                                \
                if (my_q_row < block_m && tid_k < stride) {                                                            \
                    const uint base = my_q_row * BLOCK_N + tid_k;                                                      \
                    r_tile[base] += r_tile[base + stride];                                                             \
                }                                                                                                      \
                threadgroup_barrier(mem_flags::mem_threadgroup);                                                       \
            }                                                                                                          \
            if (my_q_row < block_m && tid_k == 0) {                                                                    \
                block_sum[my_q_row] = r_tile[my_q_row * BLOCK_N];                                                      \
                float m_prev = m_tile[my_q_row];                                                                       \
                float l_prev = l_tile[my_q_row];                                                                       \
                float m_block = block_max[my_q_row];                                                                   \
                float exp_prev = 1.0f;                                                                                 \
                float exp_block = 0.0f;                                                                                \
                float m_new = m_prev;                                                                                  \
                float l_new = l_prev;                                                                                  \
                if (m_block != -INFINITY) {                                                                            \
                    m_new = max(m_prev, m_block);                                                                      \
                    exp_prev = (m_prev == -INFINITY) ? 0.0f : exp(m_prev - m_new);                                     \
                    exp_block = exp(m_block - m_new);                                                                  \
                    l_new = l_prev * exp_prev + block_sum[my_q_row] * exp_block;                                       \
                }                                                                                                      \
                m_tile[my_q_row] = m_new;                                                                              \
                l_tile[my_q_row] = l_new;                                                                              \
                exp_prev_tile[my_q_row] = exp_prev;                                                                    \
                exp_block_tile[my_q_row] = exp_block;                                                                  \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            for (uint idx = linear_tid; idx < block_n * u.head_dim; idx += total_threads) {                            \
                const uint v_local_row = idx / u.head_dim;                                                             \
                const uint d = idx % u.head_dim;                                                                       \
                if (v_local_row < block_n && d < u.head_dim) {                                                         \
                    if (kv_valid[v_local_row] != 0u) {                                                                 \
                        kv_tile[v_local_row * MAX_DIM + d] = kv_v[kv_row_base[v_local_row] + (ulong)d * s.kv_stride4]; \
                    } else {                                                                                           \
                        kv_tile[v_local_row * MAX_DIM + d] = (KV_T)0;                                                  \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (my_q_row < block_m) {                                                                                  \
                float exp_prev = exp_prev_tile[my_q_row];                                                              \
                float exp_block = exp_block_tile[my_q_row];                                                            \
                for (uint d = tid_k; d < u.head_dim; d += tgsize.x) {                                                  \
                    float acc = 0.0f;                                                                                  \
                    for (uint j = 0; j < block_n; ++j) {                                                               \
                        float weight = s_tile[my_q_row * BLOCK_N + j];                                                 \
                        if (weight != 0.0f) {                                                                          \
                            acc += weight * READ_KV(kv_tile[j * MAX_DIM + d]);                                         \
                        }                                                                                              \
                    }                                                                                                  \
                    const uint out_idx = my_q_row * MAX_DIM + d;                                                       \
                    o_tile[out_idx] = o_tile[out_idx] * exp_prev + acc * exp_block;                                    \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
        if (my_q_row < block_m) {                                                                                      \
            const uint token_index = u.token_offset + q_start + my_q_row;                                              \
            const size_t out_base = (size_t)token_index * s.out_stride0 + (size_t)q_head * s.out_stride1;              \
            float inv_l = (l_tile[my_q_row] > 0.0f) ? (1.0f / l_tile[my_q_row]) : 0.0f;                                \
            for (uint d = tid_k; d < u.head_dim; d += tgsize.x) {                                                      \
                out[out_base + (size_t)d * s.out_stride2] = WRITE_OUT(o_tile[my_q_row * MAX_DIM + d] * inv_l);         \
            }                                                                                                          \
        }                                                                                                              \
    }

DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32, float, read_float, write_float, float, FLASH_ATTN_MAX_DIM_F32, FLASH_ATTN_BLOCK_M_F32,
    FLASH_ATTN_BLOCK_N_F32
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_wide, float, read_float, write_float, float, FLASH_ATTN_MAX_DIM_F32_WIDE,
    FLASH_ATTN_BLOCK_M_F32_WIDE, FLASH_ATTN_BLOCK_N_F32_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_narrow, float, read_float, write_float, float, FLASH_ATTN_MAX_DIM_F32,
    FLASH_ATTN_BLOCK_M_F32_NARROW, FLASH_ATTN_BLOCK_N_F32_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f32_wide_narrow, float, read_float, write_float, float, FLASH_ATTN_MAX_DIM_F32_WIDE,
    FLASH_ATTN_BLOCK_M_F32_WIDE_NARROW, FLASH_ATTN_BLOCK_N_F32_WIDE_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16, half, read_half, write_half, half, FLASH_ATTN_MAX_DIM_F16, FLASH_ATTN_BLOCK_M_F16,
    FLASH_ATTN_BLOCK_N_F16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_wide, half, read_half, write_half, half, FLASH_ATTN_MAX_DIM_F16_WIDE,
    FLASH_ATTN_BLOCK_M_F16_WIDE, FLASH_ATTN_BLOCK_N_F16_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_narrow, half, read_half, write_half, half, FLASH_ATTN_MAX_DIM_F16,
    FLASH_ATTN_BLOCK_M_F16_NARROW, FLASH_ATTN_BLOCK_N_F16_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_f16_wide_narrow, half, read_half, write_half, half, FLASH_ATTN_MAX_DIM_F16_WIDE,
    FLASH_ATTN_BLOCK_M_F16_WIDE_NARROW, FLASH_ATTN_BLOCK_N_F16_WIDE_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16, ushort, read_bf16, write_bf16, ushort, FLASH_ATTN_MAX_DIM_BF16, FLASH_ATTN_BLOCK_M_BF16,
    FLASH_ATTN_BLOCK_N_BF16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_wide, ushort, read_bf16, write_bf16, ushort, FLASH_ATTN_MAX_DIM_BF16_WIDE,
    FLASH_ATTN_BLOCK_M_BF16_WIDE, FLASH_ATTN_BLOCK_N_BF16_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_narrow, ushort, read_bf16, write_bf16, ushort, FLASH_ATTN_MAX_DIM_BF16,
    FLASH_ATTN_BLOCK_M_BF16_NARROW, FLASH_ATTN_BLOCK_N_BF16_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL(
    paged_flash_attention_bf16_wide_narrow, ushort, read_bf16, write_bf16, ushort, FLASH_ATTN_MAX_DIM_BF16_WIDE,
    FLASH_ATTN_BLOCK_M_BF16_WIDE_NARROW, FLASH_ATTN_BLOCK_N_BF16_WIDE_NARROW
)

DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f32_kv_f16, float, half, read_float, read_half, write_float, FLASH_ATTN_MAX_DIM_F32,
    FLASH_ATTN_BLOCK_M_F32, FLASH_ATTN_BLOCK_N_F32
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f32_kv_f16_wide, float, half, read_float, read_half, write_float, FLASH_ATTN_MAX_DIM_F32_WIDE,
    FLASH_ATTN_BLOCK_M_F32_WIDE, FLASH_ATTN_BLOCK_N_F32_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f32_kv_f16_narrow, float, half, read_float, read_half, write_float, FLASH_ATTN_MAX_DIM_F32,
    FLASH_ATTN_BLOCK_M_F32_NARROW, FLASH_ATTN_BLOCK_N_F32_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f32_kv_f16_wide_narrow, float, half, read_float, read_half, write_float,
    FLASH_ATTN_MAX_DIM_F32_WIDE, FLASH_ATTN_BLOCK_M_F32_WIDE_NARROW, FLASH_ATTN_BLOCK_N_F32_WIDE_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f32_kv_bf16, float, ushort, read_float, read_bf16, write_float, FLASH_ATTN_MAX_DIM_F32,
    FLASH_ATTN_BLOCK_M_F32, FLASH_ATTN_BLOCK_N_F32
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f32_kv_bf16_wide, float, ushort, read_float, read_bf16, write_float,
    FLASH_ATTN_MAX_DIM_F32_WIDE, FLASH_ATTN_BLOCK_M_F32_WIDE, FLASH_ATTN_BLOCK_N_F32_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f32_kv_bf16_narrow, float, ushort, read_float, read_bf16, write_float, FLASH_ATTN_MAX_DIM_F32,
    FLASH_ATTN_BLOCK_M_F32_NARROW, FLASH_ATTN_BLOCK_N_F32_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f32_kv_bf16_wide_narrow, float, ushort, read_float, read_bf16, write_float,
    FLASH_ATTN_MAX_DIM_F32_WIDE, FLASH_ATTN_BLOCK_M_F32_WIDE_NARROW, FLASH_ATTN_BLOCK_N_F32_WIDE_NARROW
)

DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f16_kv_f32, half, float, read_half, read_float, write_half, FLASH_ATTN_MAX_DIM_F16,
    FLASH_ATTN_BLOCK_M_F16, FLASH_ATTN_BLOCK_N_F16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f16_kv_f32_wide, half, float, read_half, read_float, write_half, FLASH_ATTN_MAX_DIM_F16_WIDE,
    FLASH_ATTN_BLOCK_M_F16_WIDE, FLASH_ATTN_BLOCK_N_F16_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f16_kv_f32_narrow, half, float, read_half, read_float, write_half, FLASH_ATTN_MAX_DIM_F16,
    FLASH_ATTN_BLOCK_M_F16_NARROW, FLASH_ATTN_BLOCK_N_F16_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f16_kv_f32_wide_narrow, half, float, read_half, read_float, write_half,
    FLASH_ATTN_MAX_DIM_F16_WIDE, FLASH_ATTN_BLOCK_M_F16_WIDE_NARROW, FLASH_ATTN_BLOCK_N_F16_WIDE_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f16_kv_bf16, half, ushort, read_half, read_bf16, write_half, FLASH_ATTN_MAX_DIM_F16,
    FLASH_ATTN_BLOCK_M_F16, FLASH_ATTN_BLOCK_N_F16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f16_kv_bf16_wide, half, ushort, read_half, read_bf16, write_half, FLASH_ATTN_MAX_DIM_F16_WIDE,
    FLASH_ATTN_BLOCK_M_F16_WIDE, FLASH_ATTN_BLOCK_N_F16_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f16_kv_bf16_narrow, half, ushort, read_half, read_bf16, write_half, FLASH_ATTN_MAX_DIM_F16,
    FLASH_ATTN_BLOCK_M_F16_NARROW, FLASH_ATTN_BLOCK_N_F16_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_f16_kv_bf16_wide_narrow, half, ushort, read_half, read_bf16, write_half,
    FLASH_ATTN_MAX_DIM_F16_WIDE, FLASH_ATTN_BLOCK_M_F16_WIDE_NARROW, FLASH_ATTN_BLOCK_N_F16_WIDE_NARROW
)

DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_bf16_kv_f32, ushort, float, read_bf16, read_float, write_bf16, FLASH_ATTN_MAX_DIM_BF16,
    FLASH_ATTN_BLOCK_M_BF16, FLASH_ATTN_BLOCK_N_BF16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_bf16_kv_f32_wide, ushort, float, read_bf16, read_float, write_bf16,
    FLASH_ATTN_MAX_DIM_BF16_WIDE, FLASH_ATTN_BLOCK_M_BF16_WIDE, FLASH_ATTN_BLOCK_N_BF16_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_bf16_kv_f32_narrow, ushort, float, read_bf16, read_float, write_bf16, FLASH_ATTN_MAX_DIM_BF16,
    FLASH_ATTN_BLOCK_M_BF16_NARROW, FLASH_ATTN_BLOCK_N_BF16_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_bf16_kv_f32_wide_narrow, ushort, float, read_bf16, read_float, write_bf16,
    FLASH_ATTN_MAX_DIM_BF16_WIDE, FLASH_ATTN_BLOCK_M_BF16_WIDE_NARROW, FLASH_ATTN_BLOCK_N_BF16_WIDE_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_bf16_kv_f16, ushort, half, read_bf16, read_half, write_bf16, FLASH_ATTN_MAX_DIM_BF16,
    FLASH_ATTN_BLOCK_M_BF16, FLASH_ATTN_BLOCK_N_BF16
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_bf16_kv_f16_wide, ushort, half, read_bf16, read_half, write_bf16,
    FLASH_ATTN_MAX_DIM_BF16_WIDE, FLASH_ATTN_BLOCK_M_BF16_WIDE, FLASH_ATTN_BLOCK_N_BF16_WIDE
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_bf16_kv_f16_narrow, ushort, half, read_bf16, read_half, write_bf16, FLASH_ATTN_MAX_DIM_BF16,
    FLASH_ATTN_BLOCK_M_BF16_NARROW, FLASH_ATTN_BLOCK_N_BF16_NARROW
)
DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED(
    paged_flash_attention_bf16_kv_f16_wide_narrow, ushort, half, read_bf16, read_half, write_bf16,
    FLASH_ATTN_MAX_DIM_BF16_WIDE, FLASH_ATTN_BLOCK_M_BF16_WIDE_NARROW, FLASH_ATTN_BLOCK_N_BF16_WIDE_NARROW
)

#undef DEFINE_PAGED_FLASH_ATTENTION_KERNEL_MIXED

#undef DEFINE_PAGED_FLASH_ATTENTION_KERNEL
#undef PAGED_ATTN_MAX_CHUNKS
#undef PAGED_ATTN_MAX_DIM
#undef PAGED_ATTN_SIMD_WIDTH
