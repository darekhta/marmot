#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

#include "common/activation_utils.h"

struct MatmulQKVUniforms {
    uint N;
    uint K;
    uint M;
    uint has_bias;
    uint has_residual;
    uint activation;
    uint use_packed_weights;
    uint packed_tile_cols;
    uint packed_tile_k;
    uint packed_tiles_per_row;
    uint packed_tiles_per_col;
    uint packed_tile_stride;
    uint packed_tile_section;
    uint packed_use_vec4;
    uint has_bias_q;
    uint has_bias_k;
    uint has_bias_v;
    uint rope_enabled;
    uint rope_apply_q;
    uint rope_apply_k;
    uint rope_head_dim;
    float rope_attn_scale;
    ActivationParams activation_params;
};

struct PackQKVUniforms {
    uint rows;
    uint cols;
    uint segment_rows;
    uint tile_cols;
    uint tile_k;
    uint tiles_per_row;
    uint tiles_per_col;
    uint tile_stride;
    uint tile_section;
    uint use_vec4;
};

#define MARMOT_QKV_TILE_M 32u
#define MARMOT_QKV_TILE_N 32u
#define MARMOT_QKV_TILE_K 8u

static inline bool matmul_qkv_compute_packed_offset(
    uint col, uint k, uint segment, constant MatmulQKVUniforms &uniforms, thread uint &out_index
) {
    if (uniforms.use_packed_weights == 0u || uniforms.packed_use_vec4 == 0u) {
        return false;
    }
    uint tile_cols = uniforms.packed_tile_cols;
    uint tile_k = uniforms.packed_tile_k;
    uint tiles_per_row = uniforms.packed_tiles_per_row;
    uint tiles_per_col = uniforms.packed_tiles_per_col;
    uint tile_stride = uniforms.packed_tile_stride;
    uint tile_section = uniforms.packed_tile_section;
    if (tile_cols == 0u || tile_k == 0u || tile_stride == 0u) {
        return false;
    }
    uint tile_col = col / tile_cols;
    uint tile_k_index = k / tile_k;
    if (tile_col >= tiles_per_row || tile_k_index >= tiles_per_col) {
        return false;
    }
    uint local_col = col % tile_cols;
    uint local_k = k % tile_k;
    uint tile_index = tile_col * tiles_per_col + tile_k_index;
    out_index = tile_index * tile_stride + segment * tile_section + local_col * tile_k + local_k;
    return true;
}

static inline float4 matmul_qkv_read_weight_vec4_f32(
    device const float *packed, uint col, uint k, uint segment, constant MatmulQKVUniforms &uniforms
) {
    uint base = 0u;
    if (!matmul_qkv_compute_packed_offset(col, k, segment, uniforms, base)) {
        return float4(0.0f);
    }
    const device float4 *vec_ptr = (const device float4 *)(packed + base);
    return *vec_ptr;
}

static inline float4 matmul_qkv_read_weight_vec4_f16(
    device const half *packed, uint col, uint k, uint segment, constant MatmulQKVUniforms &uniforms
) {
    uint base = 0u;
    if (!matmul_qkv_compute_packed_offset(col, k, segment, uniforms, base)) {
        return float4(0.0f);
    }
    const device half4 *vec_ptr = (const device half4 *)(packed + base);
    half4 value = *vec_ptr;
    return float4(value);
}

static inline float4 matmul_qkv_read_weight_vec4_bf16(
    device const ushort *packed, uint col, uint k, uint segment, constant MatmulQKVUniforms &uniforms
) {
    uint base = 0u;
    if (!matmul_qkv_compute_packed_offset(col, k, segment, uniforms, base)) {
        return float4(0.0f);
    }
    const device ushort4 *vec_ptr = (const device ushort4 *)(packed + base);
    ushort4 value = *vec_ptr;
    return float4(bf16_to_float(value.x), bf16_to_float(value.y), bf16_to_float(value.z), bf16_to_float(value.w));
}

#define DEFINE_MATMUL_QKV_KERNEL(NAME, SCALAR_T, READ_FN, WRITE_FN, READ_VEC4_FN, ENABLE_VEC4)                         \
    kernel void NAME(                                                                                                  \
        device const SCALAR_T *input [[buffer(0)]], device const SCALAR_T *weight [[buffer(1)]],                       \
        device const SCALAR_T *bias [[buffer(2)]], device const SCALAR_T *residual [[buffer(3)]],                      \
        device SCALAR_T *out_q [[buffer(4)]], device SCALAR_T *out_k [[buffer(5)]],                                    \
        device SCALAR_T *out_v [[buffer(6)]], device const float *rope_positions [[buffer(7)]],                        \
        device const float *rope_freqs [[buffer(8)]], constant MatmulQKVUniforms &uniforms [[buffer(9)]],              \
        uint2 tg_pos [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]                    \
    ) {                                                                                                                \
        if (tid.x >= MARMOT_QKV_TILE_N || tid.y >= MARMOT_QKV_TILE_M) {                                                \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        threadgroup float tileInput[MARMOT_QKV_TILE_M][MARMOT_QKV_TILE_K];                                             \
        threadgroup float tileWeightQ[MARMOT_QKV_TILE_K][MARMOT_QKV_TILE_N];                                           \
        threadgroup float tileWeightK[MARMOT_QKV_TILE_K][MARMOT_QKV_TILE_N];                                           \
        threadgroup float tileWeightV[MARMOT_QKV_TILE_K][MARMOT_QKV_TILE_N];                                           \
        threadgroup float tileRopeQ[MARMOT_QKV_TILE_M][MARMOT_QKV_TILE_N];                                             \
        threadgroup float tileRopeK[MARMOT_QKV_TILE_M][MARMOT_QKV_TILE_N];                                             \
                                                                                                                       \
        uint N = uniforms.N;                                                                                           \
        uint K = uniforms.K;                                                                                           \
        uint M = uniforms.M;                                                                                           \
                                                                                                                       \
        uint tile_row_base = tg_pos.y * MARMOT_QKV_TILE_M;                                                             \
        uint tile_col_base = tg_pos.x * MARMOT_QKV_TILE_N;                                                             \
        uint row = tile_row_base + tid.y;                                                                              \
        uint col = tile_col_base + tid.x;                                                                              \
                                                                                                                       \
        float acc_q = 0.0f;                                                                                            \
        float acc_k = 0.0f;                                                                                            \
        float acc_v = 0.0f;                                                                                            \
        bool rope_enabled = (uniforms.rope_enabled != 0u) && rope_positions != nullptr && rope_freqs != nullptr;       \
        bool apply_rope_q = rope_enabled && (uniforms.rope_apply_q != 0u);                                             \
        bool apply_rope_k = rope_enabled && (uniforms.rope_apply_k != 0u);                                             \
        bool needs_rope = rope_enabled && (apply_rope_q || apply_rope_k);                                              \
        uint rope_head_dim = uniforms.rope_head_dim;                                                                   \
        if (rope_head_dim == 0u || rope_head_dim > M || (M % rope_head_dim) != 0u || (rope_head_dim & 1u) != 0u) {     \
            rope_head_dim = M;                                                                                         \
        }                                                                                                              \
                                                                                                                       \
        for (uint k0 = 0; k0 < K; k0 += MARMOT_QKV_TILE_K) {                                                           \
            uint input_col = k0 + tid.x;                                                                               \
            if (tid.x < MARMOT_QKV_TILE_K && row < N && input_col < K) {                                               \
                tileInput[tid.y][tid.x] = READ_FN(input[row * K + input_col]);                                         \
            } else {                                                                                                   \
                if (tid.x < MARMOT_QKV_TILE_K) {                                                                       \
                    tileInput[tid.y][tid.x] = 0.0f;                                                                    \
                }                                                                                                      \
            }                                                                                                          \
                                                                                                                       \
            if (tid.y < MARMOT_QKV_TILE_K) {                                                                           \
                uint weight_row = k0 + tid.y;                                                                          \
                if (col < M && weight_row < K) {                                                                       \
                    bool packed_enabled = uniforms.use_packed_weights != 0u;                                           \
                    bool can_vec = packed_enabled && uniforms.packed_use_vec4 != 0u && ENABLE_VEC4 &&                  \
                        (tid.y & 3u) == 0u && (tid.y + 3u) < MARMOT_QKV_TILE_K && (weight_row + 3u) < K;               \
                    if (can_vec) {                                                                                     \
                        float4 q_vec = READ_VEC4_FN(weight, col, weight_row, 0u, uniforms);                            \
                        float4 k_vec = READ_VEC4_FN(weight, col, weight_row, 1u, uniforms);                            \
                        float4 v_vec = READ_VEC4_FN(weight, col, weight_row, 2u, uniforms);                            \
                        tileWeightQ[tid.y + 0u][tid.x] = q_vec.x;                                                      \
                        tileWeightQ[tid.y + 1u][tid.x] = q_vec.y;                                                      \
                        tileWeightQ[tid.y + 2u][tid.x] = q_vec.z;                                                      \
                        tileWeightQ[tid.y + 3u][tid.x] = q_vec.w;                                                      \
                        tileWeightK[tid.y + 0u][tid.x] = k_vec.x;                                                      \
                        tileWeightK[tid.y + 1u][tid.x] = k_vec.y;                                                      \
                        tileWeightK[tid.y + 2u][tid.x] = k_vec.z;                                                      \
                        tileWeightK[tid.y + 3u][tid.x] = k_vec.w;                                                      \
                        tileWeightV[tid.y + 0u][tid.x] = v_vec.x;                                                      \
                        tileWeightV[tid.y + 1u][tid.x] = v_vec.y;                                                      \
                        tileWeightV[tid.y + 2u][tid.x] = v_vec.z;                                                      \
                        tileWeightV[tid.y + 3u][tid.x] = v_vec.w;                                                      \
                    } else if (packed_enabled) {                                                                       \
                        uint tile_cols = uniforms.packed_tile_cols;                                                    \
                        uint tile_k = uniforms.packed_tile_k;                                                          \
                        uint tiles_per_row = uniforms.packed_tiles_per_row;                                            \
                        uint tiles_per_col = uniforms.packed_tiles_per_col;                                            \
                        uint tile_stride = uniforms.packed_tile_stride;                                                \
                        uint tile_section = uniforms.packed_tile_section;                                              \
                        if (tile_cols != 0u && tile_k != 0u && tile_stride != 0u) {                                    \
                            uint tile_col = col / tile_cols;                                                           \
                            uint tile_k_index = weight_row / tile_k;                                                   \
                            if (tile_col < tiles_per_row && tile_k_index < tiles_per_col) {                            \
                                uint local_col = col % tile_cols;                                                      \
                                uint local_k = weight_row % tile_k;                                                    \
                                uint tile_index = tile_col * tiles_per_col + tile_k_index;                             \
                                uint base = tile_index * tile_stride;                                                  \
                                uint linear = local_col * tile_k + local_k;                                            \
                                const device SCALAR_T *packed = weight;                                                \
                                uint q_offset = base + linear;                                                         \
                                uint k_offset = base + tile_section + linear;                                          \
                                uint v_offset = base + 2u * tile_section + linear;                                     \
                                tileWeightQ[tid.y][tid.x] = READ_FN(packed[q_offset]);                                 \
                                tileWeightK[tid.y][tid.x] = READ_FN(packed[k_offset]);                                 \
                                tileWeightV[tid.y][tid.x] = READ_FN(packed[v_offset]);                                 \
                            } else {                                                                                   \
                                tileWeightQ[tid.y][tid.x] = 0.0f;                                                      \
                                tileWeightK[tid.y][tid.x] = 0.0f;                                                      \
                                tileWeightV[tid.y][tid.x] = 0.0f;                                                      \
                            }                                                                                          \
                        } else {                                                                                       \
                            tileWeightQ[tid.y][tid.x] = 0.0f;                                                          \
                            tileWeightK[tid.y][tid.x] = 0.0f;                                                          \
                            tileWeightV[tid.y][tid.x] = 0.0f;                                                          \
                        }                                                                                              \
                    } else {                                                                                           \
                        uint base = col * K + weight_row;                                                              \
                        tileWeightQ[tid.y][tid.x] = READ_FN(weight[base]);                                             \
                        tileWeightK[tid.y][tid.x] = READ_FN(weight[(M + col) * K + weight_row]);                       \
                        tileWeightV[tid.y][tid.x] = READ_FN(weight[(2 * M + col) * K + weight_row]);                   \
                    }                                                                                                  \
                } else {                                                                                               \
                    tileWeightQ[tid.y][tid.x] = 0.0f;                                                                  \
                    tileWeightK[tid.y][tid.x] = 0.0f;                                                                  \
                    tileWeightV[tid.y][tid.x] = 0.0f;                                                                  \
                }                                                                                                      \
            }                                                                                                          \
                                                                                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
                                                                                                                       \
            uint max_k = min(MARMOT_QKV_TILE_K, K - k0);                                                               \
            for (uint kk = 0; kk < max_k; ++kk) {                                                                      \
                float a = tileInput[tid.y][kk];                                                                        \
                acc_q += a * tileWeightQ[kk][tid.x];                                                                   \
                acc_k += a * tileWeightK[kk][tid.x];                                                                   \
                acc_v += a * tileWeightV[kk][tid.x];                                                                   \
            }                                                                                                          \
                                                                                                                       \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
                                                                                                                       \
        bool valid_output = (row < N && col < M);                                                                      \
        bool has_bias_flag = (uniforms.has_bias != 0u && bias != nullptr);                                             \
        bool has_residual_flag = (uniforms.has_residual != 0u && residual != nullptr);                                 \
        float bias_q = 0.0f;                                                                                           \
        float bias_k = 0.0f;                                                                                           \
        float bias_v = 0.0f;                                                                                           \
        if (valid_output && has_bias_flag) {                                                                           \
            bias_q = READ_FN(bias[col]);                                                                               \
            bias_k = READ_FN(bias[M + col]);                                                                           \
            bias_v = READ_FN(bias[(2 * M) + col]);                                                                     \
        }                                                                                                              \
                                                                                                                       \
        uint out_index = row * M + col;                                                                                \
        bool has_residual = valid_output && has_residual_flag;                                                         \
        float residual_value = has_residual ? READ_FN(residual[out_index]) : 0.0f;                                     \
        float activated_q = apply_fused_activation(uniforms.activation, acc_q + bias_q, uniforms.activation_params);   \
        float activated_k = apply_fused_activation(uniforms.activation, acc_k + bias_k, uniforms.activation_params);   \
        float activated_v = apply_fused_activation(uniforms.activation, acc_v + bias_v, uniforms.activation_params);   \
        if (has_residual) {                                                                                            \
            activated_q += residual_value;                                                                             \
            activated_k += residual_value;                                                                             \
            activated_v += residual_value;                                                                             \
        }                                                                                                              \
                                                                                                                       \
        float final_q = activated_q;                                                                                   \
        float final_k = activated_k;                                                                                   \
        float final_v = activated_v;                                                                                   \
        if (needs_rope) {                                                                                              \
            tileRopeQ[tid.y][tid.x] = activated_q;                                                                     \
            tileRopeK[tid.y][tid.x] = activated_k;                                                                     \
        }                                                                                                              \
        if (needs_rope) {                                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (valid_output) {                                                                                        \
                uint head_base = (col / rope_head_dim) * rope_head_dim;                                                \
                uint local = col - head_base;                                                                          \
                uint even_local = local & ~1u;                                                                         \
                uint odd_local = even_local + 1u;                                                                      \
                if (odd_local < rope_head_dim) {                                                                       \
                    uint even_col = head_base + even_local;                                                            \
                    uint odd_col = head_base + odd_local;                                                              \
                    uint even_tile = even_col - tile_col_base;                                                         \
                    uint odd_tile = odd_col - tile_col_base;                                                           \
                    float even_q = tileRopeQ[tid.y][even_tile];                                                        \
                    float odd_q = tileRopeQ[tid.y][odd_tile];                                                          \
                    float even_k = tileRopeK[tid.y][even_tile];                                                        \
                    float odd_k = tileRopeK[tid.y][odd_tile];                                                          \
                    float pos = rope_positions[row];                                                                   \
                    float freq = rope_freqs[even_local >> 1];                                                          \
                    float angle = pos * freq;                                                                          \
                    float cos_val = cos(angle) * uniforms.rope_attn_scale;                                             \
                    float sin_val = sin(angle) * uniforms.rope_attn_scale;                                             \
                    if (apply_rope_q) {                                                                                \
                        float rotated_even_q = even_q * cos_val - odd_q * sin_val;                                     \
                        float rotated_odd_q = even_q * sin_val + odd_q * cos_val;                                      \
                        final_q = (col == even_col) ? rotated_even_q : rotated_odd_q;                                  \
                    }                                                                                                  \
                    if (apply_rope_k) {                                                                                \
                        float rotated_even_k = even_k * cos_val - odd_k * sin_val;                                     \
                        float rotated_odd_k = even_k * sin_val + odd_k * cos_val;                                      \
                        final_k = (col == even_col) ? rotated_even_k : rotated_odd_k;                                  \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
        if (valid_output) {                                                                                            \
            out_q[out_index] = WRITE_FN(final_q);                                                                      \
            out_k[out_index] = WRITE_FN(final_k);                                                                      \
            out_v[out_index] = WRITE_FN(final_v);                                                                      \
        }                                                                                                              \
    }

DEFINE_MATMUL_QKV_KERNEL(matmul_qkv_f32_nt, float, read_float, write_float, matmul_qkv_read_weight_vec4_f32, 1)
DEFINE_MATMUL_QKV_KERNEL(matmul_qkv_f16_nt, half, read_half, write_half, matmul_qkv_read_weight_vec4_f16, 1)
DEFINE_MATMUL_QKV_KERNEL(matmul_qkv_bf16_nt, ushort, read_bf16, write_bf16, matmul_qkv_read_weight_vec4_bf16, 1)

#define DEFINE_MATMUL_QKV_SEPARATE_KERNEL(NAME, SCALAR_T, READ_FN, WRITE_FN)                                           \
    kernel void NAME(                                                                                                  \
        device const SCALAR_T *input [[buffer(0)]], device const SCALAR_T *weight_q [[buffer(1)]],                     \
        device const SCALAR_T *weight_k [[buffer(2)]], device const SCALAR_T *weight_v [[buffer(3)]],                  \
        device const SCALAR_T *bias_q [[buffer(4)]], device const SCALAR_T *bias_k [[buffer(5)]],                      \
        device const SCALAR_T *bias_v [[buffer(6)]], device const SCALAR_T *residual [[buffer(7)]],                    \
        device SCALAR_T *out_q [[buffer(8)]], device SCALAR_T *out_k [[buffer(9)]],                                    \
        device SCALAR_T *out_v [[buffer(10)]], device const float *rope_positions [[buffer(11)]],                      \
        device const float *rope_freqs [[buffer(12)]], constant MatmulQKVUniforms &uniforms [[buffer(13)]],            \
        uint2 tg_pos [[threadgroup_position_in_grid]], uint2 tid [[thread_position_in_threadgroup]]                    \
    ) {                                                                                                                \
        if (tid.x >= MARMOT_QKV_TILE_N || tid.y >= MARMOT_QKV_TILE_M) {                                                \
            return;                                                                                                    \
        }                                                                                                              \
        uint tile_col = tg_pos.x * MARMOT_QKV_TILE_N;                                                                  \
        uint tile_row = tg_pos.y * MARMOT_QKV_TILE_M;                                                                  \
        uint row = tile_row + tid.y;                                                                                   \
        uint col = tile_col + tid.x;                                                                                   \
        uint N = uniforms.N;                                                                                           \
        uint M = uniforms.M;                                                                                           \
        uint K = uniforms.K;                                                                                           \
        threadgroup float tileInput[MARMOT_QKV_TILE_M][MARMOT_QKV_TILE_K];                                             \
        threadgroup float tileWeightQ[MARMOT_QKV_TILE_K][MARMOT_QKV_TILE_N];                                           \
        threadgroup float tileWeightK[MARMOT_QKV_TILE_K][MARMOT_QKV_TILE_N];                                           \
        threadgroup float tileWeightV[MARMOT_QKV_TILE_K][MARMOT_QKV_TILE_N];                                           \
        threadgroup float tileRopeQ[MARMOT_QKV_TILE_M][MARMOT_QKV_TILE_N];                                             \
        threadgroup float tileRopeK[MARMOT_QKV_TILE_M][MARMOT_QKV_TILE_N];                                             \
        float acc_q = 0.0f;                                                                                            \
        float acc_k = 0.0f;                                                                                            \
        float acc_v = 0.0f;                                                                                            \
        bool rope_enabled = (uniforms.rope_enabled != 0u) && rope_positions != nullptr && rope_freqs != nullptr;       \
        bool apply_rope_q = rope_enabled && (uniforms.rope_apply_q != 0u);                                             \
        bool apply_rope_k = rope_enabled && (uniforms.rope_apply_k != 0u);                                             \
        bool needs_rope = rope_enabled && (apply_rope_q || apply_rope_k);                                              \
        uint rope_head_dim = uniforms.rope_head_dim;                                                                   \
        if (rope_head_dim == 0u || rope_head_dim > M || (M % rope_head_dim) != 0u || (rope_head_dim & 1u) != 0u) {     \
            rope_head_dim = M;                                                                                         \
        }                                                                                                              \
        for (uint k0 = 0; k0 < K; k0 += MARMOT_QKV_TILE_K) {                                                           \
            uint input_row = row;                                                                                      \
            uint input_col = k0 + tid.x;                                                                               \
            if (input_row < N && input_col < K) {                                                                      \
                tileInput[tid.y][tid.x] = READ_FN(input[input_row * K + input_col]);                                   \
            } else {                                                                                                   \
                tileInput[tid.y][tid.x] = 0.0f;                                                                        \
            }                                                                                                          \
            uint weight_row = k0 + tid.y;                                                                              \
            if (weight_row < K && col < M) {                                                                           \
                tileWeightQ[tid.y][tid.x] = READ_FN(weight_q[col * K + weight_row]);                                   \
                tileWeightK[tid.y][tid.x] = READ_FN(weight_k[col * K + weight_row]);                                   \
                tileWeightV[tid.y][tid.x] = READ_FN(weight_v[col * K + weight_row]);                                   \
            } else {                                                                                                   \
                tileWeightQ[tid.y][tid.x] = 0.0f;                                                                      \
                tileWeightK[tid.y][tid.x] = 0.0f;                                                                      \
                tileWeightV[tid.y][tid.x] = 0.0f;                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            uint max_k = min(MARMOT_QKV_TILE_K, K - k0);                                                               \
            for (uint kk = 0; kk < max_k; ++kk) {                                                                      \
                float a = tileInput[tid.y][kk];                                                                        \
                acc_q += a * tileWeightQ[kk][tid.x];                                                                   \
                acc_k += a * tileWeightK[kk][tid.x];                                                                   \
                acc_v += a * tileWeightV[kk][tid.x];                                                                   \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
        bool valid_output = (row < N && col < M);                                                                      \
        bool has_bias_q = (uniforms.has_bias_q != 0u && bias_q != nullptr);                                            \
        bool has_bias_k = (uniforms.has_bias_k != 0u && bias_k != nullptr);                                            \
        bool has_bias_v = (uniforms.has_bias_v != 0u && bias_v != nullptr);                                            \
        bool has_residual_flag = (uniforms.has_residual != 0u && residual != nullptr);                                 \
        float bias_q_val = 0.0f;                                                                                       \
        float bias_k_val = 0.0f;                                                                                       \
        float bias_v_val = 0.0f;                                                                                       \
        if (valid_output) {                                                                                            \
            if (has_bias_q) {                                                                                          \
                bias_q_val = READ_FN(bias_q[col]);                                                                     \
            }                                                                                                          \
            if (has_bias_k) {                                                                                          \
                bias_k_val = READ_FN(bias_k[col]);                                                                     \
            }                                                                                                          \
            if (has_bias_v) {                                                                                          \
                bias_v_val = READ_FN(bias_v[col]);                                                                     \
            }                                                                                                          \
        }                                                                                                              \
        uint out_index = row * M + col;                                                                                \
        bool has_residual = valid_output && has_residual_flag;                                                         \
        float residual_value = has_residual ? READ_FN(residual[out_index]) : 0.0f;                                     \
        float activated_q =                                                                                            \
            apply_fused_activation(uniforms.activation, acc_q + bias_q_val, uniforms.activation_params);               \
        float activated_k =                                                                                            \
            apply_fused_activation(uniforms.activation, acc_k + bias_k_val, uniforms.activation_params);               \
        float activated_v =                                                                                            \
            apply_fused_activation(uniforms.activation, acc_v + bias_v_val, uniforms.activation_params);               \
        if (has_residual) {                                                                                            \
            activated_q += residual_value;                                                                             \
            activated_k += residual_value;                                                                             \
            activated_v += residual_value;                                                                             \
        }                                                                                                              \
        float final_q = activated_q;                                                                                   \
        float final_k = activated_k;                                                                                   \
        float final_v = activated_v;                                                                                   \
        if (needs_rope) {                                                                                              \
            tileRopeQ[tid.y][tid.x] = activated_q;                                                                     \
            tileRopeK[tid.y][tid.x] = activated_k;                                                                     \
        }                                                                                                              \
        if (needs_rope) {                                                                                              \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            if (valid_output) {                                                                                        \
                uint head_base = (col / rope_head_dim) * rope_head_dim;                                                \
                uint local = col - head_base;                                                                          \
                uint even_local = local & ~1u;                                                                         \
                uint odd_local = even_local + 1u;                                                                      \
                if (odd_local < rope_head_dim) {                                                                       \
                    uint even_col = head_base + even_local;                                                            \
                    uint odd_col = head_base + odd_local;                                                              \
                    uint even_tile = even_col - tile_col;                                                              \
                    uint odd_tile = odd_col - tile_col;                                                                \
                    float even_q = tileRopeQ[tid.y][even_tile];                                                        \
                    float odd_q = tileRopeQ[tid.y][odd_tile];                                                          \
                    float even_k = tileRopeK[tid.y][even_tile];                                                        \
                    float odd_k = tileRopeK[tid.y][odd_tile];                                                          \
                    float pos = rope_positions[row];                                                                   \
                    float freq = rope_freqs[even_local >> 1];                                                          \
                    float angle = pos * freq;                                                                          \
                    float cos_val = cos(angle) * uniforms.rope_attn_scale;                                             \
                    float sin_val = sin(angle) * uniforms.rope_attn_scale;                                             \
                    if (apply_rope_q) {                                                                                \
                        float rotated_even_q = even_q * cos_val - odd_q * sin_val;                                     \
                        float rotated_odd_q = even_q * sin_val + odd_q * cos_val;                                      \
                        final_q = (col == even_col) ? rotated_even_q : rotated_odd_q;                                  \
                    }                                                                                                  \
                    if (apply_rope_k) {                                                                                \
                        float rotated_even_k = even_k * cos_val - odd_k * sin_val;                                     \
                        float rotated_odd_k = even_k * sin_val + odd_k * cos_val;                                      \
                        final_k = (col == even_col) ? rotated_even_k : rotated_odd_k;                                  \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
        if (valid_output) {                                                                                            \
            out_q[out_index] = WRITE_FN(final_q);                                                                      \
            out_k[out_index] = WRITE_FN(final_k);                                                                      \
            out_v[out_index] = WRITE_FN(final_v);                                                                      \
        }                                                                                                              \
    }

DEFINE_MATMUL_QKV_SEPARATE_KERNEL(matmul_qkv_separate_f32_nt, float, read_float, write_float)
DEFINE_MATMUL_QKV_SEPARATE_KERNEL(matmul_qkv_separate_f16_nt, half, read_half, write_half)
DEFINE_MATMUL_QKV_SEPARATE_KERNEL(matmul_qkv_separate_bf16_nt, ushort, read_bf16, write_bf16)

#undef DEFINE_MATMUL_QKV_SEPARATE_KERNEL
#undef DEFINE_MATMUL_QKV_KERNEL

#define DEFINE_PACK_QKV_KERNEL(NAME, SCALAR_T, VEC_T, ENABLE_VEC4)                                                     \
    kernel void NAME(                                                                                                  \
        device const SCALAR_T *weight [[buffer(0)]], device SCALAR_T *packed [[buffer(1)]],                            \
        constant PackQKVUniforms &uniforms [[buffer(2)]], uint2 tg_pos [[threadgroup_position_in_grid]],               \
        uint2 tid [[thread_position_in_threadgroup]]                                                                   \
    ) {                                                                                                                \
        if (ENABLE_VEC4 && uniforms.use_vec4 != 0u) {                                                                  \
            uint vec_tile_k = uniforms.tile_k >> 2;                                                                    \
            if (tid.x >= vec_tile_k || tid.y >= uniforms.tile_cols) {                                                  \
                return;                                                                                                \
            }                                                                                                          \
            uint tile_col = tg_pos.x;                                                                                  \
            uint tile_k_block = tg_pos.y;                                                                              \
            uint global_col = tile_col * uniforms.tile_cols + tid.y;                                                   \
            uint global_k = tile_k_block * uniforms.tile_k + tid.x * 4u;                                               \
            if (global_col >= uniforms.segment_rows || (global_k + 3u) >= uniforms.cols) {                             \
                return;                                                                                                \
            }                                                                                                          \
            uint tiles_per_col = uniforms.tiles_per_col == 0u ? 1u : uniforms.tiles_per_col;                           \
            uint tile_index = tile_col * tiles_per_col + tile_k_block;                                                 \
            uint base = tile_index * uniforms.tile_stride;                                                             \
            uint linear = tid.y * uniforms.tile_k + tid.x * 4u;                                                        \
            uint offset_q = base + linear;                                                                             \
            uint offset_k = offset_q + uniforms.tile_section;                                                          \
            uint offset_v = offset_k + uniforms.tile_section;                                                          \
            uint segment_rows = uniforms.segment_rows;                                                                 \
            uint q_row = global_col;                                                                                   \
            uint k_row = global_col + segment_rows;                                                                    \
            uint v_row = global_col + 2u * segment_rows;                                                               \
            const device VEC_T *src_q = (const device VEC_T *)(weight + q_row * uniforms.cols + global_k);             \
            const device VEC_T *src_k = (const device VEC_T *)(weight + k_row * uniforms.cols + global_k);             \
            const device VEC_T *src_v = (const device VEC_T *)(weight + v_row * uniforms.cols + global_k);             \
            device VEC_T *dst_q = (device VEC_T *)(packed + offset_q);                                                 \
            device VEC_T *dst_k = (device VEC_T *)(packed + offset_k);                                                 \
            device VEC_T *dst_v = (device VEC_T *)(packed + offset_v);                                                 \
            *dst_q = *src_q;                                                                                           \
            *dst_k = *src_k;                                                                                           \
            *dst_v = *src_v;                                                                                           \
            return;                                                                                                    \
        }                                                                                                              \
        if (tid.x >= uniforms.tile_k || tid.y >= uniforms.tile_cols) {                                                 \
            return;                                                                                                    \
        }                                                                                                              \
        uint tile_col = tg_pos.x;                                                                                      \
        uint tile_k_block = tg_pos.y;                                                                                  \
        uint global_col = tile_col * uniforms.tile_cols + tid.y;                                                       \
        uint global_k = tile_k_block * uniforms.tile_k + tid.x;                                                        \
        if (global_col >= uniforms.segment_rows || global_k >= uniforms.cols) {                                        \
            return;                                                                                                    \
        }                                                                                                              \
        uint tiles_per_col = uniforms.tiles_per_col == 0u ? 1u : uniforms.tiles_per_col;                               \
        uint tile_index = tile_col * tiles_per_col + tile_k_block;                                                     \
        uint base = tile_index * uniforms.tile_stride;                                                                 \
        uint linear = tid.y * uniforms.tile_k + tid.x;                                                                 \
        uint offset_q = base + linear;                                                                                 \
        uint offset_k = offset_q + uniforms.tile_section;                                                              \
        uint offset_v = offset_k + uniforms.tile_section;                                                              \
        uint segment_rows = uniforms.segment_rows;                                                                     \
        uint q_row = global_col;                                                                                       \
        uint k_row = global_col + segment_rows;                                                                        \
        uint v_row = global_col + 2u * segment_rows;                                                                   \
        packed[offset_q] = weight[q_row * uniforms.cols + global_k];                                                   \
        packed[offset_k] = weight[k_row * uniforms.cols + global_k];                                                   \
        packed[offset_v] = weight[v_row * uniforms.cols + global_k];                                                   \
    }

DEFINE_PACK_QKV_KERNEL(pack_qkv_weight_f32, float, float4, 1)
DEFINE_PACK_QKV_KERNEL(pack_qkv_weight_f16, half, half4, 1)
DEFINE_PACK_QKV_KERNEL(pack_qkv_weight_bf16, ushort, ushort4, 1)

#undef DEFINE_PACK_QKV_KERNEL
#undef DEFINE_MATMUL_QKV_KERNEL
#undef MARMOT_QKV_TILE_M
#undef MARMOT_QKV_TILE_N
#undef MARMOT_QKV_TILE_K
