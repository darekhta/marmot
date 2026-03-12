#include <metal_simdgroup_matrix>
#include <metal_stdlib>

using namespace metal;

#include "common/defines.h"

#include "common/activation_utils.h"
#include "common/quant_blocks.h"
#include "common/quant_decode.h"

typedef struct {
    uint rows;
    uint cols;
    uint input_stride;
    uint values_stride;
    uint indices_stride;
    uint k;
} metal_topk_uniforms_t;

typedef struct {
    uint rows;
    uint cols;
    uint row_stride;
} metal_moe_zero_uniforms_t;

typedef struct {
    uint count;
    uint hidden;
    uint output_rows;
    uint src_stride;
    uint out_stride;
    uint index_stride;
    uint weight_stride;
} metal_moe_scatter_uniforms_t;

typedef struct {
    uint tokens;
    uint experts_per_token;
    uint experts;
    uint id_stride0;
    uint id_stride1;
    uint weight_stride0;
    uint weight_stride1;
    uint renormalize_selected;
    float weights_scale;
} metal_moe_route_uniforms_t;

typedef struct {
    uint route_count;
    uint max_batch;
    uint active_experts;
    uint reserved;
} metal_moe_route_summary_t;

typedef struct {
    uint routes;
    uint input_cols;
    uint output_cols;
    uint input_stride;
    uint output_stride;
    uint weight_blocks;
    uint broadcast_input;
} metal_moe_decode_gate_up_uniforms_t;

typedef struct {
    uint routes;
    uint input_cols;
    uint output_cols;
    uint input_stride;
    uint output_stride;
    uint weight_blocks;
    uint activation;
} metal_moe_decode_down_uniforms_t;

typedef struct {
    uint routes;
    uint input_cols;
    uint output_cols;
    uint input_stride;
    uint output_stride;
    uint weight_blocks;
    uint activation;
    uint output_rows;
} metal_moe_indexed_down_uniforms_t;

constant float metal_topk_neg_inf = -3.402823466e+38f;
constant half metal_topk_neg_inf_f16 = half(-65504.0f);
constant uint metal_topk_small_cols = 256u;
constant int metal_topk_small_index_sentinel = 0x7fffffff;

static inline bool topk_pair_less_ascending(float lhs_value, int lhs_index, float rhs_value, int rhs_index) {
    return lhs_value < rhs_value || (lhs_value == rhs_value && lhs_index > rhs_index);
}

static inline void
topk_compare_swap_ascending(threadgroup float *values, threadgroup int *indices, uint lhs, uint rhs, bool ascending) {
    const float lhs_value = values[lhs];
    const int lhs_index = indices[lhs];
    const float rhs_value = values[rhs];
    const int rhs_index = indices[rhs];
    const bool lhs_less = topk_pair_less_ascending(lhs_value, lhs_index, rhs_value, rhs_index);
    const bool swap = ascending ? !lhs_less : lhs_less;
    if (!swap) {
        return;
    }
    values[lhs] = rhs_value;
    indices[lhs] = rhs_index;
    values[rhs] = lhs_value;
    indices[rhs] = lhs_index;
}

template <typename InputType, typename OutputType>
static inline void topk_last_axis_small_impl(
    device const InputType *input, device OutputType *values_out, device int *indices_out,
    constant metal_topk_uniforms_t &params, threadgroup float *values, threadgroup int *indices, ushort tid,
    uint2 tg_pos
) {
    if (tg_pos.y >= params.rows || params.cols > metal_topk_small_cols || params.k > 16u) {
        return;
    }

    const uint row = tg_pos.y;
    const uint input_base = row * params.input_stride;
    const uint values_base = row * params.values_stride;
    const uint indices_base = row * params.indices_stride;

    if (uint(tid) < metal_topk_small_cols) {
        if (uint(tid) < params.cols) {
            values[tid] = float(input[input_base + uint(tid)]);
            indices[tid] = int(tid);
        } else {
            values[tid] = metal_topk_neg_inf;
            indices[tid] = metal_topk_small_index_sentinel;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint size = 2u; size <= metal_topk_small_cols; size <<= 1u) {
        for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
            const uint lane = uint(tid);
            const uint peer = lane ^ stride;
            if (peer > lane && lane < metal_topk_small_cols) {
                const bool ascending = (lane & size) == 0u;
                topk_compare_swap_ascending(values, indices, lane, peer, ascending);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (uint(tid) < params.k) {
        const uint src = metal_topk_small_cols - 1u - uint(tid);
        values_out[values_base + uint(tid)] = OutputType(values[src]);
        indices_out[indices_base + uint(tid)] = indices[src];
    }
}

kernel void topk_f32_last_axis_small(
    device const float *input [[buffer(0)]], device float *values_out [[buffer(1)]],
    device int *indices_out [[buffer(2)]], constant metal_topk_uniforms_t &params [[buffer(3)]],
    threadgroup float *values [[threadgroup(0)]], threadgroup int *indices [[threadgroup(1)]],
    ushort tid [[thread_index_in_threadgroup]], uint2 tg_pos [[threadgroup_position_in_grid]]
) {
    topk_last_axis_small_impl<float, float>(input, values_out, indices_out, params, values, indices, tid, tg_pos);
}

kernel void topk_f16_last_axis_small(
    device const half *input [[buffer(0)]], device half *values_out [[buffer(1)]],
    device int *indices_out [[buffer(2)]], constant metal_topk_uniforms_t &params [[buffer(3)]],
    threadgroup float *values [[threadgroup(0)]], threadgroup int *indices [[threadgroup(1)]],
    ushort tid [[thread_index_in_threadgroup]], uint2 tg_pos [[threadgroup_position_in_grid]]
) {
    topk_last_axis_small_impl<half, half>(input, values_out, indices_out, params, values, indices, tid, tg_pos);
}

kernel void topk_f32_last_axis(
    device const float *input [[buffer(0)]], device float *values_out [[buffer(1)]],
    device int *indices_out [[buffer(2)]], constant metal_topk_uniforms_t &params [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= params.rows) {
        return;
    }

    const uint input_base = row * params.input_stride;
    const uint values_base = row * params.values_stride;
    const uint indices_base = row * params.indices_stride;

    for (uint i = 0; i < params.k; ++i) {
        values_out[values_base + i] = metal_topk_neg_inf;
        indices_out[indices_base + i] = -1;
    }

    for (uint col = 0; col < params.cols; ++col) {
        const float value = input[input_base + col];
        uint insert_pos = params.k;
        for (uint i = 0; i < params.k; ++i) {
            const float best_value = values_out[values_base + i];
            const int best_index = indices_out[indices_base + i];
            if (value > best_value || (value == best_value && int(col) < best_index)) {
                insert_pos = i;
                break;
            }
        }
        if (insert_pos == params.k) {
            continue;
        }

        for (uint i = params.k; i > insert_pos + 1; --i) {
            values_out[values_base + i - 1] = values_out[values_base + i - 2];
            indices_out[indices_base + i - 1] = indices_out[indices_base + i - 2];
        }

        values_out[values_base + insert_pos] = value;
        indices_out[indices_base + insert_pos] = int(col);
    }
}

kernel void topk_f16_last_axis(
    device const half *input [[buffer(0)]], device half *values_out [[buffer(1)]],
    device int *indices_out [[buffer(2)]], constant metal_topk_uniforms_t &params [[buffer(3)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= params.rows) {
        return;
    }

    const uint input_base = row * params.input_stride;
    const uint values_base = row * params.values_stride;
    const uint indices_base = row * params.indices_stride;

    for (uint i = 0; i < params.k; ++i) {
        values_out[values_base + i] = metal_topk_neg_inf_f16;
        indices_out[indices_base + i] = -1;
    }

    for (uint col = 0; col < params.cols; ++col) {
        const half value = input[input_base + col];
        uint insert_pos = params.k;
        for (uint i = 0; i < params.k; ++i) {
            const half best_value = values_out[values_base + i];
            const int best_index = indices_out[indices_base + i];
            if (float(value) > float(best_value) || (value == best_value && int(col) < best_index)) {
                insert_pos = i;
                break;
            }
        }
        if (insert_pos == params.k) {
            continue;
        }

        for (uint i = params.k; i > insert_pos + 1; --i) {
            values_out[values_base + i - 1] = values_out[values_base + i - 2];
            indices_out[indices_base + i - 1] = indices_out[indices_base + i - 2];
        }

        values_out[values_base + insert_pos] = value;
        indices_out[indices_base + insert_pos] = int(col);
    }
}

kernel void moe_zero_f32_2d(
    device float *out [[buffer(0)]], constant metal_moe_zero_uniforms_t &params [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    const uint total = params.rows * params.cols;
    if (id >= total) {
        return;
    }

    const uint row = id / params.cols;
    const uint col = id - row * params.cols;
    out[row * params.row_stride + col] = 0.0f;
}

kernel void moe_zero_f16_2d(
    device half *out [[buffer(0)]], constant metal_moe_zero_uniforms_t &params [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    const uint total = params.rows * params.cols;
    if (id >= total) {
        return;
    }

    const uint row = id / params.cols;
    const uint col = id - row * params.cols;
    out[row * params.row_stride + col] = half(0.0f);
}

kernel void moe_scatter_add_f32(
    device const float *src [[buffer(0)]], device const int *indices [[buffer(1)]],
    device const float *weights [[buffer(2)]], device float *out [[buffer(3)]],
    constant metal_moe_scatter_uniforms_t &params [[buffer(4)]], uint id [[thread_position_in_grid]]
) {
    const uint total = params.count * params.hidden;
    if (id >= total) {
        return;
    }

    const uint row = id / params.hidden;
    const uint col = id - row * params.hidden;
    const int token = indices[row * params.index_stride];
    if (token < 0 || uint(token) >= params.output_rows) {
        return;
    }

    const float weight = weights[row * params.weight_stride];
    out[uint(token) * params.out_stride + col] += src[row * params.src_stride + col] * weight;
}

kernel void moe_scatter_add_atomic_f32(
    device const float *src [[buffer(0)]], device const int *indices [[buffer(1)]],
    device const float *weights [[buffer(2)]], device atomic_float *out [[buffer(3)]],
    constant metal_moe_scatter_uniforms_t &params [[buffer(4)]], uint id [[thread_position_in_grid]]
) {
    const uint total = params.count * params.hidden;
    if (id >= total) {
        return;
    }

    const uint row = id / params.hidden;
    const uint col = id - row * params.hidden;
    const int token = indices[row * params.index_stride];
    if (token < 0 || uint(token) >= params.output_rows) {
        return;
    }

    const float weight = weights[row * params.weight_stride];
    const float value = src[row * params.src_stride + col] * weight;
    atomic_fetch_add_explicit(&out[uint(token) * params.out_stride + col], value, memory_order_relaxed);
}

kernel void moe_scatter_add_f16(
    device const half *src [[buffer(0)]], device const int *indices [[buffer(1)]],
    device const half *weights [[buffer(2)]], device half *out [[buffer(3)]],
    constant metal_moe_scatter_uniforms_t &params [[buffer(4)]], uint id [[thread_position_in_grid]]
) {
    const uint total = params.count * params.hidden;
    if (id >= total) {
        return;
    }

    const uint row = id / params.hidden;
    const uint col = id - row * params.hidden;
    const int token = indices[row * params.index_stride];
    if (token < 0 || uint(token) >= params.output_rows) {
        return;
    }

    const half weight = weights[row * params.weight_stride];
    out[uint(token) * params.out_stride + col] += src[row * params.src_stride + col] * weight;
}

template <typename InputType>
static inline float moe_decode_gate_up_load_scalar(device const InputType *input_row, uint gk, uint input_cols) {
    if (gk >= input_cols) {
        return 0.0f;
    }
    return float(input_row[gk]);
}

static inline float moe_decode_gate_up_q4_k_dot_block(
    const device q4_k_block &blk, thread const float *yl, thread const float *yh, float4 sumy, uint iq, uint ir
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const device ushort *sc = (device const ushort *)blk.scales + iq;
    ushort sc16[4];
    thread const uchar *sc8 = (thread const uchar *)sc16;

    sc16[0] = sc[0] & kmask1;
    sc16[1] = sc[2] & kmask1;
    sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
    sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

    const device ushort *q1 = (device const ushort *)blk.qs + 16u * iq + 4u * ir;
    const device ushort *q2 = q1 + 32;

    float4 acc1 = 0.0f;
    float4 acc2 = 0.0f;

#pragma clang loop unroll(full)
    for (ushort i = 0; i < 4; ++i) {
        const ushort q1v = q1[i];
        const ushort q2v = q2[i];

        acc1[0] += yl[2u * i + 0u] * float(q1v & 0x000Fu);
        acc1[1] += yl[2u * i + 1u] * float(q1v & 0x0F00u);
        acc1[2] += yl[2u * i + 8u] * float(q1v & 0x00F0u);
        acc1[3] += yl[2u * i + 9u] * float(q1v & 0xF000u);

        acc2[0] += yh[2u * i + 0u] * float(q2v & 0x000Fu);
        acc2[1] += yh[2u * i + 1u] * float(q2v & 0x0F00u);
        acc2[2] += yh[2u * i + 8u] * float(q2v & 0x00F0u);
        acc2[3] += yh[2u * i + 9u] * float(q2v & 0xF000u);
    }

    const float2 dm = float2(*((device const half2 *)&blk.d));
    const float d = dm[0];
    const float dmin = dm[1];
    const float inv256 = 1.0f / 256.0f;
    const float inv16 = 1.0f / 16.0f;

    const float acc = (acc1[0] + inv256 * acc1[1]) * float(sc8[0]) +
        (acc1[2] + inv256 * acc1[3]) * float(sc8[1]) * inv16 + (acc2[0] + inv256 * acc2[1]) * float(sc8[4]) +
        (acc2[2] + inv256 * acc2[3]) * float(sc8[5]) * inv16;
    const float min_acc =
        sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) + sumy[2] * float(sc8[6]) + sumy[3] * float(sc8[7]);

    return d * acc - dmin * min_acc;
}

template <typename InputType>
static inline void moe_decode_gate_up_q4_k_route_accumulate(
    const device q4_k_block *gate_row0, const device q4_k_block *gate_row1, const device q4_k_block *up_row0,
    const device q4_k_block *up_row1, bool has_m1, device const InputType *input_row, uint input_cols,
    uint weight_blocks, ushort tiisg, thread float &gate_sum0, thread float &gate_sum1, thread float &up_sum0,
    thread float &up_sum1
) {
    const uint tid = uint(tiisg);
    const uint ix = tid >> 3;
    const uint it = tid & 7u;
    const uint iq = it >> 2;
    const uint ir = it & 3u;

    float yl[16];
    float yh[16];
    const bool full_k = (input_cols & (kQK_K - 1u)) == 0u;
    if (full_k) {
        const uint nb = min(weight_blocks, input_cols / kQK_K);
        device const InputType *y4 = input_row + ix * kQK_K + 64u * iq + 8u * ir;
        const device q4_k_block *gate0 = gate_row0 + ix;
        const device q4_k_block *gate1 = gate_row1 + ix;
        const device q4_k_block *up0 = up_row0 + ix;
        const device q4_k_block *up1 = up_row1 + ix;

        for (uint ib = ix; ib < nb; ib += 4u) {
            float4 sumy = 0.0f;
            for (uint i = 0; i < 8u; ++i) {
                const float v0 = float(y4[i + 0u]);
                const float v1 = float(y4[i + 32u]);
                const float v2 = float(y4[i + 128u]);
                const float v3 = float(y4[i + 160u]);

                yl[i + 0u] = v0;
                yl[i + 8u] = v1;
                yh[i + 0u] = v2;
                yh[i + 8u] = v3;

                sumy[0] += v0;
                sumy[1] += v1;
                sumy[2] += v2;
                sumy[3] += v3;
            }

            gate_sum0 += moe_decode_gate_up_q4_k_dot_block(*gate0, yl, yh, sumy, iq, ir);
            up_sum0 += moe_decode_gate_up_q4_k_dot_block(*up0, yl, yh, sumy, iq, ir);
            if (has_m1) {
                gate_sum1 += moe_decode_gate_up_q4_k_dot_block(*gate1, yl, yh, sumy, iq, ir);
                up_sum1 += moe_decode_gate_up_q4_k_dot_block(*up1, yl, yh, sumy, iq, ir);
            }
            y4 += 4u * kQK_K;
            gate0 += 4u;
            gate1 += 4u;
            up0 += 4u;
            up1 += 4u;
        }
    } else {
        for (uint sb = ix; sb < weight_blocks; sb += 4u) {
            const uint block_start = sb * kQK_K;
            if (block_start >= input_cols) {
                break;
            }

            float4 sumy = 0.0f;
            const uint y_base = block_start + 64u * iq + 8u * ir;
            for (uint i = 0; i < 8u; ++i) {
                const float v0 = moe_decode_gate_up_load_scalar(input_row, y_base + i + 0u, input_cols);
                const float v1 = moe_decode_gate_up_load_scalar(input_row, y_base + i + 32u, input_cols);
                const float v2 = moe_decode_gate_up_load_scalar(input_row, y_base + i + 128u, input_cols);
                const float v3 = moe_decode_gate_up_load_scalar(input_row, y_base + i + 160u, input_cols);

                yl[i + 0u] = v0;
                yl[i + 8u] = v1;
                yh[i + 0u] = v2;
                yh[i + 8u] = v3;

                sumy[0] += v0;
                sumy[1] += v1;
                sumy[2] += v2;
                sumy[3] += v3;
            }

            gate_sum0 += moe_decode_gate_up_q4_k_dot_block(gate_row0[sb], yl, yh, sumy, iq, ir);
            up_sum0 += moe_decode_gate_up_q4_k_dot_block(up_row0[sb], yl, yh, sumy, iq, ir);
            if (has_m1) {
                gate_sum1 += moe_decode_gate_up_q4_k_dot_block(gate_row1[sb], yl, yh, sumy, iq, ir);
                up_sum1 += moe_decode_gate_up_q4_k_dot_block(up_row1[sb], yl, yh, sumy, iq, ir);
            }
        }
    }
}

template <typename InputType, typename OutputType>
static inline void moe_decode_gate_up_q4_k_impl(
    device const q4_k_block *gate_weight, device const q4_k_block *up_weight, device const InputType *input,
    device const int *route_experts, device OutputType *gate_out, device OutputType *up_out,
    constant metal_moe_decode_gate_up_uniforms_t &params, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint nr0 = 2u;
    constexpr uint nsg = 2u;
    constexpr uint rows_per_threadgroup = nr0 * nsg;

    const uint route = tgp.y;
    if (route >= params.routes) {
        return;
    }

    const int expert = route_experts[route];
    if (expert < 0) {
        return;
    }

    const uint m_base = tgp.x * rows_per_threadgroup + uint(sgitg) * nr0;
    if (m_base >= params.output_cols) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < params.output_cols;
    const uint expert_stride_blocks = params.output_cols * params.weight_blocks;

    device const InputType *input_row = params.broadcast_input != 0u ? input : input + route * params.input_stride;
    device const q4_k_block *gate_expert = gate_weight + uint(expert) * expert_stride_blocks;
    device const q4_k_block *up_expert = up_weight + uint(expert) * expert_stride_blocks;

    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    moe_decode_gate_up_q4_k_route_accumulate<InputType>(
        gate_expert + m0 * params.weight_blocks, gate_expert + m1 * params.weight_blocks,
        up_expert + m0 * params.weight_blocks, up_expert + m1 * params.weight_blocks, has_m1, input_row,
        params.input_cols, params.weight_blocks, tiisg, gate_sum0, gate_sum1, up_sum0, up_sum1
    );

    const float gate_out0 = simd_sum(gate_sum0);
    const float gate_out1 = simd_sum(gate_sum1);
    const float up_out0 = simd_sum(up_sum0);
    const float up_out1 = simd_sum(up_sum1);
    if (tiisg == 0) {
        gate_out[route * params.output_stride + m0] = OutputType(gate_out0);
        up_out[route * params.output_stride + m0] = OutputType(up_out0);
        if (has_m1) {
            gate_out[route * params.output_stride + m1] = OutputType(gate_out1);
            up_out[route * params.output_stride + m1] = OutputType(up_out1);
        }
    }
}

kernel void moe_decode_gate_up_q4_k_f32_f32(
    device const q4_k_block *gate_weight [[buffer(0)]], device const q4_k_block *up_weight [[buffer(1)]],
    device const float *input [[buffer(2)]], device const int *route_experts [[buffer(3)]],
    device float *gate_out [[buffer(4)]], device float *up_out [[buffer(5)]],
    constant metal_moe_decode_gate_up_uniforms_t &params [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_gate_up_q4_k_impl<float, float>(
        gate_weight, up_weight, input, route_experts, gate_out, up_out, params, tgp, tiisg, sgitg
    );
}

kernel void moe_decode_gate_up_q4_k_f16_f32(
    device const q4_k_block *gate_weight [[buffer(0)]], device const q4_k_block *up_weight [[buffer(1)]],
    device const half *input [[buffer(2)]], device const int *route_experts [[buffer(3)]],
    device float *gate_out [[buffer(4)]], device float *up_out [[buffer(5)]],
    constant metal_moe_decode_gate_up_uniforms_t &params [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_gate_up_q4_k_impl<half, float>(
        gate_weight, up_weight, input, route_experts, gate_out, up_out, params, tgp, tiisg, sgitg
    );
}

kernel void moe_decode_gate_up_q4_k_f16_f16(
    device const q4_k_block *gate_weight [[buffer(0)]], device const q4_k_block *up_weight [[buffer(1)]],
    device const half *input [[buffer(2)]], device const int *route_experts [[buffer(3)]],
    device half *gate_out [[buffer(4)]], device half *up_out [[buffer(5)]],
    constant metal_moe_decode_gate_up_uniforms_t &params [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_gate_up_q4_k_impl<half, half>(
        gate_weight, up_weight, input, route_experts, gate_out, up_out, params, tgp, tiisg, sgitg
    );
}

template <typename InputType>
static inline float moe_decode_down_load_scalar(device const InputType *input_row, uint gk, uint input_cols) {
    if (gk >= input_cols) {
        return 0.0f;
    }
    return float(input_row[gk]);
}

template <typename InputType>
static inline void moe_decode_down_q6_k_route_accumulate(
    const device q6_k_block *weight_row0, const device q6_k_block *weight_row1, bool has_m1,
    device const InputType *input_row, uint input_cols, uint weight_blocks, ushort tiisg, thread float &sum0,
    thread float &sum1
) {
    const bool full_k = (input_cols & (kQK_K - 1u)) == 0u;
    if (full_k) {
        constexpr uchar kmask1 = 0x03;
        constexpr uchar kmask2 = 0x0C;
        constexpr uchar kmask3 = 0x30;
        constexpr uchar kmask4 = 0xC0;

        const ushort tid = tiisg >> 1;
        const ushort ix = tiisg & 1;
        const ushort ip = tid >> 3;
        const ushort il = tid & 7;
        const ushort l0 = il * 4;
        const ushort is = 8 * ip + l0 / 16;

        const ushort y_offset = 128 * ip + l0;
        const ushort q_offset_l = 64 * ip + l0;
        const ushort q_offset_h = 32 * ip + l0;

        const uint nb = min(weight_blocks, input_cols / kQK_K);
        device const InputType *y = input_row + uint(ix) * kQK_K + uint(y_offset);
        float yl[16];

        for (uint ib = uint(ix); ib < nb; ib += 2u) {
#pragma clang loop unroll(full)
            for (ushort l = 0; l < 4; ++l) {
                yl[4u * uint(l) + 0u] = float(y[l + 0]);
                yl[4u * uint(l) + 1u] = float(y[l + 32]);
                yl[4u * uint(l) + 2u] = float(y[l + 64]);
                yl[4u * uint(l) + 3u] = float(y[l + 96]);
            }

            {
                const device q6_k_block &blk = weight_row0[ib];
                const device uchar *q1 = blk.ql + q_offset_l;
                const device uchar *q2 = q1 + 32;
                const device uchar *qh = blk.qh + q_offset_h;
                const device char *sc = blk.scales + is;
                float4 sums = 0.0f;
#pragma clang loop unroll(full)
                for (ushort l = 0; l < 4; ++l) {
                    sums[0] += yl[4u * uint(l) + 0u] * float(int(char((q1[l] & 0xFu) | ((qh[l] & kmask1) << 4))) - 32);
                    sums[1] += yl[4u * uint(l) + 1u] * float(int(char((q2[l] & 0xFu) | ((qh[l] & kmask2) << 2))) - 32);
                    sums[2] += yl[4u * uint(l) + 2u] * float(int(char((q1[l] >> 4) | ((qh[l] & kmask3) << 0))) - 32);
                    sums[3] += yl[4u * uint(l) + 3u] * float(int(char((q2[l] >> 4) | ((qh[l] & kmask4) >> 2))) - 32);
                }
                sum0 += float(blk.d) *
                    (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
            }

            if (has_m1) {
                const device q6_k_block &blk = weight_row1[ib];
                const device uchar *q1 = blk.ql + q_offset_l;
                const device uchar *q2 = q1 + 32;
                const device uchar *qh = blk.qh + q_offset_h;
                const device char *sc = blk.scales + is;
                float4 sums = 0.0f;
#pragma clang loop unroll(full)
                for (ushort l = 0; l < 4; ++l) {
                    sums[0] += yl[4u * uint(l) + 0u] * float(int(char((q1[l] & 0xFu) | ((qh[l] & kmask1) << 4))) - 32);
                    sums[1] += yl[4u * uint(l) + 1u] * float(int(char((q2[l] & 0xFu) | ((qh[l] & kmask2) << 2))) - 32);
                    sums[2] += yl[4u * uint(l) + 2u] * float(int(char((q1[l] >> 4) | ((qh[l] & kmask3) << 0))) - 32);
                    sums[3] += yl[4u * uint(l) + 3u] * float(int(char((q2[l] >> 4) | ((qh[l] & kmask4) >> 2))) - 32);
                }
                sum1 += float(blk.d) *
                    (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
            }

            y += 2u * kQK_K;
        }
    } else {
        const uint lane = uint(tiisg);
        const uint q_offset = lane << 3;
        const uint chunk = q_offset >> 7;
        const uint local = q_offset & 127u;
        const uint segment = local >> 5;
        const bool high_nibble = (segment & 2u) != 0u;
        const uint l_base = local & 31u;
        const uint ql_base = chunk * 64u + ((segment & 1u) != 0u ? 32u : 0u) + l_base;
        const uint qh_base = chunk * 32u + l_base;
        const uint shift_h = segment * 2u;
        const uint is = l_base >> 4;

        for (uint sb = 0; sb < weight_blocks; ++sb) {
            const uint block_start = sb * kQK_K;
            if (block_start >= input_cols) {
                break;
            }

            float4 a0 = 0.0f;
            float4 a1 = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                a0[i] = moe_decode_down_load_scalar(input_row, block_start + q_offset + i, input_cols);
                a1[i] = moe_decode_down_load_scalar(input_row, block_start + q_offset + 4u + i, input_cols);
            }

            {
                const device q6_k_block &blk = weight_row0[sb];
                const int scale_i8 = int(blk.scales[chunk * 8u + segment * 2u + is]);
                const float dl = float(blk.d) * float(scale_i8);
                const device uchar *ql = blk.ql + ql_base;
                const device uchar *qh = blk.qh + qh_base;
                float4 qv0;
                float4 qv1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uchar qlb = ql[i];
                    const uchar qhb = qh[i];
                    const uint lo4 = high_nibble ? uint(qlb >> 4) : uint(qlb & 0x0Fu);
                    const uint hi2 = (uint(qhb) >> shift_h) & 0x3u;
                    const int q6 = int(lo4 | (hi2 << 4u)) - 32;
                    const float qf = float(q6);
                    if (i < 4u) {
                        qv0[i] = qf;
                    } else {
                        qv1[i - 4u] = qf;
                    }
                }
                sum0 += dl * (dot(qv0, a0) + dot(qv1, a1));
            }

            if (has_m1) {
                const device q6_k_block &blk = weight_row1[sb];
                const int scale_i8 = int(blk.scales[chunk * 8u + segment * 2u + is]);
                const float dl = float(blk.d) * float(scale_i8);
                const device uchar *ql = blk.ql + ql_base;
                const device uchar *qh = blk.qh + qh_base;
                float4 qv0;
                float4 qv1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uchar qlb = ql[i];
                    const uchar qhb = qh[i];
                    const uint lo4 = high_nibble ? uint(qlb >> 4) : uint(qlb & 0x0Fu);
                    const uint hi2 = (uint(qhb) >> shift_h) & 0x3u;
                    const int q6 = int(lo4 | (hi2 << 4u)) - 32;
                    const float qf = float(q6);
                    if (i < 4u) {
                        qv0[i] = qf;
                    } else {
                        qv1[i - 4u] = qf;
                    }
                }
                sum1 += dl * (dot(qv0, a0) + dot(qv1, a1));
            }
        }
    }
}

static inline float moe_decode_apply_glu(float gate, float up, uint activation) {
    switch (activation) {
    case MARMOT_DEVICE_BINARY_GEGLU:
        return gelu_tanh_exact(gate) * up;
    case MARMOT_DEVICE_BINARY_SWIGLU:
    default:
        return silu_exact(gate) * up;
    }
}

template <typename InputType>
static inline float moe_decode_down_load_glu_scalar(
    device const InputType *gate_row, device const InputType *up_row, uint activation, uint gk, uint input_cols
) {
    if (gk >= input_cols) {
        return 0.0f;
    }
    return moe_decode_apply_glu(float(gate_row[gk]), float(up_row[gk]), activation);
}

template <typename InputType, typename WeightType, typename OutputType>
static inline void moe_decode_down_q6_k_impl(
    device const q6_k_block *weight, device const InputType *input, device const int *route_experts,
    device const WeightType *route_weights, device OutputType *output,
    constant metal_moe_decode_down_uniforms_t &params, uint tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint nr0 = 2u;
    constexpr uint nsg = 2u;
    constexpr uint rows_per_threadgroup = nr0 * nsg;

    const uint m_base = tgp * rows_per_threadgroup + uint(sgitg) * nr0;
    if (m_base >= params.output_cols) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < params.output_cols;
    const uint expert_stride_blocks = params.output_cols * params.weight_blocks;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint route = 0; route < params.routes; ++route) {
        const int expert = route_experts[route];
        if (expert < 0) {
            continue;
        }
        const float route_scale = float(route_weights[route]);
        if (route_scale == 0.0f) {
            continue;
        }

        device const InputType *input_row = input + route * params.input_stride;
        device const q6_k_block *expert_weight = weight + uint(expert) * expert_stride_blocks;
        float route_sum0 = 0.0f;
        float route_sum1 = 0.0f;
        moe_decode_down_q6_k_route_accumulate<InputType>(
            expert_weight + m0 * params.weight_blocks, expert_weight + m1 * params.weight_blocks, has_m1, input_row,
            params.input_cols, params.weight_blocks, tiisg, route_sum0, route_sum1
        );
        sum0 += route_scale * route_sum0;
        sum1 += route_scale * route_sum1;
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output[m0] = OutputType(out0);
        if (has_m1) {
            output[m1] = OutputType(out1);
        }
    }
}

kernel void moe_decode_down_q6_k_f32_f32(
    device const q6_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device const int *route_experts [[buffer(2)]], device const float *route_weights [[buffer(3)]],
    device float *output [[buffer(4)]], constant metal_moe_decode_down_uniforms_t &params [[buffer(5)]],
    uint tgp [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_down_q6_k_impl<float, float, float>(
        weight, input, route_experts, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_decode_down_q6_k_f16_f32(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device const int *route_experts [[buffer(2)]], device const half *route_weights [[buffer(3)]],
    device float *output [[buffer(4)]], constant metal_moe_decode_down_uniforms_t &params [[buffer(5)]],
    uint tgp [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_down_q6_k_impl<half, half, float>(
        weight, input, route_experts, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_decode_down_q6_k_f16_f16(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device const int *route_experts [[buffer(2)]], device const half *route_weights [[buffer(3)]],
    device half *output [[buffer(4)]], constant metal_moe_decode_down_uniforms_t &params [[buffer(5)]],
    uint tgp [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_down_q6_k_impl<half, half, half>(
        weight, input, route_experts, route_weights, output, params, tgp, tiisg, sgitg
    );
}

template <typename InputType>
static inline void moe_decode_down_q6_k_route_accumulate_glu(
    const device q6_k_block *weight_row0, const device q6_k_block *weight_row1, bool has_m1,
    device const InputType *gate_row, device const InputType *up_row, uint activation, uint input_cols,
    uint weight_blocks, ushort tiisg, thread float &sum0, thread float &sum1
) {
    const bool full_k = (input_cols & (kQK_K - 1u)) == 0u;
    if (full_k) {
        constexpr uchar kmask1 = 0x03;
        constexpr uchar kmask2 = 0x0C;
        constexpr uchar kmask3 = 0x30;
        constexpr uchar kmask4 = 0xC0;

        const ushort tid = tiisg >> 1;
        const ushort ix = tiisg & 1;
        const ushort ip = tid >> 3;
        const ushort il = tid & 7;
        const ushort l0 = il * 4;
        const ushort is = 8 * ip + l0 / 16;

        const ushort y_offset = 128 * ip + l0;
        const ushort q_offset_l = 64 * ip + l0;
        const ushort q_offset_h = 32 * ip + l0;

        const uint nb = min(weight_blocks, input_cols / kQK_K);
        device const InputType *gate_y = gate_row + uint(ix) * kQK_K + uint(y_offset);
        device const InputType *up_y = up_row + uint(ix) * kQK_K + uint(y_offset);
        float yl[16];

        for (uint ib = uint(ix); ib < nb; ib += 2u) {
#pragma clang loop unroll(full)
            for (ushort l = 0; l < 4; ++l) {
                yl[4u * uint(l) + 0u] = moe_decode_apply_glu(float(gate_y[l + 0]), float(up_y[l + 0]), activation);
                yl[4u * uint(l) + 1u] = moe_decode_apply_glu(float(gate_y[l + 32]), float(up_y[l + 32]), activation);
                yl[4u * uint(l) + 2u] = moe_decode_apply_glu(float(gate_y[l + 64]), float(up_y[l + 64]), activation);
                yl[4u * uint(l) + 3u] = moe_decode_apply_glu(float(gate_y[l + 96]), float(up_y[l + 96]), activation);
            }

            {
                const device q6_k_block &blk = weight_row0[ib];
                const device uchar *q1 = blk.ql + q_offset_l;
                const device uchar *q2 = q1 + 32;
                const device uchar *qh = blk.qh + q_offset_h;
                const device char *sc = blk.scales + is;
                float4 sums = 0.0f;
#pragma clang loop unroll(full)
                for (ushort l = 0; l < 4; ++l) {
                    sums[0] += yl[4u * uint(l) + 0u] * float(int(char((q1[l] & 0xFu) | ((qh[l] & kmask1) << 4))) - 32);
                    sums[1] += yl[4u * uint(l) + 1u] * float(int(char((q2[l] & 0xFu) | ((qh[l] & kmask2) << 2))) - 32);
                    sums[2] += yl[4u * uint(l) + 2u] * float(int(char((q1[l] >> 4) | ((qh[l] & kmask3) << 0))) - 32);
                    sums[3] += yl[4u * uint(l) + 3u] * float(int(char((q2[l] >> 4) | ((qh[l] & kmask4) >> 2))) - 32);
                }
                sum0 += float(blk.d) *
                    (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
            }

            if (has_m1) {
                const device q6_k_block &blk = weight_row1[ib];
                const device uchar *q1 = blk.ql + q_offset_l;
                const device uchar *q2 = q1 + 32;
                const device uchar *qh = blk.qh + q_offset_h;
                const device char *sc = blk.scales + is;
                float4 sums = 0.0f;
#pragma clang loop unroll(full)
                for (ushort l = 0; l < 4; ++l) {
                    sums[0] += yl[4u * uint(l) + 0u] * float(int(char((q1[l] & 0xFu) | ((qh[l] & kmask1) << 4))) - 32);
                    sums[1] += yl[4u * uint(l) + 1u] * float(int(char((q2[l] & 0xFu) | ((qh[l] & kmask2) << 2))) - 32);
                    sums[2] += yl[4u * uint(l) + 2u] * float(int(char((q1[l] >> 4) | ((qh[l] & kmask3) << 0))) - 32);
                    sums[3] += yl[4u * uint(l) + 3u] * float(int(char((q2[l] >> 4) | ((qh[l] & kmask4) >> 2))) - 32);
                }
                sum1 += float(blk.d) *
                    (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
            }

            gate_y += 2u * kQK_K;
            up_y += 2u * kQK_K;
        }
    } else {
        const uint lane = uint(tiisg);
        const uint q_offset = lane << 3;
        const uint chunk = q_offset >> 7;
        const uint local = q_offset & 127u;
        const uint segment = local >> 5;
        const bool high_nibble = (segment & 2u) != 0u;
        const uint l_base = local & 31u;
        const uint ql_base = chunk * 64u + ((segment & 1u) != 0u ? 32u : 0u) + l_base;
        const uint qh_base = chunk * 32u + l_base;
        const uint shift_h = segment * 2u;
        const uint is = l_base >> 4;

        for (uint sb = 0; sb < weight_blocks; ++sb) {
            const uint block_start = sb * kQK_K;
            if (block_start >= input_cols) {
                break;
            }

            float4 a0 = 0.0f;
            float4 a1 = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4u; ++i) {
                a0[i] = moe_decode_down_load_glu_scalar(
                    gate_row, up_row, activation, block_start + q_offset + i, input_cols
                );
                a1[i] = moe_decode_down_load_glu_scalar(
                    gate_row, up_row, activation, block_start + q_offset + 4u + i, input_cols
                );
            }

            {
                const device q6_k_block &blk = weight_row0[sb];
                const int scale_i8 = int(blk.scales[chunk * 8u + segment * 2u + is]);
                const float dl = float(blk.d) * float(scale_i8);
                const device uchar *ql = blk.ql + ql_base;
                const device uchar *qh = blk.qh + qh_base;
                float4 qv0;
                float4 qv1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uchar qlb = ql[i];
                    const uchar qhb = qh[i];
                    const uint lo4 = high_nibble ? uint(qlb >> 4) : uint(qlb & 0x0Fu);
                    const uint hi2 = (uint(qhb) >> shift_h) & 0x3u;
                    const int q6 = int(lo4 | (hi2 << 4u)) - 32;
                    const float qf = float(q6);
                    if (i < 4u) {
                        qv0[i] = qf;
                    } else {
                        qv1[i - 4u] = qf;
                    }
                }
                sum0 += dl * (dot(qv0, a0) + dot(qv1, a1));
            }

            if (has_m1) {
                const device q6_k_block &blk = weight_row1[sb];
                const int scale_i8 = int(blk.scales[chunk * 8u + segment * 2u + is]);
                const float dl = float(blk.d) * float(scale_i8);
                const device uchar *ql = blk.ql + ql_base;
                const device uchar *qh = blk.qh + qh_base;
                float4 qv0;
                float4 qv1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uchar qlb = ql[i];
                    const uchar qhb = qh[i];
                    const uint lo4 = high_nibble ? uint(qlb >> 4) : uint(qlb & 0x0Fu);
                    const uint hi2 = (uint(qhb) >> shift_h) & 0x3u;
                    const int q6 = int(lo4 | (hi2 << 4u)) - 32;
                    const float qf = float(q6);
                    if (i < 4u) {
                        qv0[i] = qf;
                    } else {
                        qv1[i - 4u] = qf;
                    }
                }
                sum1 += dl * (dot(qv0, a0) + dot(qv1, a1));
            }
        }
    }
}

template <typename InputType, typename WeightType, typename OutputType>
static inline void moe_decode_glu_down_q6_k_impl(
    device const q6_k_block *weight, device const InputType *gate, device const InputType *up,
    device const int *route_experts, device const WeightType *route_weights, device OutputType *output,
    constant metal_moe_decode_down_uniforms_t &params, uint tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint nr0 = 2u;
    constexpr uint nsg = 2u;
    constexpr uint rows_per_threadgroup = nr0 * nsg;

    const uint m_base = tgp * rows_per_threadgroup + uint(sgitg) * nr0;
    if (m_base >= params.output_cols) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < params.output_cols;
    const uint expert_stride_blocks = params.output_cols * params.weight_blocks;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint route = 0; route < params.routes; ++route) {
        const int expert = route_experts[route];
        if (expert < 0) {
            continue;
        }
        const float route_scale = float(route_weights[route]);
        if (route_scale == 0.0f) {
            continue;
        }

        device const InputType *gate_row = gate + route * params.input_stride;
        device const InputType *up_row = up + route * params.input_stride;
        device const q6_k_block *expert_weight = weight + uint(expert) * expert_stride_blocks;
        float route_sum0 = 0.0f;
        float route_sum1 = 0.0f;
        moe_decode_down_q6_k_route_accumulate_glu<InputType>(
            expert_weight + m0 * params.weight_blocks, expert_weight + m1 * params.weight_blocks, has_m1, gate_row,
            up_row, params.activation, params.input_cols, params.weight_blocks, tiisg, route_sum0, route_sum1
        );
        sum0 += route_scale * route_sum0;
        sum1 += route_scale * route_sum1;
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output[m0] = OutputType(out0);
        if (has_m1) {
            output[m1] = OutputType(out1);
        }
    }
}

kernel void moe_decode_glu_down_q6_k_f32_f32(
    device const q6_k_block *weight [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *up [[buffer(2)]], device const int *route_experts [[buffer(3)]],
    device const float *route_weights [[buffer(4)]], device float *output [[buffer(5)]],
    constant metal_moe_decode_down_uniforms_t &params [[buffer(6)]], uint tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_glu_down_q6_k_impl<float, float, float>(
        weight, gate, up, route_experts, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_decode_glu_down_q6_k_f16_f32(
    device const q6_k_block *weight [[buffer(0)]], device const half *gate [[buffer(1)]],
    device const half *up [[buffer(2)]], device const int *route_experts [[buffer(3)]],
    device const half *route_weights [[buffer(4)]], device float *output [[buffer(5)]],
    constant metal_moe_decode_down_uniforms_t &params [[buffer(6)]], uint tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_glu_down_q6_k_impl<half, half, float>(
        weight, gate, up, route_experts, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_decode_glu_down_q6_k_f16_f16(
    device const q6_k_block *weight [[buffer(0)]], device const half *gate [[buffer(1)]],
    device const half *up [[buffer(2)]], device const int *route_experts [[buffer(3)]],
    device const half *route_weights [[buffer(4)]], device half *output [[buffer(5)]],
    constant metal_moe_decode_down_uniforms_t &params [[buffer(6)]], uint tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_decode_glu_down_q6_k_impl<half, half, half>(
        weight, gate, up, route_experts, route_weights, output, params, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename WeightType, typename OutputType>
static inline void moe_indexed_down_q6_k_impl(
    device const q6_k_block *weight, device const InputType *input, device const int *route_indices,
    device const WeightType *route_weights, device OutputType *output,
    constant metal_moe_indexed_down_uniforms_t &params, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint nr0 = 2u;
    constexpr uint nsg = 2u;
    constexpr uint rows_per_threadgroup = nr0 * nsg;

    const uint route = tgp.y;
    if (route >= params.routes) {
        return;
    }

    const int output_index = route_indices[route];
    if (output_index < 0 || uint(output_index) >= params.output_rows) {
        return;
    }

    const float route_scale = float(route_weights[route]);
    if (route_scale == 0.0f) {
        return;
    }

    const uint m_base = tgp.x * rows_per_threadgroup + uint(sgitg) * nr0;
    if (m_base >= params.output_cols) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < params.output_cols;
    device const InputType *input_row = input + route * params.input_stride;
    device OutputType *output_row = output + uint(output_index) * params.output_stride;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    moe_decode_down_q6_k_route_accumulate<InputType>(
        weight + m0 * params.weight_blocks, weight + m1 * params.weight_blocks, has_m1, input_row, params.input_cols,
        params.weight_blocks, tiisg, sum0, sum1
    );

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output_row[m0] = OutputType(float(output_row[m0]) + route_scale * out0);
        if (has_m1) {
            output_row[m1] = OutputType(float(output_row[m1]) + route_scale * out1);
        }
    }
}

template <typename InputType, typename WeightType, typename OutputType>
static inline void moe_indexed_glu_down_q6_k_impl(
    device const q6_k_block *weight, device const InputType *gate, device const InputType *up,
    device const int *route_indices, device const WeightType *route_weights, device OutputType *output,
    constant metal_moe_indexed_down_uniforms_t &params, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint nr0 = 2u;
    constexpr uint nsg = 2u;
    constexpr uint rows_per_threadgroup = nr0 * nsg;

    const uint route = tgp.y;
    if (route >= params.routes) {
        return;
    }

    const int output_index = route_indices[route];
    if (output_index < 0 || uint(output_index) >= params.output_rows) {
        return;
    }

    const float route_scale = float(route_weights[route]);
    if (route_scale == 0.0f) {
        return;
    }

    const uint m_base = tgp.x * rows_per_threadgroup + uint(sgitg) * nr0;
    if (m_base >= params.output_cols) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < params.output_cols;
    device const InputType *gate_row = gate + route * params.input_stride;
    device const InputType *up_row = up + route * params.input_stride;
    device OutputType *output_row = output + uint(output_index) * params.output_stride;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    moe_decode_down_q6_k_route_accumulate_glu<InputType>(
        weight + m0 * params.weight_blocks, weight + m1 * params.weight_blocks, has_m1, gate_row, up_row,
        params.activation, params.input_cols, params.weight_blocks, tiisg, sum0, sum1
    );

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output_row[m0] = OutputType(float(output_row[m0]) + route_scale * out0);
        if (has_m1) {
            output_row[m1] = OutputType(float(output_row[m1]) + route_scale * out1);
        }
    }
}

kernel void moe_indexed_down_q6_k_f32_f32(
    device const q6_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device const int *route_indices [[buffer(2)]], device const float *route_weights [[buffer(3)]],
    device float *output [[buffer(4)]], constant metal_moe_indexed_down_uniforms_t &params [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_indexed_down_q6_k_impl<float, float, float>(
        weight, input, route_indices, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_indexed_down_q6_k_f16_f32(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device const int *route_indices [[buffer(2)]], device const half *route_weights [[buffer(3)]],
    device float *output [[buffer(4)]], constant metal_moe_indexed_down_uniforms_t &params [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_indexed_down_q6_k_impl<half, half, float>(
        weight, input, route_indices, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_indexed_down_q6_k_f16_f16(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device const int *route_indices [[buffer(2)]], device const half *route_weights [[buffer(3)]],
    device half *output [[buffer(4)]], constant metal_moe_indexed_down_uniforms_t &params [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_indexed_down_q6_k_impl<half, half, half>(
        weight, input, route_indices, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_indexed_glu_down_q6_k_f32_f32(
    device const q6_k_block *weight [[buffer(0)]], device const float *gate [[buffer(1)]],
    device const float *up [[buffer(2)]], device const int *route_indices [[buffer(3)]],
    device const float *route_weights [[buffer(4)]], device float *output [[buffer(5)]],
    constant metal_moe_indexed_down_uniforms_t &params [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_indexed_glu_down_q6_k_impl<float, float, float>(
        weight, gate, up, route_indices, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_indexed_glu_down_q6_k_f16_f32(
    device const q6_k_block *weight [[buffer(0)]], device const half *gate [[buffer(1)]],
    device const half *up [[buffer(2)]], device const int *route_indices [[buffer(3)]],
    device const half *route_weights [[buffer(4)]], device float *output [[buffer(5)]],
    constant metal_moe_indexed_down_uniforms_t &params [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_indexed_glu_down_q6_k_impl<half, half, float>(
        weight, gate, up, route_indices, route_weights, output, params, tgp, tiisg, sgitg
    );
}

kernel void moe_indexed_glu_down_q6_k_f16_f16(
    device const q6_k_block *weight [[buffer(0)]], device const half *gate [[buffer(1)]],
    device const half *up [[buffer(2)]], device const int *route_indices [[buffer(3)]],
    device const half *route_weights [[buffer(4)]], device half *output [[buffer(5)]],
    constant metal_moe_indexed_down_uniforms_t &params [[buffer(6)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    moe_indexed_glu_down_q6_k_impl<half, half, half>(
        weight, gate, up, route_indices, route_weights, output, params, tgp, tiisg, sgitg
    );
}

// --- Expert-batched MoE kernels using simdgroup_matrix tiled GEMM ---

static inline half4 moe_eb_load_half4(device const float *ptr) {
    return half4(*((device const float4 *)ptr));
}

static inline half4 moe_eb_load_half4(device const half *ptr) {
    return *((device const half4 *)ptr);
}

constant ushort MOE_EB_NR0 = 64;
constant ushort MOE_EB_NR1 = 32;
constant ushort MOE_EB_NK = 32;

typedef struct {
    uint output_cols;
    uint input_cols;
    uint input_stride;
    uint output_stride;
    uint weight_blocks;
    uint total_experts;
    uint activation;
} metal_moe_expert_batch_uniforms_t;

template <typename OutputType>
static inline void moe_eb_store_8x8(
    simdgroup_float8x8 mat, device OutputType *out_ptr, uint out_stride, uint out_n, uint out_m, uint max_n, uint max_m,
    threadgroup float *sg_scratch
) {
    const bool full_n = (out_n + 7u) < max_n;
    const bool full_m = (out_m + 7u) < max_m;
    simdgroup_store(mat, sg_scratch, 8, 0, false);
    if (full_n && full_m) {
#pragma clang loop unroll(full)
        for (uint r = 0; r < 8u; ++r) {
#pragma clang loop unroll(full)
            for (uint c = 0; c < 8u; ++c) {
                out_ptr[r * out_stride + c] = OutputType(sg_scratch[r * 8u + c]);
            }
        }
        return;
    }
    for (uint r = 0; r < 8u && (out_n + r) < max_n; ++r) {
        for (uint c = 0; c < 8u && (out_m + c) < max_m; ++c) {
            out_ptr[r * out_stride + c] = OutputType(sg_scratch[r * 8u + c]);
        }
    }
}

template <typename InputType, typename OutputType, typename BlockType>
static inline void moe_expert_matmul_impl(
    device const BlockType *weight, device const InputType *input, device OutputType *output,
    device const uint *route_counts, device const uint *route_offsets, device const uint *active_ids,
    constant metal_moe_expert_batch_uniforms_t &params, uint3 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint expert_id = active_ids[tgp.z];
    const uint n_routes = route_counts[expert_id];
    if (n_routes == 0u) {
        return;
    }
    const uint m0 = tgp.x * MOE_EB_NR0;
    const uint n0 = tgp.y * MOE_EB_NR1;
    if (m0 >= params.output_cols || n0 >= n_routes) {
        return;
    }
    const uint route_base = route_offsets[expert_id];
    const uint M = params.output_cols;
    const uint K = params.input_cols;

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MOE_EB_NR0 * MOE_EB_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < n_routes;
    const bool col1_active = sg_active && (sg_n0 + 8u) < n_routes;
    const uint expert_weight_base = expert_id * M * params.weight_blocks;

    for (uint k_base = 0; k_base < K; k_base += MOE_EB_NK) {
        const bool full_k_tile = (k_base + MOE_EB_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < params.weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            const device BlockType &blk = weight[expert_weight_base + gm * params.weight_blocks + block_idx];
            dequantize_q4_k_chunk(blk, short(chunk_idx), tmp);

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                const uint gk = k_base + sx_a * 8u + ly_a;
                const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
        }

        const uint lr1 = (uint)tiitg >> 2;
        const uint sx_b = (uint)tiitg & 3u;
        const uint gn = n0 + lr1;
        const uint sy_b = lr1 >> 3;
        const uint ly_b = lr1 & 7u;
        const uint k_local_base = sx_b * 8u;
        const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

        if (gn < n_routes) {
            device const InputType *input_row = input + (route_base + gn) * params.input_stride;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile) {
                *((threadgroup half4 *)(sb + sb_base)) = moe_eb_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = moe_eb_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    sb[sb_base + i] = gk < K ? half(input_row[gk]) : half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                sb[sb_base + i] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 2u; ++i) {
                        simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 8u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
                    simdgroup_load(mb[0], lsmb + 64u * 0u, 8, 0, false);
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[0], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!sg_active) {
        return;
    }

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
    device OutputType *base_ptr = output + (route_base + sg_n0) * params.output_stride + sg_m0;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        const uint out_m = sg_m0 + 8u * (i & 3u);
        const uint out_n = sg_n0 + 8u * (i >> 2);
        if (out_m >= M || out_n >= n_routes) {
            continue;
        }
        device OutputType *tile_ptr = base_ptr + 8u * (i >> 2) * params.output_stride + 8u * (i & 3u);
        moe_eb_store_8x8(mc[i], tile_ptr, params.output_stride, out_n, out_m, n_routes, M, sg_scratch);
    }
}

kernel void moe_expert_matmul_q4_k_f32_f32(
    device const q4_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device const uint *route_counts [[buffer(3)]],
    device const uint *route_offsets [[buffer(4)]], constant metal_moe_expert_batch_uniforms_t &params [[buffer(5)]],
    device const uint *active_ids [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_matmul_impl<float, float, q4_k_block>(
        weight, input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

kernel void moe_expert_matmul_q4_k_f16_f32(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], device const uint *route_counts [[buffer(3)]],
    device const uint *route_offsets [[buffer(4)]], constant metal_moe_expert_batch_uniforms_t &params [[buffer(5)]],
    device const uint *active_ids [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_matmul_impl<half, float, q4_k_block>(
        weight, input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

kernel void moe_expert_matmul_q4_k_f16_f16(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], device const uint *route_counts [[buffer(3)]],
    device const uint *route_offsets [[buffer(4)]], constant metal_moe_expert_batch_uniforms_t &params [[buffer(5)]],
    device const uint *active_ids [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_matmul_impl<half, half, q4_k_block>(
        weight, input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

template <typename InputType, typename OutputType>
static inline void moe_expert_fused_gate_up_q4_k_impl(
    device const q4_k_block *gate_weight, device const q4_k_block *up_weight, device const InputType *input,
    device OutputType *gate_output, device OutputType *up_output, device const uint *route_counts,
    device const uint *route_offsets, device const uint *active_ids, constant metal_moe_expert_batch_uniforms_t &params,
    uint3 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint expert_id = active_ids[tgp.z];
    const uint n_routes = route_counts[expert_id];
    const uint m0 = tgp.x * MOE_EB_NR0;
    const uint n0 = tgp.y * MOE_EB_NR1;
    if (m0 >= params.output_cols || n0 >= n_routes) {
        return;
    }
    const uint route_base = route_offsets[expert_id];
    const uint M = params.output_cols;
    const uint K = params.input_cols;

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MOE_EB_NR0 * MOE_EB_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half));

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < n_routes;
    const bool col1_active = sg_active && (sg_n0 + 8u) < n_routes;
    const uint expert_weight_base = expert_id * M * params.weight_blocks;

    // Pass 1: gate projection
    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k_base = 0; k_base < K; k_base += MOE_EB_NK) {
        const bool full_k_tile = (k_base + MOE_EB_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < params.weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            const device q4_k_block &blk = gate_weight[expert_weight_base + gm * params.weight_blocks + block_idx];
            dequantize_q4_k_chunk(blk, short(chunk_idx), tmp);

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                const uint gk = k_base + sx_a * 8u + ly_a;
                const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
        }

        const uint lr1 = (uint)tiitg >> 2;
        const uint sx_b = (uint)tiitg & 3u;
        const uint gn = n0 + lr1;
        const uint sy_b = lr1 >> 3;
        const uint ly_b = lr1 & 7u;
        const uint k_local_base = sx_b * 8u;
        const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

        if (gn < n_routes) {
            device const InputType *input_row = input + (route_base + gn) * params.input_stride;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile) {
                *((threadgroup half4 *)(sb + sb_base)) = moe_eb_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = moe_eb_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    sb[sb_base + i] = gk < K ? half(input_row[gk]) : half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                sb[sb_base + i] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 2u; ++i) {
                        simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 8u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
                    simdgroup_load(mb[0], lsmb + 64u * 0u, 8, 0, false);
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[0], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg_active) {
        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = gate_output + (route_base + sg_n0) * params.output_stride + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_m = sg_m0 + 8u * (i & 3u);
            const uint out_n = sg_n0 + 8u * (i >> 2);
            if (out_m >= M || out_n >= n_routes) {
                continue;
            }
            device OutputType *tile_ptr = base_ptr + 8u * (i >> 2) * params.output_stride + 8u * (i & 3u);
            moe_eb_store_8x8(mc[i], tile_ptr, params.output_stride, out_n, out_m, n_routes, M, sg_scratch);
        }
    }

    // Pass 2: up projection (input is hot in L2 cache)
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k_base = 0; k_base < K; k_base += MOE_EB_NK) {
        const bool full_k_tile = (k_base + MOE_EB_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < params.weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            const device q4_k_block &blk = up_weight[expert_weight_base + gm * params.weight_blocks + block_idx];
            dequantize_q4_k_chunk(blk, short(chunk_idx), tmp);

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                const uint gk = k_base + sx_a * 8u + ly_a;
                const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
        }

        const uint lr1 = (uint)tiitg >> 2;
        const uint sx_b = (uint)tiitg & 3u;
        const uint gn = n0 + lr1;
        const uint sy_b = lr1 >> 3;
        const uint ly_b = lr1 & 7u;
        const uint k_local_base = sx_b * 8u;
        const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

        if (gn < n_routes) {
            device const InputType *input_row = input + (route_base + gn) * params.input_stride;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile) {
                *((threadgroup half4 *)(sb + sb_base)) = moe_eb_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = moe_eb_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    sb[sb_base + i] = gk < K ? half(input_row[gk]) : half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                sb[sb_base + i] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 2u; ++i) {
                        simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 8u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
                    simdgroup_load(mb[0], lsmb + 64u * 0u, 8, 0, false);
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[0], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sg_active) {
        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = up_output + (route_base + sg_n0) * params.output_stride + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_m = sg_m0 + 8u * (i & 3u);
            const uint out_n = sg_n0 + 8u * (i >> 2);
            if (out_m >= M || out_n >= n_routes) {
                continue;
            }
            device OutputType *tile_ptr = base_ptr + 8u * (i >> 2) * params.output_stride + 8u * (i & 3u);
            moe_eb_store_8x8(mc[i], tile_ptr, params.output_stride, out_n, out_m, n_routes, M, sg_scratch);
        }
    }
}

kernel void moe_expert_fused_gate_up_q4_k_f32_f32(
    device const q4_k_block *gate_weight [[buffer(0)]], device const q4_k_block *up_weight [[buffer(1)]],
    device const float *input [[buffer(2)]], device float *gate_output [[buffer(3)]],
    device float *up_output [[buffer(4)]], device const uint *route_counts [[buffer(5)]],
    device const uint *route_offsets [[buffer(6)]], device const uint *active_ids [[buffer(7)]],
    constant metal_moe_expert_batch_uniforms_t &params [[buffer(8)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_fused_gate_up_q4_k_impl<float, float>(
        gate_weight, up_weight, input, gate_output, up_output, route_counts, route_offsets, active_ids, params, tgp,
        tiitg, sgitg, shmem
    );
}

kernel void moe_expert_fused_gate_up_q4_k_f16_f32(
    device const q4_k_block *gate_weight [[buffer(0)]], device const q4_k_block *up_weight [[buffer(1)]],
    device const half *input [[buffer(2)]], device float *gate_output [[buffer(3)]],
    device float *up_output [[buffer(4)]], device const uint *route_counts [[buffer(5)]],
    device const uint *route_offsets [[buffer(6)]], device const uint *active_ids [[buffer(7)]],
    constant metal_moe_expert_batch_uniforms_t &params [[buffer(8)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_fused_gate_up_q4_k_impl<half, float>(
        gate_weight, up_weight, input, gate_output, up_output, route_counts, route_offsets, active_ids, params, tgp,
        tiitg, sgitg, shmem
    );
}

kernel void moe_expert_fused_gate_up_q4_k_f16_f16(
    device const q4_k_block *gate_weight [[buffer(0)]], device const q4_k_block *up_weight [[buffer(1)]],
    device const half *input [[buffer(2)]], device half *gate_output [[buffer(3)]],
    device half *up_output [[buffer(4)]], device const uint *route_counts [[buffer(5)]],
    device const uint *route_offsets [[buffer(6)]], device const uint *active_ids [[buffer(7)]],
    constant metal_moe_expert_batch_uniforms_t &params [[buffer(8)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_fused_gate_up_q4_k_impl<half, half>(
        gate_weight, up_weight, input, gate_output, up_output, route_counts, route_offsets, active_ids, params, tgp,
        tiitg, sgitg, shmem
    );
}

template <typename InputType, typename OutputType>
static inline void moe_expert_matmul_q6_k_impl(
    device const q6_k_block *weight, device const InputType *input, device OutputType *output,
    device const uint *route_counts, device const uint *route_offsets, device const uint *active_ids,
    constant metal_moe_expert_batch_uniforms_t &params, uint3 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint expert_id = active_ids[tgp.z];
    if (expert_id >= params.total_experts) {
        return;
    }
    const uint n_routes = route_counts[expert_id];
    if (n_routes == 0u) {
        return;
    }
    const uint m0 = tgp.x * MOE_EB_NR0;
    const uint n0 = tgp.y * MOE_EB_NR1;
    if (m0 >= params.output_cols || n0 >= n_routes) {
        return;
    }
    const uint route_base = route_offsets[expert_id];
    const uint M = params.output_cols;
    const uint K = params.input_cols;

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MOE_EB_NR0 * MOE_EB_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < n_routes;
    const bool col1_active = sg_active && (sg_n0 + 8u) < n_routes;
    const uint expert_weight_base = expert_id * M * params.weight_blocks;

    for (uint k_base = 0; k_base < K; k_base += MOE_EB_NK) {
        const bool full_k_tile = (k_base + MOE_EB_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < params.weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            dequantize_q6_k_chunk(
                weight[expert_weight_base + gm * params.weight_blocks + block_idx], short(chunk_idx), tmp
            );

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                const uint gk = k_base + sx_a * 8u + ly_a;
                const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
        }

        const uint lr1 = (uint)tiitg >> 2;
        const uint sx_b = (uint)tiitg & 3u;
        const uint gn = n0 + lr1;
        const uint sy_b = lr1 >> 3;
        const uint ly_b = lr1 & 7u;
        const uint k_local_base = sx_b * 8u;
        const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

        if (gn < n_routes) {
            device const InputType *input_row = input + (route_base + gn) * params.input_stride;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile) {
                *((threadgroup half4 *)(sb + sb_base)) = moe_eb_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = moe_eb_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    sb[sb_base + i] = gk < K ? half(input_row[gk]) : half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                sb[sb_base + i] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 2u; ++i) {
                        simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 8u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
                    simdgroup_load(mb[0], lsmb + 64u * 0u, 8, 0, false);
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[0], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!sg_active) {
        return;
    }

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
    device OutputType *base_ptr = output + (route_base + sg_n0) * params.output_stride + sg_m0;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        const uint out_m = sg_m0 + 8u * (i & 3u);
        const uint out_n = sg_n0 + 8u * (i >> 2);
        if (out_m >= M || out_n >= n_routes) {
            continue;
        }
        device OutputType *tile_ptr = base_ptr + 8u * (i >> 2) * params.output_stride + 8u * (i & 3u);
        moe_eb_store_8x8(mc[i], tile_ptr, params.output_stride, out_n, out_m, n_routes, M, sg_scratch);
    }
}

kernel void moe_expert_matmul_q6_k_f32_f32(
    device const q6_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], device const uint *route_counts [[buffer(3)]],
    device const uint *route_offsets [[buffer(4)]], constant metal_moe_expert_batch_uniforms_t &params [[buffer(5)]],
    device const uint *active_ids [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_matmul_q6_k_impl<float, float>(
        weight, input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

kernel void moe_expert_matmul_q6_k_f16_f32(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], device const uint *route_counts [[buffer(3)]],
    device const uint *route_offsets [[buffer(4)]], constant metal_moe_expert_batch_uniforms_t &params [[buffer(5)]],
    device const uint *active_ids [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_matmul_q6_k_impl<half, float>(
        weight, input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

kernel void moe_expert_matmul_q6_k_f16_f16(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], device const uint *route_counts [[buffer(3)]],
    device const uint *route_offsets [[buffer(4)]], constant metal_moe_expert_batch_uniforms_t &params [[buffer(5)]],
    device const uint *active_ids [[buffer(6)]], uint3 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_matmul_q6_k_impl<half, half>(
        weight, input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

template <typename InputType, typename OutputType>
static inline void moe_expert_glu_down_q6_k_impl(
    device const q6_k_block *weight, device const InputType *gate_input, device const InputType *up_input,
    device OutputType *output, device const uint *route_counts, device const uint *route_offsets,
    device const uint *active_ids, constant metal_moe_expert_batch_uniforms_t &params, uint3 tgp, ushort tiitg,
    uint sgitg, threadgroup char *shmem
) {
    const uint expert_id = active_ids[tgp.z];
    if (expert_id >= params.total_experts) {
        return;
    }
    const uint n_routes = route_counts[expert_id];
    if (n_routes == 0u) {
        return;
    }
    const uint m0 = tgp.x * MOE_EB_NR0;
    const uint n0 = tgp.y * MOE_EB_NR1;
    if (m0 >= params.output_cols || n0 >= n_routes) {
        return;
    }
    const uint route_base = route_offsets[expert_id];
    const uint M = params.output_cols;
    const uint K = params.input_cols;

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MOE_EB_NR0 * MOE_EB_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < n_routes;
    const bool col1_active = sg_active && (sg_n0 + 8u) < n_routes;
    const uint expert_weight_base = expert_id * M * params.weight_blocks;

    for (uint k_base = 0; k_base < K; k_base += MOE_EB_NK) {
        const bool full_k_tile = (k_base + MOE_EB_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < params.weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            dequantize_q6_k_chunk(
                weight[expert_weight_base + gm * params.weight_blocks + block_idx], short(chunk_idx), tmp
            );

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                const uint gk = k_base + sx_a * 8u + ly_a;
                const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
        }

        const uint lr1 = (uint)tiitg >> 2;
        const uint sx_b = (uint)tiitg & 3u;
        const uint gn = n0 + lr1;
        const uint sy_b = lr1 >> 3;
        const uint ly_b = lr1 & 7u;
        const uint k_local_base = sx_b * 8u;
        const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

        if (gn < n_routes) {
            device const InputType *gate_row = gate_input + (route_base + gn) * params.input_stride;
            device const InputType *up_row = up_input + (route_base + gn) * params.input_stride;
            const uint gk_base = k_base + k_local_base;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                const uint gk = gk_base + i;
                if (gk < K) {
                    sb[sb_base + i] =
                        half(moe_decode_apply_glu(float(gate_row[gk]), float(up_row[gk]), params.activation));
                } else {
                    sb[sb_base + i] = half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                sb[sb_base + i] = half(0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
            simdgroup_half8x8 ma[4];
            simdgroup_half8x8 mb[2];

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 2u; ++i) {
                        simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                    }
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 8u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MOE_EB_NK / 8u; ++ik) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                    }
                    simdgroup_load(mb[0], lsmb + 64u * 0u, 8, 0, false);
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4u; ++i) {
                        simdgroup_multiply_accumulate(mc[i], mb[0], ma[i & 3u], mc[i]);
                    }
                    lsma += 8u * 64u;
                    lsmb += 4u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!sg_active) {
        return;
    }

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
    device OutputType *base_ptr = output + (route_base + sg_n0) * params.output_stride + sg_m0;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        const uint out_m = sg_m0 + 8u * (i & 3u);
        const uint out_n = sg_n0 + 8u * (i >> 2);
        if (out_m >= M || out_n >= n_routes) {
            continue;
        }
        device OutputType *tile_ptr = base_ptr + 8u * (i >> 2) * params.output_stride + 8u * (i & 3u);
        moe_eb_store_8x8(mc[i], tile_ptr, params.output_stride, out_n, out_m, n_routes, M, sg_scratch);
    }
}

kernel void moe_expert_glu_down_q6_k_f32_f32(
    device const q6_k_block *weight [[buffer(0)]], device const float *gate_input [[buffer(1)]],
    device const float *up_input [[buffer(2)]], device float *output [[buffer(3)]],
    device const uint *route_counts [[buffer(4)]], device const uint *route_offsets [[buffer(5)]],
    constant metal_moe_expert_batch_uniforms_t &params [[buffer(6)]], device const uint *active_ids [[buffer(7)]],
    uint3 tgp [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_glu_down_q6_k_impl<float, float>(
        weight, gate_input, up_input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

kernel void moe_expert_glu_down_q6_k_f16_f32(
    device const q6_k_block *weight [[buffer(0)]], device const half *gate_input [[buffer(1)]],
    device const half *up_input [[buffer(2)]], device float *output [[buffer(3)]],
    device const uint *route_counts [[buffer(4)]], device const uint *route_offsets [[buffer(5)]],
    constant metal_moe_expert_batch_uniforms_t &params [[buffer(6)]], device const uint *active_ids [[buffer(7)]],
    uint3 tgp [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_glu_down_q6_k_impl<half, float>(
        weight, gate_input, up_input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

kernel void moe_expert_glu_down_q6_k_f16_f16(
    device const q6_k_block *weight [[buffer(0)]], device const half *gate_input [[buffer(1)]],
    device const half *up_input [[buffer(2)]], device half *output [[buffer(3)]],
    device const uint *route_counts [[buffer(4)]], device const uint *route_offsets [[buffer(5)]],
    constant metal_moe_expert_batch_uniforms_t &params [[buffer(6)]], device const uint *active_ids [[buffer(7)]],
    uint3 tgp [[threadgroup_position_in_grid]], ushort tiitg [[thread_index_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup char shmem[(MOE_EB_NR0 * MOE_EB_NK + MOE_EB_NR1 * MOE_EB_NK) * sizeof(half) + 4u * 64u * sizeof(float)];
    moe_expert_glu_down_q6_k_impl<half, half>(
        weight, gate_input, up_input, output, route_counts, route_offsets, active_ids, params, tgp, tiitg, sgitg, shmem
    );
}

kernel void moe_route_count_i32(
    device const int *topk_ids [[buffer(0)]], device atomic_uint *expert_counts [[buffer(1)]],
    device atomic_uint *status_flag [[buffer(2)]], constant metal_moe_route_uniforms_t &params [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    const uint total = params.tokens * params.experts_per_token;
    if (id >= total) {
        return;
    }

    const uint token = id / params.experts_per_token;
    const uint slot = id - token * params.experts_per_token;
    const int expert = topk_ids[token * params.id_stride0 + slot * params.id_stride1];
    if (expert < 0 || uint(expert) >= params.experts) {
        atomic_store_explicit(&status_flag[0], 1u, memory_order_relaxed);
        return;
    }

    bool duplicate = false;
    for (uint prev = 0; prev < slot; ++prev) {
        if (topk_ids[token * params.id_stride0 + prev * params.id_stride1] == expert) {
            duplicate = true;
            break;
        }
    }
    if (duplicate) {
        return;
    }

    atomic_fetch_add_explicit(&expert_counts[uint(expert)], 1u, memory_order_relaxed);
}

kernel void moe_route_prepare_i32(
    device const atomic_uint *expert_counts [[buffer(0)]], device uint *expert_offsets [[buffer(1)]],
    device metal_moe_route_summary_t *route_summary [[buffer(2)]], constant uint &experts [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    if (id != 0) {
        return;
    }

    uint prefix = 0;
    uint max_batch = 0;
    uint active_experts = 0;
    for (uint expert = 0; expert < experts; ++expert) {
        const uint count = atomic_load_explicit(&expert_counts[expert], memory_order_relaxed);
        expert_offsets[expert] = prefix;
        prefix += count;
        if (count != 0u) {
            active_experts++;
            if (count > max_batch) {
                max_batch = count;
            }
        }
    }

    route_summary[0].route_count = prefix;
    route_summary[0].max_batch = max_batch;
    route_summary[0].active_experts = active_experts;
    route_summary[0].reserved = 0u;
}

template <typename T>
static inline T moe_route_weight_cast(float value);

template <>
inline float moe_route_weight_cast<float>(float value) {
    return value;
}

template <>
inline half moe_route_weight_cast<half>(float value) {
    return half(value);
}

template <typename T>
static inline void moe_route_pack_stable_impl(
    device const int *topk_ids [[buffer(0)]], device const T *topk_weights [[buffer(1)]],
    device const uint *expert_offsets [[buffer(2)]], device int *route_indices [[buffer(3)]],
    device T *route_weights [[buffer(4)]], device int *route_experts [[buffer(5)]],
    constant metal_moe_route_uniforms_t &params, uint expert
) {
    if (expert >= params.experts) {
        return;
    }

    uint cursor = expert_offsets[expert];
    for (uint token = 0; token < params.tokens; ++token) {
        bool matched = false;
        float combined_weight = 0.0f;
        float weight_sum = 0.0f;
        for (uint slot = 0; slot < params.experts_per_token; ++slot) {
            const float weight = float(topk_weights[token * params.weight_stride0 + slot * params.weight_stride1]);
            if (params.renormalize_selected != 0u) {
                weight_sum += weight;
            }
            const int route_expert = topk_ids[token * params.id_stride0 + slot * params.id_stride1];
            if (route_expert != int(expert)) {
                continue;
            }
            matched = true;
            combined_weight += weight;
        }
        if (!matched) {
            continue;
        }

        const float weight_norm = params.renormalize_selected != 0u
            ? (weight_sum > FLT_MIN ? params.weights_scale / weight_sum : 0.0f)
            : params.weights_scale;
        route_indices[cursor] = int(token);
        route_weights[cursor] = moe_route_weight_cast<T>(combined_weight * weight_norm);
        route_experts[cursor] = int(expert);
        ++cursor;
    }
}

kernel void moe_route_pack_stable_f32(
    device const int *topk_ids [[buffer(0)]], device const float *topk_weights [[buffer(1)]],
    device const uint *expert_offsets [[buffer(2)]], device int *route_indices [[buffer(3)]],
    device float *route_weights [[buffer(4)]], device int *route_experts [[buffer(5)]],
    constant metal_moe_route_uniforms_t &params [[buffer(6)]], uint expert [[thread_position_in_grid]]
) {
    moe_route_pack_stable_impl<float>(
        topk_ids, topk_weights, expert_offsets, route_indices, route_weights, route_experts, params, expert
    );
}

kernel void moe_route_pack_stable_f16(
    device const int *topk_ids [[buffer(0)]], device const half *topk_weights [[buffer(1)]],
    device const uint *expert_offsets [[buffer(2)]], device int *route_indices [[buffer(3)]],
    device half *route_weights [[buffer(4)]], device int *route_experts [[buffer(5)]],
    constant metal_moe_route_uniforms_t &params [[buffer(6)]], uint expert [[thread_position_in_grid]]
) {
    moe_route_pack_stable_impl<half>(
        topk_ids, topk_weights, expert_offsets, route_indices, route_weights, route_experts, params, expert
    );
}
