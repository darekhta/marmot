#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/quant_blocks.h"
#include "common/quant_decode.h"

#define MARMOT_QKV_QUANT_TILE_N 16u
#define MARMOT_QKV_QUANT_TILE_M 16u

struct MatmulQKVQuantUniforms {
    uint rope_enabled;
    uint rope_apply_q;
    uint rope_apply_k;
    uint rope_head_dim;
    float rope_attn_scale;
};

static inline uint matmul_qkv_unpack_q5_qh(const device uchar *qh) {
    return (uint)qh[0] | ((uint)qh[1] << 8u) | ((uint)qh[2] << 16u) | ((uint)qh[3] << 24u);
}

static inline void matmul_qkv_accumulate_q4_0_block(
    const device q4_0_block &wq, const device q4_0_block &wk, const device q4_0_block &wv, const device q8_0_block &act,
    thread float &acc_q, thread float &acc_k, thread float &acc_v
) {
    const uint half_block = kQuantBlockSize / 2u;
    const float act_scale = float(act.scale);
    int block_sum_q = 0;
    int block_sum_k = 0;
    int block_sum_v = 0;
    for (uint i = 0; i < kQ4PackedBytes; ++i) {
        const int act0 = int(act.qs[i]);
        const int act1 = int(act.qs[i + half_block]);
        const uchar pq = wq.qs[i];
        const uchar pk = wk.qs[i];
        const uchar pv = wv.qs[i];
        const int wq0 = int(pq & 0x0Fu) - 8;
        const int wq1 = int(pq >> 4) - 8;
        const int wk0 = int(pk & 0x0Fu) - 8;
        const int wk1 = int(pk >> 4) - 8;
        const int wv0 = int(pv & 0x0Fu) - 8;
        const int wv1 = int(pv >> 4) - 8;
        block_sum_q += wq0 * act0 + wq1 * act1;
        block_sum_k += wk0 * act0 + wk1 * act1;
        block_sum_v += wv0 * act0 + wv1 * act1;
    }
    const float scale_act = act_scale;
    acc_q += float(block_sum_q) * (float(wq.scale) * scale_act);
    acc_k += float(block_sum_k) * (float(wk.scale) * scale_act);
    acc_v += float(block_sum_v) * (float(wv.scale) * scale_act);
}

static inline void matmul_qkv_accumulate_q4_1_block(
    const device q4_1_block &wq, const device q4_1_block &wk, const device q4_1_block &wv, const device q8_0_block &act,
    thread float &acc_q, thread float &acc_k, thread float &acc_v
) {
    const uint half_block = kQuantBlockSize / 2u;
    int block_q_sum_q = 0;
    int block_q_sum_k = 0;
    int block_q_sum_v = 0;
    int block_i_sum = 0;
    for (uint i = 0; i < kQ4PackedBytes; ++i) {
        const int act0 = int(act.qs[i]);
        const int act1 = int(act.qs[i + half_block]);
        block_i_sum += act0 + act1;
        const uchar pq = wq.qs[i];
        const uchar pk = wk.qs[i];
        const uchar pv = wv.qs[i];
        block_q_sum_q += int(pq & 0x0Fu) * act0 + int(pq >> 4) * act1;
        block_q_sum_k += int(pk & 0x0Fu) * act0 + int(pk >> 4) * act1;
        block_q_sum_v += int(pv & 0x0Fu) * act0 + int(pv >> 4) * act1;
    }
    const float scale_i = float(act.scale);
    acc_q += scale_i * (float(wq.scale) * float(block_q_sum_q) + float(wq.min) * float(block_i_sum));
    acc_k += scale_i * (float(wk.scale) * float(block_q_sum_k) + float(wk.min) * float(block_i_sum));
    acc_v += scale_i * (float(wv.scale) * float(block_q_sum_v) + float(wv.min) * float(block_i_sum));
}

static inline void matmul_qkv_accumulate_q5_0_block(
    const device q5_0_block &wq, const device q5_0_block &wk, const device q5_0_block &wv, const device q8_0_block &act,
    thread float &acc_q, thread float &acc_k, thread float &acc_v
) {
    const uint half_block = kQuantBlockSize / 2u;
    const uint qh_q = matmul_qkv_unpack_q5_qh(wq.qh);
    const uint qh_k = matmul_qkv_unpack_q5_qh(wk.qh);
    const uint qh_v = matmul_qkv_unpack_q5_qh(wv.qh);
    int block_sum_q = 0;
    int block_sum_k = 0;
    int block_sum_v = 0;
    for (uint i = 0; i < kQ5PackedBytes; ++i) {
        const int act0 = int(act.qs[i]);
        const int act1 = int(act.qs[i + half_block]);
        uint lo_q = uint(wq.qs[i] & 0x0Fu);
        uint hi_q = uint(wq.qs[i] >> 4);
        uint lo_k = uint(wk.qs[i] & 0x0Fu);
        uint hi_k = uint(wk.qs[i] >> 4);
        uint lo_v = uint(wv.qs[i] & 0x0Fu);
        uint hi_v = uint(wv.qs[i] >> 4);
        lo_q |= ((qh_q >> i) & 0x1u) << 4;
        hi_q |= ((qh_q >> (i + half_block)) & 0x1u) << 4;
        lo_k |= ((qh_k >> i) & 0x1u) << 4;
        hi_k |= ((qh_k >> (i + half_block)) & 0x1u) << 4;
        lo_v |= ((qh_v >> i) & 0x1u) << 4;
        hi_v |= ((qh_v >> (i + half_block)) & 0x1u) << 4;
        const int wq0 = int(lo_q) - 16;
        const int wq1 = int(hi_q) - 16;
        const int wk0 = int(lo_k) - 16;
        const int wk1 = int(hi_k) - 16;
        const int wv0 = int(lo_v) - 16;
        const int wv1 = int(hi_v) - 16;
        block_sum_q += wq0 * act0 + wq1 * act1;
        block_sum_k += wk0 * act0 + wk1 * act1;
        block_sum_v += wv0 * act0 + wv1 * act1;
    }
    const float scale_i = float(act.scale);
    acc_q += float(block_sum_q) * (float(wq.scale) * scale_i);
    acc_k += float(block_sum_k) * (float(wk.scale) * scale_i);
    acc_v += float(block_sum_v) * (float(wv.scale) * scale_i);
}

static inline void matmul_qkv_accumulate_q5_1_block(
    const device q5_1_block &wq, const device q5_1_block &wk, const device q5_1_block &wv, const device q8_0_block &act,
    thread float &acc_q, thread float &acc_k, thread float &acc_v
) {
    const uint half_block = kQuantBlockSize / 2u;
    const uint qh_q = matmul_qkv_unpack_q5_qh(wq.qh);
    const uint qh_k = matmul_qkv_unpack_q5_qh(wk.qh);
    const uint qh_v = matmul_qkv_unpack_q5_qh(wv.qh);
    int block_q_sum_q = 0;
    int block_q_sum_k = 0;
    int block_q_sum_v = 0;
    int block_i_sum = 0;
    for (uint i = 0; i < kQ5PackedBytes; ++i) {
        const int act0 = int(act.qs[i]);
        const int act1 = int(act.qs[i + half_block]);
        block_i_sum += act0 + act1;
        uint lo_q = uint(wq.qs[i] & 0x0Fu);
        uint hi_q = uint(wq.qs[i] >> 4);
        uint lo_k = uint(wk.qs[i] & 0x0Fu);
        uint hi_k = uint(wk.qs[i] >> 4);
        uint lo_v = uint(wv.qs[i] & 0x0Fu);
        uint hi_v = uint(wv.qs[i] >> 4);
        lo_q |= ((qh_q >> i) & 0x1u) << 4;
        hi_q |= ((qh_q >> (i + half_block)) & 0x1u) << 4;
        lo_k |= ((qh_k >> i) & 0x1u) << 4;
        hi_k |= ((qh_k >> (i + half_block)) & 0x1u) << 4;
        lo_v |= ((qh_v >> i) & 0x1u) << 4;
        hi_v |= ((qh_v >> (i + half_block)) & 0x1u) << 4;
        block_q_sum_q += int(lo_q) * act0 + int(hi_q) * act1;
        block_q_sum_k += int(lo_k) * act0 + int(hi_k) * act1;
        block_q_sum_v += int(lo_v) * act0 + int(hi_v) * act1;
    }
    const float scale_i = float(act.scale);
    acc_q += scale_i * (float(wq.scale) * float(block_q_sum_q) + float(wq.min) * float(block_i_sum));
    acc_k += scale_i * (float(wk.scale) * float(block_q_sum_k) + float(wk.min) * float(block_i_sum));
    acc_v += scale_i * (float(wv.scale) * float(block_q_sum_v) + float(wv.min) * float(block_i_sum));
}

static inline void matmul_qkv_accumulate_q8_0_block(
    const device q8_0_block &wq, const device q8_0_block &wk, const device q8_0_block &wv, const device q8_0_block &act,
    thread float &acc_q, thread float &acc_k, thread float &acc_v
) {
    int block_sum_q = 0;
    int block_sum_k = 0;
    int block_sum_v = 0;
    for (uint i = 0; i < kQuantBlockSize; ++i) {
        const int act_val = int(act.qs[i]);
        block_sum_q += int(wq.qs[i]) * act_val;
        block_sum_k += int(wk.qs[i]) * act_val;
        block_sum_v += int(wv.qs[i]) * act_val;
    }
    const float scale_i = float(act.scale);
    acc_q += float(block_sum_q) * (float(wq.scale) * scale_i);
    acc_k += float(block_sum_k) * (float(wk.scale) * scale_i);
    acc_v += float(block_sum_v) * (float(wv.scale) * scale_i);
}

static inline void matmul_qkv_accumulate_q8_1_block(
    const device q8_1_block &wq, const device q8_1_block &wk, const device q8_1_block &wv, const device q8_0_block &act,
    thread float &acc_q, thread float &acc_k, thread float &acc_v
) {
    int block_sum_q = 0;
    int block_sum_k = 0;
    int block_sum_v = 0;
    for (uint i = 0; i < kQuantBlockSize; ++i) {
        const int act_val = int(act.qs[i]);
        block_sum_q += int(wq.qs[i]) * act_val;
        block_sum_k += int(wk.qs[i]) * act_val;
        block_sum_v += int(wv.qs[i]) * act_val;
    }
    const float scale_i = float(act.scale);
    acc_q += float(block_sum_q) * (float(wq.scale) * scale_i);
    acc_k += float(block_sum_k) * (float(wk.scale) * scale_i);
    acc_v += float(block_sum_v) * (float(wv.scale) * scale_i);
}

// -----------------------------------------------------------------------------
// Helper: Quantize activation column to Q8_0 (used for all quantized matmuls)
// -----------------------------------------------------------------------------

kernel void quantize_activations_column_q8_0(
    device const float *activations [[buffer(0)]], device q8_0_block *output [[buffer(1)]],
    constant uint &K [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &stride_k [[buffer(4)]],
    constant uint &stride_n [[buffer(5)]], uint2 gid [[thread_position_in_grid]]
) {
    const uint block_id = gid.x;
    const uint column_index = gid.y;
    if (column_index >= N) {
        return;
    }
    const uint blocks_per_column = (K + kQuantBlockSize - 1) / kQuantBlockSize;
    if (block_id >= blocks_per_column) {
        return;
    }

    const uint block_start = block_id * kQuantBlockSize;
    const uint block_end = min(block_start + kQuantBlockSize, K);
    const uint block_len = block_end - block_start;

    float max_abs = 0.0f;
    for (uint k = block_start; k < block_end; ++k) {
        const float val = activations[k * stride_k + column_index * stride_n];
        max_abs = max(max_abs, fabs(val));
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) {
        scale = 1.0f;
    }
    const float inv_scale = 1.0f / scale;

    device q8_0_block *dst = output + column_index * blocks_per_column + block_id;
    dst->scale = half(scale);

    for (uint i = 0; i < kQuantBlockSize; ++i) {
        if (i < block_len) {
            const uint k = block_start + i;
            const float val = activations[k * stride_k + column_index * stride_n];
            const int q = int(round(val * inv_scale));
            dst->qs[i] = char(clamp(q, -127, 127));
        } else {
            dst->qs[i] = 0;
        }
    }
}

// FP16 activations variant (used when inputs are half)
kernel void quantize_activations_column_q8_0_from_f16(
    device const half *activations [[buffer(0)]], device q8_0_block *output [[buffer(1)]],
    constant uint &K [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &stride_k [[buffer(4)]],
    constant uint &stride_n [[buffer(5)]], uint2 gid [[thread_position_in_grid]]
) {
    const uint block_id = gid.x;
    const uint column_index = gid.y;
    if (column_index >= N) {
        return;
    }
    const uint blocks_per_column = (K + kQuantBlockSize - 1) / kQuantBlockSize;
    if (block_id >= blocks_per_column) {
        return;
    }

    const uint block_start = block_id * kQuantBlockSize;
    const uint block_end = min(block_start + kQuantBlockSize, K);
    const uint block_len = block_end - block_start;

    float max_abs = 0.0f;
    for (uint k = block_start; k < block_end; ++k) {
        const float val = float(activations[k * stride_k + column_index * stride_n]);
        max_abs = max(max_abs, fabs(val));
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) {
        scale = 1.0f;
    }
    const float inv_scale = 1.0f / scale;

    device q8_0_block *dst = output + column_index * blocks_per_column + block_id;
    dst->scale = half(scale);

    for (uint i = 0; i < kQuantBlockSize; ++i) {
        if (i < block_len) {
            const uint k = block_start + i;
            const float val = float(activations[k * stride_k + column_index * stride_n]);
            const int q = int(round(val * inv_scale));
            dst->qs[i] = char(clamp(q, -127, 127));
        } else {
            dst->qs[i] = 0;
        }
    }
}

// -----------------------------------------------------------------------------
// Q4_0 × Q8_0 matmul
// PyTorch convention: input(N×K) @ weight(M×K).T = output(N×M)
// -----------------------------------------------------------------------------

kernel void matmul_q4_0_q8_0(
    device const q4_0_block *weight [[buffer(0)]], device const q8_0_block *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K_blocks [[buffer(4)]],
    constant uint &M [[buffer(5)]], uint2 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x; // output feature index (0..M-1)
    const uint n = gid.y; // batch/sample index (0..N-1)

    if (n >= N || m >= M) {
        return;
    }

    float sum = 0.0f;

    device const q8_0_block *input_row = input + n * K_blocks;
    device const q4_0_block *weight_row = weight + m * K_blocks;

    for (uint b = 0; b < K_blocks; ++b) {
        const q4_0_block w_block = weight_row[b];
        const q8_0_block i_block = input_row[b];

        int block_sum = 0;
        for (uint i = 0; i < kQ4PackedBytes; ++i) {
            const uchar packed = w_block.qs[i];
            const int w0 = int(packed & 0x0fu) - 8;
            const int w1 = int(packed >> 4) - 8;
            const int i0 = int(i_block.qs[i]);
            const int i1 = int(i_block.qs[i + kQuantBlockSize / 2]);
            block_sum += w0 * i0 + w1 * i1;
        }

        const float block_scale = float(w_block.scale) * float(i_block.scale);
        sum += float(block_sum) * block_scale;
    }

    output[n * M + m] = sum;
}

template <typename InputType>
static inline float compute_q4k_block_dot_direct(
    const device q4_k_block &w, const device InputType *input_row, uint block_start, uint block_len, uint stride_k
) {
    const float wd = float(w.d);
    const float wdmin = float(w.dmin);
    const device uchar *q = w.qs;

    float sum = 0.0f;
    uint scale_index = 0u;

    if (stride_k == 1u && block_len == kQK_K) {
#pragma clang loop unroll(full)
        for (uint group = 0; group < 4u; ++group) {
            uchar sc0 = 0u, m0 = 0u;
            uchar sc1 = 0u, m1 = 0u;
            get_scale_min_k4(scale_index++, w.scales, sc0, m0);
            get_scale_min_k4(scale_index++, w.scales, sc1, m1);

            const float scale0 = wd * float(sc0);
            const float min0 = wdmin * float(m0);
            const float scale1 = wd * float(sc1);
            const float min1 = wdmin * float(m1);

            const uint base = block_start + group * 64u;
#pragma clang loop unroll(full)
            for (uint l = 0; l < 32u; ++l) {
                const uchar packed = q[l];
                const float a0 = float(input_row[base + l]);
                const float a1 = float(input_row[base + 32u + l]);
                sum += (scale0 * float(packed & 0x0Fu) - min0) * a0;
                sum += (scale1 * float(packed >> 4) - min1) * a1;
            }
            q += 32;
        }
        return sum;
    }

    for (uint group = 0; group < 4u; ++group) {
        uchar sc0 = 0u, m0 = 0u;
        uchar sc1 = 0u, m1 = 0u;
        get_scale_min_k4(scale_index++, w.scales, sc0, m0);
        get_scale_min_k4(scale_index++, w.scales, sc1, m1);

        const float scale0 = wd * float(sc0);
        const float min0 = wdmin * float(m0);
        const float scale1 = wd * float(sc1);
        const float min1 = wdmin * float(m1);

        const uint group_base = group * 64u;
        if (group_base >= block_len) {
            break;
        }

        const uint half0_len = min(32u, block_len - group_base);
        const uint half1_len = (block_len > (group_base + 32u)) ? min(32u, block_len - (group_base + 32u)) : 0u;
        for (uint l = 0; l < 32u; ++l) {
            const uchar packed = q[l];
            if (l < half0_len) {
                const uint idx = block_start + group_base + l;
                const float a0 = float(input_row[idx * stride_k]);
                sum += (scale0 * float(packed & 0x0Fu) - min0) * a0;
            }
            if (l < half1_len) {
                const uint idx = block_start + group_base + 32u + l;
                const float a1 = float(input_row[idx * stride_k]);
                sum += (scale1 * float(packed >> 4) - min1) * a1;
            }
        }

        q += 32;
    }

    return sum;
}

template <typename InputType>
static inline float compute_q2k_block_dot_direct(
    const device q2_k_block &w, const device InputType *input_row, uint block_start, uint block_len, uint stride_k
) {
    const float d = float(w.d);
    const float dmin = float(w.dmin);
    const device uchar *qs = w.qs;

    float sum = 0.0f;

    if (stride_k == 1u && block_len == kQK_K) {
        const device InputType *row = input_row + block_start;
#pragma clang loop unroll(full)
        for (uint g = 0; g < 16u; ++g) {
            const uchar sc = w.scales[g];
            const float dl = d * float(sc & 0x0Fu);
            const float ml = dmin * float(sc >> 4);

            const ushort low_plane = ushort(qs[2u * g]) | (ushort(qs[2u * g + 1u]) << 8u);
            const ushort high_plane = ushort(qs[32u + 2u * g]) | (ushort(qs[32u + 2u * g + 1u]) << 8u);

            const uint half_group = g >> 3;
            const uint j = g & 0x7u;
            const uint base = j * 32u + half_group * 16u;

            float sum_qa = 0.0f;
            float sum_a = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint q = ((low_plane >> i) & 0x1u) | (((high_plane >> i) & 0x1u) << 1u);
                const float a_val = float(row[base + i]);
                sum_qa += float(q) * a_val;
                sum_a += a_val;
            }

            sum += dl * sum_qa - ml * sum_a;
        }
        return sum;
    }

    for (uint g = 0; g < 16u; ++g) {
        const uchar sc = w.scales[g];
        const float dl = d * float(sc & 0x0Fu);
        const float ml = dmin * float(sc >> 4);

        const ushort low_plane = ushort(qs[2u * g]) | (ushort(qs[2u * g + 1u]) << 8u);
        const ushort high_plane = ushort(qs[32u + 2u * g]) | (ushort(qs[32u + 2u * g + 1u]) << 8u);

        const uint half_group = g >> 3;
        const uint j = g & 0x7u;
        const uint base = j * 32u + half_group * 16u;
        if (base >= block_len) {
            continue;
        }

        const uint limit = min(16u, block_len - base);
        float sum_qa = 0.0f;
        float sum_a = 0.0f;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 16u; ++i) {
            if (i >= limit) {
                break;
            }
            const uint local_idx = base + i;
            const uint global_idx = block_start + local_idx;
            const uint q = ((low_plane >> i) & 0x1u) | (((high_plane >> i) & 0x1u) << 1u);
            const float a_val = float(input_row[global_idx * stride_k]);
            sum_qa += float(q) * a_val;
            sum_a += a_val;
        }

        sum += dl * sum_qa - ml * sum_a;
    }

    return sum;
}

template <typename InputType>
static inline float compute_q3k_block_dot_direct(
    const device q3_k_block &w, const device InputType *input_row, uint block_start, uint block_len, uint stride_k
) {
    uint aux[4] = {0u, 0u, 0u, 0u};
    for (uint i = 0; i < 12u; ++i) {
        const uint idx = i >> 2;
        const uint shift = (i & 3u) * 8u;
        aux[idx] |= uint(w.scales[i]) << shift;
    }

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    const uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    int scales[16];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        const uint word = aux[i >> 2];
        const uint shift = (i & 3u) * 8u;
        scales[i] = int((char)((word >> shift) & 0xFFu));
    }

    const device uchar *q = w.qs;
    const device uchar *hm = w.hmask;
    uint scale_index = 0u;
    uchar m = 1u;
    const float d_all = float(w.d);

    float sum = 0.0f;

    if (stride_k == 1u && block_len == kQK_K) {
        const device InputType *row = input_row + block_start;
#pragma clang loop unroll(full)
        for (uint n = 0; n < kQK_K; n += 128u) {
            uint shift = 0u;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 4u; ++j) {
                const float dl0 = d_all * float(scales[scale_index++] - 32);
                float acc0 = 0.0f;
#pragma clang loop unroll(full)
                for (uint l = 0; l < 16u; ++l) {
                    const uint idx = n + j * 32u + l;
                    int q_val = int((q[l] >> shift) & 0x3u);
                    if ((hm[l] & m) == 0u) {
                        q_val -= 4;
                    }
                    acc0 += float(q_val) * float(row[idx]);
                }

                const float dl1 = d_all * float(scales[scale_index++] - 32);
                float acc1 = 0.0f;
#pragma clang loop unroll(full)
                for (uint l = 0; l < 16u; ++l) {
                    const uint idx = n + j * 32u + 16u + l;
                    int q_val = int((q[l + 16u] >> shift) & 0x3u);
                    if ((hm[l + 16u] & m) == 0u) {
                        q_val -= 4;
                    }
                    acc1 += float(q_val) * float(row[idx]);
                }

                sum += dl0 * acc0;
                sum += dl1 * acc1;
                shift += 2u;
                m <<= 1u;
            }
            q += 32;
            hm += 32;
        }
        return sum;
    }

    for (uint n = 0; n < block_len; n += 128u) {
        uint shift = 0u;
        for (uint j = 0; j < 4u; ++j) {
            const float dl0 = d_all * float(scales[scale_index++] - 32);
            float acc0 = 0.0f;
#pragma clang loop unroll(full)
            for (uint l = 0; l < 16u; ++l) {
                const uint idx = n + j * 32u + l;
                if (idx >= block_len) {
                    continue;
                }
                int q_val = int((q[l] >> shift) & 0x3u);
                if ((hm[l] & m) == 0u) {
                    q_val -= 4;
                }
                const float a_val = float(input_row[(block_start + idx) * stride_k]);
                acc0 += float(q_val) * a_val;
            }

            const float dl1 = d_all * float(scales[scale_index++] - 32);
            float acc1 = 0.0f;
#pragma clang loop unroll(full)
            for (uint l = 0; l < 16u; ++l) {
                const uint idx = n + j * 32u + 16u + l;
                if (idx >= block_len) {
                    continue;
                }
                int q_val = int((q[l + 16u] >> shift) & 0x3u);
                if ((hm[l + 16u] & m) == 0u) {
                    q_val -= 4;
                }
                const float a_val = float(input_row[(block_start + idx) * stride_k]);
                acc1 += float(q_val) * a_val;
            }

            sum += dl0 * acc0;
            sum += dl1 * acc1;
            shift += 2u;
            m <<= 1u;
        }
        q += 32;
        hm += 32;
    }

    return sum;
}

template <typename InputType>
static inline float compute_q5k_block_dot_direct(
    const device q5_k_block &w, const device InputType *input_row, uint block_start, uint block_len, uint stride_k
) {
    const float wd = float(w.d);
    const float wdmin = float(w.dmin);
    const device uchar *ql = w.qs;
    const device uchar *qh = w.qh;

    float sum = 0.0f;
    uint scale_index = 0u;
    uchar u1 = 1u;
    uchar u2 = 2u;

    if (stride_k == 1u && block_len == kQK_K) {
        const device InputType *row = input_row + block_start;
#pragma clang loop unroll(full)
        for (uint base = 0; base < kQK_K; base += 64u) {
            uchar sc0 = 0u, m0 = 0u;
            uchar sc1 = 0u, m1 = 0u;
            get_scale_min_k4(scale_index++, w.scales, sc0, m0);
            get_scale_min_k4(scale_index++, w.scales, sc1, m1);

            const float scale0 = wd * float(sc0);
            const float min0 = wdmin * float(m0);
            const float scale1 = wd * float(sc1);
            const float min1 = wdmin * float(m1);

            float sum_qa0 = 0.0f;
            float sum_a0 = 0.0f;
            float sum_qa1 = 0.0f;
            float sum_a1 = 0.0f;

#pragma clang loop unroll(full)
            for (uint l = 0; l < 32u; ++l) {
                uint qv0 = uint(ql[l] & 0x0Fu);
                if ((qh[l] & u1) != 0u) {
                    qv0 += 16u;
                }
                uint qv1 = uint(ql[l] >> 4);
                if ((qh[l] & u2) != 0u) {
                    qv1 += 16u;
                }

                const float a0 = float(row[base + l]);
                const float a1 = float(row[base + 32u + l]);
                sum_qa0 += float(qv0) * a0;
                sum_a0 += a0;
                sum_qa1 += float(qv1) * a1;
                sum_a1 += a1;
            }

            sum += scale0 * sum_qa0 - min0 * sum_a0;
            sum += scale1 * sum_qa1 - min1 * sum_a1;

            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
        return sum;
    }

    for (uint group = 0; group < 4u; ++group) {
        uchar sc0 = 0u, m0 = 0u;
        uchar sc1 = 0u, m1 = 0u;
        get_scale_min_k4(scale_index++, w.scales, sc0, m0);
        get_scale_min_k4(scale_index++, w.scales, sc1, m1);

        const float scale0 = wd * float(sc0);
        const float min0 = wdmin * float(m0);
        const float scale1 = wd * float(sc1);
        const float min1 = wdmin * float(m1);

        const uint group_base = group * 64u;
        if (group_base >= block_len) {
            break;
        }

        const uint half0_len = min(32u, block_len - group_base);
        const uint half1_len = (block_len > (group_base + 32u)) ? min(32u, block_len - (group_base + 32u)) : 0u;

        float sum_qa0 = 0.0f;
        float sum_a0 = 0.0f;
        float sum_qa1 = 0.0f;
        float sum_a1 = 0.0f;

#pragma clang loop unroll(full)
        for (uint l = 0; l < 32u; ++l) {
            uint qv0 = uint(ql[l] & 0x0Fu);
            if ((qh[l] & u1) != 0u) {
                qv0 += 16u;
            }
            uint qv1 = uint(ql[l] >> 4);
            if ((qh[l] & u2) != 0u) {
                qv1 += 16u;
            }

            if (l < half0_len) {
                const uint idx = block_start + group_base + l;
                const float a0 = float(input_row[idx * stride_k]);
                sum_qa0 += float(qv0) * a0;
                sum_a0 += a0;
            }
            if (l < half1_len) {
                const uint idx = block_start + group_base + 32u + l;
                const float a1 = float(input_row[idx * stride_k]);
                sum_qa1 += float(qv1) * a1;
                sum_a1 += a1;
            }
        }

        sum += scale0 * sum_qa0 - min0 * sum_a0;
        sum += scale1 * sum_qa1 - min1 * sum_a1;

        ql += 32;
        u1 <<= 2;
        u2 <<= 2;
    }

    return sum;
}

template <typename InputType>
static inline float compute_q6k_block_dot_direct(
    const device q6_k_block &w, const device InputType *input_row, uint block_start, uint block_len, uint stride_k
) {
    const float d = float(w.d);
    const device uchar *ql = w.ql;
    const device uchar *qh = w.qh;
    const device char *sc = w.scales;

    float sum = 0.0f;

    if (stride_k == 1u && block_len == kQK_K) {
        const device InputType *row = input_row + block_start;
#pragma clang loop unroll(full)
        for (uint base = 0; base < kQK_K; base += 128u) {
#pragma clang loop unroll(full)
            for (uint l = 0; l < 32u; ++l) {
                const uint is = l >> 4;

                const int s0 = int(sc[is + 0]);
                const int s1 = int(sc[is + 2]);
                const int s2 = int(sc[is + 4]);
                const int s3 = int(sc[is + 6]);

                const int q1 = int((ql[l + 0] & 0x0Fu) | (((qh[l] >> 0) & 0x3u) << 4)) - 32;
                const int q2 = int((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 0x3u) << 4)) - 32;
                const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 0x3u) << 4)) - 32;
                const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x3u) << 4)) - 32;

                const float a0 = float(row[base + l]);
                const float a1 = float(row[base + 32u + l]);
                const float a2 = float(row[base + 64u + l]);
                const float a3 = float(row[base + 96u + l]);

                sum += d * (float(q1 * s0) * a0 + float(q2 * s1) * a1 + float(q3 * s2) * a2 + float(q4 * s3) * a3);
            }
            ql += 64;
            qh += 32;
            sc += 8;
        }
        return sum;
    }

    for (uint base = 0; base < block_len; base += 128u) {
        const uint local_len = min(128u, block_len - base);
        for (uint l = 0; l < 32u; ++l) {
            const uint is = l >> 4;

            const int s0 = int(sc[is + 0]);
            const int s1 = int(sc[is + 2]);
            const int s2 = int(sc[is + 4]);
            const int s3 = int(sc[is + 6]);

            const int q1 = int((ql[l + 0] & 0x0Fu) | (((qh[l] >> 0) & 0x3u) << 4)) - 32;
            const int q2 = int((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 0x3u) << 4)) - 32;
            const int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 0x3u) << 4)) - 32;
            const int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x3u) << 4)) - 32;

            if (l < local_len) {
                const uint idx = block_start + base + l;
                sum += d * float(q1 * s0) * float(input_row[idx * stride_k]);
            }
            if ((32u + l) < local_len) {
                const uint idx = block_start + base + 32u + l;
                sum += d * float(q2 * s1) * float(input_row[idx * stride_k]);
            }
            if ((64u + l) < local_len) {
                const uint idx = block_start + base + 64u + l;
                sum += d * float(q3 * s2) * float(input_row[idx * stride_k]);
            }
            if ((96u + l) < local_len) {
                const uint idx = block_start + base + 96u + l;
                sum += d * float(q4 * s3) * float(input_row[idx * stride_k]);
            }
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }

    return sum;
}

static inline float4 marmot_load_float4(device const float *ptr) {
    return *((device const float4 *)ptr);
}

static inline float4 marmot_load_float4(device const half *ptr) {
    return float4(*((device const half4 *)ptr));
}

static inline half4 marmot_load_half4(device const float *ptr) {
    return half4(*((device const float4 *)ptr));
}

static inline half4 marmot_load_half4(device const half *ptr) {
    return *((device const half4 *)ptr);
}

template <typename InputType>
static inline float compute_q8k_block_dot_direct(
    const device q8_k_block &w, const device InputType *input_row, uint block_start, uint block_len, uint stride_k
) {
    const float scale = w.d;
    const device char *wq = w.qs;

    float sum = 0.0f;
    if (stride_k == 1u && block_len == kQK_K) {
        const device InputType *row = input_row + block_start;
        for (uint i = 0; i < kQK_K; i += 4u) {
            const char4 q = *((device const char4 *)(wq + i));
            sum += dot(float4(q) * scale, marmot_load_float4(row + i));
        }
        return sum;
    }

    for (uint i = 0; i < block_len; ++i) {
        const float a = float(input_row[(block_start + i) * stride_k]);
        sum += (scale * float(wq[i])) * a;
    }

    return sum;
}

#define DEFINE_QK_DIRECT_KERNELS(NAME, BLOCK_T, INPUT_T, OUTPUT_T, DOT_FN)                                             \
    kernel void NAME##_opt(                                                                                            \
        device const BLOCK_T *weight [[buffer(0)]], device const INPUT_T *input [[buffer(1)]],                         \
        device OUTPUT_T *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],         \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],                        \
        uint simd_lane [[thread_index_in_simdgroup]]                                                                   \
    ) {                                                                                                                \
        const uint m = tgp.x;                                                                                          \
        const uint n = tgp.y;                                                                                          \
        if (n >= N || m >= M) {                                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        device const BLOCK_T *weight_row = weight + m * weight_blocks;                                                 \
        device const INPUT_T *input_row = input + n * stride_n;                                                        \
        float partial = 0.0f;                                                                                          \
        for (uint sb = simd_lane; sb < weight_blocks; sb += 32u) {                                                     \
            const uint block_start = sb * kQK_K;                                                                       \
            if (block_start >= K) {                                                                                    \
                break;                                                                                                 \
            }                                                                                                          \
            const uint block_len = min(kQK_K, K - block_start);                                                        \
            partial += DOT_FN(weight_row[sb], input_row, block_start, block_len, stride_k);                            \
        }                                                                                                              \
        float result = simd_sum(partial);                                                                              \
        if (simd_lane == 0u) {                                                                                         \
            output[n * M + m] = OUTPUT_T(result);                                                                      \
        }                                                                                                              \
    }                                                                                                                  \
    kernel void NAME##_opt_nr2(                                                                                        \
        device const BLOCK_T *weight [[buffer(0)]], device const INPUT_T *input [[buffer(1)]],                         \
        device OUTPUT_T *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],         \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],                        \
        uint simd_lane [[thread_index_in_simdgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]                    \
    ) {                                                                                                                \
        const uint m = tgp.x;                                                                                          \
        const uint n_base = tgp.y * 4u + (uint)sgitg * 2u;                                                             \
        if (m >= M || n_base >= N) {                                                                                   \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        device const BLOCK_T *weight_row = weight + m * weight_blocks;                                                 \
        device const INPUT_T *row0 = input + n_base * stride_n;                                                        \
        device const INPUT_T *row1 = input + (n_base + 1u) * stride_n;                                                 \
        const bool has_row1 = (n_base + 1u) < N;                                                                       \
        float partial0 = 0.0f;                                                                                         \
        float partial1 = 0.0f;                                                                                         \
        for (uint sb = simd_lane; sb < weight_blocks; sb += 32u) {                                                     \
            const uint block_start = sb * kQK_K;                                                                       \
            if (block_start >= K) {                                                                                    \
                break;                                                                                                 \
            }                                                                                                          \
            const uint block_len = min(kQK_K, K - block_start);                                                        \
            const device BLOCK_T &w_blk = weight_row[sb];                                                              \
            partial0 += DOT_FN(w_blk, row0, block_start, block_len, stride_k);                                         \
            if (has_row1) {                                                                                            \
                partial1 += DOT_FN(w_blk, row1, block_start, block_len, stride_k);                                     \
            }                                                                                                          \
        }                                                                                                              \
        float result0 = simd_sum(partial0);                                                                            \
        float result1 = simd_sum(partial1);                                                                            \
        if (simd_lane == 0u) {                                                                                         \
            output[n_base * M + m] = OUTPUT_T(result0);                                                                \
            if (has_row1) {                                                                                            \
                output[(n_base + 1u) * M + m] = OUTPUT_T(result1);                                                     \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    kernel void NAME##_small(                                                                                          \
        device const BLOCK_T *weight [[buffer(0)]], device const INPUT_T *input [[buffer(1)]],                         \
        device OUTPUT_T *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],         \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]                              \
    ) {                                                                                                                \
        const uint m = gid.x;                                                                                          \
        const uint n = gid.y;                                                                                          \
        if (n >= N || m >= M) {                                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        device const BLOCK_T *weight_row = weight + m * weight_blocks;                                                 \
        device const INPUT_T *input_row = input + n * stride_n;                                                        \
        float sum = 0.0f;                                                                                              \
        for (uint sb = 0; sb < weight_blocks; ++sb) {                                                                  \
            const uint block_start = sb * kQK_K;                                                                       \
            if (block_start >= K) {                                                                                    \
                break;                                                                                                 \
            }                                                                                                          \
            const uint block_len = min(kQK_K, K - block_start);                                                        \
            sum += DOT_FN(weight_row[sb], input_row, block_start, block_len, stride_k);                                \
        }                                                                                                              \
        output[n * M + m] = OUTPUT_T(sum);                                                                             \
    }

#define DEFINE_QK_DIRECT_SMALL_KERNEL(NAME, BLOCK_T, INPUT_T, OUTPUT_T, DOT_FN)                                        \
    kernel void NAME##_small(                                                                                          \
        device const BLOCK_T *weight [[buffer(0)]], device const INPUT_T *input [[buffer(1)]],                         \
        device OUTPUT_T *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],         \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]                              \
    ) {                                                                                                                \
        const uint m = gid.x;                                                                                          \
        const uint n = gid.y;                                                                                          \
        if (n >= N || m >= M) {                                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
                                                                                                                       \
        device const BLOCK_T *weight_row = weight + m * weight_blocks;                                                 \
        device const INPUT_T *input_row = input + n * stride_n;                                                        \
        float sum = 0.0f;                                                                                              \
        for (uint sb = 0; sb < weight_blocks; ++sb) {                                                                  \
            const uint block_start = sb * kQK_K;                                                                       \
            if (block_start >= K) {                                                                                    \
                break;                                                                                                 \
            }                                                                                                          \
            const uint block_len = min(kQK_K, K - block_start);                                                        \
            sum += DOT_FN(weight_row[sb], input_row, block_start, block_len, stride_k);                                \
        }                                                                                                              \
        output[n * M + m] = OUTPUT_T(sum);                                                                             \
    }

DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q4_k_f32_f32, q4_k_block, float, float, compute_q4k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q4_k_f16_f32, q4_k_block, half, float, compute_q4k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q4_k_f16_f16, q4_k_block, half, half, compute_q4k_block_dot_direct)
DEFINE_QK_DIRECT_KERNELS(matmul_q2_k_f32_f32, q2_k_block, float, float, compute_q2k_block_dot_direct)
DEFINE_QK_DIRECT_KERNELS(matmul_q2_k_f16_f32, q2_k_block, half, float, compute_q2k_block_dot_direct)
DEFINE_QK_DIRECT_KERNELS(matmul_q2_k_f16_f16, q2_k_block, half, half, compute_q2k_block_dot_direct)
DEFINE_QK_DIRECT_KERNELS(matmul_q3_k_f32_f32, q3_k_block, float, float, compute_q3k_block_dot_direct)
DEFINE_QK_DIRECT_KERNELS(matmul_q3_k_f16_f32, q3_k_block, half, float, compute_q3k_block_dot_direct)
DEFINE_QK_DIRECT_KERNELS(matmul_q3_k_f16_f16, q3_k_block, half, half, compute_q3k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q5_k_f32_f32, q5_k_block, float, float, compute_q5k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q5_k_f16_f32, q5_k_block, half, float, compute_q5k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q5_k_f16_f16, q5_k_block, half, half, compute_q5k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q6_k_f32_f32, q6_k_block, float, float, compute_q6k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q6_k_f16_f32, q6_k_block, half, float, compute_q6k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q6_k_f16_f16, q6_k_block, half, half, compute_q6k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q8_k_f32_f32, q8_k_block, float, float, compute_q8k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q8_k_f16_f32, q8_k_block, half, float, compute_q8k_block_dot_direct)
DEFINE_QK_DIRECT_SMALL_KERNEL(matmul_q8_k_f16_f16, q8_k_block, half, half, compute_q8k_block_dot_direct)

#undef DEFINE_QK_DIRECT_KERNELS
#undef DEFINE_QK_DIRECT_SMALL_KERNEL

static inline float marmot_sum_float4(float4 v) {
    return v.x + v.y + v.z + v.w;
}

template <typename InputType>
static inline void marmot_mv_load8(
    const device InputType *input_row, uint base, uint K, uint stride_k, thread float4 &a0, thread float4 &a1,
    thread float &sum_a
) {
    if (stride_k == 1u && (base + 7u) < K) {
        a0 = marmot_load_float4(input_row + base);
        a1 = marmot_load_float4(input_row + base + 4u);
    } else {
        float tmp[8] = {0.0f};
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint gk = base + i;
            tmp[i] = gk < K ? float(input_row[gk * stride_k]) : 0.0f;
        }
        a0 = float4(tmp[0], tmp[1], tmp[2], tmp[3]);
        a1 = float4(tmp[4], tmp[5], tmp[6], tmp[7]);
    }
    sum_a = marmot_sum_float4(a0) + marmot_sum_float4(a1);
}

template <typename InputType>
static inline float marmot_mv_load_scalar(device const InputType *input_row, uint gk, uint K, uint stride_k) {
    if (gk >= K) {
        return 0.0f;
    }
    return float(input_row[gk * stride_k]);
}

static inline float marmot_mv_q4_k_dot_block(
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

template <typename InputType, typename OutputType>
static inline void matmul_q4_k_mv_compute(
    device const q4_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 2u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < M;

    device const InputType *input_row = input + n * stride_n;
    device const q4_k_block *weight_row0 = weight + m0 * weight_blocks;
    device const q4_k_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;

    const uint tid = uint(tiisg);
    const uint ix = tid >> 3;
    const uint it = tid & 7u;
    const uint iq = it >> 2;
    const uint ir = it & 3u;

    float yl[16];
    float yh[16];
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const bool full_k = (K & (kQK_K - 1u)) == 0u;
    if (stride_k == 1u && full_k) {
        const uint nb = min(weight_blocks, K / kQK_K);
        device const InputType *y4 = input_row + ix * kQK_K + 64u * iq + 8u * ir;
        device const q4_k_block *x0 = weight_row0 + ix;
        device const q4_k_block *x1 = weight_row1 + ix;

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

            sum0 += marmot_mv_q4_k_dot_block(*x0, yl, yh, sumy, iq, ir);
            sum1 += marmot_mv_q4_k_dot_block(*x1, yl, yh, sumy, iq, ir);
            y4 += 4u * kQK_K;
            x0 += 4u;
            x1 += 4u;
        }
    } else {
        for (uint sb = ix; sb < weight_blocks; sb += 4u) {
            const uint block_start = sb * kQK_K;
            if (block_start >= K) {
                break;
            }

            float4 sumy = 0.0f;

            const uint y_base = block_start + 64u * iq + 8u * ir;
            if (stride_k == 1u && (block_start + kQK_K) <= K) {
                device const InputType *y_ptr = input_row + y_base;
                for (uint i = 0; i < 8u; ++i) {
                    const float v0 = float(y_ptr[i + 0u]);
                    const float v1 = float(y_ptr[i + 32u]);
                    const float v2 = float(y_ptr[i + 128u]);
                    const float v3 = float(y_ptr[i + 160u]);

                    yl[i + 0u] = v0;
                    yl[i + 8u] = v1;
                    yh[i + 0u] = v2;
                    yh[i + 8u] = v3;

                    sumy[0] += v0;
                    sumy[1] += v1;
                    sumy[2] += v2;
                    sumy[3] += v3;
                }
            } else {
                for (uint i = 0; i < 8u; ++i) {
                    const float v0 = marmot_mv_load_scalar(input_row, y_base + i + 0u, K, stride_k);
                    const float v1 = marmot_mv_load_scalar(input_row, y_base + i + 32u, K, stride_k);
                    const float v2 = marmot_mv_load_scalar(input_row, y_base + i + 128u, K, stride_k);
                    const float v3 = marmot_mv_load_scalar(input_row, y_base + i + 160u, K, stride_k);

                    yl[i + 0u] = v0;
                    yl[i + 8u] = v1;
                    yh[i + 0u] = v2;
                    yh[i + 8u] = v3;

                    sumy[0] += v0;
                    sumy[1] += v1;
                    sumy[2] += v2;
                    sumy[3] += v3;
                }
            }

            sum0 += marmot_mv_q4_k_dot_block(weight_row0[sb], yl, yh, sumy, iq, ir);
            sum1 += marmot_mv_q4_k_dot_block(weight_row1[sb], yl, yh, sumy, iq, ir);
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
    }
}

kernel void matmul_q4_k_f32_f32_mv(
    device const q4_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_k_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q4_k_f16_f32_mv(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_k_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q4_k_f16_f16_mv(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_k_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q4_k_dual_mv_compute(
    device const q4_k_block *weight_q, device const q4_k_block *weight_k, device const InputType *input,
    device OutputType *output_q, device OutputType *output_k, uint N, uint K, uint M, uint stride_n, uint stride_k,
    uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 2u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < M;

    device const InputType *input_row = input + n * stride_n;
    device const q4_k_block *weight_row_q0 = weight_q + m0 * weight_blocks;
    device const q4_k_block *weight_row_q1 = has_m1 ? (weight_row_q0 + weight_blocks) : weight_row_q0;
    device const q4_k_block *weight_row_k0 = weight_k + m0 * weight_blocks;
    device const q4_k_block *weight_row_k1 = has_m1 ? (weight_row_k0 + weight_blocks) : weight_row_k0;

    const uint tid = uint(tiisg);
    const uint ix = tid >> 3;
    const uint it = tid & 7u;
    const uint iq = it >> 2;
    const uint ir = it & 3u;

    float yl[16];
    float yh[16];
    float sum_q0 = 0.0f;
    float sum_q1 = 0.0f;
    float sum_k0 = 0.0f;
    float sum_k1 = 0.0f;

    const bool full_k = (K & (kQK_K - 1u)) == 0u;
    if (stride_k == 1u && full_k) {
        const uint nb = min(weight_blocks, K / kQK_K);
        device const InputType *y4 = input_row + ix * kQK_K + 64u * iq + 8u * ir;
        device const q4_k_block *xq0 = weight_row_q0 + ix;
        device const q4_k_block *xq1 = weight_row_q1 + ix;
        device const q4_k_block *xk0 = weight_row_k0 + ix;
        device const q4_k_block *xk1 = weight_row_k1 + ix;

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

            sum_q0 += marmot_mv_q4_k_dot_block(*xq0, yl, yh, sumy, iq, ir);
            sum_q1 += marmot_mv_q4_k_dot_block(*xq1, yl, yh, sumy, iq, ir);
            sum_k0 += marmot_mv_q4_k_dot_block(*xk0, yl, yh, sumy, iq, ir);
            sum_k1 += marmot_mv_q4_k_dot_block(*xk1, yl, yh, sumy, iq, ir);
            y4 += 4u * kQK_K;
            xq0 += 4u;
            xq1 += 4u;
            xk0 += 4u;
            xk1 += 4u;
        }
    } else {
        for (uint sb = ix; sb < weight_blocks; sb += 4u) {
            const uint block_start = sb * kQK_K;
            if (block_start >= K) {
                break;
            }

            float4 sumy = 0.0f;
            const uint y_base = block_start + 64u * iq + 8u * ir;
            if (stride_k == 1u && (block_start + kQK_K) <= K) {
                device const InputType *y_ptr = input_row + y_base;
                for (uint i = 0; i < 8u; ++i) {
                    const float v0 = float(y_ptr[i + 0u]);
                    const float v1 = float(y_ptr[i + 32u]);
                    const float v2 = float(y_ptr[i + 128u]);
                    const float v3 = float(y_ptr[i + 160u]);

                    yl[i + 0u] = v0;
                    yl[i + 8u] = v1;
                    yh[i + 0u] = v2;
                    yh[i + 8u] = v3;

                    sumy[0] += v0;
                    sumy[1] += v1;
                    sumy[2] += v2;
                    sumy[3] += v3;
                }
            } else {
                for (uint i = 0; i < 8u; ++i) {
                    const float v0 = marmot_mv_load_scalar(input_row, y_base + i + 0u, K, stride_k);
                    const float v1 = marmot_mv_load_scalar(input_row, y_base + i + 32u, K, stride_k);
                    const float v2 = marmot_mv_load_scalar(input_row, y_base + i + 128u, K, stride_k);
                    const float v3 = marmot_mv_load_scalar(input_row, y_base + i + 160u, K, stride_k);

                    yl[i + 0u] = v0;
                    yl[i + 8u] = v1;
                    yh[i + 0u] = v2;
                    yh[i + 8u] = v3;

                    sumy[0] += v0;
                    sumy[1] += v1;
                    sumy[2] += v2;
                    sumy[3] += v3;
                }
            }

            sum_q0 += marmot_mv_q4_k_dot_block(weight_row_q0[sb], yl, yh, sumy, iq, ir);
            sum_q1 += marmot_mv_q4_k_dot_block(weight_row_q1[sb], yl, yh, sumy, iq, ir);
            sum_k0 += marmot_mv_q4_k_dot_block(weight_row_k0[sb], yl, yh, sumy, iq, ir);
            sum_k1 += marmot_mv_q4_k_dot_block(weight_row_k1[sb], yl, yh, sumy, iq, ir);
        }
    }

    const float out_q0 = simd_sum(sum_q0);
    const float out_q1 = simd_sum(sum_q1);
    const float out_k0 = simd_sum(sum_k0);
    const float out_k1 = simd_sum(sum_k1);
    if (tiisg == 0) {
        output_q[n * M + m0] = OutputType(out_q0);
        output_k[n * M + m0] = OutputType(out_k0);
        if (has_m1) {
            output_q[n * M + m1] = OutputType(out_q1);
            output_k[n * M + m1] = OutputType(out_k1);
        }
    }
}

kernel void matmul_qkv_q4_k_dual_f32_f32_mv(
    device const q4_k_block *weight_q [[buffer(0)]], device const q4_k_block *weight_k [[buffer(1)]],
    device const q4_k_block *weight_v [[buffer(2)]], device const float *input [[buffer(3)]],
    device float *out_q [[buffer(4)]], device float *out_k [[buffer(5)]], device float *out_v [[buffer(6)]],
    constant uint &N [[buffer(7)]], constant uint &K [[buffer(8)]], constant uint &M [[buffer(9)]],
    constant uint &stride_n [[buffer(10)]], constant uint &stride_k [[buffer(11)]],
    constant uint &weight_blocks [[buffer(12)]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]], uint2 tgp [[threadgroup_position_in_grid]]
) {
    (void)weight_v;
    (void)out_v;
    matmul_q4_k_dual_mv_compute<float, float>(
        weight_q, weight_k, input, out_q, out_k, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_qkv_q4_k_dual_f16_f32_mv(
    device const q4_k_block *weight_q [[buffer(0)]], device const q4_k_block *weight_k [[buffer(1)]],
    device const q4_k_block *weight_v [[buffer(2)]], device const half *input [[buffer(3)]],
    device float *out_q [[buffer(4)]], device float *out_k [[buffer(5)]], device float *out_v [[buffer(6)]],
    constant uint &N [[buffer(7)]], constant uint &K [[buffer(8)]], constant uint &M [[buffer(9)]],
    constant uint &stride_n [[buffer(10)]], constant uint &stride_k [[buffer(11)]],
    constant uint &weight_blocks [[buffer(12)]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]], uint2 tgp [[threadgroup_position_in_grid]]
) {
    (void)weight_v;
    (void)out_v;
    matmul_q4_k_dual_mv_compute<half, float>(
        weight_q, weight_k, input, out_q, out_k, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_qkv_q4_k_dual_f16_f16_mv(
    device const q4_k_block *weight_q [[buffer(0)]], device const q4_k_block *weight_k [[buffer(1)]],
    device const q4_k_block *weight_v [[buffer(2)]], device const half *input [[buffer(3)]],
    device half *out_q [[buffer(4)]], device half *out_k [[buffer(5)]], device half *out_v [[buffer(6)]],
    constant uint &N [[buffer(7)]], constant uint &K [[buffer(8)]], constant uint &M [[buffer(9)]],
    constant uint &stride_n [[buffer(10)]], constant uint &stride_k [[buffer(11)]],
    constant uint &weight_blocks [[buffer(12)]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]], uint2 tgp [[threadgroup_position_in_grid]]
) {
    (void)weight_v;
    (void)out_v;
    matmul_q4_k_dual_mv_compute<half, half>(
        weight_q, weight_k, input, out_q, out_k, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename OutputType>
static inline void matmul_q4_k_mv_ext_r1_4_compute_f32(
    device const q4_k_block *weight, device const float *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 16;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q4_k_block *weight_row = weight + m * weight_blocks;
    device const q4_k_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const float4x4 *y0 =
        (n_base + 0u < N) ? ((device const float4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const float4x4 *y1 =
        (n_base + 1u < N) ? ((device const float4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const float4x4 *y2 =
        (n_base + 2u < N) ? ((device const float4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const float4x4 *y3 =
        (n_base + 3u < N) ? ((device const float4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        dequantize_q4_k_chunk_f32(*xq, cch, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const float4x4 yy = y0[0];
            sum0 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const float4x4 yy = y1[0];
            sum1 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const float4x4 yy = y2[0];
            sum2 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const float4x4 yy = y3[0];
            sum3 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

template <typename OutputType>
static inline void matmul_q4_k_mv_ext_r1_4_compute_f16(
    device const q4_k_block *weight, device const half *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 16;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q4_k_block *weight_row = weight + m * weight_blocks;
    device const q4_k_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const half4x4 *y0 =
        (n_base + 0u < N) ? ((device const half4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const half4x4 *y1 =
        (n_base + 1u < N) ? ((device const half4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const half4x4 *y2 =
        (n_base + 2u < N) ? ((device const half4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const half4x4 *y3 =
        (n_base + 3u < N) ? ((device const half4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        dequantize_q4_k_chunk_f32(*xq, cch, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const half4x4 yyh = y0[0];
            sum0 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const half4x4 yyh = y1[0];
            sum1 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const half4x4 yyh = y2[0];
            sum2 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const half4x4 yyh = y3[0];
            sum3 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

kernel void matmul_q4_k_f32_f32_mv_ext_r1_4(
    device const q4_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_k_mv_ext_r1_4_compute_f32<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q4_k_f16_f32_mv_ext_r1_4(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_k_mv_ext_r1_4_compute_f16<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q4_k_f16_f16_mv_ext_r1_4(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_k_mv_ext_r1_4_compute_f16<half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename OutputType>
static inline void matmul_q5_k_mv_ext_r1_4_compute_f32(
    device const q5_k_block *weight, device const float *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 16;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q5_k_block *weight_row = weight + m * weight_blocks;
    device const q5_k_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const float4x4 *y0 =
        (n_base + 0u < N) ? ((device const float4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const float4x4 *y1 =
        (n_base + 1u < N) ? ((device const float4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const float4x4 *y2 =
        (n_base + 2u < N) ? ((device const float4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const float4x4 *y3 =
        (n_base + 3u < N) ? ((device const float4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        dequantize_q5_k_chunk(*xq, cch, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const float4x4 yy = y0[0];
            sum0 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const float4x4 yy = y1[0];
            sum1 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const float4x4 yy = y2[0];
            sum2 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const float4x4 yy = y3[0];
            sum3 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

template <typename OutputType>
static inline void matmul_q5_k_mv_ext_r1_4_compute_f16(
    device const q5_k_block *weight, device const half *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 16;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q5_k_block *weight_row = weight + m * weight_blocks;
    device const q5_k_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const half4x4 *y0 =
        (n_base + 0u < N) ? ((device const half4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const half4x4 *y1 =
        (n_base + 1u < N) ? ((device const half4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const half4x4 *y2 =
        (n_base + 2u < N) ? ((device const half4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const half4x4 *y3 =
        (n_base + 3u < N) ? ((device const half4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        dequantize_q5_k_chunk(*xq, cch, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const half4x4 yyh = y0[0];
            sum0 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const half4x4 yyh = y1[0];
            sum1 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const half4x4 yyh = y2[0];
            sum2 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const half4x4 yyh = y3[0];
            sum3 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

kernel void matmul_q5_k_f32_f32_mv_ext_r1_4(
    device const q5_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_k_mv_ext_r1_4_compute_f32<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_k_f16_f32_mv_ext_r1_4(
    device const q5_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_k_mv_ext_r1_4_compute_f16<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_k_f16_f16_mv_ext_r1_4(
    device const q5_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_k_mv_ext_r1_4_compute_f16<half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename OutputType>
static inline void matmul_q6_k_mv_ext_r1_4_compute_f32(
    device const q6_k_block *weight, device const float *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 16;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q6_k_block *weight_row = weight + m * weight_blocks;
    device const q6_k_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const float4x4 *y0 =
        (n_base + 0u < N) ? ((device const float4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const float4x4 *y1 =
        (n_base + 1u < N) ? ((device const float4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const float4x4 *y2 =
        (n_base + 2u < N) ? ((device const float4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const float4x4 *y3 =
        (n_base + 3u < N) ? ((device const float4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        dequantize_q6_k_chunk(*xq, cch, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const float4x4 yy = y0[0];
            sum0 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const float4x4 yy = y1[0];
            sum1 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const float4x4 yy = y2[0];
            sum2 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const float4x4 yy = y3[0];
            sum3 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

template <typename OutputType>
static inline void matmul_q6_k_mv_ext_r1_4_compute_f16(
    device const q6_k_block *weight, device const half *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 16;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q6_k_block *weight_row = weight + m * weight_blocks;
    device const q6_k_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const half4x4 *y0 =
        (n_base + 0u < N) ? ((device const half4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const half4x4 *y1 =
        (n_base + 1u < N) ? ((device const half4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const half4x4 *y2 =
        (n_base + 2u < N) ? ((device const half4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const half4x4 *y3 =
        (n_base + 3u < N) ? ((device const half4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        dequantize_q6_k_chunk(*xq, cch, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const half4x4 yyh = y0[0];
            sum0 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const half4x4 yyh = y1[0];
            sum1 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const half4x4 yyh = y2[0];
            sum2 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const half4x4 yyh = y3[0];
            sum3 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

kernel void matmul_q6_k_f32_f32_mv_ext_r1_4(
    device const q6_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q6_k_mv_ext_r1_4_compute_f32<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q6_k_f16_f32_mv_ext_r1_4(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q6_k_mv_ext_r1_4_compute_f16<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q6_k_f16_f16_mv_ext_r1_4(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q6_k_mv_ext_r1_4_compute_f16<half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename OutputType>
static inline void matmul_q5_0_mv_ext_r1_4_compute_f32(
    device const q5_0_block *weight, device const float *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 2;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q5_0_block *weight_row = weight + m * weight_blocks;
    device const q5_0_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const float4x4 *y0 =
        (n_base + 0u < N) ? ((device const float4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const float4x4 *y1 =
        (n_base + 1u < N) ? ((device const float4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const float4x4 *y2 =
        (n_base + 2u < N) ? ((device const float4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const float4x4 *y3 =
        (n_base + 3u < N) ? ((device const float4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        const short il = (cch != 0) ? short(16) : short(0);
        dequantize_q5_0_chunk(*xq, il, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const float4x4 yy = y0[0];
            sum0 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const float4x4 yy = y1[0];
            sum1 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const float4x4 yy = y2[0];
            sum2 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const float4x4 yy = y3[0];
            sum3 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

template <typename OutputType>
static inline void matmul_q5_0_mv_ext_r1_4_compute_f16(
    device const q5_0_block *weight, device const half *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 2;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q5_0_block *weight_row = weight + m * weight_blocks;
    device const q5_0_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const half4x4 *y0 =
        (n_base + 0u < N) ? ((device const half4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const half4x4 *y1 =
        (n_base + 1u < N) ? ((device const half4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const half4x4 *y2 =
        (n_base + 2u < N) ? ((device const half4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const half4x4 *y3 =
        (n_base + 3u < N) ? ((device const half4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        const short il = (cch != 0) ? short(16) : short(0);
        dequantize_q5_0_chunk(*xq, il, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const half4x4 yyh = y0[0];
            sum0 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const half4x4 yyh = y1[0];
            sum1 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const half4x4 yyh = y2[0];
            sum2 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const half4x4 yyh = y3[0];
            sum3 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

kernel void matmul_q5_0_f32_f32_mv_ext_r1_4(
    device const q5_0_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_0_mv_ext_r1_4_compute_f32<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_0_f16_f32_mv_ext_r1_4(
    device const q5_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_0_mv_ext_r1_4_compute_f16<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_0_f16_f16_mv_ext_r1_4(
    device const q5_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_0_mv_ext_r1_4_compute_f16<half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename OutputType>
static inline void matmul_q5_1_mv_ext_r1_4_compute_f32(
    device const q5_1_block *weight, device const float *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 2;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q5_1_block *weight_row = weight + m * weight_blocks;
    device const q5_1_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const float4x4 *y0 =
        (n_base + 0u < N) ? ((device const float4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const float4x4 *y1 =
        (n_base + 1u < N) ? ((device const float4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const float4x4 *y2 =
        (n_base + 2u < N) ? ((device const float4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const float4x4 *y3 =
        (n_base + 3u < N) ? ((device const float4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        const short il = (cch != 0) ? short(16) : short(0);
        dequantize_q5_1_chunk(*xq, il, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const float4x4 yy = y0[0];
            sum0 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const float4x4 yy = y1[0];
            sum1 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const float4x4 yy = y2[0];
            sum2 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const float4x4 yy = y3[0];
            sum3 += dot(w0, yy[0]) + dot(w1, yy[1]) + dot(w2, yy[2]) + dot(w3, yy[3]);
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

template <typename OutputType>
static inline void matmul_q5_1_mv_ext_r1_4_compute_f16(
    device const q5_1_block *weight, device const half *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr ushort nsg = 2;
    constexpr ushort nxpsg = 8;
    constexpr ushort nypsg = 32 / nxpsg;
    constexpr uint r0ptg = uint(nypsg) * uint(nsg);
    constexpr uint r1ptg = 4;
    constexpr short chpb = 2;

    if (stride_k != 1u || (stride_n & 15u) != 0u || (K & 15u) != 0u) {
        return;
    }

    const ushort tx = tiisg & (nxpsg - 1);
    const ushort ty = tiisg / nxpsg;

    const uint m = tgp.x * r0ptg + uint(sgitg) * uint(nypsg) + uint(ty);
    if (m >= M) {
        return;
    }

    const uint n_base = tgp.y * r1ptg;

    device const q5_1_block *weight_row = weight + m * weight_blocks;
    device const q5_1_block *xq = weight_row + (tx / chpb);
    short cch = short(tx % chpb);

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    device const half4x4 *y0 =
        (n_base + 0u < N) ? ((device const half4x4 *)(input + (n_base + 0u) * stride_n) + tx) : nullptr;
    device const half4x4 *y1 =
        (n_base + 1u < N) ? ((device const half4x4 *)(input + (n_base + 1u) * stride_n) + tx) : nullptr;
    device const half4x4 *y2 =
        (n_base + 2u < N) ? ((device const half4x4 *)(input + (n_base + 2u) * stride_n) + tx) : nullptr;
    device const half4x4 *y3 =
        (n_base + 3u < N) ? ((device const half4x4 *)(input + (n_base + 3u) * stride_n) + tx) : nullptr;

    for (uint ich = uint(tx); (ich * 16u) < K; ich += nxpsg) {
        float4x4 lx;
        const short il = (cch != 0) ? short(16) : short(0);
        dequantize_q5_1_chunk(*xq, il, lx);

        const float4 w0 = lx[0];
        const float4 w1 = lx[1];
        const float4 w2 = lx[2];
        const float4 w3 = lx[3];

        if (y0 != nullptr) {
            const half4x4 yyh = y0[0];
            sum0 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y0 += nxpsg;
        }
        if (y1 != nullptr) {
            const half4x4 yyh = y1[0];
            sum1 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y1 += nxpsg;
        }
        if (y2 != nullptr) {
            const half4x4 yyh = y2[0];
            sum2 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y2 += nxpsg;
        }
        if (y3 != nullptr) {
            const half4x4 yyh = y3[0];
            sum3 +=
                dot(w0, float4(yyh[0])) + dot(w1, float4(yyh[1])) + dot(w2, float4(yyh[2])) + dot(w3, float4(yyh[3]));
            y3 += nxpsg;
        }

        cch += nxpsg;
        if (cch >= chpb) {
            xq += cch / chpb;
            cch %= chpb;
        }
    }

    if (tx < 4) {
        sum0 += simd_shuffle_down(sum0, 4);
        sum1 += simd_shuffle_down(sum1, 4);
        sum2 += simd_shuffle_down(sum2, 4);
        sum3 += simd_shuffle_down(sum3, 4);
    }

    if (tx < 2) {
        sum0 += simd_shuffle_down(sum0, 2);
        sum1 += simd_shuffle_down(sum1, 2);
        sum2 += simd_shuffle_down(sum2, 2);
        sum3 += simd_shuffle_down(sum3, 2);
    }

    if (tx < 1) {
        sum0 += simd_shuffle_down(sum0, 1);
        sum1 += simd_shuffle_down(sum1, 1);
        sum2 += simd_shuffle_down(sum2, 1);
        sum3 += simd_shuffle_down(sum3, 1);
    }

    if (tx != 0) {
        return;
    }

    const uint out_base = m;
    if (n_base + 0u < N) {
        output[(n_base + 0u) * M + out_base] = OutputType(sum0);
    }
    if (n_base + 1u < N) {
        output[(n_base + 1u) * M + out_base] = OutputType(sum1);
    }
    if (n_base + 2u < N) {
        output[(n_base + 2u) * M + out_base] = OutputType(sum2);
    }
    if (n_base + 3u < N) {
        output[(n_base + 3u) * M + out_base] = OutputType(sum3);
    }
}

kernel void matmul_q5_1_f32_f32_mv_ext_r1_4(
    device const q5_1_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_1_mv_ext_r1_4_compute_f32<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_1_f16_f32_mv_ext_r1_4(
    device const q5_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_1_mv_ext_r1_4_compute_f16<float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_1_f16_f16_mv_ext_r1_4(
    device const q5_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_1_mv_ext_r1_4_compute_f16<half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q5_k_mv_compute(
    device const q5_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 2u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < M;

    device const InputType *input_row = input + n * stride_n;
    device const q5_k_block *weight_row0 = weight + m0 * weight_blocks;
    device const q5_k_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const bool full_k = (stride_k == 1u) && ((K & (kQK_K - 1u)) == 0u);
    if (full_k) {
        constexpr ushort kmask1 = 0x3f3f;
        constexpr ushort kmask2 = 0x0f0f;
        constexpr ushort kmask3 = 0xc0c0;

        const ushort tid = tiisg >> 2;
        const ushort ix = tiisg & 3;
        const ushort iq = tid >> 2;
        const ushort ir = tid & 3;

        const ushort l0 = ir * 8;
        const ushort q_offset = 32 * iq + l0;
        const ushort y_offset = 64 * iq + l0;

        const uchar hm1 = uchar(1u << (2u * iq));
        const uchar hm2 = hm1 << 1;
        const uchar hm3 = hm1 << 4;
        const uchar hm4 = hm2 << 4;

        const uint nb = min(weight_blocks, K / kQK_K);
        device const InputType *y1 = input_row + uint(ix) * kQK_K + uint(y_offset);

        for (uint ib = uint(ix); ib < nb; ib += 4u) {
            float yl[16];
            float yh[16];
            float4 sumy = 0.0f;

            device const InputType *y2 = y1 + 128;

#pragma clang loop unroll(full)
            for (ushort l = 0; l < 8; ++l) {
                const float v0 = float(y1[l + 0]);
                const float v1 = float(y1[l + 32]);
                const float v2 = float(y2[l + 0]);
                const float v3 = float(y2[l + 32]);

                yl[l + 0] = v0;
                yl[l + 8] = v1;
                yh[l + 0] = v2;
                yh[l + 8] = v3;

                sumy[0] += v0;
                sumy[1] += v1;
                sumy[2] += v2;
                sumy[3] += v3;
            }

            {
                const device q5_k_block &blk = weight_row0[ib];

                const device uchar *q1 = blk.qs + q_offset;
                const device uchar *q2 = q1 + 64;
                const device uchar *qh = blk.qh + l0;
                const device half *dh = &blk.d;
                const device ushort *a = (device const ushort *)blk.scales + iq;

                ushort sc16[4];
                thread const uchar *sc8 = (thread const uchar *)sc16;
                sc16[0] = a[0] & kmask1;
                sc16[1] = a[2] & kmask1;
                sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
                sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

                float4 acc1 = 0.0f;
                float4 acc2 = 0.0f;
#pragma clang loop unroll(full)
                for (ushort l = 0; l < 8; ++l) {
                    const uchar h = qh[l];
                    acc1[0] += yl[l + 0] * float(q1[l] & 0x0Fu);
                    acc1[1] += yl[l + 8] * float(q1[l] & 0xF0u);
                    acc1[2] += yh[l + 0] * float(q2[l] & 0x0Fu);
                    acc1[3] += yh[l + 8] * float(q2[l] & 0xF0u);
                    acc2[0] += ((h & hm1) != 0u) ? yl[l + 0] : 0.0f;
                    acc2[1] += ((h & hm2) != 0u) ? yl[l + 8] : 0.0f;
                    acc2[2] += ((h & hm3) != 0u) ? yh[l + 0] : 0.0f;
                    acc2[3] += ((h & hm4) != 0u) ? yh[l + 8] : 0.0f;
                }

                sum0 += float(dh[0]) *
                        (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0]) +
                         float(sc8[1]) * (acc1[1] * (1.0f / 16.0f) + 16.0f * acc2[1]) +
                         float(sc8[4]) * (acc1[2] + 16.0f * acc2[2]) +
                         float(sc8[5]) * (acc1[3] * (1.0f / 16.0f) + 16.0f * acc2[3])) -
                    float(dh[1]) *
                        (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) + sumy[2] * float(sc8[6]) +
                         sumy[3] * float(sc8[7]));
            }

            {
                const device q5_k_block &blk = weight_row1[ib];

                const device uchar *q1 = blk.qs + q_offset;
                const device uchar *q2 = q1 + 64;
                const device uchar *qh = blk.qh + l0;
                const device half *dh = &blk.d;
                const device ushort *a = (device const ushort *)blk.scales + iq;

                ushort sc16[4];
                thread const uchar *sc8 = (thread const uchar *)sc16;
                sc16[0] = a[0] & kmask1;
                sc16[1] = a[2] & kmask1;
                sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
                sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

                float4 acc1 = 0.0f;
                float4 acc2 = 0.0f;
#pragma clang loop unroll(full)
                for (ushort l = 0; l < 8; ++l) {
                    const uchar h = qh[l];
                    acc1[0] += yl[l + 0] * float(q1[l] & 0x0Fu);
                    acc1[1] += yl[l + 8] * float(q1[l] & 0xF0u);
                    acc1[2] += yh[l + 0] * float(q2[l] & 0x0Fu);
                    acc1[3] += yh[l + 8] * float(q2[l] & 0xF0u);
                    acc2[0] += ((h & hm1) != 0u) ? yl[l + 0] : 0.0f;
                    acc2[1] += ((h & hm2) != 0u) ? yl[l + 8] : 0.0f;
                    acc2[2] += ((h & hm3) != 0u) ? yh[l + 0] : 0.0f;
                    acc2[3] += ((h & hm4) != 0u) ? yh[l + 8] : 0.0f;
                }

                sum1 += float(dh[0]) *
                        (float(sc8[0]) * (acc1[0] + 16.0f * acc2[0]) +
                         float(sc8[1]) * (acc1[1] * (1.0f / 16.0f) + 16.0f * acc2[1]) +
                         float(sc8[4]) * (acc1[2] + 16.0f * acc2[2]) +
                         float(sc8[5]) * (acc1[3] * (1.0f / 16.0f) + 16.0f * acc2[3])) -
                    float(dh[1]) *
                        (sumy[0] * float(sc8[2]) + sumy[1] * float(sc8[3]) + sumy[2] * float(sc8[6]) +
                         sumy[3] * float(sc8[7]));
            }

            y1 += 4u * kQK_K;
        }
    } else {
        const uint lane = uint(tiisg);
        const uint q_offset = lane << 3;

        const uint sub = q_offset >> 5;
        const uint group = sub >> 1;
        const bool high_nibble = (sub & 1u) != 0u;
        const uint l_base = q_offset & 31u;
        const uchar high_mask = uchar(1u << (2u * group + (high_nibble ? 1u : 0u)));

        for (uint sb = 0; sb < weight_blocks; ++sb) {
            const uint block_start = sb * kQK_K;
            if (block_start >= K) {
                break;
            }

            float4 a0 = 0.0f;
            float4 a1 = 0.0f;
            float sum_a = 0.0f;
            marmot_mv_load8(input_row, block_start + q_offset, K, stride_k, a0, a1, sum_a);

            {
                const device q5_k_block &blk = weight_row0[sb];

                uchar sc = 0u;
                uchar m = 0u;
                get_scale_min_k4(sub, blk.scales, sc, m);

                const float scale = float(blk.d) * float(sc);
                const float minv = float(blk.dmin) * float(m);

                const device uchar *ql = blk.qs + group * 32u + l_base;
                const device uchar *qh = blk.qh + l_base;

                float4 qv0;
                float4 qv1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uchar qlb = ql[i];
                    const uint q4 = high_nibble ? uint(qlb >> 4) : uint(qlb & 0x0Fu);
                    const uint q5 = q4 + (((qh[i] & high_mask) != 0u) ? 16u : 0u);
                    const float qf = float(q5);
                    if (i < 4u) {
                        qv0[i] = qf;
                    } else {
                        qv1[i - 4u] = qf;
                    }
                }

                const float sum_qa = dot(qv0, a0) + dot(qv1, a1);
                sum0 += scale * sum_qa - minv * sum_a;
            }

            {
                const device q5_k_block &blk = weight_row1[sb];

                uchar sc = 0u;
                uchar m = 0u;
                get_scale_min_k4(sub, blk.scales, sc, m);

                const float scale = float(blk.d) * float(sc);
                const float minv = float(blk.dmin) * float(m);

                const device uchar *ql = blk.qs + group * 32u + l_base;
                const device uchar *qh = blk.qh + l_base;

                float4 qv0;
                float4 qv1;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uchar qlb = ql[i];
                    const uint q4 = high_nibble ? uint(qlb >> 4) : uint(qlb & 0x0Fu);
                    const uint q5 = q4 + (((qh[i] & high_mask) != 0u) ? 16u : 0u);
                    const float qf = float(q5);
                    if (i < 4u) {
                        qv0[i] = qf;
                    } else {
                        qv1[i - 4u] = qf;
                    }
                }

                const float sum_qa = dot(qv0, a0) + dot(qv1, a1);
                sum1 += scale * sum_qa - minv * sum_a;
            }
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
    }
}

kernel void matmul_q5_k_f32_f32_mv(
    device const q5_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_k_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_k_f16_f32_mv(
    device const q5_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_k_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_k_f16_f16_mv(
    device const q5_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_k_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q6_k_mv_compute(
    device const q6_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 2u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < M;

    device const InputType *input_row = input + n * stride_n;
    device const q6_k_block *weight_row0 = weight + m0 * weight_blocks;
    device const q6_k_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const bool full_k = (stride_k == 1u) && ((K & (kQK_K - 1u)) == 0u);
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

        const uint nb = min(weight_blocks, K / kQK_K);
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

            {
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
            if (block_start >= K) {
                break;
            }

            float4 a0 = 0.0f;
            float4 a1 = 0.0f;
            float sum_a = 0.0f;
            marmot_mv_load8(input_row, block_start + q_offset, K, stride_k, a0, a1, sum_a);
            (void)sum_a;

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

                const float sum_qa = dot(qv0, a0) + dot(qv1, a1);
                sum0 += dl * sum_qa;
            }

            {
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

                const float sum_qa = dot(qv0, a0) + dot(qv1, a1);
                sum1 += dl * sum_qa;
            }
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
    }
}

template <typename InputType, typename OutputType>
static inline void matmul_q6_k_dual_mv_compute(
    device const q6_k_block *weight_q, device const q6_k_block *weight_k, device const InputType *input,
    device OutputType *output_q, device OutputType *output_k, uint N, uint K, uint M, uint stride_n, uint stride_k,
    uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 2u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < M;

    device const InputType *input_row = input + n * stride_n;
    device const q6_k_block *weight_row_q0 = weight_q + m0 * weight_blocks;
    device const q6_k_block *weight_row_q1 = has_m1 ? (weight_row_q0 + weight_blocks) : weight_row_q0;
    device const q6_k_block *weight_row_k0 = weight_k + m0 * weight_blocks;
    device const q6_k_block *weight_row_k1 = has_m1 ? (weight_row_k0 + weight_blocks) : weight_row_k0;

    float sum_q0 = 0.0f;
    float sum_q1 = 0.0f;
    float sum_k0 = 0.0f;
    float sum_k1 = 0.0f;

    const bool full_k = (stride_k == 1u) && ((K & (kQK_K - 1u)) == 0u);
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

        const uint nb = min(weight_blocks, K / kQK_K);
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
                const device q6_k_block &blk = weight_row_q0[ib];
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
                sum_q0 += float(blk.d) *
                    (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
            }
            {
                const device q6_k_block &blk = weight_row_q1[ib];
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
                sum_q1 += float(blk.d) *
                    (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
            }
            {
                const device q6_k_block &blk = weight_row_k0[ib];
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
                sum_k0 += float(blk.d) *
                    (sums[0] * float(sc[0]) + sums[1] * float(sc[2]) + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
            }
            {
                const device q6_k_block &blk = weight_row_k1[ib];
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
                sum_k1 += float(blk.d) *
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
            if (block_start >= K) {
                break;
            }

            float4 a0 = 0.0f;
            float4 a1 = 0.0f;
            float sum_a = 0.0f;
            marmot_mv_load8(input_row, block_start + q_offset, K, stride_k, a0, a1, sum_a);
            (void)sum_a;

            {
                const device q6_k_block &blk = weight_row_q0[sb];
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
                sum_q0 += dl * (dot(qv0, a0) + dot(qv1, a1));
            }
            {
                const device q6_k_block &blk = weight_row_q1[sb];
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
                sum_q1 += dl * (dot(qv0, a0) + dot(qv1, a1));
            }
            {
                const device q6_k_block &blk = weight_row_k0[sb];
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
                sum_k0 += dl * (dot(qv0, a0) + dot(qv1, a1));
            }
            {
                const device q6_k_block &blk = weight_row_k1[sb];
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
                sum_k1 += dl * (dot(qv0, a0) + dot(qv1, a1));
            }
        }
    }

    const float out_q0 = simd_sum(sum_q0);
    const float out_q1 = simd_sum(sum_q1);
    const float out_k0 = simd_sum(sum_k0);
    const float out_k1 = simd_sum(sum_k1);
    if (tiisg == 0) {
        output_q[n * M + m0] = OutputType(out_q0);
        output_k[n * M + m0] = OutputType(out_k0);
        if (has_m1) {
            output_q[n * M + m1] = OutputType(out_q1);
            output_k[n * M + m1] = OutputType(out_k1);
        }
    }
}

kernel void matmul_q6_k_f32_f32_mv(
    device const q6_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q6_k_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q6_k_f16_f32_mv(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q6_k_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q6_k_f16_f16_mv(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q6_k_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_qkv_q6_k_dual_f32_f32_mv(
    device const q6_k_block *weight_q [[buffer(0)]], device const q6_k_block *weight_k [[buffer(1)]],
    device const q6_k_block *weight_v [[buffer(2)]], device const float *input [[buffer(3)]],
    device float *out_q [[buffer(4)]], device float *out_k [[buffer(5)]], device float *out_v [[buffer(6)]],
    constant uint &N [[buffer(7)]], constant uint &K [[buffer(8)]], constant uint &M [[buffer(9)]],
    constant uint &stride_n [[buffer(10)]], constant uint &stride_k [[buffer(11)]],
    constant uint &weight_blocks [[buffer(12)]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]], uint2 tgp [[threadgroup_position_in_grid]]
) {
    (void)weight_v;
    (void)out_v;
    matmul_q6_k_dual_mv_compute<float, float>(
        weight_q, weight_k, input, out_q, out_k, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_qkv_q6_k_dual_f16_f32_mv(
    device const q6_k_block *weight_q [[buffer(0)]], device const q6_k_block *weight_k [[buffer(1)]],
    device const q6_k_block *weight_v [[buffer(2)]], device const half *input [[buffer(3)]],
    device float *out_q [[buffer(4)]], device float *out_k [[buffer(5)]], device float *out_v [[buffer(6)]],
    constant uint &N [[buffer(7)]], constant uint &K [[buffer(8)]], constant uint &M [[buffer(9)]],
    constant uint &stride_n [[buffer(10)]], constant uint &stride_k [[buffer(11)]],
    constant uint &weight_blocks [[buffer(12)]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]], uint2 tgp [[threadgroup_position_in_grid]]
) {
    (void)weight_v;
    (void)out_v;
    matmul_q6_k_dual_mv_compute<half, float>(
        weight_q, weight_k, input, out_q, out_k, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_qkv_q6_k_dual_f16_f16_mv(
    device const q6_k_block *weight_q [[buffer(0)]], device const q6_k_block *weight_k [[buffer(1)]],
    device const q6_k_block *weight_v [[buffer(2)]], device const half *input [[buffer(3)]],
    device half *out_q [[buffer(4)]], device half *out_k [[buffer(5)]], device half *out_v [[buffer(6)]],
    constant uint &N [[buffer(7)]], constant uint &K [[buffer(8)]], constant uint &M [[buffer(9)]],
    constant uint &stride_n [[buffer(10)]], constant uint &stride_k [[buffer(11)]],
    constant uint &weight_blocks [[buffer(12)]], ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]], uint2 tgp [[threadgroup_position_in_grid]]
) {
    (void)weight_v;
    (void)out_v;
    matmul_q6_k_dual_mv_compute<half, half>(
        weight_q, weight_k, input, out_q, out_k, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q8_k_mv_compute(
    device const q8_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 2u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < M;

    const uint lane = uint(tiisg);
    const uint q_offset = lane << 3;

    device const InputType *input_row = input + n * stride_n;
    device const q8_k_block *weight_row0 = weight + m0 * weight_blocks;
    device const q8_k_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (uint sb = 0; sb < weight_blocks; ++sb) {
        const uint block_start = sb * kQK_K;
        if (block_start >= K) {
            break;
        }

        float4 a0 = 0.0f;
        float4 a1 = 0.0f;
        float sum_a = 0.0f;
        marmot_mv_load8(input_row, block_start + q_offset, K, stride_k, a0, a1, sum_a);
        (void)sum_a;

        {
            const device q8_k_block &blk = weight_row0[sb];
            const float scale = blk.d;

            const device char *q = blk.qs + q_offset;
            const char4 qv0 = *((device const char4 *)(q + 0));
            const char4 qv1 = *((device const char4 *)(q + 4));
            const float sum_qa = dot(float4(qv0), a0) + dot(float4(qv1), a1);
            sum0 += scale * sum_qa;
        }

        {
            const device q8_k_block &blk = weight_row1[sb];
            const float scale = blk.d;

            const device char *q = blk.qs + q_offset;
            const char4 qv0 = *((device const char4 *)(q + 0));
            const char4 qv1 = *((device const char4 *)(q + 4));
            const float sum_qa = dot(float4(qv0), a0) + dot(float4(qv1), a1);
            sum1 += scale * sum_qa;
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
    }
}

kernel void matmul_q8_k_f32_f32_mv(
    device const q8_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q8_k_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q8_k_f16_f32_mv(
    device const q8_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q8_k_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q8_k_f16_f16_mv(
    device const q8_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q8_k_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

// -----------------------------------------------------------------------------
// Basic quant small-K kernels (direct activations)
// -----------------------------------------------------------------------------

template <typename InputType, typename OutputType>
static inline void matmul_q4_0_small_compute(
    device const q4_0_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 gid
) {
    const uint m = gid.x;
    const uint n = gid.y;
    if (n >= N || m >= M) {
        return;
    }

    device const InputType *input_row = input + n * stride_n;
    device const q4_0_block *weight_row = weight + m * weight_blocks;

    float sum = 0.0f;

    for (uint b = 0; b < weight_blocks; ++b) {
        const device q4_0_block &w_block = weight_row[b];
        const float scale = float(w_block.scale);
        const uint block_start = b * kQuantBlockSize;
        if (block_start >= K) {
            break;
        }
        const uint block_end = min(block_start + kQuantBlockSize, K);

        for (uint i = 0; i < kQ4PackedBytes; ++i) {
            const uint idx0 = block_start + i;
            const uint idx1 = block_start + i + (kQuantBlockSize / 2u);
            if (idx0 >= block_end) {
                break;
            }

            const uchar packed = w_block.qs[i];
            const int w0 = int(packed & 0x0Fu) - 8;
            sum += float(w0) * scale * float(input_row[idx0 * stride_k]);

            if (idx1 < block_end) {
                const int w1 = int(packed >> 4) - 8;
                sum += float(w1) * scale * float(input_row[idx1 * stride_k]);
            }
        }
    }

    output[n * M + m] = OutputType(sum);
}

kernel void matmul_q4_0_f32_f32_small(
    device const q4_0_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q4_0_small_compute<float, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q4_0_f16_f32_small(
    device const q4_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q4_0_small_compute<half, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q4_0_f16_f16_small(
    device const q4_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q4_0_small_compute<half, half>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

template <typename InputType, typename OutputType>
static inline void matmul_q4_1_small_compute(
    device const q4_1_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 gid
) {
    const uint m = gid.x;
    const uint n = gid.y;
    if (n >= N || m >= M) {
        return;
    }

    device const InputType *input_row = input + n * stride_n;
    device const q4_1_block *weight_row = weight + m * weight_blocks;

    float sum = 0.0f;

    for (uint b = 0; b < weight_blocks; ++b) {
        const device q4_1_block &w_block = weight_row[b];
        const float scale_w = float(w_block.scale);
        const float min_w = float(w_block.min);
        const uint block_start = b * kQuantBlockSize;
        if (block_start >= K) {
            break;
        }
        const uint block_end = min(block_start + kQuantBlockSize, K);

        for (uint i = 0; i < kQ4PackedBytes; ++i) {
            const uint idx0 = block_start + i;
            const uint idx1 = block_start + i + (kQuantBlockSize / 2u);
            if (idx0 >= block_end) {
                break;
            }

            const uchar packed = w_block.qs[i];
            const uint q0 = uint(packed & 0x0Fu);
            sum += (scale_w * float(q0) + min_w) * float(input_row[idx0 * stride_k]);

            if (idx1 < block_end) {
                const uint q1 = uint(packed >> 4);
                sum += (scale_w * float(q1) + min_w) * float(input_row[idx1 * stride_k]);
            }
        }
    }

    output[n * M + m] = OutputType(sum);
}

kernel void matmul_q4_1_f32_f32_small(
    device const q4_1_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q4_1_small_compute<float, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q4_1_f16_f32_small(
    device const q4_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q4_1_small_compute<half, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q4_1_f16_f16_small(
    device const q4_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q4_1_small_compute<half, half>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

template <typename InputType, typename OutputType>
static inline void matmul_q5_0_small_compute(
    device const q5_0_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 gid
) {
    const uint m = gid.x;
    const uint n = gid.y;
    if (n >= N || m >= M) {
        return;
    }

    device const InputType *input_row = input + n * stride_n;
    device const q5_0_block *weight_row = weight + m * weight_blocks;

    float sum = 0.0f;

    for (uint b = 0; b < weight_blocks; ++b) {
        const device q5_0_block &w_block = weight_row[b];
        const float scale_w = float(w_block.scale);
        const uint qh = marmot_unpack_q5_high_bits_32(w_block.qh);
        const uint block_start = b * kQuantBlockSize;
        if (block_start >= K) {
            break;
        }
        const uint block_end = min(block_start + kQuantBlockSize, K);

        for (uint i = 0; i < kQ5PackedBytes; ++i) {
            const uint idx0 = block_start + i;
            const uint idx1 = block_start + i + (kQuantBlockSize / 2u);
            if (idx0 >= block_end) {
                break;
            }

            const uchar packed = w_block.qs[i];
            const uint lo = uint(packed & 0x0Fu) | (((qh >> i) & 0x1u) << 4u);
            sum += float(int(lo) - 16) * scale_w * float(input_row[idx0 * stride_k]);

            if (idx1 < block_end) {
                const uint hi_idx = i + (kQuantBlockSize / 2u);
                const uint hi = uint(packed >> 4) | (((qh >> hi_idx) & 0x1u) << 4u);
                sum += float(int(hi) - 16) * scale_w * float(input_row[idx1 * stride_k]);
            }
        }
    }

    output[n * M + m] = OutputType(sum);
}

kernel void matmul_q5_0_f32_f32_small(
    device const q5_0_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q5_0_small_compute<float, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q5_0_f16_f32_small(
    device const q5_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q5_0_small_compute<half, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q5_0_f16_f16_small(
    device const q5_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q5_0_small_compute<half, half>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

template <typename InputType, typename OutputType>
static inline void matmul_q5_1_small_compute(
    device const q5_1_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 gid
) {
    const uint m = gid.x;
    const uint n = gid.y;
    if (n >= N || m >= M) {
        return;
    }

    device const InputType *input_row = input + n * stride_n;
    device const q5_1_block *weight_row = weight + m * weight_blocks;

    float sum = 0.0f;

    for (uint b = 0; b < weight_blocks; ++b) {
        const device q5_1_block &w_block = weight_row[b];
        const float scale_w = float(w_block.scale);
        const float min_w = float(w_block.min);
        const uint qh = marmot_unpack_q5_high_bits_32(w_block.qh);
        const uint block_start = b * kQuantBlockSize;
        if (block_start >= K) {
            break;
        }
        const uint block_end = min(block_start + kQuantBlockSize, K);

        for (uint i = 0; i < kQ5PackedBytes; ++i) {
            const uint idx0 = block_start + i;
            const uint idx1 = block_start + i + (kQuantBlockSize / 2u);
            if (idx0 >= block_end) {
                break;
            }

            const uchar packed = w_block.qs[i];
            const uint lo = uint(packed & 0x0Fu) | (((qh >> i) & 0x1u) << 4u);
            sum += (scale_w * float(lo) + min_w) * float(input_row[idx0 * stride_k]);

            if (idx1 < block_end) {
                const uint hi_idx = i + (kQuantBlockSize / 2u);
                const uint hi = uint(packed >> 4) | (((qh >> hi_idx) & 0x1u) << 4u);
                sum += (scale_w * float(hi) + min_w) * float(input_row[idx1 * stride_k]);
            }
        }
    }

    output[n * M + m] = OutputType(sum);
}

kernel void matmul_q5_1_f32_f32_small(
    device const q5_1_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q5_1_small_compute<float, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q5_1_f16_f32_small(
    device const q5_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q5_1_small_compute<half, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q5_1_f16_f16_small(
    device const q5_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q5_1_small_compute<half, half>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

template <typename InputType, typename OutputType>
static inline void matmul_q8_0_small_compute(
    device const q8_0_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 gid
) {
    const uint m = gid.x;
    const uint n = gid.y;
    if (n >= N || m >= M) {
        return;
    }

    device const InputType *input_row = input + n * stride_n;
    device const q8_0_block *weight_row = weight + m * weight_blocks;

    float sum = 0.0f;

    for (uint b = 0; b < weight_blocks; ++b) {
        const device q8_0_block &w_block = weight_row[b];
        const float scale = float(w_block.scale);
        const uint block_start = b * kQuantBlockSize;
        if (block_start >= K) {
            break;
        }
        const uint block_end = min(block_start + kQuantBlockSize, K);
        for (uint i = 0; i < kQuantBlockSize; ++i) {
            const uint idx = block_start + i;
            if (idx >= block_end) {
                break;
            }
            sum += float(int(w_block.qs[i])) * scale * float(input_row[idx * stride_k]);
        }
    }

    output[n * M + m] = OutputType(sum);
}

kernel void matmul_q8_0_f32_f32_small(
    device const q8_0_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q8_0_small_compute<float, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q8_0_f16_f32_small(
    device const q8_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q8_0_small_compute<half, float>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

kernel void matmul_q8_0_f16_f16_small(
    device const q8_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 gid [[thread_position_in_grid]]
) {
    matmul_q8_0_small_compute<half, half>(weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, gid);
}

// -----------------------------------------------------------------------------
// Basic quant MV kernels (direct activations)
// -----------------------------------------------------------------------------

template <typename InputType>
static inline float marmot_mv_load_yl16(device const InputType *yb, thread float yl[16]) {
    constexpr float inv256 = 1.0f / 256.0f;
    constexpr float inv16 = 1.0f / 16.0f;
    constexpr float inv4096 = 1.0f / 4096.0f;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; i += 2u) {
        const float v0 = float(yb[i + 0u]);
        const float v1 = float(yb[i + 1u]);
        const float v2 = float(yb[i + 16u]);
        const float v3 = float(yb[i + 17u]);

        sum0 += v0 + v1;
        sum1 += v2 + v3;

        yl[i + 0u] = v0;
        yl[i + 1u] = v1 * inv256;
        yl[i + 8u] = v2 * inv16;
        yl[i + 9u] = v3 * inv4096;
    }
    return sum0 + sum1;
}

static inline float marmot_mv_block_dot_y(const device q4_0_block *blk, float sumy, thread const float *yl, uint il) {
    const float d = float(blk->scale);

    const device ushort *qs = ((device const ushort *)blk) + 1 + il / 2;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; i += 2u) {
        const ushort q = qs[i / 2u];
        acc0 += yl[i + 0u] * float(q & 0x000Fu);
        acc1 += yl[i + 1u] * float(q & 0x0F00u);
        acc2 += yl[i + 8u] * float(q & 0x00F0u);
        acc3 += yl[i + 9u] * float(q & 0xF000u);
    }

    return d * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
}

static inline float marmot_mv_block_dot_y(const device q4_1_block *blk, float sumy, thread const float *yl, uint il) {
    const float d = float(blk->scale);
    const float m = float(blk->min);

    const device ushort *qs = ((device const ushort *)blk) + 2 + il / 2;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; i += 2u) {
        const ushort q = qs[i / 2u];
        acc0 += yl[i + 0u] * float(q & 0x000Fu);
        acc1 += yl[i + 1u] * float(q & 0x0F00u);
        acc2 += yl[i + 8u] * float(q & 0x00F0u);
        acc3 += yl[i + 9u] * float(q & 0xF000u);
    }

    return d * (acc0 + acc1 + acc2 + acc3) + sumy * m;
}

static inline float marmot_mv_block_dot_y(const device q5_0_block *blk, float sumy, thread const float *yl, uint il) {
    const float d = float(blk->scale);
    const uint qh = marmot_unpack_q5_high_bits_32(blk->qh);

    const device ushort *qs = ((device const ushort *)blk) + 3 + il / 2;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; i += 2u) {
        const ushort q = qs[i / 2u];
        acc0 += yl[i + 0u] * float((q & 0x000Fu) | (((qh >> (i + 0u + il)) << 4u) & 0x00010u));
        acc1 += yl[i + 1u] * float((q & 0x0F00u) | (((qh >> (i + 1u + il)) << 12u) & 0x01000u));
        acc2 += yl[i + 8u] * float((q & 0x00F0u) | (((qh >> (i + 0u + il + kQuantBlockSize / 2u)) << 8u) & 0x00100u));
        acc3 += yl[i + 9u] * float((q & 0xF000u) | (((qh >> (i + 1u + il + kQuantBlockSize / 2u)) << 16u) & 0x10000u));
    }

    return d * (sumy * -16.0f + acc0 + acc1 + acc2 + acc3);
}

static inline float marmot_mv_block_dot_y(const device q5_1_block *blk, float sumy, thread const float *yl, uint il) {
    const float d = float(blk->scale);
    const float m = float(blk->min);
    const uint qh = marmot_unpack_q5_high_bits_32(blk->qh);

    const device ushort *qs = ((device const ushort *)blk) + 4 + il / 2;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; i += 2u) {
        const ushort q = qs[i / 2u];
        acc0 += yl[i + 0u] * float((q & 0x000Fu) | (((qh >> (i + 0u + il)) << 4u) & 0x00010u));
        acc1 += yl[i + 1u] * float((q & 0x0F00u) | (((qh >> (i + 1u + il)) << 12u) & 0x01000u));
        acc2 += yl[i + 8u] * float((q & 0x00F0u) | (((qh >> (i + 0u + il + kQuantBlockSize / 2u)) << 8u) & 0x00100u));
        acc3 += yl[i + 9u] * float((q & 0xF000u) | (((qh >> (i + 1u + il + kQuantBlockSize / 2u)) << 16u) & 0x10000u));
    }

    return d * (acc0 + acc1 + acc2 + acc3) + sumy * m;
}

template <typename InputType, typename OutputType>
static inline void matmul_q4_0_mv_compute(
    device const q4_0_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 4u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const uint m2 = m_base + 2u;
    const uint m3 = m_base + 3u;
    const bool has_m1 = m1 < M;
    const bool has_m2 = m2 < M;
    const bool has_m3 = m3 < M;

    constexpr uint NQ = 16u;
    const uint lane = uint(tiisg);
    const uint ix = lane >> 1;
    const uint il = (lane & 1u) * 8u;

    device const InputType *input_row = input + n * stride_n;
    device const q4_0_block *weight_row0 = weight + m0 * weight_blocks;
    device const q4_0_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;
    device const q4_0_block *weight_row2 = has_m2 ? (weight_row0 + 2u * weight_blocks) : weight_row0;
    device const q4_0_block *weight_row3 = has_m3 ? (weight_row0 + 3u * weight_blocks) : weight_row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const bool blocks_exact = (weight_blocks * kQuantBlockSize) == K;
    const bool contig = stride_k == 1u;

    if (blocks_exact && contig) {
        device const InputType *yb = input_row + ix * kQuantBlockSize + il;
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            float yl[16];
            const float sumy = marmot_mv_load_yl16(yb, yl);
            sum0 += marmot_mv_block_dot_y(&weight_row0[ib], sumy, yl, il);
            sum1 += marmot_mv_block_dot_y(&weight_row1[ib], sumy, yl, il);
            sum2 += marmot_mv_block_dot_y(&weight_row2[ib], sumy, yl, il);
            sum3 += marmot_mv_block_dot_y(&weight_row3[ib], sumy, yl, il);
            yb += kQuantBlockSize * NQ;
        }
    } else {
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            const uint block_start = ib * kQuantBlockSize;
            if (block_start >= K) {
                break;
            }
            const uint block_end = min(block_start + kQuantBlockSize, K);

            const device q4_0_block &w0 = weight_row0[ib];
            const device q4_0_block &w1 = weight_row1[ib];
            const device q4_0_block &w2 = weight_row2[ib];
            const device q4_0_block &w3 = weight_row3[ib];
            const float scale0 = float(w0.scale);
            const float scale1 = float(w1.scale);
            const float scale2 = float(w2.scale);
            const float scale3 = float(w3.scale);

            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                const uint k0 = block_start + il + i;
                if (k0 >= block_end) {
                    break;
                }
                const float a0 = float(input_row[k0 * stride_k]);
                const uint k1 = k0 + 16u;
                const float a1 = (k1 < block_end) ? float(input_row[k1 * stride_k]) : 0.0f;

                const uchar packed0 = w0.qs[il + i];
                acc0 += float(int(packed0 & 0x0Fu) - 8) * a0 + float(int(packed0 >> 4) - 8) * a1;

                const uchar packed1 = w1.qs[il + i];
                acc1 += float(int(packed1 & 0x0Fu) - 8) * a0 + float(int(packed1 >> 4) - 8) * a1;

                const uchar packed2 = w2.qs[il + i];
                acc2 += float(int(packed2 & 0x0Fu) - 8) * a0 + float(int(packed2 >> 4) - 8) * a1;

                const uchar packed3 = w3.qs[il + i];
                acc3 += float(int(packed3 & 0x0Fu) - 8) * a0 + float(int(packed3 >> 4) - 8) * a1;
            }

            sum0 += acc0 * scale0;
            sum1 += acc1 * scale1;
            sum2 += acc2 * scale2;
            sum3 += acc3 * scale3;
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    const float out2 = simd_sum(sum2);
    const float out3 = simd_sum(sum3);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
        if (has_m2) {
            output[n * M + m2] = OutputType(out2);
        }
        if (has_m3) {
            output[n * M + m3] = OutputType(out3);
        }
    }
}

kernel void matmul_q4_0_f32_f32_mv(
    device const q4_0_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_0_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q4_0_f16_f32_mv(
    device const q4_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_0_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q4_0_f16_f16_mv(
    device const q4_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_0_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q4_1_mv_compute(
    device const q4_1_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 4u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const uint m2 = m_base + 2u;
    const uint m3 = m_base + 3u;
    const bool has_m1 = m1 < M;
    const bool has_m2 = m2 < M;
    const bool has_m3 = m3 < M;

    constexpr uint NQ = 16u;
    const uint lane = uint(tiisg);
    const uint ix = lane >> 1;
    const uint il = (lane & 1u) * 8u;

    device const InputType *input_row = input + n * stride_n;
    device const q4_1_block *weight_row0 = weight + m0 * weight_blocks;
    device const q4_1_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;
    device const q4_1_block *weight_row2 = has_m2 ? (weight_row0 + 2u * weight_blocks) : weight_row0;
    device const q4_1_block *weight_row3 = has_m3 ? (weight_row0 + 3u * weight_blocks) : weight_row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const bool blocks_exact = (weight_blocks * kQuantBlockSize) == K;
    const bool contig = stride_k == 1u;

    if (blocks_exact && contig) {
        device const InputType *yb = input_row + ix * kQuantBlockSize + il;
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            float yl[16];
            const float sumy = marmot_mv_load_yl16(yb, yl);
            sum0 += marmot_mv_block_dot_y(&weight_row0[ib], sumy, yl, il);
            sum1 += marmot_mv_block_dot_y(&weight_row1[ib], sumy, yl, il);
            sum2 += marmot_mv_block_dot_y(&weight_row2[ib], sumy, yl, il);
            sum3 += marmot_mv_block_dot_y(&weight_row3[ib], sumy, yl, il);
            yb += kQuantBlockSize * NQ;
        }
    } else {
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            const uint block_start = ib * kQuantBlockSize;
            if (block_start >= K) {
                break;
            }
            const uint block_end = min(block_start + kQuantBlockSize, K);

            const device q4_1_block &w0 = weight_row0[ib];
            const device q4_1_block &w1 = weight_row1[ib];
            const device q4_1_block &w2 = weight_row2[ib];
            const device q4_1_block &w3 = weight_row3[ib];
            const float scale0 = float(w0.scale);
            const float scale1 = float(w1.scale);
            const float scale2 = float(w2.scale);
            const float scale3 = float(w3.scale);
            const float min0 = float(w0.min);
            const float min1 = float(w1.min);
            const float min2 = float(w2.min);
            const float min3 = float(w3.min);

            float sumy = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                const uint k0 = block_start + il + i;
                if (k0 >= block_end) {
                    break;
                }
                const float a0 = float(input_row[k0 * stride_k]);
                const uint k1 = k0 + 16u;
                const float a1 = (k1 < block_end) ? float(input_row[k1 * stride_k]) : 0.0f;
                sumy += a0 + a1;

                const uchar packed0 = w0.qs[il + i];
                acc0 += float(packed0 & 0x0Fu) * a0 + float(packed0 >> 4) * a1;

                const uchar packed1 = w1.qs[il + i];
                acc1 += float(packed1 & 0x0Fu) * a0 + float(packed1 >> 4) * a1;

                const uchar packed2 = w2.qs[il + i];
                acc2 += float(packed2 & 0x0Fu) * a0 + float(packed2 >> 4) * a1;

                const uchar packed3 = w3.qs[il + i];
                acc3 += float(packed3 & 0x0Fu) * a0 + float(packed3 >> 4) * a1;
            }

            sum0 += acc0 * scale0 + sumy * min0;
            sum1 += acc1 * scale1 + sumy * min1;
            sum2 += acc2 * scale2 + sumy * min2;
            sum3 += acc3 * scale3 + sumy * min3;
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    const float out2 = simd_sum(sum2);
    const float out3 = simd_sum(sum3);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
        if (has_m2) {
            output[n * M + m2] = OutputType(out2);
        }
        if (has_m3) {
            output[n * M + m3] = OutputType(out3);
        }
    }
}

kernel void matmul_q4_1_f32_f32_mv(
    device const q4_1_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_1_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q4_1_f16_f32_mv(
    device const q4_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_1_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q4_1_f16_f16_mv(
    device const q4_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q4_1_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q5_0_mv_compute(
    device const q5_0_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 4u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const uint m2 = m_base + 2u;
    const uint m3 = m_base + 3u;
    const bool has_m1 = m1 < M;
    const bool has_m2 = m2 < M;
    const bool has_m3 = m3 < M;

    constexpr uint NQ = 16u;
    const uint lane = uint(tiisg);
    const uint ix = lane >> 1;
    const uint il = (lane & 1u) * 8u;

    device const InputType *input_row = input + n * stride_n;
    device const q5_0_block *weight_row0 = weight + m0 * weight_blocks;
    device const q5_0_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;
    device const q5_0_block *weight_row2 = has_m2 ? (weight_row0 + 2u * weight_blocks) : weight_row0;
    device const q5_0_block *weight_row3 = has_m3 ? (weight_row0 + 3u * weight_blocks) : weight_row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const bool blocks_exact = (weight_blocks * kQuantBlockSize) == K;
    const bool contig = stride_k == 1u;

    if (blocks_exact && contig) {
        device const InputType *yb = input_row + ix * kQuantBlockSize + il;
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            float yl[16];
            const float sumy = marmot_mv_load_yl16(yb, yl);
            sum0 += marmot_mv_block_dot_y(&weight_row0[ib], sumy, yl, il);
            sum1 += marmot_mv_block_dot_y(&weight_row1[ib], sumy, yl, il);
            sum2 += marmot_mv_block_dot_y(&weight_row2[ib], sumy, yl, il);
            sum3 += marmot_mv_block_dot_y(&weight_row3[ib], sumy, yl, il);
            yb += kQuantBlockSize * NQ;
        }
    } else {
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            const uint block_start = ib * kQuantBlockSize;
            if (block_start >= K) {
                break;
            }
            const uint block_end = min(block_start + kQuantBlockSize, K);

            const device q5_0_block &w0 = weight_row0[ib];
            const device q5_0_block &w1 = weight_row1[ib];
            const device q5_0_block &w2 = weight_row2[ib];
            const device q5_0_block &w3 = weight_row3[ib];
            const float scale0 = float(w0.scale);
            const float scale1 = float(w1.scale);
            const float scale2 = float(w2.scale);
            const float scale3 = float(w3.scale);
            const uint qh0 = marmot_unpack_q5_high_bits_32(w0.qh);
            const uint qh1 = marmot_unpack_q5_high_bits_32(w1.qh);
            const uint qh2 = marmot_unpack_q5_high_bits_32(w2.qh);
            const uint qh3 = marmot_unpack_q5_high_bits_32(w3.qh);

            float sumy = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                const uint k0 = block_start + il + i;
                if (k0 >= block_end) {
                    break;
                }
                const float a0 = float(input_row[k0 * stride_k]);
                const uint k1 = k0 + 16u;
                const float a1 = (k1 < block_end) ? float(input_row[k1 * stride_k]) : 0.0f;
                sumy += a0 + a1;

                const uint idx0 = il + i;
                const uint idx1 = idx0 + 16u;

                const uchar packed0 = w0.qs[idx0];
                const uint q40 = uint(packed0 & 0x0Fu);
                const uint q41 = uint(packed0 >> 4);
                const uint q50 = q40 + (((qh0 >> idx0) & 0x1u) << 4u);
                const uint q51 = q41 + (((qh0 >> idx1) & 0x1u) << 4u);
                acc0 += float(q50) * a0 + float(q51) * a1;

                const uchar packed1 = w1.qs[idx0];
                const uint q42 = uint(packed1 & 0x0Fu);
                const uint q43 = uint(packed1 >> 4);
                const uint q52 = q42 + (((qh1 >> idx0) & 0x1u) << 4u);
                const uint q53 = q43 + (((qh1 >> idx1) & 0x1u) << 4u);
                acc1 += float(q52) * a0 + float(q53) * a1;

                const uchar packed2 = w2.qs[idx0];
                const uint q44 = uint(packed2 & 0x0Fu);
                const uint q45 = uint(packed2 >> 4);
                const uint q54 = q44 + (((qh2 >> idx0) & 0x1u) << 4u);
                const uint q55 = q45 + (((qh2 >> idx1) & 0x1u) << 4u);
                acc2 += float(q54) * a0 + float(q55) * a1;

                const uchar packed3 = w3.qs[idx0];
                const uint q46 = uint(packed3 & 0x0Fu);
                const uint q47 = uint(packed3 >> 4);
                const uint q56 = q46 + (((qh3 >> idx0) & 0x1u) << 4u);
                const uint q57 = q47 + (((qh3 >> idx1) & 0x1u) << 4u);
                acc3 += float(q56) * a0 + float(q57) * a1;
            }

            sum0 += scale0 * (acc0 - 16.0f * sumy);
            sum1 += scale1 * (acc1 - 16.0f * sumy);
            sum2 += scale2 * (acc2 - 16.0f * sumy);
            sum3 += scale3 * (acc3 - 16.0f * sumy);
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    const float out2 = simd_sum(sum2);
    const float out3 = simd_sum(sum3);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
        if (has_m2) {
            output[n * M + m2] = OutputType(out2);
        }
        if (has_m3) {
            output[n * M + m3] = OutputType(out3);
        }
    }
}

kernel void matmul_q5_0_f32_f32_mv(
    device const q5_0_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_0_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_0_f16_f32_mv(
    device const q5_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_0_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_0_f16_f16_mv(
    device const q5_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_0_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q5_1_mv_compute(
    device const q5_1_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 4u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const uint m2 = m_base + 2u;
    const uint m3 = m_base + 3u;
    const bool has_m1 = m1 < M;
    const bool has_m2 = m2 < M;
    const bool has_m3 = m3 < M;

    constexpr uint NQ = 16u;
    const uint lane = uint(tiisg);
    const uint ix = lane >> 1;
    const uint il = (lane & 1u) * 8u;

    device const InputType *input_row = input + n * stride_n;
    device const q5_1_block *weight_row0 = weight + m0 * weight_blocks;
    device const q5_1_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;
    device const q5_1_block *weight_row2 = has_m2 ? (weight_row0 + 2u * weight_blocks) : weight_row0;
    device const q5_1_block *weight_row3 = has_m3 ? (weight_row0 + 3u * weight_blocks) : weight_row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const bool blocks_exact = (weight_blocks * kQuantBlockSize) == K;
    const bool contig = stride_k == 1u;

    if (blocks_exact && contig) {
        device const InputType *yb = input_row + ix * kQuantBlockSize + il;
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            float yl[16];
            const float sumy = marmot_mv_load_yl16(yb, yl);
            sum0 += marmot_mv_block_dot_y(&weight_row0[ib], sumy, yl, il);
            sum1 += marmot_mv_block_dot_y(&weight_row1[ib], sumy, yl, il);
            sum2 += marmot_mv_block_dot_y(&weight_row2[ib], sumy, yl, il);
            sum3 += marmot_mv_block_dot_y(&weight_row3[ib], sumy, yl, il);
            yb += kQuantBlockSize * NQ;
        }
    } else {
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            const uint block_start = ib * kQuantBlockSize;
            if (block_start >= K) {
                break;
            }
            const uint block_end = min(block_start + kQuantBlockSize, K);

            const device q5_1_block &w0 = weight_row0[ib];
            const device q5_1_block &w1 = weight_row1[ib];
            const device q5_1_block &w2 = weight_row2[ib];
            const device q5_1_block &w3 = weight_row3[ib];
            const float scale0 = float(w0.scale);
            const float scale1 = float(w1.scale);
            const float scale2 = float(w2.scale);
            const float scale3 = float(w3.scale);
            const float min0 = float(w0.min);
            const float min1 = float(w1.min);
            const float min2 = float(w2.min);
            const float min3 = float(w3.min);
            const uint qh0 = marmot_unpack_q5_high_bits_32(w0.qh);
            const uint qh1 = marmot_unpack_q5_high_bits_32(w1.qh);
            const uint qh2 = marmot_unpack_q5_high_bits_32(w2.qh);
            const uint qh3 = marmot_unpack_q5_high_bits_32(w3.qh);

            float sumy = 0.0f;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                const uint k0 = block_start + il + i;
                if (k0 >= block_end) {
                    break;
                }
                const float a0 = float(input_row[k0 * stride_k]);
                const uint k1 = k0 + 16u;
                const float a1 = (k1 < block_end) ? float(input_row[k1 * stride_k]) : 0.0f;
                sumy += a0 + a1;

                const uint idx0 = il + i;
                const uint idx1 = idx0 + 16u;

                const uchar packed0 = w0.qs[idx0];
                const uint q40 = uint(packed0 & 0x0Fu);
                const uint q41 = uint(packed0 >> 4);
                const uint q50 = q40 + (((qh0 >> idx0) & 0x1u) << 4u);
                const uint q51 = q41 + (((qh0 >> idx1) & 0x1u) << 4u);
                acc0 += float(q50) * a0 + float(q51) * a1;

                const uchar packed1 = w1.qs[idx0];
                const uint q42 = uint(packed1 & 0x0Fu);
                const uint q43 = uint(packed1 >> 4);
                const uint q52 = q42 + (((qh1 >> idx0) & 0x1u) << 4u);
                const uint q53 = q43 + (((qh1 >> idx1) & 0x1u) << 4u);
                acc1 += float(q52) * a0 + float(q53) * a1;

                const uchar packed2 = w2.qs[idx0];
                const uint q44 = uint(packed2 & 0x0Fu);
                const uint q45 = uint(packed2 >> 4);
                const uint q54 = q44 + (((qh2 >> idx0) & 0x1u) << 4u);
                const uint q55 = q45 + (((qh2 >> idx1) & 0x1u) << 4u);
                acc2 += float(q54) * a0 + float(q55) * a1;

                const uchar packed3 = w3.qs[idx0];
                const uint q46 = uint(packed3 & 0x0Fu);
                const uint q47 = uint(packed3 >> 4);
                const uint q56 = q46 + (((qh3 >> idx0) & 0x1u) << 4u);
                const uint q57 = q47 + (((qh3 >> idx1) & 0x1u) << 4u);
                acc3 += float(q56) * a0 + float(q57) * a1;
            }

            sum0 += acc0 * scale0 + sumy * min0;
            sum1 += acc1 * scale1 + sumy * min1;
            sum2 += acc2 * scale2 + sumy * min2;
            sum3 += acc3 * scale3 + sumy * min3;
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    const float out2 = simd_sum(sum2);
    const float out3 = simd_sum(sum3);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
        if (has_m2) {
            output[n * M + m2] = OutputType(out2);
        }
        if (has_m3) {
            output[n * M + m3] = OutputType(out3);
        }
    }
}

kernel void matmul_q5_1_f32_f32_mv(
    device const q5_1_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_1_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_1_f16_f32_mv(
    device const q5_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_1_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q5_1_f16_f16_mv(
    device const q5_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q5_1_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q8_0_mv_compute(
    device const q8_0_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiisg, ushort sgitg
) {
    constexpr uint NR0 = 2u;
    constexpr uint NSG = 2u;
    constexpr uint R0PTG = NR0 * NSG;

    const uint n = tgp.y;
    if (n >= N) {
        return;
    }

    const uint m_base = tgp.x * R0PTG + uint(sgitg) * NR0;
    if (m_base >= M) {
        return;
    }

    const uint m0 = m_base;
    const uint m1 = m_base + 1u;
    const bool has_m1 = m1 < M;

    constexpr uint NQ = 8u;
    const uint lane = uint(tiisg);
    const uint ix = lane >> 2;
    const uint il = (lane & 3u) * 8u;

    device const InputType *input_row = input + n * stride_n;
    device const q8_0_block *weight_row0 = weight + m0 * weight_blocks;
    device const q8_0_block *weight_row1 = has_m1 ? (weight_row0 + weight_blocks) : weight_row0;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const bool blocks_exact = (weight_blocks * kQuantBlockSize) == K;
    const bool contig = stride_k == 1u;

    if (blocks_exact && contig) {
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            const uint block_start = ib * kQuantBlockSize;
            device const InputType *yb = input_row + block_start + il;

            const device q8_0_block &w0 = weight_row0[ib];
            const device q8_0_block &w1 = weight_row1[ib];
            const float scale0 = float(w0.scale);
            const float scale1 = float(w1.scale);

            float acc0 = 0.0f;
            float acc1 = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                const float a = float(yb[i]);
                acc0 += float(int(w0.qs[il + i])) * a;
                acc1 += float(int(w1.qs[il + i])) * a;
            }

            sum0 += acc0 * scale0;
            sum1 += acc1 * scale1;
        }
    } else {
        for (uint ib = ix; ib < weight_blocks; ib += NQ) {
            const uint block_start = ib * kQuantBlockSize;
            if (block_start >= K) {
                break;
            }
            const uint block_end = min(block_start + kQuantBlockSize, K);

            const device q8_0_block &w0 = weight_row0[ib];
            const device q8_0_block &w1 = weight_row1[ib];
            const float scale0 = float(w0.scale);
            const float scale1 = float(w1.scale);

            float acc0 = 0.0f;
            float acc1 = 0.0f;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                const uint k0 = block_start + il + i;
                if (k0 >= block_end) {
                    break;
                }
                const float a = float(input_row[k0 * stride_k]);
                acc0 += float(int(w0.qs[il + i])) * a;
                acc1 += float(int(w1.qs[il + i])) * a;
            }

            sum0 += acc0 * scale0;
            sum1 += acc1 * scale1;
        }
    }

    const float out0 = simd_sum(sum0);
    const float out1 = simd_sum(sum1);
    if (tiisg == 0) {
        output[n * M + m0] = OutputType(out0);
        if (has_m1) {
            output[n * M + m1] = OutputType(out1);
        }
    }
}

kernel void matmul_q8_0_f32_f32_mv(
    device const q8_0_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q8_0_mv_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q8_0_f16_f32_mv(
    device const q8_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q8_0_mv_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

kernel void matmul_q8_0_f16_f16_mv(
    device const q8_0_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]], ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    matmul_q8_0_mv_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiisg, sgitg
    );
}

constant ushort MARMOT_MM_NR0 = 64;
constant ushort MARMOT_MM_NR1 = 32;
constant ushort MARMOT_MM_NR1_16 = 16;
constant ushort MARMOT_MM_NK = 32;

static inline void marmot_mm_store_8x8(
    simdgroup_float8x8 mat, device float *out_ptr, uint out_stride, uint out_row, uint out_col, uint M, uint N,
    threadgroup float *sg_scratch
) {
    const bool full_rows = (out_row + 7u) < M;
    const bool full_cols = (out_col + 7u) < N;
    if (full_rows && full_cols) {
        simdgroup_store(mat, out_ptr, out_stride, 0, false);
        return;
    }
    simdgroup_store(mat, sg_scratch, 8, 0, false);
    for (uint r = 0; r < 8u; ++r) {
        const uint gr = out_row + r;
        if (gr >= M) {
            break;
        }
        for (uint c = 0; c < 8u; ++c) {
            const uint gc = out_col + c;
            if (gc >= N) {
                break;
            }
            out_ptr[(size_t)r + (size_t)out_stride * c] = sg_scratch[c * 8u + r];
        }
    }
}

static inline void marmot_mm_store_8x8(
    simdgroup_float8x8 mat, device half *out_ptr, uint out_stride, uint out_row, uint out_col, uint M, uint N,
    threadgroup float *sg_scratch
) {
    simdgroup_store(mat, sg_scratch, 8, 0, false);
    const bool full_rows = (out_row + 7u) < M;
    const bool full_cols = (out_col + 7u) < N;
    if (full_rows && full_cols) {
#pragma clang loop unroll(full)
        for (uint c = 0; c < 8u; ++c) {
#pragma clang loop unroll(full)
            for (uint r = 0; r < 8u; ++r) {
                out_ptr[(size_t)r + (size_t)out_stride * c] = half(sg_scratch[c * 8u + r]);
            }
        }
        return;
    }
    for (uint r = 0; r < 8u; ++r) {
        const uint gr = out_row + r;
        if (gr >= M) {
            break;
        }
        for (uint c = 0; c < 8u; ++c) {
            const uint gc = out_col + c;
            if (gc >= N) {
                break;
            }
            out_ptr[(size_t)r + (size_t)out_stride * c] = half(sg_scratch[c * 8u + r]);
        }
    }
}

static inline void marmot_mm_copy_full_tile(
    threadgroup float *src, device float *dst, uint M, uint n0, uint m0, ushort tiitg, uint sgitg
) {
    const uint tiisg = (uint)tiitg & 31u;
    for (uint j = sgitg; j < (uint)MARMOT_MM_NR1; j += 4u) {
        device float *dst_row = dst + ((size_t)(n0 + j) * (size_t)M + m0);
        threadgroup float *src_row = src + j * (uint)MARMOT_MM_NR0;
        device float4 *dst4 = (device float4 *)dst_row;
        threadgroup float4 *src4 = (threadgroup float4 *)src_row;
        for (uint i = tiisg; i < (uint)MARMOT_MM_NR0 / 4u; i += 32u) {
            dst4[i] = src4[i];
        }
    }
}

static inline void
marmot_mm_copy_full_tile(threadgroup float *src, device half *dst, uint M, uint n0, uint m0, ushort tiitg, uint sgitg) {
    const uint tiisg = (uint)tiitg & 31u;
    for (uint j = sgitg; j < (uint)MARMOT_MM_NR1; j += 4u) {
        device half *dst_row = dst + ((size_t)(n0 + j) * (size_t)M + m0);
        threadgroup float *src_row = src + j * (uint)MARMOT_MM_NR0;
        device half4 *dst4 = (device half4 *)dst_row;
        threadgroup float4 *src4 = (threadgroup float4 *)src_row;
        for (uint i = tiisg; i < (uint)MARMOT_MM_NR0 / 4u; i += 32u) {
            dst4[i] = half4(src4[i]);
        }
    }
}

static inline void marmot_mm_copy_full_tile_n16(
    threadgroup float *src, device float *dst, uint M, uint n0, uint m0, ushort tiitg, uint sgitg
) {
    const uint tiisg = (uint)tiitg & 31u;
    for (uint j = sgitg; j < (uint)MARMOT_MM_NR1_16; j += 2u) {
        device float *dst_row = dst + ((size_t)(n0 + j) * (size_t)M + m0);
        threadgroup float *src_row = src + j * (uint)MARMOT_MM_NR0;
        device float4 *dst4 = (device float4 *)dst_row;
        threadgroup float4 *src4 = (threadgroup float4 *)src_row;
        for (uint i = tiisg; i < (uint)MARMOT_MM_NR0 / 4u; i += 32u) {
            dst4[i] = src4[i];
        }
    }
}

static inline void marmot_mm_copy_full_tile_n16(
    threadgroup float *src, device half *dst, uint M, uint n0, uint m0, ushort tiitg, uint sgitg
) {
    const uint tiisg = (uint)tiitg & 31u;
    for (uint j = sgitg; j < (uint)MARMOT_MM_NR1_16; j += 2u) {
        device half *dst_row = dst + ((size_t)(n0 + j) * (size_t)M + m0);
        threadgroup float *src_row = src + j * (uint)MARMOT_MM_NR0;
        device half4 *dst4 = (device half4 *)dst_row;
        threadgroup float4 *src4 = (threadgroup float4 *)src_row;
        for (uint i = tiisg; i < (uint)MARMOT_MM_NR0 / 4u; i += 32u) {
            dst4[i] = half4(src4[i]);
        }
    }
}

static inline void marmot_mm_store_8x8_fast(
    simdgroup_float8x8 mat, device float *out_ptr, uint out_stride, threadgroup float *sg_scratch
) {
    (void)sg_scratch;
    simdgroup_store(mat, out_ptr, out_stride, 0, false);
}

static inline void
marmot_mm_store_8x8_fast(simdgroup_float8x8 mat, device half *out_ptr, uint out_stride, threadgroup float *sg_scratch) {
    simdgroup_store(mat, sg_scratch, 8, 0, false);
#pragma clang loop unroll(full)
    for (uint c = 0; c < 8u; ++c) {
#pragma clang loop unroll(full)
        for (uint r = 0; r < 8u; ++r) {
            out_ptr[(size_t)r + (size_t)out_stride * c] = half(sg_scratch[c * 8u + r]);
        }
    }
}

static inline void marmot_basic_mm_dequant_chunk(const device q4_0_block &blk, short il, thread half4x4 &tmp) {
    dequantize_q4_0_chunk(blk, il, tmp);
}

static inline void marmot_basic_mm_dequant_chunk(const device q4_1_block &blk, short il, thread half4x4 &tmp) {
    dequantize_q4_1_chunk(blk, il, tmp);
}

static inline void marmot_basic_mm_dequant_chunk(const device q5_0_block &blk, short il, thread half4x4 &tmp) {
    dequantize_q5_0_chunk(blk, il, tmp);
}

static inline void marmot_basic_mm_dequant_chunk(const device q5_1_block &blk, short il, thread half4x4 &tmp) {
    dequantize_q5_1_chunk(blk, il, tmp);
}

static inline void marmot_basic_mm_dequant_chunk(const device q8_0_block &blk, short il, thread half4x4 &tmp) {
    dequantize_q8_0_chunk(blk, il, tmp);
}

template <typename BlockType, typename InputType, typename OutputType>
static inline void matmul_basic_mm_compute(
    device const BlockType *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * (uint)MARMOT_MM_NR1;
    const uint m0 = tgp.y * (uint)MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQuantBlockSize) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQuantBlockSize;

            const uint lr0 = (uint)tiitg >> 1;
            const uint il = (uint)tiitg & 1u;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;

            half4x4 tmp;
            const device BlockType &blk = weight[gm * weight_blocks + block_idx];
            marmot_basic_mm_dequant_chunk(blk, short(il), tmp);

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 4u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (sgitg & 1u) + (16u * (sgitg >> 1)) * (uint)MARMOT_MM_NR0;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQuantBlockSize;

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;

        if (gm < M && block_idx < weight_blocks) {
            half4x4 tmp;
            const device BlockType &blk = weight[gm * weight_blocks + block_idx];
            marmot_basic_mm_dequant_chunk(blk, short(il), tmp);

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

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

template <typename BlockType, typename InputType, typename OutputType>
static inline void matmul_basic_mm16_compute(
    device const BlockType *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * (uint)MARMOT_MM_NR1_16;
    const uint m0 = tgp.y * (uint)MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1_16 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (uint)sgitg;
    const uint sg_n0 = n0;
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQuantBlockSize) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1_16 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQuantBlockSize;

            const uint lr0 = (uint)tiitg;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;

            const device BlockType &blk = weight[gm * weight_blocks + block_idx];
            half4x4 tmp;

            marmot_basic_mm_dequant_chunk(blk, short(0), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            marmot_basic_mm_dequant_chunk(blk, short(1), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 2u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (uint)sgitg;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile_n16(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQuantBlockSize;

        const uint lr0 = (uint)tiitg;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;

        if (gm < M && block_idx < weight_blocks) {
            half4x4 tmp;
            const device BlockType &blk = weight[gm * weight_blocks + block_idx];

            marmot_basic_mm_dequant_chunk(blk, short(0), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                const uint gk = k_base + sx_a * 8u + ly_a;
                const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
            }

            marmot_basic_mm_dequant_chunk(blk, short(1), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
                const uint ly_a = i & 7u;
                const uint gk = k_base + sx_a * 8u + ly_a;
                const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
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
        const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

#define DEFINE_BASIC_MATMUL_MM_KERNELS(PREFIX, BLOCK_T)                                                                \
    kernel void PREFIX##_f32_f32_mm(                                                                                   \
        device const BLOCK_T *weight [[buffer(0)]], device const float *input [[buffer(1)]],                           \
        device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],            \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],                        \
        ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],                   \
        threadgroup char *shmem [[threadgroup(0)]]                                                                     \
    ) {                                                                                                                \
        matmul_basic_mm_compute<BLOCK_T, float, float>(                                                                \
            weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem                \
        );                                                                                                             \
    }                                                                                                                  \
    kernel void PREFIX##_f16_f32_mm(                                                                                   \
        device const BLOCK_T *weight [[buffer(0)]], device const half *input [[buffer(1)]],                            \
        device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],            \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],                        \
        ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],                   \
        threadgroup char *shmem [[threadgroup(0)]]                                                                     \
    ) {                                                                                                                \
        matmul_basic_mm_compute<BLOCK_T, half, float>(                                                                 \
            weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem                \
        );                                                                                                             \
    }                                                                                                                  \
    kernel void PREFIX##_f16_f16_mm(                                                                                   \
        device const BLOCK_T *weight [[buffer(0)]], device const half *input [[buffer(1)]],                            \
        device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],             \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],                        \
        ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],                   \
        threadgroup char *shmem [[threadgroup(0)]]                                                                     \
    ) {                                                                                                                \
        matmul_basic_mm_compute<BLOCK_T, half, half>(                                                                  \
            weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem                \
        );                                                                                                             \
    }                                                                                                                  \
    kernel void PREFIX##_f32_f32_mm16(                                                                                 \
        device const BLOCK_T *weight [[buffer(0)]], device const float *input [[buffer(1)]],                           \
        device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],            \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],                        \
        ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],                   \
        threadgroup char *shmem [[threadgroup(0)]]                                                                     \
    ) {                                                                                                                \
        matmul_basic_mm16_compute<BLOCK_T, float, float>(                                                              \
            weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem                \
        );                                                                                                             \
    }                                                                                                                  \
    kernel void PREFIX##_f16_f32_mm16(                                                                                 \
        device const BLOCK_T *weight [[buffer(0)]], device const half *input [[buffer(1)]],                            \
        device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],            \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],                        \
        ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],                   \
        threadgroup char *shmem [[threadgroup(0)]]                                                                     \
    ) {                                                                                                                \
        matmul_basic_mm16_compute<BLOCK_T, half, float>(                                                               \
            weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem                \
        );                                                                                                             \
    }                                                                                                                  \
    kernel void PREFIX##_f16_f16_mm16(                                                                                 \
        device const BLOCK_T *weight [[buffer(0)]], device const half *input [[buffer(1)]],                            \
        device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],             \
        constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],  \
        constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],                        \
        ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],                   \
        threadgroup char *shmem [[threadgroup(0)]]                                                                     \
    ) {                                                                                                                \
        matmul_basic_mm16_compute<BLOCK_T, half, half>(                                                                \
            weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem                \
        );                                                                                                             \
    }

DEFINE_BASIC_MATMUL_MM_KERNELS(matmul_q4_0, q4_0_block)
DEFINE_BASIC_MATMUL_MM_KERNELS(matmul_q4_1, q4_1_block)
DEFINE_BASIC_MATMUL_MM_KERNELS(matmul_q5_0, q5_0_block)
DEFINE_BASIC_MATMUL_MM_KERNELS(matmul_q5_1, q5_1_block)
DEFINE_BASIC_MATMUL_MM_KERNELS(matmul_q8_0, q8_0_block)

#undef DEFINE_BASIC_MATMUL_MM_KERNELS

template <typename InputType, typename OutputType>
static inline void matmul_q4_k_mm_compute(
    device const q4_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * MARMOT_MM_NR1;
    const uint m0 = tgp.y * MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQK_K) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQK_K;
            const uint block_offset = k_base & (kQK_K - 1u);

            const uint lr0 = (uint)tiitg >> 1;
            const uint il = (uint)tiitg & 1u;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;
            const uint chunk_idx = (block_offset >> 4) + il;

            half4x4 tmp;
            const device q4_k_block &blk = weight[gm * weight_blocks + block_idx];
            dequantize_q4_k_chunk(blk, short(chunk_idx), tmp);

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 4u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (sgitg & 1u) + (16u * (sgitg >> 1)) * (uint)MARMOT_MM_NR0;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            const device q4_k_block &blk = weight[gm * weight_blocks + block_idx];
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

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

template <typename InputType, typename OutputType>
static inline void matmul_q4_k_mm16_compute(
    device const q4_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * (uint)MARMOT_MM_NR1_16;
    const uint m0 = tgp.y * (uint)MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1_16 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (uint)sgitg;
    const uint sg_n0 = n0;
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQK_K) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1_16 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQK_K;
            const uint block_offset = k_base & (kQK_K - 1u);

            const uint lr0 = (uint)tiitg;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;

            const device q4_k_block &blk = weight[gm * weight_blocks + block_idx];
            half4x4 tmp;

            dequantize_q4_k_chunk(blk, short((block_offset >> 4) + 0u), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            dequantize_q4_k_chunk(blk, short((block_offset >> 4) + 1u), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 2u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (uint)sgitg;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile_n16(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;

        const bool has_row = (gm < M) && (block_idx < weight_blocks);
        if (has_row) {
            const device q4_k_block &blk = weight[gm * weight_blocks + block_idx];
            if (((block_offset >> 4) + 0u) < 16u) {
                half4x4 tmp;
                dequantize_q4_k_chunk(blk, short((block_offset >> 4) + 0u), tmp);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 0u + (i >> 3);
                    const uint ly_a = i & 7u;
                    const uint gk = k_base + sx_a * 8u + ly_a;
                    const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 0u + (i >> 3);
                    const uint ly_a = i & 7u;
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
                }
            }

            if (((block_offset >> 4) + 1u) < 16u) {
                half4x4 tmp;
                dequantize_q4_k_chunk(blk, short((block_offset >> 4) + 1u), tmp);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 2u + (i >> 3);
                    const uint ly_a = i & 7u;
                    const uint gk = k_base + sx_a * 8u + ly_a;
                    const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 2u + (i >> 3);
                    const uint ly_a = i & 7u;
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
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
        const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

kernel void matmul_q4_k_f32_f32_mm(
    device const q4_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q4_k_mm_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q4_k_f16_f32_mm(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q4_k_mm_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q4_k_f16_f16_mm(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q4_k_mm_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q4_k_f32_f32_mm16(
    device const q4_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q4_k_mm16_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q4_k_f16_f32_mm16(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q4_k_mm16_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q4_k_f16_f16_mm16(
    device const q4_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q4_k_mm16_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q5_k_mm_compute(
    device const q5_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * MARMOT_MM_NR1;
    const uint m0 = tgp.y * MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQK_K) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQK_K;
            const uint block_offset = k_base & (kQK_K - 1u);

            const uint lr0 = (uint)tiitg >> 1;
            const uint il = (uint)tiitg & 1u;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;
            const uint chunk_idx = (block_offset >> 4) + il;

            half4x4 tmp;
            const device q5_k_block &blk = weight[gm * weight_blocks + block_idx];
            dequantize_q5_k_chunk(blk, short(chunk_idx), tmp);

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 4u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (sgitg & 1u) + (16u * (sgitg >> 1)) * (uint)MARMOT_MM_NR0;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            const device q5_k_block &blk = weight[gm * weight_blocks + block_idx];
            dequantize_q5_k_chunk(blk, short(chunk_idx), tmp);

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

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

template <typename InputType, typename OutputType>
static inline void matmul_q5_k_mm16_compute(
    device const q5_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * (uint)MARMOT_MM_NR1_16;
    const uint m0 = tgp.y * (uint)MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1_16 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (uint)sgitg;
    const uint sg_n0 = n0;
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQK_K) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1_16 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQK_K;
            const uint block_offset = k_base & (kQK_K - 1u);

            const uint lr0 = (uint)tiitg;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;

            const device q5_k_block &blk = weight[gm * weight_blocks + block_idx];
            half4x4 tmp;

            dequantize_q5_k_chunk(blk, short((block_offset >> 4) + 0u), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            dequantize_q5_k_chunk(blk, short((block_offset >> 4) + 1u), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 2u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (uint)sgitg;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile_n16(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;

        const bool has_row = (gm < M) && (block_idx < weight_blocks);
        if (has_row) {
            const device q5_k_block &blk = weight[gm * weight_blocks + block_idx];
            if (((block_offset >> 4) + 0u) < 16u) {
                half4x4 tmp;
                dequantize_q5_k_chunk(blk, short((block_offset >> 4) + 0u), tmp);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 0u + (i >> 3);
                    const uint ly_a = i & 7u;
                    const uint gk = k_base + sx_a * 8u + ly_a;
                    const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 0u + (i >> 3);
                    const uint ly_a = i & 7u;
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
                }
            }

            if (((block_offset >> 4) + 1u) < 16u) {
                half4x4 tmp;
                dequantize_q5_k_chunk(blk, short((block_offset >> 4) + 1u), tmp);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 2u + (i >> 3);
                    const uint ly_a = i & 7u;
                    const uint gk = k_base + sx_a * 8u + ly_a;
                    const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 2u + (i >> 3);
                    const uint ly_a = i & 7u;
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
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
        const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

kernel void matmul_q5_k_f32_f32_mm(
    device const q5_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q5_k_mm_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q5_k_f16_f32_mm(
    device const q5_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q5_k_mm_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q5_k_f16_f16_mm(
    device const q5_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q5_k_mm_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q5_k_f32_f32_mm16(
    device const q5_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q5_k_mm16_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q5_k_f16_f32_mm16(
    device const q5_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q5_k_mm16_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q5_k_f16_f16_mm16(
    device const q5_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q5_k_mm16_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q6_k_mm_compute(
    device const q6_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * MARMOT_MM_NR1;
    const uint m0 = tgp.y * MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQK_K) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQK_K;
            const uint block_offset = k_base & (kQK_K - 1u);

            const uint lr0 = (uint)tiitg >> 1;
            const uint il = (uint)tiitg & 1u;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;
            const uint chunk_idx = (block_offset >> 4) + il;

            half4x4 tmp;
            const device q6_k_block &blk = weight[gm * weight_blocks + block_idx];
            dequantize_q6_k_chunk(blk, short(chunk_idx), tmp);

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 4u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (sgitg & 1u) + (16u * (sgitg >> 1)) * (uint)MARMOT_MM_NR0;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            const device q6_k_block &blk = weight[gm * weight_blocks + block_idx];
            dequantize_q6_k_chunk(blk, short(chunk_idx), tmp);

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

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

template <typename InputType, typename OutputType>
static inline void matmul_q6_k_mm16_compute(
    device const q6_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * (uint)MARMOT_MM_NR1_16;
    const uint m0 = tgp.y * (uint)MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1_16 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (uint)sgitg;
    const uint sg_n0 = n0;
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQK_K) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1_16 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQK_K;
            const uint block_offset = k_base & (kQK_K - 1u);

            const uint lr0 = (uint)tiitg;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;

            const device q6_k_block &blk = weight[gm * weight_blocks + block_idx];
            half4x4 tmp;

            dequantize_q6_k_chunk(blk, short((block_offset >> 4) + 0u), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            dequantize_q6_k_chunk(blk, short((block_offset >> 4) + 1u), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 2u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (uint)sgitg;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile_n16(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;

        const bool has_row = (gm < M) && (block_idx < weight_blocks);
        if (has_row) {
            const device q6_k_block &blk = weight[gm * weight_blocks + block_idx];
            if (((block_offset >> 4) + 0u) < 16u) {
                half4x4 tmp;
                dequantize_q6_k_chunk(blk, short((block_offset >> 4) + 0u), tmp);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 0u + (i >> 3);
                    const uint ly_a = i & 7u;
                    const uint gk = k_base + sx_a * 8u + ly_a;
                    const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 0u + (i >> 3);
                    const uint ly_a = i & 7u;
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
                }
            }

            if (((block_offset >> 4) + 1u) < 16u) {
                half4x4 tmp;
                dequantize_q6_k_chunk(blk, short((block_offset >> 4) + 1u), tmp);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 2u + (i >> 3);
                    const uint ly_a = i & 7u;
                    const uint gk = k_base + sx_a * 8u + ly_a;
                    const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 2u + (i >> 3);
                    const uint ly_a = i & 7u;
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
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
        const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

kernel void matmul_q6_k_f32_f32_mm(
    device const q6_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q6_k_mm_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q6_k_f16_f32_mm(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q6_k_mm_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q6_k_f16_f16_mm(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q6_k_mm_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q6_k_f32_f32_mm16(
    device const q6_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q6_k_mm16_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q6_k_f16_f32_mm16(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q6_k_mm16_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q6_k_f16_f16_mm16(
    device const q6_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q6_k_mm16_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

template <typename InputType, typename OutputType>
static inline void matmul_q8_k_mm_compute(
    device const q8_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * MARMOT_MM_NR1;
    const uint m0 = tgp.y * MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (sgitg & 1u);
    const uint sg_n0 = n0 + 16u * (sgitg >> 1);
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQK_K) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQK_K;
            const uint block_offset = k_base & (kQK_K - 1u);

            const uint lr0 = (uint)tiitg >> 1;
            const uint il = (uint)tiitg & 1u;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;
            const uint chunk_idx = (block_offset >> 4) + il;

            half4x4 tmp;
            const device q8_k_block &blk = weight[gm * weight_blocks + block_idx];
            dequantize_q8_k_chunk(blk, short(chunk_idx), tmp);

#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u * il + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (4u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (sgitg & 1u);
            threadgroup const half *lsmb = sb + 2u * 64u * (sgitg >> 1);

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 4u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (sgitg & 1u) + (16u * (sgitg >> 1)) * (uint)MARMOT_MM_NR0;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg >> 1;
        const uint il = (uint)tiitg & 1u;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;
        const uint chunk_idx = (block_offset >> 4) + il;

        if (gm < M && block_idx < weight_blocks && chunk_idx < 16u) {
            half4x4 tmp;
            const device q8_k_block &blk = weight[gm * weight_blocks + block_idx];
            dequantize_q8_k_chunk(blk, short(chunk_idx), tmp);

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

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

template <typename InputType, typename OutputType>
static inline void matmul_q8_k_mm16_compute(
    device const q8_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 tgp, ushort tiitg, uint sgitg, threadgroup char *shmem
) {
    const uint n0 = tgp.x * (uint)MARMOT_MM_NR1_16;
    const uint m0 = tgp.y * (uint)MARMOT_MM_NR0;
    if (n0 >= N || m0 >= M) {
        return;
    }

    threadgroup half *sa = (threadgroup half *)(shmem);
    threadgroup half *sb = (threadgroup half *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK * sizeof(half)));
    threadgroup float *scratch =
        (threadgroup float *)(shmem + (MARMOT_MM_NR0 * MARMOT_MM_NK + MARMOT_MM_NR1_16 * MARMOT_MM_NK) * sizeof(half));

    simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8u; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const uint sg_m0 = m0 + 32u * (uint)sgitg;
    const uint sg_n0 = n0;
    const bool sg_active = sg_n0 < N;
    const bool col1_active = sg_active && (sg_n0 + 8u) < N;

    const bool blocks_exact = (weight_blocks * kQK_K) == K;
    const bool contig = (stride_k == 1u) && (stride_n == K);
    const bool fast_full = contig && blocks_exact && ((K & (MARMOT_MM_NK - 1u)) == 0u) &&
        ((M & (MARMOT_MM_NR0 - 1u)) == 0u) && ((N & (MARMOT_MM_NR1_16 - 1u)) == 0u);

    if (fast_full) {
        simdgroup_half8x8 ma[4];
        simdgroup_half8x8 mb[2];

        for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
            const uint block_idx = k_base / kQK_K;
            const uint block_offset = k_base & (kQK_K - 1u);

            const uint lr0 = (uint)tiitg;
            const uint gm = m0 + lr0;
            const uint sy_a = lr0 >> 3;
            const uint lx_a = lr0 & 7u;

            const device q8_k_block &blk = weight[gm * weight_blocks + block_idx];
            half4x4 tmp;

            dequantize_q8_k_chunk(blk, short((block_offset >> 4) + 0u), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            dequantize_q8_k_chunk(blk, short((block_offset >> 4) + 1u), tmp);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = tmp[i / 4u][i % 4u];
            }

            const uint lr1 = (uint)tiitg >> 2;
            const uint sx_b = (uint)tiitg & 3u;
            const uint gn = n0 + lr1;
            const uint sy_b = lr1 >> 3;
            const uint ly_b = lr1 & 7u;
            const uint gk_base = k_base + sx_b * 8u;
            const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

            const device InputType *input_row = input + gn * stride_n;
            *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
            *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

#pragma clang loop unroll(full)
            for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4u; ++i) {
                    simdgroup_load(ma[i], lsma + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 2u; ++i) {
                    simdgroup_load(mb[i], lsmb + 64u * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);

#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i >> 2], ma[i & 3u], mc[i]);
                }

                lsma += 8u * 64u;
                lsmb += 2u * 64u;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (K <= 4096u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            threadgroup float *sc = (threadgroup float *)shmem;
            threadgroup float *temp_str = sc + 32u * (uint)sgitg;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8u; ++i) {
                simdgroup_store(
                    mc[i], temp_str + 8u * (i & 3u) + (8u * (uint)MARMOT_MM_NR0) * (i >> 2), (uint)MARMOT_MM_NR0, 0,
                    false
                );
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            marmot_mm_copy_full_tile_n16(sc, output, M, n0, m0, tiitg, sgitg);
            return;
        }

        threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;
        device OutputType *base_ptr = output + sg_n0 * M + sg_m0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            marmot_mm_store_8x8_fast(mc[i], base_ptr + 8u * (i & 3u) + (8u * M) * (i >> 2), M, sg_scratch);
        }
        return;
    }

    for (uint k_base = 0; k_base < K; k_base += MARMOT_MM_NK) {
        const bool full_k_tile = (k_base + MARMOT_MM_NK) <= K;
        const uint block_idx = k_base / kQK_K;
        const uint block_offset = k_base & (kQK_K - 1u);

        const uint lr0 = (uint)tiitg;
        const uint gm = m0 + lr0;
        const uint sy_a = lr0 >> 3;
        const uint lx_a = lr0 & 7u;

        const bool has_row = (gm < M) && (block_idx < weight_blocks);
        if (has_row) {
            const device q8_k_block &blk = weight[gm * weight_blocks + block_idx];
            if (((block_offset >> 4) + 0u) < 16u) {
                half4x4 tmp;
                dequantize_q8_k_chunk(blk, short((block_offset >> 4) + 0u), tmp);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 0u + (i >> 3);
                    const uint ly_a = i & 7u;
                    const uint gk = k_base + sx_a * 8u + ly_a;
                    const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 0u + (i >> 3);
                    const uint ly_a = i & 7u;
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
                }
            }

            if (((block_offset >> 4) + 1u) < 16u) {
                half4x4 tmp;
                dequantize_q8_k_chunk(blk, short((block_offset >> 4) + 1u), tmp);
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 2u + (i >> 3);
                    const uint ly_a = i & 7u;
                    const uint gk = k_base + sx_a * 8u + ly_a;
                    const half val = full_k_tile ? tmp[i / 4u][i % 4u] : (gk < K ? tmp[i / 4u][i % 4u] : half(0.0f));
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = val;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 16u; ++i) {
                    const uint sx_a = 2u + (i >> 3);
                    const uint ly_a = i & 7u;
                    sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
                }
            }
        } else {
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 0u + (i >> 3);
                const uint ly_a = i & 7u;
                sa[64u * (8u * sx_a + sy_a) + 8u * ly_a + lx_a] = half(0.0f);
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 16u; ++i) {
                const uint sx_a = 2u + (i >> 3);
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
        const uint sb_base = 64u * (2u * sx_b + sy_b) + 8u * ly_b;

        if (gn < N) {
            const device InputType *input_row = input + gn * stride_n;
            const uint gk_base = k_base + k_local_base;
            if (full_k_tile && stride_k == 1u) {
                *((threadgroup half4 *)(sb + sb_base)) = marmot_load_half4(input_row + gk_base);
                *((threadgroup half4 *)(sb + sb_base + 4u)) = marmot_load_half4(input_row + gk_base + 4u);
            } else {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8u; ++i) {
                    const uint gk = gk_base + i;
                    const half val = gk < K ? half(input_row[gk * stride_k]) : half(0.0f);
                    sb[sb_base + i] = val;
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

            threadgroup const half *lsma = sa + 4u * 64u * (uint)sgitg;
            threadgroup const half *lsmb = sb;

            if (col1_active) {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            } else {
#pragma clang loop unroll(full)
                for (uint ik = 0; ik < MARMOT_MM_NK / 8u; ++ik) {
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
                    lsmb += 2u * 64u;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float *sg_scratch = scratch + (uint)sgitg * 64u;

    if (col1_active) {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0 + 8u * (i >> 2);
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    } else {
#pragma clang loop unroll(full)
        for (uint i = 0; i < 4u; ++i) {
            const uint out_row = sg_m0 + 8u * (i & 3u);
            const uint out_col = sg_n0;
            if (out_row >= M || out_col >= N) {
                continue;
            }
            device OutputType *out_ptr = output + out_col * M + out_row;
            marmot_mm_store_8x8(mc[i], out_ptr, M, out_row, out_col, M, N, sg_scratch);
        }
    }
}

kernel void matmul_q8_k_f32_f32_mm(
    device const q8_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q8_k_mm_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q8_k_f16_f32_mm(
    device const q8_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q8_k_mm_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q8_k_f16_f16_mm(
    device const q8_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q8_k_mm_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q8_k_f32_f32_mm16(
    device const q8_k_block *weight [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q8_k_mm16_compute<float, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q8_k_f16_f32_mm16(
    device const q8_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q8_k_mm16_compute<half, float>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

kernel void matmul_q8_k_f16_f16_mm16(
    device const q8_k_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    constant uint &weight_blocks [[buffer(8)]], uint2 tgp [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char *shmem [[threadgroup(0)]]
) {
    matmul_q8_k_mm16_compute<half, half>(
        weight, input, output, N, K, M, stride_n, stride_k, weight_blocks, tgp, tiitg, sgitg, shmem
    );
}

#define STORE_F32(val) (val)
#define STORE_F16(val) half(val)

#define DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(NAME, WEIGHT_BLOCK_T, INPUT_T, DOT_FN, OUT_T, CAST_EXPR)                    \
    kernel void NAME(                                                                                                  \
        device const WEIGHT_BLOCK_T *weight_q [[buffer(0)]], device const WEIGHT_BLOCK_T *weight_k [[buffer(1)]],      \
        device const WEIGHT_BLOCK_T *weight_v [[buffer(2)]], device const INPUT_T *input [[buffer(3)]],                \
        device OUT_T *out_q [[buffer(4)]], device OUT_T *out_k [[buffer(5)]], device OUT_T *out_v [[buffer(6)]],       \
        constant uint &N [[buffer(7)]], constant uint &K [[buffer(8)]], constant uint &M [[buffer(9)]],                \
        constant uint &stride_n [[buffer(10)]], constant uint &stride_k [[buffer(11)]],                                \
        device const float *rope_positions [[buffer(12)]], device const float *rope_freqs [[buffer(13)]],              \
        constant MatmulQKVQuantUniforms &uniforms [[buffer(14)]], uint2 gid [[thread_position_in_grid]],               \
        ushort2 tid [[thread_position_in_threadgroup]]                                                                 \
    ) {                                                                                                                \
        const uint m = gid.x;                                                                                          \
        const uint n = gid.y;                                                                                          \
        if (n >= N || m >= M) {                                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
        const uint weight_blocks = (K + kQK_K - 1u) / kQK_K;                                                           \
        device const WEIGHT_BLOCK_T *row_q = weight_q + m * weight_blocks;                                             \
        device const WEIGHT_BLOCK_T *row_k = weight_k + m * weight_blocks;                                             \
        device const WEIGHT_BLOCK_T *row_v = weight_v + m * weight_blocks;                                             \
        device const INPUT_T *input_row = input + n * stride_n;                                                        \
        float acc_q = 0.0f;                                                                                            \
        float acc_k = 0.0f;                                                                                            \
        float acc_v = 0.0f;                                                                                            \
        for (uint sb = 0; sb < weight_blocks; ++sb) {                                                                  \
            const uint block_start = sb * kQK_K;                                                                       \
            if (block_start >= K) {                                                                                    \
                break;                                                                                                 \
            }                                                                                                          \
            const uint block_len = min(kQK_K, K - block_start);                                                        \
            acc_q += DOT_FN(row_q[sb], input_row, block_start, block_len, stride_k);                                   \
            acc_k += DOT_FN(row_k[sb], input_row, block_start, block_len, stride_k);                                   \
            acc_v += DOT_FN(row_v[sb], input_row, block_start, block_len, stride_k);                                   \
        }                                                                                                              \
        threadgroup float tileRopeQ[MARMOT_QKV_QUANT_TILE_M][MARMOT_QKV_QUANT_TILE_N];                                 \
        threadgroup float tileRopeK[MARMOT_QKV_QUANT_TILE_M][MARMOT_QKV_QUANT_TILE_N];                                 \
        bool rope_enabled = (uniforms.rope_enabled != 0u) && rope_positions != nullptr && rope_freqs != nullptr;       \
        bool apply_rope_q = rope_enabled && (uniforms.rope_apply_q != 0u);                                             \
        bool apply_rope_k = rope_enabled && (uniforms.rope_apply_k != 0u);                                             \
        bool needs_rope = apply_rope_q || apply_rope_k;                                                                \
        uint rope_head_dim = uniforms.rope_head_dim;                                                                   \
        if (rope_head_dim == 0u || rope_head_dim > M || (M % rope_head_dim) != 0u || (rope_head_dim & 1u) != 0u) {     \
            rope_head_dim = M;                                                                                         \
        }                                                                                                              \
        float final_q = acc_q;                                                                                         \
        float final_k = acc_k;                                                                                         \
        if (needs_rope) {                                                                                              \
            const uint tile_col_base = gid.x - tid.x;                                                                  \
            tileRopeQ[tid.y][tid.x] = acc_q;                                                                           \
            tileRopeK[tid.y][tid.x] = acc_k;                                                                           \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            uint head_base = (m / rope_head_dim) * rope_head_dim;                                                      \
            uint local = m - head_base;                                                                                \
            uint even_local = local & ~1u;                                                                             \
            uint odd_local = even_local + 1u;                                                                          \
            if (odd_local < rope_head_dim) {                                                                           \
                uint even_col = head_base + even_local;                                                                \
                uint odd_col = head_base + odd_local;                                                                  \
                uint even_tile = even_col - tile_col_base;                                                             \
                uint odd_tile = odd_col - tile_col_base;                                                               \
                if (even_tile < MARMOT_QKV_QUANT_TILE_N && odd_tile < MARMOT_QKV_QUANT_TILE_N) {                       \
                    float even_q = tileRopeQ[tid.y][even_tile];                                                        \
                    float odd_q = tileRopeQ[tid.y][odd_tile];                                                          \
                    float even_k = tileRopeK[tid.y][even_tile];                                                        \
                    float odd_k = tileRopeK[tid.y][odd_tile];                                                          \
                    float position = rope_positions[n];                                                                \
                    float freq = rope_freqs[even_local >> 1];                                                          \
                    float angle = position * freq;                                                                     \
                    float cos_val = cos(angle) * uniforms.rope_attn_scale;                                             \
                    float sin_val = sin(angle) * uniforms.rope_attn_scale;                                             \
                    if (apply_rope_q) {                                                                                \
                        float rotated_even_q = even_q * cos_val - odd_q * sin_val;                                     \
                        float rotated_odd_q = even_q * sin_val + odd_q * cos_val;                                      \
                        final_q = (m == even_col) ? rotated_even_q : rotated_odd_q;                                    \
                    }                                                                                                  \
                    if (apply_rope_k) {                                                                                \
                        float rotated_even_k = even_k * cos_val - odd_k * sin_val;                                     \
                        float rotated_odd_k = even_k * sin_val + odd_k * cos_val;                                      \
                        final_k = (m == even_col) ? rotated_even_k : rotated_odd_k;                                    \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
        const uint out_index = n * M + m;                                                                              \
        out_q[out_index] = CAST_EXPR(final_q);                                                                         \
        out_k[out_index] = CAST_EXPR(final_k);                                                                         \
        out_v[out_index] = CAST_EXPR(acc_v);                                                                           \
    }

#define DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(NAME, WEIGHT_BLOCK_T, INPUT_T, DOT_FN, OUT_T, CAST_EXPR)               \
    kernel void NAME(                                                                                                  \
        device const WEIGHT_BLOCK_T *weight_q [[buffer(0)]], device const WEIGHT_BLOCK_T *weight_k [[buffer(1)]],      \
        device const WEIGHT_BLOCK_T *weight_v [[buffer(2)]], device const INPUT_T *input [[buffer(3)]],                \
        device OUT_T *out_q [[buffer(4)]], device OUT_T *out_k [[buffer(5)]], device OUT_T *out_v [[buffer(6)]],       \
        constant uint &N [[buffer(7)]], constant uint &K [[buffer(8)]], constant uint &M [[buffer(9)]],                \
        constant uint &stride_n [[buffer(10)]], constant uint &stride_k [[buffer(11)]],                                \
        device const float *rope_positions [[buffer(12)]], device const float *rope_freqs [[buffer(13)]],              \
        constant MatmulQKVQuantUniforms &uniforms [[buffer(14)]], uint2 gid [[thread_position_in_grid]],               \
        ushort2 tid [[thread_position_in_threadgroup]]                                                                 \
    ) {                                                                                                                \
        (void)weight_v;                                                                                                \
        (void)out_v;                                                                                                   \
        const uint m = gid.x;                                                                                          \
        const uint n = gid.y;                                                                                          \
        if (n >= N || m >= M) {                                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
        const uint weight_blocks = (K + kQK_K - 1u) / kQK_K;                                                           \
        device const WEIGHT_BLOCK_T *row_q = weight_q + m * weight_blocks;                                             \
        device const WEIGHT_BLOCK_T *row_k = weight_k + m * weight_blocks;                                             \
        device const INPUT_T *input_row = input + n * stride_n;                                                        \
        float acc_q = 0.0f;                                                                                            \
        float acc_k = 0.0f;                                                                                            \
        for (uint sb = 0; sb < weight_blocks; ++sb) {                                                                  \
            const uint block_start = sb * kQK_K;                                                                       \
            if (block_start >= K) {                                                                                    \
                break;                                                                                                 \
            }                                                                                                          \
            const uint block_len = min(kQK_K, K - block_start);                                                        \
            acc_q += DOT_FN(row_q[sb], input_row, block_start, block_len, stride_k);                                   \
            acc_k += DOT_FN(row_k[sb], input_row, block_start, block_len, stride_k);                                   \
        }                                                                                                              \
        threadgroup float tileRopeQ[MARMOT_QKV_QUANT_TILE_M][MARMOT_QKV_QUANT_TILE_N];                                 \
        threadgroup float tileRopeK[MARMOT_QKV_QUANT_TILE_M][MARMOT_QKV_QUANT_TILE_N];                                 \
        bool rope_enabled = (uniforms.rope_enabled != 0u) && rope_positions != nullptr && rope_freqs != nullptr;       \
        bool apply_rope_q = rope_enabled && (uniforms.rope_apply_q != 0u);                                             \
        bool apply_rope_k = rope_enabled && (uniforms.rope_apply_k != 0u);                                             \
        bool needs_rope = apply_rope_q || apply_rope_k;                                                                \
        uint rope_head_dim = uniforms.rope_head_dim;                                                                   \
        if (rope_head_dim == 0u || rope_head_dim > M || (M % rope_head_dim) != 0u || (rope_head_dim & 1u) != 0u) {     \
            rope_head_dim = M;                                                                                         \
        }                                                                                                              \
        float final_q = acc_q;                                                                                         \
        float final_k = acc_k;                                                                                         \
        if (needs_rope) {                                                                                              \
            const uint tile_col_base = gid.x - tid.x;                                                                  \
            tileRopeQ[tid.y][tid.x] = acc_q;                                                                           \
            tileRopeK[tid.y][tid.x] = acc_k;                                                                           \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            uint head_base = (m / rope_head_dim) * rope_head_dim;                                                      \
            uint local = m - head_base;                                                                                \
            uint even_local = local & ~1u;                                                                             \
            uint odd_local = even_local + 1u;                                                                          \
            if (odd_local < rope_head_dim) {                                                                           \
                uint even_col = head_base + even_local;                                                                \
                uint odd_col = head_base + odd_local;                                                                  \
                uint even_tile = even_col - tile_col_base;                                                             \
                uint odd_tile = odd_col - tile_col_base;                                                               \
                if (even_tile < MARMOT_QKV_QUANT_TILE_N && odd_tile < MARMOT_QKV_QUANT_TILE_N) {                       \
                    float even_q = tileRopeQ[tid.y][even_tile];                                                        \
                    float odd_q = tileRopeQ[tid.y][odd_tile];                                                          \
                    float even_k = tileRopeK[tid.y][even_tile];                                                        \
                    float odd_k = tileRopeK[tid.y][odd_tile];                                                          \
                    float position = rope_positions[n];                                                                \
                    float freq = rope_freqs[even_local >> 1];                                                          \
                    float angle = position * freq;                                                                     \
                    float cos_val = cos(angle) * uniforms.rope_attn_scale;                                             \
                    float sin_val = sin(angle) * uniforms.rope_attn_scale;                                             \
                    if (apply_rope_q) {                                                                                \
                        float rotated_even_q = even_q * cos_val - odd_q * sin_val;                                     \
                        float rotated_odd_q = even_q * sin_val + odd_q * cos_val;                                      \
                        final_q = (m == even_col) ? rotated_even_q : rotated_odd_q;                                    \
                    }                                                                                                  \
                    if (apply_rope_k) {                                                                                \
                        float rotated_even_k = even_k * cos_val - odd_k * sin_val;                                     \
                        float rotated_odd_k = even_k * sin_val + odd_k * cos_val;                                      \
                        final_k = (m == even_col) ? rotated_even_k : rotated_odd_k;                                    \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
        const uint out_index = n * M + m;                                                                              \
        out_q[out_index] = CAST_EXPR(final_q);                                                                         \
        out_k[out_index] = CAST_EXPR(final_k);                                                                         \
    }

DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q2_k_f32_f32, q2_k_block, float, compute_q2k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q2_k_f32_f16, q2_k_block, float, compute_q2k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q2_k_f16_f32, q2_k_block, half, compute_q2k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q2_k_f16_f16, q2_k_block, half, compute_q2k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q2_k_dual_f32_f32, q2_k_block, float, compute_q2k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q2_k_dual_f32_f16, q2_k_block, float, compute_q2k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q2_k_dual_f16_f32, q2_k_block, half, compute_q2k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q2_k_dual_f16_f16, q2_k_block, half, compute_q2k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q3_k_f32_f32, q3_k_block, float, compute_q3k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q3_k_f32_f16, q3_k_block, float, compute_q3k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q3_k_f16_f32, q3_k_block, half, compute_q3k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q3_k_f16_f16, q3_k_block, half, compute_q3k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q3_k_dual_f32_f32, q3_k_block, float, compute_q3k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q3_k_dual_f32_f16, q3_k_block, float, compute_q3k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q3_k_dual_f16_f32, q3_k_block, half, compute_q3k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q3_k_dual_f16_f16, q3_k_block, half, compute_q3k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q4_k_f32_f32, q4_k_block, float, compute_q4k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q4_k_f32_f16, q4_k_block, float, compute_q4k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q4_k_f16_f32, q4_k_block, half, compute_q4k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q4_k_f16_f16, q4_k_block, half, compute_q4k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q4_k_dual_f32_f32, q4_k_block, float, compute_q4k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q4_k_dual_f32_f16, q4_k_block, float, compute_q4k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q4_k_dual_f16_f32, q4_k_block, half, compute_q4k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q4_k_dual_f16_f16, q4_k_block, half, compute_q4k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q5_k_f32_f32, q5_k_block, float, compute_q5k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q5_k_f32_f16, q5_k_block, float, compute_q5k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q5_k_f16_f32, q5_k_block, half, compute_q5k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q5_k_f16_f16, q5_k_block, half, compute_q5k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q5_k_dual_f32_f32, q5_k_block, float, compute_q5k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q5_k_dual_f32_f16, q5_k_block, float, compute_q5k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q5_k_dual_f16_f32, q5_k_block, half, compute_q5k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q5_k_dual_f16_f16, q5_k_block, half, compute_q5k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q6_k_f32_f32, q6_k_block, float, compute_q6k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q6_k_f32_f16, q6_k_block, float, compute_q6k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q6_k_f16_f32, q6_k_block, half, compute_q6k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q6_k_f16_f16, q6_k_block, half, compute_q6k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q6_k_dual_f32_f32, q6_k_block, float, compute_q6k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q6_k_dual_f32_f16, q6_k_block, float, compute_q6k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q6_k_dual_f16_f32, q6_k_block, half, compute_q6k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q6_k_dual_f16_f16, q6_k_block, half, compute_q6k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q8_k_f32_f32, q8_k_block, float, compute_q8k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q8_k_f32_f16, q8_k_block, float, compute_q8k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q8_k_f16_f32, q8_k_block, half, compute_q8k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL(
    matmul_qkv_q8_k_f16_f16, q8_k_block, half, compute_q8k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q8_k_dual_f32_f32, q8_k_block, float, compute_q8k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q8_k_dual_f32_f16, q8_k_block, float, compute_q8k_block_dot_direct, half, STORE_F16
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q8_k_dual_f16_f32, q8_k_block, half, compute_q8k_block_dot_direct, float, STORE_F32
)
DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL(
    matmul_qkv_q8_k_dual_f16_f16, q8_k_block, half, compute_q8k_block_dot_direct, half, STORE_F16
)

#define DEFINE_MATMUL_QKV_Q80_KERNEL(NAME, WEIGHT_BLOCK_T, ACCUM_FN, OUT_T, CAST_EXPR)                                 \
    kernel void NAME(                                                                                                  \
        device const WEIGHT_BLOCK_T *weight_q [[buffer(0)]], device const WEIGHT_BLOCK_T *weight_k [[buffer(1)]],      \
        device const WEIGHT_BLOCK_T *weight_v [[buffer(2)]], device const q8_0_block *input [[buffer(3)]],             \
        device OUT_T *out_q [[buffer(4)]], device OUT_T *out_k [[buffer(5)]], device OUT_T *out_v [[buffer(6)]],       \
        constant uint &N [[buffer(7)]], constant uint &K_blocks [[buffer(8)]], constant uint &M [[buffer(9)]],         \
        constant uint &weight_blocks [[buffer(10)]], constant uint &K [[buffer(11)]],                                  \
        device const float *rope_positions [[buffer(12)]], device const float *rope_freqs [[buffer(13)]],              \
        constant MatmulQKVQuantUniforms &uniforms [[buffer(14)]], uint2 gid [[thread_position_in_grid]],               \
        ushort2 tid [[thread_position_in_threadgroup]]                                                                 \
    ) {                                                                                                                \
        if (tid.x >= MARMOT_QKV_QUANT_TILE_N || tid.y >= MARMOT_QKV_QUANT_TILE_M) {                                    \
            return;                                                                                                    \
        }                                                                                                              \
        const uint m = gid.x;                                                                                          \
        const uint n = gid.y;                                                                                          \
        if (n >= N || m >= M) {                                                                                        \
            return;                                                                                                    \
        }                                                                                                              \
        device const WEIGHT_BLOCK_T *row_q = weight_q + m * weight_blocks;                                             \
        device const WEIGHT_BLOCK_T *row_k = weight_k + m * weight_blocks;                                             \
        device const WEIGHT_BLOCK_T *row_v = weight_v + m * weight_blocks;                                             \
        device const q8_0_block *input_row = input + n * K_blocks;                                                     \
        float acc_q = 0.0f;                                                                                            \
        float acc_k = 0.0f;                                                                                            \
        float acc_v = 0.0f;                                                                                            \
        for (uint sb = 0; sb < K_blocks; ++sb) {                                                                       \
            ACCUM_FN(row_q[sb], row_k[sb], row_v[sb], input_row[sb], acc_q, acc_k, acc_v);                             \
        }                                                                                                              \
        threadgroup float tileRopeQ[MARMOT_QKV_QUANT_TILE_M][MARMOT_QKV_QUANT_TILE_N];                                 \
        threadgroup float tileRopeK[MARMOT_QKV_QUANT_TILE_M][MARMOT_QKV_QUANT_TILE_N];                                 \
        bool rope_enabled = (uniforms.rope_enabled != 0u) && rope_positions != nullptr && rope_freqs != nullptr;       \
        bool apply_rope_q = rope_enabled && (uniforms.rope_apply_q != 0u);                                             \
        bool apply_rope_k = rope_enabled && (uniforms.rope_apply_k != 0u);                                             \
        bool needs_rope = apply_rope_q || apply_rope_k;                                                                \
        uint rope_head_dim = uniforms.rope_head_dim;                                                                   \
        if (rope_head_dim == 0u || rope_head_dim > M || (M % rope_head_dim) != 0u || (rope_head_dim & 1u) != 0u) {     \
            rope_head_dim = M;                                                                                         \
        }                                                                                                              \
        float final_q = acc_q;                                                                                         \
        float final_k = acc_k;                                                                                         \
        float final_v = acc_v;                                                                                         \
        if (needs_rope) {                                                                                              \
            const uint tile_col_base = gid.x - tid.x;                                                                  \
            tileRopeQ[tid.y][tid.x] = acc_q;                                                                           \
            tileRopeK[tid.y][tid.x] = acc_k;                                                                           \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
            uint head_base = (m / rope_head_dim) * rope_head_dim;                                                      \
            uint local = m - head_base;                                                                                \
            uint even_local = local & ~1u;                                                                             \
            uint odd_local = even_local + 1u;                                                                          \
            if (odd_local < rope_head_dim) {                                                                           \
                uint even_col = head_base + even_local;                                                                \
                uint odd_col = head_base + odd_local;                                                                  \
                uint even_tile = even_col - tile_col_base;                                                             \
                uint odd_tile = odd_col - tile_col_base;                                                               \
                if (even_tile < MARMOT_QKV_QUANT_TILE_N && odd_tile < MARMOT_QKV_QUANT_TILE_N) {                       \
                    float even_q = tileRopeQ[tid.y][even_tile];                                                        \
                    float odd_q = tileRopeQ[tid.y][odd_tile];                                                          \
                    float even_k = tileRopeK[tid.y][even_tile];                                                        \
                    float odd_k = tileRopeK[tid.y][odd_tile];                                                          \
                    float position = rope_positions[n];                                                                \
                    float freq = rope_freqs[even_local >> 1];                                                          \
                    float angle = position * freq;                                                                     \
                    float cos_val = cos(angle) * uniforms.rope_attn_scale;                                             \
                    float sin_val = sin(angle) * uniforms.rope_attn_scale;                                             \
                    if (apply_rope_q) {                                                                                \
                        float rotated_even_q = even_q * cos_val - odd_q * sin_val;                                     \
                        float rotated_odd_q = even_q * sin_val + odd_q * cos_val;                                      \
                        final_q = (m == even_col) ? rotated_even_q : rotated_odd_q;                                    \
                    }                                                                                                  \
                    if (apply_rope_k) {                                                                                \
                        float rotated_even_k = even_k * cos_val - odd_k * sin_val;                                     \
                        float rotated_odd_k = even_k * sin_val + odd_k * cos_val;                                      \
                        final_k = (m == even_col) ? rotated_even_k : rotated_odd_k;                                    \
                    }                                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            threadgroup_barrier(mem_flags::mem_threadgroup);                                                           \
        }                                                                                                              \
        const uint out_index = n * M + m;                                                                              \
        out_q[out_index] = CAST_EXPR(final_q);                                                                         \
        out_k[out_index] = CAST_EXPR(final_k);                                                                         \
        out_v[out_index] = CAST_EXPR(final_v);                                                                         \
    }

DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q4_0_q8_0_f32, q4_0_block, matmul_qkv_accumulate_q4_0_block, float, STORE_F32)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q4_0_q8_0_f16, q4_0_block, matmul_qkv_accumulate_q4_0_block, half, STORE_F16)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q4_1_q8_0_f32, q4_1_block, matmul_qkv_accumulate_q4_1_block, float, STORE_F32)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q4_1_q8_0_f16, q4_1_block, matmul_qkv_accumulate_q4_1_block, half, STORE_F16)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q5_0_q8_0_f32, q5_0_block, matmul_qkv_accumulate_q5_0_block, float, STORE_F32)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q5_0_q8_0_f16, q5_0_block, matmul_qkv_accumulate_q5_0_block, half, STORE_F16)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q5_1_q8_0_f32, q5_1_block, matmul_qkv_accumulate_q5_1_block, float, STORE_F32)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q5_1_q8_0_f16, q5_1_block, matmul_qkv_accumulate_q5_1_block, half, STORE_F16)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q8_0_q8_0_f32, q8_0_block, matmul_qkv_accumulate_q8_0_block, float, STORE_F32)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q8_0_q8_0_f16, q8_0_block, matmul_qkv_accumulate_q8_0_block, half, STORE_F16)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q8_1_q8_0_f32, q8_1_block, matmul_qkv_accumulate_q8_1_block, float, STORE_F32)
DEFINE_MATMUL_QKV_Q80_KERNEL(matmul_qkv_q8_1_q8_0_f16, q8_1_block, matmul_qkv_accumulate_q8_1_block, half, STORE_F16)

#undef STORE_F32
#undef STORE_F16
#undef DEFINE_MATMUL_QKV_QK_DIRECT_KERNEL
#undef DEFINE_MATMUL_QKV_QK_DIRECT_DUAL_KERNEL
#undef DEFINE_MATMUL_QKV_Q80_KERNEL

// Q4_1 × Q8_0 matmul
// PyTorch convention: input(N×K) @ weight(M×K).T = output(N×M)
// -----------------------------------------------------------------------------

kernel void matmul_q4_1_q8_0(
    device const q4_1_block *weight [[buffer(0)]], device const q8_0_block *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K_blocks [[buffer(4)]],
    constant uint &M [[buffer(5)]], uint2 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x; // output feature index (0..M-1)
    const uint n = gid.y; // batch/sample index (0..N-1)

    if (n >= N || m >= M) {
        return;
    }

    float sum = 0.0f;

    device const q8_0_block *input_row = input + n * K_blocks;
    device const q4_1_block *weight_row = weight + m * K_blocks;

    for (uint b = 0; b < K_blocks; ++b) {
        const q4_1_block w_block = weight_row[b];
        const q8_0_block i_block = input_row[b];

        int block_q_sum = 0;
        int block_i_sum = 0;
        for (uint i = 0; i < kQ4PackedBytes; ++i) {
            const uchar packed = w_block.qs[i];
            const int qw0 = int(packed & 0x0fu);
            const int qw1 = int(packed >> 4);
            const int i0 = int(i_block.qs[i]);
            const int i1 = int(i_block.qs[i + kQuantBlockSize / 2]);
            block_q_sum += qw0 * i0 + qw1 * i1;
            block_i_sum += i0 + i1;
        }

        const float scale_w = float(w_block.scale);
        const float min_w = float(w_block.min);
        const float scale_i = float(i_block.scale);
        const float block_scale = scale_w * scale_i;
        const float bias_scale = min_w * scale_i;
        sum += block_scale * float(block_q_sum) + bias_scale * float(block_i_sum);
    }

    output[n * M + m] = sum;
}

// -----------------------------------------------------------------------------
// Q8_1 × FP16 matmul -> FP32
// -----------------------------------------------------------------------------

kernel void matmul_q8_1_f16_f32(
    device const q8_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint n = gid.y;
    if (n >= N || m >= M) {
        return;
    }

    const uint K_blocks = (K + kQuantBlockSize - 1u) / kQuantBlockSize;
    device const half *input_row = input + n * stride_n;
    device const q8_1_block *weight_row = weight + m * K_blocks;

    float sum = 0.0f;

    for (uint b = 0; b < K_blocks; ++b) {
        const q8_1_block w_block = weight_row[b];
        const float scale_w = float(w_block.scale);

        const uint block_start = b * kQuantBlockSize;
        const uint block_end = min(block_start + kQuantBlockSize, K);

        for (uint i = 0; i < kQuantBlockSize && (block_start + i) < block_end; ++i) {
            const uint idx = block_start + i;
            const float w_val = scale_w * float(w_block.qs[i]);
            const float a_val = float(input_row[idx * stride_k]);
            sum += w_val * a_val;
        }
    }

    output[n * M + m] = sum;
}

// -----------------------------------------------------------------------------
// Q8_1 × FP16 matmul -> FP16
// -----------------------------------------------------------------------------

kernel void matmul_q8_1_f16_f16(
    device const q8_1_block *weight [[buffer(0)]], device const half *input [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K [[buffer(4)]],
    constant uint &M [[buffer(5)]], constant uint &stride_n [[buffer(6)]], constant uint &stride_k [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint n = gid.y;
    if (n >= N || m >= M) {
        return;
    }

    const uint K_blocks = (K + kQuantBlockSize - 1u) / kQuantBlockSize;
    device const half *input_row = input + n * stride_n;
    device const q8_1_block *weight_row = weight + m * K_blocks;

    float sum = 0.0f;

    for (uint b = 0; b < K_blocks; ++b) {
        const q8_1_block w_block = weight_row[b];
        const float scale_w = float(w_block.scale);

        const uint block_start = b * kQuantBlockSize;
        const uint block_end = min(block_start + kQuantBlockSize, K);

        for (uint i = 0; i < kQuantBlockSize && (block_start + i) < block_end; ++i) {
            const uint idx = block_start + i;
            const float w_val = scale_w * float(w_block.qs[i]);
            const float a_val = float(input_row[idx * stride_k]);
            sum += w_val * a_val;
        }
    }

    output[n * M + m] = half(sum);
}

// -----------------------------------------------------------------------------
// Q5_0 × Q8_0 matmul
// PyTorch convention: input(N×K) @ weight(M×K).T = output(N×M)
// -----------------------------------------------------------------------------

kernel void matmul_q5_0_q8_0(
    device const q5_0_block *weight [[buffer(0)]], device const q8_0_block *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K_blocks [[buffer(4)]],
    constant uint &M [[buffer(5)]], uint2 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x; // output feature index (0..M-1)
    const uint n = gid.y; // batch/sample index (0..N-1)

    if (n >= N || m >= M) {
        return;
    }

    float sum = 0.0f;

    device const q8_0_block *input_row = input + n * K_blocks;
    device const q5_0_block *weight_row = weight + m * K_blocks;

    for (uint b = 0; b < K_blocks; ++b) {
        const q5_0_block w_block = weight_row[b];
        const q8_0_block i_block = input_row[b];

        const uint qh = (uint)w_block.qh[0] | ((uint)w_block.qh[1] << 8u) | ((uint)w_block.qh[2] << 16u) |
            ((uint)w_block.qh[3] << 24u);

        int block_sum = 0;
        const uint half_block = kQuantBlockSize / 2;
        for (uint i = 0; i < kQ5PackedBytes; ++i) {
            const uchar packed = w_block.qs[i];
            uint lo = uint(packed & 0x0fu);
            uint hi = uint(packed >> 4);
            lo |= ((qh >> i) & 0x1u) << 4;
            hi |= ((qh >> (i + half_block)) & 0x1u) << 4;

            const int w0 = int(lo) - 16;
            const int w1 = int(hi) - 16;
            const int i0 = int(i_block.qs[i]);
            const int i1 = int(i_block.qs[i + half_block]);
            block_sum += w0 * i0 + w1 * i1;
        }

        const float block_scale = float(w_block.scale) * float(i_block.scale);
        sum += float(block_sum) * block_scale;
    }

    output[n * M + m] = sum;
}

// -----------------------------------------------------------------------------
// Q5_1 × Q8_0 matmul
// PyTorch convention: input(N×K) @ weight(M×K).T = output(N×M)
// -----------------------------------------------------------------------------

kernel void matmul_q5_1_q8_0(
    device const q5_1_block *weight [[buffer(0)]], device const q8_0_block *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K_blocks [[buffer(4)]],
    constant uint &M [[buffer(5)]], uint2 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x; // output feature index (0..M-1)
    const uint n = gid.y; // batch/sample index (0..N-1)

    if (n >= N || m >= M) {
        return;
    }

    float sum = 0.0f;

    device const q8_0_block *input_row = input + n * K_blocks;
    device const q5_1_block *weight_row = weight + m * K_blocks;

    for (uint b = 0; b < K_blocks; ++b) {
        const q5_1_block w_block = weight_row[b];
        const float scale_w = float(w_block.scale);
        const float min_w = float(w_block.min);
        const uint qh = (uint)w_block.qh[0] | ((uint)w_block.qh[1] << 8u) | ((uint)w_block.qh[2] << 16u) |
            ((uint)w_block.qh[3] << 24u);

        const q8_0_block i_block = input_row[b];

        int block_q_sum = 0;
        int block_i_sum = 0;
        const uint half_block = kQuantBlockSize / 2;
        for (uint i = 0; i < kQ5PackedBytes; ++i) {
            const uchar packed = w_block.qs[i];
            uint lo = uint(packed & 0x0fu);
            uint hi = uint(packed >> 4);
            lo |= ((qh >> i) & 0x1u) << 4;
            hi |= ((qh >> (i + half_block)) & 0x1u) << 4;

            const int qw0 = int(lo);
            const int qw1 = int(hi);
            const int i0 = int(i_block.qs[i]);
            const int i1 = int(i_block.qs[i + half_block]);

            block_q_sum += qw0 * i0 + qw1 * i1;
            block_i_sum += i0 + i1;
        }

        const float scale_i = float(i_block.scale);
        const float block_scale = scale_w * scale_i;
        const float bias_scale = min_w * scale_i;
        sum += block_scale * float(block_q_sum) + bias_scale * float(block_i_sum);
    }

    output[n * M + m] = sum;
}

// -----------------------------------------------------------------------------
// Q8_1 × Q8_0 matmul
// PyTorch convention: input(N×K) @ weight(M×K).T = output(N×M)
// -----------------------------------------------------------------------------

kernel void matmul_q8_1_q8_0(
    device const q8_1_block *weight [[buffer(0)]], device const q8_0_block *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K_blocks [[buffer(4)]],
    constant uint &M [[buffer(5)]], uint2 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint n = gid.y;

    if (n >= N || m >= M) {
        return;
    }

    float sum = 0.0f;

    device const q8_0_block *input_row = input + n * K_blocks;
    device const q8_1_block *weight_row = weight + m * K_blocks;

    for (uint b = 0; b < K_blocks; ++b) {
        const q8_1_block w_block = weight_row[b];
        const q8_0_block a_block = input_row[b];

        int block_sum = 0;
        for (uint i = 0; i < kQuantBlockSize; ++i) {
            const int wv = int(w_block.qs[i]);
            const int av = int(a_block.qs[i]);
            block_sum += wv * av;
        }

        const float block_scale = float(w_block.scale) * float(a_block.scale);
        sum += float(block_sum) * block_scale;
    }

    output[n * M + m] = sum;
}

// -----------------------------------------------------------------------------
// Q8_0 × Q8_0 matmul
// PyTorch convention: input(N×K) @ weight(M×K).T = output(N×M)
// -----------------------------------------------------------------------------

kernel void matmul_q8_0_q8_0(
    device const q8_0_block *weight [[buffer(0)]], device const q8_0_block *input [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &N [[buffer(3)]], constant uint &K_blocks [[buffer(4)]],
    constant uint &M [[buffer(5)]], uint2 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x; // output feature index (0..M-1)
    const uint n = gid.y; // batch/sample index (0..N-1)

    if (n >= N || m >= M) {
        return;
    }

    float sum = 0.0f;

    device const q8_0_block *input_row = input + n * K_blocks;
    device const q8_0_block *weight_row = weight + m * K_blocks;

    for (uint b = 0; b < K_blocks; ++b) {
        const q8_0_block w_block = weight_row[b];
        const q8_0_block i_block = input_row[b];

        int block_sum = 0;
        for (uint i = 0; i < kQuantBlockSize; ++i) {
            const int w = int(w_block.qs[i]);
            const int i_val = int(i_block.qs[i]);
            block_sum += w * i_val;
        }

        const float block_scale = float(w_block.scale) * float(i_block.scale);
        sum += float(block_sum) * block_scale;
    }

    output[n * M + m] = sum;
}

// -----------------------------------------------------------------------------
// Q2_K × FP32 matmul -> FP32/FP16
// -----------------------------------------------------------------------------

template <typename InputType, typename OutputType>
static inline void matmul_q2_k_compute(
    device const q2_k_block *weight, device const InputType *input, device OutputType *output, uint N, uint K, uint M,
    uint stride_n, uint stride_k, uint weight_blocks, uint2 gid
) {
    const uint m = gid.x;
    const uint n = gid.y;
    if (n >= N || m >= M) {
        return;
    }

    device const q2_k_block *weight_row = weight + m * weight_blocks;
    device const InputType *input_row = input + n * stride_n;

    float sum = 0.0f;

    for (uint sb = 0; sb < weight_blocks; ++sb) {
        const uint block_start = sb * kQK_K;
        if (block_start >= K) {
            break;
        }
        const uint block_len = min(kQK_K, K - block_start);
        const device q2_k_block &w_block = weight_row[sb];
        const float d = float(w_block.d);
        const float dmin = float(w_block.dmin);

        for (uint g = 0; g < 16u; ++g) {
            const uint half_group = g >> 3;
            const uint j = g & 0x7u;
            const uint base = j * 32u + half_group * 16u;
            if (base >= block_len) {
                continue;
            }

            const uchar sc = w_block.scales[g];
            const float dl = d * float(sc & 0x0Fu);
            const float ml = dmin * float(sc >> 4);

            const ushort low_plane = ushort(w_block.qs[2u * g]) | (ushort(w_block.qs[2u * g + 1u]) << 8u);
            const ushort high_plane = ushort(w_block.qs[32u + 2u * g]) | (ushort(w_block.qs[32u + 2u * g + 1u]) << 8u);
            const uint limit = min(16u, block_len - base);

            for (uint i = 0; i < limit; ++i) {
                const uint local_idx = base + i;
                const uint global_idx = block_start + local_idx;
                const uint q = ((low_plane >> i) & 0x1u) | (((high_plane >> i) & 0x1u) << 1u);
                const float a_val = float(input_row[global_idx * stride_k]);
                sum += (dl * float(q) - ml) * a_val;
            }
        }
    }

    output[n * M + m] = OutputType(sum);
}
