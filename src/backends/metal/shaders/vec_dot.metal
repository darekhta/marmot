#include <metal_stdlib>
using namespace metal;

constant uint kThreadsPerBlock = 32u;

#include "common/quant_blocks.h"
#include "common/quant_decode.h"

template <typename FloatPtr>
static inline void decode_q2_k_block(const device q2_k_block &blk, FloatPtr dst) {
    const float d = float(blk.d);
    const float dmin = float(blk.dmin);

    for (uint g = 0; g < 16u; ++g) {
        const uchar sc = blk.scales[g];
        const float dl = d * float(sc & 0x0Fu);
        const float ml = dmin * float(sc >> 4);

        const ushort low_plane = ushort(blk.qs[2u * g]) | (ushort(blk.qs[2u * g + 1u]) << 8u);
        const ushort high_plane = ushort(blk.qs[32u + 2u * g]) | (ushort(blk.qs[32u + 2u * g + 1u]) << 8u);

        const uint half_group = g >> 3;
        const uint j = g & 0x7u;
        const uint base = j * 32u + half_group * 16u;

        for (uint i = 0; i < 16u; ++i) {
            uint idx = base + i;
            uint q = ((low_plane >> i) & 0x1u) | (((high_plane >> i) & 0x1u) << 1u);
            dst[idx] = dl * float(q) - ml;
        }
    }
}

template <typename FloatPtr>
static inline void decode_q3_k_block(const device q3_k_block &blk, FloatPtr dst) {
    const float d_all = float(blk.d);

    uint aux[4] = {0u, 0u, 0u, 0u};
    for (uint i = 0; i < 12u; ++i) {
        uint idx = i >> 2;
        uint shift = (i & 3u) * 8u;
        aux[idx] |= uint(blk.scales[i]) << shift;
    }

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    int scales[16];
    for (uint i = 0; i < 16u; ++i) {
        uint word = aux[i >> 2];
        uint shift = (i & 3u) * 8u;
        scales[i] = int((char)((word >> shift) & 0xFFu));
    }

    const device uchar *q = blk.qs;
    const device uchar *hm = blk.hmask;
    uint scale_index = 0u;
    uchar m = 1u;

    for (uint n = 0; n < kQK_K; n += 128u) {
        uint shift = 0u;
        for (uint j = 0; j < 4u; ++j) {
            float dl = d_all * float(scales[scale_index++] - 32);
            for (uint l = 0; l < 16u; ++l) {
                uint idx = n + j * 32u + l;
                uchar q_byte = q[l];
                int q_val = int((q_byte >> shift) & 0x3u);
                uchar h_byte = hm[l];
                if ((h_byte & m) == 0u) {
                    q_val -= 4;
                }
                dst[idx] = dl * float(q_val);
            }

            dl = d_all * float(scales[scale_index++] - 32);
            for (uint l = 0; l < 16u; ++l) {
                uint idx = n + j * 32u + 16u + l;
                uchar q_byte = q[l + 16u];
                int q_val = int((q_byte >> shift) & 0x3u);
                uchar h_byte = hm[l + 16u];
                if ((h_byte & m) == 0u) {
                    q_val -= 4;
                }
                dst[idx] = dl * float(q_val);
            }

            shift += 2u;
            m <<= 1u;
        }
        q += 32;
        hm += 32;
    }
}

template <typename FloatPtr>
static inline void decode_q4_k_block(const device q4_k_block &blk, FloatPtr dst) {
    const float d = float(blk.d);
    const float dmin = float(blk.dmin);
    const device uchar *q = blk.qs;

    uint scale_index = 0u;
    for (uint base = 0; base < kQK_K; base += 64u) {
        uchar sc0 = 0u, m0 = 0u;
        get_scale_min_k4(scale_index++, blk.scales, sc0, m0);
        uchar sc1 = 0u, m1 = 0u;
        get_scale_min_k4(scale_index++, blk.scales, sc1, m1);

        const float d1 = d * float(sc0);
        const float m1f = dmin * float(m0);
        const float d2 = d * float(sc1);
        const float m2f = dmin * float(m1);

        for (uint l = 0; l < 32u; ++l) {
            dst[base + l] = d1 * float(q[l] & 0x0Fu) - m1f;
            dst[base + 32u + l] = d2 * float(q[l] >> 4) - m2f;
        }

        q += 32;
    }
}

template <typename FloatPtr>
static inline void decode_q5_k_block(const device q5_k_block &blk, FloatPtr dst) {
    const float d = float(blk.d);
    const float dmin = float(blk.dmin);
    const device uchar *ql = blk.qs;
    const device uchar *qh = blk.qh;

    uint scale_index = 0u;
    uchar u1 = 1u;
    uchar u2 = 2u;

    for (uint base = 0; base < kQK_K; base += 64u) {
        uchar sc0 = 0u, m0 = 0u;
        get_scale_min_k4(scale_index++, blk.scales, sc0, m0);
        uchar sc1 = 0u, m1 = 0u;
        get_scale_min_k4(scale_index++, blk.scales, sc1, m1);

        const float d1 = d * float(sc0);
        const float m1f = dmin * float(m0);
        const float d2 = d * float(sc1);
        const float m2f = dmin * float(m1);

        for (uint l = 0; l < 32u; ++l) {
            int qv0 = int(ql[l] & 0x0Fu);
            if ((qh[l] & u1) != 0u) {
                qv0 += 16;
            }
            int qv1 = int(ql[l] >> 4);
            if ((qh[l] & u2) != 0u) {
                qv1 += 16;
            }

            dst[base + l] = d1 * float(qv0) - m1f;
            dst[base + 32u + l] = d2 * float(qv1) - m2f;
        }

        ql += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
}

template <typename FloatPtr>
static inline void decode_q6_k_block(const device q6_k_block &blk, FloatPtr dst) {
    const float d = float(blk.d);
    const device uchar *ql = blk.ql;
    const device uchar *qh = blk.qh;
    const device char *sc = blk.scales;

    for (uint base = 0; base < kQK_K; base += 128u) {
        for (uint l = 0; l < 32u; ++l) {
            int is = int(l >> 4);

            int q1 = int((ql[l + 0] & 0x0Fu) | (((qh[l] >> 0) & 0x3u) << 4)) - 32;
            int q2 = int((ql[l + 32] & 0x0Fu) | (((qh[l] >> 2) & 0x3u) << 4)) - 32;
            int q3 = int((ql[l + 0] >> 4) | (((qh[l] >> 4) & 0x3u) << 4)) - 32;
            int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x3u) << 4)) - 32;

            dst[base + l + 0u] = d * float(sc[is + 0]) * float(q1);
            dst[base + l + 32u] = d * float(sc[is + 2]) * float(q2);
            dst[base + l + 64u] = d * float(sc[is + 4]) * float(q3);
            dst[base + l + 96u] = d * float(sc[is + 6]) * float(q4);
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

template <typename FloatPtr>
static inline void decode_q8_k_block(const device q8_k_block &blk, FloatPtr dst) {
    const float d = blk.d;
    for (uint i = 0; i < kQK_K; ++i) {
        dst[i] = d * float(blk.qs[i]);
    }
}

template <typename FloatPtr>
static inline void decode_block_dispatch(const device q2_k_block &blk, FloatPtr dst) {
    decode_q2_k_block(blk, dst);
}

template <typename FloatPtr>
static inline void decode_block_dispatch(const device q3_k_block &blk, FloatPtr dst) {
    decode_q3_k_block(blk, dst);
}

template <typename FloatPtr>
static inline void decode_block_dispatch(const device q4_k_block &blk, FloatPtr dst) {
    decode_q4_k_block(blk, dst);
}

template <typename FloatPtr>
static inline void decode_block_dispatch(const device q5_k_block &blk, FloatPtr dst) {
    decode_q5_k_block(blk, dst);
}

template <typename FloatPtr>
static inline void decode_block_dispatch(const device q6_k_block &blk, FloatPtr dst) {
    decode_q6_k_block(blk, dst);
}

template <typename FloatPtr>
static inline void decode_block_dispatch(const device q8_k_block &blk, FloatPtr dst) {
    decode_q8_k_block(blk, dst);
}

static inline float q8_block_value(const device q8_0_block &blk, uint offset) {
    int val = int(blk.qs[offset]);
    if (val >= 128) {
        val -= 256;
    }
    return float(blk.scale) * float(val);
}

kernel void vec_dot_q4_0_q8_0(
    device const q4_0_block *weights [[buffer(0)]], device const q8_0_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    const uint lane = tid % kThreadsPerBlock;

    threadgroup int shared_sums[kThreadsPerBlock];

    int local_sum = 0;
    if (lane < kQ4PackedBytes) {
        const q4_0_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const uchar packed = w_block.qs[lane];
        const uint block_half = kQuantBlockSize / 2u;
        const int w0 = int(packed & 0x0fu) - 8;
        const int w1 = int(packed >> 4) - 8;
        const int a0 = int(a_block.qs[lane]);
        const int a1 = int(a_block.qs[lane + block_half]);
        local_sum = w0 * a0 + w1 * a1;
    }

    shared_sums[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreadsPerBlock / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            shared_sums[lane] += shared_sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        const q4_0_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const float block_scale = float(w_block.scale) * float(a_block.scale);
        partials[block] = float(shared_sums[0]) * block_scale;
    }
}

kernel void vec_dot_q4_1_q8_0(
    device const q4_1_block *weights [[buffer(0)]], device const q8_0_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    const uint lane = tid % kThreadsPerBlock;

    threadgroup int shared_q[kThreadsPerBlock];
    threadgroup int shared_a[kThreadsPerBlock];

    int local_q = 0;
    int local_a = 0;
    if (lane < kQ4PackedBytes) {
        const q4_1_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const uchar packed = w_block.qs[lane];
        const uint block_half = kQuantBlockSize / 2u;
        const int qw0 = int(packed & 0x0fu);
        const int qw1 = int(packed >> 4);
        const int a0 = int(a_block.qs[lane]);
        const int a1 = int(a_block.qs[lane + block_half]);
        local_q = qw0 * a0 + qw1 * a1;
        local_a = a0 + a1;
    }

    shared_q[lane] = local_q;
    shared_a[lane] = local_a;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreadsPerBlock / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            shared_q[lane] += shared_q[lane + stride];
            shared_a[lane] += shared_a[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        const q4_1_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const float scale_w = float(w_block.scale);
        const float min_w = float(w_block.min);
        const float scale_a = float(a_block.scale);
        const float block_scale = scale_w * scale_a;
        const float bias_scale = min_w * scale_a;
        partials[block] = block_scale * float(shared_q[0]) + bias_scale * float(shared_a[0]);
    }
}

kernel void vec_dot_q5_0_q8_0(
    device const q5_0_block *weights [[buffer(0)]], device const q8_0_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    const uint lane = tid % kThreadsPerBlock;

    threadgroup int shared_sums[kThreadsPerBlock];

    int local_sum = 0;
    if (lane < kQ5PackedBytes) {
        const q5_0_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const uint qh = (uint)w_block.qh[0] | ((uint)w_block.qh[1] << 8u) | ((uint)w_block.qh[2] << 16u) |
            ((uint)w_block.qh[3] << 24u);
        const uint block_half = kQuantBlockSize / 2u;

        const uchar packed = w_block.qs[lane];
        uint lo = uint(packed & 0x0fu);
        uint hi = uint(packed >> 4);
        lo |= ((qh >> lane) & 0x1u) << 4;
        hi |= ((qh >> (lane + block_half)) & 0x1u) << 4;

        const int w0 = int(lo) - 16;
        const int w1 = int(hi) - 16;
        const int a0 = int(a_block.qs[lane]);
        const int a1 = int(a_block.qs[lane + block_half]);
        local_sum = w0 * a0 + w1 * a1;
    }

    shared_sums[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreadsPerBlock / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            shared_sums[lane] += shared_sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        const q5_0_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const float block_scale = float(w_block.scale) * float(a_block.scale);
        partials[block] = float(shared_sums[0]) * block_scale;
    }
}

kernel void vec_dot_q5_1_q8_0(
    device const q5_1_block *weights [[buffer(0)]], device const q8_0_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    const uint lane = tid % kThreadsPerBlock;

    threadgroup int shared_q[kThreadsPerBlock];
    threadgroup int shared_a[kThreadsPerBlock];

    int local_q = 0;
    int local_a = 0;
    if (lane < kQ5PackedBytes) {
        const q5_1_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const uint qh = (uint)w_block.qh[0] | ((uint)w_block.qh[1] << 8u) | ((uint)w_block.qh[2] << 16u) |
            ((uint)w_block.qh[3] << 24u);
        const uint block_half = kQuantBlockSize / 2u;

        const uchar packed = w_block.qs[lane];
        uint lo = uint(packed & 0x0fu);
        uint hi = uint(packed >> 4);
        lo |= ((qh >> lane) & 0x1u) << 4;
        hi |= ((qh >> (lane + block_half)) & 0x1u) << 4;

        const int qw0 = int(lo);
        const int qw1 = int(hi);
        const int a0 = int(a_block.qs[lane]);
        const int a1 = int(a_block.qs[lane + block_half]);

        local_q = qw0 * a0 + qw1 * a1;
        local_a = a0 + a1;
    }

    shared_q[lane] = local_q;
    shared_a[lane] = local_a;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreadsPerBlock / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            shared_q[lane] += shared_q[lane + stride];
            shared_a[lane] += shared_a[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        const q5_1_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const float scale_w = float(w_block.scale);
        const float min_w = float(w_block.min);
        const float scale_a = float(a_block.scale);
        const float block_scale = scale_w * scale_a;
        const float bias_scale = min_w * scale_a;
        partials[block] = block_scale * float(shared_q[0]) + bias_scale * float(shared_a[0]);
    }
}

kernel void vec_dot_q8_0_q8_0(
    device const q8_0_block *weights [[buffer(0)]], device const q8_0_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    const uint lane = tid % kThreadsPerBlock;

    threadgroup int shared_sums[kThreadsPerBlock];

    int local_sum = 0;
    if (lane < kQuantBlockSize) {
        const q8_0_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const int w = int(w_block.qs[lane]);
        const int a = int(a_block.qs[lane]);
        local_sum = w * a;
    }

    shared_sums[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreadsPerBlock / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            shared_sums[lane] += shared_sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        const q8_0_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const float block_scale = float(w_block.scale) * float(a_block.scale);
        partials[block] = float(shared_sums[0]) * block_scale;
    }
}

kernel void vec_dot_q8_1_q8_0(
    device const q8_1_block *weights [[buffer(0)]], device const q8_0_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    const uint lane = tid % kThreadsPerBlock;

    threadgroup int shared_sums[kThreadsPerBlock];

    int local_sum = 0;
    if (lane < kQuantBlockSize) {
        const q8_1_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const int w = int(w_block.qs[lane]);
        const int a = int(a_block.qs[lane]);
        local_sum = w * a;
    }

    shared_sums[lane] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreadsPerBlock / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            shared_sums[lane] += shared_sums[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        const q8_1_block w_block = weights[block];
        const q8_0_block a_block = activations[block];
        const float block_scale = float(w_block.scale) * float(a_block.scale);
        partials[block] = float(shared_sums[0]) * block_scale;
    }
}

// -----------------------------------------------------------------------------
// K-quant vec dot kernels (Q2_K .. Q8_K)
// -----------------------------------------------------------------------------

template <typename BlockType>
static inline void vec_dot_kquant_impl(
    device const BlockType *weights, device const q8_0_block *activations, device float *partials, uint num_blocks,
    uint tid, threadgroup float *weights_local, threadgroup float *shared_sum
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    const uint lane = tid % kThreadsPerBlock;

    if (lane == 0u) {
        decode_block_dispatch(weights[block], weights_local);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint blocks_per_super = kQK_K / kQuantBlockSize;
    const device q8_0_block *a_blocks = activations + block * blocks_per_super;

    float partial = 0.0f;
    for (uint idx = lane; idx < kQK_K; idx += kThreadsPerBlock) {
        const device q8_0_block &a_block = a_blocks[idx / kQuantBlockSize];
        float a_val = q8_block_value(a_block, idx % kQuantBlockSize);
        partial += weights_local[idx] * a_val;
    }

    shared_sum[lane] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = kThreadsPerBlock / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            shared_sum[lane] += shared_sum[lane + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0u) {
        partials[block] = shared_sum[0];
    }
}

kernel void vec_dot_q2_k_q8_k(
    device const q2_k_block *weights [[buffer(0)]], device const q8_k_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    if ((tid % kThreadsPerBlock) != 0u) {
        return;
    }

    const device q2_k_block *w_block = weights + block;
    const device q8_k_block *a_block = activations + block;

    const device uchar *scales = &(w_block->scales[0]);
    const device uchar *q2 = &(w_block->qs[0]);
    const device char *q8 = &(a_block->qs[0]);
    const device short *bsums = &(a_block->bsums[0]);

    int summs = 0;
    for (uint group = 0u; group < kQK_K / 16u; ++group) {
        const int scale_min = int(scales[group] >> 4);
        summs += int(bsums[group]) * scale_min;
    }

    const float dall = a_block->d * float(w_block->d);
    const float dmin = a_block->d * float(w_block->dmin);

    int isum = 0;
    uint scale_index = 0u;
    const uint chunks = kQK_K / 128u; // 2 chunks per block
    for (uint chunk = 0u; chunk < chunks; ++chunk) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            int scale_lo = int(scales[scale_index++] & 0x0Fu);
            int partial_lo = 0;
            for (int l = 0; l < 16; ++l) {
                const int q_val = (int(q2[l]) >> shift) & 0x3;
                partial_lo += int(q8[l]) * q_val;
            }
            isum += scale_lo * partial_lo;

            int scale_hi = int(scales[scale_index++] & 0x0Fu);
            int partial_hi = 0;
            for (int l = 16; l < 32; ++l) {
                const int q_val = (int(q2[l]) >> shift) & 0x3;
                partial_hi += int(q8[l]) * q_val;
            }
            isum += scale_hi * partial_hi;

            shift += 2;
            q8 += 32;
        }
        q2 += 32;
    }

    partials[block] = dall * float(isum) - dmin * float(summs);
}

static inline void load_scales_words(const device uchar *src, thread uint32_t *dst) {
    for (uint i = 0; i < 4u; ++i) {
        dst[i] = 0u;
    }
    for (uint i = 0; i < 12u; ++i) {
        const uint idx = i / 4u;
        const uint shift = (i % 4u) * 8u;
        dst[idx] |= (uint32_t)src[i] << shift;
    }
}

kernel void vec_dot_q3_k_q8_k(
    device const q3_k_block *weights [[buffer(0)]], device const q8_k_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    if ((tid % kThreadsPerBlock) != 0u) {
        return;
    }

    const device q3_k_block &w_block = weights[block];
    const device q8_k_block &a_block = activations[block];

    thread int8_t unpacked[kQK_K];
    thread int16_t prod8[8];
    thread int32_t accum8[8];
    for (uint i = 0; i < 8u; ++i) {
        accum8[i] = 0;
    }

    const device uchar *q3 = w_block.qs;
    const device uchar *hm = w_block.hmask;
    thread int8_t *dst = unpacked;
    uint8_t mask = 1u;
    for (uint chunk = 0; chunk < kQK_K; chunk += 128u) {
        for (uint l = 0; l < 32u; ++l) {
            int8_t val = int8_t(q3[l] & 0x3u);
            if ((hm[l] & mask) == 0u) {
                val -= 4;
            }
            dst[l] = val;
        }
        dst += 32;
        mask <<= 1;
        for (uint l = 0; l < 32u; ++l) {
            int8_t val = int8_t((q3[l] >> 2) & 0x3u);
            if ((hm[l] & mask) == 0u) {
                val -= 4;
            }
            dst[l] = val;
        }
        dst += 32;
        mask <<= 1;
        for (uint l = 0; l < 32u; ++l) {
            int8_t val = int8_t((q3[l] >> 4) & 0x3u);
            if ((hm[l] & mask) == 0u) {
                val -= 4;
            }
            dst[l] = val;
        }
        dst += 32;
        mask <<= 1;
        for (uint l = 0; l < 32u; ++l) {
            int8_t val = int8_t((q3[l] >> 6) & 0x3u);
            if ((hm[l] & mask) == 0u) {
                val -= 4;
            }
            dst[l] = val;
        }
        dst += 32;
        mask <<= 1;
        q3 += 32;
    }
    dst = unpacked;

    thread uint32_t packed[4];
    load_scales_words(w_block.scales, packed);
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t tmp = packed[2];
    packed[2] = ((packed[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    packed[3] = ((packed[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    packed[0] = (packed[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    packed[1] = (packed[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const thread int8_t *scales = (const thread int8_t *)packed;

    const device char *q8 = a_block.qs;
    for (uint group = 0; group < kQK_K / 16u; ++group) {
        for (uint l = 0; l < 8u; ++l) {
            prod8[l] = int16_t(int(q8[l]) * int(dst[l]));
        }
        const int32_t scale = int32_t(scales[group]) - 32;
        for (uint l = 0; l < 8u; ++l) {
            accum8[l] += scale * int32_t(prod8[l]);
        }
        q8 += 8;
        dst += 8;
        for (uint l = 0; l < 8u; ++l) {
            prod8[l] = int16_t(int(q8[l]) * int(dst[l]));
        }
        for (uint l = 0; l < 8u; ++l) {
            accum8[l] += scale * int32_t(prod8[l]);
        }
        q8 += 8;
        dst += 8;
    }

    const float d = a_block.d * float(w_block.d);
    float total = 0.0f;
    for (uint l = 0; l < 8u; ++l) {
        total += d * float(accum8[l]);
    }
    partials[block] = total;
}

kernel void vec_dot_q4_k_q8_k(
    device const q4_k_block *weights [[buffer(0)]], device const q8_k_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    if ((tid % kThreadsPerBlock) != 0u) {
        return;
    }

    const device q4_k_block &w_block = weights[block];
    const device q8_k_block &a_block = activations[block];

    thread int8_t unpacked[kQK_K];
    thread int16_t prod8[8];
    thread int32_t accum8[8];
    for (uint i = 0; i < 8u; ++i) {
        accum8[i] = 0;
    }

    const device uchar *q4 = w_block.qs;
    thread int8_t *dst = unpacked;
    for (uint chunk = 0; chunk < kQK_K; chunk += 64u) {
        for (uint l = 0; l < 32u; ++l) {
            dst[l] = int8_t(q4[l] & 0x0Fu);
        }
        dst += 32;
        for (uint l = 0; l < 32u; ++l) {
            dst[l] = int8_t(q4[l] >> 4);
        }
        dst += 32;
        q4 += 32;
    }
    dst = unpacked;

    thread uint32_t packed[4];
    load_scales_words(w_block.scales, packed);
    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask3 = 0x03030303u;
    packed[3] = ((packed[2] >> 4) & kmask2) | (((packed[1] >> 6) & kmask3) << 4);
    const uint32_t tmp = packed[1] & kmask1;
    packed[1] = (packed[2] & kmask2) | (((packed[0] >> 6) & kmask3) << 4);
    packed[2] = tmp;
    packed[0] &= kmask1;
    const thread uint8_t *scales = (const thread uint8_t *)&packed[0];
    const thread uint8_t *mins = (const thread uint8_t *)&packed[2];

    int sumi = 0;
    for (uint j = 0; j < kQK_K / 16u; ++j) {
        sumi += int(a_block.bsums[j]) * int(mins[j / 2u]);
    }

    const device char *q8 = a_block.qs;
    int scale_index = 0;
    for (uint block32 = 0; block32 < kQK_K / 32u; ++block32) {
        const int32_t scale = int32_t(scales[scale_index++]);
        for (uint repeat = 0; repeat < 4u; ++repeat) {
            for (uint l = 0; l < 8u; ++l) {
                prod8[l] = int16_t(int(q8[l]) * int(dst[l]));
            }
            for (uint l = 0; l < 8u; ++l) {
                accum8[l] += scale * int32_t(prod8[l]);
            }
            q8 += 8;
            dst += 8;
        }
    }

    const float d = a_block.d * float(w_block.d);
    const float dmin = a_block.d * float(w_block.dmin);
    float total = -dmin * float(sumi);
    for (uint l = 0; l < 8u; ++l) {
        total += d * float(accum8[l]);
    }
    partials[block] = total;
}

kernel void vec_dot_q5_k_q8_k(
    device const q5_k_block *weights [[buffer(0)]], device const q8_k_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    if ((tid % kThreadsPerBlock) != 0u) {
        return;
    }

    const device q5_k_block &w_block = weights[block];
    const device q8_k_block &a_block = activations[block];

    thread int8_t unpacked[kQK_K];
    thread int16_t prod8[8];
    thread int32_t accum8[8];
    for (uint i = 0; i < 8u; ++i) {
        accum8[i] = 0;
    }

    const device uchar *q4 = w_block.qs;
    const device uchar *high = w_block.qh;
    thread int8_t *dst = unpacked;
    uint8_t mask = 1u;
    for (uint chunk = 0; chunk < kQK_K; chunk += 64u) {
        for (uint l = 0; l < 32u; ++l) {
            int8_t val = int8_t(q4[l] & 0x0Fu);
            if ((high[l] & mask) != 0u) {
                val += 16;
            }
            dst[l] = val;
        }
        dst += 32;
        mask <<= 1;
        for (uint l = 0; l < 32u; ++l) {
            int8_t val = int8_t(q4[l] >> 4);
            if ((high[l] & mask) != 0u) {
                val += 16;
            }
            dst[l] = val;
        }
        dst += 32;
        mask <<= 1;
        q4 += 32;
    }
    dst = unpacked;

    thread uint32_t packed[4];
    load_scales_words(w_block.scales, packed);
    const uint32_t kmask1 = 0x3f3f3f3fu;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const uint32_t kmask3 = 0x03030303u;
    packed[3] = ((packed[2] >> 4) & kmask2) | (((packed[1] >> 6) & kmask3) << 4);
    const uint32_t tmp = packed[1] & kmask1;
    packed[1] = (packed[2] & kmask2) | (((packed[0] >> 6) & kmask3) << 4);
    packed[2] = tmp;
    packed[0] &= kmask1;
    const thread uint8_t *scales = (const thread uint8_t *)&packed[0];
    const thread uint8_t *mins = (const thread uint8_t *)&packed[2];

    int sumi = 0;
    for (uint j = 0; j < kQK_K / 16u; ++j) {
        sumi += int(a_block.bsums[j]) * int(mins[j / 2u]);
    }

    const device char *q8 = a_block.qs;
    int scale_index = 0;
    for (uint block32 = 0; block32 < kQK_K / 32u; ++block32) {
        const int32_t scale = int32_t(scales[scale_index++]);
        for (uint repeat = 0; repeat < 4u; ++repeat) {
            for (uint l = 0; l < 8u; ++l) {
                prod8[l] = int16_t(int(q8[l]) * int(dst[l]));
            }
            for (uint l = 0; l < 8u; ++l) {
                accum8[l] += scale * int32_t(prod8[l]);
            }
            q8 += 8;
            dst += 8;
        }
    }

    const float d = a_block.d * float(w_block.d);
    const float dmin = a_block.d * float(w_block.dmin);
    float total = -dmin * float(sumi);
    for (uint l = 0; l < 8u; ++l) {
        total += d * float(accum8[l]);
    }
    partials[block] = total;
}

kernel void vec_dot_q6_k_q8_k(
    device const q6_k_block *weights [[buffer(0)]], device const q8_k_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    if ((tid % kThreadsPerBlock) != 0u) {
        return;
    }

    const device q6_k_block &w_block = weights[block];
    const device q8_k_block &a_block = activations[block];

    thread int8_t unpacked[kQK_K];
    thread int16_t prod8[8];
    thread int32_t accum8[8];
    for (uint i = 0; i < 8u; ++i) {
        accum8[i] = 0;
    }

    const device uchar *ql = w_block.ql;
    const device uchar *qh = w_block.qh;
    thread int8_t *dst = unpacked;
    for (uint chunk = 0; chunk < kQK_K; chunk += 128u) {
        for (uint l = 0; l < 32u; ++l) {
            dst[l + 0u] = int8_t((ql[l + 0u] & 0x0Fu) | (((qh[l] >> 0) & 0x3u) << 4)) - 32;
            dst[l + 32u] = int8_t((ql[l + 32u] & 0x0Fu) | (((qh[l] >> 2) & 0x3u) << 4)) - 32;
            dst[l + 64u] = int8_t((ql[l + 0u] >> 4) | (((qh[l] >> 4) & 0x3u) << 4)) - 32;
            dst[l + 96u] = int8_t((ql[l + 32u] >> 4) | (((qh[l] >> 6) & 0x3u) << 4)) - 32;
        }
        dst += 128;
        ql += 64;
        qh += 32;
    }
    dst = unpacked;

    const device char *q8 = a_block.qs;
    const device char *sc = w_block.scales;
    for (uint group = 0; group < kQK_K / 16u; ++group) {
        const int scale = int(sc[group]);
        for (uint l = 0; l < 8u; ++l) {
            prod8[l] = int16_t(int(q8[l]) * int(dst[l]));
        }
        for (uint l = 0; l < 8u; ++l) {
            accum8[l] += scale * int32_t(prod8[l]);
        }
        q8 += 8;
        dst += 8;
        for (uint l = 0; l < 8u; ++l) {
            prod8[l] = int16_t(int(q8[l]) * int(dst[l]));
        }
        for (uint l = 0; l < 8u; ++l) {
            accum8[l] += scale * int32_t(prod8[l]);
        }
        q8 += 8;
        dst += 8;
    }

    const float d = a_block.d * float(w_block.d);
    float total = 0.0f;
    for (uint l = 0; l < 8u; ++l) {
        total += d * float(accum8[l]);
    }
    partials[block] = total;
}

kernel void vec_dot_q8_k_q8_k(
    device const q8_k_block *weights [[buffer(0)]], device const q8_k_block *activations [[buffer(1)]],
    device float *partials [[buffer(2)]], constant uint &num_blocks [[buffer(3)]], uint tid [[thread_position_in_grid]]
) {
    const uint block = tid / kThreadsPerBlock;
    if (block >= num_blocks) {
        return;
    }
    if ((tid % kThreadsPerBlock) != 0u) {
        return;
    }

    const device q8_k_block &w_block = weights[block];
    const device q8_k_block &a_block = activations[block];

    const float scale = w_block.d * a_block.d;
    const device char *qw = w_block.qs;
    const device char *qa = a_block.qs;

    int32_t sum = 0;
    for (uint i = 0; i < kQK_K; ++i) {
        sum += int32_t(qw[i]) * int32_t(qa[i]);
    }
    partials[block] = scale * float(sum);
}
