#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

#include "common/quant_blocks.h"
#include "common/quant_decode.h"

// Floating kernels: read source type, write destination type

#define DEF_EMB_FLOAT(NAME, WT, READ_FN, OT, WRITE_FN)                                                                 \
    kernel void NAME(                                                                                                  \
        device const WT *weights [[buffer(0)]], constant int32_t *ids [[buffer(1)]], device OT *output [[buffer(2)]],  \
        constant uint &dim [[buffer(3)]], constant uint &rows [[buffer(4)]], constant float &scale [[buffer(5)]],      \
        uint gid [[thread_position_in_grid]]                                                                           \
    ) {                                                                                                                \
        uint elements = dim * rows;                                                                                    \
        if (gid >= elements)                                                                                           \
            return;                                                                                                    \
        uint row = gid / dim;                                                                                          \
        uint col = gid - row * dim;                                                                                    \
        int id = ids[row];                                                                                             \
        if (id < 0) {                                                                                                  \
            output[gid] = WRITE_FN(0.0f);                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        uint widx = uint(id) * dim + col;                                                                              \
        float v = READ_FN(weights[widx]);                                                                              \
        output[gid] = WRITE_FN(v * scale);                                                                             \
    }

DEF_EMB_FLOAT(embedding_gather_f32_to_f32, float, read_float, float, write_float)
DEF_EMB_FLOAT(embedding_gather_f32_to_f16, float, read_float, half, write_half)
DEF_EMB_FLOAT(embedding_gather_f32_to_bf16, float, read_float, ushort, write_bf16)

DEF_EMB_FLOAT(embedding_gather_f16_to_f32, half, read_half, float, write_float)
DEF_EMB_FLOAT(embedding_gather_f16_to_f16, half, read_half, half, write_half)
DEF_EMB_FLOAT(embedding_gather_f16_to_bf16, half, read_half, ushort, write_bf16)

DEF_EMB_FLOAT(embedding_gather_bf16_to_f32, ushort, read_bf16, float, write_float)
DEF_EMB_FLOAT(embedding_gather_bf16_to_f16, ushort, read_bf16, half, write_half)
DEF_EMB_FLOAT(embedding_gather_bf16_to_bf16, ushort, read_bf16, ushort, write_bf16)

#undef DEF_EMB_FLOAT

// Quantized decoders

static inline float decode_q4_0(const device q4_0_block &blk, uint off) {
    uchar packed = blk.qs[off >> 1];
    int q = (off & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    q -= 8;
    return float(blk.scale) * float(q);
}

static inline float decode_q4_1(const device q4_1_block &blk, uint off) {
    uchar packed = blk.qs[off >> 1];
    int q = (off & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    return float(blk.scale) * float(q) + float(blk.min);
}

static inline float decode_q8_0(const device q8_0_block &blk, uint off) {
    return float(blk.scale) * float(blk.qs[off]);
}

static inline float decode_q5_0(const device q5_0_block &blk, uint off) {
    uint bits = (uint)blk.qh[0] | ((uint)blk.qh[1] << 8u) | ((uint)blk.qh[2] << 16u) | ((uint)blk.qh[3] << 24u);
    uint base = (off < 16u) ? off : (off - 16u);
    uchar packed = blk.qs[base];
    uint q4 = (off < 16u) ? (packed & 0x0fu) : (packed >> 4);
    uint bit_index = (off < 16u) ? base : (base + 16u);
    uint hi = (bits >> bit_index) & 0x1u;
    int q = (int)((q4 | (hi << 4)) - 16u);
    return float(blk.scale) * float(q);
}

static inline float decode_q5_1(const device q5_1_block &blk, uint off) {
    uint bits = (uint)blk.qh[0] | ((uint)blk.qh[1] << 8u) | ((uint)blk.qh[2] << 16u) | ((uint)blk.qh[3] << 24u);
    uint base = (off < 16u) ? off : (off - 16u);
    uchar packed = blk.qs[base];
    uint q4 = (off < 16u) ? (packed & 0x0fu) : (packed >> 4);
    uint bit_index = (off < 16u) ? base : (base + 16u);
    uint hi = (bits >> bit_index) & 0x1u;
    uint q5 = q4 | (hi << 4);
    return float(blk.scale) * float(q5) + float(blk.min);
}

// K-quant decoders (256 values per block)
static inline float decode_q2_k(const device q2_k_block &blk, uint off) {
    const float d = float(blk.d);
    const float dmin = float(blk.dmin);

    const uint j = off >> 5;
    const uint half_idx = (off >> 4) & 1u;
    const uint i = off & 15u;
    const uint g = j + (half_idx << 3);

    const uchar sc = blk.scales[g];
    const float dl = d * float(sc & 0x0Fu);
    const float ml = dmin * float(sc >> 4);

    const ushort low_plane = ushort(blk.qs[2u * g]) | (ushort(blk.qs[2u * g + 1u]) << 8u);
    const ushort high_plane = ushort(blk.qs[32u + 2u * g]) | (ushort(blk.qs[32u + 2u * g + 1u]) << 8u);
    const uint q = ((uint(low_plane) >> i) & 0x1u) | (((uint(high_plane) >> i) & 0x1u) << 1u);

    return dl * float(q) - ml;
}

static inline float decode_q3_k(const device q3_k_block &blk, uint off) {
    uint aux[4] = {0u, 0u, 0u, 0u};
    for (uint i = 0; i < kQK_K_ScaleBytes; ++i) {
        const uint idx = i >> 2;
        const uint shift = (i & 3u) * 8u;
        aux[idx] |= uint(blk.scales[i]) << shift;
    }

    const uint kmask1 = 0x03030303u;
    const uint kmask2 = 0x0f0f0f0fu;
    const uint tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    const uint half_idx = off >> 7;
    const uint r = off & 127u;
    const uint j = r >> 5;
    const uint within = r & 31u;
    const uint second = within >> 4;
    const uint l = within & 15u;
    const uint scale_idx = half_idx * 8u + j * 2u + second;

    const uint word = aux[scale_idx >> 2];
    const uint shift = (scale_idx & 3u) * 8u;
    const int scale = int((char)((word >> shift) & 0xFFu));
    const float dl = float(blk.d) * float(scale - 32);

    const device uchar *q = blk.qs + half_idx * 32u;
    const device uchar *hm = blk.hmask + half_idx * 32u;
    const uint q_shift = j * 2u;
    const uchar m = uchar(1u << j);

    int q_val = int((q[second * 16u + l] >> q_shift) & 0x3u);
    if ((hm[second * 16u + l] & m) == 0u) {
        q_val -= 4;
    }

    return dl * float(q_val);
}

static inline float decode_q4_k(const device q4_k_block &blk, uint off) {
    const uint il = off >> 4;
    const uint i = off & 15u;

    const uint is = (il >> 2) * 2u;
    const device uchar *q = blk.qs + (il >> 2) * 32u + (il & 1u) * 16u;

    const uint il_local = il & 3u;
    const uchar2 sc = get_scale_min_k4_just2(is, il_local >> 1, blk.scales);

    const float d = (il_local < 2u) ? float(blk.d) : float(blk.d) * (1.0f / 16.0f);
    const float dl = d * float(sc[0]);
    const float ml = float(blk.dmin) * float(sc[1]);

    const ushort mask = (il_local < 2u) ? 0x0Fu : 0xF0u;
    return dl * float(q[i] & mask) - ml;
}

static inline float decode_q5_k(const device q5_k_block &blk, uint off) {
    const uint il = off >> 4;
    const uint i = off & 15u;

    const uint is = (il >> 2) * 2u;
    const device uchar *q = blk.qs + (il >> 2) * 32u + (il & 1u) * 16u;
    const device uchar *qh = blk.qh + (il & 1u) * 16u;

    const uchar ul = uchar(1u << (il >> 1));
    const uint il_local = il & 3u;
    const uchar2 sc = get_scale_min_k4_just2(is, il_local >> 1, blk.scales);

    const float d = (il_local < 2u) ? float(blk.d) : float(blk.d) * (1.0f / 16.0f);
    const float dl = d * float(sc[0]);
    const float ml = float(blk.dmin) * float(sc[1]);

    const ushort mask = (il_local < 2u) ? 0x0Fu : 0xF0u;
    const float qh_val = (il_local < 2u) ? 16.0f : 256.0f;
    const float qh_term = ((qh[i] & ul) != 0u) ? qh_val : 0.0f;

    return dl * (float(q[i] & mask) + qh_term) - ml;
}

static inline float decode_q6_k(const device q6_k_block &blk, uint off) {
    const uint base = off >> 7;
    const uint within = off & 127u;
    const uint quadrant = within >> 5;
    const uint l = within & 31u;
    const uint is = l >> 4;

    const int scale = int(blk.scales[base * 8u + is + quadrant * 2u]);
    const float d_scale = float(blk.d) * float(scale);

    const uchar qh_byte = blk.qh[base * 32u + l];
    const uchar ql_byte = blk.ql[base * 64u + l + (((quadrant & 1u) != 0u) ? 32u : 0u)];

    const int ql_val = ((quadrant & 2u) != 0u) ? int(ql_byte >> 4) : int(ql_byte & 0x0Fu);
    const int qh_val = int((qh_byte >> (quadrant * 2u)) & 0x3u);
    const int q = (ql_val | (qh_val << 4)) - 32;

    return d_scale * float(q);
}

static inline float decode_q8_k(const device q8_k_block &blk, uint off) {
    return blk.d * float(blk.qs[off]);
}

#define DEF_EMB_QUANT(NAME, BT, DECODE_FN, OT, WRITE_FN)                                                               \
    kernel void NAME(                                                                                                  \
        device const BT *weights [[buffer(0)]], constant int32_t *ids [[buffer(1)]], device OT *output [[buffer(2)]],  \
        constant uint &dim [[buffer(3)]], constant uint &rows [[buffer(4)]], constant float &scale [[buffer(5)]],      \
        constant uint &blocks_per_row [[buffer(6)]], uint gid [[thread_position_in_grid]]                              \
    ) {                                                                                                                \
        uint elements = dim * rows;                                                                                    \
        if (gid >= elements)                                                                                           \
            return;                                                                                                    \
        uint row = gid / dim;                                                                                          \
        uint col = gid - row * dim;                                                                                    \
        int id = ids[row];                                                                                             \
        if (id < 0) {                                                                                                  \
            output[gid] = WRITE_FN(0.0f);                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        uint block = col / kQuantBlockSize;                                                                            \
        uint off = col % kQuantBlockSize;                                                                              \
        const device BT &blk = weights[uint(id) * blocks_per_row + block];                                             \
        float v = DECODE_FN(blk, off);                                                                                 \
        output[gid] = WRITE_FN(v * scale);                                                                             \
    }

#define DEF_EMB_QUANT_K(NAME, BT, DECODE_FN, OT, WRITE_FN)                                                             \
    kernel void NAME(                                                                                                  \
        device const BT *weights [[buffer(0)]], constant int32_t *ids [[buffer(1)]], device OT *output [[buffer(2)]],  \
        constant uint &dim [[buffer(3)]], constant uint &rows [[buffer(4)]], constant float &scale [[buffer(5)]],      \
        constant uint &blocks_per_row [[buffer(6)]], uint gid [[thread_position_in_grid]]                              \
    ) {                                                                                                                \
        uint elements = dim * rows;                                                                                    \
        if (gid >= elements)                                                                                           \
            return;                                                                                                    \
        uint row = gid / dim;                                                                                          \
        uint col = gid - row * dim;                                                                                    \
        int id = ids[row];                                                                                             \
        if (id < 0) {                                                                                                  \
            output[gid] = WRITE_FN(0.0f);                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        uint block = col / kQK_K;                                                                                      \
        uint off = col % kQK_K;                                                                                        \
        const device BT &blk = weights[uint(id) * blocks_per_row + block];                                             \
        float v = DECODE_FN(blk, off);                                                                                 \
        output[gid] = WRITE_FN(v * scale);                                                                             \
    }

DEF_EMB_QUANT(embedding_gather_q4_0_to_f32, q4_0_block, decode_q4_0, float, write_float)
DEF_EMB_QUANT(embedding_gather_q4_0_to_bf16, q4_0_block, decode_q4_0, ushort, write_bf16)

DEF_EMB_QUANT(embedding_gather_q4_1_to_f32, q4_1_block, decode_q4_1, float, write_float)
DEF_EMB_QUANT(embedding_gather_q4_1_to_bf16, q4_1_block, decode_q4_1, ushort, write_bf16)

DEF_EMB_QUANT(embedding_gather_q5_0_to_f32, q5_0_block, decode_q5_0, float, write_float)
DEF_EMB_QUANT(embedding_gather_q5_0_to_bf16, q5_0_block, decode_q5_0, ushort, write_bf16)

DEF_EMB_QUANT(embedding_gather_q5_1_to_f32, q5_1_block, decode_q5_1, float, write_float)
DEF_EMB_QUANT(embedding_gather_q5_1_to_bf16, q5_1_block, decode_q5_1, ushort, write_bf16)

DEF_EMB_QUANT(embedding_gather_q8_0_to_f32, q8_0_block, decode_q8_0, float, write_float)
DEF_EMB_QUANT(embedding_gather_q8_0_to_bf16, q8_0_block, decode_q8_0, ushort, write_bf16)

DEF_EMB_QUANT_K(embedding_gather_q2_k_to_f32, q2_k_block, decode_q2_k, float, write_float)
DEF_EMB_QUANT_K(embedding_gather_q2_k_to_f16, q2_k_block, decode_q2_k, half, write_half)
DEF_EMB_QUANT_K(embedding_gather_q2_k_to_bf16, q2_k_block, decode_q2_k, ushort, write_bf16)

DEF_EMB_QUANT_K(embedding_gather_q3_k_to_f32, q3_k_block, decode_q3_k, float, write_float)
DEF_EMB_QUANT_K(embedding_gather_q3_k_to_f16, q3_k_block, decode_q3_k, half, write_half)
DEF_EMB_QUANT_K(embedding_gather_q3_k_to_bf16, q3_k_block, decode_q3_k, ushort, write_bf16)

DEF_EMB_QUANT_K(embedding_gather_q4_k_to_f32, q4_k_block, decode_q4_k, float, write_float)
DEF_EMB_QUANT_K(embedding_gather_q4_k_to_f16, q4_k_block, decode_q4_k, half, write_half)
DEF_EMB_QUANT_K(embedding_gather_q4_k_to_bf16, q4_k_block, decode_q4_k, ushort, write_bf16)

DEF_EMB_QUANT_K(embedding_gather_q5_k_to_f32, q5_k_block, decode_q5_k, float, write_float)
DEF_EMB_QUANT_K(embedding_gather_q5_k_to_f16, q5_k_block, decode_q5_k, half, write_half)
DEF_EMB_QUANT_K(embedding_gather_q5_k_to_bf16, q5_k_block, decode_q5_k, ushort, write_bf16)

DEF_EMB_QUANT_K(embedding_gather_q6_k_to_f32, q6_k_block, decode_q6_k, float, write_float)
DEF_EMB_QUANT_K(embedding_gather_q6_k_to_f16, q6_k_block, decode_q6_k, half, write_half)
DEF_EMB_QUANT_K(embedding_gather_q6_k_to_bf16, q6_k_block, decode_q6_k, ushort, write_bf16)

DEF_EMB_QUANT_K(embedding_gather_q8_k_to_f32, q8_k_block, decode_q8_k, float, write_float)
DEF_EMB_QUANT_K(embedding_gather_q8_k_to_f16, q8_k_block, decode_q8_k, half, write_half)
DEF_EMB_QUANT_K(embedding_gather_q8_k_to_bf16, q8_k_block, decode_q8_k, ushort, write_bf16)

#undef DEF_EMB_QUANT
#undef DEF_EMB_QUANT_K

// Optimized quantized kernels: one threadgroup per (row, block), reuse headers via threadgroup memory

kernel void embedding_gather_q4_0_to_f16_opt(
    device const q4_0_block *weights [[buffer(0)]], constant int32_t *ids [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &dim [[buffer(3)]], constant uint &rows [[buffer(4)]],
    constant float &scale [[buffer(5)]], constant uint &blocks_per_row [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_idx [[threadgroup_position_in_grid]]
) {
    uint total_blocks = rows * blocks_per_row;
    if (tg_idx >= total_blocks)
        return;
    uint row = tg_idx / blocks_per_row;
    uint block = tg_idx - row * blocks_per_row;
    int id = ids[row];
    uint off = tid;
    if (off >= kQuantBlockSize)
        return;
    uint col = block * kQuantBlockSize + off;
    if (col >= dim)
        return;
    uint out_idx = row * dim + col;
    if (id < 0) {
        output[out_idx] = write_half(0.0f);
        return;
    }
    const device q4_0_block &blk = weights[uint(id) * blocks_per_row + block];
    threadgroup half tg_scale = (half)0.0h;
    threadgroup uchar tg_qs[16];
    if (tid == 0)
        tg_scale = blk.scale;
    if (tid < 16)
        tg_qs[tid] = blk.qs[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uchar packed = tg_qs[off >> 1];
    int q = (off & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    q -= 8;
    float v = float(tg_scale) * float(q) * scale;
    output[out_idx] = write_half(v);
}

kernel void embedding_gather_q4_1_to_f16_opt(
    device const q4_1_block *weights [[buffer(0)]], constant int32_t *ids [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &dim [[buffer(3)]], constant uint &rows [[buffer(4)]],
    constant float &scale [[buffer(5)]], constant uint &blocks_per_row [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_idx [[threadgroup_position_in_grid]]
) {
    uint total_blocks = rows * blocks_per_row;
    if (tg_idx >= total_blocks)
        return;
    uint row = tg_idx / blocks_per_row;
    uint block = tg_idx - row * blocks_per_row;
    int id = ids[row];
    uint off = tid;
    if (off >= kQuantBlockSize)
        return;
    uint col = block * kQuantBlockSize + off;
    if (col >= dim)
        return;
    uint out_idx = row * dim + col;
    if (id < 0) {
        output[out_idx] = write_half(0.0f);
        return;
    }
    const device q4_1_block &blk = weights[uint(id) * blocks_per_row + block];
    threadgroup half tg_scale = (half)0.0h, tg_min = (half)0.0h;
    threadgroup uchar tg_qs[16];
    if (tid == 0) {
        tg_scale = blk.scale;
        tg_min = blk.min;
    }
    if (tid < 16)
        tg_qs[tid] = blk.qs[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uchar packed = tg_qs[off >> 1];
    int q = (off & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
    float v = (float(tg_scale) * float(q) + float(tg_min)) * scale;
    output[out_idx] = write_half(v);
}

kernel void embedding_gather_q5_0_to_f16_opt(
    device const q5_0_block *weights [[buffer(0)]], constant int32_t *ids [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &dim [[buffer(3)]], constant uint &rows [[buffer(4)]],
    constant float &scale [[buffer(5)]], constant uint &blocks_per_row [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_idx [[threadgroup_position_in_grid]]
) {
    uint total_blocks = rows * blocks_per_row;
    if (tg_idx >= total_blocks)
        return;
    uint row = tg_idx / blocks_per_row;
    uint block = tg_idx - row * blocks_per_row;
    int id = ids[row];
    uint off = tid;
    if (off >= kQuantBlockSize)
        return;
    uint col = block * kQuantBlockSize + off;
    if (col >= dim)
        return;
    uint out_idx = row * dim + col;
    if (id < 0) {
        output[out_idx] = write_half(0.0f);
        return;
    }
    const device q5_0_block &blk = weights[uint(id) * blocks_per_row + block];
    threadgroup half tg_scale = (half)0.0h;
    threadgroup uchar tg_qs[16];
    threadgroup uint tg_bits = 0u;
    if (tid == 0) {
        tg_scale = blk.scale;
        tg_bits = (uint)blk.qh[0] | ((uint)blk.qh[1] << 8u) | ((uint)blk.qh[2] << 16u) | ((uint)blk.qh[3] << 24u);
    }
    if (tid < 16)
        tg_qs[tid] = blk.qs[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = (off < 16u) ? off : (off - 16u);
    uchar packed = tg_qs[base];
    uint q4 = (off < 16u) ? (packed & 0x0fu) : (packed >> 4);
    uint bit_index = (off < 16u) ? base : (base + 16u);
    uint hi = (tg_bits >> bit_index) & 0x1u;
    int q = (int)((q4 | (hi << 4)) - 16u);
    float v = float(tg_scale) * float(q) * scale;
    output[out_idx] = write_half(v);
}

kernel void embedding_gather_q5_1_to_f16_opt(
    device const q5_1_block *weights [[buffer(0)]], constant int32_t *ids [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &dim [[buffer(3)]], constant uint &rows [[buffer(4)]],
    constant float &scale [[buffer(5)]], constant uint &blocks_per_row [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_idx [[threadgroup_position_in_grid]]
) {
    uint total_blocks = rows * blocks_per_row;
    if (tg_idx >= total_blocks)
        return;
    uint row = tg_idx / blocks_per_row;
    uint block = tg_idx - row * blocks_per_row;
    int id = ids[row];
    uint off = tid;
    if (off >= kQuantBlockSize)
        return;
    uint col = block * kQuantBlockSize + off;
    if (col >= dim)
        return;
    uint out_idx = row * dim + col;
    if (id < 0) {
        output[out_idx] = write_half(0.0f);
        return;
    }
    const device q5_1_block &blk = weights[uint(id) * blocks_per_row + block];
    threadgroup half tg_scale = (half)0.0h, tg_min = (half)0.0h;
    threadgroup uchar tg_qs[16];
    threadgroup uint tg_bits = 0u;
    if (tid == 0) {
        tg_scale = blk.scale;
        tg_min = blk.min;
        tg_bits = (uint)blk.qh[0] | ((uint)blk.qh[1] << 8u) | ((uint)blk.qh[2] << 16u) | ((uint)blk.qh[3] << 24u);
    }
    if (tid < 16)
        tg_qs[tid] = blk.qs[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = (off < 16u) ? off : (off - 16u);
    uchar packed = tg_qs[base];
    uint q4 = (off < 16u) ? (packed & 0x0fu) : (packed >> 4);
    uint bit_index = (off < 16u) ? base : (base + 16u);
    uint hi = (tg_bits >> bit_index) & 0x1u;
    uint q5 = q4 | (hi << 4);
    float v = (float(tg_scale) * float(q5) + float(tg_min)) * scale;
    output[out_idx] = write_half(v);
}

kernel void embedding_gather_q8_0_to_f16_opt(
    device const q8_0_block *weights [[buffer(0)]], constant int32_t *ids [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &dim [[buffer(3)]], constant uint &rows [[buffer(4)]],
    constant float &scale [[buffer(5)]], constant uint &blocks_per_row [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]], uint tg_idx [[threadgroup_position_in_grid]]
) {
    uint total_blocks = rows * blocks_per_row;
    if (tg_idx >= total_blocks)
        return;
    uint row = tg_idx / blocks_per_row;
    uint block = tg_idx - row * blocks_per_row;
    int id = ids[row];
    uint off = tid;
    if (off >= kQuantBlockSize)
        return;
    uint col = block * kQuantBlockSize + off;
    if (col >= dim)
        return;
    uint out_idx = row * dim + col;
    if (id < 0) {
        output[out_idx] = write_half(0.0f);
        return;
    }
    const device q8_0_block &blk = weights[uint(id) * blocks_per_row + block];
    threadgroup half tg_scale = (half)0.0h;
    if (tid == 0)
        tg_scale = blk.scale;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v = float(tg_scale) * float(blk.qs[off]) * scale;
    output[out_idx] = write_half(v);
}
