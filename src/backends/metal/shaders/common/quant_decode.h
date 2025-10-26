#pragma once

#include <metal_stdlib>
using namespace metal;

#include "quant_blocks.h"

static inline uint marmot_unpack_q5_high_bits_32(const device uchar *qh) {
    return as_type<uint>(*(device const packed_uchar4 *)qh);
}

template <typename type4x4>
static inline void dequantize_q4_0_chunk(const device q4_0_block &blk, short il, thread type4x4 &reg) {
    const float scale = float(blk.scale);
    const uint shift = ((uint)il != 0u) ? 4u : 0u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        const uchar packed = blk.qs[i];
        const uint q = (uint(packed) >> shift) & 0x0Fu;
        reg[i / 4u][i % 4u] = scale * float(int(q) - 8);
    }
}

template <typename type4x4>
static inline void dequantize_q4_1_chunk(const device q4_1_block &blk, short il, thread type4x4 &reg) {
    const float scale = float(blk.scale);
    const float minv = float(blk.min);
    const uint shift = ((uint)il != 0u) ? 4u : 0u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        const uchar packed = blk.qs[i];
        const uint q = (uint(packed) >> shift) & 0x0Fu;
        reg[i / 4u][i % 4u] = scale * float(q) + minv;
    }
}

template <typename type4x4>
static inline void dequantize_q5_0_chunk(const device q5_0_block &blk, short il, thread type4x4 &reg) {
    const float scale = float(blk.scale);
    const uint shift = ((uint)il != 0u) ? 4u : 0u;
    const uint base = ((uint)il != 0u) ? 16u : 0u;
    const uint qh = marmot_unpack_q5_high_bits_32(blk.qh);

#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        const uchar packed = blk.qs[i];
        const uint q4 = (uint(packed) >> shift) & 0x0Fu;
        const uint q5 = q4 + (((qh >> (base + i)) & 0x1u) << 4u);
        reg[i / 4u][i % 4u] = scale * float(int(q5) - 16);
    }
}

template <typename type4x4>
static inline void dequantize_q5_1_chunk(const device q5_1_block &blk, short il, thread type4x4 &reg) {
    const float scale = float(blk.scale);
    const float minv = float(blk.min);
    const uint shift = ((uint)il != 0u) ? 4u : 0u;
    const uint base = ((uint)il != 0u) ? 16u : 0u;
    const uint qh = marmot_unpack_q5_high_bits_32(blk.qh);

#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        const uchar packed = blk.qs[i];
        const uint q4 = (uint(packed) >> shift) & 0x0Fu;
        const uint q5 = q4 + (((qh >> (base + i)) & 0x1u) << 4u);
        reg[i / 4u][i % 4u] = scale * float(q5) + minv;
    }
}

template <typename type4x4>
static inline void dequantize_q8_0_chunk(const device q8_0_block &blk, short il, thread type4x4 &reg) {
    const float scale = float(blk.scale);
    const uint base = ((uint)il != 0u) ? 16u : 0u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        reg[i / 4u][i % 4u] = scale * float(int(blk.qs[base + i]));
    }
}

static inline void get_scale_min_k4(uint idx, const device uchar *scales, thread uchar &d, thread uchar &m) {
    if (idx < 4u) {
        d = scales[idx] & 63u;
        m = scales[idx + 4u] & 63u;
    } else {
        d = (scales[idx + 4u] & 0x0Fu) | ((scales[idx - 4u] >> 6) << 4);
        m = (scales[idx + 4u] >> 4) | ((scales[idx] >> 6) << 4);
    }
}

static inline uchar2 get_scale_min_k4_just2(uint j, uint k, const device uchar *scales) {
    if (j < 4u) {
        const uchar d = scales[j + k] & 63u;
        const uchar m = scales[j + 4u + k] & 63u;
        return uchar2(d, m);
    }

    const uchar s0 = scales[j + 4u + k];
    const uchar d = (s0 & 0x0Fu) | ((scales[j - 4u + k] & 0xC0u) >> 2);
    const uchar m = (s0 >> 4) | ((scales[j + k] & 0xC0u) >> 2);
    return uchar2(d, m);
}

template <typename type4x4>
static inline void dequantize_q4_k_chunk(const device q4_k_block &blk, short il, thread type4x4 &reg) {
    const uint il_u = (uint)il;
    const uint is = (il_u >> 2) * 2u;

    const device uchar *q = blk.qs + (il_u >> 2) * 32u + (il_u & 1u) * 16u;

    const uint il_local = il_u & 3u;
    const uchar2 sc = get_scale_min_k4_just2(is, il_local >> 1, blk.scales);

    const float d = (il_local < 2u) ? float(blk.d) : float(blk.d) * (1.0f / 16.0f);
    const float dl = d * float(sc[0]);
    const float ml = float(blk.dmin) * float(sc[1]);

    const ushort mask = (il_local < 2u) ? 0x0Fu : 0xF0u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        reg[i / 4u][i % 4u] = half(dl * float(q[i] & mask) - ml);
    }
}

static inline void dequantize_q4_k_chunk_f32(const device q4_k_block &blk, short il, thread float4x4 &reg) {
    const uint il_u = (uint)il;
    const uint is = (il_u >> 2) * 2u;

    const device uchar *q = blk.qs + (il_u >> 2) * 32u + (il_u & 1u) * 16u;

    const uint il_local = il_u & 3u;
    const uchar2 sc = get_scale_min_k4_just2(is, il_local >> 1, blk.scales);

    const float d = (il_local < 2u) ? float(blk.d) : float(blk.d) * (1.0f / 16.0f);
    const float dl = d * float(sc[0]);
    const float ml = float(blk.dmin) * float(sc[1]);

    const ushort mask = (il_local < 2u) ? 0x0Fu : 0xF0u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        reg[i / 4u][i % 4u] = dl * float(q[i] & mask) - ml;
    }
}

template <typename type4x4>
static inline void dequantize_q8_k_chunk(const device q8_k_block &blk, short il, thread type4x4 &reg) {
    const float scale = blk.d;
    const device char *q = blk.qs + (uint)il * 16u;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        reg[i / 4u][i % 4u] = half(scale * float(q[i]));
    }
}

template <typename type4x4>
static inline void dequantize_q5_k_chunk(const device q5_k_block &blk, short il, thread type4x4 &reg) {
    const uint il_u = (uint)il;
    const uint is = (il_u >> 2) * 2u;

    const device uchar *q = blk.qs + (il_u >> 2) * 32u + (il_u & 1u) * 16u;
    const device uchar *qh = blk.qh + (il_u & 1u) * 16u;

    const uchar ul = uchar(1u << (il_u >> 1));
    const uint il_local = il_u & 3u;
    const uchar2 sc = get_scale_min_k4_just2(is, il_local >> 1, blk.scales);

    const float d = (il_local < 2u) ? float(blk.d) : float(blk.d) * (1.0f / 16.0f);
    const float dl = d * float(sc[0]);
    const float ml = float(blk.dmin) * float(sc[1]);

    const ushort mask = (il_local < 2u) ? 0x0Fu : 0xF0u;
    const float qh_val = (il_local < 2u) ? 16.0f : 256.0f;

#pragma clang loop unroll(full)
    for (uint i = 0; i < 16u; ++i) {
        const float qh_term = ((qh[i] & ul) != 0u) ? qh_val : 0.0f;
        reg[i / 4u][i % 4u] = dl * (float(q[i] & mask) + qh_term) - ml;
    }
}

template <typename type4x4>
static inline void dequantize_q6_k_chunk(const device q6_k_block &blk, short il, thread type4x4 &reg) {
    const uint il_u = (uint)il;

    const float d_all = float(blk.d);
    device const ushort *ql = (device const ushort *)blk.ql;
    device const ushort *qh = (device const ushort *)blk.qh;
    device const char *scales = (device const char *)blk.scales;

    ql = ql + 32u * (il_u >> 3) + 16u * ((il_u >> 1) & 1u) + 8u * (il_u & 1u);
    qh = qh + 16u * (il_u >> 3) + 8u * (il_u & 1u);

    const float sc = float(int(scales[(il_u & 1u) + 2u * (il_u >> 1)]));
    const uint il_local = (il_u >> 1) & 3u;

    const uint kmask1 =
        (il_local > 1u) ? ((il_local > 2u) ? 0xC0C0C0C0u : 0x30303030u) : ((il_local > 0u) ? 0x0C0C0C0Cu : 0x03030303u);
    const uint kmask2 = (il_local > 1u) ? 0xF0F0F0F0u : 0x0F0F0F0Fu;

    const float ml = d_all * sc * 32.0f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 * (1.0f / 256.0f);
    const float dl2 = dl1 * (1.0f / 256.0f);
    const float dl3 = dl2 * (1.0f / 256.0f);

    const uint shr_h = (il_local > 2u) ? 2u : 0u;
    const uint shl_h = (il_local > 1u) ? 0u : ((il_local > 0u) ? 2u : 4u);
    const uint shr_l = (il_local > 1u) ? 4u : 0u;

#pragma clang loop unroll(full)
    for (uint i = 0; i < 4u; ++i) {
        const uint low = (uint(ql[2u * i]) | (uint(ql[2u * i + 1u]) << 16u)) & kmask2;
        const uint high = (uint(qh[2u * i]) | (uint(qh[2u * i + 1u]) << 16u)) & kmask1;
        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);

        reg[i][0] = dl0 * float(q & 0xFFu) - ml;
        reg[i][1] = dl1 * float(q & 0xFF00u) - ml;
        reg[i][2] = dl2 * float(q & 0xFF0000u) - ml;
        reg[i][3] = dl3 * float(q & 0xFF000000u) - ml;
    }
}
