#pragma once

constant uint kQuantBlockSize = 32u;
constant uint kQ4PackedBytes = 16u;
constant uint kQ5PackedBytes = 16u;
constant uint kQ5HighBits = 4u;
constant uint kQ8PackedBytes = 32u;

// K-quant constants (256-value super-blocks)
constant uint kQK_K = 256u;
constant uint kQK_K_ScaleBytes = 12u;

struct q4_0_block {
    half scale;
    uchar qs[kQ4PackedBytes];
};
static_assert(sizeof(q4_0_block) == 18);

struct q4_1_block {
    half scale;
    half min;
    uchar qs[kQ4PackedBytes];
};
static_assert(sizeof(q4_1_block) == 20);

struct q5_0_block {
    half scale;
    uchar qh[kQ5HighBits];
    uchar qs[kQ5PackedBytes];
};
static_assert(sizeof(q5_0_block) == 22);

struct q5_1_block {
    half scale;
    half min;
    uchar qh[kQ5HighBits];
    uchar qs[kQ5PackedBytes];
};
static_assert(sizeof(q5_1_block) == 24);

struct q8_0_block {
    half scale;
    char qs[kQuantBlockSize];
};
static_assert(sizeof(q8_0_block) == 34);

struct q8_1_block {
    half scale;
    half sum;
    char qs[kQ8PackedBytes];
};
static_assert(sizeof(q8_1_block) == 36);

// K-quant super-blocks (256 values per block)
struct q2_k_block {
    uchar scales[kQK_K / 16]; // 16 bytes: scales and mins, quantized with 4 bits
    uchar qs[kQK_K / 4];      // 64 bytes: quants (2 bits per value)
    half d;                   // super-block scale for quantized scales
    half dmin;                // super-block scale for quantized mins
};
static_assert(sizeof(q2_k_block) == 84);

struct q3_k_block {
    uchar hmask[kQK_K / 8];         // 32 bytes: quants - high bit
    uchar qs[kQK_K / 4];            // 64 bytes: quants - low 2 bits
    uchar scales[kQK_K_ScaleBytes]; // 12 bytes: scales, quantized with 6 bits
    half d;                         // super-block scale
};
static_assert(sizeof(q3_k_block) == 110);

struct q4_k_block {
    half d;
    half dmin;
    uchar scales[kQK_K_ScaleBytes];
    uchar qs[kQK_K / 2]; // 128 bytes: 4-bit quants
};
static_assert(sizeof(q4_k_block) == 144);

struct q5_k_block {
    half d;
    half dmin;
    uchar scales[kQK_K_ScaleBytes];
    uchar qh[kQK_K / 8]; // 32 bytes: high bits
    uchar qs[kQK_K / 2]; // 128 bytes: low 4 bits
};
static_assert(sizeof(q5_k_block) == 176);

struct q6_k_block {
    uchar ql[kQK_K / 2];     // 128 bytes: lower 4 bits
    uchar qh[kQK_K / 4];     // 64 bytes: upper 2 bits
    char scales[kQK_K / 16]; // 16 bytes: scales
    half d;                  // scale (at END, unlike other K-quants!)
};
static_assert(sizeof(q6_k_block) == 210);

struct q8_k_block {
    float d;                 // delta (32-bit float, not FP16!)
    char qs[kQK_K];          // 256 bytes: quants
    short bsums[kQK_K / 16]; // 32 bytes: sum of quants in groups of 16
};
static_assert(sizeof(q8_k_block) == 292);
