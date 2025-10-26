#ifndef CPU_MATMUL_QKV_ROPE_H
#define CPU_MATMUL_QKV_ROPE_H

#include "marmot/ops_types.h"
#include "marmot/types.h"

#include <stddef.h>
#include <stdint.h>

#include <math.h>

#include "cpu_backend_internal.h"

static inline size_t cpu_matmul_qkv_resolve_head_dim(size_t dim, const marmot_rope_params_t *rope) {
    if (rope == nullptr) {
        return dim;
    }
    const size_t head_dim = (size_t)rope->head_dim;
    if (head_dim == 0 || head_dim > dim || (dim % head_dim) != 0 || (head_dim & 1u) != 0u) {
        return dim;
    }
    return head_dim;
}

static inline const float *cpu_rope_sincos_lookup(
    const float *sincos_base, size_t sincos_stride, size_t sincos_cached_positions, const int32_t *positions_i32,
    const int64_t *positions_i64, size_t index
) {
    if (sincos_base == nullptr || sincos_stride == 0 || sincos_cached_positions == 0) {
        return nullptr;
    }
    int64_t pos = -1;
    if (positions_i32 != nullptr) {
        pos = positions_i32[index];
    } else if (positions_i64 != nullptr) {
        pos = positions_i64[index];
    } else {
        return nullptr;
    }
    if (pos < 0) {
        return nullptr;
    }
    const size_t pos_index = (size_t)pos;
    if (pos_index >= sincos_cached_positions) {
        return nullptr;
    }
    return sincos_base + pos_index * sincos_stride;
}

static inline void cpu_matmul_qkv_rotate_row_f32(
    float *row, size_t dim, float position, const float *freqs, float attn_scale, marmot_rope_type_t rope_type
) {
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float angle = position * freqs[i];
        float sin_theta = 0.0f;
        float cos_theta = 0.0f;
        cpu_sincosf(angle, &sin_theta, &cos_theta);
        cos_theta *= attn_scale;
        sin_theta *= attn_scale;
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        const float even = row[even_index];
        const float odd = row[odd_index];
        row[even_index] = even * cos_theta - odd * sin_theta;
        row[odd_index] = even * sin_theta + odd * cos_theta;
    }
}

static inline void
cpu_matmul_qkv_rotate_row_f32_sincos(float *row, size_t dim, const float *sincos, marmot_rope_type_t rope_type) {
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float cos_theta = sincos[2 * i];
        const float sin_theta = sincos[2 * i + 1];
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        const float even = row[even_index];
        const float odd = row[odd_index];
        row[even_index] = even * cos_theta - odd * sin_theta;
        row[odd_index] = even * sin_theta + odd * cos_theta;
    }
}

static inline void cpu_matmul_qkv_rotate_rows_f32(
    float *row_a, float *row_b, size_t dim, float position, const float *freqs, float attn_scale,
    marmot_rope_type_t rope_type
) {
    if (row_a == nullptr && row_b == nullptr) {
        return;
    }
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float angle = position * freqs[i];
        float sin_theta = 0.0f;
        float cos_theta = 0.0f;
        cpu_sincosf(angle, &sin_theta, &cos_theta);
        cos_theta *= attn_scale;
        sin_theta *= attn_scale;
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        if (row_a != nullptr) {
            const float even = row_a[even_index];
            const float odd = row_a[odd_index];
            row_a[even_index] = even * cos_theta - odd * sin_theta;
            row_a[odd_index] = even * sin_theta + odd * cos_theta;
        }
        if (row_b != nullptr) {
            const float even = row_b[even_index];
            const float odd = row_b[odd_index];
            row_b[even_index] = even * cos_theta - odd * sin_theta;
            row_b[odd_index] = even * sin_theta + odd * cos_theta;
        }
    }
}

static inline void cpu_matmul_qkv_rotate_row_f32_headed(
    float *row, size_t dim, size_t head_dim, float position, const float *freqs, float attn_scale,
    marmot_rope_type_t rope_type
) {
    if (row == nullptr || freqs == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_row_f32(row, dim, position, freqs, attn_scale, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        cpu_matmul_qkv_rotate_row_f32(row + h * head_dim, head_dim, position, freqs, attn_scale, rope_type);
    }
}

static inline void cpu_matmul_qkv_rotate_row_f32_sincos_headed(
    float *row, size_t dim, size_t head_dim, const float *sincos, marmot_rope_type_t rope_type
) {
    if (row == nullptr || sincos == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_row_f32_sincos(row, dim, sincos, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        cpu_matmul_qkv_rotate_row_f32_sincos(row + h * head_dim, head_dim, sincos, rope_type);
    }
}

static inline void cpu_matmul_qkv_rotate_rows_f32_headed(
    float *row_a, float *row_b, size_t dim, size_t head_dim, float position, const float *freqs, float attn_scale,
    marmot_rope_type_t rope_type
) {
    if (row_a == nullptr && row_b == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_rows_f32(row_a, row_b, dim, position, freqs, attn_scale, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        float *head_a = row_a != nullptr ? row_a + h * head_dim : nullptr;
        float *head_b = row_b != nullptr ? row_b + h * head_dim : nullptr;
        cpu_matmul_qkv_rotate_rows_f32(head_a, head_b, head_dim, position, freqs, attn_scale, rope_type);
    }
}

static inline void cpu_matmul_qkv_rotate_row_f16(
    marmot_float16_t *row, size_t dim, float position, const float *freqs, float attn_scale,
    marmot_rope_type_t rope_type
) {
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float angle = position * freqs[i];
        float sin_theta = 0.0f;
        float cos_theta = 0.0f;
        cpu_sincosf(angle, &sin_theta, &cos_theta);
        cos_theta *= attn_scale;
        sin_theta *= attn_scale;
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        const float even = (float)marmot_float16_to_native(row[even_index]);
        const float odd = (float)marmot_float16_to_native(row[odd_index]);
        const float rotated_even = even * cos_theta - odd * sin_theta;
        const float rotated_odd = even * sin_theta + odd * cos_theta;
        row[even_index] = marmot_native_to_float16((_Float16)rotated_even);
        row[odd_index] = marmot_native_to_float16((_Float16)rotated_odd);
    }
}

static inline void cpu_matmul_qkv_rotate_row_f16_sincos(
    marmot_float16_t *row, size_t dim, const float *sincos, marmot_rope_type_t rope_type
) {
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float cos_theta = sincos[2 * i];
        const float sin_theta = sincos[2 * i + 1];
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        const float even = (float)marmot_float16_to_native(row[even_index]);
        const float odd = (float)marmot_float16_to_native(row[odd_index]);
        const float rotated_even = even * cos_theta - odd * sin_theta;
        const float rotated_odd = even * sin_theta + odd * cos_theta;
        row[even_index] = marmot_native_to_float16((_Float16)rotated_even);
        row[odd_index] = marmot_native_to_float16((_Float16)rotated_odd);
    }
}

static inline void cpu_matmul_qkv_rotate_rows_f16(
    marmot_float16_t *row_a, marmot_float16_t *row_b, size_t dim, float position, const float *freqs, float attn_scale,
    marmot_rope_type_t rope_type
) {
    if (row_a == nullptr && row_b == nullptr) {
        return;
    }
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float angle = position * freqs[i];
        float sin_theta = 0.0f;
        float cos_theta = 0.0f;
        cpu_sincosf(angle, &sin_theta, &cos_theta);
        cos_theta *= attn_scale;
        sin_theta *= attn_scale;
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        if (row_a != nullptr) {
            const float even = (float)marmot_float16_to_native(row_a[even_index]);
            const float odd = (float)marmot_float16_to_native(row_a[odd_index]);
            const float rotated_even = even * cos_theta - odd * sin_theta;
            const float rotated_odd = even * sin_theta + odd * cos_theta;
            row_a[even_index] = marmot_native_to_float16((_Float16)rotated_even);
            row_a[odd_index] = marmot_native_to_float16((_Float16)rotated_odd);
        }
        if (row_b != nullptr) {
            const float even = (float)marmot_float16_to_native(row_b[even_index]);
            const float odd = (float)marmot_float16_to_native(row_b[odd_index]);
            const float rotated_even = even * cos_theta - odd * sin_theta;
            const float rotated_odd = even * sin_theta + odd * cos_theta;
            row_b[even_index] = marmot_native_to_float16((_Float16)rotated_even);
            row_b[odd_index] = marmot_native_to_float16((_Float16)rotated_odd);
        }
    }
}

static inline void cpu_matmul_qkv_rotate_row_f16_headed(
    marmot_float16_t *row, size_t dim, size_t head_dim, float position, const float *freqs, float attn_scale,
    marmot_rope_type_t rope_type
) {
    if (row == nullptr || freqs == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_row_f16(row, dim, position, freqs, attn_scale, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        cpu_matmul_qkv_rotate_row_f16(row + h * head_dim, head_dim, position, freqs, attn_scale, rope_type);
    }
}

static inline void cpu_matmul_qkv_rotate_row_f16_sincos_headed(
    marmot_float16_t *row, size_t dim, size_t head_dim, const float *sincos, marmot_rope_type_t rope_type
) {
    if (row == nullptr || sincos == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_row_f16_sincos(row, dim, sincos, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        cpu_matmul_qkv_rotate_row_f16_sincos(row + h * head_dim, head_dim, sincos, rope_type);
    }
}

static inline void cpu_matmul_qkv_rotate_rows_f16_headed(
    marmot_float16_t *row_a, marmot_float16_t *row_b, size_t dim, size_t head_dim, float position, const float *freqs,
    float attn_scale, marmot_rope_type_t rope_type
) {
    if (row_a == nullptr && row_b == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_rows_f16(row_a, row_b, dim, position, freqs, attn_scale, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        marmot_float16_t *head_a = row_a != nullptr ? row_a + h * head_dim : nullptr;
        marmot_float16_t *head_b = row_b != nullptr ? row_b + h * head_dim : nullptr;
        cpu_matmul_qkv_rotate_rows_f16(head_a, head_b, head_dim, position, freqs, attn_scale, rope_type);
    }
}

static inline void cpu_matmul_qkv_rotate_row_bf16(
    marmot_bfloat16_t *row, size_t dim, float position, const float *freqs, float attn_scale,
    marmot_rope_type_t rope_type
) {
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float angle = position * freqs[i];
        float sin_theta = 0.0f;
        float cos_theta = 0.0f;
        cpu_sincosf(angle, &sin_theta, &cos_theta);
        cos_theta *= attn_scale;
        sin_theta *= attn_scale;
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        const float even = marmot_bfloat16_to_native(row[even_index]);
        const float odd = marmot_bfloat16_to_native(row[odd_index]);
        const float rotated_even = even * cos_theta - odd * sin_theta;
        const float rotated_odd = even * sin_theta + odd * cos_theta;
        row[even_index] = marmot_native_to_bfloat16(rotated_even);
        row[odd_index] = marmot_native_to_bfloat16(rotated_odd);
    }
}

static inline void cpu_matmul_qkv_rotate_row_bf16_sincos(
    marmot_bfloat16_t *row, size_t dim, const float *sincos, marmot_rope_type_t rope_type
) {
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float cos_theta = sincos[2 * i];
        const float sin_theta = sincos[2 * i + 1];
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        const float even = marmot_bfloat16_to_native(row[even_index]);
        const float odd = marmot_bfloat16_to_native(row[odd_index]);
        const float rotated_even = even * cos_theta - odd * sin_theta;
        const float rotated_odd = even * sin_theta + odd * cos_theta;
        row[even_index] = marmot_native_to_bfloat16(rotated_even);
        row[odd_index] = marmot_native_to_bfloat16(rotated_odd);
    }
}

static inline void cpu_matmul_qkv_rotate_rows_bf16(
    marmot_bfloat16_t *row_a, marmot_bfloat16_t *row_b, size_t dim, float position, const float *freqs,
    float attn_scale, marmot_rope_type_t rope_type
) {
    if (row_a == nullptr && row_b == nullptr) {
        return;
    }
    const size_t pair_count = dim / 2;
    const size_t half_dim = dim / 2;
    const bool is_neox = rope_type == MARMOT_ROPE_TYPE_NEOX;
    for (size_t i = 0; i < pair_count; ++i) {
        const float angle = position * freqs[i];
        float sin_theta = 0.0f;
        float cos_theta = 0.0f;
        cpu_sincosf(angle, &sin_theta, &cos_theta);
        cos_theta *= attn_scale;
        sin_theta *= attn_scale;
        const size_t even_index = is_neox ? i : 2 * i;
        const size_t odd_index = is_neox ? (i + half_dim) : (2 * i + 1);
        if (row_a != nullptr) {
            const float even = marmot_bfloat16_to_native(row_a[even_index]);
            const float odd = marmot_bfloat16_to_native(row_a[odd_index]);
            const float rotated_even = even * cos_theta - odd * sin_theta;
            const float rotated_odd = even * sin_theta + odd * cos_theta;
            row_a[even_index] = marmot_native_to_bfloat16(rotated_even);
            row_a[odd_index] = marmot_native_to_bfloat16(rotated_odd);
        }
        if (row_b != nullptr) {
            const float even = marmot_bfloat16_to_native(row_b[even_index]);
            const float odd = marmot_bfloat16_to_native(row_b[odd_index]);
            const float rotated_even = even * cos_theta - odd * sin_theta;
            const float rotated_odd = even * sin_theta + odd * cos_theta;
            row_b[even_index] = marmot_native_to_bfloat16(rotated_even);
            row_b[odd_index] = marmot_native_to_bfloat16(rotated_odd);
        }
    }
}

static inline void cpu_matmul_qkv_rotate_row_bf16_headed(
    marmot_bfloat16_t *row, size_t dim, size_t head_dim, float position, const float *freqs, float attn_scale,
    marmot_rope_type_t rope_type
) {
    if (row == nullptr || freqs == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_row_bf16(row, dim, position, freqs, attn_scale, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        cpu_matmul_qkv_rotate_row_bf16(row + h * head_dim, head_dim, position, freqs, attn_scale, rope_type);
    }
}

static inline void cpu_matmul_qkv_rotate_row_bf16_sincos_headed(
    marmot_bfloat16_t *row, size_t dim, size_t head_dim, const float *sincos, marmot_rope_type_t rope_type
) {
    if (row == nullptr || sincos == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_row_bf16_sincos(row, dim, sincos, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        cpu_matmul_qkv_rotate_row_bf16_sincos(row + h * head_dim, head_dim, sincos, rope_type);
    }
}

static inline void cpu_matmul_qkv_rotate_rows_bf16_headed(
    marmot_bfloat16_t *row_a, marmot_bfloat16_t *row_b, size_t dim, size_t head_dim, float position, const float *freqs,
    float attn_scale, marmot_rope_type_t rope_type
) {
    if (row_a == nullptr && row_b == nullptr) {
        return;
    }
    if (head_dim >= dim) {
        cpu_matmul_qkv_rotate_rows_bf16(row_a, row_b, dim, position, freqs, attn_scale, rope_type);
        return;
    }
    const size_t head_count = dim / head_dim;
    for (size_t h = 0; h < head_count; ++h) {
        marmot_bfloat16_t *head_a = row_a != nullptr ? row_a + h * head_dim : nullptr;
        marmot_bfloat16_t *head_b = row_b != nullptr ? row_b + h * head_dim : nullptr;
        cpu_matmul_qkv_rotate_rows_bf16(head_a, head_b, head_dim, position, freqs, attn_scale, rope_type);
    }
}

#endif // CPU_MATMUL_QKV_ROPE_H
