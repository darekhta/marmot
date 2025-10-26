#ifndef CPU_NORMALIZATION_INTERNAL_H
#define CPU_NORMALIZATION_INTERNAL_H

#include "marmot/dispatch.h"

#include "cpu_backend_internal.h"

static inline float cpu_norm_inv_sqrt_f32(float x) {
    return 1.0f / sqrtf(x);
}

static inline bool cpu_norm_no_work(size_t norm_size, size_t outer_size) {
    return norm_size == 0 || outer_size == 0;
}

static inline size_t cpu_norm_min_chunk(size_t norm_size) {
    if (norm_size < 128) {
        return 4;
    }
    if (norm_size < 256) {
        return 2;
    }
    return 1;
}

static inline size_t cpu_norm_min_outer(size_t norm_size) {
    if (norm_size <= 256) {
        return 1024;
    }
    if (norm_size <= 1024) {
        return 256;
    }
    if (norm_size >= 4096) {
        return 64;
    }
    return 256;
}

static inline void
cpu_norm_dispatch_rows(size_t outer_size, size_t norm_size, void *context, marmot_dispatch_range_fn work) {
    if (outer_size <= 1) {
        work(context, 0, outer_size);
        return;
    }
    if (outer_size < cpu_norm_min_outer(norm_size)) {
        work(context, 0, outer_size);
        return;
    }
    marmot_dispatch_parallel_for_range(
        MARMOT_DISPATCH_PRIORITY_NORMAL, outer_size, cpu_norm_min_chunk(norm_size), context, work
    );
}

typedef struct {
    const double *x;
    const double *residual;
    double *out;
    const double *weight;
    const double *bias;
    size_t norm_size;
    double eps;
} layernorm_f64_context_t;

typedef struct {
    const double *x;
    const double *residual;
    double *out;
    const double *weight;
    size_t norm_size;
    double eps;
    double weight_offset;
} rmsnorm_f64_context_t;

typedef struct {
    const float *x;
    const float *residual;
    float *out;
    const float *weight;
    const float *bias;
    size_t norm_size;
    float eps;
} layernorm_f32_context_t;

typedef struct {
    const float *x;
    const float *residual;
    float *out;
    const float *weight;
    size_t norm_size;
    float eps;
    float weight_offset;
} rmsnorm_f32_context_t;

typedef struct {
    const marmot_float16_t *x;
    const marmot_float16_t *residual;
    marmot_float16_t *out;
    const marmot_float16_t *weight;
    const marmot_float16_t *bias;
    size_t norm_size;
    float eps;
} layernorm_f16_context_t;

typedef struct {
    const marmot_float16_t *x;
    const marmot_float16_t *residual;
    marmot_float16_t *out;
    const marmot_float16_t *weight;
    size_t norm_size;
    float eps;
    float weight_offset;
} rmsnorm_f16_context_t;

typedef struct {
    const marmot_bfloat16_t *x;
    const marmot_bfloat16_t *residual;
    marmot_bfloat16_t *out;
    const marmot_bfloat16_t *weight;
    const marmot_bfloat16_t *bias;
    size_t norm_size;
    float eps;
} layernorm_bf16_context_t;

typedef struct {
    const marmot_bfloat16_t *x;
    const marmot_bfloat16_t *residual;
    marmot_bfloat16_t *out;
    const marmot_bfloat16_t *weight;
    size_t norm_size;
    float eps;
    float weight_offset;
} rmsnorm_bf16_context_t;

marmot_error_t cpu_layernorm_mixed_vector_f32(
    marmot_dtype_t dtype, const void *input_data, const void *residual_data, const float *weight, const float *bias,
    void *out_data, size_t outer_size, size_t norm_size, float eps
);

marmot_error_t cpu_rmsnorm_mixed_vector_f32(
    marmot_dtype_t dtype, const void *input_data, const void *residual_data, const float *weight, void *out_data,
    size_t outer_size, size_t norm_size, float eps, float weight_offset
);

#endif // CPU_NORMALIZATION_INTERNAL_H
