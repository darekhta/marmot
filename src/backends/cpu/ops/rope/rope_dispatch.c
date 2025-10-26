#include <math.h>
#include <string.h>

#include "cpu_backend_internal.h"
#include "rope_kernels.h"

#if HAS_AVX2
extern const cpu_rope_traits_t cpu_rope_avx_traits;
extern const cpu_rope_traits_t cpu_rope_avx_f16_traits;
extern const cpu_rope_traits_t cpu_rope_avx_bf16_traits;
#endif
#if HAS_NEON
extern const cpu_rope_traits_t cpu_rope_neon_traits;
extern const cpu_rope_traits_t cpu_rope_neon_f16_traits;
extern const cpu_rope_traits_t cpu_rope_neon_bf16_traits;
#endif

static marmot_error_t cpu_rope_validate(
    const marmot_tensor_t *x, const marmot_tensor_t *positions, const marmot_tensor_t *out, size_t *seq_len_out,
    size_t *dim_out, size_t *total_seqs_out
) {
    if (unlikely(x == nullptr || positions == nullptr || out == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in RoPE");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_dtype_t dtype = x->dtype;
    const bool supported = (dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT64 ||
                            dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_BFLOAT16
#if MARMOT_ENABLE_FP8
                            || dtype == MARMOT_DTYPE_FLOAT8_E4M3 || dtype == MARMOT_DTYPE_FLOAT8_E5M2
#endif
    );
    if (!supported || out->dtype != dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "RoPE dtype not supported on CPU backend");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    if (unlikely(x->shape.ndim < 2)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE requires at least 2D tensor");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t dim = x->shape.shape[x->shape.ndim - 1];
    if (unlikely(dim % 2 != 0)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE requires even dimension");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t seq_len = x->shape.shape[x->shape.ndim - 2];
    const size_t total_tokens = marmot_tensor_num_elements(x);
    if (unlikely(seq_len == 0 || dim == 0 || total_tokens == 0)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE input tensor must be non-empty");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t total_seqs = total_tokens / (seq_len * dim);
    const size_t expected_positions = total_seqs * seq_len;
    const size_t actual_positions = marmot_tensor_num_elements(positions);
    if (unlikely(actual_positions != expected_positions)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE positions shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (out->shape.ndim != x->shape.ndim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE output shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < x->shape.ndim; ++i) {
        if (out->shape.shape[i] != x->shape.shape[i]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE output shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    if (seq_len_out != nullptr) {
        *seq_len_out = seq_len;
    }
    if (dim_out != nullptr) {
        *dim_out = dim;
    }
    if (total_seqs_out != nullptr) {
        *total_seqs_out = total_seqs;
    }
    return MARMOT_SUCCESS;
}

static bool cpu_rope_is_contiguous(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const size_t ndim = tensor->shape.ndim;
    if (ndim == 0) {
        return true;
    }
    size_t expected_stride = 1;
    for (size_t i = ndim; i > 0; --i) {
        const size_t dim_idx = i - 1;
        if (tensor->shape.strides[dim_idx] != expected_stride) {
            return false;
        }
        expected_stride *= tensor->shape.shape[dim_idx];
    }
    return true;
}

static size_t cpu_rope_token_offset(const marmot_tensor_t *tensor, size_t token) {
    size_t offset = 0;
    size_t remaining = token;
    const size_t ndim = tensor->shape.ndim;
    for (size_t i = ndim - 1; i > 0; --i) {
        const size_t dim_idx = i - 1;
        const size_t dim = tensor->shape.shape[dim_idx];
        const size_t idx = dim == 0 ? 0 : (remaining % dim);
        offset += idx * tensor->shape.strides[dim_idx];
        remaining = dim == 0 ? 0 : (remaining / dim);
    }
    return offset;
}

static marmot_error_t cpu_rope_prepare_freqs(
    cpu_context_t *ctx, size_t dim, const marmot_rope_params_t *params, marmot_rope_freq_span_t *span_out
) {
    if (span_out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    span_out->freqs = nullptr;
    span_out->dim = 0;
    span_out->attn_scale = 1.0f;
    span_out->owns_buffer = false;

    const size_t pair_count = dim / 2;
    if (pair_count == 0) {
        return MARMOT_SUCCESS;
    }

    marmot_rope_freq_cache_t *cache = ctx != nullptr ? &ctx->rope_cache : nullptr;
    marmot_error_t status = marmot_rope_freq_cache_ensure(cache, dim, params, span_out);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_rope(const void *device_ctx, const marmot_tensor_t *x, const marmot_rope_params_t *params, marmot_tensor_t *out) {
    if (params == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE parameters are null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_tensor_t *positions = params->positions;
    if (positions == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE parameters require positions tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    size_t seq_len = 0;
    size_t dim = 0;
    size_t total_seqs = 0;
    marmot_error_t err = cpu_rope_validate(x, positions, out, &seq_len, &dim, &total_seqs);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    if (positions->dtype != MARMOT_DTYPE_FLOAT32 && positions->dtype != MARMOT_DTYPE_INT32 &&
        positions->dtype != MARMOT_DTYPE_INT64) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "RoPE positions tensor must be FLOAT32, INT32, or INT64");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t ndim = x->shape.ndim;
    if (ndim < 2 || out->shape.ndim != ndim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE requires matching input/output ranks");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (x->shape.strides[ndim - 1] != 1 || out->shape.strides[ndim - 1] != 1) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "RoPE requires contiguous inner dimension");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t x_inner_stride = x->shape.strides[ndim - 1];
    const size_t out_inner_stride = out->shape.strides[ndim - 1];
    const size_t x_row_stride = x->shape.strides[ndim - 2];
    const size_t out_row_stride = out->shape.strides[ndim - 2];
    if (x_row_stride < dim || out_row_stride < dim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE requires row strides >= feature dimension");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    const bool contiguous = cpu_rope_is_contiguous(x) && cpu_rope_is_contiguous(out);

    const size_t pair_count = dim / 2;
    const size_t total_tokens = total_seqs * seq_len;

    if (pair_count == 0) {
        const size_t bytes = marmot_tensor_size_bytes(x);
        if (bytes > 0 && out->data != x->data) {
            memcpy(out->data, x->data, bytes);
        }
        return MARMOT_SUCCESS;
    }

    cpu_context_t *ctx = get_cpu_context(device_ctx);
    const bool use_float64 = (x->dtype == MARMOT_DTYPE_FLOAT64);

    marmot_rope_freq_span_t freqs32 = {0};
    err = cpu_rope_prepare_freqs(ctx, dim, params, &freqs32);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    bool handled = false;
    bool use_sincos_cache = false;
    if (ctx != nullptr) {
        marmot_error_t cache_status =
            cpu_rope_sincos_cache_ensure(ctx, &freqs32, positions, total_tokens, &use_sincos_cache);
        if (cache_status != MARMOT_SUCCESS) {
            if (freqs32.owns_buffer) {
                free((void *)freqs32.freqs);
            }
            return cache_status;
        }
    }

    const bool allow_vector =
        contiguous && (params->rope_type == MARMOT_ROPE_TYPE_NORM || params->rope_type == MARMOT_ROPE_TYPE_NEOX);
    if (!use_float64 && allow_vector) {
#if HAS_AVX2
        if (has_avx2(device_ctx)) {
            const cpu_rope_traits_t *traits = nullptr;
            if (x->dtype == MARMOT_DTYPE_FLOAT32) {
                traits = &cpu_rope_avx_traits;
            } else if (x->dtype == MARMOT_DTYPE_FLOAT16) {
                traits = &cpu_rope_avx_f16_traits;
            } else if (x->dtype == MARMOT_DTYPE_BFLOAT16) {
                traits = &cpu_rope_avx_bf16_traits;
            }

            if (traits != nullptr && (traits->min_dim == 0 || dim >= traits->min_dim)) {
                marmot_error_t kernel_status = traits->kernel(
                    ctx, x, positions, freqs32.freqs, freqs32.attn_scale, params->rope_type, seq_len, dim, total_seqs,
                    out
                );
                if (kernel_status == MARMOT_SUCCESS) {
                    handled = true;
                } else if (kernel_status != MARMOT_ERROR_NOT_IMPLEMENTED) {
                    if (freqs32.owns_buffer) {
                        free((void *)freqs32.freqs);
                    }
                    return kernel_status;
                }
            }
        }
#endif

#if HAS_NEON
        if (!handled && has_neon(device_ctx)) {
            const cpu_rope_traits_t *traits = nullptr;
            if (x->dtype == MARMOT_DTYPE_FLOAT32) {
                traits = &cpu_rope_neon_traits;
            } else if (x->dtype == MARMOT_DTYPE_FLOAT16) {
                traits = &cpu_rope_neon_f16_traits;
            } else if (x->dtype == MARMOT_DTYPE_BFLOAT16) {
                traits = &cpu_rope_neon_bf16_traits;
            }

            if (traits != nullptr && (traits->min_dim == 0 || dim >= traits->min_dim)) {
                marmot_error_t kernel_status = traits->kernel(
                    ctx, x, positions, freqs32.freqs, freqs32.attn_scale, params->rope_type, seq_len, dim, total_seqs,
                    out
                );
                if (kernel_status == MARMOT_SUCCESS) {
                    handled = true;
                } else if (kernel_status != MARMOT_ERROR_NOT_IMPLEMENTED) {
                    if (freqs32.owns_buffer) {
                        free((void *)freqs32.freqs);
                    }
                    return kernel_status;
                }
            }
        }
#endif
    }
    if (handled) {
        if (freqs32.owns_buffer) {
            free((void *)freqs32.freqs);
        }
        return MARMOT_SUCCESS;
    }

    const void *x_data = x->data;
    void *out_data = out->data;
    const bool is_neox = params->rope_type == MARMOT_ROPE_TYPE_NEOX;
    const size_t half_dim = dim / 2;

    const cpu_rope_sincos_cache_t *sincos_cache = use_sincos_cache ? &ctx->rope_sincos_cache : nullptr;
    const float *sincos_base = sincos_cache != nullptr ? sincos_cache->sincos : nullptr;
    const size_t sincos_stride = sincos_cache != nullptr ? sincos_cache->pair_count * 2 : 0;
    const size_t sincos_cached_positions = sincos_cache != nullptr ? sincos_cache->cached_positions : 0;
    const int32_t *positions_i32 =
        use_sincos_cache && positions->dtype == MARMOT_DTYPE_INT32 ? (const int32_t *)positions->data : nullptr;
    const int64_t *positions_i64 =
        use_sincos_cache && positions->dtype == MARMOT_DTYPE_INT64 ? (const int64_t *)positions->data : nullptr;

    if (use_float64) {
        const double *x_values = (const double *)x_data;
        double *out_values = (double *)out_data;
        for (size_t token = 0; token < total_tokens; ++token) {
            const double pos = (double)cpu_rope_position_as_f32(positions, token);
            const size_t base_in = cpu_rope_token_offset(x, token);
            const size_t base_out = cpu_rope_token_offset(out, token);
            const double attn_scale = (double)freqs32.attn_scale;

            for (size_t i = 0; i < pair_count; ++i) {
                const double angle = pos * (double)freqs32.freqs[i];
                const double cos_theta = cos(angle) * attn_scale;
                const double sin_theta = sin(angle) * attn_scale;

                const size_t even_offset = is_neox ? i : 2 * i;
                const size_t odd_offset = is_neox ? (i + half_dim) : (2 * i + 1);
                const size_t even_index = base_in + even_offset * x_inner_stride;
                const size_t odd_index = base_in + odd_offset * x_inner_stride;
                const double x_even = x_values[even_index];
                const double x_odd = x_values[odd_index];

                const double rotated_even = x_even * cos_theta - x_odd * sin_theta;
                const double rotated_odd = x_even * sin_theta + x_odd * cos_theta;

                out_values[base_out + even_offset * out_inner_stride] = rotated_even;
                out_values[base_out + odd_offset * out_inner_stride] = rotated_odd;
            }
        }
    } else {
        for (size_t token = 0; token < total_tokens; ++token) {
            const float *sincos = nullptr;
            if (positions_i32 != nullptr) {
                const int32_t pos = positions_i32[token];
                if (pos >= 0 && (size_t)pos < sincos_cached_positions) {
                    sincos = sincos_base + (size_t)pos * sincos_stride;
                }
            } else if (positions_i64 != nullptr) {
                const int64_t pos = positions_i64[token];
                if (pos >= 0 && (size_t)pos < sincos_cached_positions) {
                    sincos = sincos_base + (size_t)pos * sincos_stride;
                }
            }
            const float pos = sincos != nullptr ? 0.0f : cpu_rope_position_as_f32(positions, token);
            const size_t base_in = cpu_rope_token_offset(x, token);
            const size_t base_out = cpu_rope_token_offset(out, token);

            for (size_t i = 0; i < pair_count; ++i) {
                float cos_theta = 0.0f;
                float sin_theta = 0.0f;
                if (sincos != nullptr) {
                    cos_theta = sincos[2 * i];
                    sin_theta = sincos[2 * i + 1];
                } else {
                    const float angle = pos * freqs32.freqs[i];
                    cpu_sincosf(angle, &sin_theta, &cos_theta);
                    cos_theta *= freqs32.attn_scale;
                    sin_theta *= freqs32.attn_scale;
                }

                const size_t even_offset = is_neox ? i : 2 * i;
                const size_t odd_offset = is_neox ? (i + half_dim) : (2 * i + 1);
                const size_t even_index = base_in + even_offset * x_inner_stride;
                const size_t odd_index = base_in + odd_offset * x_inner_stride;
                const float x_even = cpu_load_as_f32(x->dtype, x_data, even_index);
                const float x_odd = cpu_load_as_f32(x->dtype, x_data, odd_index);

                const float rotated_even = x_even * cos_theta - x_odd * sin_theta;
                const float rotated_odd = x_even * sin_theta + x_odd * cos_theta;

                cpu_store_from_f32(out->dtype, out_data, base_out + even_offset * out_inner_stride, rotated_even);
                cpu_store_from_f32(out->dtype, out_data, base_out + odd_offset * out_inner_stride, rotated_odd);
            }
        }
    }

    if (freqs32.owns_buffer) {
        free((void *)freqs32.freqs);
    }

    return MARMOT_SUCCESS;
}
