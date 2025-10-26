#include "cpu_backend_internal.h"
#include "matmul_qkv_rope.h"

marmot_error_t cpu_matmul_qkv_kernel_f32_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
    (void)rope_params;
    if (input == nullptr || weight == nullptr || out_q == nullptr || out_k == nullptr || out_v == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const float *input_data = (const float *)input->data;
    const float *weight_data = (const float *)weight->data;
    const float *bias_data = bias != nullptr ? (const float *)bias->data : nullptr;

    const size_t in_row_stride = input->shape.strides[0];
    const size_t in_col_stride = input->shape.strides[1];
    const size_t weight_row_stride = weight->shape.strides[0];
    const size_t weight_col_stride = weight->shape.strides[1];
    const size_t out_q_row_stride = out_q->shape.strides[0];
    const size_t out_q_col_stride = out_q->shape.strides[1];
    const size_t out_k_row_stride = out_k->shape.strides[0];
    const size_t out_k_col_stride = out_k->shape.strides[1];
    const size_t out_v_row_stride = out_v->shape.strides[0];
    const size_t out_v_col_stride = out_v->shape.strides[1];
    const size_t bias_stride = bias != nullptr ? bias->shape.strides[0] : 0;

    float *out_q_data = (float *)out_q->data;
    float *out_k_data = (float *)out_k->data;
    float *out_v_data = (float *)out_v->data;

    const marmot_tensor_t *positions = rope_params != nullptr ? rope_params->positions : nullptr;
    const bool apply_rope_q = rope_params != nullptr && rope_params->apply_to_q;
    const bool apply_rope_k = rope_params != nullptr && rope_params->apply_to_k;
    const size_t rope_head_dim = cpu_matmul_qkv_resolve_head_dim(M, rope_params);
    const float *rope_freqs = nullptr;
    const float *sincos_base = nullptr;
    size_t sincos_stride = 0;
    size_t sincos_cached_positions = 0;
    const int32_t *positions_i32 = nullptr;
    const int64_t *positions_i64 = nullptr;
    float *rope_temp = nullptr;
    float rope_attn_scale = 1.0f;
    if (rope_params != nullptr && (apply_rope_q || apply_rope_k)) {
        marmot_rope_freq_span_t span = {0};
        cpu_context_t *ctx = (cpu_context_t *)device_ctx;
        marmot_error_t freq_status = marmot_rope_freq_cache_ensure(
            ctx != nullptr ? &ctx->rope_cache : nullptr, rope_head_dim, rope_params, &span
        );
        if (freq_status != MARMOT_SUCCESS) {
            return freq_status;
        }
        rope_freqs = span.freqs;
        rope_temp = span.owns_buffer ? (float *)span.freqs : nullptr;
        rope_attn_scale = span.attn_scale;
        if (ctx != nullptr && positions != nullptr) {
            bool use_sincos_cache = false;
            marmot_error_t cache_status = cpu_rope_sincos_cache_ensure(ctx, &span, positions, N, &use_sincos_cache);
            if (cache_status != MARMOT_SUCCESS) {
                if (rope_temp != nullptr) {
                    free(rope_temp);
                }
                return cache_status;
            }
            if (use_sincos_cache) {
                sincos_base = ctx->rope_sincos_cache.sincos;
                sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
                sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
                if (positions->dtype == MARMOT_DTYPE_INT32) {
                    positions_i32 = (const int32_t *)positions->data;
                } else if (positions->dtype == MARMOT_DTYPE_INT64) {
                    positions_i64 = (const int64_t *)positions->data;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        const float *input_row = input_data + n * in_row_stride;
        float *out_q_row = out_q_data + n * out_q_row_stride;
        float *out_k_row = out_k_data + n * out_k_row_stride;
        float *out_v_row = out_v_data + n * out_v_row_stride;

        for (size_t m = 0; m < M; ++m) {
            const size_t row_offset = m;

            const float *wq_row = weight_data + row_offset * weight_row_stride;
            const float *wk_row = weight_data + (M + row_offset) * weight_row_stride;
            const float *wv_row = weight_data + (2 * M + row_offset) * weight_row_stride;

            float acc_q = bias_data != nullptr ? bias_data[row_offset * bias_stride] : 0.0f;
            float acc_k = bias_data != nullptr ? bias_data[(M + row_offset) * bias_stride] : 0.0f;
            float acc_v = bias_data != nullptr ? bias_data[(2 * M + row_offset) * bias_stride] : 0.0f;

            for (size_t k = 0; k < K; ++k) {
                const float a = input_row[k * in_col_stride];
                acc_q += a * wq_row[k * weight_col_stride];
                acc_k += a * wk_row[k * weight_col_stride];
                acc_v += a * wv_row[k * weight_col_stride];
            }

            out_q_row[row_offset * out_q_col_stride] = acc_q;
            out_k_row[row_offset * out_k_col_stride] = acc_k;
            out_v_row[row_offset * out_v_col_stride] = acc_v;
        }

        if (rope_freqs != nullptr && positions != nullptr) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f32_sincos_headed(
                        out_q_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
                if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f32_sincos_headed(
                        out_k_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                if (apply_rope_q && apply_rope_k) {
                    cpu_matmul_qkv_rotate_rows_f32_headed(
                        out_q_row, out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f32_headed(
                        out_q_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f32_headed(
                        out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                }
            }
        }
    }

    if (rope_temp != nullptr) {
        free(rope_temp);
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_qkv_kernel_f16_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
    const marmot_float16_t *input_data = (const marmot_float16_t *)input->data;
    const marmot_float16_t *weight_data = (const marmot_float16_t *)weight->data;
    const marmot_float16_t *bias_data = bias != nullptr ? (const marmot_float16_t *)bias->data : nullptr;
    marmot_float16_t *out_q_data = (marmot_float16_t *)out_q->data;
    marmot_float16_t *out_k_data = (marmot_float16_t *)out_k->data;
    marmot_float16_t *out_v_data = (marmot_float16_t *)out_v->data;

    const size_t in_row_stride = input->shape.strides[0];
    const size_t in_col_stride = input->shape.strides[1];
    const size_t weight_row_stride = weight->shape.strides[0];
    const size_t weight_col_stride = weight->shape.strides[1];
    const size_t out_q_row_stride = out_q->shape.strides[0];
    const size_t out_q_col_stride = out_q->shape.strides[1];
    const size_t out_k_row_stride = out_k->shape.strides[0];
    const size_t out_k_col_stride = out_k->shape.strides[1];
    const size_t out_v_row_stride = out_v->shape.strides[0];
    const size_t out_v_col_stride = out_v->shape.strides[1];
    const size_t bias_stride = bias != nullptr ? bias->shape.strides[0] : 0;

    const marmot_tensor_t *positions = rope_params != nullptr ? rope_params->positions : nullptr;
    const bool apply_rope_q = rope_params != nullptr && rope_params->apply_to_q;
    const bool apply_rope_k = rope_params != nullptr && rope_params->apply_to_k;
    const size_t rope_head_dim = cpu_matmul_qkv_resolve_head_dim(M, rope_params);
    const float *rope_freqs = nullptr;
    const float *sincos_base = nullptr;
    size_t sincos_stride = 0;
    size_t sincos_cached_positions = 0;
    const int32_t *positions_i32 = nullptr;
    const int64_t *positions_i64 = nullptr;
    float *rope_temp = nullptr;
    float rope_attn_scale = 1.0f;
    if (rope_params != nullptr && (apply_rope_q || apply_rope_k)) {
        marmot_rope_freq_span_t span = {0};
        cpu_context_t *ctx = (cpu_context_t *)device_ctx;
        marmot_error_t freq_status = marmot_rope_freq_cache_ensure(
            ctx != nullptr ? &ctx->rope_cache : nullptr, rope_head_dim, rope_params, &span
        );
        if (freq_status != MARMOT_SUCCESS) {
            return freq_status;
        }
        rope_freqs = span.freqs;
        rope_temp = span.owns_buffer ? (float *)span.freqs : nullptr;
        rope_attn_scale = span.attn_scale;
        if (ctx != nullptr && positions != nullptr) {
            bool use_sincos_cache = false;
            marmot_error_t cache_status = cpu_rope_sincos_cache_ensure(ctx, &span, positions, N, &use_sincos_cache);
            if (cache_status != MARMOT_SUCCESS) {
                if (rope_temp != nullptr) {
                    free(rope_temp);
                }
                return cache_status;
            }
            if (use_sincos_cache) {
                sincos_base = ctx->rope_sincos_cache.sincos;
                sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
                sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
                if (positions->dtype == MARMOT_DTYPE_INT32) {
                    positions_i32 = (const int32_t *)positions->data;
                } else if (positions->dtype == MARMOT_DTYPE_INT64) {
                    positions_i64 = (const int64_t *)positions->data;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        const marmot_float16_t *input_row = input_data + n * in_row_stride;
        marmot_float16_t *out_q_row = out_q_data + n * out_q_row_stride;
        marmot_float16_t *out_k_row = out_k_data + n * out_k_row_stride;
        marmot_float16_t *out_v_row = out_v_data + n * out_v_row_stride;

        for (size_t m = 0; m < M; ++m) {
            const size_t row_offset = m;
            const marmot_float16_t *wq_row = weight_data + row_offset * weight_row_stride;
            const marmot_float16_t *wk_row = weight_data + (M + row_offset) * weight_row_stride;
            const marmot_float16_t *wv_row = weight_data + (2 * M + row_offset) * weight_row_stride;

            float acc_q =
                bias_data != nullptr ? (float)marmot_float16_to_native(bias_data[row_offset * bias_stride]) : 0.0f;
            float acc_k = bias_data != nullptr
                ? (float)marmot_float16_to_native(bias_data[(M + row_offset) * bias_stride])
                : 0.0f;
            float acc_v = bias_data != nullptr
                ? (float)marmot_float16_to_native(bias_data[(2 * M + row_offset) * bias_stride])
                : 0.0f;

            for (size_t k = 0; k < K; ++k) {
                const float a = (float)marmot_float16_to_native(input_row[k * in_col_stride]);
                acc_q += a * (float)marmot_float16_to_native(wq_row[k * weight_col_stride]);
                acc_k += a * (float)marmot_float16_to_native(wk_row[k * weight_col_stride]);
                acc_v += a * (float)marmot_float16_to_native(wv_row[k * weight_col_stride]);
            }

            out_q_row[row_offset * out_q_col_stride] = marmot_native_to_float16((_Float16)acc_q);
            out_k_row[row_offset * out_k_col_stride] = marmot_native_to_float16((_Float16)acc_k);
            out_v_row[row_offset * out_v_col_stride] = marmot_native_to_float16((_Float16)acc_v);
        }

        if (rope_freqs != nullptr && positions != nullptr) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f16_sincos_headed(
                        out_q_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
                if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f16_sincos_headed(
                        out_k_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                if (apply_rope_q && apply_rope_k) {
                    cpu_matmul_qkv_rotate_rows_f16_headed(
                        out_q_row, out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_f16_headed(
                        out_q_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_f16_headed(
                        out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                }
            }
        }
    }

    if (rope_temp != nullptr) {
        free(rope_temp);
    }

    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_qkv_kernel_bf16_scalar(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    const marmot_tensor_t *bias, marmot_tensor_t *out_q, marmot_tensor_t *out_k, marmot_tensor_t *out_v,
    const marmot_rope_params_t *rope_params
) {
    const marmot_bfloat16_t *input_data = (const marmot_bfloat16_t *)input->data;
    const marmot_bfloat16_t *weight_data = (const marmot_bfloat16_t *)weight->data;
    const marmot_bfloat16_t *bias_data = bias != nullptr ? (const marmot_bfloat16_t *)bias->data : nullptr;
    marmot_bfloat16_t *out_q_data = (marmot_bfloat16_t *)out_q->data;
    marmot_bfloat16_t *out_k_data = (marmot_bfloat16_t *)out_k->data;
    marmot_bfloat16_t *out_v_data = (marmot_bfloat16_t *)out_v->data;

    const size_t in_row_stride = input->shape.strides[0];
    const size_t in_col_stride = input->shape.strides[1];
    const size_t weight_row_stride = weight->shape.strides[0];
    const size_t weight_col_stride = weight->shape.strides[1];
    const size_t out_q_row_stride = out_q->shape.strides[0];
    const size_t out_q_col_stride = out_q->shape.strides[1];
    const size_t out_k_row_stride = out_k->shape.strides[0];
    const size_t out_k_col_stride = out_k->shape.strides[1];
    const size_t out_v_row_stride = out_v->shape.strides[0];
    const size_t out_v_col_stride = out_v->shape.strides[1];
    const size_t bias_stride = bias != nullptr ? bias->shape.strides[0] : 0;

    const marmot_tensor_t *positions = rope_params != nullptr ? rope_params->positions : nullptr;
    const bool apply_rope_q = rope_params != nullptr && rope_params->apply_to_q;
    const bool apply_rope_k = rope_params != nullptr && rope_params->apply_to_k;
    const size_t rope_head_dim = cpu_matmul_qkv_resolve_head_dim(M, rope_params);
    const float *rope_freqs = nullptr;
    const float *sincos_base = nullptr;
    size_t sincos_stride = 0;
    size_t sincos_cached_positions = 0;
    const int32_t *positions_i32 = nullptr;
    const int64_t *positions_i64 = nullptr;
    float *rope_temp = nullptr;
    float rope_attn_scale = 1.0f;
    if (rope_params != nullptr && (apply_rope_q || apply_rope_k)) {
        marmot_rope_freq_span_t span = {0};
        cpu_context_t *ctx = (cpu_context_t *)device_ctx;
        marmot_error_t freq_status = marmot_rope_freq_cache_ensure(
            ctx != nullptr ? &ctx->rope_cache : nullptr, rope_head_dim, rope_params, &span
        );
        if (freq_status != MARMOT_SUCCESS) {
            return freq_status;
        }
        rope_freqs = span.freqs;
        rope_temp = span.owns_buffer ? (float *)span.freqs : nullptr;
        rope_attn_scale = span.attn_scale;
        if (ctx != nullptr && positions != nullptr) {
            bool use_sincos_cache = false;
            marmot_error_t cache_status = cpu_rope_sincos_cache_ensure(ctx, &span, positions, N, &use_sincos_cache);
            if (cache_status != MARMOT_SUCCESS) {
                if (rope_temp != nullptr) {
                    free(rope_temp);
                }
                return cache_status;
            }
            if (use_sincos_cache) {
                sincos_base = ctx->rope_sincos_cache.sincos;
                sincos_stride = ctx->rope_sincos_cache.pair_count * 2;
                sincos_cached_positions = ctx->rope_sincos_cache.cached_positions;
                if (positions->dtype == MARMOT_DTYPE_INT32) {
                    positions_i32 = (const int32_t *)positions->data;
                } else if (positions->dtype == MARMOT_DTYPE_INT64) {
                    positions_i64 = (const int64_t *)positions->data;
                }
            }
        }
    }

    for (size_t n = 0; n < N; ++n) {
        const marmot_bfloat16_t *input_row = input_data + n * in_row_stride;
        marmot_bfloat16_t *out_q_row = out_q_data + n * out_q_row_stride;
        marmot_bfloat16_t *out_k_row = out_k_data + n * out_k_row_stride;
        marmot_bfloat16_t *out_v_row = out_v_data + n * out_v_row_stride;

        for (size_t m = 0; m < M; ++m) {
            const size_t row_offset = m;
            const marmot_bfloat16_t *wq_row = weight_data + row_offset * weight_row_stride;
            const marmot_bfloat16_t *wk_row = weight_data + (M + row_offset) * weight_row_stride;
            const marmot_bfloat16_t *wv_row = weight_data + (2 * M + row_offset) * weight_row_stride;

            float acc_q = bias_data != nullptr ? marmot_bfloat16_to_native(bias_data[row_offset * bias_stride]) : 0.0f;
            float acc_k =
                bias_data != nullptr ? marmot_bfloat16_to_native(bias_data[(M + row_offset) * bias_stride]) : 0.0f;
            float acc_v =
                bias_data != nullptr ? marmot_bfloat16_to_native(bias_data[(2 * M + row_offset) * bias_stride]) : 0.0f;

            for (size_t k = 0; k < K; ++k) {
                const float a = marmot_bfloat16_to_native(input_row[k * in_col_stride]);
                acc_q += a * marmot_bfloat16_to_native(wq_row[k * weight_col_stride]);
                acc_k += a * marmot_bfloat16_to_native(wk_row[k * weight_col_stride]);
                acc_v += a * marmot_bfloat16_to_native(wv_row[k * weight_col_stride]);
            }

            out_q_row[row_offset * out_q_col_stride] = marmot_native_to_bfloat16(acc_q);
            out_k_row[row_offset * out_k_col_stride] = marmot_native_to_bfloat16(acc_k);
            out_v_row[row_offset * out_v_col_stride] = marmot_native_to_bfloat16(acc_v);
        }

        if (rope_freqs != nullptr && positions != nullptr) {
            const float *sincos = cpu_rope_sincos_lookup(
                sincos_base, sincos_stride, sincos_cached_positions, positions_i32, positions_i64, n
            );
            if (sincos != nullptr) {
                if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_bf16_sincos_headed(
                        out_q_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
                if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_bf16_sincos_headed(
                        out_k_row, M, rope_head_dim, sincos, rope_params->rope_type
                    );
                }
            } else {
                const float pos = cpu_rope_position_as_f32(positions, n);
                if (apply_rope_q && apply_rope_k) {
                    cpu_matmul_qkv_rotate_rows_bf16_headed(
                        out_q_row, out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_q) {
                    cpu_matmul_qkv_rotate_row_bf16_headed(
                        out_q_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                } else if (apply_rope_k) {
                    cpu_matmul_qkv_rotate_row_bf16_headed(
                        out_k_row, M, rope_head_dim, pos, rope_freqs, rope_attn_scale, rope_params->rope_type
                    );
                }
            }
        }
    }

    if (rope_temp != nullptr) {
        free(rope_temp);
    }

    return MARMOT_SUCCESS;
}
