#include <math.h>

#include "backend/test_backend_utils.h"

static float test_moe_silu(float x) {
    return x / (1.0f + expf(-x));
}

static void matmul_row_vector(const float *x, size_t in_dim, const float *weight, size_t out_dim, float *out) {
    for (size_t row = 0; row < out_dim; ++row) {
        float acc = 0.0f;
        for (size_t col = 0; col < in_dim; ++col) {
            acc += x[col] * weight[row * in_dim + col];
        }
        out[row] = acc;
    }
}

static marmot_error_t quantize_matrix(
    const marmot_context_t *ctx, marmot_quant_kind_t kind, const float *src_data, size_t rows, size_t cols,
    void *dst_bytes, size_t dst_size
) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    if (traits == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t block_bytes = traits->header_bytes + traits->payload_bytes;
    const size_t blocks_per_row = (cols + traits->block_values - 1) / traits->block_values;
    const size_t row_bytes = blocks_per_row * block_bytes;
    if (dst_size != rows * row_bytes) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t row_shape[] = {cols};
    for (size_t row = 0; row < rows; ++row) {
        marmot_tensor_t *row_fp32 = marmot_tensor_create(ctx, row_shape, 1, MARMOT_DTYPE_FLOAT32);
        marmot_tensor_t *row_q = marmot_tensor_create(ctx, &row_bytes, 1, traits->storage_dtype);
        if (row_fp32 == nullptr || row_q == nullptr) {
            marmot_tensor_destroy(row_q);
            marmot_tensor_destroy(row_fp32);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        marmot_error_t status =
            marmot_tensor_copy_from_host_buffer(ctx, row_fp32, src_data + row * cols, cols * sizeof(float));
        if (status == MARMOT_SUCCESS) {
            switch (kind) {
            case MARMOT_QUANT_KIND_Q4_K:
                status = marmot_quantize_q4_k(ctx, row_fp32, row_q);
                break;
            case MARMOT_QUANT_KIND_Q6_K:
                status = marmot_quantize_q6_k(ctx, row_fp32, row_q);
                break;
            case MARMOT_QUANT_KIND_Q8_0:
                status = marmot_quantize_q8_0(ctx, row_fp32, row_q);
                break;
            default:
                status = MARMOT_ERROR_NOT_IMPLEMENTED;
                break;
            }
        }
        if (status == MARMOT_SUCCESS) {
            status = marmot_tensor_copy_to_host_buffer(ctx, row_q, (uint8_t *)dst_bytes + row * row_bytes, row_bytes);
        }

        marmot_tensor_destroy(row_q);
        marmot_tensor_destroy(row_fp32);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_tensor_t *create_quantized_expert_tensor(
    marmot_test_env_t *env, marmot_quant_kind_t kind, const float *expert_data, size_t rows, size_t cols, size_t experts
) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    assert_non_null(traits);

    const size_t block_bytes = traits->header_bytes + traits->payload_bytes;
    const size_t blocks_per_row = (cols + traits->block_values - 1) / traits->block_values;
    const size_t slice_bytes = rows * blocks_per_row * block_bytes;
    const size_t total_bytes = slice_bytes * experts;

    size_t storage_shape[] = {total_bytes};
    marmot_tensor_t *tensor = marmot_tensor_create(env->ctx, storage_shape, 1, traits->storage_dtype);
    assert_non_null(tensor);

    tensor->shape.ndim = 3;
    tensor->shape.shape[0] = cols;
    tensor->shape.shape[1] = rows;
    tensor->shape.shape[2] = experts;
    tensor->shape.strides[2] = 1;
    tensor->shape.strides[1] = experts;
    tensor->shape.strides[0] = rows * experts;
    tensor->quant_kind = kind;
    tensor->quant_layout = MARMOT_QUANT_LAYOUT_GGUF;
    tensor->packed_data = nullptr;
    tensor->packed_src_data = nullptr;
    tensor->packed_bytes = 0;
    tensor->packed_row_bytes = 0;
    tensor->packed_rows = 0;

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);

    uint8_t *packed = (uint8_t *)malloc(total_bytes);
    assert_non_null(packed);
    for (size_t expert = 0; expert < experts; ++expert) {
        const float *src = expert_data + expert * rows * cols;
        assert_int_equal(
            quantize_matrix(cpu_ctx, kind, src, rows, cols, packed + expert * slice_bytes, slice_bytes), MARMOT_SUCCESS
        );
    }

    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, tensor, packed, total_bytes), MARMOT_SUCCESS);

    free(packed);
    marmot_destroy(cpu_ctx);
    return tensor;
}

static void exercise_topk_case(marmot_test_env_t *env, marmot_dtype_t dtype) {
    const size_t shape[] = {2, 5};
    const float input_data[] = {
        1.0f, 4.0f, 4.0f, 2.0f, -1.0f, 0.0f, -3.0f, 5.0f, 5.0f, 1.0f,
    };
    const float expected_values[] = {
        4.0f, 4.0f, 2.0f, 5.0f, 5.0f, 1.0f,
    };
    const int32_t expected_indices[] = {
        1, 2, 3, 2, 3, 4,
    };

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, shape, 2, dtype);

    const size_t out_shape[] = {2, 3};
    marmot_tensor_t *values = marmot_tensor_create(env->ctx, out_shape, 2, dtype);
    marmot_tensor_t *indices = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_INT32);
    assert_non_null(input);
    assert_non_null(values);
    assert_non_null(indices);
    marmot_test_convert_f32_span(env, input, input_data, 10);

    assert_int_equal(
        marmot_topk(
            env->ctx,
            &(marmot_topk_desc_t){
                .x = input,
                .values_out = values,
                .indices_out = indices,
                .axis = -1,
                .k = 3,
            }
        ),
        MARMOT_SUCCESS
    );

    float actual_values[6];
    marmot_test_fetch_span(env, actual_values, MARMOT_DTYPE_FLOAT32, values, 6);
    marmot_test_expect_close_array(actual_values, expected_values, 6, 1e-6f);

    const marmot_int32_t *actual_indices = marmot_tensor_data_i32(env->ctx, indices);
    assert_non_null(actual_indices);
    for (size_t i = 0; i < 6; ++i) {
        assert_int_equal(actual_indices[i].value, expected_indices[i]);
    }

    marmot_test_tensor_destroy_all(3, indices, values, input);
}

static void exercise_topk(marmot_test_env_t *env) {
    exercise_topk_case(env, MARMOT_DTYPE_FLOAT32);
}

static void exercise_topk_f16(marmot_test_env_t *env) {
    exercise_topk_case(env, MARMOT_DTYPE_FLOAT16);
}

static void exercise_moe_experts_case(
    marmot_test_env_t *env, marmot_dtype_t dtype, const float topk_weights_data[4],
    marmot_router_weight_policy_t router_weight_policy
) {
    const size_t hidden_shape[] = {2, 2};
    const float hidden_data[] = {
        1.0f,
        -0.5f,
        0.25f,
        0.75f,
    };

    const float gate_expert0[] = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const float gate_expert1[] = {
        0.2f,
        -0.4f,
        0.7f,
        0.1f,
    };
    const float up_expert0[] = {
        0.5f,
        0.0f,
        0.0f,
        2.0f,
    };
    const float up_expert1[] = {
        1.2f,
        0.3f,
        -0.6f,
        0.8f,
    };
    const float down_expert0[] = {
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const float down_expert1[] = {
        0.6f,
        -0.2f,
        0.1f,
        0.9f,
    };
    const float gate_exps_storage[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.2f, -0.4f, 0.7f, 0.1f,
    };
    const float up_exps_storage[] = {
        0.5f, 0.0f, 0.0f, 2.0f, 1.2f, 0.3f, -0.6f, 0.8f,
    };
    const float down_exps_storage[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.6f, -0.2f, 0.1f, 0.9f,
    };
    const marmot_int32_t topk_ids_data[] = {
        MARMOT_I32(0),
        MARMOT_I32(1),
        MARMOT_I32(1),
        MARMOT_I32(0),
    };
    const float weights_scale = 1.5f;

    marmot_tensor_t *hidden = marmot_tensor_create(env->ctx, hidden_shape, 2, dtype);

    const size_t gate_shape[] = {2, 2, 2};
    const size_t down_shape[] = {2, 2, 2};
    const size_t topk_shape[] = {2, 2};

    marmot_tensor_t *gate_exps = marmot_tensor_create(env->ctx, gate_shape, 3, dtype);
    marmot_tensor_t *up_exps = marmot_tensor_create(env->ctx, gate_shape, 3, dtype);
    marmot_tensor_t *down_exps = marmot_tensor_create(env->ctx, down_shape, 3, dtype);
    marmot_tensor_t *topk_ids = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *topk_weights = marmot_tensor_create(env->ctx, topk_shape, 2, dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, hidden_shape, 2, dtype);
    assert_non_null(hidden);
    assert_non_null(gate_exps);
    assert_non_null(up_exps);
    assert_non_null(down_exps);
    assert_non_null(topk_ids);
    assert_non_null(topk_weights);
    assert_non_null(out);
    marmot_test_convert_f32_span(env, hidden, hidden_data, 4);

    marmot_test_convert_f32_span(env, gate_exps, gate_exps_storage, 8);
    marmot_test_convert_f32_span(env, up_exps, up_exps_storage, 8);
    marmot_test_convert_f32_span(env, down_exps, down_exps_storage, 8);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_ids, topk_ids_data, sizeof(topk_ids_data)), MARMOT_SUCCESS
    );
    marmot_test_convert_f32_span(env, topk_weights, topk_weights_data, 4);

    assert_int_equal(
        marmot_moe_experts(
            env->ctx,
            &(marmot_moe_experts_desc_t){
                .hidden_states = hidden,
                .gate_exps = gate_exps,
                .up_exps = up_exps,
                .down_exps = down_exps,
                .topk_ids = topk_ids,
                .topk_weights = topk_weights,
                .out = out,
                .ffn_type = MARMOT_FFN_SWIGLU,
                .weights_scale = weights_scale,
                .router_weight_policy = router_weight_policy,
            }
        ),
        MARMOT_SUCCESS
    );

    float expected[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float *expert_gate[2] = {gate_expert0, gate_expert1};
    const float *expert_up[2] = {up_expert0, up_expert1};
    const float *expert_down[2] = {down_expert0, down_expert1};
    for (size_t token = 0; token < 2; ++token) {
        const float *x = hidden_data + token * 2;
        float *out_row = expected + token * 2;
        float weight_norm = weights_scale;
        if (router_weight_policy == MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED) {
            const float sum = topk_weights_data[token * 2] + topk_weights_data[token * 2 + 1];
            weight_norm = sum > 0.0f ? weights_scale / sum : 0.0f;
        }
        for (size_t slot = 0; slot < 2; ++slot) {
            const int32_t expert_idx = topk_ids_data[token * 2 + slot].value;
            float gate_vals[2];
            float up_vals[2];
            float fused[2];
            float down_vals[2];
            matmul_row_vector(x, 2, expert_gate[expert_idx], 2, gate_vals);
            matmul_row_vector(x, 2, expert_up[expert_idx], 2, up_vals);
            for (size_t i = 0; i < 2; ++i) {
                fused[i] = test_moe_silu(gate_vals[i]) * up_vals[i];
            }
            matmul_row_vector(fused, 2, expert_down[expert_idx], 2, down_vals);
            const float weight = topk_weights_data[token * 2 + slot] * weight_norm;
            for (size_t i = 0; i < 2; ++i) {
                out_row[i] += weight * down_vals[i];
            }
        }
    }

    float actual[4];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT32, out, 4);
    marmot_test_expect_close_array(actual, expected, 4, dtype == MARMOT_DTYPE_FLOAT16 ? 3e-2f : 1e-5f);

    marmot_test_tensor_destroy_all(6, out, topk_weights, topk_ids, down_exps, up_exps, gate_exps);
    marmot_tensor_destroy(hidden);
}

static void exercise_moe_experts(marmot_test_env_t *env) {
    const float topk_weights_data[] = {
        0.75f,
        0.25f,
        1.0f,
        0.5f,
    };
    exercise_moe_experts_case(
        env, MARMOT_DTYPE_FLOAT32, topk_weights_data, MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED
    );
}

static void exercise_moe_experts_f16(marmot_test_env_t *env) {
    const float topk_weights_data[] = {
        0.75f,
        0.25f,
        1.0f,
        0.5f,
    };
    exercise_moe_experts_case(
        env, MARMOT_DTYPE_FLOAT16, topk_weights_data, MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED
    );
}

static void exercise_moe_experts_renorm(marmot_test_env_t *env) {
    const float topk_weights_data[] = {
        0.60f,
        0.20f,
        0.15f,
        0.45f,
    };
    exercise_moe_experts_case(
        env, MARMOT_DTYPE_FLOAT32, topk_weights_data, MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED
    );
}

static void run_moe_experts_duplicate_route_case(marmot_test_env_t *env, const char *route_mode, float *actual_out) {
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }

    const size_t hidden_shape[] = {1, 2};
    const size_t gate_shape[] = {2, 2, 2};
    const size_t down_shape[] = {2, 2, 2};
    const size_t topk_shape[] = {1, 4};
    const float hidden_data[] = {
        0.75f,
        -0.25f,
    };
    const float gate_exps_storage[] = {
        0.5f, 0.0f, 0.0f, 0.5f, 1.0f, -0.5f, 0.25f, 0.75f,
    };
    const float up_exps_storage[] = {
        0.2f, 0.0f, 0.0f, 0.3f, 0.8f, 0.4f, -0.2f, 0.6f,
    };
    const float down_exps_storage[] = {
        0.4f, 0.0f, 0.0f, 0.2f, 0.6f, -0.1f, 0.3f, 0.5f,
    };
    const marmot_int32_t topk_ids_data[] = {
        MARMOT_I32(1),
        MARMOT_I32(1),
        MARMOT_I32(1),
        MARMOT_I32(1),
    };
    const float topk_weights_data[] = {
        0.10f,
        0.20f,
        0.30f,
        0.40f,
    };

    marmot_tensor_t *hidden = marmot_test_tensor_from_array(env, hidden_shape, 2, hidden_data);
    marmot_tensor_t *gate_exps = marmot_test_tensor_from_array(env, gate_shape, 3, gate_exps_storage);
    marmot_tensor_t *up_exps = marmot_test_tensor_from_array(env, gate_shape, 3, up_exps_storage);
    marmot_tensor_t *down_exps = marmot_test_tensor_from_array(env, down_shape, 3, down_exps_storage);
    marmot_tensor_t *topk_ids = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *topk_weights = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(hidden);
    assert_non_null(gate_exps);
    assert_non_null(up_exps);
    assert_non_null(down_exps);
    assert_non_null(topk_ids);
    assert_non_null(topk_weights);
    assert_non_null(out);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_ids, topk_ids_data, sizeof(topk_ids_data)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_weights, topk_weights_data, sizeof(topk_weights_data)),
        MARMOT_SUCCESS
    );

#ifdef __APPLE__
    const char *saved_route_mode = getenv("MARMOT_MOE_ROUTE_MODE");
    char saved_route_mode_buf[16] = {0};
    if (saved_route_mode != nullptr && saved_route_mode[0] != '\0') {
        snprintf(saved_route_mode_buf, sizeof(saved_route_mode_buf), "%s", saved_route_mode);
    }
    if (route_mode != nullptr) {
        assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", route_mode, 1), 0);
    }
#endif

    marmot_error_t status = marmot_moe_experts(
        env->ctx,
        &(marmot_moe_experts_desc_t){
            .hidden_states = hidden,
            .gate_exps = gate_exps,
            .up_exps = up_exps,
            .down_exps = down_exps,
            .topk_ids = topk_ids,
            .topk_weights = topk_weights,
            .out = out,
            .ffn_type = MARMOT_FFN_SWIGLU,
            .weights_scale = 1.0f,
            .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        }
    );
    if (status != MARMOT_SUCCESS) {
        fail_msg("duplicate-route moe failed: %s (%s)", marmot_error_string(status), marmot_get_last_error_detail());
    }

    float gate_vals[2];
    float up_vals[2];
    float fused[2];
    float down_vals[2];
    matmul_row_vector(hidden_data, 2, gate_exps_storage + 4, 2, gate_vals);
    matmul_row_vector(hidden_data, 2, up_exps_storage + 4, 2, up_vals);
    for (size_t i = 0; i < 2; ++i) {
        fused[i] = test_moe_silu(gate_vals[i]) * up_vals[i];
    }
    matmul_row_vector(fused, 2, down_exps_storage + 4, 2, down_vals);
    const float total_weight =
        topk_weights_data[0] + topk_weights_data[1] + topk_weights_data[2] + topk_weights_data[3];
    float expected[2] = {
        total_weight * down_vals[0],
        total_weight * down_vals[1],
    };

    float actual[2];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT32, out, 2);
    marmot_test_expect_close_array(actual, expected, 2, 1e-5f);
    if (actual_out != nullptr) {
        memcpy(actual_out, actual, sizeof(actual));
    }

    marmot_test_tensor_destroy_all(6, out, topk_weights, topk_ids, down_exps, up_exps, gate_exps);
    marmot_tensor_destroy(hidden);

#ifdef __APPLE__
    if (route_mode != nullptr) {
        if (saved_route_mode_buf[0] != '\0') {
            assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", saved_route_mode_buf, 1), 0);
        } else {
            assert_int_equal(unsetenv("MARMOT_MOE_ROUTE_MODE"), 0);
        }
    }
#endif
}

static void exercise_moe_experts_quantized_q8_0(marmot_test_env_t *env) {
    enum {
        tokens = 2,
        hidden = 33,
        ff_length = 2,
        experts = 2,
        experts_per_token = 2,
    };

    float hidden_data[tokens * hidden];
    float gate_storage[experts * ff_length * hidden];
    float up_storage[experts * ff_length * hidden];
    float down_storage[experts * hidden * ff_length];
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t i = 0; i < hidden; ++i) {
            hidden_data[token * hidden + i] = ((float)((int)(token * hidden + i) % 9) - 4.0f) * 0.125f;
        }
    }
    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row = 0; row < ff_length; ++row) {
            for (size_t col = 0; col < hidden; ++col) {
                const size_t idx = expert * ff_length * hidden + row * hidden + col;
                gate_storage[idx] = 0.015f * (float)(expert + 1) * (float)(row + 1) * (float)((int)(col % 7) - 3);
                up_storage[idx] = 0.020f * (float)(expert + 2) * (float)((int)(col % 5) - 2);
            }
        }
        for (size_t row = 0; row < hidden; ++row) {
            for (size_t col = 0; col < ff_length; ++col) {
                const size_t idx = expert * hidden * ff_length + row * ff_length + col;
                down_storage[idx] = 0.030f * (float)(expert + 1) * (float)(col + 1) * (float)((int)(row % 11) - 5);
            }
        }
    }

    const marmot_int32_t topk_ids_data[] = {
        MARMOT_I32(0),
        MARMOT_I32(1),
        MARMOT_I32(1),
        MARMOT_I32(0),
    };
    const float topk_weights_data[] = {
        0.75f,
        0.25f,
        0.40f,
        0.60f,
    };

    const size_t hidden_shape[] = {tokens, hidden};
    const size_t topk_shape[] = {tokens, experts_per_token};
    marmot_tensor_t *hidden_tensor = marmot_test_tensor_from_array(env, hidden_shape, 2, hidden_data);
    marmot_tensor_t *gate_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q8_0, gate_storage, ff_length, hidden, experts);
    marmot_tensor_t *up_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q8_0, up_storage, ff_length, hidden, experts);
    marmot_tensor_t *down_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q8_0, down_storage, hidden, ff_length, experts);
    marmot_tensor_t *topk_ids = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *topk_weights = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(hidden_tensor);
    assert_non_null(gate_exps);
    assert_non_null(up_exps);
    assert_non_null(down_exps);
    assert_non_null(topk_ids);
    assert_non_null(topk_weights);
    assert_non_null(out);

    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_ids, topk_ids_data, sizeof(topk_ids_data)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_weights, topk_weights_data, sizeof(topk_weights_data)),
        MARMOT_SUCCESS
    );

    marmot_error_t status = marmot_moe_experts(
        env->ctx,
        &(marmot_moe_experts_desc_t){
            .hidden_states = hidden_tensor,
            .gate_exps = gate_exps,
            .up_exps = up_exps,
            .down_exps = down_exps,
            .topk_ids = topk_ids,
            .topk_weights = topk_weights,
            .out = out,
            .ffn_type = MARMOT_FFN_SWIGLU,
            .weights_scale = 1.0f,
            .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        }
    );
    if (status != MARMOT_SUCCESS) {
        fail_msg("quantized q8_0 moe failed: %s (%s)", marmot_error_string(status), marmot_get_last_error_detail());
    }

    float expected[tokens * hidden];
    memset(expected, 0, sizeof(expected));
    for (size_t token = 0; token < tokens; ++token) {
        const float *x = hidden_data + token * hidden;
        float *out_row = expected + token * hidden;
        for (size_t slot = 0; slot < experts_per_token; ++slot) {
            const int32_t expert_idx = topk_ids_data[token * experts_per_token + slot].value;
            float gate_vals[ff_length];
            float up_vals[ff_length];
            float fused[ff_length];
            float down_vals[hidden];
            matmul_row_vector(x, hidden, gate_storage + expert_idx * ff_length * hidden, ff_length, gate_vals);
            matmul_row_vector(x, hidden, up_storage + expert_idx * ff_length * hidden, ff_length, up_vals);
            for (size_t i = 0; i < ff_length; ++i) {
                fused[i] = test_moe_silu(gate_vals[i]) * up_vals[i];
            }
            matmul_row_vector(fused, ff_length, down_storage + expert_idx * hidden * ff_length, hidden, down_vals);
            for (size_t i = 0; i < hidden; ++i) {
                out_row[i] += topk_weights_data[token * experts_per_token + slot] * down_vals[i];
            }
        }
    }

    float actual[tokens * hidden];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT32, out, tokens * hidden);
    marmot_test_expect_close_array(actual, expected, tokens * hidden, 3e-2f);

    marmot_test_tensor_destroy_all(6, out, topk_weights, topk_ids, down_exps, up_exps, gate_exps);
    marmot_tensor_destroy(hidden_tensor);
}

static void
run_moe_experts_quantized_q8_0_grouped_case(marmot_test_env_t *env, const char *route_mode, float *actual_out) {
    enum {
        tokens = 5,
        hidden = 33,
        ff_length = 2,
        experts = 2,
        experts_per_token = 2,
    };

    float hidden_data[tokens * hidden];
    float gate_storage[experts * ff_length * hidden];
    float up_storage[experts * ff_length * hidden];
    float down_storage[experts * hidden * ff_length];
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t i = 0; i < hidden; ++i) {
            hidden_data[token * hidden + i] = ((float)((int)((token + 3) * (i + 5)) % 19) - 9.0f) * 0.0625f;
        }
    }
    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row = 0; row < ff_length; ++row) {
            for (size_t col = 0; col < hidden; ++col) {
                const size_t idx = expert * ff_length * hidden + row * hidden + col;
                gate_storage[idx] =
                    0.0125f * (float)(expert + 1) * (float)(row + 1) * (float)((int)((col + expert) % 9) - 4);
                up_storage[idx] = 0.0175f * (float)(expert + 2) * (float)((int)((col + row) % 7) - 3);
            }
        }
        for (size_t row = 0; row < hidden; ++row) {
            for (size_t col = 0; col < ff_length; ++col) {
                const size_t idx = expert * hidden * ff_length + row * ff_length + col;
                down_storage[idx] =
                    0.0225f * (float)(expert + 1) * (float)(col + 1) * (float)((int)((row + 2) % 13) - 6);
            }
        }
    }

    const marmot_int32_t topk_ids_data[tokens * experts_per_token] = {
        MARMOT_I32(0), MARMOT_I32(1), MARMOT_I32(1), MARMOT_I32(0), MARMOT_I32(0),
        MARMOT_I32(1), MARMOT_I32(1), MARMOT_I32(0), MARMOT_I32(0), MARMOT_I32(1),
    };
    const float topk_weights_data[tokens * experts_per_token] = {
        0.70f, 0.30f, 0.55f, 0.45f, 0.80f, 0.20f, 0.35f, 0.65f, 0.60f, 0.40f,
    };

    const size_t hidden_shape[] = {tokens, hidden};
    const size_t topk_shape[] = {tokens, experts_per_token};
    marmot_tensor_t *hidden_tensor = marmot_test_tensor_from_array(env, hidden_shape, 2, hidden_data);
    marmot_tensor_t *gate_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q8_0, gate_storage, ff_length, hidden, experts);
    marmot_tensor_t *up_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q8_0, up_storage, ff_length, hidden, experts);
    marmot_tensor_t *down_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q8_0, down_storage, hidden, ff_length, experts);
    marmot_tensor_t *topk_ids = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *topk_weights = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(hidden_tensor);
    assert_non_null(gate_exps);
    assert_non_null(up_exps);
    assert_non_null(down_exps);
    assert_non_null(topk_ids);
    assert_non_null(topk_weights);
    assert_non_null(out);

    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_ids, topk_ids_data, sizeof(topk_ids_data)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_weights, topk_weights_data, sizeof(topk_weights_data)),
        MARMOT_SUCCESS
    );

#ifdef __APPLE__
    const char *saved_route_mode = getenv("MARMOT_MOE_ROUTE_MODE");
    char saved_route_mode_buf[16] = {0};
    if (saved_route_mode != nullptr && saved_route_mode[0] != '\0') {
        snprintf(saved_route_mode_buf, sizeof(saved_route_mode_buf), "%s", saved_route_mode);
    }
    if (env->backend == MARMOT_BACKEND_METAL && route_mode != nullptr) {
        assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", route_mode, 1), 0);
    }
#endif

    marmot_error_t status = marmot_moe_experts(
        env->ctx,
        &(marmot_moe_experts_desc_t){
            .hidden_states = hidden_tensor,
            .gate_exps = gate_exps,
            .up_exps = up_exps,
            .down_exps = down_exps,
            .topk_ids = topk_ids,
            .topk_weights = topk_weights,
            .out = out,
            .ffn_type = MARMOT_FFN_SWIGLU,
            .weights_scale = 1.0f,
            .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        }
    );
    if (status != MARMOT_SUCCESS) {
        fail_msg(
            "grouped quantized q8_0 moe failed: %s (%s)", marmot_error_string(status), marmot_get_last_error_detail()
        );
    }

    float expected[tokens * hidden];
    memset(expected, 0, sizeof(expected));
    for (size_t token = 0; token < tokens; ++token) {
        const float *x = hidden_data + token * hidden;
        float *out_row = expected + token * hidden;
        for (size_t slot = 0; slot < experts_per_token; ++slot) {
            const int32_t expert_idx = topk_ids_data[token * experts_per_token + slot].value;
            float gate_vals[ff_length];
            float up_vals[ff_length];
            float fused[ff_length];
            float down_vals[hidden];
            matmul_row_vector(x, hidden, gate_storage + expert_idx * ff_length * hidden, ff_length, gate_vals);
            matmul_row_vector(x, hidden, up_storage + expert_idx * ff_length * hidden, ff_length, up_vals);
            for (size_t i = 0; i < ff_length; ++i) {
                fused[i] = test_moe_silu(gate_vals[i]) * up_vals[i];
            }
            matmul_row_vector(fused, ff_length, down_storage + expert_idx * hidden * ff_length, hidden, down_vals);
            for (size_t i = 0; i < hidden; ++i) {
                out_row[i] += topk_weights_data[token * experts_per_token + slot] * down_vals[i];
            }
        }
    }

    float actual[tokens * hidden];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT32, out, tokens * hidden);
    marmot_test_expect_close_array(actual, expected, tokens * hidden, 3e-2f);
    if (actual_out != nullptr) {
        memcpy(actual_out, actual, sizeof(actual));
    }

    marmot_test_tensor_destroy_all(6, out, topk_weights, topk_ids, down_exps, up_exps, gate_exps);
    marmot_tensor_destroy(hidden_tensor);

#ifdef __APPLE__
    if (env->backend == MARMOT_BACKEND_METAL && route_mode != nullptr) {
        if (saved_route_mode_buf[0] != '\0') {
            assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", saved_route_mode_buf, 1), 0);
        } else {
            assert_int_equal(unsetenv("MARMOT_MOE_ROUTE_MODE"), 0);
        }
    }
#endif
}

static void
run_moe_experts_quantized_q4k_q6k_decode_case(marmot_test_env_t *env, const char *route_mode, float *actual_out) {
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }

    enum {
        tokens = 1,
        hidden = 33,
        ff_length = 32,
        experts = 4,
        experts_per_token = 4,
    };

    float hidden_data[tokens * hidden];
    float gate_storage[experts * ff_length * hidden];
    float up_storage[experts * ff_length * hidden];
    float down_storage[experts * hidden * ff_length];
    for (size_t i = 0; i < hidden; ++i) {
        hidden_data[i] = ((float)((int)((i + 11) * 7) % 23) - 11.0f) * 0.0625f;
    }
    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row = 0; row < ff_length; ++row) {
            for (size_t col = 0; col < hidden; ++col) {
                const size_t idx = expert * ff_length * hidden + row * hidden + col;
                gate_storage[idx] = 0.0085f * (float)(expert + 1) * (float)((int)((row + col + expert) % 13) - 6);
                up_storage[idx] = 0.0105f * (float)(expert + 2) * (float)((int)((row * 3 + col + 1) % 11) - 5);
            }
        }
        for (size_t row = 0; row < hidden; ++row) {
            for (size_t col = 0; col < ff_length; ++col) {
                const size_t idx = expert * hidden * ff_length + row * ff_length + col;
                down_storage[idx] = 0.0060f * (float)(expert + 1) * (float)((int)((row + col * 2 + expert) % 17) - 8);
            }
        }
    }

    const marmot_int32_t topk_ids_data[] = {
        MARMOT_I32(0),
        MARMOT_I32(1),
        MARMOT_I32(2),
        MARMOT_I32(3),
    };
    const float topk_weights_data[] = {
        0.40f,
        0.30f,
        0.20f,
        0.10f,
    };

    const size_t hidden_shape[] = {tokens, hidden};
    const size_t topk_shape[] = {tokens, experts_per_token};
    marmot_tensor_t *hidden_tensor = marmot_test_tensor_from_array(env, hidden_shape, 2, hidden_data);
    marmot_tensor_t *gate_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q4_K, gate_storage, ff_length, hidden, experts);
    marmot_tensor_t *up_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q4_K, up_storage, ff_length, hidden, experts);
    marmot_tensor_t *down_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q6_K, down_storage, hidden, ff_length, experts);
    marmot_tensor_t *topk_ids = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *topk_weights = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(hidden_tensor);
    assert_non_null(gate_exps);
    assert_non_null(up_exps);
    assert_non_null(down_exps);
    assert_non_null(topk_ids);
    assert_non_null(topk_weights);
    assert_non_null(out);

    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_ids, topk_ids_data, sizeof(topk_ids_data)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_weights, topk_weights_data, sizeof(topk_weights_data)),
        MARMOT_SUCCESS
    );

#ifdef __APPLE__
    const char *saved_route_mode = getenv("MARMOT_MOE_ROUTE_MODE");
    char saved_route_mode_buf[16] = {0};
    if (saved_route_mode != nullptr && saved_route_mode[0] != '\0') {
        snprintf(saved_route_mode_buf, sizeof(saved_route_mode_buf), "%s", saved_route_mode);
    }
    if (route_mode != nullptr) {
        assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", route_mode, 1), 0);
    }
#endif

    marmot_error_t status = marmot_moe_experts(
        env->ctx,
        &(marmot_moe_experts_desc_t){
            .hidden_states = hidden_tensor,
            .gate_exps = gate_exps,
            .up_exps = up_exps,
            .down_exps = down_exps,
            .topk_ids = topk_ids,
            .topk_weights = topk_weights,
            .out = out,
            .ffn_type = MARMOT_FFN_SWIGLU,
            .weights_scale = 1.0f,
            .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        }
    );
    if (status != MARMOT_SUCCESS) {
        fail_msg(
            "decode grouped q4_k/q6_k moe failed: %s (%s)", marmot_error_string(status), marmot_get_last_error_detail()
        );
    }

    float expected[hidden];
    memset(expected, 0, sizeof(expected));
    for (size_t slot = 0; slot < experts_per_token; ++slot) {
        const int32_t expert_idx = topk_ids_data[slot].value;
        float gate_vals[ff_length];
        float up_vals[ff_length];
        float fused[ff_length];
        float down_vals[hidden];
        matmul_row_vector(hidden_data, hidden, gate_storage + expert_idx * ff_length * hidden, ff_length, gate_vals);
        matmul_row_vector(hidden_data, hidden, up_storage + expert_idx * ff_length * hidden, ff_length, up_vals);
        for (size_t i = 0; i < ff_length; ++i) {
            fused[i] = test_moe_silu(gate_vals[i]) * up_vals[i];
        }
        matmul_row_vector(fused, ff_length, down_storage + expert_idx * hidden * ff_length, hidden, down_vals);
        for (size_t i = 0; i < hidden; ++i) {
            expected[i] += topk_weights_data[slot] * down_vals[i];
        }
    }

    float actual[hidden];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT32, out, hidden);
    marmot_test_expect_close_array(actual, expected, hidden, 8e-2f);
    if (actual_out != nullptr) {
        memcpy(actual_out, actual, sizeof(actual));
    }

    marmot_test_tensor_destroy_all(6, out, topk_weights, topk_ids, down_exps, up_exps, gate_exps);
    marmot_tensor_destroy(hidden_tensor);

#ifdef __APPLE__
    if (route_mode != nullptr) {
        if (saved_route_mode_buf[0] != '\0') {
            assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", saved_route_mode_buf, 1), 0);
        } else {
            assert_int_equal(unsetenv("MARMOT_MOE_ROUTE_MODE"), 0);
        }
    }
#endif
}

static void
run_moe_experts_quantized_q4k_q6k_prefill_case(marmot_test_env_t *env, const char *route_mode, float *actual_out) {
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }

    enum {
        tokens = 4,
        hidden = 33,
        ff_length = 32,
        experts = 3,
        experts_per_token = 2,
    };

    float hidden_data[tokens * hidden];
    float gate_storage[experts * ff_length * hidden];
    float up_storage[experts * ff_length * hidden];
    float down_storage[experts * hidden * ff_length];
    for (size_t token = 0; token < tokens; ++token) {
        for (size_t i = 0; i < hidden; ++i) {
            hidden_data[token * hidden + i] = ((float)((int)((token + 5) * (i + 7)) % 29) - 14.0f) * 0.04375f;
        }
    }
    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t row = 0; row < ff_length; ++row) {
            for (size_t col = 0; col < hidden; ++col) {
                const size_t idx = expert * ff_length * hidden + row * hidden + col;
                gate_storage[idx] = 0.0075f * (float)(expert + 1) * (float)((int)((row + col * 2 + expert) % 17) - 8);
                up_storage[idx] = 0.0090f * (float)(expert + 2) * (float)((int)((row * 3 + col + expert) % 19) - 9);
            }
        }
        for (size_t row = 0; row < hidden; ++row) {
            for (size_t col = 0; col < ff_length; ++col) {
                const size_t idx = expert * hidden * ff_length + row * ff_length + col;
                down_storage[idx] = 0.0055f * (float)(expert + 1) * (float)((int)((row + col * 5 + expert) % 23) - 11);
            }
        }
    }

    const marmot_int32_t topk_ids_data[tokens * experts_per_token] = {
        MARMOT_I32(0), MARMOT_I32(1), MARMOT_I32(1), MARMOT_I32(2),
        MARMOT_I32(2), MARMOT_I32(0), MARMOT_I32(0), MARMOT_I32(2),
    };
    const float topk_weights_data[tokens * experts_per_token] = {
        0.65f, 0.35f, 0.55f, 0.45f, 0.70f, 0.30f, 0.40f, 0.60f,
    };

    const size_t hidden_shape[] = {tokens, hidden};
    const size_t topk_shape[] = {tokens, experts_per_token};
    marmot_tensor_t *hidden_tensor = marmot_test_tensor_from_array(env, hidden_shape, 2, hidden_data);
    marmot_tensor_t *gate_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q4_K, gate_storage, ff_length, hidden, experts);
    marmot_tensor_t *up_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q4_K, up_storage, ff_length, hidden, experts);
    marmot_tensor_t *down_exps =
        create_quantized_expert_tensor(env, MARMOT_QUANT_KIND_Q6_K, down_storage, hidden, ff_length, experts);
    marmot_tensor_t *topk_ids = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *topk_weights = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(hidden_tensor);
    assert_non_null(gate_exps);
    assert_non_null(up_exps);
    assert_non_null(down_exps);
    assert_non_null(topk_ids);
    assert_non_null(topk_weights);
    assert_non_null(out);

    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_ids, topk_ids_data, sizeof(topk_ids_data)), MARMOT_SUCCESS
    );
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, topk_weights, topk_weights_data, sizeof(topk_weights_data)),
        MARMOT_SUCCESS
    );

#ifdef __APPLE__
    const char *saved_route_mode = getenv("MARMOT_MOE_ROUTE_MODE");
    char saved_route_mode_buf[16] = {0};
    if (saved_route_mode != nullptr && saved_route_mode[0] != '\0') {
        snprintf(saved_route_mode_buf, sizeof(saved_route_mode_buf), "%s", saved_route_mode);
    }
    if (route_mode != nullptr) {
        assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", route_mode, 1), 0);
    }
#endif

    marmot_error_t status = marmot_moe_experts(
        env->ctx,
        &(marmot_moe_experts_desc_t){
            .hidden_states = hidden_tensor,
            .gate_exps = gate_exps,
            .up_exps = up_exps,
            .down_exps = down_exps,
            .topk_ids = topk_ids,
            .topk_weights = topk_weights,
            .out = out,
            .ffn_type = MARMOT_FFN_SWIGLU,
            .weights_scale = 1.0f,
            .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        }
    );
    if (status != MARMOT_SUCCESS) {
        fail_msg(
            "prefill grouped q4_k/q6_k moe failed: %s (%s)", marmot_error_string(status), marmot_get_last_error_detail()
        );
    }

    float expected[tokens * hidden];
    memset(expected, 0, sizeof(expected));
    for (size_t token = 0; token < tokens; ++token) {
        const float *x = hidden_data + token * hidden;
        float *out_row = expected + token * hidden;
        for (size_t slot = 0; slot < experts_per_token; ++slot) {
            const int32_t expert_idx = topk_ids_data[token * experts_per_token + slot].value;
            float gate_vals[ff_length];
            float up_vals[ff_length];
            float fused[ff_length];
            float down_vals[hidden];
            matmul_row_vector(x, hidden, gate_storage + expert_idx * ff_length * hidden, ff_length, gate_vals);
            matmul_row_vector(x, hidden, up_storage + expert_idx * ff_length * hidden, ff_length, up_vals);
            for (size_t i = 0; i < ff_length; ++i) {
                fused[i] = test_moe_silu(gate_vals[i]) * up_vals[i];
            }
            matmul_row_vector(fused, ff_length, down_storage + expert_idx * hidden * ff_length, hidden, down_vals);
            for (size_t i = 0; i < hidden; ++i) {
                out_row[i] += topk_weights_data[token * experts_per_token + slot] * down_vals[i];
            }
        }
    }

    float actual[tokens * hidden];
    marmot_test_fetch_span(env, actual, MARMOT_DTYPE_FLOAT32, out, tokens * hidden);
    marmot_test_expect_close_array(actual, expected, tokens * hidden, 9e-2f);
    if (actual_out != nullptr) {
        memcpy(actual_out, actual, sizeof(actual));
    }

    marmot_test_tensor_destroy_all(6, out, topk_weights, topk_ids, down_exps, up_exps, gate_exps);
    marmot_tensor_destroy(hidden_tensor);

#ifdef __APPLE__
    if (route_mode != nullptr) {
        if (saved_route_mode_buf[0] != '\0') {
            assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", saved_route_mode_buf, 1), 0);
        } else {
            assert_int_equal(unsetenv("MARMOT_MOE_ROUTE_MODE"), 0);
        }
    }
#endif
}

static void exercise_moe_experts_quantized_q4k_q6k_decode_grouped(marmot_test_env_t *env) {
    run_moe_experts_quantized_q4k_q6k_decode_case(env, nullptr, nullptr);
}

static void exercise_moe_experts_quantized_q4k_q6k_prefill_grouped(marmot_test_env_t *env) {
    run_moe_experts_quantized_q4k_q6k_prefill_case(env, nullptr, nullptr);
}

static void exercise_moe_experts_quantized_q8_0_grouped(marmot_test_env_t *env) {
    run_moe_experts_quantized_q8_0_grouped_case(env, nullptr, nullptr);
}

static void test_topk_default(void **state) {
    exercise_topk((marmot_test_env_t *)(*state));
}

static void test_topk_f16_default(void **state) {
    exercise_topk_f16((marmot_test_env_t *)(*state));
}

static void test_moe_experts_default(void **state) {
    exercise_moe_experts((marmot_test_env_t *)(*state));
}

static void test_moe_experts_f16_default(void **state) {
    exercise_moe_experts_f16((marmot_test_env_t *)(*state));
}

static void test_moe_experts_renorm_default(void **state) {
    exercise_moe_experts_renorm((marmot_test_env_t *)(*state));
}

static void test_moe_experts_quantized_q8_0_default(void **state) {
    exercise_moe_experts_quantized_q8_0((marmot_test_env_t *)(*state));
}

static void test_moe_experts_quantized_q8_0_grouped_default(void **state) {
    exercise_moe_experts_quantized_q8_0_grouped((marmot_test_env_t *)(*state));
}

static void test_moe_experts_quantized_q8_0_grouped_route_modes_match(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }

    enum {
        tokens = 5,
        hidden = 33,
    };
    float host_actual[tokens * hidden];
    float gpu_actual[tokens * hidden];

    run_moe_experts_quantized_q8_0_grouped_case(env, "host", host_actual);
    run_moe_experts_quantized_q8_0_grouped_case(env, "gpu", gpu_actual);
    marmot_test_expect_close_array(gpu_actual, host_actual, tokens * hidden, 3e-2f);
}

static void test_moe_experts_quantized_q4k_q6k_decode_grouped_default(void **state) {
    exercise_moe_experts_quantized_q4k_q6k_decode_grouped((marmot_test_env_t *)(*state));
}

static void test_moe_experts_quantized_q4k_q6k_decode_grouped_route_modes_match(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }

    enum {
        hidden = 33,
    };
    float host_actual[hidden];
    float gpu_actual[hidden];

    run_moe_experts_quantized_q4k_q6k_decode_case(env, "host", host_actual);
    run_moe_experts_quantized_q4k_q6k_decode_case(env, "gpu", gpu_actual);
    marmot_test_expect_close_array(gpu_actual, host_actual, hidden, 8e-2f);
}

static void test_moe_experts_quantized_q4k_q6k_prefill_grouped_default(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }
    exercise_moe_experts_quantized_q4k_q6k_prefill_grouped(env);
}

static void test_moe_experts_quantized_q4k_q6k_prefill_grouped_route_modes_match(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }

    enum {
        tokens = 4,
        hidden = 33,
    };
    float host_actual[tokens * hidden];
    float gpu_actual[tokens * hidden];

    run_moe_experts_quantized_q4k_q6k_prefill_case(env, "host", host_actual);
    run_moe_experts_quantized_q4k_q6k_prefill_case(env, "gpu", gpu_actual);
    marmot_test_expect_close_array(gpu_actual, host_actual, tokens * hidden, 9e-2f);
}

static void test_moe_experts_duplicate_routes_host(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }
    run_moe_experts_duplicate_route_case(env, "host", nullptr);
}

static void test_moe_experts_duplicate_routes_route_modes_match(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }

    float host_actual[2];
    float gpu_actual[2];

    run_moe_experts_duplicate_route_case(env, "host", host_actual);
    run_moe_experts_duplicate_route_case(env, "gpu", gpu_actual);
    marmot_test_expect_close_array(gpu_actual, host_actual, 2, 1e-5f);
}

static void test_topk_to_moe_route_modes_match(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
    }

    const size_t hidden_shape[] = {2, 2};
    const size_t gate_shape[] = {2, 2, 4};
    const size_t router_shape[] = {2, 4};
    const size_t topk_shape[] = {2, 2};
    const float hidden_data[] = {
        0.75f,
        -0.25f,
        0.10f,
        0.90f,
    };
    const float gate_exps_storage[] = {
        0.20f, -0.10f, 0.35f,  0.40f, 0.50f, 0.15f,  -0.25f, 0.30f,
        0.10f, 0.45f,  -0.05f, 0.60f, 0.25f, -0.20f, 0.55f,  0.35f,
    };
    const float up_exps_storage[] = {
        0.40f, 0.30f,  -0.10f, 0.20f, 0.15f, 0.50f, 0.25f,  -0.05f,
        0.35f, -0.15f, 0.45f,  0.10f, 0.30f, 0.20f, -0.25f, 0.55f,
    };
    const float down_exps_storage[] = {
        0.60f, -0.20f, 0.15f, 0.35f, -0.10f, 0.45f, 0.50f, 0.25f,
        0.30f, -0.05f, 0.40f, 0.20f, -0.15f, 0.55f, 0.10f, 0.65f,
    };
    const float router_cases[][8] = {
        {
            0.95f,
            0.80f,
            0.25f,
            0.10f,
            0.15f,
            0.75f,
            0.65f,
            0.05f,
        },
        {
            0.10f,
            0.35f,
            0.92f,
            0.60f,
            0.88f,
            0.15f,
            0.40f,
            0.73f,
        },
    };

    marmot_tensor_t *hidden = marmot_test_tensor_from_array(env, hidden_shape, 2, hidden_data);
    marmot_tensor_t *gate_exps = marmot_test_tensor_from_array(env, gate_shape, 3, gate_exps_storage);
    marmot_tensor_t *up_exps = marmot_test_tensor_from_array(env, gate_shape, 3, up_exps_storage);
    marmot_tensor_t *down_exps = marmot_test_tensor_from_array(env, gate_shape, 3, down_exps_storage);
    marmot_tensor_t *router_logits = marmot_tensor_create(env->ctx, router_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *topk_values = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *topk_ids = marmot_tensor_create(env->ctx, topk_shape, 2, MARMOT_DTYPE_INT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(hidden);
    assert_non_null(gate_exps);
    assert_non_null(up_exps);
    assert_non_null(down_exps);
    assert_non_null(router_logits);
    assert_non_null(topk_values);
    assert_non_null(topk_ids);
    assert_non_null(out);

#ifdef __APPLE__
    const char *saved_route_mode = getenv("MARMOT_MOE_ROUTE_MODE");
    char saved_route_mode_buf[16] = {0};
    if (saved_route_mode != nullptr && saved_route_mode[0] != '\0') {
        snprintf(saved_route_mode_buf, sizeof(saved_route_mode_buf), "%s", saved_route_mode);
    }
#endif

    for (size_t case_idx = 0; case_idx < sizeof(router_cases) / sizeof(router_cases[0]); ++case_idx) {
        marmot_test_convert_f32_span(env, router_logits, router_cases[case_idx], 8);
        assert_int_equal(
            marmot_topk(
                env->ctx,
                &(marmot_topk_desc_t){
                    .x = router_logits,
                    .values_out = topk_values,
                    .indices_out = topk_ids,
                    .axis = -1,
                    .k = 2,
                }
            ),
            MARMOT_SUCCESS
        );

        const marmot_int32_t *topk_indices = marmot_tensor_data_i32(env->ctx, topk_ids);
        assert_non_null(topk_indices);
        for (size_t token = 0; token < 2; ++token) {
            assert_true(topk_indices[token * 2].value != topk_indices[token * 2 + 1].value);
        }

        float host_actual[4];
        float gpu_actual[4];

        assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", "host", 1), 0);
        assert_int_equal(
            marmot_moe_experts(
                env->ctx,
                &(marmot_moe_experts_desc_t){
                    .hidden_states = hidden,
                    .gate_exps = gate_exps,
                    .up_exps = up_exps,
                    .down_exps = down_exps,
                    .topk_ids = topk_ids,
                    .topk_weights = topk_values,
                    .out = out,
                    .ffn_type = MARMOT_FFN_SWIGLU,
                    .weights_scale = 1.0f,
                    .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED,
                }
            ),
            MARMOT_SUCCESS
        );
        marmot_test_fetch_span(env, host_actual, MARMOT_DTYPE_FLOAT32, out, 4);

        assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", "gpu", 1), 0);
        assert_int_equal(
            marmot_moe_experts(
                env->ctx,
                &(marmot_moe_experts_desc_t){
                    .hidden_states = hidden,
                    .gate_exps = gate_exps,
                    .up_exps = up_exps,
                    .down_exps = down_exps,
                    .topk_ids = topk_ids,
                    .topk_weights = topk_values,
                    .out = out,
                    .ffn_type = MARMOT_FFN_SWIGLU,
                    .weights_scale = 1.0f,
                    .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_RENORMALIZE_SELECTED,
                }
            ),
            MARMOT_SUCCESS
        );
        marmot_test_fetch_span(env, gpu_actual, MARMOT_DTYPE_FLOAT32, out, 4);
        marmot_test_expect_close_array(gpu_actual, host_actual, 4, 1e-5f);
    }

    marmot_test_tensor_destroy_all(7, out, topk_ids, topk_values, router_logits, down_exps, up_exps, gate_exps);
    marmot_tensor_destroy(hidden);

#ifdef __APPLE__
    if (saved_route_mode_buf[0] != '\0') {
        assert_int_equal(setenv("MARMOT_MOE_ROUTE_MODE", saved_route_mode_buf, 1), 0);
    } else {
        assert_int_equal(unsetenv("MARMOT_MOE_ROUTE_MODE"), 0);
    }
#endif
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_topk_scalar(void **state) {
    marmot_test_run_with_cpu_scalar((marmot_test_env_t *)(*state), exercise_topk);
}

static void test_topk_f16_scalar(void **state) {
    marmot_test_run_with_cpu_scalar((marmot_test_env_t *)(*state), exercise_topk_f16);
}

static void test_moe_experts_scalar(void **state) {
    marmot_test_run_with_cpu_scalar((marmot_test_env_t *)(*state), exercise_moe_experts);
}

static void test_moe_experts_f16_scalar(void **state) {
    marmot_test_run_with_cpu_scalar((marmot_test_env_t *)(*state), exercise_moe_experts_f16);
}

static void test_moe_experts_renorm_scalar(void **state) {
    marmot_test_run_with_cpu_scalar((marmot_test_env_t *)(*state), exercise_moe_experts_renorm);
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_topk_default, marmot_test_backend_setup, marmot_test_backend_teardown),
        cmocka_unit_test_setup_teardown(test_topk_f16_default, marmot_test_backend_setup, marmot_test_backend_teardown),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_f16_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_renorm_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_quantized_q8_0_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_quantized_q8_0_grouped_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_quantized_q8_0_grouped_route_modes_match, marmot_test_backend_setup,
            marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_quantized_q4k_q6k_decode_grouped_default, marmot_test_backend_setup,
            marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_quantized_q4k_q6k_decode_grouped_route_modes_match, marmot_test_backend_setup,
            marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_quantized_q4k_q6k_prefill_grouped_default, marmot_test_backend_setup,
            marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_quantized_q4k_q6k_prefill_grouped_route_modes_match, marmot_test_backend_setup,
            marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_duplicate_routes_host, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_duplicate_routes_route_modes_match, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_topk_to_moe_route_modes_match, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_topk_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_topk_f16_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_f16_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_moe_experts_renorm_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
