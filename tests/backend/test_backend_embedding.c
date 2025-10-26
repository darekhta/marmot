#include "marmot/quant_block.h"

#include "backend/test_backend_utils.h"
#include "utils/dtype_ref.h"

static void fetch_embedding(const marmot_test_env_t *env, const marmot_tensor_t *tensor, float *dst, size_t count) {
    assert_non_null(env);
    assert_non_null(tensor);
    assert_non_null(dst);
    const size_t bytes = count * sizeof(float);
    marmot_error_t err = marmot_tensor_copy_to_host_buffer(env->ctx, tensor, dst, bytes);
    assert_int_equal(err, MARMOT_SUCCESS);
}

static void
fetch_embedding_as_f32(const marmot_test_env_t *env, const marmot_tensor_t *tensor, float *dst, size_t count) {
    assert_non_null(env);
    assert_non_null(tensor);
    assert_non_null(dst);
    const void *data = marmot_tensor_data(env->ctx, (marmot_tensor_t *)tensor);
    assert_non_null(data);
    if (tensor->dtype == MARMOT_DTYPE_FLOAT32) {
        memcpy(dst, data, count * sizeof(float));
        return;
    }
    if (tensor->dtype == MARMOT_DTYPE_FLOAT64) {
        const double *src = (const double *)data;
        for (size_t i = 0; i < count; ++i) {
            dst[i] = (float)src[i];
        }
        return;
    }
    if (tensor->dtype == MARMOT_DTYPE_FLOAT16) {
        const marmot_float16_t *src = (const marmot_float16_t *)data;
        for (size_t i = 0; i < count; ++i) {
            dst[i] = marmot_f16_to_f32_ref(src[i]);
        }
        return;
    }
    if (tensor->dtype == MARMOT_DTYPE_BFLOAT16) {
        const marmot_bfloat16_t *src = (const marmot_bfloat16_t *)data;
        for (size_t i = 0; i < count; ++i) {
            dst[i] = marmot_bf16_to_f32_ref(src[i]);
        }
        return;
    }
    fail_msg("Unsupported out dtype for fetch_embedding_as_f32: %d", tensor->dtype);
}

static void test_embedding_float_basic(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);

    if (env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    const size_t vocab = 4;
    const size_t dim = 3;
    const size_t weight_shape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(weights);

    float *w_data = (float *)weights->data;
    const float seed[] = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
    };
    memcpy(w_data, seed, sizeof(seed));
    marmot_test_commit_tensor(env, weights);

    const size_t token_shape[] = {2, 2};
    marmot_tensor_t *token_ids = marmot_tensor_create(env->ctx, token_shape, 2, MARMOT_DTYPE_INT32);
    assert_non_null(token_ids);
    int32_t *t_data = (int32_t *)token_ids->data;
    const int32_t indices[] = {2, 0, 1, 3};
    memcpy(t_data, indices, sizeof(indices));
    marmot_test_commit_tensor(env, token_ids);
    marmot_test_commit_tensor(env, token_ids);

    const size_t out_shape[] = {2, 2, dim};
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 3, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_embedding_desc_t desc = marmot_embedding_desc_default();
    desc.weights = weights;
    desc.token_ids = token_ids;
    desc.out = out;
    desc.dtype_out = MARMOT_DTYPE_FLOAT32;
    desc.padding_id = -1;
    desc.bounds_check = true;

    marmot_test_commit_tensor(env, weights);
    marmot_error_t err = marmot_embedding_lookup(env->ctx, &desc);
    assert_int_equal(err, MARMOT_SUCCESS);

    float out_buf[4 * dim];
    fetch_embedding(env, out, out_buf, 4 * dim);

    const float expected[] = {
        6.0f, 7.0f, 8.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 9.0f, 10.0f, 11.0f,
    };
    marmot_test_expect_close_array(out_buf, expected, 4 * dim, 1e-6f);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(token_ids);
    marmot_tensor_destroy(weights);
}

static void test_embedding_scale(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);

    if (env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    const size_t vocab = 2;
    const size_t dim = 2;
    const size_t weight_shape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(weights);
    float *w_data = (float *)weights->data;
    const float seed[] = {1.0f, -2.0f, 3.0f, -4.0f};
    memcpy(w_data, seed, sizeof(seed));
    marmot_test_commit_tensor(env, weights);

    const size_t token_shape[] = {2};
    marmot_tensor_t *token_ids = marmot_tensor_create(env->ctx, token_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(token_ids);
    int32_t *t_data = (int32_t *)token_ids->data;
    const int32_t indices[] = {0, 1};
    memcpy(t_data, indices, sizeof(indices));
    marmot_test_commit_tensor(env, token_ids);

    const size_t out_shape[] = {2, dim};
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_embedding_gather_desc_t desc = marmot_embedding_gather_desc_default();
    desc.weights = weights;
    desc.token_ids = token_ids;
    desc.out = out;
    desc.dtype_out = MARMOT_DTYPE_FLOAT32;
    desc.scale = 2.0f;
    desc.padding_id = -1;
    desc.bounds_check = true;

    marmot_error_t err = marmot_embedding_gather(env->ctx, &desc);
    assert_int_equal(err, MARMOT_SUCCESS);

    float out_buf[4];
    fetch_embedding(env, out, out_buf, 4);
    const float expected[] = {2.0f, -4.0f, 6.0f, -8.0f};
    marmot_test_expect_close_array(out_buf, expected, 4, 1e-6f);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(token_ids);
    marmot_tensor_destroy(weights);
}

static void test_embedding_device_token_ids_stale_host(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);

    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    const size_t vocab = 4;
    const size_t dim = 2;

    const size_t weight_shape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(weights);
    float *w_data = (float *)weights->data;
    for (size_t i = 0; i < vocab; ++i) {
        w_data[i * dim] = (float)(i * 10u + 1u);
        w_data[i * dim + 1] = (float)(i * 10u + 2u);
    }
    marmot_test_commit_tensor(env, weights);

    const size_t token_shape[] = {1};
    marmot_tensor_t *token_ids = marmot_tensor_create(env->ctx, token_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(token_ids);
    ((int32_t *)token_ids->data)[0] = 0;
    marmot_test_commit_tensor(env, token_ids);

    marmot_tensor_t *input = marmot_tensor_create(env->ctx, token_shape, 1, MARMOT_DTYPE_UINT64);
    assert_non_null(input);
    ((uint64_t *)input->data)[0] = 2;
    marmot_test_commit_tensor(env, input);

    marmot_tensor_t *indices = marmot_tensor_create(env->ctx, token_shape, 1, MARMOT_DTYPE_UINT32);
    assert_non_null(indices);
    ((uint32_t *)indices->data)[0] = 0;
    marmot_test_commit_tensor(env, indices);

    marmot_error_t err = marmot_scatter_u64_to_i32(env->ctx, input, indices, token_ids);
    assert_int_equal(err, MARMOT_SUCCESS);

    assert_int_equal(((int32_t *)token_ids->data)[0], 0);
    assert_int_equal((int)token_ids->memory_location, (int)MARMOT_MEMORY_DEVICE);
    assert_true(token_ids->needs_sync);

    const size_t out_shape[] = {1, dim};
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_embedding_gather_desc_t desc = marmot_embedding_gather_desc_default();
    desc.weights = weights;
    desc.token_ids = token_ids;
    desc.out = out;
    desc.dtype_out = MARMOT_DTYPE_FLOAT32;
    desc.padding_id = -1;
    desc.bounds_check = false;

    err = marmot_embedding_gather(env->ctx, &desc);
    assert_int_equal(err, MARMOT_SUCCESS);

    float out_buf[2];
    fetch_embedding(env, out, out_buf, 2);

    const float expected[] = {21.0f, 22.0f};
    marmot_test_expect_close_array(out_buf, expected, 2, 1e-6f);

    marmot_test_tensor_destroy_all(5, out, indices, input, token_ids, weights);
}

static void run_embedding_float64_basic(marmot_test_env_t *env) {
    if (env == nullptr) {
        return;
    }

    const size_t vocab = 4;
    const size_t dim = 3;
    const size_t weight_shape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT64);
    assert_non_null(weights);

    double *w_data = (double *)weights->data;
    const double seed[] = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
    };
    memcpy(w_data, seed, sizeof(seed));

    const size_t token_shape[] = {2, 2};
    marmot_tensor_t *token_ids = marmot_tensor_create(env->ctx, token_shape, 2, MARMOT_DTYPE_INT32);
    assert_non_null(token_ids);
    int32_t *t_data = (int32_t *)token_ids->data;
    const int32_t indices[] = {2, 0, 1, 3};
    memcpy(t_data, indices, sizeof(indices));

    const size_t out_shape[] = {2, 2, dim};
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 3, MARMOT_DTYPE_FLOAT64);
    assert_non_null(out);

    marmot_embedding_desc_t desc = marmot_embedding_desc_default();
    desc.weights = weights;
    desc.token_ids = token_ids;
    desc.out = out;
    desc.dtype_out = MARMOT_DTYPE_FLOAT64;
    desc.padding_id = -1;
    desc.bounds_check = true;

    marmot_error_t err = marmot_embedding_lookup(env->ctx, &desc);
    assert_int_equal(err, MARMOT_SUCCESS);

    float out_buf[4 * dim];
    fetch_embedding_as_f32(env, out, out_buf, 4 * dim);

    const float expected[] = {
        6.0f, 7.0f, 8.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 9.0f, 10.0f, 11.0f,
    };
    marmot_test_expect_close_array(out_buf, expected, 4 * dim, 1e-6f);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(token_ids);
    marmot_tensor_destroy(weights);
}

static void test_embedding_float64_basic(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);

    if (env->backend == MARMOT_BACKEND_METAL) {
        marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx == nullptr) {
            skip();
            return;
        }
        marmot_test_env_t cpu_env = {
            .backend = MARMOT_BACKEND_CPU,
            .ctx = cpu_ctx,
        };
        run_embedding_float64_basic(&cpu_env);
        marmot_destroy(cpu_ctx);
        return;
    }
    if (env->backend != MARMOT_BACKEND_CPU) {
        skip();
        return;
    }

    run_embedding_float64_basic(env);
}

static void test_embedding_padding_and_bounds(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);

    if (env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    const size_t vocab = 3;
    const size_t dim = 2;
    const size_t weight_shape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(weights);
    float *w_data = (float *)weights->data;
    const float base[] = {
        0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f,
    };
    memcpy(w_data, base, sizeof(base));
    marmot_test_commit_tensor(env, weights);

    const size_t token_shape[] = {1, 4};
    marmot_tensor_t *token_ids = marmot_tensor_create(env->ctx, token_shape, 2, MARMOT_DTYPE_INT32);
    assert_non_null(token_ids);
    int32_t *t_data = (int32_t *)token_ids->data;
    const int32_t ids_padding[] = {0, 2, 99, 1};
    memcpy(t_data, ids_padding, sizeof(ids_padding));
    marmot_test_commit_tensor(env, token_ids);

    const size_t out_shape[] = {1, 4, dim};
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 3, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    marmot_embedding_desc_t desc = marmot_embedding_desc_default();
    desc.weights = weights;
    desc.token_ids = token_ids;
    desc.out = out;
    desc.dtype_out = MARMOT_DTYPE_FLOAT32;
    desc.padding_id = 2;
    desc.bounds_check = false;

    marmot_error_t err = marmot_embedding_lookup(env->ctx, &desc);
    assert_int_equal(err, MARMOT_SUCCESS);

    float out_buf[4 * dim];
    fetch_embedding(env, out, out_buf, 4 * dim);

    const float expected[] = {
        0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.5f, -1.5f,
    };
    marmot_test_expect_close_array(out_buf, expected, 4 * dim, 1e-6f);

    // Now enable bounds checking to ensure error is raised for invalid id.
    desc.bounds_check = true;
    err = marmot_embedding_lookup(env->ctx, &desc);
    assert_int_equal(err, MARMOT_ERROR_INVALID_ARGUMENT);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(token_ids);
    marmot_tensor_destroy(weights);
}

// Removed duplicate Q4_0 bespoke test; covered by golden fixtures

[[maybe_unused]] static void test_embedding_quantized_variants(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);
    if (env->backend != MARMOT_BACKEND_CPU && env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    const marmot_quant_kind_t kinds[] = {
        MARMOT_QUANT_KIND_Q4_1,
        MARMOT_QUANT_KIND_Q8_0,
    };

    const size_t vocab = 3;
    const size_t dim = 64; // multiple blocks
    const size_t weight_shape[] = {vocab, dim};

    float base[vocab * dim];
    for (size_t i = 0; i < vocab * dim; ++i) {
        base[i] = (float)i * 0.125f - 1.0f;
    }

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);

    for (size_t k = 0; k < sizeof(kinds) / sizeof(kinds[0]); ++k) {
        marmot_quant_kind_t kind = kinds[k];

        marmot_tensor_t *weights_cpu = marmot_tensor_create(nullptr, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
        assert_non_null(weights_cpu);
        memcpy(weights_cpu->data, base, sizeof(base));

        marmot_tensor_t *weights_q_cpu = marmot_tensor_create_quantized(nullptr, weight_shape, 2, kind);
        assert_non_null(weights_q_cpu);

        marmot_error_t qerr = MARMOT_SUCCESS;
        switch (kind) {
        case MARMOT_QUANT_KIND_Q4_1:
            qerr = marmot_quantize_q4_1(cpu_ctx, weights_cpu, weights_q_cpu);
            break;
        case MARMOT_QUANT_KIND_Q5_0:
            qerr = marmot_quantize_q5_0(cpu_ctx, weights_cpu, weights_q_cpu);
            break;
        case MARMOT_QUANT_KIND_Q5_1:
            qerr = marmot_quantize_q5_1(cpu_ctx, weights_cpu, weights_q_cpu);
            break;
        case MARMOT_QUANT_KIND_Q8_0:
            qerr = marmot_quantize_q8_0(cpu_ctx, weights_cpu, weights_q_cpu);
            break;
        default:
            qerr = MARMOT_ERROR_INVALID_ARGUMENT;
            break;
        }
        assert_int_equal(qerr, MARMOT_SUCCESS);

        marmot_tensor_t *weights_q = marmot_tensor_create_quantized(env->ctx, weight_shape, 2, kind);
        assert_non_null(weights_q);
        const size_t weight_bytes = marmot_tensor_size_bytes(weights_q_cpu);
        memcpy(weights_q->data, weights_q_cpu->data, weight_bytes);

        const size_t token_shape[] = {1, 3};
        marmot_tensor_t *token_ids = marmot_tensor_create(env->ctx, token_shape, 2, MARMOT_DTYPE_INT32);
        assert_non_null(token_ids);
        int32_t *t_data = (int32_t *)token_ids->data;
        const int32_t ids[] = {0, 2, 1};
        memcpy(t_data, ids, sizeof(ids));

        const size_t out_shape[] = {1, 3, dim};
        marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 3, MARMOT_DTYPE_FLOAT32);
        assert_non_null(out);

        marmot_embedding_desc_t desc = marmot_embedding_desc_default();
        desc.weights = weights_q;
        desc.token_ids = token_ids;
        desc.out = out;
        desc.dtype_out = MARMOT_DTYPE_FLOAT32;
        desc.bounds_check = true;

        marmot_error_t err = marmot_embedding_lookup(env->ctx, &desc);
        if (err != MARMOT_SUCCESS) {
            printf("embedding_lookup variants(kind=%d) failed: %s\n", (int)kind, marmot_error_string(err));
        }
        assert_int_equal(err, MARMOT_SUCCESS);

        float out_buf[3 * dim];
        fetch_embedding_as_f32(env, out, out_buf, 3 * dim);

        float expected_buf[3 * dim];
        const size_t blocks_per_row = (dim + 31) / 32;
        for (size_t row = 0; row < 3; ++row) {
            const size_t token_id = (size_t)ids[row];
            const uint8_t *row_bytes = nullptr;
            switch (kind) {
            case MARMOT_QUANT_KIND_Q4_1:
                row_bytes =
                    (const uint8_t *)((const marmot_q4_1_block_t *)weights_q_cpu->data + token_id * blocks_per_row);
                break;
            case MARMOT_QUANT_KIND_Q5_0:
                row_bytes =
                    (const uint8_t *)((const marmot_q5_0_block_t *)weights_q_cpu->data + token_id * blocks_per_row);
                break;
            case MARMOT_QUANT_KIND_Q5_1:
                row_bytes =
                    (const uint8_t *)((const marmot_q5_1_block_t *)weights_q_cpu->data + token_id * blocks_per_row);
                break;
            case MARMOT_QUANT_KIND_Q8_0:
                row_bytes =
                    (const uint8_t *)((const marmot_q8_0_block_t *)weights_q_cpu->data + token_id * blocks_per_row);
                break;
            default:
                row_bytes = (const uint8_t *)weights_q_cpu->data;
                break;
            }
            for (size_t col = 0; col < dim; ++col) {
                const size_t block_index = col / 32;
                const size_t offset = col % 32;
                switch (kind) {
                case MARMOT_QUANT_KIND_Q4_1: {
                    const marmot_q4_1_block_t *blk = ((const marmot_q4_1_block_t *)row_bytes) + block_index;
                    const uint8_t packed = blk->qs[offset >> 1];
                    const int q = (offset & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
                    const float scale = marmot_f16_to_f32_ref(blk->scale);
                    const float minv = marmot_f16_to_f32_ref(blk->min);
                    expected_buf[row * dim + col] = scale * (float)q + minv;
                    break;
                }
                case MARMOT_QUANT_KIND_Q5_0: {
                    const marmot_q5_0_block_t *blk = ((const marmot_q5_0_block_t *)row_bytes) + block_index;
                    const size_t byte_index = offset % 16;
                    const uint8_t packed = blk->qs[byte_index];
                    uint32_t qh_bits;
                    memcpy(&qh_bits, blk->qh, sizeof(uint32_t));
                    int q;
                    if (offset < 16) {
                        uint8_t lo = (uint8_t)(packed & 0x0f);
                        lo |= (uint8_t)(((qh_bits >> byte_index) & 0x1u) << 4);
                        q = (int)lo - 16;
                    } else {
                        uint8_t hi = (uint8_t)(packed >> 4);
                        hi |= (uint8_t)(((qh_bits >> (byte_index + 16)) & 0x1u) << 4);
                        q = (int)hi - 16;
                    }
                    const float scale = marmot_f16_to_f32_ref(blk->scale);
                    expected_buf[row * dim + col] = scale * (float)q;
                    break;
                }
                case MARMOT_QUANT_KIND_Q5_1: {
                    const marmot_q5_1_block_t *blk = ((const marmot_q5_1_block_t *)row_bytes) + block_index;
                    const size_t byte_index = offset % 16;
                    const uint8_t packed = blk->qs[byte_index];
                    uint32_t qh_bits;
                    memcpy(&qh_bits, blk->qh, sizeof(uint32_t));
                    int q;
                    if (offset < 16) {
                        uint8_t lo = (uint8_t)(packed & 0x0f);
                        lo |= (uint8_t)(((qh_bits >> byte_index) & 0x1u) << 4);
                        q = (int)lo;
                    } else {
                        uint8_t hi = (uint8_t)(packed >> 4);
                        hi |= (uint8_t)(((qh_bits >> (byte_index + 16)) & 0x1u) << 4);
                        q = (int)hi;
                    }
                    const float scale = marmot_f16_to_f32_ref(blk->scale);
                    const float minv = marmot_f16_to_f32_ref(blk->min);
                    expected_buf[row * dim + col] = scale * (float)q + minv;
                    break;
                }
                case MARMOT_QUANT_KIND_Q8_0: {
                    const marmot_q8_0_block_t *blk = ((const marmot_q8_0_block_t *)row_bytes) + block_index;
                    const int8_t q = blk->qs[offset];
                    const float scale = marmot_f16_to_f32_ref(blk->scale);
                    expected_buf[row * dim + col] = scale * (float)q;
                    break;
                }
                default:
                    expected_buf[row * dim + col] = 0.0f;
                    break;
                }
            }
        }

        marmot_test_expect_close_array(out_buf, expected_buf, 3 * dim, 7e-3f);

        marmot_tensor_destroy(out);
        marmot_tensor_destroy(token_ids);
        marmot_tensor_destroy(weights_q);
        marmot_tensor_destroy(weights_q_cpu);
        marmot_tensor_destroy(weights_cpu);
    }

    marmot_destroy(cpu_ctx);
}

static void test_embedding_token_id_dtypes(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);
    marmot_context_t *cpu_ctx = nullptr;
    marmot_test_env_t cpu_env;
    if (env->backend != MARMOT_BACKEND_CPU) {
        cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx == nullptr) {
            skip();
            return;
        }
        cpu_env.backend = MARMOT_BACKEND_CPU;
        cpu_env.ctx = cpu_ctx;
        env = &cpu_env;
    }

    const size_t vocab = 4;
    const size_t dim = 5;
    const size_t weight_shape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(weights);
    float *w_data = (float *)weights->data;
    for (size_t i = 0; i < vocab * dim; ++i)
        w_data[i] = (float)(i + 1);

    const int32_t ids32[] = {3, 1, 0, 2, 1};
    const size_t L = sizeof(ids32) / sizeof(ids32[0]);
    const size_t out_shape[] = {L, dim};

    // Baseline with INT32
    marmot_tensor_t *tok32 = marmot_tensor_create(env->ctx, &L, 1, MARMOT_DTYPE_INT32);
    memcpy(tok32->data, ids32, sizeof(ids32));
    marmot_tensor_t *out32 = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_embedding_desc_t desc = marmot_embedding_desc_default();
    desc.weights = weights;
    desc.token_ids = tok32;
    desc.out = out32;
    desc.dtype_out = MARMOT_DTYPE_FLOAT32;
    assert_int_equal(marmot_embedding_lookup(env->ctx, &desc), MARMOT_SUCCESS);
    float ref[L * dim];
    fetch_embedding_as_f32(env, out32, ref, L * dim);

    // Check UINT32, INT16, UINT16
    struct {
        marmot_dtype_t dtype;
        const char *label;
    } cases[] = {
        {MARMOT_DTYPE_UINT32, "u32"},
        {MARMOT_DTYPE_INT16, "i16"},
        {MARMOT_DTYPE_UINT16, "u16"},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        marmot_tensor_t *tok = marmot_tensor_create(env->ctx, &L, 1, cases[i].dtype);
        assert_non_null(tok);
        // Populate with same ids via memcpy and truncation rules
        if (cases[i].dtype == MARMOT_DTYPE_UINT32) {
            uint32_t tmp[L];
            for (size_t j = 0; j < L; ++j)
                tmp[j] = (uint32_t)ids32[j];
            memcpy(tok->data, tmp, sizeof(tmp));
        } else if (cases[i].dtype == MARMOT_DTYPE_INT16) {
            int16_t tmp[L];
            for (size_t j = 0; j < L; ++j)
                tmp[j] = (int16_t)ids32[j];
            memcpy(tok->data, tmp, sizeof(tmp));
        } else {
            uint16_t tmp[L];
            for (size_t j = 0; j < L; ++j)
                tmp[j] = (uint16_t)ids32[j];
            memcpy(tok->data, tmp, sizeof(tmp));
        }

        marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);
        assert_non_null(out);
        marmot_embedding_desc_t d = marmot_embedding_desc_default();
        d.weights = weights;
        d.token_ids = tok;
        d.out = out;
        d.dtype_out = MARMOT_DTYPE_FLOAT32;
        assert_int_equal(marmot_embedding_lookup(env->ctx, &d), MARMOT_SUCCESS);
        float got[L * dim];
        fetch_embedding_as_f32(env, out, got, L * dim);
        marmot_test_expect_close_array(got, ref, L * dim, 1e-6f);
        marmot_tensor_destroy(out);
        marmot_tensor_destroy(tok);
    }

    marmot_tensor_destroy(out32);
    marmot_tensor_destroy(tok32);
    marmot_tensor_destroy(weights);
    if (cpu_ctx != nullptr) {
        marmot_destroy(cpu_ctx);
    }
}

static void test_embedding_output_dtypes(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);
    marmot_context_t *cpu_ctx = nullptr;
    marmot_test_env_t cpu_env;
    if (env->backend != MARMOT_BACKEND_CPU) {
        cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx == nullptr) {
            skip();
            return;
        }
        cpu_env.backend = MARMOT_BACKEND_CPU;
        cpu_env.ctx = cpu_ctx;
        env = &cpu_env;
    }

    const size_t vocab = 4;
    const size_t dim = 7;
    const size_t weight_shape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    float *w = (float *)weights->data;
    for (size_t i = 0; i < vocab * dim; ++i)
        w[i] = (float)(i) * 0.25f - 2.0f;

    const size_t token_shape[] = {3};
    marmot_tensor_t *tok = marmot_tensor_create(env->ctx, token_shape, 1, MARMOT_DTYPE_INT32);
    int32_t *t = (int32_t *)tok->data;
    t[0] = 3;
    t[1] = 1;
    t[2] = 0;

    const size_t out_shape[] = {3, dim};

    // Reference f32
    marmot_tensor_t *out_f32 = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_embedding_desc_t ref = marmot_embedding_desc_default();
    ref.weights = weights;
    ref.token_ids = tok;
    ref.out = out_f32;
    ref.dtype_out = MARMOT_DTYPE_FLOAT32;
    assert_int_equal(marmot_embedding_lookup(env->ctx, &ref), MARMOT_SUCCESS);
    float ref_buf[3 * dim];
    fetch_embedding_as_f32(env, out_f32, ref_buf, 3 * dim);

    // BF16
    marmot_tensor_t *out_bf16 = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_BFLOAT16);
    marmot_embedding_desc_t d1 = marmot_embedding_desc_default();
    d1.weights = weights;
    d1.token_ids = tok;
    d1.out = out_bf16;
    d1.dtype_out = MARMOT_DTYPE_BFLOAT16;
    assert_int_equal(marmot_embedding_lookup(env->ctx, &d1), MARMOT_SUCCESS);
    float got_bf16[3 * dim];
    fetch_embedding_as_f32(env, out_bf16, got_bf16, 3 * dim);
    marmot_test_expect_close_array(got_bf16, ref_buf, 3 * dim, 1e-3f);

    // F16
    marmot_tensor_t *out_f16 = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT16);
    marmot_embedding_desc_t d2 = marmot_embedding_desc_default();
    d2.weights = weights;
    d2.token_ids = tok;
    d2.out = out_f16;
    d2.dtype_out = MARMOT_DTYPE_FLOAT16;
    assert_int_equal(marmot_embedding_lookup(env->ctx, &d2), MARMOT_SUCCESS);
    float got_f16[3 * dim];
    fetch_embedding_as_f32(env, out_f16, got_f16, 3 * dim);
    marmot_test_expect_close_array(got_f16, ref_buf, 3 * dim, 3e-3f);

    marmot_tensor_destroy(out_f16);
    marmot_tensor_destroy(out_bf16);
    marmot_tensor_destroy(out_f32);
    marmot_tensor_destroy(tok);
    marmot_tensor_destroy(weights);
    if (cpu_ctx != nullptr) {
        marmot_destroy(cpu_ctx);
    }
}

static void test_embedding_ragged_basic(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);

    marmot_context_t *cpu_ctx = nullptr;
    marmot_test_env_t cpu_env;
    if (env->backend != MARMOT_BACKEND_CPU) {
        cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx == nullptr) {
            skip();
            return;
        }
        cpu_env.backend = MARMOT_BACKEND_CPU;
        cpu_env.ctx = cpu_ctx;
        env = &cpu_env;
    }

    const size_t vocab = 4;
    const size_t dim = 2;
    const size_t weight_shape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(weights);
    float *w_data = (float *)weights->data;
    const float seed[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f};
    memcpy(w_data, seed, sizeof(seed));

    const size_t token_shape[] = {5};
    marmot_tensor_t *token_ids = marmot_tensor_create(env->ctx, token_shape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(token_ids);
    int32_t *t_data = (int32_t *)token_ids->data;
    const int32_t ids[] = {3, 1, 0, 2, 1};
    memcpy(t_data, ids, sizeof(ids));

    const size_t out_shape[] = {5, dim};
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, out_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    const int32_t offsets[] = {0, 2, 3, 5};

    marmot_embedding_desc_t desc = marmot_embedding_desc_default();
    desc.weights = weights;
    desc.token_ids = token_ids;
    desc.out = out;
    desc.dtype_out = MARMOT_DTYPE_FLOAT32;
    desc.ragged = true;
    desc.row_offsets = offsets;
    desc.num_row_offsets = sizeof(offsets) / sizeof(offsets[0]);

    marmot_error_t err = marmot_embedding_lookup(env->ctx, &desc);
    assert_int_equal(err, MARMOT_SUCCESS);

    float out_buf[5 * dim];
    fetch_embedding(env, out, out_buf, 5 * dim);

    const float expected[] = {
        6.0f, 7.0f, 2.0f, 3.0f, 0.0f, 1.0f, 4.0f, 5.0f, 2.0f, 3.0f,
    };
    marmot_test_expect_close_array(out_buf, expected, 5 * dim, 1e-6f);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(token_ids);
    marmot_tensor_destroy(weights);
    if (cpu_ctx != nullptr) {
        marmot_destroy(cpu_ctx);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_embedding_float_basic, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(test_embedding_scale, marmot_test_backend_setup, marmot_test_backend_teardown),
        cmocka_unit_test_setup_teardown(
            test_embedding_device_token_ids_stale_host, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_float64_basic, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_padding_and_bounds, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_token_id_dtypes, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_output_dtypes, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_embedding_ragged_basic, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };

    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
