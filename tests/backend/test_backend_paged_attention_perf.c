#include "marmot/config.h"
#include "marmot/inference/kv_pool.h"
#include "marmot/ops/paged_attention.h"
#include "marmot/tensor.h"

#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include <string.h>
#include <time.h>

#include "backend/test_backend_utils.h"

typedef struct {
    size_t seq_len;
    size_t head_dim;
    size_t block_size;
} paged_attention_perf_case_t;

static const uint32_t kTokenFlagPrefill = 1u << 0;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static size_t ceil_div(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

static const char *backend_name(marmot_backend_type_t backend) {
    switch (backend) {
    case MARMOT_BACKEND_CPU:
        return "cpu";
    case MARMOT_BACKEND_METAL:
        return "metal";
    case MARMOT_BACKEND_CUDA:
        return "cuda";
    default:
        return "unknown";
    }
}

static const char *dtype_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT16:
        return "f16";
    case MARMOT_DTYPE_BFLOAT16:
        return "bf16";
    case MARMOT_DTYPE_FLOAT32:
        return "f32";
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return "fp8_e4m3";
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return "fp8_e5m2";
#endif
    default:
        return "unknown";
    }
}

static size_t choose_iters(size_t tokens, size_t kv_len, size_t head_dim, size_t heads) {
    double work = (double)tokens * (double)kv_len * (double)head_dim * (double)heads;
    if (work < 2.0e6) {
        return 80;
    }
    if (work < 1.0e7) {
        return 30;
    }
    if (work < 5.0e7) {
        return 12;
    }
    return 6;
}

static void fill_pattern(float *dst, size_t count, uint32_t seed, float scale) {
    uint32_t x = seed;
    for (size_t i = 0; i < count; ++i) {
        x = x * 1664525u + 1013904223u;
        float v = (float)(x & 0xFFFFu) * (1.0f / 65535.0f);
        dst[i] = (v - 0.5f) * scale;
    }
}

static marmot_dtype_t pick_paged_attention_dtype(const marmot_test_env_t *env) {
    const char *override = getenv("MARMOT_PAGED_ATTENTION_PERF_DTYPE");
    if (override != nullptr && override[0] != '\0') {
        char lowered[32];
        marmot_test_to_lower(lowered, sizeof(lowered), override);
        if (strcmp(lowered, "f16") == 0 || strcmp(lowered, "float16") == 0) {
            return MARMOT_DTYPE_FLOAT16;
        }
        if (strcmp(lowered, "bf16") == 0 || strcmp(lowered, "bfloat16") == 0) {
            return MARMOT_DTYPE_BFLOAT16;
        }
        if (strcmp(lowered, "f32") == 0 || strcmp(lowered, "float32") == 0) {
            return MARMOT_DTYPE_FLOAT32;
        }
    }
    if (env->backend == MARMOT_BACKEND_METAL) {
        return MARMOT_DTYPE_FLOAT16;
    }
    return MARMOT_DTYPE_FLOAT32;
}

static void run_paged_attention_prefill_case(
    marmot_test_env_t *env, marmot_dtype_t activation_dtype, marmot_dtype_t kv_dtype,
    const paged_attention_perf_case_t *tc
) {
    const size_t num_q_heads = 8;
    const size_t num_kv_heads = 8;
    const size_t token_count = tc->seq_len;
    const size_t kv_len = tc->seq_len;

    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = env->backend;
    opts.max_seqs = 1;
    opts.max_seq_len = token_count;
    opts.block_size = tc->block_size;
    opts.num_blocks = ceil_div(token_count, tc->block_size);
    opts.num_layers = 1;
    opts.num_kv_heads = num_kv_heads;
    opts.head_dim = tc->head_dim;
    opts.kv_dtype = kv_dtype;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(env->ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq), MARMOT_SUCCESS);

    marmot_kv_append_plan_t plan = {0};
    marmot_kv_slot_t *slots = (marmot_kv_slot_t *)malloc(sizeof(marmot_kv_slot_t) * token_count);
    assert_non_null(slots);

    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq, token_count, &plan, slots, &start_pos), MARMOT_SUCCESS);

    const size_t meta_shape[2] = {token_count, 4};
    marmot_tensor_t *token_meta = marmot_tensor_create(env->ctx, meta_shape, 2, MARMOT_DTYPE_UINT32);
    assert_non_null(token_meta);

    marmot_uint32_t *meta_data = marmot_tensor_data_u32_mut(env->ctx, token_meta);
    assert_non_null(meta_data);
    for (size_t t = 0; t < token_count; ++t) {
        meta_data[t * 4 + 0].value = seq;
        meta_data[t * 4 + 1].value = (uint32_t)(start_pos + t);
        meta_data[t * 4 + 2].value = slots[t];
        meta_data[t * 4 + 3].value = kTokenFlagPrefill;
    }

    const size_t q_shape[3] = {token_count, num_q_heads, tc->head_dim};
    const size_t kv_shape[3] = {token_count, num_kv_heads, tc->head_dim};
    marmot_tensor_t *q = marmot_tensor_create(env->ctx, q_shape, 3, activation_dtype);
    marmot_tensor_t *k_new = marmot_tensor_create(env->ctx, kv_shape, 3, activation_dtype);
    marmot_tensor_t *v_new = marmot_tensor_create(env->ctx, kv_shape, 3, activation_dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, q_shape, 3, activation_dtype);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(out);

    const size_t q_count = token_count * num_q_heads * tc->head_dim;
    const size_t kv_count = token_count * num_kv_heads * tc->head_dim;
    float *q_host = (float *)malloc(sizeof(float) * q_count);
    float *k_host = (float *)malloc(sizeof(float) * kv_count);
    float *v_host = (float *)malloc(sizeof(float) * kv_count);
    assert_non_null(q_host);
    assert_non_null(k_host);
    assert_non_null(v_host);

    fill_pattern(q_host, q_count, 1u, 1.0f);
    fill_pattern(k_host, kv_count, 2u, 0.8f);
    fill_pattern(v_host, kv_count, 3u, 1.2f);

    marmot_test_convert_f32_span(env, q, q_host, q_count);
    marmot_test_convert_f32_span(env, k_new, k_host, kv_count);
    marmot_test_convert_f32_span(env, v_new, v_host, kv_count);
    marmot_test_commit_tensor(env, q);
    marmot_test_commit_tensor(env, k_new);
    marmot_test_commit_tensor(env, v_new);

    free(q_host);
    free(k_host);
    free(v_host);

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    assert_int_equal(marmot_kv_pool_get_tensors(pool, &kv_k, &kv_v, &block_table), MARMOT_SUCCESS);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);

    marmot_tensor_t *kv_k_scale = nullptr;
    marmot_tensor_t *kv_v_scale = nullptr;
#if MARMOT_ENABLE_FP8
    marmot_paged_attention_kv_scale_ext_t scale_ext = {0};
#endif

    marmot_paged_attention_desc_t desc = marmot_paged_attention_desc_default();
    desc.token_count = token_count;
    desc.layer_idx = 0;
    desc.num_q_heads = (uint32_t)num_q_heads;
    desc.num_kv_heads = (uint32_t)num_kv_heads;
    desc.head_dim = (uint32_t)tc->head_dim;
    desc.block_size = (uint32_t)tc->block_size;
    desc.scale = 1.0f / sqrtf((float)tc->head_dim);
    desc.token_meta = token_meta;
    desc.q = q;
    desc.k_new = k_new;
    desc.v_new = v_new;
    desc.kv_k = kv_k;
    desc.kv_v = kv_v;
    desc.block_table = block_table;
    desc.out = out;
#if MARMOT_ENABLE_FP8
    if (kv_dtype == MARMOT_DTYPE_FLOAT8_E4M3) {
        assert_int_equal(marmot_kv_pool_get_scale_tensors(pool, &kv_k_scale, &kv_v_scale), MARMOT_SUCCESS);
        assert_non_null(kv_k_scale);
        assert_non_null(kv_v_scale);
        scale_ext.struct_size = sizeof(scale_ext);
        scale_ext.struct_version = MARMOT_PAGED_ATTENTION_KV_SCALE_EXT_VERSION;
        scale_ext.kv_k_scale = kv_k_scale;
        scale_ext.kv_v_scale = kv_v_scale;
        desc.pnext = &scale_ext;
    }
#endif

    const size_t warmup = 3;
    for (size_t i = 0; i < warmup; ++i) {
        marmot_error_t err = marmot_paged_attention(env->ctx, &desc);
        assert_int_equal(err, MARMOT_SUCCESS);
        err = marmot_device_synchronize(env->ctx);
        assert_int_equal(err, MARMOT_SUCCESS);
    }

    const size_t iters = choose_iters(token_count, kv_len, tc->head_dim, num_q_heads);
    double t0 = now_seconds();
    for (size_t i = 0; i < iters; ++i) {
        marmot_error_t err = marmot_paged_attention(env->ctx, &desc);
        assert_int_equal(err, MARMOT_SUCCESS);
        err = marmot_device_synchronize(env->ctx);
        assert_int_equal(err, MARMOT_SUCCESS);
    }
    double t1 = now_seconds();

    double avg_us = ((t1 - t0) / (double)iters) * 1.0e6;
    printf(
        "\n[paged attention prefill %s] dtype=%s kv=%s tokens=%zu kv_len=%zu block=%zu heads=%zu dim=%zu iters=%zu "
        "avg=%.3f us",
        backend_name(env->backend), dtype_name(activation_dtype), dtype_name(kv_dtype), token_count, kv_len,
        tc->block_size, num_q_heads, tc->head_dim, iters, avg_us
    );
    if (env->backend == MARMOT_BACKEND_METAL) {
        bool expect_flash = kv_len >= 128 && tc->head_dim <= 256;
        printf(" expect_flash=%s", expect_flash ? "yes" : "no");
    }
    printf("\n");

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq), MARMOT_SUCCESS);

    free(slots);
    marmot_kv_pool_destroy(pool);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(v_new);
    marmot_tensor_destroy(k_new);
    marmot_tensor_destroy(q);
    marmot_tensor_destroy(token_meta);
}

static void test_paged_attention_prefill_perf_optional(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);

    const char *run = getenv("MARMOT_RUN_PAGED_ATTENTION_PERF");
    if (run == nullptr || run[0] == '\0') {
        skip();
        return;
    }
    if (env->backend != MARMOT_BACKEND_METAL) {
        const char *run_cpu = getenv("MARMOT_RUN_PAGED_ATTENTION_PERF_CPU");
        if (run_cpu == nullptr || run_cpu[0] == '\0') {
            skip();
            return;
        }
    }

    const marmot_dtype_t dtype = pick_paged_attention_dtype(env);
    const size_t seq_lens[] = {64, 128, 256, 512};
    const size_t head_dims[] = {64, 128, 256};
    const size_t block_sizes[] = {16, 32};

    paged_attention_perf_case_t tc = {
        .seq_len = 0,
        .head_dim = 0,
        .block_size = 0,
    };

    for (size_t s = 0; s < sizeof(seq_lens) / sizeof(seq_lens[0]); ++s) {
        tc.seq_len = seq_lens[s];
        for (size_t d = 0; d < sizeof(head_dims) / sizeof(head_dims[0]); ++d) {
            tc.head_dim = head_dims[d];
            for (size_t b = 0; b < sizeof(block_sizes) / sizeof(block_sizes[0]); ++b) {
                tc.block_size = block_sizes[b];
                run_paged_attention_prefill_case(env, dtype, dtype, &tc);
            }
        }
    }
#if MARMOT_ENABLE_FP8
    const char *run_fp8 = getenv("MARMOT_RUN_PAGED_ATTENTION_PERF_FP8");
    if (run_fp8 != nullptr && run_fp8[0] != '\0' && env->backend == MARMOT_BACKEND_CPU) {
        for (size_t s = 0; s < sizeof(seq_lens) / sizeof(seq_lens[0]); ++s) {
            tc.seq_len = seq_lens[s];
            for (size_t d = 0; d < sizeof(head_dims) / sizeof(head_dims[0]); ++d) {
                tc.head_dim = head_dims[d];
                for (size_t b = 0; b < sizeof(block_sizes) / sizeof(block_sizes[0]); ++b) {
                    tc.block_size = block_sizes[b];
                    run_paged_attention_prefill_case(env, dtype, MARMOT_DTYPE_FLOAT8_E4M3, &tc);
                }
            }
        }
    }
#endif

    if (env->backend == MARMOT_BACKEND_METAL) {
        const size_t sweep_seq_lens[] = {96, 112, 128, 144, 160, 192};
        const size_t sweep_head_dims[] = {64, 128};
        const size_t sweep_block_sizes[] = {16, 32};
        paged_attention_perf_case_t sweep = {
            .seq_len = 0,
            .head_dim = 0,
            .block_size = 0,
        };

        for (size_t s = 0; s < sizeof(sweep_seq_lens) / sizeof(sweep_seq_lens[0]); ++s) {
            sweep.seq_len = sweep_seq_lens[s];
            for (size_t d = 0; d < sizeof(sweep_head_dims) / sizeof(sweep_head_dims[0]); ++d) {
                sweep.head_dim = sweep_head_dims[d];
                for (size_t b = 0; b < sizeof(sweep_block_sizes) / sizeof(sweep_block_sizes[0]); ++b) {
                    sweep.block_size = sweep_block_sizes[b];
                    run_paged_attention_prefill_case(env, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, &sweep);
                }
            }
        }
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_paged_attention_prefill_perf_optional, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
