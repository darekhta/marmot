#include "marmot/inference/kv_pool.h"
#include "marmot/ops/paged_attention.h"
#include "marmot/quant_traits.h"

#include <string.h>

#include "backend/test_backend_utils.h"

#ifdef __APPLE__

typedef marmot_error_t (*marmot_quantize_fn)(const marmot_context_t *, const marmot_tensor_t *, marmot_tensor_t *);

typedef struct {
    marmot_quant_kind_t kind;
    marmot_quantize_fn quantize;
    size_t k_values;
} marmot_quant_case_t;

typedef struct {
    const char *name;
    char *value;
} marmot_env_guard_t;

static const uint32_t kTokenFlagPrefill = 1u << 0;
static const uint32_t kTokenFlagDecode = 1u << 1;

static char *marmot_env_strdup(const char *value) {
    if (value == nullptr) {
        return nullptr;
    }
    const size_t len = strlen(value) + 1u;
    char *copy = (char *)malloc(len);
    if (copy != nullptr) {
        memcpy(copy, value, len);
    }
    return copy;
}

static marmot_env_guard_t marmot_env_guard(const char *name) {
    marmot_env_guard_t guard = {name, marmot_env_strdup(getenv(name))};
    return guard;
}

static void marmot_env_restore(marmot_env_guard_t guard) {
    if (guard.value == nullptr) {
        unsetenv(guard.name);
        return;
    }
    setenv(guard.name, guard.value, 1);
    free(guard.value);
}

static void fill_span(float *dst, size_t count, float scale, float offset) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = offset + scale * (float)i;
    }
}

static void run_quant_matmul_case(
    marmot_context_t *ctx, marmot_quant_kind_t kind, marmot_quantize_fn quantize, size_t N, size_t K, size_t M
) {
    marmot_test_env_t env = {.backend = MARMOT_BACKEND_METAL, .ctx = ctx};

    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    assert_non_null(traits);

    const size_t input_count = N * K;
    const size_t weight_count = M * K;
    float *input_vals = (float *)malloc(input_count * sizeof(float));
    float *weight_vals = (float *)malloc(weight_count * sizeof(float));
    assert_non_null(input_vals);
    assert_non_null(weight_vals);
    fill_span(input_vals, input_count, 0.01f, 0.1f);
    fill_span(weight_vals, weight_count, 0.005f, -0.2f);

    size_t shape_input[] = {N, K};
    size_t shape_weight[] = {M, K};
    size_t shape_output[] = {N, M};

    marmot_tensor_t *input = marmot_test_tensor_from_array(&env, shape_input, 2, input_vals);
    marmot_tensor_t *weight_fp32 = marmot_test_tensor_from_array(&env, shape_weight, 2, weight_vals);

    const size_t blocks_per_row = (K + traits->block_values - 1u) / traits->block_values;
    const size_t num_blocks = M * blocks_per_row;
    const size_t block_bytes = traits->header_bytes + traits->payload_bytes;
    size_t quant_bytes = num_blocks * block_bytes;
    marmot_tensor_t *weight_q = marmot_tensor_create(ctx, &quant_bytes, 1, MARMOT_DTYPE_UINT8);
    assert_non_null(weight_q);

    assert_int_equal(quantize(ctx, weight_fp32, weight_q), MARMOT_SUCCESS);

    weight_q->quant_kind = kind;
    weight_q->quant_layout = traits->layout;
    weight_q->shape.ndim = 2;
    weight_q->shape.shape[0] = M;
    weight_q->shape.shape[1] = K;
    weight_q->shape.strides[0] = K;
    weight_q->shape.strides[1] = 1;

    marmot_tensor_t *out = marmot_tensor_create(ctx, shape_output, 2, MARMOT_DTYPE_FLOAT32);
    assert_non_null(out);

    assert_int_equal(marmot_linear(ctx, input, weight_q, nullptr, out), MARMOT_SUCCESS);

    marmot_test_tensor_destroy_all(4, out, weight_q, weight_fp32, input);
    free(weight_vals);
    free(input_vals);
}

static void run_paged_attention_case(marmot_context_t *ctx) {
    marmot_test_env_t env = {.backend = MARMOT_BACKEND_METAL, .ctx = ctx};

    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = MARMOT_BACKEND_METAL;
    opts.max_seqs = 1;
    opts.max_seq_len = 4;
    opts.block_size = 4;
    opts.num_blocks = 1;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 2;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq), MARMOT_SUCCESS);

    marmot_kv_append_plan_t plan = {0};
    marmot_kv_slot_t slots[2] = {0};
    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq, 2, &plan, slots, &start_pos), MARMOT_SUCCESS);

    const size_t meta_shape[2] = {2, 4};
    marmot_tensor_t *token_meta = marmot_tensor_create(ctx, meta_shape, 2, MARMOT_DTYPE_UINT32);
    assert_non_null(token_meta);

    marmot_uint32_t meta_data[8];
    for (size_t t = 0; t < 2; ++t) {
        meta_data[t * 4 + 0].value = seq;
        meta_data[t * 4 + 1].value = (uint32_t)(start_pos + t);
        meta_data[t * 4 + 2].value = slots[t];
        meta_data[t * 4 + 3].value = (t == 0) ? kTokenFlagDecode : kTokenFlagPrefill;
    }
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(ctx, token_meta, meta_data, sizeof(meta_data)), MARMOT_SUCCESS
    );

    const size_t q_shape[3] = {2, 1, 2};
    marmot_tensor_t *q = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *k_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *v_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(out);

    float q_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float v_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    marmot_test_convert_f32_span(&env, q, q_vals, 4);
    marmot_test_convert_f32_span(&env, k_new, k_vals, 4);
    marmot_test_convert_f32_span(&env, v_new, v_vals, 4);
    marmot_test_commit_tensor(&env, q);
    marmot_test_commit_tensor(&env, k_new);
    marmot_test_commit_tensor(&env, v_new);

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    assert_int_equal(marmot_kv_pool_get_tensors(pool, &kv_k, &kv_v, &block_table), MARMOT_SUCCESS);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);

    marmot_paged_attention_desc_t desc = marmot_paged_attention_desc_default();
    desc.token_count = 2;
    desc.layer_idx = 0;
    desc.num_q_heads = 1;
    desc.num_kv_heads = 1;
    desc.head_dim = 2;
    desc.block_size = (uint32_t)opts.block_size;
    desc.scale = 1.0f;
    desc.token_meta = token_meta;
    desc.q = q;
    desc.k_new = k_new;
    desc.v_new = v_new;
    desc.kv_k = kv_k;
    desc.kv_v = kv_v;
    desc.block_table = block_table;
    desc.out = out;

    assert_int_equal(marmot_paged_attention(ctx, &desc), MARMOT_SUCCESS);
    assert_int_equal(marmot_device_synchronize(ctx), MARMOT_SUCCESS);

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq), MARMOT_SUCCESS);

    marmot_kv_pool_destroy(pool);
    marmot_test_tensor_destroy_all(5, out, v_new, k_new, q, token_meta);
}

static void test_metal_compat_simdgroup_mm_off(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env == nullptr || env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    marmot_env_guard_t guard = marmot_env_guard("MARMOT_METAL_SIMDGROUP_MM");
    setenv("MARMOT_METAL_SIMDGROUP_MM", "0", 1);

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_METAL);
    if (ctx == nullptr) {
        marmot_env_restore(guard);
        skip();
        return;
    }

    const marmot_quant_case_t cases[] = {
        {MARMOT_QUANT_KIND_Q4_0, marmot_quantize_q4_0, 64},  {MARMOT_QUANT_KIND_Q5_0, marmot_quantize_q5_0, 64},
        {MARMOT_QUANT_KIND_Q8_0, marmot_quantize_q8_0, 64},  {MARMOT_QUANT_KIND_Q4_K, marmot_quantize_q4_k, 256},
        {MARMOT_QUANT_KIND_Q5_K, marmot_quantize_q5_k, 256}, {MARMOT_QUANT_KIND_Q6_K, marmot_quantize_q6_k, 256},
        {MARMOT_QUANT_KIND_Q8_K, marmot_quantize_q8_k, 256},
    };

    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        run_quant_matmul_case(ctx, cases[i].kind, cases[i].quantize, 4, cases[i].k_values, 8);
    }

    marmot_destroy(ctx);
    marmot_env_restore(guard);
}

static void test_metal_compat_decode_simd_groups_small(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env == nullptr || env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    marmot_env_guard_t guard = marmot_env_guard("MARMOT_METAL_DECODE_SIMD_GROUPS");
    setenv("MARMOT_METAL_DECODE_SIMD_GROUPS", "4", 1);

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_METAL);
    if (ctx == nullptr) {
        marmot_env_restore(guard);
        skip();
        return;
    }

    run_paged_attention_case(ctx);

    marmot_destroy(ctx);
    marmot_env_restore(guard);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_metal_compat_simdgroup_mm_off, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
        cmocka_unit_test_setup_teardown(
            test_metal_compat_decode_simd_groups_small, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}

#else

int main(void) {
    return 0;
}

#endif
