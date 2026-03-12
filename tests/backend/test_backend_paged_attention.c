#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/inference/kv_pool.h"
#include "marmot/ops/paged_attention.h"
#include "marmot/tensor.h"

#include <stdlib.h>

#include <math.h>
#include <string.h>

#include "backend/test_backend_utils.h"

static const uint32_t kTokenFlagPrefill = 1u << 0;
static const uint32_t kTokenFlagDecode = 1u << 1;

static size_t
kv_index_5d(const marmot_tensor_t *tensor, size_t block, size_t layer, size_t head, size_t offset, size_t d) {
    return block * tensor->shape.strides[0] + layer * tensor->shape.strides[1] + head * tensor->shape.strides[2] +
        offset * tensor->shape.strides[3] + d * tensor->shape.strides[4];
}

static uint32_t log2_u32(uint32_t value) {
    uint32_t shift = 0;
    while (value > 1u) {
        value >>= 1u;
        shift++;
    }
    return shift;
}

static size_t ceil_div(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

static void fill_pattern(float *dst, size_t count, uint32_t seed, float scale) {
    uint32_t x = seed;
    for (size_t i = 0; i < count; ++i) {
        x = x * 1664525u + 1013904223u;
        float v = (float)(x & 0xFFFFu) * (1.0f / 65535.0f);
        dst[i] = (v - 0.5f) * scale;
    }
}

static void fill_token_meta_prefill(
    marmot_uint32_t *meta, size_t token_count, marmot_seq_slot_t seq, size_t start_pos, const marmot_kv_slot_t *slots
) {
    for (size_t t = 0; t < token_count; ++t) {
        meta[t * 4 + 0].value = seq;
        meta[t * 4 + 1].value = (uint32_t)(start_pos + t);
        meta[t * 4 + 2].value = slots[t];
        meta[t * 4 + 3].value = kTokenFlagPrefill;
    }
}

static void run_paged_attention_prefill_case(
    const marmot_test_env_t *env, marmot_dtype_t activation_dtype, marmot_dtype_t kv_dtype, size_t token_count,
    size_t num_q_heads, size_t num_kv_heads, size_t head_dim, size_t block_size, const float *q_host,
    const float *k_host, const float *v_host, float *out_host
) {
    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = env->backend;
    opts.max_seqs = 1;
    opts.max_seq_len = token_count;
    opts.block_size = block_size;
    opts.num_blocks = ceil_div(token_count, block_size);
    opts.num_layers = 1;
    opts.num_kv_heads = num_kv_heads;
    opts.head_dim = head_dim;
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

    const size_t meta_count = token_count * 4;
    marmot_uint32_t *meta_host = (marmot_uint32_t *)malloc(sizeof(marmot_uint32_t) * meta_count);
    assert_non_null(meta_host);
    fill_token_meta_prefill(meta_host, token_count, seq, start_pos, slots);
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, token_meta, meta_host, sizeof(marmot_uint32_t) * meta_count),
        MARMOT_SUCCESS
    );

    const size_t q_shape[3] = {token_count, num_q_heads, head_dim};
    const size_t kv_shape[3] = {token_count, num_kv_heads, head_dim};
    marmot_tensor_t *q = marmot_tensor_create(env->ctx, q_shape, 3, activation_dtype);
    marmot_tensor_t *k_new = marmot_tensor_create(env->ctx, kv_shape, 3, activation_dtype);
    marmot_tensor_t *v_new = marmot_tensor_create(env->ctx, kv_shape, 3, activation_dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, q_shape, 3, activation_dtype);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(out);

    const size_t q_count = token_count * num_q_heads * head_dim;
    const size_t kv_count = token_count * num_kv_heads * head_dim;
    marmot_test_convert_f32_span(env, q, q_host, q_count);
    marmot_test_convert_f32_span(env, k_new, k_host, kv_count);
    marmot_test_convert_f32_span(env, v_new, v_host, kv_count);
    marmot_test_commit_tensor(env, q);
    marmot_test_commit_tensor(env, k_new);
    marmot_test_commit_tensor(env, v_new);

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    assert_int_equal(marmot_kv_pool_get_tensors(pool, &kv_k, &kv_v, &block_table), MARMOT_SUCCESS);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);

    marmot_paged_attention_desc_t desc = marmot_paged_attention_desc_default();
    desc.token_count = token_count;
    desc.layer_idx = 0;
    desc.num_q_heads = (uint32_t)num_q_heads;
    desc.num_kv_heads = (uint32_t)num_kv_heads;
    desc.head_dim = (uint32_t)head_dim;
    desc.block_size = (uint32_t)block_size;
    desc.scale = 1.0f / sqrtf((float)head_dim);
    desc.token_meta = token_meta;
    desc.q = q;
    desc.k_new = k_new;
    desc.v_new = v_new;
    desc.kv_k = kv_k;
    desc.kv_v = kv_v;
    desc.block_table = block_table;
    desc.out = out;

    marmot_error_t err = marmot_paged_attention(env->ctx, &desc);
    if (err != MARMOT_SUCCESS) {
        fail_msg("paged_attention failed: %s %s", marmot_error_string(err), marmot_get_last_error_detail());
    }
    assert_int_equal(marmot_device_synchronize(env->ctx), MARMOT_SUCCESS);

    marmot_test_fetch_f32_span(env, out_host, out, q_count);

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq), MARMOT_SUCCESS);

    free(slots);
    free(meta_host);
    marmot_kv_pool_destroy(pool);
    marmot_test_tensor_destroy_all(5, out, v_new, k_new, q, token_meta);
}

static void test_paged_attention_basic(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = env->backend;
    opts.max_seqs = 1;
    opts.max_seq_len = 4;
    opts.block_size = 4;
    opts.num_blocks = 1;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 2;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(env->ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq), MARMOT_SUCCESS);

    marmot_kv_append_plan_t plan = {0};
    marmot_kv_slot_t slots[2] = {0};
    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq, 2, &plan, slots, &start_pos), MARMOT_SUCCESS);

    const size_t meta_shape[2] = {2, 4};
    marmot_tensor_t *token_meta = marmot_tensor_create(env->ctx, meta_shape, 2, MARMOT_DTYPE_UINT32);
    assert_non_null(token_meta);

    marmot_uint32_t *meta_data = marmot_tensor_data_u32_mut(env->ctx, token_meta);
    assert_non_null(meta_data);
    for (size_t t = 0; t < 2; ++t) {
        meta_data[t * 4 + 0].value = seq;
        meta_data[t * 4 + 1].value = (uint32_t)(start_pos + t);
        meta_data[t * 4 + 2].value = slots[t];
        meta_data[t * 4 + 3].value = (t == 0) ? kTokenFlagDecode : kTokenFlagPrefill;
    }

    const size_t q_shape[3] = {2, 1, 2};
    marmot_tensor_t *q = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *k_new = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *v_new = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(out);

    float q_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float v_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    memcpy(q->data, q_vals, sizeof(q_vals));
    memcpy(k_new->data, k_vals, sizeof(k_vals));
    memcpy(v_new->data, v_vals, sizeof(v_vals));

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

    assert_int_equal(marmot_paged_attention(env->ctx, &desc), MARMOT_SUCCESS);

    const float *out_data = marmot_tensor_data_f32(env->ctx, out);
    assert_non_null(out_data);
    assert_true(fabsf(out_data[0] - 1.0f) < 1e-4f);
    assert_true(fabsf(out_data[1] - 0.0f) < 1e-4f);

    const float exp1 = expf(1.0f);
    const float w0 = 1.0f / (1.0f + exp1);
    const float w1 = exp1 / (1.0f + exp1);
    assert_true(fabsf(out_data[2] - w0) < 1e-4f);
    assert_true(fabsf(out_data[3] - w1) < 1e-4f);

    const uint32_t shift = log2_u32((uint32_t)opts.block_size);
    const uint32_t block_id = slots[0] >> shift;
    assert_int_equal(block_id, slots[1] >> shift);

    const float *kv_k_data = marmot_tensor_data_f32(env->ctx, kv_k);
    const float *kv_v_data = marmot_tensor_data_f32(env->ctx, kv_v);
    assert_non_null(kv_k_data);
    assert_non_null(kv_v_data);

    size_t idx0 = kv_index_5d(kv_k, block_id, 0, 0, 0, 0);
    size_t idx1 = kv_index_5d(kv_k, block_id, 0, 0, 0, 1);
    size_t idx2 = kv_index_5d(kv_k, block_id, 0, 0, 1, 0);
    size_t idx3 = kv_index_5d(kv_k, block_id, 0, 0, 1, 1);
    assert_true(fabsf(kv_k_data[idx0] - 1.0f) < 1e-4f);
    assert_true(fabsf(kv_k_data[idx1] - 0.0f) < 1e-4f);
    assert_true(fabsf(kv_k_data[idx2] - 0.0f) < 1e-4f);
    assert_true(fabsf(kv_k_data[idx3] - 1.0f) < 1e-4f);

    assert_true(fabsf(kv_v_data[idx0] - 1.0f) < 1e-4f);
    assert_true(fabsf(kv_v_data[idx1] - 0.0f) < 1e-4f);
    assert_true(fabsf(kv_v_data[idx2] - 0.0f) < 1e-4f);
    assert_true(fabsf(kv_v_data[idx3] - 1.0f) < 1e-4f);

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq), MARMOT_SUCCESS);

    marmot_kv_pool_destroy(pool);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(v_new);
    marmot_tensor_destroy(k_new);
    marmot_tensor_destroy(q);
    marmot_tensor_destroy(token_meta);
}

static void test_paged_attention_mixed_kv_dtype(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = env->backend;
    opts.max_seqs = 1;
    opts.max_seq_len = 4;
    opts.block_size = 4;
    opts.num_blocks = 1;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 2;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(env->ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq), MARMOT_SUCCESS);

    marmot_kv_append_plan_t plan = {0};
    marmot_kv_slot_t slots[2] = {0};
    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq, 2, &plan, slots, &start_pos), MARMOT_SUCCESS);

    const size_t meta_shape[2] = {2, 4};
    marmot_tensor_t *token_meta = marmot_tensor_create(env->ctx, meta_shape, 2, MARMOT_DTYPE_UINT32);
    assert_non_null(token_meta);

    marmot_uint32_t *meta_data = marmot_tensor_data_u32_mut(env->ctx, token_meta);
    assert_non_null(meta_data);
    for (size_t t = 0; t < 2; ++t) {
        meta_data[t * 4 + 0].value = seq;
        meta_data[t * 4 + 1].value = (uint32_t)(start_pos + t);
        meta_data[t * 4 + 2].value = slots[t];
        meta_data[t * 4 + 3].value = (t == 0) ? kTokenFlagDecode : kTokenFlagPrefill;
    }

    const size_t q_shape[3] = {2, 1, 2};
    marmot_tensor_t *q = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *k_new = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *v_new = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(out);

    float q_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float v_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    marmot_float16_t q_vals_f16[4];
    marmot_float16_t k_vals_f16[4];
    marmot_float16_t v_vals_f16[4];
    for (size_t i = 0; i < 4; ++i) {
        q_vals_f16[i] = marmot_f32_to_f16_ref(q_vals[i]);
        k_vals_f16[i] = marmot_f32_to_f16_ref(k_vals[i]);
        v_vals_f16[i] = marmot_f32_to_f16_ref(v_vals[i]);
    }
    memcpy(q->data, q_vals_f16, sizeof(q_vals_f16));
    memcpy(k_new->data, k_vals_f16, sizeof(k_vals_f16));
    memcpy(v_new->data, v_vals_f16, sizeof(v_vals_f16));

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

    assert_int_equal(marmot_paged_attention(env->ctx, &desc), MARMOT_SUCCESS);

    const marmot_float16_t *out_data = marmot_tensor_data_f16(env->ctx, out);
    assert_non_null(out_data);
    assert_true(fabsf(marmot_f16_to_f32_ref(out_data[0]) - 1.0f) < 5e-3f);
    assert_true(fabsf(marmot_f16_to_f32_ref(out_data[1]) - 0.0f) < 5e-3f);

    const float exp1 = expf(1.0f);
    const float w0 = 1.0f / (1.0f + exp1);
    const float w1 = exp1 / (1.0f + exp1);
    assert_true(fabsf(marmot_f16_to_f32_ref(out_data[2]) - w0) < 5e-3f);
    assert_true(fabsf(marmot_f16_to_f32_ref(out_data[3]) - w1) < 5e-3f);

    const uint32_t shift = log2_u32((uint32_t)opts.block_size);
    const uint32_t block_id = slots[0] >> shift;
    assert_int_equal(block_id, slots[1] >> shift);

    const float *kv_k_data = marmot_tensor_data_f32(env->ctx, kv_k);
    const float *kv_v_data = marmot_tensor_data_f32(env->ctx, kv_v);
    assert_non_null(kv_k_data);
    assert_non_null(kv_v_data);

    size_t idx0 = kv_index_5d(kv_k, block_id, 0, 0, 0, 0);
    size_t idx1 = kv_index_5d(kv_k, block_id, 0, 0, 0, 1);
    size_t idx2 = kv_index_5d(kv_k, block_id, 0, 0, 1, 0);
    size_t idx3 = kv_index_5d(kv_k, block_id, 0, 0, 1, 1);
    assert_true(fabsf(kv_k_data[idx0] - 1.0f) < 1e-4f);
    assert_true(fabsf(kv_k_data[idx1] - 0.0f) < 1e-4f);
    assert_true(fabsf(kv_k_data[idx2] - 0.0f) < 1e-4f);
    assert_true(fabsf(kv_k_data[idx3] - 1.0f) < 1e-4f);

    assert_true(fabsf(kv_v_data[idx0] - 1.0f) < 1e-4f);
    assert_true(fabsf(kv_v_data[idx1] - 0.0f) < 1e-4f);
    assert_true(fabsf(kv_v_data[idx2] - 0.0f) < 1e-4f);
    assert_true(fabsf(kv_v_data[idx3] - 1.0f) < 1e-4f);

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq), MARMOT_SUCCESS);

    marmot_kv_pool_destroy(pool);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(v_new);
    marmot_tensor_destroy(k_new);
    marmot_tensor_destroy(q);
    marmot_tensor_destroy(token_meta);
}

static void test_paged_attention_invalid_token_meta_ranges(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = env->backend;
    opts.max_seqs = 1;
    opts.max_seq_len = 4;
    opts.block_size = 4;
    opts.num_blocks = 1;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 2;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(env->ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq), MARMOT_SUCCESS);

    marmot_kv_append_plan_t plan = {0};
    marmot_kv_slot_t slots[1] = {0};
    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq, 1, &plan, slots, &start_pos), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan), MARMOT_SUCCESS);

    const size_t meta_shape[2] = {1, 4};
    marmot_tensor_t *token_meta = marmot_tensor_create(env->ctx, meta_shape, 2, MARMOT_DTYPE_UINT32);
    assert_non_null(token_meta);

    marmot_uint32_t *meta_data = marmot_tensor_data_u32_mut(env->ctx, token_meta);
    assert_non_null(meta_data);
    meta_data[0].value = 1u;
    meta_data[1].value = (uint32_t)start_pos;
    meta_data[2].value = slots[0];
    meta_data[3].value = kTokenFlagPrefill;

    const size_t q_shape[3] = {1, 1, 2};
    marmot_tensor_t *q = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *k_new = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *v_new = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(out);

    float q_vals[2] = {1.0f, 0.0f};
    float k_vals[2] = {1.0f, 0.0f};
    float v_vals[2] = {1.0f, 0.0f};
    memcpy(q->data, q_vals, sizeof(q_vals));
    memcpy(k_new->data, k_vals, sizeof(k_vals));
    memcpy(v_new->data, v_vals, sizeof(v_vals));

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    assert_int_equal(marmot_kv_pool_get_tensors(pool, &kv_k, &kv_v, &block_table), MARMOT_SUCCESS);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);

    marmot_paged_attention_desc_t desc = marmot_paged_attention_desc_default();
    desc.token_count = 1;
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

    assert_int_equal(marmot_paged_attention(env->ctx, &desc), MARMOT_ERROR_DIMENSION_MISMATCH);

    meta_data[0].value = seq;
    meta_data[2].value = (uint32_t)(opts.num_blocks << log2_u32((uint32_t)opts.block_size));
    assert_int_equal(marmot_paged_attention(env->ctx, &desc), MARMOT_ERROR_DIMENSION_MISMATCH);

    assert_int_equal(marmot_kv_pool_release_seq(pool, seq), MARMOT_SUCCESS);
    marmot_kv_pool_destroy(pool);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(v_new);
    marmot_tensor_destroy(k_new);
    marmot_tensor_destroy(q);
    marmot_tensor_destroy(token_meta);
}

static void test_paged_attention_prefill_mixed_kv_flash_metal(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    if (env->backend != MARMOT_BACKEND_METAL) {
        skip();
        return;
    }

    const size_t token_count = 128;
    const size_t num_q_heads = 4;
    const size_t num_kv_heads = 4;
    const size_t head_dim = 64;
    const size_t block_size = 16;
    const size_t q_count = token_count * num_q_heads * head_dim;
    const size_t kv_count = token_count * num_kv_heads * head_dim;

    float *q_host = (float *)malloc(sizeof(float) * q_count);
    float *k_host = (float *)malloc(sizeof(float) * kv_count);
    float *v_host = (float *)malloc(sizeof(float) * kv_count);
    float *out_metal = (float *)malloc(sizeof(float) * q_count);
    float *out_cpu = (float *)malloc(sizeof(float) * q_count);
    assert_non_null(q_host);
    assert_non_null(k_host);
    assert_non_null(v_host);
    assert_non_null(out_metal);
    assert_non_null(out_cpu);

    fill_pattern(q_host, q_count, 17u, 1.0f);
    fill_pattern(k_host, kv_count, 29u, 0.8f);
    fill_pattern(v_host, kv_count, 43u, 1.2f);

    run_paged_attention_prefill_case(
        env, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, token_count, num_q_heads, num_kv_heads, head_dim, block_size,
        q_host, k_host, v_host, out_metal
    );

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);
    marmot_test_env_t cpu_env = {
        .backend = MARMOT_BACKEND_CPU,
        .ctx = cpu_ctx,
    };
    run_paged_attention_prefill_case(
        &cpu_env, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, token_count, num_q_heads, num_kv_heads, head_dim,
        block_size, q_host, k_host, v_host, out_cpu
    );
    marmot_destroy(cpu_ctx);

    marmot_test_expect_close_array(out_metal, out_cpu, q_count, 3e-2f);

    free(out_cpu);
    free(out_metal);
    free(v_host);
    free(k_host);
    free(q_host);
}

static void test_kv_pool_cow_partial(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = env->backend;
    opts.max_seqs = 2;
    opts.max_seq_len = 8;
    opts.block_size = 4;
    opts.num_blocks = 3;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 1;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(env->ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq_a = 0;
    marmot_seq_slot_t seq_b = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq_a), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq_b), MARMOT_SUCCESS);

    marmot_kv_append_plan_t plan_a = {0};
    marmot_kv_slot_t slots_a[6] = {0};
    size_t start_pos_a = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq_a, 6, &plan_a, slots_a, &start_pos_a), MARMOT_SUCCESS);
    assert_int_equal(start_pos_a, 0);
    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan_a), MARMOT_SUCCESS);

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    assert_int_equal(marmot_kv_pool_get_tensors(pool, &kv_k, &kv_v, &block_table), MARMOT_SUCCESS);
    assert_non_null(block_table);

    const marmot_uint32_t *table = marmot_tensor_data_u32(env->ctx, block_table);
    assert_non_null(table);

    const size_t max_blocks_per_seq = (opts.max_seq_len + opts.block_size - 1) / opts.block_size;
    marmot_block_id_t blocks[2] = {0};
    blocks[0] = table[(size_t)seq_a * max_blocks_per_seq + 0].value;
    blocks[1] = table[(size_t)seq_a * max_blocks_per_seq + 1].value;
    assert_true(blocks[0] != MARMOT_BLOCK_ID_INVALID);
    assert_true(blocks[1] != MARMOT_BLOCK_ID_INVALID);

    marmot_kv_prefix_plan_t plan_b = {0};
    const size_t prefix_len = 6;
    assert_int_equal(marmot_kv_pool_prepare_prefix_attach(pool, seq_b, blocks, 2, prefix_len, &plan_b), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_commit_prefix_attach(pool, &plan_b), MARMOT_SUCCESS);

    marmot_kv_append_plan_t plan_b_append = {0};
    marmot_kv_slot_t slot_b = 0;
    size_t start_pos_b = 0;
    assert_int_equal(
        marmot_kv_pool_prepare_append(pool, seq_b, 1, &plan_b_append, &slot_b, &start_pos_b), MARMOT_SUCCESS
    );
    assert_int_equal(start_pos_b, 6);
    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan_b_append), MARMOT_SUCCESS);

    const marmot_uint32_t *table_after = marmot_tensor_data_u32(env->ctx, block_table);
    assert_non_null(table_after);
    const marmot_block_id_t block_a = table_after[(size_t)seq_a * max_blocks_per_seq + 1].value;
    const marmot_block_id_t block_b = table_after[(size_t)seq_b * max_blocks_per_seq + 1].value;
    assert_true(block_a != MARMOT_BLOCK_ID_INVALID);
    assert_true(block_b != MARMOT_BLOCK_ID_INVALID);
    assert_int_not_equal(block_a, block_b);

    assert_int_equal(marmot_kv_pool_release_seq(pool, seq_b), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq_a), MARMOT_SUCCESS);
    marmot_kv_pool_destroy(pool);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_paged_attention_basic),
        cmocka_unit_test(test_paged_attention_mixed_kv_dtype),
        cmocka_unit_test(test_paged_attention_invalid_token_meta_ranges),
        cmocka_unit_test(test_paged_attention_prefill_mixed_kv_flash_metal),
        cmocka_unit_test(test_kv_pool_cow_partial),
    };

    return cmocka_run_group_tests(tests, marmot_test_backend_setup, marmot_test_backend_teardown);
}
