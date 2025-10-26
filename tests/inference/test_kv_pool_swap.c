/* clang-format off */
#include "marmot/config.h"
#include "marmot/device.h"
#include "marmot/inference/kv_pool.h"
#include "marmot/tensor.h"

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
/* clang-format on */

static size_t
kv_index_5d(const marmot_tensor_t *tensor, size_t block, size_t layer, size_t head, size_t offset, size_t d) {
    return block * tensor->shape.strides[0] + layer * tensor->shape.strides[1] + head * tensor->shape.strides[2] +
        offset * tensor->shape.strides[3] + d * tensor->shape.strides[4];
}

static size_t kv_index_3d(const marmot_tensor_t *tensor, size_t block, size_t layer, size_t head) {
    return block * tensor->shape.strides[0] + layer * tensor->shape.strides[1] + head * tensor->shape.strides[2];
}

static uint32_t log2_u32(uint32_t value) {
    uint32_t shift = 0;
    while (value > 1u) {
        value >>= 1u;
        shift++;
    }
    return shift;
}

static marmot_block_id_t
block_table_at(const marmot_tensor_t *block_table, marmot_seq_slot_t seq, size_t logical_block) {
    const marmot_uint32_t *data = (const marmot_uint32_t *)block_table->data;
    return data[(size_t)seq * block_table->shape.strides[0] + logical_block * block_table->shape.strides[1]].value;
}

static void test_kv_pool_deterministic_alloc(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (ctx == nullptr) {
        skip();
    }

    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.flags = MARMOT_KV_POOL_FLAG_DETERMINISTIC_ALLOC;
    opts.backend = MARMOT_BACKEND_CPU;
    opts.max_seqs = 1;
    opts.max_seq_len = 4;
    opts.block_size = 2;
    opts.num_blocks = 4;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 1;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq), MARMOT_SUCCESS);

    marmot_kv_slot_t slots[3] = {0};
    marmot_kv_append_plan_t plan = {0};
    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq, 3, &plan, slots, &start_pos), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq), MARMOT_SUCCESS);

    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq), MARMOT_SUCCESS);
    marmot_kv_slot_t slot = 0;
    marmot_kv_append_plan_t plan2 = {0};
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq, 2, &plan2, &slot, &start_pos), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan2), MARMOT_SUCCESS);

    const uint32_t shift = log2_u32((uint32_t)opts.block_size);
    const uint32_t block_id = slot >> shift;
    assert_int_equal(block_id, 0);

    assert_int_equal(marmot_kv_pool_release_seq(pool, seq), MARMOT_SUCCESS);
    marmot_kv_pool_destroy(pool);
    marmot_destroy(ctx);
}

static void test_kv_pool_swap_roundtrip(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (ctx == nullptr) {
        skip();
    }

    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = MARMOT_BACKEND_CPU;
    opts.max_seqs = 2;
    opts.max_seq_len = 4;
    opts.block_size = 2;
    opts.num_blocks = 2;
    opts.num_swap_blocks = 2;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 1;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq_a = 0;
    marmot_seq_slot_t seq_b = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq_a), MARMOT_SUCCESS);

    marmot_kv_slot_t slots_a[3] = {0};
    marmot_kv_append_plan_t plan_a = {0};
    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq_a, 3, &plan_a, slots_a, &start_pos), MARMOT_SUCCESS);

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    assert_int_equal(marmot_kv_pool_get_tensors(pool, &kv_k, &kv_v, &block_table), MARMOT_SUCCESS);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);

    float *kv_k_data = marmot_tensor_data_f32_mut(ctx, kv_k);
    float *kv_v_data = marmot_tensor_data_f32_mut(ctx, kv_v);
    assert_non_null(kv_k_data);
    assert_non_null(kv_v_data);

    const uint32_t shift = log2_u32((uint32_t)opts.block_size);
    for (size_t i = 0; i < 3; ++i) {
        const uint32_t block_id = slots_a[i] >> shift;
        const uint32_t offset = slots_a[i] & (opts.block_size - 1);
        const size_t idx = kv_index_5d(kv_k, block_id, 0, 0, offset, 0);
        kv_k_data[idx] = 100.0f + (float)i;
        kv_v_data[idx] = 200.0f + (float)i;
    }

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan_a), MARMOT_SUCCESS);

    assert_int_equal(marmot_kv_pool_swap_out_seq(pool, seq_a), MARMOT_SUCCESS);
    bool swapped = false;
    assert_int_equal(marmot_kv_pool_is_seq_swapped(pool, seq_a, &swapped), MARMOT_SUCCESS);
    assert_true(swapped);

    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq_b), MARMOT_SUCCESS);
    marmot_kv_slot_t slots_b[3] = {0};
    marmot_kv_append_plan_t plan_b = {0};
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq_b, 3, &plan_b, slots_b, &start_pos), MARMOT_SUCCESS);

    for (size_t i = 0; i < 3; ++i) {
        const uint32_t block_id = slots_b[i] >> shift;
        const uint32_t offset = slots_b[i] & (opts.block_size - 1);
        const size_t idx = kv_index_5d(kv_k, block_id, 0, 0, offset, 0);
        kv_k_data[idx] = 10.0f + (float)i;
        kv_v_data[idx] = 20.0f + (float)i;
    }

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan_b), MARMOT_SUCCESS);

    assert_int_equal(marmot_kv_pool_swap_in_seq(pool, seq_a), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_is_seq_swapped(pool, seq_a, &swapped), MARMOT_SUCCESS);
    assert_false(swapped);

    const marmot_tensor_t *block_table_tensor = block_table;
    for (size_t pos = 0; pos < 3; ++pos) {
        const size_t logical_block = pos / opts.block_size;
        const size_t offset = pos % opts.block_size;
        const marmot_block_id_t block_id = block_table_at(block_table_tensor, seq_a, logical_block);
        const size_t idx = kv_index_5d(kv_k, block_id, 0, 0, offset, 0);
        assert_true(fabsf(kv_k_data[idx] - (100.0f + (float)pos)) < 1e-6f);
        assert_true(fabsf(kv_v_data[idx] - (200.0f + (float)pos)) < 1e-6f);
    }

    assert_int_equal(marmot_kv_pool_release_seq(pool, seq_b), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq_a), MARMOT_SUCCESS);

    marmot_kv_pool_destroy(pool);
    marmot_destroy(ctx);
}

#if MARMOT_ENABLE_FP8
static void test_kv_pool_swap_roundtrip_fp8_scales(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (ctx == nullptr) {
        skip();
    }

    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = MARMOT_BACKEND_CPU;
    opts.max_seqs = 2;
    opts.max_seq_len = 4;
    opts.block_size = 2;
    opts.num_blocks = 2;
    opts.num_swap_blocks = 2;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 1;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT8_E4M3;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq_a = 0;
    marmot_seq_slot_t seq_b = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq_a), MARMOT_SUCCESS);

    marmot_kv_slot_t slots_a[3] = {0};
    marmot_kv_append_plan_t plan_a = {0};
    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq_a, 3, &plan_a, slots_a, &start_pos), MARMOT_SUCCESS);

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    marmot_tensor_t *kv_k_scale = nullptr;
    marmot_tensor_t *kv_v_scale = nullptr;
    assert_int_equal(marmot_kv_pool_get_tensors(pool, &kv_k, &kv_v, &block_table), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_get_scale_tensors(pool, &kv_k_scale, &kv_v_scale), MARMOT_SUCCESS);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);
    assert_non_null(kv_k_scale);
    assert_non_null(kv_v_scale);

    marmot_float8_e4m3_t *kv_k_data = marmot_tensor_data_fp8_e4m3_mut(ctx, kv_k);
    marmot_float8_e4m3_t *kv_v_data = marmot_tensor_data_fp8_e4m3_mut(ctx, kv_v);
    float *kv_k_scale_data = marmot_tensor_data_f32_mut(ctx, kv_k_scale);
    float *kv_v_scale_data = marmot_tensor_data_f32_mut(ctx, kv_v_scale);
    assert_non_null(kv_k_data);
    assert_non_null(kv_v_data);
    assert_non_null(kv_k_scale_data);
    assert_non_null(kv_v_scale_data);

    float expected_k_scale[2] = {0};
    float expected_v_scale[2] = {0};
    const uint32_t shift = log2_u32((uint32_t)opts.block_size);
    const marmot_tensor_t *block_table_tensor = block_table;
    for (size_t logical_block = 0; logical_block < 2; ++logical_block) {
        const marmot_block_id_t block_id = block_table_at(block_table_tensor, seq_a, logical_block);
        const size_t scale_index = kv_index_3d(kv_k_scale, block_id, 0, 0);
        expected_k_scale[logical_block] = 1.25f + (float)logical_block;
        expected_v_scale[logical_block] = 2.25f + (float)logical_block;
        kv_k_scale_data[scale_index] = expected_k_scale[logical_block];
        kv_v_scale_data[scale_index] = expected_v_scale[logical_block];
    }

    marmot_float8_e4m3_t expected_k[3];
    marmot_float8_e4m3_t expected_v[3];
    for (size_t i = 0; i < 3; ++i) {
        expected_k[i] = marmot_make_fp8_e4m3((uint8_t)(0x20u + (uint8_t)i));
        expected_v[i] = marmot_make_fp8_e4m3((uint8_t)(0x40u + (uint8_t)i));
        const uint32_t block_id = slots_a[i] >> shift;
        const uint32_t offset = slots_a[i] & (opts.block_size - 1);
        const size_t idx = kv_index_5d(kv_k, block_id, 0, 0, offset, 0);
        kv_k_data[idx] = expected_k[i];
        kv_v_data[idx] = expected_v[i];
    }

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan_a), MARMOT_SUCCESS);

    assert_int_equal(marmot_kv_pool_swap_out_seq(pool, seq_a), MARMOT_SUCCESS);
    bool swapped = false;
    assert_int_equal(marmot_kv_pool_is_seq_swapped(pool, seq_a, &swapped), MARMOT_SUCCESS);
    assert_true(swapped);

    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq_b), MARMOT_SUCCESS);
    marmot_kv_slot_t slots_b[3] = {0};
    marmot_kv_append_plan_t plan_b = {0};
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq_b, 3, &plan_b, slots_b, &start_pos), MARMOT_SUCCESS);
    for (size_t i = 0; i < 3; ++i) {
        const uint32_t block_id = slots_b[i] >> shift;
        const uint32_t offset = slots_b[i] & (opts.block_size - 1);
        const size_t idx = kv_index_5d(kv_k, block_id, 0, 0, offset, 0);
        kv_k_data[idx] = marmot_make_fp8_e4m3((uint8_t)(0x60u + (uint8_t)i));
        kv_v_data[idx] = marmot_make_fp8_e4m3((uint8_t)(0x70u + (uint8_t)i));
    }
    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan_b), MARMOT_SUCCESS);

    assert_int_equal(marmot_kv_pool_swap_in_seq(pool, seq_a), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_is_seq_swapped(pool, seq_a, &swapped), MARMOT_SUCCESS);
    assert_false(swapped);

    for (size_t pos = 0; pos < 3; ++pos) {
        const size_t logical_block = pos / opts.block_size;
        const size_t offset = pos % opts.block_size;
        const marmot_block_id_t block_id = block_table_at(block_table_tensor, seq_a, logical_block);
        const size_t idx = kv_index_5d(kv_k, block_id, 0, 0, offset, 0);
        assert_int_equal(kv_k_data[idx].bits, expected_k[pos].bits);
        assert_int_equal(kv_v_data[idx].bits, expected_v[pos].bits);
    }

    for (size_t logical_block = 0; logical_block < 2; ++logical_block) {
        const marmot_block_id_t block_id = block_table_at(block_table_tensor, seq_a, logical_block);
        const size_t scale_index = kv_index_3d(kv_k_scale, block_id, 0, 0);
        assert_true(fabsf(kv_k_scale_data[scale_index] - expected_k_scale[logical_block]) < 1e-6f);
        assert_true(fabsf(kv_v_scale_data[scale_index] - expected_v_scale[logical_block]) < 1e-6f);
    }

    assert_int_equal(marmot_kv_pool_release_seq(pool, seq_b), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq_a), MARMOT_SUCCESS);

    marmot_kv_pool_destroy(pool);
    marmot_destroy(ctx);
}

static void test_kv_pool_clone_preserves_scales(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (ctx == nullptr) {
        skip();
    }

    marmot_kv_pool_options_t opts;
    assert_int_equal(marmot_kv_pool_options_init(&opts), MARMOT_SUCCESS);
    opts.backend = MARMOT_BACKEND_CPU;
    opts.max_seqs = 2;
    opts.max_seq_len = 4;
    opts.block_size = 4;
    opts.num_blocks = 2;
    opts.num_layers = 1;
    opts.num_kv_heads = 1;
    opts.head_dim = 1;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT8_E4M3;

    marmot_kv_pool_t *pool = nullptr;
    assert_int_equal(marmot_kv_pool_create(ctx, &opts, &pool), MARMOT_SUCCESS);
    assert_non_null(pool);

    marmot_seq_slot_t seq_a = 0;
    marmot_seq_slot_t seq_b = 0;
    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq_a), MARMOT_SUCCESS);

    marmot_kv_slot_t slots_a[4] = {0};
    marmot_kv_append_plan_t plan_a = {0};
    size_t start_pos = 0;
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq_a, 4, &plan_a, slots_a, &start_pos), MARMOT_SUCCESS);

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    marmot_tensor_t *kv_k_scale = nullptr;
    marmot_tensor_t *kv_v_scale = nullptr;
    assert_int_equal(marmot_kv_pool_get_tensors(pool, &kv_k, &kv_v, &block_table), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_get_scale_tensors(pool, &kv_k_scale, &kv_v_scale), MARMOT_SUCCESS);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);
    assert_non_null(kv_k_scale);
    assert_non_null(kv_v_scale);

    marmot_float8_e4m3_t *kv_k_data = marmot_tensor_data_fp8_e4m3_mut(ctx, kv_k);
    marmot_float8_e4m3_t *kv_v_data = marmot_tensor_data_fp8_e4m3_mut(ctx, kv_v);
    float *kv_k_scale_data = marmot_tensor_data_f32_mut(ctx, kv_k_scale);
    float *kv_v_scale_data = marmot_tensor_data_f32_mut(ctx, kv_v_scale);
    assert_non_null(kv_k_data);
    assert_non_null(kv_v_data);
    assert_non_null(kv_k_scale_data);
    assert_non_null(kv_v_scale_data);

    const uint32_t shift = log2_u32((uint32_t)opts.block_size);
    const marmot_tensor_t *block_table_tensor = block_table;
    const marmot_block_id_t block_id = block_table_at(block_table_tensor, seq_a, 0);
    const size_t scale_index = kv_index_3d(kv_k_scale, block_id, 0, 0);
    kv_k_scale_data[scale_index] = 1.5f;
    kv_v_scale_data[scale_index] = 2.5f;

    const size_t idx0 = kv_index_5d(kv_k, block_id, 0, 0, 0, 0);
    kv_k_data[idx0] = marmot_make_fp8_e4m3(0x22u);
    kv_v_data[idx0] = marmot_make_fp8_e4m3(0x33u);

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan_a), MARMOT_SUCCESS);

    assert_int_equal(marmot_kv_pool_acquire_seq(pool, &seq_b), MARMOT_SUCCESS);
    marmot_block_id_t prefix_blocks[1] = {block_id};
    marmot_kv_prefix_plan_t prefix_plan = {0};
    assert_int_equal(
        marmot_kv_pool_prepare_prefix_attach(pool, seq_b, prefix_blocks, 1, opts.block_size - 1, &prefix_plan),
        MARMOT_SUCCESS
    );
    assert_int_equal(marmot_kv_pool_commit_prefix_attach(pool, &prefix_plan), MARMOT_SUCCESS);

    marmot_kv_slot_t slot_b = 0;
    marmot_kv_append_plan_t plan_b = {0};
    assert_int_equal(marmot_kv_pool_prepare_append(pool, seq_b, 1, &plan_b, &slot_b, &start_pos), MARMOT_SUCCESS);
    const marmot_block_id_t clone_block_id = slot_b >> shift;
    assert_true(clone_block_id != block_id);

    const size_t clone_scale_index = kv_index_3d(kv_k_scale, clone_block_id, 0, 0);
    assert_true(fabsf(kv_k_scale_data[clone_scale_index] - 1.5f) < 1e-6f);
    assert_true(fabsf(kv_v_scale_data[clone_scale_index] - 2.5f) < 1e-6f);

    const size_t clone_idx0 = kv_index_5d(kv_k, clone_block_id, 0, 0, 0, 0);
    assert_int_equal(kv_k_data[clone_idx0].bits, 0x22u);
    assert_int_equal(kv_v_data[clone_idx0].bits, 0x33u);

    assert_int_equal(marmot_kv_pool_commit_append(pool, &plan_b), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq_b), MARMOT_SUCCESS);
    assert_int_equal(marmot_kv_pool_release_seq(pool, seq_a), MARMOT_SUCCESS);

    marmot_kv_pool_destroy(pool);
    marmot_destroy(ctx);
}
#else
static void test_kv_pool_swap_roundtrip_fp8_scales(void **state) {
    (void)state;
    skip();
}

static void test_kv_pool_clone_preserves_scales(void **state) {
    (void)state;
    skip();
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_kv_pool_deterministic_alloc),
        cmocka_unit_test(test_kv_pool_swap_roundtrip),
        cmocka_unit_test(test_kv_pool_swap_roundtrip_fp8_scales),
        cmocka_unit_test(test_kv_pool_clone_preserves_scales),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
