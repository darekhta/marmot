/* clang-format off */
#include "marmot/device.h"
#include "marmot/inference/engine.h"
#include "marmot/inference/llm.h"
#include "marmot/inference/model.h"

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

#include "test_fixture_utils.h"

static _Thread_local char g_fixture_path[MARMOT_TEST_PATH_MAX];

static uint32_t log2_u32(uint32_t value) {
    uint32_t shift = 0;
    while (value > 1u) {
        value >>= 1u;
        shift++;
    }
    return shift;
}

static void test_serving_engine_batch_pack(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = marmot_test_get_fixture_path(fixture->filename, g_fixture_path, sizeof(g_fixture_path));
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            skip();
        }

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        if (marmot_model_load_file(path, &model_opts, &model) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);
        assert_true(info.n_vocab > 3);

        marmot_serving_engine_options_t opts;
        assert_int_equal(marmot_serving_engine_options_init(&opts), MARMOT_SUCCESS);
        opts.max_seqs = 1;
        opts.max_batch_seqs = 1;
        opts.max_num_tokens = 4;
        opts.max_seq_len = 8;
        opts.block_size = 4;
        opts.num_kv_blocks = 4;
        opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

        marmot_serving_engine_t *engine = nullptr;
        assert_int_equal(marmot_serving_engine_create(ctx, model, &opts, &engine), MARMOT_SUCCESS);
        assert_non_null(engine);

        marmot_token_id_t prompt[3] = {1, 2, 3};
        marmot_llm_generate_options_t gen_opts;
        assert_int_equal(marmot_llm_generate_options_init(&gen_opts), MARMOT_SUCCESS);
        gen_opts.max_new_tokens = 1;

        marmot_llm_sampling_options_t sampling_opts;
        assert_int_equal(marmot_llm_sampling_options_init(&sampling_opts), MARMOT_SUCCESS);

        marmot_request_id_t request_id = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt, 3, &gen_opts, &sampling_opts, &request_id), MARMOT_SUCCESS
        );

        size_t steps_done = 0;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_engine_batch_view_t batch;
        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);

        assert_int_equal(batch.token_count, 3);
        assert_int_equal(batch.sample_count, 1);
        assert_non_null(batch.token_ids);
        assert_non_null(batch.token_meta);
        assert_non_null(batch.sample_indices);
        assert_non_null(batch.sample_request_ids);

        assert_int_equal(batch.token_ids[0], prompt[0]);
        assert_int_equal(batch.token_ids[1], prompt[1]);
        assert_int_equal(batch.token_ids[2], prompt[2]);
        assert_int_equal(batch.sample_indices[0], 2);
        assert_int_equal(batch.sample_request_ids[0], request_id);

        const uint32_t shift = log2_u32((uint32_t)opts.block_size);
        const uint32_t first_block = batch.token_meta[2] >> shift;
        for (size_t t = 0; t < batch.token_count; ++t) {
            const uint32_t seq_slot = batch.token_meta[t * 4 + 0];
            const uint32_t pos = batch.token_meta[t * 4 + 1];
            const uint32_t kv_slot = batch.token_meta[t * 4 + 2];
            const uint32_t block_id = kv_slot >> shift;
            const uint32_t offset = kv_slot & (opts.block_size - 1);
            assert_int_equal(seq_slot, 0);
            assert_int_equal(pos, t);
            assert_int_equal(offset, t);
            assert_int_equal(block_id, first_block);
        }

        marmot_serving_engine_destroy(engine);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

static void test_serving_engine_decode_first(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = marmot_test_get_fixture_path(fixture->filename, g_fixture_path, sizeof(g_fixture_path));
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            skip();
        }

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        if (marmot_model_load_file(path, &model_opts, &model) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);
        if (info.n_vocab <= 8) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_serving_engine_options_t opts;
        assert_int_equal(marmot_serving_engine_options_init(&opts), MARMOT_SUCCESS);
        opts.max_seqs = 2;
        opts.max_batch_seqs = 2;
        opts.max_num_tokens = 5;
        opts.max_seq_len = 16;
        opts.block_size = 4;
        opts.num_kv_blocks = 8;
        opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

        marmot_serving_engine_t *engine = nullptr;
        assert_int_equal(marmot_serving_engine_create(ctx, model, &opts, &engine), MARMOT_SUCCESS);
        assert_non_null(engine);

        marmot_llm_generate_options_t gen_opts;
        assert_int_equal(marmot_llm_generate_options_init(&gen_opts), MARMOT_SUCCESS);
        gen_opts.max_new_tokens = 2;

        marmot_llm_sampling_options_t sampling_opts;
        assert_int_equal(marmot_llm_sampling_options_init(&sampling_opts), MARMOT_SUCCESS);

        marmot_token_id_t prompt_a[1] = {1};
        marmot_token_id_t prompt_b[8] = {1, 2, 3, 4, 5, 6, 7, 8};

        marmot_request_id_t req_a = 0;
        marmot_request_id_t req_b = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt_a, 1, &gen_opts, &sampling_opts, &req_a), MARMOT_SUCCESS
        );
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt_b, 8, &gen_opts, &sampling_opts, &req_b), MARMOT_SUCCESS
        );
        (void)req_a;
        (void)req_b;

        size_t steps_done = 0;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_engine_batch_view_t batch;
        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);
        assert_int_equal(batch.token_count, 5);
        assert_int_equal(batch.sample_count, 2);
        assert_non_null(batch.token_meta);

        const uint32_t kPrefill = 1u << 0;
        const uint32_t kDecode = 1u << 1;
        const uint32_t first_flags = batch.token_meta[0 * 4 + 3];
        assert_true((first_flags & kDecode) != 0);
        assert_true((first_flags & kPrefill) == 0);

        for (size_t i = 1; i < batch.token_count; ++i) {
            const uint32_t flags = batch.token_meta[i * 4 + 3];
            assert_true((flags & kPrefill) != 0);
            assert_true((flags & kDecode) == 0);
        }

        marmot_serving_engine_destroy(engine);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

static void test_serving_engine_kv_watermark(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = marmot_test_get_fixture_path(fixture->filename, g_fixture_path, sizeof(g_fixture_path));
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            skip();
        }

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        if (marmot_model_load_file(path, &model_opts, &model) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);
        if (info.n_vocab <= 8) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_serving_engine_options_t opts;
        assert_int_equal(marmot_serving_engine_options_init(&opts), MARMOT_SUCCESS);
        opts.max_seqs = 2;
        opts.max_batch_seqs = 2;
        opts.max_num_tokens = 4;
        opts.max_seq_len = 8;
        opts.block_size = 4;
        opts.num_kv_blocks = 2;
        opts.kv_dtype = MARMOT_DTYPE_FLOAT32;
        opts.kv_block_watermark = 0.5f;

        marmot_serving_engine_t *engine = nullptr;
        assert_int_equal(marmot_serving_engine_create(ctx, model, &opts, &engine), MARMOT_SUCCESS);
        assert_non_null(engine);

        marmot_llm_generate_options_t gen_opts;
        assert_int_equal(marmot_llm_generate_options_init(&gen_opts), MARMOT_SUCCESS);
        gen_opts.max_new_tokens = 2;

        marmot_llm_sampling_options_t sampling_opts;
        assert_int_equal(marmot_llm_sampling_options_init(&sampling_opts), MARMOT_SUCCESS);

        marmot_token_id_t prompt_a[4] = {1, 2, 3, 4};
        marmot_token_id_t prompt_b[4] = {1, 2, 3, 4};

        marmot_request_id_t req_a = 0;
        marmot_request_id_t req_b = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt_a, 4, &gen_opts, &sampling_opts, &req_a), MARMOT_SUCCESS
        );

        size_t steps_done = 0;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt_b, 4, &gen_opts, &sampling_opts, &req_b), MARMOT_SUCCESS
        );
        (void)req_b;

        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_engine_batch_view_t batch;
        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);
        assert_int_equal(batch.token_count, 1);
        assert_int_equal(batch.sample_count, 1);
        assert_non_null(batch.token_meta);
        assert_non_null(batch.sample_request_ids);

        const uint32_t kPrefill = 1u << 0;
        const uint32_t kDecode = 1u << 1;
        const uint32_t flags = batch.token_meta[0 * 4 + 3];
        assert_true((flags & kDecode) != 0);
        assert_true((flags & kPrefill) == 0);
        assert_int_equal(batch.sample_request_ids[0], req_a);

        marmot_serving_engine_destroy(engine);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

static void test_serving_engine_prefix_cache(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = marmot_test_get_fixture_path(fixture->filename, g_fixture_path, sizeof(g_fixture_path));
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            skip();
        }

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        if (marmot_model_load_file(path, &model_opts, &model) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);
        if (info.n_vocab <= 8) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_serving_engine_options_t opts;
        assert_int_equal(marmot_serving_engine_options_init(&opts), MARMOT_SUCCESS);
        opts.max_seqs = 1;
        opts.max_batch_seqs = 1;
        opts.max_num_tokens = 8;
        opts.max_seq_len = 16;
        opts.block_size = 4;
        opts.num_kv_blocks = 8;
        opts.kv_dtype = MARMOT_DTYPE_FLOAT32;
        opts.flags = MARMOT_SERVING_ENGINE_FLAG_ENABLE_PREFIX_CACHE;
        opts.prefill_chunk_size = 8;

        marmot_serving_engine_t *engine = nullptr;
        assert_int_equal(marmot_serving_engine_create(ctx, model, &opts, &engine), MARMOT_SUCCESS);
        assert_non_null(engine);

        marmot_llm_generate_options_t gen_opts;
        assert_int_equal(marmot_llm_generate_options_init(&gen_opts), MARMOT_SUCCESS);
        gen_opts.max_new_tokens = 1;

        marmot_llm_sampling_options_t sampling_opts;
        assert_int_equal(marmot_llm_sampling_options_init(&sampling_opts), MARMOT_SUCCESS);
        sampling_opts.temperature = 0.0f;

        marmot_token_id_t prompt[8] = {1, 2, 3, 4, 5, 6, 7, 8};

        marmot_serving_request_ext_t ext = {
            .struct_size = sizeof(ext),
            .struct_version = MARMOT_SERVING_REQUEST_EXT_VERSION,
            .flags = 0,
            .priority = 0,
            .cache_salt = "salt",
            .cache_salt_len = 4,
            .retention_blocks = 0,
            .num_samples = 0,
            .sample_user_data = nullptr,
            .sample_user_data_len = 0,
            .out_request_ids = nullptr,
            .out_request_ids_capacity = 0,
            .pnext = nullptr,
        };
        gen_opts.pnext = &ext;

        marmot_request_id_t req_id = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt, 8, &gen_opts, &sampling_opts, &req_id), MARMOT_SUCCESS
        );
        (void)req_id;

        size_t steps_done = 0;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_request_ext_t ext_same = ext;
        gen_opts.pnext = &ext_same;

        marmot_request_id_t req_cached = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt, 8, &gen_opts, &sampling_opts, &req_cached), MARMOT_SUCCESS
        );
        (void)req_cached;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_engine_batch_view_t batch;
        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);
        assert_int_equal(batch.token_count, 4);
        assert_int_equal(batch.sample_count, 1);
        for (size_t i = 0; i < batch.token_count; ++i) {
            const uint32_t pos = batch.token_meta[i * 4 + 1];
            assert_int_equal(pos, 4 + i);
        }

        marmot_serving_request_ext_t ext_other = ext;
        ext_other.cache_salt = "other";
        ext_other.cache_salt_len = 5;
        gen_opts.pnext = &ext_other;

        marmot_request_id_t req_uncached = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt, 8, &gen_opts, &sampling_opts, &req_uncached), MARMOT_SUCCESS
        );
        (void)req_uncached;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);
        assert_int_equal(batch.token_count, 8);
        assert_int_equal(batch.sample_count, 1);
        assert_int_equal(batch.token_meta[1], 0);

        marmot_serving_engine_destroy(engine);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

static void test_serving_engine_retention_blocks(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = marmot_test_get_fixture_path(fixture->filename, g_fixture_path, sizeof(g_fixture_path));
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            skip();
        }

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        if (marmot_model_load_file(path, &model_opts, &model) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);
        if (info.n_vocab <= 12) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_serving_engine_options_t opts;
        assert_int_equal(marmot_serving_engine_options_init(&opts), MARMOT_SUCCESS);
        opts.max_seqs = 1;
        opts.max_batch_seqs = 1;
        opts.max_num_tokens = 8;
        opts.max_seq_len = 8;
        opts.block_size = 4;
        opts.num_kv_blocks = 2;
        opts.kv_dtype = MARMOT_DTYPE_FLOAT32;
        opts.prefill_chunk_size = 8;
        opts.kv_block_watermark = 0.0f;
        opts.flags = MARMOT_SERVING_ENGINE_FLAG_ENABLE_PREFIX_CACHE;

        marmot_serving_engine_t *engine = nullptr;
        assert_int_equal(marmot_serving_engine_create(ctx, model, &opts, &engine), MARMOT_SUCCESS);
        assert_non_null(engine);

        marmot_llm_generate_options_t gen_opts;
        assert_int_equal(marmot_llm_generate_options_init(&gen_opts), MARMOT_SUCCESS);
        gen_opts.max_new_tokens = 0;

        marmot_llm_sampling_options_t sampling_opts;
        assert_int_equal(marmot_llm_sampling_options_init(&sampling_opts), MARMOT_SUCCESS);
        sampling_opts.temperature = 0.0f;

        marmot_token_id_t prompt_a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        marmot_token_id_t prompt_b[4] = {9, 10, 11, 12};

        marmot_serving_request_ext_t ext = {
            .struct_size = sizeof(ext),
            .struct_version = MARMOT_SERVING_REQUEST_EXT_VERSION,
            .flags = 0,
            .priority = 0,
            .cache_salt = "salt",
            .cache_salt_len = 4,
            .retention_blocks = 1,
            .num_samples = 0,
            .sample_user_data = nullptr,
            .sample_user_data_len = 0,
            .out_request_ids = nullptr,
            .out_request_ids_capacity = 0,
            .pnext = nullptr,
        };
        gen_opts.pnext = &ext;

        marmot_request_id_t req_a = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt_a, 8, &gen_opts, &sampling_opts, &req_a), MARMOT_SUCCESS
        );
        (void)req_a;

        size_t steps_done = 0;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_request_ext_t ext_b = ext;
        ext_b.retention_blocks = 0;
        gen_opts.pnext = &ext_b;

        marmot_request_id_t req_b = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt_b, 4, &gen_opts, &sampling_opts, &req_b), MARMOT_SUCCESS
        );
        (void)req_b;

        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_engine_batch_view_t batch;
        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);
        assert_int_equal(batch.token_count, 4);
        assert_int_equal(batch.sample_count, 0);
        assert_non_null(batch.token_meta);

        const uint32_t shift = log2_u32((uint32_t)opts.block_size);
        const uint32_t block_id = batch.token_meta[2] >> shift;
        assert_int_equal(block_id, 1);

        marmot_serving_engine_destroy(engine);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

static void test_serving_engine_parallel_sampling(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = marmot_test_get_fixture_path(fixture->filename, g_fixture_path, sizeof(g_fixture_path));
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            skip();
        }

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        if (marmot_model_load_file(path, &model_opts, &model) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);
        if (info.n_vocab <= 8) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_serving_engine_options_t opts;
        assert_int_equal(marmot_serving_engine_options_init(&opts), MARMOT_SUCCESS);
        opts.max_seqs = 2;
        opts.max_batch_seqs = 2;
        opts.max_num_tokens = 8;
        opts.max_seq_len = 16;
        opts.block_size = 4;
        opts.num_kv_blocks = 8;
        opts.kv_dtype = MARMOT_DTYPE_FLOAT32;
        opts.prefill_chunk_size = 8;

        marmot_serving_engine_t *engine = nullptr;
        assert_int_equal(marmot_serving_engine_create(ctx, model, &opts, &engine), MARMOT_SUCCESS);
        assert_non_null(engine);

        marmot_llm_generate_options_t gen_opts;
        assert_int_equal(marmot_llm_generate_options_init(&gen_opts), MARMOT_SUCCESS);
        gen_opts.max_new_tokens = 2;
        gen_opts.stop_on_eos = false;

        marmot_llm_sampling_options_t sampling_opts;
        assert_int_equal(marmot_llm_sampling_options_init(&sampling_opts), MARMOT_SUCCESS);
        sampling_opts.temperature = 0.0f;

        marmot_token_id_t prompt[4] = {1, 2, 3, 4};
        marmot_request_id_t out_ids[2] = {0, 0};

        marmot_serving_request_ext_t ext = {
            .struct_size = sizeof(ext),
            .struct_version = MARMOT_SERVING_REQUEST_EXT_VERSION,
            .flags = 0,
            .priority = 0,
            .cache_salt = nullptr,
            .cache_salt_len = 0,
            .retention_blocks = 0,
            .num_samples = 2,
            .sample_user_data = nullptr,
            .sample_user_data_len = 0,
            .out_request_ids = out_ids,
            .out_request_ids_capacity = 2,
            .pnext = nullptr,
        };
        gen_opts.pnext = &ext;

        marmot_request_id_t req_id = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt, 4, &gen_opts, &sampling_opts, &req_id), MARMOT_SUCCESS
        );
        assert_int_equal(out_ids[0], req_id);
        assert_true(out_ids[1] != 0);
        assert_true(out_ids[1] != req_id);

        size_t steps_done = 0;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        assert_int_equal(marmot_serving_engine_request_state(engine, out_ids[0]), MARMOT_LLM_REQUEST_STATE_DECODING);
        assert_int_equal(marmot_serving_engine_request_state(engine, out_ids[1]), MARMOT_LLM_REQUEST_STATE_DECODING);

        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_engine_batch_view_t batch;
        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);
        assert_int_equal(batch.token_count, 2);
        assert_int_equal(batch.sample_count, 2);

        marmot_serving_engine_destroy(engine);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

static void test_serving_engine_recompute_preemption(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = marmot_test_get_fixture_path(fixture->filename, g_fixture_path, sizeof(g_fixture_path));
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            skip();
        }

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        if (marmot_model_load_file(path, &model_opts, &model) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);
        if (info.n_vocab <= 4) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_serving_engine_options_t opts;
        assert_int_equal(marmot_serving_engine_options_init(&opts), MARMOT_SUCCESS);
        opts.max_seqs = 2;
        opts.max_batch_seqs = 2;
        opts.max_num_tokens = 8;
        opts.max_seq_len = 8;
        opts.block_size = 4;
        opts.num_kv_blocks = 2;
        opts.kv_dtype = MARMOT_DTYPE_FLOAT32;
        opts.prefill_chunk_size = 8;
        opts.kv_block_watermark = 0.0f;

        marmot_serving_engine_t *engine = nullptr;
        assert_int_equal(marmot_serving_engine_create(ctx, model, &opts, &engine), MARMOT_SUCCESS);
        assert_non_null(engine);

        marmot_llm_generate_options_t gen_opts;
        assert_int_equal(marmot_llm_generate_options_init(&gen_opts), MARMOT_SUCCESS);
        gen_opts.max_new_tokens = 2;

        marmot_llm_sampling_options_t sampling_opts;
        assert_int_equal(marmot_llm_sampling_options_init(&sampling_opts), MARMOT_SUCCESS);
        sampling_opts.temperature = 0.0f;

        marmot_token_id_t prompt[4] = {1, 2, 3, 4};

        marmot_serving_request_ext_t high_prio = {
            .struct_size = sizeof(high_prio),
            .struct_version = MARMOT_SERVING_REQUEST_EXT_VERSION,
            .flags = 0,
            .priority = 10,
            .cache_salt = nullptr,
            .cache_salt_len = 0,
            .retention_blocks = 0,
            .num_samples = 0,
            .sample_user_data = nullptr,
            .sample_user_data_len = 0,
            .out_request_ids = nullptr,
            .out_request_ids_capacity = 0,
            .pnext = nullptr,
        };
        marmot_serving_request_ext_t low_prio = high_prio;
        low_prio.priority = 0;

        gen_opts.pnext = &high_prio;
        marmot_request_id_t req_high = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt, 4, &gen_opts, &sampling_opts, &req_high), MARMOT_SUCCESS
        );

        gen_opts.pnext = &low_prio;
        marmot_request_id_t req_low = 0;
        assert_int_equal(
            marmot_serving_engine_submit(engine, prompt, 4, &gen_opts, &sampling_opts, &req_low), MARMOT_SUCCESS
        );
        (void)req_high;
        (void)req_low;

        size_t steps_done = 0;
        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        marmot_serving_engine_batch_view_t batch;
        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);
        assert_int_equal(batch.token_count, 1);
        assert_int_equal(batch.sample_count, 1);

        assert_int_equal(marmot_serving_engine_step(engine, 1, &steps_done), MARMOT_SUCCESS);
        assert_int_equal(steps_done, 1);

        assert_int_equal(marmot_serving_engine_last_batch(engine, &batch), MARMOT_SUCCESS);
        assert_int_equal(batch.token_count, 5);
        assert_int_equal(batch.sample_count, 1);
        for (size_t i = 0; i < batch.token_count; ++i) {
            const uint32_t pos = batch.token_meta[i * 4 + 1];
            assert_int_equal(pos, i);
        }

        marmot_serving_engine_destroy(engine);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_serving_engine_batch_pack),
        cmocka_unit_test(test_serving_engine_decode_first),
        cmocka_unit_test(test_serving_engine_kv_watermark),
        cmocka_unit_test(test_serving_engine_prefix_cache),
        cmocka_unit_test(test_serving_engine_retention_blocks),
        cmocka_unit_test(test_serving_engine_parallel_sampling),
        cmocka_unit_test(test_serving_engine_recompute_preemption),
    };

    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
