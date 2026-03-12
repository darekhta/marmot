/* clang-format off */
#include "marmot/device.h"
#include "marmot/graph/architecture.h"
#include "marmot/inference/engine.h"
#include "marmot/inference/llm.h"
#include "marmot/inference/model.h"
#include "marmot/tokenizer.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
/* clang-format on */

#include "golden_llm_outputs.h"
#include "test_fixture_utils.h"

static _Thread_local char g_fixture_path[MARMOT_TEST_PATH_MAX];

typedef struct {
    marmot_token_id_t *tokens;
    size_t capacity;
    size_t count;
} token_capture_t;

static const char *request_state_name(marmot_llm_request_state_t state) {
    switch (state) {
    case MARMOT_LLM_REQUEST_STATE_INVALID:
        return "invalid";
    case MARMOT_LLM_REQUEST_STATE_PENDING:
        return "pending";
    case MARMOT_LLM_REQUEST_STATE_PREFILL:
        return "prefill";
    case MARMOT_LLM_REQUEST_STATE_DECODING:
        return "decoding";
    case MARMOT_LLM_REQUEST_STATE_DONE:
        return "done";
    case MARMOT_LLM_REQUEST_STATE_FAILED:
        return "failed";
    case MARMOT_LLM_REQUEST_STATE_CANCELED:
        return "canceled";
    default:
        return "unknown";
    }
}

static const char *get_fixture_path(const char *filename) {
    return marmot_test_get_fixture_path(filename, g_fixture_path, sizeof(g_fixture_path));
}

static void capture_token(void *user_data, marmot_token_id_t token_id) {
    token_capture_t *capture = (token_capture_t *)user_data;
    if (capture == nullptr || capture->tokens == nullptr) {
        return;
    }
    if (capture->count < capture->capacity) {
        capture->tokens[capture->count++] = token_id;
    }
}

static void init_engine_options(
    marmot_serving_engine_options_t *opts, size_t max_seq_len, size_t max_num_tokens, size_t block_size,
    marmot_dtype_t kv_dtype
) {
    assert_int_equal(marmot_serving_engine_options_init(opts), MARMOT_SUCCESS);
    opts->flags = MARMOT_SERVING_ENGINE_FLAG_DETERMINISTIC_KV;
    opts->max_seqs = 1;
    opts->max_batch_seqs = 1;
    opts->max_seq_len = max_seq_len;
    opts->max_num_tokens = max_num_tokens;
    opts->block_size = block_size;
    size_t blocks_per_seq = (max_seq_len + opts->block_size - 1) / opts->block_size;
    if (blocks_per_seq == 0) {
        blocks_per_seq = 1;
    }
    opts->num_kv_blocks = blocks_per_seq * opts->max_seqs;
    if (opts->num_kv_blocks < 4) {
        opts->num_kv_blocks = 4;
    }
    opts->kv_dtype = kv_dtype;
    opts->prefill_chunk_size = 0;
}

static void init_engine_options_multi(
    marmot_serving_engine_options_t *opts, size_t max_seqs, size_t max_batch_seqs, size_t max_seq_len,
    size_t max_num_tokens, size_t block_size, marmot_dtype_t kv_dtype
) {
    assert_int_equal(marmot_serving_engine_options_init(opts), MARMOT_SUCCESS);
    opts->flags = MARMOT_SERVING_ENGINE_FLAG_DETERMINISTIC_KV;
    opts->max_seqs = max_seqs;
    opts->max_batch_seqs = max_batch_seqs;
    opts->max_seq_len = max_seq_len;
    opts->max_num_tokens = max_num_tokens;
    opts->block_size = block_size;
    size_t blocks_per_seq = (max_seq_len + opts->block_size - 1) / opts->block_size;
    if (blocks_per_seq == 0) {
        blocks_per_seq = 1;
    }
    opts->num_kv_blocks = blocks_per_seq * opts->max_seqs;
    if (opts->num_kv_blocks < 4) {
        opts->num_kv_blocks = 4;
    }
    opts->kv_dtype = kv_dtype;
    opts->prefill_chunk_size = 0;
}

static marmot_error_t
encode_prompt(marmot_tokenizer_t *tok, const char *prompt, marmot_token_id_t **out_tokens, size_t *out_len) {
    if (tok == nullptr || prompt == nullptr || out_tokens == nullptr || out_len == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_tokens = nullptr;
    *out_len = 0;

    marmot_tokenizer_encode_options_t enc_opts;
    if (marmot_tokenizer_encode_options_init(&enc_opts) != MARMOT_SUCCESS) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    enc_opts.add_bos = true;
    enc_opts.add_eos = false;

    size_t prompt_len = 0;
    marmot_error_t err = marmot_tokenizer_encode(tok, prompt, strlen(prompt), &enc_opts, nullptr, &prompt_len);
    if (err != MARMOT_SUCCESS || prompt_len == 0) {
        return err != MARMOT_SUCCESS ? err : MARMOT_ERROR_INVALID_OPERATION;
    }

    marmot_token_id_t *tokens = malloc(prompt_len * sizeof(*tokens));
    if (tokens == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    size_t cap = prompt_len;
    err = marmot_tokenizer_encode(tok, prompt, strlen(prompt), &enc_opts, tokens, &cap);
    if (err != MARMOT_SUCCESS || cap != prompt_len) {
        free(tokens);
        return err != MARMOT_SUCCESS ? err : MARMOT_ERROR_INVALID_OPERATION;
    }

    *out_tokens = tokens;
    *out_len = prompt_len;
    return MARMOT_SUCCESS;
}

static marmot_error_t run_serving_generate_with_limits(
    marmot_context_t *ctx, marmot_model_t *model, marmot_backend_type_t backend, const marmot_token_id_t *prompt_tokens,
    size_t prompt_len, size_t block_size, size_t max_new_tokens, size_t max_num_tokens_override,
    marmot_token_id_t *out_tokens, size_t out_capacity, size_t *out_len
) {
    if (out_len == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_len = 0;

    const size_t total_len = prompt_len + max_new_tokens;
    const size_t max_seq_len = total_len > 0 ? total_len : 1;
    size_t max_num_tokens = max_num_tokens_override == 0 ? max_seq_len : max_num_tokens_override;
    if (max_num_tokens == 0) {
        max_num_tokens = 1;
    }
    if (max_num_tokens > max_seq_len) {
        max_num_tokens = max_seq_len;
    }

    marmot_dtype_t kv_dtype = MARMOT_DTYPE_FLOAT32;
    marmot_model_info_t info;
    if (marmot_model_get_info(model, &info) == MARMOT_SUCCESS) {
        const marmot_architecture_t arch = marmot_architecture_from_string(info.architecture);
        kv_dtype = marmot_activation_dtype_for_architecture(arch, backend);
    } else if (backend == MARMOT_BACKEND_METAL) {
        kv_dtype = MARMOT_DTYPE_FLOAT16;
    }

    marmot_serving_engine_options_t engine_opts;
    init_engine_options(&engine_opts, max_seq_len, max_num_tokens, block_size, kv_dtype);

    marmot_serving_engine_t *engine = nullptr;
    marmot_error_t status = marmot_serving_engine_create(ctx, model, &engine_opts, &engine);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    token_capture_t capture = {
        .tokens = out_tokens,
        .capacity = out_capacity,
        .count = 0,
    };

    marmot_llm_generate_options_t gen_opts;
    status = marmot_llm_generate_options_init(&gen_opts);
    if (status != MARMOT_SUCCESS) {
        marmot_serving_engine_destroy(engine);
        return status;
    }
    gen_opts.max_new_tokens = max_new_tokens;
    gen_opts.stop_on_eos = false;
    gen_opts.on_token = capture_token;
    gen_opts.user_data = &capture;

    marmot_llm_sampling_options_t sampling_opts;
    status = marmot_llm_sampling_options_init(&sampling_opts);
    if (status != MARMOT_SUCCESS) {
        marmot_serving_engine_destroy(engine);
        return status;
    }
    sampling_opts.temperature = 0.0f;

    marmot_request_id_t request_id = 0;
    status = marmot_serving_engine_submit(engine, prompt_tokens, prompt_len, &gen_opts, &sampling_opts, &request_id);
    if (status != MARMOT_SUCCESS) {
        marmot_serving_engine_destroy(engine);
        return status;
    }

    const size_t max_steps = max_seq_len + 16;
    marmot_llm_request_state_t state = MARMOT_LLM_REQUEST_STATE_INVALID;
    for (size_t step = 0; step < max_steps; ++step) {
        size_t steps_done = 0;
        status = marmot_serving_engine_step(engine, 1, &steps_done);
        if (status != MARMOT_SUCCESS) {
            break;
        }
        state = marmot_serving_engine_request_state(engine, request_id);
        if (state == MARMOT_LLM_REQUEST_STATE_INVALID || state == MARMOT_LLM_REQUEST_STATE_DONE) {
            break;
        }
        if (state == MARMOT_LLM_REQUEST_STATE_FAILED || state == MARMOT_LLM_REQUEST_STATE_CANCELED) {
            char msg[96];
            snprintf(msg, sizeof(msg), "request state %s", request_state_name(state));
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, msg);
            status = MARMOT_ERROR_INVALID_OPERATION;
            break;
        }
        if (capture.count >= max_new_tokens) {
            break;
        }
    }

    *out_len = capture.count;
    marmot_serving_engine_destroy(engine);
    return status;
}

static marmot_error_t run_serving_generate(
    marmot_context_t *ctx, marmot_model_t *model, marmot_backend_type_t backend, const marmot_token_id_t *prompt_tokens,
    size_t prompt_len, size_t block_size, size_t max_new_tokens, marmot_token_id_t *out_tokens, size_t out_capacity,
    size_t *out_len
) {
    return run_serving_generate_with_limits(
        ctx, model, backend, prompt_tokens, prompt_len, block_size, max_new_tokens, 0, out_tokens, out_capacity, out_len
    );
}

static marmot_error_t run_serving_generate_batched(
    marmot_context_t *ctx, marmot_model_t *model, marmot_backend_type_t backend, const marmot_token_id_t *prompt_a,
    size_t prompt_a_len, const marmot_token_id_t *prompt_b, size_t prompt_b_len, size_t max_new_tokens,
    size_t block_size, marmot_token_id_t *out_a, size_t out_a_capacity, size_t *out_a_len, marmot_token_id_t *out_b,
    size_t out_b_capacity, size_t *out_b_len
) {
    if (out_a_len == nullptr || out_b_len == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_a_len = 0;
    *out_b_len = 0;

    const size_t total_prompt = prompt_a_len + prompt_b_len;
    size_t max_seq_len = total_prompt + max_new_tokens;
    if (max_seq_len == 0) {
        max_seq_len = 1;
    }
    size_t max_num_tokens = total_prompt;
    if (max_num_tokens < 2) {
        max_num_tokens = 2;
    }
    if (max_num_tokens > max_seq_len) {
        max_num_tokens = max_seq_len;
    }

    marmot_dtype_t kv_dtype = MARMOT_DTYPE_FLOAT32;
    marmot_model_info_t info;
    if (marmot_model_get_info(model, &info) == MARMOT_SUCCESS) {
        const marmot_architecture_t arch = marmot_architecture_from_string(info.architecture);
        kv_dtype = marmot_activation_dtype_for_architecture(arch, backend);
    } else if (backend == MARMOT_BACKEND_METAL) {
        kv_dtype = MARMOT_DTYPE_FLOAT16;
    }

    marmot_serving_engine_options_t engine_opts;
    init_engine_options_multi(&engine_opts, 2, 2, max_seq_len, max_num_tokens, block_size, kv_dtype);

    marmot_serving_engine_t *engine = nullptr;
    marmot_error_t status = marmot_serving_engine_create(ctx, model, &engine_opts, &engine);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    token_capture_t capture_a = {
        .tokens = out_a,
        .capacity = out_a_capacity,
        .count = 0,
    };
    token_capture_t capture_b = {
        .tokens = out_b,
        .capacity = out_b_capacity,
        .count = 0,
    };

    marmot_llm_generate_options_t gen_opts_a;
    status = marmot_llm_generate_options_init(&gen_opts_a);
    if (status != MARMOT_SUCCESS) {
        marmot_serving_engine_destroy(engine);
        return status;
    }
    gen_opts_a.max_new_tokens = max_new_tokens;
    gen_opts_a.stop_on_eos = false;
    gen_opts_a.on_token = capture_token;
    gen_opts_a.user_data = &capture_a;

    marmot_llm_generate_options_t gen_opts_b;
    status = marmot_llm_generate_options_init(&gen_opts_b);
    if (status != MARMOT_SUCCESS) {
        marmot_serving_engine_destroy(engine);
        return status;
    }
    gen_opts_b.max_new_tokens = max_new_tokens;
    gen_opts_b.stop_on_eos = false;
    gen_opts_b.on_token = capture_token;
    gen_opts_b.user_data = &capture_b;

    marmot_llm_sampling_options_t sampling_opts;
    status = marmot_llm_sampling_options_init(&sampling_opts);
    if (status != MARMOT_SUCCESS) {
        marmot_serving_engine_destroy(engine);
        return status;
    }
    sampling_opts.temperature = 0.0f;

    marmot_request_id_t req_a = 0;
    status = marmot_serving_engine_submit(engine, prompt_a, prompt_a_len, &gen_opts_a, &sampling_opts, &req_a);
    if (status != MARMOT_SUCCESS) {
        marmot_serving_engine_destroy(engine);
        return status;
    }

    marmot_request_id_t req_b = 0;
    status = marmot_serving_engine_submit(engine, prompt_b, prompt_b_len, &gen_opts_b, &sampling_opts, &req_b);
    if (status != MARMOT_SUCCESS) {
        marmot_serving_engine_destroy(engine);
        return status;
    }

    const size_t max_steps = total_prompt + max_new_tokens * 2 + 16;
    marmot_llm_request_state_t state_a = MARMOT_LLM_REQUEST_STATE_INVALID;
    marmot_llm_request_state_t state_b = MARMOT_LLM_REQUEST_STATE_INVALID;
    for (size_t step = 0; step < max_steps; ++step) {
        size_t steps_done = 0;
        status = marmot_serving_engine_step(engine, 1, &steps_done);
        if (status != MARMOT_SUCCESS) {
            break;
        }

        state_a = marmot_serving_engine_request_state(engine, req_a);
        state_b = marmot_serving_engine_request_state(engine, req_b);
        if (state_a == MARMOT_LLM_REQUEST_STATE_FAILED || state_a == MARMOT_LLM_REQUEST_STATE_CANCELED) {
            char msg[128];
            snprintf(msg, sizeof(msg), "request A state %s", request_state_name(state_a));
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, msg);
            status = MARMOT_ERROR_INVALID_OPERATION;
            break;
        }
        if (state_b == MARMOT_LLM_REQUEST_STATE_FAILED || state_b == MARMOT_LLM_REQUEST_STATE_CANCELED) {
            char msg[128];
            snprintf(msg, sizeof(msg), "request B state %s", request_state_name(state_b));
            marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, msg);
            status = MARMOT_ERROR_INVALID_OPERATION;
            break;
        }

        if (state_a == MARMOT_LLM_REQUEST_STATE_DONE && state_b == MARMOT_LLM_REQUEST_STATE_DONE) {
            break;
        }
    }

    *out_a_len = capture_a.count;
    *out_b_len = capture_b.count;
    marmot_serving_engine_destroy(engine);
    return status;
}

// ============================================================================
// Smoke test: Generate 1 token from all fixtures (CPU)
// ============================================================================
static void test_llm_smoke_all_fixtures_cpu(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = get_fixture_path(fixture->filename);
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        assert_non_null(ctx);

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        marmot_error_t load_err = marmot_model_load_file(path, &model_opts, &model);
        if (load_err != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);
        assert_true(info.n_vocab > 0);

        marmot_token_id_t prompt_token = info.n_vocab > 1 ? (marmot_token_id_t)1 : (marmot_token_id_t)0;
        marmot_token_id_t out_tokens[1] = {MARMOT_TOKEN_ID_INVALID};
        size_t out_len = 0;

        marmot_error_t gen_err =
            run_serving_generate(ctx, model, MARMOT_BACKEND_CPU, &prompt_token, 1, 16, 1, out_tokens, 1, &out_len);
        if (gen_err != MARMOT_SUCCESS) {
            fprintf(stderr, "Warning: %s failed to generate\n", fixture->filename);
        } else {
            assert_int_equal(out_len, 1);
            assert_true(out_tokens[0] >= 0);
            assert_true((size_t)out_tokens[0] < info.n_vocab);
        }

        marmot_model_destroy(model);
        marmot_destroy(ctx);
    }
}

// ============================================================================
// Golden output test: Compare generated tokens against golden values
// ============================================================================
static void test_llm_golden_outputs_cpu(void **state) {
    (void)state;

    for (size_t i = 0; i < MARMOT_GOLDEN_LLM_OUTPUT_COUNT; ++i) {
        const marmot_golden_llm_output_t *golden = &MARMOT_GOLDEN_LLM_OUTPUTS[i];
        const char *path = get_fixture_path(golden->model_filename);

        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (ctx == nullptr) {
            continue;
        }

        marmot_tokenizer_options_t tok_opts;
        if (marmot_tokenizer_options_init(&tok_opts) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_tokenizer_t *tok = nullptr;
        if (marmot_tokenizer_create_from_gguf_file(path, &tok_opts, &tok) != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_tokenizer_encode_options_t enc_opts;
        if (marmot_tokenizer_encode_options_init(&enc_opts) != MARMOT_SUCCESS) {
            marmot_tokenizer_destroy(tok);
            marmot_destroy(ctx);
            continue;
        }
        enc_opts.add_bos = true;
        enc_opts.add_eos = false;

        size_t prompt_len = 0;
        if (marmot_tokenizer_encode(tok, golden->prompt, strlen(golden->prompt), &enc_opts, nullptr, &prompt_len) !=
            MARMOT_SUCCESS) {
            marmot_tokenizer_destroy(tok);
            marmot_destroy(ctx);
            continue;
        }

        marmot_token_id_t *prompt_tokens = malloc(prompt_len * sizeof(*prompt_tokens));
        if (prompt_tokens == nullptr) {
            marmot_tokenizer_destroy(tok);
            marmot_destroy(ctx);
            continue;
        }

        size_t cap = prompt_len;
        if (marmot_tokenizer_encode(tok, golden->prompt, strlen(golden->prompt), &enc_opts, prompt_tokens, &cap) !=
            MARMOT_SUCCESS) {
            free(prompt_tokens);
            marmot_tokenizer_destroy(tok);
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_options_t model_opts;
        if (marmot_model_options_init(&model_opts) != MARMOT_SUCCESS) {
            free(prompt_tokens);
            marmot_tokenizer_destroy(tok);
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_t *model = nullptr;
        if (marmot_model_load_file(path, &model_opts, &model) != MARMOT_SUCCESS) {
            free(prompt_tokens);
            marmot_tokenizer_destroy(tok);
            marmot_destroy(ctx);
            continue;
        }

        marmot_token_id_t *out_tokens = malloc(golden->expected_len * sizeof(*out_tokens));
        assert_non_null(out_tokens);
        size_t out_len = 0;

        marmot_error_t gen_err = run_serving_generate(
            ctx, model, MARMOT_BACKEND_CPU, prompt_tokens, prompt_len, 16, golden->expected_len, out_tokens,
            golden->expected_len, &out_len
        );

        if (gen_err == MARMOT_SUCCESS) {
            assert_int_equal(out_len, golden->expected_len);
            for (size_t j = 0; j < golden->expected_len; ++j) {
                if (out_tokens[j] != golden->expected_tokens[j]) {
                    fprintf(
                        stderr, "Golden mismatch for %s at token %zu: expected %d, got %d\n", golden->model_filename, j,
                        golden->expected_tokens[j], out_tokens[j]
                    );
                }
                assert_int_equal(out_tokens[j], golden->expected_tokens[j]);
            }
        } else {
            fprintf(stderr, "Warning: %s failed to generate for golden test\n", golden->model_filename);
        }

        free(out_tokens);
        free(prompt_tokens);
        marmot_model_destroy(model);
        marmot_tokenizer_destroy(tok);
        marmot_destroy(ctx);
    }
}

// ============================================================================
// Invariance test: single-seq vs batched generation (CPU)
// ============================================================================
static void test_llm_batch_invariance_cpu(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = get_fixture_path(fixture->filename);
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

        marmot_tokenizer_options_t tok_opts;
        assert_int_equal(marmot_tokenizer_options_init(&tok_opts), MARMOT_SUCCESS);

        marmot_tokenizer_t *tok = nullptr;
        if (marmot_tokenizer_create_from_gguf_file(path, &tok_opts, &tok) != MARMOT_SUCCESS) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_token_id_t *prompt_a = nullptr;
        marmot_token_id_t *prompt_b = nullptr;
        size_t prompt_a_len = 0;
        size_t prompt_b_len = 0;
        if (encode_prompt(tok, "Hello", &prompt_a, &prompt_a_len) != MARMOT_SUCCESS ||
            encode_prompt(tok, "The capital of France is", &prompt_b, &prompt_b_len) != MARMOT_SUCCESS) {
            free(prompt_a);
            free(prompt_b);
            marmot_tokenizer_destroy(tok);
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        const size_t max_new_tokens = 8;
        marmot_token_id_t single_tokens[8] = {MARMOT_TOKEN_ID_INVALID};
        size_t single_len = 0;
        marmot_error_t single_status = run_serving_generate(
            ctx, model, MARMOT_BACKEND_CPU, prompt_a, prompt_a_len, 16, max_new_tokens, single_tokens, max_new_tokens,
            &single_len
        );
        if (single_status != MARMOT_SUCCESS) {
            fprintf(
                stderr, "Batch invariance single-seq failed: %s (%s)\n", marmot_error_string(single_status),
                marmot_get_last_error_detail()
            );
        }
        assert_int_equal(single_status, MARMOT_SUCCESS);
        assert_int_equal(single_len, max_new_tokens);

        marmot_token_id_t batch_a[8] = {MARMOT_TOKEN_ID_INVALID};
        marmot_token_id_t batch_b[8] = {MARMOT_TOKEN_ID_INVALID};
        size_t batch_a_len = 0;
        size_t batch_b_len = 0;
        marmot_error_t batch_status = run_serving_generate_batched(
            ctx, model, MARMOT_BACKEND_CPU, prompt_a, prompt_a_len, prompt_b, prompt_b_len, max_new_tokens, 16, batch_a,
            max_new_tokens, &batch_a_len, batch_b, max_new_tokens, &batch_b_len
        );
        if (batch_status != MARMOT_SUCCESS) {
            fprintf(
                stderr, "Batch invariance batched failed: %s (%s)\n", marmot_error_string(batch_status),
                marmot_get_last_error_detail()
            );
        }
        assert_int_equal(batch_status, MARMOT_SUCCESS);
        assert_int_equal(batch_a_len, max_new_tokens);
        assert_int_equal(batch_b_len, max_new_tokens);

        for (size_t i = 0; i < max_new_tokens; ++i) {
            assert_int_equal(single_tokens[i], batch_a[i]);
        }

        free(prompt_a);
        free(prompt_b);
        marmot_tokenizer_destroy(tok);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

// ============================================================================
// Invariance test: token capacity change should not affect outputs (CPU)
// ============================================================================
static void test_llm_token_capacity_invariance_cpu(void **state) {
    (void)state;

    const char *debug_layer_env = getenv("MARMOT_DEBUG_LAYER_INVARIANCE");
    bool debug_layer = debug_layer_env != nullptr && debug_layer_env[0] != '\0';
    char *prev_layer = nullptr;
    if (debug_layer) {
        const char *existing = getenv("MARMOT_DEBUG_LAYER");
        if (existing != nullptr && existing[0] != '\0') {
            const size_t len = strlen(existing);
            prev_layer = malloc(len + 1);
            if (prev_layer != nullptr) {
                memcpy(prev_layer, existing, len);
                prev_layer[len] = '\0';
            } else {
                debug_layer = false;
            }
        }
    }

    const char *debug_qkv_env = getenv("MARMOT_DEBUG_QKV_INDEX_INVARIANCE");
    bool debug_qkv = debug_qkv_env != nullptr && debug_qkv_env[0] != '\0';
    char *prev_qkv = nullptr;
    if (debug_qkv) {
        const char *existing = getenv("MARMOT_DEBUG_QKV_INDEX");
        if (existing != nullptr && existing[0] != '\0') {
            const size_t len = strlen(existing);
            prev_qkv = malloc(len + 1);
            if (prev_qkv != nullptr) {
                memcpy(prev_qkv, existing, len);
                prev_qkv[len] = '\0';
            } else {
                debug_qkv = false;
            }
        }
    }

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = get_fixture_path(fixture->filename);
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

        marmot_tokenizer_options_t tok_opts;
        assert_int_equal(marmot_tokenizer_options_init(&tok_opts), MARMOT_SUCCESS);

        marmot_tokenizer_t *tok = nullptr;
        if (marmot_tokenizer_create_from_gguf_file(path, &tok_opts, &tok) != MARMOT_SUCCESS) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_token_id_t *prompt_tokens = nullptr;
        size_t prompt_len = 0;
        if (encode_prompt(tok, "Hello", &prompt_tokens, &prompt_len) != MARMOT_SUCCESS) {
            marmot_tokenizer_destroy(tok);
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        const size_t max_new_tokens = 8;
        const size_t max_seq_len = prompt_len + max_new_tokens;
        size_t small_cap = prompt_len;
        if (small_cap == 0) {
            small_cap = 1;
        }

        marmot_token_id_t tokens_small[8] = {MARMOT_TOKEN_ID_INVALID};
        marmot_token_id_t tokens_large[8] = {MARMOT_TOKEN_ID_INVALID};
        size_t len_small = 0;
        size_t len_large = 0;

        if (debug_layer) {
            fprintf(stderr, "marmot debug token_capacity=small max_num_tokens=%zu\n", small_cap);
            setenv("MARMOT_DEBUG_LAYER", debug_layer_env, 1);
        }
        if (debug_qkv) {
            setenv("MARMOT_DEBUG_QKV_INDEX", debug_qkv_env, 1);
        }
        marmot_error_t small_status = run_serving_generate_with_limits(
            ctx, model, MARMOT_BACKEND_CPU, prompt_tokens, prompt_len, 16, max_new_tokens, small_cap, tokens_small,
            max_new_tokens, &len_small
        );
        if (debug_layer) {
            if (prev_layer != nullptr) {
                setenv("MARMOT_DEBUG_LAYER", prev_layer, 1);
            } else {
                unsetenv("MARMOT_DEBUG_LAYER");
            }
        }
        if (debug_qkv) {
            if (prev_qkv != nullptr) {
                setenv("MARMOT_DEBUG_QKV_INDEX", prev_qkv, 1);
            } else {
                unsetenv("MARMOT_DEBUG_QKV_INDEX");
            }
        }
        if (small_status != MARMOT_SUCCESS) {
            fprintf(
                stderr, "Token capacity small failed: %s (%s)\n", marmot_error_string(small_status),
                marmot_get_last_error_detail()
            );
        }
        assert_int_equal(small_status, MARMOT_SUCCESS);
        assert_int_equal(len_small, max_new_tokens);

        if (debug_layer) {
            fprintf(stderr, "marmot debug token_capacity=large max_num_tokens=%zu\n", max_seq_len);
            setenv("MARMOT_DEBUG_LAYER", debug_layer_env, 1);
        }
        if (debug_qkv) {
            setenv("MARMOT_DEBUG_QKV_INDEX", debug_qkv_env, 1);
        }
        marmot_error_t large_status = run_serving_generate_with_limits(
            ctx, model, MARMOT_BACKEND_CPU, prompt_tokens, prompt_len, 16, max_new_tokens, max_seq_len, tokens_large,
            max_new_tokens, &len_large
        );
        if (debug_layer) {
            if (prev_layer != nullptr) {
                setenv("MARMOT_DEBUG_LAYER", prev_layer, 1);
            } else {
                unsetenv("MARMOT_DEBUG_LAYER");
            }
        }
        if (debug_qkv) {
            if (prev_qkv != nullptr) {
                setenv("MARMOT_DEBUG_QKV_INDEX", prev_qkv, 1);
            } else {
                unsetenv("MARMOT_DEBUG_QKV_INDEX");
            }
        }
        if (large_status != MARMOT_SUCCESS) {
            fprintf(
                stderr, "Token capacity large failed: %s (%s)\n", marmot_error_string(large_status),
                marmot_get_last_error_detail()
            );
        }
        assert_int_equal(large_status, MARMOT_SUCCESS);
        assert_int_equal(len_large, max_new_tokens);

        if (prev_layer != nullptr) {
            free(prev_layer);
            prev_layer = nullptr;
        }
        if (prev_qkv != nullptr) {
            free(prev_qkv);
            prev_qkv = nullptr;
        }

        if (debug_layer) {
            fprintf(stderr, "marmot debug tokens small:");
            for (size_t i = 0; i < max_new_tokens; ++i) {
                fprintf(stderr, " %u", (unsigned)tokens_small[i]);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "marmot debug tokens large:");
            for (size_t i = 0; i < max_new_tokens; ++i) {
                fprintf(stderr, " %u", (unsigned)tokens_large[i]);
            }
            fprintf(stderr, "\n");
            for (size_t i = 0; i < max_new_tokens; ++i) {
                if (tokens_small[i] != tokens_large[i]) {
                    fprintf(
                        stderr, "marmot debug token mismatch i=%zu small=%u large=%u\n", i, (unsigned)tokens_small[i],
                        (unsigned)tokens_large[i]
                    );
                    break;
                }
            }
        }

        for (size_t i = 0; i < max_new_tokens; ++i) {
            assert_int_equal(tokens_small[i], tokens_large[i]);
        }

        free(prompt_tokens);
        marmot_tokenizer_destroy(tok);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    if (prev_layer != nullptr) {
        free(prev_layer);
    }
    if (prev_qkv != nullptr) {
        free(prev_qkv);
    }
    skip();
}

// ============================================================================
// Invariance test: block size change should not affect outputs (CPU)
// ============================================================================
static void test_llm_block_size_invariance_cpu(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture = nullptr;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = get_fixture_path(fixture->filename);
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
        if (marmot_model_get_info(model, &info) != MARMOT_SUCCESS) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }
        if (info.n_vocab < 4) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        const size_t block_a = 8;
        const size_t block_b = 16;
        size_t prompt_len = block_b + 1;
        const size_t max_new_tokens = 6;
        if (info.context_length > 0 && prompt_len + max_new_tokens > info.context_length) {
            if (info.context_length <= max_new_tokens + 1) {
                marmot_model_destroy(model);
                marmot_destroy(ctx);
                continue;
            }
            prompt_len = info.context_length - max_new_tokens;
        }
        if (prompt_len <= block_a) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }

        marmot_token_id_t *prompt_tokens = malloc(prompt_len * sizeof(*prompt_tokens));
        if (prompt_tokens == nullptr) {
            marmot_model_destroy(model);
            marmot_destroy(ctx);
            continue;
        }
        const size_t range = info.n_vocab > 1 ? info.n_vocab - 1 : 1;
        for (size_t i = 0; i < prompt_len; ++i) {
            prompt_tokens[i] = (marmot_token_id_t)(1 + (i % range));
        }

        marmot_token_id_t out_a[6] = {MARMOT_TOKEN_ID_INVALID};
        marmot_token_id_t out_b[6] = {MARMOT_TOKEN_ID_INVALID};
        size_t out_a_len = 0;
        size_t out_b_len = 0;

        marmot_error_t status_a = run_serving_generate(
            ctx, model, MARMOT_BACKEND_CPU, prompt_tokens, prompt_len, block_a, max_new_tokens, out_a, max_new_tokens,
            &out_a_len
        );
        assert_int_equal(status_a, MARMOT_SUCCESS);
        assert_int_equal(out_a_len, max_new_tokens);

        marmot_error_t status_b = run_serving_generate(
            ctx, model, MARMOT_BACKEND_CPU, prompt_tokens, prompt_len, block_b, max_new_tokens, out_b, max_new_tokens,
            &out_b_len
        );
        assert_int_equal(status_b, MARMOT_SUCCESS);
        assert_int_equal(out_b_len, max_new_tokens);

        for (size_t i = 0; i < max_new_tokens; ++i) {
            assert_int_equal(out_a[i], out_b[i]);
        }

        free(prompt_tokens);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        return;
    }

    skip();
}

// ============================================================================
// Metal backend tests (macOS only)
// ============================================================================
#ifdef __APPLE__
static void test_llm_smoke_all_fixtures_metal(void **state) {
    (void)state;

    const marmot_test_fixture_t *fixture;
    MARMOT_TEST_FOR_EACH_FIXTURE(fixture) {
        const char *path = get_fixture_path(fixture->filename);
        if (!marmot_test_fixture_exists(path)) {
            continue;
        }

        marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_METAL);
        if (ctx == nullptr) {
            continue;
        }

        marmot_model_options_t model_opts;
        assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

        marmot_model_t *model = nullptr;
        marmot_error_t load_err = marmot_model_load_file(path, &model_opts, &model);
        if (load_err != MARMOT_SUCCESS) {
            marmot_destroy(ctx);
            continue;
        }

        marmot_model_info_t info;
        assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);

        marmot_token_id_t prompt_token = info.n_vocab > 1 ? (marmot_token_id_t)1 : (marmot_token_id_t)0;
        marmot_token_id_t out_tokens[4] = {
            MARMOT_TOKEN_ID_INVALID, MARMOT_TOKEN_ID_INVALID, MARMOT_TOKEN_ID_INVALID, MARMOT_TOKEN_ID_INVALID
        };
        size_t out_len = 0;

        marmot_error_t gen_err =
            run_serving_generate(ctx, model, MARMOT_BACKEND_METAL, &prompt_token, 1, 16, 4, out_tokens, 4, &out_len);
        if (gen_err == MARMOT_SUCCESS) {
            fprintf(stderr, "[metal_decode] %s: tokens=[", fixture->filename);
            for (size_t i = 0; i < out_len; ++i) {
                fprintf(stderr, "%s%d", i ? ", " : "", out_tokens[i]);
            }
            fprintf(stderr, "] (len=%zu)\n", out_len);
            assert_true(out_len >= 1);
            assert_true(out_tokens[0] >= 0);
            assert_true((size_t)out_tokens[0] < info.n_vocab);
        }

        // CPU comparison
        marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
        if (cpu_ctx != nullptr) {
            marmot_token_id_t cpu_tokens[4] = {
                MARMOT_TOKEN_ID_INVALID, MARMOT_TOKEN_ID_INVALID, MARMOT_TOKEN_ID_INVALID, MARMOT_TOKEN_ID_INVALID
            };
            size_t cpu_len = 0;
            marmot_error_t cpu_err = run_serving_generate(
                cpu_ctx, model, MARMOT_BACKEND_CPU, &prompt_token, 1, 16, 4, cpu_tokens, 4, &cpu_len
            );
            if (cpu_err == MARMOT_SUCCESS) {
                fprintf(stderr, "[cpu_decode]   %s: tokens=[", fixture->filename);
                for (size_t i = 0; i < cpu_len; ++i) {
                    fprintf(stderr, "%s%d", i ? ", " : "", cpu_tokens[i]);
                }
                fprintf(stderr, "] (len=%zu)\n", cpu_len);
            }
            marmot_destroy(cpu_ctx);
        }

        marmot_model_destroy(model);
        marmot_destroy(ctx);
    }
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_llm_smoke_all_fixtures_cpu),    cmocka_unit_test(test_llm_golden_outputs_cpu),
        cmocka_unit_test(test_llm_batch_invariance_cpu),      cmocka_unit_test(test_llm_token_capacity_invariance_cpu),
        cmocka_unit_test(test_llm_block_size_invariance_cpu),
#ifdef __APPLE__
        cmocka_unit_test(test_llm_smoke_all_fixtures_metal),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
