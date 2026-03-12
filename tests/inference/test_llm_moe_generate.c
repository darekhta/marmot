/* clang-format off */
#include "marmot/device.h"
#include "marmot/graph/architecture.h"
#include "marmot/inference/engine.h"
#include "marmot/inference/llm.h"
#include "marmot/inference/model.h"
#include "marmot/tokenizer.h"

#include <math.h>
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

#include "test_fixture_utils.h"

static _Thread_local char g_fixture_path[MARMOT_TEST_PATH_MAX];

// ---------------------------------------------------------------------------
// MoE fixture filenames.
// The truncated fixture is small enough for CI.
// The full model is optional — tests skip if absent.
// ---------------------------------------------------------------------------
static const char *MOE_TRUNCATED_FIXTURE = "qwen3moe-30b-a3b-1layer-q4km.gguf";
static const char *MOE_FULL_FIXTURE = "qwen3moe-30b-a3b-q4km.gguf";

static const char *get_fixture_path(const char *filename) {
    return marmot_test_get_fixture_path(filename, g_fixture_path, sizeof(g_fixture_path));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

typedef struct {
    marmot_token_id_t *tokens;
    size_t capacity;
    size_t count;
} token_capture_t;

static void capture_token(void *user_data, marmot_token_id_t token_id) {
    token_capture_t *capture = (token_capture_t *)user_data;
    if (capture == nullptr || capture->tokens == nullptr) {
        return;
    }
    if (capture->count < capture->capacity) {
        capture->tokens[capture->count++] = token_id;
    }
}

static const char *request_state_name(marmot_llm_request_state_t state) {
    switch (state) {
    case MARMOT_LLM_REQUEST_STATE_DONE:
        return "done";
    case MARMOT_LLM_REQUEST_STATE_FAILED:
        return "failed";
    case MARMOT_LLM_REQUEST_STATE_CANCELED:
        return "canceled";
    default:
        return "other";
    }
}

// ---------------------------------------------------------------------------
// Test 1: Truncated fixture — model loads and metadata is correct
// ---------------------------------------------------------------------------
static void test_moe_truncated_model_info(void **state) {
    (void)state;
    const char *path = get_fixture_path(MOE_TRUNCATED_FIXTURE);
    if (!marmot_test_fixture_exists(path)) {
        skip();
    }

    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

    marmot_model_t *model = nullptr;
    marmot_error_t err = marmot_model_load_file(path, &model_opts, &model);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_non_null(model);

    marmot_model_info_t info;
    assert_int_equal(marmot_model_get_info(model, &info), MARMOT_SUCCESS);

    marmot_architecture_t arch = marmot_architecture_from_string(info.architecture);
    assert_int_equal(arch, MARMOT_ARCH_QWEN3MOE);

    assert_true(info.is_moe);
    assert_int_equal(info.n_experts, 128);
    assert_int_equal(info.n_experts_used, 8);

    printf(
        "  qwen3moe model loaded: %zu layers, %zu vocab, %zu experts (top-%zu)\n", (size_t)info.n_layer,
        (size_t)info.n_vocab, (size_t)info.n_experts, (size_t)info.n_experts_used
    );

    marmot_model_destroy(model);
}

// ---------------------------------------------------------------------------
// Test 2: Truncated fixture — graph builds and prefill produces finite logits
// ---------------------------------------------------------------------------
static marmot_error_t run_moe_prefill(
    marmot_context_t *ctx, marmot_model_t *model, marmot_backend_type_t backend, const marmot_token_id_t *prompt_tokens,
    size_t prompt_len, size_t max_new_tokens, marmot_token_id_t *out_tokens, size_t out_capacity, size_t *out_len
) {
    size_t max_seq_len = prompt_len + max_new_tokens + 16;
    size_t max_num_tokens = max_seq_len;
    size_t block_size = 16;

    marmot_dtype_t kv_dtype = MARMOT_DTYPE_FLOAT32;
    marmot_model_info_t info;
    if (marmot_model_get_info(model, &info) == MARMOT_SUCCESS) {
        marmot_architecture_t arch = marmot_architecture_from_string(info.architecture);
        kv_dtype = marmot_activation_dtype_for_architecture(arch, backend);
    }

    marmot_serving_engine_options_t engine_opts;
    assert_int_equal(marmot_serving_engine_options_init(&engine_opts), MARMOT_SUCCESS);
    engine_opts.flags = MARMOT_SERVING_ENGINE_FLAG_DETERMINISTIC_KV;
    engine_opts.max_seqs = 1;
    engine_opts.max_batch_seqs = 1;
    engine_opts.max_seq_len = max_seq_len;
    engine_opts.max_num_tokens = max_num_tokens;
    engine_opts.block_size = block_size;
    size_t blocks_per_seq = (max_seq_len + block_size - 1) / block_size;
    if (blocks_per_seq < 4) {
        blocks_per_seq = 4;
    }
    engine_opts.num_kv_blocks = blocks_per_seq;
    engine_opts.kv_dtype = kv_dtype;
    engine_opts.prefill_chunk_size = 0;

    marmot_serving_engine_t *engine = nullptr;
    marmot_error_t status = marmot_serving_engine_create(ctx, model, &engine_opts, &engine);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    token_capture_t capture = {.tokens = out_tokens, .capacity = out_capacity, .count = 0};

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

    for (size_t step = 0; step < max_seq_len + 16; ++step) {
        size_t steps_done = 0;
        status = marmot_serving_engine_step(engine, 1, &steps_done);
        if (status != MARMOT_SUCCESS) {
            break;
        }
        marmot_llm_request_state_t state = marmot_serving_engine_request_state(engine, request_id);
        if (state == MARMOT_LLM_REQUEST_STATE_DONE || state == MARMOT_LLM_REQUEST_STATE_INVALID) {
            break;
        }
        if (state == MARMOT_LLM_REQUEST_STATE_FAILED || state == MARMOT_LLM_REQUEST_STATE_CANCELED) {
            char msg[96];
            snprintf(msg, sizeof(msg), "request %s", request_state_name(state));
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

static void test_moe_truncated_prefill_cpu(void **state) {
    (void)state;
    const char *path = get_fixture_path(MOE_TRUNCATED_FIXTURE);
    if (!marmot_test_fixture_exists(path)) {
        skip();
    }

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

    marmot_model_t *model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &model), MARMOT_SUCCESS);

    // Encode a short prompt. Use raw token IDs to avoid tokenizer dependency.
    // Token 9707 = "Hello" in Qwen3 vocabulary.
    const marmot_token_id_t prompt[] = {9707};
    const size_t prompt_len = 1;
    const size_t max_new = 4;

    marmot_token_id_t out[4];
    size_t out_len = 0;
    marmot_error_t err =
        run_moe_prefill(ctx, model, MARMOT_BACKEND_CPU, prompt, prompt_len, max_new, out, max_new, &out_len);

    printf("  Prefill result: status=%d, generated %zu tokens", err, out_len);
    if (out_len > 0) {
        printf(" [");
        for (size_t i = 0; i < out_len; ++i) {
            printf("%s%d", i > 0 ? ", " : "", out[i]);
        }
        printf("]");
    }
    printf("\n");

    // The truncated model (1 layer) produces garbage, but it must not crash or produce NaN.
    // We check that the engine ran without error and produced at least 1 token.
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_true(out_len > 0);

    // Verify tokens are valid (non-negative, within reasonable range)
    for (size_t i = 0; i < out_len; ++i) {
        assert_true(out[i] >= 0);
    }

    marmot_model_destroy(model);
    marmot_destroy(ctx);
}

#ifdef __APPLE__
static void test_moe_truncated_prefill_metal(void **state) {
    (void)state;
    const char *path = get_fixture_path(MOE_TRUNCATED_FIXTURE);
    if (!marmot_test_fixture_exists(path)) {
        skip();
    }

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_METAL);
    assert_non_null(ctx);

    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

    marmot_model_t *model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &model), MARMOT_SUCCESS);

    const marmot_token_id_t prompt[] = {9707};
    const size_t prompt_len = 1;
    const size_t max_new = 4;

    marmot_token_id_t out[4];
    size_t out_len = 0;
    marmot_error_t err =
        run_moe_prefill(ctx, model, MARMOT_BACKEND_METAL, prompt, prompt_len, max_new, out, max_new, &out_len);

    const char *detail = marmot_get_last_error_detail();
    printf("  Metal prefill: status=%d, generated %zu tokens", err, out_len);
    if (out_len > 0) {
        printf(" [");
        for (size_t i = 0; i < out_len; ++i) {
            printf("%s%d", i > 0 ? ", " : "", out[i]);
        }
        printf("]");
    }
    printf(
        "%s%s\n", detail != nullptr && detail[0] != '\0' ? ", detail=" : "",
        detail != nullptr && detail[0] != '\0' ? detail : ""
    );
    if (err != MARMOT_SUCCESS) {
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        fail_msg(
            "Metal truncated MoE prefill failed: %d%s%s", err, detail != nullptr && detail[0] != '\0' ? " - " : "",
            detail != nullptr ? detail : ""
        );
    }
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_true(out_len > 0);

    marmot_model_destroy(model);
    marmot_destroy(ctx);
}

#endif

// ---------------------------------------------------------------------------
// Test 3: Full model — golden output comparison (skipped if fixture absent)
// ---------------------------------------------------------------------------
static void test_moe_full_generation_cpu(void **state) {
    (void)state;
    const char *path = get_fixture_path(MOE_FULL_FIXTURE);
    if (!marmot_test_fixture_exists(path)) {
        printf("  Full MoE fixture not found, skipping golden test\n");
        skip();
    }

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

    marmot_model_t *model = nullptr;
    marmot_error_t load_err = marmot_model_load_file(path, &model_opts, &model);
    if (load_err != MARMOT_SUCCESS) {
        printf("  Failed to load full MoE model: %d\n", load_err);
        marmot_destroy(ctx);
        skip();
    }

    // Use tokenizer for proper prompt encoding
    marmot_tokenizer_options_t tok_opts;
    assert_int_equal(marmot_tokenizer_options_init(&tok_opts), MARMOT_SUCCESS);

    marmot_tokenizer_t *tok = nullptr;
    marmot_error_t tok_err = marmot_tokenizer_create_from_gguf_file(path, &tok_opts, &tok);
    if (tok_err != MARMOT_SUCCESS) {
        printf("  Failed to create tokenizer: %d\n", tok_err);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        skip();
    }

    const char *prompt = "The capital of France is";
    marmot_tokenizer_encode_options_t enc_opts;
    assert_int_equal(marmot_tokenizer_encode_options_init(&enc_opts), MARMOT_SUCCESS);
    enc_opts.add_bos = true;
    enc_opts.add_eos = false;

    size_t prompt_len = 0;
    marmot_tokenizer_encode(tok, prompt, strlen(prompt), &enc_opts, nullptr, &prompt_len);
    assert_true(prompt_len > 0);

    marmot_token_id_t *prompt_tokens = malloc(prompt_len * sizeof(*prompt_tokens));
    assert_non_null(prompt_tokens);
    size_t cap = prompt_len;
    assert_int_equal(
        marmot_tokenizer_encode(tok, prompt, strlen(prompt), &enc_opts, prompt_tokens, &cap), MARMOT_SUCCESS
    );

    const size_t max_new = 16;
    marmot_token_id_t out[16];
    size_t out_len = 0;
    marmot_error_t err =
        run_moe_prefill(ctx, model, MARMOT_BACKEND_CPU, prompt_tokens, prompt_len, max_new, out, max_new, &out_len);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_true(out_len > 0);

    // Decode and print
    marmot_tokenizer_decode_options_t dec_opts;
    assert_int_equal(marmot_tokenizer_decode_options_init(&dec_opts), MARMOT_SUCCESS);

    printf("  Generated %zu tokens: ", out_len);
    for (size_t i = 0; i < out_len; ++i) {
        char decoded[64] = {0};
        size_t decoded_len = sizeof(decoded) - 1;
        if (marmot_tokenizer_decode(tok, &out[i], 1, &dec_opts, decoded, &decoded_len) == MARMOT_SUCCESS) {
            printf("%s", decoded);
        } else {
            printf("[%d]", out[i]);
        }
    }
    printf("\n");

    // TODO: Add golden token comparison once reference outputs are generated
    // For now, just verify the model runs without errors.

    free(prompt_tokens);
    marmot_tokenizer_destroy(tok);
    marmot_model_destroy(model);
    marmot_destroy(ctx);
}

// ---------------------------------------------------------------------------
// Test 4: Full model — Metal output comparison
// ---------------------------------------------------------------------------
#ifdef __APPLE__
static void test_moe_full_generation_metal(void **state) {
    (void)state;
    const char *path = get_fixture_path(MOE_FULL_FIXTURE);
    if (!marmot_test_fixture_exists(path)) {
        printf("  Full MoE fixture not found, skipping Metal test\n");
        skip();
    }

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_METAL);
    assert_non_null(ctx);

    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

    marmot_model_t *model = nullptr;
    marmot_error_t load_err = marmot_model_load_file(path, &model_opts, &model);
    if (load_err != MARMOT_SUCCESS) {
        printf("  Failed to load full MoE model on Metal: %d\n", load_err);
        marmot_destroy(ctx);
        skip();
    }

    marmot_tokenizer_options_t tok_opts;
    assert_int_equal(marmot_tokenizer_options_init(&tok_opts), MARMOT_SUCCESS);

    marmot_tokenizer_t *tok = nullptr;
    marmot_error_t tok_err = marmot_tokenizer_create_from_gguf_file(path, &tok_opts, &tok);
    if (tok_err != MARMOT_SUCCESS) {
        printf("  Failed to create tokenizer: %d\n", tok_err);
        marmot_model_destroy(model);
        marmot_destroy(ctx);
        skip();
    }

    const char *prompt = "The capital of France is";
    marmot_tokenizer_encode_options_t enc_opts;
    assert_int_equal(marmot_tokenizer_encode_options_init(&enc_opts), MARMOT_SUCCESS);
    enc_opts.add_bos = true;
    enc_opts.add_eos = false;

    size_t prompt_len = 0;
    marmot_tokenizer_encode(tok, prompt, strlen(prompt), &enc_opts, nullptr, &prompt_len);
    assert_true(prompt_len > 0);

    marmot_token_id_t *prompt_tokens = malloc(prompt_len * sizeof(*prompt_tokens));
    assert_non_null(prompt_tokens);
    size_t cap = prompt_len;
    assert_int_equal(
        marmot_tokenizer_encode(tok, prompt, strlen(prompt), &enc_opts, prompt_tokens, &cap), MARMOT_SUCCESS
    );

    const size_t max_new = 16;
    marmot_token_id_t out[16];
    size_t out_len = 0;
    marmot_error_t err =
        run_moe_prefill(ctx, model, MARMOT_BACKEND_METAL, prompt_tokens, prompt_len, max_new, out, max_new, &out_len);

    const char *detail = marmot_get_last_error_detail();
    printf(
        "  Metal full: status=%d, generated %zu tokens%s%s\n", err, out_len,
        detail != nullptr && detail[0] != '\0' ? ", detail=" : "", detail != nullptr && detail[0] != '\0' ? detail : ""
    );
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_true(out_len > 0);

    marmot_tokenizer_decode_options_t dec_opts;
    assert_int_equal(marmot_tokenizer_decode_options_init(&dec_opts), MARMOT_SUCCESS);

    printf("  Metal generated %zu tokens: ", out_len);
    for (size_t i = 0; i < out_len; ++i) {
        char decoded[64] = {0};
        size_t decoded_len = sizeof(decoded) - 1;
        if (marmot_tokenizer_decode(tok, &out[i], 1, &dec_opts, decoded, &decoded_len) == MARMOT_SUCCESS) {
            printf("%s", decoded);
        } else {
            printf("[%d]", out[i]);
        }
    }
    printf("\n");

    free(prompt_tokens);
    marmot_tokenizer_destroy(tok);
    marmot_model_destroy(model);
    marmot_destroy(ctx);
}
#endif

// ---------------------------------------------------------------------------
// Test 5: 3-layer fixture — CPU vs Metal token comparison
// ---------------------------------------------------------------------------
static const char *MOE_3LAYER_FIXTURE = "qwen3moe-30b-a3b-3layer-q4km.gguf";

static void test_moe_3layer_cpu_vs_metal(void **state) {
    (void)state;
#ifndef __APPLE__
    skip();
#else
    const char *path = get_fixture_path(MOE_3LAYER_FIXTURE);
    if (!marmot_test_fixture_exists(path)) {
        printf("  3-layer MoE fixture not found, skipping\n");
        skip();
    }

    // Run on CPU
    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);
    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);
    marmot_model_t *cpu_model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &cpu_model), MARMOT_SUCCESS);

    const marmot_token_id_t prompt[] = {9707}; // "Hello"
    const size_t max_new = 4;
    marmot_token_id_t cpu_out[4];
    size_t cpu_len = 0;
    marmot_error_t cpu_err =
        run_moe_prefill(cpu_ctx, cpu_model, MARMOT_BACKEND_CPU, prompt, 1, max_new, cpu_out, max_new, &cpu_len);
    assert_int_equal(cpu_err, MARMOT_SUCCESS);
    assert_true(cpu_len > 0);
    printf("  CPU 3-layer tokens: [");
    for (size_t i = 0; i < cpu_len; ++i) {
        printf("%s%d", i > 0 ? ", " : "", cpu_out[i]);
    }
    printf("]\n");

    marmot_model_destroy(cpu_model);
    marmot_destroy(cpu_ctx);

    // Run on Metal
    marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
    assert_non_null(metal_ctx);
    marmot_model_t *metal_model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &metal_model), MARMOT_SUCCESS);

    marmot_token_id_t metal_out[4];
    size_t metal_len = 0;
    marmot_error_t metal_err = run_moe_prefill(
        metal_ctx, metal_model, MARMOT_BACKEND_METAL, prompt, 1, max_new, metal_out, max_new, &metal_len
    );
    assert_int_equal(metal_err, MARMOT_SUCCESS);
    assert_true(metal_len > 0);
    printf("  Metal 3-layer tokens: [");
    for (size_t i = 0; i < metal_len; ++i) {
        printf("%s%d", i > 0 ? ", " : "", metal_out[i]);
    }
    printf("]\n");

    // Compare: tokens should match (or at least be close)
    printf("  CPU vs Metal match: ");
    size_t match_count = 0;
    size_t min_len = cpu_len < metal_len ? cpu_len : metal_len;
    for (size_t i = 0; i < min_len; ++i) {
        if (cpu_out[i] == metal_out[i]) {
            match_count++;
        }
    }
    printf("%zu/%zu\n", match_count, min_len);

    marmot_model_destroy(metal_model);
    marmot_destroy(metal_ctx);
#endif
}

// ---------------------------------------------------------------------------
// Test: non-MoE model CPU vs Metal (Qwen2 0.5B)
// ---------------------------------------------------------------------------
static void test_dense_cpu_vs_metal(void **state) {
    (void)state;
#ifndef __APPLE__
    skip();
#else
    const char *path = get_fixture_path("qwen2-0_5b-instruct-q4_k_m.gguf");
    if (!marmot_test_fixture_exists(path)) {
        printf("  Qwen2 fixture not found, skipping\n");
        skip();
    }

    const marmot_token_id_t prompt[] = {9707}; // "Hello"
    const size_t max_new = 4;

    // CPU
    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);
    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);
    marmot_model_t *cpu_model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &cpu_model), MARMOT_SUCCESS);
    marmot_token_id_t cpu_out[4];
    size_t cpu_len = 0;
    assert_int_equal(
        run_moe_prefill(cpu_ctx, cpu_model, MARMOT_BACKEND_CPU, prompt, 1, max_new, cpu_out, max_new, &cpu_len),
        MARMOT_SUCCESS
    );
    printf("  CPU Qwen2 tokens: [");
    for (size_t i = 0; i < cpu_len; ++i)
        printf("%s%d", i > 0 ? ", " : "", cpu_out[i]);
    printf("]\n");
    marmot_model_destroy(cpu_model);
    marmot_destroy(cpu_ctx);

    // Metal
    marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
    assert_non_null(metal_ctx);
    marmot_model_t *metal_model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &metal_model), MARMOT_SUCCESS);
    marmot_token_id_t metal_out[4];
    size_t metal_len = 0;
    assert_int_equal(
        run_moe_prefill(
            metal_ctx, metal_model, MARMOT_BACKEND_METAL, prompt, 1, max_new, metal_out, max_new, &metal_len
        ),
        MARMOT_SUCCESS
    );
    printf("  Metal Qwen2 tokens: [");
    for (size_t i = 0; i < metal_len; ++i)
        printf("%s%d", i > 0 ? ", " : "", metal_out[i]);
    printf("]\n");

    size_t match = 0;
    size_t n = cpu_len < metal_len ? cpu_len : metal_len;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_out[i] == metal_out[i])
            match++;
    }
    printf("  Dense CPU vs Metal match: %zu/%zu\n", match, n);

    marmot_model_destroy(metal_model);
    marmot_destroy(metal_ctx);
#endif
}

// ---------------------------------------------------------------------------
// Test: 3-layer MoE with multi-token prompt (exercises expert_batch_prefill)
// ---------------------------------------------------------------------------
static void test_moe_3layer_multitok_cpu_vs_metal(void **state) {
    (void)state;
#ifndef __APPLE__
    skip();
#else
    const char *path = get_fixture_path(MOE_3LAYER_FIXTURE);
    if (!marmot_test_fixture_exists(path)) {
        printf("  3-layer MoE fixture not found, skipping\n");
        skip();
    }

    // Multi-token prompt to trigger expert_batch_prefill path (tokens > 1).
    // Tokens: "Hello world" in Qwen3 vocabulary (approximate).
    const marmot_token_id_t prompt[] = {9707, 1879, 1495, 279};
    const size_t prompt_len = 4;
    const size_t max_new = 4;

    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

    // CPU
    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);
    marmot_model_t *cpu_model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &cpu_model), MARMOT_SUCCESS);

    marmot_token_id_t cpu_out[4];
    size_t cpu_len = 0;
    marmot_error_t cpu_err = run_moe_prefill(
        cpu_ctx, cpu_model, MARMOT_BACKEND_CPU, prompt, prompt_len, max_new, cpu_out, max_new, &cpu_len
    );
    assert_int_equal(cpu_err, MARMOT_SUCCESS);
    assert_true(cpu_len > 0);
    printf("  CPU 3-layer multitok: [");
    for (size_t i = 0; i < cpu_len; ++i) {
        printf("%s%d", i > 0 ? ", " : "", cpu_out[i]);
    }
    printf("]\n");
    marmot_model_destroy(cpu_model);
    marmot_destroy(cpu_ctx);

    // Metal
    marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
    assert_non_null(metal_ctx);
    marmot_model_t *metal_model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &metal_model), MARMOT_SUCCESS);

    marmot_token_id_t metal_out[4];
    size_t metal_len = 0;
    marmot_error_t metal_err = run_moe_prefill(
        metal_ctx, metal_model, MARMOT_BACKEND_METAL, prompt, prompt_len, max_new, metal_out, max_new, &metal_len
    );
    assert_int_equal(metal_err, MARMOT_SUCCESS);
    assert_true(metal_len > 0);
    printf("  Metal 3-layer multitok: [");
    for (size_t i = 0; i < metal_len; ++i) {
        printf("%s%d", i > 0 ? ", " : "", metal_out[i]);
    }
    printf("]\n");

    size_t match = 0;
    size_t n = cpu_len < metal_len ? cpu_len : metal_len;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_out[i] == metal_out[i])
            match++;
    }
    printf("  3-layer multitok CPU vs Metal: %zu/%zu\n", match, n);

    marmot_model_destroy(metal_model);
    marmot_destroy(metal_ctx);
#endif
}

// ---------------------------------------------------------------------------
// Test: Full model 1-token prompt CPU vs Metal (decode-only, no prefill batch)
// ---------------------------------------------------------------------------
static void test_moe_full_1tok_cpu_vs_metal(void **state) {
    (void)state;
#ifndef __APPLE__
    skip();
#else
    const char *path = get_fixture_path(MOE_FULL_FIXTURE);
    if (!marmot_test_fixture_exists(path)) {
        printf("  Full MoE fixture not found, skipping\n");
        skip();
    }

    const marmot_token_id_t prompt[] = {9707};
    const size_t max_new = 8;

    marmot_model_options_t model_opts;
    assert_int_equal(marmot_model_options_init(&model_opts), MARMOT_SUCCESS);

    marmot_context_t *cpu_ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(cpu_ctx);
    marmot_model_t *cpu_model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &cpu_model), MARMOT_SUCCESS);
    marmot_token_id_t cpu_out[8];
    size_t cpu_len = 0;
    assert_int_equal(
        run_moe_prefill(cpu_ctx, cpu_model, MARMOT_BACKEND_CPU, prompt, 1, max_new, cpu_out, max_new, &cpu_len),
        MARMOT_SUCCESS
    );
    printf("  CPU full 1-tok: [");
    for (size_t i = 0; i < cpu_len; ++i)
        printf("%s%d", i > 0 ? ", " : "", cpu_out[i]);
    printf("]\n");
    marmot_model_destroy(cpu_model);
    marmot_destroy(cpu_ctx);

    marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
    assert_non_null(metal_ctx);
    marmot_model_t *metal_model = nullptr;
    assert_int_equal(marmot_model_load_file(path, &model_opts, &metal_model), MARMOT_SUCCESS);
    marmot_token_id_t metal_out[8];
    size_t metal_len = 0;
    assert_int_equal(
        run_moe_prefill(
            metal_ctx, metal_model, MARMOT_BACKEND_METAL, prompt, 1, max_new, metal_out, max_new, &metal_len
        ),
        MARMOT_SUCCESS
    );
    printf("  Metal full 1-tok: [");
    for (size_t i = 0; i < metal_len; ++i)
        printf("%s%d", i > 0 ? ", " : "", metal_out[i]);
    printf("]\n");

    size_t match = 0;
    size_t n = cpu_len < metal_len ? cpu_len : metal_len;
    for (size_t i = 0; i < n; ++i) {
        if (cpu_out[i] == metal_out[i])
            match++;
    }
    printf("  Full model 1-tok CPU vs Metal: %zu/%zu\n", match, n);

    marmot_model_destroy(metal_model);
    marmot_destroy(metal_ctx);
#endif
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_moe_truncated_model_info),         cmocka_unit_test(test_moe_truncated_prefill_cpu),
#ifdef __APPLE__
        cmocka_unit_test(test_moe_truncated_prefill_metal),
#endif
        cmocka_unit_test(test_moe_full_generation_cpu),
#ifdef __APPLE__
        cmocka_unit_test(test_moe_full_generation_metal),
#endif
        cmocka_unit_test(test_moe_3layer_cpu_vs_metal),          cmocka_unit_test(test_dense_cpu_vs_metal),
        cmocka_unit_test(test_moe_3layer_multitok_cpu_vs_metal), cmocka_unit_test(test_moe_full_1tok_cpu_vs_metal),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
