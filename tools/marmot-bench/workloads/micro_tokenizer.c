#include "../bench_workloads.h"

#include "marmot/tokenizer.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    marmot_tokenizer_t *tok;
    marmot_token_id_t *token_ids;
    size_t token_capacity;
    const char *text;
    size_t text_len;
    marmot_tokenizer_encode_options_t encode_opts;
} tokenizer_state_t;

static const char *default_gguf_path(void) {
    const char *env = getenv("MARMOT_BENCH_GGUF");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
    return "tests/fixtures/gguf/multiarch/tinyllama-q4_k_m.gguf";
}

static marmot_error_t tokenizer_setup(
    marmot_backend_type_t backend,
    marmot_context_t *ctx,
    marmot_graph_t **graph,
    marmot_tensor_t ***inputs,
    size_t *num_inputs,
    marmot_tensor_t ***outputs,
    size_t *num_outputs,
    void *user_data
) {
    (void)backend;
    (void)ctx;
    (void)user_data;

    *inputs = nullptr;
    *outputs = nullptr;
    *num_inputs = 0;
    *num_outputs = 0;

    tokenizer_state_t *state = calloc(1, sizeof(*state));
    if (state == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    state->text =
        "The quick brown fox jumps over the lazy dog.\n"
        "In 2025, Marmot aims to be a fast, clean inference stack.\n";
    state->text_len = strlen(state->text);

    marmot_tokenizer_options_t tok_opts;
    marmot_error_t err = marmot_tokenizer_options_init(&tok_opts);
    if (err != MARMOT_SUCCESS) {
        free(state);
        return err;
    }
    tok_opts.flags |= MARMOT_TOKENIZER_FLAG_ENABLE_CACHE;

    err = marmot_tokenizer_create_from_gguf_file(default_gguf_path(), &tok_opts, &state->tok);
    if (err != MARMOT_SUCCESS) {
        free(state);
        return err;
    }

    err = marmot_tokenizer_encode_options_init(&state->encode_opts);
    if (err != MARMOT_SUCCESS) {
        marmot_tokenizer_destroy(state->tok);
        free(state);
        return err;
    }

    state->token_capacity = 0;
    err = marmot_tokenizer_encode(state->tok, state->text, state->text_len, &state->encode_opts, nullptr, &state->token_capacity);
    if (err != MARMOT_SUCCESS) {
        marmot_tokenizer_destroy(state->tok);
        free(state);
        return err;
    }

    state->token_ids = calloc(state->token_capacity, sizeof(marmot_token_id_t));
    if (state->token_ids == nullptr) {
        marmot_tokenizer_destroy(state->tok);
        free(state);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    *graph = (marmot_graph_t *)state;
    return MARMOT_SUCCESS;
}

static marmot_error_t tokenizer_execute(
    marmot_context_t *ctx,
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs
) {
    (void)ctx;
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;

    tokenizer_state_t *state = (tokenizer_state_t *)graph;
    size_t n = state->token_capacity;
    return marmot_tokenizer_encode(state->tok, state->text, state->text_len, &state->encode_opts, state->token_ids, &n);
}

static void tokenizer_teardown(
    marmot_graph_t *graph,
    marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs,
    void *user_data
) {
    (void)inputs;
    (void)num_inputs;
    (void)outputs;
    (void)num_outputs;
    (void)user_data;

    tokenizer_state_t *state = (tokenizer_state_t *)graph;
    if (state == nullptr) {
        return;
    }

    free(state->token_ids);
    if (state->tok != nullptr) {
        marmot_tokenizer_destroy(state->tok);
    }
    free(state);
}

static marmot_bench_workload_t *create_tokenizer_workload(void) {
    marmot_bench_workload_t *w = calloc(1, sizeof(*w));
    if (w == nullptr) {
        return nullptr;
    }

    w->desc.name = "tokenizer_encode";
    w->desc.category = MARMOT_BENCH_CATEGORY_MICRO;
    w->desc.primary_dtype = MARMOT_DTYPE_INT32;
    w->desc.flops = 0;
    w->desc.bytes_read = 0;
    w->desc.bytes_written = 0;
    memset(&w->desc.signature, 0, sizeof(w->desc.signature));

    w->setup = tokenizer_setup;
    w->execute = tokenizer_execute;
    w->teardown = tokenizer_teardown;
    w->user_data = nullptr;
    return w;
}

void marmot_bench_register_tokenizer_workloads(marmot_bench_suite_t *suite) {
    marmot_bench_workload_t *w = create_tokenizer_workload();
    if (w != nullptr) {
        marmot_bench_suite_add(suite, w);
    }
}

