#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/architecture.h"
#include "marmot/graph/gguf_model.h"
#include "marmot/graph/graph.h"
#include "marmot/inference/kv_pool.h"
#include "marmot/ops/conversion.h"
#include "marmot/ops/neural.h"
#include "marmot/tokenizer.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t kTokenFlagDecode = 1u << 1;

static void die(const char *op, marmot_error_t err) {
    const char *detail = marmot_get_last_error_detail();
    fprintf(stderr, "%s: %s", op, marmot_error_string(err));
    if (detail != nullptr && detail[0] != '\0') {
        fprintf(stderr, " (%s)", detail);
    }
    fprintf(stderr, "\n");
    exit(1);
}

static size_t ceil_div(size_t value, size_t divisor) {
    return (value + divisor - 1) / divisor;
}

static char *join_args(int argc, char **argv, int start) {
    size_t total = 0;
    for (int i = start; i < argc; ++i) {
        total += strlen(argv[i]) + 1;
    }
    if (total == 0) {
        char *out = malloc(1);
        if (out != nullptr) {
            out[0] = '\0';
        }
        return out;
    }

    char *out = malloc(total);
    if (out == nullptr) {
        return nullptr;
    }
    out[0] = '\0';
    for (int i = start; i < argc; ++i) {
        if (i > start) {
            strcat(out, " ");
        }
        strcat(out, argv[i]);
    }
    return out;
}

static marmot_token_id_t argmax_logits(
    const marmot_context_t *ctx, marmot_tensor_t *logits, size_t n_vocab, float *scratch
) {
    const float *logits_f32 = nullptr;
    if (logits->dtype == MARMOT_DTYPE_FLOAT32) {
        logits_f32 = marmot_tensor_data_f32(ctx, logits);
    } else if (logits->dtype == MARMOT_DTYPE_FLOAT16) {
        const marmot_float16_t *f16 = marmot_tensor_data_f16(ctx, logits);
        if (f16 == nullptr) {
            return MARMOT_TOKEN_ID_INVALID;
        }
        if (marmot_convert_f16_to_f32(ctx, scratch, f16, n_vocab) != MARMOT_SUCCESS) {
            return MARMOT_TOKEN_ID_INVALID;
        }
        logits_f32 = scratch;
    } else {
        return MARMOT_TOKEN_ID_INVALID;
    }

    size_t best = 0;
    float best_value = logits_f32[0];
    for (size_t i = 1; i < n_vocab; ++i) {
        if (logits_f32[i] > best_value) {
            best_value = logits_f32[i];
            best = i;
        }
    }
    if (best > (size_t)INT32_MAX) {
        return MARMOT_TOKEN_ID_INVALID;
    }
    return (marmot_token_id_t)best;
}

static marmot_error_t copy_decode_inputs(
    const marmot_context_t *ctx, marmot_tensor_t *token_id_tensor, marmot_tensor_t *positions,
    marmot_tensor_t *token_meta, marmot_tensor_t *sample_indices, marmot_seq_slot_t seq, size_t position,
    marmot_kv_slot_t slot, marmot_token_id_t token_id
) {
    if (position > (size_t)UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "position exceeds packed graph uint32 limit");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_int32_t token_id_host[1] = {MARMOT_I32(token_id)};
    const float positions_host[1] = {(float)position};
    const marmot_uint32_t token_meta_host[4] = {
        MARMOT_U32(seq),
        MARMOT_U32(position),
        MARMOT_U32(slot),
        MARMOT_U32(kTokenFlagDecode),
    };
    const marmot_uint32_t sample_indices_host[1] = {MARMOT_U32(0)};

    marmot_error_t err =
        marmot_tensor_copy_from_host_buffer(ctx, token_id_tensor, token_id_host, sizeof(token_id_host));
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    err = marmot_tensor_copy_from_host_buffer(ctx, positions, positions_host, sizeof(positions_host));
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    err = marmot_tensor_copy_from_host_buffer(ctx, token_meta, token_meta_host, sizeof(token_meta_host));
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return marmot_tensor_copy_from_host_buffer(ctx, sample_indices, sample_indices_host, sizeof(sample_indices_host));
}

static marmot_error_t
refresh_block_table_snapshot(const marmot_context_t *ctx, const marmot_tensor_t *block_table, marmot_tensor_t *snapshot) {
    if (snapshot == nullptr) {
        return MARMOT_SUCCESS;
    }
    if (block_table == nullptr || block_table->data == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "block table snapshot requires host-backed block table");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return marmot_tensor_copy_from_host_buffer(ctx, snapshot, block_table->data, marmot_tensor_size_bytes(block_table));
}

static marmot_error_t run_decode_step(
    const marmot_context_t *ctx, const marmot_tensor_t *token_embedding, marmot_graph_t *graph,
    marmot_tensor_t *token_id_tensor, marmot_tensor_t *hidden, marmot_tensor_t *positions,
    marmot_tensor_t *token_meta, marmot_tensor_t *sample_indices, marmot_tensor_t *logits,
    const marmot_tensor_t *block_table, marmot_tensor_t *block_table_snapshot, const marmot_tensor_t *kv_k,
    const marmot_tensor_t *kv_v, marmot_dtype_t activation_dtype, float embedding_scale, marmot_seq_slot_t seq,
    size_t position, marmot_kv_slot_t slot, marmot_token_id_t token_id
) {
    marmot_error_t err = copy_decode_inputs(
        ctx, token_id_tensor, positions, token_meta, sample_indices, seq, position, slot, token_id
    );
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    err = refresh_block_table_snapshot(ctx, block_table, block_table_snapshot);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    marmot_embedding_gather_desc_t gather = marmot_embedding_gather_desc_default();
    gather.weights = token_embedding;
    gather.token_ids = token_id_tensor;
    gather.out = hidden;
    gather.dtype_out = activation_dtype;
    gather.scale = embedding_scale;
    gather.bounds_check = true;
    gather.allow_quant_decode_on_the_fly = MARMOT_PREFERENCE_ENABLE;
    err = marmot_embedding_gather(ctx, &gather);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    const marmot_tensor_t *graph_block_table = block_table_snapshot != nullptr ? block_table_snapshot : block_table;
    const marmot_tensor_t *inputs[7] = {
        hidden,
        positions,
        token_meta,
        graph_block_table,
        kv_k,
        kv_v,
        sample_indices,
    };
    marmot_tensor_t *outputs[1] = {logits};

    err = marmot_graph_execute(graph, ctx, inputs, 7, outputs, 1);
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    return marmot_device_synchronize(ctx);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s [--metal] <model.gguf> <prompt...> [max_new_tokens]\n", argv[0]);
        return 2;
    }

    marmot_backend_type_t backend = MARMOT_BACKEND_CPU;
    int arg_start = 1;

    if (argc > 1 && strcmp(argv[1], "--metal") == 0) {
        backend = MARMOT_BACKEND_METAL;
        arg_start = 2;
        if (argc < 4) {
            fprintf(stderr, "usage: %s [--metal] <model.gguf> <prompt...> [max_new_tokens]\n", argv[0]);
            return 2;
        }
    }

    const char *gguf_path = argv[arg_start];
    size_t max_new_tokens = 32;
    if (argc >= arg_start + 3) {
        const char *maybe_steps = argv[argc - 1];
        char *end = nullptr;
        unsigned long parsed = strtoul(maybe_steps, &end, 10);
        if (end != nullptr && *end == '\0' && parsed > 0) {
            max_new_tokens = (size_t)parsed;
            argc -= 1;
        }
    }

    char *prompt = join_args(argc, argv, arg_start + 1);
    if (prompt == nullptr) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    marmot_tokenizer_options_t tok_opts;
    marmot_error_t err = marmot_tokenizer_options_init(&tok_opts);
    if (err != MARMOT_SUCCESS) {
        die("tokenizer options init", err);
    }

    marmot_tokenizer_t *tokenizer = nullptr;
    err = marmot_tokenizer_create_from_gguf_file(gguf_path, &tok_opts, &tokenizer);
    if (err != MARMOT_SUCCESS) {
        die("load tokenizer", err);
    }

    marmot_tokenizer_special_ids_t special_ids;
    err = marmot_tokenizer_get_special_ids(tokenizer, &special_ids);
    if (err != MARMOT_SUCCESS) {
        die("get special ids", err);
    }

    marmot_tokenizer_encode_options_t enc_opts;
    err = marmot_tokenizer_encode_options_init(&enc_opts);
    if (err != MARMOT_SUCCESS) {
        die("encode options init", err);
    }
    enc_opts.add_bos = special_ids.has_bos;

    size_t prompt_tokens_len = 0;
    err = marmot_tokenizer_encode(tokenizer, prompt, strlen(prompt), &enc_opts, nullptr, &prompt_tokens_len);
    if (err != MARMOT_SUCCESS) {
        die("tokenize prompt", err);
    }
    if (prompt_tokens_len == 0) {
        fprintf(stderr, "prompt tokenized to empty sequence\n");
        return 1;
    }
    if (max_new_tokens > SIZE_MAX - prompt_tokens_len) {
        die("max_new_tokens overflow", MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_token_id_t *prompt_tokens = calloc(prompt_tokens_len, sizeof(*prompt_tokens));
    if (prompt_tokens == nullptr) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    size_t prompt_tokens_cap = prompt_tokens_len;
    err = marmot_tokenizer_encode(tokenizer, prompt, strlen(prompt), &enc_opts, prompt_tokens, &prompt_tokens_cap);
    if (err != MARMOT_SUCCESS) {
        die("tokenize prompt", err);
    }

    marmot_context_t *ctx = marmot_init(backend);
    if (ctx == nullptr) {
        die("marmot_init", marmot_get_last_error());
    }

    fprintf(stderr, "Backend: %s\n", backend == MARMOT_BACKEND_METAL ? "Metal" : "CPU");

    marmot_gguf_model_t *model = nullptr;
    err = marmot_gguf_model_load(gguf_path, backend, &model);
    if (err != MARMOT_SUCCESS) {
        die("load gguf model", err);
    }

    marmot_gguf_model_meta_t meta;
    if (!marmot_gguf_model_metadata(model, &meta)) {
        die("read model metadata", MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const size_t max_total_tokens = prompt_tokens_len + max_new_tokens;
    if (max_total_tokens == 0 || max_total_tokens > meta.context_length) {
        fprintf(
            stderr, "requested %zu total tokens, but model context length is %zu\n", max_total_tokens, meta.context_length
        );
        return 1;
    }

    const marmot_tensor_t *token_embedding = marmot_gguf_model_tensor(model, "token_embd.weight");
    if (token_embedding == nullptr) {
        die("lookup token_embd.weight", MARMOT_ERROR_INVALID_ARGUMENT);
    }

    const marmot_dtype_t activation_dtype = marmot_activation_dtype_for_architecture(meta.architecture, backend);
    const marmot_dtype_t kv_dtype = marmot_activation_dtype_for_architecture(meta.architecture, backend);
    const size_t block_size = 16;
    const size_t max_blocks_per_seq = ceil_div(max_total_tokens, block_size);

    marmot_packed_graph_options_t graph_opts;
    err = marmot_packed_graph_options_init(&graph_opts);
    if (err != MARMOT_SUCCESS) {
        die("packed graph options init", err);
    }
    graph_opts.flags = MARMOT_PACKED_GRAPH_FLAG_NONE;
    graph_opts.token_count = 1;
    graph_opts.sample_count = 1;
    graph_opts.max_seqs = 1;
    graph_opts.max_seq_len = max_total_tokens;
    graph_opts.block_size = block_size;
    graph_opts.num_kv_blocks = max_blocks_per_seq;
    graph_opts.kv_dtype = kv_dtype;

    marmot_graph_t *graph = nullptr;
    err = marmot_graph_from_model_packed(model, backend, &graph_opts, &graph);
    if (err != MARMOT_SUCCESS) {
        die("graph from model packed", err);
    }

    const size_t token_shape[1] = {1};
    const size_t hidden_shape[2] = {1, meta.n_embd};
    const size_t token_meta_shape[2] = {1, 4};
    const size_t logits_shape[2] = {1, meta.n_vocab};

    marmot_tensor_t *token_id_tensor = marmot_tensor_create(ctx, token_shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *hidden = marmot_tensor_create(ctx, hidden_shape, 2, activation_dtype);
    marmot_tensor_t *positions = marmot_tensor_create(ctx, token_shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *token_meta = marmot_tensor_create(ctx, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *sample_indices = marmot_tensor_create(ctx, token_shape, 1, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *logits = marmot_tensor_create(ctx, logits_shape, 2, activation_dtype);
    if (token_id_tensor == nullptr || hidden == nullptr || positions == nullptr || token_meta == nullptr ||
        sample_indices == nullptr || logits == nullptr) {
        die("allocate tensors", MARMOT_ERROR_OUT_OF_MEMORY);
    }

    marmot_kv_pool_options_t kv_opts;
    err = marmot_kv_pool_options_init(&kv_opts);
    if (err != MARMOT_SUCCESS) {
        die("kv pool options init", err);
    }
    kv_opts.backend = backend;
    kv_opts.max_seqs = 1;
    kv_opts.max_seq_len = max_total_tokens;
    kv_opts.block_size = block_size;
    kv_opts.num_blocks = max_blocks_per_seq;
    kv_opts.num_layers = meta.n_layer;
    kv_opts.num_kv_heads = meta.n_head_kv != 0 ? meta.n_head_kv : meta.n_head;
    kv_opts.head_dim = meta.head_dim != 0 ? meta.head_dim : meta.n_embd / meta.n_head;
    kv_opts.kv_dtype = kv_dtype;

    marmot_kv_pool_t *kv_pool = nullptr;
    err = marmot_kv_pool_create(ctx, &kv_opts, &kv_pool);
    if (err != MARMOT_SUCCESS) {
        die("kv pool create", err);
    }

    marmot_seq_slot_t seq = 0;
    err = marmot_kv_pool_acquire_seq(kv_pool, &seq);
    if (err != MARMOT_SUCCESS) {
        die("kv pool acquire seq", err);
    }

    marmot_tensor_t *kv_k = nullptr;
    marmot_tensor_t *kv_v = nullptr;
    marmot_tensor_t *block_table = nullptr;
    err = marmot_kv_pool_get_tensors(kv_pool, &kv_k, &kv_v, &block_table);
    if (err != MARMOT_SUCCESS) {
        die("kv pool get tensors", err);
    }
    if (kv_k == nullptr || kv_v == nullptr || block_table == nullptr) {
        die("kv pool tensors unavailable", MARMOT_ERROR_INVALID_OPERATION);
    }

    marmot_tensor_t *block_table_snapshot = nullptr;
    if (backend == MARMOT_BACKEND_METAL) {
        const size_t block_table_shape[2] = {block_table->shape.shape[0], block_table->shape.shape[1]};
        block_table_snapshot = marmot_tensor_create(ctx, block_table_shape, 2, block_table->dtype);
        if (block_table_snapshot == nullptr) {
            die("allocate block table snapshot", MARMOT_ERROR_OUT_OF_MEMORY);
        }
    }

    float *logits_f32 = malloc(meta.n_vocab * sizeof(*logits_f32));
    if (logits_f32 == nullptr) {
        die("allocate logits scratch", MARMOT_ERROR_OUT_OF_MEMORY);
    }

    marmot_token_id_t *all_tokens = calloc(max_total_tokens + 1, sizeof(*all_tokens));
    if (all_tokens == nullptr) {
        die("allocate output token buffer", MARMOT_ERROR_OUT_OF_MEMORY);
    }
    size_t all_len = 0;
    for (size_t i = 0; i < prompt_tokens_len; ++i) {
        all_tokens[all_len++] = prompt_tokens[i];
    }

    for (size_t i = 0; i < prompt_tokens_len; ++i) {
        marmot_kv_append_plan_t plan = {0};
        marmot_kv_slot_t slot = 0;
        size_t start_pos = 0;
        err = marmot_kv_pool_prepare_append(kv_pool, seq, 1, &plan, &slot, &start_pos);
        if (err != MARMOT_SUCCESS) {
            die("kv pool prepare prompt append", err);
        }

        err = run_decode_step(
            ctx, token_embedding, graph, token_id_tensor, hidden, positions, token_meta, sample_indices, logits,
            block_table, block_table_snapshot, kv_k, kv_v, activation_dtype, meta.embedding_scale, seq, start_pos, slot,
            prompt_tokens[i]
        );
        if (err != MARMOT_SUCCESS) {
            marmot_kv_pool_abort_append(kv_pool, &plan);
            die("prompt decode step", err);
        }

        err = marmot_kv_pool_commit_append(kv_pool, &plan);
        if (err != MARMOT_SUCCESS) {
            die("kv pool commit prompt append", err);
        }
    }

    for (size_t step = 0; step < max_new_tokens; ++step) {
        marmot_token_id_t next = argmax_logits(ctx, logits, meta.n_vocab, logits_f32);
        if (next == MARMOT_TOKEN_ID_INVALID) {
            die("argmax logits", MARMOT_ERROR_INVALID_OPERATION);
        }

        all_tokens[all_len++] = next;
        if (special_ids.has_eos && next == special_ids.eos_id) {
            break;
        }

        marmot_kv_append_plan_t plan = {0};
        marmot_kv_slot_t slot = 0;
        size_t start_pos = 0;
        err = marmot_kv_pool_prepare_append(kv_pool, seq, 1, &plan, &slot, &start_pos);
        if (err != MARMOT_SUCCESS) {
            die("kv pool prepare decode append", err);
        }

        err = run_decode_step(
            ctx, token_embedding, graph, token_id_tensor, hidden, positions, token_meta, sample_indices, logits,
            block_table, block_table_snapshot, kv_k, kv_v, activation_dtype, meta.embedding_scale, seq, start_pos, slot,
            next
        );
        if (err != MARMOT_SUCCESS) {
            marmot_kv_pool_abort_append(kv_pool, &plan);
            die("decode step", err);
        }

        err = marmot_kv_pool_commit_append(kv_pool, &plan);
        if (err != MARMOT_SUCCESS) {
            die("kv pool commit decode append", err);
        }
    }

    marmot_tokenizer_decode_options_t dec_opts;
    err = marmot_tokenizer_decode_options_init(&dec_opts);
    if (err != MARMOT_SUCCESS) {
        die("decode options init", err);
    }
    dec_opts.skip_special = true;

    size_t decoded_len = 0;
    err = marmot_tokenizer_decode(tokenizer, all_tokens, all_len, &dec_opts, nullptr, &decoded_len);
    if (err != MARMOT_SUCCESS) {
        die("decode output", err);
    }

    char *decoded = malloc(decoded_len);
    if (decoded == nullptr) {
        die("allocate decoded buffer", MARMOT_ERROR_OUT_OF_MEMORY);
    }
    size_t decoded_cap = decoded_len;
    err = marmot_tokenizer_decode(tokenizer, all_tokens, all_len, &dec_opts, decoded, &decoded_cap);
    if (err != MARMOT_SUCCESS) {
        die("decode output", err);
    }

    fprintf(stdout, "%s\n", decoded);

    free(decoded);
    free(all_tokens);
    free(logits_f32);
    marmot_tensor_destroy(block_table_snapshot);
    err = marmot_kv_pool_release_seq(kv_pool, seq);
    if (err != MARMOT_SUCCESS) {
        die("kv pool release seq", err);
    }
    marmot_kv_pool_destroy(kv_pool);
    marmot_tensor_destroy(logits);
    marmot_tensor_destroy(sample_indices);
    marmot_tensor_destroy(token_meta);
    marmot_tensor_destroy(positions);
    marmot_tensor_destroy(hidden);
    marmot_tensor_destroy(token_id_tensor);
    marmot_graph_destroy(graph);
    marmot_gguf_model_destroy(model);
    marmot_destroy(ctx);
    marmot_tokenizer_destroy(tokenizer);
    free(prompt_tokens);
    free(prompt);
    return 0;
}
