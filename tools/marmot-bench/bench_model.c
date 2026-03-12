#include "bench_model.h"

#include "marmot/device.h"
#include "marmot/graph/architecture.h"
#include "marmot/inference/engine.h"
#include "marmot/inference/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/context/context_internal.h"

static marmot_error_t
marmot_bench_apply_cpu_thread_override(const marmot_bench_model_config_t *config, marmot_context_t *ctx) {
    if (config == nullptr || ctx == nullptr || config->backend != MARMOT_BACKEND_CPU || config->n_threads == 0) {
        return MARMOT_SUCCESS;
    }
    return marmot_context_set_thread_count(ctx, config->n_threads);
}

void marmot_bench_model_config_init(marmot_bench_model_config_t *config) {
    config->model_path = nullptr;
    config->ctx_size = 4096;
    config->batch_size = 512;
    config->ubatch_size = 512;
    config->max_seqs = 1;
    config->max_batch_seqs = 1;
    config->kv_type_k = MARMOT_DTYPE_FLOAT16;
    config->kv_type_v = MARMOT_DTYPE_FLOAT16;
    config->gpu_layers = 99; // Default: offload all layers
    config->flash_attn = true;
    config->create_engine = true;
    config->n_threads = 4;
    config->backend = MARMOT_BACKEND_METAL;
}

marmot_error_t marmot_bench_model_load(const marmot_bench_model_config_t *config, marmot_bench_model_t *out) {
    if (config == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (config->model_path == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "model_path is required");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    out->model = nullptr;
    out->engine = nullptr;
    out->ctx = nullptr;
    out->config = *config;
    out->loaded = false;
    memset(&out->info, 0, sizeof(out->info));

    // Initialize context for the specified backend
    out->ctx = marmot_init(config->backend);
    if (out->ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Failed to initialize backend");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    marmot_error_t thread_err = marmot_bench_apply_cpu_thread_override(config, out->ctx);
    if (thread_err != MARMOT_SUCCESS) {
        marmot_destroy(out->ctx);
        out->ctx = nullptr;
        return thread_err;
    }

    // Load the GGUF model
    marmot_model_options_t model_opts;
    marmot_error_t err = marmot_model_options_init(&model_opts);
    if (err != MARMOT_SUCCESS) {
        marmot_destroy(out->ctx);
        out->ctx = nullptr;
        return err;
    }

    marmot_model_t *model = nullptr;
    err = marmot_model_load_file(config->model_path, &model_opts, &model);
    if (err != MARMOT_SUCCESS) {
        marmot_destroy(out->ctx);
        out->ctx = nullptr;
        return err;
    }
    out->model = model;

    // Get model info
    err = marmot_model_get_info(model, &out->info);
    if (err != MARMOT_SUCCESS) {
        marmot_model_destroy(model);
        marmot_destroy(out->ctx);
        out->model = nullptr;
        out->ctx = nullptr;
        return err;
    }

    if (!config->create_engine) {
        out->loaded = true;
        return MARMOT_SUCCESS;
    }

    // Create serving engine for benchmarking
    marmot_serving_engine_options_t engine_opts;
    err = marmot_serving_engine_options_init(&engine_opts);
    if (err != MARMOT_SUCCESS) {
        marmot_model_destroy(model);
        marmot_destroy(out->ctx);
        out->model = nullptr;
        out->ctx = nullptr;
        return err;
    }

    engine_opts.flags |= MARMOT_SERVING_ENGINE_FLAG_ENABLE_PREFIX_CACHE;
    engine_opts.max_seqs = config->max_seqs;
    engine_opts.max_batch_seqs = config->max_batch_seqs;
    engine_opts.max_seq_len = config->ctx_size;
    engine_opts.max_num_tokens = config->batch_size;
    if (engine_opts.max_num_tokens == 0 || engine_opts.max_num_tokens > engine_opts.max_seq_len) {
        engine_opts.max_num_tokens = engine_opts.max_seq_len;
    }
    if (engine_opts.block_size == 0) {
        engine_opts.block_size = 16;
    }
    size_t blocks_per_seq = (engine_opts.max_seq_len + engine_opts.block_size - 1) / engine_opts.block_size;
    if (blocks_per_seq == 0) {
        blocks_per_seq = 1;
    }
    engine_opts.num_kv_blocks = blocks_per_seq * engine_opts.max_seqs;
    if (engine_opts.num_kv_blocks < 4) {
        engine_opts.num_kv_blocks = 4;
    }

    const marmot_architecture_t arch = marmot_architecture_from_string(out->info.architecture);
    const marmot_dtype_t activation_dtype = marmot_activation_dtype_for_architecture(arch, config->backend);
    engine_opts.kv_dtype = activation_dtype;
    out->config.kv_type_k = activation_dtype;
    out->config.kv_type_v = activation_dtype;

    marmot_serving_engine_t *engine = nullptr;
    err = marmot_serving_engine_create(out->ctx, model, &engine_opts, &engine);
    if (err != MARMOT_SUCCESS) {
        marmot_model_destroy(model);
        marmot_destroy(out->ctx);
        out->model = nullptr;
        out->ctx = nullptr;
        return err;
    }
    out->engine = engine;

    out->loaded = true;
    return MARMOT_SUCCESS;
}

void marmot_bench_model_free(marmot_bench_model_t *model) {
    if (model == nullptr) {
        return;
    }

    if (model->engine != nullptr) {
        marmot_serving_engine_destroy((marmot_serving_engine_t *)model->engine);
        model->engine = nullptr;
    }

    if (model->model != nullptr) {
        marmot_model_destroy((marmot_model_t *)model->model);
        model->model = nullptr;
    }

    if (model->ctx != nullptr) {
        marmot_destroy(model->ctx);
        model->ctx = nullptr;
    }

    model->loaded = false;
}

static char model_info_buffer[512];

const char *marmot_bench_model_info(const marmot_bench_model_t *model) {
    if (model == nullptr || !model->loaded) {
        return "no model";
    }

    const char *path = model->config.model_path;
    const char *basename = path;

    // Find last path separator
    for (const char *p = path; *p; p++) {
        if (*p == '/' || *p == '\\') {
            basename = p + 1;
        }
    }

    // Include model architecture and parameter count if available
    if (model->info.n_layer > 0) {
        double params_b = (double)(model->info.n_embd * model->info.n_layer * 12) / 1e9; // rough estimate
        snprintf(
            model_info_buffer, sizeof(model_info_buffer), "%s (%s, ~%.1fB, ctx=%zu, batch=%zu, seqs=%zu, bseqs=%zu)",
            basename, model->info.architecture, params_b, model->config.ctx_size, model->config.batch_size,
            model->config.max_seqs, model->config.max_batch_seqs
        );
    } else {
        snprintf(
            model_info_buffer, sizeof(model_info_buffer), "%s (ctx=%zu, batch=%zu, seqs=%zu, bseqs=%zu)", basename,
            model->config.ctx_size, model->config.batch_size, model->config.max_seqs, model->config.max_batch_seqs
        );
    }

    return model_info_buffer;
}

marmot_error_t marmot_bench_parse_kv_type(const char *str, marmot_dtype_t *out) {
    if (str == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (strcmp(str, "f16") == 0 || strcmp(str, "fp16") == 0) {
        *out = MARMOT_DTYPE_FLOAT16;
        return MARMOT_SUCCESS;
    }

    if (strcmp(str, "f32") == 0 || strcmp(str, "fp32") == 0) {
        *out = MARMOT_DTYPE_FLOAT32;
        return MARMOT_SUCCESS;
    }

    if (strcmp(str, "bf16") == 0 || strcmp(str, "bfloat16") == 0) {
        *out = MARMOT_DTYPE_BFLOAT16;
        return MARMOT_SUCCESS;
    }

    if (strcmp(str, "q8_0") == 0 || strcmp(str, "q8") == 0 || strcmp(str, "i8") == 0) {
        *out = MARMOT_DTYPE_INT8;
        return MARMOT_SUCCESS;
    }

    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unknown KV cache type");
    return MARMOT_ERROR_INVALID_ARGUMENT;
}
