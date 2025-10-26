#include "marmot/error.h"
#include "marmot/inference/engine.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>

#include "inference/frontends/serving_engine.hpp"
#include "inference_handles.hpp"

namespace {

using marmot::inference::ServingEngine;

using EngineHandle = MarmotServingEngineHandle;
using ModelHandle = MarmotModelHandle;

[[nodiscard]] EngineHandle *from_api(marmot_serving_engine_t *ptr) {
    return reinterpret_cast<EngineHandle *>(ptr);
}

[[nodiscard]] const EngineHandle *from_api(const marmot_serving_engine_t *ptr) {
    return reinterpret_cast<const EngineHandle *>(ptr);
}

[[nodiscard]] const ModelHandle *from_api(const marmot_model_t *ptr) {
    return reinterpret_cast<const ModelHandle *>(ptr);
}

void set_last_error(EngineHandle *handle, marmot_error_t code, std::string_view message) {
    if (handle == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(handle->last_error_mutex);
    handle->last_error = marmot_error_info_t{
        .code = code,
        .message = {0},
        .file = nullptr,
        .line = 0,
        .function = nullptr,
    };

    if (!message.empty()) {
        const size_t n = std::min(message.size(), sizeof(handle->last_error.message) - 1);
        std::memcpy(handle->last_error.message, message.data(), n);
        handle->last_error.message[n] = '\0';
    }

    if (code != MARMOT_SUCCESS) {
        if (handle->last_error.message[0] != '\0') {
            marmot_set_error(code, handle->last_error.message);
        } else {
            marmot_set_error(code, "serving engine error");
        }
    }
}

void set_success(EngineHandle *handle) {
    set_last_error(handle, MARMOT_SUCCESS, {});
}

[[nodiscard]] marmot_error_t validate_engine_options(const marmot_serving_engine_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_SERVING_ENGINE_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_serving_engine_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

[[nodiscard]] marmot_error_t validate_generate_options(const marmot_llm_generate_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_LLM_GENERATE_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_llm_generate_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

[[nodiscard]] marmot_error_t validate_sampling_options(const marmot_llm_sampling_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_LLM_SAMPLING_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_llm_sampling_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

} // namespace

extern "C" {

marmot_error_t marmot_serving_engine_options_init(marmot_serving_engine_options_t *opts) {
    try {
        if (opts == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(marmot_serving_engine_options_t);
        opts->struct_version = MARMOT_SERVING_ENGINE_OPTIONS_VERSION;
        opts->flags = 0;
        opts->max_seqs = 16;
        opts->max_batch_seqs = 8;
        opts->max_num_tokens = 512;
        opts->max_seq_len = 2048;
        opts->block_size = 16;
        opts->num_kv_blocks = 256;
        opts->num_swap_blocks = 0;
        opts->kv_dtype = MARMOT_DTYPE_FLOAT16;
        opts->kv_block_watermark = 0.01f;
        opts->prefill_chunk_size = 0;
        opts->pnext = nullptr;
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_options_init threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_serving_engine_create(
    const marmot_context_t *ctx, const marmot_model_t *model, const marmot_serving_engine_options_t *opts,
    marmot_serving_engine_t **out_engine
) {
    try {
        if (out_engine == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        *out_engine = nullptr;

        marmot_error_t status = validate_engine_options(opts);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        const ModelHandle *model_handle = from_api(model);
        if (ctx == nullptr || model_handle == nullptr || model_handle->impl == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        auto handle = std::unique_ptr<EngineHandle>(new (std::nothrow) EngineHandle());
        if (!handle) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        marmot_error_t init_status = MARMOT_ERROR_INVALID_OPERATION;
        std::string error;
        auto engine = ServingEngine::create(ctx, model_handle->impl, *opts, init_status, error);
        if (!engine || init_status != MARMOT_SUCCESS) {
            if (error.empty()) {
                error = "failed to create serving engine";
            }
            marmot_set_error(init_status, error.c_str());
            set_last_error(handle.get(), init_status, error);
            return init_status;
        }

        handle->impl = std::move(engine);
        set_success(handle.get());
        *out_engine = reinterpret_cast<marmot_serving_engine_t *>(handle.release());
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_create threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

void marmot_serving_engine_destroy(marmot_serving_engine_t *engine) {
    try {
        delete from_api(engine);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_destroy threw");
    }
}

marmot_error_t marmot_serving_engine_submit(
    marmot_serving_engine_t *engine, const marmot_token_id_t *prompt_tokens, size_t prompt_len,
    const marmot_llm_generate_options_t *gen_opts, const marmot_llm_sampling_options_t *sampling_opts,
    marmot_request_id_t *out_request_id
) {
    try {
        EngineHandle *handle = from_api(engine);
        if (handle == nullptr || handle->impl == nullptr || out_request_id == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        *out_request_id = 0;

        marmot_error_t status = validate_generate_options(gen_opts);
        if (status != MARMOT_SUCCESS) {
            set_last_error(handle, status, "invalid generate options");
            return status;
        }
        status = validate_sampling_options(sampling_opts);
        if (status != MARMOT_SUCCESS) {
            set_last_error(handle, status, "invalid sampling options");
            return status;
        }

        std::string error;
        marmot_request_id_t request_id = 0;
        marmot_error_t submit_status =
            handle->impl->submit(prompt_tokens, prompt_len, *gen_opts, *sampling_opts, request_id, error);
        if (submit_status != MARMOT_SUCCESS) {
            if (error.empty()) {
                error = "serving engine submit failed";
            }
            set_last_error(handle, submit_status, error);
            return submit_status;
        }

        *out_request_id = request_id;
        set_success(handle);
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_submit threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_serving_engine_step(marmot_serving_engine_t *engine, size_t max_steps, size_t *out_steps_done) {
    try {
        EngineHandle *handle = from_api(engine);
        if (handle == nullptr || handle->impl == nullptr || out_steps_done == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        size_t steps_done = 0;
        std::string error;
        marmot_error_t status = handle->impl->step(max_steps, steps_done, error);
        *out_steps_done = steps_done;
        if (status != MARMOT_SUCCESS) {
            if (error.empty()) {
                error = "serving engine step failed";
            }
            set_last_error(handle, status, error);
            return status;
        }

        set_success(handle);
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_step threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_llm_request_state_t
marmot_serving_engine_request_state(const marmot_serving_engine_t *engine, marmot_request_id_t request_id) {
    try {
        const EngineHandle *handle = from_api(engine);
        if (handle == nullptr || handle->impl == nullptr) {
            return MARMOT_LLM_REQUEST_STATE_INVALID;
        }
        return handle->impl->request_state(request_id);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_request_state threw");
        return MARMOT_LLM_REQUEST_STATE_INVALID;
    }
}

marmot_error_t marmot_serving_engine_request_cancel(marmot_serving_engine_t *engine, marmot_request_id_t request_id) {
    try {
        EngineHandle *handle = from_api(engine);
        if (handle == nullptr || handle->impl == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        std::string error;
        marmot_error_t status = handle->impl->request_cancel(request_id, error);
        if (status != MARMOT_SUCCESS) {
            if (error.empty()) {
                error = "serving engine cancel failed";
            }
            set_last_error(handle, status, error);
            return status;
        }

        set_success(handle);
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_request_cancel threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_serving_engine_request_release(marmot_serving_engine_t *engine, marmot_request_id_t request_id) {
    try {
        EngineHandle *handle = from_api(engine);
        if (handle == nullptr || handle->impl == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        std::string error;
        marmot_error_t status = handle->impl->request_release(request_id, error);
        if (status != MARMOT_SUCCESS) {
            if (error.empty()) {
                error = "serving engine release failed";
            }
            set_last_error(handle, status, error);
            return status;
        }

        set_success(handle);
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_request_release threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t
marmot_serving_engine_last_batch(const marmot_serving_engine_t *engine, marmot_serving_engine_batch_view_t *out_batch) {
    try {
        const EngineHandle *handle = from_api(engine);
        if (handle == nullptr || handle->impl == nullptr || out_batch == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        *out_batch = marmot_serving_engine_batch_view_t{};
        marmot_error_t status = handle->impl->last_batch_view(*out_batch);
        if (status != MARMOT_SUCCESS) {
            set_last_error(const_cast<EngineHandle *>(handle), status, "serving engine last batch failed");
            return status;
        }

        set_success(const_cast<EngineHandle *>(handle));
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_serving_engine_last_batch threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

} // extern "C"
