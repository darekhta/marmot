#include "marmot/error.h"
#include "marmot/inference/model.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>

#include "inference/model/model.hpp"
#include "inference_handles.hpp"

namespace {

using marmot::inference::Model;

using Handle = MarmotModelHandle;

[[nodiscard]] Handle *from_api(marmot_model_t *ptr) {
    return reinterpret_cast<Handle *>(ptr);
}

[[nodiscard]] const Handle *from_api(const marmot_model_t *ptr) {
    return reinterpret_cast<const Handle *>(ptr);
}

void set_last_error(Handle *handle, marmot_error_t code, std::string_view message) {
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
}

void set_success(Handle *handle) {
    set_last_error(handle, MARMOT_SUCCESS, {});
}

[[nodiscard]] marmot_error_t validate_options(const marmot_model_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_MODEL_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_model_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

} // namespace

extern "C" {

marmot_error_t marmot_model_options_init(marmot_model_options_t *opts) {
    try {
        if (opts == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(marmot_model_options_t);
        opts->struct_version = MARMOT_MODEL_OPTIONS_VERSION;
        opts->flags = MARMOT_MODEL_FLAG_STRICT_VALIDATION;
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_model_options_init threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t
marmot_model_load_file(const char *path, const marmot_model_options_t *opts, marmot_model_t **out_model) {
    try {
        if (out_model == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        *out_model = nullptr;

        marmot_error_t status = validate_options(opts);
        if (status != MARMOT_SUCCESS) {
            return status;
        }

        auto handle = std::unique_ptr<Handle>(new (std::nothrow) Handle());
        if (!handle) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }

        marmot_error_t init_status = MARMOT_ERROR_INVALID_OPERATION;
        std::string error;
        auto model = Model::load_file(path, *opts, init_status, error);
        if (!model || init_status != MARMOT_SUCCESS) {
            if (error.empty()) {
                error = "failed to create model";
            }
            marmot_set_error(init_status, error.c_str());
            set_last_error(handle.get(), init_status, error);
            return init_status;
        }

        handle->impl = std::shared_ptr<Model>(std::move(model));
        set_success(handle.get());
        *out_model = reinterpret_cast<marmot_model_t *>(handle.release());
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_model_load_file threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

void marmot_model_destroy(marmot_model_t *model) {
    try {
        delete from_api(model);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_model_destroy threw");
    }
}

marmot_error_t marmot_model_get_info(const marmot_model_t *model, marmot_model_info_t *out_info) {
    try {
        const Handle *handle = from_api(model);
        if (handle == nullptr || handle->impl == nullptr || out_info == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        *out_info = handle->impl->info();
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_model_get_info threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

const marmot_error_info_t *marmot_model_last_error(const marmot_model_t *model) {
    try {
        const Handle *handle = from_api(model);
        if (handle == nullptr) {
            return marmot_get_last_error_info();
        }
        std::lock_guard<std::mutex> lock(handle->last_error_mutex);
        return &handle->last_error;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_model_last_error threw");
        return marmot_get_last_error_info();
    }
}

} // extern "C"
