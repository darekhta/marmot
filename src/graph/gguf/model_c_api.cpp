#include "marmot/error.h"

#include <memory>

#include "model.hpp"

struct marmot_gguf_model {
    std::unique_ptr<marmot::gguf::Model> inner;
};

extern "C" {

marmot_error_t
marmot_gguf_model_load(const char *path, marmot_backend_type_t backend, marmot_gguf_model_t **out_model) {
    if (path == nullptr || out_model == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (backend != MARMOT_BACKEND_CPU && backend != MARMOT_BACKEND_METAL && backend != MARMOT_BACKEND_CUDA) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_gguf_t *gguf = marmot_gguf_load(path);
    if (gguf == nullptr) {
        return marmot_get_last_error();
    }

    try {
        auto model = std::make_unique<marmot::gguf::Model>(gguf, backend);
        *out_model = new marmot_gguf_model{std::move(model)};
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_gguf_unload(gguf);
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_model_load threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

void marmot_gguf_model_destroy(marmot_gguf_model_t *model) {
    delete model;
}

const marmot_gguf_t *marmot_gguf_model_file(const marmot_gguf_model_t *model) {
    try {
        return model ? model->inner->file() : nullptr;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_model_file threw");
        return nullptr;
    }
}

const marmot_tensor_t *marmot_gguf_model_tensor(const marmot_gguf_model_t *model, const char *name) {
    try {
        return (model && name) ? model->inner->get_tensor(name) : nullptr;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_model_tensor threw");
        return nullptr;
    }
}

const marmot_gguf_tensor_t *marmot_gguf_model_tensor_info(const marmot_gguf_model_t *model, size_t index) {
    try {
        return model ? model->inner->get_tensor_info(index) : nullptr;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_model_tensor_info threw");
        return nullptr;
    }
}

size_t marmot_gguf_model_tensor_count(const marmot_gguf_model_t *model) {
    try {
        return model ? model->inner->tensor_count() : 0;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_model_tensor_count threw");
        return 0;
    }
}

bool marmot_gguf_model_metadata(const marmot_gguf_model_t *model, marmot_gguf_model_meta_t *out) {
    try {
        return model ? model->inner->get_metadata(out) : false;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_model_metadata threw");
        return false;
    }
}

} // extern "C"
