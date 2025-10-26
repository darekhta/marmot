#include "marmot/graph/gguf_loader.h"

#include "marmot/error.h"
#include "marmot/graph/gguf_model.h"

#include <bit>
#include <cstddef>
#include <cstring>
#include <memory>
#include <new>
#include <span>

#include "graph/gguf/gguf_internal.hpp"
#include "graph/gguf/graph_loader.hpp"

namespace {

using marmot::gguf::GraphLoader;
using marmot::gguf::LoaderOptions;

struct MarmotGgufLoaderHandle {
    std::unique_ptr<GraphLoader> impl;
    mutable marmot_error_info_t last_error{};
};

using LoaderHandle = MarmotGgufLoaderHandle;

[[nodiscard]] LoaderHandle *from_api(marmot_gguf_loader_t *ptr) {
    return reinterpret_cast<LoaderHandle *>(ptr);
}

[[nodiscard]] const LoaderHandle *from_api(const marmot_gguf_loader_t *ptr) {
    return reinterpret_cast<const LoaderHandle *>(ptr);
}

[[nodiscard]] marmot_error_t validate_options(const marmot_gguf_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_GGUF_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_gguf_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->packed_opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const marmot_packed_graph_options_t *packed = opts->packed_opts;
    if (packed->struct_version != MARMOT_PACKED_GRAPH_OPTIONS_VERSION ||
        packed->struct_size < sizeof(marmot_packed_graph_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (packed->token_count == 0 || packed->max_seqs == 0 || packed->max_seq_len == 0 || packed->block_size == 0 ||
        packed->num_kv_blocks == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (packed->token_count > packed->max_seq_len) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (packed->sample_count > packed->token_count) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (packed->block_size <= 1 || !std::has_single_bit(packed->block_size)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

[[nodiscard]] LoaderOptions to_loader_options(const marmot_gguf_options_t *opts) {
    LoaderOptions options{};
    options.flags = opts->flags;
    options.backend = opts->backend;
    options.allocator = opts->allocator;
    options.pnext = opts->pnext;
    options.packed_opts = *opts->packed_opts;
    return options;
}

} // namespace

extern "C" {

marmot_error_t marmot_gguf_options_init(marmot_gguf_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    std::memset(opts, 0, sizeof(*opts));
    opts->struct_size = sizeof(marmot_gguf_options_t);
    opts->struct_version = MARMOT_GGUF_OPTIONS_VERSION;
    opts->backend = MARMOT_BACKEND_CPU;
    opts->packed_opts = nullptr;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_gguf_loader_create(const marmot_gguf_options_t *opts, marmot_gguf_loader_t **out_loader) {
    if (out_loader == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_loader = nullptr;

    marmot_error_t status = validate_options(opts);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    auto handle = std::unique_ptr<LoaderHandle>(new (std::nothrow) LoaderHandle());
    if (!handle) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    LoaderOptions options = to_loader_options(opts);
    handle->impl = std::make_unique<GraphLoader>(options);
    handle->last_error = marmot_error_info_t{
        .code = MARMOT_SUCCESS,
        .message = {0},
        .file = nullptr,
        .line = 0,
        .function = nullptr,
    };

    *out_loader = reinterpret_cast<marmot_gguf_loader_t *>(handle.release());
    return MARMOT_SUCCESS;
}

void marmot_gguf_loader_destroy(marmot_gguf_loader_t *loader) {
    delete from_api(loader);
}

marmot_error_t
marmot_gguf_loader_load_file(marmot_gguf_loader_t *loader, const char *path, marmot_graph_t **out_graph) {
    LoaderHandle *handle = from_api(loader);
    if (handle == nullptr || handle->impl == nullptr || out_graph == nullptr || path == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_error_t status = handle->impl->load_file(path, out_graph);
    handle->last_error = handle->impl->last_error().to_info();
    return status;
}

marmot_error_t
marmot_gguf_loader_load_memory(marmot_gguf_loader_t *loader, const void *data, size_t len, marmot_graph_t **out_graph) {
    LoaderHandle *handle = from_api(loader);
    if (handle == nullptr || handle->impl == nullptr || out_graph == nullptr || data == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    std::span<const std::byte> bytes(reinterpret_cast<const std::byte *>(data), len);
    marmot_error_t status = handle->impl->load_memory(bytes, out_graph);
    handle->last_error = handle->impl->last_error().to_info();
    return status;
}

marmot_error_t marmot_gguf_loader_query_capabilities(const marmot_gguf_loader_t *loader, marmot_gguf_caps_t *out_caps) {
    (void)loader;
    if (out_caps == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    std::memset(out_caps, 0, sizeof(*out_caps));
    out_caps->struct_size = sizeof(marmot_gguf_caps_t);
    out_caps->struct_version = MARMOT_GGUF_CAPS_VERSION;
    out_caps->supported_flags =
        MARMOT_GGUF_FLAG_STRICT_VALIDATION | MARMOT_GGUF_FLAG_ALLOW_UNKNOWN_OPS | MARMOT_GGUF_FLAG_AUTO_BACKEND;
    out_caps->supported_version_min = marmot::gguf::kVersionSupported;
    out_caps->supported_version_max = marmot::gguf::kVersionSupported;
    return MARMOT_SUCCESS;
}

const marmot_error_info_t *marmot_gguf_loader_last_error(const marmot_gguf_loader_t *loader) {
    const LoaderHandle *handle = from_api(loader);
    if (handle == nullptr || handle->impl == nullptr) {
        return nullptr;
    }
    handle->last_error = handle->impl->last_error().to_info();
    return &handle->last_error;
}

marmot_gguf_t *marmot_gguf_load(const char *path) {
    try {
        return marmot::gguf::load_file(path);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_load threw");
        return nullptr;
    }
}

void marmot_gguf_unload(marmot_gguf_t *gguf) {
    try {
        marmot::gguf::unload_file(gguf);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_unload threw");
    }
}

const marmot_gguf_kv_t *marmot_gguf_find_kv(const marmot_gguf_t *gguf, const char *key) {
    try {
        return marmot::gguf::find_kv(gguf, key);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_find_kv threw");
        return nullptr;
    }
}

const marmot_gguf_tensor_t *marmot_gguf_find_tensor(const marmot_gguf_t *gguf, const char *name) {
    try {
        return marmot::gguf::find_tensor(gguf, name);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_gguf_find_tensor threw");
        return nullptr;
    }
}

} // extern "C"
