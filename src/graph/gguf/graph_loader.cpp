#include "graph_loader.hpp"

#include "marmot/graph/gguf_loader.h"

namespace marmot::gguf {

GraphLoader::GraphLoader(LoaderOptions options)
    : options_(options),
      builder_(options.backend, (options.flags & MARMOT_GGUF_FLAG_AUTO_BACKEND) != 0, options.packed_opts) {
    tensor_resolver_.set_flags(options.flags);
}

marmot_error_t GraphLoader::load_file(const char *path, marmot_graph_t **out_graph) {
    error_.clear();

    auto gguf = reader_.load_file(path, error_);
    if (!gguf) {
        return error_.code();
    }

    marmot_error_t validate_status = validator_.validate(gguf.get(), error_);
    if (validate_status != MARMOT_SUCCESS) {
        return validate_status;
    }

    return builder_.build_from_file(path, out_graph, error_);
}

marmot_error_t GraphLoader::load_memory(std::span<const std::byte> data, marmot_graph_t **out_graph) {
    error_.clear();

    auto gguf = reader_.load_memory(data, error_);
    if (!gguf) {
        return error_.code();
    }

    marmot_error_t validate_status = validator_.validate(gguf.get(), error_);
    if (validate_status != MARMOT_SUCCESS) {
        return validate_status;
    }

    error_.set(MARMOT_ERROR_NOT_IMPLEMENTED, "memory-backed GGUF loading is not implemented");
    (void)out_graph;
    return error_.code();
}

} // namespace marmot::gguf
