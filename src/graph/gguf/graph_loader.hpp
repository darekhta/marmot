#pragma once

#include "marmot/allocator.h"
#include "marmot/graph/gguf_model.h"
#include "marmot/types.h"

#include <cstddef>
#include <span>

#include "error.hpp"
#include "gguf_reader.hpp"
#include "gguf_validator.hpp"
#include "graph_builder.hpp"
#include "op_registry.hpp"
#include "tensor_resolver.hpp"

namespace marmot::gguf {

struct LoaderOptions {
    uint64_t flags{0};
    marmot_backend_type_t backend{MARMOT_BACKEND_CPU};
    const marmot_allocator *allocator{nullptr};
    marmot_packed_graph_options_t packed_opts{};
    const void *pnext{nullptr};
};

class GraphLoader {
  public:
    explicit GraphLoader(LoaderOptions options);

    [[nodiscard]] marmot_error_t load_file(const char *path, marmot_graph_t **out_graph);
    [[nodiscard]] marmot_error_t load_memory(std::span<const std::byte> data, marmot_graph_t **out_graph);
    [[nodiscard]] const Error &last_error() const noexcept {
        return error_;
    }

  private:
    LoaderOptions options_{};
    GgufReader reader_{};
    GgufValidator validator_{};
    GraphBuilder builder_;
    TensorResolver tensor_resolver_{};
    OpRegistry op_registry_{};
    Error error_{};
};

} // namespace marmot::gguf
