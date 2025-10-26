#pragma once

#include "marmot/graph/gguf_model.h"
#include "marmot/graph/graph.h"
#include "marmot/graph/graph_types.h"
#include "marmot/tensor.h"

#include <cstddef>
#include <string_view>

#include "error.hpp"

namespace marmot::gguf {

class GraphBuilder {
  public:
    GraphBuilder(marmot_backend_type_t backend, bool auto_backend, const marmot_packed_graph_options_t &packed_opts)
        : backend_(backend), auto_backend_(auto_backend), packed_opts_(packed_opts) {}

    [[nodiscard]] marmot_error_t build_from_file(const char *path, marmot_graph_t **out_graph, Error &err) const;

  private:
    marmot_backend_type_t backend_;
    bool auto_backend_;
    marmot_packed_graph_options_t packed_opts_{};
};

} // namespace marmot::gguf
