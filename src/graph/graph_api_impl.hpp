#pragma once

#include "marmot/error.h"
#include "marmot/graph/graph.h"
#include "marmot/graph/graph_types.h"
#include "marmot/graph/op_signature.h"
#include "marmot/types.h"

#include "graph_handle.hpp"

namespace marmot::graph {

class GraphApiFacade {
  public:
    [[nodiscard]] static marmot_graph_t *create();
    static void destroy(marmot_graph_t *graph);

    [[nodiscard]] static marmot_backend_type_t backend(const marmot_graph_t *graph);

    [[nodiscard]] static marmot_error_t
    add_input(marmot_graph_t *graph, const marmot_graph_tensor_desc_t *desc, marmot_value_id_t *out_value_id);

    [[nodiscard]] static marmot_error_t add_op(
        marmot_graph_t *graph, const char *op_name, const marmot_op_signature_t *signature,
        const marmot_value_id_t *input_ids, size_t num_inputs, const marmot_graph_tensor_desc_t *output_descs,
        size_t num_outputs, marmot_value_id_t *out_value_ids
    );

    [[nodiscard]] static marmot_error_t finalize(marmot_graph_t *graph, marmot_backend_type_t backend);
    [[nodiscard]] static marmot_error_t finalize_auto(marmot_graph_t *graph, marmot_backend_type_t *out_backend);
    [[nodiscard]] static marmot_error_t finalize_with_options(
        marmot_graph_t *graph, const marmot_graph_finalize_options_t *opts, marmot_backend_type_t *out_backend
    );

    [[nodiscard]] static marmot_error_t execute(
        marmot_graph_t *graph, const marmot_context_t *ctx, const marmot_tensor_t **inputs, size_t num_inputs,
        marmot_tensor_t **outputs, size_t num_outputs
    );

    [[nodiscard]] static marmot_error_t dump_json(const marmot_graph_t *graph, const char *path);
};

} // namespace marmot::graph
