#include "graph_api_impl.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <span>

namespace marmot::graph {

marmot_graph_t *GraphApiFacade::create() {
    try {
        return new marmot_graph();
    } catch (...) {
        return nullptr;
    }
}

void GraphApiFacade::destroy(marmot_graph_t *graph) {
    if (graph == nullptr) {
        return;
    }
    if (graph->external_cleanup) {
        graph->external_cleanup(graph->external_state);
    }
    delete graph;
}

marmot_backend_type_t GraphApiFacade::backend(const marmot_graph_t *graph) {
    if (graph == nullptr) {
        return MARMOT_BACKEND_CPU;
    }
    return graph->inner.backend();
}

marmot_error_t GraphApiFacade::add_input(
    marmot_graph_t *graph, const marmot_graph_tensor_desc_t *desc, marmot_value_id_t *out_value_id
) {
    if (graph == nullptr || desc == nullptr || out_value_id == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    auto result = graph->inner.add_input(*desc);
    if (!result) {
        return result.error();
    }
    *out_value_id = *result;
    return MARMOT_SUCCESS;
}

marmot_error_t GraphApiFacade::add_op(
    marmot_graph_t *graph, const char *op_name, const marmot_op_signature_t *signature,
    const marmot_value_id_t *input_ids, size_t num_inputs, const marmot_graph_tensor_desc_t *output_descs,
    size_t num_outputs, marmot_value_id_t *out_value_ids
) {
    if (graph == nullptr || op_name == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if ((num_inputs > 0 && input_ids == nullptr) ||
        (num_outputs > 0 && (output_descs == nullptr || out_value_ids == nullptr))) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    std::span inputs(input_ids, num_inputs);
    std::span outputs(output_descs, num_outputs);

    auto result = graph->inner.add_op(op_name, signature, inputs, outputs);
    if (!result) {
        return result.error();
    }

    if (result->size() != num_outputs) {
        return MARMOT_ERROR_INVALID_OPERATION;
    }
    std::copy(result->begin(), result->end(), out_value_ids);

    return MARMOT_SUCCESS;
}

marmot_error_t
GraphApiFacade::set_last_node_moe_params(marmot_graph_t *graph, marmot_ffn_type_t ffn_type, float weights_scale) {
    if (graph == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!std::isfinite(weights_scale)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return graph->inner.set_last_node_moe_params_checked(ffn_type, weights_scale);
}

marmot_error_t GraphApiFacade::finalize(marmot_graph_t *graph, marmot_backend_type_t backend) {
    if (graph == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return graph->inner.finalize(backend);
}

marmot_error_t GraphApiFacade::finalize_auto(marmot_graph_t *graph, marmot_backend_type_t *out_backend) {
    if (graph == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return graph->inner.finalize_auto(out_backend);
}

marmot_error_t GraphApiFacade::finalize_with_options(
    marmot_graph_t *graph, const marmot_graph_finalize_options_t *opts, marmot_backend_type_t *out_backend
) {
    if (graph == nullptr || opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_GRAPH_FINALIZE_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_graph_finalize_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if ((opts->flags & MARMOT_GRAPH_FINALIZE_FLAG_AUTO_BACKEND) != 0) {
        return graph->inner.finalize_auto_with_policy(opts->routing_policy, out_backend);
    }

    marmot_error_t status = graph->inner.finalize(opts->backend);
    if (status == MARMOT_SUCCESS && out_backend != nullptr) {
        *out_backend = opts->backend;
    }
    return status;
}

marmot_error_t GraphApiFacade::execute(
    marmot_graph_t *graph, const marmot_context_t *ctx, const marmot_tensor_t **inputs, size_t num_inputs,
    marmot_tensor_t **outputs, size_t num_outputs
) {
    if (graph == nullptr || ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    std::span<const marmot_tensor_t *const> input_span(inputs, num_inputs);
    std::span<marmot_tensor_t *const> output_span(outputs, num_outputs);

    return graph->inner.execute(ctx, input_span, output_span);
}

marmot_error_t GraphApiFacade::dump_json(const marmot_graph_t *graph, const char *path) {
    if (graph == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return graph->inner.dump_json(path);
}

} // namespace marmot::graph
