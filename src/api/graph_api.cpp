#include "marmot/error.h"
#include "marmot/graph/graph.h"

#include <cstring>

#include "graph/graph_api_impl.hpp"

extern "C" {

extern marmot_routing_policy_t marmot_routing_policy_from_env(void);

marmot_graph_t *marmot_graph_create(void) {
    try {
        return marmot::graph::GraphApiFacade::create();
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_create threw");
        return nullptr;
    }
}

void marmot_graph_destroy(marmot_graph_t *graph) {
    try {
        marmot::graph::GraphApiFacade::destroy(graph);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_destroy threw");
    }
}

marmot_backend_type_t marmot_graph_get_backend(const marmot_graph_t *graph) {
    try {
        return marmot::graph::GraphApiFacade::backend(graph);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_get_backend threw");
        return MARMOT_BACKEND_CPU;
    }
}

marmot_error_t
marmot_graph_add_input(marmot_graph_t *graph, const marmot_graph_tensor_desc_t *desc, marmot_value_id_t *out_value_id) {
    try {
        return marmot::graph::GraphApiFacade::add_input(graph, desc, out_value_id);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_add_input threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_graph_add_op(
    marmot_graph_t *graph, const char *op_name, const marmot_op_signature_t *signature,
    const marmot_value_id_t *input_ids, size_t num_inputs, const marmot_graph_tensor_desc_t *output_descs,
    size_t num_outputs, marmot_value_id_t *out_value_ids
) {
    try {
        return marmot::graph::GraphApiFacade::add_op(
            graph, op_name, signature, input_ids, num_inputs, output_descs, num_outputs, out_value_ids
        );
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_add_op threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_graph_finalize(marmot_graph_t *graph, marmot_backend_type_t backend) {
    try {
        return marmot::graph::GraphApiFacade::finalize(graph, backend);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_finalize threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_graph_finalize_options_init(marmot_graph_finalize_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    std::memset(opts, 0, sizeof(*opts));
    opts->struct_size = sizeof(marmot_graph_finalize_options_t);
    opts->struct_version = MARMOT_GRAPH_FINALIZE_OPTIONS_VERSION;
    opts->flags = 0;
    opts->routing_policy = MARMOT_ROUTING_AUTO;
    opts->backend = MARMOT_BACKEND_CPU;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_graph_finalize_with_options(
    marmot_graph_t *graph, const marmot_graph_finalize_options_t *opts, marmot_backend_type_t *out_backend
) {
    try {
        return marmot::graph::GraphApiFacade::finalize_with_options(graph, opts, out_backend);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_finalize_with_options threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_graph_finalize_auto(marmot_graph_t *graph, marmot_backend_type_t *out_backend) {
    try {
        marmot_graph_finalize_options_t opts{};
        marmot_error_t init_status = marmot_graph_finalize_options_init(&opts);
        if (init_status != MARMOT_SUCCESS) {
            return init_status;
        }
        opts.flags |= MARMOT_GRAPH_FINALIZE_FLAG_AUTO_BACKEND;
        opts.routing_policy = marmot_routing_policy_from_env();
        return marmot::graph::GraphApiFacade::finalize_with_options(graph, &opts, out_backend);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_finalize_auto threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_graph_execute(
    marmot_graph_t *graph, const marmot_context_t *ctx, const marmot_tensor_t **inputs, size_t num_inputs,
    marmot_tensor_t **outputs, size_t num_outputs
) {
    try {
        return marmot::graph::GraphApiFacade::execute(graph, ctx, inputs, num_inputs, outputs, num_outputs);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_execute threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_graph_dump_json(const marmot_graph_t *graph, const char *path) {
    try {
        return marmot::graph::GraphApiFacade::dump_json(graph, path);
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_graph_dump_json threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

} // extern "C"
