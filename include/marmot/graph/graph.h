#ifndef MARMOT_GRAPH_GRAPH_H
#define MARMOT_GRAPH_GRAPH_H

#include "marmot/error.h"
#include "marmot/graph/graph_types.h"
#include "marmot/graph/op_signature.h"
#include "marmot/types.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct marmot_graph marmot_graph_t;

marmot_graph_t *marmot_graph_create(void);
void marmot_graph_destroy(marmot_graph_t *graph);

MARMOT_NODISCARD marmot_backend_type_t marmot_graph_get_backend(const marmot_graph_t *graph);

//------------------------------------------------------------------------------
// Finalize options
//------------------------------------------------------------------------------

#define MARMOT_GRAPH_FINALIZE_OPTIONS_VERSION 1

typedef enum {
    MARMOT_GRAPH_FINALIZE_FLAG_AUTO_BACKEND = 1u << 0,
} marmot_graph_finalize_flags_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;
    marmot_routing_policy_t routing_policy;
    marmot_backend_type_t backend;
    const void *pnext;
    uint64_t reserved[4];
} marmot_graph_finalize_options_t;

MARMOT_NODISCARD marmot_error_t marmot_graph_finalize_options_init(marmot_graph_finalize_options_t *opts);

MARMOT_NODISCARD marmot_error_t marmot_graph_finalize_with_options(
    marmot_graph_t *graph, const marmot_graph_finalize_options_t *opts, marmot_backend_type_t *out_backend
);

MARMOT_NODISCARD marmot_error_t
marmot_graph_add_input(marmot_graph_t *graph, const marmot_graph_tensor_desc_t *desc, marmot_value_id_t *out_value_id);

MARMOT_NODISCARD marmot_error_t marmot_graph_add_op(
    marmot_graph_t *graph, const char *op_name, const marmot_op_signature_t *signature,
    const marmot_value_id_t *input_ids, size_t num_inputs, const marmot_graph_tensor_desc_t *output_descs,
    size_t num_outputs, marmot_value_id_t *out_value_ids
);

MARMOT_NODISCARD marmot_error_t marmot_graph_finalize(marmot_graph_t *graph, marmot_backend_type_t backend);

MARMOT_NODISCARD marmot_error_t marmot_graph_finalize_auto(marmot_graph_t *graph, marmot_backend_type_t *out_backend);

MARMOT_NODISCARD marmot_error_t marmot_graph_execute(
    marmot_graph_t *graph, const marmot_context_t *ctx, const marmot_tensor_t **inputs, size_t num_inputs,
    marmot_tensor_t **outputs, size_t num_outputs
);

MARMOT_NODISCARD marmot_error_t marmot_graph_dump_json(const marmot_graph_t *graph, const char *path);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_GRAPH_GRAPH_H
