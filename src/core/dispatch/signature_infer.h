#ifndef MARMOT_CORE_DISPATCH_SIGNATURE_INFER_H
#define MARMOT_CORE_DISPATCH_SIGNATURE_INFER_H

#include "marmot/graph/graph_types.h"
#include "marmot/graph/op_signature.h"

#ifdef __cplusplus
extern "C" {
#endif

bool marmot_signature_infer_elementwise(
    const marmot_graph_tensor_desc_t *input_desc, const marmot_graph_tensor_desc_t *weight_desc,
    const marmot_graph_tensor_desc_t *output_desc, marmot_op_signature_t *sig
);

bool marmot_signature_infer_matmul(
    const marmot_graph_tensor_desc_t *input_desc, const marmot_graph_tensor_desc_t *weight_desc,
    const marmot_graph_tensor_desc_t *output_desc, marmot_op_signature_t *sig
);

bool marmot_signature_infer_paged_attention(
    const marmot_graph_tensor_desc_t *q_desc, const marmot_graph_tensor_desc_t *kv_desc,
    const marmot_graph_tensor_desc_t *output_desc, marmot_op_signature_t *sig
);

bool marmot_signature_infer_passthrough(
    const marmot_graph_tensor_desc_t *input_desc, const marmot_graph_tensor_desc_t *output_desc,
    marmot_op_signature_t *sig
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_DISPATCH_SIGNATURE_INFER_H
