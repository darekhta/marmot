#include "signature_infer.h"

#include "signature_utils.h"

static uint32_t marmot_graph_num_elements(const marmot_graph_tensor_desc_t *desc) {
    if (desc == nullptr || desc->ndim == 0) {
        return 0;
    }
    size_t n_elems = 1;
    for (uint32_t i = 0; i < desc->ndim; ++i) {
        n_elems *= desc->shape[i];
    }
    return (uint32_t)n_elems;
}

bool marmot_signature_infer_elementwise(
    const marmot_graph_tensor_desc_t *input_desc, const marmot_graph_tensor_desc_t *weight_desc,
    const marmot_graph_tensor_desc_t *output_desc, marmot_op_signature_t *sig
) {
    if (input_desc == nullptr || output_desc == nullptr || sig == nullptr) {
        return false;
    }

    if (sig->input_dtype >= MARMOT_DTYPE_COUNT) {
        sig->input_dtype = input_desc->dtype;
    }
    if (sig->output_dtype >= MARMOT_DTYPE_COUNT) {
        sig->output_dtype = output_desc->dtype;
    }
    if (sig->accum_dtype >= MARMOT_DTYPE_COUNT) {
        sig->accum_dtype = marmot_elementwise_accum_dtype(input_desc->dtype);
    }

    if (weight_desc != nullptr) {
        if (sig->weight_dtype >= MARMOT_DTYPE_COUNT) {
            sig->weight_dtype = weight_desc->dtype;
        }
    } else {
        if (sig->weight_dtype >= MARMOT_DTYPE_COUNT) {
            sig->weight_dtype = input_desc->dtype;
        }
    }

    sig->dims.elementwise.n_elems = marmot_graph_num_elements(input_desc);
    return true;
}

bool marmot_signature_infer_matmul(
    const marmot_graph_tensor_desc_t *input_desc, const marmot_graph_tensor_desc_t *weight_desc,
    const marmot_graph_tensor_desc_t *output_desc, marmot_op_signature_t *sig
) {
    if (input_desc == nullptr || weight_desc == nullptr || output_desc == nullptr || sig == nullptr) {
        return false;
    }
    if (input_desc->ndim < 2 || weight_desc->ndim < 2 || output_desc->ndim < 2) {
        return false;
    }

    const size_t input_rows = input_desc->shape[0];
    const size_t input_k = input_desc->shape[1];
    const size_t weight_m = weight_desc->shape[0];
    const size_t weight_k = weight_desc->shape[1];
    const size_t output_rows = output_desc->shape[0];
    const size_t output_cols = output_desc->shape[1];

    bool use_nt_layout = false;
    if (input_rows == output_rows && weight_m == output_cols && input_k == weight_k) {
        use_nt_layout = true;
    } else if (input_rows == output_rows && weight_m == input_k && weight_k == output_cols) {
        use_nt_layout = false;
    } else {
        return false;
    }

    if (sig->input_dtype >= MARMOT_DTYPE_COUNT) {
        sig->input_dtype = input_desc->dtype;
    }
    if (sig->weight_dtype >= MARMOT_DTYPE_COUNT) {
        sig->weight_dtype = weight_desc->dtype;
    }
    if (sig->output_dtype >= MARMOT_DTYPE_COUNT) {
        sig->output_dtype = output_desc->dtype;
    }
    if (sig->accum_dtype >= MARMOT_DTYPE_COUNT) {
        sig->accum_dtype = marmot_matmul_accum_dtype(input_desc->dtype);
    }
    if (sig->matmul_layout == MARMOT_MATMUL_LAYOUT_INVALID) {
        sig->matmul_layout = use_nt_layout ? MARMOT_MATMUL_LAYOUT_NT : MARMOT_MATMUL_LAYOUT_NN;
    }
    if (sig->weight_layout == MARMOT_WEIGHT_LAYOUT_INVALID) {
        sig->weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE;
    }

    sig->dims.matmul.N = (uint32_t)input_rows;
    sig->dims.matmul.K = (uint32_t)input_k;
    sig->dims.matmul.M = (uint32_t)output_cols;
    return true;
}

bool marmot_signature_infer_paged_attention(
    const marmot_graph_tensor_desc_t *q_desc, const marmot_graph_tensor_desc_t *kv_desc,
    const marmot_graph_tensor_desc_t *output_desc, marmot_op_signature_t *sig
) {
    if (q_desc == nullptr || kv_desc == nullptr || output_desc == nullptr || sig == nullptr) {
        return false;
    }

    if (sig->input_dtype >= MARMOT_DTYPE_COUNT) {
        sig->input_dtype = q_desc->dtype;
    }
    if (sig->weight_dtype >= MARMOT_DTYPE_COUNT) {
        sig->weight_dtype = kv_desc->dtype;
    }
    if (sig->output_dtype >= MARMOT_DTYPE_COUNT) {
        sig->output_dtype = output_desc->dtype;
    }
    if (sig->accum_dtype >= MARMOT_DTYPE_COUNT) {
        sig->accum_dtype = marmot_elementwise_accum_dtype(q_desc->dtype);
    }

    sig->dims.elementwise.n_elems = marmot_graph_num_elements(q_desc);
    return true;
}

bool marmot_signature_infer_passthrough(
    const marmot_graph_tensor_desc_t *input_desc, const marmot_graph_tensor_desc_t *output_desc,
    marmot_op_signature_t *sig
) {
    if (input_desc == nullptr || output_desc == nullptr || sig == nullptr) {
        return false;
    }

    if (sig->input_dtype >= MARMOT_DTYPE_COUNT) {
        sig->input_dtype = input_desc->dtype;
    }
    if (sig->weight_dtype >= MARMOT_DTYPE_COUNT) {
        sig->weight_dtype = input_desc->dtype;
    }
    if (sig->output_dtype >= MARMOT_DTYPE_COUNT) {
        sig->output_dtype = output_desc->dtype;
    }
    if (sig->accum_dtype >= MARMOT_DTYPE_COUNT) {
        sig->accum_dtype = marmot_elementwise_accum_dtype(input_desc->dtype);
    }

    return true;
}
