#include "marmot/ops_utils.h"

#include "marmot/error.h"

#include <string.h>

static void marmot_shape_clear(marmot_shape_t *shape) {
    if (shape == nullptr) {
        return;
    }
    memset(shape, 0, sizeof(*shape));
}

static void marmot_shape_set_contiguous(marmot_shape_t *shape) {
    if (shape == nullptr || shape->ndim == 0) {
        return;
    }
    shape->strides[shape->ndim - 1] = 1;
    for (size_t i = shape->ndim - 1; i > 0; --i) {
        shape->strides[i - 1] = shape->strides[i] * shape->shape[i];
    }
}

static bool marmot_shapes_equal(const marmot_shape_t *lhs, const marmot_shape_t *rhs) {
    if (lhs == nullptr || rhs == nullptr) {
        return false;
    }
    if (lhs->ndim != rhs->ndim) {
        return false;
    }
    for (size_t i = 0; i < lhs->ndim; ++i) {
        if (lhs->shape[i] != rhs->shape[i]) {
            return false;
        }
    }
    return true;
}

static marmot_error_t marmot_shape_copy(const marmot_shape_t *input, marmot_shape_t *out) {
    if (input == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null shape pointer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->ndim > MARMOT_MAX_DIMS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Shape rank exceeds MARMOT_MAX_DIMS");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_shape_clear(out);
    out->ndim = input->ndim;
    if (input->ndim > 0) {
        memcpy(out->shape, input->shape, input->ndim * sizeof(size_t));
    }
    marmot_shape_set_contiguous(out);
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_infer_unary_output_shape(const marmot_shape_t *input, marmot_shape_t *out) {
    return marmot_shape_copy(input, out);
}

marmot_error_t
marmot_infer_binary_output_shape(const marmot_shape_t *lhs, const marmot_shape_t *rhs, marmot_shape_t *out) {
    if (lhs == nullptr || rhs == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null shape pointer for binary inference");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!marmot_shapes_equal(lhs, rhs)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Binary elementwise shapes must match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    return marmot_shape_copy(lhs, out);
}

marmot_error_t marmot_infer_ternary_output_shape(
    const marmot_shape_t *a, const marmot_shape_t *b, const marmot_shape_t *c, marmot_shape_t *out
) {
    if (a == nullptr || b == nullptr || c == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null shape pointer for ternary inference");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!marmot_shapes_equal(a, b) || !marmot_shapes_equal(a, c)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Ternary elementwise shapes must match");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    return marmot_shape_copy(a, out);
}

marmot_error_t
marmot_infer_linear_output_shape(const marmot_shape_t *input, const marmot_shape_t *weight, marmot_shape_t *out) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null shape pointer for linear inference");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->ndim != 2 || weight->ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Linear inference requires 2D input and weight shapes");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (input->shape[1] != weight->shape[1]) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Linear K dimension mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    marmot_shape_clear(out);
    out->ndim = 2;
    out->shape[0] = input->shape[0];
    out->shape[1] = weight->shape[0];
    marmot_shape_set_contiguous(out);
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_infer_matmul_output_shape(const marmot_shape_t *a, const marmot_shape_t *b, marmot_shape_t *out) {
    if (a == nullptr || b == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null shape pointer for matmul inference");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (a->ndim > MARMOT_MAX_DIMS || b->ndim > MARMOT_MAX_DIMS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul input rank exceeds MARMOT_MAX_DIMS");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (a->ndim == 0 || b->ndim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul inference requires non-empty shapes");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t a_nd = a->ndim;
    const size_t b_nd = b->ndim;
    const bool a_vec = (a_nd == 1);
    const bool b_vec = (b_nd == 1);

    size_t a_batch_dims = a_nd > 2 ? a_nd - 2 : 0;
    size_t b_batch_dims = b_nd > 2 ? b_nd - 2 : 0;
    if (a_vec) {
        a_batch_dims = 0;
    }
    if (b_vec) {
        b_batch_dims = 0;
    }

    size_t batch_rank = a_batch_dims > b_batch_dims ? a_batch_dims : b_batch_dims;
    const size_t a_batch_offset = batch_rank > a_batch_dims ? batch_rank - a_batch_dims : 0;
    const size_t b_batch_offset = batch_rank > b_batch_dims ? batch_rank - b_batch_dims : 0;

    size_t batch_dims[MARMOT_MAX_DIMS] = {0};
    for (size_t i = 0; i < batch_rank; ++i) {
        const size_t a_dim = (i < a_batch_offset) ? 1 : a->shape[i - a_batch_offset];
        const size_t b_dim = (i < b_batch_offset) ? 1 : b->shape[i - b_batch_offset];
        if ((a_dim != 1 && b_dim != 1 && a_dim != b_dim)) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul batch dimensions are not broadcastable");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        batch_dims[i] = (a_dim > b_dim) ? a_dim : b_dim;
    }

    size_t M = 0;
    size_t K = 0;
    size_t N = 0;
    if (a_vec && b_vec) {
        K = a->shape[0];
        if (K != b->shape[0]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul vector dimensions mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        M = 1;
        N = 1;
    } else if (a_vec) {
        if (b_nd < 2) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul expects 2D B for vector @ matrix");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        K = a->shape[0];
        N = b->shape[b_nd - 1];
        if (K != b->shape[b_nd - 2]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul K dimension mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        M = 1;
    } else if (b_vec) {
        if (a_nd < 2) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul expects 2D A for matrix @ vector");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        M = a->shape[a_nd - 2];
        K = a->shape[a_nd - 1];
        if (K != b->shape[0]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul K dimension mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        N = 1;
    } else {
        if (a_nd < 2 || b_nd < 2) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul expects rank >= 2 inputs");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        M = a->shape[a_nd - 2];
        K = a->shape[a_nd - 1];
        N = b->shape[b_nd - 1];
        if (K != b->shape[b_nd - 2]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul K dimension mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    size_t trailing_nd = 2;
    size_t trailing_dims[2] = {M, N};
    if (a_vec && b_vec && M == 1 && N == 1) {
        trailing_nd = 1;
        trailing_dims[0] = 1;
    } else if (M == 1) {
        trailing_nd = 1;
        trailing_dims[0] = N;
    } else if (N == 1) {
        trailing_nd = 1;
        trailing_dims[0] = M;
    }

    const size_t out_nd = batch_rank + trailing_nd;
    if (out_nd > MARMOT_MAX_DIMS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul output rank exceeds MARMOT_MAX_DIMS");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_shape_clear(out);
    out->ndim = out_nd;
    for (size_t i = 0; i < batch_rank; ++i) {
        out->shape[i] = batch_dims[i];
    }
    for (size_t i = 0; i < trailing_nd; ++i) {
        out->shape[batch_rank + i] = trailing_dims[i];
    }
    marmot_shape_set_contiguous(out);
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_infer_matmul_qkv_output_shape(
    const marmot_shape_t *input, marmot_matmul_qkv_layout_t layout, const marmot_shape_t *fused_weight,
    const marmot_shape_t *wq, const marmot_shape_t *wk, const marmot_shape_t *wv, marmot_shape_t *out_q,
    marmot_shape_t *out_k, marmot_shape_t *out_v
) {
    if (input == nullptr || out_q == nullptr || out_k == nullptr || out_v == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null shape pointer for QKV inference");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV input must be 2D");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const size_t N = input->shape[0];
    const size_t K = input->shape[1];
    size_t M = 0;

    switch (layout) {
    case MARMOT_QKV_LAYOUT_FUSED:
        if (fused_weight == nullptr || fused_weight->ndim != 2) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fused QKV weight must be 2D");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (fused_weight->shape[0] % 3 != 0) {
            marmot_set_error(
                MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight first dimension must be divisible by 3"
            );
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (fused_weight->shape[1] != K) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight K dimension mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        M = fused_weight->shape[0] / 3;
        break;
    case MARMOT_QKV_LAYOUT_SEPARATE:
        if (wq == nullptr || wk == nullptr || wv == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Separate QKV weights are required");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (wq->ndim != 2 || wk->ndim != 2 || wv->ndim != 2) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must be 2D");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (wq->shape[1] != K || wk->shape[1] != K || wv->shape[1] != K) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weight K dimension mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (wq->shape[0] != wk->shape[0] || wq->shape[0] != wv->shape[0]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must share output dimension");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        M = wq->shape[0];
        break;
    default:
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "QKV layout must be fused or separate");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_shape_clear(out_q);
    marmot_shape_clear(out_k);
    marmot_shape_clear(out_v);
    out_q->ndim = 2;
    out_k->ndim = 2;
    out_v->ndim = 2;
    out_q->shape[0] = N;
    out_k->shape[0] = N;
    out_v->shape[0] = N;
    out_q->shape[1] = M;
    out_k->shape[1] = M;
    out_v->shape[1] = M;
    marmot_shape_set_contiguous(out_q);
    marmot_shape_set_contiguous(out_k);
    marmot_shape_set_contiguous(out_v);
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_infer_embedding_output_shape(
    const marmot_shape_t *weights, const marmot_shape_t *token_ids, marmot_shape_t *out
) {
    if (weights == nullptr || token_ids == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null shape pointer for embedding inference");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (weights->ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Embedding weights must be 2D");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (token_ids->ndim == 0) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Embedding token_ids must be rank >= 1");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (token_ids->ndim + 1 > MARMOT_MAX_DIMS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding output rank exceeds MARMOT_MAX_DIMS");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const size_t dim = weights->shape[1];
    if (dim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding dimension must be non-zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_shape_clear(out);
    out->ndim = token_ids->ndim + 1;
    for (size_t i = 0; i < token_ids->ndim; ++i) {
        out->shape[i] = token_ids->shape[i];
    }
    out->shape[out->ndim - 1] = dim;
    marmot_shape_set_contiguous(out);
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_infer_reduction_output_shape(
    const marmot_shape_t *input, const int32_t *axes, size_t num_axes, bool keepdims, marmot_shape_t *out
) {
    if (input == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null shape pointer for reduction inference");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->ndim > MARMOT_MAX_DIMS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Input rank exceeds MARMOT_MAX_DIMS");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t ndim = input->ndim;
    bool reduce_axes[MARMOT_MAX_DIMS] = {0};
    size_t reduced = 0;

    if (axes == nullptr || num_axes == 0) {
        for (size_t i = 0; i < ndim; ++i) {
            reduce_axes[i] = true;
        }
        reduced = ndim;
    } else {
        for (size_t i = 0; i < num_axes; ++i) {
            int32_t axis = axes[i];
            if (axis < 0) {
                axis += (int32_t)ndim;
            }
            if (axis < 0 || axis >= (int32_t)ndim) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reduction axis out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            if (reduce_axes[axis]) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Duplicate reduction axis");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            reduce_axes[axis] = true;
            reduced += 1;
        }
    }

    marmot_shape_clear(out);
    if (keepdims) {
        out->ndim = ndim;
        for (size_t i = 0; i < ndim; ++i) {
            out->shape[i] = reduce_axes[i] ? 1 : input->shape[i];
        }
    } else {
        if (reduced > ndim) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reduction axes exceed input rank");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        out->ndim = ndim - reduced;
        size_t out_i = 0;
        for (size_t i = 0; i < ndim; ++i) {
            if (!reduce_axes[i]) {
                out->shape[out_i++] = input->shape[i];
            }
        }
    }
    marmot_shape_set_contiguous(out);
    return MARMOT_SUCCESS;
}
