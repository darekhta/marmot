#include "marmot/error.h"
#include "marmot/graph/op_signature.h"
#include "marmot/ops/manipulation.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/rope.h"
marmot_error_t cpu_matmul_quant_prepack(const void *device_ctx, const marmot_tensor_t *weight);
marmot_error_t
cpu_matmul_quant_pin_range(const void *device_ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows);

#include "core/dispatch/dispatch_build.h"
#include "core/dispatch/dispatch_execute.h"
#include "core/helpers/quant.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static bool matmul_op_requires_bias(marmot_op_id_t op_id) {
    switch (op_id) {
    case MARMOT_OP_MATMUL_BIAS:
    case MARMOT_OP_MATMUL_BIAS_RELU:
    case MARMOT_OP_MATMUL_BIAS_GELU:
    case MARMOT_OP_MATMUL_BIAS_SILU:
        return true;
    default:
        return false;
    }
}

static marmot_error_t matmul_validate_epilogue(marmot_op_id_t op_id, const marmot_matmul_epilogue_t *epilogue) {
    const bool needs_bias = matmul_op_requires_bias(op_id);
    const bool has_bias = epilogue != nullptr && epilogue->bias != nullptr;
    if (needs_bias && !has_bias) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul bias operation requires a bias epilogue");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!needs_bias && has_bias) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul operation does not accept a bias epilogue");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

typedef struct {
    size_t batch_rank;
    size_t broadcast_dims[MARMOT_MAX_DIMS];
    size_t a_batch_strides[MARMOT_MAX_DIMS];
    size_t b_batch_strides[MARMOT_MAX_DIMS];
    size_t out_batch_strides[MARMOT_MAX_DIMS];
    size_t batch_count;
    size_t M;
    size_t N;
    size_t K;
    bool is_batched;
} marmot_batched_matmul_plan_t;

static marmot_error_t marmot_matmul_linear_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, marmot_op_id_t op_id
) {
    if (ctx == nullptr || input == nullptr || weight == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to matmul");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const bool input_quantized = marmot_tensor_is_block_quantized_weight(input);
    const bool weight_quantized = marmot_tensor_is_block_quantized_weight(weight);

    if (input_quantized) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized inputs/activations are not supported for matmul yet");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

#if MARMOT_ENABLE_FP8
    const bool input_is_fp8 = input->dtype == MARMOT_DTYPE_FLOAT8_E4M3 || input->dtype == MARMOT_DTYPE_FLOAT8_E5M2;
    if (input_is_fp8) {
        if (weight->dtype != input->dtype) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 matmul requires matching input and weight dtype");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (out->dtype != MARMOT_DTYPE_FLOAT32) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 matmul requires FLOAT32 output");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
    }
#endif

    if (weight_quantized) {
        if (weight->quant_kind == MARMOT_QUANT_KIND_GENERIC) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Generic quantized matmul not available");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
    }

    marmot_error_t epilogue_status = matmul_validate_epilogue(op_id, epilogue);
    if (epilogue_status != MARMOT_SUCCESS) {
        return epilogue_status;
    }

    marmot_qscheme_id_t qscheme_id =
        weight_quantized ? marmot_quant_kind_to_qscheme(weight->quant_kind) : MARMOT_QSCHEME_NONE;
    marmot_weight_layout_t weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;

    marmot_op_signature_t sig = {0};
    marmot_kernel_args_matmul_t packed = {0};
    marmot_error_t build_status = marmot_matmul_build(
        ctx, MARMOT_MATMUL_LAYOUT_NT, op_id, input, weight, epilogue, out, qscheme_id, weight_layout, &sig, &packed
    );
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }
    return marmot_execute_signature(ctx, &sig, &packed, "Matmul");
}

marmot_error_t marmot_linear_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    marmot_op_id_t op_id = MARMOT_OP_MATMUL;
    if (epilogue != nullptr && epilogue->bias != nullptr) {
        op_id = MARMOT_OP_MATMUL_BIAS;
    }
    return marmot_matmul_linear_impl(ctx, input, weight, epilogue, out, op_id);
}

static bool marmot_matmul_compute_broadcast_plan(
    const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *out, marmot_batched_matmul_plan_t *plan
) {
    if (a == nullptr || b == nullptr || out == nullptr || plan == nullptr) {
        return false;
    }

    const size_t a_nd = marmot_tensor_ndim(a);
    const size_t b_nd = marmot_tensor_ndim(b);
    const size_t out_nd = marmot_tensor_ndim(out);

    const bool a_vec = (a_nd == 1);
    const bool b_vec = (b_nd == 1);

    size_t a_batch_dims = a_nd > 2 ? a_nd - 2 : 0;
    size_t b_batch_dims = b_nd > 2 ? b_nd - 2 : 0;
    size_t out_batch_dims = out_nd > 2 ? out_nd - 2 : 0;

    if (a_vec) {
        a_batch_dims = 0;
    }
    if (b_vec) {
        b_batch_dims = 0;
    }

    size_t batch_rank = a_batch_dims > b_batch_dims ? a_batch_dims : b_batch_dims;
    if (out_batch_dims > batch_rank) {
        batch_rank = out_batch_dims;
    }
    plan->batch_rank = batch_rank;
    plan->is_batched = batch_rank > 0;
    plan->batch_count = 1;

    const size_t a_batch_offset = batch_rank > a_batch_dims ? batch_rank - a_batch_dims : 0;
    const size_t b_batch_offset = batch_rank > b_batch_dims ? batch_rank - b_batch_dims : 0;
    const size_t out_batch_offset = batch_rank > out_batch_dims ? batch_rank - out_batch_dims : 0;

    const size_t a_dtype_size = marmot_dtype_size(a->dtype);
    const size_t b_dtype_size = marmot_dtype_size(b->dtype);
    const size_t out_dtype_size = marmot_dtype_size(out->dtype);

    for (size_t i = 0; i < batch_rank; ++i) {
        const size_t a_dim = (i < a_batch_offset) ? 1 : a->shape.shape[i - a_batch_offset];
        const size_t b_dim = (i < b_batch_offset) ? 1 : b->shape.shape[i - b_batch_offset];
        const size_t out_dim = (i < out_batch_offset) ? 1 : out->shape.shape[i - out_batch_offset];
        const size_t broadcast_dim = a_dim > b_dim ? a_dim : b_dim;

        if ((a_dim != 1 && b_dim != 1 && a_dim != b_dim) || out_dim != broadcast_dim) {
            return false;
        }

        plan->broadcast_dims[i] = broadcast_dim;
        plan->a_batch_strides[i] = (a_dim == 1) ? 0 : a->shape.strides[i - a_batch_offset] * a_dtype_size;
        plan->b_batch_strides[i] = (b_dim == 1) ? 0 : b->shape.strides[i - b_batch_offset] * b_dtype_size;
        plan->out_batch_strides[i] = (out_dim == 1) ? 0 : out->shape.strides[i - out_batch_offset] * out_dtype_size;

        plan->batch_count *= broadcast_dim;
    }

    // Determine matmul core dims (M, K, N) accounting for vector cases
    if (a_vec && b_vec) {
        plan->M = 1;
        plan->K = marmot_tensor_shape_at(a, 0);
        plan->N = 1;
        if (plan->K != marmot_tensor_shape_at(b, 0)) {
            return false;
        }
    } else if (a_vec) {
        if (b_nd < 2) {
            return false;
        }
        plan->M = 1;
        plan->K = marmot_tensor_shape_at(a, 0);
        plan->N = marmot_tensor_shape_at(b, b_nd - 1);
        if (plan->K != marmot_tensor_shape_at(b, b_nd - 2)) {
            return false;
        }
    } else if (b_vec) {
        if (a_nd < 2) {
            return false;
        }
        plan->M = marmot_tensor_shape_at(a, a_nd - 2);
        plan->K = marmot_tensor_shape_at(a, a_nd - 1);
        plan->N = 1;
        if (plan->K != marmot_tensor_shape_at(b, 0)) {
            return false;
        }
    } else {
        if (a_nd < 2 || b_nd < 2) {
            return false;
        }
        plan->M = marmot_tensor_shape_at(a, a_nd - 2);
        plan->K = marmot_tensor_shape_at(a, a_nd - 1);
        plan->N = marmot_tensor_shape_at(b, b_nd - 1);
        if (plan->K != marmot_tensor_shape_at(b, b_nd - 2)) {
            return false;
        }
    }
    return true;
}

static marmot_error_t marmot_matmul_nn_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, marmot_op_id_t op_id
) {
    if (ctx == nullptr || input == nullptr || weight == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to matmul");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (marmot_tensor_is_block_quantized_weight(input) || marmot_tensor_is_block_quantized_weight(weight)) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized torch-style matmul not available");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_error_t epilogue_status = matmul_validate_epilogue(op_id, epilogue);
    if (epilogue_status != MARMOT_SUCCESS) {
        return epilogue_status;
    }

    marmot_op_signature_t sig = {0};
    marmot_kernel_args_matmul_t packed = {0};
    marmot_error_t build_status = marmot_matmul_build(
        ctx, MARMOT_MATMUL_LAYOUT_NN, op_id, input, weight, epilogue, out, MARMOT_QSCHEME_NONE,
        MARMOT_WEIGHT_LAYOUT_INVALID, &sig, &packed
    );
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "Matmul");
}

static marmot_error_t marmot_matmul_impl_with_op(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, marmot_op_id_t op_id
) {
    if (ctx == nullptr || a == nullptr || b == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to matmul");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_error_t epilogue_status = matmul_validate_epilogue(op_id, epilogue);
    if (epilogue_status != MARMOT_SUCCESS) {
        return epilogue_status;
    }

    if (marmot_tensor_is_block_quantized_weight(b)) {
        marmot_set_error(
            MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized weights are not supported for torch-style matmul yet"
        );
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (a->dtype != b->dtype || a->dtype != out->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Matmul tensors must share dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    marmot_batched_matmul_plan_t plan = {0};
    if (!marmot_matmul_compute_broadcast_plan(a, b, out, &plan)) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Matmul tensor shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    const size_t out_nd = marmot_tensor_ndim(out);
    size_t expected_trailing_nd = 2;
    size_t expected_trailing[2] = {plan.M, plan.N};
    if (plan.N == 1 && plan.M == 1 && marmot_tensor_ndim(a) == 1 && marmot_tensor_ndim(b) == 1) {
        expected_trailing_nd = 1;
        expected_trailing[0] = 1;
    } else if (plan.M == 1) {
        expected_trailing_nd = 1;
        expected_trailing[0] = plan.N;
    } else if (plan.N == 1) {
        expected_trailing_nd = 1;
        expected_trailing[0] = plan.M;
    }

    const size_t expected_out_nd = plan.batch_rank + expected_trailing_nd;
    if (out_nd != expected_out_nd) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Output rank mismatch for matmul");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < plan.batch_rank; ++i) {
        const size_t out_dim = out->shape.shape[i];
        if (out_dim != plan.broadcast_dims[i]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Output batch dimension mismatch for matmul");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }
    for (size_t i = 0; i < expected_trailing_nd; ++i) {
        if (out->shape.shape[plan.batch_rank + i] != expected_trailing[i]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Output trailing dimension mismatch for matmul");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }
    if (plan.batch_count == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid batch configuration for matmul");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const char *a_bytes = (const char *)a->data;
    const char *b_bytes = (const char *)b->data;
    char *out_bytes = (char *)out->data;
    const size_t elem_size = marmot_dtype_size(a->dtype);

    // For Metal backend, we need to allocate temporary tensors because
    // Metal's newBufferWithBytesNoCopy requires page-aligned pointers.
    // Pointer-offset views into the middle of a tensor don't work.
    const bool needs_temp_tensors = (ctx != nullptr && ctx->policy.matmul_requires_temp_tensors);

    size_t a_slice_shape[2] = {plan.M, plan.K};
    size_t b_slice_shape[2] = {plan.K, plan.N};
    size_t out_slice_shape[2] = {plan.M, plan.N};

    marmot_tensor_t *a_temp = nullptr;
    marmot_tensor_t *b_temp = nullptr;
    marmot_tensor_t *out_temp = nullptr;

    if (needs_temp_tensors) {
        a_temp = marmot_tensor_create(ctx, a_slice_shape, 2, a->dtype);
        b_temp = marmot_tensor_create(ctx, b_slice_shape, 2, b->dtype);
        out_temp = marmot_tensor_create(ctx, out_slice_shape, 2, out->dtype);
        if (a_temp == nullptr || b_temp == nullptr || out_temp == nullptr) {
            if (a_temp != nullptr)
                marmot_tensor_destroy(a_temp);
            if (b_temp != nullptr)
                marmot_tensor_destroy(b_temp);
            if (out_temp != nullptr)
                marmot_tensor_destroy(out_temp);
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate temporary tensors for batched matmul");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    marmot_error_t final_status = MARMOT_SUCCESS;

    for (size_t batch = 0; batch < plan.batch_count; ++batch) {
        size_t tmp = batch;
        size_t a_offset = 0;
        size_t b_offset = 0;
        size_t out_offset = 0;
        for (size_t dim = plan.batch_rank; dim-- > 0;) {
            const size_t dim_size = plan.broadcast_dims[dim];
            const size_t idx = dim_size > 0 ? (tmp % dim_size) : 0;
            tmp /= (dim_size == 0 ? 1 : dim_size);
            a_offset += idx * plan.a_batch_strides[dim];
            b_offset += idx * plan.b_batch_strides[dim];
            out_offset += idx * plan.out_batch_strides[dim];
        }

        const marmot_tensor_t *a_slice = nullptr;
        const marmot_tensor_t *b_slice = nullptr;
        marmot_tensor_t *out_slice = nullptr;

        marmot_tensor_t a_view;
        marmot_tensor_t b_view;
        marmot_tensor_t out_view;

        if (needs_temp_tensors) {
            // Copy slices to temporary tensors using proper device API
            const size_t a_slice_bytes = plan.M * plan.K * elem_size;
            const size_t b_slice_bytes = plan.K * plan.N * elem_size;
            marmot_error_t copy_err =
                marmot_tensor_copy_from_host_buffer(ctx, a_temp, a_bytes + a_offset, a_slice_bytes);
            if (copy_err != MARMOT_SUCCESS) {
                final_status = copy_err;
                break;
            }
            copy_err = marmot_tensor_copy_from_host_buffer(ctx, b_temp, b_bytes + b_offset, b_slice_bytes);
            if (copy_err != MARMOT_SUCCESS) {
                final_status = copy_err;
                break;
            }
            a_slice = a_temp;
            b_slice = b_temp;
            out_slice = out_temp;
        } else {
            // Use views with offset pointers (works for CPU)
            a_view = *a;
            b_view = *b;
            out_view = *out;

            a_view.data = (void *)(a_bytes + a_offset);
            b_view.data = (void *)(b_bytes + b_offset);
            out_view.data = (void *)(out_bytes + out_offset);

            a_view.shape.ndim = 2;
            a_view.shape.shape[0] = plan.M;
            a_view.shape.shape[1] = plan.K;
            a_view.shape.strides[0] = plan.K;
            a_view.shape.strides[1] = 1;

            b_view.shape.ndim = 2;
            b_view.shape.shape[0] = plan.K;
            b_view.shape.shape[1] = plan.N;
            b_view.shape.strides[0] = plan.N;
            b_view.shape.strides[1] = 1;

            out_view.shape.ndim = 2;
            out_view.shape.shape[0] = plan.M;
            out_view.shape.shape[1] = plan.N;
            out_view.shape.strides[0] = plan.N;
            out_view.shape.strides[1] = 1;

            a_slice = &a_view;
            b_slice = &b_view;
            out_slice = &out_view;
        }

        marmot_error_t status = marmot_matmul_nn_impl(ctx, a_slice, b_slice, epilogue, out_slice, op_id);
        if (status == MARMOT_ERROR_NOT_IMPLEMENTED) {
            size_t bt_shape[2] = {plan.N, plan.K};
            marmot_tensor_t *bt = marmot_tensor_create(ctx, bt_shape, 2, b_slice->dtype);
            if (bt == nullptr) {
                final_status = marmot_get_last_error();
                break;
            }
            int perm[2] = {1, 0};
            marmot_error_t transpose_status = marmot_transpose(ctx, b_slice, bt, perm);
            if (transpose_status == MARMOT_SUCCESS) {
                status = marmot_matmul_linear_impl(ctx, a_slice, bt, epilogue, out_slice, op_id);
            } else {
                status = transpose_status;
            }
            marmot_tensor_destroy(bt);
        }

        if (needs_temp_tensors && status == MARMOT_SUCCESS) {
            // Sync GPU and copy result back to output tensor using device API
            const size_t out_slice_bytes = plan.M * plan.N * elem_size;
            marmot_error_t copy_err =
                marmot_tensor_copy_to_host_buffer(ctx, out_temp, out_bytes + out_offset, out_slice_bytes);
            if (copy_err != MARMOT_SUCCESS) {
                final_status = copy_err;
                break;
            }
            // Invalidate residency on the output tensor's slice
            if (ctx->ops != nullptr && ctx->ops->memcpy_to_device != nullptr) {
                (void)ctx->ops->memcpy_to_device(
                    ctx->device_ctx, out_bytes + out_offset, out_bytes + out_offset, out_slice_bytes
                );
            }
        }

        if (status != MARMOT_SUCCESS) {
            final_status = status;
            break;
        }
    }

    if (needs_temp_tensors) {
        marmot_tensor_destroy(a_temp);
        marmot_tensor_destroy(b_temp);
        marmot_tensor_destroy(out_temp);
    }

    return final_status;
}

marmot_error_t marmot_matmul_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    return marmot_matmul_impl_with_op(ctx, a, b, epilogue, out, MARMOT_OP_MATMUL);
}

marmot_error_t marmot_matmul_bias_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    return marmot_matmul_impl_with_op(ctx, a, b, epilogue, out, MARMOT_OP_MATMUL_BIAS);
}

marmot_error_t marmot_matmul_bias_relu_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    return marmot_matmul_impl_with_op(ctx, a, b, epilogue, out, MARMOT_OP_MATMUL_BIAS_RELU);
}

marmot_error_t marmot_matmul_bias_gelu_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    return marmot_matmul_impl_with_op(ctx, a, b, epilogue, out, MARMOT_OP_MATMUL_BIAS_GELU);
}

marmot_error_t marmot_matmul_bias_silu_impl(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
) {
    return marmot_matmul_impl_with_op(ctx, a, b, epilogue, out, MARMOT_OP_MATMUL_BIAS_SILU);
}

marmot_error_t marmot_matmul_prepack_quant_weight_impl(const marmot_context_t *ctx, const marmot_tensor_t *weight) {
    if (ctx == nullptr || weight == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to matmul prepack");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!marmot_quant_kind_is_block_quantized(weight->quant_kind)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Prepack only supports block-quantized weights");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ctx->backend_type == MARMOT_BACKEND_CPU) {
        return cpu_matmul_quant_prepack(ctx->device_ctx, weight);
    }
    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Matmul prepack not available for this backend");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t marmot_cpu_pin_quant_weight_range(
    const marmot_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows
) {
    if (ctx == nullptr || src == nullptr || bytes == 0 || row_bytes == 0 || rows == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to CPU quant weight pin");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ctx->backend_type != MARMOT_BACKEND_CPU) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "CPU quant weight pin only supports the CPU backend");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return cpu_matmul_quant_pin_range(ctx->device_ctx, src, bytes, row_bytes, rows);
}

static bool marmot_matmul_qkv_desc_is_separate(const marmot_matmul_qkv_desc_t *desc) {
    return desc != nullptr && desc->layout == MARMOT_QKV_LAYOUT_SEPARATE;
}

static marmot_error_t marmot_matmul_qkv_validate_layout(const marmot_matmul_qkv_desc_t *desc) {
    if (desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    switch (desc->layout) {
    case MARMOT_QKV_LAYOUT_FUSED:
        if (desc->fused.weight == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fused QKV layout requires the fused weight tensor");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return MARMOT_SUCCESS;
    case MARMOT_QKV_LAYOUT_SEPARATE:
        if (desc->separate.wq == nullptr || desc->separate.wk == nullptr || desc->separate.wv == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Separate QKV layout requires wq, wk, and wv tensors");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        return MARMOT_SUCCESS;
    default:
        marmot_set_error(
            MARMOT_ERROR_INVALID_ARGUMENT,
            "Fused QKV descriptor must set layout to MARMOT_QKV_LAYOUT_FUSED or MARMOT_QKV_LAYOUT_SEPARATE"
        );
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
}

static marmot_error_t marmot_matmul_qkv_validate_rope(const marmot_matmul_qkv_desc_t *desc) {
    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null descriptor for fused QKV matmul");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_rope_params_t *rope = desc->rope_params;
    if (rope == nullptr) {
        return MARMOT_SUCCESS;
    }

    if (rope->positions == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE parameters require a positions tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!rope->apply_to_q && !rope->apply_to_k) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE parameters must enable Q and/or K rotation");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (rope->apply_to_q && desc->out_q == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE requested for Q but Q output tensor is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (rope->apply_to_k && desc->out_k == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE requested for K but K output tensor is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *input = desc->input;
    if (input == nullptr || input->shape.ndim == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE requires a valid input tensor to infer sequence rows");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const size_t rows = input->shape.shape[0];

    size_t dim = 0;
    if (desc->layout == MARMOT_QKV_LAYOUT_FUSED) {
        const marmot_tensor_t *weight = desc->fused.weight;
        if (weight == nullptr || weight->shape.ndim != 2 || weight->shape.shape[0] % 3 != 0) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE requires a valid fused weight tensor");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        dim = weight->shape.shape[0] / 3;
    } else {
        const marmot_tensor_t *target_weight = rope->apply_to_q ? desc->separate.wq : desc->separate.wk;
        if (target_weight == nullptr || target_weight->shape.ndim != 2) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE requires Q/K weights to infer head dimension");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        dim = target_weight->shape.shape[0];
    }

    if (dim == 0 || (dim & 1u) != 0u) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE requires an even head dimension");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    const marmot_tensor_t *positions = rope->positions;
    if (positions->dtype != MARMOT_DTYPE_INT32 && positions->dtype != MARMOT_DTYPE_INT64 &&
        positions->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "RoPE positions tensor must be INT32, INT64, or FLOAT32");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t pos_elements = marmot_tensor_num_elements(positions);
    if (pos_elements != rows) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "RoPE positions count must match Q/K rows");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t marmot_matmul_qkv_apply_rope(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc) {
    if (desc == nullptr || desc->rope_params == nullptr) {
        return MARMOT_SUCCESS;
    }

    const marmot_rope_params_t *rope = desc->rope_params;
    if (rope->apply_to_q) {
        marmot_error_t err = marmot_rope(ctx, desc->out_q, rope, desc->out_q);
        if (err != MARMOT_SUCCESS) {
            return err;
        }
    }

    if (rope->apply_to_k) {
        marmot_error_t err = marmot_rope(ctx, desc->out_k, rope, desc->out_k);
        if (err != MARMOT_SUCCESS) {
            return err;
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t marmot_matmul_qkv_fallback(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc) {
    if (ctx == nullptr || desc == nullptr || desc->input == nullptr || desc->out_q == nullptr ||
        desc->out_k == nullptr || desc->out_v == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_error_t layout_ok = marmot_matmul_qkv_validate_layout(desc);
    if (layout_ok != MARMOT_SUCCESS) {
        return layout_ok;
    }

    const bool uses_separate_weights = marmot_matmul_qkv_desc_is_separate(desc);
    const marmot_matmul_epilogue_t *desc_epilogue = desc->epilogue;
    const marmot_tensor_t *input = desc->input;
    if (input->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV input must be 2D");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    const size_t N = input->shape.shape[0];
    const size_t K = input->shape.shape[1];

    const marmot_tensor_t *weight_slices[3] = {nullptr, nullptr, nullptr};
    marmot_tensor_t weight_views[3] = {{0}};
    const marmot_tensor_t *bias_slices[3] = {nullptr, nullptr, nullptr};
    marmot_tensor_t bias_views[3] = {{0}};
    size_t M = 0;

    if (!uses_separate_weights) {
        const marmot_tensor_t *weight = desc->fused.weight;
        if (weight == nullptr || weight->shape.ndim != 2) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight tensor must be 2D");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (weight->shape.shape[0] % 3 != 0) {
            marmot_set_error(
                MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight first dimension must be divisible by 3"
            );
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (weight->shape.shape[1] != K) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight K dimension mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (weight->shape.strides[1] != 1) {
            marmot_set_error(
                MARMOT_ERROR_NOT_IMPLEMENTED, "Fused QKV weight tensor must be contiguous in the last axis"
            );
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        if (marmot_tensor_is_block_quantized_weight(weight)) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized fused QKV weights are not supported yet");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }

        const marmot_tensor_t *bias = desc->fused.bias;
        const bool has_bias = (bias != nullptr);
        if (has_bias) {
            if (bias->shape.ndim != 1 || bias->shape.shape[0] != weight->shape.shape[0]) {
                marmot_set_error(
                    MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV bias must be 1D and match weight first dimension"
                );
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            if (bias->shape.strides[0] != 1) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Fused QKV bias must be contiguous");
                return MARMOT_ERROR_NOT_IMPLEMENTED;
            }
        }

        M = weight->shape.shape[0] / 3;
        const size_t element_size = marmot_dtype_size(weight->dtype);
        const size_t row_stride = weight->shape.strides[0];
        const char *weight_bytes = (const char *)weight->data;

        const char *bias_bytes = has_bias ? (const char *)bias->data : nullptr;
        const size_t bias_stride = has_bias ? bias->shape.strides[0] : 0;
        const size_t bias_elt_size = has_bias ? marmot_dtype_size(bias->dtype) : 0;

        if (desc_epilogue != nullptr && desc_epilogue->bias != nullptr && has_bias) {
            marmot_set_error(
                MARMOT_ERROR_INVALID_ARGUMENT,
                "Fused QKV bias may be provided either via descriptor bias tensor or epilogue, not both"
            );
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        for (size_t slice = 0; slice < 3; ++slice) {
            weight_views[slice] = *weight;
            weight_views[slice].shape.shape[0] = M;
            weight_views[slice].shape.shape[1] = K;
            weight_views[slice].data = (void *)(weight_bytes + slice * M * row_stride * element_size);
            weight_slices[slice] = &weight_views[slice];
            if (has_bias) {
                bias_views[slice] = *bias;
                bias_views[slice].shape.shape[0] = M;
                bias_views[slice].data = (void *)(bias_bytes + slice * M * bias_stride * bias_elt_size);
                bias_slices[slice] = &bias_views[slice];
            }
        }
    } else {
        const marmot_tensor_t *wq = desc->separate.wq;
        const marmot_tensor_t *wk = desc->separate.wk;
        const marmot_tensor_t *wv = desc->separate.wv;
        if (wq->shape.ndim != 2 || wk->shape.ndim != 2 || wv->shape.ndim != 2) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must be 2D");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (wq->shape.shape[1] != K || wk->shape.shape[1] != K || wv->shape.shape[1] != K) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must share the input K dimension");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (wq->shape.shape[0] != wk->shape.shape[0] || wq->shape.shape[0] != wv->shape.shape[0]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must share the output dimension");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        M = wq->shape.shape[0];
        weight_slices[0] = wq;
        weight_slices[1] = wk;
        weight_slices[2] = wv;

        const marmot_tensor_t *biases[3] = {desc->separate.bq, desc->separate.bk, desc->separate.bv};
        for (size_t slice = 0; slice < 3; ++slice) {
            const marmot_tensor_t *bias = biases[slice];
            if (bias == nullptr) {
                continue;
            }
            if (bias->shape.ndim != 1 || bias->shape.shape[0] != M) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV bias tensors must be 1D and match M");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
            bias_slices[slice] = bias;
        }

        if (desc_epilogue != nullptr && desc_epilogue->bias != nullptr) {
            marmot_set_error(
                MARMOT_ERROR_INVALID_ARGUMENT,
                "Separate QKV epilogue bias must be provided via the descriptor bias tensors (bq/bk/bv)"
            );
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    }

    if (desc->out_q->shape.ndim != 2 || desc->out_k->shape.ndim != 2 || desc->out_v->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV outputs must be 2D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (desc->out_q->shape.shape[0] != N || desc->out_k->shape.shape[0] != N || desc->out_v->shape.shape[0] != N ||
        desc->out_q->shape.shape[1] != M || desc->out_k->shape.shape[1] != M || desc->out_v->shape.shape[1] != M) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV output shapes must match input N and projection M");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    marmot_tensor_t *outs[3] = {desc->out_q, desc->out_k, desc->out_v};
    for (size_t slice = 0; slice < 3; ++slice) {
        const marmot_tensor_t *weight_slice = weight_slices[slice];
        if (weight_slice == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Missing weight slice for QKV fallback");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        marmot_matmul_epilogue_t ep = {.bias = bias_slices[slice]};

        marmot_matmul_epilogue_t fused_ep = ep;
        const marmot_matmul_epilogue_t *ep_to_use = nullptr;
        if (desc_epilogue != nullptr) {
            fused_ep = *desc_epilogue;
            fused_ep.bias = ep.bias != nullptr ? ep.bias : desc_epilogue->bias;
            ep_to_use = &fused_ep;
        } else if (ep.bias != nullptr) {
            ep_to_use = &ep;
        }

        marmot_error_t status = marmot_linear(ctx, desc->input, weight_slice, ep_to_use, outs[slice]);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t marmot_matmul_qkv_dispatch(
    const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, bool from_fused, marmot_op_id_t op_id
) {
    (void)from_fused;
    marmot_weight_layout_t weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE;

    marmot_op_signature_t sig = {0};
    marmot_kernel_args_qkv_t packed = {0};
    marmot_error_t build_status =
        marmot_matmul_qkv_build(ctx, desc, op_id, MARMOT_MATMUL_LAYOUT_NT, weight_layout, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }
    return marmot_execute_signature(ctx, &sig, &packed, "Matmul QKV");
}

typedef struct {
    marmot_matmul_qkv_desc_t desc;
    marmot_tensor_t w_views[3];
    marmot_tensor_t b_views[3];
    bool uses_views;
} marmot_matmul_qkv_canonical_desc_t;

static marmot_error_t
marmot_matmul_qkv_canonicalize(const marmot_matmul_qkv_desc_t *in, marmot_matmul_qkv_canonical_desc_t *out) {
    if (in == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out = (marmot_matmul_qkv_canonical_desc_t){0};
    out->desc = *in;
    if (in->layout != MARMOT_QKV_LAYOUT_FUSED) {
        return MARMOT_SUCCESS;
    }

    const marmot_tensor_t *weight = in->fused.weight;
    if (weight == nullptr || weight->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight tensor must be 2D");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (marmot_tensor_is_block_quantized_weight(weight)) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized fused QKV weights are not supported yet");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (weight->shape.shape[0] % 3 != 0) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight first dimension must be divisible by 3");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (weight->shape.strides[1] != 1) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Fused QKV weight tensor must be contiguous in the last axis");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const size_t M = weight->shape.shape[0] / 3;
    const size_t K = weight->shape.shape[1];
    const size_t row_stride = weight->shape.strides[0];
    const size_t element_size = marmot_dtype_size(weight->dtype);
    const char *weight_bytes = (const char *)weight->data;

    const marmot_tensor_t *bias = in->fused.bias;
    const bool has_bias = (bias != nullptr);
    size_t bias_stride = 0;
    size_t bias_elem_size = 0;
    const char *bias_bytes = nullptr;
    if (has_bias) {
        if (bias->shape.ndim != 1 || bias->shape.shape[0] != weight->shape.shape[0]) {
            marmot_set_error(
                MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV bias must be 1D and match weight first dimension"
            );
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (bias->shape.strides[0] != 1) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Fused QKV bias must be contiguous");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        bias_stride = bias->shape.strides[0];
        bias_elem_size = marmot_dtype_size(bias->dtype);
        bias_bytes = (const char *)bias->data;
    }

    for (size_t slice = 0; slice < 3; ++slice) {
        out->w_views[slice] = *weight;
        out->w_views[slice].shape.ndim = 2;
        out->w_views[slice].shape.shape[0] = M;
        out->w_views[slice].shape.shape[1] = K;
        out->w_views[slice].shape.strides[0] = row_stride;
        out->w_views[slice].shape.strides[1] = 1;
        out->w_views[slice].data = (void *)(weight_bytes + slice * M * row_stride * element_size);

        if (has_bias) {
            out->b_views[slice] = *bias;
            out->b_views[slice].shape.ndim = 1;
            out->b_views[slice].shape.shape[0] = M;
            out->b_views[slice].shape.strides[0] = bias_stride;
            out->b_views[slice].data = (void *)(bias_bytes + slice * M * bias_stride * bias_elem_size);
        }
    }

    out->desc.layout = MARMOT_QKV_LAYOUT_SEPARATE;
    out->desc.separate.wq = &out->w_views[0];
    out->desc.separate.wk = &out->w_views[1];
    out->desc.separate.wv = &out->w_views[2];
    out->desc.separate.bq = has_bias ? &out->b_views[0] : nullptr;
    out->desc.separate.bk = has_bias ? &out->b_views[1] : nullptr;
    out->desc.separate.bv = has_bias ? &out->b_views[2] : nullptr;
    out->uses_views = true;
    return MARMOT_SUCCESS;
}

static marmot_error_t marmot_matmul_qkv_impl_with_op_id(
    const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, marmot_op_id_t op_id
) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Matmul QKV requires non-null context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null descriptor for fused QKV matmul");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_error_t layout_status = marmot_matmul_qkv_validate_layout(desc);
    if (layout_status != MARMOT_SUCCESS) {
        return layout_status;
    }

    const bool wants_rope = desc->rope_params != nullptr;
    if (wants_rope) {
        marmot_error_t err = marmot_matmul_qkv_validate_rope(desc);
        if (err != MARMOT_SUCCESS) {
            return err;
        }
    }

    marmot_matmul_qkv_canonical_desc_t canonical = {0};
    marmot_error_t canon_status = marmot_matmul_qkv_canonicalize(desc, &canonical);
    if (canon_status != MARMOT_SUCCESS) {
        return canon_status;
    }

    marmot_error_t universal_status = marmot_matmul_qkv_dispatch(ctx, &canonical.desc, canonical.uses_views, op_id);
    if (universal_status == MARMOT_SUCCESS) {
        return universal_status;
    }

    marmot_error_t status = marmot_matmul_qkv_fallback(ctx, desc);
    if (status == MARMOT_SUCCESS && wants_rope) {
        status = marmot_matmul_qkv_apply_rope(ctx, desc);
    }
    return status;
}

marmot_error_t marmot_matmul_qkv_impl(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc) {
    return marmot_matmul_qkv_impl_with_op_id(ctx, desc, MARMOT_OP_QKV_SHARED_INPUT);
}

marmot_error_t marmot_matmul_qkv_shared_input_impl(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc) {
    return marmot_matmul_qkv_impl_with_op_id(ctx, desc, MARMOT_OP_QKV_SHARED_INPUT);
}

marmot_error_t marmot_matmul_qkv_projection_impl(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc) {
    return marmot_matmul_qkv_impl_with_op_id(ctx, desc, MARMOT_OP_QKV_PROJECTION);
}
