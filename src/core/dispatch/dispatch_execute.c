#include "dispatch_execute.h"

#include "marmot/op_metadata.gen.h"

#include "core/bytecode/bytecode_compile.h"
#include "core/dispatch/dispatch_build.h"
#include "graph/kernel_dispatch_args.gen.h"

static bool marmot_fused_binary_activation_info(
    marmot_op_id_t op_id, marmot_op_id_t *base_op_out, marmot_op_id_t *activation_op_out
) {
    if (base_op_out == nullptr || activation_op_out == nullptr) {
        return false;
    }
    switch (op_id) {
    case MARMOT_OP_ADD_RELU:
        *base_op_out = MARMOT_OP_ADD;
        *activation_op_out = MARMOT_OP_RELU;
        return true;
    case MARMOT_OP_ADD_GELU:
        *base_op_out = MARMOT_OP_ADD;
        *activation_op_out = MARMOT_OP_GELU;
        return true;
    case MARMOT_OP_ADD_SILU:
        *base_op_out = MARMOT_OP_ADD;
        *activation_op_out = MARMOT_OP_SILU;
        return true;
    default:
        break;
    }
    return false;
}

static bool marmot_fused_matmul_activation_info(marmot_op_id_t op_id, marmot_op_id_t *activation_op_out) {
    if (activation_op_out == nullptr) {
        return false;
    }
    switch (op_id) {
    case MARMOT_OP_MATMUL_BIAS:
        *activation_op_out = MARMOT_OP_INVALID;
        return true;
    case MARMOT_OP_MATMUL_BIAS_RELU:
        *activation_op_out = MARMOT_OP_RELU;
        return true;
    case MARMOT_OP_MATMUL_BIAS_GELU:
        *activation_op_out = MARMOT_OP_GELU;
        return true;
    case MARMOT_OP_MATMUL_BIAS_SILU:
        *activation_op_out = MARMOT_OP_SILU;
        return true;
    default:
        break;
    }
    return false;
}

static marmot_error_t marmot_execute_binary_activation_fallback(
    const marmot_context_t *ctx, const marmot_op_signature_t *sig, const void *args, marmot_op_id_t base_op,
    marmot_op_id_t activation_op, const char *op_name
) {
    (void)op_name;
    const marmot_kernel_args_binary_t *packed = args;
    marmot_op_signature_t base_sig = *sig;
    base_sig.op_id = base_op;
    base_sig.activation = MARMOT_DEVICE_UNARY_COUNT;
    marmot_error_t status = marmot_bc_try_execute_signature(ctx, &base_sig, args);
    if (status != MARMOT_SUCCESS || activation_op == MARMOT_OP_INVALID) {
        return status;
    }

    const marmot_tensor_t *output = packed->output;
    if (output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null output for fused binary fallback");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_device_unary_op_t device_op = marmot_op_metadata_unary_from_op_id(activation_op);
    if (device_op == MARMOT_DEVICE_UNARY_COUNT) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported activation for fused fallback");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_op_signature_t unary_sig = {0};
    marmot_kernel_args_unary_t unary_packed = {0};
    marmot_error_t build_status = marmot_unary_build(
        ctx, device_op, activation_op, output, nullptr, (marmot_tensor_t *)output, &unary_sig, &unary_packed
    );
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_bc_try_execute_signature(ctx, &unary_sig, &unary_packed);
}

static marmot_error_t marmot_execute_matmul_activation_fallback(
    const marmot_context_t *ctx, const marmot_op_signature_t *sig, const void *args, marmot_op_id_t activation_op,
    const char *op_name
) {
    (void)op_name;
    const marmot_kernel_args_matmul_t *packed = args;
    const marmot_tensor_t *input = packed->input;
    const marmot_tensor_t *weight = packed->weight;
    const marmot_matmul_epilogue_t *epilogue = packed->epilogue;
    marmot_tensor_t *output = packed->output;
    if (input == nullptr || weight == nullptr || output == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor for matmul fallback");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_matmul_epilogue_t *epilogue_ptr = epilogue;

    marmot_op_signature_t base_sig = *sig;
    base_sig.op_id = MARMOT_OP_MATMUL;
    base_sig.activation = MARMOT_DEVICE_UNARY_COUNT;

    marmot_kernel_args_matmul_t packed_matmul = {
        .ctx = ctx,
        .input = input,
        .weight = weight,
        .epilogue = epilogue_ptr,
        .output = output,
    };
    marmot_error_t status = marmot_bc_try_execute_signature(ctx, &base_sig, &packed_matmul);
    if (status != MARMOT_SUCCESS || activation_op == MARMOT_OP_INVALID) {
        return status;
    }

    marmot_device_unary_op_t device_op = marmot_op_metadata_unary_from_op_id(activation_op);
    if (device_op == MARMOT_DEVICE_UNARY_COUNT) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported activation for matmul fallback");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_op_signature_t unary_sig = {0};
    marmot_kernel_args_unary_t unary_packed = {0};
    marmot_error_t build_status =
        marmot_unary_build(ctx, device_op, activation_op, output, nullptr, output, &unary_sig, &unary_packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }

    return marmot_bc_try_execute_signature(ctx, &unary_sig, &unary_packed);
}

static bool marmot_tensors_alias(const marmot_tensor_t *a, const marmot_tensor_t *b) {
    if (a == nullptr || b == nullptr) {
        return false;
    }
    if (a == b) {
        return true;
    }
    return a->data != nullptr && a->data == b->data;
}

static marmot_error_t marmot_execute_fma_fallback(const marmot_context_t *ctx, const void *args, const char *op_name) {
    (void)op_name;
    const marmot_kernel_args_ternary_t *packed = args;
    const marmot_tensor_t *a = packed->input_a;
    const marmot_tensor_t *b = packed->input_b;
    const marmot_tensor_t *c = packed->input_c;
    marmot_tensor_t *out = packed->output;
    if (a == nullptr || b == nullptr || c == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor for FMA fallback");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_tensor_t *mul_out = out;
    marmot_tensor_t *temp = nullptr;
    if (marmot_tensors_alias(out, c)) {
        temp = marmot_tensor_create(ctx, out->shape.shape, out->shape.ndim, out->dtype);
        if (temp == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate temporary tensor for FMA fallback");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        mul_out = temp;
    }

    marmot_op_signature_t mul_sig = {0};
    marmot_kernel_args_binary_t mul_packed = {0};
    marmot_error_t status = marmot_binary_build(ctx, MARMOT_OP_MUL, a, b, mul_out, false, true, &mul_sig, &mul_packed);
    if (status == MARMOT_SUCCESS) {
        status = marmot_bc_try_execute_signature(ctx, &mul_sig, &mul_packed);
    }
    if (status != MARMOT_SUCCESS) {
        if (temp != nullptr) {
            marmot_tensor_destroy(temp);
        }
        return status;
    }

    marmot_op_signature_t add_sig = {0};
    marmot_kernel_args_binary_t add_packed = {0};
    status = marmot_binary_build(ctx, MARMOT_OP_ADD, mul_out, c, out, false, true, &add_sig, &add_packed);
    if (status == MARMOT_SUCCESS) {
        status = marmot_bc_try_execute_signature(ctx, &add_sig, &add_packed);
    }

    if (temp != nullptr) {
        marmot_tensor_destroy(temp);
    }
    return status;
}

marmot_error_t marmot_execute_signature(
    const marmot_context_t *ctx, const marmot_op_signature_t *sig, const void *args, const char *op_name
) {
    if (ctx == nullptr || sig == nullptr || args == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null argument to operation dispatch");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const char *label = op_name != nullptr ? op_name : "operation";
    marmot_op_signature_t sig_copy = *sig;
    sig_copy.variant_flags &= ctx->policy.variant_flags_mask;
    marmot_error_t status = marmot_bc_try_execute_signature(ctx, &sig_copy, args);
    if (status != MARMOT_ERROR_NOT_IMPLEMENTED) {
        return status;
    }

    marmot_op_id_t base_op = MARMOT_OP_INVALID;
    marmot_op_id_t activation_op = MARMOT_OP_INVALID;
    if (marmot_fused_binary_activation_info(sig->op_id, &base_op, &activation_op)) {
        return marmot_execute_binary_activation_fallback(ctx, &sig_copy, args, base_op, activation_op, label);
    }

    if (marmot_fused_matmul_activation_info(sig->op_id, &activation_op)) {
        return marmot_execute_matmul_activation_fallback(ctx, &sig_copy, args, activation_op, label);
    }

    if (sig->op_id == MARMOT_OP_FMA) {
        return marmot_execute_fma_fallback(ctx, args, label);
    }

    return status;
}
