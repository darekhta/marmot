#include "core/helpers/elementwise.h"

#include <stdint.h>

#include "internal/metal_kernel_runtime.h"
#include "internal/stride_helpers.h"
#include "metal_backend_internal.h"
#include "metal_unary_tables.gen.h"

typedef struct {
    const char *arith_kernel;
    const char *arith_row_kernel;
    const char *bitwise_kernel;
    const char *compare_kernel;
} metal_elementwise_binary_kernels_t;

typedef struct {
    marmot_dtype_t dtype;
    metal_elementwise_binary_kernels_t binary;
} metal_elementwise_traits_t;

static const metal_elementwise_traits_t k_metal_elementwise_f32_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .binary = {
        .arith_kernel = "elementwise_arith_f32",
        .arith_row_kernel = "elementwise_arith_f32_row",
        .bitwise_kernel = nullptr,
        .compare_kernel = "elementwise_compare_f32",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_f16_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .binary = {
        .arith_kernel = "elementwise_arith_f16",
        .arith_row_kernel = "elementwise_arith_f16_row",
        .bitwise_kernel = nullptr,
        .compare_kernel = "elementwise_compare_f16",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_bf16_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .binary = {
        .arith_kernel = "elementwise_arith_bf16",
        .arith_row_kernel = "elementwise_arith_bf16_row",
        .bitwise_kernel = nullptr,
        .compare_kernel = "elementwise_compare_bf16",
    },
};

#if MARMOT_ENABLE_FP8
static const metal_elementwise_traits_t k_metal_elementwise_fp8_e4m3_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E4M3,
    .binary = {
        .arith_kernel = "elementwise_arith_fp8_e4m3",
        .arith_row_kernel = "elementwise_arith_fp8_e4m3_row",
        .bitwise_kernel = nullptr,
        .compare_kernel = "elementwise_compare_fp8_e4m3",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_fp8_e5m2_traits = {
    .dtype = MARMOT_DTYPE_FLOAT8_E5M2,
    .binary = {
        .arith_kernel = "elementwise_arith_fp8_e5m2",
        .arith_row_kernel = "elementwise_arith_fp8_e5m2_row",
        .bitwise_kernel = nullptr,
        .compare_kernel = "elementwise_compare_fp8_e5m2",
    },
};
#endif

static const metal_elementwise_traits_t k_metal_elementwise_i32_traits = {
    .dtype = MARMOT_DTYPE_INT32,
    .binary = {
        .arith_kernel = "elementwise_arith_i32",
        .arith_row_kernel = nullptr,
        .bitwise_kernel = "elementwise_arith_i32",
        .compare_kernel = "elementwise_compare_i32",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_i16_traits = {
    .dtype = MARMOT_DTYPE_INT16,
    .binary = {
        .arith_kernel = "elementwise_arith_i16",
        .arith_row_kernel = nullptr,
        .bitwise_kernel = "elementwise_arith_i16",
        .compare_kernel = "elementwise_compare_i16",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_i8_traits = {
    .dtype = MARMOT_DTYPE_INT8,
    .binary = {
        .arith_kernel = "elementwise_arith_i8",
        .arith_row_kernel = nullptr,
        .bitwise_kernel = "elementwise_arith_i8",
        .compare_kernel = "elementwise_compare_i8",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_u32_traits = {
    .dtype = MARMOT_DTYPE_UINT32,
    .binary = {
        .arith_kernel = "elementwise_arith_u32",
        .arith_row_kernel = nullptr,
        .bitwise_kernel = "elementwise_arith_u32",
        .compare_kernel = "elementwise_compare_u32",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_u16_traits = {
    .dtype = MARMOT_DTYPE_UINT16,
    .binary = {
        .arith_kernel = "elementwise_arith_u16",
        .arith_row_kernel = nullptr,
        .bitwise_kernel = "elementwise_arith_u16",
        .compare_kernel = "elementwise_compare_u16",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_u8_traits = {
    .dtype = MARMOT_DTYPE_UINT8,
    .binary = {
        .arith_kernel = "elementwise_arith_u8",
        .arith_row_kernel = nullptr,
        .bitwise_kernel = "elementwise_arith_u8",
        .compare_kernel = "elementwise_compare_u8",
    },
};

static const metal_elementwise_traits_t k_metal_elementwise_u64_traits = {
    .dtype = MARMOT_DTYPE_UINT64,
    .binary = {
        .arith_kernel = "elementwise_arith_u64",
        .arith_row_kernel = nullptr,
        .bitwise_kernel = "elementwise_arith_u64",
        .compare_kernel = "elementwise_compare_u64",
    },
};

#ifdef __APPLE__

static const metal_elementwise_traits_t *metal_elementwise_select_traits(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return &k_metal_elementwise_f32_traits;
    case MARMOT_DTYPE_FLOAT16:
        return &k_metal_elementwise_f16_traits;
    case MARMOT_DTYPE_BFLOAT16:
        return &k_metal_elementwise_bf16_traits;
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return &k_metal_elementwise_fp8_e4m3_traits;
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return &k_metal_elementwise_fp8_e5m2_traits;
#endif
    case MARMOT_DTYPE_INT32:
        return &k_metal_elementwise_i32_traits;
    case MARMOT_DTYPE_INT16:
        return &k_metal_elementwise_i16_traits;
    case MARMOT_DTYPE_INT8:
        return &k_metal_elementwise_i8_traits;
    case MARMOT_DTYPE_UINT32:
        return &k_metal_elementwise_u32_traits;
    case MARMOT_DTYPE_UINT16:
        return &k_metal_elementwise_u16_traits;
    case MARMOT_DTYPE_UINT8:
        return &k_metal_elementwise_u8_traits;
    case MARMOT_DTYPE_UINT64:
        return &k_metal_elementwise_u64_traits;
    default:
        return nullptr;
    }
}

static bool metal_is_arithmetic_binary(marmot_device_binary_op_t op) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
    case MARMOT_DEVICE_BINARY_SUB:
    case MARMOT_DEVICE_BINARY_MUL:
    case MARMOT_DEVICE_BINARY_DIV:
    case MARMOT_DEVICE_BINARY_MIN:
    case MARMOT_DEVICE_BINARY_MAX:
    case MARMOT_DEVICE_BINARY_POW:
    case MARMOT_DEVICE_BINARY_MOD:
    case MARMOT_DEVICE_BINARY_SWIGLU:
    case MARMOT_DEVICE_BINARY_GEGLU:
        return true;
    default:
        return false;
    }
}

static bool metal_is_bitwise_binary(marmot_device_binary_op_t op) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_BITWISE_AND:
    case MARMOT_DEVICE_BINARY_BITWISE_OR:
    case MARMOT_DEVICE_BINARY_BITWISE_XOR:
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT:
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT:
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL:
        return true;
    default:
        return false;
    }
}

static bool metal_is_compare_binary(marmot_device_binary_op_t op) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return true;
    default:
        return false;
    }
}

static bool metal_dtype_is_integer(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_INT32:
    case MARMOT_DTYPE_INT16:
    case MARMOT_DTYPE_INT8:
    case MARMOT_DTYPE_UINT8:
    case MARMOT_DTYPE_UINT16:
    case MARMOT_DTYPE_UINT32:
    case MARMOT_DTYPE_UINT64:
        return true;
    default:
        return false;
    }
}

// Check if mixed-precision binary op is supported (similar to matmul bias dtype support)
static bool metal_binary_mixed_precision_supported(marmot_dtype_t dtype_a, marmot_dtype_t dtype_b) {
    // Allow F32 + F16/BF16 in either order (we'll convert the F32 input)
    if (dtype_a == dtype_b) {
        return true;
    }
    if (dtype_a == MARMOT_DTYPE_FLOAT32 && (dtype_b == MARMOT_DTYPE_FLOAT16 || dtype_b == MARMOT_DTYPE_BFLOAT16)) {
        return true;
    }
    if (dtype_b == MARMOT_DTYPE_FLOAT32 && (dtype_a == MARMOT_DTYPE_FLOAT16 || dtype_a == MARMOT_DTYPE_BFLOAT16)) {
        return true;
    }
    return false;
}

static marmot_error_t metal_elementwise_validate_binary(
    const void *device_ctx, marmot_device_binary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_tensor_t *out
) {
    if (device_ctx == nullptr || a == nullptr || b == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!metal_binary_mixed_precision_supported(a->dtype, b->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Binary op input dtypes not compatible");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (metal_is_compare_binary(op)) {
        if (out->dtype != MARMOT_DTYPE_UINT8) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
    } else {
        // For arithmetic ops, output dtype must match the non-F32 input (or both if same dtype)
        marmot_dtype_t expected_out = (a->dtype == MARMOT_DTYPE_FLOAT32) ? b->dtype : a->dtype;
        if (out->dtype != expected_out) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Binary op output dtype must match lower precision input");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_elementwise_binary_common(
    metal_context_t *ctx, marmot_device_binary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    const metal_elementwise_traits_t *traits = metal_elementwise_select_traits(a->dtype);
    if (traits == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const auto a_stride = marmot::metal::get_stride_info(a);
    const auto b_stride = marmot::metal::get_stride_info(b);
    const auto out_stride = marmot::metal::get_stride_info(out);
    const bool any_strided = (!a_stride.is_contiguous || !b_stride.is_contiguous || !out_stride.is_contiguous);
    if (any_strided) {
        const bool row_strided = a_stride.is_row_strided && b_stride.is_row_strided && out_stride.is_row_strided &&
            a->shape.ndim == 2 && b->shape.ndim == 2 && out->shape.ndim == 2 &&
            a->shape.shape[0] == b->shape.shape[0] && a->shape.shape[1] == b->shape.shape[1] &&
            a->shape.shape[0] == out->shape.shape[0] && a->shape.shape[1] == out->shape.shape[1];
        if (!row_strided || traits->binary.arith_row_kernel == nullptr || !metal_is_arithmetic_binary(op)) {
            marmot_set_error(
                MARMOT_ERROR_NOT_IMPLEMENTED, "Metal binary op requires contiguous or row-strided tensors"
            );
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }

        const uint32_t rows = (uint32_t)a->shape.shape[0];
        const uint32_t cols = (uint32_t)a->shape.shape[1];
        return metal_elementwise_run_binary_kernel_row_strided(
            ctx, a, b, out, traits->binary.arith_row_kernel, op, rows, cols, a_stride.row_stride, b_stride.row_stride,
            out_stride.row_stride
        );
    }

    if (metal_is_compare_binary(op)) {
        if (traits->binary.compare_kernel == nullptr) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        return metal_elementwise_run_binary_kernel(ctx, a, b, out, traits->binary.compare_kernel, op);
    }

    if (metal_is_bitwise_binary(op)) {
        if (!metal_dtype_is_integer(a->dtype)) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (traits->binary.bitwise_kernel == nullptr) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        return metal_elementwise_run_binary_kernel(ctx, a, b, out, traits->binary.bitwise_kernel, op);
    }

    if (metal_is_arithmetic_binary(op)) {
        if (traits->binary.arith_kernel == nullptr) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        return metal_elementwise_run_binary_kernel(ctx, a, b, out, traits->binary.arith_kernel, op);
    }

    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported Metal binary operation");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t metal_binary_dispatch(
    const void *device_ctx, marmot_device_binary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t validate = metal_elementwise_validate_binary(device_ctx, op, a, b, out);
    if (validate != MARMOT_SUCCESS) {
        return validate;
    }

    size_t problem_bytes = out != nullptr ? marmot_tensor_size_bytes(out) : 0;
    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_BINARY, "binary", problem_bytes, true, "gpu");

    // Handle mixed-precision: convert F32 input to match F16/BF16
    const bool a_needs_convert = (a->dtype == MARMOT_DTYPE_FLOAT32 && b->dtype != MARMOT_DTYPE_FLOAT32);
    const bool b_needs_convert = (b->dtype == MARMOT_DTYPE_FLOAT32 && a->dtype != MARMOT_DTYPE_FLOAT32);

    if (!a_needs_convert && !b_needs_convert) {
        // Same dtype, no conversion needed
        return metal_elementwise_binary_common(ctx, op, a, b, out);
    }

    // Determine target dtype and which input needs conversion
    const marmot_dtype_t target_dtype = out->dtype;
    const marmot_tensor_t *convert_input = a_needs_convert ? a : b;
    const size_t n_elements = marmot_tensor_num_elements(convert_input);

    // Allocate temporary buffer for converted input
    size_t converted_bytes = n_elements * marmot_dtype_size(target_dtype);
    marmot_allocation_t converted_alloc =
        {.ptr = nullptr, .size = 0, .alignment = 0, .type = MARMOT_ALLOC_GPU_SHARED, .alloc_id = 0};
    if (metal_allocate_tracked(ctx, converted_bytes, MARMOT_ALLOC_GPU_SHARED, &converted_alloc) != MARMOT_SUCCESS) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    // Convert F32 to target dtype
    marmot_error_t convert_status = metal_convert_dispatch(
        ctx, target_dtype, converted_alloc.ptr, convert_input->dtype, convert_input->data, n_elements
    );
    if (convert_status != MARMOT_SUCCESS) {
        metal_allocator_ops.free(ctx, &converted_alloc);
        return convert_status;
    }

    // Create a temporary tensor descriptor pointing to converted data
    marmot_tensor_t converted_tensor = *convert_input;
    converted_tensor.dtype = target_dtype;
    converted_tensor.data = converted_alloc.ptr;

    // Call binary op with matching dtypes
    const marmot_tensor_t *effective_a = a_needs_convert ? &converted_tensor : a;
    const marmot_tensor_t *effective_b = b_needs_convert ? &converted_tensor : b;
    marmot_error_t result = metal_elementwise_binary_common(ctx, op, effective_a, effective_b, out);

    // Free temporary buffer
    metal_allocator_ops.free(ctx, &converted_alloc);

    return result;
}

static marmot_error_t metal_unary_dispatch_impl(
    metal_context_t *ctx, marmot_device_unary_op_t op, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out
) {
    if (x == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->dtype != out->dtype) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (op < 0 || op >= MARMOT_DEVICE_UNARY_COUNT) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->dtype < 0 || x->dtype >= MARMOT_DTYPE_COUNT) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    const char *kernel_name = k_metal_unary_kernel_name[x->dtype][op];
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    bool needs_params = k_metal_unary_needs_params[op];
    const char *vec4_kernel = k_metal_unary_vec4_kernel_name[x->dtype][op];
    const char *fused_bias_kernel = k_metal_unary_fused_bias_kernel_name[x->dtype];

    marmot_activation_params_t prepared_params;
    const marmot_activation_params_t *effective_params = params;
    if (needs_params || params != nullptr) {
        marmot_error_t prep_err = marmot_unary_prepare_activation_params(op, params, &prepared_params);
        if (prep_err != MARMOT_SUCCESS) {
            return prep_err;
        }
        effective_params = &prepared_params;
    }

    metal_activation_params_t args = {
        .alpha = effective_params != nullptr ? effective_params->alpha : 0.0f,
        .beta = effective_params != nullptr ? effective_params->beta : 0.0f,
        .gamma = effective_params != nullptr ? effective_params->gamma : 0.0f,
        .delta = 0.0f,
    };

    if (effective_params != nullptr && effective_params->bias != nullptr && fused_bias_kernel != nullptr) {
        marmot_error_t fused_result = metal_elementwise_run_fused_bias_activation(
            ctx, x->dtype, fused_bias_kernel, op, x, effective_params->bias, out, &args
        );
        return fused_result;
    }

    return metal_elementwise_run_unary_kernel(
        ctx, x, out, kernel_name, vec4_kernel, (needs_params || effective_params != nullptr) ? &args : nullptr,
        needs_params
    );
}

marmot_error_t metal_elementwise_validate_unary(
    const void *device_ctx, marmot_device_unary_op_t op, const marmot_tensor_t *x, const marmot_tensor_t *out
) {
    if (device_ctx == nullptr || x == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    (void)op;
    if (x->dtype != out->dtype) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_unary_dispatch(
    const void *device_ctx, marmot_device_unary_op_t op, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t validate = metal_elementwise_validate_unary(device_ctx, op, x, out);
    if (validate != MARMOT_SUCCESS) {
        return validate;
    }

    size_t problem_bytes = out != nullptr ? marmot_tensor_size_bytes(out) : 0;
    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_UNARY, "unary", problem_bytes, true, "gpu");

    return metal_unary_dispatch_impl(ctx, op, x, params, out);
}

marmot_error_t metal_elementwise_binary_impl(
    const void *device_ctx, marmot_device_binary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
) {
    return metal_binary_dispatch(device_ctx, op, a, b, out);
}

marmot_error_t metal_elementwise_unary_impl(
    const void *device_ctx, marmot_device_unary_op_t op, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t validate = metal_elementwise_validate_unary(device_ctx, op, x, out);
    if (validate != MARMOT_SUCCESS) {
        return validate;
    }
    return metal_unary_dispatch_impl(ctx, op, x, params, out);
}

marmot_error_t metal_elementwise_ternary_impl(
    const void *device_ctx, marmot_device_ternary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_tensor_t *c, marmot_tensor_t *out
) {
    return metal_ternary_dispatch(device_ctx, op, a, b, c, out);
}

#endif // __APPLE__
