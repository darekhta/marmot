#include "marmot/graph/op_signature.h"

#include <stdint.h>

#include "core/dispatch/dispatch_build.h"
#include "core/dispatch/dispatch_execute.h"
#include "core/dispatch/fusion_flags.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static marmot_error_t validate_convert_inputs(
    const marmot_context_t *ctx, const void *src, void *dst, size_t n, marmot_dtype_t src_dtype,
    marmot_dtype_t dst_dtype
) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Dtype conversion requires non-null context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (src == nullptr || dst == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in dtype conversion");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }
    if (marmot_dtype_is_packed(src_dtype) || marmot_dtype_is_packed(dst_dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Packed dtypes require specialized conversions");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t src_stride = marmot_dtype_size(src_dtype);
    size_t dst_stride = marmot_dtype_size(dst_dtype);
    if (src_stride == 0 || dst_stride == 0) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported dtype conversion");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t src_bytes = src_stride * n;
    size_t dst_bytes = dst_stride * n;
    if (marmot_buffers_overlap(dst, dst_bytes, src, src_bytes)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Source and destination buffers must not overlap");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_convert_build(
    const marmot_context_t *ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src,
    size_t n, marmot_op_signature_t *sig_out, marmot_kernel_args_convert_t *packed_out
) {
    if (sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Convert build requires non-null output buffers");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_error_t status = validate_convert_inputs(ctx, src, dst, n, src_dtype, dst_dtype);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_CONVERT,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = src_dtype,
        .weight_dtype = src_dtype,
        .output_dtype = dst_dtype,
        .accum_dtype = MARMOT_DTYPE_COUNT,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)n}},
    };

    *packed_out = (marmot_kernel_args_convert_t){
        .ctx = ctx,
        .dst = dst,
        .src = src,
        .n = n,
        .dst_dtype = dst_dtype,
        .src_dtype = src_dtype,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_convert_dispatch(
    const marmot_context_t *ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src,
    size_t n
) {
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_convert_t packed = {0};
    marmot_error_t build_status = marmot_convert_build(ctx, dst_dtype, dst, src_dtype, src, n, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }
    if (n == 0) {
        return MARMOT_SUCCESS;
    }

    return marmot_execute_signature(ctx, &sig, &packed, "Convert");
}
