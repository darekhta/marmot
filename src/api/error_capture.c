#include "marmot/error_capture.h"

#include "marmot/device.h"
#include "marmot/ops/matmul.h"
#include "marmot/ops/neural.h"
#include "marmot/ops/rope.h"
#include "marmot/tensor.h"

#include <stdio.h>

#include <string.h>

static void marmot_error_info_success(marmot_error_info_t *out_error_info) {
    if (out_error_info == nullptr) {
        return;
    }
    *out_error_info = (marmot_error_info_t){
        .code = MARMOT_SUCCESS,
        .message = {0},
        .file = nullptr,
        .line = 0,
        .function = nullptr,
    };
}

static void marmot_error_info_capture(marmot_error_info_t *out_error_info, marmot_error_t fallback_code) {
    if (out_error_info == nullptr) {
        return;
    }

    const marmot_error_info_t *info = marmot_get_last_error_info();
    *out_error_info = info ? *info
                           : (marmot_error_info_t){
                                 .code = MARMOT_SUCCESS,
                                 .message = {0},
                                 .file = nullptr,
                                 .line = 0,
                                 .function = nullptr,
                             };

    if (out_error_info->code == MARMOT_SUCCESS && fallback_code != MARMOT_SUCCESS) {
        out_error_info->code = fallback_code;
        const char *detail = marmot_get_last_error_detail();
        if (detail != nullptr && detail[0] != '\0') {
            snprintf(out_error_info->message, sizeof(out_error_info->message), "%s", detail);
        } else {
            out_error_info->message[0] = '\0';
        }
        out_error_info->file = nullptr;
        out_error_info->line = 0;
        out_error_info->function = nullptr;
    }
}

marmot_error_t
marmot_init_capture(marmot_backend_type_t backend, marmot_context_t **out_ctx, marmot_error_info_t *out_error_info) {
    if (out_ctx == nullptr) {
        marmot_error_info_success(out_error_info);
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    *out_ctx = nullptr;
    marmot_error_info_success(out_error_info);

    marmot_context_t *ctx = marmot_init(backend);
    if (ctx == nullptr) {
        marmot_error_t code = marmot_get_last_error();
        if (code == MARMOT_SUCCESS) {
            code = MARMOT_ERROR_INVALID_OPERATION;
        }
        marmot_error_info_capture(out_error_info, code);
        return code;
    }

    *out_ctx = ctx;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_allocator_get_usage_capture(
    const marmot_context_t *ctx, marmot_allocator_usage_t *usage, marmot_error_info_t *out_error_info
) {
    marmot_error_info_success(out_error_info);
    marmot_error_t status = marmot_allocator_get_usage(ctx, usage);
    if (status != MARMOT_SUCCESS) {
        marmot_error_info_capture(out_error_info, status);
    }
    return status;
}

marmot_error_t marmot_tensor_create_capture(
    const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_dtype_t dtype, marmot_tensor_t **out_tensor,
    marmot_error_info_t *out_error_info
) {
    if (out_tensor == nullptr) {
        marmot_error_info_success(out_error_info);
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    *out_tensor = nullptr;
    marmot_error_info_success(out_error_info);

    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, dtype);
    if (tensor == nullptr) {
        marmot_error_t code = marmot_get_last_error();
        if (code == MARMOT_SUCCESS) {
            code = MARMOT_ERROR_INVALID_OPERATION;
        }
        marmot_error_info_capture(out_error_info, code);
        return code;
    }

    *out_tensor = tensor;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_matmul_capture(
    const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, marmot_error_info_t *out_error_info
) {
    marmot_error_info_success(out_error_info);
    marmot_error_t status = marmot_matmul(ctx, a, b, epilogue, out);
    if (status != MARMOT_SUCCESS) {
        marmot_error_info_capture(out_error_info, status);
    }
    return status;
}

marmot_error_t marmot_linear_capture(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, marmot_error_info_t *out_error_info
) {
    marmot_error_info_success(out_error_info);
    marmot_error_t status = marmot_linear(ctx, input, weight, epilogue, out);
    if (status != MARMOT_SUCCESS) {
        marmot_error_info_capture(out_error_info, status);
    }
    return status;
}

marmot_error_t marmot_layernorm_capture(
    const marmot_context_t *ctx, const marmot_layernorm_desc_t *desc, marmot_error_info_t *out_error_info
) {
    marmot_error_info_success(out_error_info);
    marmot_error_t status = marmot_layernorm(ctx, desc);
    if (status != MARMOT_SUCCESS) {
        marmot_error_info_capture(out_error_info, status);
    }
    return status;
}
