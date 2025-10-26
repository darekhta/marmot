#include <stdlib.h>

#include <string.h>

#include "internal/metal_matmul_internal.h"
#include "internal/metal_matmul_qkv_shared.h"
#include "metal_packed_weight.h"
#include "ops/matmul_kernels.h"

#ifdef __APPLE__

static bool metal_matmul_qkv_tensor_is_quantized(const marmot_tensor_t *tensor) {
    return tensor != nullptr && tensor->quant_kind != MARMOT_QUANT_KIND_GENERIC;
}

static bool metal_matmul_qkv_desc_has_quantized_weights(const marmot_matmul_qkv_desc_t *desc) {
    if (desc == nullptr || desc->layout != MARMOT_QKV_LAYOUT_SEPARATE) {
        return false;
    }
    const marmot_tensor_t *wq = desc->separate.wq;
    const marmot_tensor_t *wk = desc->separate.wk;
    const marmot_tensor_t *wv = desc->separate.wv;
    if (!metal_matmul_qkv_tensor_is_quantized(wq) || !metal_matmul_qkv_tensor_is_quantized(wk) ||
        !metal_matmul_qkv_tensor_is_quantized(wv)) {
        return false;
    }
    return wq->quant_kind == wk->quant_kind && wq->quant_kind == wv->quant_kind &&
        wq->quant_layout == wk->quant_layout && wq->quant_layout == wv->quant_layout;
}

static marmot_error_t
metal_matmul_qkv_validate_bias_tensor(const marmot_tensor_t *bias, const marmot_tensor_t *out, size_t M) {
    if (bias == nullptr) {
        return MARMOT_SUCCESS;
    }
    if (!metal_matmul_bias_dtype_supported(out->dtype, bias->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "QKV bias dtype must match output or be FP32");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (bias->shape.ndim != 1 || bias->shape.shape[0] != M || bias->shape.strides[0] != 1) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV bias must be contiguous and match projection size");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t
metal_matmul_qkv_validate_desc(const marmot_matmul_qkv_desc_t *desc, metal_matmul_qkv_dims_t *dims) {
    if (desc == nullptr || desc->input == nullptr || desc->out_q == nullptr || desc->out_k == nullptr ||
        desc->out_v == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *input = desc->input;
    const marmot_tensor_t *out_q = desc->out_q;
    const marmot_tensor_t *out_k = desc->out_k;
    const marmot_tensor_t *out_v = desc->out_v;
    if (input->shape.ndim != 2 || out_q->shape.ndim != 2 || out_k->shape.ndim != 2 || out_v->shape.ndim != 2) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV expects 2D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    const size_t N = input->shape.shape[0];
    const size_t K = input->shape.shape[1];
    size_t M = 0;

    if (desc->layout == MARMOT_QKV_LAYOUT_FUSED) {
        const marmot_tensor_t *weight = desc->fused.weight;
        if (weight == nullptr || weight->shape.ndim != 2) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Fused QKV weight tensor must be 2D");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (weight->quant_kind != MARMOT_QUANT_KIND_GENERIC) {
            marmot_set_error(
                MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized fused QKV weights are not supported on Metal yet"
            );
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        const size_t fused_rows = weight->shape.shape[0];
        if (fused_rows % 3 != 0) {
            marmot_set_error(
                MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV weight first dimension must be divisible by 3"
            );
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        M = fused_rows / 3;
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
        if (input->dtype != weight->dtype || input->dtype != out_q->dtype || input->dtype != out_k->dtype ||
            input->dtype != out_v->dtype) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Fused QKV tensors must share dtype");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        const marmot_tensor_t *bias = desc->fused.bias;
        if (bias != nullptr) {
            if (bias->dtype != input->dtype) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Fused QKV bias dtype must match activations");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
            if (bias->shape.ndim != 1 || bias->shape.shape[0] != fused_rows || bias->shape.strides[0] != 1) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Fused QKV bias must be contiguous and match rows");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
        }
        if (desc->epilogue != nullptr && desc->epilogue->bias != nullptr && bias == nullptr) {
            marmot_set_error(
                MARMOT_ERROR_INVALID_ARGUMENT, "Fused QKV epilogue bias must be supplied via the descriptor bias tensor"
            );
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    } else if (desc->layout == MARMOT_QKV_LAYOUT_SEPARATE) {
        const marmot_tensor_t *wq = desc->separate.wq;
        const marmot_tensor_t *wk = desc->separate.wk;
        const marmot_tensor_t *wv = desc->separate.wv;
        if (wq == nullptr || wk == nullptr || wv == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Separate QKV layout requires wq, wk, and wv tensors");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (wq->shape.ndim != 2 || wk->shape.ndim != 2 || wv->shape.ndim != 2) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must be 2D");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        const bool weights_quantized = metal_matmul_qkv_desc_has_quantized_weights(desc);
        if (wq->shape.shape[1] != K || wk->shape.shape[1] != K || wv->shape.shape[1] != K) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must share the input dimension");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (wq->shape.shape[0] != wk->shape.shape[0] || wq->shape.shape[0] != wv->shape.shape[0]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Separate QKV weights must share the output dimension");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (wq->shape.strides[1] != 1 || wk->shape.strides[1] != 1 || wv->shape.strides[1] != 1) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Separate QKV weights must be contiguous in the last axis");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        M = wq->shape.shape[0];
        if (!weights_quantized) {
            if (wq->quant_kind != MARMOT_QUANT_KIND_GENERIC || wk->quant_kind != MARMOT_QUANT_KIND_GENERIC ||
                wv->quant_kind != MARMOT_QUANT_KIND_GENERIC) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized separate QKV weights are not supported yet");
                return MARMOT_ERROR_NOT_IMPLEMENTED;
            }
            if (input->dtype != wq->dtype || input->dtype != wk->dtype || input->dtype != wv->dtype ||
                input->dtype != out_q->dtype || input->dtype != out_k->dtype || input->dtype != out_v->dtype) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Separate QKV tensors must share dtype");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
        } else {
            if (!(input->dtype == MARMOT_DTYPE_FLOAT32 || input->dtype == MARMOT_DTYPE_FLOAT16)) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantized QKV requires FP16/FP32 activations");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
            if (!(out_q->dtype == MARMOT_DTYPE_FLOAT32 || out_q->dtype == MARMOT_DTYPE_FLOAT16)) {
                marmot_set_error(
                    MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantized QKV outputs must be FP16 or FP32 on Metal backend"
                );
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
            if (out_q->dtype != out_k->dtype || out_q->dtype != out_v->dtype) {
                marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Quantized QKV outputs must share dtype");
                return MARMOT_ERROR_UNSUPPORTED_DTYPE;
            }
        }
        const marmot_tensor_t *biases[3] = {desc->separate.bq, desc->separate.bk, desc->separate.bv};
        for (size_t i = 0; i < 3; ++i) {
            const marmot_tensor_t *bias = biases[i];
            if (bias == nullptr) {
                continue;
            }
            marmot_error_t bias_status =
                metal_matmul_qkv_validate_bias_tensor(bias, i == 0 ? out_q : (i == 1 ? out_k : out_v), M);
            if (bias_status != MARMOT_SUCCESS) {
                return bias_status;
            }
        }
        if (desc->epilogue != nullptr && desc->epilogue->bias != nullptr) {
            marmot_set_error(
                MARMOT_ERROR_INVALID_ARGUMENT,
                "Separate QKV epilogue bias must be provided via the descriptor bias tensors (bq/bk/bv)"
            );
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    } else {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid QKV layout");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (out_q->shape.shape[0] != N || out_q->shape.shape[1] != M || out_k->shape.shape[0] != N ||
        out_k->shape.shape[1] != M || out_v->shape.shape[0] != N || out_v->shape.shape[1] != M) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "QKV output shapes must match input batch and projection");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (dims != nullptr) {
        dims->N = N;
        dims->K = K;
        dims->M = M;
    }
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_matmul_qkv_run_fallback(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, const metal_matmul_qkv_dims_t *dims
) {
    const marmot_tensor_t *weight = desc->fused.weight;
    const marmot_tensor_t *bias = desc->fused.bias;
    const bool has_bias = (bias != nullptr);
    const size_t M = dims->M;
    const size_t K = dims->K;
    const size_t row_stride = weight->shape.strides[0];
    const size_t element_size = marmot_dtype_size(weight->dtype);
    const char *weight_bytes = (const char *)weight->data;

    const char *bias_bytes = has_bias ? (const char *)bias->data : nullptr;
    const size_t bias_stride = has_bias ? bias->shape.strides[0] : 0;
    const size_t bias_elt_size = has_bias ? marmot_dtype_size(bias->dtype) : 0;

    marmot_tensor_t *outputs[3] = {desc->out_q, desc->out_k, desc->out_v};

    for (size_t slice = 0; slice < 3; ++slice) {
        marmot_tensor_t weight_view = *weight;
        weight_view.shape.shape[0] = M;
        weight_view.shape.shape[1] = K;
        weight_view.data = (void *)(weight_bytes + slice * M * row_stride * element_size);

        marmot_matmul_epilogue_t ep = {};
        marmot_tensor_t bias_view;
        memset(&bias_view, 0, sizeof(bias_view));
        if (has_bias) {
            bias_view = *bias;
            bias_view.shape.shape[0] = M;
            bias_view.data = (void *)(bias_bytes + slice * M * bias_stride * bias_elt_size);
            ep.bias = &bias_view;
        }

        const marmot_matmul_epilogue_t *ep_ptr = ep.bias != nullptr ? &ep : nullptr;

        marmot_error_t status = metal_matmul_direct(ctx, desc->input, &weight_view, ep_ptr, outputs[slice]);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
    }
    return MARMOT_SUCCESS;
}

typedef struct {
    uint32_t N;
    uint32_t K;
    uint32_t M;
    uint32_t has_bias;
    uint32_t has_residual;
    uint32_t activation;
    uint32_t use_packed_weights;
    uint32_t packed_tile_cols;
    uint32_t packed_tile_k;
    uint32_t packed_tiles_per_row;
    uint32_t packed_tiles_per_col;
    uint32_t packed_tile_stride;
    uint32_t packed_tile_section;
    uint32_t packed_use_vec4;
    uint32_t has_bias_q;
    uint32_t has_bias_k;
    uint32_t has_bias_v;
    uint32_t rope_enabled;
    uint32_t rope_apply_q;
    uint32_t rope_apply_k;
    uint32_t rope_head_dim;
    float rope_attn_scale;
    metal_matmul_activation_params_t params;
} metal_matmul_qkv_uniforms_t;

static const char *metal_matmul_qkv_dense_kernel_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "matmul_qkv_f32_nt";
    case MARMOT_DTYPE_FLOAT16:
        return "matmul_qkv_f16_nt";
    case MARMOT_DTYPE_BFLOAT16:
        return "matmul_qkv_bf16_nt";
    default:
        return nullptr;
    }
}

static marmot_error_t
metal_matmul_qkv_dense_entry(metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, marmot_dtype_t dtype) {
    if (ctx == nullptr || desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    metal_matmul_qkv_dims_t dims;
    memset(&dims, 0, sizeof(dims));
    marmot_error_t status = metal_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    const char *kernel_name = metal_matmul_qkv_dense_kernel_name(dtype);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const marmot_rope_params_t *rope = desc->rope_params;
    const bool wants_rope = rope != nullptr && (rope->apply_to_q || rope->apply_to_k);

    marmot_matmul_qkv_desc_t fused_desc;
    memset(&fused_desc, 0, sizeof(fused_desc));
    metal_matmul_qkv_packed_weights_t packed;
    memset(&packed, 0, sizeof(packed));
    const marmot_matmul_qkv_desc_t *gpu_desc = desc;
    if (desc->layout == MARMOT_QKV_LAYOUT_SEPARATE) {
        fused_desc = *desc;
        status = metal_matmul_qkv_prepare_fused_desc(ctx, desc, &dims, &fused_desc, &packed);
        if (status != MARMOT_SUCCESS) {
            metal_matmul_qkv_release_packed_weights(&packed);
            return status;
        }
        gpu_desc = &fused_desc;
    }

    marmot_error_t kernel_status = metal_matmul_qkv_run_kernel(ctx, gpu_desc, &dims, kernel_name);
    if (kernel_status == MARMOT_SUCCESS) {
        metal_matmul_qkv_release_packed_weights(&packed);
        if (wants_rope && rope->rope_type == MARMOT_ROPE_TYPE_NEOX) {
            marmot_error_t rope_status = metal_matmul_qkv_apply_rope_gpu(ctx, gpu_desc, &dims);
            if (rope_status != MARMOT_SUCCESS) {
                return rope_status;
            }
        }
        return kernel_status;
    }
    if (kernel_status != MARMOT_ERROR_NOT_IMPLEMENTED && kernel_status != MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        metal_matmul_qkv_release_packed_weights(&packed);
        return kernel_status;
    }

    marmot_error_t fb_status = metal_matmul_qkv_run_fallback(ctx, gpu_desc, &dims);
    metal_matmul_qkv_release_packed_weights(&packed);
    if (fb_status == MARMOT_SUCCESS && wants_rope) {
        marmot_error_t rope_status = metal_matmul_qkv_apply_rope_gpu(ctx, gpu_desc, &dims);
        if (rope_status != MARMOT_SUCCESS) {
            return rope_status;
        }
    }
    return fb_status;
}

marmot_error_t metal_matmul_qkv_dense_f32_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_dense_entry((metal_context_t *)device_ctx, desc, MARMOT_DTYPE_FLOAT32);
}

marmot_error_t metal_matmul_qkv_dense_f16_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_dense_entry((metal_context_t *)device_ctx, desc, MARMOT_DTYPE_FLOAT16);
}

marmot_error_t metal_matmul_qkv_dense_bf16_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_dense_entry((metal_context_t *)device_ctx, desc, MARMOT_DTYPE_BFLOAT16);
}

static marmot_error_t metal_matmul_qkv_separate_entry(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, [[maybe_unused]] const char *kernel_name
) {
    if (ctx == nullptr || desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    metal_matmul_qkv_dims_t dims;
    memset(&dims, 0, sizeof(dims));
    marmot_error_t status = metal_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (desc->layout != MARMOT_QKV_LAYOUT_SEPARATE) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    const marmot_rope_params_t *rope = desc->rope_params;
    const bool wants_rope = rope != nullptr && (rope->apply_to_q || rope->apply_to_k);

    marmot_matmul_qkv_desc_t fused_desc = *desc;
    metal_matmul_qkv_packed_weights_t packed;
    memset(&packed, 0, sizeof(packed));
    marmot_error_t prep_status = metal_matmul_qkv_prepare_fused_desc(ctx, desc, &dims, &fused_desc, &packed);
    if (prep_status != MARMOT_SUCCESS) {
        metal_matmul_qkv_release_packed_weights(&packed);
        return prep_status;
    }

    const char *dense_kernel = metal_matmul_qkv_dense_kernel_name(desc->input->dtype);
    marmot_error_t gpu_status = metal_matmul_qkv_run_kernel(ctx, &fused_desc, &dims, dense_kernel);
    if (gpu_status == MARMOT_SUCCESS) {
        metal_matmul_qkv_release_packed_weights(&packed);
        if (wants_rope && rope->rope_type == MARMOT_ROPE_TYPE_NEOX) {
            marmot_error_t rope_status = metal_matmul_qkv_apply_rope_gpu(ctx, &fused_desc, &dims);
            if (rope_status != MARMOT_SUCCESS) {
                return rope_status;
            }
        }
        return gpu_status;
    }
    if (gpu_status != MARMOT_ERROR_NOT_IMPLEMENTED && gpu_status != MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        metal_matmul_qkv_release_packed_weights(&packed);
        return gpu_status;
    }

    marmot_error_t fb_status = metal_matmul_qkv_run_fallback(ctx, &fused_desc, &dims);
    metal_matmul_qkv_release_packed_weights(&packed);
    if (fb_status == MARMOT_SUCCESS && wants_rope) {
        marmot_error_t rope_status = metal_matmul_qkv_apply_rope_gpu(ctx, &fused_desc, &dims);
        if (rope_status != MARMOT_SUCCESS) {
            return rope_status;
        }
    }
    return fb_status;
}

static marmot_error_t
metal_matmul_qkv_quant_entry(metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, marmot_quant_kind_t kind) {
    if (ctx == nullptr || desc == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    metal_matmul_qkv_dims_t dims;
    memset(&dims, 0, sizeof(dims));
    marmot_error_t status = metal_matmul_qkv_validate_desc(desc, &dims);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    if (desc->layout != MARMOT_QKV_LAYOUT_SEPARATE) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (!metal_matmul_qkv_desc_has_quantized_weights(desc) || desc->separate.wq->quant_kind != kind) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    marmot_error_t quant_status = metal_matmul_qkv_run_quantized(ctx, desc, &dims);
    if (quant_status == MARMOT_SUCCESS) {
        return quant_status;
    }
    if (quant_status != MARMOT_ERROR_NOT_IMPLEMENTED && quant_status != MARMOT_ERROR_UNSUPPORTED_DTYPE) {
        return quant_status;
    }
    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized matmul_qkv not supported on Metal backend");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

marmot_error_t metal_matmul_qkv_separate_f16_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_separate_entry((metal_context_t *)device_ctx, desc, "matmul_qkv_separate_f16_nt");
}

marmot_error_t metal_matmul_qkv_separate_f32_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_separate_entry((metal_context_t *)device_ctx, desc, "matmul_qkv_separate_f32_nt");
}

marmot_error_t metal_matmul_qkv_separate_bf16_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_separate_entry((metal_context_t *)device_ctx, desc, "matmul_qkv_separate_bf16_nt");
}

marmot_error_t metal_matmul_qkv_q4_0_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q4_0);
}

marmot_error_t metal_matmul_qkv_q4_0_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q4_0);
}

marmot_error_t metal_matmul_qkv_q4_1_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q4_1);
}

marmot_error_t metal_matmul_qkv_q4_1_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q4_1);
}

marmot_error_t metal_matmul_qkv_q5_0_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q5_0);
}

marmot_error_t metal_matmul_qkv_q5_0_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q5_0);
}

marmot_error_t metal_matmul_qkv_q5_1_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q5_1);
}

marmot_error_t metal_matmul_qkv_q5_1_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q5_1);
}

marmot_error_t metal_matmul_qkv_q8_0_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q8_0);
}

marmot_error_t metal_matmul_qkv_q8_0_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q8_0);
}

marmot_error_t metal_matmul_qkv_q8_1_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q8_1);
}

marmot_error_t metal_matmul_qkv_q8_1_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q8_1);
}

marmot_error_t metal_matmul_qkv_q2_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q2_K);
}

marmot_error_t metal_matmul_qkv_q2_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q2_K);
}

marmot_error_t metal_matmul_qkv_q3_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q3_K);
}

marmot_error_t metal_matmul_qkv_q3_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q3_K);
}

marmot_error_t metal_matmul_qkv_q4_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q4_K);
}

marmot_error_t metal_matmul_qkv_q4_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q4_K);
}

marmot_error_t metal_matmul_qkv_q5_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q5_K);
}

marmot_error_t metal_matmul_qkv_q5_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q5_K);
}

marmot_error_t metal_matmul_qkv_q6_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q6_K);
}

marmot_error_t metal_matmul_qkv_q6_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q6_K);
}

marmot_error_t metal_matmul_qkv_q8_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q8_K);
}

marmot_error_t metal_matmul_qkv_q8_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc) {
    return metal_matmul_qkv_quant_entry((metal_context_t *)device_ctx, desc, MARMOT_QUANT_KIND_Q8_K);
}

#endif // __APPLE__
