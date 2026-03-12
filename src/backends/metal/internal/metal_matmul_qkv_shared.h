#ifndef METAL_MATMUL_QKV_SHARED_H
#define METAL_MATMUL_QKV_SHARED_H

#include "marmot/ops/matmul.h"

#include "metal_backend_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct metal_matmul_qkv_dims {
    size_t N;
    size_t K;
    size_t M;
} metal_matmul_qkv_dims_t;

static inline bool metal_matmul_qkv_epilogue_has_effect(const marmot_matmul_epilogue_t *epilogue) {
    if (epilogue == nullptr) {
        return false;
    }
    return epilogue->bias != nullptr || epilogue->enable_output_cast;
}

typedef struct {
    marmot_matmul_epilogue_t storage;
    const marmot_matmul_epilogue_t *ep;
    const marmot_rope_params_t *rope;
    size_t feature_dim;
    bool bias_scalar;
} metal_matmul_qkv_epilogue_config_t;

static inline void metal_matmul_qkv_init_identity_epilogue(marmot_matmul_epilogue_t *ep, marmot_dtype_t dtype) {
    ep->bias = nullptr;
    ep->enable_output_cast = false;
    ep->output_dtype = dtype;
}

static inline marmot_error_t metal_matmul_qkv_prepare_epilogue(
    const marmot_tensor_t *out, const marmot_matmul_epilogue_t *base_ep, const marmot_tensor_t *bias_override,
    const marmot_rope_params_t *rope_params, bool apply_rope_to_q, bool apply_rope_to_k, bool detach_rope,
    bool validate_support, metal_matmul_qkv_epilogue_config_t *cfg
) {
    if (cfg == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    (void)detach_rope;
    metal_matmul_qkv_init_identity_epilogue(&cfg->storage, out->dtype);
    if (base_ep != nullptr) {
        cfg->storage = *base_ep;
    }
    if (bias_override != nullptr) {
        cfg->storage.bias = bias_override;
    }
    cfg->ep = nullptr;
    cfg->rope = nullptr;
    cfg->feature_dim = 0;
    cfg->bias_scalar = false;

    if (rope_params != nullptr) {
        const bool apply_rope =
            (apply_rope_to_q && rope_params->apply_to_q) || (apply_rope_to_k && rope_params->apply_to_k);
        cfg->rope = apply_rope ? rope_params : nullptr;
    }

    if (!metal_matmul_qkv_epilogue_has_effect(&cfg->storage)) {
        return MARMOT_SUCCESS;
    }

    if (validate_support) {
        if (!metal_matmul_epilogue_supported(out, &cfg->storage, &cfg->feature_dim, &cfg->bias_scalar)) {
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
    }
    cfg->ep = &cfg->storage;
    return MARMOT_SUCCESS;
}

static inline bool
metal_matmul_qkv_epilogue_inline_supported(const marmot_tensor_t *out, const metal_matmul_qkv_epilogue_config_t *cfg) {
    (void)out;
    (void)cfg;
    return false;
}

static inline uint32_t metal_matmul_qkv_resolve_head_dim(size_t dim, const marmot_rope_params_t *rope) {
    if (rope == nullptr) {
        return (uint32_t)dim;
    }
    const uint32_t head_dim = rope->head_dim;
    if (head_dim == 0 || head_dim > dim || (dim % head_dim) != 0 || (head_dim & 1u) != 0u) {
        return (uint32_t)dim;
    }
    return head_dim;
}

// QKV kernel dispatch (from metal_matmul_qkv_kernel.h)
marmot_error_t metal_matmul_qkv_run_kernel(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, const metal_matmul_qkv_dims_t *dims,
    const char *kernel_name
);

marmot_error_t metal_matmul_qkv_apply_rope_gpu(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, const metal_matmul_qkv_dims_t *dims
);

// QKV weight packing (from metal_matmul_qkv_pack.h)
typedef struct {
    bool owns_weight;
    bool owns_bias;
    marmot_tensor_t weight_tensor;
    marmot_tensor_t bias_tensor;
    void *weight_storage;
    void *bias_storage;
    void *cache_entry;
} metal_matmul_qkv_packed_weights_t;

marmot_error_t metal_matmul_qkv_prepare_fused_desc(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *src, const metal_matmul_qkv_dims_t *dims,
    marmot_matmul_qkv_desc_t *dst, metal_matmul_qkv_packed_weights_t *packed
);
void metal_matmul_qkv_release_packed_weights(metal_matmul_qkv_packed_weights_t *packed);

// Quantized QKV dispatch (from metal_matmul_qkv_quant.h)
marmot_error_t metal_matmul_qkv_run_quantized(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *desc, const metal_matmul_qkv_dims_t *dims
);
marmot_error_t metal_matmul_qkv_run_quantized_dual_output(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight_q,
    const marmot_tensor_t *weight_k, marmot_tensor_t *out_q, marmot_tensor_t *out_k
);

#ifdef __cplusplus
}
#endif

#endif // METAL_MATMUL_QKV_SHARED_H
