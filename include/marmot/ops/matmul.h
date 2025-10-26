#ifndef MARMOT_OPS_MATMUL_H
#define MARMOT_OPS_MATMUL_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// Linear projection (GGUF/nn.Linear semantics): input(N×K) @ weight(M×K)^T = out(N×M) with optional epilogue
MARMOT_NODISCARD marmot_error_t marmot_linear(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

// Matrix multiplication: A(M×K) @ B(K×N) = out(M×N) (PyTorch convention, 2D only for now)
MARMOT_NODISCARD marmot_error_t marmot_matmul(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

// Fused matmul with bias
MARMOT_NODISCARD marmot_error_t marmot_matmul_bias(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_matmul_bias_relu(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_matmul_bias_gelu(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
MARMOT_NODISCARD marmot_error_t marmot_matmul_bias_silu(
    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);

// Prepack a block-quantized weight tensor for faster CPU matmul; no-op/NI on other backends.
MARMOT_NODISCARD marmot_error_t
marmot_matmul_prepack_quant_weight(const marmot_context_t *ctx, const marmot_tensor_t *weight);

static inline marmot_rope_params_t marmot_rope_params_default(void) {
    marmot_rope_params_t params;
    params.positions = nullptr;
    params.scaling_type = MARMOT_ROPE_SCALING_NONE;
    params.rope_type = MARMOT_ROPE_TYPE_NORM;
    params.theta = 10000.0f;
    params.freq_scale = 1.0f;
    params.ext_factor = 0.0f;
    params.attn_factor = 1.0f;
    params.beta_fast = 32.0f;
    params.beta_slow = 1.0f;
    params.orig_ctx_len = 0;
    params.head_dim = 0;
    params.apply_to_q = true;
    params.apply_to_k = true;
    return params;
}

typedef enum {
    MARMOT_QKV_LAYOUT_UNSPECIFIED = 0,
    MARMOT_QKV_LAYOUT_FUSED = 1,
    MARMOT_QKV_LAYOUT_SEPARATE = 2,
} marmot_matmul_qkv_layout_t;

typedef struct marmot_matmul_qkv_fused {
    const marmot_tensor_t *weight;
    const marmot_tensor_t *bias;
} marmot_matmul_qkv_fused_t;

typedef struct marmot_matmul_qkv_separate {
    const marmot_tensor_t *wq;
    const marmot_tensor_t *wk;
    const marmot_tensor_t *wv;
    const marmot_tensor_t *bq;
    const marmot_tensor_t *bk;
    const marmot_tensor_t *bv;
} marmot_matmul_qkv_separate_t;

typedef struct marmot_matmul_qkv_desc {
    const marmot_tensor_t *input;
    marmot_matmul_qkv_layout_t layout;
    union {
        marmot_matmul_qkv_fused_t fused;
        marmot_matmul_qkv_separate_t separate;
    };
    const marmot_matmul_epilogue_t *epilogue;
    marmot_tensor_t *out_q;
    marmot_tensor_t *out_k;
    marmot_tensor_t *out_v;
    const marmot_rope_params_t *rope_params;
} marmot_matmul_qkv_desc_t;

static inline marmot_matmul_qkv_desc_t marmot_matmul_qkv_desc_default(void) {
    marmot_matmul_qkv_desc_t desc;
    desc.input = nullptr;
    desc.layout = MARMOT_QKV_LAYOUT_UNSPECIFIED;
    desc.fused.weight = nullptr;
    desc.fused.bias = nullptr;
    desc.separate.wq = nullptr;
    desc.separate.wk = nullptr;
    desc.separate.wv = nullptr;
    desc.separate.bq = nullptr;
    desc.separate.bk = nullptr;
    desc.separate.bv = nullptr;
    desc.epilogue = nullptr;
    desc.out_q = nullptr;
    desc.out_k = nullptr;
    desc.out_v = nullptr;
    desc.rope_params = nullptr;
    return desc;
}

MARMOT_NODISCARD marmot_error_t marmot_matmul_qkv(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc);
MARMOT_NODISCARD marmot_error_t
marmot_matmul_qkv_shared_input(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc);
MARMOT_NODISCARD marmot_error_t
marmot_matmul_qkv_projection(const marmot_context_t *ctx, const marmot_matmul_qkv_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_MATMUL_H
