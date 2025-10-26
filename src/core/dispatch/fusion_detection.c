#include "fusion_detection.h"

static marmot_op_id_t matmul_bias_activation_op(marmot_op_id_t activation) {
    switch (activation) {
    case MARMOT_OP_RELU:
        return MARMOT_OP_MATMUL_BIAS_RELU;
    case MARMOT_OP_GELU:
        return MARMOT_OP_MATMUL_BIAS_GELU;
    case MARMOT_OP_SILU:
        return MARMOT_OP_MATMUL_BIAS_SILU;
    default:
        return MARMOT_OP_INVALID;
    }
}

marmot_op_id_t marmot_detect_fused_op_id(const marmot_fusion_context_t *ctx) {
    if (ctx == nullptr) {
        return MARMOT_OP_INVALID;
    }

    if (ctx->current_op == MARMOT_OP_ADD && ctx->intermediate_is_temporary) {
        if (ctx->next_op == MARMOT_OP_RELU) {
            return MARMOT_OP_ADD_RELU;
        }
        if (ctx->next_op == MARMOT_OP_GELU) {
            return MARMOT_OP_ADD_GELU;
        }
        if (ctx->next_op == MARMOT_OP_SILU) {
            return MARMOT_OP_ADD_SILU;
        }
    }

    if (ctx->current_op == MARMOT_OP_MATMUL && ctx->intermediate_is_temporary) {
        if (ctx->next_op == MARMOT_OP_ADD) {
            if (ctx->next_intermediate_is_temporary) {
                marmot_op_id_t activation_fused = matmul_bias_activation_op(ctx->next_next_op);
                if (activation_fused != MARMOT_OP_INVALID) {
                    return activation_fused;
                }
            }
            return MARMOT_OP_MATMUL_BIAS;
        }
    }

    if (ctx->current_op == MARMOT_OP_MUL && ctx->intermediate_is_temporary) {
        if (ctx->next_op == MARMOT_OP_ADD) {
            return MARMOT_OP_FMA;
        }
    }

    return MARMOT_OP_INVALID;
}

marmot_fusion_pattern_t marmot_detect_fusion_pattern(const marmot_fusion_context_t *ctx) {
    marmot_op_id_t fused = marmot_detect_fused_op_id(ctx);
    switch (fused) {
    case MARMOT_OP_ADD_RELU:
        return MARMOT_FUSION_PATTERN_ADD_RELU;
    case MARMOT_OP_ADD_GELU:
        return MARMOT_FUSION_PATTERN_ADD_GELU;
    case MARMOT_OP_ADD_SILU:
        return MARMOT_FUSION_PATTERN_ADD_SILU;
    case MARMOT_OP_MATMUL_BIAS:
        return MARMOT_FUSION_PATTERN_MATMUL_BIAS;
    case MARMOT_OP_MATMUL_BIAS_RELU:
        return MARMOT_FUSION_PATTERN_MATMUL_BIAS_RELU;
    case MARMOT_OP_MATMUL_BIAS_GELU:
        return MARMOT_FUSION_PATTERN_MATMUL_BIAS_GELU;
    case MARMOT_OP_MATMUL_BIAS_SILU:
        return MARMOT_FUSION_PATTERN_MATMUL_BIAS_SILU;
    default:
        break;
    }

    return MARMOT_FUSION_PATTERN_NONE;
}

bool marmot_backend_supports_fusion(marmot_backend_type_t backend_type, marmot_fusion_flags_t flags) {
    if (flags == MARMOT_FUSION_NONE) {
        return true;
    }

    uint32_t supported = 0;
    switch (backend_type) {
    case MARMOT_BACKEND_CPU:
        supported = MARMOT_FUSION_RESIDUAL_ADD;
        break;
    case MARMOT_BACKEND_METAL:
#if MARMOT_ENABLE_METAL
        supported = MARMOT_FUSION_RESIDUAL_ADD;
#endif
        break;
    default:
        supported = 0;
        break;
    }

    return (supported & flags) != 0;
}
