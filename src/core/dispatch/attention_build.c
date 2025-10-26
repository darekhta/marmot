#include "marmot/config.h"
#include "marmot/error.h"
#include "marmot/graph/op_signature.h"
#include "marmot/ops/paged_attention.h"

#include "core/dispatch/fusion_flags.h"
#include "core/tensor/tensor_utils.h"
#include "graph/kernel_dispatch_args.gen.h"

static bool marmot_paged_attention_activation_dtype_supported(marmot_dtype_t dtype) {
    return dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_BFLOAT16;
}

static bool marmot_paged_attention_kv_dtype_supported(marmot_dtype_t dtype) {
    if (dtype == MARMOT_DTYPE_FLOAT32 || dtype == MARMOT_DTYPE_FLOAT16 || dtype == MARMOT_DTYPE_BFLOAT16) {
        return true;
    }
#if MARMOT_ENABLE_FP8
    return dtype == MARMOT_DTYPE_FLOAT8_E4M3;
#else
    (void)dtype;
    return false;
#endif
}

marmot_error_t marmot_paged_attention_build(
    const marmot_context_t *ctx, const marmot_paged_attention_desc_t *desc, marmot_op_signature_t *sig_out,
    marmot_kernel_args_paged_attention_t *packed_out
) {
    if (ctx == nullptr || desc == nullptr || sig_out == nullptr || packed_out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention requires context and descriptor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!marmot_paged_attention_desc_is_valid(desc)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention descriptor invalid");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->num_q_heads == 0 || desc->num_kv_heads == 0 || desc->head_dim == 0 || desc->block_size == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention requires non-zero dimensions");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (!marmot_is_power_of_two_u32(desc->block_size)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention block_size must be power of two");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->num_q_heads % desc->num_kv_heads != 0) {
        marmot_set_error(
            MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention requires num_q_heads divisible by num_kv_heads"
        );
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *token_meta = desc->token_meta;
    const marmot_tensor_t *q = desc->q;
    const marmot_tensor_t *k_new = desc->k_new;
    const marmot_tensor_t *v_new = desc->v_new;
    const marmot_tensor_t *kv_k = desc->kv_k;
    const marmot_tensor_t *kv_v = desc->kv_v;
    const marmot_tensor_t *block_table = desc->block_table;
    const marmot_tensor_t *out = desc->out;
    const marmot_paged_attention_kv_scale_ext_t *scale_ext =
        (desc->pnext != nullptr) ? (const marmot_paged_attention_kv_scale_ext_t *)desc->pnext : nullptr;
    const marmot_tensor_t *kv_k_scale = nullptr;
    const marmot_tensor_t *kv_v_scale = nullptr;
    if (scale_ext != nullptr && scale_ext->struct_version == MARMOT_PAGED_ATTENTION_KV_SCALE_EXT_VERSION &&
        scale_ext->struct_size >= sizeof(*scale_ext)) {
        kv_k_scale = scale_ext->kv_k_scale;
        kv_v_scale = scale_ext->kv_v_scale;
    }

    if (token_meta->dtype != MARMOT_DTYPE_UINT32 || block_table->dtype != MARMOT_DTYPE_UINT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention requires uint32 metadata");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (!marmot_paged_attention_activation_dtype_supported(q->dtype) ||
        !marmot_paged_attention_activation_dtype_supported(k_new->dtype) ||
        !marmot_paged_attention_activation_dtype_supported(v_new->dtype) ||
        !marmot_paged_attention_kv_dtype_supported(kv_k->dtype) ||
        !marmot_paged_attention_kv_dtype_supported(kv_v->dtype) ||
        !marmot_paged_attention_activation_dtype_supported(out->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention dtype not supported");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (q->dtype != k_new->dtype || q->dtype != v_new->dtype || q->dtype != out->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention requires matching q/k_new/v_new/out dtypes");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (kv_k->dtype != kv_v->dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention requires matching kv_k/kv_v dtypes");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
#if MARMOT_ENABLE_FP8
    if (kv_k->dtype == MARMOT_DTYPE_FLOAT8_E5M2) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention FP8 E5M2 not supported");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (kv_k->dtype == MARMOT_DTYPE_FLOAT8_E4M3) {
        if (kv_k_scale == nullptr || kv_v_scale == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention FP8 KV requires scale tensors");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (kv_k_scale->dtype != MARMOT_DTYPE_FLOAT32 || kv_v_scale->dtype != MARMOT_DTYPE_FLOAT32) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "paged_attention FP8 scale dtype must be float32");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (kv_k_scale->shape.ndim != 3 || kv_v_scale->shape.ndim != 3) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention FP8 scale tensors must be 3D");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (kv_k_scale->shape.shape[0] != kv_k->shape.shape[0] || kv_k_scale->shape.shape[1] != kv_k->shape.shape[1] ||
            kv_k_scale->shape.shape[2] != kv_k->shape.shape[2]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention FP8 scale tensor shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        if (kv_v_scale->shape.shape[0] != kv_v->shape.shape[0] || kv_v_scale->shape.shape[1] != kv_v->shape.shape[1] ||
            kv_v_scale->shape.shape[2] != kv_v->shape.shape[2]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "paged_attention FP8 scale tensor shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    } else
#endif
        if (kv_k_scale != nullptr || kv_v_scale != nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention scale tensors require FP8 KV");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_PAGED_ATTENTION,
        .profile_id = MARMOT_PROFILE_INVALID,
        .input_dtype = q->dtype,
        .weight_dtype = kv_k->dtype,
        .output_dtype = out->dtype,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .qscheme_id = MARMOT_QSCHEME_NONE,
        .quant_block = {0},
        .weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID,
        .epilogue_flags = MARMOT_EPILOGUE_NONE,
        .activation = MARMOT_DEVICE_UNARY_COUNT,
        .variant_flags = MARMOT_FUSION_NONE,
        .dims = {.elementwise = {.n_elems = (uint32_t)marmot_tensor_num_elements(q)}},
    };

    *packed_out = (marmot_kernel_args_paged_attention_t){
        .ctx = ctx,
        .token_meta = token_meta,
        .q = q,
        .k_new = k_new,
        .v_new = v_new,
        .kv_k = (marmot_tensor_t *)kv_k,
        .kv_v = (marmot_tensor_t *)kv_v,
        .block_table = block_table,
        .kv_k_scale = (marmot_tensor_t *)kv_k_scale,
        .kv_v_scale = (marmot_tensor_t *)kv_v_scale,
        .out = (marmot_tensor_t *)out,
        .token_count = desc->token_count,
        .layer_idx = desc->layer_idx,
        .num_q_heads = desc->num_q_heads,
        .num_kv_heads = desc->num_kv_heads,
        .head_dim = desc->head_dim,
        .block_size = desc->block_size,
        .scale = desc->scale,
    };
    *sig_out = sig;
    return MARMOT_SUCCESS;
}
