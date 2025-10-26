#ifndef MARMOT_OPS_PAGED_ATTENTION_H
#define MARMOT_OPS_PAGED_ATTENTION_H

#include "../ops_types.h"
#include "../tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MARMOT_PAGED_ATTENTION_DESC_VERSION 1
#define MARMOT_PAGED_ATTENTION_KV_SCALE_EXT_VERSION 1

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    const marmot_tensor_t *kv_k_scale;
    const marmot_tensor_t *kv_v_scale;
} marmot_paged_attention_kv_scale_ext_t;

static inline marmot_paged_attention_kv_scale_ext_t marmot_paged_attention_kv_scale_ext_default(void) {
    marmot_paged_attention_kv_scale_ext_t ext = {
        .struct_size = sizeof(marmot_paged_attention_kv_scale_ext_t),
        .struct_version = MARMOT_PAGED_ATTENTION_KV_SCALE_EXT_VERSION,
        .kv_k_scale = nullptr,
        .kv_v_scale = nullptr,
    };
    return ext;
}

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;

    uint32_t token_count;
    uint32_t layer_idx;
    uint32_t num_q_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t block_size;

    float scale;

    const marmot_tensor_t *token_meta;
    const marmot_tensor_t *q;
    const marmot_tensor_t *k_new;
    const marmot_tensor_t *v_new;
    const marmot_tensor_t *kv_k;
    const marmot_tensor_t *kv_v;
    const marmot_tensor_t *block_table;

    marmot_tensor_t *out;

    const void *pnext;
    uint64_t reserved[4];
} marmot_paged_attention_desc_t;

static inline marmot_paged_attention_desc_t marmot_paged_attention_desc_default(void) {
    marmot_paged_attention_desc_t desc = {
        .struct_size = sizeof(marmot_paged_attention_desc_t),
        .struct_version = MARMOT_PAGED_ATTENTION_DESC_VERSION,
        .flags = 0,
        .token_count = 0,
        .layer_idx = 0,
        .num_q_heads = 0,
        .num_kv_heads = 0,
        .head_dim = 0,
        .block_size = 0,
        .scale = 1.0f,
        .token_meta = nullptr,
        .q = nullptr,
        .k_new = nullptr,
        .v_new = nullptr,
        .kv_k = nullptr,
        .kv_v = nullptr,
        .block_table = nullptr,
        .out = nullptr,
        .pnext = nullptr,
        .reserved = {0, 0, 0, 0},
    };
    return desc;
}

static inline bool marmot_paged_attention_desc_is_valid(const marmot_paged_attention_desc_t *desc) {
    if (desc == nullptr) {
        return false;
    }
    if (desc->struct_version != MARMOT_PAGED_ATTENTION_DESC_VERSION) {
        return false;
    }
    if (desc->struct_size < sizeof(marmot_paged_attention_desc_t)) {
        return false;
    }
    return desc->token_meta != nullptr && desc->q != nullptr && desc->k_new != nullptr && desc->v_new != nullptr &&
        desc->kv_k != nullptr && desc->kv_v != nullptr && desc->block_table != nullptr && desc->out != nullptr;
}

MARMOT_NODISCARD marmot_error_t
marmot_paged_attention(const marmot_context_t *ctx, const marmot_paged_attention_desc_t *desc);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_OPS_PAGED_ATTENTION_H
