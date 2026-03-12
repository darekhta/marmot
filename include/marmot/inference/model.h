#ifndef MARMOT_INFERENCE_MODEL_H
#define MARMOT_INFERENCE_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "../error.h"
#include "../macros.h"
#include "../types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MARMOT_MODEL_OPTIONS_VERSION 1

typedef enum {
    MARMOT_MODEL_FLAG_STRICT_VALIDATION = 1u << 0,
} marmot_model_flags_t;

typedef struct marmot_model marmot_model_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;
    const void *pnext;
    uint64_t reserved[4];
} marmot_model_options_t;

typedef struct {
    char architecture[32];
    size_t context_length;
    size_t n_vocab;
    size_t n_embd;
    size_t n_layer;
    size_t n_head;
    size_t n_head_kv;
    size_t ff_length;
    size_t rope_dimension;
    float rope_freq_base;
    marmot_rope_type_t rope_type;
    marmot_rope_scaling_type_t rope_scaling_type;
    float rope_freq_scale;
    float rope_ext_factor;
    float rope_attn_factor;
    float rope_beta_fast;
    float rope_beta_slow;
    uint32_t rope_orig_ctx_len;
    float rms_norm_eps;
    bool is_moe;
    size_t n_experts;
    size_t n_experts_used;
} marmot_model_info_t;

MARMOT_NODISCARD marmot_error_t marmot_model_options_init(marmot_model_options_t *opts);

MARMOT_NODISCARD marmot_error_t
marmot_model_load_file(const char *path, const marmot_model_options_t *opts, marmot_model_t **out_model);

void marmot_model_destroy(marmot_model_t *model);

MARMOT_NODISCARD marmot_error_t marmot_model_get_info(const marmot_model_t *model, marmot_model_info_t *out_info);

MARMOT_NODISCARD const marmot_error_info_t *marmot_model_last_error(const marmot_model_t *model);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_INFERENCE_MODEL_H
