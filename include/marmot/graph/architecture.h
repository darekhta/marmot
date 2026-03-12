#ifndef MARMOT_GRAPH_ARCHITECTURE_H
#define MARMOT_GRAPH_ARCHITECTURE_H

#include "marmot/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    marmot_architecture_t arch_id;
    const char *name;
    const char *gguf_arch_name;
    marmot_ffn_type_t ffn_type;
    bool is_moe;
    bool has_attention_bias;
    bool has_qk_norm;
    bool uses_gemma_norm;
    bool embedding_scale_sqrt_dim;
    uint32_t n_experts;
    uint32_t n_experts_used;
    uint32_t n_shared_experts;
    marmot_rope_type_t rope_type;
    float default_rope_base;
    float expert_weights_scale;
    marmot_router_weight_policy_t router_weight_policy;
    const char *metadata_prefix;
    marmot_dtype_t metal_activation_dtype; // F16 for most, F32 for Gemma
} marmot_architecture_traits_t;

typedef struct {
    const char *context_length;
    const char *embedding_length;
    const char *block_count;
    const char *head_count;
    const char *head_count_kv;
    const char *ff_length;
    const char *rope_dimension;
    const char *rms_eps;
    const char *rope_base;
    const char *rope_scaling_type;
    const char *rope_scaling_factor;
    const char *rope_attn_factor;
    const char *rope_ext_factor;
    const char *rope_beta_fast;
    const char *rope_beta_slow;
    const char *rope_orig_ctx_len;
    const char *key_length;   // head dimension for keys (optional, Qwen3+)
    const char *value_length; // head dimension for values (optional, Qwen3+)
    const char *expert_count;
    const char *expert_used_count;
    const char *shared_expert_count;
    const char *expert_weights_scale;
    const char *expert_gating_func;
} marmot_metadata_key_map_t;

marmot_architecture_t marmot_architecture_from_string(const char *name);
const char *marmot_architecture_to_string(marmot_architecture_t arch);
const marmot_architecture_traits_t *marmot_get_architecture_traits(marmot_architecture_t arch);
const marmot_metadata_key_map_t *marmot_get_metadata_keys(marmot_architecture_t arch);
marmot_dtype_t marmot_activation_dtype_for_architecture(marmot_architecture_t arch, marmot_backend_type_t backend);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_GRAPH_ARCHITECTURE_H
