#include "marmot/graph/architecture.h"

#include "marmot/device.h"

#include <cstring>

namespace {

constexpr marmot_architecture_traits_t kArchitectureTraits[] = {
    {
        .arch_id = MARMOT_ARCH_LLAMA,
        .name = "llama",
        .gguf_arch_name = "llama",
        .ffn_type = MARMOT_FFN_SWIGLU,
        .is_moe = false,
        .has_attention_bias = false,
        .has_qk_norm = false,
        .uses_gemma_norm = false, // GGUF converter pre-shifts Gemma norm weights by +1.
        .embedding_scale_sqrt_dim = false,
        .n_experts = 0,
        .n_experts_used = 0,
        .n_shared_experts = 0,
        .rope_type = MARMOT_ROPE_TYPE_NORM,
        .default_rope_base = 10000.0f,
        .expert_weights_scale = 1.0f,
        .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        .metadata_prefix = "llama",
        .metal_activation_dtype = MARMOT_DTYPE_FLOAT16,
    },
    {
        .arch_id = MARMOT_ARCH_MISTRAL,
        .name = "mistral",
        .gguf_arch_name = "llama",
        .ffn_type = MARMOT_FFN_SWIGLU,
        .is_moe = false,
        .has_attention_bias = false,
        .has_qk_norm = false,
        .uses_gemma_norm = false,
        .embedding_scale_sqrt_dim = false,
        .n_experts = 0,
        .n_experts_used = 0,
        .n_shared_experts = 0,
        .rope_type = MARMOT_ROPE_TYPE_NORM,
        .default_rope_base = 1000000.0f,
        .expert_weights_scale = 1.0f,
        .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        .metadata_prefix = "llama",
        .metal_activation_dtype = MARMOT_DTYPE_FLOAT16,
    },
    {
        .arch_id = MARMOT_ARCH_QWEN2,
        .name = "qwen2",
        .gguf_arch_name = "qwen2",
        .ffn_type = MARMOT_FFN_SWIGLU,
        .is_moe = false,
        .has_attention_bias = true,
        .has_qk_norm = false,
        .uses_gemma_norm = false,
        .embedding_scale_sqrt_dim = false,
        .n_experts = 0,
        .n_experts_used = 0,
        .n_shared_experts = 0,
        .rope_type = MARMOT_ROPE_TYPE_NEOX,
        .default_rope_base = 10000.0f,
        .expert_weights_scale = 1.0f,
        .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        .metadata_prefix = "qwen2",
        .metal_activation_dtype = MARMOT_DTYPE_FLOAT16,
    },
    {
        .arch_id = MARMOT_ARCH_PHI3,
        .name = "phi3",
        .gguf_arch_name = "phi3",
        .ffn_type = MARMOT_FFN_SWIGLU,
        .is_moe = false,
        .has_attention_bias = false,
        .has_qk_norm = false,
        .uses_gemma_norm = false,
        .embedding_scale_sqrt_dim = false,
        .n_experts = 0,
        .n_experts_used = 0,
        .n_shared_experts = 0,
        .rope_type = MARMOT_ROPE_TYPE_NEOX,
        .default_rope_base = 10000.0f,
        .expert_weights_scale = 1.0f,
        .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        .metadata_prefix = "phi3",
        .metal_activation_dtype = MARMOT_DTYPE_FLOAT16,
    },
    {
        .arch_id = MARMOT_ARCH_GEMMA,
        .name = "gemma",
        .gguf_arch_name = "gemma",
        .ffn_type = MARMOT_FFN_GEGLU,
        .is_moe = false,
        .has_attention_bias = false,
        .has_qk_norm = false,
        .uses_gemma_norm = false,
        .embedding_scale_sqrt_dim = true,
        .n_experts = 0,
        .n_experts_used = 0,
        .n_shared_experts = 0,
        .rope_type = MARMOT_ROPE_TYPE_NEOX,
        .default_rope_base = 10000.0f,
        .expert_weights_scale = 1.0f,
        .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        .metadata_prefix = "gemma",
        .metal_activation_dtype = MARMOT_DTYPE_FLOAT32, // Gemma needs F32 for numerical stability
    },
    {
        .arch_id = MARMOT_ARCH_QWEN3,
        .name = "qwen3",
        .gguf_arch_name = "qwen3",
        .ffn_type = MARMOT_FFN_SWIGLU,
        .is_moe = false,
        .has_attention_bias = false,
        .has_qk_norm = true,
        .uses_gemma_norm = false,
        .embedding_scale_sqrt_dim = false,
        .n_experts = 0,
        .n_experts_used = 0,
        .n_shared_experts = 0,
        .rope_type = MARMOT_ROPE_TYPE_NEOX,
        .default_rope_base = 10000.0f,
        .expert_weights_scale = 1.0f,
        .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED,
        .metadata_prefix = "qwen3",
        .metal_activation_dtype = MARMOT_DTYPE_FLOAT16,
    },
    {
        .arch_id = MARMOT_ARCH_QWEN3MOE,
        .name = "qwen3moe",
        .gguf_arch_name = "qwen3moe",
        .ffn_type = MARMOT_FFN_SWIGLU,
        .is_moe = true,
        .has_attention_bias = false,
        .has_qk_norm = true,
        .uses_gemma_norm = false,
        .embedding_scale_sqrt_dim = false,
        .n_experts = 0,
        .n_experts_used = 0,
        .n_shared_experts = 0,
        .rope_type = MARMOT_ROPE_TYPE_NEOX,
        .default_rope_base = 10000.0f,
        .expert_weights_scale = 1.0f,
        .router_weight_policy = MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED_SCALED,
        .metadata_prefix = "qwen3moe",
        .metal_activation_dtype = MARMOT_DTYPE_FLOAT32,
    },
};

constexpr size_t kArchitectureTraitsCount = sizeof(kArchitectureTraits) / sizeof(kArchitectureTraits[0]);

constexpr marmot_metadata_key_map_t kLlamaMetadataKeys = {
    .context_length = "llama.context_length",
    .embedding_length = "llama.embedding_length",
    .block_count = "llama.block_count",
    .head_count = "llama.attention.head_count",
    .head_count_kv = "llama.attention.head_count_kv",
    .ff_length = "llama.feed_forward_length",
    .rope_dimension = "llama.rope.dimension_count",
    .rms_eps = "llama.attention.layer_norm_rms_epsilon",
    .rope_base = "llama.rope.freq_base",
    .rope_scaling_type = "llama.rope.scaling.type",
    .rope_scaling_factor = "llama.rope.scaling.factor",
    .rope_attn_factor = "llama.rope.scaling.attn_factor",
    .rope_ext_factor = "llama.rope.scaling.yarn_ext_factor",
    .rope_beta_fast = "llama.rope.scaling.yarn_beta_fast",
    .rope_beta_slow = "llama.rope.scaling.yarn_beta_slow",
    .rope_orig_ctx_len = "llama.rope.scaling.original_context_length",
    .key_length = nullptr,
    .value_length = nullptr,
    .expert_count = nullptr,
    .expert_used_count = nullptr,
    .shared_expert_count = nullptr,
    .expert_weights_scale = nullptr,
    .expert_gating_func = nullptr,
};

constexpr marmot_metadata_key_map_t kQwen2MetadataKeys = {
    .context_length = "qwen2.context_length",
    .embedding_length = "qwen2.embedding_length",
    .block_count = "qwen2.block_count",
    .head_count = "qwen2.attention.head_count",
    .head_count_kv = "qwen2.attention.head_count_kv",
    .ff_length = "qwen2.feed_forward_length",
    .rope_dimension = "qwen2.rope.dimension_count",
    .rms_eps = "qwen2.attention.layer_norm_rms_epsilon",
    .rope_base = "qwen2.rope.freq_base",
    .rope_scaling_type = "qwen2.rope.scaling.type",
    .rope_scaling_factor = "qwen2.rope.scaling.factor",
    .rope_attn_factor = "qwen2.rope.scaling.attn_factor",
    .rope_ext_factor = "qwen2.rope.scaling.yarn_ext_factor",
    .rope_beta_fast = "qwen2.rope.scaling.yarn_beta_fast",
    .rope_beta_slow = "qwen2.rope.scaling.yarn_beta_slow",
    .rope_orig_ctx_len = "qwen2.rope.scaling.original_context_length",
    .key_length = nullptr,
    .value_length = nullptr,
    .expert_count = nullptr,
    .expert_used_count = nullptr,
    .shared_expert_count = nullptr,
    .expert_weights_scale = nullptr,
    .expert_gating_func = nullptr,
};

constexpr marmot_metadata_key_map_t kPhi3MetadataKeys = {
    .context_length = "phi3.context_length",
    .embedding_length = "phi3.embedding_length",
    .block_count = "phi3.block_count",
    .head_count = "phi3.attention.head_count",
    .head_count_kv = "phi3.attention.head_count_kv",
    .ff_length = "phi3.feed_forward_length",
    .rope_dimension = "phi3.rope.dimension_count",
    .rms_eps = "phi3.attention.layer_norm_rms_epsilon",
    .rope_base = "phi3.rope.freq_base",
    .rope_scaling_type = "phi3.rope.scaling.type",
    .rope_scaling_factor = "phi3.rope.scaling.factor",
    .rope_attn_factor = "phi3.rope.scaling.attn_factor",
    .rope_ext_factor = "phi3.rope.scaling.yarn_ext_factor",
    .rope_beta_fast = "phi3.rope.scaling.yarn_beta_fast",
    .rope_beta_slow = "phi3.rope.scaling.yarn_beta_slow",
    .rope_orig_ctx_len = "phi3.rope.scaling.original_context_length",
    .key_length = nullptr,
    .value_length = nullptr,
    .expert_count = nullptr,
    .expert_used_count = nullptr,
    .shared_expert_count = nullptr,
    .expert_weights_scale = nullptr,
    .expert_gating_func = nullptr,
};

constexpr marmot_metadata_key_map_t kGemmaMetadataKeys = {
    .context_length = "gemma.context_length",
    .embedding_length = "gemma.embedding_length",
    .block_count = "gemma.block_count",
    .head_count = "gemma.attention.head_count",
    .head_count_kv = "gemma.attention.head_count_kv",
    .ff_length = "gemma.feed_forward_length",
    .rope_dimension = "gemma.rope.dimension_count",
    .rms_eps = "gemma.attention.layer_norm_rms_epsilon",
    .rope_base = "gemma.rope.freq_base",
    .rope_scaling_type = "gemma.rope.scaling.type",
    .rope_scaling_factor = "gemma.rope.scaling.factor",
    .rope_attn_factor = "gemma.rope.scaling.attn_factor",
    .rope_ext_factor = "gemma.rope.scaling.yarn_ext_factor",
    .rope_beta_fast = "gemma.rope.scaling.yarn_beta_fast",
    .rope_beta_slow = "gemma.rope.scaling.yarn_beta_slow",
    .rope_orig_ctx_len = "gemma.rope.scaling.original_context_length",
    .key_length = "gemma.attention.key_length",
    .value_length = "gemma.attention.value_length",
    .expert_count = nullptr,
    .expert_used_count = nullptr,
    .shared_expert_count = nullptr,
    .expert_weights_scale = nullptr,
    .expert_gating_func = nullptr,
};

constexpr marmot_metadata_key_map_t kQwen3MetadataKeys = {
    .context_length = "qwen3.context_length",
    .embedding_length = "qwen3.embedding_length",
    .block_count = "qwen3.block_count",
    .head_count = "qwen3.attention.head_count",
    .head_count_kv = "qwen3.attention.head_count_kv",
    .ff_length = "qwen3.feed_forward_length",
    .rope_dimension = "qwen3.rope.dimension_count",
    .rms_eps = "qwen3.attention.layer_norm_rms_epsilon",
    .rope_base = "qwen3.rope.freq_base",
    .rope_scaling_type = "qwen3.rope.scaling.type",
    .rope_scaling_factor = "qwen3.rope.scaling.factor",
    .rope_attn_factor = "qwen3.rope.scaling.attn_factor",
    .rope_ext_factor = "qwen3.rope.scaling.yarn_ext_factor",
    .rope_beta_fast = "qwen3.rope.scaling.yarn_beta_fast",
    .rope_beta_slow = "qwen3.rope.scaling.yarn_beta_slow",
    .rope_orig_ctx_len = "qwen3.rope.scaling.original_context_length",
    .key_length = "qwen3.attention.key_length",
    .value_length = "qwen3.attention.value_length",
    .expert_count = nullptr,
    .expert_used_count = nullptr,
    .shared_expert_count = nullptr,
    .expert_weights_scale = nullptr,
    .expert_gating_func = nullptr,
};

constexpr marmot_metadata_key_map_t kQwen3MoeMetadataKeys = {
    .context_length = "qwen3moe.context_length",
    .embedding_length = "qwen3moe.embedding_length",
    .block_count = "qwen3moe.block_count",
    .head_count = "qwen3moe.attention.head_count",
    .head_count_kv = "qwen3moe.attention.head_count_kv",
    .ff_length = "qwen3moe.feed_forward_length",
    .rope_dimension = "qwen3moe.rope.dimension_count",
    .rms_eps = "qwen3moe.attention.layer_norm_rms_epsilon",
    .rope_base = "qwen3moe.rope.freq_base",
    .rope_scaling_type = "qwen3moe.rope.scaling.type",
    .rope_scaling_factor = "qwen3moe.rope.scaling.factor",
    .rope_attn_factor = "qwen3moe.rope.scaling.attn_factor",
    .rope_ext_factor = "qwen3moe.rope.scaling.yarn_ext_factor",
    .rope_beta_fast = "qwen3moe.rope.scaling.yarn_beta_fast",
    .rope_beta_slow = "qwen3moe.rope.scaling.yarn_beta_slow",
    .rope_orig_ctx_len = "qwen3moe.rope.scaling.original_context_length",
    .key_length = "qwen3moe.attention.key_length",
    .value_length = "qwen3moe.attention.value_length",
    .expert_count = "qwen3moe.expert_count",
    .expert_used_count = "qwen3moe.expert_used_count",
    .shared_expert_count = "qwen3moe.expert_shared_count",
    .expert_weights_scale = "qwen3moe.expert_weights_scale",
    .expert_gating_func = "qwen3moe.expert_gating_func",
};

const marmot_metadata_key_map_t *kMetadataKeyMaps[MARMOT_ARCH_COUNT] = {
    nullptr,                // MARMOT_ARCH_UNKNOWN
    &kLlamaMetadataKeys,    // MARMOT_ARCH_LLAMA
    &kLlamaMetadataKeys,    // MARMOT_ARCH_MISTRAL (uses llama keys)
    &kQwen2MetadataKeys,    // MARMOT_ARCH_QWEN2
    &kPhi3MetadataKeys,     // MARMOT_ARCH_PHI3
    &kGemmaMetadataKeys,    // MARMOT_ARCH_GEMMA
    &kQwen3MetadataKeys,    // MARMOT_ARCH_QWEN3
    &kQwen3MoeMetadataKeys, // MARMOT_ARCH_QWEN3MOE
};

} // namespace

extern "C" {

marmot_architecture_t marmot_architecture_from_string(const char *name) {
    if (name == nullptr) {
        return MARMOT_ARCH_UNKNOWN;
    }
    for (size_t i = 0; i < kArchitectureTraitsCount; ++i) {
        if (std::strcmp(kArchitectureTraits[i].gguf_arch_name, name) == 0) {
            return kArchitectureTraits[i].arch_id;
        }
    }
    return MARMOT_ARCH_UNKNOWN;
}

const char *marmot_architecture_to_string(marmot_architecture_t arch) {
    for (size_t i = 0; i < kArchitectureTraitsCount; ++i) {
        if (kArchitectureTraits[i].arch_id == arch) {
            return kArchitectureTraits[i].name;
        }
    }
    return "unknown";
}

const marmot_architecture_traits_t *marmot_get_architecture_traits(marmot_architecture_t arch) {
    for (size_t i = 0; i < kArchitectureTraitsCount; ++i) {
        if (kArchitectureTraits[i].arch_id == arch) {
            return &kArchitectureTraits[i];
        }
    }
    return nullptr;
}

const marmot_metadata_key_map_t *marmot_get_metadata_keys(marmot_architecture_t arch) {
    if (arch >= MARMOT_ARCH_COUNT) {
        return nullptr;
    }
    return kMetadataKeyMaps[arch];
}

marmot_dtype_t marmot_activation_dtype_for_architecture(marmot_architecture_t arch, marmot_backend_type_t backend) {
    if (backend != MARMOT_BACKEND_METAL) {
        return MARMOT_DTYPE_FLOAT32;
    }
    const marmot_architecture_traits_t *traits = marmot_get_architecture_traits(arch);
    if (traits == nullptr) {
        return MARMOT_DTYPE_FLOAT16;
    }
    return traits->metal_activation_dtype;
}

} // extern "C"
