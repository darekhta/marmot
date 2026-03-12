#include "model.hpp"

#include "marmot/graph/architecture.h"
#include "marmot/tensor.h"

#include <cmath>

namespace marmot::gguf {

std::optional<size_t> Model::read_u64(const char *key) const {
    const marmot_gguf_kv_t *kv = marmot_gguf_find_kv(gguf_, key);
    if (!kv)
        return std::nullopt;

    switch (kv->value.type) {
    case MARMOT_GGUF_TYPE_UINT32:
        return kv->value.data.uint32_value;
    case MARMOT_GGUF_TYPE_INT32:
        return static_cast<size_t>(kv->value.data.int32_value);
    case MARMOT_GGUF_TYPE_UINT64:
        return kv->value.data.uint64_value;
    case MARMOT_GGUF_TYPE_INT64:
        if (kv->value.data.int64_value < 0)
            return std::nullopt;
        return static_cast<size_t>(kv->value.data.int64_value);
    default:
        return std::nullopt;
    }
}

std::optional<float> Model::read_f32(const char *key) const {
    const marmot_gguf_kv_t *kv = marmot_gguf_find_kv(gguf_, key);
    if (!kv)
        return std::nullopt;

    if (kv->value.type == MARMOT_GGUF_TYPE_FLOAT32) {
        return kv->value.data.float32_value;
    }
    if (kv->value.type == MARMOT_GGUF_TYPE_FLOAT64) {
        return static_cast<float>(kv->value.data.float64_value);
    }
    return std::nullopt;
}

std::optional<std::string_view> Model::read_string(const char *key) const {
    const marmot_gguf_kv_t *kv = marmot_gguf_find_kv(gguf_, key);
    if (!kv || kv->value.type != MARMOT_GGUF_TYPE_STRING) {
        return std::nullopt;
    }
    return std::string_view(kv->value.data.string_value.data, kv->value.data.string_value.length);
}

static marmot_rope_scaling_type_t parse_rope_scaling_type(std::string_view name) {
    if (name == "none") {
        return MARMOT_ROPE_SCALING_NONE;
    }
    if (name == "linear") {
        return MARMOT_ROPE_SCALING_LINEAR;
    }
    if (name == "yarn") {
        return MARMOT_ROPE_SCALING_YARN;
    }
    if (name == "longrope") {
        return MARMOT_ROPE_SCALING_LONGROPE;
    }
    return MARMOT_ROPE_SCALING_NONE;
}

static std::optional<marmot_router_weight_policy_t> parse_router_weight_policy(std::string_view name, float scale) {
    if (name == "softmax") {
        if (scale != 1.0f) {
            return MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED_SCALED;
        }
        return MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED;
    }
    return std::nullopt;
}

static std::optional<marmot_router_weight_policy_t> parse_router_weight_policy(uint32_t kind, float scale) {
    switch (kind) {
    case 1:
        if (scale != 1.0f) {
            return MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED_SCALED;
        }
        return MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED;
    default:
        return std::nullopt;
    }
}

bool Model::get_metadata(marmot_gguf_model_meta_t *out) const {
    if (!out)
        return false;
    std::memset(out, 0, sizeof(*out));

    const char *arch_name = get_architecture(gguf_);
    marmot_architecture_t arch_id = marmot_architecture_from_string(arch_name);
    if (arch_id == MARMOT_ARCH_UNKNOWN) {
        return false;
    }

    const marmot_metadata_key_map_t *keys = marmot_get_metadata_keys(arch_id);
    if (keys == nullptr) {
        return false;
    }

    auto context_length = read_u64(keys->context_length);
    auto n_embd = read_u64(keys->embedding_length);
    auto n_layer = read_u64(keys->block_count);
    auto n_head = read_u64(keys->head_count);
    auto n_head_kv = read_u64(keys->head_count_kv);
    auto ff_length = read_u64(keys->ff_length);
    auto rope_dim = read_u64(keys->rope_dimension);
    auto rms_eps = read_f32(keys->rms_eps);
    auto rope_base = read_f32(keys->rope_base);

    if (!context_length || !n_embd || !n_layer || !n_head || !n_head_kv || !ff_length || !rms_eps) {
        return false;
    }

    // Get architecture traits for defaults
    const marmot_architecture_traits_t *arch_traits = marmot_get_architecture_traits(arch_id);
    if (arch_traits == nullptr) {
        return false;
    }

    // Read optional key_length (head dimension, used in Qwen3+)
    std::optional<size_t> key_length;
    if (keys->key_length != nullptr) {
        key_length = read_u64(keys->key_length);
    }

    // Compute head_dim: use key_length if present, otherwise compute from n_embd/n_head
    size_t head_dim = key_length.value_or(*n_embd / *n_head);

    // Compute rope_dimension if not present (common in Qwen2 GGUF files)
    size_t rope_dimension = rope_dim.value_or(head_dim);

    out->architecture = arch_id;

    out->context_length = *context_length;
    out->n_embd = *n_embd;
    out->n_layer = *n_layer;
    out->n_head = *n_head;
    out->n_head_kv = *n_head_kv;
    out->ff_length = *ff_length;
    out->n_experts = arch_traits->n_experts;
    out->n_experts_used = arch_traits->n_experts_used;
    out->n_shared_experts = arch_traits->n_shared_experts;
    out->rope_dimension = rope_dimension;
    out->head_dim = head_dim;
    out->rms_norm_eps = *rms_eps;
    out->rope_freq_base = rope_base.value_or(arch_traits->default_rope_base);
    out->expert_weights_scale = arch_traits->expert_weights_scale;
    out->rope_type = arch_traits->rope_type;
    out->router_weight_policy = arch_traits->router_weight_policy;
    // Gemma scales input embeddings by sqrt(n_embd) at runtime before the transformer stack.
    out->embedding_scale = arch_traits->embedding_scale_sqrt_dim ? std::sqrt(static_cast<float>(*n_embd)) : 1.0f;
    out->is_moe = arch_traits->is_moe;

    if (keys->expert_count != nullptr) {
        if (auto value = read_u64(keys->expert_count)) {
            out->n_experts = *value;
        }
    }
    if (keys->expert_used_count != nullptr) {
        if (auto value = read_u64(keys->expert_used_count)) {
            out->n_experts_used = *value;
        }
    }
    if (keys->shared_expert_count != nullptr) {
        if (auto value = read_u64(keys->shared_expert_count)) {
            out->n_shared_experts = *value;
        }
    }
    if (keys->expert_weights_scale != nullptr) {
        if (auto value = read_f32(keys->expert_weights_scale)) {
            out->expert_weights_scale = *value;
        }
    }
    if (keys->expert_gating_func != nullptr) {
        if (auto value = read_string(keys->expert_gating_func)) {
            if (auto policy = parse_router_weight_policy(*value, out->expert_weights_scale)) {
                out->router_weight_policy = *policy;
            }
        } else if (auto value = read_u64(keys->expert_gating_func)) {
            if (auto policy = parse_router_weight_policy((uint32_t)*value, out->expert_weights_scale)) {
                out->router_weight_policy = *policy;
            }
        }
    }

    std::string_view rope_scaling_name = "linear";
    if (auto rope_scaling_value = read_string(keys->rope_scaling_type)) {
        rope_scaling_name = *rope_scaling_value;
    }
    marmot_rope_scaling_type_t scaling_type = parse_rope_scaling_type(rope_scaling_name);

    float rope_scaling_factor = 0.0f;
    if (auto scaling_factor = read_f32(keys->rope_scaling_factor)) {
        rope_scaling_factor = *scaling_factor;
    }
    float rope_attn_factor = 1.0f;
    if (auto attn_factor = read_f32(keys->rope_attn_factor)) {
        rope_attn_factor = *attn_factor;
    }

    float rope_freq_scale = rope_scaling_factor == 0.0f ? 1.0f : 1.0f / rope_scaling_factor;
    if (scaling_type == MARMOT_ROPE_SCALING_NONE) {
        rope_freq_scale = 1.0f;
    }

    uint32_t rope_orig_ctx_len = static_cast<uint32_t>(*context_length);
    if (auto orig_ctx_len = read_u64(keys->rope_orig_ctx_len)) {
        rope_orig_ctx_len = static_cast<uint32_t>(*orig_ctx_len);
    }

    float yarn_log_multiplier = 0.0f;
    // Note: yarn_log_multiplier is llama-specific, not in key map
    if (auto yarn_log_mul = read_f32("llama.rope.scaling.yarn_log_multiplier")) {
        yarn_log_multiplier = *yarn_log_mul;
    }

    float yarn_ext_factor = -1.0f;
    if (auto ext_factor = read_f32(keys->rope_ext_factor)) {
        yarn_ext_factor = *ext_factor;
    }

    float yarn_attn_factor = 1.0f;
    if (auto attn_factor = read_f32(keys->rope_attn_factor)) {
        yarn_attn_factor = *attn_factor;
    }

    float yarn_beta_fast = 32.0f;
    if (auto beta_fast = read_f32(keys->rope_beta_fast)) {
        yarn_beta_fast = *beta_fast;
    }

    float yarn_beta_slow = 1.0f;
    if (auto beta_slow = read_f32(keys->rope_beta_slow)) {
        yarn_beta_slow = *beta_slow;
    }

    float ext_factor = yarn_ext_factor;
    if (ext_factor < 0.0f) {
        ext_factor = scaling_type == MARMOT_ROPE_SCALING_YARN ? 1.0f : 0.0f;
    }

    float attn_factor = yarn_attn_factor;
    if (ext_factor != 0.0f) {
        auto get_mscale = [](float scale, float mscale) -> float {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * std::logf(scale) + 1.0f);
        };

        const float factor = rope_freq_scale == 0.0f ? 1.0f : 1.0f / rope_freq_scale;
        if (yarn_log_multiplier != 0.0f) {
            const float mscale = 1.0f;
            const float mscale_all_dims = yarn_log_multiplier;
            attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);
        } else {
            attn_factor = get_mscale(factor, 1.0f);
        }

        attn_factor *= 1.0f / (1.0f + 0.1f * std::logf(factor));
    }
    attn_factor *= rope_attn_factor;

    out->rope_scaling_type = scaling_type;
    out->rope_freq_scale = rope_freq_scale;
    out->rope_ext_factor = ext_factor;
    out->rope_attn_factor = attn_factor;
    out->rope_beta_fast = yarn_beta_fast;
    out->rope_beta_slow = yarn_beta_slow;
    out->rope_orig_ctx_len = rope_orig_ctx_len;

    const marmot_gguf_kv_t *vocab_tokens = marmot_gguf_find_kv(gguf_, "tokenizer.ggml.tokens");
    if (vocab_tokens && vocab_tokens->value.type == MARMOT_GGUF_TYPE_ARRAY &&
        vocab_tokens->value.data.array_value.type == MARMOT_GGUF_TYPE_STRING) {
        out->n_vocab = vocab_tokens->value.data.array_value.length;
    } else {
        const marmot_gguf_tensor_t *output = marmot_gguf_find_tensor(gguf_, "output.weight");
        if (output && output->tensor && output->tensor->shape.ndim == 2) {
            size_t vocab_dim = output->tensor->shape.shape[0];
            if (output->tensor->shape.shape[1] > vocab_dim) {
                vocab_dim = output->tensor->shape.shape[1];
            }
            out->n_vocab = vocab_dim;
        }
    }

    return out->n_vocab > 0;
}

const char *get_architecture(const marmot_gguf_t *gguf) {
    const marmot_gguf_kv_t *arch = marmot_gguf_find_kv(gguf, "general.architecture");
    if (arch == nullptr || arch->value.type != MARMOT_GGUF_TYPE_STRING) {
        return nullptr;
    }
    return arch->value.data.string_value.data;
}

} // namespace marmot::gguf
