#pragma once

#include "marmot/device.h"
#include "marmot/graph/gguf_model.h"
#include "marmot/ops/matmul.h"
#include "marmot/tensor.h"

#include <cstring>

namespace marmot::inference {

extern "C" marmot_error_t marmot_cpu_pin_quant_weight_range(
    const marmot_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows
);

inline bool should_prepack_cpu_decode_tensor(const marmot_gguf_tensor_t *info, bool prefer_embed_fallback) noexcept {
    if (info == nullptr || info->name == nullptr || info->tensor == nullptr) {
        return false;
    }
    if (info->tensor->shape.ndim != 2) {
        return false;
    }
    if (!marmot_quant_kind_is_block_quantized(info->tensor->quant_kind)) {
        return false;
    }
    return std::strcmp(info->name, "output.weight") == 0 ||
        (prefer_embed_fallback && std::strcmp(info->name, "token_embd.weight") == 0);
}

inline bool tensor_name_has_suffix(const char *name, const char *suffix) noexcept {
    if (name == nullptr || suffix == nullptr) {
        return false;
    }
    const size_t name_len = std::strlen(name);
    const size_t suffix_len = std::strlen(suffix);
    return name_len >= suffix_len && std::memcmp(name + name_len - suffix_len, suffix, suffix_len) == 0;
}

inline bool should_prepack_cpu_attention_tensor(const marmot_gguf_tensor_t *info) noexcept {
    if (info == nullptr || info->name == nullptr || info->tensor == nullptr) {
        return false;
    }
    if (info->tensor->shape.ndim != 2) {
        return false;
    }
    if (!marmot_quant_kind_is_block_quantized(info->tensor->quant_kind)) {
        return false;
    }
    return tensor_name_has_suffix(info->name, ".attn_q.weight") ||
        tensor_name_has_suffix(info->name, ".attn_k.weight") || tensor_name_has_suffix(info->name, ".attn_v.weight") ||
        tensor_name_has_suffix(info->name, ".attn_output.weight");
}

struct CpuMoePrepackView {
    marmot_tensor_t tensor{};
    size_t rows_per_expert{0};
    size_t experts{0};
};

inline bool should_prepack_cpu_moe_bank(const marmot_gguf_tensor_t *info) noexcept {
    return info != nullptr && info->name != nullptr &&
        (tensor_name_has_suffix(info->name, ".ffn_gate_exps.weight") ||
         tensor_name_has_suffix(info->name, ".ffn_up_exps.weight"));
}

inline size_t quant_row_bytes(marmot_quant_kind_t kind, size_t cols) noexcept {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    if (traits == nullptr || traits->block_values == 0) {
        return 0;
    }
    const size_t block_bytes = traits->header_bytes + traits->payload_bytes;
    const size_t blocks_per_row = (cols + traits->block_values - 1) / traits->block_values;
    return blocks_per_row * block_bytes;
}

inline bool make_cpu_moe_prepack_view(const marmot_gguf_tensor_t *info, CpuMoePrepackView *out) noexcept {
    if (info == nullptr || out == nullptr || info->name == nullptr || info->tensor == nullptr) {
        return false;
    }
    const marmot_tensor_t *tensor = info->tensor;
    if (tensor->shape.ndim != 3 || !marmot_quant_kind_is_block_quantized(tensor->quant_kind)) {
        return false;
    }

    const bool is_moe_bank = tensor_name_has_suffix(info->name, ".ffn_gate_exps.weight") ||
        tensor_name_has_suffix(info->name, ".ffn_up_exps.weight") ||
        tensor_name_has_suffix(info->name, ".ffn_down_exps.weight");
    if (!is_moe_bank) {
        return false;
    }

    const size_t rows_per_expert = tensor->shape.shape[1];
    const size_t cols = tensor->shape.shape[0];
    const size_t experts = tensor->shape.shape[2];
    const size_t total_rows = rows_per_expert * experts;
    const size_t row_bytes = quant_row_bytes(tensor->quant_kind, cols);
    if (rows_per_expert == 0 || cols == 0 || experts == 0 || row_bytes == 0) {
        return false;
    }

    std::memset(out, 0, sizeof(*out));
    out->rows_per_expert = rows_per_expert;
    out->experts = experts;
    out->tensor.shape.ndim = 2;
    out->tensor.shape.shape[0] = total_rows;
    out->tensor.shape.shape[1] = cols;
    out->tensor.shape.strides[0] = cols;
    out->tensor.shape.strides[1] = 1;
    out->tensor.dtype = tensor->dtype;
    out->tensor.data = tensor->data;
    out->tensor.capacity_bytes = total_rows * row_bytes;
    out->tensor.owns_data = false;
    out->tensor.quant_kind = tensor->quant_kind;
    out->tensor.quant_layout = tensor->quant_layout;
    out->tensor.backend = tensor->backend;
    out->tensor.ctx = tensor->ctx;
    out->tensor.memory_location = tensor->memory_location;
    out->tensor.needs_sync = tensor->needs_sync;
    return true;
}

inline void prepack_cpu_model_weights(const marmot_context_t *ctx, const marmot_gguf_model_t *model, bool emit_logits) {
    if (ctx == nullptr || model == nullptr || ctx->backend_type != MARMOT_BACKEND_CPU) {
        return;
    }

    bool has_output_weight = false;
    const size_t tensor_count = marmot_gguf_model_tensor_count(model);
    for (size_t i = 0; i < tensor_count; ++i) {
        const marmot_gguf_tensor_t *info = marmot_gguf_model_tensor_info(model, i);
        if (info != nullptr && info->name != nullptr && std::strcmp(info->name, "output.weight") == 0) {
            has_output_weight = true;
            break;
        }
    }
    for (size_t i = 0; i < tensor_count; ++i) {
        const marmot_gguf_tensor_t *info = marmot_gguf_model_tensor_info(model, i);
        if (should_prepack_cpu_decode_tensor(info, !has_output_weight)) {
            if (emit_logits) {
                (void)marmot_matmul_prepack_quant_weight(ctx, info->tensor);
            }
            continue;
        }
        if (should_prepack_cpu_attention_tensor(info)) {
            (void)marmot_matmul_prepack_quant_weight(ctx, info->tensor);
            continue;
        }

        CpuMoePrepackView prepack_view{};
        if (!make_cpu_moe_prepack_view(info, &prepack_view)) {
            continue;
        }

        const size_t row_bytes = quant_row_bytes(prepack_view.tensor.quant_kind, prepack_view.tensor.shape.shape[1]);
        const size_t rows = prepack_view.tensor.shape.shape[0];
        const size_t bytes = rows * row_bytes;
        if (row_bytes == 0 || rows == 0 || bytes == 0) {
            continue;
        }
        if (should_prepack_cpu_moe_bank(info)) {
            const marmot_error_t status = marmot_matmul_prepack_quant_weight(ctx, &prepack_view.tensor);
            if (status == MARMOT_SUCCESS) {
                continue;
            }
        }
        (void)marmot_cpu_pin_quant_weight_range(ctx, prepack_view.tensor.data, bytes, row_bytes, rows);
    }
    marmot_clear_error();
}

inline void
prepack_cpu_decode_weights(const marmot_context_t *ctx, const marmot_gguf_model_t *model, bool emit_logits) {
    prepack_cpu_model_weights(ctx, model, emit_logits);
}

} // namespace marmot::inference
