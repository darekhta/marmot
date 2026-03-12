#include "model.hpp"

#include "marmot/error.h"
#include "marmot/graph/architecture.h"
#include "marmot/graph/gguf_loader.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <string_view>

namespace marmot::inference {

static bool ends_with_case_insensitive(std::string_view value, std::string_view suffix) noexcept {
    if (suffix.empty()) {
        return true;
    }
    if (value.size() < suffix.size()) {
        return false;
    }

    const size_t offset = value.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); ++i) {
        unsigned char a = (unsigned char)value[offset + i];
        unsigned char b = (unsigned char)suffix[i];
        if (a >= (unsigned char)'A' && a <= (unsigned char)'Z') {
            a = (unsigned char)(a - (unsigned char)'A' + (unsigned char)'a');
        }
        if (b >= (unsigned char)'A' && b <= (unsigned char)'Z') {
            b = (unsigned char)(b - (unsigned char)'A' + (unsigned char)'a');
        }
        if (a != b) {
            return false;
        }
    }
    return true;
}

static void copy_architecture(char *dst, size_t dst_len, const marmot_gguf_t *file) {
    if (dst == nullptr || dst_len == 0) {
        return;
    }

    dst[0] = '\0';
    if (file == nullptr) {
        return;
    }

    const marmot_gguf_kv_t *kv = marmot_gguf_find_kv(file, "general.architecture");
    if (kv == nullptr || kv->value.type != MARMOT_GGUF_TYPE_STRING || kv->value.data.string_value.data == nullptr) {
        return;
    }

    const marmot_gguf_string_t s = kv->value.data.string_value;
    if (s.data == nullptr || s.length == 0) {
        return;
    }

    const size_t n = std::min(s.length, dst_len - 1);
    std::memcpy(dst, s.data, n);
    dst[n] = '\0';
}

std::unique_ptr<Model>
Model::load_file(const char *path, const marmot_model_options_t &opts, marmot_error_t &status, std::string &error) {
    (void)opts;
    status = MARMOT_ERROR_INVALID_OPERATION;
    error.clear();

    if (path == nullptr) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "model path is null";
        return nullptr;
    }

    const std::string_view path_view(path);
    if (ends_with_case_insensitive(path_view, ".safetensors") || ends_with_case_insensitive(path_view, ".safetensor")) {
        status = MARMOT_ERROR_NOT_IMPLEMENTED;
        error = "safetensors models are not supported yet";
        return nullptr;
    }

    marmot_gguf_model_t *gguf_raw = nullptr;
    marmot_error_t load_status = marmot_gguf_model_load(path, MARMOT_BACKEND_CPU, &gguf_raw);
    if (load_status != MARMOT_SUCCESS || gguf_raw == nullptr) {
        status = load_status != MARMOT_SUCCESS ? load_status : marmot_get_last_error();
        const char *detail = marmot_get_last_error_detail();
        if (detail != nullptr && detail[0] != '\0') {
            error = detail;
        } else {
            error = "failed to load GGUF model";
        }
        return nullptr;
    }

    GgufOwner gguf(gguf_raw);

    marmot_gguf_model_meta_t meta{};
    if (!marmot_gguf_model_metadata(gguf.get(), &meta)) {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "GGUF model metadata missing or invalid";
        return nullptr;
    }

    marmot_model_info_t info{};
    copy_architecture(info.architecture, sizeof(info.architecture), marmot_gguf_model_file(gguf.get()));
    info.context_length = meta.context_length;
    info.n_vocab = meta.n_vocab;
    info.n_embd = meta.n_embd;
    info.n_layer = meta.n_layer;
    info.n_head = meta.n_head;
    info.n_head_kv = meta.n_head_kv;
    info.ff_length = meta.ff_length;
    info.rope_dimension = meta.rope_dimension;
    info.rope_freq_base = meta.rope_freq_base;
    info.rope_type = meta.rope_type;
    info.rope_scaling_type = meta.rope_scaling_type;
    info.rope_freq_scale = meta.rope_freq_scale;
    info.rope_ext_factor = meta.rope_ext_factor;
    info.rope_attn_factor = meta.rope_attn_factor;
    info.rope_beta_fast = meta.rope_beta_fast;
    info.rope_beta_slow = meta.rope_beta_slow;
    info.rope_orig_ctx_len = meta.rope_orig_ctx_len;
    info.rms_norm_eps = meta.rms_norm_eps;
    info.is_moe = meta.is_moe;
    info.n_experts = meta.n_experts;
    info.n_experts_used = meta.n_experts_used;

    if (info.architecture[0] == '\0') {
        status = MARMOT_ERROR_INVALID_ARGUMENT;
        error = "GGUF missing general.architecture";
        return nullptr;
    }

    if ((opts.flags & MARMOT_MODEL_FLAG_STRICT_VALIDATION) != 0) {
        marmot_architecture_t arch_id = marmot_architecture_from_string(info.architecture);
        if (arch_id == MARMOT_ARCH_UNKNOWN) {
            status = MARMOT_ERROR_NOT_IMPLEMENTED;
            error = std::string("unsupported architecture: ") + info.architecture;
            return nullptr;
        }
    }

    status = MARMOT_SUCCESS;
    return std::unique_ptr<Model>(new (std::nothrow) Model(path, std::move(gguf), info));
}

} // namespace marmot::inference
