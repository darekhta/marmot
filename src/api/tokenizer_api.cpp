#include "marmot/error.h"
#include "marmot/tokenizer.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer/tokenizer.hpp"

namespace {

using marmot::tokenizer::Tokenizer;

struct MarmotTokenizerHandle {
    std::unique_ptr<Tokenizer> impl;
    mutable std::mutex last_error_mutex{};
    mutable marmot_error_info_t last_error{};
};

using Handle = MarmotTokenizerHandle;

[[nodiscard]] Handle *from_api(marmot_tokenizer_t *ptr) {
    return reinterpret_cast<Handle *>(ptr);
}

[[nodiscard]] const Handle *from_api(const marmot_tokenizer_t *ptr) {
    return reinterpret_cast<const Handle *>(ptr);
}

void set_last_error(Handle *handle, marmot_error_t code, std::string_view message) {
    if (handle == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(handle->last_error_mutex);
    handle->last_error = marmot_error_info_t{
        .code = code,
        .message = {0},
        .file = nullptr,
        .line = 0,
        .function = nullptr,
    };

    if (!message.empty()) {
        const size_t n = std::min(message.size(), sizeof(handle->last_error.message) - 1);
        std::memcpy(handle->last_error.message, message.data(), n);
        handle->last_error.message[n] = '\0';
    }
}

void set_success(Handle *handle) {
    set_last_error(handle, MARMOT_SUCCESS, {});
}

[[nodiscard]] marmot_error_t validate_options(const marmot_tokenizer_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_TOKENIZER_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_tokenizer_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

[[nodiscard]] marmot_error_t validate_encode_options(const marmot_tokenizer_encode_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_TOKENIZER_ENCODE_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_tokenizer_encode_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

[[nodiscard]] marmot_error_t validate_decode_options(const marmot_tokenizer_decode_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_version != MARMOT_TOKENIZER_DECODE_OPTIONS_VERSION) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (opts->struct_size < sizeof(marmot_tokenizer_decode_options_t)) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

} // namespace

extern "C" {

marmot_error_t marmot_tokenizer_options_init(marmot_tokenizer_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    std::memset(opts, 0, sizeof(*opts));
    opts->struct_size = sizeof(marmot_tokenizer_options_t);
    opts->struct_version = MARMOT_TOKENIZER_OPTIONS_VERSION;
    opts->flags = MARMOT_TOKENIZER_FLAG_STRICT_VALIDATION;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tokenizer_encode_options_init(marmot_tokenizer_encode_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    std::memset(opts, 0, sizeof(*opts));
    opts->struct_size = sizeof(marmot_tokenizer_encode_options_t);
    opts->struct_version = MARMOT_TOKENIZER_ENCODE_OPTIONS_VERSION;
    opts->add_bos = false;
    opts->add_eos = false;
    opts->allow_special = false;
    opts->max_tokens = 0;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tokenizer_decode_options_init(marmot_tokenizer_decode_options_t *opts) {
    if (opts == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    std::memset(opts, 0, sizeof(*opts));
    opts->struct_size = sizeof(marmot_tokenizer_decode_options_t);
    opts->struct_version = MARMOT_TOKENIZER_DECODE_OPTIONS_VERSION;
    opts->skip_special = false;
    opts->strip_space_prefix = true;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tokenizer_create_from_gguf_file(
    const char *path, const marmot_tokenizer_options_t *opts, marmot_tokenizer_t **out_tokenizer
) {
    if (out_tokenizer == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_tokenizer = nullptr;

    marmot_error_t status = validate_options(opts);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    auto handle = std::unique_ptr<Handle>(new (std::nothrow) Handle());
    if (!handle) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_error_t init_status = MARMOT_ERROR_INVALID_OPERATION;
    std::string error;

    handle->impl = Tokenizer::from_gguf_file(path, *opts, init_status, error);
    if (!handle->impl || init_status != MARMOT_SUCCESS) {
        if (error.empty()) {
            error = "failed to create tokenizer";
        }
        marmot_set_error(init_status, error.c_str());
        set_last_error(handle.get(), init_status, error);
        return init_status;
    }

    set_success(handle.get());
    *out_tokenizer = reinterpret_cast<marmot_tokenizer_t *>(handle.release());
    return MARMOT_SUCCESS;
}

void marmot_tokenizer_destroy(marmot_tokenizer_t *tokenizer) {
    delete from_api(tokenizer);
}

size_t marmot_tokenizer_vocab_size(const marmot_tokenizer_t *tokenizer) {
    const Handle *handle = from_api(tokenizer);
    if (handle == nullptr || handle->impl == nullptr) {
        return 0;
    }
    return handle->impl->vocab_size();
}

marmot_error_t
marmot_tokenizer_get_special_ids(const marmot_tokenizer_t *tokenizer, marmot_tokenizer_special_ids_t *out_special_ids) {
    const Handle *handle = from_api(tokenizer);
    if (handle == nullptr || handle->impl == nullptr || out_special_ids == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    *out_special_ids = handle->impl->special_ids();
    return MARMOT_SUCCESS;
}

marmot_error_t
marmot_tokenizer_get_behavior(const marmot_tokenizer_t *tokenizer, marmot_tokenizer_behavior_t *out_behavior) {
    const Handle *handle = from_api(tokenizer);
    if (handle == nullptr || handle->impl == nullptr || out_behavior == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    *out_behavior = handle->impl->behavior();
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tokenizer_piece_to_token(
    const marmot_tokenizer_t *tokenizer, const char *piece, size_t piece_len, marmot_token_id_t *out_token_id
) {
    const Handle *handle = from_api(tokenizer);
    if (handle == nullptr || handle->impl == nullptr || out_token_id == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (piece == nullptr && piece_len != 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_token_id_t id = MARMOT_TOKEN_ID_INVALID;
    marmot_error_t status = handle->impl->piece_to_token(std::string_view(piece ? piece : "", piece_len), id);
    if (status != MARMOT_SUCCESS) {
        marmot_set_error(status, "piece not in vocab");
        set_last_error(const_cast<Handle *>(handle), status, "piece not in vocab");
        return status;
    }

    set_success(const_cast<Handle *>(handle));
    *out_token_id = id;
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tokenizer_token_to_piece(
    const marmot_tokenizer_t *tokenizer, marmot_token_id_t token_id, char *out_piece, size_t *inout_len
) {
    const Handle *handle = from_api(tokenizer);
    if (handle == nullptr || handle->impl == nullptr || inout_len == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const std::string_view piece = handle->impl->token_to_piece(token_id);
    if (piece.data() == nullptr && piece.size() > 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "token has null piece");
        set_last_error(const_cast<Handle *>(handle), MARMOT_ERROR_INVALID_OPERATION, "token has null piece");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    const size_t required = piece.size() + 1;
    if (out_piece == nullptr) {
        *inout_len = required;
        set_success(const_cast<Handle *>(handle));
        return MARMOT_SUCCESS;
    }
    if (*inout_len < required) {
        *inout_len = required;
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "buffer too small");
        set_last_error(const_cast<Handle *>(handle), MARMOT_ERROR_INVALID_ARGUMENT, "buffer too small");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!piece.empty()) {
        std::memcpy(out_piece, piece.data(), piece.size());
    }
    out_piece[piece.size()] = '\0';
    *inout_len = required;
    set_success(const_cast<Handle *>(handle));
    return MARMOT_SUCCESS;
}

marmot_error_t
marmot_tokenizer_chat_template(const marmot_tokenizer_t *tokenizer, char *out_template, size_t *inout_len) {
    const Handle *handle = from_api(tokenizer);
    if (handle == nullptr || handle->impl == nullptr || inout_len == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const std::string_view tpl = handle->impl->chat_template();
    const size_t required = tpl.size() + 1;

    if (out_template == nullptr) {
        *inout_len = required;
        set_success(const_cast<Handle *>(handle));
        return MARMOT_SUCCESS;
    }
    if (*inout_len < required) {
        *inout_len = required;
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "buffer too small");
        set_last_error(const_cast<Handle *>(handle), MARMOT_ERROR_INVALID_ARGUMENT, "buffer too small");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!tpl.empty()) {
        std::memcpy(out_template, tpl.data(), tpl.size());
    }
    out_template[tpl.size()] = '\0';
    *inout_len = required;
    set_success(const_cast<Handle *>(handle));
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tokenizer_encode(
    const marmot_tokenizer_t *tokenizer, const char *text, size_t text_len,
    const marmot_tokenizer_encode_options_t *opts, marmot_token_id_t *out_token_ids, size_t *inout_len
) {
    Handle *handle = from_api(const_cast<marmot_tokenizer_t *>(tokenizer));
    if (handle == nullptr || handle->impl == nullptr || inout_len == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (text == nullptr && text_len != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "text is null");
        set_last_error(handle, MARMOT_ERROR_INVALID_ARGUMENT, "text is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_tokenizer_encode_options_t local_opts{};
    if (opts == nullptr) {
        marmot_error_t init_status = marmot_tokenizer_encode_options_init(&local_opts);
        if (init_status != MARMOT_SUCCESS) {
            return init_status;
        }
        opts = &local_opts;
    }
    marmot_error_t opts_status = validate_encode_options(opts);
    if (opts_status != MARMOT_SUCCESS) {
        marmot_set_error(opts_status, "invalid encode options");
        set_last_error(handle, opts_status, "invalid encode options");
        return opts_status;
    }

    std::vector<marmot_token_id_t> tokens;
    std::string error;
    marmot_error_t status = handle->impl->encode(std::string_view(text ? text : "", text_len), *opts, tokens, error);
    if (status != MARMOT_SUCCESS) {
        if (error.empty()) {
            error = "encode failed";
        }
        marmot_set_error(status, error.c_str());
        set_last_error(handle, status, error);
        return status;
    }

    const size_t required = tokens.size();
    if (out_token_ids == nullptr) {
        *inout_len = required;
        set_success(handle);
        return MARMOT_SUCCESS;
    }
    if (*inout_len < required) {
        *inout_len = required;
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "buffer too small");
        set_last_error(handle, MARMOT_ERROR_INVALID_ARGUMENT, "buffer too small");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!tokens.empty()) {
        std::memcpy(out_token_ids, tokens.data(), tokens.size() * sizeof(tokens[0]));
    }
    *inout_len = required;
    set_success(handle);
    return MARMOT_SUCCESS;
}

marmot_error_t marmot_tokenizer_decode(
    const marmot_tokenizer_t *tokenizer, const marmot_token_id_t *token_ids, size_t token_ids_len,
    const marmot_tokenizer_decode_options_t *opts, char *out_text, size_t *inout_len
) {
    Handle *handle = from_api(const_cast<marmot_tokenizer_t *>(tokenizer));
    if (handle == nullptr || handle->impl == nullptr || inout_len == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (token_ids == nullptr && token_ids_len != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "token_ids is null");
        set_last_error(handle, MARMOT_ERROR_INVALID_ARGUMENT, "token_ids is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_tokenizer_decode_options_t local_opts{};
    if (opts == nullptr) {
        marmot_error_t init_status = marmot_tokenizer_decode_options_init(&local_opts);
        if (init_status != MARMOT_SUCCESS) {
            return init_status;
        }
        opts = &local_opts;
    }
    marmot_error_t opts_status = validate_decode_options(opts);
    if (opts_status != MARMOT_SUCCESS) {
        marmot_set_error(opts_status, "invalid decode options");
        set_last_error(handle, opts_status, "invalid decode options");
        return opts_status;
    }

    std::string decoded;
    std::string error;
    std::span<const marmot_token_id_t> ids(token_ids, token_ids_len);
    marmot_error_t status = handle->impl->decode(ids, *opts, decoded, error);
    if (status != MARMOT_SUCCESS) {
        if (error.empty()) {
            error = "decode failed";
        }
        marmot_set_error(status, error.c_str());
        set_last_error(handle, status, error);
        return status;
    }

    const size_t required = decoded.size() + 1;
    if (out_text == nullptr) {
        *inout_len = required;
        set_success(handle);
        return MARMOT_SUCCESS;
    }
    if (*inout_len < required) {
        *inout_len = required;
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "buffer too small");
        set_last_error(handle, MARMOT_ERROR_INVALID_ARGUMENT, "buffer too small");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!decoded.empty()) {
        std::memcpy(out_text, decoded.data(), decoded.size());
    }
    out_text[decoded.size()] = '\0';
    *inout_len = required;
    set_success(handle);
    return MARMOT_SUCCESS;
}

const marmot_error_info_t *marmot_tokenizer_last_error(const marmot_tokenizer_t *tokenizer) {
    const Handle *handle = from_api(tokenizer);
    if (handle == nullptr) {
        return nullptr;
    }

    static thread_local marmot_error_info_t info;
    {
        std::lock_guard<std::mutex> lock(handle->last_error_mutex);
        info = handle->last_error;
    }
    return &info;
}

} // extern "C"
