#include "vocab.hpp"

#include "marmot/graph/gguf_loader.h"

#include <algorithm>
#include <cctype>
#include <cstring>

#include "utf8.hpp"

namespace marmot::tokenizer {

std::string_view Vocab::piece(marmot_token_id_t id) const noexcept {
    if (!is_valid_id(id)) {
        return {};
    }
    return pieces_[(size_t)id];
}

bool Vocab::piece_to_id(std::string_view piece, marmot_token_id_t &out_id) const noexcept {
    auto it = piece_to_id_.find(piece);
    if (it == piece_to_id_.end()) {
        return false;
    }
    out_id = it->second;
    return true;
}

bool Vocab::is_byte_token(marmot_token_id_t id) const noexcept {
    if (!is_valid_id(id)) {
        return false;
    }

    // SPM-style: check token type 6
    if (!token_types_.empty() && token_types_[(size_t)id] == 6) {
        return true;
    }

    // GPT-2 style: check if this token is in the byte_to_id_ mapping
    if (byte_encoding_type_ == ByteEncodingType::Gpt2Unicode) {
        for (size_t b = 0; b < 256; ++b) {
            if (byte_to_id_[b] == id) {
                return true;
            }
        }
    }

    return false;
}

std::optional<uint8_t> Vocab::byte_value(marmot_token_id_t id) const noexcept {
    if (!is_valid_id(id)) {
        return std::nullopt;
    }

    const std::string_view piece = pieces_[(size_t)id];

    // Try SPM-style <0xHH> format first
    if (std::optional<uint8_t> parsed = parse_byte_token(piece)) {
        return *parsed;
    }

    // Try GPT-2 encoding
    if (byte_encoding_type_ == ByteEncodingType::Gpt2Unicode && piece.size() <= 2) {
        const uint8_t *bytes = reinterpret_cast<const uint8_t *>(piece.data());
        Utf8DecodeResult decoded = utf8_decode(bytes, piece.size());
        if (decoded.valid && decoded.length == piece.size()) {
            return gpt2_utf8_to_byte(decoded.codepoint);
        }
    }

    return std::nullopt;
}

marmot_token_id_t Vocab::byte_token_id(uint8_t byte) const noexcept {
    marmot_token_id_t id = byte_to_id_[byte];
    return id;
}

void Vocab::force_gpt2_byte_encoding() noexcept {
    if (byte_encoding_type_ != ByteEncodingType::None) {
        return;
    }

    size_t mapped = 0;
    for (size_t i = 0; i < byte_to_id_.size(); ++i) {
        if (byte_to_id_[i] != MARMOT_TOKEN_ID_INVALID) {
            mapped++;
        }
    }
    if (mapped > 0) {
        byte_encoding_type_ = ByteEncodingType::Gpt2Unicode;
    }
}

bool Vocab::is_special_token(marmot_token_id_t id) const noexcept {
    if (!is_valid_id(id) || token_types_.empty()) {
        return false;
    }

    const int32_t type = token_types_[(size_t)id];
    if (type == 1 || type == 6) {
        return false;
    }
    return true;
}

std::optional<uint8_t> Vocab::parse_byte_token(std::string_view piece) noexcept {
    if (piece.size() != 6) {
        return std::nullopt;
    }
    if (piece[0] != '<' || piece[1] != '0' || piece[2] != 'x' || piece[5] != '>') {
        return std::nullopt;
    }

    auto hex = [](char c) -> int {
        if (c >= '0' && c <= '9') {
            return c - '0';
        }
        if (c >= 'a' && c <= 'f') {
            return 10 + (c - 'a');
        }
        if (c >= 'A' && c <= 'F') {
            return 10 + (c - 'A');
        }
        return -1;
    };

    const int hi = hex(piece[3]);
    const int lo = hex(piece[4]);
    if (hi < 0 || lo < 0) {
        return std::nullopt;
    }
    return (uint8_t)((hi << 4) | lo);
}

static bool is_visible_special_piece(std::string_view piece) noexcept {
    return piece == "<think>" || piece == "</think>";
}

static bool is_unused_special_piece(std::string_view piece) noexcept {
    constexpr std::string_view prefix = "<unused";
    if (piece.size() <= prefix.size() + 1) {
        return false;
    }
    if (!piece.starts_with(prefix) || piece.back() != '>') {
        return false;
    }
    for (size_t i = prefix.size(); i + 1 < piece.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(piece[i]);
        if (!std::isdigit(c)) {
            return false;
        }
    }
    return true;
}

static const marmot_gguf_kv_t *find_kv(const marmot_gguf_t *gguf, const char *key) {
    return marmot_gguf_find_kv(gguf, key);
}

std::optional<Vocab> Vocab::from_gguf(
    const marmot_gguf_t *gguf, std::string_view &out_chat_template, std::string &out_error, bool strict_validation
) {
    out_error.clear();
    out_chat_template = {};
    if (gguf == nullptr) {
        out_error = "gguf is null";
        return std::nullopt;
    }

    const marmot_gguf_kv_t *tokens_kv = find_kv(gguf, "tokenizer.ggml.tokens");
    if (tokens_kv == nullptr || tokens_kv->value.type != MARMOT_GGUF_TYPE_ARRAY ||
        tokens_kv->value.data.array_value.type != MARMOT_GGUF_TYPE_STRING) {
        out_error = "missing tokenizer.ggml.tokens";
        return std::nullopt;
    }

    const marmot_gguf_array_t tokens_arr = tokens_kv->value.data.array_value;
    if (tokens_arr.length == 0 || tokens_arr.data.string_values == nullptr) {
        out_error = "tokenizer.ggml.tokens is empty";
        return std::nullopt;
    }

    const marmot_gguf_kv_t *types_kv = find_kv(gguf, "tokenizer.ggml.token_type");
    std::vector<int32_t> types;
    if (types_kv != nullptr && types_kv->value.type == MARMOT_GGUF_TYPE_ARRAY &&
        types_kv->value.data.array_value.type == MARMOT_GGUF_TYPE_INT32) {
        const marmot_gguf_array_t types_arr = types_kv->value.data.array_value;
        if (types_arr.length != tokens_arr.length) {
            out_error = "tokenizer.ggml.token_type length mismatch";
            return std::nullopt;
        }
        types.assign(types_arr.data.int32_values, types_arr.data.int32_values + types_arr.length);
    } else {
        types.assign(tokens_arr.length, 1);
        if (strict_validation) {
            out_error = "missing tokenizer.ggml.token_type";
            return std::nullopt;
        }
    }

    const marmot_gguf_kv_t *chat_kv = find_kv(gguf, "tokenizer.chat_template");
    if (chat_kv != nullptr && chat_kv->value.type == MARMOT_GGUF_TYPE_STRING) {
        out_chat_template =
            std::string_view(chat_kv->value.data.string_value.data, chat_kv->value.data.string_value.length);
    }

    Vocab vocab;
    vocab.pieces_.reserve(tokens_arr.length);
    vocab.token_types_ = std::move(types);
    vocab.piece_to_id_.reserve((size_t)((double)tokens_arr.length * 1.3));
    vocab.byte_to_id_.fill(MARMOT_TOKEN_ID_INVALID);

    for (size_t i = 0; i < tokens_arr.length; ++i) {
        const marmot_gguf_string_t s = tokens_arr.data.string_values[i];
        const std::string_view piece(s.data != nullptr ? s.data : "", s.length);
        vocab.pieces_.push_back(piece);
        vocab.piece_to_id_.emplace(piece, (marmot_token_id_t)i);

        if (is_visible_special_piece(piece)) {
            vocab.token_types_[(size_t)i] = 1;
        }
        if (vocab.token_types_[(size_t)i] == 1 && is_unused_special_piece(piece)) {
            vocab.token_types_[(size_t)i] = 3;
        }

        std::optional<uint8_t> byte = parse_byte_token(piece);
        if (byte && vocab.token_types_[(size_t)i] == 6 && vocab.byte_to_id_[*byte] == MARMOT_TOKEN_ID_INVALID) {
            vocab.byte_to_id_[*byte] = (marmot_token_id_t)i;
        }
    }

    auto it = vocab.piece_to_id_.find(kWhitespaceMarker);
    if (it != vocab.piece_to_id_.end()) {
        vocab.whitespace_marker_id_ = it->second;
    }

    // Detect byte encoding type
    // First check if we have all SPM-style <0xHH> byte tokens
    bool has_all_spm_bytes = true;
    for (size_t b = 0; b < 256 && has_all_spm_bytes; ++b) {
        if (vocab.byte_to_id_[b] == MARMOT_TOKEN_ID_INVALID) {
            has_all_spm_bytes = false;
        }
    }

    if (has_all_spm_bytes) {
        vocab.byte_encoding_type_ = ByteEncodingType::SpmHex;
    } else {
        // Try GPT-2 byte encoding (used by Qwen, GPT-2, etc.)
        vocab.byte_to_id_.fill(MARMOT_TOKEN_ID_INVALID);
        size_t gpt2_bytes_found = 0;

        for (size_t byte = 0; byte < 256; ++byte) {
            std::string encoded = gpt2_byte_to_utf8((uint8_t)byte);
            marmot_token_id_t id;
            if (vocab.piece_to_id(encoded, id)) {
                vocab.byte_to_id_[byte] = id;
                gpt2_bytes_found++;
            }
        }

        // Accept GPT-2 encoding if we have most bytes (200+)
        // Some models like Qwen2 don't include all 256 byte tokens
        // (missing C1 control codes 127-160), but have enough for normal text
        vocab.byte_encoding_type_ = (gpt2_bytes_found >= 200) ? ByteEncodingType::Gpt2Unicode : ByteEncodingType::None;
    }

    // Validate: fail only if no byte encoding is available and strict validation is enabled
    if (vocab.byte_encoding_type_ == ByteEncodingType::None && strict_validation) {
        out_error = "missing byte fallback tokens";
        return std::nullopt;
    }

    return vocab;
}

} // namespace marmot::tokenizer
