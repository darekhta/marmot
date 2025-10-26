#pragma once

#include "marmot/graph/gguf_loader.h"
#include "marmot/tokenizer.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace marmot::tokenizer {

enum class ByteEncodingType {
    SpmHex,      // <0xHH> format (SPM/UGM tokenizers: LLaMA, Gemma, Phi-3)
    Gpt2Unicode, // GPT-2 bytes_to_unicode mapping (BPE tokenizers: Qwen, GPT-2)
    None,        // No byte fallback available
};

class Vocab {
  public:
    Vocab() = default;

    [[nodiscard]] size_t size() const noexcept {
        return pieces_.size();
    }

    [[nodiscard]] std::string_view piece(marmot_token_id_t id) const noexcept;
    [[nodiscard]] bool piece_to_id(std::string_view piece, marmot_token_id_t &out_id) const noexcept;

    [[nodiscard]] bool is_valid_id(marmot_token_id_t id) const noexcept {
        return id >= 0 && (size_t)id < pieces_.size();
    }

    [[nodiscard]] bool is_byte_token(marmot_token_id_t id) const noexcept;
    [[nodiscard]] std::optional<uint8_t> byte_value(marmot_token_id_t id) const noexcept;
    [[nodiscard]] marmot_token_id_t byte_token_id(uint8_t byte) const noexcept;

    [[nodiscard]] bool is_special_token(marmot_token_id_t id) const noexcept;

    [[nodiscard]] std::optional<marmot_token_id_t> whitespace_marker_id() const noexcept {
        return whitespace_marker_id_ >= 0 ? std::optional<marmot_token_id_t>(whitespace_marker_id_) : std::nullopt;
    }

    [[nodiscard]] ByteEncodingType byte_encoding_type() const noexcept {
        return byte_encoding_type_;
    }

    void force_gpt2_byte_encoding() noexcept;

    static constexpr std::string_view kWhitespaceMarker = "▁";

    [[nodiscard]] static std::optional<uint8_t> parse_byte_token(std::string_view piece) noexcept;

    [[nodiscard]] static std::optional<Vocab> from_gguf(
        const marmot_gguf_t *gguf, std::string_view &out_chat_template, std::string &out_error, bool strict_validation
    );

  private:
    std::vector<std::string_view> pieces_{};
    std::vector<int32_t> token_types_{};
    std::unordered_map<std::string_view, marmot_token_id_t> piece_to_id_{};
    std::array<marmot_token_id_t, 256> byte_to_id_{};
    marmot_token_id_t whitespace_marker_id_{MARMOT_TOKEN_ID_INVALID};
    ByteEncodingType byte_encoding_type_{ByteEncodingType::None};
};

} // namespace marmot::tokenizer
