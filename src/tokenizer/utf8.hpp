#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace marmot::tokenizer {

struct Utf8DecodeResult {
    uint32_t codepoint;
    size_t length;
    bool valid;
};

[[nodiscard]] Utf8DecodeResult utf8_decode(const uint8_t *data, size_t len) noexcept;

// GPT-2 byte-level BPE encoding: maps bytes to Unicode characters
// Used by Qwen, GPT-2, and other BPE tokenizers
[[nodiscard]] std::string gpt2_byte_to_utf8(uint8_t byte) noexcept;
[[nodiscard]] std::optional<uint8_t> gpt2_utf8_to_byte(uint32_t codepoint) noexcept;

} // namespace marmot::tokenizer
