#include "utf8.hpp"

#include <array>

namespace marmot::tokenizer {

static constexpr bool is_cont(uint8_t byte) noexcept {
    return (byte & 0xC0u) == 0x80u;
}

// GPT-2 bytes_to_unicode() mapping:
// - ASCII printable (33-126) and Latin-1 supplement (161-172, 174-255) map directly
// - Other bytes (0-32, 127-160, 173) shift to Unicode 256+ to avoid control chars
static constexpr bool is_gpt2_direct_byte(uint8_t byte) noexcept {
    return (byte >= 33 && byte <= 126) || // ASCII printable: '!' to '~'
        (byte >= 161 && byte <= 172) ||   // Latin-1: inverted exclamation to feminine ordinal
        (byte >= 174 && byte <= 255);     // Latin-1: registered sign to latin small y with diaeresis
}

static const std::array<uint16_t, 256> &gpt2_bytes_to_unicode_table() {
    static const std::array<uint16_t, 256> table = [] {
        std::array<uint16_t, 256> map{};
        uint16_t extra = 0;
        for (uint16_t b = 0; b < 256; ++b) {
            const uint8_t byte = (uint8_t)b;
            if (is_gpt2_direct_byte(byte)) {
                map[b] = byte;
            } else {
                map[b] = (uint16_t)(256 + extra);
                ++extra;
            }
        }
        return map;
    }();
    return table;
}

static const std::array<int16_t, 512> &gpt2_unicode_to_bytes_table() {
    static const std::array<int16_t, 512> table = [] {
        std::array<int16_t, 512> map{};
        map.fill(-1);
        const std::array<uint16_t, 256> &forward = gpt2_bytes_to_unicode_table();
        for (size_t b = 0; b < forward.size(); ++b) {
            const uint16_t cp = forward[b];
            if (cp < map.size()) {
                map[cp] = (int16_t)b;
            }
        }
        return map;
    }();
    return table;
}

std::string gpt2_byte_to_utf8(uint8_t byte) noexcept {
    const uint32_t cp = gpt2_bytes_to_unicode_table()[byte];
    std::string result;
    if (cp < 0x80u) {
        result.push_back((char)cp);
    } else {
        result.push_back((char)(0xC0u | (cp >> 6)));
        result.push_back((char)(0x80u | (cp & 0x3Fu)));
    }
    return result;
}

std::optional<uint8_t> gpt2_utf8_to_byte(uint32_t codepoint) noexcept {
    const std::array<int16_t, 512> &table = gpt2_unicode_to_bytes_table();
    if (codepoint < table.size()) {
        const int16_t value = table[codepoint];
        if (value >= 0) {
            return (uint8_t)value;
        }
    }
    return std::nullopt;
}

Utf8DecodeResult utf8_decode(const uint8_t *data, size_t len) noexcept {
    if (data == nullptr || len == 0) {
        return Utf8DecodeResult{.codepoint = 0, .length = 0, .valid = false};
    }

    const uint8_t b0 = data[0];
    if (b0 < 0x80u) {
        return Utf8DecodeResult{.codepoint = b0, .length = 1, .valid = true};
    }

    if ((b0 & 0xE0u) == 0xC0u) {
        if (len < 2 || !is_cont(data[1])) {
            return Utf8DecodeResult{.codepoint = b0, .length = 1, .valid = false};
        }
        const uint32_t cp = ((uint32_t)(b0 & 0x1Fu) << 6) | (uint32_t)(data[1] & 0x3Fu);
        if (cp < 0x80u) {
            return Utf8DecodeResult{.codepoint = b0, .length = 1, .valid = false};
        }
        return Utf8DecodeResult{.codepoint = cp, .length = 2, .valid = true};
    }

    if ((b0 & 0xF0u) == 0xE0u) {
        if (len < 3 || !is_cont(data[1]) || !is_cont(data[2])) {
            return Utf8DecodeResult{.codepoint = b0, .length = 1, .valid = false};
        }
        const uint32_t cp =
            ((uint32_t)(b0 & 0x0Fu) << 12) | ((uint32_t)(data[1] & 0x3Fu) << 6) | (uint32_t)(data[2] & 0x3Fu);
        if (cp < 0x800u || (cp >= 0xD800u && cp <= 0xDFFFu)) {
            return Utf8DecodeResult{.codepoint = b0, .length = 1, .valid = false};
        }
        return Utf8DecodeResult{.codepoint = cp, .length = 3, .valid = true};
    }

    if ((b0 & 0xF8u) == 0xF0u) {
        if (len < 4 || !is_cont(data[1]) || !is_cont(data[2]) || !is_cont(data[3])) {
            return Utf8DecodeResult{.codepoint = b0, .length = 1, .valid = false};
        }
        const uint32_t cp = ((uint32_t)(b0 & 0x07u) << 18) | ((uint32_t)(data[1] & 0x3Fu) << 12) |
            ((uint32_t)(data[2] & 0x3Fu) << 6) | (uint32_t)(data[3] & 0x3Fu);
        if (cp < 0x10000u || cp > 0x10FFFFu) {
            return Utf8DecodeResult{.codepoint = b0, .length = 1, .valid = false};
        }
        return Utf8DecodeResult{.codepoint = cp, .length = 4, .valid = true};
    }

    return Utf8DecodeResult{.codepoint = b0, .length = 1, .valid = false};
}

} // namespace marmot::tokenizer
