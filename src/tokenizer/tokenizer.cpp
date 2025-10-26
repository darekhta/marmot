#include "tokenizer.hpp"

#include "marmot/graph/gguf_loader.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <new>

#include "utf8.hpp"

namespace marmot::tokenizer {

namespace {

struct CodepointFlags {
    bool is_letter = false;
    bool is_number = false;
    bool is_whitespace = false;
    bool valid = false;
};

constexpr uint32_t kOutOfRange = 0xFFFFFFFF;

bool is_unicode_whitespace(uint32_t cpt) noexcept {
    switch (cpt) {
    case 0x000009:
    case 0x00000A:
    case 0x00000B:
    case 0x00000C:
    case 0x00000D:
    case 0x000020:
    case 0x000085:
    case 0x0000A0:
    case 0x001680:
    case 0x002000:
    case 0x002001:
    case 0x002002:
    case 0x002003:
    case 0x002004:
    case 0x002005:
    case 0x002006:
    case 0x002007:
    case 0x002008:
    case 0x002009:
    case 0x00200A:
    case 0x002028:
    case 0x002029:
    case 0x00202F:
    case 0x00205F:
    case 0x003000:
        return true;
    default:
        return false;
    }
}

CodepointFlags classify_codepoint(uint32_t cpt) noexcept {
    CodepointFlags flags{};
    flags.valid = true;
    if (cpt <= 0x7F) {
        const unsigned char ch = (unsigned char)cpt;
        flags.is_whitespace = std::isspace(ch) != 0;
        flags.is_letter = std::isalpha(ch) != 0;
        flags.is_number = std::isdigit(ch) != 0;
        return flags;
    }

    if (is_unicode_whitespace(cpt)) {
        flags.is_whitespace = true;
        return flags;
    }

    flags.is_letter = true;
    return flags;
}

uint32_t tolower_ascii(uint32_t cpt) noexcept {
    if (cpt >= 'A' && cpt <= 'Z') {
        return cpt + ('a' - 'A');
    }
    return cpt;
}

struct DecodedText {
    std::vector<uint32_t> cpts;
    std::vector<size_t> byte_offsets;
};

DecodedText decode_text(std::string_view text) {
    DecodedText decoded;
    decoded.cpts.reserve(text.size());
    decoded.byte_offsets.reserve(text.size() + 1);

    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(text.data());
    size_t offset = 0;
    while (offset < text.size()) {
        decoded.byte_offsets.push_back(offset);
        const Utf8DecodeResult res = utf8_decode(bytes + offset, text.size() - offset);
        const size_t step = res.valid && res.length > 0 ? res.length : 1;
        const uint32_t cpt = res.valid ? res.codepoint : bytes[offset];
        decoded.cpts.push_back(cpt);
        offset += step;
    }
    decoded.byte_offsets.push_back(text.size());
    return decoded;
}

struct Slice {
    size_t offset = 0;
    size_t length = 0;
};

std::vector<Slice> split_gpt2(std::string_view text) {
    std::vector<Slice> slices;
    if (text.empty()) {
        return slices;
    }

    const DecodedText decoded = decode_text(text);
    const auto &cpts = decoded.cpts;
    const auto &byte_offsets = decoded.byte_offsets;

    std::vector<CodepointFlags> flags;
    flags.reserve(cpts.size());
    for (uint32_t cpt : cpts) {
        flags.push_back(classify_codepoint(cpt));
    }

    auto get_cpt = [&](size_t pos) -> uint32_t { return pos < cpts.size() ? cpts[pos] : kOutOfRange; };
    auto get_flags = [&](size_t pos) -> CodepointFlags { return pos < flags.size() ? flags[pos] : CodepointFlags{}; };

    size_t prev_end = 0;
    auto add_slice = [&](size_t end) -> size_t {
        if (end > prev_end) {
            const size_t start_byte = byte_offsets[prev_end];
            const size_t end_byte = byte_offsets[end];
            slices.push_back({start_byte, end_byte - start_byte});
        }
        const size_t len = end - prev_end;
        prev_end = end;
        return len;
    };

    for (size_t pos = 0; pos < cpts.size();) {
        const uint32_t cpt = get_cpt(pos);
        const CodepointFlags f = get_flags(pos);

        if (cpt == '\'' && pos + 1 < cpts.size()) {
            const uint32_t cpt_next = tolower_ascii(get_cpt(pos + 1));
            if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                pos += add_slice(pos + 2);
                continue;
            }
            if (pos + 2 < cpts.size()) {
                const uint32_t cpt_next_next = tolower_ascii(get_cpt(pos + 2));
                if ((cpt_next == 'r' && cpt_next_next == 'e') || (cpt_next == 'v' && cpt_next_next == 'e') ||
                    (cpt_next == 'l' && cpt_next_next == 'l')) {
                    pos += add_slice(pos + 3);
                    continue;
                }
            }
        }

        CodepointFlags f2 = (cpt == ' ') ? get_flags(pos + 1) : f;
        if (f2.valid && f2.is_letter) {
            pos += (cpt == ' ');
            while (get_flags(pos).is_letter) {
                pos++;
            }
            add_slice(pos);
            continue;
        }
        if (f2.valid && f2.is_number) {
            pos += (cpt == ' ');
            while (get_flags(pos).is_number) {
                pos++;
            }
            add_slice(pos);
            continue;
        }
        if (f2.valid && !f2.is_whitespace && !f2.is_letter && !f2.is_number) {
            pos += (cpt == ' ');
            while (true) {
                CodepointFlags f3 = get_flags(pos);
                if (!f3.valid || f3.is_whitespace || f3.is_letter || f3.is_number) {
                    break;
                }
                pos++;
            }
            add_slice(pos);
            continue;
        }

        size_t num_ws = 0;
        while (get_flags(pos + num_ws).is_whitespace) {
            num_ws++;
        }

        if (num_ws > 1 && get_cpt(pos + num_ws) != kOutOfRange) {
            pos += num_ws - 1;
            add_slice(pos);
            continue;
        }
        if (num_ws > 0) {
            pos += num_ws;
            add_slice(pos);
            continue;
        }

        add_slice(++pos);
    }

    return slices;
}

std::vector<Slice> split_qwen2(std::string_view text) {
    std::vector<Slice> slices;
    if (text.empty()) {
        return slices;
    }

    const DecodedText decoded = decode_text(text);
    const auto &cpts = decoded.cpts;
    const auto &byte_offsets = decoded.byte_offsets;

    std::vector<CodepointFlags> flags;
    flags.reserve(cpts.size());
    for (uint32_t cpt : cpts) {
        flags.push_back(classify_codepoint(cpt));
    }

    auto get_cpt = [&](size_t pos) -> uint32_t { return pos < cpts.size() ? cpts[pos] : kOutOfRange; };
    auto get_flags = [&](size_t pos) -> CodepointFlags { return pos < flags.size() ? flags[pos] : CodepointFlags{}; };

    size_t prev_end = 0;
    auto add_slice = [&](size_t end) -> size_t {
        if (end > prev_end) {
            const size_t start_byte = byte_offsets[prev_end];
            const size_t end_byte = byte_offsets[end];
            slices.push_back({start_byte, end_byte - start_byte});
        }
        const size_t len = end - prev_end;
        prev_end = end;
        return len;
    };

    for (size_t pos = 0; pos < cpts.size();) {
        const uint32_t cpt = get_cpt(pos);
        const CodepointFlags f = get_flags(pos);

        if (cpt == '\'' && pos + 1 < cpts.size()) {
            const uint32_t cpt_next = tolower_ascii(get_cpt(pos + 1));
            if (cpt_next == 's' || cpt_next == 't' || cpt_next == 'm' || cpt_next == 'd') {
                pos += add_slice(pos + 2);
                continue;
            }
            if (pos + 2 < cpts.size()) {
                const uint32_t cpt_next_next = tolower_ascii(get_cpt(pos + 2));
                if ((cpt_next == 'r' && cpt_next_next == 'e') || (cpt_next == 'v' && cpt_next_next == 'e') ||
                    (cpt_next == 'l' && cpt_next_next == 'l')) {
                    pos += add_slice(pos + 3);
                    continue;
                }
            }
        }

        if (!(cpt == '\r' || cpt == '\n' || f.is_number)) {
            if (f.is_letter || get_flags(pos + 1).is_letter) {
                pos++;
                while (get_flags(pos).is_letter) {
                    pos++;
                }
                add_slice(pos);
                continue;
            }
        }

        if (f.is_number) {
            add_slice(++pos);
            continue;
        }

        CodepointFlags f2 = (cpt == ' ') ? get_flags(pos + 1) : f;
        if (f2.valid && !f2.is_whitespace && !f2.is_letter && !f2.is_number) {
            pos += (cpt == ' ');
            while (true) {
                CodepointFlags f3 = get_flags(pos);
                if (!f3.valid || f3.is_whitespace || f3.is_letter || f3.is_number) {
                    break;
                }
                pos++;
            }
            uint32_t cpt2 = get_cpt(pos);
            while (cpt2 == '\r' || cpt2 == '\n') {
                cpt2 = get_cpt(++pos);
            }
            add_slice(pos);
            continue;
        }

        size_t num_ws = 0;
        size_t last_end_r_or_n = 0;
        while (get_flags(pos + num_ws).is_whitespace) {
            const uint32_t cpt2 = get_cpt(pos + num_ws);
            if (cpt2 == '\r' || cpt2 == '\n') {
                last_end_r_or_n = pos + num_ws + 1;
            }
            num_ws++;
        }

        if (last_end_r_or_n > 0) {
            pos = last_end_r_or_n;
            add_slice(pos);
            continue;
        }

        if (num_ws > 1 && get_cpt(pos + num_ws) != kOutOfRange) {
            pos += num_ws - 1;
            add_slice(pos);
            continue;
        }

        if (num_ws > 0) {
            pos += num_ws;
            add_slice(pos);
            continue;
        }

        add_slice(++pos);
    }

    return slices;
}

std::string encode_gpt2_bytes(std::string_view text) {
    std::string encoded;
    encoded.reserve(text.size() * 2);

    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(text.data());
    for (size_t i = 0; i < text.size(); ++i) {
        encoded.append(gpt2_byte_to_utf8(bytes[i]));
    }

    return encoded;
}

std::string decode_gpt2_bytes(std::string_view text) {
    std::string decoded;
    decoded.reserve(text.size());

    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(text.data());
    size_t offset = 0;
    while (offset < text.size()) {
        const Utf8DecodeResult res = utf8_decode(bytes + offset, text.size() - offset);
        const size_t step = res.valid && res.length > 0 ? res.length : 1;
        if (res.valid) {
            if (std::optional<uint8_t> byte = gpt2_utf8_to_byte(res.codepoint)) {
                decoded.push_back((char)*byte);
            } else {
                decoded.append(reinterpret_cast<const char *>(bytes + offset), step);
            }
        } else {
            decoded.push_back((char)bytes[offset]);
        }
        offset += step;
    }

    return decoded;
}

} // namespace

void Tokenizer::GgufDeleter::operator()(marmot_gguf_t *gguf) const noexcept {
    marmot_gguf_unload(gguf);
}

Tokenizer::Tokenizer(GgufPtr gguf) : gguf_(std::move(gguf)) {}

static bool validate_options(const marmot_tokenizer_options_t &opts, std::string &out_error) {
    out_error.clear();
    if (opts.struct_version != MARMOT_TOKENIZER_OPTIONS_VERSION) {
        out_error = "options version mismatch";
        return false;
    }
    if (opts.struct_size < sizeof(marmot_tokenizer_options_t)) {
        out_error = "options size mismatch";
        return false;
    }
    return true;
}

std::optional<marmot_token_id_t> Tokenizer::read_token_id(const marmot_gguf_t *gguf, const char *key) noexcept {
    const marmot_gguf_kv_t *kv = marmot_gguf_find_kv(gguf, key);
    if (kv == nullptr) {
        return std::nullopt;
    }

    switch (kv->value.type) {
    case MARMOT_GGUF_TYPE_UINT32:
        if (kv->value.data.uint32_value > (uint32_t)std::numeric_limits<marmot_token_id_t>::max()) {
            return std::nullopt;
        }
        return (marmot_token_id_t)kv->value.data.uint32_value;
    case MARMOT_GGUF_TYPE_INT32:
        return kv->value.data.int32_value;
    case MARMOT_GGUF_TYPE_UINT64:
        if (kv->value.data.uint64_value > (uint64_t)std::numeric_limits<marmot_token_id_t>::max()) {
            return std::nullopt;
        }
        return (marmot_token_id_t)kv->value.data.uint64_value;
    case MARMOT_GGUF_TYPE_INT64:
        if (kv->value.data.int64_value < 0 ||
            kv->value.data.int64_value > std::numeric_limits<marmot_token_id_t>::max()) {
            return std::nullopt;
        }
        return (marmot_token_id_t)kv->value.data.int64_value;
    default:
        return std::nullopt;
    }
}

std::optional<std::string_view> Tokenizer::read_string(const marmot_gguf_t *gguf, const char *key) noexcept {
    const marmot_gguf_kv_t *kv = marmot_gguf_find_kv(gguf, key);
    if (kv == nullptr || kv->value.type != MARMOT_GGUF_TYPE_STRING) {
        return std::nullopt;
    }
    return std::string_view(kv->value.data.string_value.data, kv->value.data.string_value.length);
}

std::optional<bool> Tokenizer::read_bool(const marmot_gguf_t *gguf, const char *key) noexcept {
    const marmot_gguf_kv_t *kv = marmot_gguf_find_kv(gguf, key);
    if (kv == nullptr || kv->value.type != MARMOT_GGUF_TYPE_BOOL) {
        return std::nullopt;
    }
    return kv->value.data.bool_value;
}

std::string Tokenizer::normalize_sentencepiece(const Vocab &vocab, std::string_view text, bool add_dummy_prefix) {
    const std::string_view marker = Vocab::kWhitespaceMarker;
    if (text.empty()) {
        return {};
    }
    if (vocab.whitespace_marker_id()) {
        std::string out;
        out.reserve(text.size() + (add_dummy_prefix ? marker.size() : 0));
        if (add_dummy_prefix) {
            out.append(marker);
        }

        for (unsigned char c : text) {
            if (c == ' ') {
                out.append(marker);
            } else {
                out.push_back((char)c);
            }
        }
        return out;
    }

    std::string ascii;
    ascii.reserve(text.size() + (add_dummy_prefix ? 1 : 0));
    if (add_dummy_prefix) {
        ascii.push_back(' ');
    }
    for (unsigned char c : text) {
        ascii.push_back((char)c);
    }
    return ascii;
}

void Tokenizer::replace_whitespace_marker(std::string &text) {
    const std::string_view marker = Vocab::kWhitespaceMarker;
    if (marker.empty() || text.find(marker) == std::string::npos) {
        return;
    }

    std::string out;
    out.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        if (i + marker.size() <= text.size() && std::memcmp(text.data() + i, marker.data(), marker.size()) == 0) {
            out.push_back(' ');
            i += marker.size();
            continue;
        }
        out.push_back(text[i]);
        ++i;
    }
    text = std::move(out);
}

std::string Tokenizer::cache_key(std::string_view text, const marmot_tokenizer_encode_options_t &opts) {
    std::string key;
    key.reserve(text.size() + 8);
    key.append(text.data(), text.size());
    key.push_back('\0');
    key.push_back(opts.add_bos ? 1 : 0);
    key.push_back(opts.add_eos ? 1 : 0);
    key.push_back(opts.allow_special ? 1 : 0);
    key.append(reinterpret_cast<const char *>(&opts.max_tokens), sizeof(opts.max_tokens));
    return key;
}

std::unique_ptr<Tokenizer> Tokenizer::from_gguf_file(
    const char *path, const marmot_tokenizer_options_t &opts, marmot_error_t &out_status, std::string &out_error
) {
    out_error.clear();
    out_status = MARMOT_ERROR_INVALID_OPERATION;

    if (path == nullptr) {
        out_status = MARMOT_ERROR_INVALID_ARGUMENT;
        out_error = "path is null";
        return nullptr;
    }

    if (!validate_options(opts, out_error)) {
        out_status = MARMOT_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }

    const bool strict_validation = (opts.flags & MARMOT_TOKENIZER_FLAG_STRICT_VALIDATION) != 0;

    marmot_gguf_t *raw = marmot_gguf_load(path);
    if (raw == nullptr) {
        out_status = marmot_get_last_error();
        out_error = marmot_get_last_error_detail() ? marmot_get_last_error_detail() : "failed to load gguf";
        return nullptr;
    }

    GgufPtr gguf(raw);

    std::string_view chat_template;
    auto maybe_vocab = Vocab::from_gguf(gguf.get(), chat_template, out_error, strict_validation);
    if (!maybe_vocab) {
        out_status = MARMOT_ERROR_INVALID_ARGUMENT;
        return nullptr;
    }
    Vocab vocab = std::move(*maybe_vocab);

    std::optional<std::string_view> model = read_string(gguf.get(), "tokenizer.ggml.model");
    if (!model) {
        out_status = MARMOT_ERROR_INVALID_ARGUMENT;
        out_error = "missing tokenizer.ggml.model";
        return nullptr;
    }
    const std::optional<std::string_view> pre_tokenizer = read_string(gguf.get(), "tokenizer.ggml.pre");
    const std::optional<bool> add_bos_token = read_bool(gguf.get(), "tokenizer.ggml.add_bos_token");
    const std::optional<bool> add_eos_token = read_bool(gguf.get(), "tokenizer.ggml.add_eos_token");
    const std::optional<bool> add_space_prefix = read_bool(gguf.get(), "tokenizer.ggml.add_space_prefix");
    const std::optional<std::string_view> arch_name = read_string(gguf.get(), "general.architecture");

    std::unique_ptr<Tokenizer> tokenizer(new (std::nothrow) Tokenizer(std::move(gguf)));
    if (!tokenizer) {
        out_status = MARMOT_ERROR_OUT_OF_MEMORY;
        out_error = "out of memory";
        return nullptr;
    }

    tokenizer->vocab_ = std::move(vocab);
    tokenizer->chat_template_ = chat_template;
    tokenizer->model_ = *model;
    tokenizer->pre_tokenizer_ = pre_tokenizer.value_or(std::string_view{});
    if (add_space_prefix) {
        tokenizer->add_space_prefix_ = *add_space_prefix;
    } else if (arch_name && *arch_name == "gemma") {
        tokenizer->add_space_prefix_ = false;
    }
    if (pre_tokenizer) {
        const std::string_view pre = *pre_tokenizer;
        if (pre == "qwen2" || pre == "gpt2" || pre == "gpt-2") {
            tokenizer->vocab_.force_gpt2_byte_encoding();
        }
    }
    tokenizer->cache_enabled_ = (opts.flags & MARMOT_TOKENIZER_FLAG_ENABLE_CACHE) != 0;
    if (tokenizer->cache_enabled_) {
        tokenizer->cache_.reserve(kCacheCapacity);
    }

    tokenizer->special_ids_ = marmot_tokenizer_special_ids_t{
        .has_bos = false,
        .has_eos = false,
        .has_unk = false,
        .has_pad = false,
        .bos_id = MARMOT_TOKEN_ID_INVALID,
        .eos_id = MARMOT_TOKEN_ID_INVALID,
        .unk_id = MARMOT_TOKEN_ID_INVALID,
        .pad_id = MARMOT_TOKEN_ID_INVALID,
    };

    if (auto bos = read_token_id(tokenizer->gguf_.get(), "tokenizer.ggml.bos_token_id")) {
        tokenizer->special_ids_.bos_id = *bos;
        tokenizer->special_ids_.has_bos = tokenizer->vocab_.is_valid_id(*bos);
    }
    if (auto eos = read_token_id(tokenizer->gguf_.get(), "tokenizer.ggml.eos_token_id")) {
        tokenizer->special_ids_.eos_id = *eos;
        tokenizer->special_ids_.has_eos = tokenizer->vocab_.is_valid_id(*eos);
    }
    if (auto unk = read_token_id(tokenizer->gguf_.get(), "tokenizer.ggml.unknown_token_id")) {
        tokenizer->special_ids_.unk_id = *unk;
        tokenizer->special_ids_.has_unk = tokenizer->vocab_.is_valid_id(*unk);
    }
    if (auto pad = read_token_id(tokenizer->gguf_.get(), "tokenizer.ggml.padding_token_id")) {
        tokenizer->special_ids_.pad_id = *pad;
        tokenizer->special_ids_.has_pad = tokenizer->vocab_.is_valid_id(*pad);
    }
    if (add_bos_token) {
        tokenizer->has_add_bos_token_ = true;
        tokenizer->add_bos_token_ = *add_bos_token;
    }
    if (add_eos_token) {
        tokenizer->has_add_eos_token_ = true;
        tokenizer->add_eos_token_ = *add_eos_token;
    }

    tokenizer->special_pieces_.clear();
    tokenizer->special_pieces_.reserve(64);
    for (marmot_token_id_t id = 0; id < (marmot_token_id_t)tokenizer->vocab_.size(); ++id) {
        if (!tokenizer->vocab_.is_special_token(id)) {
            continue;
        }
        const std::string_view piece = tokenizer->vocab_.piece(id);
        if (piece.empty()) {
            continue;
        }
        tokenizer->special_pieces_.push_back(SpecialPiece{.piece = piece, .id = id});
    }
    std::sort(
        tokenizer->special_pieces_.begin(), tokenizer->special_pieces_.end(),
        [](const SpecialPiece &a, const SpecialPiece &b) { return a.piece.size() > b.piece.size(); }
    );

    tokenizer->whitespace_run_ids_.clear();
    if (tokenizer->vocab_.whitespace_marker_id()) {
        const std::string_view marker = Vocab::kWhitespaceMarker;
        for (marmot_token_id_t id = 0; id < (marmot_token_id_t)tokenizer->vocab_.size(); ++id) {
            if (tokenizer->vocab_.is_special_token(id) || tokenizer->vocab_.is_byte_token(id)) {
                continue;
            }
            const std::string_view piece = tokenizer->vocab_.piece(id);
            if (piece.empty() || piece.size() % marker.size() != 0) {
                continue;
            }
            const size_t count = piece.size() / marker.size();
            if (count == 0) {
                continue;
            }
            bool is_run = true;
            for (size_t i = 0; i < count; ++i) {
                if (std::memcmp(piece.data() + i * marker.size(), marker.data(), marker.size()) != 0) {
                    is_run = false;
                    break;
                }
            }
            if (!is_run) {
                continue;
            }

            if (tokenizer->whitespace_run_ids_.size() <= count) {
                tokenizer->whitespace_run_ids_.resize(count + 1, MARMOT_TOKEN_ID_INVALID);
            }
            if (tokenizer->whitespace_run_ids_[count] == MARMOT_TOKEN_ID_INVALID) {
                tokenizer->whitespace_run_ids_[count] = id;
            }
        }
    }

    std::string model_error;
    bool has_merges = false;
    const marmot_gguf_kv_t *merges_kv = marmot_gguf_find_kv(tokenizer->gguf_.get(), "tokenizer.ggml.merges");
    if (merges_kv && merges_kv->value.type == MARMOT_GGUF_TYPE_ARRAY &&
        merges_kv->value.data.array_value.type == MARMOT_GGUF_TYPE_STRING &&
        merges_kv->value.data.array_value.length > 0) {
        has_merges = true;
    }

    if (has_merges) {
        auto model_bpe = BpeModel::from_gguf(tokenizer->gguf_.get(), tokenizer->vocab_, model_error, strict_validation);
        if (!model_bpe) {
            out_status = MARMOT_ERROR_INVALID_ARGUMENT;
            out_error = !model_error.empty() ? model_error : "failed to load bpe model";
            return nullptr;
        }
        tokenizer->model_kind_ = ModelKind::Bpe;
        tokenizer->bpe_ = std::move(*model_bpe);
    } else if (tokenizer->model_ == "wordpiece") {
        tokenizer->model_kind_ = ModelKind::WordPiece;
        tokenizer->wordpiece_ = WordPieceModel{};
    } else {
        auto model_unigram =
            UnigramModel::from_gguf(tokenizer->gguf_.get(), tokenizer->vocab_, model_error, strict_validation);
        if (!model_unigram) {
            out_status = MARMOT_ERROR_INVALID_ARGUMENT;
            out_error = !model_error.empty() ? model_error : "failed to load unigram model";
            return nullptr;
        }
        tokenizer->model_kind_ = ModelKind::Unigram;
        tokenizer->unigram_ = std::move(*model_unigram);
    }

    out_status = MARMOT_SUCCESS;
    return tokenizer;
}

marmot_error_t Tokenizer::encode(
    std::string_view text, const marmot_tokenizer_encode_options_t &opts, std::vector<marmot_token_id_t> &out_token_ids,
    std::string &out_error
) const {
    if (!cache_enabled_) {
        return encode_uncached(text, opts, out_token_ids, out_error);
    }

    const std::string key = cache_key(text, opts);
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        for (auto &entry : cache_) {
            if (entry.key == key) {
                out_token_ids = entry.tokens;
                out_error.clear();
                return MARMOT_SUCCESS;
            }
        }
    }

    std::vector<marmot_token_id_t> tokens;
    marmot_error_t status = encode_uncached(text, opts, tokens, out_error);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        if (cache_.size() >= kCacheCapacity) {
            cache_.erase(cache_.begin());
        }
        cache_.push_back(CacheEntry{.key = key, .tokens = tokens});
    }

    out_token_ids = std::move(tokens);
    return MARMOT_SUCCESS;
}

marmot_error_t Tokenizer::encode_uncached(
    std::string_view text, const marmot_tokenizer_encode_options_t &opts, std::vector<marmot_token_id_t> &out_token_ids,
    std::string &out_error
) const {
    out_error.clear();
    out_token_ids.clear();

    std::vector<marmot_token_id_t> ids;
    std::vector<marmot_token_id_t> scratch;
    scratch.reserve(128);
    std::vector<marmot_token_id_t> word_scratch;
    word_scratch.reserve(128);
    bool emitted_dummy_prefix = false;

    auto encode_segment = [&](std::string_view segment) -> marmot_error_t {
        scratch.clear();
        word_scratch.clear();
        if (segment.empty()) {
            return MARMOT_SUCCESS;
        }

        if (model_kind_ == ModelKind::Bpe) {
            if (model_ == "llama") {
                const bool add_dummy_prefix = add_space_prefix_ && !emitted_dummy_prefix;
                const std::string normalized = normalize_sentencepiece(vocab_, segment, add_dummy_prefix);
                if (add_dummy_prefix) {
                    emitted_dummy_prefix = true;
                }
                if (!vocab_.whitespace_marker_id()) {
                    return bpe_->encode(vocab_, normalized, scratch, out_error);
                }

                const std::string_view marker = Vocab::kWhitespaceMarker;
                const size_t marker_len = marker.size();
                const auto marker_id = vocab_.whitespace_marker_id();
                if (!marker_id) {
                    out_error = "whitespace marker missing";
                    return MARMOT_ERROR_INVALID_OPERATION;
                }

                auto emit_whitespace = [&](size_t count) -> marmot_error_t {
                    while (count > 0) {
                        size_t best = 0;
                        const size_t max_len = whitespace_run_ids_.empty() ? 1 : whitespace_run_ids_.size() - 1;
                        for (size_t candidate = std::min(count, max_len); candidate >= 1; --candidate) {
                            if (candidate < whitespace_run_ids_.size() &&
                                whitespace_run_ids_[candidate] != MARMOT_TOKEN_ID_INVALID) {
                                best = candidate;
                                break;
                            }
                            if (candidate == 1) {
                                break;
                            }
                        }

                        if (best == 0) {
                            scratch.push_back(*marker_id);
                            count -= 1;
                            continue;
                        }

                        scratch.push_back(whitespace_run_ids_[best]);
                        count -= best;
                    }
                    return MARMOT_SUCCESS;
                };

                auto is_marker_at = [&](size_t pos) -> bool {
                    return pos + marker_len <= normalized.size() &&
                        std::memcmp(normalized.data() + pos, marker.data(), marker_len) == 0;
                };

                std::string with_marker;
                with_marker.reserve(marker_len + 32);

                bool pending_marker = false;
                size_t pos = 0;
                while (pos < normalized.size()) {
                    if (is_marker_at(pos)) {
                        size_t run = 0;
                        while (is_marker_at(pos)) {
                            run += 1;
                            pos += marker_len;
                        }

                        if (pending_marker) {
                            run += 1;
                            pending_marker = false;
                        }

                        if (run == 1 && pos < normalized.size()) {
                            pending_marker = true;
                            continue;
                        }

                        marmot_error_t ws_status = emit_whitespace(run);
                        if (ws_status != MARMOT_SUCCESS) {
                            return ws_status;
                        }
                        continue;
                    }

                    const size_t start = pos;
                    while (pos < normalized.size() && !is_marker_at(pos)) {
                        pos += 1;
                    }
                    const std::string_view word(normalized.data() + start, pos - start);

                    if (pending_marker) {
                        with_marker.clear();
                        with_marker.append(marker);
                        with_marker.append(word);
                        marmot_error_t word_status = bpe_->encode(vocab_, with_marker, word_scratch, out_error);
                        if (word_status != MARMOT_SUCCESS) {
                            return word_status;
                        }
                        scratch.insert(scratch.end(), word_scratch.begin(), word_scratch.end());
                        pending_marker = false;
                        continue;
                    }

                    marmot_error_t word_status = bpe_->encode(vocab_, word, word_scratch, out_error);
                    if (word_status != MARMOT_SUCCESS) {
                        return word_status;
                    }
                    scratch.insert(scratch.end(), word_scratch.begin(), word_scratch.end());
                }

                if (pending_marker) {
                    marmot_error_t ws_status = emit_whitespace(1);
                    if (ws_status != MARMOT_SUCCESS) {
                        return ws_status;
                    }
                }

                return MARMOT_SUCCESS;
            }
            auto encode_piece = [&](std::string_view piece) -> marmot_error_t {
                word_scratch.clear();
                std::string encoded;
                std::string_view input = piece;
                if (vocab_.byte_encoding_type() == ByteEncodingType::Gpt2Unicode) {
                    encoded = encode_gpt2_bytes(piece);
                    input = encoded;
                }
                marmot_error_t status = bpe_->encode(vocab_, input, word_scratch, out_error);
                if (status != MARMOT_SUCCESS) {
                    return status;
                }
                scratch.insert(scratch.end(), word_scratch.begin(), word_scratch.end());
                return MARMOT_SUCCESS;
            };

            if (pre_tokenizer_ == "qwen2") {
                const std::vector<Slice> slices = split_qwen2(segment);
                if (slices.empty()) {
                    return bpe_->encode(vocab_, segment, scratch, out_error);
                }
                for (const Slice &slice : slices) {
                    const std::string_view piece(segment.data() + slice.offset, slice.length);
                    marmot_error_t status = encode_piece(piece);
                    if (status != MARMOT_SUCCESS) {
                        return status;
                    }
                }
                return MARMOT_SUCCESS;
            }
            if (pre_tokenizer_ == "gpt2" || pre_tokenizer_ == "gpt-2") {
                const std::vector<Slice> slices = split_gpt2(segment);
                if (slices.empty()) {
                    return bpe_->encode(vocab_, segment, scratch, out_error);
                }
                for (const Slice &slice : slices) {
                    const std::string_view piece(segment.data() + slice.offset, slice.length);
                    marmot_error_t status = encode_piece(piece);
                    if (status != MARMOT_SUCCESS) {
                        return status;
                    }
                }
                return MARMOT_SUCCESS;
            }

            return bpe_->encode(vocab_, segment, scratch, out_error);
        }
        if (model_kind_ == ModelKind::Unigram) {
            if (model_ == "llama") {
                const bool add_dummy_prefix = add_space_prefix_ && !emitted_dummy_prefix;
                const std::string normalized = normalize_sentencepiece(vocab_, segment, add_dummy_prefix);
                if (add_dummy_prefix) {
                    emitted_dummy_prefix = true;
                }
                return unigram_->encode(vocab_, normalized, scratch, out_error);
            }
            return unigram_->encode(vocab_, segment, scratch, out_error);
        }

        const marmot_token_id_t unk_id = special_ids_.has_unk ? special_ids_.unk_id : MARMOT_TOKEN_ID_INVALID;
        return wordpiece_->encode(vocab_, segment, unk_id, scratch, out_error);
    };

    auto append_scratch = [&]() { ids.insert(ids.end(), scratch.begin(), scratch.end()); };

    if (opts.allow_special && !special_pieces_.empty() && !text.empty()) {
        size_t cursor = 0;
        size_t segment_start = 0;

        while (cursor < text.size()) {
            std::optional<marmot_token_id_t> match_id;
            size_t match_len = 0;

            for (const SpecialPiece &special : special_pieces_) {
                if (special.piece.size() > text.size() - cursor) {
                    continue;
                }
                if (std::memcmp(text.data() + cursor, special.piece.data(), special.piece.size()) == 0) {
                    match_id = special.id;
                    match_len = special.piece.size();
                    break;
                }
            }

            if (!match_id) {
                ++cursor;
                continue;
            }

            const std::string_view segment = text.substr(segment_start, cursor - segment_start);
            marmot_error_t seg_status = encode_segment(segment);
            if (seg_status != MARMOT_SUCCESS) {
                return seg_status;
            }
            append_scratch();

            ids.push_back(*match_id);

            cursor += match_len;
            segment_start = cursor;
        }

        const std::string_view tail = text.substr(segment_start);
        marmot_error_t tail_status = encode_segment(tail);
        if (tail_status != MARMOT_SUCCESS) {
            return tail_status;
        }
        append_scratch();
    } else {
        marmot_error_t status = encode_segment(text);
        if (status != MARMOT_SUCCESS) {
            return status;
        }
        append_scratch();
    }

    if (opts.add_bos && special_ids_.has_bos) {
        ids.insert(ids.begin(), special_ids_.bos_id);
    }
    if (opts.add_eos && special_ids_.has_eos) {
        ids.push_back(special_ids_.eos_id);
    }

    if (opts.max_tokens > 0 && ids.size() > opts.max_tokens) {
        out_error = "max_tokens exceeded";
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    out_token_ids = std::move(ids);
    return MARMOT_SUCCESS;
}

marmot_error_t Tokenizer::decode(
    std::span<const marmot_token_id_t> token_ids, const marmot_tokenizer_decode_options_t &opts, std::string &out_text,
    std::string &out_error
) const {
    out_error.clear();
    out_text.clear();

    std::string raw;
    raw.reserve(token_ids.size() * 4);
    const bool gpt2_bytes = vocab_.byte_encoding_type() == ByteEncodingType::Gpt2Unicode;

    for (marmot_token_id_t id : token_ids) {
        if (!vocab_.is_valid_id(id)) {
            out_error = "invalid token id";
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (opts.skip_special && vocab_.is_special_token(id)) {
            continue;
        }

        if (!gpt2_bytes) {
            if (std::optional<uint8_t> byte = vocab_.byte_value(id)) {
                raw.push_back((char)*byte);
                continue;
            }
        }
        raw.append(vocab_.piece(id));
    }

    if (gpt2_bytes) {
        raw = decode_gpt2_bytes(raw);
    }
    if (vocab_.whitespace_marker_id()) {
        replace_whitespace_marker(raw);
    }
    if (opts.strip_space_prefix && model_ == "llama" && add_space_prefix_ && !raw.empty() && raw[0] == ' ') {
        raw.erase(raw.begin());
    }

    out_text = std::move(raw);
    return MARMOT_SUCCESS;
}

} // namespace marmot::tokenizer
