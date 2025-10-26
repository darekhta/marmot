#pragma once

#include "marmot/tokenizer.h"

#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "model_bpe.hpp"
#include "model_unigram.hpp"
#include "model_wordpiece.hpp"
#include "vocab.hpp"

namespace marmot::tokenizer {

class Tokenizer {
  public:
    Tokenizer() = delete;

    [[nodiscard]] static std::unique_ptr<Tokenizer> from_gguf_file(
        const char *path, const marmot_tokenizer_options_t &opts, marmot_error_t &out_status, std::string &out_error
    );

    [[nodiscard]] size_t vocab_size() const noexcept {
        return vocab_.size();
    }

    [[nodiscard]] const Vocab &vocab() const noexcept {
        return vocab_;
    }

    [[nodiscard]] std::string_view chat_template() const noexcept {
        return chat_template_;
    }

    [[nodiscard]] marmot_tokenizer_special_ids_t special_ids() const noexcept {
        return special_ids_;
    }

    [[nodiscard]] marmot_tokenizer_behavior_t behavior() const noexcept {
        return {
            .has_add_bos = has_add_bos_token_,
            .add_bos = add_bos_token_,
            .has_add_eos = has_add_eos_token_,
            .add_eos = add_eos_token_,
        };
    }

    [[nodiscard]] marmot_error_t piece_to_token(std::string_view piece, marmot_token_id_t &out_id) const noexcept {
        return vocab_.piece_to_id(piece, out_id) ? MARMOT_SUCCESS : MARMOT_ERROR_INVALID_ARGUMENT;
    }

    [[nodiscard]] std::string_view token_to_piece(marmot_token_id_t id) const noexcept {
        return vocab_.piece(id);
    }

    [[nodiscard]] marmot_error_t encode(
        std::string_view text, const marmot_tokenizer_encode_options_t &opts,
        std::vector<marmot_token_id_t> &out_token_ids, std::string &out_error
    ) const;

    [[nodiscard]] marmot_error_t decode(
        std::span<const marmot_token_id_t> token_ids, const marmot_tokenizer_decode_options_t &opts,
        std::string &out_text, std::string &out_error
    ) const;

  private:
    struct GgufDeleter {
        void operator()(marmot_gguf_t *gguf) const noexcept;
    };
    using GgufPtr = std::unique_ptr<marmot_gguf_t, GgufDeleter>;

    struct SpecialPiece {
        std::string_view piece;
        marmot_token_id_t id;
    };

    enum class ModelKind {
        Bpe,
        Unigram,
        WordPiece,
    };

    struct CacheEntry {
        std::string key;
        std::vector<marmot_token_id_t> tokens;
    };

    explicit Tokenizer(GgufPtr gguf);

    [[nodiscard]] marmot_error_t encode_uncached(
        std::string_view text, const marmot_tokenizer_encode_options_t &opts,
        std::vector<marmot_token_id_t> &out_token_ids, std::string &out_error
    ) const;

    [[nodiscard]] static std::optional<marmot_token_id_t>
    read_token_id(const marmot_gguf_t *gguf, const char *key) noexcept;

    [[nodiscard]] static std::optional<std::string_view>
    read_string(const marmot_gguf_t *gguf, const char *key) noexcept;

    [[nodiscard]] static std::optional<bool> read_bool(const marmot_gguf_t *gguf, const char *key) noexcept;

    [[nodiscard]] static std::string
    normalize_sentencepiece(const Vocab &vocab, std::string_view text, bool add_dummy_prefix);

    static void replace_whitespace_marker(std::string &text);

    [[nodiscard]] static std::string cache_key(std::string_view text, const marmot_tokenizer_encode_options_t &opts);

    GgufPtr gguf_{};
    Vocab vocab_{};
    std::string_view model_{};
    std::string_view pre_tokenizer_{};
    std::string_view chat_template_{};
    bool add_space_prefix_{true};
    marmot_tokenizer_special_ids_t special_ids_{};
    bool has_add_bos_token_{false};
    bool add_bos_token_{false};
    bool has_add_eos_token_{false};
    bool add_eos_token_{false};
    ModelKind model_kind_{ModelKind::Bpe};
    std::optional<BpeModel> bpe_{};
    std::optional<UnigramModel> unigram_{};
    std::optional<WordPieceModel> wordpiece_{};
    std::vector<SpecialPiece> special_pieces_{};
    std::vector<marmot_token_id_t> whitespace_run_ids_{};
    bool cache_enabled_{false};

    static constexpr size_t kCacheCapacity = 128;
    mutable std::mutex cache_mutex_{};
    mutable std::vector<CacheEntry> cache_{};
};

} // namespace marmot::tokenizer
