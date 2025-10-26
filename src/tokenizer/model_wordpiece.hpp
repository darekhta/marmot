#pragma once

#include "marmot/tokenizer.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace marmot::tokenizer {

class Vocab;

class WordPieceModel {
  public:
    WordPieceModel() = default;

    [[nodiscard]] marmot_error_t encode(
        const Vocab &vocab, std::string_view text, marmot_token_id_t unk_id, std::vector<marmot_token_id_t> &out,
        std::string &out_error
    ) const;

    static constexpr std::string_view kContinuationPrefix = "##";
};

} // namespace marmot::tokenizer
