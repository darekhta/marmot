#pragma once

#include "marmot/graph/gguf_loader.h"
#include "marmot/tokenizer.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace marmot::tokenizer {

class Vocab;

class BpeModel {
  public:
    BpeModel() = default;

    struct MergeInfo {
        uint32_t rank;
        marmot_token_id_t merged_id;
    };

    [[nodiscard]] static std::optional<BpeModel>
    from_gguf(const marmot_gguf_t *gguf, const Vocab &vocab, std::string &out_error, bool strict_validation);

    [[nodiscard]] marmot_error_t encode(
        const Vocab &vocab, std::string_view normalized, std::vector<marmot_token_id_t> &out, std::string &out_error
    ) const;

  private:
    std::unordered_map<uint64_t, MergeInfo> merges_{};
};

} // namespace marmot::tokenizer
