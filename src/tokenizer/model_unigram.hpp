#pragma once

#include "marmot/graph/gguf_loader.h"
#include "marmot/tokenizer.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace marmot::tokenizer {

class Vocab;

class UnigramModel {
  public:
    UnigramModel() = default;

    [[nodiscard]] static std::optional<UnigramModel>
    from_gguf(const marmot_gguf_t *gguf, const Vocab &vocab, std::string &out_error, bool strict_validation);

    [[nodiscard]] marmot_error_t encode(
        const Vocab &vocab, std::string_view normalized, std::vector<marmot_token_id_t> &out, std::string &out_error
    ) const;

  private:
    struct TrieNode {
        int32_t head_edge;
        marmot_token_id_t token_id;
    };

    struct TrieEdge {
        uint8_t ch;
        int32_t next_node;
        int32_t next_edge;
    };

    [[nodiscard]] int32_t find_child(int32_t node, uint8_t ch) const noexcept;
    [[nodiscard]] int32_t add_child(int32_t node, uint8_t ch);

    std::vector<TrieNode> nodes_{};
    std::vector<TrieEdge> edges_{};
    std::vector<float> scores_{};
};

} // namespace marmot::tokenizer
