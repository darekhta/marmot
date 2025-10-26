#include "model_unigram.hpp"

#include "marmot/graph/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "vocab.hpp"

namespace marmot::tokenizer {

static const marmot_gguf_kv_t *find_kv(const marmot_gguf_t *gguf, const char *key) {
    return marmot_gguf_find_kv(gguf, key);
}

int32_t UnigramModel::find_child(int32_t node, uint8_t ch) const noexcept {
    if (node < 0 || node >= (int32_t)nodes_.size()) {
        return -1;
    }
    for (int32_t edge = nodes_[(size_t)node].head_edge; edge >= 0; edge = edges_[(size_t)edge].next_edge) {
        if (edges_[(size_t)edge].ch == ch) {
            return edges_[(size_t)edge].next_node;
        }
    }
    return -1;
}

int32_t UnigramModel::add_child(int32_t node, uint8_t ch) {
    int32_t child = find_child(node, ch);
    if (child >= 0) {
        return child;
    }

    child = (int32_t)nodes_.size();
    nodes_.push_back(TrieNode{.head_edge = -1, .token_id = MARMOT_TOKEN_ID_INVALID});
    const int32_t edge_idx = (int32_t)edges_.size();
    edges_.push_back(
        TrieEdge{
            .ch = ch,
            .next_node = child,
            .next_edge = nodes_[(size_t)node].head_edge,
        }
    );
    nodes_[(size_t)node].head_edge = edge_idx;
    return child;
}

std::optional<UnigramModel>
UnigramModel::from_gguf(const marmot_gguf_t *gguf, const Vocab &vocab, std::string &out_error, bool strict_validation) {
    out_error.clear();
    if (gguf == nullptr) {
        out_error = "gguf is null";
        return std::nullopt;
    }

    const marmot_gguf_kv_t *scores_kv = find_kv(gguf, "tokenizer.ggml.scores");
    if (scores_kv == nullptr || scores_kv->value.type != MARMOT_GGUF_TYPE_ARRAY ||
        scores_kv->value.data.array_value.type != MARMOT_GGUF_TYPE_FLOAT32) {
        out_error = "missing tokenizer.ggml.scores";
        return std::nullopt;
    }

    const marmot_gguf_array_t scores_arr = scores_kv->value.data.array_value;
    if (scores_arr.length != vocab.size() || scores_arr.data.float32_values == nullptr) {
        out_error = "tokenizer.ggml.scores length mismatch";
        return std::nullopt;
    }

    bool any_nonzero = false;
    for (size_t i = 0; i < scores_arr.length; ++i) {
        if (scores_arr.data.float32_values[i] != 0.0f) {
            any_nonzero = true;
            break;
        }
    }

    if (!any_nonzero && strict_validation) {
        out_error = "tokenizer.ggml.scores are all zero";
        return std::nullopt;
    }

    UnigramModel model;
    model.nodes_.push_back(TrieNode{.head_edge = -1, .token_id = MARMOT_TOKEN_ID_INVALID});
    model.scores_.assign(scores_arr.data.float32_values, scores_arr.data.float32_values + scores_arr.length);

    for (marmot_token_id_t id = 0; id < (marmot_token_id_t)vocab.size(); ++id) {
        if (vocab.is_special_token(id) || vocab.is_byte_token(id)) {
            continue;
        }
        const std::string_view piece = vocab.piece(id);
        if (piece.empty()) {
            continue;
        }

        int32_t node = 0;
        for (uint8_t ch : piece) {
            node = model.add_child(node, ch);
        }
        if (model.nodes_[(size_t)node].token_id == MARMOT_TOKEN_ID_INVALID) {
            model.nodes_[(size_t)node].token_id = id;
        }
    }

    if (model.edges_.empty()) {
        out_error = "unigram trie is empty";
        return std::nullopt;
    }

    return model;
}

marmot_error_t UnigramModel::encode(
    const Vocab &vocab, std::string_view normalized, std::vector<marmot_token_id_t> &out, std::string &out_error
) const {
    out_error.clear();
    out.clear();
    if (normalized.empty()) {
        return MARMOT_SUCCESS;
    }

    const size_t n = normalized.size();
    constexpr double kNegInf = -std::numeric_limits<double>::infinity();

    std::vector<double> dp(n + 1, kNegInf);
    std::vector<int32_t> prev(n + 1, -1);
    std::vector<marmot_token_id_t> token(n + 1, MARMOT_TOKEN_ID_INVALID);
    dp[0] = 0.0;

    for (size_t i = 0; i < n; ++i) {
        if (dp[i] == kNegInf) {
            continue;
        }

        bool matched = false;
        int32_t node = 0;
        for (size_t j = i; j < n; ++j) {
            node = find_child(node, (uint8_t)normalized[j]);
            if (node < 0) {
                break;
            }

            const marmot_token_id_t id = nodes_[(size_t)node].token_id;
            if (id == MARMOT_TOKEN_ID_INVALID) {
                continue;
            }

            const size_t next = j + 1;
            const double candidate = dp[i] + (double)scores_[(size_t)id];
            if (candidate > dp[next]) {
                dp[next] = candidate;
                prev[next] = (int32_t)i;
                token[next] = id;
                matched = true;
            }
        }

        if (!matched) {
            const marmot_token_id_t byte_id = vocab.byte_token_id((uint8_t)normalized[i]);
            if (byte_id == MARMOT_TOKEN_ID_INVALID) {
                out_error = "byte fallback missing";
                return MARMOT_ERROR_INVALID_OPERATION;
            }
            if (dp[i] >= dp[i + 1]) {
                dp[i + 1] = dp[i];
                prev[i + 1] = (int32_t)i;
                token[i + 1] = byte_id;
            }
        }
    }

    if (dp[n] == kNegInf) {
        out_error = "unigram tokenization failed";
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    std::vector<marmot_token_id_t> rev;
    for (int32_t at = (int32_t)n; at > 0;) {
        const marmot_token_id_t id = token[(size_t)at];
        const int32_t p = prev[(size_t)at];
        if (id == MARMOT_TOKEN_ID_INVALID || p < 0) {
            out_error = "unigram backtrack failed";
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        rev.push_back(id);
        at = p;
    }

    std::reverse(rev.begin(), rev.end());
    out = std::move(rev);
    return MARMOT_SUCCESS;
}

} // namespace marmot::tokenizer
