#include "model_bpe.hpp"

#include "marmot/graph/gguf_loader.h"

#include <algorithm>
#include <queue>

#include "utf8.hpp"
#include "vocab.hpp"

namespace marmot::tokenizer {

static uint64_t pair_key(marmot_token_id_t lhs, marmot_token_id_t rhs) {
    return ((uint64_t)(uint32_t)lhs << 32) | (uint32_t)rhs;
}

static const marmot_gguf_kv_t *find_kv(const marmot_gguf_t *gguf, const char *key) {
    return marmot_gguf_find_kv(gguf, key);
}

std::optional<BpeModel>
BpeModel::from_gguf(const marmot_gguf_t *gguf, const Vocab &vocab, std::string &out_error, bool strict_validation) {
    out_error.clear();
    if (gguf == nullptr) {
        out_error = "gguf is null";
        return std::nullopt;
    }

    const marmot_gguf_kv_t *merges_kv = find_kv(gguf, "tokenizer.ggml.merges");
    if (merges_kv == nullptr || merges_kv->value.type != MARMOT_GGUF_TYPE_ARRAY ||
        merges_kv->value.data.array_value.type != MARMOT_GGUF_TYPE_STRING) {
        out_error = "missing tokenizer.ggml.merges";
        return std::nullopt;
    }

    const marmot_gguf_array_t merges_arr = merges_kv->value.data.array_value;
    if (merges_arr.length == 0 || merges_arr.data.string_values == nullptr) {
        out_error = "tokenizer.ggml.merges is empty";
        return std::nullopt;
    }

    BpeModel model;
    model.merges_.reserve((size_t)((double)merges_arr.length * 1.3));

    std::string merged_piece;
    for (size_t i = 0; i < merges_arr.length; ++i) {
        const marmot_gguf_string_t merge_str = merges_arr.data.string_values[i];
        const std::string_view merge(merge_str.data != nullptr ? merge_str.data : "", merge_str.length);

        const size_t sep = merge.find(' ');
        if (sep == std::string_view::npos || sep == 0 || sep + 1 >= merge.size()) {
            if (strict_validation) {
                out_error = "invalid merge entry";
                return std::nullopt;
            }
            continue;
        }

        const std::string_view left = merge.substr(0, sep);
        const std::string_view right = merge.substr(sep + 1);

        marmot_token_id_t left_id = MARMOT_TOKEN_ID_INVALID;
        marmot_token_id_t right_id = MARMOT_TOKEN_ID_INVALID;
        if (!vocab.piece_to_id(left, left_id) || !vocab.piece_to_id(right, right_id)) {
            if (strict_validation) {
                out_error = "merge references unknown vocab piece";
                return std::nullopt;
            }
            continue;
        }

        merged_piece.clear();
        merged_piece.reserve(left.size() + right.size());
        merged_piece.append(left);
        merged_piece.append(right);

        marmot_token_id_t merged_id = MARMOT_TOKEN_ID_INVALID;
        if (!vocab.piece_to_id(std::string_view(merged_piece), merged_id)) {
            if (strict_validation) {
                out_error = "merge result missing from vocab";
                return std::nullopt;
            }
            continue;
        }

        const uint64_t key = pair_key(left_id, right_id);
        auto it = model.merges_.find(key);
        if (it == model.merges_.end()) {
            model.merges_.emplace(key, MergeInfo{.rank = (uint32_t)i, .merged_id = merged_id});
            continue;
        }
        if ((uint32_t)i < it->second.rank) {
            it->second.rank = (uint32_t)i;
            it->second.merged_id = merged_id;
        }
    }

    if (model.merges_.empty()) {
        out_error = "no usable merges";
        return std::nullopt;
    }

    return model;
}

struct MergeCandidate {
    uint32_t rank;
    int32_t left;
    int32_t right;
    uint32_t version;
    uint32_t right_version;
};

struct MergeCandidateGreater {
    bool operator()(const MergeCandidate &a, const MergeCandidate &b) const noexcept {
        if (a.rank != b.rank) {
            return a.rank > b.rank;
        }
        return a.left > b.left;
    }
};

struct BpeNode {
    marmot_token_id_t id;
    int32_t prev;
    int32_t next;
    uint32_t version;
    bool alive;
};

static bool push_candidate(
    const std::unordered_map<uint64_t, BpeModel::MergeInfo> &merges, const std::vector<BpeNode> &nodes, int32_t left,
    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidateGreater> &queue
) {
    if (left < 0 || left >= (int32_t)nodes.size()) {
        return false;
    }
    if (!nodes[(size_t)left].alive) {
        return false;
    }
    const int32_t right = nodes[(size_t)left].next;
    if (right < 0 || right >= (int32_t)nodes.size()) {
        return false;
    }
    if (!nodes[(size_t)right].alive) {
        return false;
    }

    const marmot_token_id_t lhs = nodes[(size_t)left].id;
    const marmot_token_id_t rhs = nodes[(size_t)right].id;
    if (lhs < 0 || rhs < 0) {
        return false;
    }

    auto it = merges.find(pair_key(lhs, rhs));
    if (it == merges.end()) {
        return false;
    }

    queue.push(
        MergeCandidate{
            .rank = it->second.rank,
            .left = left,
            .right = right,
            .version = nodes[(size_t)left].version,
            .right_version = nodes[(size_t)right].version,
        }
    );
    return true;
}

marmot_error_t BpeModel::encode(
    const Vocab &vocab, std::string_view normalized, std::vector<marmot_token_id_t> &out, std::string &out_error
) const {
    out_error.clear();
    out.clear();
    if (normalized.empty()) {
        return MARMOT_SUCCESS;
    }

    std::vector<marmot_token_id_t> ids;
    ids.reserve(normalized.size());

    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(normalized.data());
    size_t offset = 0;
    while (offset < normalized.size()) {
        const Utf8DecodeResult decoded = utf8_decode(bytes + offset, normalized.size() - offset);
        const size_t n = decoded.valid && decoded.length > 0 ? decoded.length : 1;
        const std::string_view piece = normalized.substr(offset, n);

        marmot_token_id_t id = MARMOT_TOKEN_ID_INVALID;
        if (decoded.valid && vocab.piece_to_id(piece, id)) {
            ids.push_back(id);
            offset += n;
            continue;
        }

        for (size_t i = 0; i < n; ++i) {
            const marmot_token_id_t byte_id = vocab.byte_token_id(bytes[offset + i]);
            if (byte_id == MARMOT_TOKEN_ID_INVALID) {
                out_error = "byte fallback missing";
                return MARMOT_ERROR_INVALID_OPERATION;
            }
            ids.push_back(byte_id);
        }
        offset += n;
    }

    if (ids.size() <= 1) {
        out = std::move(ids);
        return MARMOT_SUCCESS;
    }

    std::vector<BpeNode> nodes;
    nodes.resize(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        nodes[i] = BpeNode{
            .id = ids[i],
            .prev = i == 0 ? -1 : (int32_t)(i - 1),
            .next = (i + 1) < ids.size() ? (int32_t)(i + 1) : -1,
            .version = 0,
            .alive = true,
        };
    }

    std::priority_queue<MergeCandidate, std::vector<MergeCandidate>, MergeCandidateGreater> queue;
    for (int32_t i = 0; i + 1 < (int32_t)nodes.size(); ++i) {
        (void)push_candidate(merges_, nodes, i, queue);
    }

    while (!queue.empty()) {
        const MergeCandidate candidate = queue.top();
        queue.pop();

        if (candidate.left < 0 || candidate.left >= (int32_t)nodes.size()) {
            continue;
        }
        BpeNode &left = nodes[(size_t)candidate.left];
        if (!left.alive || left.version != candidate.version) {
            continue;
        }
        const int32_t right_idx = left.next;
        if (right_idx != candidate.right || right_idx < 0 || right_idx >= (int32_t)nodes.size()) {
            continue;
        }
        BpeNode &right = nodes[(size_t)right_idx];
        if (!right.alive || right.version != candidate.right_version) {
            continue;
        }

        auto it = merges_.find(pair_key(left.id, right.id));
        if (it == merges_.end()) {
            continue;
        }

        left.id = it->second.merged_id;
        right.alive = false;

        const int32_t next_idx = right.next;
        left.next = next_idx;
        if (next_idx >= 0) {
            nodes[(size_t)next_idx].prev = candidate.left;
        }
        left.version++;

        if (left.prev >= 0) {
            (void)push_candidate(merges_, nodes, left.prev, queue);
        }
        if (left.next >= 0) {
            (void)push_candidate(merges_, nodes, candidate.left, queue);
        }
    }

    out.reserve(nodes.size());
    for (int32_t idx = 0; idx >= 0; idx = nodes[(size_t)idx].next) {
        if (nodes[(size_t)idx].alive) {
            out.push_back(nodes[(size_t)idx].id);
        }
        if (nodes[(size_t)idx].next < 0) {
            break;
        }
    }

    return MARMOT_SUCCESS;
}

} // namespace marmot::tokenizer
