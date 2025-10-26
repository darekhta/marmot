#include "model_wordpiece.hpp"

#include <string>

#include "vocab.hpp"

namespace marmot::tokenizer {

marmot_error_t WordPieceModel::encode(
    const Vocab &vocab, std::string_view text, marmot_token_id_t unk_id, std::vector<marmot_token_id_t> &out,
    std::string &out_error
) const {
    out_error.clear();
    out.clear();

    if (text.empty()) {
        return MARMOT_SUCCESS;
    }

    size_t pos = 0;
    while (pos < text.size()) {
        while (pos < text.size() && text[pos] == ' ') {
            ++pos;
        }
        if (pos >= text.size()) {
            break;
        }
        size_t end = pos;
        while (end < text.size() && text[end] != ' ') {
            ++end;
        }

        const std::string_view word = text.substr(pos, end - pos);
        size_t wpos = 0;
        bool ok = true;
        while (wpos < word.size()) {
            size_t best_end = word.size();
            marmot_token_id_t best_id = MARMOT_TOKEN_ID_INVALID;

            while (best_end > wpos) {
                std::string piece;
                if (wpos > 0) {
                    piece.reserve(kContinuationPrefix.size() + (best_end - wpos));
                    piece.append(kContinuationPrefix);
                    piece.append(word.substr(wpos, best_end - wpos));
                } else {
                    piece.assign(word.substr(wpos, best_end - wpos));
                }

                if (vocab.piece_to_id(piece, best_id)) {
                    break;
                }

                best_id = MARMOT_TOKEN_ID_INVALID;
                --best_end;
            }

            if (best_id == MARMOT_TOKEN_ID_INVALID) {
                ok = false;
                break;
            }

            out.push_back(best_id);
            wpos = best_end;
        }

        if (!ok) {
            if (unk_id == MARMOT_TOKEN_ID_INVALID) {
                out_error = "wordpiece tokenization failed";
                return MARMOT_ERROR_INVALID_OPERATION;
            }
            out.push_back(unk_id);
        }

        pos = end;
    }

    return MARMOT_SUCCESS;
}

} // namespace marmot::tokenizer
