#include "error.hpp"

#include <algorithm>
#include <cstring>

namespace marmot::gguf {

marmot_error_info_t Error::to_info() const noexcept {
    marmot_error_info_t info{
        .code = code_,
        .message = {0},
        .file = nullptr,
        .line = 0,
        .function = nullptr,
    };

    if (!message_.empty()) {
        const size_t copied = std::min(message_.size(), sizeof(info.message) - 1);
        std::memcpy(info.message, message_.data(), copied);
        info.message[copied] = '\0';
    }

    return info;
}

} // namespace marmot::gguf
