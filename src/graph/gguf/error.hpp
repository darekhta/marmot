#pragma once

#include "marmot/error.h"

#include <string>
#include <string_view>

namespace marmot::gguf {

class Error {
  public:
    Error() = default;

    [[nodiscard]] bool has_error() const noexcept {
        return code_ != MARMOT_SUCCESS;
    }
    [[nodiscard]] marmot_error_t code() const noexcept {
        return code_;
    }
    [[nodiscard]] std::string_view message() const noexcept {
        return message_;
    }

    void clear() noexcept {
        code_ = MARMOT_SUCCESS;
        message_.clear();
    }

    void set(marmot_error_t code, std::string_view message) {
        code_ = code;
        message_ = std::string(message);
    }

    [[nodiscard]] marmot_error_info_t to_info() const noexcept;

  private:
    marmot_error_t code_{MARMOT_SUCCESS};
    std::string message_{};
};

} // namespace marmot::gguf
