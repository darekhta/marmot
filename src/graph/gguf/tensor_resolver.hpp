#pragma once

#include <cstdint>

namespace marmot::gguf {

class TensorResolver {
  public:
    TensorResolver() = default;

    void set_flags(uint64_t flags) noexcept {
        flags_ = flags;
    }
    [[nodiscard]] uint64_t flags() const noexcept {
        return flags_;
    }

  private:
    uint64_t flags_{0};
};

} // namespace marmot::gguf
