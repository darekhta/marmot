#pragma once

namespace marmot::gguf {

class OpRegistry {
  public:
    OpRegistry() = default;

    void set_custom_registry(const void *custom) noexcept {
        custom_registry_ = custom;
    }
    [[nodiscard]] const void *custom_registry() const noexcept {
        return custom_registry_;
    }

  private:
    const void *custom_registry_{nullptr};
};

} // namespace marmot::gguf
