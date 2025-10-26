#pragma once

#include "marmot/graph/gguf_loader.h"
#include "marmot/graph/gguf_model.h"

#include <cstring>
#include <memory>
#include <optional>
#include <string_view>

namespace marmot::gguf {

class Model {
  public:
    Model(marmot_gguf_t *gguf, marmot_backend_type_t backend) : gguf_(gguf), backend_(backend) {}

    ~Model() {
        if (gguf_) {
            marmot_gguf_unload(gguf_);
        }
    }

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    Model(Model &&) = default;
    Model &operator=(Model &&) = default;

    [[nodiscard]] const marmot_gguf_t *file() const {
        return gguf_;
    }
    [[nodiscard]] marmot_backend_type_t backend() const {
        return backend_;
    }

    [[nodiscard]] const marmot_tensor_t *get_tensor(const char *name) const {
        const marmot_gguf_tensor_t *info = marmot_gguf_find_tensor(gguf_, name);
        return info ? info->tensor : nullptr;
    }

    [[nodiscard]] const marmot_gguf_tensor_t *get_tensor_info(size_t index) const {
        if (index >= gguf_->tensor_count)
            return nullptr;
        return &gguf_->tensors[index];
    }

    [[nodiscard]] size_t tensor_count() const {
        return gguf_->tensor_count;
    }

    [[nodiscard]] bool get_metadata(marmot_gguf_model_meta_t *out) const;

  private:
    [[nodiscard]] std::optional<size_t> read_u64(const char *key) const;
    [[nodiscard]] std::optional<float> read_f32(const char *key) const;
    [[nodiscard]] std::optional<std::string_view> read_string(const char *key) const;

    marmot_gguf_t *gguf_;
    marmot_backend_type_t backend_;
};

[[nodiscard]] const char *get_architecture(const marmot_gguf_t *gguf);

} // namespace marmot::gguf
