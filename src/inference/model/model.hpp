#pragma once

#include "marmot/graph/gguf_model.h"
#include "marmot/inference/model.h"

#include <memory>
#include <string>

namespace marmot::inference {

class Model {
  public:
    struct GgufDeleter {
        void operator()(marmot_gguf_model_t *model) const noexcept {
            marmot_gguf_model_destroy(model);
        }
    };

    using GgufOwner = std::unique_ptr<marmot_gguf_model_t, GgufDeleter>;

    static std::unique_ptr<Model>
    load_file(const char *path, const marmot_model_options_t &opts, marmot_error_t &status, std::string &error);

    [[nodiscard]] const std::string &path() const noexcept {
        return path_;
    }

    [[nodiscard]] const marmot_model_info_t &info() const noexcept {
        return info_;
    }

    [[nodiscard]] const marmot_gguf_model_t *gguf() const noexcept {
        return gguf_.get();
    }

  private:
    Model(std::string path, GgufOwner gguf, const marmot_model_info_t &info)
        : path_(std::move(path)), gguf_(std::move(gguf)), info_(info) {}

    std::string path_;
    GgufOwner gguf_;
    marmot_model_info_t info_{};
};

} // namespace marmot::inference
