#pragma once

#include "marmot/graph/gguf_loader.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

#include "error.hpp"

namespace marmot::gguf {

using GgufFilePtr = std::unique_ptr<marmot_gguf_t, void (*)(marmot_gguf_t *)>;

class GgufReader {
  public:
    [[nodiscard]] GgufFilePtr load_file(const char *path, Error &err) const;
    [[nodiscard]] GgufFilePtr load_memory(std::span<const std::byte> data, Error &err) const;
};

} // namespace marmot::gguf
