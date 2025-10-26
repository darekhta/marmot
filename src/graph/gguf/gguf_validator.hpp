#pragma once

#include "marmot/graph/gguf_loader.h"

#include "error.hpp"

namespace marmot::gguf {

class GgufValidator {
  public:
    [[nodiscard]] marmot_error_t validate(const marmot_gguf_t *gguf, Error &err) const;
};

} // namespace marmot::gguf
