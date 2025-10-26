#include "gguf_validator.hpp"

#include "gguf_internal.hpp"

namespace marmot::gguf {

marmot_error_t GgufValidator::validate(const marmot_gguf_t *gguf, Error &err) const {
    err.clear();
    if (gguf == nullptr) {
        err.set(MARMOT_ERROR_INVALID_ARGUMENT, "gguf is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (gguf->version != kVersionSupported) {
        err.set(MARMOT_ERROR_NOT_IMPLEMENTED, "unsupported GGUF version");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (gguf->alignment == 0) {
        err.set(MARMOT_ERROR_INVALID_ARGUMENT, "alignment is zero");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    return MARMOT_SUCCESS;
}

} // namespace marmot::gguf
