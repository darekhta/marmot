#include "gguf_reader.hpp"

#include "gguf_internal.hpp"

namespace marmot::gguf {

GgufFilePtr GgufReader::load_file(const char *path, Error &err) const {
    err.clear();
    if (path == nullptr) {
        err.set(MARMOT_ERROR_INVALID_ARGUMENT, "path is null");
        return GgufFilePtr(nullptr, &marmot::gguf::unload_file);
    }

    marmot_gguf_t *raw = marmot::gguf::load_file(path);
    if (raw == nullptr) {
        err.set(MARMOT_ERROR_INVALID_OPERATION, "failed to load GGUF file");
        return GgufFilePtr(nullptr, &marmot::gguf::unload_file);
    }

    return GgufFilePtr(raw, &marmot::gguf::unload_file);
}

GgufFilePtr GgufReader::load_memory(std::span<const std::byte> data, Error &err) const {
    err.set(MARMOT_ERROR_NOT_IMPLEMENTED, "GGUF memory loading not implemented");
    (void)data;
    return GgufFilePtr(nullptr, &marmot::gguf::unload_file);
}

} // namespace marmot::gguf
