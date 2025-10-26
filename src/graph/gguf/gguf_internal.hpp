#pragma once

#include "marmot/graph/gguf_loader.h"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>

namespace marmot::gguf {

constexpr size_t kDefaultAlignment = 32;
constexpr uint32_t kMagic = 0x46554747;
constexpr uint32_t kVersionSupported = 3;

enum class ParseError {
    EndOfData,
    InvalidType,
    InvalidLength,
    AllocationFailed,
    UnsupportedVersion,
    MagicMismatch,
    InvalidTensorRank,
    InvalidDimension,
    TensorDataOutOfBounds,
};

constexpr std::string_view to_string(ParseError e) {
    switch (e) {
    case ParseError::EndOfData:
        return "Unexpected end of data";
    case ParseError::InvalidType:
        return "Invalid type";
    case ParseError::InvalidLength:
        return "Invalid length";
    case ParseError::AllocationFailed:
        return "Allocation failed";
    case ParseError::UnsupportedVersion:
        return "Unsupported GGUF version";
    case ParseError::MagicMismatch:
        return "GGUF magic mismatch";
    case ParseError::InvalidTensorRank:
        return "Invalid tensor rank";
    case ParseError::InvalidDimension:
        return "Invalid dimension";
    case ParseError::TensorDataOutOfBounds:
        return "Tensor data out of bounds";
    }
    return "Unknown error";
}

template <typename T>
using Result = std::expected<T, ParseError>;

struct ScopedFd {
    int value{-1};

    ScopedFd() = default;
    explicit ScopedFd(int fd) : value(fd) {}
    ScopedFd(const ScopedFd &) = delete;
    ScopedFd &operator=(const ScopedFd &) = delete;
    ScopedFd(ScopedFd &&other) noexcept : value(std::exchange(other.value, -1)) {}
    ScopedFd &operator=(ScopedFd &&other) noexcept {
        if (this != &other) {
            reset();
            value = std::exchange(other.value, -1);
        }
        return *this;
    }
    ~ScopedFd() {
        reset();
    }

    void reset() noexcept {
        if (value >= 0) {
            close(value);
            value = -1;
        }
    }

    [[nodiscard]] int release() noexcept {
        return std::exchange(value, -1);
    }
};

struct MappedRegion {
    void *ptr{nullptr};
    size_t size{0};

    MappedRegion() = default;
    MappedRegion(void *mapping, size_t length) : ptr(mapping), size(length) {}
    MappedRegion(const MappedRegion &) = delete;
    MappedRegion &operator=(const MappedRegion &) = delete;
    MappedRegion(MappedRegion &&other) noexcept
        : ptr(std::exchange(other.ptr, nullptr)), size(std::exchange(other.size, 0)) {}
    MappedRegion &operator=(MappedRegion &&other) noexcept {
        if (this != &other) {
            reset();
            ptr = std::exchange(other.ptr, nullptr);
            size = std::exchange(other.size, 0);
        }
        return *this;
    }
    ~MappedRegion() {
        reset();
    }

    [[nodiscard]] bool valid() const noexcept {
        return ptr != nullptr && ptr != MAP_FAILED;
    }

    void reset() noexcept {
        if (valid()) {
            munmap(ptr, size);
        }
        ptr = nullptr;
        size = 0;
    }

    [[nodiscard]] void *release() noexcept {
        size = 0;
        return std::exchange(ptr, nullptr);
    }
};

template <typename T>
concept Trivial = std::is_trivially_copyable_v<T>;

struct ByteCursor {
    std::span<const uint8_t> bytes{};
    size_t offset{0};

    [[nodiscard]] bool can_read(size_t length) const noexcept {
        return offset <= bytes.size() && bytes.size() - offset >= length;
    }

    [[nodiscard]] const uint8_t *advance(size_t length) noexcept {
        if (!can_read(length)) {
            return nullptr;
        }
        const uint8_t *p = bytes.data() + offset;
        offset += length;
        return p;
    }

    template <Trivial T>
    [[nodiscard]] Result<T> read() noexcept {
        const uint8_t *src = advance(sizeof(T));
        if (src == nullptr) {
            return std::unexpected(ParseError::EndOfData);
        }
        T out;
        std::memcpy(&out, src, sizeof(T));
        return out;
    }

    template <Trivial T>
    [[nodiscard]] Result<T> read(T &out) noexcept {
        auto result = read<T>();
        if (!result) {
            return result;
        }
        out = *result;
        return out;
    }

    [[nodiscard]] Result<std::span<const uint8_t>> read_span(size_t length) noexcept {
        const uint8_t *src = advance(length);
        if (src == nullptr) {
            return std::unexpected(ParseError::EndOfData);
        }
        return std::span<const uint8_t>(src, length);
    }
};

struct FreeDeleter {
    void operator()(void *p) const noexcept {
        free(p);
    }
};

struct KvArrayDeleter {
    size_t count{0};
    void operator()(marmot_gguf_kv_t *p) const noexcept;
};

struct TensorArrayDeleter {
    size_t count{0};
    void operator()(marmot_gguf_tensor_t *p) const noexcept;
};

void free_string(marmot_gguf_string_t *str);
void free_value(marmot_gguf_value_t *val);
void free_kv_array(marmot_gguf_kv_t *kv, size_t count);
void free_tensor_array(marmot_gguf_tensor_t *tensors, size_t count);

Result<void> read_string(ByteCursor &cursor, marmot_gguf_string_t *out);
Result<void> parse_value(ByteCursor &cursor, marmot_gguf_value_t *out);
Result<void> parse_array(ByteCursor &cursor, marmot_gguf_array_t *out);

class GgufFile {
  public:
    GgufFile() = default;
    GgufFile(ScopedFd fd, MappedRegion mapping, marmot_gguf_t *raw) noexcept
        : fd_(std::move(fd)), mapping_(std::move(mapping)), raw_(raw) {}

    ~GgufFile() {
        reset();
    }

    GgufFile(const GgufFile &) = delete;
    GgufFile &operator=(const GgufFile &) = delete;
    GgufFile(GgufFile &&other) noexcept {
        move_from(std::move(other));
    }
    GgufFile &operator=(GgufFile &&other) noexcept {
        if (this != &other) {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }

    [[nodiscard]] marmot_gguf_t *get() const noexcept {
        return raw_;
    }
    [[nodiscard]] marmot_gguf_t *release() noexcept {
        (void)fd_.release();
        (void)mapping_.release();
        auto *tmp = raw_;
        raw_ = nullptr;
        return tmp;
    }

    void reset() noexcept {
        if (raw_ != nullptr) {
            free_kv_array(raw_->kv, raw_->kv_count);
            free_tensor_array(raw_->tensors, raw_->tensor_count);
            if (raw_->data != nullptr && raw_->data != MAP_FAILED) {
                munmap(raw_->data, raw_->size);
            }
            if (raw_->fd >= 0) {
                close(raw_->fd);
            }
            delete[] raw_->kv;
            delete[] raw_->tensors;
            delete raw_;
        }
        raw_ = nullptr;
        mapping_.reset();
        fd_.reset();
    }

    [[nodiscard]] bool valid() const noexcept {
        return raw_ != nullptr;
    }

  private:
    void move_from(GgufFile &&other) noexcept {
        fd_ = std::move(other.fd_);
        mapping_ = std::move(other.mapping_);
        raw_ = std::exchange(other.raw_, nullptr);
    }

    ScopedFd fd_{};
    MappedRegion mapping_{};
    marmot_gguf_t *raw_{nullptr};
};

// Loader entry points (C++ impl; C wrappers live in loader_c_api.cpp)
marmot_gguf_t *load_file(const char *path);
void unload_file(marmot_gguf_t *gguf);
const marmot_gguf_kv_t *find_kv(const marmot_gguf_t *gguf, const char *key);
const marmot_gguf_tensor_t *find_tensor(const marmot_gguf_t *gguf, const char *name);

struct GgmlTypeMapping {
    marmot_quant_kind_t quant_kind;
    marmot_dtype_t dtype;
    marmot_qscheme_id_t qscheme;
    bool supported;
};

[[nodiscard]] GgmlTypeMapping map_ggml_type(uint32_t ggml_type);
[[nodiscard]] bool set_tensor_data_views(marmot_gguf_t *gguf, size_t metadata_end_offset);
[[nodiscard]] constexpr size_t compute_align(size_t offset, size_t alignment) {
    if (alignment == 0) {
        return offset;
    }
    const size_t remainder = offset % alignment;
    return remainder == 0 ? offset : offset + (alignment - remainder);
}

} // namespace marmot::gguf
