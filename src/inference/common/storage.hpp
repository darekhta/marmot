#pragma once

#include "marmot/tensor.h"
#include "marmot/types.h"

#include <cstddef>
#include <expected>
#include <memory>
#include <span>

#include "tensor_ptr.hpp"

namespace marmot::inference {

class StorageBlock {
  public:
    [[nodiscard]] static std::expected<StorageBlock, marmot_error_t> create(
        marmot_backend_type_t backend, std::span<const size_t> shape, marmot_dtype_t dtype,
        const marmot_context_t *ctx = nullptr
    );

    [[nodiscard]] static std::expected<StorageBlock, marmot_error_t> create_quantized(
        marmot_backend_type_t backend, std::span<const size_t> shape, marmot_quant_kind_t quant_kind,
        const marmot_context_t *ctx = nullptr
    );

    StorageBlock() = default;
    ~StorageBlock() = default;

    StorageBlock(StorageBlock &&) noexcept = default;
    StorageBlock &operator=(StorageBlock &&) noexcept = default;

    StorageBlock(const StorageBlock &) = delete;
    StorageBlock &operator=(const StorageBlock &) = delete;

    [[nodiscard]] marmot_tensor_t *tensor() noexcept {
        return tensor_.get();
    }
    [[nodiscard]] const marmot_tensor_t *tensor() const noexcept {
        return tensor_.get();
    }

    [[nodiscard]] void *data() noexcept;
    [[nodiscard]] const void *data() const noexcept;

    [[nodiscard]] size_t size_bytes() const noexcept;
    [[nodiscard]] bool valid() const noexcept {
        return tensor_ != nullptr;
    }

    explicit operator bool() const noexcept {
        return valid();
    }

  private:
    TensorPtr tensor_;
    explicit StorageBlock(TensorPtr tensor);
};

class SharedStorageBlock {
  public:
    [[nodiscard]] static std::expected<SharedStorageBlock, marmot_error_t> create(
        marmot_backend_type_t backend, std::span<const size_t> shape, marmot_dtype_t dtype,
        const marmot_context_t *ctx = nullptr
    );

    [[nodiscard]] static std::expected<SharedStorageBlock, marmot_error_t> create_quantized(
        marmot_backend_type_t backend, std::span<const size_t> shape, marmot_quant_kind_t quant_kind,
        const marmot_context_t *ctx = nullptr
    );

    SharedStorageBlock() = default;
    ~SharedStorageBlock() = default;

    SharedStorageBlock(const SharedStorageBlock &) = default;
    SharedStorageBlock &operator=(const SharedStorageBlock &) = default;

    SharedStorageBlock(SharedStorageBlock &&) noexcept = default;
    SharedStorageBlock &operator=(SharedStorageBlock &&) noexcept = default;

    [[nodiscard]] marmot_tensor_t *tensor() noexcept {
        return tensor_.get();
    }
    [[nodiscard]] const marmot_tensor_t *tensor() const noexcept {
        return tensor_.get();
    }

    [[nodiscard]] void *data() noexcept;
    [[nodiscard]] const void *data() const noexcept;

    [[nodiscard]] size_t size_bytes() const noexcept;
    [[nodiscard]] bool valid() const noexcept {
        return tensor_ != nullptr;
    }

    [[nodiscard]] bool shared() const noexcept {
        return tensor_.use_count() > 1;
    }

    explicit operator bool() const noexcept {
        return valid();
    }

  private:
    std::shared_ptr<marmot_tensor_t> tensor_;
    explicit SharedStorageBlock(std::shared_ptr<marmot_tensor_t> tensor);
};

} // namespace marmot::inference
