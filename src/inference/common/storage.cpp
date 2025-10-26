#include "storage.hpp"

#include "marmot/quant_block.h"

namespace marmot::inference {

void TensorDeleter::operator()(marmot_tensor_t *ptr) const noexcept {
    if (ptr != nullptr) {
        marmot_tensor_destroy(ptr);
    }
}

StorageBlock::StorageBlock(TensorPtr tensor) : tensor_(std::move(tensor)) {}

SharedStorageBlock::SharedStorageBlock(std::shared_ptr<marmot_tensor_t> tensor) : tensor_(std::move(tensor)) {}

std::expected<StorageBlock, marmot_error_t> StorageBlock::create(
    marmot_backend_type_t backend, std::span<const size_t> shape, marmot_dtype_t dtype, const marmot_context_t *ctx
) {
    if (shape.empty() || shape.size() > MARMOT_MAX_DIMS) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape.data(), shape.size(), dtype);
    if (tensor == nullptr) {
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }

    tensor->backend = backend;
    tensor->memory_location = MARMOT_MEMORY_HOST;
    tensor->needs_sync = false;

    return StorageBlock(TensorPtr(tensor));
}

std::expected<SharedStorageBlock, marmot_error_t> SharedStorageBlock::create(
    marmot_backend_type_t backend, std::span<const size_t> shape, marmot_dtype_t dtype, const marmot_context_t *ctx
) {
    if (shape.empty() || shape.size() > MARMOT_MAX_DIMS) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape.data(), shape.size(), dtype);
    if (tensor == nullptr) {
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }

    tensor->backend = backend;
    tensor->memory_location = MARMOT_MEMORY_HOST;
    tensor->needs_sync = false;

    return SharedStorageBlock(std::shared_ptr<marmot_tensor_t>(tensor, TensorDeleter{}));
}

std::expected<StorageBlock, marmot_error_t> StorageBlock::create_quantized(
    marmot_backend_type_t backend, std::span<const size_t> shape, marmot_quant_kind_t quant_kind,
    const marmot_context_t *ctx
) {
    if (shape.empty() || shape.size() > MARMOT_MAX_DIMS) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_dtype_t storage_dtype = MARMOT_DTYPE_UINT8;
    marmot_quant_layout_t layout = MARMOT_QUANT_LAYOUT_GENERIC;
    if (quant_kind != MARMOT_QUANT_KIND_GENERIC) {
        const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(quant_kind);
        if (traits == nullptr || !traits->is_block_quantized) {
            return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
        }
        storage_dtype = traits->storage_dtype;
        layout = traits->layout;
    }

    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape.data(), shape.size(), storage_dtype);
    if (tensor == nullptr) {
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }

    tensor->backend = backend;
    tensor->quant_kind = quant_kind;
    tensor->quant_layout = layout;
    tensor->memory_location = MARMOT_MEMORY_HOST;
    tensor->needs_sync = false;

    return StorageBlock(TensorPtr(tensor));
}

std::expected<SharedStorageBlock, marmot_error_t> SharedStorageBlock::create_quantized(
    marmot_backend_type_t backend, std::span<const size_t> shape, marmot_quant_kind_t quant_kind,
    const marmot_context_t *ctx
) {
    if (shape.empty() || shape.size() > MARMOT_MAX_DIMS) {
        return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
    }

    marmot_dtype_t storage_dtype = MARMOT_DTYPE_UINT8;
    marmot_quant_layout_t layout = MARMOT_QUANT_LAYOUT_GENERIC;
    if (quant_kind != MARMOT_QUANT_KIND_GENERIC) {
        const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(quant_kind);
        if (traits == nullptr || !traits->is_block_quantized) {
            return std::unexpected(MARMOT_ERROR_INVALID_ARGUMENT);
        }
        storage_dtype = traits->storage_dtype;
        layout = traits->layout;
    }

    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape.data(), shape.size(), storage_dtype);
    if (tensor == nullptr) {
        return std::unexpected(MARMOT_ERROR_OUT_OF_MEMORY);
    }

    tensor->backend = backend;
    tensor->quant_kind = quant_kind;
    tensor->quant_layout = layout;
    tensor->memory_location = MARMOT_MEMORY_HOST;
    tensor->needs_sync = false;

    return SharedStorageBlock(std::shared_ptr<marmot_tensor_t>(tensor, TensorDeleter{}));
}

void *StorageBlock::data() noexcept {
    return tensor_ ? tensor_->data : nullptr;
}

const void *StorageBlock::data() const noexcept {
    return tensor_ ? tensor_->data : nullptr;
}

size_t StorageBlock::size_bytes() const noexcept {
    return tensor_ ? marmot_tensor_size_bytes(tensor_.get()) : 0;
}

void *SharedStorageBlock::data() noexcept {
    return tensor_ ? tensor_->data : nullptr;
}

const void *SharedStorageBlock::data() const noexcept {
    return tensor_ ? tensor_->data : nullptr;
}

size_t SharedStorageBlock::size_bytes() const noexcept {
    return tensor_ ? marmot_tensor_size_bytes(tensor_.get()) : 0;
}

} // namespace marmot::inference
