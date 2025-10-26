#include "marmot/config.h"
#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/macros.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <new>
#include <ranges>
#include <sys/stat.h>
#include <vector>

#include "gguf_internal.hpp"

namespace marmot::gguf {

void free_string(marmot_gguf_string_t *str) {
    if (str == nullptr) {
        return;
    }
    delete[] str->data;
    str->data = nullptr;
    str->length = 0;
}

void free_value(marmot_gguf_value_t *val) {
    if (val == nullptr) {
        return;
    }
    if (val->type == MARMOT_GGUF_TYPE_STRING) {
        free_string(&val->data.string_value);
        return;
    }
    if (val->type != MARMOT_GGUF_TYPE_ARRAY) {
        return;
    }
    marmot_gguf_array_t *arr = &val->data.array_value;
    if (arr->data.raw == nullptr) {
        return;
    }
    if (arr->type == MARMOT_GGUF_TYPE_STRING) {
        for (size_t i = 0; i < arr->length; ++i) {
            free_string(&arr->data.string_values[i]);
        }
        delete[] arr->data.string_values;
    } else {
        delete[] static_cast<char *>(arr->data.raw);
    }
    arr->data.raw = nullptr;
    arr->length = 0;
}

void free_kv_array(marmot_gguf_kv_t *kv, size_t count) {
    if (kv == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        free_value(&kv[i].value);
        delete[] kv[i].key;
        kv[i].key = nullptr;
    }
}

void free_tensor_array(marmot_gguf_tensor_t *tensors, size_t count) {
    if (tensors == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        delete[] tensors[i].name;
        tensors[i].name = nullptr;
        marmot_tensor_destroy(tensors[i].tensor);
        tensors[i].tensor = nullptr;
    }
}

void KvArrayDeleter::operator()(marmot_gguf_kv_t *ptr) const noexcept {
    if (ptr != nullptr) {
        free_kv_array(ptr, count);
        delete[] ptr;
    }
}

void TensorArrayDeleter::operator()(marmot_gguf_tensor_t *ptr) const noexcept {
    if (ptr != nullptr) {
        free_tensor_array(ptr, count);
        delete[] ptr;
    }
}

Result<void> read_string(ByteCursor &cursor, marmot_gguf_string_t *out) {
    auto len_result = cursor.read<uint64_t>();
    if (!len_result)
        return std::unexpected(len_result.error());

    uint64_t len_u64 = *len_result;
    if (len_u64 > SIZE_MAX - 1) {
        return std::unexpected(ParseError::InvalidLength);
    }
    size_t len = static_cast<size_t>(len_u64);

    auto span_result = cursor.read_span(len);
    if (!span_result) {
        return std::unexpected(span_result.error());
    }

    auto buf = std::make_unique<char[]>(len + 1);
    std::memcpy(buf.get(), span_result->data(), len);
    buf[len] = '\0';

    out->data = buf.release();
    out->length = len;
    return {};
}

Result<void> parse_array(ByteCursor &cursor, marmot_gguf_array_t *out) {
    uint32_t element_type_u32 = 0;
    auto type_ok = cursor.read(element_type_u32);
    if (!type_ok || element_type_u32 > MARMOT_GGUF_TYPE_FLOAT64 || element_type_u32 == MARMOT_GGUF_TYPE_ARRAY) {
        return std::unexpected(ParseError::InvalidType);
    }
    uint64_t length_u64 = 0;
    auto len_ok = cursor.read(length_u64);
    if (!len_ok || length_u64 > SIZE_MAX) {
        return std::unexpected(ParseError::InvalidLength);
    }
    const size_t length = static_cast<size_t>(length_u64);

    out->type = static_cast<marmot_gguf_value_type_t>(element_type_u32);
    out->length = length;
    if (length == 0) {
        out->data.raw = nullptr;
        return {};
    }

    const size_t elem_size = (element_type_u32 == MARMOT_GGUF_TYPE_UINT8 || element_type_u32 == MARMOT_GGUF_TYPE_INT8 ||
                              element_type_u32 == MARMOT_GGUF_TYPE_BOOL)
        ? sizeof(uint8_t)
        : (element_type_u32 == MARMOT_GGUF_TYPE_UINT16 || element_type_u32 == MARMOT_GGUF_TYPE_INT16) ? sizeof(uint16_t)
        : (element_type_u32 == MARMOT_GGUF_TYPE_UINT32 || element_type_u32 == MARMOT_GGUF_TYPE_INT32 ||
           element_type_u32 == MARMOT_GGUF_TYPE_FLOAT32)
        ? sizeof(uint32_t)
        : sizeof(uint64_t);

    std::unique_ptr<char[]> data_owner;

    if (out->type != MARMOT_GGUF_TYPE_STRING) {
        if (length > SIZE_MAX / elem_size) {
            return std::unexpected(ParseError::InvalidLength);
        }
        data_owner = std::make_unique<char[]>(length * elem_size);
        std::memset(data_owner.get(), 0, length * elem_size);
    } else {
        data_owner = std::unique_ptr<char[]>(reinterpret_cast<char *>(new marmot_gguf_string_t[length]()));
    }

    for (size_t i = 0; i < length; ++i) {
        switch (out->type) {
        case MARMOT_GGUF_TYPE_UINT8:
        case MARMOT_GGUF_TYPE_BOOL: {
            uint8_t v = 0;
            if (!cursor.read(v))
                return std::unexpected(ParseError::EndOfData);
            if (out->type == MARMOT_GGUF_TYPE_BOOL) {
                reinterpret_cast<bool *>(data_owner.get())[i] = v != 0;
            } else {
                reinterpret_cast<uint8_t *>(data_owner.get())[i] = v;
            }
            break;
        }
        case MARMOT_GGUF_TYPE_INT8: {
            uint8_t v = 0;
            if (!cursor.read(v))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<int8_t *>(data_owner.get())[i] = static_cast<int8_t>(v);
            break;
        }
        case MARMOT_GGUF_TYPE_UINT16: {
            uint16_t v = 0;
            if (!cursor.read(v))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<uint16_t *>(data_owner.get())[i] = v;
            break;
        }
        case MARMOT_GGUF_TYPE_INT16: {
            uint16_t v = 0;
            if (!cursor.read(v))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<int16_t *>(data_owner.get())[i] = static_cast<int16_t>(v);
            break;
        }
        case MARMOT_GGUF_TYPE_UINT32: {
            uint32_t v = 0;
            if (!cursor.read(v))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<uint32_t *>(data_owner.get())[i] = v;
            break;
        }
        case MARMOT_GGUF_TYPE_INT32: {
            uint32_t tmp = 0;
            if (!cursor.read(tmp))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<int32_t *>(data_owner.get())[i] = static_cast<int32_t>(tmp);
            break;
        }
        case MARMOT_GGUF_TYPE_UINT64: {
            uint64_t v = 0;
            if (!cursor.read(v))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<uint64_t *>(data_owner.get())[i] = v;
            break;
        }
        case MARMOT_GGUF_TYPE_INT64: {
            uint64_t v = 0;
            if (!cursor.read(v))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<int64_t *>(data_owner.get())[i] = static_cast<int64_t>(v);
            break;
        }
        case MARMOT_GGUF_TYPE_FLOAT32: {
            float f = 0.0f;
            if (!cursor.read(f))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<float *>(data_owner.get())[i] = f;
            break;
        }
        case MARMOT_GGUF_TYPE_FLOAT64: {
            double f = 0.0;
            if (!cursor.read(f))
                return std::unexpected(ParseError::EndOfData);
            reinterpret_cast<double *>(data_owner.get())[i] = f;
            break;
        }
        case MARMOT_GGUF_TYPE_STRING: {
            auto *strings = reinterpret_cast<marmot_gguf_string_t *>(data_owner.get());
            auto s_ok = read_string(cursor, &strings[i]);
            if (!s_ok) {
                for (size_t j = 0; j < i; ++j) {
                    free_string(&strings[j]);
                }
                return s_ok;
            }
            break;
        }
        default:
            return std::unexpected(ParseError::InvalidType);
        }
    }

    out->data.raw = data_owner.release();
    return {};
}

Result<void> parse_value(ByteCursor &cursor, marmot_gguf_value_t *out) {
    uint32_t type = 0;
    auto type_ok = cursor.read(type);
    if (!type_ok || type > MARMOT_GGUF_TYPE_FLOAT64) {
        return std::unexpected(ParseError::InvalidType);
    }
    out->type = static_cast<marmot_gguf_value_type_t>(type);

    switch (out->type) {
    case MARMOT_GGUF_TYPE_UINT8:
    case MARMOT_GGUF_TYPE_INT8: {
        uint8_t tmp = 0;
        if (!cursor.read(tmp)) {
            return std::unexpected(ParseError::EndOfData);
        }
        if (out->type == MARMOT_GGUF_TYPE_UINT8) {
            out->data.uint8_value = tmp;
        } else {
            out->data.int8_value = static_cast<int8_t>(tmp);
        }
        return {};
    }
    case MARMOT_GGUF_TYPE_BOOL: {
        uint8_t tmp = 0;
        if (!cursor.read(tmp)) {
            return std::unexpected(ParseError::EndOfData);
        }
        out->data.bool_value = tmp != 0;
        return {};
    }
    case MARMOT_GGUF_TYPE_UINT16:
    case MARMOT_GGUF_TYPE_INT16: {
        uint16_t tmp = 0;
        if (!cursor.read(tmp)) {
            return std::unexpected(ParseError::EndOfData);
        }
        if (out->type == MARMOT_GGUF_TYPE_UINT16) {
            out->data.uint16_value = tmp;
        } else {
            out->data.int16_value = static_cast<int16_t>(tmp);
        }
        return {};
    }
    case MARMOT_GGUF_TYPE_UINT32:
    case MARMOT_GGUF_TYPE_INT32: {
        uint32_t tmp = 0;
        if (!cursor.read(tmp)) {
            return std::unexpected(ParseError::EndOfData);
        }
        if (out->type == MARMOT_GGUF_TYPE_UINT32) {
            out->data.uint32_value = tmp;
        } else {
            out->data.int32_value = static_cast<int32_t>(tmp);
        }
        return {};
    }
    case MARMOT_GGUF_TYPE_UINT64:
    case MARMOT_GGUF_TYPE_INT64: {
        uint64_t tmp = 0;
        if (!cursor.read(tmp)) {
            return std::unexpected(ParseError::EndOfData);
        }
        if (out->type == MARMOT_GGUF_TYPE_UINT64) {
            out->data.uint64_value = tmp;
        } else {
            out->data.int64_value = static_cast<int64_t>(tmp);
        }
        return {};
    }
    case MARMOT_GGUF_TYPE_FLOAT32: {
        float tmp = 0.0f;
        if (!cursor.read(tmp)) {
            return std::unexpected(ParseError::EndOfData);
        }
        out->data.float32_value = tmp;
        return {};
    }
    case MARMOT_GGUF_TYPE_FLOAT64: {
        double tmp = 0.0;
        if (!cursor.read(tmp)) {
            return std::unexpected(ParseError::EndOfData);
        }
        out->data.float64_value = tmp;
        return {};
    }
    case MARMOT_GGUF_TYPE_STRING: {
        return read_string(cursor, &out->data.string_value);
    }
    case MARMOT_GGUF_TYPE_ARRAY: {
        return parse_array(cursor, &out->data.array_value);
    }
    default:
        return std::unexpected(ParseError::InvalidType);
    }
}

GgmlTypeMapping map_ggml_type(uint32_t ggml_type) {
    switch (ggml_type) {
    case 0:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_GENERIC,
            .dtype = MARMOT_DTYPE_FLOAT32,
            .qscheme = MARMOT_QSCHEME_NONE,
            .supported = true
        };
    case 1:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_GENERIC,
            .dtype = MARMOT_DTYPE_FLOAT16,
            .qscheme = MARMOT_QSCHEME_NONE,
            .supported = true
        };
    case 30:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_GENERIC,
            .dtype = MARMOT_DTYPE_BFLOAT16,
            .qscheme = MARMOT_QSCHEME_NONE,
            .supported = true
        };
    case 24:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_GENERIC,
            .dtype = MARMOT_DTYPE_INT8,
            .qscheme = MARMOT_QSCHEME_NONE,
            .supported = true
        };
    case 25:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_GENERIC,
            .dtype = MARMOT_DTYPE_INT16,
            .qscheme = MARMOT_QSCHEME_NONE,
            .supported = true
        };
    case 26:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_GENERIC,
            .dtype = MARMOT_DTYPE_INT32,
            .qscheme = MARMOT_QSCHEME_NONE,
            .supported = true
        };
    case 27:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_GENERIC,
            .dtype = MARMOT_DTYPE_INT64,
            .qscheme = MARMOT_QSCHEME_NONE,
            .supported = true
        };
    case 2:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q4_0,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q4_0,
            .supported = true
        };
    case 3:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q4_1,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q4_1,
            .supported = true
        };
    case 6:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q5_0,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q5_0,
            .supported = true
        };
    case 7:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q5_1,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q5_1,
            .supported = true
        };
    case 8:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q8_0,
            .dtype = MARMOT_DTYPE_INT8,
            .qscheme = MARMOT_QSCHEME_Q8_0,
            .supported = true
        };
    case 9:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q8_1,
            .dtype = MARMOT_DTYPE_INT8,
            .qscheme = MARMOT_QSCHEME_Q8_1,
            .supported = true
        };
    case 10:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q2_K,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q2_K,
            .supported = true
        };
    case 11:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q3_K,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q3_K,
            .supported = true
        };
    case 12:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q4_K,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q4_K,
            .supported = true
        };
    case 13:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q5_K,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q5_K,
            .supported = true
        };
    case 14:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q6_K,
            .dtype = MARMOT_DTYPE_UINT8,
            .qscheme = MARMOT_QSCHEME_Q6_K,
            .supported = true
        };
    case 15:
        return GgmlTypeMapping{
            .quant_kind = MARMOT_QUANT_KIND_Q8_K,
            .dtype = MARMOT_DTYPE_INT8,
            .qscheme = MARMOT_QSCHEME_Q8_K,
            .supported = true
        };
    default:
        return GgmlTypeMapping{.supported = false};
    }
}

bool set_tensor_data_views(marmot_gguf_t *gguf, size_t metadata_end_offset) {
    if (gguf == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF handle is null");
        return false;
    }
    if (gguf->alignment == 0) {
        gguf->alignment = kDefaultAlignment;
    }
    size_t data_offset = compute_align(metadata_end_offset, gguf->alignment);
    if (data_offset > gguf->size) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF tensor data offset out of range");
        return false;
    }
    size_t data_size = gguf->size - data_offset;
    gguf->tensor_data_offset = data_offset;

    if (gguf->tensor_count == 0) {
        return true;
    }

    std::vector<std::pair<uint64_t, size_t>> offset_pairs(gguf->tensor_count);
    for (size_t i = 0; i < gguf->tensor_count; ++i) {
        offset_pairs[i] = {gguf->tensors[i].data_offset, i};
    }
    std::ranges::sort(offset_pairs, [](const auto &a, const auto &b) { return a.first < b.first; });

    for (size_t idx = 0; idx < offset_pairs.size(); ++idx) {
        const uint64_t current_offset = offset_pairs[idx].first;
        const uint64_t next_offset =
            idx + 1 < offset_pairs.size() ? offset_pairs[idx + 1].first : static_cast<uint64_t>(data_size);
        if (current_offset > data_size || next_offset > data_size || next_offset < current_offset) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF tensor span exceeds file bounds");
            return false;
        }
        const size_t length = static_cast<size_t>(next_offset - current_offset);
        const size_t tensor_index = offset_pairs[idx].second;
        marmot_gguf_tensor_t *tensor_info = &gguf->tensors[tensor_index];
        tensor_info->byte_length = length;

        marmot_tensor_t *tensor = tensor_info->tensor;
        if (tensor == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF tensor entry missing tensor object");
            return false;
        }

        const GgmlTypeMapping mapping = map_ggml_type(tensor_info->ggml_type);
        if (!mapping.supported) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported GGUF tensor type");
            return false;
        }

        if (tensor->shape.ndim == 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF tensor missing shape");
            return false;
        }
        tensor->shape.strides[tensor->shape.ndim - 1] = 1;
        for (ssize_t d = static_cast<ssize_t>(tensor->shape.ndim) - 2; d >= 0; --d) {
            tensor->shape.strides[d] = tensor->shape.strides[d + 1] * tensor->shape.shape[d + 1];
        }
        tensor->dtype = mapping.dtype;
        tensor->backend = MARMOT_BACKEND_CPU;
        tensor->ctx = gguf->ctx;
        tensor->owns_data = false;
        tensor->quant_params = nullptr;
        tensor->quant_kind = mapping.quant_kind;
        tensor->quant_layout =
            mapping.quant_kind == MARMOT_QUANT_KIND_GENERIC ? MARMOT_QUANT_LAYOUT_GENERIC : MARMOT_QUANT_LAYOUT_GGUF;
        tensor->memory_location = MARMOT_MEMORY_HOST;
        tensor->needs_sync = false;
        tensor->data = static_cast<uint8_t *>(gguf->data) + data_offset + current_offset;

        const size_t expected_bytes = marmot_tensor_size_bytes(tensor);
        if (expected_bytes == 0 || expected_bytes > length) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF tensor storage does not match expected size");
            return false;
        }
        tensor->capacity_bytes = expected_bytes;

        tensor_info->qscheme_id = mapping.qscheme;
        tensor_info->byte_length = length;
    }

    return true;
}

marmot_gguf_t *load_file(const char *path) {
    if (path == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF path is nullptr");
        return nullptr;
    }

    try {
        ScopedFd fd{open(path, O_RDONLY)};
        if (fd.value < 0) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to open GGUF file");
            return nullptr;
        }

        struct stat st{};
        if (fstat(fd.value, &st) != 0 || st.st_size <= 0 || st.st_size > SSIZE_MAX) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to stat GGUF file");
            return nullptr;
        }
        const size_t file_size = static_cast<size_t>(st.st_size);

        MappedRegion mapping{mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd.value, 0), file_size};
        if (!mapping.valid()) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to mmap GGUF file");
            return nullptr;
        }

        ByteCursor cursor{std::span<const uint8_t>(static_cast<const uint8_t *>(mapping.ptr), file_size), 0};

        uint32_t magic = 0;
        if (!cursor.read(magic) || magic != gguf::kMagic) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF magic mismatch");
            return nullptr;
        }

        uint32_t version = 0;
        if (!cursor.read(version) || version != gguf::kVersionSupported) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported GGUF version");
            return nullptr;
        }

        uint64_t tensor_count_u64 = 0;
        uint64_t kv_count_u64 = 0;
        if (!cursor.read(tensor_count_u64) || !cursor.read(kv_count_u64)) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to read GGUF counts");
            return nullptr;
        }
        if (tensor_count_u64 > SIZE_MAX || kv_count_u64 > SIZE_MAX) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF counts exceed platform limits");
            return nullptr;
        }

        auto result = std::unique_ptr<marmot_gguf_t>(new marmot_gguf_t());

        result->version = version;
        result->alignment = gguf::kDefaultAlignment;
        result->tensor_count = static_cast<size_t>(tensor_count_u64);
        result->kv_count = static_cast<size_t>(kv_count_u64);
        result->data = mapping.ptr;
        result->size = file_size;
        result->fd = -1;
        result->ctx = nullptr;

        auto kv_owner = std::unique_ptr<marmot_gguf_kv_t[], KvArrayDeleter>(
            result->kv_count > 0 ? new marmot_gguf_kv_t[result->kv_count]() : nullptr, KvArrayDeleter{result->kv_count}
        );

        auto tensor_owner = std::unique_ptr<marmot_gguf_tensor_t[], TensorArrayDeleter>(
            result->tensor_count > 0 ? new marmot_gguf_tensor_t[result->tensor_count]() : nullptr,
            TensorArrayDeleter{result->tensor_count}
        );

        if ((result->kv_count > 0 && kv_owner == nullptr) || (result->tensor_count > 0 && tensor_owner == nullptr)) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Unable to allocate GGUF tables");
            return nullptr;
        }
        result->kv = kv_owner.get();
        result->tensors = tensor_owner.get();

        for (size_t i = 0; i < result->kv_count; ++i) {
            uint64_t key_len = 0;
            if (!cursor.read(key_len) || key_len == 0 || key_len > SIZE_MAX) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid GGUF key length");
                return nullptr;
            }

            auto key_span = cursor.read_span(static_cast<size_t>(key_len));
            if (!key_span) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF key exceeds file bounds");
                return nullptr;
            }

            auto key = std::make_unique<char[]>(static_cast<size_t>(key_len) + 1);
            std::memcpy(key.get(), key_span->data(), static_cast<size_t>(key_len));
            key[key_len] = '\0';

            kv_owner.get()[i].key = key.release();
            auto parsed = gguf::parse_value(cursor, &kv_owner.get()[i].value);
            if (!parsed) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unable to parse GGUF value");
                return nullptr;
            }

            if (kv_owner.get()[i].key != nullptr && std::strcmp(kv_owner.get()[i].key, "general.alignment") == 0 &&
                kv_owner.get()[i].value.type == MARMOT_GGUF_TYPE_UINT32) {
                size_t align_value = static_cast<size_t>(kv_owner.get()[i].value.data.uint32_value);
                if (align_value != 0) {
                    result->alignment = align_value;
                }
            }
        }

        for (size_t i = 0; i < result->tensor_count; ++i) {
            uint64_t name_len = 0;
            if (!cursor.read(name_len) || name_len == 0 || name_len > SIZE_MAX) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid GGUF tensor name");
                return nullptr;
            }

            auto name_span = cursor.read_span(static_cast<size_t>(name_len));
            if (!name_span) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "GGUF tensor name exceeds file bounds");
                return nullptr;
            }

            auto name = std::make_unique<char[]>(static_cast<size_t>(name_len) + 1);
            std::memcpy(name.get(), name_span->data(), static_cast<size_t>(name_len));
            name[name_len] = '\0';

            uint32_t ndim = 0;
            if (!cursor.read(ndim) || ndim == 0 || ndim > MARMOT_MAX_DIMS) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid GGUF tensor rank");
                return nullptr;
            }
            uint64_t raw_shape[MARMOT_MAX_DIMS] = {0};
            for (uint32_t d = 0; d < ndim; ++d) {
                uint64_t dim = 0;
                if (!cursor.read(dim) || dim == 0 || dim > SIZE_MAX) {
                    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid GGUF tensor dimension");
                    return nullptr;
                }
                raw_shape[d] = dim;
            }

            uint32_t ggml_type = 0;
            if (!cursor.read(ggml_type)) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid GGUF tensor type");
                return nullptr;
            }

            uint64_t data_off = 0;
            if (!cursor.read(data_off)) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid GGUF tensor offset");
                return nullptr;
            }

            auto placeholder = std::unique_ptr<marmot_tensor_t, gguf::FreeDeleter>(
                static_cast<marmot_tensor_t *>(calloc(1, sizeof(marmot_tensor_t)))
            );
            if (placeholder == nullptr) {
                marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Unable to allocate tensor placeholder");
                return nullptr;
            }
            placeholder->shape.ndim = ndim;
            if (ndim == 2) {
                placeholder->shape.shape[0] = static_cast<size_t>(raw_shape[1]);
                placeholder->shape.shape[1] = static_cast<size_t>(raw_shape[0]);
            } else {
                for (uint32_t d = 0; d < ndim; ++d) {
                    placeholder->shape.shape[d] = static_cast<size_t>(raw_shape[d]);
                }
            }

            tensor_owner.get()[i].name = name.release();
            tensor_owner.get()[i].tensor = placeholder.release();
            tensor_owner.get()[i].ggml_type = ggml_type;
            tensor_owner.get()[i].data_offset = data_off;
        }

        if (!set_tensor_data_views(result.get(), cursor.offset)) {
            return nullptr;
        }

        result->data = mapping.release();
        result->fd = fd.release();
        result->kv = kv_owner.release();
        result->tensors = tensor_owner.release();
        return result.release();
    } catch (const std::bad_alloc &) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "GGUF loader out of memory");
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "GGUF loader threw unexpected exception");
    }
    return nullptr;
}

void unload_file(marmot_gguf_t *gguf) {
    if (gguf == nullptr) {
        return;
    }

    free_kv_array(gguf->kv, gguf->kv_count);
    free_tensor_array(gguf->tensors, gguf->tensor_count);

    if (gguf->data != nullptr && gguf->data != MAP_FAILED) {
        if (gguf->ctx != nullptr && gguf->ctx->ops != nullptr && gguf->ctx->ops->on_host_range_freed != nullptr) {
            gguf->ctx->ops->on_host_range_freed(gguf->ctx->device_ctx, gguf->data, gguf->size);
        }
        munmap(gguf->data, gguf->size);
        gguf->data = nullptr;
    }
    if (gguf->fd >= 0) {
        close(gguf->fd);
    }

    delete[] gguf->kv;
    delete[] gguf->tensors;
    delete gguf;
}

const marmot_gguf_kv_t *find_kv(const marmot_gguf_t *gguf, const char *key) {
    if (gguf == nullptr || key == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < gguf->kv_count; ++i) {
        if (gguf->kv[i].key != nullptr && std::strcmp(gguf->kv[i].key, key) == 0) {
            return &gguf->kv[i];
        }
    }
    return nullptr;
}

const marmot_gguf_tensor_t *find_tensor(const marmot_gguf_t *gguf, const char *name) {
    if (gguf == nullptr || name == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < gguf->tensor_count; ++i) {
        if (gguf->tensors[i].name != nullptr && std::strcmp(gguf->tensors[i].name, name) == 0) {
            return &gguf->tensors[i];
        }
    }
    return nullptr;
}

} // namespace marmot::gguf
