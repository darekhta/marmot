#include <stdint.h>

#include <string.h>

#include "tensor_internal.h"

static inline void fill_raw(void *dst, const void *value, size_t value_size, size_t count) {
    uint8_t *ptr = (uint8_t *)dst;
    for (size_t i = 0; i < count; ++i) {
        memcpy(ptr + i * value_size, value, value_size);
    }
}

void marmot_tensor_fill_f32(marmot_tensor_t *tensor, float value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_FLOAT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT32 tensor");
        return;
    }
    float *data = (float *)tensor->data;
    size_t n = marmot_tensor_num_elements(tensor);
    for (size_t i = 0; i < n; ++i) {
        data[i] = value;
    }
    mark_host_written(tensor);
}

void marmot_tensor_fill_f16(marmot_tensor_t *tensor, marmot_float16_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_FLOAT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT16 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_bf16(marmot_tensor_t *tensor, marmot_bfloat16_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_BFLOAT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected BFLOAT16 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_i32(marmot_tensor_t *tensor, marmot_int32_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_INT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT32 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_i16(marmot_tensor_t *tensor, marmot_int16_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_INT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT16 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_i8(marmot_tensor_t *tensor, marmot_int8_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_INT8)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected INT8 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_u8(marmot_tensor_t *tensor, marmot_uint8_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_UINT8)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT8 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_u16(marmot_tensor_t *tensor, marmot_uint16_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_UINT16)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT16 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_u32(marmot_tensor_t *tensor, marmot_uint32_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_UINT32)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT32 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_u64(marmot_tensor_t *tensor, marmot_uint64_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_UINT64)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected UINT64 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

#if MARMOT_ENABLE_FP8
void marmot_tensor_fill_fp8_e4m3(marmot_tensor_t *tensor, marmot_float8_e4m3_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_FLOAT8_E4M3)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT8_E4M3 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}

void marmot_tensor_fill_fp8_e5m2(marmot_tensor_t *tensor, marmot_float8_e5m2_t value) {
    if (unlikely(tensor == nullptr || tensor->dtype != MARMOT_DTYPE_FLOAT8_E5M2)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Expected FLOAT8_E5M2 tensor");
        return;
    }
    fill_raw(tensor->data, &value, sizeof(value), marmot_tensor_num_elements(tensor));
    mark_host_written(tensor);
}
#endif

// ===================================================================
// Full tensor helpers
// ===================================================================

marmot_tensor_t *marmot_tensor_full_f32(const marmot_context_t *ctx, const size_t *shape, size_t ndim, float value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_FLOAT32);
    if (tensor != nullptr) {
        marmot_tensor_fill_f32(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_f16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_float16_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_FLOAT16);
    if (tensor != nullptr) {
        marmot_tensor_fill_f16(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_bf16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_bfloat16_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_BFLOAT16);
    if (tensor != nullptr) {
        marmot_tensor_fill_bf16(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_i32(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int32_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_INT32);
    if (tensor != nullptr) {
        marmot_tensor_fill_i32(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_i16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int16_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_INT16);
    if (tensor != nullptr) {
        marmot_tensor_fill_i16(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_i8(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_int8_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_INT8);
    if (tensor != nullptr) {
        marmot_tensor_fill_i8(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_u8(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint8_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_UINT8);
    if (tensor != nullptr) {
        marmot_tensor_fill_u8(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_u16(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint16_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_UINT16);
    if (tensor != nullptr) {
        marmot_tensor_fill_u16(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_u32(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint32_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_UINT32);
    if (tensor != nullptr) {
        marmot_tensor_fill_u32(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_u64(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_uint64_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_UINT64);
    if (tensor != nullptr) {
        marmot_tensor_fill_u64(tensor, value);
    }
    return tensor;
}

#if MARMOT_ENABLE_FP8
marmot_tensor_t *
marmot_tensor_full_fp8_e4m3(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_float8_e4m3_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_FLOAT8_E4M3);
    if (tensor != nullptr) {
        marmot_tensor_fill_fp8_e4m3(tensor, value);
    }
    return tensor;
}

marmot_tensor_t *
marmot_tensor_full_fp8_e5m2(const marmot_context_t *ctx, const size_t *shape, size_t ndim, marmot_float8_e5m2_t value) {
    marmot_tensor_t *tensor = marmot_tensor_create(ctx, shape, ndim, MARMOT_DTYPE_FLOAT8_E5M2);
    if (tensor != nullptr) {
        marmot_tensor_fill_fp8_e5m2(tensor, value);
    }
    return tensor;
}
#endif
