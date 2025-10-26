#include "ops/embedding/embedding_internal.h"

#if HAS_AVX2

#include <stdlib.h>

#include <errno.h>
#include <string.h>

static size_t cpu_embedding_prefetch_distance(void) {
    constexpr size_t k_max_prefetch = 16;
    static size_t cached = (size_t)-1;
    if (cached != (size_t)-1) {
        return cached;
    }

    const char *env = getenv("MARMOT_EMBEDDING_PREFETCH_DISTANCE");
    if (env == nullptr || env[0] == '\0') {
        cached = 0;
        return cached;
    }

    errno = 0;
    char *end = nullptr;
    unsigned long value = strtoul(env, &end, 10);
    if (errno != 0 || end == env) {
        cached = 0;
        return cached;
    }

    if (value > k_max_prefetch) {
        value = k_max_prefetch;
    }
    cached = (size_t)value;
    return cached;
}

static bool cpu_embedding_token_dtype_supported(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_INT16:
    case MARMOT_DTYPE_UINT16:
    case MARMOT_DTYPE_INT32:
    case MARMOT_DTYPE_UINT32:
    case MARMOT_DTYPE_INT64:
    case MARMOT_DTYPE_UINT64:
        return true;
    default:
        return false;
    }
}

static bool cpu_embedding_load_token_id(const marmot_tensor_t *token_ids, size_t index, int64_t *out_value) {
    const size_t offset = index * token_ids->shape.strides[0];
    switch (token_ids->dtype) {
    case MARMOT_DTYPE_INT16:
        *out_value = ((const int16_t *)token_ids->data)[offset];
        return true;
    case MARMOT_DTYPE_UINT16:
        *out_value = (int64_t)((const uint16_t *)token_ids->data)[offset];
        return true;
    case MARMOT_DTYPE_INT32:
        *out_value = ((const int32_t *)token_ids->data)[offset];
        return true;
    case MARMOT_DTYPE_UINT32:
        *out_value = (int64_t)((const uint32_t *)token_ids->data)[offset];
        return true;
    case MARMOT_DTYPE_INT64:
        *out_value = ((const marmot_int64_t *)token_ids->data)[offset].value;
        return true;
    case MARMOT_DTYPE_UINT64:
        *out_value = (int64_t)((const marmot_uint64_t *)token_ids->data)[offset].value;
        return true;
    default:
        return false;
    }
}

static inline void cpu_embedding_copy_f32_avx2(const float *src, float *dst, size_t dim, float scale) {
    size_t i = 0;
    const __m256 scale_vec = _mm256_set1_ps(scale);
    for (; i + 8 <= dim; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        v = _mm256_mul_ps(v, scale_vec);
        _mm256_storeu_ps(dst + i, v);
    }
    for (; i < dim; ++i) {
        dst[i] = src[i] * scale;
    }
}

static inline void cpu_embedding_copy_u16_avx2(const uint16_t *src, uint16_t *dst, size_t dim) {
    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(src + i));
        _mm256_storeu_si256((__m256i *)(dst + i), v);
    }
    for (; i < dim; ++i) {
        dst[i] = src[i];
    }
}

marmot_error_t cpu_embedding_gather_avx2(const void *device_ctx, const marmot_embedding_gather_desc_t *desc) {
    if (desc == nullptr || desc->weights == nullptr || desc->token_ids == nullptr || desc->out == nullptr) {
        return cpu_embedding_gather_scalar(device_ctx, desc);
    }

    const marmot_tensor_t *weights = desc->weights;
    const marmot_tensor_t *token_ids = desc->token_ids;
    marmot_tensor_t *out = desc->out;

    if (weights->quant_kind != MARMOT_QUANT_KIND_GENERIC) {
        return cpu_embedding_gather_scalar(device_ctx, desc);
    }

    const bool f32_path = weights->dtype == MARMOT_DTYPE_FLOAT32 && out->dtype == MARMOT_DTYPE_FLOAT32;
    const bool f16_path =
        weights->dtype == MARMOT_DTYPE_FLOAT16 && out->dtype == MARMOT_DTYPE_FLOAT16 && desc->scale == 1.0f;
    const bool bf16_path =
        weights->dtype == MARMOT_DTYPE_BFLOAT16 && out->dtype == MARMOT_DTYPE_BFLOAT16 && desc->scale == 1.0f;
    if (!f32_path && !f16_path && !bf16_path) {
        return cpu_embedding_gather_scalar(device_ctx, desc);
    }

    if (weights->shape.ndim != 2 || token_ids->shape.ndim != 1 || out->shape.ndim != 2) {
        return cpu_embedding_gather_scalar(device_ctx, desc);
    }

    const size_t vocab = weights->shape.shape[0];
    const size_t dim = weights->shape.shape[1];
    const size_t token_count = token_ids->shape.shape[0];

    if (out->shape.shape[0] != token_count || out->shape.shape[1] != dim) {
        return cpu_embedding_gather_scalar(device_ctx, desc);
    }

    if (weights->shape.strides[1] != 1 || out->shape.strides[1] != 1) {
        return cpu_embedding_gather_scalar(device_ctx, desc);
    }

    if (!cpu_embedding_token_dtype_supported(token_ids->dtype)) {
        return cpu_embedding_gather_scalar(device_ctx, desc);
    }

    const size_t weight_row_stride = weights->shape.strides[0];
    const size_t out_row_stride = out->shape.strides[0];
    const float scale = desc->scale;
    const int32_t padding_id = desc->padding_id;
    const bool bounds_check = desc->bounds_check;
    const size_t prefetch_distance = cpu_embedding_prefetch_distance();
    const size_t dtype_bytes = f32_path ? sizeof(float) : sizeof(uint16_t);
    const uint8_t *weights_bytes = (const uint8_t *)weights->data;
    uint8_t *out_bytes = (uint8_t *)out->data;

    for (size_t idx = 0; idx < token_count; ++idx) {
        if (prefetch_distance > 0) {
            const size_t prefetch_idx = idx + prefetch_distance;
            if (prefetch_idx < token_count) {
                int64_t prefetch_token = 0;
                if (cpu_embedding_load_token_id(token_ids, prefetch_idx, &prefetch_token)) {
                    if (prefetch_token >= 0 && (size_t)prefetch_token < vocab &&
                        !(padding_id >= 0 && prefetch_token == padding_id)) {
                        const uint8_t *prefetch_ptr =
                            weights_bytes + (size_t)prefetch_token * weight_row_stride * dtype_bytes;
                        MARMOT_PREFETCH(prefetch_ptr);
                    }
                }
            }
        }

        int64_t token_value = 0;
        if (!cpu_embedding_load_token_id(token_ids, idx, &token_value)) {
            return cpu_embedding_gather_scalar(device_ctx, desc);
        }

        uint8_t *out_row_bytes = out_bytes + idx * out_row_stride * dtype_bytes;
        if (padding_id >= 0 && token_value == padding_id) {
            memset(out_row_bytes, 0, dim * dtype_bytes);
            continue;
        }

        if (token_value < 0 || (size_t)token_value >= vocab) {
            if (bounds_check) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding id out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            memset(out_row_bytes, 0, dim * dtype_bytes);
            continue;
        }

        const uint8_t *src_row_bytes = weights_bytes + (size_t)token_value * weight_row_stride * dtype_bytes;
        if (f32_path) {
            cpu_embedding_copy_f32_avx2((const float *)src_row_bytes, (float *)out_row_bytes, dim, scale);
        } else {
            cpu_embedding_copy_u16_avx2((const uint16_t *)src_row_bytes, (uint16_t *)out_row_bytes, dim);
        }
    }

    return MARMOT_SUCCESS;
}

#endif
