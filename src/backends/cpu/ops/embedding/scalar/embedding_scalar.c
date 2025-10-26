#include "marmot/dispatch.h"
#include "marmot/quant_block.h"

#include <stdlib.h>

#include <errno.h>

#include "core/helpers/quant.h"
#include "cpu_backend_internal.h"
#include "ops/convert/convert_registry.h"
#include "ops/embedding/embedding_internal.h"

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

static bool cpu_embedding_is_block_quant_weight(const marmot_tensor_t *tensor) {
    if (tensor == nullptr) {
        return false;
    }
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(tensor->quant_kind);
    if (traits == nullptr) {
        return false;
    }
    if (!traits->is_block_quantized) {
        return false;
    }
    if (!marmot_quant_storage_dtype_compatible(traits, tensor->dtype)) {
        return false;
    }
    if (tensor->quant_layout != traits->layout) {
        return false;
    }
    if (tensor->shape.ndim != 2) {
        return false;
    }
    return true;
}

typedef struct {
    const void *device_ctx;
    const marmot_tensor_t *weights;
    const marmot_tensor_t *token_ids;
    marmot_tensor_t *out;
    const void *weights_data;
    const uint8_t *weights_bytes;
    const void *token_data;
    void *out_data;
    size_t token_stride;
    size_t weight_row_stride;
    size_t weight_inner_stride;
    size_t out_row_stride;
    size_t out_inner_stride;
    size_t vocab;
    size_t dim;
    marmot_dtype_t out_dtype;
    float scale;
    cpu_convert_fn convert_fn;
    bool is_quantized;
    const marmot_quant_traits_t *quant_traits;
    size_t block_size;
    size_t blocks_per_row;
    size_t block_bytes;
    size_t row_stride_bytes;
    int32_t padding_id;
    bool bounds_check;
    size_t prefetch_distance;
    size_t start;
    size_t end;
    marmot_error_t status;
    size_t error_token_index;
    int64_t error_token_value;
} cpu_embedding_worker_args_t;

typedef bool (*cpu_embedding_token_loader_t)(
    const cpu_embedding_worker_args_t *args, size_t logical_index, int64_t *out_value
);

static bool cpu_embedding_load_token_i32(const cpu_embedding_worker_args_t *args, size_t idx, int64_t *out_value) {
    const size_t offset = idx * args->token_stride;
    *out_value = ((const int32_t *)args->token_data)[offset];
    return true;
}

static bool cpu_embedding_load_token_u32(const cpu_embedding_worker_args_t *args, size_t idx, int64_t *out_value) {
    const size_t offset = idx * args->token_stride;
    *out_value = (int64_t)((const uint32_t *)args->token_data)[offset];
    return true;
}

static bool cpu_embedding_load_token_i16(const cpu_embedding_worker_args_t *args, size_t idx, int64_t *out_value) {
    const size_t offset = idx * args->token_stride;
    *out_value = ((const int16_t *)args->token_data)[offset];
    return true;
}

static bool cpu_embedding_load_token_u16(const cpu_embedding_worker_args_t *args, size_t idx, int64_t *out_value) {
    const size_t offset = idx * args->token_stride;
    *out_value = (int64_t)((const uint16_t *)args->token_data)[offset];
    return true;
}

static bool cpu_embedding_load_token_i64(const cpu_embedding_worker_args_t *args, size_t idx, int64_t *out_value) {
    const size_t offset = idx * args->token_stride;
    *out_value = ((const marmot_int64_t *)args->token_data)[offset].value;
    return true;
}

static bool cpu_embedding_load_token_u64(const cpu_embedding_worker_args_t *args, size_t idx, int64_t *out_value) {
    const size_t offset = idx * args->token_stride;
    *out_value = (int64_t)((const marmot_uint64_t *)args->token_data)[offset].value;
    return true;
}

static inline bool
cpu_embedding_load_token_id(const cpu_embedding_worker_args_t *args, size_t logical_index, int64_t *out_value) {
    static const cpu_embedding_token_loader_t k_token_loaders[MARMOT_DTYPE_COUNT] = {
        [MARMOT_DTYPE_INT16] = cpu_embedding_load_token_i16, [MARMOT_DTYPE_UINT16] = cpu_embedding_load_token_u16,
        [MARMOT_DTYPE_INT32] = cpu_embedding_load_token_i32, [MARMOT_DTYPE_UINT32] = cpu_embedding_load_token_u32,
        [MARMOT_DTYPE_INT64] = cpu_embedding_load_token_i64, [MARMOT_DTYPE_UINT64] = cpu_embedding_load_token_u64,
    };

    marmot_dtype_t dtype = args->token_ids->dtype;
    if (dtype >= MARMOT_DTYPE_COUNT) {
        return false;
    }
    cpu_embedding_token_loader_t loader = k_token_loaders[dtype];
    if (loader == nullptr) {
        return false;
    }
    return loader(args, logical_index, out_value);
}

static inline void cpu_embedding_store_value(const cpu_embedding_worker_args_t *args, size_t out_idx, double value) {
    const double scaled = value * (double)args->scale;
    if (args->out_dtype == MARMOT_DTYPE_FLOAT64) {
        cpu_store_from_f64(args->out_dtype, args->out_data, out_idx, scaled);
    } else {
        cpu_store_from_f32(args->out_dtype, args->out_data, out_idx, (float)scaled);
    }
}

static void cpu_embedding_zero_row(const cpu_embedding_worker_args_t *args, size_t row_index) {
    const size_t dim = args->dim;
    const size_t row_stride = args->out_row_stride;
    const size_t inner_stride = args->out_inner_stride;
    uint8_t *out_bytes = (uint8_t *)args->out_data;
    const size_t dtype_bytes = marmot_dtype_size(args->out_dtype);

    if (inner_stride == 1) {
        uint8_t *row_ptr = out_bytes + row_index * row_stride * dtype_bytes;
        memset(row_ptr, 0, dim * dtype_bytes);
        return;
    }

    const size_t base = row_index * row_stride;
    for (size_t d = 0; d < dim; ++d) {
        const size_t out_idx = base + d * inner_stride;
        cpu_embedding_store_value(args, out_idx, 0.0);
    }
}

static inline void cpu_embedding_store_contiguous_f32(
    const cpu_embedding_worker_args_t *args, size_t out_offset, const float *values, size_t count
) {
    float *out_ptr = (float *)args->out_data + out_offset;
    const float scale = args->scale;
    if (scale == 1.0f) {
        memcpy(out_ptr, values, count * sizeof(float));
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        out_ptr[i] = values[i] * scale;
    }
}

static inline void cpu_embedding_store_contiguous_f16(
    const cpu_embedding_worker_args_t *args, size_t out_offset, const float *values, size_t count
) {
    marmot_float16_t *out_ptr = (marmot_float16_t *)args->out_data + out_offset;
    const float scale = args->scale;
    for (size_t i = 0; i < count; ++i) {
        out_ptr[i] = marmot_native_to_float16((_Float16)(values[i] * scale));
    }
}

static inline void cpu_embedding_store_contiguous_bf16(
    const cpu_embedding_worker_args_t *args, size_t out_offset, const float *values, size_t count
) {
    marmot_bfloat16_t *out_ptr = (marmot_bfloat16_t *)args->out_data + out_offset;
    const float scale = args->scale;
    for (size_t i = 0; i < count; ++i) {
        out_ptr[i] = marmot_native_to_bfloat16(values[i] * scale);
    }
}

static inline void cpu_embedding_decode_q4_0_interleaved(const marmot_q4_0_block_t *block, size_t count, float *dst) {
    const float scale = (float)marmot_float16_to_native(block->scale);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t packed = block->qs[i >> 1];
        int32_t q = (i & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
        q -= 8;
        dst[i] = scale * (float)q;
    }
}

static inline void cpu_embedding_decode_q4_1_interleaved(const marmot_q4_1_block_t *block, size_t count, float *dst) {
    const float scale = (float)marmot_float16_to_native(block->scale);
    const float min = (float)marmot_float16_to_native(block->min);
    for (size_t i = 0; i < count; ++i) {
        const uint8_t packed = block->qs[i >> 1];
        const uint8_t q = (uint8_t)((i & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f));
        dst[i] = scale * (float)q + min;
    }
}

static marmot_error_t cpu_embedding_copy_row_quantized(
    const cpu_embedding_worker_args_t *args, const uint8_t *row_ptr, size_t out_row_index
) {
    float decoded[MARMOT_QK_K_VALUES];
    const size_t dim = args->dim;
    const size_t block_size = args->block_size;
    const size_t out_row_offset = out_row_index * args->out_row_stride;
    const marmot_quant_traits_t *traits = args->quant_traits;

    if (traits == nullptr || traits->dequantize_block == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const marmot_quant_kind_t kind = traits->kind;
    const bool interleaved_q4 = (kind == MARMOT_QUANT_KIND_Q4_0) || (kind == MARMOT_QUANT_KIND_Q4_1);

    for (size_t block_idx = 0; block_idx < args->blocks_per_row; ++block_idx) {
        const size_t base_col = block_idx * block_size;
        if (base_col >= dim) {
            break;
        }

        const size_t remaining = dim - base_col;
        const size_t count = remaining < block_size ? remaining : block_size;
        const uint8_t *block_ptr = row_ptr + block_idx * args->block_bytes;

        if (interleaved_q4) {
            if (kind == MARMOT_QUANT_KIND_Q4_0) {
                cpu_embedding_decode_q4_0_interleaved((const marmot_q4_0_block_t *)block_ptr, count, decoded);
            } else {
                cpu_embedding_decode_q4_1_interleaved((const marmot_q4_1_block_t *)block_ptr, count, decoded);
            }
        } else {
            marmot_error_t err = traits->dequantize_block(block_ptr, (uint32_t)count, decoded, nullptr);
            if (err != MARMOT_SUCCESS) {
                return err;
            }
        }

        if (args->out_inner_stride == 1) {
            const size_t out_offset = out_row_offset + base_col;
            switch (args->out_dtype) {
            case MARMOT_DTYPE_FLOAT32:
                cpu_embedding_store_contiguous_f32(args, out_offset, decoded, count);
                continue;
            case MARMOT_DTYPE_FLOAT16:
                cpu_embedding_store_contiguous_f16(args, out_offset, decoded, count);
                continue;
            case MARMOT_DTYPE_BFLOAT16:
                cpu_embedding_store_contiguous_bf16(args, out_offset, decoded, count);
                continue;
            default:
                break;
            }
        }

        for (size_t j = 0; j < count; ++j) {
            const size_t out_idx = out_row_offset + (base_col + j) * args->out_inner_stride;
            cpu_embedding_store_value(args, out_idx, (double)decoded[j]);
        }
    }

    return MARMOT_SUCCESS;
}

static void
cpu_embedding_copy_row_floating(const cpu_embedding_worker_args_t *args, size_t token_row, size_t out_row_index) {
    const size_t dim = args->dim;
    const size_t weight_row_offset = token_row * args->weight_row_stride;
    const size_t out_row_offset = out_row_index * args->out_row_stride;
    const size_t inner_stride = args->out_inner_stride;
    const marmot_dtype_t weight_dtype = args->weights->dtype;
    const size_t dtype_bytes = marmot_dtype_size(weight_dtype);
    const size_t out_dtype_bytes = marmot_dtype_size(args->out_dtype);
    const bool needs_double = (weight_dtype == MARMOT_DTYPE_FLOAT64) || (args->out_dtype == MARMOT_DTYPE_FLOAT64);

    if (weight_dtype == args->out_dtype && args->weight_inner_stride == 1 && inner_stride == 1 && args->scale == 1.0f) {
        const uint8_t *src = (const uint8_t *)args->weights_data + weight_row_offset * dtype_bytes;
        uint8_t *dst = (uint8_t *)args->out_data + out_row_offset * dtype_bytes;
        memcpy(dst, src, dim * dtype_bytes);
        return;
    }

    if (args->convert_fn != nullptr && args->weight_inner_stride == 1 && inner_stride == 1 && args->scale == 1.0f) {
        const uint8_t *src = (const uint8_t *)args->weights_data + weight_row_offset * dtype_bytes;
        uint8_t *dst = (uint8_t *)args->out_data + out_row_offset * out_dtype_bytes;
        args->convert_fn(args->device_ctx, dst, src, dim);
        return;
    }

    if (inner_stride == 1) {
        const float scale = args->scale;
        const size_t weight_stride = args->weight_inner_stride;
        size_t w_idx = weight_row_offset;

        switch (args->out_dtype) {
        case MARMOT_DTYPE_FLOAT32: {
            float *out_ptr = (float *)args->out_data + out_row_offset;
            for (size_t d = 0; d < dim; ++d) {
                const float value = cpu_load_as_f32(weight_dtype, args->weights_data, w_idx);
                out_ptr[d] = value * scale;
                w_idx += weight_stride;
            }
            return;
        }
        case MARMOT_DTYPE_FLOAT16: {
            marmot_float16_t *out_ptr = (marmot_float16_t *)args->out_data + out_row_offset;
            for (size_t d = 0; d < dim; ++d) {
                const float value = cpu_load_as_f32(weight_dtype, args->weights_data, w_idx);
                out_ptr[d] = marmot_native_to_float16((_Float16)(value * scale));
                w_idx += weight_stride;
            }
            return;
        }
        case MARMOT_DTYPE_BFLOAT16: {
            marmot_bfloat16_t *out_ptr = (marmot_bfloat16_t *)args->out_data + out_row_offset;
            for (size_t d = 0; d < dim; ++d) {
                const float value = cpu_load_as_f32(weight_dtype, args->weights_data, w_idx);
                out_ptr[d] = marmot_native_to_bfloat16(value * scale);
                w_idx += weight_stride;
            }
            return;
        }
        default:
            break;
        }
    }

    for (size_t d = 0; d < dim; ++d) {
        const size_t w_idx = weight_row_offset + d * args->weight_inner_stride;
        const size_t out_idx = out_row_offset + d * inner_stride;
        if (needs_double) {
            const double value = cpu_load_as_f64(weight_dtype, args->weights_data, w_idx);
            cpu_embedding_store_value(args, out_idx, value);
        } else {
            const float value = cpu_load_as_f32(weight_dtype, args->weights_data, w_idx);
            cpu_embedding_store_value(args, out_idx, (double)value);
        }
    }
}

static void cpu_embedding_run_worker(cpu_embedding_worker_args_t *args) {
    args->status = MARMOT_SUCCESS;
    const size_t weight_dtype_bytes = marmot_dtype_size(args->weights->dtype);
    for (size_t idx = args->start; idx < args->end; ++idx) {
        if (args->prefetch_distance > 0) {
            const size_t prefetch_idx = idx + args->prefetch_distance;
            if (prefetch_idx < args->end) {
                int64_t prefetch_token = 0;
                if (cpu_embedding_load_token_id(args, prefetch_idx, &prefetch_token)) {
                    if (prefetch_token >= 0 && (size_t)prefetch_token < args->vocab &&
                        !(args->padding_id >= 0 && prefetch_token == args->padding_id)) {
                        if (args->is_quantized) {
                            const uint8_t *prefetch_ptr =
                                args->weights_bytes + (size_t)prefetch_token * args->row_stride_bytes;
                            MARMOT_PREFETCH(prefetch_ptr);
                        } else {
                            const uint8_t *prefetch_ptr = (const uint8_t *)args->weights_data +
                                (size_t)prefetch_token * args->weight_row_stride * weight_dtype_bytes;
                            MARMOT_PREFETCH(prefetch_ptr);
                        }
                    }
                }
            }
        }

        int64_t token_value = 0;
        if (!cpu_embedding_load_token_id(args, idx, &token_value)) {
            args->status = MARMOT_ERROR_UNSUPPORTED_DTYPE;
            args->error_token_index = idx;
            args->error_token_value = token_value;
            break;
        }

        if (args->padding_id >= 0 && token_value == args->padding_id) {
            cpu_embedding_zero_row(args, idx);
            continue;
        }

        if (token_value < 0 || (size_t)token_value >= args->vocab) {
            if (args->bounds_check) {
                args->status = MARMOT_ERROR_INVALID_ARGUMENT;
                args->error_token_index = idx;
                args->error_token_value = token_value;
                break;
            }
            cpu_embedding_zero_row(args, idx);
            continue;
        }

        const size_t token_row = (size_t)token_value;
        if (args->is_quantized) {
            const uint8_t *row_ptr = args->weights_bytes + token_row * args->row_stride_bytes;
            marmot_error_t err = cpu_embedding_copy_row_quantized(args, row_ptr, idx);
            if (err != MARMOT_SUCCESS) {
                args->status = err;
                args->error_token_index = idx;
                args->error_token_value = token_value;
                break;
            }
        } else {
            cpu_embedding_copy_row_floating(args, token_row, idx);
        }
    }
}

typedef struct {
    const cpu_embedding_worker_args_t *template_args;
} cpu_embedding_dispatch_ctx_t;

static marmot_error_t cpu_embedding_dispatch_range(void *ctx, size_t start, size_t end) {
    const cpu_embedding_dispatch_ctx_t *c = (const cpu_embedding_dispatch_ctx_t *)ctx;
    cpu_embedding_worker_args_t args = *c->template_args;
    args.start = start;
    args.end = end;
    args.status = MARMOT_SUCCESS;
    cpu_embedding_run_worker(&args);
    return args.status;
}

marmot_error_t cpu_embedding_gather_scalar(const void *device_ctx, const marmot_embedding_gather_desc_t *desc) {
    (void)device_ctx;
    const marmot_tensor_t *weights = desc->weights;
    const marmot_tensor_t *token_ids_original = desc->token_ids;
    marmot_tensor_t *out_original = desc->out;

    const size_t dim = weights->shape.shape[1];
    const size_t vocab = weights->shape.shape[0];
    const size_t token_count = marmot_tensor_num_elements(token_ids_original);

    marmot_tensor_t token_view;
    const marmot_tensor_t *token_ids = token_ids_original;
    if (token_ids_original->shape.ndim != 1) {
        memset(&token_view, 0, sizeof(token_view));
        memcpy(&token_view, token_ids_original, sizeof(token_view));
        token_view.shape.ndim = 1;
        token_view.shape.shape[0] = token_count;
        token_view.shape.strides[0] = 1;
        for (size_t i = 1; i < MARMOT_MAX_DIMS; ++i) {
            token_view.shape.shape[i] = 0;
            token_view.shape.strides[i] = 0;
        }
        token_ids = &token_view;
    }

    marmot_tensor_t out_view;
    marmot_tensor_t *out_tensor = out_original;
    if (out_original->shape.ndim != 2 || out_original->shape.shape[1] != dim) {
        memset(&out_view, 0, sizeof(out_view));
        memcpy(&out_view, out_original, sizeof(out_view));
        out_view.shape.ndim = 2;
        out_view.shape.shape[0] = token_count;
        out_view.shape.shape[1] = dim;
        out_view.shape.strides[1] = 1;
        out_view.shape.strides[0] = dim;
        for (size_t i = 2; i < MARMOT_MAX_DIMS; ++i) {
            out_view.shape.shape[i] = 0;
            out_view.shape.strides[i] = 0;
        }
        out_tensor = &out_view;
    }

    const bool is_quantized = cpu_embedding_is_block_quant_weight(weights);
    const marmot_quant_traits_t *quant_traits = nullptr;
    size_t block_size = 0;
    size_t block_bytes = 0;
    size_t row_stride_bytes = 0;
    size_t blocks_per_row = 0;
    cpu_convert_fn convert_fn = nullptr;

    if (is_quantized) {
        const marmot_quant_kind_traits_t *kind_traits = marmot_get_quant_kind_traits(weights->quant_kind);
        quant_traits = marmot_get_quant_traits(weights->quant_kind);
        if (kind_traits == nullptr || !kind_traits->is_block_quantized ||
            kind_traits->layout != MARMOT_QUANT_LAYOUT_GGUF) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Unsupported quantized embedding layout");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }
        if (quant_traits == nullptr || quant_traits->dequantize_block == nullptr || quant_traits->block_size == 0 ||
            quant_traits->block_bytes == 0 || quant_traits->layout != kind_traits->layout) {
            marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Quantized embedding kind not supported on CPU");
            return MARMOT_ERROR_NOT_IMPLEMENTED;
        }

        block_size = (size_t)quant_traits->block_size;
        block_bytes = (size_t)quant_traits->block_bytes;
        if (block_size > MARMOT_QK_K_VALUES) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quantized embedding block size unsupported");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }

        blocks_per_row = (dim + block_size - 1) / block_size;
        row_stride_bytes = blocks_per_row * block_bytes;
    } else if (desc->scale == 1.0f) {
        const cpu_context_t *cpu_ctx = (const cpu_context_t *)device_ctx;
        if (cpu_ctx != nullptr) {
            convert_fn = cpu_convert_resolve_fn(cpu_ctx, out_tensor->dtype, weights->dtype, nullptr);
        }
    }

    cpu_embedding_worker_args_t template_args = {
        .device_ctx = device_ctx,
        .weights = weights,
        .token_ids = token_ids,
        .out = out_tensor,
        .weights_data = weights->data,
        .weights_bytes = (const uint8_t *)weights->data,
        .token_data = token_ids->data,
        .out_data = out_tensor->data,
        .token_stride = token_ids->shape.strides[0],
        .weight_row_stride = weights->shape.strides[0],
        .weight_inner_stride = weights->shape.strides[1],
        .out_row_stride = out_tensor->shape.strides[0],
        .out_inner_stride = out_tensor->shape.strides[1],
        .vocab = vocab,
        .dim = dim,
        .out_dtype = out_tensor->dtype,
        .scale = desc->scale,
        .convert_fn = convert_fn,
        .is_quantized = is_quantized,
        .quant_traits = quant_traits,
        .block_size = block_size,
        .blocks_per_row = blocks_per_row,
        .block_bytes = block_bytes,
        .row_stride_bytes = row_stride_bytes,
        .padding_id = desc->padding_id,
        .bounds_check = desc->bounds_check,
        .prefetch_distance = cpu_embedding_prefetch_distance(),
        .start = 0,
        .end = token_count,
        .status = MARMOT_SUCCESS,
        .error_token_index = 0,
        .error_token_value = 0,
    };

    cpu_embedding_dispatch_ctx_t dctx = {
        .template_args = &template_args,
    };
    marmot_error_t status = marmot_dispatch_parallel_for_range_with_error(
        MARMOT_DISPATCH_PRIORITY_HIGH, token_count, 64, &dctx, cpu_embedding_dispatch_range
    );
    if (status != MARMOT_SUCCESS) {
        if (status == MARMOT_ERROR_INVALID_ARGUMENT) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Embedding id out of range");
        } else if (status == MARMOT_ERROR_UNSUPPORTED_DTYPE) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Unsupported token dtype for CPU embedding gather");
        } else {
            marmot_set_error(status, "CPU embedding gather failed");
        }
    }
    return status;
}
