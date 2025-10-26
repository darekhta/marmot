#include "metal_packed_weight.h"

#ifdef __APPLE__

#include <limits.h>

#include "metal_backend_internal.h"

@implementation MarmotMetalPackedWeightRecord
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        _packedBuffer = nil;
        _packedByteLength = 0;
        _sourceRows = 0;
        _sourceCols = 0;
        _sourceDtype = MARMOT_DTYPE_COUNT;
        _sourceVersion = 0;
        _layout = METAL_PACKED_LAYOUT_SOA;
        metal_packed_layout_config_t defaultConfig;
        defaultConfig.tile_cols = 0;
        defaultConfig.tile_k = 0;
        defaultConfig.tile_stride = 0;
        defaultConfig.tile_section = 0;
        defaultConfig.tiles_per_row = 0;
        defaultConfig.tiles_per_col = 0;
        defaultConfig.use_vec4 = false;
        defaultConfig.element_size = 0;
        _config = defaultConfig;
    }
    return self;
}

- (void)dealloc {
    if (_packedBuffer != nil) {
        [_packedBuffer release];
        _packedBuffer = nil;
    }
    [super dealloc];
}
@end

static NSValue *metal_key_for_pointer(const void *ptr) {
    void *value = (void *)ptr;
    return [[NSValue alloc] initWithBytes:&value objCType:@encode(void *)];
}

typedef struct {
    uint32_t rows;
    uint32_t cols;
    uint32_t segment_rows;
    uint32_t tile_cols;
    uint32_t tile_k;
    uint32_t tiles_per_row;
    uint32_t tiles_per_col;
    uint32_t tile_stride;
    uint32_t tile_section;
    uint32_t use_vec4;
} metal_pack_qkv_uniforms_t;

static void metal_packed_weight_select_tiles(
    metal_context_t *ctx, marmot_dtype_t dtype, size_t segment_rows, size_t *out_tile_cols, size_t *out_tile_k
) {
    size_t cols = 32;
    size_t tile_k = 8;
    if (ctx != nullptr && ctx->packed_weight_tiles_overridden) {
        if (ctx->packed_weight_tile_cols != 0) {
            cols = ctx->packed_weight_tile_cols;
        }
        if (ctx->packed_weight_tile_k != 0) {
            tile_k = ctx->packed_weight_tile_k;
        }
    } else if (dtype == MARMOT_DTYPE_FLOAT32) {
        if (segment_rows >= 4096) {
            cols = 24;
            tile_k = 8;
        }
    }
    if (cols == 0) {
        cols = 32;
    }
    if (tile_k == 0) {
        tile_k = 8;
    }
    if (out_tile_cols != nullptr) {
        *out_tile_cols = cols;
    }
    if (out_tile_k != nullptr) {
        *out_tile_k = tile_k;
    }
}

static metal_packed_layout_config_t metal_packed_weight_default_layout(
    metal_context_t *ctx, const marmot_tensor_t *weight, size_t segment_rows, size_t cols, size_t element_size,
    bool enable_vec4
) {
    size_t default_tile_cols = 32;
    size_t default_tile_k = 8;
    metal_packed_weight_select_tiles(
        ctx, weight != nullptr ? weight->dtype : MARMOT_DTYPE_FLOAT32, segment_rows, &default_tile_cols, &default_tile_k
    );
    metal_packed_layout_config_t config = {};
    config.tile_cols = default_tile_cols;
    config.tile_k = default_tile_k;
    config.use_vec4 = enable_vec4;
    config.element_size = element_size;
    size_t tile_section = config.tile_cols * config.tile_k;
    config.tile_section = tile_section;
    config.tile_stride = tile_section * 3;
    config.tiles_per_row = (segment_rows + config.tile_cols - 1) / config.tile_cols;
    config.tiles_per_col = (cols + config.tile_k - 1) / config.tile_k;
    if (!enable_vec4 || (config.tile_k % 4 != 0) || (cols % 4 != 0)) {
        config.use_vec4 = false;
    }
    return config;
}

static size_t metal_packed_weight_buffer_size(const metal_packed_layout_config_t *config) {
    if (config == nullptr) {
        return 0;
    }
    size_t tiles = config->tiles_per_row * config->tiles_per_col;
    return tiles * config->tile_stride * config->element_size;
}

static size_t metal_packed_weight_dtype_min_dim(size_t base, marmot_dtype_t dtype) {
    size_t dim = base == 0 ? 1 : base;
    if (dtype == MARMOT_DTYPE_FLOAT16) {
        dim = dim > 4 ? (dim * 3) / 4 : dim;
    } else if (dtype == MARMOT_DTYPE_BFLOAT16) {
        dim = dim > 8 ? (dim * 7) / 8 : dim;
    }
    if (dim == 0) {
        dim = 1;
    }
    return dim;
}

static size_t metal_packed_weight_dtype_min_total(size_t base, marmot_dtype_t dtype) {
    if (base == 0) {
        return 0;
    }
    if (dtype == MARMOT_DTYPE_FLOAT16) {
        return base / 2;
    }
    if (dtype == MARMOT_DTYPE_BFLOAT16) {
        return (base * 3) / 4;
    }
    return base;
}

static void metal_packed_weight_touch_locked(metal_context_t *ctx, NSValue *key) {
    if (ctx == nullptr || ctx->packed_weight_lru_keys == nil || key == nil) {
        return;
    }
    NSUInteger idx = [ctx->packed_weight_lru_keys indexOfObject:key];
    if (idx != NSNotFound) {
        [ctx->packed_weight_lru_keys removeObjectAtIndex:idx];
    }
    [ctx->packed_weight_lru_keys addObject:key];
}

static void metal_packed_weight_remove_entry_locked(metal_context_t *ctx, NSValue *key) {
    if (ctx == nullptr || ctx->packed_weight_cache == nil || key == nil) {
        return;
    }
    MarmotMetalPackedWeightRecord *record = ctx->packed_weight_cache[key];
    if (record == nil) {
        return;
    }
    size_t length = record.packedByteLength;
    if (ctx->packed_weight_cache_bytes >= length) {
        ctx->packed_weight_cache_bytes -= length;
    } else {
        ctx->packed_weight_cache_bytes = 0;
    }
    [ctx->packed_weight_cache removeObjectForKey:key];
    if (ctx->packed_weight_lru_keys != nil) {
        NSUInteger idx = [ctx->packed_weight_lru_keys indexOfObject:key];
        if (idx != NSNotFound) {
            [ctx->packed_weight_lru_keys removeObjectAtIndex:idx];
        }
    }
}

static void metal_packed_weight_enforce_limit_locked(metal_context_t *ctx, NSValue *protected_key) {
    if (ctx == nullptr || ctx->packed_weight_cache_limit_bytes == 0) {
        return;
    }
    while (ctx->packed_weight_cache_bytes > ctx->packed_weight_cache_limit_bytes &&
           ctx->packed_weight_lru_keys != nil && ctx->packed_weight_lru_keys.count > 0) {
        NSValue *oldest = [ctx->packed_weight_lru_keys objectAtIndex:0];
        if (protected_key != nil && [oldest isEqual:protected_key]) {
            if (ctx->packed_weight_lru_keys.count == 1) {
                break;
            }
            [ctx->packed_weight_lru_keys removeObjectAtIndex:0];
            [ctx->packed_weight_lru_keys addObject:oldest];
            continue;
        }
        metal_packed_weight_remove_entry_locked(ctx, oldest);
    }
}

static bool
metal_packed_weight_should_pack(metal_context_t *ctx, const marmot_tensor_t *weight, size_t segment_rows, size_t cols) {
    if (ctx == nullptr || weight == nullptr || weight->data == nullptr) {
        return false;
    }
    if (!ctx->enable_packed_weights) {
        return false;
    }
    if (!(weight->dtype == MARMOT_DTYPE_FLOAT32 || weight->dtype == MARMOT_DTYPE_FLOAT16 ||
          weight->dtype == MARMOT_DTYPE_BFLOAT16)) {
        return false;
    }
    size_t min_dim = metal_packed_weight_dtype_min_dim(ctx->packed_weight_min_dim, weight->dtype);
    if (segment_rows < min_dim || cols < min_dim) {
        return false;
    }
    size_t min_total = metal_packed_weight_dtype_min_total(ctx->packed_weight_min_elements, weight->dtype);
    if (min_total > 0) {
        size_t total;
        if (cols != 0 && segment_rows > SIZE_MAX / cols) {
            total = SIZE_MAX;
        } else {
            total = segment_rows * cols;
        }
        if (total < min_total) {
            return false;
        }
    }
    return true;
}

static marmot_error_t metal_pack_qkv_weight_gpu(
    metal_context_t *ctx, const marmot_tensor_t *weight, id<MTLBuffer> packed_buffer,
    const metal_packed_layout_config_t *config
) {
    if (ctx == nullptr || weight == nullptr || packed_buffer == nil || config == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    id<MTLBuffer> bufferWeight = metal_buffer_acquire(ctx, weight->data, marmot_tensor_size_bytes(weight));
    if (bufferWeight == nil) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const char *kernel_name = nullptr;
    switch (weight->dtype) {
    case MARMOT_DTYPE_FLOAT32:
        kernel_name = "pack_qkv_weight_f32";
        break;
    case MARMOT_DTYPE_FLOAT16:
        kernel_name = "pack_qkv_weight_f16";
        break;
    case MARMOT_DTYPE_BFLOAT16:
        kernel_name = "pack_qkv_weight_bf16";
        break;
    default:
        kernel_name = nullptr;
        break;
    }
    if (kernel_name == nullptr) {
        [bufferWeight release];
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferWeight release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    size_t segment_rows = weight->shape.shape[0] / 3;
    metal_pack_qkv_uniforms_t uniforms = {
        .rows = (uint32_t)weight->shape.shape[0],
        .cols = (uint32_t)weight->shape.shape[1],
        .segment_rows = (uint32_t)segment_rows,
        .tile_cols = (uint32_t)config->tile_cols,
        .tile_k = (uint32_t)config->tile_k,
        .tiles_per_row = (uint32_t)config->tiles_per_row,
        .tiles_per_col = (uint32_t)config->tiles_per_col,
        .tile_stride = (uint32_t)config->tile_stride,
        .tile_section = (uint32_t)config->tile_section,
        .use_vec4 = config->use_vec4 ? 1u : 0u,
    };

    [encoder setBuffer:bufferWeight offset:0 atIndex:0];
    [encoder setBuffer:packed_buffer offset:0 atIndex:1];
    [encoder setBytes:&uniforms length:sizeof(uniforms) atIndex:2];

    MTLSize threads = MTLSizeMake(config->tile_k, config->tile_cols, 1);
    MTLSize threadgroups =
        MTLSizeMake(config->tiles_per_row, config->tiles_per_col == 0 ? 1 : config->tiles_per_col, 1);

    if (threadgroups.width == 0) {
        threadgroups.width = 1;
    }
    if (threadgroups.height == 0) {
        threadgroups.height = 1;
    }

    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads];
    metal_command_stream_flush(ctx, false);
    [pipeline release];
    [bufferWeight release];
    return MARMOT_SUCCESS;
}

static MarmotMetalPackedWeightRecord *
metal_packed_weight_lookup_locked(metal_context_t *ctx, const void *ptr, bool remove_on_mismatch) {
    if (ctx == nullptr || ctx->packed_weight_cache == nil || ptr == nullptr) {
        return nil;
    }
    NSValue *key = metal_key_for_pointer(ptr);
    MarmotMetalPackedWeightRecord *record = ctx->packed_weight_cache[key];
    if (record == nil) {
        [key release];
        return nil;
    }
    if (remove_on_mismatch) {
        metal_packed_weight_remove_entry_locked(ctx, key);
    } else {
        metal_packed_weight_touch_locked(ctx, key);
    }
    [key release];
    return record;
}

static void metal_packed_weight_store(metal_context_t *ctx, const void *ptr, MarmotMetalPackedWeightRecord *record) {
    if (ctx == nullptr || ctx->packed_weight_cache == nil || ptr == nullptr || record == nil) {
        return;
    }
    NSValue *key = metal_key_for_pointer(ptr);
    ctx->packed_weight_cache[key] = record;
    size_t length = record.packedByteLength;
    if (SIZE_MAX - ctx->packed_weight_cache_bytes < length) {
        ctx->packed_weight_cache_bytes = SIZE_MAX;
    } else {
        ctx->packed_weight_cache_bytes += length;
    }
    metal_packed_weight_touch_locked(ctx, key);
    metal_packed_weight_enforce_limit_locked(ctx, key);
    [key release];
}

marmot_error_t metal_packed_weight_acquire(
    metal_context_t *ctx, const marmot_tensor_t *weight, MarmotMetalPackedWeightRecord **out_record
) {
    if (out_record == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    *out_record = nil;
    if (ctx == nullptr || weight == nullptr || weight->data == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const size_t total_rows = weight->shape.shape[0];
    if (total_rows % 3 != 0) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    const size_t segment_rows = total_rows / 3;
    const size_t cols = weight->shape.shape[1];
    if (!metal_packed_weight_should_pack(ctx, weight, segment_rows, cols)) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    pthread_mutex_lock(&ctx->packed_weight_mutex);
    MarmotMetalPackedWeightRecord *existing = metal_packed_weight_lookup_locked(ctx, weight->data, false);
    if (existing != nil) {
        bool matches = existing.sourceRows == weight->shape.shape[0] && existing.sourceCols == cols &&
            existing.sourceDtype == weight->dtype;
        if (!matches) {
            NSValue *key = metal_key_for_pointer(weight->data);
            [ctx->packed_weight_cache removeObjectForKey:key];
            [key release];
            existing = nil;
        }
    }
    pthread_mutex_unlock(&ctx->packed_weight_mutex);
    if (existing != nil) {
        *out_record = [existing retain];
        return MARMOT_SUCCESS;
    }

    MarmotMetalPackedWeightRecord *record = [[MarmotMetalPackedWeightRecord alloc] init];
    record.sourceRows = weight->shape.shape[0];
    record.sourceCols = cols;
    record.sourceDtype = weight->dtype;
    record.layout = METAL_PACKED_LAYOUT_TILED_SOA;
    size_t element_size = marmot_dtype_size(weight->dtype);
    bool enable_vec4 =
        (weight->dtype == MARMOT_DTYPE_FLOAT32 || weight->dtype == MARMOT_DTYPE_FLOAT16 ||
         weight->dtype == MARMOT_DTYPE_BFLOAT16);
    metal_packed_layout_config_t config =
        metal_packed_weight_default_layout(ctx, weight, segment_rows, cols, element_size, enable_vec4);

    size_t buffer_size = metal_packed_weight_buffer_size(&config);
    if (buffer_size == 0) {
        [record release];
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    id<MTLBuffer> packed = [ctx->device newBufferWithLength:buffer_size options:MTLResourceStorageModePrivate];
    if (packed == nil) {
        [record release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    record.packedBuffer = packed;
    record.config = config;
    [packed release];
    record.packedByteLength = buffer_size;

    marmot_error_t status = metal_pack_qkv_weight_gpu(ctx, weight, record.packedBuffer, &config);
    if (status != MARMOT_SUCCESS) {
        [record release];
        return status;
    }

    pthread_mutex_lock(&ctx->packed_weight_mutex);
    metal_packed_weight_store(ctx, weight->data, record);
    pthread_mutex_unlock(&ctx->packed_weight_mutex);

    *out_record = record;
    return MARMOT_SUCCESS;
}

void metal_packed_weight_invalidate(metal_context_t *ctx, const void *ptr) {
    if (ctx == nullptr || ctx->packed_weight_cache == nil || ptr == nullptr) {
        return;
    }
    pthread_mutex_lock(&ctx->packed_weight_mutex);
    NSValue *key = metal_key_for_pointer(ptr);
    metal_packed_weight_remove_entry_locked(ctx, key);
    pthread_mutex_unlock(&ctx->packed_weight_mutex);
    [key release];
}

void metal_packed_weight_clear(metal_context_t *ctx) {
    if (ctx == nullptr || ctx->packed_weight_cache == nil) {
        return;
    }
    pthread_mutex_lock(&ctx->packed_weight_mutex);
    [ctx->packed_weight_cache removeAllObjects];
    ctx->packed_weight_cache_bytes = 0;
    if (ctx->packed_weight_lru_keys != nil) {
        [ctx->packed_weight_lru_keys removeAllObjects];
    }
    pthread_mutex_unlock(&ctx->packed_weight_mutex);
}

#endif // __APPLE__
