#include <stdlib.h>

#include <string.h>

#include "internal/metal_matmul_qkv_shared.h"
#include "metal_packed_weight.h"

#ifdef __APPLE__

@interface MarmotMetalQKVPackedWeights : NSObject
- (instancetype)initWithPackedWeights:(metal_matmul_qkv_packed_weights_t *)packed
                          weightBytes:(size_t)weightBytes
                            biasBytes:(size_t)biasBytes;
- (void)exportPackedWeights:(metal_matmul_qkv_packed_weights_t *)outPacked;
- (size_t)totalBytes;
@end

@implementation MarmotMetalQKVPackedWeights {
    metal_matmul_qkv_packed_weights_t _stored;
    size_t _weightBytes;
    size_t _biasBytes;
}

- (instancetype)initWithPackedWeights:(metal_matmul_qkv_packed_weights_t *)packed
                          weightBytes:(size_t)weightBytes
                            biasBytes:(size_t)biasBytes {
    self = [super init];
    if (self == nil) {
        return nil;
    }
    memset(&_stored, 0, sizeof(_stored));
    if (packed != nullptr) {
        _stored = *packed;
        _stored.cache_entry = nil;
        _weightBytes = weightBytes;
        _biasBytes = biasBytes;
        packed->weight_storage = nullptr;
        packed->bias_storage = nullptr;
        packed->owns_weight = false;
        packed->owns_bias = false;
    }
    return self;
}

- (void)dealloc {
    metal_matmul_qkv_release_packed_weights(&_stored);
    [super dealloc];
}

- (void)exportPackedWeights:(metal_matmul_qkv_packed_weights_t *)outPacked {
    if (outPacked == nullptr) {
        return;
    }
    metal_matmul_qkv_packed_weights_t copy = _stored;
    copy.cache_entry = nil;
    copy.owns_weight = false;
    copy.owns_bias = false;
    *outPacked = copy;
}

- (size_t)totalBytes {
    size_t sum = _weightBytes;
    if (SIZE_MAX - sum < _biasBytes) {
        return SIZE_MAX;
    }
    return sum + _biasBytes;
}

@end

typedef struct {
    const void *wq;
    const void *wk;
    const void *wv;
    const void *bq;
    const void *bk;
    const void *bv;
    uint32_t dtype;
    uint32_t m;
    uint32_t k;
} metal_matmul_qkv_cache_key_t;

static NSValue *
metal_matmul_qkv_cache_make_key(const marmot_matmul_qkv_desc_t *src, const metal_matmul_qkv_dims_t *dims) {
    if (src == nullptr || dims == nullptr) {
        return nil;
    }
    metal_matmul_qkv_cache_key_t key = {
        .wq = src->separate.wq != nullptr ? src->separate.wq->data : nullptr,
        .wk = src->separate.wk != nullptr ? src->separate.wk->data : nullptr,
        .wv = src->separate.wv != nullptr ? src->separate.wv->data : nullptr,
        .bq = src->separate.bq != nullptr ? src->separate.bq->data : nullptr,
        .bk = src->separate.bk != nullptr ? src->separate.bk->data : nullptr,
        .bv = src->separate.bv != nullptr ? src->separate.bv->data : nullptr,
        .dtype = (uint32_t)src->input->dtype,
        .m = (uint32_t)dims->M,
        .k = (uint32_t)dims->K,
    };
    return [NSValue valueWithBytes:&key objCType:@encode(metal_matmul_qkv_cache_key_t)];
}

static void metal_matmul_qkv_cache_touch_locked(metal_context_t *ctx, NSValue *key) {
    if (ctx == nullptr || key == nil || ctx->qkv_fused_cache_lru_keys == nil) {
        return;
    }
    NSUInteger idx = [ctx->qkv_fused_cache_lru_keys indexOfObject:key];
    if (idx != NSNotFound) {
        [ctx->qkv_fused_cache_lru_keys removeObjectAtIndex:idx];
    }
    [ctx->qkv_fused_cache_lru_keys addObject:key];
}

static void metal_matmul_qkv_cache_enforce_limit_locked(metal_context_t *ctx) {
    if (ctx == nullptr || ctx->qkv_fused_cache == nil || ctx->qkv_fused_cache_lru_keys == nil) {
        return;
    }
    while (ctx->qkv_fused_cache_limit_bytes > 0 && ctx->qkv_fused_cache_bytes > ctx->qkv_fused_cache_limit_bytes &&
           ctx->qkv_fused_cache_lru_keys.count > 0) {
        NSValue *victim = [ctx->qkv_fused_cache_lru_keys firstObject];
        [ctx->qkv_fused_cache_lru_keys removeObjectAtIndex:0];
        MarmotMetalQKVPackedWeights *entry = ctx->qkv_fused_cache[victim];
        if (entry != nil) {
            size_t bytes = [entry totalBytes];
            if (ctx->qkv_fused_cache_bytes > bytes) {
                ctx->qkv_fused_cache_bytes -= bytes;
            } else {
                ctx->qkv_fused_cache_bytes = 0;
            }
            [ctx->qkv_fused_cache removeObjectForKey:victim];
        }
    }
}

static MarmotMetalQKVPackedWeights *metal_matmul_qkv_cache_lookup(metal_context_t *ctx, NSValue *key) {
    if (ctx == nullptr || key == nil || ctx->qkv_fused_cache == nil || ctx->qkv_fused_cache_limit_bytes == 0) {
        return nil;
    }
    pthread_mutex_lock(&ctx->qkv_fused_cache_mutex);
    MarmotMetalQKVPackedWeights *entry = ctx->qkv_fused_cache[key];
    if (entry != nil) {
        [entry retain];
        metal_matmul_qkv_cache_touch_locked(ctx, key);
    }
    pthread_mutex_unlock(&ctx->qkv_fused_cache_mutex);
    return entry;
}

static void metal_matmul_qkv_cache_store(metal_context_t *ctx, NSValue *key, MarmotMetalQKVPackedWeights *entry) {
    if (ctx == nullptr || key == nil || entry == nil || ctx->qkv_fused_cache == nil ||
        ctx->qkv_fused_cache_limit_bytes == 0) {
        return;
    }
    pthread_mutex_lock(&ctx->qkv_fused_cache_mutex);
    ctx->qkv_fused_cache[key] = entry;
    metal_matmul_qkv_cache_touch_locked(ctx, key);
    size_t bytes = [entry totalBytes];
    if (SIZE_MAX - ctx->qkv_fused_cache_bytes < bytes) {
        ctx->qkv_fused_cache_bytes = SIZE_MAX;
    } else {
        ctx->qkv_fused_cache_bytes += bytes;
    }
    metal_matmul_qkv_cache_enforce_limit_locked(ctx);
    pthread_mutex_unlock(&ctx->qkv_fused_cache_mutex);
}

void metal_matmul_qkv_release_packed_weights(metal_matmul_qkv_packed_weights_t *packed) {
    if (packed == nullptr) {
        return;
    }
    MarmotMetalQKVPackedWeights *entry = (MarmotMetalQKVPackedWeights *)packed->cache_entry;
    if (entry != nil) {
        [entry release];
        packed->cache_entry = nullptr;
    }
    if (packed->owns_weight && packed->weight_storage != nullptr) {
        free(packed->weight_storage);
    }
    if (packed->owns_bias && packed->bias_storage != nullptr) {
        free(packed->bias_storage);
    }
    memset(packed, 0, sizeof(*packed));
}

marmot_error_t metal_matmul_qkv_prepare_fused_desc(
    metal_context_t *ctx, const marmot_matmul_qkv_desc_t *src, const metal_matmul_qkv_dims_t *dims,
    marmot_matmul_qkv_desc_t *dst, metal_matmul_qkv_packed_weights_t *packed
) {
    if (ctx == nullptr || src == nullptr || dims == nullptr || dst == nullptr || packed == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    memset(packed, 0, sizeof(*packed));
    const bool cache_enabled = ctx->qkv_fused_cache_limit_bytes > 0;
    NSValue *cache_key = nil;
    if (cache_enabled) {
        cache_key = metal_matmul_qkv_cache_make_key(src, dims);
        if (cache_key != nil) {
            MarmotMetalQKVPackedWeights *cached = metal_matmul_qkv_cache_lookup(ctx, cache_key);
            if (cached != nil) {
                [cached exportPackedWeights:packed];
                packed->cache_entry = [cached retain];
                [cached release];
                *dst = *src;
                dst->layout = MARMOT_QKV_LAYOUT_FUSED;
                dst->fused.weight = &packed->weight_tensor;
                dst->fused.bias = packed->bias_tensor.data != nullptr ? &packed->bias_tensor : nullptr;
                return MARMOT_SUCCESS;
            }
        }
    }

    const size_t dtype_size = marmot_dtype_size(src->input->dtype);
    const size_t tile_bytes = 3 * dims->M * dims->K * dtype_size;
    void *weight_storage = malloc(tile_bytes);
    if (weight_storage == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    packed->weight_storage = weight_storage;
    packed->owns_weight = true;
    marmot_tensor_t weight_tensor = {};
    weight_tensor.data = weight_storage;
    weight_tensor.dtype = src->input->dtype;
    weight_tensor.quant_kind = MARMOT_QUANT_KIND_GENERIC;
    weight_tensor.quant_layout = MARMOT_QUANT_LAYOUT_GENERIC;
    weight_tensor.shape.ndim = 2;
    weight_tensor.shape.shape[0] = 3 * dims->M;
    weight_tensor.shape.shape[1] = dims->K;
    weight_tensor.shape.strides[0] = dims->K;
    weight_tensor.shape.strides[1] = 1;
    packed->weight_tensor = weight_tensor;

    const marmot_tensor_t *weights[3] = {src->separate.wq, src->separate.wk, src->separate.wv};
    for (size_t slice = 0; slice < 3; ++slice) {
        const marmot_tensor_t *w = weights[slice];
        const char *src_bytes = (const char *)w->data;
        char *dst_bytes = (char *)weight_storage + slice * dims->M * dims->K * dtype_size;
        for (size_t row = 0; row < dims->M; ++row) {
            memcpy(
                dst_bytes + row * dims->K * dtype_size, src_bytes + row * w->shape.strides[0] * dtype_size,
                dims->K * dtype_size
            );
        }
    }

    const marmot_tensor_t *biases[3] = {src->separate.bq, src->separate.bk, src->separate.bv};
    bool has_bias = (biases[0] != nullptr) || (biases[1] != nullptr) || (biases[2] != nullptr);
    if (has_bias) {
        const size_t bias_elems = dims->M;
        const size_t bias_bytes = 3 * bias_elems * dtype_size;
        void *bias_storage = malloc(bias_bytes);
        if (bias_storage == nullptr) {
            metal_matmul_qkv_release_packed_weights(packed);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        for (size_t slice = 0; slice < 3; ++slice) {
            const marmot_tensor_t *bias_tensor = biases[slice];
            char *dst_ptr = (char *)bias_storage + slice * dims->M * dtype_size;
            if (bias_tensor != nullptr) {
                memcpy(dst_ptr, bias_tensor->data, dims->M * dtype_size);
            } else {
                memset(dst_ptr, 0, dims->M * dtype_size);
            }
        }
        packed->bias_storage = bias_storage;
        packed->owns_bias = true;
        marmot_tensor_t bias_tensor = {};
        bias_tensor.data = bias_storage;
        bias_tensor.dtype = src->input->dtype;
        bias_tensor.quant_kind = MARMOT_QUANT_KIND_GENERIC;
        bias_tensor.quant_layout = MARMOT_QUANT_LAYOUT_GENERIC;
        bias_tensor.shape.ndim = 1;
        bias_tensor.shape.shape[0] = 3 * dims->M;
        bias_tensor.shape.strides[0] = 1;
        packed->bias_tensor = bias_tensor;
    }

    if (cache_enabled && cache_key != nil) {
        size_t bias_bytes = packed->owns_bias ? (3 * dims->M * dtype_size) : 0;
        MarmotMetalQKVPackedWeights *entry = [[MarmotMetalQKVPackedWeights alloc] initWithPackedWeights:packed
                                                                                            weightBytes:tile_bytes
                                                                                              biasBytes:bias_bytes];
        if (entry != nil) {
            metal_matmul_qkv_cache_store(ctx, cache_key, entry);
            [entry exportPackedWeights:packed];
            packed->cache_entry = [entry retain];
            [entry release];
        }
    }

    *dst = *src;
    dst->layout = MARMOT_QKV_LAYOUT_FUSED;
    dst->fused.weight = &packed->weight_tensor;
    dst->fused.bias = packed->bias_tensor.data != nullptr ? &packed->bias_tensor : nullptr;
    return MARMOT_SUCCESS;
}

#endif // __APPLE__
