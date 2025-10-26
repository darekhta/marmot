#include "core/tensor/tensor_debug.h"
#include "internal/stride_helpers.h"
#include "metal_backend_internal.h"
#include "metal_packed_weight.h"
#include "utils/dtype_ref.h"

#ifdef __APPLE__

#include <stdlib.h>

#include <algorithm>
#include <atomic>
#include <execinfo.h>

// Track residency operations
static std::atomic<size_t> g_private_buffer_releases{0};
static std::atomic<size_t> g_shared_buffer_creates{0};
static std::atomic<size_t> g_shared_buffer_bytes{0};
#import <os/log.h>
#include <string.h>

// Result of targeted residency sync for a specific memory range.
typedef enum metal_residency_sync_result {
    METAL_RESIDENCY_SYNC_RESULT_CLEAN = 0,    // Range found, no sync needed
    METAL_RESIDENCY_SYNC_RESULT_SYNCED = 1,   // Range found and synced
    METAL_RESIDENCY_SYNC_RESULT_UNTRACKED = 2 // Range not in residency map
} metal_residency_sync_result_t;

extern const marmot_allocator_ops_t metal_allocator_ops;

@interface MarmotMetalResidencyRecord : NSObject
@property(nonatomic, retain) id<MTLBuffer> privateBuffer;
@property(nonatomic, assign) size_t byteLength;
@property(nonatomic, assign) BOOL sharedDirty;
@property(nonatomic, assign) BOOL privateDirty;
@property(nonatomic, assign) marmot_dtype_t storageDtype;
@property(nonatomic, assign) marmot_dtype_t computeDtype;
@property(nonatomic, assign) size_t storageByteLength;
@property(nonatomic, assign) size_t elementCount;
@end

@implementation MarmotMetalResidencyRecord
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        _privateBuffer = nil;
        _byteLength = 0;
        _sharedDirty = YES;
        _privateDirty = NO;
        _storageDtype = MARMOT_DTYPE_COUNT;
        _computeDtype = MARMOT_DTYPE_COUNT;
        _storageByteLength = 0;
        _elementCount = 0;
    }
    return self;
}

- (void)dealloc {
    if (_privateBuffer != nil) {
        [_privateBuffer release];
        _privateBuffer = nil;
    }
    [super dealloc];
}
@end

@interface MarmotMetalBiasCacheEntry : NSObject
@property(nonatomic, assign) metal_context_t *ctx;
@property(nonatomic, assign) marmot_allocation_t allocation;
@property(nonatomic, assign) size_t elements;
@end

static NSValue *metal_key_for_pointer(const void *ptr);
static size_t metal_allocator_pool_bucket_for(size_t size);
static void metal_residency_mark_shared_dirty(metal_context_t *ctx, const void *ptr);
static metal_residency_sync_result_t metal_residency_sync_range(metal_context_t *ctx, const void *ptr, size_t bytes);

@implementation MarmotMetalBiasCacheEntry
- (void)dealloc {
    if (self.ctx != nullptr && self.allocation.ptr != nullptr) {
        metal_allocator_ops.free(self.ctx, &_allocation);
        _allocation.ptr = nullptr;
    }
    [super dealloc];
}
@end

@interface MarmotMetalAllocatorDeferredGroup : NSObject
@property(nonatomic, assign) metal_context_t *ctx;
@property(nonatomic, retain) id<MTLCommandBuffer> commandBuffer;
@property(nonatomic, assign) metal_pool_entry_t *head;
@property(nonatomic, assign) metal_pool_entry_t *tail;
@end

@implementation MarmotMetalAllocatorDeferredGroup
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        _ctx = nullptr;
        _commandBuffer = nil;
        _head = nullptr;
        _tail = nullptr;
    }
    return self;
}

- (void)dealloc {
    if (_commandBuffer != nil) {
        [_commandBuffer release];
        _commandBuffer = nil;
    }
    [super dealloc];
}
@end

@interface MarmotMetalMemoryState : NSObject {
  @public
    pthread_mutex_t mutex;
    NSMutableArray<MarmotMetalAllocatorDeferredGroup *> *groups;
    MarmotMetalAllocatorDeferredGroup *active_group;
    MarmotMetalAllocatorDeferredGroup *barrier_group;
}
@end

@implementation MarmotMetalMemoryState
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        pthread_mutex_init(&mutex, nullptr);
        groups = [[NSMutableArray alloc] init];
        active_group = nil;
        barrier_group = nil;
    }
    return self;
}

- (void)dealloc {
    if (groups != nil) {
        [groups removeAllObjects];
        [groups release];
        groups = nil;
    }
    pthread_mutex_destroy(&mutex);
    [super dealloc];
}
@end

static pthread_mutex_t metal_memory_states_mutex = PTHREAD_MUTEX_INITIALIZER;
static NSMutableDictionary<NSValue *, MarmotMetalMemoryState *> *metal_memory_states = nil;

static MarmotMetalMemoryState *metal_memory_state_acquire(metal_context_t *ctx, bool create) {
    if (ctx == nullptr) {
        return nil;
    }

    pthread_mutex_lock(&metal_memory_states_mutex);
    if (metal_memory_states == nil && create) {
        metal_memory_states = [[NSMutableDictionary alloc] init];
    }
    if (metal_memory_states == nil) {
        pthread_mutex_unlock(&metal_memory_states_mutex);
        return nil;
    }

    NSValue *key = metal_key_for_pointer(ctx);
    MarmotMetalMemoryState *state = metal_memory_states[key];
    if (state == nil && create) {
        state = [[MarmotMetalMemoryState alloc] init];
        metal_memory_states[key] = state;
        [state release];
    }
    [key release];

    pthread_mutex_unlock(&metal_memory_states_mutex);
    return state;
}

static void metal_allocator_pool_recycle_entries(metal_context_t *ctx, metal_pool_entry_t *entry) {
    if (ctx == nullptr) {
        return;
    }

    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    while (entry != nullptr) {
        metal_pool_entry_t *next = entry->next;
        size_t bucket = metal_allocator_pool_bucket_for(entry->size);
        if (bucket >= METAL_ALLOCATOR_POOL_BUCKET_COUNT) {
            if (ctx->allocator_stats.pooled_bytes >= entry->size) {
                ctx->allocator_stats.pooled_bytes -= entry->size;
            } else {
                ctx->allocator_stats.pooled_bytes = 0;
            }
            if (entry->buffer != nil) {
                [entry->buffer release];
            }
            free(entry);
            entry = next;
            continue;
        }

        entry->next = ctx->allocator_stats.pool[bucket].head;
        ctx->allocator_stats.pool[bucket].head = entry;
        ctx->allocator_stats.pool[bucket].count++;
        entry = next;
    }
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
}

static void metal_allocator_pool_complete_group(MarmotMetalAllocatorDeferredGroup *group) {
    if (group == nil) {
        return;
    }

    metal_context_t *ctx = group.ctx;
    metal_pool_entry_t *entries = nullptr;

    MarmotMetalMemoryState *state = metal_memory_state_acquire(ctx, false);
    if (state != nil) {
        pthread_mutex_lock(&state->mutex);
        NSUInteger idx = [state->groups indexOfObjectIdenticalTo:group];
        if (idx != NSNotFound) {
            [state->groups removeObjectAtIndex:idx];
        }
        if (state->active_group == group) {
            state->active_group = nil;
        }
        if (state->barrier_group == group) {
            state->barrier_group = nil;
        }
        entries = group.head;
        group.head = nullptr;
        group.tail = nullptr;
        pthread_mutex_unlock(&state->mutex);
    } else {
        entries = group.head;
        group.head = nullptr;
        group.tail = nullptr;
    }

    if (entries != nullptr) {
        metal_allocator_pool_recycle_entries(ctx, entries);
    }
}

static NSValue *metal_key_for_pointer(const void *ptr) {
    void *value = (void *)ptr;
    return [[NSValue alloc] initWithBytes:&value objCType:@encode(void *)];
}

static const size_t metal_allocator_pool_bucket_sizes[METAL_ALLOCATOR_POOL_BUCKET_COUNT] = {
    4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608
};

static const char *metal_residency_convert_kernel_name(marmot_dtype_t src_dtype, marmot_dtype_t dst_dtype) {
    if (src_dtype == dst_dtype) {
        return nullptr;
    }
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_FLOAT16)
        return "convert_f32_to_f16";
    if (src_dtype == MARMOT_DTYPE_FLOAT16 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return "convert_f16_to_f32";
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_BFLOAT16)
        return "convert_f32_to_bf16";
    if (src_dtype == MARMOT_DTYPE_BFLOAT16 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return "convert_bf16_to_f32";
    if (src_dtype == MARMOT_DTYPE_FLOAT16 && dst_dtype == MARMOT_DTYPE_BFLOAT16)
        return "convert_f16_to_bf16";
    if (src_dtype == MARMOT_DTYPE_BFLOAT16 && dst_dtype == MARMOT_DTYPE_FLOAT16)
        return "convert_bf16_to_f16";
#if MARMOT_ENABLE_FP8
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_FLOAT8_E4M3)
        return "convert_f32_to_fp8_e4m3";
    if (src_dtype == MARMOT_DTYPE_FLOAT8_E4M3 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return "convert_fp8_e4m3_to_f32";
    if (src_dtype == MARMOT_DTYPE_FLOAT32 && dst_dtype == MARMOT_DTYPE_FLOAT8_E5M2)
        return "convert_f32_to_fp8_e5m2";
    if (src_dtype == MARMOT_DTYPE_FLOAT8_E5M2 && dst_dtype == MARMOT_DTYPE_FLOAT32)
        return "convert_fp8_e5m2_to_f32";
#endif
    return nullptr;
}

static marmot_error_t metal_residency_convert_buffer(
    metal_context_t *ctx, id<MTLBuffer> src_buffer, marmot_dtype_t src_dtype, id<MTLBuffer> dst_buffer,
    marmot_dtype_t dst_dtype, size_t elements
) {
    if (elements == 0 || src_buffer == nil || dst_buffer == nil || ctx == nullptr) {
        return MARMOT_SUCCESS;
    }
    if (src_dtype == dst_dtype) {
        // Should be handled by caller, but treat as raw copy fallback.
        return MARMOT_SUCCESS;
    }
    const char *kernel_name = metal_residency_convert_kernel_name(src_dtype, dst_dtype);
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    size_t src_stride = marmot_dtype_size(src_dtype);
    size_t dst_stride = marmot_dtype_size(dst_dtype);
    if (src_stride == 0 || dst_stride == 0) {
        [pipeline release];
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    size_t processed = 0;
    while (processed < elements) {
        size_t remaining = elements - processed;
        uint32_t chunk = remaining > UINT32_MAX ? UINT32_MAX : (uint32_t)remaining;
        NSUInteger src_offset = (NSUInteger)(processed * src_stride);
        NSUInteger dst_offset = (NSUInteger)(processed * dst_stride);
        [encoder setBuffer:src_buffer offset:src_offset atIndex:0];
        [encoder setBuffer:dst_buffer offset:dst_offset atIndex:1];
        [encoder setBytes:&chunk length:sizeof(uint32_t) atIndex:2];
        MTLSize threadgroupSize = metal_threads_for_elements(pipeline, (NSUInteger)chunk, 512);
        [encoder dispatchThreads:MTLSizeMake(chunk, 1, 1) threadsPerThreadgroup:threadgroupSize];
        processed += chunk;
    }

    [pipeline release];
    metal_command_stream_flush(ctx, false);
    return MARMOT_SUCCESS;
}

static size_t metal_allocator_pool_bucket_for(size_t size) {
    for (size_t i = 0; i < METAL_ALLOCATOR_POOL_BUCKET_COUNT; ++i) {
        if (size <= metal_allocator_pool_bucket_sizes[i]) {
            return i;
        }
    }
    return METAL_ALLOCATOR_POOL_BUCKET_COUNT;
}

static size_t metal_allocator_pool_bucket_size(size_t bucket) {
    if (bucket >= METAL_ALLOCATOR_POOL_BUCKET_COUNT) {
        return 0;
    }
    return metal_allocator_pool_bucket_sizes[bucket];
}

static id<MTLBuffer> metal_buffer_detach(metal_context_t *ctx, void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return nil;
    }
    pthread_mutex_lock(&ctx->buffer_mutex);
    NSValue *key = metal_key_for_pointer(ptr);
    id<MTLBuffer> buffer = ctx->buffer_registry[key];
    if (buffer != nil) {
        [buffer retain];
        [ctx->buffer_registry removeObjectForKey:key];
    }
    pthread_mutex_unlock(&ctx->buffer_mutex);
    [key release];
    return buffer;
}

static void metal_buffer_attach(metal_context_t *ctx, void *ptr, id<MTLBuffer> buffer) {
    if (ctx == nullptr || ptr == nullptr || buffer == nil) {
        return;
    }
    pthread_mutex_lock(&ctx->buffer_mutex);
    NSValue *key = metal_key_for_pointer(ptr);
    ctx->buffer_registry[key] = buffer;
    pthread_mutex_unlock(&ctx->buffer_mutex);
    [key release];
}

static bool metal_allocator_pool_acquire(
    metal_context_t *ctx, marmot_alloc_type_t type, size_t request_size, void **out_ptr, size_t *out_size
) {
    if (ctx == nullptr || out_ptr == nullptr || out_size == nullptr) {
        return false;
    }
    if (type != MARMOT_ALLOC_GPU_SHARED && type != MARMOT_ALLOC_GPU_PRIVATE) {
        return false;
    }
    size_t bucket = metal_allocator_pool_bucket_for(request_size);
    if (bucket >= METAL_ALLOCATOR_POOL_BUCKET_COUNT) {
        return false;
    }

    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    metal_pool_entry_t *prev = nullptr;
    metal_pool_entry_t *entry = ctx->allocator_stats.pool[bucket].head;
    while (entry != nullptr && entry->type != type) {
        prev = entry;
        entry = entry->next;
    }
    if (entry == nullptr) {
        ctx->allocator_stats.pool_misses++;
        pthread_mutex_unlock(&ctx->allocator_stats.mutex);
        return false;
    }
    if (prev == nullptr) {
        ctx->allocator_stats.pool[bucket].head = entry->next;
    } else {
        prev->next = entry->next;
    }
    ctx->allocator_stats.pool[bucket].count--;
    if (ctx->allocator_stats.pooled_bytes >= entry->size) {
        ctx->allocator_stats.pooled_bytes -= entry->size;
    } else {
        ctx->allocator_stats.pooled_bytes = 0;
    }
    ctx->allocator_stats.pool_hits++;
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);

    metal_buffer_attach(ctx, entry->ptr, entry->buffer);
    *out_ptr = entry->ptr;
    *out_size = entry->size;
    [entry->buffer release];
    free(entry);
    return true;
}

static bool metal_allocator_pool_release(metal_context_t *ctx, void *ptr, size_t size, marmot_alloc_type_t type) {
    if (ctx == nullptr || ptr == nullptr || size == 0) {
        return false;
    }
    if (type != MARMOT_ALLOC_GPU_SHARED && type != MARMOT_ALLOC_GPU_PRIVATE) {
        return false;
    }
    size_t bucket = metal_allocator_pool_bucket_for(size);
    if (bucket >= METAL_ALLOCATOR_POOL_BUCKET_COUNT) {
        return false;
    }

    size_t bucket_size = metal_allocator_pool_bucket_size(bucket);
    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    if (ctx->allocator_stats.pooled_bytes + bucket_size > ctx->allocator_stats.pool_capacity_bytes) {
        pthread_mutex_unlock(&ctx->allocator_stats.mutex);
        return false;
    }
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);

    id<MTLBuffer> buffer = metal_buffer_detach(ctx, ptr);
    if (buffer == nil) {
        return false;
    }

    metal_pool_entry_t *entry = (metal_pool_entry_t *)malloc(sizeof(metal_pool_entry_t));
    if (entry == nullptr) {
        metal_buffer_attach(ctx, ptr, buffer);
        [buffer release];
        return false;
    }
    entry->ptr = ptr;
    entry->buffer = buffer;
    entry->size = bucket_size;
    entry->type = type;
    entry->next = nullptr;

    const bool should_defer = (ctx->active_command_buffer != nil) || ctx->has_in_flight_work;

    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    if (ctx->allocator_stats.pooled_bytes + bucket_size > ctx->allocator_stats.pool_capacity_bytes) {
        pthread_mutex_unlock(&ctx->allocator_stats.mutex);
        metal_buffer_attach(ctx, ptr, buffer);
        [buffer release];
        free(entry);
        return false;
    }
    ctx->allocator_stats.pooled_bytes += bucket_size;
    if (!should_defer) {
        entry->next = ctx->allocator_stats.pool[bucket].head;
        ctx->allocator_stats.pool[bucket].head = entry;
        ctx->allocator_stats.pool[bucket].count++;
        pthread_mutex_unlock(&ctx->allocator_stats.mutex);

        metal_residency_invalidate(ctx, ptr);
        return true;
    }
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);

    MarmotMetalMemoryState *state = metal_memory_state_acquire(ctx, true);
    if (state == nil) {
        pthread_mutex_lock(&ctx->allocator_stats.mutex);
        if (ctx->allocator_stats.pooled_bytes >= bucket_size) {
            ctx->allocator_stats.pooled_bytes -= bucket_size;
        } else {
            ctx->allocator_stats.pooled_bytes = 0;
        }
        pthread_mutex_unlock(&ctx->allocator_stats.mutex);
        metal_buffer_attach(ctx, ptr, buffer);
        [buffer release];
        free(entry);
        return false;
    }

    pthread_mutex_lock(&state->mutex);
    MarmotMetalAllocatorDeferredGroup *group = nil;
    if (ctx->active_command_buffer != nil) {
        state->barrier_group = nil;
        if (state->active_group != nil && state->active_group.commandBuffer == ctx->active_command_buffer) {
            group = state->active_group;
        } else {
            group = [[MarmotMetalAllocatorDeferredGroup alloc] init];
            group.ctx = ctx;
            group.commandBuffer = ctx->active_command_buffer;
            [state->groups addObject:group];
            state->active_group = group;
            [group.commandBuffer addCompletedHandler:^(id<MTLCommandBuffer> command_buffer) {
              (void)command_buffer;
              metal_allocator_pool_complete_group(group);
            }];
            [group release];
        }
    } else {
        state->active_group = nil;
        if (state->barrier_group != nil) {
            group = state->barrier_group;
        } else {
            id<MTLCommandBuffer> barrier = nil;
            if (ctx->queue != nil) {
                barrier = [ctx->queue commandBuffer];
            }
            if (barrier == nil) {
                pthread_mutex_unlock(&state->mutex);
                pthread_mutex_lock(&ctx->allocator_stats.mutex);
                if (ctx->allocator_stats.pooled_bytes >= bucket_size) {
                    ctx->allocator_stats.pooled_bytes -= bucket_size;
                } else {
                    ctx->allocator_stats.pooled_bytes = 0;
                }
                pthread_mutex_unlock(&ctx->allocator_stats.mutex);
                metal_buffer_attach(ctx, ptr, buffer);
                [buffer release];
                free(entry);
                return false;
            }

            group = [[MarmotMetalAllocatorDeferredGroup alloc] init];
            group.ctx = ctx;
            [barrier retain];
            group.commandBuffer = barrier;
            [barrier release];
            [state->groups addObject:group];
            state->barrier_group = group;
            [barrier addCompletedHandler:^(id<MTLCommandBuffer> command_buffer) {
              (void)command_buffer;
              metal_allocator_pool_complete_group(group);
            }];
            [barrier commit];
            [group release];
        }
    }

    entry->next = nullptr;
    if (group.head == nullptr) {
        group.head = entry;
        group.tail = entry;
    } else {
        group.tail->next = entry;
        group.tail = entry;
    }
    pthread_mutex_unlock(&state->mutex);

    metal_residency_invalidate(ctx, ptr);
    return true;
}

void metal_allocator_pool_reclaim_deferred(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }

    MarmotMetalMemoryState *state = metal_memory_state_acquire(ctx, false);
    if (state != nil) {
        metal_pool_entry_t *head = nullptr;
        metal_pool_entry_t *tail = nullptr;

        pthread_mutex_lock(&state->mutex);
        for (MarmotMetalAllocatorDeferredGroup *group in state->groups) {
            if (group == nil || group.head == nullptr) {
                continue;
            }
            if (tail != nullptr) {
                tail->next = group.head;
            } else {
                head = group.head;
            }
            tail = group.tail;
            group.head = nullptr;
            group.tail = nullptr;
        }
        pthread_mutex_unlock(&state->mutex);

        if (head != nullptr) {
            metal_allocator_pool_recycle_entries(ctx, head);
        }
    }
}

static marmot_error_t metal_allocator_tracker_add(metal_context_t *ctx, const marmot_allocation_t *info) {
    if (ctx == nullptr || info == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    metal_allocation_entry_t *entry = (metal_allocation_entry_t *)malloc(sizeof(metal_allocation_entry_t));
    if (entry == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    entry->ptr = info->ptr;
    entry->info = *info;
    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    entry->next = ctx->allocator_stats.active_head;
    ctx->allocator_stats.active_head = entry;
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
    return MARMOT_SUCCESS;
}

static metal_allocation_entry_t *metal_allocator_tracker_take(metal_context_t *ctx, void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return nullptr;
    }
    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    metal_allocation_entry_t *prev = nullptr;
    metal_allocation_entry_t *node = ctx->allocator_stats.active_head;
    while (node != nullptr) {
        if (node->ptr == ptr) {
            if (prev == nullptr) {
                ctx->allocator_stats.active_head = node->next;
            } else {
                prev->next = node->next;
            }
            pthread_mutex_unlock(&ctx->allocator_stats.mutex);
            return node;
        }
        prev = node;
        node = node->next;
    }
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
    return nullptr;
}

static void metal_allocator_record_alloc(metal_context_t *ctx, size_t bytes, marmot_allocation_t *alloc) {
    if (ctx == nullptr || alloc == nullptr) {
        return;
    }
    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    ctx->allocator_stats.current_bytes += bytes;
    if (ctx->allocator_stats.current_bytes > ctx->allocator_stats.peak_bytes) {
        ctx->allocator_stats.peak_bytes = ctx->allocator_stats.current_bytes;
    }
    ctx->allocator_stats.total_allocations++;
    ctx->allocator_stats.active_allocations++;
    if (ctx->allocator_stats.active_allocations > ctx->allocator_stats.peak_allocations) {
        ctx->allocator_stats.peak_allocations = ctx->allocator_stats.active_allocations;
    }
    alloc->alloc_id = ctx->allocator_stats.next_alloc_id++;
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
}

static void metal_allocator_record_free(metal_context_t *ctx, size_t bytes) {
    if (ctx == nullptr) {
        return;
    }
    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    if (ctx->allocator_stats.current_bytes >= bytes) {
        ctx->allocator_stats.current_bytes -= bytes;
    } else {
        ctx->allocator_stats.current_bytes = 0;
    }
    if (ctx->allocator_stats.active_allocations > 0) {
        ctx->allocator_stats.active_allocations--;
    }
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
}

static marmot_error_t
metal_create_buffer(metal_context_t *ctx, size_t size, MTLResourceOptions options, void **out_ptr) {
    if (ctx == nullptr || out_ptr == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    size_t actual_size = size == 0 ? 1 : size;
    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:actual_size options:options];
    if (buffer == nil) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    void *contents = buffer.contents;
    if (options == MTLResourceStorageModePrivate) {
        contents = buffer;
    } else if (contents == nullptr) {
        [buffer release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    pthread_mutex_lock(&ctx->buffer_mutex);
    NSValue *key = metal_key_for_pointer(contents);
    ctx->buffer_registry[key] = buffer;
    pthread_mutex_unlock(&ctx->buffer_mutex);
    [key release];

    [buffer release];

    *out_ptr = contents;
    return MARMOT_SUCCESS;
}

static void metal_release_buffer(metal_context_t *ctx, void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return;
    }
    id<MTLBuffer> buffer = metal_buffer_detach(ctx, ptr);
    metal_residency_invalidate(ctx, ptr);
    if (buffer != nil) {
        [buffer release];
    }
}

marmot_error_t
metal_allocate_tracked(metal_context_t *ctx, size_t size, marmot_alloc_type_t type, marmot_allocation_t *out) {
    if (ctx == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    MTLResourceOptions options = MTLResourceStorageModeShared;
    marmot_alloc_type_t effective_type = type;
    switch (type) {
    case MARMOT_ALLOC_GPU_PRIVATE:
        options = MTLResourceStorageModePrivate;
        break;
    case MARMOT_ALLOC_GPU_SHARED:
        options = MTLResourceStorageModeShared;
        break;
    default:
        effective_type = MARMOT_ALLOC_GPU_SHARED;
        options = MTLResourceStorageModeShared;
        break;
    }

    size_t request_size = (size == 0) ? 1 : size;
    size_t bucket = metal_allocator_pool_bucket_for(request_size);
    if (bucket < METAL_ALLOCATOR_POOL_BUCKET_COUNT) {
        request_size = metal_allocator_pool_bucket_size(bucket);
    }

    void *ptr = nullptr;
    if (metal_allocator_pool_acquire(ctx, effective_type, request_size, &ptr, &request_size)) {
        out->ptr = ptr;
        out->size = request_size;
        out->alignment = alignof(max_align_t);
        out->type = effective_type;
        metal_allocator_record_alloc(ctx, out->size, out);
        marmot_error_t track_err = metal_allocator_tracker_add(ctx, out);
        if (track_err != MARMOT_SUCCESS) {
            metal_allocator_record_free(ctx, out->size);
            metal_allocator_pool_release(ctx, out->ptr, out->size, out->type);
            return track_err;
        }
        return MARMOT_SUCCESS;
    }

    marmot_error_t err = metal_create_buffer(ctx, request_size, options, &ptr);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    out->ptr = ptr;
    out->size = (request_size == 0) ? 1 : request_size;
    out->alignment = alignof(max_align_t);
    out->type = effective_type;
    metal_allocator_record_alloc(ctx, out->size, out);
    marmot_error_t track_err = metal_allocator_tracker_add(ctx, out);
    if (track_err != MARMOT_SUCCESS) {
        metal_allocator_record_free(ctx, out->size);
        metal_release_buffer(ctx, ptr);
        return track_err;
    }
    return MARMOT_SUCCESS;
}

static NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *
metal_residency_table_acquire_locked(metal_context_t *ctx, const void *ptr, bool create) {
    NSValue *key = metal_key_for_pointer(ptr);
    NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table = ctx->residency_map[key];
    if (table == nil && create) {
        table = [[NSMutableDictionary alloc] init];
        ctx->residency_map[key] = table;
        [table release];
    }
    [key release];
    return table;
}

static MarmotMetalResidencyRecord *metal_residency_record_acquire_locked(
    metal_context_t *ctx, const void *ptr, marmot_dtype_t compute_dtype, bool create, bool *out_is_new
) {
    if (out_is_new != nullptr) {
        *out_is_new = false;
    }
    NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table =
        metal_residency_table_acquire_locked(ctx, ptr, create);
    if (table == nil) {
        return nil;
    }
    NSNumber *dtypeKey = @(compute_dtype);
    MarmotMetalResidencyRecord *record = table[dtypeKey];
    if (record == nil && create) {
        record = [[MarmotMetalResidencyRecord alloc] init];
        table[dtypeKey] = record;
        [record release];
        if (out_is_new != nullptr) {
            *out_is_new = true;
        }
    }
    return record;
}

static void metal_residency_table_remove_locked(metal_context_t *ctx, const void *ptr) {
    NSValue *key = metal_key_for_pointer(ptr);
    NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table = ctx->residency_map[key];
    if (table != nil) {
        // Explicitly set privateBuffer to nil on each record to release GPU memory.
        // This ensures MTLBuffer resources are freed even if records have high retain counts.
        for (NSNumber *dtypeKey in [table allKeys]) {
            MarmotMetalResidencyRecord *rec = table[dtypeKey];
            if (rec != nil && rec.privateBuffer != nil) {
                g_private_buffer_releases++;
                rec.privateBuffer = nil;
            }
        }
        [table removeAllObjects];
    }
    [ctx->residency_map removeObjectForKey:key];
    if (ctx->residency_dirty != nil) {
        [ctx->residency_dirty removeObjectForKey:key];
    }
    [key release];
}

marmot_error_t metal_alloc(const void *device_ctx, size_t size, void **ptr) {
    if (device_ctx == nullptr || ptr == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_allocation_t allocation = {};
    marmot_error_t err = metal_allocate_tracked(ctx, size, MARMOT_ALLOC_GPU_SHARED, &allocation);
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    *ptr = allocation.ptr;
    return MARMOT_SUCCESS;
}

void metal_free(const void *device_ctx, void *ptr) {
    if (device_ctx == nullptr || ptr == nullptr) {
        return;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    metal_allocation_entry_t *entry = metal_allocator_tracker_take(ctx, ptr);
    if (entry == nullptr) {
        metal_release_buffer(ctx, ptr);
        return;
    }
    metal_allocator_record_free(ctx, entry->info.size);
    if (!metal_allocator_pool_release(ctx, entry->info.ptr, entry->info.size, entry->info.type)) {
        metal_release_buffer(ctx, entry->info.ptr);
    }
    free(entry);
}

marmot_error_t metal_memcpy_to_device(const void *device_ctx, void *dst, const void *src, size_t size) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (dst == nullptr || src == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (dst != src && size > 0) {
        memcpy(dst, src, size);
    }
    if (ctx != nullptr) {
        metal_residency_mark_shared_dirty(ctx, dst);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_memcpy_from_device(const void *device_ctx, void *dst, const void *src, size_t size) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (dst == nullptr || src == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ctx != nullptr) {
        metal_residency_sync_result_t result = metal_residency_sync_range(ctx, src, size);
        if (result == METAL_RESIDENCY_SYNC_RESULT_SYNCED) {
            metal_command_stream_flush(ctx, true);
        } else if (result == METAL_RESIDENCY_SYNC_RESULT_UNTRACKED) {
            if (!metal_command_stream_wait_for_shared_read(ctx, src, size)) {
                metal_command_stream_flush(ctx, true);
            }
        } else {
            (void)metal_command_stream_wait_for_shared_read(ctx, src, size);
        }
    }
    if (dst != src && size > 0) {
        memcpy(dst, src, size);
    }
    return MARMOT_SUCCESS;
}

id<MTLBuffer> metal_buffer_lookup(metal_context_t *ctx, void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return nil;
    }
    pthread_mutex_lock(&ctx->buffer_mutex);
    NSValue *key = metal_key_for_pointer(ptr);
    id<MTLBuffer> buffer = ctx->buffer_registry[key];
    if (buffer != nil) {
        [buffer retain];
    }
    pthread_mutex_unlock(&ctx->buffer_mutex);
    [key release];
    return buffer;
}

id<MTLBuffer> metal_buffer_acquire(metal_context_t *ctx, const void *ptr, size_t length) {
    if (ctx == nullptr || ptr == nullptr || length == 0) {
        return nil;
    }

    id<MTLBuffer> buffer = metal_buffer_lookup(ctx, (void *)ptr);
    if (buffer != nil) {
        return buffer;
    }

    id<MTLBuffer> base_buffer = nil;
    pthread_mutex_lock(&ctx->buffer_mutex);
    for (NSValue *key in ctx->buffer_registry) {
        void *base_ptr = [key pointerValue];
        if (base_ptr == nullptr) {
            continue;
        }

        id<MTLBuffer> candidate = ctx->buffer_registry[key];
        if (candidate == nil) {
            continue;
        }
        void *candidate_ptr = candidate.contents;
        if (candidate_ptr == nullptr || candidate_ptr != base_ptr) {
            continue;
        }

        const NSUInteger candidate_bytes = candidate.length;
        const uintptr_t base_addr = (uintptr_t)base_ptr;
        const uintptr_t ptr_addr = (uintptr_t)ptr;
        if (ptr_addr < base_addr) {
            continue;
        }
        const uintptr_t offset = ptr_addr - base_addr;
        if (offset > candidate_bytes) {
            continue;
        }
        if (length > (size_t)(candidate_bytes - offset)) {
            continue;
        }

        base_buffer = [candidate retain];
        break;
    }
    pthread_mutex_unlock(&ctx->buffer_mutex);

    if (base_buffer != nil) {
        id<MTLBuffer> wrapper = [ctx->device newBufferWithBytesNoCopy:(void *)ptr
                                                               length:length
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:^(void *buffer_ptr, NSUInteger buffer_length) {
                                                            (void)buffer_ptr;
                                                            (void)buffer_length;
                                                            [base_buffer release];
                                                          }];
        if (wrapper == nil) {
            [base_buffer release];
        } else {
            g_shared_buffer_creates++;
            g_shared_buffer_bytes += length;
        }
        return wrapper;
    }

    id<MTLBuffer> result = [ctx->device newBufferWithBytesNoCopy:(void *)ptr
                                                          length:length
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
    if (result != nil) {
        g_shared_buffer_creates++;
        g_shared_buffer_bytes += length;
        // Cache the buffer in the registry for reuse
        metal_buffer_attach(ctx, (void *)ptr, result);
    }
    return result;
}

typedef struct {
    const void *base_ptr;
    size_t offset_bytes;
    MarmotMetalResidencyRecord *record;
} metal_residency_view_t;

static bool metal_residency_find_view_locked(
    metal_context_t *ctx, const void *ptr, size_t bytes, marmot_dtype_t compute_dtype, metal_residency_view_t *view
) {
    if (ctx == nullptr || ptr == nullptr || bytes == 0 || view == nullptr) {
        return false;
    }
    for (NSValue *key in ctx->residency_map) {
        void *base_ptr = [key pointerValue];
        if (base_ptr == nullptr) {
            continue;
        }
        NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table = ctx->residency_map[key];
        if (table == nil || table.count == 0) {
            continue;
        }
        // Get any record to check storageByteLength (all records in table share the same value)
        // Use fast enumeration to avoid allocating NSArray/NSEnumerator on hot path
        MarmotMetalResidencyRecord *any_record = nil;
        for (NSNumber *dtype_key in table) {
            any_record = table[dtype_key];
            break;
        }
        if (any_record == nil || any_record.storageByteLength == 0) {
            continue;
        }
        const uintptr_t base_addr = (uintptr_t)base_ptr;
        const uintptr_t ptr_addr = (uintptr_t)ptr;
        if (ptr_addr < base_addr) {
            continue;
        }
        const size_t offset = (size_t)(ptr_addr - base_addr);
        if (offset > any_record.storageByteLength) {
            continue;
        }
        if (bytes > (size_t)(any_record.storageByteLength - offset)) {
            continue;
        }
        view->base_ptr = base_ptr;
        view->offset_bytes = offset;
        view->record = table[@(compute_dtype)];
        return true;
    }
    return false;
}

metal_tensor_buffer_t metal_buffer_acquire_view(
    metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype, size_t bytes
) {
    metal_tensor_buffer_t view = {nil, 0, false};
    if (ctx == nullptr || tensor == nullptr || tensor->data == nullptr || bytes == 0) {
        return view;
    }

    metal_residency_view_t resolved{};
    pthread_mutex_lock(&ctx->residency_mutex);
    bool has_view = metal_residency_find_view_locked(ctx, tensor->data, bytes, compute_dtype, &resolved);
    if (has_view && resolved.record != nil && resolved.record.privateBuffer != nil && !resolved.record.sharedDirty) {
        const size_t elem_size = marmot_dtype_size(tensor->dtype);
        const size_t compute_size = marmot_dtype_size(compute_dtype);
        if (elem_size != 0 && compute_size != 0 && (resolved.offset_bytes % elem_size) == 0) {
            const size_t offset_elems = resolved.offset_bytes / elem_size;
            if (resolved.record.elementCount >= offset_elems &&
                resolved.record.elementCount - offset_elems >= marmot_tensor_num_elements(tensor)) {
                view.buffer = [resolved.record.privateBuffer retain];
                view.offset = offset_elems * compute_size;
                view.is_private = true;
            }
        }
    }
    pthread_mutex_unlock(&ctx->residency_mutex);

    if (view.buffer != nil) {
        return view;
    }

    id<MTLBuffer> buffer = metal_residency_acquire_existing(ctx, tensor, compute_dtype);
    if (buffer == nil) {
        buffer = metal_residency_acquire_compute(ctx, tensor, compute_dtype, nullptr);
        if (buffer != nil) {
            view.is_private = true;
        }
    } else {
        view.is_private = true;
    }
    if (buffer == nil) {
        buffer = metal_buffer_acquire(ctx, tensor->data, bytes);
    }
    view.buffer = buffer;
    return view;
}

marmot_error_t
metal_copy_regions(metal_context_t *ctx, const metal_buffer_copy_region_t *regions, size_t region_count) {
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    if (regions == nullptr || region_count == 0) {
        return MARMOT_SUCCESS;
    }

    id<MTLBlitCommandEncoder> blitEncoder = metal_command_acquire_blit_encoder(ctx);
    if (blitEncoder == nil) {
        metal_command_stream_discard(ctx);
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    for (size_t i = 0; i < region_count; ++i) {
        const metal_buffer_copy_region_t *region = &regions[i];
        if (region->src == nil || region->dst == nil || region->size == 0) {
            continue;
        }
        [blitEncoder copyFromBuffer:region->src
                       sourceOffset:region->src_offset
                           toBuffer:region->dst
                  destinationOffset:region->dst_offset
                               size:region->size];
    }
    metal_command_stream_flush(ctx, false);
    return MARMOT_SUCCESS;
}

marmot_error_t metal_allocator_usage(const void *device_ctx, marmot_allocator_usage_t *usage) {
    if (device_ctx == nullptr || usage == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    usage->current_bytes = ctx->allocator_stats.current_bytes;
    usage->peak_bytes = ctx->allocator_stats.peak_bytes;
    usage->pooled_bytes = ctx->allocator_stats.pooled_bytes;
    usage->active_allocations = ctx->allocator_stats.active_allocations;
    usage->peak_allocations = ctx->allocator_stats.peak_allocations;
    usage->total_allocations = ctx->allocator_stats.total_allocations;
    usage->pool_hits = ctx->allocator_stats.pool_hits;
    usage->pool_misses = ctx->allocator_stats.pool_misses;
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
    return MARMOT_SUCCESS;
}

static size_t metal_tensor_span_elements(const marmot_tensor_t *tensor) {
    size_t span = marmot::metal::tensor_span_elements(tensor);
    if (span == 0) {
        span = marmot_tensor_num_elements(tensor);
    }
    return span;
}

static size_t metal_tensor_span_bytes_for_dtype(const marmot_tensor_t *tensor, marmot_dtype_t dtype) {
    if (tensor == nullptr) {
        return 0;
    }
    const size_t elements = metal_tensor_span_elements(tensor);
    if (elements == 0) {
        return 0;
    }
    if (dtype == MARMOT_DTYPE_INT4 || dtype == MARMOT_DTYPE_UINT4) {
        return (elements + 1) / 2;
    }
    const size_t dtype_size = marmot_dtype_size(dtype);
    if (dtype_size == 0 || elements > SIZE_MAX / dtype_size) {
        return 0;
    }
    return elements * dtype_size;
}

static size_t metal_tensor_storage_bytes(const marmot_tensor_t *tensor) {
    size_t quant_bytes = marmot_tensor_quant_storage_bytes(tensor);
    if (quant_bytes != 0) {
        return quant_bytes;
    }
    size_t span_bytes = metal_tensor_span_bytes_for_dtype(tensor, tensor->dtype);
    if (span_bytes != 0) {
        return span_bytes;
    }
    return marmot_tensor_size_bytes(tensor);
}

static size_t metal_tensor_bytes_for_dtype(const marmot_tensor_t *tensor, marmot_dtype_t dtype) {
    // Prefer quantized storage sizing when applicable (row-major blocks for weights)
    size_t quant_bytes = marmot_tensor_quant_storage_bytes(tensor);
    if (quant_bytes != 0) {
        return quant_bytes;
    }

    size_t span_bytes = metal_tensor_span_bytes_for_dtype(tensor, dtype);
    if (span_bytes != 0) {
        return span_bytes;
    }

    size_t elements = marmot_tensor_num_elements(tensor);
    size_t dtype_size = marmot_dtype_size(dtype);
    if (dtype_size == 0) {
        dtype_size = marmot_dtype_size(tensor->dtype);
    }
    size_t total = elements * dtype_size;
    if (total == 0) {
        total = marmot_tensor_size_bytes(tensor);
    }
    return total;
}

// Track residency operations
static std::atomic<size_t> g_residency_hits{0};
static std::atomic<size_t> g_residency_misses{0};
static std::atomic<size_t> g_residency_total_bytes{0};
static std::atomic<size_t> g_intermediate_allocs{0};
static std::atomic<size_t> g_intermediate_bytes{0};
static std::atomic<size_t> g_intermediate_frees{0};
static std::atomic<size_t> g_residency_invalidations{0};
static std::atomic<size_t> g_private_buffer_allocs{0};
static std::atomic<size_t> g_private_buffer_bytes{0};

void metal_debug_print_stats_full(metal_context_t *ctx, const char *label) {
    fprintf(stderr, "=== METAL STATS [%s] ===\n", label ? label : "");
    fprintf(
        stderr, "  Model weight misses: %zu (%.1f MB)\n", g_residency_misses.load(),
        g_residency_total_bytes.load() / (1024.0 * 1024.0)
    );
    fprintf(stderr, "  Model weight hits: %zu\n", g_residency_hits.load());
    fprintf(
        stderr, "  Intermediate allocs: %zu (%.1f MB)\n", g_intermediate_allocs.load(),
        g_intermediate_bytes.load() / (1024.0 * 1024.0)
    );
    fprintf(stderr, "  Intermediate frees: %zu\n", g_intermediate_frees.load());
    fprintf(stderr, "  Residency invalidations: %zu\n", g_residency_invalidations.load());
    size_t net_intermediate = g_intermediate_allocs.load() - g_intermediate_frees.load();
    fprintf(stderr, "  NET intermediate: %zd\n", (ssize_t)net_intermediate);
    fprintf(
        stderr, "  Private buffers: allocs=%zu (%.1f MB) releases=%zu NET=%.1f MB\n", g_private_buffer_allocs.load(),
        g_private_buffer_bytes.load() / (1024.0 * 1024.0), g_private_buffer_releases.load(),
        (g_private_buffer_bytes.load() -
         g_private_buffer_releases.load() *
             (g_private_buffer_bytes.load() / std::max(g_private_buffer_allocs.load(), (size_t)1))) /
            (1024.0 * 1024.0)
    );

    if (ctx != nullptr) {
        size_t residency_count = 0;
        size_t residency_private_bytes = 0;
        pthread_mutex_lock(&ctx->residency_mutex);
        NSDictionary<NSValue *, NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *> *residency_snapshot =
            [ctx->residency_map copy];
        residency_count = residency_snapshot != nil ? residency_snapshot.count : 0;
        pthread_mutex_unlock(&ctx->residency_mutex);
        for (NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table in [residency_snapshot allValues]) {
            if (table == nil) {
                continue;
            }
            for (MarmotMetalResidencyRecord *record in [table allValues]) {
                if (record != nil) {
                    residency_private_bytes += record.byteLength;
                }
            }
        }
        [residency_snapshot release];

        size_t buffer_count = 0;
        size_t buffer_bytes = 0;
        pthread_mutex_lock(&ctx->buffer_mutex);
        NSDictionary<NSValue *, id<MTLBuffer>> *buffer_snapshot = [ctx->buffer_registry copy];
        buffer_count = buffer_snapshot != nil ? buffer_snapshot.count : 0;
        pthread_mutex_unlock(&ctx->buffer_mutex);
        for (id<MTLBuffer> buffer in [buffer_snapshot allValues]) {
            buffer_bytes += (size_t)[buffer length];
        }
        [buffer_snapshot release];

        size_t packed_cache_bytes = 0;
        size_t packed_cache_count = 0;
        if (ctx->packed_weight_cache != nil) {
            pthread_mutex_lock(&ctx->packed_weight_mutex);
            packed_cache_bytes = ctx->packed_weight_cache_bytes;
            packed_cache_count = ctx->packed_weight_cache.count;
            pthread_mutex_unlock(&ctx->packed_weight_mutex);
        }

        size_t qkv_cache_bytes = 0;
        size_t qkv_cache_count = 0;
        if (ctx->qkv_fused_cache != nil) {
            pthread_mutex_lock(&ctx->qkv_fused_cache_mutex);
            qkv_cache_bytes = ctx->qkv_fused_cache_bytes;
            qkv_cache_count = ctx->qkv_fused_cache.count;
            pthread_mutex_unlock(&ctx->qkv_fused_cache_mutex);
        }

        size_t bias_cache_tables = 0;
        size_t bias_cache_entries = 0;
        size_t bias_cache_bytes = 0;
        if (ctx->bias_cache != nil) {
            pthread_mutex_lock(&ctx->bias_cache_mutex);
            bias_cache_tables = ctx->bias_cache.count;
            NSArray<NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *> *tables =
                [[ctx->bias_cache allValues] copy];
            pthread_mutex_unlock(&ctx->bias_cache_mutex);

            for (NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *table in tables) {
                if (table == nil) {
                    continue;
                }
                bias_cache_entries += table.count;
                NSArray<MarmotMetalBiasCacheEntry *> *entries = [[table allValues] copy];
                for (MarmotMetalBiasCacheEntry *entry in entries) {
                    if (entry != nil) {
                        bias_cache_bytes += entry.allocation.size;
                    }
                }
                [entries release];
            }
            [tables release];
        }

        size_t deferred_groups = 0;
        size_t deferred_entries = 0;
        MarmotMetalMemoryState *state = metal_memory_state_acquire(ctx, false);
        if (state != nil) {
            pthread_mutex_lock(&state->mutex);
            deferred_groups = state->groups.count;
            for (MarmotMetalAllocatorDeferredGroup *group in state->groups) {
                metal_pool_entry_t *entry = group.head;
                while (entry != nullptr) {
                    deferred_entries++;
                    entry = entry->next;
                }
            }
            pthread_mutex_unlock(&state->mutex);
        }

        fprintf(stderr, "  Residency map entries: %zu\n", residency_count);
        fprintf(stderr, "  Residency private bytes: %.1f MB\n", residency_private_bytes / (1024.0 * 1024.0));
        fprintf(stderr, "  Buffer registry entries: %zu\n", buffer_count);
        fprintf(stderr, "  Buffer registry bytes: %.1f MB\n", buffer_bytes / (1024.0 * 1024.0));
        fprintf(stderr, "  Allocator current_bytes: %.1f MB\n", ctx->allocator_stats.current_bytes / (1024.0 * 1024.0));
        fprintf(stderr, "  Allocator pooled_bytes: %.1f MB\n", ctx->allocator_stats.pooled_bytes / (1024.0 * 1024.0));
        fprintf(
            stderr, "  Packed weight cache: %.1f MB (%zu entries)\n", packed_cache_bytes / (1024.0 * 1024.0),
            packed_cache_count
        );
        fprintf(
            stderr, "  QKV fused cache: %.1f MB (%zu entries)\n", qkv_cache_bytes / (1024.0 * 1024.0), qkv_cache_count
        );
        fprintf(
            stderr, "  Bias cache: %.1f MB (%zu entries, %zu tables)\n", bias_cache_bytes / (1024.0 * 1024.0),
            bias_cache_entries, bias_cache_tables
        );
        fprintf(stderr, "  Deferred pool: groups=%zu entries=%zu\n", deferred_groups, deferred_entries);
        fprintf(
            stderr, "  Command buffers: active=%s in_flight=%s batch_depth=%u\n",
            ctx->active_command_buffer != nil ? "yes" : "no", ctx->has_in_flight_work ? "yes" : "no",
            ctx->command_batch_depth
        );

        marmot_tensor_debug_stats_t tensor_stats = {};
        marmot_tensor_debug_snapshot(&tensor_stats);
        fprintf(
            stderr, "  Host tensors: live=%.1f MB peak=%.1f MB alloc=%.1f MB free=%.1f MB\n",
            tensor_stats.live_bytes / (1024.0 * 1024.0), tensor_stats.peak_bytes / (1024.0 * 1024.0),
            tensor_stats.alloc_bytes / (1024.0 * 1024.0), tensor_stats.free_bytes / (1024.0 * 1024.0)
        );
        fprintf(stderr, "  Host tensors: allocs=%zu frees=%zu\n", tensor_stats.allocs, tensor_stats.frees);
    }
    fprintf(
        stderr, "  Shared buffers: creates=%zu (%.1f MB)\n", g_shared_buffer_creates.load(),
        g_shared_buffer_bytes.load() / (1024.0 * 1024.0)
    );
}

void metal_debug_print_stats(const char *label) {
    metal_debug_print_stats_full(nullptr, label);
}

marmot_error_t metal_residency_make_private(
    metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype, id<MTLBuffer> *out_buffer,
    bool *out_is_new
) {
    if (ctx == nullptr || tensor == nullptr || tensor->data == nullptr || out_buffer == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t tensor_bytes = metal_tensor_storage_bytes(tensor);
    size_t private_bytes = metal_tensor_bytes_for_dtype(tensor, compute_dtype);
    size_t element_count = metal_tensor_span_elements(tensor);
    if (private_bytes == 0 || tensor_bytes == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    bool record_was_created = false;
    pthread_mutex_lock(&ctx->residency_mutex);
    MarmotMetalResidencyRecord *record =
        metal_residency_record_acquire_locked(ctx, tensor->data, compute_dtype, true, &record_was_created);

    // Track allocations (counters only, no logging)
    if (record_was_created) {
        if (tensor->ctx == nullptr) {
            g_residency_misses++;
            g_residency_total_bytes += private_bytes;
        } else {
            g_intermediate_allocs++;
            g_intermediate_bytes += private_bytes;
        }
    } else if (record != nil && record.privateBuffer != nil && tensor->ctx == nullptr) {
        g_residency_hits++;
    }
    if (record == nil) {
        pthread_mutex_unlock(&ctx->residency_mutex);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    bool dtype_changed = (record.storageDtype != tensor->dtype);
    bool storage_size_changed = (record.storageByteLength != tensor_bytes);
    record.storageDtype = tensor->dtype;
    record.computeDtype = compute_dtype;
    record.storageByteLength = tensor_bytes;
    record.elementCount = element_count;

    bool needs_allocation = (record.privateBuffer == nil) || (record.byteLength < private_bytes);
    if (needs_allocation) {
        id<MTLBuffer> newBuffer = [ctx->device newBufferWithLength:private_bytes options:MTLResourceStorageModePrivate];
        g_private_buffer_allocs++;
        g_private_buffer_bytes += private_bytes;
        if (newBuffer == nil) {
            pthread_mutex_unlock(&ctx->residency_mutex);
            os_log_error(
                OS_LOG_DEFAULT, "Metal: failed to allocate private residency buffer (%zu bytes)", private_bytes
            );
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        record.privateBuffer = newBuffer;
        [newBuffer release];
        record.byteLength = private_bytes;
        record.sharedDirty = YES;
        record.privateDirty = NO;
    }

    bool has_host_storage = tensor->data != nullptr && tensor_bytes > 0;
    bool should_upload = has_host_storage && (record.sharedDirty || dtype_changed || storage_size_changed);
    id<MTLBuffer> privateBuffer = [record.privateBuffer retain];
    pthread_mutex_unlock(&ctx->residency_mutex);

    if (privateBuffer == nil) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    if (should_upload && element_count > 0) {
        id<MTLBuffer> sharedBuffer = metal_buffer_acquire(ctx, tensor->data, tensor_bytes);
        if (sharedBuffer == nil) {
            os_log_error(
                OS_LOG_DEFAULT, "Metal residency: missing shared buffer for host upload (%zu bytes)", tensor_bytes
            );
            [privateBuffer release];
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }

        marmot_error_t upload_status = MARMOT_SUCCESS;
        if (tensor->dtype != compute_dtype) {
            upload_status = metal_residency_convert_buffer(
                ctx, sharedBuffer, tensor->dtype, privateBuffer, compute_dtype, element_count
            );
        } else {
            id<MTLBlitCommandEncoder> blit = metal_command_acquire_blit_encoder(ctx);
            if (blit == nil) {
                [sharedBuffer release];
                [privateBuffer release];
                metal_command_stream_discard(ctx);
                return MARMOT_ERROR_BACKEND_INIT_FAILED;
            }
            size_t copy_bytes = private_bytes < tensor_bytes ? private_bytes : tensor_bytes;
            [blit copyFromBuffer:sharedBuffer
                     sourceOffset:0
                         toBuffer:privateBuffer
                destinationOffset:0
                             size:copy_bytes];
            metal_command_stream_flush(ctx, false);
        }
        [sharedBuffer release];

        if (upload_status != MARMOT_SUCCESS) {
            [privateBuffer release];
            return upload_status;
        }

        pthread_mutex_lock(&ctx->residency_mutex);
        MarmotMetalResidencyRecord *locked =
            metal_residency_record_acquire_locked(ctx, tensor->data, compute_dtype, false, nullptr);
        if (locked != nil) {
            locked.sharedDirty = NO;
            locked.privateDirty = NO;
        }
        pthread_mutex_unlock(&ctx->residency_mutex);
    } else {
        // No host upload; mark compute side dirty so caller knows data lives in private buffer.
        pthread_mutex_lock(&ctx->residency_mutex);
        MarmotMetalResidencyRecord *locked =
            metal_residency_record_acquire_locked(ctx, tensor->data, compute_dtype, false, nullptr);
        if (locked != nil) {
            locked.sharedDirty = NO;
            locked.privateDirty = YES;
            NSValue *key = metal_key_for_pointer(tensor->data);
            NSMutableSet<NSNumber *> *dirty = ctx->residency_dirty[key];
            if (dirty == nil) {
                dirty = [[NSMutableSet alloc] init];
                ctx->residency_dirty[key] = dirty;
                [dirty release];
            }
            [dirty addObject:@(compute_dtype)];
            [key release];
        }
        pthread_mutex_unlock(&ctx->residency_mutex);
    }

    if (out_is_new != nullptr) {
        *out_is_new = needs_allocation || record_was_created;
    }
    *out_buffer = privateBuffer;
    return MARMOT_SUCCESS;
}

id<MTLBuffer> metal_residency_acquire_compute(
    metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype, bool *out_is_staging
) {
    if (ctx == nullptr || tensor == nullptr) {
        return nil;
    }

    id<MTLBuffer> buffer = nil;
    bool is_new = false;
    marmot_error_t err = metal_residency_make_private(ctx, tensor, compute_dtype, &buffer, &is_new);
    if (err != MARMOT_SUCCESS) {
        return nil;
    }
    if (out_is_staging != nullptr) {
        *out_is_staging = is_new;
    }
    return buffer;
}

id<MTLBuffer>
metal_residency_acquire_existing(metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype) {
    if (ctx == nullptr || tensor == nullptr || tensor->data == nullptr) {
        return nil;
    }

    const size_t private_bytes = metal_tensor_bytes_for_dtype(tensor, compute_dtype);
    if (private_bytes == 0) {
        return nil;
    }

    pthread_mutex_lock(&ctx->residency_mutex);
    NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table =
        metal_residency_table_acquire_locked(ctx, tensor->data, false);
    MarmotMetalResidencyRecord *record = table != nil ? table[@(compute_dtype)] : nil;
    id<MTLBuffer> buffer = nil;
    if (record != nil && record.privateBuffer != nil && !record.sharedDirty && record.byteLength >= private_bytes) {
        buffer = [record.privateBuffer retain];
    }
    pthread_mutex_unlock(&ctx->residency_mutex);
    return buffer;
}

void metal_residency_mark_dirty(metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype) {
    if (ctx == nullptr || tensor == nullptr || tensor->data == nullptr) {
        return;
    }

    ((marmot_tensor_t *)tensor)->memory_location = MARMOT_MEMORY_DEVICE;
    ((marmot_tensor_t *)tensor)->needs_sync = true;
    pthread_mutex_lock(&ctx->residency_mutex);
    NSValue *key = metal_key_for_pointer(tensor->data);
    NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table = ctx->residency_map[key];
    MarmotMetalResidencyRecord *record = table != nil ? table[@(compute_dtype)] : nil;
    if (record != nil && record.privateBuffer != nil) {
        record.privateDirty = YES;
        record.sharedDirty = NO;
        NSMutableSet<NSNumber *> *dirty = ctx->residency_dirty[key];
        if (dirty == nil) {
            dirty = [[NSMutableSet alloc] init];
            ctx->residency_dirty[key] = dirty;
            [dirty release];
        }
        [dirty addObject:@(compute_dtype)];
    }
    pthread_mutex_unlock(&ctx->residency_mutex);
    [key release];
}

static void metal_residency_mark_shared_dirty(metal_context_t *ctx, const void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return;
    }

    pthread_mutex_lock(&ctx->residency_mutex);
    NSValue *key = metal_key_for_pointer(ptr);
    NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table = ctx->residency_map[key];
    if (table != nil) {
        for (NSNumber *dtype_num in table) {
            MarmotMetalResidencyRecord *record = table[dtype_num];
            if (record != nil) {
                record.sharedDirty = YES;
                record.privateDirty = NO;
            }
        }
        if (ctx->residency_dirty != nil) {
            [ctx->residency_dirty removeObjectForKey:key];
        }
    }
    pthread_mutex_unlock(&ctx->residency_mutex);
    [key release];
}

static void metal_residency_sync_single_buffer(
    metal_context_t *ctx, const void *tensor_data, marmot_dtype_t compute_dtype, MarmotMetalResidencyRecord *record
) {
    if (record == nil || record.privateBuffer == nil || !record.privateDirty) {
        return;
    }

    (void)compute_dtype;

    size_t copy_bytes = record.byteLength;
    id<MTLBuffer> privateBuffer = [record.privateBuffer retain];

    id<MTLBuffer> sharedBuffer = metal_buffer_lookup(ctx, (void *)tensor_data);
    if (sharedBuffer == nil) {
        size_t target_bytes = record.storageByteLength != 0 ? record.storageByteLength : record.byteLength;
        sharedBuffer = metal_buffer_acquire(ctx, (void *)tensor_data, target_bytes);
        if (sharedBuffer == nil) {
            [privateBuffer release];
            return;
        }
    }

    marmot_error_t sync_status = MARMOT_SUCCESS;
    bool needs_conversion = record.computeDtype != record.storageDtype && record.storageDtype != MARMOT_DTYPE_COUNT &&
        record.computeDtype != MARMOT_DTYPE_COUNT;

    if (needs_conversion && record.elementCount > 0) {
        sync_status = metal_residency_convert_buffer(
            ctx, privateBuffer, record.computeDtype, sharedBuffer, record.storageDtype, record.elementCount
        );
        copy_bytes = record.storageByteLength;
    } else {
        id<MTLBlitCommandEncoder> blit = metal_command_acquire_blit_encoder(ctx);
        if (blit == nil) {
            sync_status = MARMOT_ERROR_BACKEND_INIT_FAILED;
        } else {
            size_t shared_size = [sharedBuffer length];
            if (copy_bytes > shared_size) {
                copy_bytes = shared_size;
            }
            [blit copyFromBuffer:privateBuffer
                     sourceOffset:0
                         toBuffer:sharedBuffer
                destinationOffset:0
                             size:copy_bytes];
        }
    }
    if (sync_status == MARMOT_SUCCESS) {
        record.privateDirty = NO;
        record.sharedDirty = NO;
    }
    [sharedBuffer release];
    [privateBuffer release];
}

// Sync only the residency entry covering [ptr, ptr+bytes), avoiding global sync.
// Returns CLEAN if no sync needed, SYNCED if blit was encoded, UNTRACKED if range not found.
static metal_residency_sync_result_t metal_residency_sync_range(metal_context_t *ctx, const void *ptr, size_t bytes) {
    if (ctx == nullptr || ptr == nullptr || bytes == 0) {
        return METAL_RESIDENCY_SYNC_RESULT_CLEAN;
    }

    const bool was_syncing = ctx->syncing;
    ctx->syncing = true;

    pthread_mutex_lock(&ctx->residency_mutex);

    // Step 1: Find the residency table containing this pointer range.
    const void *base_ptr = ptr;
    NSValue *key = metal_key_for_pointer(ptr);
    NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table = ctx->residency_map[key];

    if (table == nil) {
        // Exact match failed - search for a buffer that contains this range.
        [key release];
        key = nil;

        for (NSValue *candidate in ctx->residency_map) {
            void *cand_ptr = [candidate pointerValue];
            if (cand_ptr == nullptr) {
                continue;
            }
            NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *cand_table = ctx->residency_map[candidate];
            if (cand_table == nil || cand_table.count == 0) {
                continue;
            }

            // Get buffer size from any record (all share same storage).
            MarmotMetalResidencyRecord *record = [cand_table objectEnumerator].nextObject;
            if (record == nil || record.storageByteLength == 0) {
                continue;
            }

            // Check if [ptr, ptr+bytes) falls within [cand_ptr, cand_ptr+storageByteLength).
            const uintptr_t cand_start = (uintptr_t)cand_ptr;
            const uintptr_t cand_end = cand_start + record.storageByteLength;
            const uintptr_t range_start = (uintptr_t)ptr;
            const uintptr_t range_end = range_start + bytes;

            if (range_start >= cand_start && range_end <= cand_end) {
                key = candidate;
                table = cand_table;
                base_ptr = cand_ptr;
                break;
            }
        }
    }

    if (table == nil) {
        pthread_mutex_unlock(&ctx->residency_mutex);
        ctx->syncing = was_syncing;
        return METAL_RESIDENCY_SYNC_RESULT_UNTRACKED;
    }

    // Step 2: Sync any dirty private buffers for this entry.
    bool did_sync = false;
    for (NSNumber *dtype_num in table) {
        MarmotMetalResidencyRecord *record = table[dtype_num];
        if (record != nil && record.privateDirty && record.privateBuffer != nil) {
            marmot_dtype_t dtype = (marmot_dtype_t)[dtype_num intValue];
            metal_residency_sync_single_buffer(ctx, base_ptr, dtype, record);
            did_sync = true;
        }
    }

    // Clear dirty tracking for this entry.
    if (ctx->residency_dirty != nil) {
        [ctx->residency_dirty removeObjectForKey:key];
    }

    pthread_mutex_unlock(&ctx->residency_mutex);
    ctx->syncing = was_syncing;

    return did_sync ? METAL_RESIDENCY_SYNC_RESULT_SYNCED : METAL_RESIDENCY_SYNC_RESULT_CLEAN;
}

void metal_residency_sync_dirty_buffers(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }

    bool was_syncing = ctx->syncing;
    ctx->syncing = true;

    pthread_mutex_lock(&ctx->residency_mutex);
    NSMutableDictionary<NSValue *, NSMutableSet<NSNumber *> *> *dirty_copy = [ctx->residency_dirty copy];
    [ctx->residency_dirty removeAllObjects];
    pthread_mutex_unlock(&ctx->residency_mutex);

    if (dirty_copy == nil) {
        ctx->syncing = was_syncing;
        return;
    }

    for (NSValue *key in dirty_copy) {
        NSMutableSet<NSNumber *> *dtypes = dirty_copy[key];
        const void *ptr = [key pointerValue];
        if (dtypes == nil) {
            continue;
        }

        for (NSNumber *dtype_num in dtypes) {
            marmot_dtype_t dtype = (marmot_dtype_t)[dtype_num intValue];
            pthread_mutex_lock(&ctx->residency_mutex);
            NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *table = ctx->residency_map[key];
            MarmotMetalResidencyRecord *record = table != nil ? table[dtype_num] : nil;
            if (record != nil && record.privateDirty) {
                metal_residency_sync_single_buffer(ctx, ptr, dtype, record);
            }
            pthread_mutex_unlock(&ctx->residency_mutex);
        }
    }

    if ([dirty_copy respondsToSelector:@selector(release)]) {
        [dirty_copy release];
    }

    if (ctx->active_command_buffer != nil) {
        metal_command_stream_flush(ctx, true);
    }
    ctx->syncing = was_syncing;
}

void metal_residency_invalidate(metal_context_t *ctx, const void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return;
    }

    g_residency_invalidations++;
    g_intermediate_frees++;

    pthread_mutex_lock(&ctx->residency_mutex);
    metal_residency_table_remove_locked(ctx, ptr);
    pthread_mutex_unlock(&ctx->residency_mutex);

    metal_command_stream_forget_shared_ptr(ctx, ptr);

    // Also clean up any shared buffer in buffer_registry for this pointer
    id<MTLBuffer> buffer = metal_buffer_detach(ctx, (void *)ptr);
    if (buffer != nil) {
        [buffer release];
    }

    metal_bias_cache_remove(ctx, ptr);
    metal_packed_weight_invalidate(ctx, ptr);
}

void metal_residency_invalidate_mapped_range(metal_context_t *ctx, const void *start, size_t length) {
    if (ctx == nullptr || start == nullptr || length == 0) {
        return;
    }

    const uintptr_t start_addr = (uintptr_t)start;
    const uintptr_t end_addr = start_addr + length;
    if (end_addr <= start_addr) {
        return;
    }

    pthread_mutex_lock(&ctx->residency_mutex);
    NSArray<NSValue *> *residency_keys = [[ctx->residency_map allKeys] copy];
    if (residency_keys != nil) {
        for (NSValue *key in residency_keys) {
            void *ptr = [key pointerValue];
            const uintptr_t addr = (uintptr_t)ptr;
            if (addr >= start_addr && addr < end_addr) {
                [ctx->residency_map removeObjectForKey:key];
                if (ctx->residency_dirty != nil) {
                    [ctx->residency_dirty removeObjectForKey:key];
                }
                metal_command_stream_forget_shared_ptr(ctx, ptr);
            }
        }
        [residency_keys release];
    }
    pthread_mutex_unlock(&ctx->residency_mutex);

    if (ctx->bias_cache != nil) {
        pthread_mutex_lock(&ctx->bias_cache_mutex);
        NSArray<NSValue *> *bias_keys = [[ctx->bias_cache allKeys] copy];
        if (bias_keys != nil) {
            for (NSValue *key in bias_keys) {
                void *ptr = [key pointerValue];
                const uintptr_t addr = (uintptr_t)ptr;
                if (addr >= start_addr && addr < end_addr) {
                    NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *table = ctx->bias_cache[key];
                    if (table != nil) {
                        [table removeAllObjects];
                    }
                    [ctx->bias_cache removeObjectForKey:key];
                }
            }
            [bias_keys release];
        }
        pthread_mutex_unlock(&ctx->bias_cache_mutex);
    }

    if (ctx->packed_weight_cache != nil) {
        pthread_mutex_lock(&ctx->packed_weight_mutex);
        NSArray<NSValue *> *packed_keys = [[ctx->packed_weight_cache allKeys] copy];
        if (packed_keys != nil) {
            for (NSValue *key in packed_keys) {
                void *ptr = [key pointerValue];
                const uintptr_t addr = (uintptr_t)ptr;
                if (addr < start_addr || addr >= end_addr) {
                    continue;
                }

                MarmotMetalPackedWeightRecord *record = ctx->packed_weight_cache[key];
                if (record != nil) {
                    size_t bytes = record.packedByteLength;
                    if (ctx->packed_weight_cache_bytes >= bytes) {
                        ctx->packed_weight_cache_bytes -= bytes;
                    } else {
                        ctx->packed_weight_cache_bytes = 0;
                    }
                }

                [ctx->packed_weight_cache removeObjectForKey:key];
                if (ctx->packed_weight_lru_keys != nil) {
                    NSUInteger idx = [ctx->packed_weight_lru_keys indexOfObject:key];
                    if (idx != NSNotFound) {
                        [ctx->packed_weight_lru_keys removeObjectAtIndex:idx];
                    }
                }
            }
            [packed_keys release];
        }
        pthread_mutex_unlock(&ctx->packed_weight_mutex);
    }
}

static NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *
metal_bias_cache_lookup_table_locked(metal_context_t *ctx, const void *ptr, bool create_if_missing) {
    if (ctx->bias_cache == nil) {
        return nil;
    }
    NSValue *ptr_key = metal_key_for_pointer(ptr);
    NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *table = ctx->bias_cache[ptr_key];
    if (table == nil && create_if_missing) {
        table = [[NSMutableDictionary alloc] init];
        ctx->bias_cache[ptr_key] = table;
        [table release];
    }
    [ptr_key release];
    return table;
}

bool metal_bias_cache_fetch(
    metal_context_t *ctx, const void *ptr, marmot_dtype_t dst_dtype, marmot_allocation_t *out_allocation,
    size_t *out_elements
) {
    if (ctx == nullptr || ptr == nullptr || out_allocation == nullptr || out_elements == nullptr) {
        return false;
    }
    bool found = false;
    pthread_mutex_lock(&ctx->bias_cache_mutex);
    NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *table =
        metal_bias_cache_lookup_table_locked(ctx, ptr, false);
    MarmotMetalBiasCacheEntry *entry = table != nil ? table[@(dst_dtype)] : nil;
    if (entry != nil) {
        *out_allocation = entry.allocation;
        *out_elements = entry.elements;
        found = true;
    }
    pthread_mutex_unlock(&ctx->bias_cache_mutex);
    return found;
}

void metal_bias_cache_store(
    metal_context_t *ctx, const void *ptr, marmot_dtype_t dst_dtype, const marmot_allocation_t *allocation,
    size_t elements
) {
    if (ctx == nullptr || ptr == nullptr || allocation == nullptr || allocation->ptr == nullptr) {
        return;
    }

    MarmotMetalBiasCacheEntry *entry = [[MarmotMetalBiasCacheEntry alloc] init];
    entry.ctx = ctx;
    entry.allocation = *allocation;
    entry.elements = elements;

    MarmotMetalBiasCacheEntry *existing = nil;
    pthread_mutex_lock(&ctx->bias_cache_mutex);
    NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *table =
        metal_bias_cache_lookup_table_locked(ctx, ptr, true);
    NSNumber *dtype_key = @(dst_dtype);
    existing = table[dtype_key];
    if (existing != nil) {
        [existing retain];
    }
    table[dtype_key] = entry;
    pthread_mutex_unlock(&ctx->bias_cache_mutex);

    if (existing != nil) {
        [existing release];
    }
    [entry release];
}

void metal_bias_cache_remove(metal_context_t *ctx, const void *ptr) {
    if (ctx == nullptr || ptr == nullptr || ctx->bias_cache == nil) {
        return;
    }
    pthread_mutex_lock(&ctx->bias_cache_mutex);
    NSValue *key = metal_key_for_pointer(ptr);
    NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *table = ctx->bias_cache[key];
    if (table != nil) {
        [table retain];
        [ctx->bias_cache removeObjectForKey:key];
    }
    pthread_mutex_unlock(&ctx->bias_cache_mutex);
    [key release];

    if (table != nil) {
        [table release];
    }
}

void metal_bias_cache_clear(metal_context_t *ctx) {
    if (ctx == nullptr || ctx->bias_cache == nil) {
        return;
    }
    pthread_mutex_lock(&ctx->bias_cache_mutex);
    NSArray<NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *> *tables =
        [[ctx->bias_cache allValues] retain];
    [ctx->bias_cache removeAllObjects];
    pthread_mutex_unlock(&ctx->bias_cache_mutex);

    NSMutableArray<MarmotMetalBiasCacheEntry *> *entries = [[NSMutableArray alloc] init];
    for (NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *table in tables) {
        [entries addObjectsFromArray:[table allValues]];
    }
    [tables release];

    for (MarmotMetalBiasCacheEntry *entry in entries) {
        [entry retain];
    }
    for (MarmotMetalBiasCacheEntry *entry in entries) {
        marmot_allocation_t allocation = entry.allocation;
        metal_allocator_ops.free(ctx, &allocation);
        allocation.ptr = nullptr;
        entry.allocation = allocation;
        [entry release];
    }
    [entries release];
}

static marmot_error_t metal_allocator_copy_buffers(metal_context_t *ctx, void *dst_ptr, void *src_ptr, size_t bytes) {
    if (ctx == nullptr || dst_ptr == nullptr || src_ptr == nullptr || bytes == 0) {
        return (bytes == 0) ? MARMOT_SUCCESS : MARMOT_ERROR_INVALID_ARGUMENT;
    }

    id<MTLBuffer> src = metal_buffer_lookup(ctx, src_ptr);
    id<MTLBuffer> dst = metal_buffer_lookup(ctx, dst_ptr);
    if (src == nil || dst == nil) {
        if (src != nil) {
            [src release];
        }
        if (dst != nil) {
            [dst release];
        }
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_buffer_copy_region_t region = {
        .src = src,
        .src_offset = 0,
        .dst = dst,
        .dst_offset = 0,
        .size = bytes,
    };
    marmot_error_t err = metal_copy_regions(ctx, &region, 1);
    [src release];
    [dst release];
    return err;
}

static marmot_error_t metal_allocator_alloc(
    void *allocator_ctx, size_t size, size_t alignment, marmot_alloc_type_t type, marmot_allocation_t *out
) {
    metal_context_t *ctx = (metal_context_t *)allocator_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    (void)alignment;
    marmot_allocation_t allocation = {};
    marmot_error_t err = metal_allocate_tracked(ctx, size, type, &allocation);
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    *out = allocation;
    return MARMOT_SUCCESS;
}

static void metal_allocator_free(void *allocator_ctx, marmot_allocation_t *alloc) {
    if (allocator_ctx == nullptr || alloc == nullptr || alloc->ptr == nullptr) {
        return;
    }
    metal_context_t *ctx = (metal_context_t *)allocator_ctx;
    metal_free(ctx, alloc->ptr);
    alloc->ptr = nullptr;
    alloc->size = 0;
}

static marmot_error_t metal_allocator_realloc(void *allocator_ctx, marmot_allocation_t *alloc, size_t new_size) {
    if (allocator_ctx == nullptr || alloc == nullptr || alloc->ptr == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t target_size = (new_size == 0) ? 1 : new_size;
    metal_context_t *ctx = (metal_context_t *)allocator_ctx;
    marmot_allocation_t replacement = {};
    marmot_error_t err = metal_allocate_tracked(ctx, target_size, alloc->type, &replacement);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    size_t to_copy = (target_size < alloc->size) ? target_size : alloc->size;
    if (to_copy > 0) {
        marmot_error_t copy_err = MARMOT_SUCCESS;
        if (alloc->type == MARMOT_ALLOC_GPU_SHARED && replacement.type == MARMOT_ALLOC_GPU_SHARED) {
            memcpy(replacement.ptr, alloc->ptr, to_copy);
        } else {
            copy_err = metal_allocator_copy_buffers(ctx, replacement.ptr, alloc->ptr, to_copy);
        }
        if (copy_err != MARMOT_SUCCESS) {
            metal_free(ctx, replacement.ptr);
            return copy_err;
        }
    }

    metal_free(ctx, alloc->ptr);
    *alloc = replacement;
    return MARMOT_SUCCESS;
}

static size_t metal_allocator_get_peak_usage(void *allocator_ctx) {
    if (allocator_ctx == nullptr) {
        return 0;
    }
    metal_context_t *ctx = (metal_context_t *)allocator_ctx;
    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    size_t value = ctx->allocator_stats.peak_bytes;
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
    return value;
}

static size_t metal_allocator_get_current_usage(void *allocator_ctx) {
    if (allocator_ctx == nullptr) {
        return 0;
    }
    metal_context_t *ctx = (metal_context_t *)allocator_ctx;
    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    size_t value = ctx->allocator_stats.current_bytes;
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
    return value;
}

const marmot_allocator_ops_t metal_allocator_ops MARMOT_MAYBE_UNUSED = {
    .alloc = metal_allocator_alloc,
    .free = metal_allocator_free,
    .realloc = metal_allocator_realloc,
    .get_peak_usage = metal_allocator_get_peak_usage,
    .get_current_usage = metal_allocator_get_current_usage,
};

#endif // __APPLE__
