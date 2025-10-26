#include "core/dispatch/kernel_query.h"
#include "internal/metal_kernel_query.gen.h"
#include "internal/metal_kernel_runtime.h"
#include "internal/metal_matmul_internal.h"
#include "metal_backend_internal.h"
#include "metal_caps.h"
#include "metal_packed_weight.h"
#include "ops/matmul_kernels.h"

#ifdef __APPLE__

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <limits.h>
#import <os/log.h>
#include <string.h>

extern const marmot_allocator_ops_t metal_allocator_ops;

static inline os_log_t metal_log() {
    static os_log_t logger = nullptr;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      logger = os_log_create("marmot", "metal");
    });
    return logger;
}

@interface NSObject (MarmotMetalCounters)
- (void)sampleCountersInBuffer:(id<MTLCounterSampleBuffer>)buffer atSampleIndex:(NSUInteger)sampleIndex;
- (void)resolveCounters:(id<MTLCounterSampleBuffer>)sampleBuffer
                  range:(NSRange)range
      destinationBuffer:(id<MTLBuffer>)destinationBuffer
      destinationOffset:(NSUInteger)offset;
@end

static const char *metal_routing_category_name(metal_routing_category_t category) {
    switch (category) {
    case METAL_ROUTING_CATEGORY_BINARY:
        return "binary";
    case METAL_ROUTING_CATEGORY_UNARY:
        return "unary";
    case METAL_ROUTING_CATEGORY_TERNARY:
        return "ternary";
    case METAL_ROUTING_CATEGORY_REDUCTION:
        return "reduction";
    case METAL_ROUTING_CATEGORY_NORMALIZATION:
        return "normalization";
    case METAL_ROUTING_CATEGORY_MATMUL:
        return "matmul";
    case METAL_ROUTING_CATEGORY_VEC_DOT:
        return "vec_dot";
    case METAL_ROUTING_CATEGORY_TENSOR:
        return "tensor";
    case METAL_ROUTING_CATEGORY_MISC:
        return "misc";
    case METAL_ROUTING_CATEGORY_COUNT:
        break;
    }
    return "unknown";
}

void metal_routing_log_decision(
    metal_context_t *ctx, metal_routing_category_t category, const char *op_name, size_t problem_bytes, bool using_gpu,
    const char *reason
) {
    if (ctx == nullptr || !ctx->routing_debug) {
        return;
    }
    double mib = problem_bytes / (1024.0 * 1024.0);
    os_log_info(
        metal_log(), "routing[%{public}s]: %{public}s -> %{public}s (%.3f MiB, reason=%{public}s)",
        metal_routing_category_name(category), op_name != nullptr ? op_name : "op", using_gpu ? "GPU" : "CPU", mib,
        reason != nullptr ? reason : "unspecified"
    );
}

typedef struct {
    uint64_t timestamp;
    uint32_t timestampFrequency;
    uint32_t reserved;
} metal_counter_result_timestamp_t;

static void metal_profiling_init(metal_context_t *ctx) {
    ctx->profiling_enabled = false;
    ctx->profiling_active = false;
    ctx->profiling_start_index = 0;
    ctx->profiling_end_index = 0;
    ctx->timestamp_buffer = nil;
    ctx->timestamp_resolve_buffer = nil;
    ctx->profiling_label[0] = '\0';
    ctx->profiling_last_gpu_time_ns = 0;
    ctx->profiling_wall_start_ns = 0;

    const char *env = getenv("MARMOT_PROFILE_GPU");
    if (env == nullptr) {
        return;
    }

    if (![ctx->device respondsToSelector:@selector(counterSets)]) {
        os_log_info(metal_log(), "Metal profiling: device does not expose counter sets");
        return;
    }

    NSArray<id<MTLCounterSet>> *sets = [ctx->device counterSets];
    id<MTLCounterSet> timestampSet = nil;
    for (id<MTLCounterSet> set in sets) {
        if ([[set name] containsString:@"timestamp"]) {
            timestampSet = set;
            break;
        }
    }
    if (timestampSet == nil) {
        os_log_info(metal_log(), "Metal profiling: no timestamp counter set available");
        return;
    }

    MTLCounterSampleBufferDescriptor *descriptor = [[MTLCounterSampleBufferDescriptor alloc] init];
    descriptor.counterSet = timestampSet;
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.sampleCount = 4;

    NSError *error = nil;
    id<MTLCounterSampleBuffer> buffer = [ctx->device newCounterSampleBufferWithDescriptor:descriptor error:&error];
    if (buffer == nil) {
        os_log_info(metal_log(), "Metal profiling: failed to create counter sample buffer (%{public}@)", error);
        [descriptor release];
        return;
    }

    size_t resolve_length = descriptor.sampleCount * sizeof(metal_counter_result_timestamp_t);
    id<MTLBuffer> resolve_buffer = [ctx->device newBufferWithLength:resolve_length
                                                            options:MTLResourceStorageModeShared];
    if (resolve_buffer == nil) {
        os_log_info(metal_log(), "Metal profiling: failed to allocate resolve buffer");
        [buffer release];
        [descriptor release];
        return;
    }

    ctx->timestamp_buffer = buffer;
    ctx->timestamp_resolve_buffer = resolve_buffer;
    ctx->profiling_enabled = true;

    [descriptor release];
    fprintf(stderr, "[gpu profile] Metal GPU timestamp profiling enabled\n");
}

void metal_profiling_reset(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }
    ctx->profiling_active = false;
    ctx->profiling_start_index = 0;
    ctx->profiling_end_index = 0;
    ctx->profiling_label[0] = '\0';
    ctx->profiling_wall_start_ns = 0;
}

void metal_profiling_set_label(metal_context_t *ctx, const char *label) {
    if (ctx == nullptr || !ctx->profiling_enabled || label == nullptr) {
        return;
    }
    size_t len = strnlen(label, sizeof(ctx->profiling_label) - 1);
    memcpy(ctx->profiling_label, label, len);
    ctx->profiling_label[len] = '\0';
}

static uint64_t metal_profiling_wall_clock_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

void metal_profiling_begin(metal_context_t *ctx) {
    if (ctx == nullptr || !ctx->profiling_enabled || ctx->active_command_buffer == nil || ctx->profiling_active) {
        return;
    }
    // Use GPU counter sampling if available
    if ([ctx->active_command_buffer respondsToSelector:@selector(sampleCountersInBuffer:atSampleIndex:)] &&
        ctx->timestamp_buffer != nil) {
        [(NSObject *)ctx->active_command_buffer sampleCountersInBuffer:ctx->timestamp_buffer atSampleIndex:0];
    }
    // Also record wall-clock time as fallback
    ctx->profiling_wall_start_ns = metal_profiling_wall_clock_ns();
    ctx->profiling_active = true;
    ctx->profiling_start_index = 0;
    ctx->profiling_end_index = 1;
}

void metal_profiling_end(metal_context_t *ctx) {
    if (ctx == nullptr || !ctx->profiling_enabled || !ctx->profiling_active || ctx->active_command_buffer == nil) {
        return;
    }
    if (![ctx->active_command_buffer respondsToSelector:@selector(sampleCountersInBuffer:atSampleIndex:)]) {
        return;
    }
    [(NSObject *)ctx->active_command_buffer sampleCountersInBuffer:ctx->timestamp_buffer
                                                     atSampleIndex:ctx->profiling_end_index];
}

void metal_profiling_commit(metal_context_t *ctx) {
    if (ctx == nullptr || !ctx->profiling_enabled || !ctx->profiling_active || ctx->active_command_buffer == nil) {
        return;
    }
    NSRange range = NSMakeRange(ctx->profiling_start_index, ctx->profiling_end_index - ctx->profiling_start_index + 1);
    if (![ctx->active_command_buffer
            respondsToSelector:@selector(resolveCounters:range:destinationBuffer:destinationOffset:)]) {
        return;
    }
    [(NSObject *)ctx->active_command_buffer resolveCounters:ctx->timestamp_buffer
                                                      range:range
                                          destinationBuffer:ctx->timestamp_resolve_buffer
                                          destinationOffset:0];
}

void metal_profiling_complete(metal_context_t *ctx) {
    if (ctx == nullptr || !ctx->profiling_enabled || !ctx->profiling_active) {
        return;
    }

    uint64_t ns = 0;
    bool used_gpu_counter = false;

    // Try to get GPU timestamp if available
    if (ctx->timestamp_resolve_buffer != nil) {
        const metal_counter_result_timestamp_t *samples =
            (const metal_counter_result_timestamp_t *)[ctx->timestamp_resolve_buffer contents];
        if (samples != nullptr) {
            const metal_counter_result_timestamp_t *start_sample = &samples[ctx->profiling_start_index];
            const metal_counter_result_timestamp_t *end_sample = &samples[ctx->profiling_end_index];
            if (start_sample->timestampFrequency != 0) {
                uint64_t delta = end_sample->timestamp > start_sample->timestamp
                    ? end_sample->timestamp - start_sample->timestamp
                    : 0;
                ns = (delta * 1000000000ull) / start_sample->timestampFrequency;
                used_gpu_counter = (ns > 0);
            }
        }
    }

    // Fall back to wall-clock timing (includes dispatch overhead but better than nothing)
    if (!used_gpu_counter && ctx->profiling_wall_start_ns > 0) {
        uint64_t end_ns = metal_profiling_wall_clock_ns();
        ns = end_ns - ctx->profiling_wall_start_ns;
    }

    ctx->profiling_last_gpu_time_ns = ns;

    // Print timing
    if (ctx->profiling_label[0] != '\0' && ns > 0) {
        double us = (double)ns / 1.0e3;
        double ms = (double)ns / 1.0e6;
        const char *suffix = used_gpu_counter ? "" : " (wall)";
        if (ms >= 0.1) {
            fprintf(stderr, "[gpu profile] %s: %.3f ms%s\n", ctx->profiling_label, ms, suffix);
        } else {
            fprintf(stderr, "[gpu profile] %s: %.1f µs%s\n", ctx->profiling_label, us, suffix);
        }
    }

    metal_profiling_reset(ctx);
}

id<MTLCommandBuffer> metal_command_acquire_buffer(metal_context_t *ctx) {
    if (ctx == nullptr || ctx->queue == nil) {
        return nil;
    }

    if (ctx->active_command_buffer != nil) {
        return ctx->active_command_buffer;
    }

    id<MTLCommandBuffer> commandBuffer = [ctx->queue commandBuffer];
    if (commandBuffer == nil) {
        return nil;
    }

    ctx->active_command_buffer = [commandBuffer retain];
    return ctx->active_command_buffer;
}

id<MTLComputeCommandEncoder>
metal_command_acquire_compute_encoder(metal_context_t *ctx, id<MTLComputePipelineState> pipeline) {
    if (ctx == nullptr || pipeline == nil) {
        return nil;
    }

    if (ctx->active_blit_encoder != nil) {
        [ctx->active_blit_encoder endEncoding];
        [ctx->active_blit_encoder release];
        ctx->active_blit_encoder = nil;
    }

    id<MTLCommandBuffer> commandBuffer = metal_command_acquire_buffer(ctx);
    if (commandBuffer == nil) {
        return nil;
    }

    if (ctx->active_compute_encoder == nil) {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
            if (ctx->active_command_buffer != nil) {
                [ctx->active_command_buffer release];
                ctx->active_command_buffer = nil;
            }
            return nil;
        }
        ctx->active_compute_encoder = [encoder retain];
    }

    [ctx->active_compute_encoder setComputePipelineState:pipeline];
    return ctx->active_compute_encoder;
}

id<MTLBlitCommandEncoder> metal_command_acquire_blit_encoder(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return nil;
    }

    if (ctx->active_compute_encoder != nil) {
        [ctx->active_compute_encoder endEncoding];
        [ctx->active_compute_encoder release];
        ctx->active_compute_encoder = nil;
    }

    id<MTLCommandBuffer> commandBuffer = metal_command_acquire_buffer(ctx);
    if (commandBuffer == nil) {
        return nil;
    }

    if (ctx->active_blit_encoder == nil) {
        id<MTLBlitCommandEncoder> encoder = [commandBuffer blitCommandEncoder];
        if (encoder == nil) {
            if (ctx->active_command_buffer != nil) {
                [ctx->active_command_buffer release];
                ctx->active_command_buffer = nil;
            }
            return nil;
        }
        ctx->active_blit_encoder = [encoder retain];
    }

    return ctx->active_blit_encoder;
}

void metal_command_stream_track_shared_write(metal_context_t *ctx, const void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return;
    }

    pthread_mutex_lock(&ctx->command_serial_mutex);
    void *value = (void *)ptr;
    NSValue *key = [[NSValue alloc] initWithBytes:&value objCType:@encode(void *)];
    if (ctx->pending_shared_writes != nil) {
        [ctx->pending_shared_writes addObject:key];
    }
    [key release];
    pthread_mutex_unlock(&ctx->command_serial_mutex);
}

bool metal_command_stream_wait_for_shared_read(metal_context_t *ctx, const void *ptr, size_t bytes) {
    if (ctx == nullptr || ptr == nullptr || bytes == 0) {
        return true;
    }

    if (ctx->active_command_buffer != nil && ctx->command_batch_depth == 0) {
        metal_command_stream_flush(ctx, false);
    }

    pthread_mutex_lock(&ctx->command_serial_mutex);
    void *value = (void *)ptr;
    NSValue *key = [[NSValue alloc] initWithBytes:&value objCType:@encode(void *)];
    NSNumber *serial_number = ctx->shared_write_serial != nil ? ctx->shared_write_serial[key] : nil;
    [key release];
    if (serial_number == nil) {
        pthread_mutex_unlock(&ctx->command_serial_mutex);
        return false;
    }

    const uint64_t serial = (uint64_t)[serial_number unsignedLongLongValue];
    while (ctx->completed_command_serial < serial) {
        pthread_cond_wait(&ctx->command_serial_cond, &ctx->command_serial_mutex);
    }
    pthread_mutex_unlock(&ctx->command_serial_mutex);
    return true;
}

void metal_command_stream_forget_shared_ptr(metal_context_t *ctx, const void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return;
    }

    pthread_mutex_lock(&ctx->command_serial_mutex);
    void *value = (void *)ptr;
    NSValue *key = [[NSValue alloc] initWithBytes:&value objCType:@encode(void *)];
    if (ctx->pending_shared_writes != nil) {
        [ctx->pending_shared_writes removeObject:key];
    }
    if (ctx->shared_write_serial != nil) {
        [ctx->shared_write_serial removeObjectForKey:key];
    }
    [key release];
    pthread_mutex_unlock(&ctx->command_serial_mutex);
}

void metal_command_stream_flush(metal_context_t *ctx, bool wait_for_completion) {
    if (ctx == nullptr) {
        return;
    }

    if (ctx->command_batch_depth > 0 && !wait_for_completion && !(ctx->profiling_enabled && ctx->profiling_active)) {
        return;
    }

    bool invoked_from_sync = ctx->syncing;

    @autoreleasepool {
        if (ctx->active_compute_encoder != nil) {
            [ctx->active_compute_encoder endEncoding];
            [ctx->active_compute_encoder release];
            ctx->active_compute_encoder = nil;
        }

        if (ctx->active_blit_encoder != nil) {
            [ctx->active_blit_encoder endEncoding];
            [ctx->active_blit_encoder release];
            ctx->active_blit_encoder = nil;
        }

        if (ctx->active_command_buffer != nil) {
            metal_profiling_commit(ctx);
            bool should_wait =
                wait_for_completion || ctx->bench_sync || (ctx->profiling_enabled && ctx->profiling_active);
            uint64_t serial = 0;
            pthread_mutex_lock(&ctx->command_serial_mutex);
            serial = ctx->next_command_serial++;
            ctx->last_submitted_command_serial = serial;
            if (ctx->pending_shared_writes != nil && ctx->pending_shared_writes.count > 0) {
                if (ctx->shared_write_serial == nil) {
                    ctx->shared_write_serial = [[NSMutableDictionary alloc] init];
                }
                for (NSValue *key in ctx->pending_shared_writes) {
                    ctx->shared_write_serial[key] = @(serial);
                }
                [ctx->pending_shared_writes removeAllObjects];
            }
            if (!should_wait) {
                ctx->has_in_flight_work = true;
            }
            pthread_mutex_unlock(&ctx->command_serial_mutex);

            if (!should_wait) {
                metal_context_t *completion_ctx = ctx;
                const uint64_t completion_serial = serial;
                [ctx->active_command_buffer addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
                  (void)buffer;
                  pthread_mutex_lock(&completion_ctx->command_serial_mutex);
                  if (completion_serial > completion_ctx->completed_command_serial) {
                      completion_ctx->completed_command_serial = completion_serial;
                  }
                  if (completion_ctx->completed_command_serial >= completion_ctx->last_submitted_command_serial) {
                      completion_ctx->has_in_flight_work = false;
                  }
                  pthread_cond_broadcast(&completion_ctx->command_serial_cond);
                  pthread_mutex_unlock(&completion_ctx->command_serial_mutex);
                }];
            }
            [ctx->active_command_buffer commit];
            if (ctx->trace_batch) {
                ctx->trace_commit_count++;
            }
            if (should_wait) {
                [ctx->active_command_buffer waitUntilCompleted];
                metal_profiling_complete(ctx);
                pthread_mutex_lock(&ctx->command_serial_mutex);
                if (serial > ctx->completed_command_serial) {
                    ctx->completed_command_serial = serial;
                }
                if (ctx->completed_command_serial >= ctx->last_submitted_command_serial) {
                    ctx->has_in_flight_work = false;
                }
                pthread_cond_broadcast(&ctx->command_serial_cond);
                pthread_mutex_unlock(&ctx->command_serial_mutex);
            }
            [ctx->active_command_buffer release];
            ctx->active_command_buffer = nil;
            if (!should_wait) {
                metal_profiling_reset(ctx);
            } else {
                metal_allocator_pool_reclaim_deferred(ctx);
            }
        } else if (wait_for_completion && ctx->has_in_flight_work) {
            pthread_mutex_lock(&ctx->command_serial_mutex);
            const uint64_t target_serial = ctx->last_submitted_command_serial;
            while (ctx->completed_command_serial < target_serial) {
                pthread_cond_wait(&ctx->command_serial_cond, &ctx->command_serial_mutex);
            }
            ctx->has_in_flight_work = false;
            pthread_mutex_unlock(&ctx->command_serial_mutex);
            metal_allocator_pool_reclaim_deferred(ctx);
        }
    }

    if (ctx->bench_sync && !invoked_from_sync) {
        ctx->syncing = true;
        metal_residency_sync_dirty_buffers(ctx);
        ctx->syncing = false;
    }
}

void metal_command_batch_begin(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->command_batch_depth == UINT32_MAX) {
        return;
    }
    if (ctx->command_batch_depth == 0 && ctx->trace_batch) {
        ctx->trace_batch_start_commit_count = ctx->trace_commit_count;
    }
    ctx->command_batch_depth++;
}

void metal_command_batch_end(metal_context_t *ctx, bool commit) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->command_batch_depth == 0) {
        return;
    }
    ctx->command_batch_depth--;
    if (ctx->command_batch_depth != 0) {
        return;
    }
    if (commit) {
        metal_command_stream_flush(ctx, false);
        if (ctx->trace_batch) {
            uint64_t commits = ctx->trace_commit_count - ctx->trace_batch_start_commit_count;
            fprintf(stderr, "[metal trace] graph batch commits=%llu\n", (unsigned long long)commits);
        }
        return;
    }
    metal_command_stream_discard(ctx);
    if (ctx->trace_batch) {
        uint64_t commits = ctx->trace_commit_count - ctx->trace_batch_start_commit_count;
        fprintf(stderr, "[metal trace] graph batch commits=%llu (discard)\n", (unsigned long long)commits);
    }
}

// Vtable wrappers for graph batching (match marmot_device_ops signature)
static marmot_error_t metal_graph_batch_begin(void *device_ctx) {
    metal_command_batch_begin(static_cast<metal_context_t *>(device_ctx));
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_graph_batch_end(void *device_ctx, bool commit) {
    metal_command_batch_end(static_cast<metal_context_t *>(device_ctx), commit);
    return MARMOT_SUCCESS;
}

void metal_command_stream_discard(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }

    if (ctx->active_compute_encoder != nil) {
        [ctx->active_compute_encoder endEncoding];
        [ctx->active_compute_encoder release];
        ctx->active_compute_encoder = nil;
    }

    if (ctx->active_blit_encoder != nil) {
        [ctx->active_blit_encoder endEncoding];
        [ctx->active_blit_encoder release];
        ctx->active_blit_encoder = nil;
    }

    if (ctx->active_command_buffer != nil) {
        [ctx->active_command_buffer release];
        ctx->active_command_buffer = nil;
    }

    pthread_mutex_lock(&ctx->command_serial_mutex);
    if (ctx->pending_shared_writes != nil) {
        [ctx->pending_shared_writes removeAllObjects];
    }
    pthread_mutex_unlock(&ctx->command_serial_mutex);

    metal_profiling_reset(ctx);
}

//------------------------------------------------------------------------------
// Context helpers
//------------------------------------------------------------------------------

static marmot_error_t metal_load_library(metal_context_t *ctx) {
    if (ctx->library != nil) {
        return MARMOT_SUCCESS;
    }

    NSError *error = nil;
    dispatch_data_t libraryData = dispatch_data_create(
        marmot_metal_lib, marmot_metal_lib_len, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        DISPATCH_DATA_DESTRUCTOR_DEFAULT
    );

    ctx->library = [ctx->device newLibraryWithData:libraryData error:&error];
    dispatch_release(libraryData);

    if (ctx->library == nil) {
        os_log_error(metal_log(), "Metal: failed to load metallib (%{public}@)", error);
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    return MARMOT_SUCCESS;
}

static marmot_error_t metal_create_device(metal_context_t *ctx) {
    ctx->device = MTLCreateSystemDefaultDevice();
    if (ctx->device == nil) {
        os_log_error(metal_log(), "Metal: no compatible GPU device found");
        return MARMOT_ERROR_DEVICE_NOT_AVAILABLE;
    }

    ctx->queue = [ctx->device newCommandQueue];
    if (ctx->queue == nil) {
        os_log_error(metal_log(), "Metal: failed to create command queue");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    return MARMOT_SUCCESS;
}

marmot_error_t metal_context_init(metal_context_t **out_ctx) {
    if (out_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)calloc(1, sizeof(metal_context_t));
    if (ctx == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    marmot_error_t err = metal_create_device(ctx);
    if (err != MARMOT_SUCCESS) {
        free(ctx);
        return err;
    }

    ctx->buffer_registry = [[NSMutableDictionary alloc] init];
    pthread_mutex_init(&ctx->buffer_mutex, nullptr);

    pthread_mutex_init(&ctx->allocator_stats.mutex, nullptr);
    ctx->allocator_stats.current_bytes = 0;
    ctx->allocator_stats.peak_bytes = 0;
    ctx->allocator_stats.next_alloc_id = 1;
    ctx->allocator_stats.active_head = nullptr;
    ctx->allocator_stats.pooled_bytes = 0;
    ctx->allocator_stats.pool_capacity_bytes = 128ULL * 1024ULL * 1024ULL;
    ctx->allocator_stats.active_allocations = 0;
    ctx->allocator_stats.peak_allocations = 0;
    ctx->allocator_stats.total_allocations = 0;
    ctx->allocator_stats.pool_hits = 0;
    ctx->allocator_stats.pool_misses = 0;

    ctx->pipeline_cache = [[NSMutableDictionary alloc] init];
    pthread_mutex_init(&ctx->pipeline_mutex, nullptr);

    ctx->residency_map = [[NSMutableDictionary alloc] init];
    ctx->residency_dirty = [[NSMutableDictionary alloc] init];
    pthread_mutex_init(&ctx->residency_mutex, nullptr);

    ctx->bias_cache = [[NSMutableDictionary alloc] init];
    pthread_mutex_init(&ctx->bias_cache_mutex, nullptr);
    ctx->packed_weight_cache = [[NSMutableDictionary alloc] init];
    ctx->packed_weight_lru_keys = [[NSMutableOrderedSet alloc] init];
    ctx->packed_weight_cache_bytes = 0;
    pthread_mutex_init(&ctx->packed_weight_mutex, nullptr);
    ctx->qkv_fused_cache = [[NSMutableDictionary alloc] init];
    ctx->qkv_fused_cache_lru_keys = [[NSMutableOrderedSet alloc] init];
    ctx->qkv_fused_cache_bytes = 0;
    pthread_mutex_init(&ctx->qkv_fused_cache_mutex, nullptr);
    const char *packed_env = getenv("MARMOT_METAL_ENABLE_PACKED_WEIGHTS");
    ctx->enable_packed_weights = (packed_env == nullptr || strcmp(packed_env, "0") != 0);
    ctx->packed_weight_min_dim = 256;
    ctx->packed_weight_min_elements = 256ULL * 256ULL;
    const char *packed_min_dim_env = getenv("MARMOT_METAL_PACKED_MIN_DIM");
    if (packed_min_dim_env != nullptr && packed_min_dim_env[0] != '\0') {
        char *endptr = nullptr;
        unsigned long long value = strtoull(packed_min_dim_env, &endptr, 10);
        if (endptr != packed_min_dim_env && value > 0) {
            ctx->packed_weight_min_dim = (size_t)value;
        }
    }
    bool packed_min_elements_overridden = false;
    const char *packed_min_elements_env = getenv("MARMOT_METAL_PACKED_MIN_ELEMENTS");
    if (packed_min_elements_env != nullptr && packed_min_elements_env[0] != '\0') {
        char *endptr = nullptr;
        unsigned long long value = strtoull(packed_min_elements_env, &endptr, 10);
        if (endptr != packed_min_elements_env && value > 0) {
            if (value > (unsigned long long)SIZE_MAX) {
                ctx->packed_weight_min_elements = SIZE_MAX;
            } else {
                ctx->packed_weight_min_elements = (size_t)value;
            }
            packed_min_elements_overridden = true;
        }
    }
    if (!packed_min_elements_overridden) {
        unsigned long long dim = (unsigned long long)(ctx->packed_weight_min_dim == 0 ? 1 : ctx->packed_weight_min_dim);
        unsigned long long square = dim * dim;
        if (square == 0) {
            ctx->packed_weight_min_elements = 0;
        } else if (square > (unsigned long long)SIZE_MAX) {
            ctx->packed_weight_min_elements = SIZE_MAX;
        } else {
            ctx->packed_weight_min_elements = (size_t)square;
        }
    }
    ctx->packed_weight_tile_cols = 32;
    ctx->packed_weight_tile_k = 8;
    ctx->packed_weight_tiles_overridden = false;
    const char *tile_cols_env = getenv("MARMOT_METAL_PACKED_TILE_COLS");
    if (tile_cols_env != nullptr && tile_cols_env[0] != '\0') {
        char *endptr = nullptr;
        unsigned long long value = strtoull(tile_cols_env, &endptr, 10);
        if (endptr != tile_cols_env && value > 0) {
            ctx->packed_weight_tile_cols = (size_t)value;
            ctx->packed_weight_tiles_overridden = true;
        }
    }
    const char *tile_k_env = getenv("MARMOT_METAL_PACKED_TILE_K");
    if (tile_k_env != nullptr && tile_k_env[0] != '\0') {
        char *endptr = nullptr;
        unsigned long long value = strtoull(tile_k_env, &endptr, 10);
        if (endptr != tile_k_env && value > 0) {
            ctx->packed_weight_tile_k = (size_t)value;
            ctx->packed_weight_tiles_overridden = true;
        }
    }
    ctx->packed_weight_cache_limit_bytes = 512ULL * 1024ULL * 1024ULL; // 512 MB default
    ctx->qkv_fused_cache_limit_bytes = 256ULL * 1024ULL * 1024ULL;     // 256 MB default
    const char *packed_cache_limit_env = getenv("MARMOT_METAL_PACKED_CACHE_LIMIT_MB");
    if (packed_cache_limit_env != nullptr && packed_cache_limit_env[0] != '\0') {
        char *endptr = nullptr;
        unsigned long long value = strtoull(packed_cache_limit_env, &endptr, 10);
        if (endptr != packed_cache_limit_env) {
            if (value == 0) {
                ctx->packed_weight_cache_limit_bytes = 0;
            } else {
                unsigned long long bytes = value * 1024ULL * 1024ULL;
                if (bytes > (unsigned long long)SIZE_MAX) {
                    ctx->packed_weight_cache_limit_bytes = SIZE_MAX;
                } else {
                    ctx->packed_weight_cache_limit_bytes = (size_t)bytes;
                }
            }
        }
    }
    const char *qkv_cache_limit_env = getenv("MARMOT_METAL_QKV_FUSED_CACHE_LIMIT_MB");
    if (qkv_cache_limit_env != nullptr && qkv_cache_limit_env[0] != '\0') {
        char *endptr = nullptr;
        unsigned long long value = strtoull(qkv_cache_limit_env, &endptr, 10);
        if (endptr != qkv_cache_limit_env) {
            if (value == 0) {
                ctx->qkv_fused_cache_limit_bytes = 0;
            } else {
                unsigned long long bytes = value * 1024ULL * 1024ULL;
                if (bytes > (unsigned long long)SIZE_MAX) {
                    ctx->qkv_fused_cache_limit_bytes = SIZE_MAX;
                } else {
                    ctx->qkv_fused_cache_limit_bytes = (size_t)bytes;
                }
            }
        }
    }

    ctx->prefer_half_storage = (getenv("MARMOT_METAL_FORCE_F32") == nullptr);
    ctx->quant_activation_mode = MARMOT_QUANT_ACTIVATION_AUTO;
    ctx->bench_sync = false; // Async by default for performance
    ctx->has_in_flight_work = false;
    ctx->active_command_buffer = nil;
    ctx->active_compute_encoder = nil;
    ctx->active_blit_encoder = nil;
    pthread_mutex_init(&ctx->command_serial_mutex, nullptr);
    pthread_cond_init(&ctx->command_serial_cond, nullptr);
    ctx->next_command_serial = 1;
    ctx->last_submitted_command_serial = 0;
    ctx->completed_command_serial = 0;
    ctx->shared_write_serial = [[NSMutableDictionary alloc] init];
    ctx->pending_shared_writes = [[NSMutableSet alloc] init];
    const char *trace_env = getenv("MARMOT_METAL_TRACE_BATCH");
    ctx->trace_batch = (trace_env != nullptr && trace_env[0] != '\0' && strcmp(trace_env, "0") != 0);
    ctx->trace_commit_count = 0;
    ctx->trace_batch_start_commit_count = 0;

    // Create a small dummy buffer for unused shader bindings
    ctx->dummy_buffer = [ctx->device newBufferWithLength:16 options:MTLResourceStorageModeShared];

    metal_norm_context_build(ctx);

    err = metal_load_library(ctx);
    if (err != MARMOT_SUCCESS) {
        metal_context_destroy(ctx);
        return err;
    }

    metal_profiling_init(ctx);

    *out_ctx = ctx;
    return MARMOT_SUCCESS;
}

static marmot_error_t metal_configure(const void *device_ctx, const marmot_context_t *owner_ctx) {
    if (device_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    ctx->quant_activation_mode =
        owner_ctx != nullptr ? owner_ctx->policy.quant_activation_mode : MARMOT_QUANT_ACTIVATION_AUTO;
    ctx->routing_debug = (getenv("MARMOT_DEBUG_ROUTING") != nullptr);
    marmot_device_caps_t detected = marmot_metal_detect_caps(ctx->device);
    ctx->device_caps = detected;
    if (owner_ctx != nullptr) {
        ((marmot_context_t *)owner_ctx)->device_caps = detected;
    }
    return MARMOT_SUCCESS;
}

void metal_context_destroy(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return;
    }

    if (ctx->pipeline_cache != nil) {
        [ctx->pipeline_cache removeAllObjects];
        [ctx->pipeline_cache release];
    }
    pthread_mutex_destroy(&ctx->pipeline_mutex);

    metal_residency_sync_dirty_buffers(ctx);
    metal_command_stream_flush(ctx, true);
    metal_bias_cache_clear(ctx);
    metal_packed_weight_clear(ctx);

    pthread_mutex_lock(&ctx->allocator_stats.mutex);
    metal_allocation_entry_t *allocation = ctx->allocator_stats.active_head;
    ctx->allocator_stats.active_head = nullptr;
    pthread_mutex_unlock(&ctx->allocator_stats.mutex);
    while (allocation != nullptr) {
        metal_allocation_entry_t *next = allocation->next;
        metal_allocator_ops.free(ctx, &allocation->info);
        free(allocation);
        allocation = next;
    }
    for (size_t i = 0; i < METAL_ALLOCATOR_POOL_BUCKET_COUNT; ++i) {
        metal_pool_entry_t *entry = ctx->allocator_stats.pool[i].head;
        ctx->allocator_stats.pool[i].head = nullptr;
        while (entry != nullptr) {
            metal_pool_entry_t *next = entry->next;
            if (entry->buffer != nil) {
                [entry->buffer release];
            }
            free(entry);
            entry = next;
        }
    }
    metal_pool_entry_t *deferred = ctx->allocator_stats.deferred_head;
    ctx->allocator_stats.deferred_head = nullptr;
    ctx->allocator_stats.deferred_tail = nullptr;
    ctx->allocator_stats.deferred_count = 0;
    while (deferred != nullptr) {
        metal_pool_entry_t *next = deferred->next;
        if (deferred->buffer != nil) {
            [deferred->buffer release];
        }
        free(deferred);
        deferred = next;
    }
    pthread_mutex_destroy(&ctx->allocator_stats.mutex);

    if (ctx->residency_map != nil) {
        NSMutableDictionary<NSValue *, NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *> *residency_map =
            ctx->residency_map;
        ctx->residency_map = nil;
        [residency_map removeAllObjects];
        [residency_map release];
    }
    if (ctx->residency_dirty != nil) {
        NSMutableDictionary<NSValue *, NSMutableSet<NSNumber *> *> *residency_dirty = ctx->residency_dirty;
        ctx->residency_dirty = nil;
        [residency_dirty removeAllObjects];
        [residency_dirty release];
    }
    pthread_mutex_destroy(&ctx->residency_mutex);

    if (ctx->bias_cache != nil) {
        [ctx->bias_cache removeAllObjects];
        [ctx->bias_cache release];
    }
    pthread_mutex_destroy(&ctx->bias_cache_mutex);
    if (ctx->packed_weight_cache != nil) {
        [ctx->packed_weight_cache removeAllObjects];
        [ctx->packed_weight_cache release];
    }
    if (ctx->packed_weight_lru_keys != nil) {
        [ctx->packed_weight_lru_keys removeAllObjects];
        [ctx->packed_weight_lru_keys release];
    }
    pthread_mutex_destroy(&ctx->packed_weight_mutex);
    if (ctx->qkv_fused_cache != nil) {
        [ctx->qkv_fused_cache removeAllObjects];
        [ctx->qkv_fused_cache release];
    }
    if (ctx->qkv_fused_cache_lru_keys != nil) {
        [ctx->qkv_fused_cache_lru_keys removeAllObjects];
        [ctx->qkv_fused_cache_lru_keys release];
    }
    ctx->qkv_fused_cache_bytes = 0;
    pthread_mutex_destroy(&ctx->qkv_fused_cache_mutex);

    metal_command_stream_discard(ctx);

    if (ctx->shared_write_serial != nil) {
        [ctx->shared_write_serial removeAllObjects];
        [ctx->shared_write_serial release];
    }
    if (ctx->pending_shared_writes != nil) {
        [ctx->pending_shared_writes removeAllObjects];
        [ctx->pending_shared_writes release];
    }
    pthread_cond_destroy(&ctx->command_serial_cond);
    pthread_mutex_destroy(&ctx->command_serial_mutex);

    if (ctx->buffer_registry != nil) {
        [ctx->buffer_registry removeAllObjects];
        [ctx->buffer_registry release];
    }
    pthread_mutex_destroy(&ctx->buffer_mutex);

    if (ctx->library != nil) {
        [ctx->library release];
    }
    if (ctx->queue != nil) {
        [ctx->queue release];
    }
    if (ctx->device != nil) {
        [ctx->device release];
    }

    if (ctx->reduction_partials_buffer != nil) {
        [ctx->reduction_partials_buffer release];
    }
    if (ctx->embedding_ids_buffer != nil) {
        [ctx->embedding_ids_buffer release];
    }

    if (ctx->timestamp_buffer != nil) {
        [ctx->timestamp_buffer release];
    }
    if (ctx->timestamp_resolve_buffer != nil) {
        [ctx->timestamp_resolve_buffer release];
    }
    if (ctx->rope_cache.buffer != nil) {
        [ctx->rope_cache.buffer release];
        ctx->rope_cache.buffer = nil;
        ctx->rope_cache.capacity_bytes = 0;
        ctx->rope_cache.dim = 0;
        ctx->rope_cache.theta = 0.0f;
        ctx->rope_cache.freq_scale = 1.0f;
        ctx->rope_cache.ext_factor = 0.0f;
        ctx->rope_cache.attn_factor = 1.0f;
        ctx->rope_cache.beta_fast = 0.0f;
        ctx->rope_cache.beta_slow = 0.0f;
        ctx->rope_cache.orig_ctx_len = 0;
        ctx->rope_cache.scaling_type = MARMOT_ROPE_SCALING_NONE;
        ctx->rope_cache.attn_scale = 1.0f;
    }

    if (ctx->dummy_buffer != nil) {
        [ctx->dummy_buffer release];
    }

    free(ctx);
}

id<MTLComputePipelineState> metal_pipeline_get_ns(metal_context_t *ctx, NSString *fn) {
    if (ctx == nullptr || fn == nil) {
        return nil;
    }
    pthread_mutex_lock(&ctx->pipeline_mutex);
    id<MTLComputePipelineState> pipeline = ctx->pipeline_cache[fn];
    if (pipeline != nil) {
        [pipeline retain];
        pthread_mutex_unlock(&ctx->pipeline_mutex);
        return pipeline;
    }

    id<MTLFunction> function = [ctx->library newFunctionWithName:fn];
    if (function == nil) {
        pthread_mutex_unlock(&ctx->pipeline_mutex);
        os_log_error(metal_log(), "Metal: missing kernel function %{public}@", fn);
        return nil;
    }

    NSError *error = nil;
    pipeline = [ctx->device newComputePipelineStateWithFunction:function error:&error];
    [function release];

    if (pipeline == nil) {
        pthread_mutex_unlock(&ctx->pipeline_mutex);
        os_log_error(metal_log(), "Metal: failed to create pipeline %{public}@ (%{public}@)", fn, error);
        return nil;
    }

    ctx->pipeline_cache[fn] = pipeline;
    pthread_mutex_unlock(&ctx->pipeline_mutex);

    id<MTLComputePipelineState> result = [pipeline retain];
    [pipeline release];
    return result;
}

id<MTLComputePipelineState> metal_pipeline_get(metal_context_t *ctx, const char *function_name) {
    if (ctx == nullptr || function_name == nullptr) {
        return nil;
    }

    if (ctx->pipeline_last_name == function_name && ctx->pipeline_last != nil) {
        return [ctx->pipeline_last retain];
    }

    NSString *fn = [[NSString alloc] initWithUTF8String:function_name];
    if (fn == nil) {
        return nil;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get_ns(ctx, fn);
    [fn release];
    if (pipeline != nil) {
        ctx->pipeline_last_name = function_name;
        ctx->pipeline_last = pipeline;
    }
    return pipeline;
}

size_t metal_round_up(size_t value, size_t alignment) {
    if (alignment == 0) {
        return value;
    }
    size_t remainder = value % alignment;
    if (remainder == 0) {
        return value;
    }
    return value + (alignment - remainder);
}

id<MTLBuffer> metal_get_dummy_buffer(metal_context_t *ctx) {
    if (ctx == nullptr) {
        return nil;
    }
    return ctx->dummy_buffer;
}

NSUInteger metal_threadgroup_size_1d(id<MTLComputePipelineState> pipeline, NSUInteger maximum) {
    NSUInteger width = pipeline.threadExecutionWidth;
    if (width == 0) {
        width = 1;
    }
    NSUInteger limit = pipeline.maxTotalThreadsPerThreadgroup;
    if (limit == 0) {
        limit = width;
    }

    NSUInteger chosen = width;
    while (chosen < maximum && chosen + width <= limit) {
        chosen += width;
    }
    if (chosen > limit) {
        chosen = limit;
    }
    if (chosen > maximum) {
        chosen = maximum;
    }
    if (chosen == 0) {
        chosen = 1;
    }
    return chosen;
}

MTLSize metal_threads_for_elements(id<MTLComputePipelineState> pipeline, NSUInteger elements, NSUInteger maximum) {
    NSUInteger threads = metal_threadgroup_size_1d(pipeline, maximum);
    if (threads > elements && elements > 0) {
        threads = elements;
    }
    if (threads == 0) {
        threads = 1;
    }
    return MTLSizeMake(threads, 1, 1);
}

//------------------------------------------------------------------------------
// Backend entry points
//------------------------------------------------------------------------------

static marmot_error_t metal_init(void **device_ctx) {
    if (device_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = nullptr;
    marmot_error_t err = metal_context_init(&ctx);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    *device_ctx = ctx;
    return MARMOT_SUCCESS;
}

static void metal_destroy_wrapper(const void *device_ctx) {
    metal_context_destroy((metal_context_t *)device_ctx);
}

static const char *metal_fma_kernel_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "fma_f32";
    case MARMOT_DTYPE_FLOAT16:
        return "fma_f16";
    case MARMOT_DTYPE_BFLOAT16:
        return "fma_bf16";
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return "fma_fp8_e4m3";
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return "fma_fp8_e5m2";
#endif
    default:
        return nullptr;
    }
}

static marmot_error_t metal_run_fma_kernel(
    metal_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c,
    marmot_tensor_t *out, const char *kernel_name
) {
    if (kernel_name == nullptr) {
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    size_t elements = marmot_tensor_num_elements(a);
    if (elements != marmot_tensor_num_elements(b) || elements != marmot_tensor_num_elements(c) ||
        elements != marmot_tensor_num_elements(out)) {
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    size_t bytes = marmot_dtype_size(a->dtype) * elements;
    size_t out_bytes = marmot_dtype_size(out->dtype) * elements;

    id<MTLBuffer> bufferA = metal_buffer_acquire(ctx, a->data, bytes);
    id<MTLBuffer> bufferB = metal_buffer_acquire(ctx, b->data, bytes);
    id<MTLBuffer> bufferC = metal_buffer_acquire(ctx, c->data, bytes);

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferOut = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferOut == nil) {
        bufferOut = metal_buffer_acquire(ctx, out->data, out_bytes);
    } else {
        out_private = true;
    }
    if (bufferA == nil || bufferB == nil || bufferC == nil || bufferOut == nil) {
        if (bufferA != nil)
            [bufferA release];
        if (bufferB != nil)
            [bufferB release];
        if (bufferC != nil)
            [bufferC release];
        if (bufferOut != nil)
            [bufferOut release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, kernel_name);
    if (pipeline == nil) {
        [bufferA release];
        [bufferB release];
        [bufferC release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferA release];
        [bufferB release];
        [bufferC release];
        [bufferOut release];
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    [encoder setBuffer:bufferA offset:0 atIndex:0];
    [encoder setBuffer:bufferB offset:0 atIndex:1];
    [encoder setBuffer:bufferC offset:0 atIndex:2];
    [encoder setBuffer:bufferOut offset:0 atIndex:3];

    MTLSize gridSize = MTLSizeMake(elements, 1, 1);
    NSUInteger threads = metal_threadgroup_size_1d(pipeline, (NSUInteger)elements);
    MTLSize threadgroupSize = MTLSizeMake(threads, 1, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];

    metal_command_stream_flush(ctx, false);

    [pipeline release];
    [bufferA release];
    [bufferB release];
    [bufferC release];
    [bufferOut release];
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_elementwise_validate_ternary(
    const void *device_ctx, marmot_device_ternary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_tensor_t *c, const marmot_tensor_t *out
) {
    if (device_ctx == nullptr || a == nullptr || b == nullptr || c == nullptr || out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    switch (op) {
    case MARMOT_DEVICE_TERNARY_FMA:
        if (a->dtype != b->dtype || a->dtype != c->dtype || a->dtype != out->dtype) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        break;
    case MARMOT_DEVICE_TERNARY_WHERE:
        if (a->dtype != MARMOT_DTYPE_UINT8) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (b->dtype != c->dtype || b->dtype != out->dtype) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        break;
    default:
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal ternary op not implemented");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_ternary_dispatch(
    const void *device_ctx, marmot_device_ternary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_tensor_t *c, marmot_tensor_t *out
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    marmot_error_t validate = metal_elementwise_validate_ternary(device_ctx, op, a, b, c, out);
    if (validate != MARMOT_SUCCESS) {
        return validate;
    }

    if (op == MARMOT_DEVICE_TERNARY_FMA) {
        const char *kernel_name = metal_fma_kernel_name(a->dtype);
        if (kernel_name == nullptr) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        const size_t problem_bytes = marmot_tensor_size_bytes(out);
        metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_TERNARY, "fma", problem_bytes, true, "gpu");
        return metal_run_fma_kernel(ctx, a, b, c, out, kernel_name);
    }

    if (op == MARMOT_DEVICE_TERNARY_WHERE) {
        const size_t problem_bytes = marmot_tensor_size_bytes(out);
        metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_TERNARY, "where", problem_bytes, true, "gpu");
        const char *kernel_name = metal_kernel_name_for_where(out->dtype);
        if (kernel_name == nullptr) {
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        return metal_elementwise_run_where_kernel(ctx, a, b, c, out, kernel_name);
    }

    marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Metal ternary op not implemented");
    return MARMOT_ERROR_NOT_IMPLEMENTED;
}

// Debug function in metal_memory.mm
extern void metal_debug_print_stats_full(metal_context_t *ctx, const char *label);

static marmot_error_t metal_synchronize(const void *device_ctx) {
    if (device_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    metal_residency_sync_dirty_buffers(ctx);
    metal_command_stream_flush(ctx, true);
    return MARMOT_SUCCESS;
}

static void metal_on_host_ptr_freed(void *device_ctx, const void *ptr) {
    if (device_ctx == nullptr || ptr == nullptr) {
        return;
    }
    metal_residency_invalidate((metal_context_t *)device_ctx, ptr);
}

static void metal_on_host_range_freed(void *device_ctx, const void *start, size_t length) {
    if (device_ctx == nullptr || start == nullptr || length == 0) {
        return;
    }
    metal_residency_invalidate_mapped_range((metal_context_t *)device_ctx, start, length);
}

static const marmot_device_ops_t metal_ops = {
    .init = metal_init,
    .destroy = metal_destroy_wrapper,
    .configure = metal_configure,
    .alloc = metal_alloc,
    .free = metal_free,
    .memcpy_to_device = metal_memcpy_to_device,
    .memcpy_from_device = metal_memcpy_from_device,
    .synchronize = metal_synchronize,
    .allocator_usage = metal_allocator_usage,
    .graph_batch_begin = metal_graph_batch_begin,
    .graph_batch_end = metal_graph_batch_end,
    .on_host_ptr_freed = metal_on_host_ptr_freed,
    .on_host_range_freed = metal_on_host_range_freed,
};

extern "C" const marmot_device_ops_t *marmot_get_metal_ops(void) {
    return &metal_ops;
}

extern "C" bool marmot_metal_default_preferences(const marmot_device_caps_t *caps, marmot_backend_preferences_t *out) {
    if (out == nullptr) {
        return false;
    }
    marmot_dtype_t quant_dtype = MARMOT_DTYPE_FLOAT16;
    if (caps != nullptr && !caps->has_fp16_compute) {
        quant_dtype = MARMOT_DTYPE_FLOAT32;
    }
    *out = (marmot_backend_preferences_t){
        .policy =
            {
                .embedding_quant_output_dtype = quant_dtype,
                .quant_activation_mode = MARMOT_QUANT_ACTIVATION_AUTO,
                .variant_flags_mask = UINT32_MAX,
                .embedding_prefer_gpu_private = true,
                .embedding_allow_quant_decode_on_the_fly = true,
                .matmul_requires_temp_tensors = true,
                .matmul_prefer_packed_weights = false,
            },
        .routing_policy = MARMOT_ROUTING_AUTO,
    };
    return true;
}

#endif // __APPLE__
