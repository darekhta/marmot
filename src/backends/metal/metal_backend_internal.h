#pragma once

#include "marmot/allocator.h"
#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/ops/paged_attention.h"
#include "marmot/tensor.h"
#include "marmot/types.h"

#ifdef __APPLE__

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>

#include <dispatch/dispatch.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// Embedded metallib compiled at build time
extern const uint8_t marmot_metal_lib[];
extern const size_t marmot_metal_lib_len;

extern const marmot_allocator_ops_t metal_allocator_ops;

@class MarmotMetalResidencyRecord;
@class MarmotMetalBiasCacheEntry;
@class MarmotMetalPackedWeightRecord;
@class MarmotMetalQKVPackedWeights;

typedef enum {
    METAL_ROUTING_CATEGORY_BINARY = 0,
    METAL_ROUTING_CATEGORY_UNARY = 1,
    METAL_ROUTING_CATEGORY_TERNARY = 2,
    METAL_ROUTING_CATEGORY_REDUCTION = 3,
    METAL_ROUTING_CATEGORY_NORMALIZATION = 4,
    METAL_ROUTING_CATEGORY_MATMUL = 5,
    METAL_ROUTING_CATEGORY_VEC_DOT = 6,
    METAL_ROUTING_CATEGORY_TENSOR = 7,
    METAL_ROUTING_CATEGORY_MISC = 8,
    METAL_ROUTING_CATEGORY_COUNT = 9,
} metal_routing_category_t;

#define METAL_ALLOCATOR_POOL_BUCKET_COUNT 12

typedef struct metal_allocation_entry {
    void *ptr;
    marmot_allocation_t info;
    struct metal_allocation_entry *next;
} metal_allocation_entry_t;

typedef struct metal_pool_entry {
    void *ptr;
    id<MTLBuffer> buffer;
    size_t size;
    marmot_alloc_type_t type;
    struct metal_pool_entry *next;
} metal_pool_entry_t;

typedef struct {
    metal_pool_entry_t *head;
    size_t count;
} metal_pool_bucket_t;

typedef marmot_error_t (*metal_layernorm_fn)(
    const void *device_ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    const marmot_tensor_t *bias, marmot_tensor_t *out, float eps
);

typedef marmot_error_t (*metal_rmsnorm_fn)(
    const void *device_ctx, const marmot_tensor_t *x, const marmot_tensor_t *residual, const marmot_tensor_t *weight,
    marmot_tensor_t *out, float eps
);

typedef marmot_error_t (*metal_softmax_fn)(const void *device_ctx, const marmot_softmax_desc_t *desc);

typedef enum {
    METAL_NORM_IMPL_GPU = 0,
} metal_norm_impl_kind_t;

#define METAL_NORM_IMPL_COUNT 1

typedef struct {
    metal_layernorm_fn layernorm;
    metal_rmsnorm_fn rmsnorm;
    metal_softmax_fn softmax;
    const char *layernorm_kernel;
    const char *fused_layernorm_kernel;
    const char *rmsnorm_kernel;
    const char *fused_rmsnorm_kernel;
    const char *softmax_kernel;
    const char *softmax_strided_kernel;
    const char *impl_name;
} metal_norm_ops_t;

typedef struct {
    metal_norm_ops_t impls[METAL_NORM_IMPL_COUNT];
    bool has_impl[METAL_NORM_IMPL_COUNT];
} metal_norm_entry_t;

typedef struct {
    marmot_dtype_t dtype;
    metal_norm_impl_kind_t impl_kind;
    metal_norm_ops_t ops;
} metal_norm_traits_t;

typedef struct metal_moe_workspace {
    marmot_allocation_t route_counts_alloc;
    marmot_allocation_t route_offsets_alloc;
    marmot_allocation_t route_status_alloc;
    marmot_allocation_t route_summary_alloc;
    marmot_allocation_t route_indices_alloc;
    marmot_allocation_t route_experts_alloc;
    marmot_allocation_t route_weights_alloc;
    marmot_allocation_t hidden_batch_alloc;
    marmot_allocation_t gate_batch_alloc;
    marmot_allocation_t up_batch_alloc;
    marmot_allocation_t fused_batch_alloc;
    marmot_allocation_t down_batch_alloc;
    size_t *expert_counts;
    size_t *expert_offsets;
    size_t *expert_cursor;
    size_t *expert_order;
    size_t experts_capacity;
    bool in_use;
    uint64_t release_serial;
    struct metal_moe_workspace *next;
} metal_moe_workspace_t;

typedef struct metal_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    // Active command stream state (for batching encoders per command buffer)
    id<MTLCommandBuffer> active_command_buffer;
    id<MTLComputeCommandEncoder> active_compute_encoder;
    id<MTLBlitCommandEncoder> active_blit_encoder;
    bool bench_sync; // when true, always wait after commit
    uint32_t command_batch_depth;
    bool has_in_flight_work;
    pthread_mutex_t command_serial_mutex;
    pthread_cond_t command_serial_cond;
    uint64_t next_command_serial;
    uint64_t last_submitted_command_serial;
    uint64_t completed_command_serial;
    NSMutableDictionary<NSValue *, NSNumber *> *shared_write_serial;
    NSMutableSet<NSValue *> *pending_shared_writes;

    // Buffer registry to track allocations backed by Metal buffers
    NSMutableDictionary<NSValue *, id<MTLBuffer>> *buffer_registry;
    pthread_mutex_t buffer_mutex;

    struct {
        pthread_mutex_t mutex;
        size_t current_bytes;
        size_t peak_bytes;
        uint64_t next_alloc_id;
        metal_allocation_entry_t *active_head;
        size_t pooled_bytes;
        size_t pool_capacity_bytes;
        size_t active_allocations;
        size_t peak_allocations;
        uint64_t total_allocations;
        uint64_t pool_hits;
        uint64_t pool_misses;
        metal_pool_entry_t *deferred_head;
        metal_pool_entry_t *deferred_tail;
        size_t deferred_count;
        metal_pool_bucket_t pool[METAL_ALLOCATOR_POOL_BUCKET_COUNT];
    } allocator_stats;

    // Pipeline cache keyed by function name (NSString *)
    NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipeline_cache;
    pthread_mutex_t pipeline_mutex;
    const char *pipeline_last_name;
    id<MTLComputePipelineState> pipeline_last;

    // Residency tracking for tensor storage/compute views
    NSMutableDictionary<NSValue *, NSMutableDictionary<NSNumber *, MarmotMetalResidencyRecord *> *> *residency_map;
    NSMutableDictionary<NSValue *, NSMutableSet<NSNumber *> *> *residency_dirty;
    pthread_mutex_t residency_mutex;

    NSMutableDictionary<NSValue *, NSMutableDictionary<NSNumber *, MarmotMetalBiasCacheEntry *> *> *bias_cache;
    pthread_mutex_t bias_cache_mutex;
    NSMutableDictionary<NSValue *, MarmotMetalPackedWeightRecord *> *packed_weight_cache;
    pthread_mutex_t packed_weight_mutex;
    NSMutableOrderedSet<NSValue *> *packed_weight_lru_keys;
    size_t packed_weight_cache_bytes;
    size_t packed_weight_cache_limit_bytes;

    NSMutableDictionary<NSValue *, MarmotMetalQKVPackedWeights *> *qkv_fused_cache;
    NSMutableOrderedSet<NSValue *> *qkv_fused_cache_lru_keys;
    size_t qkv_fused_cache_bytes;
    size_t qkv_fused_cache_limit_bytes;
    pthread_mutex_t qkv_fused_cache_mutex;

    marmot_device_caps_t device_caps;

    // Scratch buffers
    id<MTLBuffer> reduction_partials_buffer;
    size_t reduction_partials_capacity;
    id<MTLBuffer> embedding_ids_buffer;
    size_t embedding_ids_capacity;
    metal_moe_workspace_t *moe_workspaces;

    metal_norm_entry_t norm_table[MARMOT_DTYPE_COUNT];

    bool prefer_half_storage;
    marmot_quant_activation_mode_t quant_activation_mode;
    bool routing_debug;
    bool trace_batch;
    uint64_t trace_commit_count;
    uint64_t trace_batch_start_commit_count;
    struct {
        size_t dim;
        float theta;
        float freq_scale;
        float ext_factor;
        float attn_factor;
        float beta_fast;
        float beta_slow;
        uint32_t orig_ctx_len;
        marmot_rope_scaling_type_t scaling_type;
        float attn_scale;
        size_t capacity_bytes;
        id<MTLBuffer> buffer;
    } rope_cache;
    bool profiling_enabled;
    bool profiling_active;
    uint32_t profiling_start_index;
    uint32_t profiling_end_index;
    id<MTLCounterSampleBuffer> timestamp_buffer;
    id<MTLBuffer> timestamp_resolve_buffer;
    char profiling_label[32];
    uint64_t profiling_last_gpu_time_ns;
    uint64_t profiling_wall_start_ns;
    bool syncing;
    bool enable_packed_weights;
    size_t packed_weight_min_dim;
    size_t packed_weight_min_elements;
    size_t packed_weight_tile_cols;
    size_t packed_weight_tile_k;
    bool packed_weight_tiles_overridden;

    // Dummy buffer for shader bindings that require a buffer but aren't accessed
    id<MTLBuffer> dummy_buffer;
} metal_context_t;

// Context helpers ------------------------------------------------------------

marmot_error_t metal_context_init(metal_context_t **out_ctx);
void metal_context_destroy(metal_context_t *ctx);

id<MTLComputePipelineState> metal_pipeline_get(metal_context_t *ctx, const char *function_name);
id<MTLComputePipelineState> metal_pipeline_get_ns(metal_context_t *ctx, NSString *function_name);
id<MTLComputePipelineState>
metal_pipeline_get_with_u32_constant(metal_context_t *ctx, const char *function_name, uint32_t index, uint32_t value);
void metal_moe_workspace_pool_destroy(metal_context_t *ctx);

// Command stream helpers ----------------------------------------------------

id<MTLCommandBuffer> metal_command_acquire_buffer(metal_context_t *ctx);
id<MTLComputeCommandEncoder>
metal_command_acquire_compute_encoder(metal_context_t *ctx, id<MTLComputePipelineState> pipeline);
id<MTLBlitCommandEncoder> metal_command_acquire_blit_encoder(metal_context_t *ctx);
void metal_command_batch_begin(metal_context_t *ctx);
void metal_command_batch_end(metal_context_t *ctx, bool commit);
void metal_command_stream_flush(metal_context_t *ctx, bool wait_for_completion);
void metal_command_stream_discard(metal_context_t *ctx);
void metal_command_stream_track_shared_write(metal_context_t *ctx, const void *ptr);
bool metal_command_stream_wait_for_shared_read(metal_context_t *ctx, const void *ptr, size_t bytes);
void metal_command_stream_forget_shared_ptr(metal_context_t *ctx, const void *ptr);

// Shared utility -------------------------------------------------------------

size_t metal_round_up(size_t value, size_t alignment);

// Returns a small dummy buffer for unused shader bindings that require non-nil buffer
id<MTLBuffer> metal_get_dummy_buffer(metal_context_t *ctx);
NSUInteger metal_threadgroup_size_1d(id<MTLComputePipelineState> pipeline, NSUInteger maximum);
MTLSize metal_threads_for_elements(id<MTLComputePipelineState> pipeline, NSUInteger elements, NSUInteger maximum);
id<MTLBuffer> metal_buffer_acquire(metal_context_t *ctx, const void *ptr, size_t length);
typedef struct {
    id<MTLBuffer> buffer;
    size_t offset;
    bool is_private;
} metal_tensor_buffer_t;

#define METAL_MATMUL_QUANT_HINT_PREFER_MV (1u << 0)

metal_tensor_buffer_t metal_buffer_acquire_view(
    metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype, size_t bytes
);
id<MTLBuffer> metal_residency_acquire_compute(
    metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype, bool *out_is_staging
);
id<MTLBuffer>
metal_residency_acquire_existing(metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype);
void metal_residency_mark_dirty(metal_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t compute_dtype);
void metal_residency_mark_shared_write(metal_context_t *ctx, const void *ptr);
void metal_residency_sync_shared_range(metal_context_t *ctx, const void *ptr, size_t bytes);
void metal_residency_sync_dirty_buffers(metal_context_t *ctx);
void metal_residency_invalidate(metal_context_t *ctx, const void *ptr);
void metal_residency_invalidate_mapped_range(metal_context_t *ctx, const void *start, size_t length);

bool metal_matmul_epilogue_supported(
    const marmot_tensor_t *out, const marmot_matmul_epilogue_t *epilogue, size_t *feature_dim, bool *bias_is_scalar
);
marmot_error_t metal_matmul_apply_epilogue(
    metal_context_t *ctx, const marmot_tensor_t *out, id<MTLBuffer> bufferOut, size_t buffer_out_offset,
    size_t total_elements, size_t feature_dim, bool bias_is_scalar, const marmot_matmul_epilogue_t *epilogue
);
marmot_error_t metal_matmul_generic(
    metal_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out, const char *kernel_name
);
// All compute operations are routed via universal dispatch.
// Backend helpers must not call back into C-API paths.
size_t metal_matmul_min3(size_t a, size_t b, size_t c);
bool metal_matmul_residual_matches(const marmot_tensor_t *out, const marmot_tensor_t *residual);
bool metal_bias_cache_fetch(
    metal_context_t *ctx, const void *ptr, marmot_dtype_t dst_dtype, marmot_allocation_t *out_allocation,
    size_t *out_elements
);
void metal_bias_cache_store(
    metal_context_t *ctx, const void *ptr, marmot_dtype_t dst_dtype, const marmot_allocation_t *allocation,
    size_t elements
);
void metal_bias_cache_remove(metal_context_t *ctx, const void *ptr);
void metal_bias_cache_clear(metal_context_t *ctx);
void metal_routing_log_decision(
    metal_context_t *ctx, metal_routing_category_t category, const char *op_name, size_t problem_bytes, bool using_gpu,
    const char *reason
);
void metal_norm_context_build(metal_context_t *ctx);

void metal_profiling_set_label(metal_context_t *ctx, const char *label);
void metal_profiling_begin(metal_context_t *ctx);
void metal_profiling_end(metal_context_t *ctx);
void metal_profiling_commit(metal_context_t *ctx);
void metal_profiling_complete(metal_context_t *ctx);
void metal_profiling_reset(metal_context_t *ctx);

// Memory operations ----------------------------------------------------------

marmot_error_t metal_alloc(const void *device_ctx, size_t size, void **ptr);
void metal_free(const void *device_ctx, void *ptr);
marmot_error_t metal_memcpy_to_device(const void *device_ctx, void *dst, const void *src, size_t size);
marmot_error_t metal_memcpy_from_device(const void *device_ctx, void *dst, const void *src, size_t size);
id<MTLBuffer> metal_buffer_lookup(metal_context_t *ctx, void *ptr);
marmot_error_t metal_allocator_usage(const void *device_ctx, marmot_allocator_usage_t *usage);
void metal_allocator_pool_reclaim_deferred(metal_context_t *ctx);
marmot_error_t
metal_allocate_tracked(metal_context_t *ctx, size_t size, marmot_alloc_type_t type, marmot_allocation_t *out);

typedef struct {
    id<MTLBuffer> src;
    size_t src_offset;
    id<MTLBuffer> dst;
    size_t dst_offset;
    size_t size;
} metal_buffer_copy_region_t;

marmot_error_t metal_copy_regions(metal_context_t *ctx, const metal_buffer_copy_region_t *regions, size_t region_count);

// Tensor ops / math ----------------------------------------------------------

marmot_error_t metal_matmul_qkv_dense_f16_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_dense_f32_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_dense_bf16_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_separate_f16_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_separate_f32_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_separate_bf16_nt(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q4_0_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q4_0_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q4_1_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q4_1_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q5_0_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q5_0_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q5_1_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q5_1_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q8_0_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q8_0_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q8_1_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q8_1_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q2_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q2_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q3_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q3_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q4_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q4_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q5_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q5_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q6_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q6_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q8_k_f32(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_qkv_q8_k_f16(const void *device_ctx, const marmot_matmul_qkv_desc_t *desc);
marmot_error_t metal_matmul_quantized(
    const void *device_ctx, const marmot_tensor_t *weights, const marmot_tensor_t *activations,
    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out
);
void metal_matmul_quant_push_hints(uint32_t hints);
void metal_matmul_quant_pop_hints(uint32_t hints);
marmot_error_t metal_vec_dot(const void *device_ctx, const marmot_vec_dot_descriptor_t *desc, float *result);
marmot_error_t metal_layernorm_impl(const void *device_ctx, const marmot_layernorm_desc_t *desc);
marmot_error_t metal_rmsnorm_impl(const void *device_ctx, const marmot_rmsnorm_desc_t *desc);
marmot_error_t metal_rmsnorm_gemma_impl(const void *device_ctx, const marmot_rmsnorm_desc_t *desc);
marmot_error_t metal_rmsnorm(const void *device_ctx, const marmot_rmsnorm_desc_t *desc);
marmot_error_t metal_softmax_impl(const void *device_ctx, const marmot_softmax_desc_t *desc);
marmot_error_t metal_topk_impl(const void *device_ctx, const marmot_topk_desc_t *desc);
marmot_error_t metal_moe_experts_impl(const void *device_ctx, const marmot_moe_experts_desc_t *desc);
marmot_error_t metal_embedding_gather(const void *device_ctx, const marmot_embedding_gather_desc_t *desc);

marmot_error_t metal_binary_dispatch(
    const void *device_ctx, marmot_device_binary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
);
marmot_error_t metal_unary_dispatch(
    const void *device_ctx, marmot_device_unary_op_t op, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out
);

marmot_error_t metal_paged_attention_impl(const void *device_ctx, const marmot_paged_attention_desc_t *desc);
marmot_error_t metal_ternary_dispatch(
    const void *device_ctx, marmot_device_ternary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_tensor_t *c, marmot_tensor_t *out
);
marmot_error_t metal_elementwise_binary_impl(
    const void *device_ctx, marmot_device_binary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    marmot_tensor_t *out
);
marmot_error_t metal_elementwise_unary_impl(
    const void *device_ctx, marmot_device_unary_op_t op, const marmot_tensor_t *x,
    const marmot_activation_params_t *params, marmot_tensor_t *out
);
marmot_error_t metal_elementwise_ternary_impl(
    const void *device_ctx, marmot_device_ternary_op_t op, const marmot_tensor_t *a, const marmot_tensor_t *b,
    const marmot_tensor_t *c, marmot_tensor_t *out
);
marmot_error_t metal_reduction_sum_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_mean_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_max_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_min_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_variance_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_std_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_norm_l1_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_norm_l2_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_prod_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_argmax_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_argmin_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_any_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_all_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
);
marmot_error_t metal_reduction_dispatch(
    const void *device_ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input, marmot_tensor_t *out_values,
    marmot_tensor_t *out_indices, const marmot_reduction_params_t *params
);

marmot_error_t metal_reshape(
    const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *new_shape, size_t new_ndim
);
marmot_error_t metal_contiguous(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out);
marmot_error_t metal_view(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, size_t byte_offset);
marmot_error_t metal_transpose(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const int *perm);
marmot_error_t metal_concat(
    const void *device_ctx, const marmot_tensor_t *const *tensors, size_t num_tensors, marmot_tensor_t *out, int axis
);
marmot_error_t metal_slice(
    const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *starts, const size_t *sizes
);
marmot_error_t metal_gather_rows(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *out
);
marmot_error_t metal_scatter_u64_to_i32(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *out
);

marmot_error_t
metal_rope(const void *device_ctx, const marmot_tensor_t *x, const marmot_rope_params_t *params, marmot_tensor_t *out);

// Quantization ---------------------------------------------------------------

marmot_error_t metal_compute_quant_params(
    const void *device_ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size,
    marmot_quant_params_t *out_params
);
marmot_error_t metal_quantize_dispatch(
    const void *device_ctx, marmot_quant_kind_t kind, marmot_quant_layout_t layout, const marmot_tensor_t *input,
    const marmot_quant_params_t *quant_params, marmot_tensor_t *output
);
marmot_error_t metal_dequantize_dispatch(
    const void *device_ctx, marmot_quant_kind_t kind, marmot_quant_layout_t layout, const marmot_tensor_t *input,
    marmot_tensor_t *output
);

MARMOT_NODISCARD marmot_error_t metal_convert_dispatch(
    const void *device_ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src, size_t n
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __APPLE__
