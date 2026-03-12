#include "metal_backend_internal.h"

#ifdef __APPLE__

#include <stdint.h>

#include <math.h>
#include <string.h>
#include <sys/types.h>

#define METAL_REDUCTION_MAX_THREADGROUP_SIZE 256u

typedef struct {
    bool keepdims;
    size_t reduce_ndim;
    size_t reduce_axes[MARMOT_MAX_DIMS];
    size_t reduce_shape[MARMOT_MAX_DIMS];
    size_t reduce_linear_strides[MARMOT_MAX_DIMS];
    size_t reduce_total;
    size_t kept_ndim;
    size_t kept_axes[MARMOT_MAX_DIMS];
    size_t expected_out_ndim;
    size_t expected_out_shape[MARMOT_MAX_DIMS];
} metal_reduction_geometry_t;

typedef struct {
    float sum;
    float prod;
    float sum_abs;
    float sum_sq;
    float mean;
    float m2;
    float max_value;
    float min_value;
    uint64_t max_index;
    uint64_t min_index;
    uint32_t have_max;
    uint32_t have_min;
    uint32_t any_flag;
    uint32_t all_flag;
    uint64_t count;
} metal_reduction_partial_t;

static const size_t METAL_REDUCTION_PARTIAL_SIZE = sizeof(metal_reduction_partial_t);

static inline void metal_reduction_sort_axes(size_t *axes, size_t count) {
    for (size_t i = 1; i < count; ++i) {
        size_t key = axes[i];
        size_t j = i;
        while (j > 0 && axes[j - 1] > key) {
            axes[j] = axes[j - 1];
            --j;
        }
        axes[j] = key;
    }
}

static marmot_error_t metal_prepare_reduction_geometry(
    const marmot_tensor_t *input, const marmot_reduction_params_t *params, const marmot_tensor_t *out_values,
    const marmot_tensor_t *out_indices, bool require_indices, metal_reduction_geometry_t *geo
) {
    if (input == nullptr || out_values == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor pointer in reduction setup");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t ndim = input->shape.ndim;
    if (ndim == 0 || ndim > MARMOT_MAX_DIMS) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unsupported tensor rank for reduction");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    bool reduce_mask[MARMOT_MAX_DIMS] = {false};
    size_t reduced_axes[MARMOT_MAX_DIMS];
    size_t reduce_ndim = 0;

    geo->keepdims = params != nullptr && params->keepdims;

    if (params == nullptr || params->axes == nullptr || params->num_axes == 0) {
        for (size_t axis = 0; axis < ndim; ++axis) {
            reduce_mask[axis] = true;
            reduced_axes[reduce_ndim++] = axis;
        }
    } else {
        for (size_t i = 0; i < params->num_axes; ++i) {
            int32_t axis = params->axes[i];
            if (axis < 0) {
                axis += (int32_t)ndim;
            }
            if (axis < 0 || axis >= (int32_t)ndim) {
                marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Reduction axis out of range");
                return MARMOT_ERROR_INVALID_ARGUMENT;
            }
            size_t axis_u = (size_t)axis;
            if (!reduce_mask[axis_u]) {
                reduce_mask[axis_u] = true;
                reduced_axes[reduce_ndim++] = axis_u;
            }
        }
    }

    metal_reduction_sort_axes(reduced_axes, reduce_ndim);

    geo->reduce_ndim = reduce_ndim;
    geo->reduce_total = 1;
    for (size_t i = 0; i < reduce_ndim; ++i) {
        size_t axis = reduced_axes[i];
        geo->reduce_axes[i] = axis;
        geo->reduce_shape[i] = input->shape.shape[axis];
        geo->reduce_total *= input->shape.shape[axis];
    }

    size_t stride = 1;
    for (ssize_t i = (ssize_t)reduce_ndim - 1; i >= 0; --i) {
        geo->reduce_linear_strides[i] = stride;
        stride *= geo->reduce_shape[i];
    }

    geo->kept_ndim = 0;
    for (size_t axis = 0; axis < ndim; ++axis) {
        if (!reduce_mask[axis]) {
            geo->kept_axes[geo->kept_ndim++] = axis;
        }
    }

    if (geo->keepdims) {
        geo->expected_out_ndim = ndim;
        for (size_t axis = 0; axis < ndim; ++axis) {
            geo->expected_out_shape[axis] = reduce_mask[axis] ? 1 : input->shape.shape[axis];
        }
    } else if (geo->kept_ndim > 0) {
        geo->expected_out_ndim = geo->kept_ndim;
        for (size_t i = 0; i < geo->kept_ndim; ++i) {
            geo->expected_out_shape[i] = input->shape.shape[geo->kept_axes[i]];
        }
    } else {
        geo->expected_out_ndim = 1;
        geo->expected_out_shape[0] = 1;
    }

    if (out_values->shape.ndim != geo->expected_out_ndim) {
        marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Reduction output rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    for (size_t axis = 0; axis < geo->expected_out_ndim; ++axis) {
        if (out_values->shape.shape[axis] != geo->expected_out_shape[axis]) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Reduction output shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    if (require_indices) {
        if (out_indices == nullptr) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Arg reduction requires indices tensor");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        if (out_indices->dtype != MARMOT_DTYPE_UINT64) {
            marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Arg reduction indices must be UINT64");
            return MARMOT_ERROR_UNSUPPORTED_DTYPE;
        }
        if (out_indices->shape.ndim != geo->expected_out_ndim) {
            marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Arg reduction indices rank mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
        for (size_t axis = 0; axis < geo->expected_out_ndim; ++axis) {
            if (out_indices->shape.shape[axis] != geo->expected_out_shape[axis]) {
                marmot_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Arg reduction indices shape mismatch");
                return MARMOT_ERROR_DIMENSION_MISMATCH;
            }
        }
    }

    return MARMOT_SUCCESS;
}

typedef struct {
    uint32_t op;
    uint32_t has_indices;
    uint32_t keepdims;
    uint32_t unbiased;
    uint32_t input_ndim;
    uint32_t out_ndim;
    uint32_t reduce_ndim;
    uint32_t kept_ndim;
    uint32_t indices_ndim;
    uint32_t threads_per_group;
    uint64_t out_elements;
    uint64_t reduce_total;
    float epsilon;
    float pad[3];
    uint64_t input_shape[MARMOT_MAX_DIMS];
    uint64_t input_strides[MARMOT_MAX_DIMS];
    uint64_t out_shape[MARMOT_MAX_DIMS];
    uint64_t out_strides[MARMOT_MAX_DIMS];
    uint64_t indices_strides[MARMOT_MAX_DIMS];
    uint64_t reduce_axes[MARMOT_MAX_DIMS];
    uint64_t reduce_shape[MARMOT_MAX_DIMS];
    uint64_t reduce_linear_strides[MARMOT_MAX_DIMS];
    uint64_t kept_axes[MARMOT_MAX_DIMS];
    uint32_t chunks_per_output;
    uint32_t elements_per_chunk;
    uint32_t stage_reserved0;
    uint32_t stage_reserved1;
} metal_reduction_uniforms_t;

static const char *metal_reduction_stage1_kernel_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "reduction_stage1_f32";
    case MARMOT_DTYPE_FLOAT16:
        return "reduction_stage1_f16";
    case MARMOT_DTYPE_BFLOAT16:
        return "reduction_stage1_bf16";
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return "reduction_stage1_fp8_e4m3";
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return "reduction_stage1_fp8_e5m2";
#endif
    case MARMOT_DTYPE_INT32:
        return "reduction_stage1_i32";
    case MARMOT_DTYPE_INT16:
        return "reduction_stage1_i16";
    case MARMOT_DTYPE_INT8:
        return "reduction_stage1_i8";
    case MARMOT_DTYPE_UINT8:
        return "reduction_stage1_u8";
    case MARMOT_DTYPE_UINT16:
        return "reduction_stage1_u16";
    case MARMOT_DTYPE_UINT32:
        return "reduction_stage1_u32";
    case MARMOT_DTYPE_UINT64:
        return "reduction_stage1_u64";
    default:
        return nullptr;
    }
}

static const char *metal_reduction_stage2_kernel_name(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT32:
        return "reduction_stage2_f32";
    case MARMOT_DTYPE_FLOAT16:
        return "reduction_stage2_f16";
    case MARMOT_DTYPE_BFLOAT16:
        return "reduction_stage2_bf16";
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
        return "reduction_stage2_fp8_e4m3";
    case MARMOT_DTYPE_FLOAT8_E5M2:
        return "reduction_stage2_fp8_e5m2";
#endif
    case MARMOT_DTYPE_INT32:
        return "reduction_stage2_i32";
    case MARMOT_DTYPE_INT16:
        return "reduction_stage2_i16";
    case MARMOT_DTYPE_INT8:
        return "reduction_stage2_i8";
    case MARMOT_DTYPE_UINT8:
        return "reduction_stage2_u8";
    case MARMOT_DTYPE_UINT16:
        return "reduction_stage2_u16";
    case MARMOT_DTYPE_UINT32:
        return "reduction_stage2_u32";
    case MARMOT_DTYPE_UINT64:
        return "reduction_stage2_u64";
    default:
        return nullptr;
    }
}

static bool metal_reduction_requires_indices(marmot_device_reduction_op_t op) {
    return op == MARMOT_DEVICE_REDUCTION_ARGMAX || op == MARMOT_DEVICE_REDUCTION_ARGMIN;
}

static marmot_error_t metal_reduction_run(
    metal_context_t *ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input, marmot_tensor_t *out_values,
    marmot_tensor_t *out_indices, const marmot_reduction_params_t *params
) {
    if (ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null Metal context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input == nullptr || out_values == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor for Metal reduction");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!marmot_dtype_supports_reduction(input->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Input dtype not supported for reductions");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (!marmot_dtype_supports_reduction(out_values->dtype)) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Output dtype not supported for reductions");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const bool require_indices = metal_reduction_requires_indices(op);
    metal_reduction_geometry_t geo;
    marmot_error_t status =
        metal_prepare_reduction_geometry(input, params, out_values, out_indices, require_indices, &geo);
    if (status != MARMOT_SUCCESS) {
        return status;
    }

    if (geo.reduce_total == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Reduction over zero elements is undefined");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    const size_t problem_bytes = marmot_tensor_size_bytes(input);

    const char *stage1_kernel = metal_reduction_stage1_kernel_name(input->dtype);
    const char *stage2_kernel = metal_reduction_stage2_kernel_name(out_values->dtype);
    if (stage1_kernel == nullptr || stage2_kernel == nullptr) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Metal reduction kernel not available for dtype");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_REDUCTION, "reduction", problem_bytes, true, "gpu");

    metal_reduction_uniforms_t base_uniforms;
    memset(&base_uniforms, 0, sizeof(base_uniforms));
    base_uniforms.op = (uint32_t)op;
    base_uniforms.has_indices = require_indices ? 1u : 0u;
    base_uniforms.keepdims = geo.keepdims ? 1u : 0u;
    base_uniforms.unbiased = (params != nullptr && params->unbiased) ? 1u : 0u;
    base_uniforms.input_ndim = (uint32_t)input->shape.ndim;
    base_uniforms.out_ndim = (uint32_t)out_values->shape.ndim;
    base_uniforms.reduce_ndim = (uint32_t)geo.reduce_ndim;
    base_uniforms.kept_ndim = (uint32_t)geo.kept_ndim;
    base_uniforms.indices_ndim = out_indices != nullptr ? (uint32_t)out_indices->shape.ndim : 0u;
    base_uniforms.out_elements = marmot_tensor_num_elements(out_values);
    base_uniforms.reduce_total = geo.reduce_total;
    base_uniforms.epsilon = params != nullptr ? params->epsilon : 0.0f;
    base_uniforms.chunks_per_output = 0u;
    base_uniforms.elements_per_chunk = 0u;
    base_uniforms.stage_reserved0 = 0u;
    base_uniforms.stage_reserved1 = 0u;

    if (base_uniforms.out_elements == 0) {
        return MARMOT_SUCCESS;
    }
    if (base_uniforms.out_elements > (uint64_t)UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Metal reduction output size exceeds GPU grid limits");
        return MARMOT_ERROR_INVALID_OPERATION;
    }

    for (size_t i = 0; i < input->shape.ndim; ++i) {
        base_uniforms.input_shape[i] = input->shape.shape[i];
        base_uniforms.input_strides[i] = input->shape.strides[i];
    }
    for (size_t i = 0; i < out_values->shape.ndim; ++i) {
        base_uniforms.out_shape[i] = out_values->shape.shape[i];
        base_uniforms.out_strides[i] = out_values->shape.strides[i];
    }
    if (out_indices != nullptr) {
        for (size_t i = 0; i < out_indices->shape.ndim; ++i) {
            base_uniforms.indices_strides[i] = out_indices->shape.strides[i];
        }
    }
    for (size_t i = 0; i < geo.reduce_ndim; ++i) {
        base_uniforms.reduce_axes[i] = geo.reduce_axes[i];
        base_uniforms.reduce_shape[i] = geo.reduce_shape[i];
        base_uniforms.reduce_linear_strides[i] = geo.reduce_linear_strides[i];
    }
    for (size_t i = 0; i < geo.kept_ndim; ++i) {
        base_uniforms.kept_axes[i] = geo.kept_axes[i];
    }

    size_t input_bytes = marmot_dtype_size(input->dtype) * marmot_tensor_num_elements(input);
    size_t out_bytes = marmot_dtype_size(out_values->dtype) * base_uniforms.out_elements;
    size_t indices_bytes = require_indices ? sizeof(uint64_t) * base_uniforms.out_elements : 0;
    static constexpr size_t kSharedIndicesMaxBytes = 4096;

    id<MTLBuffer> buffer_input = metal_residency_acquire_existing(ctx, input, input->dtype);
    if (buffer_input == nil) {
        buffer_input = metal_residency_acquire_compute(ctx, input, input->dtype, nullptr);
    }
    if (buffer_input == nil) {
        buffer_input = metal_buffer_acquire(ctx, input->data, input_bytes);
    }

    bool out_values_private = false;
    bool out_indices_private = false;
    bool out_values_is_new = false;
    bool out_indices_is_new = false;
    bool out_indices_shared = false;

    id<MTLBuffer> buffer_output =
        metal_residency_acquire_compute(ctx, out_values, out_values->dtype, &out_values_is_new);
    if (buffer_output == nil) {
        buffer_output = metal_buffer_acquire(ctx, out_values->data, out_bytes);
    } else {
        out_values_private = true;
    }

    id<MTLBuffer> buffer_indices = nil;
    id<MTLBuffer> buffer_partials = nil;
    if (require_indices) {
        if (indices_bytes > 0 && indices_bytes <= kSharedIndicesMaxBytes) {
            buffer_indices = metal_buffer_acquire(ctx, out_indices->data, indices_bytes);
            out_indices_shared = buffer_indices != nil;
        }
        if (buffer_indices == nil) {
            buffer_indices = metal_residency_acquire_compute(ctx, out_indices, out_indices->dtype, &out_indices_is_new);
            if (buffer_indices != nil) {
                out_indices_private = true;
            }
        }
        if (buffer_indices == nil) {
            buffer_indices = metal_buffer_acquire(ctx, out_indices->data, indices_bytes);
            out_indices_shared = buffer_indices != nil;
        }
    }

    if (buffer_input == nil || buffer_output == nil || (require_indices && buffer_indices == nil)) {
        if (buffer_input != nil)
            [buffer_input release];
        if (buffer_output != nil)
            [buffer_output release];
        if (buffer_indices != nil)
            [buffer_indices release];
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLComputePipelineState> stage1_pipeline = metal_pipeline_get(ctx, stage1_kernel);
    id<MTLComputePipelineState> stage2_pipeline = metal_pipeline_get(ctx, stage2_kernel);
    if (stage1_pipeline == nil || stage2_pipeline == nil) {
        if (stage1_pipeline != nil)
            [stage1_pipeline release];
        if (stage2_pipeline != nil)
            [stage2_pipeline release];
        [buffer_input release];
        [buffer_output release];
        if (buffer_indices != nil)
            [buffer_indices release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Failed to create Metal reduction pipelines");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    marmot_error_t result = MARMOT_SUCCESS;
    bool commands_active = false;

    size_t partial_bytes = 0;
    size_t partial_bytes_aligned = 0;
    __uint128_t partial_bytes128 = 0;
    metal_reduction_uniforms_t stage1_uniforms;
    metal_reduction_uniforms_t stage2_uniforms;
    NSUInteger stage1_groups = 0;
    NSUInteger stage2_groups = 0;
    MTLSize stage1_threadgroups = MTLSizeMake(1, 1, 1);
    MTLSize stage1_threads_per_group = MTLSizeMake(1, 1, 1);
    MTLSize stage2_threadgroups = MTLSizeMake(1, 1, 1);
    MTLSize stage2_threads_per_group = MTLSizeMake(1, 1, 1);
    id<MTLComputeCommandEncoder> encoder = nil;

    NSUInteger stage1_threads = 1;
    NSUInteger stage2_threads = 1;

    stage1_threads = metal_threadgroup_size_1d(stage1_pipeline, METAL_REDUCTION_MAX_THREADGROUP_SIZE);
    if (stage1_threads == 0) {
        stage1_threads = 1;
    }
    uint32_t stage1_threads_u32 = (uint32_t)stage1_threads;
    if (stage1_threads_u32 == 0) {
        stage1_threads_u32 = 1;
    }

    uint64_t reduce_total = base_uniforms.reduce_total;
    uint64_t chunk_target = (uint64_t)stage1_threads_u32 * 8u;
    if (chunk_target == 0) {
        chunk_target = (uint64_t)stage1_threads_u32;
    }
    if (chunk_target == 0) {
        chunk_target = 1ull;
    }

    uint64_t chunks_per_output64 = (reduce_total + chunk_target - 1ull) / chunk_target;
    if (chunks_per_output64 == 0ull) {
        chunks_per_output64 = 1ull;
    }
    if (chunks_per_output64 > UINT32_MAX) {
        [stage1_pipeline release];
        [stage2_pipeline release];
        [buffer_input release];
        [buffer_output release];
        if (buffer_indices != nil)
            [buffer_indices release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Reduction requires too many partial chunks");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    uint32_t chunks_per_output = (uint32_t)chunks_per_output64;
    uint64_t elements_per_chunk64 = (reduce_total + chunks_per_output64 - 1ull) / chunks_per_output64;
    if (elements_per_chunk64 == 0ull) {
        elements_per_chunk64 = 1ull;
    }
    if (elements_per_chunk64 > UINT32_MAX) {
        [stage1_pipeline release];
        [stage2_pipeline release];
        [buffer_input release];
        [buffer_output release];
        if (buffer_indices != nil)
            [buffer_indices release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Reduction chunk size exceeds limits");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    uint32_t elements_per_chunk = (uint32_t)elements_per_chunk64;
    if (elements_per_chunk == 0u) {
        elements_per_chunk = stage1_threads_u32;
    }

    if (chunks_per_output == 0u) {
        chunks_per_output = 1u;
    }

    __uint128_t total_partials128 = (__uint128_t)base_uniforms.out_elements * (__uint128_t)chunks_per_output;
    if (total_partials128 == 0) {
        [stage1_pipeline release];
        [stage2_pipeline release];
        [buffer_input release];
        [buffer_output release];
        if (buffer_indices != nil)
            [buffer_indices release];
        return MARMOT_SUCCESS;
    }

    uint64_t total_partials = (uint64_t)total_partials128;
    if (total_partials > (uint64_t)UINT32_MAX) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "Metal reduction partial grid exceeds GPU limits");
        result = MARMOT_ERROR_INVALID_OPERATION;
        goto cleanup;
    }
    partial_bytes128 = total_partials128 * (__uint128_t)METAL_REDUCTION_PARTIAL_SIZE;
    if (partial_bytes128 > SIZE_MAX) {
        [stage1_pipeline release];
        [stage2_pipeline release];
        [buffer_input release];
        [buffer_output release];
        if (buffer_indices != nil)
            [buffer_indices release];
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Reduction partial buffer exceeds host limits");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    partial_bytes = (size_t)partial_bytes128;
    partial_bytes_aligned = metal_round_up(partial_bytes, 256);
    if (partial_bytes_aligned == 0) {
        partial_bytes_aligned = 256;
    }

    if (ctx->reduction_partials_buffer != nil && ctx->reduction_partials_capacity < partial_bytes_aligned) {
        [ctx->reduction_partials_buffer release];
        ctx->reduction_partials_buffer = nil;
        ctx->reduction_partials_capacity = 0;
    }
    if (ctx->reduction_partials_buffer == nil) {
        id<MTLBuffer> new_buffer = [ctx->device newBufferWithLength:partial_bytes_aligned
                                                            options:MTLResourceStorageModePrivate];
        if (new_buffer == nil) {
            [stage1_pipeline release];
            [stage2_pipeline release];
            [buffer_input release];
            [buffer_output release];
            if (buffer_indices != nil)
                [buffer_indices release];
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Failed to allocate Metal reduction partial buffer");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        ctx->reduction_partials_buffer = new_buffer;
        ctx->reduction_partials_capacity = partial_bytes_aligned;
    }

    buffer_partials = [ctx->reduction_partials_buffer retain];
    if (buffer_partials == nil) {
        [stage1_pipeline release];
        [stage2_pipeline release];
        [buffer_input release];
        [buffer_output release];
        if (buffer_indices != nil)
            [buffer_indices release];
        marmot_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Failed to acquire reduction partial buffer");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    stage1_uniforms = base_uniforms;
    stage1_uniforms.threads_per_group = stage1_threads_u32;
    stage1_uniforms.chunks_per_output = chunks_per_output;
    stage1_uniforms.elements_per_chunk = elements_per_chunk;

    stage2_threads = metal_threadgroup_size_1d(stage2_pipeline, METAL_REDUCTION_MAX_THREADGROUP_SIZE);
    if (stage2_threads == 0) {
        stage2_threads = 1;
    }
    if (stage2_threads > chunks_per_output) {
        stage2_threads = chunks_per_output;
    }
    if (stage2_threads == 0) {
        stage2_threads = 1;
    }

    stage2_uniforms = base_uniforms;
    stage2_uniforms.threads_per_group = (uint32_t)stage2_threads;
    stage2_uniforms.chunks_per_output = chunks_per_output;
    stage2_uniforms.elements_per_chunk = elements_per_chunk;

    metal_profiling_set_label(ctx, "reduction");
    metal_profiling_begin(ctx);

    encoder = metal_command_acquire_compute_encoder(ctx, stage1_pipeline);
    if (encoder == nil) {
        result = MARMOT_ERROR_BACKEND_INIT_FAILED;
        goto cleanup;
    }
    commands_active = true;

    [encoder setBuffer:buffer_input offset:0 atIndex:1];
    [encoder setBuffer:buffer_partials offset:0 atIndex:2];
    [encoder setBytes:&stage1_uniforms length:sizeof(stage1_uniforms) atIndex:0];

    stage1_groups = (NSUInteger)total_partials;
    if (stage1_groups == 0) {
        stage1_groups = 1;
    }
    stage1_threadgroups = MTLSizeMake(stage1_groups, 1, 1);
    stage1_threads_per_group = MTLSizeMake(stage1_threads, 1, 1);
    [encoder dispatchThreadgroups:stage1_threadgroups threadsPerThreadgroup:stage1_threads_per_group];

    [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

    encoder = metal_command_acquire_compute_encoder(ctx, stage2_pipeline);
    if (encoder == nil) {
        result = MARMOT_ERROR_BACKEND_INIT_FAILED;
        goto cleanup;
    }

    [encoder setBuffer:buffer_partials offset:0 atIndex:1];
    [encoder setBuffer:buffer_output offset:0 atIndex:2];
    if (require_indices) {
        [encoder setBuffer:buffer_indices offset:0 atIndex:3];
    } else {
        [encoder setBuffer:metal_get_dummy_buffer(ctx) offset:0 atIndex:3];
    }
    [encoder setBytes:&stage2_uniforms length:sizeof(stage2_uniforms) atIndex:0];

    stage2_groups = (NSUInteger)base_uniforms.out_elements;
    if (stage2_groups == 0) {
        stage2_groups = 1;
    }
    stage2_threadgroups = MTLSizeMake(stage2_groups, 1, 1);
    stage2_threads_per_group = MTLSizeMake(stage2_threads, 1, 1);
    [encoder dispatchThreadgroups:stage2_threadgroups threadsPerThreadgroup:stage2_threads_per_group];

    metal_profiling_end(ctx);

    if (out_indices_shared && out_indices != nullptr) {
        metal_command_stream_track_shared_write(ctx, out_indices->data);
    }

    metal_command_stream_flush(ctx, false);
    commands_active = false;

    if (out_values_private) {
        metal_residency_mark_dirty(ctx, out_values, out_values->dtype);
    }
    if (out_indices != nullptr) {
        if (out_indices_private) {
            metal_residency_mark_dirty(ctx, out_indices, out_indices->dtype);
        } else if (out_indices_shared) {
            // GPU wrote to the shared buffer — mark shared as authoritative so any
            // stale private buffer is re-uploaded on next acquire_compute.
            metal_residency_mark_shared_write(ctx, out_indices->data);
            ((marmot_tensor_t *)out_indices)->memory_location = MARMOT_MEMORY_DEVICE;
            ((marmot_tensor_t *)out_indices)->needs_sync = true;
        }
    }

cleanup:
    if (commands_active) {
        metal_command_stream_discard(ctx);
    }

    if (stage1_pipeline != nil) {
        [stage1_pipeline release];
    }
    if (stage2_pipeline != nil) {
        [stage2_pipeline release];
    }

    if (buffer_input != nil) {
        [buffer_input release];
    }
    if (buffer_output != nil) {
        [buffer_output release];
    }
    if (buffer_indices != nil) {
        [buffer_indices release];
    }
    if (buffer_partials != nil) {
        [buffer_partials release];
    }

    return result;
}

marmot_error_t metal_reduction_sum_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_SUM, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_mean_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_MEAN, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_max_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_MAX, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_min_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_MIN, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_variance_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_VARIANCE, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_std_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_STD, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_norm_l1_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_NORM_L1, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_norm_l2_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_NORM_L2, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_prod_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_PROD, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_argmax_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_ARGMAX, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_argmin_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_ARGMIN, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_any_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_ANY, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_all_impl(
    const void *device_ctx, const marmot_tensor_t *input, marmot_tensor_t *out_values, marmot_tensor_t *out_indices,
    const marmot_reduction_params_t *params
) {
    metal_context_t *ctx = (metal_context_t *)device_ctx;
    return metal_reduction_run(ctx, MARMOT_DEVICE_REDUCTION_ALL, input, out_values, out_indices, params);
}

marmot_error_t metal_reduction_dispatch(
    const void *device_ctx, marmot_device_reduction_op_t op, const marmot_tensor_t *input, marmot_tensor_t *out_values,
    marmot_tensor_t *out_indices, const marmot_reduction_params_t *params
) {
    if (device_ctx == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null Metal context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    switch (op) {
    case MARMOT_DEVICE_REDUCTION_SUM:
        return metal_reduction_sum_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_MEAN:
        return metal_reduction_mean_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_MAX:
        return metal_reduction_max_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_MIN:
        return metal_reduction_min_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_VARIANCE:
        return metal_reduction_variance_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_STD:
        return metal_reduction_std_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_NORM_L1:
        return metal_reduction_norm_l1_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_NORM_L2:
        return metal_reduction_norm_l2_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_PROD:
        return metal_reduction_prod_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_ARGMAX:
        return metal_reduction_argmax_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_ARGMIN:
        return metal_reduction_argmin_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_ANY:
        return metal_reduction_any_impl(device_ctx, input, out_values, out_indices, params);
    case MARMOT_DEVICE_REDUCTION_ALL:
        return metal_reduction_all_impl(device_ctx, input, out_values, out_indices, params);
    default:
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Reduction operation not implemented on Metal");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
}

#endif // __APPLE__
