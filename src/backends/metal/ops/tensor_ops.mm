#include "metal_backend_internal.h"

#ifdef __APPLE__

#include <stdint.h>
#include <stdlib.h>

#include <string.h>

#include "internal/stride_helpers.h"

typedef struct {
    uint32_t ndim;
    uint32_t elem_size;
    uint32_t total;
    uint32_t dst_divisors[8];
    uint32_t src_strides_perm[8];
} metal_transpose_params_t;

typedef struct {
    uint32_t ndim;
    uint32_t dtype_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t total_elements;
    uint64_t src_strides[MARMOT_MAX_DIMS];
    uint64_t dst_strides[MARMOT_MAX_DIMS];
    uint64_t starts[MARMOT_MAX_DIMS];
    uint64_t out_shape[MARMOT_MAX_DIMS];
} metal_copy_uniforms_t;

static inline void metal_set_error(marmot_error_t err, const char *msg) {
    marmot_set_error(err, msg);
}

static marmot_error_t metal_tensor_simple_copy(
    metal_context_t *ctx, id<MTLBuffer> bufferSrc, id<MTLBuffer> bufferDst, size_t bytes, const marmot_tensor_t *out,
    bool out_private
) {
    if (bytes == 0) {
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        return MARMOT_SUCCESS;
    }

    id<MTLBlitCommandEncoder> blit = metal_command_acquire_blit_encoder(ctx);
    if (blit == nil) {
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal: unable to acquire blit encoder");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    [blit copyFromBuffer:bufferSrc sourceOffset:0 toBuffer:bufferDst destinationOffset:0 size:bytes];
    metal_command_stream_flush(ctx, false);

    if (out_private && out != nullptr) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }

    if (bufferSrc != nil) {
        [bufferSrc release];
    }
    if (bufferDst != nil) {
        [bufferDst release];
    }
    return MARMOT_SUCCESS;
}

marmot_error_t metal_reshape(
    const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *new_shape, size_t new_ndim
) {
    (void)device_ctx;

    if (x == nullptr || out == nullptr || new_shape == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in reshape");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (x->shape.ndim == 0 || new_ndim == 0) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid ndim in reshape");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t old_elements = marmot_tensor_num_elements(x);
    size_t new_elements = 1;
    for (size_t i = 0; i < new_ndim; ++i) {
        new_elements *= new_shape[i];
    }

    if (old_elements != new_elements) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Reshape element count mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (out->owns_data && out->data != nullptr) {
        if (ctx != nullptr) {
            metal_residency_invalidate(ctx, out->data);
        }
        free(out->data);
    }
    if (out->packed_data != nullptr) {
        free(out->packed_data);
    }
    if (out->quant_params != nullptr && out->quant_params != x->quant_params) {
        free(out->quant_params);
    }

    out->shape.ndim = (uint32_t)new_ndim;
    for (size_t i = 0; i < new_ndim; ++i) {
        out->shape.shape[i] = new_shape[i];
    }

    bool has_stride = false;
    for (size_t i = 0; i < new_ndim; ++i) {
        if (out->shape.strides[i] != 0) {
            has_stride = true;
            break;
        }
    }
    if (!has_stride) {
        out->shape.strides[new_ndim - 1] = 1;
        for (size_t i = new_ndim - 1; i-- > 0;) {
            out->shape.strides[i] = out->shape.strides[i + 1] * out->shape.shape[i + 1];
        }
    }

    out->dtype = x->dtype;
    out->data = x->data;
    out->capacity_bytes = x->capacity_bytes;
    out->owns_data = false;
    out->quant_params = x->quant_params;
    out->quant_kind = x->quant_kind;
    out->quant_layout = x->quant_layout;
    out->backend = x->backend;
    out->memory_location = x->memory_location;
    out->needs_sync = x->needs_sync;
    out->packed_data = nullptr;
    out->packed_src_data = nullptr;
    out->packed_bytes = 0;
    out->packed_row_bytes = 0;
    out->packed_rows = 0;
    return MARMOT_SUCCESS;
}

marmot_error_t metal_view(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, size_t byte_offset) {
    if (x == nullptr || out == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in view");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->dtype != out->dtype) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires matching dtypes");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->backend != out->backend) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires matching backends");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const size_t elem_size = marmot_dtype_size(x->dtype);
    if (elem_size != 0 && (byte_offset % elem_size) != 0) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View byte offset must align to dtype size");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    marmot_tensor_t out_probe = *out;
    out_probe.quant_kind = x->quant_kind;
    out_probe.quant_layout = x->quant_layout;
    size_t out_bytes = marmot_tensor_size_bytes(&out_probe);
    if (x->capacity_bytes < byte_offset) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View byte offset exceeds input capacity");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (out_bytes > x->capacity_bytes - byte_offset) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "View exceeds input capacity");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (out_bytes != 0 && x->data == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "View requires non-null input data");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (out->owns_data && out->data != nullptr) {
        if (ctx != nullptr) {
            metal_residency_invalidate(ctx, out->data);
        }
        free(out->data);
    }
    if (out->packed_data != nullptr) {
        free(out->packed_data);
    }
    if (out->quant_params != nullptr && out->quant_params != x->quant_params) {
        free(out->quant_params);
    }

    out->data = static_cast<uint8_t *>(x->data) + byte_offset;
    out->capacity_bytes = x->capacity_bytes > byte_offset ? x->capacity_bytes - byte_offset : 0;
    out->owns_data = false;
    out->quant_params = x->quant_params;
    out->quant_kind = x->quant_kind;
    out->quant_layout = x->quant_layout;
    out->backend = x->backend;
    out->memory_location = x->memory_location;
    out->needs_sync = x->needs_sync;
    out->packed_data = nullptr;
    out->packed_src_data = nullptr;
    out->packed_bytes = 0;
    out->packed_row_bytes = 0;
    out->packed_rows = 0;
    return MARMOT_SUCCESS;
}

marmot_error_t metal_contiguous(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out) {
    if (x == nullptr || out == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in contiguous");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (x->dtype != out->dtype) {
        metal_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Contiguous requires matching dtypes");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (x->shape.ndim != out->shape.ndim) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Contiguous rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    for (size_t i = 0; i < x->shape.ndim; ++i) {
        if (x->shape.shape[i] != out->shape.shape[i]) {
            metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Contiguous shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    const size_t total_elements = marmot_tensor_num_elements(out);
    if (total_elements == 0) {
        return MARMOT_SUCCESS;
    }
    const size_t elem_size = marmot_dtype_size(out->dtype);
    if (elem_size == 0) {
        metal_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "Contiguous dtype unsupported");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (x->data == nullptr || out->data == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Contiguous requires non-null tensor data");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const bool src_contig = marmot::metal::is_contiguous(x);
    const bool dst_contig = marmot::metal::is_contiguous(out);
    const size_t src_bytes = marmot::metal::tensor_span_bytes(x);
    const size_t dst_bytes = marmot::metal::tensor_span_bytes(out);
    if (src_bytes == 0 || dst_bytes == 0) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Contiguous tensor span invalid");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_TENSOR, "contiguous", dst_bytes, true, "gpu");

    metal_tensor_buffer_t viewSrc = metal_buffer_acquire_view(ctx, x, x->dtype, src_bytes);
    id<MTLBuffer> bufferSrc = viewSrc.buffer;
    const size_t offsetSrc = viewSrc.offset;

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferDst = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferDst == nil) {
        bufferDst = metal_buffer_acquire(ctx, out->data, dst_bytes);
    } else {
        out_private = true;
    }

    if (bufferSrc == nil || bufferDst == nil) {
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal contiguous: unable to acquire buffers");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    if (src_contig && dst_contig && offsetSrc == 0) {
        if (x->data == out->data) {
            [bufferSrc release];
            [bufferDst release];
            return MARMOT_SUCCESS;
        }
        return metal_tensor_simple_copy(ctx, bufferSrc, bufferDst, dst_bytes, out, out_private);
    }

    if (x->shape.ndim > MARMOT_MAX_DIMS) {
        [bufferSrc release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Contiguous supports up to 8 dimensions");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    metal_copy_uniforms_t params = {
        .ndim = (uint32_t)x->shape.ndim,
        .dtype_bytes = (uint32_t)elem_size,
        .reserved0 = 0,
        .reserved1 = 0,
        .total_elements = (uint64_t)total_elements,
        .src_strides = {0},
        .dst_strides = {0},
        .starts = {0},
        .out_shape = {0},
    };
    for (size_t i = 0; i < x->shape.ndim; ++i) {
        params.src_strides[i] = (uint64_t)x->shape.strides[i];
        params.dst_strides[i] = (uint64_t)out->shape.strides[i];
        params.out_shape[i] = (uint64_t)out->shape.shape[i];
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, "tensor_slice_generic");
    if (pipeline == nil) {
        [bufferSrc release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal contiguous: missing pipeline");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferSrc release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal contiguous: unable to acquire encoder");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    [encoder setBytes:&params length:sizeof(params) atIndex:0];
    [encoder setBuffer:bufferSrc offset:offsetSrc atIndex:1];
    [encoder setBuffer:bufferDst offset:0 atIndex:2];

    NSUInteger total_u = (NSUInteger)total_elements;
    MTLSize tpg = metal_threads_for_elements(pipeline, total_u, 512);
    [encoder dispatchThreads:MTLSizeMake(total_u, 1, 1) threadsPerThreadgroup:tpg];

    metal_command_stream_flush(ctx, false);
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }

    [pipeline release];
    [bufferSrc release];
    [bufferDst release];
    return MARMOT_SUCCESS;
}

marmot_error_t
metal_transpose(const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const int *perm) {
    if (x == nullptr || out == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null pointer in transpose");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t ndim = x->shape.ndim;
    if (ndim != out->shape.ndim) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Transpose rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    if (ndim == 0) {
        return MARMOT_SUCCESS;
    }

    int perm_buffer[8];
    if (ndim > 8) {
        metal_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Transpose supports up to 8 dimensions");
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }

    if (perm != nullptr) {
        for (size_t i = 0; i < ndim; ++i) {
            perm_buffer[i] = perm[i];
        }
    } else {
        for (size_t i = 0; i < ndim; ++i) {
            perm_buffer[i] = (int)(ndim - 1 - i);
        }
    }

    bool seen[8] = {false};
    for (size_t i = 0; i < ndim; ++i) {
        int axis = perm_buffer[i];
        if (axis < 0 || axis >= (int)ndim || seen[axis]) {
            metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid permutation for transpose");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        seen[axis] = true;
        if (out->shape.shape[i] != x->shape.shape[axis]) {
            metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Transpose output shape mismatch");
            return MARMOT_ERROR_DIMENSION_MISMATCH;
        }
    }

    bool is_identity = true;
    for (size_t i = 0; i < ndim; ++i) {
        if (perm_buffer[i] != (int)i) {
            is_identity = false;
            break;
        }
    }

    size_t elem_size = marmot_dtype_size(x->dtype);
    size_t total = marmot_tensor_num_elements(x);
    if (total == 0) {
        return MARMOT_SUCCESS;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    size_t bytes = total * elem_size;

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_TENSOR, "transpose", bytes, true, "gpu");

    id<MTLBuffer> bufferSrc = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    if (bufferSrc == nil) {
        bufferSrc = metal_buffer_acquire(ctx, x->data, bytes);
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferDst = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferDst == nil) {
        bufferDst = metal_buffer_acquire(ctx, out->data, bytes);
    } else {
        out_private = true;
    }
    if (bufferSrc == nil || bufferDst == nil) {
        if (bufferSrc != nil)
            [bufferSrc release];
        if (bufferDst != nil)
            [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal transpose: unable to acquire buffers");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    if (is_identity) {
        if (x->data == out->data) {
            if (bufferSrc != nil) {
                [bufferSrc release];
            }
            if (bufferDst != nil) {
                [bufferDst release];
            }
            return MARMOT_SUCCESS;
        }
        marmot_error_t copy_result = metal_tensor_simple_copy(ctx, bufferSrc, bufferDst, bytes, out, out_private);
        return copy_result;
    }

    metal_transpose_params_t params = {
        .ndim = (uint32_t)ndim,
        .elem_size = (uint32_t)elem_size,
        .total = (uint32_t)total,
        .dst_divisors = {0},
        .src_strides_perm = {0},
    };

    uint32_t src_strides[8];
    src_strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; --i) {
        src_strides[i] = src_strides[i + 1] * (uint32_t)x->shape.shape[i + 1];
    }

    uint32_t dst_div = 1;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        params.dst_divisors[i] = dst_div;
        dst_div *= (uint32_t)out->shape.shape[i];
    }

    for (size_t i = 0; i < ndim; ++i) {
        params.src_strides_perm[i] = src_strides[perm_buffer[i]];
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, "tensor_transpose_nd");
    if (pipeline == nil) {
        [bufferSrc release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal transpose: missing ND kernel");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferSrc release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal transpose: unable to acquire encoder");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }
    [encoder setBuffer:bufferSrc offset:0 atIndex:0];
    [encoder setBuffer:bufferDst offset:0 atIndex:1];
    [encoder setBytes:&params length:sizeof(params) atIndex:2];

    NSUInteger total_u = (NSUInteger)total;
    MTLSize tpg = metal_threads_for_elements(pipeline, total_u, 512);
    [encoder dispatchThreads:MTLSizeMake(total_u, 1, 1) threadsPerThreadgroup:tpg];

    metal_command_stream_flush(ctx, false);
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }

    [pipeline release];
    [bufferSrc release];
    [bufferDst release];
    return MARMOT_SUCCESS;
}

marmot_error_t metal_concat(
    const void *device_ctx, const marmot_tensor_t *const *tensors, size_t num_tensors, marmot_tensor_t *out, int axis
) {
    if (tensors == nullptr || out == nullptr || num_tensors == 0) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid concat inputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    const marmot_tensor_t *ref = tensors[0];
    if (ref == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null tensor in concat");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (axis < 0) {
        axis += (int)ref->shape.ndim;
    }

    if (axis < 0 || axis >= (int)ref->shape.ndim) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Concat axis out of range");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t ndim = ref->shape.ndim;
    if (out->shape.ndim != ndim) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Concat rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    size_t elem_size = marmot_dtype_size(ref->dtype);
    size_t inner = 1;
    for (size_t i = (size_t)axis + 1; i < ndim; ++i) {
        inner *= ref->shape.shape[i];
    }
    size_t outer = 1;
    for (size_t i = 0; i < (size_t)axis; ++i) {
        outer *= ref->shape.shape[i];
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    size_t out_bytes = marmot_tensor_size_bytes(out);

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_TENSOR, "concat", out_bytes, true, "gpu");

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferDst = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferDst == nil) {
        bufferDst = metal_buffer_acquire(ctx, out->data, marmot_tensor_size_bytes(out));
    } else {
        out_private = true;
    }
    if (bufferDst == nil) {
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal concat: unable to acquire destination buffer");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    size_t total_regions = outer * num_tensors;
    metal_buffer_copy_region_t *regions =
        (metal_buffer_copy_region_t *)malloc(total_regions * sizeof(metal_buffer_copy_region_t));
    if (regions == nullptr) {
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal concat: unable to allocate copy region buffer");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    id<MTLBuffer> *source_buffers = (id<MTLBuffer> *)malloc(num_tensors * sizeof(id<MTLBuffer>));
    if (source_buffers == nullptr) {
        free(regions);
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal concat: unable to allocate source buffer table");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    size_t out_axis = out->shape.shape[axis];
    size_t dst_axis_stride = out_axis * inner;
    size_t axis_offset_stride = 0;
    size_t region_count = 0;
    size_t source_count = 0;

    for (size_t t = 0; t < num_tensors; ++t) {
        const marmot_tensor_t *tensor = tensors[t];
        if (tensor == nullptr || tensor->dtype != ref->dtype || tensor->shape.ndim != ndim) {
            metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Concat tensor mismatch");
            goto concat_cleanup_error_dim;
        }
        for (size_t dim = 0; dim < ndim; ++dim) {
            if (dim == (size_t)axis) {
                continue;
            }
            if (tensor->shape.shape[dim] != ref->shape.shape[dim]) {
                metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Concat tensor shape mismatch");
                goto concat_cleanup_error_dim;
            }
        }

        size_t axis_len = tensor->shape.shape[axis];
        size_t copy_elems = axis_len * inner;
        size_t copy_bytes = copy_elems * elem_size;
        id<MTLBuffer> bufferSrc = metal_residency_acquire_compute(ctx, tensor, tensor->dtype, nullptr);
        if (bufferSrc == nil) {
            bufferSrc = metal_buffer_acquire(ctx, tensor->data, marmot_tensor_size_bytes(tensor));
        }
        if (bufferSrc == nil) {
            metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal concat: unable to acquire source buffer");
            goto concat_cleanup_error_src;
        }
        source_buffers[source_count++] = bufferSrc;

        for (size_t outer_idx = 0; outer_idx < outer; ++outer_idx) {
            if (copy_bytes == 0) {
                continue;
            }
            size_t dst_offset_elems = outer_idx * dst_axis_stride + axis_offset_stride;
            size_t src_offset_elems = outer_idx * copy_elems;
            size_t dst_offset_bytes = dst_offset_elems * elem_size;
            size_t src_offset_bytes = src_offset_elems * elem_size;
            metal_buffer_copy_region_t region;
            region.src = bufferSrc;
            region.src_offset = src_offset_bytes;
            region.dst = bufferDst;
            region.dst_offset = dst_offset_bytes;
            region.size = copy_bytes;
            regions[region_count++] = region;
        }

        axis_offset_stride += copy_elems;
    }

    if (region_count > 0) {
        marmot_error_t copy_err = metal_copy_regions(ctx, regions, region_count);
        if (copy_err != MARMOT_SUCCESS) {
            metal_set_error(copy_err, "Metal concat: batched buffer copy failed");
            goto concat_cleanup_error_copy;
        }
    }

    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }

    for (size_t i = 0; i < source_count; ++i) {
        [source_buffers[i] release];
    }
    [bufferDst release];
    free(source_buffers);
    free(regions);
    return MARMOT_SUCCESS;

concat_cleanup_error_copy:
    for (size_t i = 0; i < source_count; ++i) {
        [source_buffers[i] release];
    }
    [bufferDst release];
    free(source_buffers);
    free(regions);
    return MARMOT_ERROR_BACKEND_INIT_FAILED;

concat_cleanup_error_src:
    for (size_t i = 0; i < source_count; ++i) {
        [source_buffers[i] release];
    }
    [bufferDst release];
    free(source_buffers);
    free(regions);
    return MARMOT_ERROR_BACKEND_INIT_FAILED;

concat_cleanup_error_dim:
    [bufferDst release];
    free(source_buffers);
    free(regions);
    return MARMOT_ERROR_DIMENSION_MISMATCH;
}

marmot_error_t metal_slice(
    const void *device_ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *starts, const size_t *sizes
) {
    if (x == nullptr || out == nullptr || starts == nullptr || sizes == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid slice inputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (x->shape.ndim != out->shape.ndim) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "Slice rank mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    size_t ndim = x->shape.ndim;
    for (size_t i = 0; i < ndim; ++i) {
        if (starts[i] + sizes[i] > x->shape.shape[i] || sizes[i] != out->shape.shape[i]) {
            metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Slice range out of bounds");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    }

    size_t elem_size = marmot_dtype_size(x->dtype);
    if (elem_size == 0) {
        return MARMOT_SUCCESS;
    }

    size_t total_elements = marmot_tensor_num_elements(out);
    if (total_elements == 0) {
        return MARMOT_SUCCESS;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    size_t bytes = elem_size * total_elements;

    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_TENSOR, "slice", bytes, true, "gpu");

    id<MTLBuffer> bufferSrc = metal_residency_acquire_compute(ctx, x, x->dtype, nullptr);
    if (bufferSrc == nil) {
        bufferSrc = metal_buffer_acquire(ctx, x->data, marmot_tensor_size_bytes(x));
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferDst = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferDst == nil) {
        bufferDst = metal_buffer_acquire(ctx, out->data, marmot_tensor_size_bytes(out));
    } else {
        out_private = true;
    }
    bool is_full_slice = true;
    for (size_t i = 0; i < ndim; ++i) {
        if (starts[i] != 0 || sizes[i] != x->shape.shape[i]) {
            is_full_slice = false;
            break;
        }
    }

    if (is_full_slice) {
        if (x->data == out->data) {
            if (bufferSrc != nil) {
                [bufferSrc release];
            }
            if (bufferDst != nil) {
                [bufferDst release];
            }
            return MARMOT_SUCCESS;
        }
        if (bufferSrc == nil || bufferDst == nil) {
            if (bufferSrc != nil) {
                [bufferSrc release];
            }
            if (bufferDst != nil) {
                [bufferDst release];
            }
            metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal slice: unable to acquire buffers");
            return MARMOT_ERROR_BACKEND_INIT_FAILED;
        }
        size_t copy_bytes = marmot_tensor_size_bytes(out);
        size_t src_bytes = marmot_tensor_size_bytes(x);
        if (src_bytes < copy_bytes) {
            copy_bytes = src_bytes;
        }
        return metal_tensor_simple_copy(ctx, bufferSrc, bufferDst, copy_bytes, out, out_private);
    }

    if (bufferSrc == nil || bufferDst == nil) {
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal slice: unable to acquire buffers");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    size_t block_elems = sizes[ndim - 1];
    size_t block_start_dim = ndim - 1;
    while (block_start_dim > 0) {
        size_t prev_dim = block_start_dim - 1;
        size_t stride = x->shape.strides[prev_dim];
        if (block_elems == stride) {
            block_elems *= sizes[prev_dim];
            block_start_dim = prev_dim;
        } else {
            break;
        }
    }

    size_t outer_dims = block_start_dim;
    size_t outer_count = 1;
    for (size_t i = 0; i < outer_dims; ++i) {
        outer_count *= sizes[i];
    }
    size_t block_bytes = block_elems * elem_size;

    size_t base_src_offset = 0;
    for (size_t i = 0; i < ndim; ++i) {
        base_src_offset += starts[i] * x->shape.strides[i];
    }

    metal_buffer_copy_region_t *regions =
        (metal_buffer_copy_region_t *)malloc((outer_count == 0 ? 1 : outer_count) * sizeof(metal_buffer_copy_region_t));
    if (regions == nullptr) {
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        metal_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal slice: unable to allocate copy regions");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    size_t *indices = nullptr;
    if (outer_dims > 0) {
        indices = (size_t *)calloc(outer_dims, sizeof(size_t));
        if (indices == nullptr) {
            free(regions);
            if (bufferSrc != nil) {
                [bufferSrc release];
            }
            if (bufferDst != nil) {
                [bufferDst release];
            }
            metal_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Metal slice: unable to allocate index buffer");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    size_t region_count = outer_count == 0 ? 1 : outer_count;
    for (size_t r = 0; r < region_count; ++r) {
        size_t src_offset = base_src_offset;
        size_t dst_offset = 0;
        for (size_t dim = 0; dim < outer_dims; ++dim) {
            size_t idx_val = indices[dim];
            src_offset += idx_val * x->shape.strides[dim];
            dst_offset += idx_val * out->shape.strides[dim];
        }

        metal_buffer_copy_region_t region;
        region.src = bufferSrc;
        region.src_offset = src_offset * elem_size;
        region.dst = bufferDst;
        region.dst_offset = dst_offset * elem_size;
        region.size = block_bytes;
        regions[r] = region;

        if (outer_dims > 0) {
            for (size_t dim = outer_dims; dim-- > 0;) {
                indices[dim]++;
                if (indices[dim] < sizes[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }
    }

    marmot_error_t copy_status = metal_copy_regions(ctx, regions, region_count);
    if (copy_status != MARMOT_SUCCESS) {
        metal_set_error(copy_status, "Metal slice: region copy failed");
        if (indices != nullptr) {
            free(indices);
        }
        free(regions);
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        return copy_status;
    }

    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }

    if (indices != nullptr) {
        free(indices);
    }
    free(regions);
    if (bufferSrc != nil) {
        [bufferSrc release];
    }
    if (bufferDst != nil) {
        [bufferDst release];
    }
    return MARMOT_SUCCESS;
}

typedef struct {
    uint32_t rows_out;
    uint32_t cols;
    uint32_t input_rows;
    uint32_t dtype_bytes;
    uint64_t input_stride0;
    uint64_t input_stride1;
    uint64_t output_stride0;
    uint64_t output_stride1;
    uint64_t index_stride0;
    uint32_t index_is_signed;
} metal_gather_rows_uniforms_t;

typedef struct {
    uint32_t count;
    uint32_t dst_size;
    uint64_t src_stride;
    uint64_t index_stride;
    uint64_t dst_stride;
} metal_scatter_u64_to_i32_uniforms_t;

marmot_error_t metal_gather_rows(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *out
) {
    if (input == nullptr || indices == nullptr || out == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid gather_rows inputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->shape.ndim != 2 || out->shape.ndim != 2) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "gather_rows expects 2D input/output tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (indices->shape.ndim != 1) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "gather_rows indices must be 1D");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (input->dtype != out->dtype) {
        metal_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "gather_rows requires matching input/output dtypes");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t rows_out = out->shape.shape[0];
    const size_t cols = out->shape.shape[1];
    if (rows_out != indices->shape.shape[0] || cols != input->shape.shape[1]) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "gather_rows output shape mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (rows_out == 0 || cols == 0) {
        return MARMOT_SUCCESS;
    }
    const size_t dtype_bytes = marmot_dtype_size(input->dtype);
    if (dtype_bytes == 0) {
        return MARMOT_SUCCESS;
    }

    if (indices->dtype != MARMOT_DTYPE_INT32 && indices->dtype != MARMOT_DTYPE_UINT32) {
        metal_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "gather_rows indices dtype unsupported");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t bytes = rows_out * cols * dtype_bytes;
    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_TENSOR, "gather_rows", bytes, true, "gpu");

    id<MTLBuffer> bufferSrc = metal_residency_acquire_existing(ctx, input, input->dtype);
    if (bufferSrc == nil) {
        bufferSrc = metal_buffer_acquire(ctx, input->data, marmot_tensor_size_bytes(input));
    }

    id<MTLBuffer> bufferIdx = metal_residency_acquire_existing(ctx, indices, indices->dtype);
    if (bufferIdx == nil) {
        bufferIdx = metal_buffer_acquire(ctx, indices->data, marmot_tensor_size_bytes(indices));
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferDst = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferDst == nil) {
        bufferDst = metal_buffer_acquire(ctx, out->data, marmot_tensor_size_bytes(out));
    } else {
        out_private = true;
    }

    if (bufferSrc == nil || bufferIdx == nil || bufferDst == nil) {
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferIdx != nil) {
            [bufferIdx release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal gather_rows: unable to acquire buffers");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, "tensor_gather_rows_generic");
    if (pipeline == nil) {
        [bufferSrc release];
        [bufferIdx release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal gather_rows: missing pipeline");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferSrc release];
        [bufferIdx release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal gather_rows: unable to acquire encoder");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_gather_rows_uniforms_t params = {
        .rows_out = (uint32_t)rows_out,
        .cols = (uint32_t)cols,
        .input_rows = (uint32_t)input->shape.shape[0],
        .dtype_bytes = (uint32_t)dtype_bytes,
        .input_stride0 = (uint64_t)input->shape.strides[0],
        .input_stride1 = (uint64_t)input->shape.strides[1],
        .output_stride0 = (uint64_t)out->shape.strides[0],
        .output_stride1 = (uint64_t)out->shape.strides[1],
        .index_stride0 = (uint64_t)indices->shape.strides[0],
        .index_is_signed = indices->dtype == MARMOT_DTYPE_INT32 ? 1u : 0u,
    };

    [encoder setBytes:&params length:sizeof(params) atIndex:0];
    [encoder setBuffer:bufferSrc offset:0 atIndex:1];
    [encoder setBuffer:bufferIdx offset:0 atIndex:2];
    [encoder setBuffer:bufferDst offset:0 atIndex:3];

    const uint32_t total = (uint32_t)(rows_out * cols);
    MTLSize grid = MTLSizeMake(total, 1, 1);
    MTLSize tpg = metal_threads_for_elements(pipeline, total, 512);
    [encoder dispatchThreads:grid threadsPerThreadgroup:tpg];

    metal_command_stream_flush(ctx, false);
    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }

    [pipeline release];
    [bufferSrc release];
    [bufferIdx release];
    [bufferDst release];
    return MARMOT_SUCCESS;
}

marmot_error_t metal_scatter_u64_to_i32(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices, marmot_tensor_t *out
) {
    if (input == nullptr || indices == nullptr || out == nullptr) {
        metal_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid scatter_u64_to_i32 inputs");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->shape.ndim != 1 || indices->shape.ndim != 1 || out->shape.ndim != 1) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "scatter_u64_to_i32 expects 1D tensors");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }
    if (input->dtype != MARMOT_DTYPE_UINT64 || out->dtype != MARMOT_DTYPE_INT32 ||
        indices->dtype != MARMOT_DTYPE_UINT32) {
        metal_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "scatter_u64_to_i32 dtype mismatch");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }

    const size_t count = input->shape.shape[0];
    const size_t dst_size = out->shape.shape[0];
    if (count == 0 || dst_size == 0) {
        return MARMOT_SUCCESS;
    }
    if (indices->shape.shape[0] != count) {
        metal_set_error(MARMOT_ERROR_DIMENSION_MISMATCH, "scatter_u64_to_i32 input/indices length mismatch");
        return MARMOT_ERROR_DIMENSION_MISMATCH;
    }

    metal_context_t *ctx = (metal_context_t *)device_ctx;
    if (ctx == nullptr) {
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    const size_t bytes = count * sizeof(uint64_t) + count * sizeof(uint32_t) + dst_size * sizeof(int32_t);
    metal_routing_log_decision(ctx, METAL_ROUTING_CATEGORY_TENSOR, "scatter_u64_to_i32", bytes, true, "gpu");

    id<MTLBuffer> bufferSrc = metal_residency_acquire_compute(ctx, input, input->dtype, nullptr);
    if (bufferSrc == nil) {
        bufferSrc = metal_buffer_acquire(ctx, input->data, marmot_tensor_size_bytes(input));
    }

    id<MTLBuffer> bufferIdx = metal_residency_acquire_compute(ctx, indices, indices->dtype, nullptr);
    if (bufferIdx == nil) {
        bufferIdx = metal_buffer_acquire(ctx, indices->data, marmot_tensor_size_bytes(indices));
    }

    bool out_private = false;
    bool out_is_new = false;
    id<MTLBuffer> bufferDst = metal_residency_acquire_compute(ctx, out, out->dtype, &out_is_new);
    if (bufferDst == nil) {
        bufferDst = metal_buffer_acquire(ctx, out->data, marmot_tensor_size_bytes(out));
    } else {
        out_private = true;
    }

    if (bufferSrc == nil || bufferIdx == nil || bufferDst == nil) {
        if (bufferSrc != nil) {
            [bufferSrc release];
        }
        if (bufferIdx != nil) {
            [bufferIdx release];
        }
        if (bufferDst != nil) {
            [bufferDst release];
        }
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal scatter_u64_to_i32: unable to acquire buffers");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputePipelineState> pipeline = metal_pipeline_get(ctx, "tensor_scatter_u64_to_i32_generic");
    if (pipeline == nil) {
        [bufferSrc release];
        [bufferIdx release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal scatter_u64_to_i32: missing pipeline");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    id<MTLComputeCommandEncoder> encoder = metal_command_acquire_compute_encoder(ctx, pipeline);
    if (encoder == nil) {
        [pipeline release];
        [bufferSrc release];
        [bufferIdx release];
        [bufferDst release];
        metal_set_error(MARMOT_ERROR_BACKEND_INIT_FAILED, "Metal scatter_u64_to_i32: unable to acquire encoder");
        return MARMOT_ERROR_BACKEND_INIT_FAILED;
    }

    metal_scatter_u64_to_i32_uniforms_t params = {
        .count = (uint32_t)count,
        .dst_size = (uint32_t)dst_size,
        .src_stride = input->shape.strides[0],
        .index_stride = indices->shape.strides[0],
        .dst_stride = out->shape.strides[0],
    };

    [encoder setBytes:&params length:sizeof(params) atIndex:0];
    [encoder setBuffer:bufferSrc offset:0 atIndex:1];
    [encoder setBuffer:bufferIdx offset:0 atIndex:2];
    [encoder setBuffer:bufferDst offset:0 atIndex:3];

    MTLSize grid = MTLSizeMake((NSUInteger)count, 1, 1);
    NSUInteger threads = metal_threadgroup_size_1d(pipeline, (NSUInteger)count);
    if (threads == 0) {
        threads = 1;
    }
    MTLSize threadgroup = MTLSizeMake(threads, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];

    metal_command_stream_flush(ctx, false);

    if (out_private) {
        metal_residency_mark_dirty(ctx, out, out->dtype);
    }

    [pipeline release];
    [bufferSrc release];
    [bufferIdx release];
    [bufferDst release];
    return MARMOT_SUCCESS;
}

#endif // __APPLE__
