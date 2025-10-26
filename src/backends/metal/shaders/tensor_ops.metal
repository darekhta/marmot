#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

// -----------------------------------------------------------------------------

struct CopyParams {
    ulong src_offset;
    ulong dst_offset;
    ulong count;
};

#define K_SLICE_MAX_DIMS 8u

struct SliceUniforms {
    uint ndim;
    uint dtype_bytes;
    uint reserved0;
    uint reserved1;
    ulong total_elements;
    ulong src_strides[K_SLICE_MAX_DIMS];
    ulong dst_strides[K_SLICE_MAX_DIMS];
    ulong starts[K_SLICE_MAX_DIMS];
    ulong out_shape[K_SLICE_MAX_DIMS];
};

kernel void tensor_slice_generic(
    constant SliceUniforms &params [[buffer(0)]], constant uchar *src [[buffer(1)]], device uchar *dst [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total_elements) {
        return;
    }

    ulong coords[K_SLICE_MAX_DIMS];
    for (uint i = 0u; i < K_SLICE_MAX_DIMS; ++i) {
        coords[i] = 0ul;
    }

    ulong remaining = gid;
    for (int dim = int(params.ndim) - 1; dim >= 0; --dim) {
        ulong extent = params.out_shape[dim];
        ulong coord = 0ul;
        if (extent != 0ul) {
            coord = remaining % extent;
            remaining /= extent;
        }
        coords[dim] = coord;
    }

    ulong src_offset = 0ul;
    ulong dst_offset = 0ul;
    for (uint i = 0u; i < params.ndim; ++i) {
        src_offset += (coords[i] + params.starts[i]) * params.src_strides[i];
        dst_offset += coords[i] * params.dst_strides[i];
    }

    src_offset *= params.dtype_bytes;
    dst_offset *= params.dtype_bytes;

    for (uint b = 0u; b < params.dtype_bytes; ++b) {
        dst[dst_offset + b] = src[src_offset + b];
    }
}

struct GatherRowsUniforms {
    uint rows_out;
    uint cols;
    uint input_rows;
    uint dtype_bytes;
    ulong input_stride0;
    ulong input_stride1;
    ulong output_stride0;
    ulong output_stride1;
    ulong index_stride0;
    uint index_is_signed;
};

struct ScatterU64ToI32Uniforms {
    uint count;
    uint dst_size;
    ulong src_stride;
    ulong index_stride;
    ulong dst_stride;
};

kernel void tensor_gather_rows_generic(
    constant GatherRowsUniforms &params [[buffer(0)]], device const uchar *src [[buffer(1)]],
    device const uint *indices [[buffer(2)]], device uchar *dst [[buffer(3)]], uint gid [[thread_position_in_grid]]
) {
    const uint total = params.rows_out * params.cols;
    if (gid >= total || params.cols == 0u) {
        return;
    }

    const uint row = gid / params.cols;
    const uint col = gid - row * params.cols;
    const uint idx_bits = indices[(ulong)row * params.index_stride0];
    int idx_signed = params.index_is_signed != 0u ? int(idx_bits) : int(idx_bits);
    if (idx_signed < 0 || (uint)idx_signed >= params.input_rows) {
        const ulong dst_offset =
            ((ulong)row * params.output_stride0 + (ulong)col * params.output_stride1) * (ulong)params.dtype_bytes;
        for (uint b = 0u; b < params.dtype_bytes; ++b) {
            dst[dst_offset + b] = 0;
        }
        return;
    }

    const uint idx = (uint)idx_signed;
    const ulong src_offset =
        ((ulong)idx * params.input_stride0 + (ulong)col * params.input_stride1) * (ulong)params.dtype_bytes;
    const ulong dst_offset =
        ((ulong)row * params.output_stride0 + (ulong)col * params.output_stride1) * (ulong)params.dtype_bytes;
    for (uint b = 0u; b < params.dtype_bytes; ++b) {
        dst[dst_offset + b] = src[src_offset + b];
    }
}

kernel void tensor_scatter_u64_to_i32_generic(
    constant ScatterU64ToI32Uniforms &params [[buffer(0)]], device const ulong *src [[buffer(1)]],
    device const uint *indices [[buffer(2)]], device int *dst [[buffer(3)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }
    const uint idx = indices[(ulong)gid * params.index_stride];
    if (idx >= params.dst_size) {
        return;
    }
    const ulong src_index = (ulong)gid * params.src_stride;
    dst[(ulong)idx * params.dst_stride] = (int)src[src_index];
}

kernel void tensor_copy_range(
    device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]], constant CopyParams &params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) {
        return;
    }
    ulong src_index = params.src_offset + gid;
    ulong dst_index = params.dst_offset + gid;
    dst[dst_index] = src[src_index];
}

kernel void transpose_2d_bytes(
    device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]], constant uint &rows [[buffer(2)]],
    constant uint &cols [[buffer(3)]], constant uint &elem_size [[buffer(4)]],
    constant uint &do_transpose [[buffer(5)]], uint gid [[thread_position_in_grid]]
) {
    uint total = rows * cols;
    if (gid >= total) {
        return;
    }

    uint row = gid / cols;
    uint col = gid % cols;
    uint src_linear = row * cols + col;
    uint dst_linear = do_transpose ? (col * rows + row) : src_linear;

    ulong src_byte = (ulong)src_linear * (ulong)elem_size;
    ulong dst_byte = (ulong)dst_linear * (ulong)elem_size;

    for (uint b = 0; b < elem_size; ++b) {
        dst[dst_byte + b] = src[src_byte + b];
    }
}

// -----------------------------------------------------------------------------
// General N-D transpose
// -----------------------------------------------------------------------------

struct TransposeParams {
    uint ndim;
    uint elem_size;
    uint total;
    uint dst_divisors[8];
    uint src_strides_perm[8];
};

kernel void tensor_transpose_nd(
    device const uchar *src [[buffer(0)]], device uchar *dst [[buffer(1)]],
    constant TransposeParams &params [[buffer(2)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total) {
        return;
    }

    uint coords[8];
    uint remaining = gid;
    for (uint axis = 0; axis < params.ndim; ++axis) {
        uint stride = (axis + 1 < params.ndim) ? params.dst_divisors[axis] : 1;
        uint value;
        if (axis + 1 < params.ndim) {
            value = remaining / stride;
            remaining -= value * stride;
        } else {
            value = remaining;
        }
        coords[axis] = value;
    }

    uint src_index = 0;
    for (uint axis = 0; axis < params.ndim; ++axis) {
        src_index += coords[axis] * params.src_strides_perm[axis];
    }

    uint dst_byte = gid * params.elem_size;
    uint src_byte = src_index * params.elem_size;
    for (uint b = 0; b < params.elem_size; ++b) {
        dst[dst_byte + b] = src[src_byte + b];
    }
}
