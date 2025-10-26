#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

#define K_REDUCTION_MAX_DIMS 8u
#define K_REDUCTION_MAX_THREADGROUP_SIZE 256u

struct ReductionUniforms {
    uint op;
    uint has_indices;
    uint keepdims;
    uint unbiased;
    uint input_ndim;
    uint out_ndim;
    uint reduce_ndim;
    uint kept_ndim;
    uint indices_ndim;
    uint threads_per_group;
    ulong out_elements;
    ulong reduce_total;
    float epsilon;
    float pad[3];
    ulong input_shape[K_REDUCTION_MAX_DIMS];
    ulong input_strides[K_REDUCTION_MAX_DIMS];
    ulong out_shape[K_REDUCTION_MAX_DIMS];
    ulong out_strides[K_REDUCTION_MAX_DIMS];
    ulong indices_strides[K_REDUCTION_MAX_DIMS];
    ulong reduce_axes[K_REDUCTION_MAX_DIMS];
    ulong reduce_shape[K_REDUCTION_MAX_DIMS];
    ulong reduce_linear_strides[K_REDUCTION_MAX_DIMS];
    ulong kept_axes[K_REDUCTION_MAX_DIMS];
    uint chunks_per_output;
    uint elements_per_chunk;
    uint stage_reserved0;
    uint stage_reserved1;
};

struct ReductionPartial {
    float sum;
    float prod;
    float sum_abs;
    float sum_sq;
    float mean;
    float m2;
    float max_value;
    float min_value;
    ulong max_index;
    ulong min_index;
    uint have_max;
    uint have_min;
    uint any_flag;
    uint all_flag;
    ulong count;
};

static inline float reduction_sanitize(float value) {
    return isnan(value) ? 0.0f : value;
}

static inline float reduction_clamp(float value, float minv, float maxv) {
    return fmin(fmax(value, minv), maxv);
}

static inline ReductionPartial reduction_partial_identity() {
    ReductionPartial partial;
    partial.sum = 0.0f;
    partial.prod = 1.0f;
    partial.sum_abs = 0.0f;
    partial.sum_sq = 0.0f;
    partial.mean = 0.0f;
    partial.m2 = 0.0f;
    partial.max_value = 0.0f;
    partial.min_value = 0.0f;
    partial.max_index = 0ul;
    partial.min_index = 0ul;
    partial.have_max = 0u;
    partial.have_min = 0u;
    partial.any_flag = 0u;
    partial.all_flag = 1u;
    partial.count = 0ul;
    return partial;
}

static inline void reduction_partial_accumulate(thread ReductionPartial &partial, float value, ulong linear_index) {
    partial.sum += value;
    partial.prod *= value;
    partial.sum_abs += fabs(value);
    partial.sum_sq += value * value;
    partial.count += 1ul;

    float delta = value - partial.mean;
    partial.mean += delta / (float)partial.count;
    float delta2 = value - partial.mean;
    partial.m2 += delta * delta2;

    if (partial.have_max == 0u || value > partial.max_value ||
        (value == partial.max_value && linear_index < partial.max_index)) {
        partial.max_value = value;
        partial.max_index = linear_index;
        partial.have_max = 1u;
    }
    if (partial.have_min == 0u || value < partial.min_value ||
        (value == partial.min_value && linear_index < partial.min_index)) {
        partial.min_value = value;
        partial.min_index = linear_index;
        partial.have_min = 1u;
    }

    if (value != 0.0f) {
        partial.any_flag = 1u;
    } else {
        partial.all_flag = 0u;
    }
}

template <typename PartRef>
static inline void reduction_partial_merge_impl(thread ReductionPartial &accum, PartRef part_ref) {
    ReductionPartial part = part_ref;
    if (part.count == 0ul) {
        return;
    }

    if (accum.count == 0ul) {
        accum = part;
        return;
    }

    accum.sum += part.sum;
    accum.prod *= part.prod;
    accum.sum_abs += part.sum_abs;
    accum.sum_sq += part.sum_sq;
    accum.any_flag |= part.any_flag;
    accum.all_flag = (accum.all_flag != 0u && part.all_flag != 0u) ? 1u : 0u;

    float count_a = (float)accum.count;
    float count_b = (float)part.count;
    float total = count_a + count_b;
    float mean_a = accum.mean;
    float mean_b = part.mean;
    float delta = mean_b - mean_a;
    float new_mean = (count_a * mean_a + count_b * mean_b) / total;
    float new_m2 = accum.m2 + part.m2 + delta * delta * (count_a * count_b / total);
    accum.mean = new_mean;
    accum.m2 = new_m2;
    accum.count += part.count;

    if (part.have_max != 0u &&
        (accum.have_max == 0u || part.max_value > accum.max_value ||
         (part.max_value == accum.max_value && part.max_index < accum.max_index))) {
        accum.max_value = part.max_value;
        accum.max_index = part.max_index;
        accum.have_max = 1u;
    }
    if (part.have_min != 0u &&
        (accum.have_min == 0u || part.min_value < accum.min_value ||
         (part.min_value == accum.min_value && part.min_index < accum.min_index))) {
        accum.min_value = part.min_value;
        accum.min_index = part.min_index;
        accum.have_min = 1u;
    }
}

static inline void reduction_partial_merge(thread ReductionPartial &accum, threadgroup const ReductionPartial &part) {
    reduction_partial_merge_impl(accum, part);
}

static inline int reduction_saturate_i32(float value) {
    value = reduction_sanitize(value);
    const float minv = -2147483648.0f;
    const float maxv = 2147483647.0f;
    value = reduction_clamp(value, minv, maxv);
    float rounded = rint(value);
    rounded = reduction_clamp(rounded, minv, maxv);
    return int(rounded);
}

static inline short reduction_saturate_i16(float value) {
    value = reduction_sanitize(value);
    const float minv = -32768.0f;
    const float maxv = 32767.0f;
    value = reduction_clamp(value, minv, maxv);
    float rounded = rint(value);
    rounded = reduction_clamp(rounded, minv, maxv);
    return short(rounded);
}

static inline char reduction_saturate_i8(float value) {
    value = reduction_sanitize(value);
    const float minv = -128.0f;
    const float maxv = 127.0f;
    value = reduction_clamp(value, minv, maxv);
    float rounded = rint(value);
    rounded = reduction_clamp(rounded, minv, maxv);
    return char(rounded);
}

static inline uint reduction_saturate_u32(float value) {
    value = reduction_sanitize(value);
    const float maxv = 4294967295.0f;
    value = fmax(value, 0.0f);
    value = fmin(value, maxv);
    float rounded = rint(value);
    if (rounded < 0.0f) {
        rounded = 0.0f;
    }
    if (rounded > maxv) {
        rounded = maxv;
    }
    return uint(rounded);
}

static inline ushort reduction_saturate_u16(float value) {
    value = reduction_sanitize(value);
    const float maxv = 65535.0f;
    value = fmax(value, 0.0f);
    value = fmin(value, maxv);
    float rounded = rint(value);
    if (rounded < 0.0f) {
        rounded = 0.0f;
    }
    if (rounded > maxv) {
        rounded = maxv;
    }
    return ushort(rounded);
}

static inline uchar reduction_saturate_u8(float value) {
    value = reduction_sanitize(value);
    const float maxv = 255.0f;
    value = fmax(value, 0.0f);
    value = fmin(value, maxv);
    float rounded = rint(value);
    if (rounded < 0.0f) {
        rounded = 0.0f;
    }
    if (rounded > maxv) {
        rounded = maxv;
    }
    return uchar(rounded);
}

static inline ulong reduction_saturate_u64(float value) {
    value = reduction_sanitize(value);
    const float maxv = 18446744073709551615.0f;
    value = fmax(value, 0.0f);
    value = fmin(value, maxv);
    float rounded = rint(value);
    if (rounded < 0.0f) {
        rounded = 0.0f;
    }
    if (rounded > maxv) {
        rounded = maxv;
    }
    return ulong(rounded);
}

template <typename Ptr>
struct ReductionLoaderFloat {
    Ptr data;
    inline float operator()(ulong offset) const {
        return data[offset];
    }
};

template <typename Ptr>
struct ReductionLoaderHalf {
    Ptr data;
    inline float operator()(ulong offset) const {
        return data[offset];
    }
};

template <typename Ptr>
struct ReductionLoaderBF16 {
    Ptr data;
    inline float operator()(ulong offset) const {
        return read_bf16(data[offset]);
    }
};

#if MARMOT_ENABLE_FP8
template <typename Ptr>
struct ReductionLoaderFP8E4M3 {
    Ptr data;
    inline float operator()(ulong offset) const {
        return read_fp8_e4m3(data[offset]);
    }
};

template <typename Ptr>
struct ReductionLoaderFP8E5M2 {
    Ptr data;
    inline float operator()(ulong offset) const {
        return read_fp8_e5m2(data[offset]);
    }
};
#endif

template <typename Ptr, typename ValueType>
struct ReductionLoaderInt {
    Ptr data;
    inline float operator()(ulong offset) const {
        return (float)ValueType(data[offset]);
    }
};

template <typename Ptr>
struct ReductionLoaderUInt64 {
    Ptr data;
    inline float operator()(ulong offset) const {
        return (float)data[offset];
    }
};

template <typename Ptr>
struct ReductionStoreFloat {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = (float)value;
    }
};

template <typename Ptr>
struct ReductionStoreHalf {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = half(value);
    }
};

template <typename Ptr>
struct ReductionStoreBF16 {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = write_bf16((float)value);
    }
};

#if MARMOT_ENABLE_FP8
template <typename Ptr>
struct ReductionStoreFP8E4M3 {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = write_fp8_e4m3((float)value);
    }
};

template <typename Ptr>
struct ReductionStoreFP8E5M2 {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = write_fp8_e5m2((float)value);
    }
};
#endif

template <typename Ptr, typename ValueType>
struct ReductionStoreInt;

template <typename Ptr>
struct ReductionStoreInt<Ptr, int> {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = reduction_saturate_i32(value);
    }
};

template <typename Ptr>
struct ReductionStoreInt<Ptr, short> {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = reduction_saturate_i16(value);
    }
};

template <typename Ptr>
struct ReductionStoreInt<Ptr, char> {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = reduction_saturate_i8(value);
    }
};

template <typename Ptr, typename ValueType>
struct ReductionStoreUInt;

template <typename Ptr>
struct ReductionStoreUInt<Ptr, uint> {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = reduction_saturate_u32(value);
    }
};

template <typename Ptr>
struct ReductionStoreUInt<Ptr, ushort> {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = reduction_saturate_u16(value);
    }
};

template <typename Ptr>
struct ReductionStoreUInt<Ptr, uchar> {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = reduction_saturate_u8(value);
    }
};

template <typename Ptr>
struct ReductionStoreUInt64 {
    Ptr data;
    inline void operator()(ulong offset, float value) const {
        data[offset] = reduction_saturate_u64(value);
    }
};

template <typename Loader>
static inline void reduction_stage1_kernel_core(
    constant ReductionUniforms &params, Loader load, device ReductionPartial *partials,
    threadgroup ReductionPartial *partials_scratch, threadgroup ulong *base_offset_scratch, uint tid, uint group_linear
) {
    uint threads_per_group = params.threads_per_group == 0u ? 1u : params.threads_per_group;
    threads_per_group = min(threads_per_group, (uint)K_REDUCTION_MAX_THREADGROUP_SIZE);

    uint chunks_per_output = params.chunks_per_output == 0u ? 1u : params.chunks_per_output;
    ulong total_chunks = params.out_elements * (ulong)chunks_per_output;
    ulong group_id = (ulong)group_linear;
    if (group_id >= total_chunks) {
        if (tid == 0u) {
            partials[group_id] = reduction_partial_identity();
        }
        return;
    }

    ulong output_index = group_id / (ulong)chunks_per_output;
    if (output_index >= params.out_elements) {
        if (tid == 0u) {
            partials[group_id] = reduction_partial_identity();
        }
        return;
    }

    uint elements_per_chunk = params.elements_per_chunk == 0u ? threads_per_group : params.elements_per_chunk;
    if (elements_per_chunk == 0u) {
        elements_per_chunk = 1u;
    }
    ulong chunk_index = group_id % (ulong)chunks_per_output;
    ulong chunk_start = chunk_index * (ulong)elements_per_chunk;
    if (chunk_start >= params.reduce_total) {
        if (tid == 0u) {
            partials[group_id] = reduction_partial_identity();
        }
        return;
    }
    ulong chunk_end = chunk_start + (ulong)elements_per_chunk;
    if (chunk_end > params.reduce_total) {
        chunk_end = params.reduce_total;
    }

    if (tid == 0u) {
        ulong out_coords[K_REDUCTION_MAX_DIMS];
        for (uint i = 0u; i < K_REDUCTION_MAX_DIMS; ++i) {
            out_coords[i] = 0ul;
        }

        if (params.out_ndim > 0u) {
            ulong linear = output_index;
            for (int dim = int(params.out_ndim) - 1; dim >= 0; --dim) {
                ulong extent = params.out_shape[dim];
                ulong coord = 0ul;
                if (extent != 0ul) {
                    coord = linear % extent;
                    linear /= extent;
                }
                out_coords[dim] = coord;
            }
        }

        ulong base = 0ul;
        if (params.keepdims != 0u) {
            for (uint axis = 0u; axis < params.input_ndim; ++axis) {
                ulong idx = params.out_ndim > 0u ? out_coords[axis] : 0ul;
                base += idx * params.input_strides[axis];
            }
        } else if (params.kept_ndim != 0u) {
            for (uint i = 0u; i < params.kept_ndim; ++i) {
                ulong axis = params.kept_axes[i];
                ulong idx = params.out_ndim > 0u ? out_coords[i] : 0ul;
                base += idx * params.input_strides[axis];
            }
        }
        *base_offset_scratch = base;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    ReductionPartial partial = reduction_partial_identity();

    for (ulong linear = chunk_start + (ulong)tid; linear < chunk_end; linear += (ulong)threads_per_group) {
        ulong offset = *base_offset_scratch;
        ulong remainder = linear;
        for (uint r = 0u; r < params.reduce_ndim; ++r) {
            ulong stride = params.reduce_linear_strides[r];
            ulong coord = 0ul;
            if (stride != 0ul) {
                coord = remainder / stride;
                remainder %= stride;
            }
            offset += coord * params.input_strides[params.reduce_axes[r]];
        }

        float value = load(offset);
        reduction_partial_accumulate(partial, value, linear);
    }

    partials_scratch[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0u) {
        return;
    }

    ReductionPartial accum = reduction_partial_identity();
    accum.prod = 1.0f;
    accum.all_flag = 1u;

    for (uint i = 0u; i < threads_per_group; ++i) {
        reduction_partial_merge(accum, partials_scratch[i]);
    }

    partials[group_id] = accum;
}

template <typename Store>
static inline void reduction_stage2_kernel_core(
    constant ReductionUniforms &params, Store store, device const ReductionPartial *partials, device ulong *indices,
    threadgroup ReductionPartial *partials_scratch, threadgroup ulong *out_offset_scratch,
    threadgroup ulong *indices_offset_scratch, uint tid, uint group_index
) {
    if (group_index >= params.out_elements) {
        return;
    }

    uint threads_per_group = params.threads_per_group == 0u ? 1u : params.threads_per_group;
    threads_per_group = min(threads_per_group, (uint)K_REDUCTION_MAX_THREADGROUP_SIZE);
    uint chunks_per_output = params.chunks_per_output == 0u ? 1u : params.chunks_per_output;
    ulong total_chunks = params.out_elements * (ulong)chunks_per_output;

    if (tid == 0u) {
        ulong out_coords[K_REDUCTION_MAX_DIMS];
        for (uint i = 0u; i < K_REDUCTION_MAX_DIMS; ++i) {
            out_coords[i] = 0ul;
        }

        if (params.out_ndim > 0u) {
            ulong linear = (ulong)group_index;
            for (int dim = int(params.out_ndim) - 1; dim >= 0; --dim) {
                ulong extent = params.out_shape[dim];
                ulong coord = 0ul;
                if (extent != 0ul) {
                    coord = linear % extent;
                    linear /= extent;
                }
                out_coords[dim] = coord;
            }
        }

        ulong computed_out = 0ul;
        for (uint i = 0u; i < params.out_ndim; ++i) {
            computed_out += out_coords[i] * params.out_strides[i];
        }
        *out_offset_scratch = computed_out;

        ulong idx_offset = 0ul;
        if (params.has_indices != 0u && params.indices_ndim > 0u) {
            uint dims = min(params.indices_ndim, params.out_ndim);
            for (uint i = 0u; i < dims; ++i) {
                idx_offset += out_coords[i] * params.indices_strides[i];
            }
        }
        *indices_offset_scratch = idx_offset;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    ReductionPartial accum = reduction_partial_identity();
    accum.prod = 1.0f;
    accum.all_flag = 1u;

    for (uint chunk = tid; chunk < chunks_per_output; chunk += threads_per_group) {
        ulong partial_index = (ulong)group_index * (ulong)chunks_per_output + (ulong)chunk;
        if (partial_index < total_chunks) {
            ReductionPartial part = partials[partial_index];
            reduction_partial_merge_impl(accum, part);
        }
    }

    partials_scratch[tid] = accum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0u) {
        return;
    }

    ReductionPartial final_partial = reduction_partial_identity();
    final_partial.prod = 1.0f;
    final_partial.all_flag = 1u;

    for (uint i = 0u; i < threads_per_group; ++i) {
        reduction_partial_merge(final_partial, partials_scratch[i]);
    }

    float result = 0.0f;
    ulong arg_index = 0ul;

    switch (params.op) {
    case MARMOT_DEVICE_REDUCTION_SUM:
        result = final_partial.sum;
        break;
    case MARMOT_DEVICE_REDUCTION_MEAN:
        result = final_partial.count > 0ul ? (final_partial.sum / (float)final_partial.count) : 0.0f;
        break;
    case MARMOT_DEVICE_REDUCTION_PROD:
        result = final_partial.count > 0ul ? final_partial.prod : 0.0f;
        break;
    case MARMOT_DEVICE_REDUCTION_MAX:
    case MARMOT_DEVICE_REDUCTION_ARGMAX:
        result = final_partial.have_max != 0u ? final_partial.max_value : 0.0f;
        arg_index = final_partial.have_max != 0u ? final_partial.max_index : 0ul;
        break;
    case MARMOT_DEVICE_REDUCTION_MIN:
    case MARMOT_DEVICE_REDUCTION_ARGMIN:
        result = final_partial.have_min != 0u ? final_partial.min_value : 0.0f;
        arg_index = final_partial.have_min != 0u ? final_partial.min_index : 0ul;
        break;
    case MARMOT_DEVICE_REDUCTION_ANY:
        result = final_partial.any_flag != 0u ? 1.0f : 0.0f;
        break;
    case MARMOT_DEVICE_REDUCTION_ALL:
        result = final_partial.all_flag != 0u ? 1.0f : 0.0f;
        break;
    case MARMOT_DEVICE_REDUCTION_VARIANCE:
    case MARMOT_DEVICE_REDUCTION_STD: {
        float denom;
        if (params.unbiased != 0u && final_partial.count > 1ul) {
            denom = (float)(final_partial.count - 1ul);
        } else {
            denom = (float)final_partial.count;
        }
        if (denom <= 0.0f) {
            denom = 1.0f;
        }
        float variance = (final_partial.count > 0ul ? (final_partial.m2 / denom) : 0.0f) + params.epsilon;
        if (params.op == MARMOT_DEVICE_REDUCTION_STD) {
            variance = sqrt(variance);
        }
        result = variance;
        break;
    }
    case MARMOT_DEVICE_REDUCTION_NORM_L1:
        result = final_partial.sum_abs;
        break;
    case MARMOT_DEVICE_REDUCTION_NORM_L2:
        result = sqrt(final_partial.sum_sq);
        break;
    default:
        result = 0.0f;
        break;
    }

    store(*out_offset_scratch, result);

    if (params.has_indices != 0u && indices != nullptr) {
        ulong write_offset = (params.indices_ndim == 0u) ? 0ul : *indices_offset_scratch;
        if (params.op == MARMOT_DEVICE_REDUCTION_ARGMAX) {
            indices[write_offset] = arg_index;
        } else if (params.op == MARMOT_DEVICE_REDUCTION_ARGMIN) {
            indices[write_offset] = arg_index;
        } else {
            indices[write_offset] = 0ul;
        }
    }
}

kernel void reduction_stage1_f32(
    constant ReductionUniforms &params [[buffer(0)]], constant float *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderFloat<constant float *> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_f16(
    constant ReductionUniforms &params [[buffer(0)]], constant half *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderHalf<constant half *> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_bf16(
    constant ReductionUniforms &params [[buffer(0)]], constant ushort *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderBF16<constant ushort *> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_i32(
    constant ReductionUniforms &params [[buffer(0)]], constant int *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderInt<constant int *, int> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_i16(
    constant ReductionUniforms &params [[buffer(0)]], constant short *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderInt<constant short *, short> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_i8(
    constant ReductionUniforms &params [[buffer(0)]], constant char *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderInt<constant char *, char> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_u8(
    constant ReductionUniforms &params [[buffer(0)]], constant uchar *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderInt<constant uchar *, uchar> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_u16(
    constant ReductionUniforms &params [[buffer(0)]], constant ushort *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderInt<constant ushort *, ushort> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_u32(
    constant ReductionUniforms &params [[buffer(0)]], constant uint *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderInt<constant uint *, uint> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_u64(
    constant ReductionUniforms &params [[buffer(0)]], constant ulong *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderUInt64<constant ulong *> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

#if MARMOT_ENABLE_FP8
kernel void reduction_stage1_fp8_e4m3(
    constant ReductionUniforms &params [[buffer(0)]], constant uchar *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderFP8E4M3<constant uchar *> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

kernel void reduction_stage1_fp8_e5m2(
    constant ReductionUniforms &params [[buffer(0)]], constant uchar *input [[buffer(1)]],
    device ReductionPartial *partials [[buffer(2)]], uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong base_offset;
    ReductionLoaderFP8E5M2<constant uchar *> loader = {input};
    reduction_stage1_kernel_core(params, loader, partials, partials_scratch, &base_offset, tid3.x, group_pos.x);
}

#endif

kernel void reduction_stage2_f32(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device float *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreFloat<device float *> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_f16(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device half *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreHalf<device half *> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_bf16(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device ushort *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreBF16<device ushort *> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_i32(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device int *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreInt<device int *, int> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_i16(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device short *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreInt<device short *, short> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_i8(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device char *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreInt<device char *, char> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_u8(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device uchar *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreUInt<device uchar *, uchar> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_u16(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device ushort *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreUInt<device ushort *, ushort> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_u32(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device uint *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreUInt<device uint *, uint> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_u64(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device ulong *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreUInt64<device ulong *> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

#if MARMOT_ENABLE_FP8
kernel void reduction_stage2_fp8_e4m3(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device uchar *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreFP8E4M3<device uchar *> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

kernel void reduction_stage2_fp8_e5m2(
    constant ReductionUniforms &params [[buffer(0)]], device const ReductionPartial *partials [[buffer(1)]],
    device uchar *output [[buffer(2)]], device ulong *indices [[buffer(3)]],
    uint3 tid3 [[thread_position_in_threadgroup]], uint3 group_pos [[threadgroup_position_in_grid]]
) {
    threadgroup ReductionPartial partials_scratch[K_REDUCTION_MAX_THREADGROUP_SIZE];
    threadgroup ulong out_offset;
    threadgroup ulong indices_offset;
    ReductionStoreFP8E5M2<device uchar *> storer = {output};
    reduction_stage2_kernel_core(
        params, storer, partials, indices, partials_scratch, &out_offset, &indices_offset, tid3.x, group_pos.x
    );
}

#endif
