#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

// -----------------------------------------------------------------------------

// LayerNorm kernels
kernel void layernorm_f32(
    constant float *x [[buffer(0)]], constant float *weight [[buffer(1)]], constant float *bias [[buffer(2)]],
    device float *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant float *x_row = x + row * dim;
    device float *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = x_row[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float normalized = (x_row[i] - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        out_row[i] = normalized;
    }
}

kernel void layernorm_f16(
    constant half *x [[buffer(0)]], constant half *weight [[buffer(1)]], constant half *bias [[buffer(2)]],
    device half *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant half *x_row = x + row * dim;
    device half *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = float(x_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float normalized = (float(x_row[i]) - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= float(weight[i]);
        }
        if (bias != nullptr) {
            normalized += float(bias[i]);
        }
        out_row[i] = half(normalized);
    }
}

kernel void layernorm_f16_wf32(
    constant half *x [[buffer(0)]], constant float *weight [[buffer(1)]], constant float *bias [[buffer(2)]],
    device half *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant half *x_row = x + row * dim;
    device half *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = float(x_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float normalized = (float(x_row[i]) - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        out_row[i] = half(normalized);
    }
}

kernel void layernorm_bf16(
    constant ushort *x [[buffer(0)]], constant ushort *weight [[buffer(1)]], constant ushort *bias [[buffer(2)]],
    device ushort *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant ushort *x_row = x + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = bf16_to_float(x_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float normalized = (bf16_to_float(x_row[i]) - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= bf16_to_float(weight[i]);
        }
        if (bias != nullptr) {
            normalized += bf16_to_float(bias[i]);
        }
        out_row[i] = float_to_bf16(normalized);
    }
}

kernel void layernorm_bf16_wf32(
    constant ushort *x [[buffer(0)]], constant float *weight [[buffer(1)]], constant float *bias [[buffer(2)]],
    device ushort *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant ushort *x_row = x + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = bf16_to_float(x_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float normalized = (bf16_to_float(x_row[i]) - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        out_row[i] = float_to_bf16(normalized);
    }
}

kernel void fused_residual_layernorm_f32(
    constant float *x [[buffer(0)]], constant float *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    constant float *bias [[buffer(3)]], device float *out [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant float *x_row = x + row * dim;
    constant float *res_row = residual + row * dim;
    device float *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = x_row[i] + res_row[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float v = x_row[i] + res_row[i];
        float normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        out_row[i] = normalized;
    }
}

kernel void fused_residual_layernorm_f16(
    constant half *x [[buffer(0)]], constant half *residual [[buffer(1)]], constant half *weight [[buffer(2)]],
    constant half *bias [[buffer(3)]], device half *out [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant half *x_row = x + row * dim;
    constant half *res_row = residual + row * dim;
    device half *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = read_half(x_row[i]) + read_half(res_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float v = read_half(x_row[i]) + read_half(res_row[i]);
        float normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= read_half(weight[i]);
        }
        if (bias != nullptr) {
            normalized += read_half(bias[i]);
        }
        out_row[i] = write_half(normalized);
    }
}

kernel void fused_residual_layernorm_f16_wf32(
    constant half *x [[buffer(0)]], constant half *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    constant float *bias [[buffer(3)]], device half *out [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant half *x_row = x + row * dim;
    constant half *res_row = residual + row * dim;
    device half *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = read_half(x_row[i]) + read_half(res_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float v = read_half(x_row[i]) + read_half(res_row[i]);
        float normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        out_row[i] = write_half(normalized);
    }
}

kernel void fused_residual_layernorm_bf16(
    constant ushort *x [[buffer(0)]], constant ushort *residual [[buffer(1)]], constant ushort *weight [[buffer(2)]],
    constant ushort *bias [[buffer(3)]], device ushort *out [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant ushort *x_row = x + row * dim;
    constant ushort *res_row = residual + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float v = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        float normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= read_bf16(weight[i]);
        }
        if (bias != nullptr) {
            normalized += read_bf16(bias[i]);
        }
        out_row[i] = write_bf16(normalized);
    }
}

kernel void fused_residual_layernorm_bf16_wf32(
    constant ushort *x [[buffer(0)]], constant ushort *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    constant float *bias [[buffer(3)]], device ushort *out [[buffer(4)]], constant uint &dim [[buffer(5)]],
    constant float &eps [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant ushort *x_row = x + row * dim;
    constant ushort *res_row = residual + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float v = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        float normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= weight[i];
        }
        if (bias != nullptr) {
            normalized += bias[i];
        }
        out_row[i] = write_bf16(normalized);
    }
}

static inline float4 rmsnorm_load_float4(constant float *ptr) {
    return float4(*((constant packed_float4 *)ptr));
}

static inline float4 rmsnorm_load_half4(constant half *ptr) {
    return float4(*((constant packed_half4 *)ptr));
}

static inline float
rmsnorm_local_sum_sq_f32(constant float *x_row, constant float *res_row, uint dim, uint tid, uint tsize) {
    const uint aligned_dim = dim & ~3u;
    float local_sum_sq = 0.0f;
    if (res_row != nullptr) {
        for (uint i = tid * 4u; i < aligned_dim; i += tsize * 4u) {
            float4 v = rmsnorm_load_float4(x_row + i) + rmsnorm_load_float4(res_row + i);
            local_sum_sq += dot(v, v);
        }
        for (uint i = aligned_dim + tid; i < dim; i += tsize) {
            float v = x_row[i] + res_row[i];
            local_sum_sq += v * v;
        }
    } else {
        for (uint i = tid * 4u; i < aligned_dim; i += tsize * 4u) {
            float4 v = rmsnorm_load_float4(x_row + i);
            local_sum_sq += dot(v, v);
        }
        for (uint i = aligned_dim + tid; i < dim; i += tsize) {
            float v = x_row[i];
            local_sum_sq += v * v;
        }
    }
    return local_sum_sq;
}

static inline float
rmsnorm_local_sum_sq_f16(constant half *x_row, constant half *res_row, uint dim, uint tid, uint tsize) {
    const uint aligned_dim = dim & ~3u;
    float local_sum_sq = 0.0f;
    if (res_row != nullptr) {
        for (uint i = tid * 4u; i < aligned_dim; i += tsize * 4u) {
            float4 v = rmsnorm_load_half4(x_row + i) + rmsnorm_load_half4(res_row + i);
            local_sum_sq += dot(v, v);
        }
        for (uint i = aligned_dim + tid; i < dim; i += tsize) {
            float v = read_half(x_row[i]) + read_half(res_row[i]);
            local_sum_sq += v * v;
        }
    } else {
        for (uint i = tid * 4u; i < aligned_dim; i += tsize * 4u) {
            float4 v = rmsnorm_load_half4(x_row + i);
            local_sum_sq += dot(v, v);
        }
        for (uint i = aligned_dim + tid; i < dim; i += tsize) {
            float v = read_half(x_row[i]);
            local_sum_sq += v * v;
        }
    }
    return local_sum_sq;
}

static inline float rmsnorm_reduce_sum_sq(
    float local_sum_sq, threadgroup float *shared_sum, uint simd_lane_id, uint simd_group_id, uint tsize
) {
    float sum_sq = simd_sum(local_sum_sq);
    if (simd_lane_id == 0u) {
        shared_sum[simd_group_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0u) {
        const uint simd_groups = (tsize + 31u) / 32u;
        float total = (simd_lane_id < simd_groups) ? shared_sum[simd_lane_id] : 0.0f;
        total = simd_sum(total);
        if (simd_lane_id == 0u) {
            shared_sum[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared_sum[0];
}

static inline float
softmax_reduce_max(float local_max, threadgroup float *shared_max, uint simd_lane_id, uint simd_group_id, uint tsize) {
    float max_val = simd_max(local_max);
    if (simd_lane_id == 0u) {
        shared_max[simd_group_id] = max_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0u) {
        const uint simd_groups = (tsize + 31u) / 32u;
        float total = (simd_lane_id < simd_groups) ? shared_max[simd_lane_id] : -INFINITY;
        total = simd_max(total);
        if (simd_lane_id == 0u) {
            shared_max[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared_max[0];
}

static inline float
softmax_reduce_sum(float local_sum, threadgroup float *shared_sum, uint simd_lane_id, uint simd_group_id, uint tsize) {
    float sum = simd_sum(local_sum);
    if (simd_lane_id == 0u) {
        shared_sum[simd_group_id] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0u) {
        const uint simd_groups = (tsize + 31u) / 32u;
        float total = (simd_lane_id < simd_groups) ? shared_sum[simd_lane_id] : 0.0f;
        total = simd_sum(total);
        if (simd_lane_id == 0u) {
            shared_sum[0] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared_sum[0];
}

kernel void fused_residual_rmsnorm_f32(
    constant float *x [[buffer(0)]], constant float *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    device float *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant float *x_row = x + row * dim;
    constant float *res_row = residual + row * dim;
    device float *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f32(x_row, res_row, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = x_row[i] + res_row[i];
        float value = v * norm;
        if (weight != nullptr) {
            value *= weight[i];
        }
        out_row[i] = value;
    }
}

kernel void fused_residual_rmsnorm_f16(
    constant half *x [[buffer(0)]], constant half *residual [[buffer(1)]], constant half *weight [[buffer(2)]],
    device half *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant half *x_row = x + row * dim;
    constant half *res_row = residual + row * dim;
    device half *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f16(x_row, res_row, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = read_half(x_row[i]) + read_half(res_row[i]);
        float value = v * norm;
        if (weight != nullptr) {
            value *= read_half(weight[i]);
        }
        out_row[i] = write_half(value);
    }
}

kernel void fused_residual_rmsnorm_f16_wf32(
    constant half *x [[buffer(0)]], constant half *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    device half *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant half *x_row = x + row * dim;
    constant half *res_row = residual + row * dim;
    device half *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f16(x_row, res_row, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = read_half(x_row[i]) + read_half(res_row[i]);
        float value = v * norm;
        if (weight != nullptr) {
            value *= weight[i];
        }
        out_row[i] = write_half(value);
    }
}

kernel void fused_residual_rmsnorm_bf16(
    constant ushort *x [[buffer(0)]], constant ushort *residual [[buffer(1)]], constant ushort *weight [[buffer(2)]],
    device ushort *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant ushort *x_row = x + row * dim;
    constant ushort *res_row = residual + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        float value = v * norm;
        if (weight != nullptr) {
            value *= read_bf16(weight[i]);
        }
        out_row[i] = write_bf16(value);
    }
}

kernel void fused_residual_rmsnorm_bf16_wf32(
    constant ushort *x [[buffer(0)]], constant ushort *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    device ushort *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant ushort *x_row = x + row * dim;
    constant ushort *res_row = residual + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        float value = v * norm;
        if (weight != nullptr) {
            value *= weight[i];
        }
        out_row[i] = write_bf16(value);
    }
}

kernel void fused_residual_gemma_rmsnorm_f32(
    constant float *x [[buffer(0)]], constant float *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    device float *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant float *x_row = x + row * dim;
    constant float *res_row = residual + row * dim;
    device float *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f32(x_row, res_row, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = x_row[i] + res_row[i];
        float value = v * norm;
        if (weight != nullptr) {
            value *= weight[i] + 1.0f;
        }
        out_row[i] = value;
    }
}

kernel void fused_residual_gemma_rmsnorm_f16(
    constant half *x [[buffer(0)]], constant half *residual [[buffer(1)]], constant half *weight [[buffer(2)]],
    device half *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant half *x_row = x + row * dim;
    constant half *res_row = residual + row * dim;
    device half *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f16(x_row, res_row, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = read_half(x_row[i]) + read_half(res_row[i]);
        float value = v * norm;
        if (weight != nullptr) {
            value *= read_half(weight[i]) + 1.0f;
        }
        out_row[i] = write_half(value);
    }
}

kernel void fused_residual_gemma_rmsnorm_f16_wf32(
    constant half *x [[buffer(0)]], constant half *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    device half *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant half *x_row = x + row * dim;
    constant half *res_row = residual + row * dim;
    device half *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f16(x_row, res_row, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = read_half(x_row[i]) + read_half(res_row[i]);
        float value = v * norm;
        if (weight != nullptr) {
            value *= weight[i] + 1.0f;
        }
        out_row[i] = write_half(value);
    }
}

kernel void fused_residual_gemma_rmsnorm_bf16(
    constant ushort *x [[buffer(0)]], constant ushort *residual [[buffer(1)]], constant ushort *weight [[buffer(2)]],
    device ushort *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant ushort *x_row = x + row * dim;
    constant ushort *res_row = residual + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        float value = v * norm;
        if (weight != nullptr) {
            value *= read_bf16(weight[i]) + 1.0f;
        }
        out_row[i] = write_bf16(value);
    }
}

kernel void fused_residual_gemma_rmsnorm_bf16_wf32(
    constant ushort *x [[buffer(0)]], constant ushort *residual [[buffer(1)]], constant float *weight [[buffer(2)]],
    device ushort *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant ushort *x_row = x + row * dim;
    constant ushort *res_row = residual + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float v = read_bf16(x_row[i]) + read_bf16(res_row[i]);
        float value = v * norm;
        if (weight != nullptr) {
            value *= weight[i] + 1.0f;
        }
        out_row[i] = write_bf16(value);
    }
}

kernel void layernorm_fp8_e4m3(
    constant uchar *x [[buffer(0)]], constant uchar *weight [[buffer(1)]], constant uchar *bias [[buffer(2)]],
    device uchar *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant uchar *x_row = x + row * dim;
    device uchar *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = fp8_e4m3_to_float(x_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float normalized = (fp8_e4m3_to_float(x_row[i]) - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= fp8_e4m3_to_float(weight[i]);
        }
        if (bias != nullptr) {
            normalized += fp8_e4m3_to_float(bias[i]);
        }
        out_row[i] = float_to_fp8_e4m3(normalized);
    }
}

kernel void layernorm_fp8_e5m2(
    constant uchar *x [[buffer(0)]], constant uchar *weight [[buffer(1)]], constant uchar *bias [[buffer(2)]],
    device uchar *out [[buffer(3)]], constant uint &dim [[buffer(4)]], constant float &eps [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    constant uchar *x_row = x + row * dim;
    device uchar *out_row = out + row * dim;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = fp8_e5m2_to_float(x_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tsize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(dim);
    float variance = (shared_sum_sq[0] / float(dim)) - (mean * mean);
    float inv_std = rsqrt(variance + eps);

    for (uint i = tid; i < dim; i += tsize) {
        float normalized = (fp8_e5m2_to_float(x_row[i]) - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= fp8_e5m2_to_float(weight[i]);
        }
        if (bias != nullptr) {
            normalized += fp8_e5m2_to_float(bias[i]);
        }
        out_row[i] = float_to_fp8_e5m2(normalized);
    }
}

// RMSNorm kernels
kernel void rmsnorm_f32(
    constant float *x [[buffer(0)]], constant float *weight [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant float *x_row = x + row * dim;
    device float *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f32(x_row, nullptr, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = x_row[i] * norm;
        if (weight != nullptr) {
            value *= weight[i];
        }
        out_row[i] = value;
    }
}

kernel void rmsnorm_f16(
    constant half *x [[buffer(0)]], constant half *weight [[buffer(1)]], device half *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant half *x_row = x + row * dim;
    device half *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f16(x_row, nullptr, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= float(weight[i]);
        }
        out_row[i] = half(value);
    }
}

kernel void rmsnorm_f16_wf32(
    constant half *x [[buffer(0)]], constant float *weight [[buffer(1)]], device half *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant half *x_row = x + row * dim;
    device half *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f16(x_row, nullptr, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= weight[i];
        }
        out_row[i] = half(value);
    }
}

kernel void rmsnorm_bf16(
    constant ushort *x [[buffer(0)]], constant ushort *weight [[buffer(1)]], device ushort *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant ushort *x_row = x + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = bf16_to_float(x_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = bf16_to_float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= bf16_to_float(weight[i]);
        }
        out_row[i] = float_to_bf16(value);
    }
}

kernel void rmsnorm_bf16_wf32(
    constant ushort *x [[buffer(0)]], constant float *weight [[buffer(1)]], device ushort *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant ushort *x_row = x + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = bf16_to_float(x_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = bf16_to_float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= weight[i];
        }
        out_row[i] = float_to_bf16(value);
    }
}

kernel void rmsnorm_fp8_e4m3(
    constant uchar *x [[buffer(0)]], constant uchar *weight [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant uchar *x_row = x + row * dim;
    device uchar *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = fp8_e4m3_to_float(x_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = fp8_e4m3_to_float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= fp8_e4m3_to_float(weight[i]);
        }
        out_row[i] = float_to_fp8_e4m3(value);
    }
}

kernel void rmsnorm_fp8_e5m2(
    constant uchar *x [[buffer(0)]], constant uchar *weight [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant uchar *x_row = x + row * dim;
    device uchar *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = fp8_e5m2_to_float(x_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = fp8_e5m2_to_float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= fp8_e5m2_to_float(weight[i]);
        }
        out_row[i] = float_to_fp8_e5m2(value);
    }
}

kernel void gemma_rmsnorm_f32(
    constant float *x [[buffer(0)]], constant float *weight [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant float *x_row = x + row * dim;
    device float *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f32(x_row, nullptr, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = x_row[i] * norm;
        if (weight != nullptr) {
            value *= weight[i] + 1.0f;
        }
        out_row[i] = value;
    }
}

kernel void gemma_rmsnorm_f16(
    constant half *x [[buffer(0)]], constant half *weight [[buffer(1)]], device half *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant half *x_row = x + row * dim;
    device half *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f16(x_row, nullptr, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= float(weight[i]) + 1.0f;
        }
        out_row[i] = half(value);
    }
}

kernel void gemma_rmsnorm_f16_wf32(
    constant half *x [[buffer(0)]], constant float *weight [[buffer(1)]], device half *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant half *x_row = x + row * dim;
    device half *out_row = out + row * dim;

    float local_sum_sq = rmsnorm_local_sum_sq_f16(x_row, nullptr, dim, tid, tsize);
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= weight[i] + 1.0f;
        }
        out_row[i] = half(value);
    }
}

kernel void gemma_rmsnorm_bf16(
    constant ushort *x [[buffer(0)]], constant ushort *weight [[buffer(1)]], device ushort *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant ushort *x_row = x + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = bf16_to_float(x_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = bf16_to_float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= bf16_to_float(weight[i]) + 1.0f;
        }
        out_row[i] = float_to_bf16(value);
    }
}

kernel void gemma_rmsnorm_bf16_wf32(
    constant ushort *x [[buffer(0)]], constant float *weight [[buffer(1)]], device ushort *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant ushort *x_row = x + row * dim;
    device ushort *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = bf16_to_float(x_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = bf16_to_float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= weight[i] + 1.0f;
        }
        out_row[i] = float_to_bf16(value);
    }
}

kernel void gemma_rmsnorm_fp8_e4m3(
    constant uchar *x [[buffer(0)]], constant uchar *weight [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant uchar *x_row = x + row * dim;
    device uchar *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = fp8_e4m3_to_float(x_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = fp8_e4m3_to_float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= fp8_e4m3_to_float(weight[i]) + 1.0f;
        }
        out_row[i] = float_to_fp8_e4m3(value);
    }
}

kernel void gemma_rmsnorm_fp8_e5m2(
    constant uchar *x [[buffer(0)]], constant uchar *weight [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &dim [[buffer(3)]], constant float &eps [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]], uint tsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]], uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
    threadgroup float shared_sum[256];

    constant uchar *x_row = x + row * dim;
    device uchar *out_row = out + row * dim;

    float local_sum_sq = 0.0f;
    for (uint i = tid; i < dim; i += tsize) {
        float val = fp8_e5m2_to_float(x_row[i]);
        local_sum_sq += val * val;
    }
    float sum_sq = rmsnorm_reduce_sum_sq(local_sum_sq, shared_sum, simd_lane_id, simd_group_id, tsize);
    float norm = rsqrt(sum_sq / float(dim) + eps);
    for (uint i = tid; i < dim; i += tsize) {
        float value = fp8_e5m2_to_float(x_row[i]) * norm;
        if (weight != nullptr) {
            value *= fp8_e5m2_to_float(weight[i]) + 1.0f;
        }
        out_row[i] = float_to_fp8_e5m2(value);
    }
}

// -----------------------------------------------------------------------------
// Softmax kernels (two-pass for numerical stability)
// -----------------------------------------------------------------------------

#define DEFINE_SOFTMAX_KERNEL_STAGED(NAME, READ_PTR, WRITE_PTR, READ_FN, WRITE_FN)                                     \
    kernel void NAME(                                                                                                  \
        constant READ_PTR *x [[buffer(0)]], device WRITE_PTR *out [[buffer(1)]], constant uint &dim [[buffer(2)]],     \
        uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],                        \
        uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],                       \
        uint simd_group_id [[simdgroup_index_in_threadgroup]]                                                          \
    ) {                                                                                                                \
        threadgroup float shared_max[256];                                                                             \
        threadgroup float shared_sum[256];                                                                             \
                                                                                                                       \
        constant READ_PTR *x_row = x + row * dim;                                                                      \
        device WRITE_PTR *out_row = out + row * dim;                                                                   \
                                                                                                                       \
        float local_max = -INFINITY;                                                                                   \
        for (uint i = tid; i < dim; i += tsize) {                                                                      \
            float val = READ_FN(x_row[i]);                                                                             \
            local_max = fmax(local_max, val);                                                                          \
        }                                                                                                              \
                                                                                                                       \
        float max_val = softmax_reduce_max(local_max, shared_max, simd_lane_id, simd_group_id, tsize);                 \
                                                                                                                       \
        float local_sum = 0.0f;                                                                                        \
        for (uint i = tid; i < dim; i += tsize) {                                                                      \
            float val = READ_FN(x_row[i]);                                                                             \
            float exp_val = exp(val - max_val);                                                                        \
            local_sum += exp_val;                                                                                      \
            out_row[i] = WRITE_FN(exp_val);                                                                            \
        }                                                                                                              \
                                                                                                                       \
        float inv_sum = 1.0f / softmax_reduce_sum(local_sum, shared_sum, simd_lane_id, simd_group_id, tsize);          \
        for (uint i = tid; i < dim; i += tsize) {                                                                      \
            float soft = READ_FN(out_row[i]) * inv_sum;                                                                \
            out_row[i] = WRITE_FN(soft);                                                                               \
        }                                                                                                              \
    }

#define DEFINE_SOFTMAX_KERNEL(NAME, READ_PTR, WRITE_PTR, READ_FN, WRITE_FN)                                            \
    kernel void NAME(                                                                                                  \
        constant READ_PTR *x [[buffer(0)]], device WRITE_PTR *out [[buffer(1)]], constant uint &dim [[buffer(2)]],     \
        uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],                        \
        uint tsize [[threads_per_threadgroup]], uint simd_lane_id [[thread_index_in_simdgroup]],                       \
        uint simd_group_id [[simdgroup_index_in_threadgroup]]                                                          \
    ) {                                                                                                                \
        threadgroup float shared_max[256];                                                                             \
        threadgroup float shared_sum[256];                                                                             \
                                                                                                                       \
        constant READ_PTR *x_row = x + row * dim;                                                                      \
        device WRITE_PTR *out_row = out + row * dim;                                                                   \
                                                                                                                       \
        float local_max = -INFINITY;                                                                                   \
        for (uint i = tid; i < dim; i += tsize) {                                                                      \
            float val = READ_FN(x_row[i]);                                                                             \
            local_max = fmax(local_max, val);                                                                          \
        }                                                                                                              \
                                                                                                                       \
        float max_val = softmax_reduce_max(local_max, shared_max, simd_lane_id, simd_group_id, tsize);                 \
                                                                                                                       \
        float local_sum = 0.0f;                                                                                        \
        for (uint i = tid; i < dim; i += tsize) {                                                                      \
            float val = READ_FN(x_row[i]);                                                                             \
            local_sum += exp(val - max_val);                                                                           \
        }                                                                                                              \
                                                                                                                       \
        float inv_sum = 1.0f / softmax_reduce_sum(local_sum, shared_sum, simd_lane_id, simd_group_id, tsize);          \
        for (uint i = tid; i < dim; i += tsize) {                                                                      \
            float val = READ_FN(x_row[i]);                                                                             \
            float soft = exp(val - max_val) * inv_sum;                                                                 \
            out_row[i] = WRITE_FN(soft);                                                                               \
        }                                                                                                              \
    }

DEFINE_SOFTMAX_KERNEL_STAGED(softmax_f32, float, float, float, float)
DEFINE_SOFTMAX_KERNEL(softmax_f16, half, half, read_half, write_half)
DEFINE_SOFTMAX_KERNEL(softmax_bf16, ushort, ushort, read_bf16, write_bf16)
DEFINE_SOFTMAX_KERNEL(softmax_fp8_e4m3, uchar, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_SOFTMAX_KERNEL(softmax_fp8_e5m2, uchar, uchar, read_fp8_e5m2, write_fp8_e5m2)

#undef DEFINE_SOFTMAX_KERNEL_STAGED
#undef DEFINE_SOFTMAX_KERNEL

#define DEFINE_SOFTMAX_STRIDED_KERNEL(NAME, READ_PTR, WRITE_PTR, READ_FN, WRITE_FN)                                    \
    kernel void NAME(                                                                                                  \
        device const READ_PTR *x [[buffer(0)]], device WRITE_PTR *out [[buffer(1)]],                                   \
        constant uint &rows [[buffer(2)]], constant uint &axis_size [[buffer(3)]],                                     \
        constant uint &inner_stride [[buffer(4)]], uint gid [[thread_position_in_grid]]                                \
    ) {                                                                                                                \
        if (gid >= rows) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        uint outer = gid / inner_stride;                                                                               \
        uint offset = outer * axis_size * inner_stride + (gid % inner_stride);                                         \
                                                                                                                       \
        float max_val = -INFINITY;                                                                                     \
        for (uint i = 0; i < axis_size; ++i) {                                                                         \
            float val = READ_FN(x[offset + i * inner_stride]);                                                         \
            max_val = fmax(max_val, val);                                                                              \
        }                                                                                                              \
                                                                                                                       \
        float sum = 0.0f;                                                                                              \
        for (uint i = 0; i < axis_size; ++i) {                                                                         \
            float val = exp(READ_FN(x[offset + i * inner_stride]) - max_val);                                          \
            sum += val;                                                                                                \
        }                                                                                                              \
        float inv_sum = 1.0f / sum;                                                                                    \
                                                                                                                       \
        for (uint i = 0; i < axis_size; ++i) {                                                                         \
            float val = exp(READ_FN(x[offset + i * inner_stride]) - max_val) * inv_sum;                                \
            out[offset + i * inner_stride] = WRITE_FN(val);                                                            \
        }                                                                                                              \
    }

DEFINE_SOFTMAX_STRIDED_KERNEL(softmax_strided_f32, float, float, (float), (float))
DEFINE_SOFTMAX_STRIDED_KERNEL(softmax_strided_f16, half, half, read_half, write_half)
DEFINE_SOFTMAX_STRIDED_KERNEL(softmax_strided_bf16, ushort, ushort, read_bf16, write_bf16)
DEFINE_SOFTMAX_STRIDED_KERNEL(softmax_strided_fp8_e4m3, uchar, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_SOFTMAX_STRIDED_KERNEL(softmax_strided_fp8_e5m2, uchar, uchar, read_fp8_e5m2, write_fp8_e5m2)

#undef DEFINE_SOFTMAX_STRIDED_KERNEL
