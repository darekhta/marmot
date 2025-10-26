#include <metal_stdlib>
using namespace metal;

kernel void metal_add_relu_fused_f32(
    device const float *input_a [[buffer(0)]], device const float *input_b [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &n_elements [[buffer(3)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_elements) {
        return;
    }
    float sum = input_a[gid] + input_b[gid];
    output[gid] = max(sum, 0.0f);
}

kernel void metal_add_relu_fused_f16(
    device const half *input_a [[buffer(0)]], device const half *input_b [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &n_elements [[buffer(3)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_elements) {
        return;
    }
    half sum = input_a[gid] + input_b[gid];
    output[gid] = max(sum, half(0.0));
}

inline float gelu_approx(float x) {
    const float sqrt_2_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;
    float inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    return 0.5f * x * (1.0f + tanh(inner));
}

kernel void metal_add_gelu_fused_f32(
    device const float *input_a [[buffer(0)]], device const float *input_b [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &n_elements [[buffer(3)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_elements) {
        return;
    }
    float sum = input_a[gid] + input_b[gid];
    output[gid] = gelu_approx(sum);
}

kernel void metal_add_gelu_fused_f16(
    device const half *input_a [[buffer(0)]], device const half *input_b [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &n_elements [[buffer(3)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_elements) {
        return;
    }
    float sum_f32 = float(input_a[gid]) + float(input_b[gid]);
    output[gid] = half(gelu_approx(sum_f32));
}

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

kernel void metal_add_silu_fused_f32(
    device const float *input_a [[buffer(0)]], device const float *input_b [[buffer(1)]],
    device float *output [[buffer(2)]], constant uint &n_elements [[buffer(3)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_elements) {
        return;
    }
    float sum = input_a[gid] + input_b[gid];
    output[gid] = silu(sum);
}

kernel void metal_add_silu_fused_f16(
    device const half *input_a [[buffer(0)]], device const half *input_b [[buffer(1)]],
    device half *output [[buffer(2)]], constant uint &n_elements [[buffer(3)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_elements) {
        return;
    }
    float sum_f32 = float(input_a[gid]) + float(input_b[gid]);
    output[gid] = half(silu(sum_f32));
}

kernel void metal_mul_add_fused_f32(
    device const float *input_a [[buffer(0)]], device const float *input_b [[buffer(1)]],
    device const float *input_c [[buffer(2)]], device float *output [[buffer(3)]],
    constant uint &n_elements [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= n_elements) {
        return;
    }
    output[gid] = fma(input_a[gid], input_b[gid], input_c[gid]);
}
