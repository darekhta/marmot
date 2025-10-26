#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

#include "common/activation_utils.h"

#define DEFINE_UNARY_KERNEL(NAME, READ_T, WRITE_T, READ_FN, WRITE_FN)                                                  \
    kernel void NAME(                                                                                                  \
        constant READ_T *input [[buffer(0)]], device WRITE_T *output [[buffer(1)]],                                    \
        uint id [[thread_position_in_grid]]                                                                            \
    ) {                                                                                                                \
        float x = READ_FN(input[id]);                                                                                  \
        float y = unary_op(x);                                                                                         \
        output[id] = WRITE_FN(y);                                                                                      \
    }

// ReLU
#undef DEFINE_UNARY_KERNEL
#define DEFINE_RELU_KERNEL(NAME, READ_T, WRITE_T, READ_FN, WRITE_FN)                                                   \
    kernel void NAME(                                                                                                  \
        constant READ_T *input [[buffer(0)]], device WRITE_T *output [[buffer(1)]],                                    \
        uint id [[thread_position_in_grid]]                                                                            \
    ) {                                                                                                                \
        float x = READ_FN(input[id]);                                                                                  \
        float y = relu_op(x);                                                                                          \
        output[id] = WRITE_FN(y);                                                                                      \
    }

DEFINE_RELU_KERNEL(relu_f32, float, float, float, float)
DEFINE_RELU_KERNEL(relu_f16, half, half, read_half, write_half)
DEFINE_RELU_KERNEL(relu_bf16, ushort, ushort, read_bf16, write_bf16)
DEFINE_RELU_KERNEL(relu_fp8_e4m3, uchar, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_RELU_KERNEL(relu_fp8_e5m2, uchar, uchar, read_fp8_e5m2, write_fp8_e5m2)

#undef DEFINE_RELU_KERNEL

// ReLU vec4 variants
kernel void relu_f32_vec4(
    constant float4 *input [[buffer(0)]], device float4 *output [[buffer(1)]], constant uint &total_vec4 [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    float4 x = input[id];
    output[id] = fmax(x, 0.0f);
}

kernel void relu_f16_vec4(
    constant half4 *input [[buffer(0)]], device half4 *output [[buffer(1)]], constant uint &total_vec4 [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    half4 x = input[id];
    output[id] = fmax(x, half4(0.0h));
}

kernel void relu_bf16_vec4(
    constant ushort4 *input [[buffer(0)]], device ushort4 *output [[buffer(1)]],
    constant uint &total_vec4 [[buffer(2)]], uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    ushort4 x = input[id];
    float4 xf = float4(read_bf16(x.x), read_bf16(x.y), read_bf16(x.z), read_bf16(x.w));
    float4 result = fmax(xf, 0.0f);
    output[id] = ushort4(write_bf16(result.x), write_bf16(result.y), write_bf16(result.z), write_bf16(result.w));
}

// GELU
#define DEFINE_GELU_KERNEL(NAME, READ_T, WRITE_T, READ_FN, WRITE_FN)                                                   \
    kernel void NAME(                                                                                                  \
        constant READ_T *input [[buffer(0)]], device WRITE_T *output [[buffer(1)]],                                    \
        uint id [[thread_position_in_grid]]                                                                            \
    ) {                                                                                                                \
        float x = READ_FN(input[id]);                                                                                  \
        float y = gelu_exact(x);                                                                                       \
        output[id] = WRITE_FN(y);                                                                                      \
    }

DEFINE_GELU_KERNEL(gelu_f32, float, float, float, float)
DEFINE_GELU_KERNEL(gelu_f16, half, half, read_half, write_half)
DEFINE_GELU_KERNEL(gelu_bf16, ushort, ushort, read_bf16, write_bf16)
DEFINE_GELU_KERNEL(gelu_fp8_e4m3, uchar, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_GELU_KERNEL(gelu_fp8_e5m2, uchar, uchar, read_fp8_e5m2, write_fp8_e5m2)

#undef DEFINE_GELU_KERNEL

// GELU vec4 variants
kernel void gelu_f32_vec4(
    constant float4 *input [[buffer(0)]], device float4 *output [[buffer(1)]], constant uint &total_vec4 [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    float4 x = input[id];
    output[id] = float4(gelu_exact(x.x), gelu_exact(x.y), gelu_exact(x.z), gelu_exact(x.w));
}

kernel void gelu_f16_vec4(
    constant half4 *input [[buffer(0)]], device half4 *output [[buffer(1)]], constant uint &total_vec4 [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    half4 x = input[id];
    float4 xf = float4(x);
    float4 result = float4(gelu_exact(xf.x), gelu_exact(xf.y), gelu_exact(xf.z), gelu_exact(xf.w));
    output[id] = half4(result);
}

kernel void gelu_bf16_vec4(
    constant ushort4 *input [[buffer(0)]], device ushort4 *output [[buffer(1)]],
    constant uint &total_vec4 [[buffer(2)]], uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    ushort4 x = input[id];
    float4 xf = float4(read_bf16(x.x), read_bf16(x.y), read_bf16(x.z), read_bf16(x.w));
    float4 result = float4(gelu_exact(xf.x), gelu_exact(xf.y), gelu_exact(xf.z), gelu_exact(xf.w));
    output[id] = ushort4(write_bf16(result.x), write_bf16(result.y), write_bf16(result.z), write_bf16(result.w));
}

// SiLU
#define DEFINE_SILU_KERNEL(NAME, READ_T, WRITE_T, READ_FN, WRITE_FN)                                                   \
    kernel void NAME(                                                                                                  \
        constant READ_T *input [[buffer(0)]], device WRITE_T *output [[buffer(1)]],                                    \
        uint id [[thread_position_in_grid]]                                                                            \
    ) {                                                                                                                \
        float x = READ_FN(input[id]);                                                                                  \
        float y = silu_exact(x);                                                                                       \
        output[id] = WRITE_FN(y);                                                                                      \
    }

DEFINE_SILU_KERNEL(silu_f32, float, float, float, float)
DEFINE_SILU_KERNEL(silu_f16, half, half, read_half, write_half)
DEFINE_SILU_KERNEL(silu_bf16, ushort, ushort, read_bf16, write_bf16)
DEFINE_SILU_KERNEL(silu_fp8_e4m3, uchar, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_SILU_KERNEL(silu_fp8_e5m2, uchar, uchar, read_fp8_e5m2, write_fp8_e5m2)

#undef DEFINE_SILU_KERNEL

// SiLU vec4 variants
kernel void silu_f32_vec4(
    constant float4 *input [[buffer(0)]], device float4 *output [[buffer(1)]], constant uint &total_vec4 [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    float4 x = input[id];
    output[id] = float4(silu_exact(x.x), silu_exact(x.y), silu_exact(x.z), silu_exact(x.w));
}

kernel void silu_f16_vec4(
    constant half4 *input [[buffer(0)]], device half4 *output [[buffer(1)]], constant uint &total_vec4 [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    half4 x = input[id];
    float4 xf = float4(x);
    float4 result = float4(silu_exact(xf.x), silu_exact(xf.y), silu_exact(xf.z), silu_exact(xf.w));
    output[id] = half4(result);
}

kernel void silu_bf16_vec4(
    constant ushort4 *input [[buffer(0)]], device ushort4 *output [[buffer(1)]],
    constant uint &total_vec4 [[buffer(2)]], uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    ushort4 x = input[id];
    float4 xf = float4(read_bf16(x.x), read_bf16(x.y), read_bf16(x.z), read_bf16(x.w));
    float4 result = float4(silu_exact(xf.x), silu_exact(xf.y), silu_exact(xf.z), silu_exact(xf.w));
    output[id] = ushort4(write_bf16(result.x), write_bf16(result.y), write_bf16(result.z), write_bf16(result.w));
}

kernel void fused_bias_activation_f32(
    device const float *input [[buffer(0)]], device const float *bias [[buffer(1)]],
    device const float *residual [[buffer(2)]], device float *output [[buffer(3)]],
    constant FusedBiasActivationUniforms &uniforms [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= uniforms.total_elements) {
        return;
    }
    bool has_bias = (uniforms.flags & FusedBiasFlagHasBias) != 0u;
    bool has_residual = (uniforms.flags & FusedBiasFlagHasResidual) != 0u;
    uint bias_index = 0u;
    if (has_bias) {
        bias_index = (uniforms.flags & FusedBiasFlagScalarBias) != 0u ? 0u : (gid % uniforms.bias_length);
    }
    float x = input[gid];
    if (has_bias) {
        x += bias[bias_index];
    }
    float y = apply_fused_activation(uniforms.activation, x, uniforms.params);
    if (has_residual) {
        y += residual[gid];
    }
    output[gid] = y;
}

kernel void fused_bias_activation_f16(
    device const half *input [[buffer(0)]], device const half *bias [[buffer(1)]],
    device const half *residual [[buffer(2)]], device half *output [[buffer(3)]],
    constant FusedBiasActivationUniforms &uniforms [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= uniforms.total_elements) {
        return;
    }
    bool has_bias = (uniforms.flags & FusedBiasFlagHasBias) != 0u;
    bool has_residual = (uniforms.flags & FusedBiasFlagHasResidual) != 0u;
    uint bias_index = 0u;
    if (has_bias) {
        bias_index = (uniforms.flags & FusedBiasFlagScalarBias) != 0u ? 0u : (gid % uniforms.bias_length);
    }
    float x = read_half(input[gid]);
    if (has_bias) {
        x += read_half(bias[bias_index]);
    }
    float result = apply_fused_activation(uniforms.activation, x, uniforms.params);
    if (has_residual) {
        result += read_half(residual[gid]);
    }
    output[gid] = write_half(result);
}

kernel void fused_bias_activation_bf16(
    device const ushort *input [[buffer(0)]], device const ushort *bias [[buffer(1)]],
    device const ushort *residual [[buffer(2)]], device ushort *output [[buffer(3)]],
    constant FusedBiasActivationUniforms &uniforms [[buffer(4)]], uint gid [[thread_position_in_grid]]
) {
    if (gid >= uniforms.total_elements) {
        return;
    }
    bool has_bias = (uniforms.flags & FusedBiasFlagHasBias) != 0u;
    bool has_residual = (uniforms.flags & FusedBiasFlagHasResidual) != 0u;
    uint bias_index = 0u;
    if (has_bias) {
        bias_index = (uniforms.flags & FusedBiasFlagScalarBias) != 0u ? 0u : (gid % uniforms.bias_length);
    }
    float x = read_bf16(input[gid]);
    if (has_bias) {
        x += read_bf16(bias[bias_index]);
    }
    float result = apply_fused_activation(uniforms.activation, x, uniforms.params);
    if (has_residual) {
        result += read_bf16(residual[gid]);
    }
    output[gid] = write_bf16(result);
}

static inline uint fused_bias_index(uint gid, uint bias_length, bool scalar_bias) {
    return scalar_bias ? 0u : (gid % bias_length);
}

kernel void fused_bias_rope_activation_f32(
    device const float *input [[buffer(0)]], device const float *bias [[buffer(1)]],
    device const float *residual [[buffer(2)]], device float *output [[buffer(3)]],
    device const float *positions [[buffer(4)]], device const float *freqs [[buffer(5)]],
    constant FusedBiasActivationUniforms &uniforms [[buffer(6)]], uint gid [[thread_position_in_grid]]
) {
    if ((uniforms.flags & FusedBiasFlagHasRope) == 0u) {
        return;
    }
    uint pairs = uniforms.rope_pairs;
    uint rows = uniforms.rope_rows;
    if (pairs == 0u || rows == 0u) {
        return;
    }
    uint total_pairs = pairs * rows;
    if (gid >= total_pairs) {
        return;
    }
    bool has_bias = (uniforms.flags & FusedBiasFlagHasBias) != 0u;
    bool scalar_bias = (uniforms.flags & FusedBiasFlagScalarBias) != 0u;
    bool has_residual = (uniforms.flags & FusedBiasFlagHasResidual) != 0u;
    uint bias_length = uniforms.bias_length;
    uint dim = uniforms.rope_dim;
    uint row = gid / pairs;
    uint pair = gid % pairs;
    uint half_dim = dim >> 1;
    bool is_neox = uniforms.rope_type != 0u;
    uint even_index = row * dim + (is_neox ? pair : (pair * 2u));
    uint odd_index = row * dim + (is_neox ? (pair + half_dim) : (pair * 2u + 1u));

    float even = input[even_index];
    float odd = input[odd_index];
    if (has_bias) {
        uint bias_idx_even = fused_bias_index(even_index, bias_length, scalar_bias);
        uint bias_idx_odd = fused_bias_index(odd_index, bias_length, scalar_bias);
        even += bias[bias_idx_even];
        odd += bias[bias_idx_odd];
    }

    float pos = positions[row];
    float angle = pos * freqs[pair];
    float cos_theta = cos(angle) * uniforms.rope_attn_scale;
    float sin_theta = sin(angle) * uniforms.rope_attn_scale;
    float rotated_even = even * cos_theta - odd * sin_theta;
    float rotated_odd = even * sin_theta + odd * cos_theta;

    float value_even = apply_fused_activation(uniforms.activation, rotated_even, uniforms.params);
    float value_odd = apply_fused_activation(uniforms.activation, rotated_odd, uniforms.params);
    if (has_residual) {
        value_even += residual[even_index];
        value_odd += residual[odd_index];
    }
    output[even_index] = value_even;
    output[odd_index] = value_odd;
}

kernel void fused_bias_rope_activation_f16(
    device const half *input [[buffer(0)]], device const half *bias [[buffer(1)]],
    device const half *residual [[buffer(2)]], device half *output [[buffer(3)]],
    device const float *positions [[buffer(4)]], device const float *freqs [[buffer(5)]],
    constant FusedBiasActivationUniforms &uniforms [[buffer(6)]], uint gid [[thread_position_in_grid]]
) {
    if ((uniforms.flags & FusedBiasFlagHasRope) == 0u) {
        return;
    }
    uint pairs = uniforms.rope_pairs;
    uint rows = uniforms.rope_rows;
    if (pairs == 0u || rows == 0u) {
        return;
    }
    uint total_pairs = pairs * rows;
    if (gid >= total_pairs) {
        return;
    }
    bool has_bias = (uniforms.flags & FusedBiasFlagHasBias) != 0u;
    bool scalar_bias = (uniforms.flags & FusedBiasFlagScalarBias) != 0u;
    bool has_residual = (uniforms.flags & FusedBiasFlagHasResidual) != 0u;
    uint bias_length = uniforms.bias_length;
    uint dim = uniforms.rope_dim;
    uint row = gid / pairs;
    uint pair = gid % pairs;
    uint half_dim = dim >> 1;
    bool is_neox = uniforms.rope_type != 0u;
    uint even_index = row * dim + (is_neox ? pair : (pair * 2u));
    uint odd_index = row * dim + (is_neox ? (pair + half_dim) : (pair * 2u + 1u));

    float even = read_half(input[even_index]);
    float odd = read_half(input[odd_index]);
    if (has_bias) {
        uint bias_idx_even = fused_bias_index(even_index, bias_length, scalar_bias);
        uint bias_idx_odd = fused_bias_index(odd_index, bias_length, scalar_bias);
        even += read_half(bias[bias_idx_even]);
        odd += read_half(bias[bias_idx_odd]);
    }
    float pos = positions[row];
    float angle = pos * freqs[pair];
    float cos_theta = cos(angle) * uniforms.rope_attn_scale;
    float sin_theta = sin(angle) * uniforms.rope_attn_scale;
    float rotated_even = even * cos_theta - odd * sin_theta;
    float rotated_odd = even * sin_theta + odd * cos_theta;
    float value_even = apply_fused_activation(uniforms.activation, rotated_even, uniforms.params);
    float value_odd = apply_fused_activation(uniforms.activation, rotated_odd, uniforms.params);
    if (has_residual) {
        value_even += read_half(residual[even_index]);
        value_odd += read_half(residual[odd_index]);
    }
    output[even_index] = write_half(value_even);
    output[odd_index] = write_half(value_odd);
}

kernel void fused_bias_rope_activation_bf16(
    device const ushort *input [[buffer(0)]], device const ushort *bias [[buffer(1)]],
    device const ushort *residual [[buffer(2)]], device ushort *output [[buffer(3)]],
    device const float *positions [[buffer(4)]], device const float *freqs [[buffer(5)]],
    constant FusedBiasActivationUniforms &uniforms [[buffer(6)]], uint gid [[thread_position_in_grid]]
) {
    if ((uniforms.flags & FusedBiasFlagHasRope) == 0u) {
        return;
    }
    uint pairs = uniforms.rope_pairs;
    uint rows = uniforms.rope_rows;
    if (pairs == 0u || rows == 0u) {
        return;
    }
    uint total_pairs = pairs * rows;
    if (gid >= total_pairs) {
        return;
    }
    bool has_bias = (uniforms.flags & FusedBiasFlagHasBias) != 0u;
    bool scalar_bias = (uniforms.flags & FusedBiasFlagScalarBias) != 0u;
    bool has_residual = (uniforms.flags & FusedBiasFlagHasResidual) != 0u;
    uint bias_length = uniforms.bias_length;
    uint dim = uniforms.rope_dim;
    uint row = gid / pairs;
    uint pair = gid % pairs;
    uint half_dim = dim >> 1;
    bool is_neox = uniforms.rope_type != 0u;
    uint even_index = row * dim + (is_neox ? pair : (pair * 2u));
    uint odd_index = row * dim + (is_neox ? (pair + half_dim) : (pair * 2u + 1u));

    float even = read_bf16(input[even_index]);
    float odd = read_bf16(input[odd_index]);
    if (has_bias) {
        uint bias_idx_even = fused_bias_index(even_index, bias_length, scalar_bias);
        uint bias_idx_odd = fused_bias_index(odd_index, bias_length, scalar_bias);
        even += read_bf16(bias[bias_idx_even]);
        odd += read_bf16(bias[bias_idx_odd]);
    }
    float pos = positions[row];
    float angle = pos * freqs[pair];
    float cos_theta = cos(angle) * uniforms.rope_attn_scale;
    float sin_theta = sin(angle) * uniforms.rope_attn_scale;
    float rotated_even = even * cos_theta - odd * sin_theta;
    float rotated_odd = even * sin_theta + odd * cos_theta;
    float value_even = apply_fused_activation(uniforms.activation, rotated_even, uniforms.params);
    float value_odd = apply_fused_activation(uniforms.activation, rotated_odd, uniforms.params);
    if (has_residual) {
        value_even += read_bf16(residual[even_index]);
        value_odd += read_bf16(residual[odd_index]);
    }
    output[even_index] = write_bf16(value_even);
    output[odd_index] = write_bf16(value_odd);
}

#define DEFINE_ABS_INT_KERNEL(NAME, TYPE)                                                                              \
    kernel void NAME(                                                                                                  \
        constant TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]], uint id [[thread_position_in_grid]]     \
    ) {                                                                                                                \
        output[id] = abs_int_exact<TYPE>(input[id]);                                                                   \
    }

#define DEFINE_ABS_UINT_KERNEL(NAME, TYPE)                                                                             \
    kernel void NAME(                                                                                                  \
        constant TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]], uint id [[thread_position_in_grid]]     \
    ) {                                                                                                                \
        output[id] = input[id];                                                                                        \
    }

#define DEFINE_NEG_INT_KERNEL(NAME, TYPE)                                                                              \
    kernel void NAME(                                                                                                  \
        constant TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]], uint id [[thread_position_in_grid]]     \
    ) {                                                                                                                \
        output[id] = negate_int_exact<TYPE>(input[id]);                                                                \
    }

#define DEFINE_SIGN_INT_KERNEL(NAME, TYPE)                                                                             \
    kernel void NAME(                                                                                                  \
        constant TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]], uint id [[thread_position_in_grid]]     \
    ) {                                                                                                                \
        output[id] = sign_int_exact<TYPE>(input[id]);                                                                  \
    }

#define DEFINE_SIGN_UINT_KERNEL(NAME, TYPE)                                                                            \
    kernel void NAME(                                                                                                  \
        constant TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]], uint id [[thread_position_in_grid]]     \
    ) {                                                                                                                \
        output[id] = sign_uint_exact<TYPE>(input[id]);                                                                 \
    }

#define DEFINE_BITWISE_NOT_KERNEL(NAME, TYPE)                                                                          \
    kernel void NAME(                                                                                                  \
        constant TYPE *input [[buffer(0)]], device TYPE *output [[buffer(1)]], uint id [[thread_position_in_grid]]     \
    ) {                                                                                                                \
        output[id] = ~input[id];                                                                                       \
    }

DEFINE_ABS_INT_KERNEL(abs_i32, int)
DEFINE_ABS_INT_KERNEL(abs_i16, short)
DEFINE_ABS_INT_KERNEL(abs_i8, char)
DEFINE_ABS_UINT_KERNEL(abs_u8, uchar)
DEFINE_ABS_UINT_KERNEL(abs_u16, ushort)
DEFINE_ABS_UINT_KERNEL(abs_u32, uint)
DEFINE_ABS_UINT_KERNEL(abs_u64, ulong)

DEFINE_NEG_INT_KERNEL(neg_i32, int)
DEFINE_NEG_INT_KERNEL(neg_i16, short)
DEFINE_NEG_INT_KERNEL(neg_i8, char)

DEFINE_SIGN_INT_KERNEL(sign_i32, int)
DEFINE_SIGN_INT_KERNEL(sign_i16, short)
DEFINE_SIGN_INT_KERNEL(sign_i8, char)
DEFINE_SIGN_UINT_KERNEL(sign_u8, uchar)
DEFINE_SIGN_UINT_KERNEL(sign_u16, ushort)
DEFINE_SIGN_UINT_KERNEL(sign_u32, uint)
DEFINE_SIGN_UINT_KERNEL(sign_u64, ulong)

DEFINE_BITWISE_NOT_KERNEL(bitwise_not_i32, int)
DEFINE_BITWISE_NOT_KERNEL(bitwise_not_i16, short)
DEFINE_BITWISE_NOT_KERNEL(bitwise_not_i8, char)
DEFINE_BITWISE_NOT_KERNEL(bitwise_not_u8, uchar)
DEFINE_BITWISE_NOT_KERNEL(bitwise_not_u16, ushort)
DEFINE_BITWISE_NOT_KERNEL(bitwise_not_u32, uint)
DEFINE_BITWISE_NOT_KERNEL(bitwise_not_u64, ulong)

#undef DEFINE_ABS_INT_KERNEL
#undef DEFINE_ABS_UINT_KERNEL
#undef DEFINE_NEG_INT_KERNEL
#undef DEFINE_SIGN_INT_KERNEL
#undef DEFINE_SIGN_UINT_KERNEL
#undef DEFINE_BITWISE_NOT_KERNEL

#define DEFINE_PARAMLESS_ACTIVATION_BASE(NAME, FUNC)                                                                   \
    kernel void NAME##_f32(                                                                                            \
        constant float *input [[buffer(0)]], device float *output [[buffer(1)]], uint id [[thread_position_in_grid]]   \
    ) {                                                                                                                \
        float x = input[id];                                                                                           \
        float y = FUNC(x);                                                                                             \
        output[id] = y;                                                                                                \
    }                                                                                                                  \
    kernel void NAME##_f16(                                                                                            \
        constant half *input [[buffer(0)]], device half *output [[buffer(1)]], uint id [[thread_position_in_grid]]     \
    ) {                                                                                                                \
        float x = read_half(input[id]);                                                                                \
        float y = FUNC(x);                                                                                             \
        output[id] = write_half(y);                                                                                    \
    }                                                                                                                  \
    kernel void NAME##_bf16(                                                                                           \
        constant ushort *input [[buffer(0)]], device ushort *output [[buffer(1)]], uint id [[thread_position_in_grid]] \
    ) {                                                                                                                \
        float x = read_bf16(input[id]);                                                                                \
        float y = FUNC(x);                                                                                             \
        output[id] = write_bf16(y);                                                                                    \
    }

#if MARMOT_ENABLE_FP8
#define DEFINE_PARAMLESS_ACTIVATION_FP8(NAME, FUNC)                                                                    \
    kernel void NAME##_fp8_e4m3(                                                                                       \
        constant uchar *input [[buffer(0)]], device uchar *output [[buffer(1)]], uint id [[thread_position_in_grid]]   \
    ) {                                                                                                                \
        float x = read_fp8_e4m3(input[id]);                                                                            \
        float y = FUNC(x);                                                                                             \
        output[id] = write_fp8_e4m3(y);                                                                                \
    }                                                                                                                  \
    kernel void NAME##_fp8_e5m2(                                                                                       \
        constant uchar *input [[buffer(0)]], device uchar *output [[buffer(1)]], uint id [[thread_position_in_grid]]   \
    ) {                                                                                                                \
        float x = read_fp8_e5m2(input[id]);                                                                            \
        float y = FUNC(x);                                                                                             \
        output[id] = write_fp8_e5m2(y);                                                                                \
    }
#endif

DEFINE_PARAMLESS_ACTIVATION_BASE(gelu_tanh, gelu_tanh_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(sigmoid, sigmoid_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(tanh_act, tanh_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(mish, mish_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(abs, abs_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(neg, neg_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(sign, sign_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(sqrt, sqrt_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(exp, exp_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_BASE(log, log_scalar_exact)

#if MARMOT_ENABLE_FP8
DEFINE_PARAMLESS_ACTIVATION_FP8(gelu_tanh, gelu_tanh_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(sigmoid, sigmoid_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(tanh_act, tanh_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(mish, mish_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(abs, abs_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(neg, neg_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(sign, sign_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(sqrt, sqrt_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(exp, exp_scalar_exact)
DEFINE_PARAMLESS_ACTIVATION_FP8(log, log_scalar_exact)
#endif

#undef DEFINE_PARAMLESS_ACTIVATION_BASE
#if MARMOT_ENABLE_FP8
#undef DEFINE_PARAMLESS_ACTIVATION_FP8
#endif

#define DEFINE_PARAM_ACTIVATION_BASE(NAME, FUNC)                                                                       \
    kernel void NAME##_f32(                                                                                            \
        constant float *input [[buffer(0)]], device float *output [[buffer(1)]],                                       \
        constant ActivationParams &params [[buffer(2)]], uint id [[thread_position_in_grid]]                           \
    ) {                                                                                                                \
        float x = input[id];                                                                                           \
        float y = FUNC(x, params);                                                                                     \
        output[id] = y;                                                                                                \
    }                                                                                                                  \
    kernel void NAME##_f16(                                                                                            \
        constant half *input [[buffer(0)]], device half *output [[buffer(1)]],                                         \
        constant ActivationParams &params [[buffer(2)]], uint id [[thread_position_in_grid]]                           \
    ) {                                                                                                                \
        float x = read_half(input[id]);                                                                                \
        float y = FUNC(x, params);                                                                                     \
        output[id] = write_half(y);                                                                                    \
    }                                                                                                                  \
    kernel void NAME##_bf16(                                                                                           \
        constant ushort *input [[buffer(0)]], device ushort *output [[buffer(1)]],                                     \
        constant ActivationParams &params [[buffer(2)]], uint id [[thread_position_in_grid]]                           \
    ) {                                                                                                                \
        float x = read_bf16(input[id]);                                                                                \
        float y = FUNC(x, params);                                                                                     \
        output[id] = write_bf16(y);                                                                                    \
    }

#if MARMOT_ENABLE_FP8
#define DEFINE_PARAM_ACTIVATION_FP8(NAME, FUNC)                                                                        \
    kernel void NAME##_fp8_e4m3(                                                                                       \
        constant uchar *input [[buffer(0)]], device uchar *output [[buffer(1)]],                                       \
        constant ActivationParams &params [[buffer(2)]], uint id [[thread_position_in_grid]]                           \
    ) {                                                                                                                \
        float x = read_fp8_e4m3(input[id]);                                                                            \
        float y = FUNC(x, params);                                                                                     \
        output[id] = write_fp8_e4m3(y);                                                                                \
    }                                                                                                                  \
    kernel void NAME##_fp8_e5m2(                                                                                       \
        constant uchar *input [[buffer(0)]], device uchar *output [[buffer(1)]],                                       \
        constant ActivationParams &params [[buffer(2)]], uint id [[thread_position_in_grid]]                           \
    ) {                                                                                                                \
        float x = read_fp8_e5m2(input[id]);                                                                            \
        float y = FUNC(x, params);                                                                                     \
        output[id] = write_fp8_e5m2(y);                                                                                \
    }
#endif

DEFINE_PARAM_ACTIVATION_BASE(elu, elu_with_params)
DEFINE_PARAM_ACTIVATION_BASE(selu, selu_with_params)
DEFINE_PARAM_ACTIVATION_BASE(leaky_relu, leaky_relu_with_params)
DEFINE_PARAM_ACTIVATION_BASE(prelu, prelu_with_params)

#if MARMOT_ENABLE_FP8
DEFINE_PARAM_ACTIVATION_FP8(elu, elu_with_params)
DEFINE_PARAM_ACTIVATION_FP8(selu, selu_with_params)
DEFINE_PARAM_ACTIVATION_FP8(leaky_relu, leaky_relu_with_params)
DEFINE_PARAM_ACTIVATION_FP8(prelu, prelu_with_params)
#endif

#undef DEFINE_PARAM_ACTIVATION_BASE
#if MARMOT_ENABLE_FP8
#undef DEFINE_PARAM_ACTIVATION_FP8
#endif
