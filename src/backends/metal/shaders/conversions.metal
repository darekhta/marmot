#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

// -----------------------------------------------------------------------------
// Dtype conversion kernels (vectorized conversions)
// -----------------------------------------------------------------------------

kernel void convert_f32_to_f16(
    constant float *src [[buffer(0)]], device half *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = half(src[id]);
    }
}

kernel void convert_f16_to_f32(
    constant half *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = float(src[id]);
    }
}

kernel void convert_f32_to_bf16(
    constant float *src [[buffer(0)]], device ushort *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = float_to_bf16(src[id]);
    }
}

kernel void convert_bf16_to_f32(
    constant ushort *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = bf16_to_float(src[id]);
    }
}

kernel void convert_f16_to_bf16(
    constant half *src [[buffer(0)]], device ushort *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = float_to_bf16(float(src[id]));
    }
}

kernel void convert_bf16_to_f16(
    constant ushort *src [[buffer(0)]], device half *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = half(bf16_to_float(src[id]));
    }
}

kernel void convert_f32_to_fp8_e4m3(
    constant float *src [[buffer(0)]], device uchar *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = float_to_fp8_e4m3(src[id]);
    }
}

kernel void convert_fp8_e4m3_to_f32(
    constant uchar *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = fp8_e4m3_to_float(src[id]);
    }
}

kernel void convert_f32_to_fp8_e5m2(
    constant float *src [[buffer(0)]], device uchar *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = float_to_fp8_e5m2(src[id]);
    }
}

kernel void convert_fp8_e5m2_to_f32(
    constant uchar *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = fp8_e5m2_to_float(src[id]);
    }
}

kernel void convert_f32_to_i64(
    constant float *src [[buffer(0)]], device long *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = (long)src[id];
    }
}

kernel void convert_i64_to_f32(
    constant long *src [[buffer(0)]], device float *dst [[buffer(1)]], constant uint &count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id < count) {
        dst[id] = (float)src[id];
    }
}
