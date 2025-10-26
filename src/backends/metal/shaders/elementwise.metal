#include <metal_stdlib>
using namespace metal;

#include "common/defines.h"

#include "common/math_utils.h"

#include "common/dtype_utils.h"

#include "common/activation_utils.h"
#include "common/stride_utils.h"

static inline float apply_binary_float(uint op, float lhs, float rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
        return lhs + rhs;
    case MARMOT_DEVICE_BINARY_SUB:
        return lhs - rhs;
    case MARMOT_DEVICE_BINARY_MUL:
        return lhs * rhs;
    case MARMOT_DEVICE_BINARY_DIV:
        return lhs / rhs;
    case MARMOT_DEVICE_BINARY_MIN:
        return fmin(lhs, rhs);
    case MARMOT_DEVICE_BINARY_MAX:
        return fmax(lhs, rhs);
    case MARMOT_DEVICE_BINARY_POW:
        return pow(lhs, rhs);
    case MARMOT_DEVICE_BINARY_MOD:
        return fmod(lhs, rhs);
    case MARMOT_DEVICE_BINARY_SWIGLU:
        return silu_exact(lhs) * rhs;
    case MARMOT_DEVICE_BINARY_GEGLU:
        return gelu_tanh_exact(lhs) * rhs;
    default:
        return 0.0f;
    }
}

static inline int apply_binary_i32(uint op, int lhs, int rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
        return lhs + rhs;
    case MARMOT_DEVICE_BINARY_SUB:
        return lhs - rhs;
    case MARMOT_DEVICE_BINARY_MUL:
        return lhs * rhs;
    case MARMOT_DEVICE_BINARY_DIV:
        return lhs / rhs;
    case MARMOT_DEVICE_BINARY_MIN:
        return lhs < rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_MAX:
        return lhs > rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_POW:
        return int_pow_signed<int>(lhs, rhs);
    case MARMOT_DEVICE_BINARY_MOD:
        return rhs == 0 ? 0 : lhs % rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_AND:
        return lhs & rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_OR:
        return lhs | rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_XOR:
        return lhs ^ rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT: {
        uint amount = normalize_shift_signed(rhs, 32u);
        if (amount >= 32u) {
            return 0;
        }
        return lhs << amount;
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT: {
        uint amount = normalize_shift_signed(rhs, 32u);
        if (amount >= 32u) {
            return lhs < 0 ? -1 : 0;
        }
        return lhs >> amount;
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL: {
        uint amount = normalize_shift_signed(rhs, 32u);
        if (amount >= 32u) {
            return 0;
        }
        uint bits = (uint)lhs;
        return (int)(bits >> amount);
    }
    default:
        return 0;
    }
}

static inline short apply_binary_i16(uint op, short lhs, short rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
        return short(int(lhs) + int(rhs));
    case MARMOT_DEVICE_BINARY_SUB:
        return short(int(lhs) - int(rhs));
    case MARMOT_DEVICE_BINARY_MUL:
        return short(int(lhs) * int(rhs));
    case MARMOT_DEVICE_BINARY_DIV:
        return short(int(lhs) / int(rhs));
    case MARMOT_DEVICE_BINARY_MIN:
        return lhs < rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_MAX:
        return lhs > rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_POW:
        return short(int_pow_signed<int>(int(lhs), int(rhs)));
    case MARMOT_DEVICE_BINARY_MOD:
        return rhs == 0 ? short(0) : short(int(lhs) % int(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_AND:
        return short(int(lhs) & int(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_OR:
        return short(int(lhs) | int(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_XOR:
        return short(int(lhs) ^ int(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT: {
        uint amount = normalize_shift_signed(int(rhs), 16u);
        if (amount >= 16u) {
            return short(0);
        }
        return short(int(lhs) << amount);
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT: {
        uint amount = normalize_shift_signed(int(rhs), 16u);
        if (amount >= 16u) {
            return lhs < 0 ? short(-1) : short(0);
        }
        return short(int(lhs) >> amount);
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL: {
        uint amount = normalize_shift_signed(int(rhs), 16u);
        if (amount >= 16u) {
            return short(0);
        }
        uint bits = uint(ushort(lhs));
        return short(bits >> amount);
    }
    default:
        return 0;
    }
}

static inline char apply_binary_i8(uint op, char lhs, char rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
        return char(int(lhs) + int(rhs));
    case MARMOT_DEVICE_BINARY_SUB:
        return char(int(lhs) - int(rhs));
    case MARMOT_DEVICE_BINARY_MUL:
        return char(int(lhs) * int(rhs));
    case MARMOT_DEVICE_BINARY_DIV:
        return char(int(lhs) / int(rhs));
    case MARMOT_DEVICE_BINARY_MIN:
        return lhs < rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_MAX:
        return lhs > rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_POW:
        return char(int_pow_signed<int>(int(lhs), int(rhs)));
    case MARMOT_DEVICE_BINARY_MOD:
        return rhs == 0 ? char(0) : char(int(lhs) % int(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_AND:
        return char(int(lhs) & int(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_OR:
        return char(int(lhs) | int(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_XOR:
        return char(int(lhs) ^ int(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT: {
        uint amount = normalize_shift_signed(int(rhs), 8u);
        if (amount >= 8u) {
            return char(0);
        }
        return char(int(lhs) << amount);
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT: {
        uint amount = normalize_shift_signed(int(rhs), 8u);
        if (amount >= 8u) {
            return lhs < 0 ? char(-1) : char(0);
        }
        return char(int(lhs) >> amount);
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL: {
        uint amount = normalize_shift_signed(int(rhs), 8u);
        if (amount >= 8u) {
            return char(0);
        }
        uint bits = uint(uchar(lhs));
        return char(bits >> amount);
    }
    default:
        return 0;
    }
}

static inline uint apply_binary_u32(uint op, uint lhs, uint rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
        return lhs + rhs;
    case MARMOT_DEVICE_BINARY_SUB:
        return lhs - rhs;
    case MARMOT_DEVICE_BINARY_MUL:
        return lhs * rhs;
    case MARMOT_DEVICE_BINARY_DIV:
        return lhs / rhs;
    case MARMOT_DEVICE_BINARY_MIN:
        return lhs < rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_MAX:
        return lhs > rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_POW:
        return int_pow_unsigned<uint>(lhs, rhs);
    case MARMOT_DEVICE_BINARY_MOD:
        return rhs == 0u ? 0u : lhs % rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_AND:
        return lhs & rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_OR:
        return lhs | rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_XOR:
        return lhs ^ rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT: {
        uint amount = normalize_shift_unsigned(rhs, 32u);
        if (amount >= 32u) {
            return 0u;
        }
        return lhs << amount;
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT:
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL: {
        uint amount = normalize_shift_unsigned(rhs, 32u);
        if (amount >= 32u) {
            return 0u;
        }
        return lhs >> amount;
    }
    default:
        return 0u;
    }
}

static inline ushort apply_binary_u16(uint op, ushort lhs, ushort rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
        return ushort(uint(lhs) + uint(rhs));
    case MARMOT_DEVICE_BINARY_SUB:
        return ushort(uint(lhs) - uint(rhs));
    case MARMOT_DEVICE_BINARY_MUL:
        return ushort(uint(lhs) * uint(rhs));
    case MARMOT_DEVICE_BINARY_DIV:
        return ushort(uint(lhs) / uint(rhs));
    case MARMOT_DEVICE_BINARY_MIN:
        return lhs < rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_MAX:
        return lhs > rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_POW:
        return ushort(int_pow_unsigned<uint>(uint(lhs), uint(rhs)));
    case MARMOT_DEVICE_BINARY_MOD:
        return rhs == 0 ? ushort(0) : ushort(uint(lhs) % uint(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_AND:
        return ushort(uint(lhs) & uint(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_OR:
        return ushort(uint(lhs) | uint(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_XOR:
        return ushort(uint(lhs) ^ uint(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT: {
        uint amount = normalize_shift_unsigned(uint(rhs), 16u);
        if (amount >= 16u) {
            return ushort(0);
        }
        return ushort(uint(lhs) << amount);
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT:
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL: {
        uint amount = normalize_shift_unsigned(uint(rhs), 16u);
        if (amount >= 16u) {
            return ushort(0);
        }
        return ushort(uint(lhs) >> amount);
    }
    default:
        return 0;
    }
}

static inline uchar apply_binary_u8(uint op, uchar lhs, uchar rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
        return uchar(uint(lhs) + uint(rhs));
    case MARMOT_DEVICE_BINARY_SUB:
        return uchar(uint(lhs) - uint(rhs));
    case MARMOT_DEVICE_BINARY_MUL:
        return uchar(uint(lhs) * uint(rhs));
    case MARMOT_DEVICE_BINARY_DIV:
        return uchar(uint(lhs) / uint(rhs));
    case MARMOT_DEVICE_BINARY_MIN:
        return lhs < rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_MAX:
        return lhs > rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_POW:
        return uchar(int_pow_unsigned<uint>(uint(lhs), uint(rhs)));
    case MARMOT_DEVICE_BINARY_MOD:
        return rhs == 0 ? uchar(0) : uchar(uint(lhs) % uint(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_AND:
        return uchar(uint(lhs) & uint(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_OR:
        return uchar(uint(lhs) | uint(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_XOR:
        return uchar(uint(lhs) ^ uint(rhs));
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT: {
        uint amount = normalize_shift_unsigned(uint(rhs), 8u);
        if (amount >= 8u) {
            return uchar(0);
        }
        return uchar(uint(lhs) << amount);
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT:
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL: {
        uint amount = normalize_shift_unsigned(uint(rhs), 8u);
        if (amount >= 8u) {
            return uchar(0);
        }
        return uchar(uint(lhs) >> amount);
    }
    default:
        return (uchar)0;
    }
}

static inline ulong apply_binary_u64(uint op, ulong lhs, ulong rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_ADD:
        return lhs + rhs;
    case MARMOT_DEVICE_BINARY_SUB:
        return lhs - rhs;
    case MARMOT_DEVICE_BINARY_MUL:
        return lhs * rhs;
    case MARMOT_DEVICE_BINARY_DIV:
        return lhs / rhs;
    case MARMOT_DEVICE_BINARY_MIN:
        return lhs < rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_MAX:
        return lhs > rhs ? lhs : rhs;
    case MARMOT_DEVICE_BINARY_POW:
        return int_pow_unsigned<ulong>(lhs, rhs);
    case MARMOT_DEVICE_BINARY_MOD:
        return rhs == 0ul ? 0ul : lhs % rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_AND:
        return lhs & rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_OR:
        return lhs | rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_XOR:
        return lhs ^ rhs;
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT: {
        uint amount = normalize_shift_unsigned((uint)min(rhs, (ulong)UINT_MAX), 64u);
        if (amount >= 64u) {
            return 0ul;
        }
        return lhs << amount;
    }
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT:
    case MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL: {
        uint amount = normalize_shift_unsigned((uint)min(rhs, (ulong)UINT_MAX), 64u);
        if (amount >= 64u) {
            return 0ul;
        }
        return lhs >> amount;
    }
    default:
        return 0ul;
    }
}

static inline uchar apply_binary_compare_float(uint op, float lhs, float rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
        return (uchar)(lhs == rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
        return (uchar)(lhs != rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
        return (uchar)(lhs < rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
        return (uchar)(lhs <= rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
        return (uchar)(lhs > rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return (uchar)(lhs >= rhs);
    default:
        return (uchar)0;
    }
}

static inline uchar apply_binary_compare_i32(uint op, int lhs, int rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
        return (uchar)(lhs == rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
        return (uchar)(lhs != rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
        return (uchar)(lhs < rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
        return (uchar)(lhs <= rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
        return (uchar)(lhs > rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return (uchar)(lhs >= rhs);
    default:
        return (uchar)0;
    }
}

static inline uchar apply_binary_compare_i16(uint op, short lhs, short rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
        return (uchar)(lhs == rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
        return (uchar)(lhs != rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
        return (uchar)(lhs < rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
        return (uchar)(lhs <= rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
        return (uchar)(lhs > rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return (uchar)(lhs >= rhs);
    default:
        return (uchar)0;
    }
}

static inline uchar apply_binary_compare_i8(uint op, char lhs, char rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
        return (uchar)(lhs == rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
        return (uchar)(lhs != rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
        return (uchar)(lhs < rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
        return (uchar)(lhs <= rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
        return (uchar)(lhs > rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return (uchar)(lhs >= rhs);
    default:
        return (uchar)0;
    }
}

static inline uchar apply_binary_compare_u32(uint op, uint lhs, uint rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
        return (uchar)(lhs == rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
        return (uchar)(lhs != rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
        return (uchar)(lhs < rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
        return (uchar)(lhs <= rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
        return (uchar)(lhs > rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return (uchar)(lhs >= rhs);
    default:
        return (uchar)0;
    }
}

static inline uchar apply_binary_compare_u16(uint op, ushort lhs, ushort rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
        return (uchar)(lhs == rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
        return (uchar)(lhs != rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
        return (uchar)(lhs < rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
        return (uchar)(lhs <= rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
        return (uchar)(lhs > rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return (uchar)(lhs >= rhs);
    default:
        return (uchar)0;
    }
}

static inline uchar apply_binary_compare_u8(uint op, uchar lhs, uchar rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
        return (uchar)(lhs == rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
        return (uchar)(lhs != rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
        return (uchar)(lhs < rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
        return (uchar)(lhs <= rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
        return (uchar)(lhs > rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return (uchar)(lhs >= rhs);
    default:
        return (uchar)0;
    }
}

static inline uchar apply_binary_compare_u64(uint op, ulong lhs, ulong rhs) {
    switch (op) {
    case MARMOT_DEVICE_BINARY_COMPARE_EQ:
        return (uchar)(lhs == rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_NE:
        return (uchar)(lhs != rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LT:
        return (uchar)(lhs < rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_LE:
        return (uchar)(lhs <= rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GT:
        return (uchar)(lhs > rhs);
    case MARMOT_DEVICE_BINARY_COMPARE_GE:
        return (uchar)(lhs >= rhs);
    default:
        return (uchar)0;
    }
}

kernel void elementwise_arith_f32(
    constant float *a [[buffer(0)]], constant float *b [[buffer(1)]], device float *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_float(a[id]);
    float rhs = read_float(b[id]);
    out[id] = write_float(apply_binary_float(op, lhs, rhs));
}

kernel void elementwise_arith_f32_vec4(
    constant float4 *a [[buffer(0)]], constant float4 *b [[buffer(1)]], device float4 *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], constant uint &total_vec4 [[buffer(4)]], uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    float4 lhs = a[id];
    float4 rhs = b[id];
    out[id] = float4(
        apply_binary_float(op, lhs.x, rhs.x), apply_binary_float(op, lhs.y, rhs.y),
        apply_binary_float(op, lhs.z, rhs.z), apply_binary_float(op, lhs.w, rhs.w)
    );
}

kernel void elementwise_arith_f16(
    constant half *a [[buffer(0)]], constant half *b [[buffer(1)]], device half *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_half(a[id]);
    float rhs = read_half(b[id]);
    float value = apply_binary_float(op, lhs, rhs);
    out[id] = write_half(value);
}

kernel void elementwise_arith_f16_vec4(
    constant half4 *a [[buffer(0)]], constant half4 *b [[buffer(1)]], device half4 *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], constant uint &total_vec4 [[buffer(4)]], uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    half4 lhs_h = a[id];
    half4 rhs_h = b[id];
    float4 lhs = float4(lhs_h);
    float4 rhs = float4(rhs_h);
    float4 result = float4(
        apply_binary_float(op, lhs.x, rhs.x), apply_binary_float(op, lhs.y, rhs.y),
        apply_binary_float(op, lhs.z, rhs.z), apply_binary_float(op, lhs.w, rhs.w)
    );
    out[id] = half4(result);
}

kernel void elementwise_arith_bf16(
    constant ushort *a [[buffer(0)]], constant ushort *b [[buffer(1)]], device ushort *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_bf16(a[id]);
    float rhs = read_bf16(b[id]);
    float value = apply_binary_float(op, lhs, rhs);
    out[id] = write_bf16(value);
}

kernel void elementwise_arith_bf16_vec4(
    constant ushort4 *a [[buffer(0)]], constant ushort4 *b [[buffer(1)]], device ushort4 *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], constant uint &total_vec4 [[buffer(4)]], uint id [[thread_position_in_grid]]
) {
    if (id >= total_vec4)
        return;
    ushort4 lhs_bf = a[id];
    ushort4 rhs_bf = b[id];
    float4 result = float4(
        apply_binary_float(op, read_bf16(lhs_bf.x), read_bf16(rhs_bf.x)),
        apply_binary_float(op, read_bf16(lhs_bf.y), read_bf16(rhs_bf.y)),
        apply_binary_float(op, read_bf16(lhs_bf.z), read_bf16(rhs_bf.z)),
        apply_binary_float(op, read_bf16(lhs_bf.w), read_bf16(rhs_bf.w))
    );
    out[id] = ushort4(write_bf16(result.x), write_bf16(result.y), write_bf16(result.z), write_bf16(result.w));
}

#define DEFINE_ELEMENTWISE_ARITH_ROW(NAME, VALUE_T, READ_FN, WRITE_FN)                                                 \
    kernel void NAME(                                                                                                  \
        device const VALUE_T *a [[buffer(0)]], device const VALUE_T *b [[buffer(1)]],                                  \
        device VALUE_T *out [[buffer(2)]], constant uint &op [[buffer(3)]], constant uint &rows [[buffer(4)]],         \
        constant uint &cols [[buffer(5)]], constant size_t &a_row_stride [[buffer(6)]],                                \
        constant size_t &b_row_stride [[buffer(7)]], constant size_t &out_row_stride [[buffer(8)]],                    \
        uint id [[thread_position_in_grid]]                                                                            \
    ) {                                                                                                                \
        uint total = rows * cols;                                                                                      \
        if (id >= total) {                                                                                             \
            return;                                                                                                    \
        }                                                                                                              \
        uint row = id / cols;                                                                                          \
        uint col = id - row * cols;                                                                                    \
        size_t a_idx = elem_to_loc_row_strided<size_t>((size_t)row, (size_t)col, a_row_stride);                        \
        size_t b_idx = elem_to_loc_row_strided<size_t>((size_t)row, (size_t)col, b_row_stride);                        \
        size_t out_idx = elem_to_loc_row_strided<size_t>((size_t)row, (size_t)col, out_row_stride);                    \
        float lhs = READ_FN(a[a_idx]);                                                                                 \
        float rhs = READ_FN(b[b_idx]);                                                                                 \
        out[out_idx] = WRITE_FN(apply_binary_float(op, lhs, rhs));                                                     \
    }

DEFINE_ELEMENTWISE_ARITH_ROW(elementwise_arith_f32_row, float, read_float, write_float)
DEFINE_ELEMENTWISE_ARITH_ROW(elementwise_arith_f16_row, half, read_half, write_half)
DEFINE_ELEMENTWISE_ARITH_ROW(elementwise_arith_bf16_row, ushort, read_bf16, write_bf16)
#if MARMOT_ENABLE_FP8
DEFINE_ELEMENTWISE_ARITH_ROW(elementwise_arith_fp8_e4m3_row, uchar, read_fp8_e4m3, write_fp8_e4m3)
DEFINE_ELEMENTWISE_ARITH_ROW(elementwise_arith_fp8_e5m2_row, uchar, read_fp8_e5m2, write_fp8_e5m2)
#endif

#undef DEFINE_ELEMENTWISE_ARITH_ROW

#if MARMOT_ENABLE_FP8
kernel void elementwise_arith_fp8_e4m3(
    constant uchar *a [[buffer(0)]], constant uchar *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_fp8_e4m3(a[id]);
    float rhs = read_fp8_e4m3(b[id]);
    float value = apply_binary_float(op, lhs, rhs);
    out[id] = write_fp8_e4m3(value);
}

kernel void elementwise_arith_fp8_e5m2(
    constant uchar *a [[buffer(0)]], constant uchar *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_fp8_e5m2(a[id]);
    float rhs = read_fp8_e5m2(b[id]);
    float value = apply_binary_float(op, lhs, rhs);
    out[id] = write_fp8_e5m2(value);
}
#endif

kernel void elementwise_arith_i32(
    constant int *a [[buffer(0)]], constant int *b [[buffer(1)]], device int *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_i32(op, a[id], b[id]);
}

kernel void elementwise_arith_i16(
    constant short *a [[buffer(0)]], constant short *b [[buffer(1)]], device short *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_i16(op, a[id], b[id]);
}

kernel void elementwise_arith_i8(
    constant char *a [[buffer(0)]], constant char *b [[buffer(1)]], device char *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_i8(op, a[id], b[id]);
}

kernel void elementwise_arith_u32(
    constant uint *a [[buffer(0)]], constant uint *b [[buffer(1)]], device uint *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_u32(op, a[id], b[id]);
}

kernel void elementwise_arith_u16(
    constant ushort *a [[buffer(0)]], constant ushort *b [[buffer(1)]], device ushort *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_u16(op, a[id], b[id]);
}

kernel void elementwise_arith_u8(
    constant uchar *a [[buffer(0)]], constant uchar *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_u8(op, a[id], b[id]);
}

kernel void elementwise_arith_u64(
    constant ulong *a [[buffer(0)]], constant ulong *b [[buffer(1)]], device ulong *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_u64(op, a[id], b[id]);
}

kernel void elementwise_compare_f32(
    constant float *a [[buffer(0)]], constant float *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_float(a[id]);
    float rhs = read_float(b[id]);
    out[id] = apply_binary_compare_float(op, lhs, rhs);
}

kernel void elementwise_compare_f16(
    constant half *a [[buffer(0)]], constant half *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_half(a[id]);
    float rhs = read_half(b[id]);
    out[id] = apply_binary_compare_float(op, lhs, rhs);
}

kernel void elementwise_compare_bf16(
    constant ushort *a [[buffer(0)]], constant ushort *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_bf16(a[id]);
    float rhs = read_bf16(b[id]);
    out[id] = apply_binary_compare_float(op, lhs, rhs);
}

#if MARMOT_ENABLE_FP8
kernel void elementwise_compare_fp8_e4m3(
    constant uchar *a [[buffer(0)]], constant uchar *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_fp8_e4m3(a[id]);
    float rhs = read_fp8_e4m3(b[id]);
    out[id] = apply_binary_compare_float(op, lhs, rhs);
}

kernel void elementwise_compare_fp8_e5m2(
    constant uchar *a [[buffer(0)]], constant uchar *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float lhs = read_fp8_e5m2(a[id]);
    float rhs = read_fp8_e5m2(b[id]);
    out[id] = apply_binary_compare_float(op, lhs, rhs);
}
#endif

kernel void elementwise_compare_i32(
    constant int *a [[buffer(0)]], constant int *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_compare_i32(op, a[id], b[id]);
}

kernel void elementwise_compare_i16(
    constant short *a [[buffer(0)]], constant short *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_compare_i16(op, a[id], b[id]);
}

kernel void elementwise_compare_i8(
    constant char *a [[buffer(0)]], constant char *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_compare_i8(op, a[id], b[id]);
}

kernel void elementwise_compare_u32(
    constant uint *a [[buffer(0)]], constant uint *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_compare_u32(op, a[id], b[id]);
}

kernel void elementwise_compare_u16(
    constant ushort *a [[buffer(0)]], constant ushort *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_compare_u16(op, a[id], b[id]);
}

kernel void elementwise_compare_u8(
    constant uchar *a [[buffer(0)]], constant uchar *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_compare_u8(op, a[id], b[id]);
}

kernel void elementwise_compare_u64(
    constant ulong *a [[buffer(0)]], constant ulong *b [[buffer(1)]], device uchar *out [[buffer(2)]],
    constant uint &op [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = apply_binary_compare_u64(op, a[id], b[id]);
}
// -----------------------------------------------------------------------------

kernel void fma_f32(
    constant float *a [[buffer(0)]], constant float *b [[buffer(1)]], constant float *c [[buffer(2)]],
    device float *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = fma(a[id], b[id], c[id]);
}

kernel void fma_f16(
    constant half *a [[buffer(0)]], constant half *b [[buffer(1)]], constant half *c [[buffer(2)]],
    device half *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float av = read_half(a[id]);
    float bv = read_half(b[id]);
    float cv = read_half(c[id]);
    float result = fma(av, bv, cv);
    out[id] = write_half(result);
}

kernel void fma_bf16(
    constant ushort *a [[buffer(0)]], constant ushort *b [[buffer(1)]], constant ushort *c [[buffer(2)]],
    device ushort *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float av = read_bf16(a[id]);
    float bv = read_bf16(b[id]);
    float cv = read_bf16(c[id]);
    float result = fma(av, bv, cv);
    out[id] = write_bf16(result);
}
#if MARMOT_ENABLE_FP8
kernel void fma_fp8_e4m3(
    constant uchar *a [[buffer(0)]], constant uchar *b [[buffer(1)]], constant uchar *c [[buffer(2)]],
    device uchar *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float av = read_fp8_e4m3(a[id]);
    float bv = read_fp8_e4m3(b[id]);
    float cv = read_fp8_e4m3(c[id]);
    float result = fma(av, bv, cv);
    out[id] = write_fp8_e4m3(result);
}

kernel void fma_fp8_e5m2(
    constant uchar *a [[buffer(0)]], constant uchar *b [[buffer(1)]], constant uchar *c [[buffer(2)]],
    device uchar *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    float av = read_fp8_e5m2(a[id]);
    float bv = read_fp8_e5m2(b[id]);
    float cv = read_fp8_e5m2(c[id]);
    float result = fma(av, bv, cv);
    out[id] = write_fp8_e5m2(result);
}
#endif

kernel void where_metal_f32(
    constant uchar *mask [[buffer(0)]], constant float *a [[buffer(1)]], constant float *b [[buffer(2)]],
    device float *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = mask[id] ? a[id] : b[id];
}

kernel void where_metal_f16(
    constant uchar *mask [[buffer(0)]], constant half *a [[buffer(1)]], constant half *b [[buffer(2)]],
    device half *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = mask[id] ? a[id] : b[id];
}

kernel void where_metal_bf16(
    constant uchar *mask [[buffer(0)]], constant ushort *a [[buffer(1)]], constant ushort *b [[buffer(2)]],
    device ushort *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = mask[id] ? a[id] : b[id];
}

#if MARMOT_ENABLE_FP8
kernel void where_metal_fp8_e4m3(
    constant uchar *mask [[buffer(0)]], constant uchar *a [[buffer(1)]], constant uchar *b [[buffer(2)]],
    device uchar *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = mask[id] ? a[id] : b[id];
}

kernel void where_metal_fp8_e5m2(
    constant uchar *mask [[buffer(0)]], constant uchar *a [[buffer(1)]], constant uchar *b [[buffer(2)]],
    device uchar *out [[buffer(3)]], uint id [[thread_position_in_grid]]
) {
    out[id] = mask[id] ? a[id] : b[id];
}
#endif
