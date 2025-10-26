#include "dtype_ref.h"

#include <stdbool.h>
#include <stdint.h>

#include <math.h>
#include <string.h>

// ===================================================================
// Scalar Reference Implementations - NO SIMD
// ===================================================================

// FLOAT32 -> FLOAT16 (IEEE 754 half precision)
marmot_float16_t marmot_f32_to_f16_ref(float value) {
    uint32_t f32_bits;
    memcpy(&f32_bits, &value, sizeof(float));

    uint32_t sign = (f32_bits >> 16) & 0x8000;
    uint32_t exponent = (f32_bits >> 23) & 0xFF;
    uint32_t mantissa = f32_bits & 0x7FFFFF;

    uint16_t result_bits;

    // Handle special cases
    if (exponent == 0xFF) { // Inf or NaN
        if (mantissa == 0) {
            result_bits = (uint16_t)(sign | 0x7C00); // Inf
        } else {
            result_bits = (uint16_t)(sign | 0x7C00 | (mantissa >> 13)); // NaN
        }
    } else {
        int32_t new_exp = (int32_t)exponent - 127 + 15; // Rebias exponent

        if (new_exp <= 0) { // Underflow -> denormal or zero
            if (new_exp < -10) {
                result_bits = (uint16_t)sign; // Too small, flush to zero
            } else {
                mantissa = (mantissa | 0x800000) >> (1 - new_exp);
                result_bits = (uint16_t)(sign | (mantissa >> 13));
            }
        } else if (new_exp >= 0x1F) { // Overflow -> infinity
            result_bits = (uint16_t)(sign | 0x7C00);
        } else {
            // Normal case
            result_bits = (uint16_t)(sign | (new_exp << 10) | (mantissa >> 13));
        }
    }

    return (marmot_float16_t){.bits = result_bits};
}

// FLOAT16 -> FLOAT32
float marmot_f16_to_f32_ref(marmot_float16_t value) {
    uint16_t bits = value.bits;
    uint32_t sign = ((uint32_t)(bits & 0x8000u)) << 16;
    uint32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x3FF;

    uint32_t f32_bits;

    if (exponent == 0) { // Zero or denormal
        if (mantissa == 0) {
            f32_bits = sign; // Zero
        } else {
            // Denormal - normalize it
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f32_bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) { // Inf or NaN
        f32_bits = sign | 0x7F800000 | (mantissa << 13);
    } else { // Normal
        f32_bits = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }

    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// FLOAT32 -> BFLOAT16 (truncate mantissa)
marmot_bfloat16_t marmot_f32_to_bf16_ref(float value) {
    uint32_t f32_bits;
    memcpy(&f32_bits, &value, sizeof(float));

    // Round to nearest even (RNE)
    uint32_t rounding_bias = 0x7FFF + ((f32_bits >> 16) & 1);
    uint32_t rounded = f32_bits + rounding_bias;

    // Truncate to bfloat16 (upper 16 bits of float32)
    return (marmot_bfloat16_t){.bits = (uint16_t)(rounded >> 16)};
}

// BFLOAT16 -> FLOAT32 (zero-extend mantissa)
float marmot_bf16_to_f32_ref(marmot_bfloat16_t value) {
    uint32_t f32_bits = ((uint32_t)value.bits) << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

// FLOAT16 -> BFLOAT16 (via FLOAT32)
marmot_bfloat16_t marmot_f16_to_bf16_ref(marmot_float16_t value) {
    return marmot_f32_to_bf16_ref(marmot_f16_to_f32_ref(value));
}

// BFLOAT16 -> FLOAT16 (via FLOAT32)
marmot_float16_t marmot_bf16_to_f16_ref(marmot_bfloat16_t value) {
    return marmot_f32_to_f16_ref(marmot_bf16_to_f32_ref(value));
}

#if MARMOT_ENABLE_FP8

static inline uint8_t
marmot_fp8_quantize(float value, int exp_bits, int mant_bits, int bias, bool has_infinity, float max_finite) {
    if (isnan(value)) {
        uint8_t payload = 1u;
        uint8_t exp_mask = (uint8_t)((1u << exp_bits) - 1u);
        return (uint8_t)((0u << 7) | (exp_mask << mant_bits) | payload);
    }

    uint32_t sign = value < 0 ? 1u : 0u;
    float abs_value = fabsf(value);
    uint8_t sign_bit = (uint8_t)(sign << 7);
    uint8_t exp_mask = (uint8_t)((1u << exp_bits) - 1u);
    uint8_t max_exponent = (uint8_t)(exp_mask - 1u);
    uint8_t mant_mask = (uint8_t)((1u << mant_bits) - 1u);

    if (isinf(value) || abs_value > max_finite) {
        if (has_infinity) {
            return (uint8_t)(sign_bit | (exp_mask << mant_bits));
        }
        return (uint8_t)(sign_bit | (max_exponent << mant_bits) | mant_mask);
    }

    if (abs_value == 0.0f) {
        return sign_bit;
    }

    int exponent;
    float mantissa = frexpf(abs_value, &exponent); // abs_value = mantissa * 2^exponent, mantissa in [0.5, 1)
    mantissa *= 2.0f;
    exponent -= 1;

    int fp_exp = exponent + bias;
    if (fp_exp > 0 && fp_exp < (int)max_exponent + 1) {
        const float scaled = (mantissa - 1.0f) * (float)(1 << mant_bits);
        int rounded = (int)lrintf(scaled);
        if (rounded == (1 << mant_bits)) {
            rounded = 0;
            fp_exp += 1;
            if (fp_exp >= (int)max_exponent + 1) {
                if (has_infinity) {
                    return (uint8_t)(sign_bit | (exp_mask << mant_bits));
                }
                return (uint8_t)(sign_bit | (max_exponent << mant_bits) | mant_mask);
            }
        }
        return (uint8_t)(sign_bit | ((uint8_t)fp_exp << mant_bits) | (uint8_t)(rounded & mant_mask));
    }

    // Subnormal handling
    const int emin = 1 - bias;
    float scaled = ldexpf(abs_value, -(emin) + mant_bits);
    int rounded = (int)lrintf(scaled);
    if (rounded == 0) {
        return sign_bit;
    }
    if (rounded > mant_mask) {
        rounded = mant_mask;
    }
    return (uint8_t)(sign_bit | (uint8_t)(rounded & mant_mask));
}

static inline float marmot_fp8_dequantize(uint8_t bits, int exp_bits, int mant_bits, int bias, bool has_infinity) {
    uint8_t sign = (bits >> 7) & 0x1;
    uint8_t exp = (bits >> mant_bits) & ((1u << exp_bits) - 1u);
    uint8_t mant = bits & ((1u << mant_bits) - 1u);
    uint8_t exp_mask = (uint8_t)((1u << exp_bits) - 1u);
    float sign_scale = sign ? -1.0f : 1.0f;

    if (exp == exp_mask) {
        if (has_infinity && mant == 0) {
            return sign ? -INFINITY : INFINITY;
        }
        return nanf("");
    }

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        float fraction = (float)mant / (float)(1 << mant_bits);
        int emin = 1 - bias;
        return sign_scale * ldexpf(fraction, emin);
    }

    float fraction = 1.0f + ((float)mant / (float)(1 << mant_bits));
    int real_exp = (int)exp - bias;
    return sign_scale * ldexpf(fraction, real_exp);
}

marmot_float8_e4m3_t marmot_f32_to_fp8_e4m3_ref(float value) {
    const int exp_bits = 4;
    const int mant_bits = 3;
    const int bias = 7;
    const float max_finite = 240.0f; // (1.875 * 2^7)
    uint8_t encoded = marmot_fp8_quantize(value, exp_bits, mant_bits, bias, false, max_finite);
    return (marmot_float8_e4m3_t){.bits = encoded};
}

float marmot_fp8_e4m3_to_f32_ref(marmot_float8_e4m3_t value) {
    const int exp_bits = 4;
    const int mant_bits = 3;
    const int bias = 7;
    return marmot_fp8_dequantize(value.bits, exp_bits, mant_bits, bias, false);
}

marmot_float8_e5m2_t marmot_f32_to_fp8_e5m2_ref(float value) {
    const int exp_bits = 5;
    const int mant_bits = 2;
    const int bias = 15;
    const float max_finite = 57344.0f; // (1.75 * 2^15)
    uint8_t encoded = marmot_fp8_quantize(value, exp_bits, mant_bits, bias, true, max_finite);
    return (marmot_float8_e5m2_t){.bits = encoded};
}

float marmot_fp8_e5m2_to_f32_ref(marmot_float8_e5m2_t value) {
    const int exp_bits = 5;
    const int mant_bits = 2;
    const int bias = 15;
    return marmot_fp8_dequantize(value.bits, exp_bits, mant_bits, bias, true);
}

#endif // MARMOT_ENABLE_FP8
