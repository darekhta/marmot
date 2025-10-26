#pragma once

static inline float bf16_to_float(ushort bits) {
    uint value = (uint)bits << 16;
    return as_type<float>(value);
}

static inline ushort float_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    uint lsb = (bits >> 16) & 1u;
    uint rounding = 0x7FFFu + lsb;
    bits += rounding;
    return (ushort)(bits >> 16);
}

static inline uchar
fp8_quantize(float value, int exp_bits, int mant_bits, int bias, bool has_infinity, float max_finite) {
    if (isnan(value)) {
        uchar payload = 1u;
        uchar exp_mask = (uchar)((1u << exp_bits) - 1u);
        return (uchar)(exp_mask << mant_bits | payload);
    }

    uint sign = value < 0.0f ? 1u : 0u;
    float abs_value = fabs(value);
    uchar sign_bit = (uchar)(sign << 7);
    uchar exp_mask = (uchar)((1u << exp_bits) - 1u);
    uchar max_exponent = (uchar)(exp_mask - 1u);
    uchar mant_mask = (uchar)((1u << mant_bits) - 1u);

    if (isinf(value) || abs_value > max_finite) {
        if (has_infinity) {
            return (uchar)(sign_bit | (exp_mask << mant_bits));
        }
        return (uchar)(sign_bit | (max_exponent << mant_bits) | mant_mask);
    }

    if (abs_value == 0.0f) {
        return sign_bit;
    }

    int exponent = 0;
    float mantissa = frexp(abs_value, exponent);
    mantissa *= 2.0f;
    exponent -= 1;

    int fp_exp = exponent + bias;
    if (fp_exp > 0 && fp_exp < (int)max_exponent + 1) {
        float scaled = (mantissa - 1.0f) * (float)(1 << mant_bits);
        int rounded = int(rint(scaled));
        if (rounded == (1 << mant_bits)) {
            rounded = 0;
            fp_exp += 1;
            if (fp_exp >= (int)max_exponent + 1) {
                if (has_infinity) {
                    return (uchar)(sign_bit | (exp_mask << mant_bits));
                }
                return (uchar)(sign_bit | (max_exponent << mant_bits) | mant_mask);
            }
        }
        return (uchar)(sign_bit | ((uchar)fp_exp << mant_bits) | (uchar)(rounded & mant_mask));
    }

    int emin = 1 - bias;
    float scaled = ldexp(abs_value, -(emin) + mant_bits);
    int rounded = int(rint(scaled));
    if (rounded == 0) {
        return sign_bit;
    }
    if (rounded > mant_mask) {
        rounded = mant_mask;
    }
    return (uchar)(sign_bit | (uchar)(rounded & mant_mask));
}

static inline float fp8_dequantize(uchar bits, int exp_bits, int mant_bits, int bias, bool has_infinity) {
    uint sign = (bits >> 7) & 0x1;
    uint exp = (bits >> mant_bits) & ((1u << exp_bits) - 1u);
    uint mant = bits & ((1u << mant_bits) - 1u);
    uint exp_mask = (1u << exp_bits) - 1u;
    float sign_scale = sign ? -1.0f : 1.0f;

    if (exp == exp_mask) {
        if (has_infinity && mant == 0u) {
            return sign ? -INFINITY : INFINITY;
        }
        return make_nan();
    }

    if (exp == 0) {
        if (mant == 0u) {
            return sign ? -0.0f : 0.0f;
        }
        float fraction = (float)mant / (float)(1 << mant_bits);
        int emin = 1 - bias;
        return sign_scale * ldexp(fraction, emin);
    }

    float fraction = 1.0f + ((float)mant / (float)(1 << mant_bits));
    int real_exp = int(exp) - bias;
    return sign_scale * ldexp(fraction, real_exp);
}

static inline uchar float_to_fp8_e4m3(float value) {
    const int exp_bits = 4;
    const int mant_bits = 3;
    const int bias = 7;
    const float max_finite = 240.0f;
    return fp8_quantize(value, exp_bits, mant_bits, bias, false, max_finite);
}

static inline float fp8_e4m3_to_float(uchar bits) {
    const int exp_bits = 4;
    const int mant_bits = 3;
    const int bias = 7;
    return fp8_dequantize(bits, exp_bits, mant_bits, bias, false);
}

static inline uchar float_to_fp8_e5m2(float value) {
    const int exp_bits = 5;
    const int mant_bits = 2;
    const int bias = 15;
    const float max_finite = 57344.0f;
    return fp8_quantize(value, exp_bits, mant_bits, bias, true, max_finite);
}

static inline float fp8_e5m2_to_float(uchar bits) {
    const int exp_bits = 5;
    const int mant_bits = 2;
    const int bias = 15;
    return fp8_dequantize(bits, exp_bits, mant_bits, bias, true);
}

static inline float read_float(float value) {
    return value;
}

static inline float write_float(float value) {
    return value;
}

static inline float read_half(half value) {
    return float(value);
}

static inline half write_half(float value) {
    return half(value);
}

static inline float read_bf16(ushort bits) {
    return bf16_to_float(bits);
}

static inline ushort write_bf16(float value) {
    return float_to_bf16(value);
}

static inline float read_fp8_e4m3(uchar bits) {
    return fp8_e4m3_to_float(bits);
}

static inline uchar write_fp8_e4m3(float value) {
    return float_to_fp8_e4m3(value);
}

static inline float read_fp8_e5m2(uchar bits) {
    return fp8_e5m2_to_float(bits);
}

static inline uchar write_fp8_e5m2(float value) {
    return float_to_fp8_e5m2(value);
}
