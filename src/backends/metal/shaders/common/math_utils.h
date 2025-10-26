#pragma once

static inline float make_nan() {
    return as_type<float>(0x7fc00000u);
}

static inline float erf_approx(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabs(x);
    const float p = 0.3275911f;
    const float a1 = 0.254829592f;
    const float a2 = -0.284496736f;
    const float a3 = 1.421413741f;
    const float a4 = -1.453152027f;
    const float a5 = 1.061405429f;
    float t = 1.0f / (1.0f + p * x);
    float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
    return sign * y;
}

static inline float abs_scalar_exact(float x) {
    return fabs(x);
}

static inline float neg_scalar_exact(float x) {
    return -x;
}

static inline float sign_scalar_exact(float x) {
    if (x > 0.0f) {
        return 1.0f;
    }
    if (x < 0.0f) {
        return -1.0f;
    }
    return 0.0f;
}

static inline float sqrt_scalar_exact(float x) {
    return sqrt(x);
}

static inline float exp_scalar_exact(float x) {
    return exp(x);
}

static inline float log_scalar_exact(float x) {
    return log(x);
}

template <typename T>
static inline T int_pow_signed(T base, T exp) {
    int e = (int)exp;
    if (e < 0) {
        if (base == (T)1) {
            return (T)1;
        }
        if (base == (T)-1) {
            return (T)((e & 1) ? -1 : 1);
        }
        return (T)0;
    }
    T result = (T)1;
    T factor = base;
    while (e > 0) {
        if (e & 1) {
            result = (T)(result * factor);
        }
        e >>= 1;
        if (e != 0) {
            factor = (T)(factor * factor);
        }
    }
    return result;
}

template <typename T>
static inline T int_pow_unsigned(T base, T exp) {
    unsigned int e = (unsigned int)exp;
    T result = (T)1;
    T factor = base;
    while (e > 0) {
        if (e & 1u) {
            result = (T)(result * factor);
        }
        e >>= 1u;
        if (e != 0u) {
            factor = (T)(factor * factor);
        }
    }
    return result;
}

static inline uint normalize_shift_signed(int rhs, uint bits) {
    if (rhs <= 0) {
        return 0u;
    }
    uint amount = (uint)rhs;
    return amount >= bits ? bits : amount;
}

static inline uint normalize_shift_unsigned(uint rhs, uint bits) {
    return rhs >= bits ? bits : rhs;
}

template <typename T>
static inline T abs_int_exact(T x) {
    const uint bits = sizeof(T) * 8;
    const uint mask = 1u << (bits - 1);
    T min_val = (T)(-((int)mask));
    if (x == min_val) {
        return x;
    }
    return x < (T)0 ? (T)(-x) : x;
}

template <typename T>
static inline T negate_int_exact(T x) {
    const uint bits = sizeof(T) * 8;
    const uint mask = 1u << (bits - 1);
    T min_val = (T)(-((int)mask));
    if (x == min_val) {
        return x;
    }
    return (T)(-x);
}

template <typename T>
static inline T sign_int_exact(T x) {
    return (T)((x > (T)0) - (x < (T)0));
}

template <typename T>
static inline T sign_uint_exact(T x) {
    return x > (T)0 ? (T)1 : (T)0;
}
