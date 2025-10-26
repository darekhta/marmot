#ifndef CPU_NEON_MATH_H
#define CPU_NEON_MATH_H

#include "cpu_backend_internal.h"

#if HAS_NEON

static inline float32x4_t cpu_neon_floor_vec(float32x4_t x) {
#if defined(__aarch64__)
    return vrndmq_f32(x);
#else
    int32x4_t xi = vcvtq_s32_f32(x);
    float32x4_t xf = vcvtq_f32_s32(xi);
    uint32x4_t mask = vcgtq_f32(xf, x);
    float32x4_t one = vdupq_n_f32(1.0f);
    float32x4_t xf_minus_one = vsubq_f32(xf, one);
    return vbslq_f32(mask, xf_minus_one, xf);
#endif
}

static inline float32x4_t cpu_neon_div_vec(float32x4_t num, float32x4_t den) {
#if defined(__aarch64__)
    return vdivq_f32(num, den);
#else
    float32x4_t recip = vrecpeq_f32(den);
    recip = vmulq_f32(vrecpsq_f32(den, recip), recip);
    recip = vmulq_f32(vrecpsq_f32(den, recip), recip);
    return vmulq_f32(num, recip);
#endif
}

// NEON exp approximation based on Cephes polynomial coefficients.
// Original: http://gruntthepeon.free.fr/ssemath/ (zlib license)
static inline float32x4_t cpu_neon_exp_vec(float32x4_t x) {
    const float32x4_t exp_hi = vdupq_n_f32(88.3762626647949f);
    const float32x4_t exp_lo = vdupq_n_f32(-88.3762626647949f);
    const float32x4_t log2ef = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t exp_c1 = vdupq_n_f32(0.693359375f);
    const float32x4_t exp_c2 = vdupq_n_f32(-2.12194440e-4f);

    x = vminq_f32(x, exp_hi);
    x = vmaxq_f32(x, exp_lo);

    float32x4_t fx = vmlaq_f32(vdupq_n_f32(0.5f), x, log2ef);
    fx = cpu_neon_floor_vec(fx);

    float32x4_t g = vmlsq_f32(x, fx, exp_c1);
    g = vmlsq_f32(g, fx, exp_c2);

    float32x4_t z = vmulq_f32(g, g);
    float32x4_t y = vdupq_n_f32(1.9875691500E-4f);
    y = vmlaq_f32(vdupq_n_f32(1.3981999507E-3f), g, y);
    y = vmlaq_f32(vdupq_n_f32(8.3334519073E-3f), g, y);
    y = vmlaq_f32(vdupq_n_f32(4.1665795894E-2f), g, y);
    y = vmlaq_f32(vdupq_n_f32(1.6666665459E-1f), g, y);
    y = vmlaq_f32(vdupq_n_f32(5.0000001201E-1f), g, y);
    y = vmlaq_f32(g, z, y);
    y = vaddq_f32(y, vdupq_n_f32(1.0f));

    int32x4_t mm = vcvtq_s32_f32(fx);
    mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
    mm = vshlq_n_s32(mm, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(mm);

    return vmulq_f32(y, pow2n);
}

#endif // HAS_NEON

#endif // CPU_NEON_MATH_H
