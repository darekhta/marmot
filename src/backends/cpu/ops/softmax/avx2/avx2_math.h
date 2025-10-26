#ifndef CPU_SOFTMAX_AVX2_MATH_H
#define CPU_SOFTMAX_AVX2_MATH_H

#include "cpu_backend_internal.h"

#if HAS_AVX2
static inline __m256 cpu_avx2_exp_vec(__m256 x) {
    const __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);
    const __m256 log2ef = _mm256_set1_ps(1.44269504088896341f);
    const __m256 exp_c1 = _mm256_set1_ps(0.693359375f);
    const __m256 exp_c2 = _mm256_set1_ps(-2.12194440e-4f);

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    __m256 fx = _mm256_fmadd_ps(x, log2ef, _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);

    __m256 g = _mm256_fnmadd_ps(fx, exp_c1, x);
    g = _mm256_fnmadd_ps(fx, exp_c2, g);

    __m256 z = _mm256_mul_ps(g, g);
    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, g, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, z, g);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));

    __m256i mm = _mm256_cvtps_epi32(fx);
    mm = _mm256_add_epi32(mm, _mm256_set1_epi32(0x7f));
    mm = _mm256_slli_epi32(mm, 23);
    __m256 pow2n = _mm256_castsi256_ps(mm);

    return _mm256_mul_ps(y, pow2n);
}
#endif

#endif // CPU_SOFTMAX_AVX2_MATH_H
