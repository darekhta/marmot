#include "cpu_backend_internal.h"
#include "ops/matmul/matmul_epilogue.h"

#if HAS_AVX2

static inline float horizontal_sum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

#if defined(__F16C__)
static void matmul_f16_avx2_native(
    const marmot_float16_t *a, const marmot_float16_t *b_row_major, marmot_float16_t *out, size_t N, size_t K, size_t M
) {
    for (size_t n = 0; n < N; n++) {
        const marmot_float16_t *a_row = a + n * K;
        for (size_t m = 0; m < M; m++) {
            const marmot_float16_t *b_row = b_row_major + m * K;
            __m256 acc = _mm256_setzero_ps();
            size_t k = 0;

            for (; k + 8 <= K; k += 8) {
                const __m128i a_half = _mm_loadu_si128((const __m128i *)(a_row + k));
                const __m128i b_half = _mm_loadu_si128((const __m128i *)(b_row + k));
                const __m256 a_f32 = _mm256_cvtph_ps(a_half);
                const __m256 b_f32 = _mm256_cvtph_ps(b_half);
#ifdef __FMA__
                acc = _mm256_fmadd_ps(a_f32, b_f32, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a_f32, b_f32));
#endif
            }

            float sum = horizontal_sum_m256(acc);
            for (; k < K; k++) {
                float av = (float)marmot_float16_to_native(a_row[k]);
                float bv = (float)marmot_float16_to_native(b_row[k]);
                sum += av * bv;
            }

            out[n * M + m] = marmot_native_to_float16((_Float16)sum);
        }
    }
}
#else
static void matmul_f16_avx2_fallback(
    const marmot_float16_t *a, const marmot_float16_t *b_row_major, marmot_float16_t *out, size_t N, size_t K, size_t M
) {
    const size_t CHUNK = 128;
    float a32[CHUNK];
    float b32[CHUNK];

    for (size_t n = 0; n < N; n++) {
        const marmot_float16_t *a_row = a + n * K;
        for (size_t m = 0; m < M; m++) {
            const marmot_float16_t *b_row = b_row_major + m * K;
            __m256 vacc = _mm256_setzero_ps();
            float tail = 0.0f;

            for (size_t k0 = 0; k0 < K; k0 += CHUNK) {
                const size_t len = (k0 + CHUNK <= K) ? CHUNK : (K - k0);
                cpu_convert_f16_to_f32(nullptr, a32, a_row + k0, len);
                cpu_convert_f16_to_f32(nullptr, b32, b_row + k0, len);

                size_t i = 0;
                for (; i + 8 <= len; i += 8) {
                    __m256 av = _mm256_loadu_ps(a32 + i);
                    __m256 bv = _mm256_loadu_ps(b32 + i);
#ifdef __FMA__
                    vacc = _mm256_fmadd_ps(av, bv, vacc);
#else
                    vacc = _mm256_add_ps(vacc, _mm256_mul_ps(av, bv));
#endif
                }
                for (; i < len; ++i) {
                    tail += a32[i] * b32[i];
                }
            }

            float sum = horizontal_sum_m256(vacc) + tail;
            out[n * M + m] = marmot_native_to_float16((_Float16)sum);
        }
    }
}
#endif

static void matmul_bf16_avx2_impl(
    const marmot_bfloat16_t *a, const marmot_bfloat16_t *b_row_major, marmot_bfloat16_t *out, size_t N, size_t K,
    size_t M
) {
    for (size_t n = 0; n < N; n++) {
        const marmot_bfloat16_t *a_row = a + n * K;
        for (size_t m = 0; m < M; m++) {
            const marmot_bfloat16_t *b_row = b_row_major + m * K;
            __m256 acc = _mm256_setzero_ps();
            size_t k = 0;

            for (; k + 8 <= K; k += 8) {
                __m128i a_vec = _mm_loadu_si128((const __m128i *)(a_row + k));
                __m128i b_vec = _mm_loadu_si128((const __m128i *)(b_row + k));

                __m256i a_u32 = _mm256_cvtepu16_epi32(a_vec);
                __m256i b_u32 = _mm256_cvtepu16_epi32(b_vec);
                a_u32 = _mm256_slli_epi32(a_u32, 16);
                b_u32 = _mm256_slli_epi32(b_u32, 16);

                __m256 a_f32 = _mm256_castsi256_ps(a_u32);
                __m256 b_f32 = _mm256_castsi256_ps(b_u32);

#ifdef __FMA__
                acc = _mm256_fmadd_ps(a_f32, b_f32, acc);
#else
                acc = _mm256_add_ps(acc, _mm256_mul_ps(a_f32, b_f32));
#endif
            }

            float sum = horizontal_sum_m256(acc);
            for (; k < K; k++) {
                float av = marmot_bf16_to_f32_ref(a_row[k]);
                float bv = marmot_bf16_to_f32_ref(b_row[k]);
                sum += av * bv;
            }

            out[n * M + m] = marmot_f32_to_bf16_ref(sum);
        }
    }
}

marmot_error_t cpu_matmul_f16_avx2(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
#if defined(__F16C__)
    matmul_f16_avx2_native(
        (const marmot_float16_t *)input->data, (const marmot_float16_t *)weight->data, (marmot_float16_t *)out->data, N,
        K, M
    );
#else
    matmul_f16_avx2_fallback(
        (const marmot_float16_t *)input->data, (const marmot_float16_t *)weight->data, (marmot_float16_t *)out->data, N,
        K, M
    );
#endif
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_bf16_avx2(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    matmul_bf16_avx2_impl(
        (const marmot_bfloat16_t *)input->data, (const marmot_bfloat16_t *)weight->data, (marmot_bfloat16_t *)out->data,
        N, K, M
    );
    return MARMOT_SUCCESS;
}

#endif // HAS_AVX2
