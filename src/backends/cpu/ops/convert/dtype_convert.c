#include "cpu_backend_internal.h"
#include "ops/convert/convert_registry.h"

#define CPU_CONVERT_DEFINE_SHIM(func, dst_type, src_type)                                                              \
    static void func##_shim(const void *device_ctx, void *dst, const void *src, size_t n) {                            \
        func(device_ctx, (dst_type *)dst, (const src_type *)src, n);                                                   \
    }

// ==================================================================
// CPU Backend Dtype Conversion Operations
// ==================================================================
// SIMD-optimized dtype conversions: F32 <-> F16 <-> BF16
// Uses NEON on ARM, F16C/AVX2 on x86, scalar fallback otherwise
// ==================================================================

// FLOAT32 -> FLOAT16 (vectorized with SIMD)
void cpu_convert_f32_to_f16(
    [[maybe_unused]] const void *device_ctx, marmot_float16_t *dst, const float *src, size_t n
) {
    size_t i = 0;

#if HAS_NEON
    // NEON: Process 4 floats at a time
    for (; i + 4 <= n; i += 4) {
        float32x4_t f32_vec = vld1q_f32(src + i);
        float16x4_t f16_vec = vcvt_f16_f32(f32_vec);
        vst1_u16((uint16_t *)(dst + i), vreinterpret_u16_f16(f16_vec));
    }
#elif HAS_F16C
    // AVX2 + F16C: Process 8 floats at a time
    for (; i + 8 <= n; i += 8) {
        __m256 f32_vec = _mm256_loadu_ps(src + i);
        __m128i f16_vec = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128((__m128i *)(dst + i), f16_vec);
    }
#endif

    // Scalar fallback for remaining elements
    for (; i < n; i++) {
        dst[i] = marmot_f32_to_f16_ref(src[i]);
    }
}

// FLOAT16 -> FLOAT32 (vectorized with SIMD)
void cpu_convert_f16_to_f32(
    [[maybe_unused]] const void *device_ctx, float *dst, const marmot_float16_t *src, size_t n
) {
    size_t i = 0;

#if HAS_NEON
    // NEON: Process 4 floats at a time
    for (; i + 4 <= n; i += 4) {
        float16x4_t f16_vec = vreinterpret_f16_u16(vld1_u16((const uint16_t *)(src + i)));
        float32x4_t f32_vec = vcvt_f32_f16(f16_vec);
        vst1q_f32(dst + i, f32_vec);
    }
#elif HAS_F16C
    // AVX2 + F16C: Process 8 floats at a time
    for (; i + 8 <= n; i += 8) {
        __m128i f16_vec = _mm_loadu_si128((const __m128i *)(src + i));
        __m256 f32_vec = _mm256_cvtph_ps(f16_vec);
        _mm256_storeu_ps(dst + i, f32_vec);
    }
#endif

    // Scalar fallback
    for (; i < n; i++) {
        dst[i] = marmot_f16_to_f32_ref(src[i]);
    }
}

// FLOAT32 -> BFLOAT16 (vectorized with SIMD)
void cpu_convert_f32_to_bf16(
    [[maybe_unused]] const void *device_ctx, marmot_bfloat16_t *dst, const float *src, size_t n
) {
    size_t i = 0;

#if HAS_NEON
    // NEON: Process 4 floats at a time
    const uint32x4_t rounding_bias = vdupq_n_u32(0x7FFF);
    for (; i + 4 <= n; i += 4) {
        uint32x4_t f32_bits = vreinterpretq_u32_f32(vld1q_f32(src + i));

        // Round to nearest even
        uint32x4_t lsb = vshrq_n_u32(f32_bits, 16);
        lsb = vandq_u32(lsb, vdupq_n_u32(1));
        uint32x4_t bias = vaddq_u32(rounding_bias, lsb);
        uint32x4_t rounded = vaddq_u32(f32_bits, bias);

        // Truncate to upper 16 bits
        uint16x4_t bf16_vec = vmovn_u32(vshrq_n_u32(rounded, 16));
        vst1_u16((uint16_t *)(dst + i), bf16_vec);
    }
#elif HAS_AVX2
    // AVX2: Process 8 floats at a time
    const __m256i rounding_bias = _mm256_set1_epi32(0x7FFF);
    for (; i + 8 <= n; i += 8) {
        __m256i f32_bits = _mm256_castps_si256(_mm256_loadu_ps(src + i));

        // Round to nearest even
        __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(f32_bits, 16), _mm256_set1_epi32(1));
        __m256i bias = _mm256_add_epi32(rounding_bias, lsb);
        __m256i rounded = _mm256_add_epi32(f32_bits, bias);

        // Truncate to upper 16 bits (pack 32-bit to 16-bit)
        __m256i shifted = _mm256_srli_epi32(rounded, 16);
        __m128i lo = _mm256_castsi256_si128(shifted);
        __m128i hi = _mm256_extracti128_si256(shifted, 1);
        __m128i bf16_vec = _mm_packus_epi32(lo, hi);

        // Fix lane ordering after pack
        bf16_vec = _mm_permute4x64_epi64(bf16_vec, 0xD8);
        _mm_storeu_si128((__m128i *)(dst + i), bf16_vec);
    }
#endif

    // Scalar fallback
    for (; i < n; i++) {
        dst[i] = marmot_f32_to_bf16_ref(src[i]);
    }
}

// BFLOAT16 -> FLOAT32 (vectorized with SIMD)
void cpu_convert_bf16_to_f32(
    [[maybe_unused]] const void *device_ctx, float *dst, const marmot_bfloat16_t *src, size_t n
) {
    size_t i = 0;

#if HAS_NEON
    // NEON: Process 4 bfloat16s at a time
    for (; i + 4 <= n; i += 4) {
        uint16x4_t bf16_vec = vld1_u16((const uint16_t *)(src + i));
        uint32x4_t f32_bits = vshll_n_u16(bf16_vec, 16);
        vst1q_f32(dst + i, vreinterpretq_f32_u32(f32_bits));
    }
#elif HAS_AVX2
    // AVX2: Process 8 bfloat16s at a time
    for (; i + 8 <= n; i += 8) {
        __m128i bf16_vec = _mm_loadu_si128((const __m128i *)(src + i));

        // Zero-extend to 32-bit
        __m256i f32_bits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(bf16_vec), 16);
        _mm256_storeu_ps(dst + i, _mm256_castsi256_ps(f32_bits));
    }
#endif

    // Scalar fallback
    for (; i < n; i++) {
        dst[i] = marmot_bf16_to_f32_ref(src[i]);
    }
}

void cpu_convert_f32_to_f64([[maybe_unused]] const void *device_ctx, double *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (double)src[i];
    }
}

void cpu_convert_f64_to_f32([[maybe_unused]] const void *device_ctx, float *dst, const double *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (float)src[i];
    }
}

void cpu_convert_f32_to_i64([[maybe_unused]] const void *device_ctx, marmot_int64_t *dst, const float *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i].value = (int64_t)src[i];
    }
}

void cpu_convert_i64_to_f32([[maybe_unused]] const void *device_ctx, float *dst, const marmot_int64_t *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (float)src[i].value;
    }
}

void cpu_convert_f64_to_i64([[maybe_unused]] const void *device_ctx, marmot_int64_t *dst, const double *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i].value = (int64_t)src[i];
    }
}

void cpu_convert_i64_to_f64([[maybe_unused]] const void *device_ctx, double *dst, const marmot_int64_t *src, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        dst[i] = (double)src[i].value;
    }
}

// FLOAT16 -> BFLOAT16 (via FLOAT32 temp buffer)
void cpu_convert_f16_to_bf16(const void *device_ctx, marmot_bfloat16_t *dst, const marmot_float16_t *src, size_t n) {
    // Use temporary F32 buffer for conversion
    const size_t STACK_BUFFER_SIZE = 256;
    float temp_buffer[STACK_BUFFER_SIZE];

    for (size_t offset = 0; offset < n; offset += STACK_BUFFER_SIZE) {
        size_t chunk_size = (n - offset < STACK_BUFFER_SIZE) ? (n - offset) : STACK_BUFFER_SIZE;
        cpu_convert_f16_to_f32(device_ctx, temp_buffer, src + offset, chunk_size);
        cpu_convert_f32_to_bf16(device_ctx, dst + offset, temp_buffer, chunk_size);
    }
}

// BFLOAT16 -> FLOAT16 (via FLOAT32 temp buffer)
void cpu_convert_bf16_to_f16(const void *device_ctx, marmot_float16_t *dst, const marmot_bfloat16_t *src, size_t n) {
    // Use temporary F32 buffer for conversion
    const size_t STACK_BUFFER_SIZE = 256;
    float temp_buffer[STACK_BUFFER_SIZE];

    for (size_t offset = 0; offset < n; offset += STACK_BUFFER_SIZE) {
        size_t chunk_size = (n - offset < STACK_BUFFER_SIZE) ? (n - offset) : STACK_BUFFER_SIZE;
        cpu_convert_bf16_to_f32(device_ctx, temp_buffer, src + offset, chunk_size);
        cpu_convert_f32_to_f16(device_ctx, dst + offset, temp_buffer, chunk_size);
    }
}

#if MARMOT_ENABLE_FP8

#if (HAS_AVX2 && HAS_F16C) || HAS_NEON
static inline void cpu_fp8_e4m3_scalar_block(marmot_float8_e4m3_t *dst, const float *src, size_t count) {
    for (size_t lane = 0; lane < count; ++lane) {
        dst[lane] = marmot_f32_to_fp8_e4m3_ref(src[lane]);
    }
}

static inline void cpu_fp8_e5m2_scalar_block(marmot_float8_e5m2_t *dst, const float *src, size_t count) {
    for (size_t lane = 0; lane < count; ++lane) {
        dst[lane] = marmot_f32_to_fp8_e5m2_ref(src[lane]);
    }
}
#endif

void cpu_convert_f32_to_fp8_e4m3(const void *device_ctx, marmot_float8_e4m3_t *dst, const float *src, size_t n) {
    size_t i = 0;

#if HAS_AVX2 && HAS_F16C
    if (has_avx2(device_ctx) && has_f16c(device_ctx)) {
        while (i + 8U <= n) {
            __m256 f32_vec = _mm256_loadu_ps(src + i);
            __m128i f16_vec = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);

            __m128i sign = _mm_and_si128(_mm_srli_epi16(f16_vec, 8), _mm_set1_epi16(0x80));
            __m128i exp = _mm_and_si128(_mm_srli_epi16(f16_vec, 10), _mm_set1_epi16(0x1F));
            __m128i mant = _mm_and_si128(f16_vec, _mm_set1_epi16(0x3FF));

            __m128i exp_zero = _mm_cmpeq_epi16(exp, _mm_setzero_si128());
            __m128i exp_special = _mm_cmpeq_epi16(exp, _mm_set1_epi16(0x1F));

            __m128i new_exp = _mm_sub_epi16(exp, _mm_set1_epi16(8));
            __m128i exp_low = _mm_cmpgt_epi16(_mm_set1_epi16(1), new_exp);   // new_exp <= 0
            __m128i exp_high = _mm_cmpgt_epi16(new_exp, _mm_set1_epi16(14)); // new_exp >= 15

            __m128i fast_mask = _mm_andnot_si128(exp_zero, _mm_andnot_si128(exp_special, _mm_set1_epi16(-1)));
            fast_mask = _mm_andnot_si128(_mm_or_si128(exp_low, exp_high), fast_mask);

            if (_mm_movemask_epi8(fast_mask) != 0xFFFF) {
                float fallback_vals[8];
                _mm256_storeu_ps(fallback_vals, f32_vec);
                cpu_fp8_e4m3_scalar_block(dst + i, fallback_vals, 8);
                i += 8;
                continue;
            }

            __m128i mant_round = _mm_add_epi16(mant, _mm_set1_epi16(1 << 6));
            mant_round = _mm_srli_epi16(mant_round, 7);

            __m128i overflow = _mm_cmpeq_epi16(mant_round, _mm_set1_epi16(1 << 3));
            mant_round = _mm_andnot_si128(overflow, mant_round);

            __m128i overflow_inc = _mm_and_si128(overflow, _mm_set1_epi16(1));
            __m128i new_exp_inc = _mm_add_epi16(new_exp, overflow_inc);
            __m128i exp_clamped = _mm_min_epi16(new_exp_inc, _mm_set1_epi16(14));

            __m128i exp_out = _mm_and_si128(exp_clamped, _mm_set1_epi16(0xF));
            __m128i mant_out = _mm_min_epi16(mant_round, _mm_set1_epi16(7));

            __m128i packed = _mm_or_si128(sign, _mm_or_si128(_mm_slli_epi16(exp_out, 3), mant_out));
            __m128i bytes = _mm_packus_epi16(packed, _mm_setzero_si128());

            uint8_t tmp[16];
            _mm_storeu_si128((__m128i *)tmp, bytes);
            for (int lane = 0; lane < 8; ++lane) {
                dst[i + (size_t)lane].bits = tmp[lane];
            }
            i += 8;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        while (i + 4U <= n) {
            float32x4_t f32_vec = vld1q_f32(src + i);
            float16x4_t f16_vec = vcvt_f16_f32(f32_vec);
            uint16x4_t bits = vreinterpret_u16_f16(f16_vec);

            uint16x4_t sign = vand_u16(vshr_n_u16(bits, 8), vdup_n_u16(0x80));
            uint16x4_t exp = vand_u16(vshr_n_u16(bits, 10), vdup_n_u16(0x1F));
            uint16x4_t mant = vand_u16(bits, vdup_n_u16(0x3FF));

            uint16x4_t exp_zero = vceq_u16(exp, vdup_n_u16(0));
            uint16x4_t exp_special = vceq_u16(exp, vdup_n_u16(0x1F));

            int16x4_t new_exp = vsub_s16(vreinterpret_s16_u16(exp), vdup_n_s16(8));
            uint16x4_t exp_low = vmvn_u16(vcgt_s16(new_exp, vdup_n_s16(0)));
            uint16x4_t exp_high = vcge_s16(new_exp, vdup_n_s16(15));

            uint16x4_t fast_mask = vand_u16(vmvn_u16(exp_zero), vmvn_u16(exp_special));
            fast_mask = vand_u16(fast_mask, vmvn_u16(vorr_u16(exp_low, exp_high)));

            uint16_t mask_arr[4];
            vst1_u16(mask_arr, fast_mask);
            if (mask_arr[0] != 0xFFFF || mask_arr[1] != 0xFFFF || mask_arr[2] != 0xFFFF || mask_arr[3] != 0xFFFF) {
                float fallback_vals[4];
                vst1q_f32(fallback_vals, f32_vec);
                cpu_fp8_e4m3_scalar_block(dst + i, fallback_vals, 4);
                i += 4;
                continue;
            }

            uint16x4_t mant_round = vadd_u16(mant, vdup_n_u16(1 << 6));
            mant_round = vshr_n_u16(mant_round, 7);

            uint16x4_t overflow = vceq_u16(mant_round, vdup_n_u16(1 << 3));
            mant_round = vbic_u16(mant_round, overflow);

            int16x4_t new_exp_inc = vadd_s16(new_exp, vreinterpret_s16_u16(vand_u16(overflow, vdup_n_u16(1))));
            int16x4_t exp_clamped = vmin_s16(new_exp_inc, vdup_n_s16(14));

            uint16x4_t exp_out = vand_u16(vreinterpret_u16_s16(exp_clamped), vdup_n_u16(0xF));
            uint16x4_t mant_out = vmin_u16(mant_round, vdup_n_u16(7));

            uint16x4_t packed = vorr_u16(sign, vorr_u16(vshl_n_u16(exp_out, 3), mant_out));
            uint16_t tmp[4];
            vst1_u16(tmp, packed);
            for (int lane = 0; lane < 4; ++lane) {
                dst[i + (size_t)lane].bits = (uint8_t)tmp[lane];
            }
            i += 4;
        }
    }
#endif

    for (; i < n; ++i) {
        dst[i] = marmot_f32_to_fp8_e4m3_ref(src[i]);
    }
}

void cpu_convert_f32_to_fp8_e5m2(const void *device_ctx, marmot_float8_e5m2_t *dst, const float *src, size_t n) {
    size_t i = 0;

#if HAS_AVX2 && HAS_F16C
    if (has_avx2(device_ctx) && has_f16c(device_ctx)) {
        while (i + 8U <= n) {
            __m256 f32_vec = _mm256_loadu_ps(src + i);
            __m128i f16_vec = _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT);

            __m128i sign = _mm_and_si128(_mm_srli_epi16(f16_vec, 8), _mm_set1_epi16(0x80));
            __m128i exp = _mm_and_si128(_mm_srli_epi16(f16_vec, 10), _mm_set1_epi16(0x1F));
            __m128i mant = _mm_and_si128(f16_vec, _mm_set1_epi16(0x3FF));

            __m128i exp_zero = _mm_cmpeq_epi16(exp, _mm_setzero_si128());
            __m128i exp_special = _mm_cmpeq_epi16(exp, _mm_set1_epi16(0x1F));

            __m128i fast_mask = _mm_andnot_si128(exp_zero, _mm_andnot_si128(exp_special, _mm_set1_epi16(-1)));
            if (_mm_movemask_epi8(fast_mask) != 0xFFFF) {
                float fallback_vals[8];
                _mm256_storeu_ps(fallback_vals, f32_vec);
                cpu_fp8_e5m2_scalar_block(dst + i, fallback_vals, 8);
                i += 8;
                continue;
            }

            __m128i mant_round = _mm_add_epi16(mant, _mm_set1_epi16(1 << 7));
            mant_round = _mm_srli_epi16(mant_round, 8);

            __m128i overflow = _mm_cmpeq_epi16(mant_round, _mm_set1_epi16(1 << 2));
            mant_round = _mm_andnot_si128(overflow, mant_round);

            __m128i exp_inc = _mm_add_epi16(exp, _mm_and_si128(overflow, _mm_set1_epi16(1)));
            __m128i exp_clamped = _mm_min_epi16(exp_inc, _mm_set1_epi16(31));

            __m128i exp_is_inf = _mm_cmpeq_epi16(exp_clamped, _mm_set1_epi16(31));
            __m128i mant_out = _mm_min_epi16(mant_round, _mm_set1_epi16(3));
            mant_out = _mm_andnot_si128(exp_is_inf, mant_out);

            __m128i packed = _mm_or_si128(sign, _mm_or_si128(_mm_slli_epi16(exp_clamped, 2), mant_out));
            __m128i bytes = _mm_packus_epi16(packed, _mm_setzero_si128());

            uint8_t tmp[16];
            _mm_storeu_si128((__m128i *)tmp, bytes);
            for (int lane = 0; lane < 8; ++lane) {
                dst[i + (size_t)lane].bits = tmp[lane];
            }
            i += 8;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        while (i + 4U <= n) {
            float32x4_t f32_vec = vld1q_f32(src + i);
            float16x4_t f16_vec = vcvt_f16_f32(f32_vec);
            uint16x4_t bits = vreinterpret_u16_f16(f16_vec);

            uint16x4_t sign = vand_u16(vshr_n_u16(bits, 8), vdup_n_u16(0x80));
            uint16x4_t exp = vand_u16(vshr_n_u16(bits, 10), vdup_n_u16(0x1F));
            uint16x4_t mant = vand_u16(bits, vdup_n_u16(0x3FF));

            uint16x4_t exp_zero = vceq_u16(exp, vdup_n_u16(0));
            uint16x4_t exp_special = vceq_u16(exp, vdup_n_u16(0x1F));

            uint16x4_t fast_mask = vand_u16(vmvn_u16(exp_zero), vmvn_u16(exp_special));
            uint16_t mask_arr[4];
            vst1_u16(mask_arr, fast_mask);
            if (mask_arr[0] != 0xFFFF || mask_arr[1] != 0xFFFF || mask_arr[2] != 0xFFFF || mask_arr[3] != 0xFFFF) {
                float fallback_vals[4];
                vst1q_f32(fallback_vals, f32_vec);
                cpu_fp8_e5m2_scalar_block(dst + i, fallback_vals, 4);
                i += 4;
                continue;
            }

            uint16x4_t mant_round = vadd_u16(mant, vdup_n_u16(1 << 7));
            mant_round = vshr_n_u16(mant_round, 8);

            uint16x4_t overflow = vceq_u16(mant_round, vdup_n_u16(1 << 2));
            mant_round = vbic_u16(mant_round, overflow);

            uint16x4_t exp_inc = vadd_u16(exp, vand_u16(overflow, vdup_n_u16(1)));
            uint16x4_t exp_clamped = vmin_u16(exp_inc, vdup_n_u16(31));

            uint16x4_t exp_is_inf = vceq_u16(exp_clamped, vdup_n_u16(31));
            uint16x4_t mant_out = vmin_u16(mant_round, vdup_n_u16(3));
            mant_out = vbic_u16(mant_out, exp_is_inf);

            uint16x4_t packed = vorr_u16(sign, vorr_u16(vshl_n_u16(exp_clamped, 2), mant_out));
            uint16_t tmp[4];
            vst1_u16(tmp, packed);
            for (int lane = 0; lane < 4; ++lane) {
                dst[i + (size_t)lane].bits = (uint8_t)tmp[lane];
            }
            i += 4;
        }
    }
#endif

    for (; i < n; ++i) {
        dst[i] = marmot_f32_to_fp8_e5m2_ref(src[i]);
    }
}

void cpu_convert_fp8_e4m3_to_f32(const void *device_ctx, float *dst, const marmot_float8_e4m3_t *src, size_t n) {
    size_t i = 0;

#if HAS_AVX2 && HAS_F16C
    if (has_avx2(device_ctx) && has_f16c(device_ctx)) {
        while (i + 8U <= n) {
            __m128i raw_bytes = _mm_loadl_epi64((const __m128i *)&src[i]);
            __m128i raw_u16 = _mm_cvtepu8_epi16(raw_bytes);

            __m128i exp = _mm_and_si128(_mm_srli_epi16(raw_u16, 3), _mm_set1_epi16(0xF));
            __m128i mant = _mm_and_si128(raw_u16, _mm_set1_epi16(0x7));
            __m128i sign = _mm_and_si128(raw_u16, _mm_set1_epi16(0x80));

            __m128i exp_gt_zero = _mm_cmpgt_epi16(exp, _mm_setzero_si128());
            __m128i exp_lt_15 = _mm_cmpgt_epi16(_mm_set1_epi16(15), exp);
            __m128i fast_mask = _mm_and_si128(exp_gt_zero, exp_lt_15);

            if (_mm_movemask_epi8(fast_mask) != 0xFFFF) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    dst[i + lane] = marmot_fp8_e4m3_to_f32_ref(src[i + lane]);
                }
                i += 8;
                continue;
            }

            __m128i sign_f16 = _mm_slli_epi16(sign, 8);
            __m128i exp_f16 = _mm_add_epi16(exp, _mm_set1_epi16(8));
            __m128i mant_f16 = _mm_slli_epi16(mant, 7);

            __m128i f16_bits = _mm_or_si128(sign_f16, _mm_or_si128(_mm_slli_epi16(exp_f16, 10), mant_f16));
            __m256 f32_vec = _mm256_cvtph_ps(f16_bits);
            _mm256_storeu_ps(dst + i, f32_vec);
            i += 8;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        while (i + 8U <= n) {
            uint8x8_t raw = vld1_u8((const uint8_t *)&src[i]);
            uint16x8_t raw_u16 = vmovl_u8(raw);

            uint16x8_t exp = vandq_u16(vshrq_n_u16(raw_u16, 3), vdupq_n_u16(0xF));
            uint16x8_t mant = vandq_u16(raw_u16, vdupq_n_u16(0x7));
            uint16x8_t sign = vandq_u16(raw_u16, vdupq_n_u16(0x80));

            uint16x8_t exp_gt_zero = vcgtq_u16(exp, vdupq_n_u16(0));
            uint16x8_t exp_lt_15 = vcgtq_u16(vdupq_n_u16(15), exp);
            uint16x8_t fast_mask = vandq_u16(exp_gt_zero, exp_lt_15);

            uint16_t mask_arr[8];
            vst1q_u16(mask_arr, fast_mask);
            bool all_fast = true;
            for (int lane = 0; lane < 8; ++lane) {
                if (mask_arr[lane] != 0xFFFF) {
                    all_fast = false;
                    break;
                }
            }
            if (!all_fast) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    dst[i + lane] = marmot_fp8_e4m3_to_f32_ref(src[i + lane]);
                }
                i += 8;
                continue;
            }

            uint16x8_t sign_f16 = vshlq_n_u16(sign, 8);
            uint16x8_t exp_f16 = vaddq_u16(exp, vdupq_n_u16(8));
            uint16x8_t mant_f16 = vshlq_n_u16(mant, 7);

            uint16x8_t f16_bits = vorrq_u16(sign_f16, vorrq_u16(vshlq_n_u16(exp_f16, 10), mant_f16));

            float16x4_t f16_lo = vreinterpret_f16_u16(vget_low_u16(f16_bits));
            float16x4_t f16_hi = vreinterpret_f16_u16(vget_high_u16(f16_bits));

            float32x4_t f32_lo = vcvt_f32_f16(f16_lo);
            float32x4_t f32_hi = vcvt_f32_f16(f16_hi);

            vst1q_f32(dst + i, f32_lo);
            vst1q_f32(dst + i + 4, f32_hi);
            i += 8;
        }
    }
#endif

    for (; i < n; ++i) {
        dst[i] = marmot_fp8_e4m3_to_f32_ref(src[i]);
    }
}

void cpu_convert_fp8_e5m2_to_f32(const void *device_ctx, float *dst, const marmot_float8_e5m2_t *src, size_t n) {
    size_t i = 0;

#if HAS_AVX2 && HAS_F16C
    if (has_avx2(device_ctx) && has_f16c(device_ctx)) {
        while (i + 8U <= n) {
            __m128i raw_bytes = _mm_loadl_epi64((const __m128i *)&src[i]);
            __m128i raw_u16 = _mm_cvtepu8_epi16(raw_bytes);

            __m128i exp = _mm_and_si128(_mm_srli_epi16(raw_u16, 2), _mm_set1_epi16(0x1F));
            __m128i mant = _mm_and_si128(raw_u16, _mm_set1_epi16(0x3));
            __m128i sign = _mm_and_si128(raw_u16, _mm_set1_epi16(0x80));

            __m128i exp_gt_zero = _mm_cmpgt_epi16(exp, _mm_setzero_si128());
            __m128i exp_lt_31 = _mm_cmpgt_epi16(_mm_set1_epi16(31), exp);
            __m128i fast_mask = _mm_and_si128(exp_gt_zero, exp_lt_31);

            if (_mm_movemask_epi8(fast_mask) != 0xFFFF) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    dst[i + lane] = marmot_fp8_e5m2_to_f32_ref(src[i + lane]);
                }
                i += 8;
                continue;
            }

            __m128i sign_f16 = _mm_slli_epi16(sign, 8);
            __m128i exp_f16 = exp;
            __m128i mant_f16 = _mm_slli_epi16(mant, 8);

            __m128i f16_bits = _mm_or_si128(sign_f16, _mm_or_si128(_mm_slli_epi16(exp_f16, 10), mant_f16));
            __m256 f32_vec = _mm256_cvtph_ps(f16_bits);
            _mm256_storeu_ps(dst + i, f32_vec);
            i += 8;
        }
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        while (i + 8U <= n) {
            uint8x8_t raw = vld1_u8((const uint8_t *)&src[i]);
            uint16x8_t raw_u16 = vmovl_u8(raw);

            uint16x8_t exp = vandq_u16(vshrq_n_u16(raw_u16, 2), vdupq_n_u16(0x1F));
            uint16x8_t mant = vandq_u16(raw_u16, vdupq_n_u16(0x3));
            uint16x8_t sign = vandq_u16(raw_u16, vdupq_n_u16(0x80));

            uint16x8_t exp_gt_zero = vcgtq_u16(exp, vdupq_n_u16(0));
            uint16x8_t exp_lt_31 = vcgtq_u16(vdupq_n_u16(31), exp);
            uint16x8_t fast_mask = vandq_u16(exp_gt_zero, exp_lt_31);

            uint16_t mask_arr[8];
            vst1q_u16(mask_arr, fast_mask);
            bool all_fast = true;
            for (int lane = 0; lane < 8; ++lane) {
                if (mask_arr[lane] != 0xFFFF) {
                    all_fast = false;
                    break;
                }
            }
            if (!all_fast) {
                for (size_t lane = 0; lane < 8; ++lane) {
                    dst[i + lane] = marmot_fp8_e5m2_to_f32_ref(src[i + lane]);
                }
                i += 8;
                continue;
            }

            uint16x8_t sign_f16 = vshlq_n_u16(sign, 8);
            uint16x8_t exp_f16 = exp;
            uint16x8_t mant_f16 = vshlq_n_u16(mant, 8);

            uint16x8_t f16_bits = vorrq_u16(sign_f16, vorrq_u16(vshlq_n_u16(exp_f16, 10), mant_f16));

            float16x4_t f16_lo = vreinterpret_f16_u16(vget_low_u16(f16_bits));
            float16x4_t f16_hi = vreinterpret_f16_u16(vget_high_u16(f16_bits));

            float32x4_t f32_lo = vcvt_f32_f16(f16_lo);
            float32x4_t f32_hi = vcvt_f32_f16(f16_hi);

            vst1q_f32(dst + i, f32_lo);
            vst1q_f32(dst + i + 4, f32_hi);
            i += 8;
        }
    }
#endif

    for (; i < n; ++i) {
        dst[i] = marmot_fp8_e5m2_to_f32_ref(src[i]);
    }
}

void cpu_convert_f16_to_fp8_e4m3(
    const void *device_ctx, marmot_float8_e4m3_t *dst, const marmot_float16_t *src, size_t n
) {
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_f16_to_f32(device_ctx, tmp, src + offset, chunk);
        cpu_convert_f32_to_fp8_e4m3(device_ctx, dst + offset, tmp, chunk);
    }
}

void cpu_convert_fp8_e4m3_to_f16(
    const void *device_ctx, marmot_float16_t *dst, const marmot_float8_e4m3_t *src, size_t n
) {
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_fp8_e4m3_to_f32(device_ctx, tmp, src + offset, chunk);
        cpu_convert_f32_to_f16(device_ctx, dst + offset, tmp, chunk);
    }
}

void cpu_convert_f16_to_fp8_e5m2(
    const void *device_ctx, marmot_float8_e5m2_t *dst, const marmot_float16_t *src, size_t n
) {
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_f16_to_f32(device_ctx, tmp, src + offset, chunk);
        cpu_convert_f32_to_fp8_e5m2(device_ctx, dst + offset, tmp, chunk);
    }
}

void cpu_convert_fp8_e5m2_to_f16(
    const void *device_ctx, marmot_float16_t *dst, const marmot_float8_e5m2_t *src, size_t n
) {
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_fp8_e5m2_to_f32(device_ctx, tmp, src + offset, chunk);
        cpu_convert_f32_to_f16(device_ctx, dst + offset, tmp, chunk);
    }
}

void cpu_convert_bf16_to_fp8_e4m3(
    const void *device_ctx, marmot_float8_e4m3_t *dst, const marmot_bfloat16_t *src, size_t n
) {
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_bf16_to_f32(device_ctx, tmp, src + offset, chunk);
        cpu_convert_f32_to_fp8_e4m3(device_ctx, dst + offset, tmp, chunk);
    }
}

void cpu_convert_fp8_e4m3_to_bf16(
    const void *device_ctx, marmot_bfloat16_t *dst, const marmot_float8_e4m3_t *src, size_t n
) {
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_fp8_e4m3_to_f32(device_ctx, tmp, src + offset, chunk);
        cpu_convert_f32_to_bf16(device_ctx, dst + offset, tmp, chunk);
    }
}

void cpu_convert_bf16_to_fp8_e5m2(
    const void *device_ctx, marmot_float8_e5m2_t *dst, const marmot_bfloat16_t *src, size_t n
) {
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_bf16_to_f32(device_ctx, tmp, src + offset, chunk);
        cpu_convert_f32_to_fp8_e5m2(device_ctx, dst + offset, tmp, chunk);
    }
}

void cpu_convert_fp8_e5m2_to_bf16(
    const void *device_ctx, marmot_bfloat16_t *dst, const marmot_float8_e5m2_t *src, size_t n
) {
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_fp8_e5m2_to_f32(device_ctx, tmp, src + offset, chunk);
        cpu_convert_f32_to_bf16(device_ctx, dst + offset, tmp, chunk);
    }
}

#endif // MARMOT_ENABLE_FP8

static void cpu_convert_f64_to_f16_via_f32(const void *device_ctx, void *dst, const void *src, size_t n) {
    const double *src_f64 = (const double *)src;
    marmot_float16_t *dst_f16 = (marmot_float16_t *)dst;
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_f64_to_f32(device_ctx, tmp, src_f64 + offset, chunk);
        cpu_convert_f32_to_f16(device_ctx, dst_f16 + offset, tmp, chunk);
    }
}

static void cpu_convert_f64_to_bf16_via_f32(const void *device_ctx, void *dst, const void *src, size_t n) {
    const double *src_f64 = (const double *)src;
    marmot_bfloat16_t *dst_bf16 = (marmot_bfloat16_t *)dst;
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_f64_to_f32(device_ctx, tmp, src_f64 + offset, chunk);
        cpu_convert_f32_to_bf16(device_ctx, dst_bf16 + offset, tmp, chunk);
    }
}

static void cpu_convert_f16_to_f64_via_f32(const void *device_ctx, void *dst, const void *src, size_t n) {
    double *dst_f64 = (double *)dst;
    const marmot_float16_t *src_f16 = (const marmot_float16_t *)src;
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_f16_to_f32(device_ctx, tmp, src_f16 + offset, chunk);
        cpu_convert_f32_to_f64(device_ctx, dst_f64 + offset, tmp, chunk);
    }
}

static void cpu_convert_bf16_to_f64_via_f32(const void *device_ctx, void *dst, const void *src, size_t n) {
    double *dst_f64 = (double *)dst;
    const marmot_bfloat16_t *src_bf16 = (const marmot_bfloat16_t *)src;
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_bf16_to_f32(device_ctx, tmp, src_bf16 + offset, chunk);
        cpu_convert_f32_to_f64(device_ctx, dst_f64 + offset, tmp, chunk);
    }
}

#if MARMOT_ENABLE_ACCELERATE
static void cpu_convert_f32_to_f16_accelerate(const void *device_ctx, void *dst, const void *src, size_t n) {
    (void)device_ctx;
    if (n == 0) {
        return;
    }
    vImage_Buffer src_buf = {
        .data = (void *)src,
        .height = 1,
        .width = n,
        .rowBytes = n * sizeof(float),
    };
    vImage_Buffer dst_buf = {
        .data = (void *)((marmot_float16_t *)dst),
        .height = 1,
        .width = n,
        .rowBytes = n * sizeof(marmot_float16_t),
    };
    vImage_Error err = vImageConvert_PlanarFtoPlanar16F(&src_buf, &dst_buf, kvImageNoFlags);
    if (err != kvImageNoError) {
        cpu_convert_f32_to_f16(device_ctx, (marmot_float16_t *)dst, (const float *)src, n);
    }
}

static void cpu_convert_f16_to_f32_accelerate(const void *device_ctx, void *dst, const void *src, size_t n) {
    (void)device_ctx;
    if (n == 0) {
        return;
    }
    vImage_Buffer src_buf = {
        .data = (void *)((const marmot_float16_t *)src),
        .height = 1,
        .width = n,
        .rowBytes = n * sizeof(marmot_float16_t),
    };
    vImage_Buffer dst_buf = {
        .data = (void *)dst,
        .height = 1,
        .width = n,
        .rowBytes = n * sizeof(float),
    };
    vImage_Error err = vImageConvert_Planar16FtoPlanarF(&src_buf, &dst_buf, kvImageNoFlags);
    if (err != kvImageNoError) {
        cpu_convert_f16_to_f32(device_ctx, (float *)dst, (const marmot_float16_t *)src, n);
    }
}
#endif

#if MARMOT_ENABLE_FP8
static void cpu_convert_f64_to_fp8_e4m3_via_f32(const void *device_ctx, void *dst, const void *src, size_t n) {
    const double *src_f64 = (const double *)src;
    marmot_float8_e4m3_t *dst_fp8 = (marmot_float8_e4m3_t *)dst;
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_f64_to_f32(device_ctx, tmp, src_f64 + offset, chunk);
        cpu_convert_f32_to_fp8_e4m3(device_ctx, dst_fp8 + offset, tmp, chunk);
    }
}

static void cpu_convert_f64_to_fp8_e5m2_via_f32(const void *device_ctx, void *dst, const void *src, size_t n) {
    const double *src_f64 = (const double *)src;
    marmot_float8_e5m2_t *dst_fp8 = (marmot_float8_e5m2_t *)dst;
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_f64_to_f32(device_ctx, tmp, src_f64 + offset, chunk);
        cpu_convert_f32_to_fp8_e5m2(device_ctx, dst_fp8 + offset, tmp, chunk);
    }
}

static void cpu_convert_fp8_e4m3_to_f64_via_f32(const void *device_ctx, void *dst, const void *src, size_t n) {
    double *dst_f64 = (double *)dst;
    const marmot_float8_e4m3_t *src_fp8 = (const marmot_float8_e4m3_t *)src;
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_fp8_e4m3_to_f32(device_ctx, tmp, src_fp8 + offset, chunk);
        cpu_convert_f32_to_f64(device_ctx, dst_f64 + offset, tmp, chunk);
    }
}

static void cpu_convert_fp8_e5m2_to_f64_via_f32(const void *device_ctx, void *dst, const void *src, size_t n) {
    double *dst_f64 = (double *)dst;
    const marmot_float8_e5m2_t *src_fp8 = (const marmot_float8_e5m2_t *)src;
    const size_t kBlock = 256;
    float tmp[kBlock];
    for (size_t offset = 0; offset < n; offset += kBlock) {
        size_t chunk = min_size(kBlock, n - offset);
        cpu_convert_fp8_e5m2_to_f32(device_ctx, tmp, src_fp8 + offset, chunk);
        cpu_convert_f32_to_f64(device_ctx, dst_f64 + offset, tmp, chunk);
    }
}
#endif // MARMOT_ENABLE_FP8

CPU_CONVERT_DEFINE_SHIM(cpu_convert_f32_to_f16, marmot_float16_t, float)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f16_to_f32, float, marmot_float16_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f32_to_bf16, marmot_bfloat16_t, float)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_bf16_to_f32, float, marmot_bfloat16_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f16_to_bf16, marmot_bfloat16_t, marmot_float16_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_bf16_to_f16, marmot_float16_t, marmot_bfloat16_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f32_to_f64, double, float)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f64_to_f32, float, double)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f32_to_i64, marmot_int64_t, float)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_i64_to_f32, float, marmot_int64_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f64_to_i64, marmot_int64_t, double)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_i64_to_f64, double, marmot_int64_t)
#if MARMOT_ENABLE_FP8
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f32_to_fp8_e4m3, marmot_float8_e4m3_t, float)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_fp8_e4m3_to_f32, float, marmot_float8_e4m3_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f32_to_fp8_e5m2, marmot_float8_e5m2_t, float)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_fp8_e5m2_to_f32, float, marmot_float8_e5m2_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f16_to_fp8_e4m3, marmot_float8_e4m3_t, marmot_float16_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_fp8_e4m3_to_f16, marmot_float16_t, marmot_float8_e4m3_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_f16_to_fp8_e5m2, marmot_float8_e5m2_t, marmot_float16_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_fp8_e5m2_to_f16, marmot_float16_t, marmot_float8_e5m2_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_bf16_to_fp8_e4m3, marmot_float8_e4m3_t, marmot_bfloat16_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_fp8_e4m3_to_bf16, marmot_bfloat16_t, marmot_float8_e4m3_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_bf16_to_fp8_e5m2, marmot_float8_e5m2_t, marmot_bfloat16_t)
CPU_CONVERT_DEFINE_SHIM(cpu_convert_fp8_e5m2_to_bf16, marmot_bfloat16_t, marmot_float8_e5m2_t)
#endif

#define CPU_CONVERT_TRAIT(symbol, SRC, DST, KIND, FN, LABEL)                                                           \
    const cpu_convert_traits_t symbol = {                                                                              \
        .src = (SRC),                                                                                                  \
        .dst = (DST),                                                                                                  \
        .impl_kind = (KIND),                                                                                           \
        .fn = (FN),                                                                                                    \
        .impl_name = (LABEL),                                                                                          \
    };                                                                                                                 \
    CPU_CONVERT_REGISTER_TRAITS(symbol)
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_f16_scalar_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f32_to_f16_shim, "scalar:f32->f16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_f32_scalar_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f16_to_f32_shim, "scalar:f16->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_bf16_scalar_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_BFLOAT16, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f32_to_bf16_shim, "scalar:f32->bf16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_bf16_to_f32_scalar_traits, MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_bf16_to_f32_shim, "scalar:bf16->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_bf16_scalar_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_BFLOAT16, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f16_to_bf16_shim, "scalar:f16->bf16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_bf16_to_f16_scalar_traits, MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_bf16_to_f16_shim, "scalar:bf16->f16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_f64_scalar_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT64, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f32_to_f64_shim, "scalar:f32->f64"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f64_to_f32_scalar_traits, MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f64_to_f32_shim, "scalar:f64->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_i64_scalar_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_INT64, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f32_to_i64_shim, "scalar:f32->i64"
)
CPU_CONVERT_TRAIT(
    cpu_convert_i64_to_f32_scalar_traits, MARMOT_DTYPE_INT64, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_i64_to_f32_shim, "scalar:i64->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f64_to_i64_scalar_traits, MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_INT64, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f64_to_i64_shim, "scalar:f64->i64"
)
CPU_CONVERT_TRAIT(
    cpu_convert_i64_to_f64_scalar_traits, MARMOT_DTYPE_INT64, MARMOT_DTYPE_FLOAT64, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_i64_to_f64_shim, "scalar:i64->f64"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f64_to_f16_custom_traits, MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_CUSTOM,
    cpu_convert_f64_to_f16_via_f32, "via-f32:f64->f16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f64_to_bf16_custom_traits, MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_BFLOAT16, CPU_CONVERT_IMPL_CUSTOM,
    cpu_convert_f64_to_bf16_via_f32, "via-f32:f64->bf16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_f64_custom_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT64, CPU_CONVERT_IMPL_CUSTOM,
    cpu_convert_f16_to_f64_via_f32, "via-f32:f16->f64"
)
CPU_CONVERT_TRAIT(
    cpu_convert_bf16_to_f64_custom_traits, MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT64, CPU_CONVERT_IMPL_CUSTOM,
    cpu_convert_bf16_to_f64_via_f32, "via-f32:bf16->f64"
)

#if MARMOT_ENABLE_ACCELERATE
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_f16_accel_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_ACCELERATE,
    cpu_convert_f32_to_f16_accelerate, "accelerate:f32->f16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_f32_accel_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_ACCELERATE,
    cpu_convert_f16_to_f32_accelerate, "accelerate:f16->f32"
)
#endif

#if HAS_NEON
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_f16_neon_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_NEON,
    cpu_convert_f32_to_f16_shim, "neon:f32->f16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_f32_neon_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_NEON,
    cpu_convert_f16_to_f32_shim, "neon:f16->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_bf16_neon_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_BFLOAT16, CPU_CONVERT_IMPL_NEON,
    cpu_convert_f32_to_bf16_shim, "neon:f32->bf16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_bf16_to_f32_neon_traits, MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_NEON,
    cpu_convert_bf16_to_f32_shim, "neon:bf16->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_bf16_neon_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_BFLOAT16, CPU_CONVERT_IMPL_NEON,
    cpu_convert_f16_to_bf16_shim, "neon:f16->bf16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_bf16_to_f16_neon_traits, MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_NEON,
    cpu_convert_bf16_to_f16_shim, "neon:bf16->f16"
)
#endif

#if HAS_AVX2
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_f16_avx2_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_AVX2,
    cpu_convert_f32_to_f16_shim, "avx2:f32->f16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_f32_avx2_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_AVX2,
    cpu_convert_f16_to_f32_shim, "avx2:f16->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_bf16_avx2_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_BFLOAT16, CPU_CONVERT_IMPL_AVX2,
    cpu_convert_f32_to_bf16_shim, "avx2:f32->bf16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_bf16_to_f32_avx2_traits, MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_AVX2,
    cpu_convert_bf16_to_f32_shim, "avx2:bf16->f32"
)
#endif

#if MARMOT_ENABLE_FP8
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_fp8_e4m3_scalar_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT8_E4M3, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f32_to_fp8_e4m3_shim, "scalar:f32->fp8_e4m3"
)
CPU_CONVERT_TRAIT(
    cpu_convert_fp8_e4m3_to_f32_scalar_traits, MARMOT_DTYPE_FLOAT8_E4M3, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_fp8_e4m3_to_f32_shim, "scalar:fp8_e4m3->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f32_to_fp8_e5m2_scalar_traits, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT8_E5M2, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f32_to_fp8_e5m2_shim, "scalar:f32->fp8_e5m2"
)
CPU_CONVERT_TRAIT(
    cpu_convert_fp8_e5m2_to_f32_scalar_traits, MARMOT_DTYPE_FLOAT8_E5M2, MARMOT_DTYPE_FLOAT32, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_fp8_e5m2_to_f32_shim, "scalar:fp8_e5m2->f32"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_fp8_e4m3_scalar_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT8_E4M3, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f16_to_fp8_e4m3_shim, "scalar:f16->fp8_e4m3"
)
CPU_CONVERT_TRAIT(
    cpu_convert_fp8_e4m3_to_f16_scalar_traits, MARMOT_DTYPE_FLOAT8_E4M3, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_fp8_e4m3_to_f16_shim, "scalar:fp8_e4m3->f16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f16_to_fp8_e5m2_scalar_traits, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT8_E5M2, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_f16_to_fp8_e5m2_shim, "scalar:f16->fp8_e5m2"
)
CPU_CONVERT_TRAIT(
    cpu_convert_fp8_e5m2_to_f16_scalar_traits, MARMOT_DTYPE_FLOAT8_E5M2, MARMOT_DTYPE_FLOAT16, CPU_CONVERT_IMPL_SCALAR,
    cpu_convert_fp8_e5m2_to_f16_shim, "scalar:fp8_e5m2->f16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_bf16_to_fp8_e4m3_scalar_traits, MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT8_E4M3,
    CPU_CONVERT_IMPL_SCALAR, cpu_convert_bf16_to_fp8_e4m3_shim, "scalar:bf16->fp8_e4m3"
)
CPU_CONVERT_TRAIT(
    cpu_convert_fp8_e4m3_to_bf16_scalar_traits, MARMOT_DTYPE_FLOAT8_E4M3, MARMOT_DTYPE_BFLOAT16,
    CPU_CONVERT_IMPL_SCALAR, cpu_convert_fp8_e4m3_to_bf16_shim, "scalar:fp8_e4m3->bf16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_bf16_to_fp8_e5m2_scalar_traits, MARMOT_DTYPE_BFLOAT16, MARMOT_DTYPE_FLOAT8_E5M2,
    CPU_CONVERT_IMPL_SCALAR, cpu_convert_bf16_to_fp8_e5m2_shim, "scalar:bf16->fp8_e5m2"
)
CPU_CONVERT_TRAIT(
    cpu_convert_fp8_e5m2_to_bf16_scalar_traits, MARMOT_DTYPE_FLOAT8_E5M2, MARMOT_DTYPE_BFLOAT16,
    CPU_CONVERT_IMPL_SCALAR, cpu_convert_fp8_e5m2_to_bf16_shim, "scalar:fp8_e5m2->bf16"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f64_to_fp8_e4m3_custom_traits, MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_FLOAT8_E4M3, CPU_CONVERT_IMPL_CUSTOM,
    cpu_convert_f64_to_fp8_e4m3_via_f32, "via-f32:f64->fp8_e4m3"
)
CPU_CONVERT_TRAIT(
    cpu_convert_f64_to_fp8_e5m2_custom_traits, MARMOT_DTYPE_FLOAT64, MARMOT_DTYPE_FLOAT8_E5M2, CPU_CONVERT_IMPL_CUSTOM,
    cpu_convert_f64_to_fp8_e5m2_via_f32, "via-f32:f64->fp8_e5m2"
)
CPU_CONVERT_TRAIT(
    cpu_convert_fp8_e4m3_to_f64_custom_traits, MARMOT_DTYPE_FLOAT8_E4M3, MARMOT_DTYPE_FLOAT64, CPU_CONVERT_IMPL_CUSTOM,
    cpu_convert_fp8_e4m3_to_f64_via_f32, "via-f32:fp8_e4m3->f64"
)
CPU_CONVERT_TRAIT(
    cpu_convert_fp8_e5m2_to_f64_custom_traits, MARMOT_DTYPE_FLOAT8_E5M2, MARMOT_DTYPE_FLOAT64, CPU_CONVERT_IMPL_CUSTOM,
    cpu_convert_fp8_e5m2_to_f64_via_f32, "via-f32:fp8_e5m2->f64"
)
#endif
