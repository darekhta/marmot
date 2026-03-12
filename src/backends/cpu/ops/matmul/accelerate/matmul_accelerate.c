#include "cpu_backend_internal.h"
#include "ops/matmul/matmul_epilogue.h"
#include "ops/matmul/neon/neon_matmul_params.h"

#if MARMOT_ENABLE_ACCELERATE

static inline size_t accel_min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

#if HAS_NEON
static inline void bf16_convert_8_to_f32(const marmot_bfloat16_t *src, float *dst) {
    uint16x8_t bf16 = vld1q_u16((const uint16_t *)src);
    uint32x4_t lo_u32 = vshll_n_u16(vget_low_u16(bf16), 16);
    uint32x4_t hi_u32 = vshll_n_u16(vget_high_u16(bf16), 16);
    vst1q_f32(dst, vreinterpretq_f32_u32(lo_u32));
    vst1q_f32(dst + 4, vreinterpretq_f32_u32(hi_u32));
}
#endif

static inline void accel_convert_bf16_row(const marmot_bfloat16_t *src, float *dst, size_t count) {
#if HAS_NEON
    size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        bf16_convert_8_to_f32(src + k, dst + k);
    }
    for (; k < count; ++k) {
        dst[k] = marmot_bf16_to_f32_ref(src[k]);
    }
#else
    for (size_t k = 0; k < count; ++k) {
        dst[k] = marmot_bf16_to_f32_ref(src[k]);
    }
#endif
}

#if HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static inline void f16_convert_8_to_f32(const marmot_float16_t *src, float *dst) {
    uint16x8_t bits = vld1q_u16((const uint16_t *)src);
    float16x8_t h = vreinterpretq_f16_u16(bits);
    float32x4_t lo = vcvt_f32_f16(vget_low_f16(h));
    float32x4_t hi = vcvt_f32_f16(vget_high_f16(h));
    vst1q_f32(dst, lo);
    vst1q_f32(dst + 4, hi);
}
#endif

static inline void accel_convert_f16_row(const marmot_float16_t *src, float *dst, size_t count) {
#if HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        f16_convert_8_to_f32(src + k, dst + k);
    }
    for (; k < count; ++k) {
        dst[k] = marmot_f16_to_f32_ref(src[k]);
    }
#else
    for (size_t k = 0; k < count; ++k) {
        dst[k] = marmot_f16_to_f32_ref(src[k]);
    }
#endif
}

#if HAS_NEON
static inline void f32_convert_8_to_bf16(const float *src, marmot_bfloat16_t *dst) {
    const uint32x4_t bias = vdupq_n_u32(0x7FFF);
    const uint32x4_t lsb_mask = vdupq_n_u32(1);

    float32x4_t lo_f32 = vld1q_f32(src);
    float32x4_t hi_f32 = vld1q_f32(src + 4);
    uint32x4_t lo_bits = vreinterpretq_u32_f32(lo_f32);
    uint32x4_t hi_bits = vreinterpretq_u32_f32(hi_f32);
    uint32x4_t lo_round = vaddq_u32(lo_bits, vaddq_u32(bias, vandq_u32(vshrq_n_u32(lo_bits, 16), lsb_mask)));
    uint32x4_t hi_round = vaddq_u32(hi_bits, vaddq_u32(bias, vandq_u32(vshrq_n_u32(hi_bits, 16), lsb_mask)));
    uint16x8_t packed = vcombine_u16(vshrn_n_u32(lo_round, 16), vshrn_n_u32(hi_round, 16));
    vst1q_u16((uint16_t *)dst, packed);
}
#endif

static inline void accel_convert_f32_to_bf16_row(const float *src, marmot_bfloat16_t *dst, size_t count) {
#if HAS_NEON
    size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        f32_convert_8_to_bf16(src + k, dst + k);
    }
    for (; k < count; ++k) {
        dst[k] = marmot_f32_to_bf16_ref(src[k]);
    }
#else
    for (size_t k = 0; k < count; ++k) {
        dst[k] = marmot_f32_to_bf16_ref(src[k]);
    }
#endif
}

static inline void accel_convert_f32_to_f16_row(const float *src, marmot_float16_t *dst, size_t count) {
#if HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        float32x4_t lo = vld1q_f32(src + k);
        float32x4_t hi = vld1q_f32(src + k + 4);
        float16x4_t lo_h = vcvt_f16_f32(lo);
        float16x4_t hi_h = vcvt_f16_f32(hi);
        vst1_u16((uint16_t *)(dst + k), vreinterpret_u16_f16(lo_h));
        vst1_u16((uint16_t *)(dst + k + 4), vreinterpret_u16_f16(hi_h));
    }
    for (; k < count; ++k) {
        dst[k] = marmot_f32_to_f16_ref(src[k]);
    }
#else
    for (size_t k = 0; k < count; ++k) {
        dst[k] = marmot_f32_to_f16_ref(src[k]);
    }
#endif
}

#if MARMOT_ENABLE_FP8
#if HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static inline void fp8_e4m3_convert_8_to_f32(const marmot_float8_e4m3_t *src, float *dst) {
    uint8x8_t raw = vld1_u8((const uint8_t *)src);
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
            dst[lane] = marmot_fp8_e4m3_to_f32_ref(src[lane]);
        }
        return;
    }

    uint16x8_t sign_f16 = vshlq_n_u16(sign, 8);
    uint16x8_t exp_f16 = vaddq_u16(exp, vdupq_n_u16(8));
    uint16x8_t mant_f16 = vshlq_n_u16(mant, 7);
    uint16x8_t f16_bits = vorrq_u16(sign_f16, vorrq_u16(vshlq_n_u16(exp_f16, 10), mant_f16));

    float16x4_t f16_lo = vreinterpret_f16_u16(vget_low_u16(f16_bits));
    float16x4_t f16_hi = vreinterpret_f16_u16(vget_high_u16(f16_bits));
    vst1q_f32(dst, vcvt_f32_f16(f16_lo));
    vst1q_f32(dst + 4, vcvt_f32_f16(f16_hi));
}

static inline void fp8_e5m2_convert_8_to_f32(const marmot_float8_e5m2_t *src, float *dst) {
    uint8x8_t raw = vld1_u8((const uint8_t *)src);
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
            dst[lane] = marmot_fp8_e5m2_to_f32_ref(src[lane]);
        }
        return;
    }

    uint16x8_t sign_f16 = vshlq_n_u16(sign, 8);
    uint16x8_t exp_f16 = exp;
    uint16x8_t mant_f16 = vshlq_n_u16(mant, 8);
    uint16x8_t f16_bits = vorrq_u16(sign_f16, vorrq_u16(vshlq_n_u16(exp_f16, 10), mant_f16));

    float16x4_t f16_lo = vreinterpret_f16_u16(vget_low_u16(f16_bits));
    float16x4_t f16_hi = vreinterpret_f16_u16(vget_high_u16(f16_bits));
    vst1q_f32(dst, vcvt_f32_f16(f16_lo));
    vst1q_f32(dst + 4, vcvt_f32_f16(f16_hi));
}
#endif

static inline void accel_convert_fp8_e4m3_row(const marmot_float8_e4m3_t *src, float *dst, size_t count) {
#if HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        fp8_e4m3_convert_8_to_f32(src + k, dst + k);
    }
    for (; k < count; ++k) {
        dst[k] = marmot_fp8_e4m3_to_f32_ref(src[k]);
    }
#else
    for (size_t k = 0; k < count; ++k) {
        dst[k] = marmot_fp8_e4m3_to_f32_ref(src[k]);
    }
#endif
}

static inline void accel_convert_fp8_e5m2_row(const marmot_float8_e5m2_t *src, float *dst, size_t count) {
#if HAS_NEON && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        fp8_e5m2_convert_8_to_f32(src + k, dst + k);
    }
    for (; k < count; ++k) {
        dst[k] = marmot_fp8_e5m2_to_f32_ref(src[k]);
    }
#else
    for (size_t k = 0; k < count; ++k) {
        dst[k] = marmot_fp8_e5m2_to_f32_ref(src[k]);
    }
#endif
}
#endif

typedef struct {
    void (*pack_a)(const void *, size_t, size_t, size_t, size_t, size_t, float *);
    void (*pack_b_nt)(const void *, size_t, size_t, size_t, size_t, size_t, float *);
    void (*pack_b_nn)(const void *, size_t, size_t, size_t, size_t, size_t, float *);
    void (*store)(const float *, void *, size_t, size_t, size_t, size_t, size_t);
} accel_pack_ops_t;

static void accel_pack_a_panel_bf16(
    const void *input, size_t lda, size_t n_start, size_t n_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_bfloat16_t *src = (const marmot_bfloat16_t *)input;
    for (size_t n = 0; n < n_block; ++n) {
        const marmot_bfloat16_t *row = src + (n_start + n) * lda + k_start;
        accel_convert_bf16_row(row, dst + n * k_block, k_block);
    }
}

static void accel_pack_b_panel_nt_bf16(
    const void *weight, size_t ldw, size_t m_start, size_t m_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_bfloat16_t *src = (const marmot_bfloat16_t *)weight;
    for (size_t m = 0; m < m_block; ++m) {
        const marmot_bfloat16_t *row = src + (m_start + m) * ldw + k_start;
        accel_convert_bf16_row(row, dst + m * k_block, k_block);
    }
}

static void accel_pack_b_panel_nn_bf16(
    const void *weight, size_t ldw, size_t m_start, size_t m_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_bfloat16_t *src = (const marmot_bfloat16_t *)weight;
    for (size_t k = 0; k < k_block; ++k) {
        const marmot_bfloat16_t *row = src + (k_start + k) * ldw + m_start;
        accel_convert_bf16_row(row, dst + k * m_block, m_block);
    }
}

static void accel_pack_a_panel_f16(
    const void *input, size_t lda, size_t n_start, size_t n_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float16_t *src = (const marmot_float16_t *)input;
    for (size_t n = 0; n < n_block; ++n) {
        const marmot_float16_t *row = src + (n_start + n) * lda + k_start;
        accel_convert_f16_row(row, dst + n * k_block, k_block);
    }
}

static void accel_pack_b_panel_nt_f16(
    const void *weight, size_t ldw, size_t m_start, size_t m_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float16_t *src = (const marmot_float16_t *)weight;
    for (size_t m = 0; m < m_block; ++m) {
        const marmot_float16_t *row = src + (m_start + m) * ldw + k_start;
        accel_convert_f16_row(row, dst + m * k_block, k_block);
    }
}

static void accel_pack_b_panel_nn_f16(
    const void *weight, size_t ldw, size_t m_start, size_t m_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float16_t *src = (const marmot_float16_t *)weight;
    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float16_t *row = src + (k_start + k) * ldw + m_start;
        accel_convert_f16_row(row, dst + k * m_block, m_block);
    }
}

#if MARMOT_ENABLE_FP8
static void accel_pack_a_panel_fp8_e4m3(
    const void *input, size_t lda, size_t n_start, size_t n_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float8_e4m3_t *src = (const marmot_float8_e4m3_t *)input;
    for (size_t n = 0; n < n_block; ++n) {
        const marmot_float8_e4m3_t *row = src + (n_start + n) * lda + k_start;
        accel_convert_fp8_e4m3_row(row, dst + n * k_block, k_block);
    }
}

static void accel_pack_b_panel_nt_fp8_e4m3(
    const void *weight, size_t ldw, size_t m_start, size_t m_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float8_e4m3_t *src = (const marmot_float8_e4m3_t *)weight;
    for (size_t m = 0; m < m_block; ++m) {
        const marmot_float8_e4m3_t *row = src + (m_start + m) * ldw + k_start;
        accel_convert_fp8_e4m3_row(row, dst + m * k_block, k_block);
    }
}

static void accel_pack_b_panel_nn_fp8_e4m3(
    const void *weight, size_t ldw, size_t m_start, size_t m_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float8_e4m3_t *src = (const marmot_float8_e4m3_t *)weight;
    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float8_e4m3_t *row = src + (k_start + k) * ldw + m_start;
        accel_convert_fp8_e4m3_row(row, dst + k * m_block, m_block);
    }
}

static void accel_pack_a_panel_fp8_e5m2(
    const void *input, size_t lda, size_t n_start, size_t n_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float8_e5m2_t *src = (const marmot_float8_e5m2_t *)input;
    for (size_t n = 0; n < n_block; ++n) {
        const marmot_float8_e5m2_t *row = src + (n_start + n) * lda + k_start;
        accel_convert_fp8_e5m2_row(row, dst + n * k_block, k_block);
    }
}

static void accel_pack_b_panel_nt_fp8_e5m2(
    const void *weight, size_t ldw, size_t m_start, size_t m_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float8_e5m2_t *src = (const marmot_float8_e5m2_t *)weight;
    for (size_t m = 0; m < m_block; ++m) {
        const marmot_float8_e5m2_t *row = src + (m_start + m) * ldw + k_start;
        accel_convert_fp8_e5m2_row(row, dst + m * k_block, k_block);
    }
}

static void accel_pack_b_panel_nn_fp8_e5m2(
    const void *weight, size_t ldw, size_t m_start, size_t m_block, size_t k_start, size_t k_block, float *dst
) {
    const marmot_float8_e5m2_t *src = (const marmot_float8_e5m2_t *)weight;
    for (size_t k = 0; k < k_block; ++k) {
        const marmot_float8_e5m2_t *row = src + (k_start + k) * ldw + m_start;
        accel_convert_fp8_e5m2_row(row, dst + k * m_block, m_block);
    }
}
#endif

static void accel_store_panel_bf16(
    const float *src, void *out, size_t ldo, size_t n_start, size_t n_block, size_t m_start, size_t m_block
) {
    marmot_bfloat16_t *dst_base = (marmot_bfloat16_t *)out;
    for (size_t n = 0; n < n_block; ++n) {
        const float *row_src = src + n * m_block;
        marmot_bfloat16_t *row_dst = dst_base + (n_start + n) * ldo + m_start;
        accel_convert_f32_to_bf16_row(row_src, row_dst, m_block);
    }
}

static void accel_store_panel_f16(
    const float *src, void *out, size_t ldo, size_t n_start, size_t n_block, size_t m_start, size_t m_block
) {
    marmot_float16_t *dst_base = (marmot_float16_t *)out;
    for (size_t n = 0; n < n_block; ++n) {
        const float *row_src = src + n * m_block;
        marmot_float16_t *row_dst = dst_base + (n_start + n) * ldo + m_start;
        accel_convert_f32_to_f16_row(row_src, row_dst, m_block);
    }
}

static void accel_store_panel_f32(
    const float *src, void *out, size_t ldo, size_t n_start, size_t n_block, size_t m_start, size_t m_block
) {
    float *dst_base = (float *)out;
    for (size_t n = 0; n < n_block; ++n) {
        const float *row_src = src + n * m_block;
        float *row_dst = dst_base + (n_start + n) * ldo + m_start;
        memcpy(row_dst, row_src, m_block * sizeof(float));
    }
}

static marmot_error_t accel_run_blocked(
    const accel_pack_ops_t *ops, const void *input, const void *weight, void *out, size_t N, size_t K, size_t M,
    bool layout_nt
) {
    const marmot_neon_f32_params_t *params = marmot_neon_f32_get_params();
    const size_t a_elems = params->block_n * params->block_k;
    const size_t b_elems = layout_nt ? params->block_m * params->block_k : params->block_k * params->block_m;
    const size_t c_elems = params->block_n * params->block_m;

    float *A32 = (float *)marmot_aligned_alloc(64, a_elems * sizeof(float));
    float *B32 = (float *)marmot_aligned_alloc(64, b_elems * sizeof(float));
    float *C32 = (float *)marmot_aligned_alloc(64, c_elems * sizeof(float));
    if (A32 == nullptr || B32 == nullptr || C32 == nullptr) {
        free(A32);
        free(B32);
        free(C32);
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Accelerate matmul scratch allocation failed");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    const size_t lda = K;
    const size_t ldw = layout_nt ? K : M;
    for (size_t m_outer = 0; m_outer < M; m_outer += params->block_m) {
        const size_t m_block = accel_min_size(params->block_m, M - m_outer);
        for (size_t n_outer = 0; n_outer < N; n_outer += params->block_n) {
            const size_t n_block = accel_min_size(params->block_n, N - n_outer);
            memset(C32, 0, n_block * m_block * sizeof(float));

            for (size_t k_outer = 0; k_outer < K; k_outer += params->block_k) {
                const size_t k_block = accel_min_size(params->block_k, K - k_outer);
                ops->pack_a(input, lda, n_outer, n_block, k_outer, k_block, A32);
                if (layout_nt) {
                    ops->pack_b_nt(weight, ldw, m_outer, m_block, k_outer, k_block, B32);
                } else {
                    ops->pack_b_nn(weight, ldw, m_outer, m_block, k_outer, k_block, B32);
                }

                const int m_int = (int)m_block;
                const int n_int = (int)n_block;
                const int k_int = (int)k_block;
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, layout_nt ? CblasTrans : CblasNoTrans, n_int, m_int, k_int, 1.0f, A32,
                    k_int, B32, layout_nt ? k_int : m_int, 1.0f, C32, m_int
                );
            }
            ops->store(C32, out, M, n_outer, n_block, m_outer, m_block);
        }
    }

    free(A32);
    free(B32);
    free(C32);
    return MARMOT_SUCCESS;
}

static marmot_error_t accel_validate_same_dtype(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_dtype_t dtype,
    const char *message
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Accelerate matmul received null tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->dtype != dtype || weight->dtype != dtype || out->dtype != dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, message);
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    return MARMOT_SUCCESS;
}

#if MARMOT_ENABLE_FP8
static marmot_error_t accel_validate_fp8(
    const marmot_tensor_t *input, const marmot_tensor_t *weight, const marmot_tensor_t *out, marmot_dtype_t dtype
) {
    if (input == nullptr || weight == nullptr || out == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "FP8 Accelerate matmul received null tensor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (input->dtype != dtype || weight->dtype != dtype) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 Accelerate matmul requires matching input and weight");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    if (out->dtype != MARMOT_DTYPE_FLOAT32) {
        marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 Accelerate matmul requires FLOAT32 output");
        return MARMOT_ERROR_UNSUPPORTED_DTYPE;
    }
    return MARMOT_SUCCESS;
}
#endif

static const accel_pack_ops_t ACCEL_OPS_BF16 = {
    .pack_a = accel_pack_a_panel_bf16,
    .pack_b_nt = accel_pack_b_panel_nt_bf16,
    .pack_b_nn = accel_pack_b_panel_nn_bf16,
    .store = accel_store_panel_bf16,
};

static const accel_pack_ops_t ACCEL_OPS_F16 = {
    .pack_a = accel_pack_a_panel_f16,
    .pack_b_nt = accel_pack_b_panel_nt_f16,
    .pack_b_nn = accel_pack_b_panel_nn_f16,
    .store = accel_store_panel_f16,
};

#if MARMOT_ENABLE_FP8
static const accel_pack_ops_t ACCEL_OPS_FP8_E4M3 = {
    .pack_a = accel_pack_a_panel_fp8_e4m3,
    .pack_b_nt = accel_pack_b_panel_nt_fp8_e4m3,
    .pack_b_nn = accel_pack_b_panel_nn_fp8_e4m3,
    .store = accel_store_panel_f32,
};

static const accel_pack_ops_t ACCEL_OPS_FP8_E5M2 = {
    .pack_a = accel_pack_a_panel_fp8_e5m2,
    .pack_b_nt = accel_pack_b_panel_nt_fp8_e5m2,
    .pack_b_nn = accel_pack_b_panel_nn_fp8_e5m2,
    .store = accel_store_panel_f32,
};
#endif

marmot_error_t cpu_matmul_bf16_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    marmot_error_t status = accel_validate_same_dtype(
        input, weight, out, MARMOT_DTYPE_BFLOAT16, "BF16 Accelerate matmul requires BF16 input, weight, and output"
    );
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return accel_run_blocked(&ACCEL_OPS_BF16, input->data, weight->data, out->data, N, K, M, true);
}

marmot_error_t cpu_matmul_bf16_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    marmot_error_t status = accel_validate_same_dtype(
        input, weight, out, MARMOT_DTYPE_BFLOAT16, "BF16 Accelerate matmul requires BF16 input, weight, and output"
    );
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return accel_run_blocked(&ACCEL_OPS_BF16, input->data, weight->data, out->data, N, K, M, false);
}

marmot_error_t cpu_matmul_f16_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    marmot_error_t status = accel_validate_same_dtype(
        input, weight, out, MARMOT_DTYPE_FLOAT16, "FP16 Accelerate matmul requires FP16 input, weight, and output"
    );
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return accel_run_blocked(&ACCEL_OPS_F16, input->data, weight->data, out->data, N, K, M, true);
}

marmot_error_t cpu_matmul_f16_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    marmot_error_t status = accel_validate_same_dtype(
        input, weight, out, MARMOT_DTYPE_FLOAT16, "FP16 Accelerate matmul requires FP16 input, weight, and output"
    );
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return accel_run_blocked(&ACCEL_OPS_F16, input->data, weight->data, out->data, N, K, M, false);
}

#if MARMOT_ENABLE_FP8
marmot_error_t cpu_matmul_fp8_e4m3_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    marmot_error_t status = accel_validate_fp8(input, weight, out, MARMOT_DTYPE_FLOAT8_E4M3);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return accel_run_blocked(&ACCEL_OPS_FP8_E4M3, input->data, weight->data, out->data, N, K, M, true);
}

marmot_error_t cpu_matmul_fp8_e4m3_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    marmot_error_t status = accel_validate_fp8(input, weight, out, MARMOT_DTYPE_FLOAT8_E4M3);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return accel_run_blocked(&ACCEL_OPS_FP8_E4M3, input->data, weight->data, out->data, N, K, M, false);
}

marmot_error_t cpu_matmul_fp8_e5m2_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    marmot_error_t status = accel_validate_fp8(input, weight, out, MARMOT_DTYPE_FLOAT8_E5M2);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return accel_run_blocked(&ACCEL_OPS_FP8_E5M2, input->data, weight->data, out->data, N, K, M, true);
}

marmot_error_t cpu_matmul_fp8_e5m2_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    marmot_error_t status = accel_validate_fp8(input, weight, out, MARMOT_DTYPE_FLOAT8_E5M2);
    if (status != MARMOT_SUCCESS) {
        return status;
    }
    return accel_run_blocked(&ACCEL_OPS_FP8_E5M2, input->data, weight->data, out->data, N, K, M, false);
}
#else
marmot_error_t cpu_matmul_fp8_e4m3_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 Accelerate matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e5m2_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 Accelerate matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e4m3_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 Accelerate matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}

marmot_error_t cpu_matmul_fp8_e5m2_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    (void)input;
    (void)weight;
    (void)N;
    (void)K;
    (void)M;
    (void)out;
    marmot_set_error(MARMOT_ERROR_UNSUPPORTED_DTYPE, "FP8 Accelerate matmul not enabled");
    return MARMOT_ERROR_UNSUPPORTED_DTYPE;
}
#endif

marmot_error_t cpu_matmul_f32_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    const float *input_data = (const float *)input->data;
    const float *weight_data = (const float *)weight->data;
    float *out_data = (float *)out->data;
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, (int)N, (int)M, (int)K, 1.0f, input_data, (int)K, weight_data, (int)K,
        0.0f, out_data, (int)M
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_f64_accelerate(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    const double *input_data = (const double *)input->data;
    const double *weight_data = (const double *)weight->data;
    double *out_data = (double *)out->data;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans, (int)N, (int)M, (int)K, 1.0, input_data, (int)K, weight_data, (int)K,
        0.0, out_data, (int)M
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_f32_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    const float *input_data = (const float *)input->data;
    const float *weight_data = (const float *)weight->data;
    float *out_data = (float *)out->data;
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)M, (int)K, 1.0f, input_data, (int)K, weight_data,
        (int)M, 0.0f, out_data, (int)M
    );
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_matmul_f64_accelerate_nn(
    const void *device_ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight, size_t N, size_t K, size_t M,
    marmot_tensor_t *out
) {
    (void)device_ctx;
    const double *input_data = (const double *)input->data;
    const double *weight_data = (const double *)weight->data;
    double *out_data = (double *)out->data;
    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)N, (int)M, (int)K, 1.0, input_data, (int)K, weight_data, (int)M,
        0.0, out_data, (int)M
    );
    return MARMOT_SUCCESS;
}

#endif // MARMOT_ENABLE_ACCELERATE
