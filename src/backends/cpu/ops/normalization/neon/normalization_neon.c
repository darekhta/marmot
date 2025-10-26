#include "ops/normalization/normalization_internal.h"

#if MARMOT_ENABLE_NEON
#if HAS_NEON
static inline float neon_sum_float32x4(float32x4_t v) {
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    lo = vadd_f32(lo, hi);
    lo = vpadd_f32(lo, lo);
    return vget_lane_f32(lo, 0);
#endif
}
#endif
#if HAS_NEON
static inline void
compute_mean_variance_neon(const float *data, const float *residual, size_t n, float *mean_out, float *var_out) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_vec = vdupq_n_f32(0.0f);
    size_t i = 0;

    if (residual != nullptr) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(data + i);
            float32x4_t r = vld1q_f32(residual + i);
            v = vaddq_f32(v, r);
            sum_vec = vaddq_f32(sum_vec, v);
#if defined(__aarch64__)
            sum_sq_vec = vfmaq_f32(sum_sq_vec, v, v);
#else
            sum_sq_vec = vmlaq_f32(sum_sq_vec, v, v);
#endif
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            float32x4_t v = vld1q_f32(data + i);
            sum_vec = vaddq_f32(sum_vec, v);
#if defined(__aarch64__)
            sum_sq_vec = vfmaq_f32(sum_sq_vec, v, v);
#else
            sum_sq_vec = vmlaq_f32(sum_sq_vec, v, v);
#endif
        }
    }

    float sum = neon_sum_float32x4(sum_vec);
    float sum_sq = neon_sum_float32x4(sum_sq_vec);

    for (; i < n; i++) {
        float v = data[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        sum += v;
        sum_sq += v * v;
    }

    float inv_n = 1.0f / (float)n;
    float mean = sum * inv_n;
    float variance = sum_sq * inv_n - mean * mean;
    if (variance < 0.0f) {
        variance = 0.0f;
    }
    *mean_out = mean;
    *var_out = variance;
}

// NEON-optimized normalization
static inline void normalize_neon(
    const float *x, const float *residual, float *out, size_t n, float mean, float inv_std, const float *weight,
    const float *bias
) {
    float32x4_t mean_vec = vdupq_n_f32(mean);
    float32x4_t inv_std_vec = vdupq_n_f32(inv_std);
    size_t i = 0;

    if (residual != nullptr) {
        if (weight != nullptr && bias != nullptr) {
            for (; i + 4 <= n; i += 4) {
                float32x4_t x_vec = vld1q_f32(x + i);
                float32x4_t r_vec = vld1q_f32(residual + i);
                float32x4_t w_vec = vld1q_f32(weight + i);
                float32x4_t b_vec = vld1q_f32(bias + i);

                x_vec = vaddq_f32(x_vec, r_vec);
                float32x4_t normalized = vsubq_f32(x_vec, mean_vec);
                normalized = vmulq_f32(normalized, inv_std_vec);
                normalized = vmlaq_f32(b_vec, normalized, w_vec);

                vst1q_f32(out + i, normalized);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = ((v - mean) * inv_std) * weight[i] + bias[i];
            }
        } else if (weight != nullptr) {
            for (; i + 4 <= n; i += 4) {
                float32x4_t x_vec = vld1q_f32(x + i);
                float32x4_t r_vec = vld1q_f32(residual + i);
                float32x4_t w_vec = vld1q_f32(weight + i);

                x_vec = vaddq_f32(x_vec, r_vec);
                float32x4_t normalized = vsubq_f32(x_vec, mean_vec);
                normalized = vmulq_f32(normalized, inv_std_vec);
                normalized = vmulq_f32(normalized, w_vec);

                vst1q_f32(out + i, normalized);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = ((v - mean) * inv_std) * weight[i];
            }
        } else if (bias != nullptr) {
            for (; i + 4 <= n; i += 4) {
                float32x4_t x_vec = vld1q_f32(x + i);
                float32x4_t r_vec = vld1q_f32(residual + i);
                float32x4_t b_vec = vld1q_f32(bias + i);

                x_vec = vaddq_f32(x_vec, r_vec);
                float32x4_t normalized = vsubq_f32(x_vec, mean_vec);
                normalized = vmulq_f32(normalized, inv_std_vec);
                normalized = vaddq_f32(normalized, b_vec);

                vst1q_f32(out + i, normalized);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = (v - mean) * inv_std + bias[i];
            }
        } else {
            for (; i + 4 <= n; i += 4) {
                float32x4_t x_vec = vld1q_f32(x + i);
                float32x4_t r_vec = vld1q_f32(residual + i);

                x_vec = vaddq_f32(x_vec, r_vec);
                float32x4_t normalized = vsubq_f32(x_vec, mean_vec);
                normalized = vmulq_f32(normalized, inv_std_vec);

                vst1q_f32(out + i, normalized);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = (v - mean) * inv_std;
            }
        }
    } else if (weight != nullptr && bias != nullptr) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t w_vec = vld1q_f32(weight + i);
            float32x4_t b_vec = vld1q_f32(bias + i);

            float32x4_t normalized = vsubq_f32(x_vec, mean_vec);
            normalized = vmulq_f32(normalized, inv_std_vec);
            normalized = vmlaq_f32(b_vec, normalized, w_vec);

            vst1q_f32(out + i, normalized);
        }
        for (; i < n; i++) {
            out[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
        }
    } else if (weight != nullptr) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t w_vec = vld1q_f32(weight + i);

            float32x4_t normalized = vsubq_f32(x_vec, mean_vec);
            normalized = vmulq_f32(normalized, inv_std_vec);
            normalized = vmulq_f32(normalized, w_vec);

            vst1q_f32(out + i, normalized);
        }
        for (; i < n; i++) {
            out[i] = ((x[i] - mean) * inv_std) * weight[i];
        }
    } else if (bias != nullptr) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t b_vec = vld1q_f32(bias + i);

            float32x4_t normalized = vsubq_f32(x_vec, mean_vec);
            normalized = vmulq_f32(normalized, inv_std_vec);
            normalized = vaddq_f32(normalized, b_vec);

            vst1q_f32(out + i, normalized);
        }
        for (; i < n; i++) {
            out[i] = (x[i] - mean) * inv_std + bias[i];
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);

            float32x4_t normalized = vsubq_f32(x_vec, mean_vec);
            normalized = vmulq_f32(normalized, inv_std_vec);

            vst1q_f32(out + i, normalized);
        }
        for (; i < n; i++) {
            out[i] = (x[i] - mean) * inv_std;
        }
    }
}
#endif
#if HAS_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static void layernorm_row_f16_neon(
    const marmot_float16_t *x_row, const marmot_float16_t *residual_row, const marmot_float16_t *weight,
    const marmot_float16_t *bias, size_t norm_size, float eps, marmot_float16_t *out_row
) {
    float32x4_t sum_lo = vdupq_n_f32(0.0f);
    float32x4_t sum_hi = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_lo = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_hi = vdupq_n_f32(0.0f);
    size_t i = 0;

    if (residual_row != nullptr) {
        for (; i + 8 <= norm_size; i += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + i));
            uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + i));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float16x8_t r_f16 = vreinterpretq_f16_u16(r_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            float32x4_t r_lo = vcvt_f32_f16(vget_low_f16(r_f16));
            float32x4_t r_hi = vcvt_f32_f16(vget_high_f16(r_f16));
            x_lo = vaddq_f32(x_lo, r_lo);
            x_hi = vaddq_f32(x_hi, r_hi);
            sum_lo = vaddq_f32(sum_lo, x_lo);
            sum_hi = vaddq_f32(sum_hi, x_hi);
#if defined(__aarch64__)
            sum_sq_lo = vfmaq_f32(sum_sq_lo, x_lo, x_lo);
            sum_sq_hi = vfmaq_f32(sum_sq_hi, x_hi, x_hi);
#else
            sum_sq_lo = vmlaq_f32(sum_sq_lo, x_lo, x_lo);
            sum_sq_hi = vmlaq_f32(sum_sq_hi, x_hi, x_hi);
#endif
        }
    } else {
        for (; i + 8 <= norm_size; i += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + i));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            sum_lo = vaddq_f32(sum_lo, x_lo);
            sum_hi = vaddq_f32(sum_hi, x_hi);
#if defined(__aarch64__)
            sum_sq_lo = vfmaq_f32(sum_sq_lo, x_lo, x_lo);
            sum_sq_hi = vfmaq_f32(sum_sq_hi, x_hi, x_hi);
#else
            sum_sq_lo = vmlaq_f32(sum_sq_lo, x_lo, x_lo);
            sum_sq_hi = vmlaq_f32(sum_sq_hi, x_hi, x_hi);
#endif
        }
    }

    float sum = neon_sum_float32x4(sum_lo) + neon_sum_float32x4(sum_hi);
    float sum_sq = neon_sum_float32x4(sum_sq_lo) + neon_sum_float32x4(sum_sq_hi);
    for (; i < norm_size; i++) {
        float v = (float)marmot_float16_to_native(x_row[i]);
        if (residual_row != nullptr) {
            v += (float)marmot_float16_to_native(residual_row[i]);
        }
        sum += v;
        sum_sq += v * v;
    }

    float mean = sum / (float)norm_size;
    float variance = (sum_sq / (float)norm_size) - (mean * mean);
    if (variance < 0.0f) {
        variance = 0.0f;
    }
    float inv_std = cpu_norm_inv_sqrt_f32(variance + eps);
    float32x4_t mean_vec = vdupq_n_f32(mean);
    float32x4_t inv_std_vec = vdupq_n_f32(inv_std);

    size_t k = 0;
    if (residual_row != nullptr) {
        if (weight != nullptr && bias != nullptr) {
            for (; k + 8 <= norm_size; k += 8) {
                uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
                uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + k));
                uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
                uint16x8_t b_bits = vld1q_u16((const uint16_t *)(bias + k));
                float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
                float16x8_t r_f16 = vreinterpretq_f16_u16(r_bits);
                float16x8_t w_f16 = vreinterpretq_f16_u16(w_bits);
                float16x8_t b_f16 = vreinterpretq_f16_u16(b_bits);
                float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
                float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
                float32x4_t r_lo = vcvt_f32_f16(vget_low_f16(r_f16));
                float32x4_t r_hi = vcvt_f32_f16(vget_high_f16(r_f16));
                float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(w_f16));
                float32x4_t w_hi = vcvt_f32_f16(vget_high_f16(w_f16));
                float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b_f16));
                float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b_f16));
                x_lo = vaddq_f32(x_lo, r_lo);
                x_hi = vaddq_f32(x_hi, r_hi);
                float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
                float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);
                norm_lo = vmlaq_f32(b_lo, norm_lo, w_lo);
                norm_hi = vmlaq_f32(b_hi, norm_hi, w_hi);
                float16x4_t out_lo = vcvt_f16_f32(norm_lo);
                float16x4_t out_hi = vcvt_f16_f32(norm_hi);
                float16x8_t packed = vcombine_f16(out_lo, out_hi);
                vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
            }
        } else if (weight != nullptr) {
            for (; k + 8 <= norm_size; k += 8) {
                uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
                uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + k));
                uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
                float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
                float16x8_t r_f16 = vreinterpretq_f16_u16(r_bits);
                float16x8_t w_f16 = vreinterpretq_f16_u16(w_bits);
                float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
                float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
                float32x4_t r_lo = vcvt_f32_f16(vget_low_f16(r_f16));
                float32x4_t r_hi = vcvt_f32_f16(vget_high_f16(r_f16));
                float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(w_f16));
                float32x4_t w_hi = vcvt_f32_f16(vget_high_f16(w_f16));
                x_lo = vaddq_f32(x_lo, r_lo);
                x_hi = vaddq_f32(x_hi, r_hi);
                float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
                float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);
                norm_lo = vmulq_f32(norm_lo, w_lo);
                norm_hi = vmulq_f32(norm_hi, w_hi);
                float16x4_t out_lo = vcvt_f16_f32(norm_lo);
                float16x4_t out_hi = vcvt_f16_f32(norm_hi);
                float16x8_t packed = vcombine_f16(out_lo, out_hi);
                vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
            }
        } else if (bias != nullptr) {
            for (; k + 8 <= norm_size; k += 8) {
                uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
                uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + k));
                uint16x8_t b_bits = vld1q_u16((const uint16_t *)(bias + k));
                float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
                float16x8_t r_f16 = vreinterpretq_f16_u16(r_bits);
                float16x8_t b_f16 = vreinterpretq_f16_u16(b_bits);
                float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
                float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
                float32x4_t r_lo = vcvt_f32_f16(vget_low_f16(r_f16));
                float32x4_t r_hi = vcvt_f32_f16(vget_high_f16(r_f16));
                float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b_f16));
                float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b_f16));
                x_lo = vaddq_f32(x_lo, r_lo);
                x_hi = vaddq_f32(x_hi, r_hi);
                float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
                float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);
                norm_lo = vaddq_f32(norm_lo, b_lo);
                norm_hi = vaddq_f32(norm_hi, b_hi);
                float16x4_t out_lo = vcvt_f16_f32(norm_lo);
                float16x4_t out_hi = vcvt_f16_f32(norm_hi);
                float16x8_t packed = vcombine_f16(out_lo, out_hi);
                vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
            }
        } else {
            for (; k + 8 <= norm_size; k += 8) {
                uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
                uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + k));
                float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
                float16x8_t r_f16 = vreinterpretq_f16_u16(r_bits);
                float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
                float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
                float32x4_t r_lo = vcvt_f32_f16(vget_low_f16(r_f16));
                float32x4_t r_hi = vcvt_f32_f16(vget_high_f16(r_f16));
                x_lo = vaddq_f32(x_lo, r_lo);
                x_hi = vaddq_f32(x_hi, r_hi);
                float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
                float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);
                float16x4_t out_lo = vcvt_f16_f32(norm_lo);
                float16x4_t out_hi = vcvt_f16_f32(norm_hi);
                float16x8_t packed = vcombine_f16(out_lo, out_hi);
                vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
            }
        }
    } else if (weight != nullptr && bias != nullptr) {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
            uint16x8_t b_bits = vld1q_u16((const uint16_t *)(bias + k));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float16x8_t w_f16 = vreinterpretq_f16_u16(w_bits);
            float16x8_t b_f16 = vreinterpretq_f16_u16(b_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(w_f16));
            float32x4_t w_hi = vcvt_f32_f16(vget_high_f16(w_f16));
            float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b_f16));
            float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b_f16));
            float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
            float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);
            norm_lo = vmlaq_f32(b_lo, norm_lo, w_lo);
            norm_hi = vmlaq_f32(b_hi, norm_hi, w_hi);
            float16x4_t out_lo = vcvt_f16_f32(norm_lo);
            float16x4_t out_hi = vcvt_f16_f32(norm_hi);
            float16x8_t packed = vcombine_f16(out_lo, out_hi);
            vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
        }
    } else if (weight != nullptr) {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float16x8_t w_f16 = vreinterpretq_f16_u16(w_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(w_f16));
            float32x4_t w_hi = vcvt_f32_f16(vget_high_f16(w_f16));
            float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
            float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);
            norm_lo = vmulq_f32(norm_lo, w_lo);
            norm_hi = vmulq_f32(norm_hi, w_hi);
            float16x4_t out_lo = vcvt_f16_f32(norm_lo);
            float16x4_t out_hi = vcvt_f16_f32(norm_hi);
            float16x8_t packed = vcombine_f16(out_lo, out_hi);
            vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
        }
    } else if (bias != nullptr) {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            uint16x8_t b_bits = vld1q_u16((const uint16_t *)(bias + k));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float16x8_t b_f16 = vreinterpretq_f16_u16(b_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            float32x4_t b_lo = vcvt_f32_f16(vget_low_f16(b_f16));
            float32x4_t b_hi = vcvt_f32_f16(vget_high_f16(b_f16));
            float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
            float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);
            norm_lo = vaddq_f32(norm_lo, b_lo);
            norm_hi = vaddq_f32(norm_hi, b_hi);
            float16x4_t out_lo = vcvt_f16_f32(norm_lo);
            float16x4_t out_hi = vcvt_f16_f32(norm_hi);
            float16x8_t packed = vcombine_f16(out_lo, out_hi);
            vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
        }
    } else {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
            float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);
            float16x4_t out_lo = vcvt_f16_f32(norm_lo);
            float16x4_t out_hi = vcvt_f16_f32(norm_hi);
            float16x8_t packed = vcombine_f16(out_lo, out_hi);
            vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
        }
    }

    for (; k < norm_size; k++) {
        float v = (float)marmot_float16_to_native(x_row[k]);
        if (residual_row != nullptr) {
            v += (float)marmot_float16_to_native(residual_row[k]);
        }
        float normalized = (v - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= (float)marmot_float16_to_native(weight[k]);
        }
        if (bias != nullptr) {
            normalized += (float)marmot_float16_to_native(bias[k]);
        }
        out_row[k] = marmot_f32_to_f16_ref(normalized);
    }
}
#endif
#if HAS_NEON
static void layernorm_row_bf16_neon(
    const marmot_bfloat16_t *x_row, const marmot_bfloat16_t *residual_row, const marmot_bfloat16_t *weight,
    const marmot_bfloat16_t *bias, size_t norm_size, float eps, marmot_bfloat16_t *out_row
) {
    float32x4_t sum_lo = vdupq_n_f32(0.0f);
    float32x4_t sum_hi = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_lo = vdupq_n_f32(0.0f);
    float32x4_t sum_sq_hi = vdupq_n_f32(0.0f);
    size_t i = 0;

    if (residual_row != nullptr) {
        for (; i + 8 <= norm_size; i += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + i));
            uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + i));
            float32x4_t x_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x_bits), 16));
            float32x4_t x_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x_bits), 16));
            float32x4_t r_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r_bits), 16));
            float32x4_t r_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r_bits), 16));
            x_lo = vaddq_f32(x_lo, r_lo);
            x_hi = vaddq_f32(x_hi, r_hi);
            sum_lo = vaddq_f32(sum_lo, x_lo);
            sum_hi = vaddq_f32(sum_hi, x_hi);
#if defined(__aarch64__)
            sum_sq_lo = vfmaq_f32(sum_sq_lo, x_lo, x_lo);
            sum_sq_hi = vfmaq_f32(sum_sq_hi, x_hi, x_hi);
#else
            sum_sq_lo = vmlaq_f32(sum_sq_lo, x_lo, x_lo);
            sum_sq_hi = vmlaq_f32(sum_sq_hi, x_hi, x_hi);
#endif
        }
    } else {
        for (; i + 8 <= norm_size; i += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + i));
            float32x4_t x_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x_bits), 16));
            float32x4_t x_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x_bits), 16));
            sum_lo = vaddq_f32(sum_lo, x_lo);
            sum_hi = vaddq_f32(sum_hi, x_hi);
#if defined(__aarch64__)
            sum_sq_lo = vfmaq_f32(sum_sq_lo, x_lo, x_lo);
            sum_sq_hi = vfmaq_f32(sum_sq_hi, x_hi, x_hi);
#else
            sum_sq_lo = vmlaq_f32(sum_sq_lo, x_lo, x_lo);
            sum_sq_hi = vmlaq_f32(sum_sq_hi, x_hi, x_hi);
#endif
        }
    }

    float sum = neon_sum_float32x4(sum_lo) + neon_sum_float32x4(sum_hi);
    float sum_sq = neon_sum_float32x4(sum_sq_lo) + neon_sum_float32x4(sum_sq_hi);
    for (; i < norm_size; i++) {
        float v = marmot_bf16_to_f32_ref(x_row[i]);
        if (residual_row != nullptr) {
            v += marmot_bf16_to_f32_ref(residual_row[i]);
        }
        sum += v;
        sum_sq += v * v;
    }

    float mean = sum / (float)norm_size;
    float variance = (sum_sq / (float)norm_size) - (mean * mean);
    if (variance < 0.0f) {
        variance = 0.0f;
    }

    float inv_std = cpu_norm_inv_sqrt_f32(variance + eps);
    float32x4_t inv_std_vec = vdupq_n_f32(inv_std);
    float32x4_t mean_vec = vdupq_n_f32(mean);

    size_t k = 0;
    if (residual_row != nullptr) {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + k));
            float32x4_t x_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x_bits), 16));
            float32x4_t x_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x_bits), 16));
            float32x4_t r_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r_bits), 16));
            float32x4_t r_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r_bits), 16));
            x_lo = vaddq_f32(x_lo, r_lo);
            x_hi = vaddq_f32(x_hi, r_hi);

            float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
            float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);

            if (weight != nullptr) {
                uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
                norm_lo = vmulq_f32(norm_lo, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_bits), 16)));
                norm_hi = vmulq_f32(norm_hi, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_bits), 16)));
            }

            if (bias != nullptr) {
                uint16x8_t b_bits = vld1q_u16((const uint16_t *)(bias + k));
                norm_lo = vaddq_f32(norm_lo, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b_bits), 16)));
                norm_hi = vaddq_f32(norm_hi, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b_bits), 16)));
            }

            uint32x4_t norm_lo_bits = vreinterpretq_u32_f32(norm_lo);
            uint32x4_t norm_hi_bits = vreinterpretq_u32_f32(norm_hi);
            uint32x4_t bias_bits = vdupq_n_u32(0x7FFF);
            uint32x4_t lsb_lo = vandq_u32(vshrq_n_u32(norm_lo_bits, 16), vdupq_n_u32(1));
            uint32x4_t lsb_hi = vandq_u32(vshrq_n_u32(norm_hi_bits, 16), vdupq_n_u32(1));
            uint32x4_t rounded_lo = vaddq_u32(norm_lo_bits, vaddq_u32(bias_bits, lsb_lo));
            uint32x4_t rounded_hi = vaddq_u32(norm_hi_bits, vaddq_u32(bias_bits, lsb_hi));
            uint16x4_t out_lo = vmovn_u32(vshrq_n_u32(rounded_lo, 16));
            uint16x4_t out_hi = vmovn_u32(vshrq_n_u32(rounded_hi, 16));
            uint16x8_t packed = vcombine_u16(out_lo, out_hi);
            vst1q_u16((uint16_t *)(out_row + k), packed);
        }
    } else {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            float32x4_t x_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x_bits), 16));
            float32x4_t x_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x_bits), 16));

            float32x4_t norm_lo = vmulq_f32(vsubq_f32(x_lo, mean_vec), inv_std_vec);
            float32x4_t norm_hi = vmulq_f32(vsubq_f32(x_hi, mean_vec), inv_std_vec);

            if (weight != nullptr) {
                uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
                norm_lo = vmulq_f32(norm_lo, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_bits), 16)));
                norm_hi = vmulq_f32(norm_hi, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_bits), 16)));
            }

            if (bias != nullptr) {
                uint16x8_t b_bits = vld1q_u16((const uint16_t *)(bias + k));
                norm_lo = vaddq_f32(norm_lo, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b_bits), 16)));
                norm_hi = vaddq_f32(norm_hi, vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(b_bits), 16)));
            }

            uint32x4_t norm_lo_bits = vreinterpretq_u32_f32(norm_lo);
            uint32x4_t norm_hi_bits = vreinterpretq_u32_f32(norm_hi);
            uint32x4_t bias_bits = vdupq_n_u32(0x7FFF);
            uint32x4_t lsb_lo = vandq_u32(vshrq_n_u32(norm_lo_bits, 16), vdupq_n_u32(1));
            uint32x4_t lsb_hi = vandq_u32(vshrq_n_u32(norm_hi_bits, 16), vdupq_n_u32(1));
            uint32x4_t rounded_lo = vaddq_u32(norm_lo_bits, vaddq_u32(bias_bits, lsb_lo));
            uint32x4_t rounded_hi = vaddq_u32(norm_hi_bits, vaddq_u32(bias_bits, lsb_hi));
            uint16x4_t out_lo = vmovn_u32(vshrq_n_u32(rounded_lo, 16));
            uint16x4_t out_hi = vmovn_u32(vshrq_n_u32(rounded_hi, 16));
            uint16x8_t packed = vcombine_u16(out_lo, out_hi);
            vst1q_u16((uint16_t *)(out_row + k), packed);
        }
    }

    for (; k < norm_size; k++) {
        float value = marmot_bf16_to_f32_ref(x_row[k]);
        if (residual_row != nullptr) {
            value += marmot_bf16_to_f32_ref(residual_row[k]);
        }
        float normalized = (value - mean) * inv_std;
        if (weight != nullptr) {
            normalized *= marmot_bf16_to_f32_ref(weight[k]);
        }
        if (bias != nullptr) {
            normalized += marmot_bf16_to_f32_ref(bias[k]);
        }
        out_row[k] = marmot_f32_to_bf16_ref(normalized);
    }
}
#endif // HAS_NEON
#if HAS_NEON
static float compute_mean_square_neon(const float *x, const float *residual, size_t n) {
    size_t i = 0;
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    if (residual != nullptr) {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t r_vec = vld1q_f32(residual + i);
            x_vec = vaddq_f32(x_vec, r_vec);
#if defined(__aarch64__)
            sum_vec = vfmaq_f32(sum_vec, x_vec, x_vec);
#else
            sum_vec = vmlaq_f32(sum_vec, x_vec, x_vec);
#endif
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
#if defined(__aarch64__)
            sum_vec = vfmaq_f32(sum_vec, x_vec, x_vec);
#else
            sum_vec = vmlaq_f32(sum_vec, x_vec, x_vec);
#endif
        }
    }

    float sum = neon_sum_float32x4(sum_vec);
    for (; i < n; i++) {
        float v = x[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        sum += v * v;
    }

    return sum / (float)n;
}

static void normalize_rms_neon(
    const float *x, const float *residual, float *out, size_t n, float inv_rms, const float *weight, float weight_offset
) {
    size_t i = 0;
    float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);

    if (residual != nullptr) {
        if (weight != nullptr) {
            float32x4_t offset_vec = vdupq_n_f32(weight_offset);
            for (; i + 4 <= n; i += 4) {
                float32x4_t x_vec = vld1q_f32(x + i);
                float32x4_t r_vec = vld1q_f32(residual + i);
                float32x4_t w_vec = vld1q_f32(weight + i);
                x_vec = vaddq_f32(x_vec, r_vec);
                w_vec = vaddq_f32(w_vec, offset_vec);
                float32x4_t out_vec = vmulq_f32(vmulq_f32(x_vec, inv_rms_vec), w_vec);
                vst1q_f32(out + i, out_vec);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = v * inv_rms * (weight[i] + weight_offset);
            }
        } else {
            for (; i + 4 <= n; i += 4) {
                float32x4_t x_vec = vld1q_f32(x + i);
                float32x4_t r_vec = vld1q_f32(residual + i);
                x_vec = vaddq_f32(x_vec, r_vec);
                float32x4_t out_vec = vmulq_f32(x_vec, inv_rms_vec);
                vst1q_f32(out + i, out_vec);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = v * inv_rms;
            }
        }
    } else if (weight != nullptr) {
        float32x4_t offset_vec = vdupq_n_f32(weight_offset);
        for (; i + 4 <= n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t w_vec = vld1q_f32(weight + i);
            w_vec = vaddq_f32(w_vec, offset_vec);
            float32x4_t out_vec = vmulq_f32(vmulq_f32(x_vec, inv_rms_vec), w_vec);
            vst1q_f32(out + i, out_vec);
        }
        for (; i < n; i++) {
            out[i] = x[i] * inv_rms * (weight[i] + weight_offset);
        }
    } else {
        for (; i + 4 <= n; i += 4) {
            float32x4_t x_vec = vld1q_f32(x + i);
            float32x4_t out_vec = vmulq_f32(x_vec, inv_rms_vec);
            vst1q_f32(out + i, out_vec);
        }
        for (; i < n; i++) {
            out[i] = x[i] * inv_rms;
        }
    }
}
#endif
#if HAS_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static void rmsnorm_row_f16_neon(
    const marmot_float16_t *x_row, const marmot_float16_t *residual_row, const marmot_float16_t *weight,
    size_t norm_size, float eps, float weight_offset, marmot_float16_t *out_row
) {
    float32x4_t sum_lo = vdupq_n_f32(0.0f);
    float32x4_t sum_hi = vdupq_n_f32(0.0f);
    size_t i = 0;

    if (residual_row != nullptr) {
        for (; i + 8 <= norm_size; i += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + i));
            uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + i));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float16x8_t r_f16 = vreinterpretq_f16_u16(r_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            float32x4_t r_lo = vcvt_f32_f16(vget_low_f16(r_f16));
            float32x4_t r_hi = vcvt_f32_f16(vget_high_f16(r_f16));
            x_lo = vaddq_f32(x_lo, r_lo);
            x_hi = vaddq_f32(x_hi, r_hi);
#if defined(__aarch64__)
            sum_lo = vfmaq_f32(sum_lo, x_lo, x_lo);
            sum_hi = vfmaq_f32(sum_hi, x_hi, x_hi);
#else
            sum_lo = vmlaq_f32(sum_lo, x_lo, x_lo);
            sum_hi = vmlaq_f32(sum_hi, x_hi, x_hi);
#endif
        }
    } else {
        for (; i + 8 <= norm_size; i += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + i));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
#if defined(__aarch64__)
            sum_lo = vfmaq_f32(sum_lo, x_lo, x_lo);
            sum_hi = vfmaq_f32(sum_hi, x_hi, x_hi);
#else
            sum_lo = vmlaq_f32(sum_lo, x_lo, x_lo);
            sum_hi = vmlaq_f32(sum_hi, x_hi, x_hi);
#endif
        }
    }

    float sum = neon_sum_float32x4(sum_lo) + neon_sum_float32x4(sum_hi);
    for (; i < norm_size; i++) {
        float value = (float)marmot_float16_to_native(x_row[i]);
        if (residual_row != nullptr) {
            value += (float)marmot_float16_to_native(residual_row[i]);
        }
        sum += value * value;
    }

    float mean_sq = sum / (float)norm_size;
    float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + eps);
    float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
    float32x4_t offset_vec = vdupq_n_f32(weight_offset);

    size_t k = 0;
    if (residual_row != nullptr) {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + k));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float16x8_t r_f16 = vreinterpretq_f16_u16(r_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            float32x4_t r_lo = vcvt_f32_f16(vget_low_f16(r_f16));
            float32x4_t r_hi = vcvt_f32_f16(vget_high_f16(r_f16));
            x_lo = vaddq_f32(x_lo, r_lo);
            x_hi = vaddq_f32(x_hi, r_hi);
            float32x4_t out_lo = vmulq_f32(x_lo, inv_rms_vec);
            float32x4_t out_hi = vmulq_f32(x_hi, inv_rms_vec);

            if (weight != nullptr) {
                uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
                float16x8_t w_f16 = vreinterpretq_f16_u16(w_bits);
                float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(w_f16));
                float32x4_t w_hi = vcvt_f32_f16(vget_high_f16(w_f16));
                w_lo = vaddq_f32(w_lo, offset_vec);
                w_hi = vaddq_f32(w_hi, offset_vec);
                out_lo = vmulq_f32(out_lo, w_lo);
                out_hi = vmulq_f32(out_hi, w_hi);
            }

            float16x4_t out_lo_f16 = vcvt_f16_f32(out_lo);
            float16x4_t out_hi_f16 = vcvt_f16_f32(out_hi);
            float16x8_t packed = vcombine_f16(out_lo_f16, out_hi_f16);
            vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
        }
    } else {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            float16x8_t x_f16 = vreinterpretq_f16_u16(x_bits);
            float32x4_t x_lo = vcvt_f32_f16(vget_low_f16(x_f16));
            float32x4_t x_hi = vcvt_f32_f16(vget_high_f16(x_f16));
            float32x4_t out_lo = vmulq_f32(x_lo, inv_rms_vec);
            float32x4_t out_hi = vmulq_f32(x_hi, inv_rms_vec);

            if (weight != nullptr) {
                uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
                float16x8_t w_f16 = vreinterpretq_f16_u16(w_bits);
                float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(w_f16));
                float32x4_t w_hi = vcvt_f32_f16(vget_high_f16(w_f16));
                w_lo = vaddq_f32(w_lo, offset_vec);
                w_hi = vaddq_f32(w_hi, offset_vec);
                out_lo = vmulq_f32(out_lo, w_lo);
                out_hi = vmulq_f32(out_hi, w_hi);
            }

            float16x4_t out_lo_f16 = vcvt_f16_f32(out_lo);
            float16x4_t out_hi_f16 = vcvt_f16_f32(out_hi);
            float16x8_t packed = vcombine_f16(out_lo_f16, out_hi_f16);
            vst1q_u16((uint16_t *)(out_row + k), vreinterpretq_u16_f16(packed));
        }
    }

    for (; k < norm_size; k++) {
        float value = (float)marmot_float16_to_native(x_row[k]);
        if (residual_row != nullptr) {
            value += (float)marmot_float16_to_native(residual_row[k]);
        }
        value *= inv_rms;
        if (weight != nullptr) {
            value *= (float)marmot_float16_to_native(weight[k]) + weight_offset;
        }
        out_row[k] = marmot_f32_to_f16_ref(value);
    }
}
#endif
#if HAS_NEON
static void rmsnorm_row_bf16_neon(
    const marmot_bfloat16_t *x_row, const marmot_bfloat16_t *residual_row, const marmot_bfloat16_t *weight,
    size_t norm_size, float eps, float weight_offset, marmot_bfloat16_t *out_row
) {
    float32x4_t sum_lo = vdupq_n_f32(0.0f);
    float32x4_t sum_hi = vdupq_n_f32(0.0f);
    size_t i = 0;

    if (residual_row != nullptr) {
        for (; i + 8 <= norm_size; i += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + i));
            uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + i));
            float32x4_t x_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x_bits), 16));
            float32x4_t x_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x_bits), 16));
            float32x4_t r_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r_bits), 16));
            float32x4_t r_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r_bits), 16));
            x_lo = vaddq_f32(x_lo, r_lo);
            x_hi = vaddq_f32(x_hi, r_hi);
#if defined(__aarch64__)
            sum_lo = vfmaq_f32(sum_lo, x_lo, x_lo);
            sum_hi = vfmaq_f32(sum_hi, x_hi, x_hi);
#else
            sum_lo = vmlaq_f32(sum_lo, x_lo, x_lo);
            sum_hi = vmlaq_f32(sum_hi, x_hi, x_hi);
#endif
        }
    } else {
        for (; i + 8 <= norm_size; i += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + i));
            float32x4_t x_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x_bits), 16));
            float32x4_t x_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x_bits), 16));
#if defined(__aarch64__)
            sum_lo = vfmaq_f32(sum_lo, x_lo, x_lo);
            sum_hi = vfmaq_f32(sum_hi, x_hi, x_hi);
#else
            sum_lo = vmlaq_f32(sum_lo, x_lo, x_lo);
            sum_hi = vmlaq_f32(sum_hi, x_hi, x_hi);
#endif
        }
    }

    float sum = neon_sum_float32x4(sum_lo) + neon_sum_float32x4(sum_hi);
    for (; i < norm_size; i++) {
        float value = marmot_bf16_to_f32_ref(x_row[i]);
        if (residual_row != nullptr) {
            value += marmot_bf16_to_f32_ref(residual_row[i]);
        }
        sum += value * value;
    }

    float mean_sq = sum / (float)norm_size;
    float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + eps);
    float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);

    size_t k = 0;
    float32x4_t offset_vec = vdupq_n_f32(weight_offset);
    if (residual_row != nullptr) {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            uint16x8_t r_bits = vld1q_u16((const uint16_t *)(residual_row + k));
            float32x4_t x_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x_bits), 16));
            float32x4_t x_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x_bits), 16));
            float32x4_t r_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(r_bits), 16));
            float32x4_t r_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(r_bits), 16));
            x_lo = vaddq_f32(x_lo, r_lo);
            x_hi = vaddq_f32(x_hi, r_hi);
            float32x4_t out_lo = vmulq_f32(x_lo, inv_rms_vec);
            float32x4_t out_hi = vmulq_f32(x_hi, inv_rms_vec);

            if (weight != nullptr) {
                uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
                float32x4_t w_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_bits), 16));
                float32x4_t w_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_bits), 16));
                w_lo = vaddq_f32(w_lo, offset_vec);
                w_hi = vaddq_f32(w_hi, offset_vec);
                out_lo = vmulq_f32(out_lo, w_lo);
                out_hi = vmulq_f32(out_hi, w_hi);
            }

            uint32x4_t out_lo_bits = vreinterpretq_u32_f32(out_lo);
            uint32x4_t out_hi_bits = vreinterpretq_u32_f32(out_hi);
            uint32x4_t bias_bits = vdupq_n_u32(0x7FFF);
            uint32x4_t lsb_lo = vandq_u32(vshrq_n_u32(out_lo_bits, 16), vdupq_n_u32(1));
            uint32x4_t lsb_hi = vandq_u32(vshrq_n_u32(out_hi_bits, 16), vdupq_n_u32(1));
            uint32x4_t rounded_lo = vaddq_u32(out_lo_bits, vaddq_u32(bias_bits, lsb_lo));
            uint32x4_t rounded_hi = vaddq_u32(out_hi_bits, vaddq_u32(bias_bits, lsb_hi));
            uint16x4_t out_lo_f16 = vmovn_u32(vshrq_n_u32(rounded_lo, 16));
            uint16x4_t out_hi_f16 = vmovn_u32(vshrq_n_u32(rounded_hi, 16));
            uint16x8_t packed = vcombine_u16(out_lo_f16, out_hi_f16);
            vst1q_u16((uint16_t *)(out_row + k), packed);
        }
    } else {
        for (; k + 8 <= norm_size; k += 8) {
            uint16x8_t x_bits = vld1q_u16((const uint16_t *)(x_row + k));
            float32x4_t x_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(x_bits), 16));
            float32x4_t x_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(x_bits), 16));
            float32x4_t out_lo = vmulq_f32(x_lo, inv_rms_vec);
            float32x4_t out_hi = vmulq_f32(x_hi, inv_rms_vec);

            if (weight != nullptr) {
                uint16x8_t w_bits = vld1q_u16((const uint16_t *)(weight + k));
                float32x4_t w_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(w_bits), 16));
                float32x4_t w_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(w_bits), 16));
                w_lo = vaddq_f32(w_lo, offset_vec);
                w_hi = vaddq_f32(w_hi, offset_vec);
                out_lo = vmulq_f32(out_lo, w_lo);
                out_hi = vmulq_f32(out_hi, w_hi);
            }

            uint32x4_t out_lo_bits = vreinterpretq_u32_f32(out_lo);
            uint32x4_t out_hi_bits = vreinterpretq_u32_f32(out_hi);
            uint32x4_t bias_bits = vdupq_n_u32(0x7FFF);
            uint32x4_t lsb_lo = vandq_u32(vshrq_n_u32(out_lo_bits, 16), vdupq_n_u32(1));
            uint32x4_t lsb_hi = vandq_u32(vshrq_n_u32(out_hi_bits, 16), vdupq_n_u32(1));
            uint32x4_t rounded_lo = vaddq_u32(out_lo_bits, vaddq_u32(bias_bits, lsb_lo));
            uint32x4_t rounded_hi = vaddq_u32(out_hi_bits, vaddq_u32(bias_bits, lsb_hi));
            uint16x4_t out_lo_f16 = vmovn_u32(vshrq_n_u32(rounded_lo, 16));
            uint16x4_t out_hi_f16 = vmovn_u32(vshrq_n_u32(rounded_hi, 16));
            uint16x8_t packed = vcombine_u16(out_lo_f16, out_hi_f16);
            vst1q_u16((uint16_t *)(out_row + k), packed);
        }
    }

    for (; k < norm_size; k++) {
        float value = marmot_bf16_to_f32_ref(x_row[k]);
        if (residual_row != nullptr) {
            value += marmot_bf16_to_f32_ref(residual_row[k]);
        }
        value *= inv_rms;
        if (weight != nullptr) {
            value *= marmot_bf16_to_f32_ref(weight[k]) + weight_offset;
        }
        out_row[k] = marmot_f32_to_bf16_ref(value);
    }
}
#endif // HAS_NEON
#if HAS_NEON
static void layernorm_f32_neon_range(void *context, size_t start, size_t end) {
    layernorm_f32_context_t *ctx = (layernorm_f32_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const float *x_row = ctx->x + row * ctx->norm_size;
        const float *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        float *out_row = ctx->out + row * ctx->norm_size;
        float mean = 0.0f;
        float variance = 0.0f;
        compute_mean_variance_neon(x_row, res_row, ctx->norm_size, &mean, &variance);
        float inv_std = cpu_norm_inv_sqrt_f32(variance + ctx->eps);
        normalize_neon(x_row, res_row, out_row, ctx->norm_size, mean, inv_std, ctx->weight, ctx->bias);
    }
}

static void rmsnorm_f32_neon_range(void *context, size_t start, size_t end) {
    rmsnorm_f32_context_t *ctx = (rmsnorm_f32_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const float *x_row = ctx->x + row * ctx->norm_size;
        const float *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        float *out_row = ctx->out + row * ctx->norm_size;
        float mean_sq = compute_mean_square_neon(x_row, res_row, ctx->norm_size);
        float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + ctx->eps);
        normalize_rms_neon(x_row, res_row, out_row, ctx->norm_size, inv_rms, ctx->weight, ctx->weight_offset);
    }
}
#endif
#if HAS_NEON
static marmot_error_t cpu_layernorm_f32_neon(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    layernorm_f32_context_t ctx = {
        .x = (const float *)params->x,
        .residual = (const float *)params->residual,
        .out = (float *)params->out,
        .weight = (const float *)params->weight,
        .bias = (const float *)params->bias,
        .norm_size = params->norm_size,
        .eps = params->eps,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_f32_neon_range);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_f32_neon(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    rmsnorm_f32_context_t ctx = {
        .x = (const float *)params->x,
        .residual = (const float *)params->residual,
        .out = (float *)params->out,
        .weight = (const float *)params->weight,
        .norm_size = params->norm_size,
        .eps = params->eps,
        .weight_offset = params->weight_offset,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_f32_neon_range);
    return MARMOT_SUCCESS;
}
#endif
#if HAS_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static void layernorm_f16_neon_range(void *context, size_t start, size_t end) {
    layernorm_f16_context_t *ctx = (layernorm_f16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_float16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_float16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_float16_t *out_row = ctx->out + row * ctx->norm_size;
        layernorm_row_f16_neon(x_row, res_row, ctx->weight, ctx->bias, ctx->norm_size, ctx->eps, out_row);
    }
}

static void rmsnorm_f16_neon_range(void *context, size_t start, size_t end) {
    rmsnorm_f16_context_t *ctx = (rmsnorm_f16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_float16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_float16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_float16_t *out_row = ctx->out + row * ctx->norm_size;
        rmsnorm_row_f16_neon(x_row, res_row, ctx->weight, ctx->norm_size, ctx->eps, ctx->weight_offset, out_row);
    }
}
#endif
#if HAS_NEON
static void layernorm_bf16_neon_range(void *context, size_t start, size_t end) {
    layernorm_bf16_context_t *ctx = (layernorm_bf16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_bfloat16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_bfloat16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_bfloat16_t *out_row = ctx->out + row * ctx->norm_size;
        layernorm_row_bf16_neon(x_row, res_row, ctx->weight, ctx->bias, ctx->norm_size, ctx->eps, out_row);
    }
}
#endif
#if HAS_NEON
static void rmsnorm_bf16_neon_range(void *context, size_t start, size_t end) {
    rmsnorm_bf16_context_t *ctx = (rmsnorm_bf16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_bfloat16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_bfloat16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_bfloat16_t *out_row = ctx->out + row * ctx->norm_size;
        rmsnorm_row_bf16_neon(x_row, res_row, ctx->weight, ctx->norm_size, ctx->eps, ctx->weight_offset, out_row);
    }
}
#endif
#if HAS_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static marmot_error_t cpu_layernorm_f16_neon(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    layernorm_f16_context_t ctx = {
        .x = (const marmot_float16_t *)params->x,
        .residual = (const marmot_float16_t *)params->residual,
        .out = (marmot_float16_t *)params->out,
        .weight = (const marmot_float16_t *)params->weight,
        .bias = (const marmot_float16_t *)params->bias,
        .norm_size = params->norm_size,
        .eps = params->eps,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_f16_neon_range);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_f16_neon(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    rmsnorm_f16_context_t ctx = {
        .x = (const marmot_float16_t *)params->x,
        .residual = (const marmot_float16_t *)params->residual,
        .out = (marmot_float16_t *)params->out,
        .weight = (const marmot_float16_t *)params->weight,
        .norm_size = params->norm_size,
        .eps = params->eps,
        .weight_offset = params->weight_offset,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_f16_neon_range);
    return MARMOT_SUCCESS;
}
#endif
#if HAS_NEON
static marmot_error_t cpu_layernorm_bf16_neon(const void *device_ctx, const cpu_layernorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    layernorm_bf16_context_t ctx = {
        .x = (const marmot_bfloat16_t *)params->x,
        .residual = (const marmot_bfloat16_t *)params->residual,
        .out = (marmot_bfloat16_t *)params->out,
        .weight = (const marmot_bfloat16_t *)params->weight,
        .bias = (const marmot_bfloat16_t *)params->bias,
        .norm_size = params->norm_size,
        .eps = params->eps,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_bf16_neon_range);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_bf16_neon(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
    (void)device_ctx;
    if (params == nullptr || params->x == nullptr || params->out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (cpu_norm_no_work(params->norm_size, params->outer_size)) {
        return MARMOT_SUCCESS;
    }
    rmsnorm_bf16_context_t ctx = {
        .x = (const marmot_bfloat16_t *)params->x,
        .residual = (const marmot_bfloat16_t *)params->residual,
        .out = (marmot_bfloat16_t *)params->out,
        .weight = (const marmot_bfloat16_t *)params->weight,
        .norm_size = params->norm_size,
        .eps = params->eps,
        .weight_offset = params->weight_offset,
    };
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_bf16_neon_range);
    return MARMOT_SUCCESS;
}
#endif
#if HAS_NEON
const cpu_norm_traits_t cpu_norm_f32_neon_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_NORM_IMPL_NEON,
    .ops = {
        .layernorm = cpu_layernorm_f32_neon,
        .rmsnorm = cpu_rmsnorm_f32_neon,
        .impl_name = "f32_neon",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_f32_neon_traits)
#endif
#if HAS_NEON && defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
const cpu_norm_traits_t cpu_norm_f16_neon_traits = {
    .dtype = MARMOT_DTYPE_FLOAT16,
    .impl_kind = CPU_NORM_IMPL_NEON,
    .ops = {
        .layernorm = cpu_layernorm_f16_neon,
        .rmsnorm = cpu_rmsnorm_f16_neon,
        .impl_name = "f16_neon",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_f16_neon_traits)
#endif
#if HAS_NEON
const cpu_norm_traits_t cpu_norm_bf16_neon_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = CPU_NORM_IMPL_NEON,
    .ops = {
        .layernorm = cpu_layernorm_bf16_neon,
        .rmsnorm = cpu_rmsnorm_bf16_neon,
        .impl_name = "bf16_neon",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_bf16_neon_traits)
#endif

#endif // MARMOT_ENABLE_NEON
