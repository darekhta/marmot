#include "ops/normalization/normalization_internal.h"

#if MARMOT_ENABLE_AVX2
#if HAS_AVX2
static inline float horizontal_sum_m256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif
#if HAS_AVX2
static inline void
compute_mean_variance_avx2(const float *data, const float *residual, size_t n, float *mean_out, float *var_out) {
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 sum_sq_vec = _mm256_setzero_ps();
    size_t i = 0;

    if (residual != nullptr) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            __m256 r = _mm256_loadu_ps(residual + i);
            v = _mm256_add_ps(v, r);
            sum_vec = _mm256_add_ps(sum_vec, v);
#ifdef __FMA__
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
#else
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, _mm256_mul_ps(v, v));
#endif
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            sum_vec = _mm256_add_ps(sum_vec, v);
#ifdef __FMA__
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
#else
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, _mm256_mul_ps(v, v));
#endif
        }
    }

    float sum = horizontal_sum_m256(sum_vec);
    float sum_sq = horizontal_sum_m256(sum_sq_vec);

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

static inline float compute_mean_square_avx2(const float *data, const float *residual, size_t n) {
    __m256 sum_sq_vec = _mm256_setzero_ps();
    size_t i = 0;

    if (residual != nullptr) {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
            __m256 r = _mm256_loadu_ps(residual + i);
            v = _mm256_add_ps(v, r);
#ifdef __FMA__
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
#else
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, _mm256_mul_ps(v, v));
#endif
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 v = _mm256_loadu_ps(data + i);
#ifdef __FMA__
            sum_sq_vec = _mm256_fmadd_ps(v, v, sum_sq_vec);
#else
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, _mm256_mul_ps(v, v));
#endif
        }
    }

    float sum_sq = horizontal_sum_m256(sum_sq_vec);
    for (; i < n; i++) {
        float v = data[i];
        if (residual != nullptr) {
            v += residual[i];
        }
        sum_sq += v * v;
    }

    return sum_sq / (float)n;
}

// AVX2-optimized normalization
static inline void normalize_avx2(
    const float *x, const float *residual, float *out, size_t n, float mean, float inv_std, const float *weight,
    const float *bias
) {
    __m256 mean_vec = _mm256_set1_ps(mean);
    __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    size_t i = 0;

    if (residual != nullptr) {
        if (weight != nullptr && bias != nullptr) {
            for (; i + 8 <= n; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 r_vec = _mm256_loadu_ps(residual + i);
                __m256 w_vec = _mm256_loadu_ps(weight + i);
                __m256 b_vec = _mm256_loadu_ps(bias + i);

                x_vec = _mm256_add_ps(x_vec, r_vec);
                __m256 normalized = _mm256_sub_ps(x_vec, mean_vec);
                normalized = _mm256_mul_ps(normalized, inv_std_vec);
                normalized = _mm256_fmadd_ps(normalized, w_vec, b_vec);

                _mm256_storeu_ps(out + i, normalized);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = ((v - mean) * inv_std) * weight[i] + bias[i];
            }
        } else if (weight != nullptr) {
            for (; i + 8 <= n; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 r_vec = _mm256_loadu_ps(residual + i);
                __m256 w_vec = _mm256_loadu_ps(weight + i);

                x_vec = _mm256_add_ps(x_vec, r_vec);
                __m256 normalized = _mm256_sub_ps(x_vec, mean_vec);
                normalized = _mm256_mul_ps(normalized, inv_std_vec);
                normalized = _mm256_mul_ps(normalized, w_vec);

                _mm256_storeu_ps(out + i, normalized);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = ((v - mean) * inv_std) * weight[i];
            }
        } else if (bias != nullptr) {
            for (; i + 8 <= n; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 r_vec = _mm256_loadu_ps(residual + i);
                __m256 b_vec = _mm256_loadu_ps(bias + i);

                x_vec = _mm256_add_ps(x_vec, r_vec);
                __m256 normalized = _mm256_sub_ps(x_vec, mean_vec);
                normalized = _mm256_mul_ps(normalized, inv_std_vec);
                normalized = _mm256_add_ps(normalized, b_vec);

                _mm256_storeu_ps(out + i, normalized);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = (v - mean) * inv_std + bias[i];
            }
        } else {
            for (; i + 8 <= n; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 r_vec = _mm256_loadu_ps(residual + i);

                x_vec = _mm256_add_ps(x_vec, r_vec);
                __m256 normalized = _mm256_sub_ps(x_vec, mean_vec);
                normalized = _mm256_mul_ps(normalized, inv_std_vec);

                _mm256_storeu_ps(out + i, normalized);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = (v - mean) * inv_std;
            }
        }
    } else if (weight != nullptr && bias != nullptr) {
        for (; i + 8 <= n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 w_vec = _mm256_loadu_ps(weight + i);
            __m256 b_vec = _mm256_loadu_ps(bias + i);

            __m256 normalized = _mm256_sub_ps(x_vec, mean_vec);
            normalized = _mm256_mul_ps(normalized, inv_std_vec);
            normalized = _mm256_fmadd_ps(normalized, w_vec, b_vec);

            _mm256_storeu_ps(out + i, normalized);
        }
        for (; i < n; i++) {
            out[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
        }
    } else if (weight != nullptr) {
        for (; i + 8 <= n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 w_vec = _mm256_loadu_ps(weight + i);

            __m256 normalized = _mm256_sub_ps(x_vec, mean_vec);
            normalized = _mm256_mul_ps(normalized, inv_std_vec);
            normalized = _mm256_mul_ps(normalized, w_vec);

            _mm256_storeu_ps(out + i, normalized);
        }
        for (; i < n; i++) {
            out[i] = ((x[i] - mean) * inv_std) * weight[i];
        }
    } else if (bias != nullptr) {
        for (; i + 8 <= n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 b_vec = _mm256_loadu_ps(bias + i);

            __m256 normalized = _mm256_sub_ps(x_vec, mean_vec);
            normalized = _mm256_mul_ps(normalized, inv_std_vec);
            normalized = _mm256_add_ps(normalized, b_vec);

            _mm256_storeu_ps(out + i, normalized);
        }
        for (; i < n; i++) {
            out[i] = (x[i] - mean) * inv_std + bias[i];
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);

            __m256 normalized = _mm256_sub_ps(x_vec, mean_vec);
            normalized = _mm256_mul_ps(normalized, inv_std_vec);

            _mm256_storeu_ps(out + i, normalized);
        }
        for (; i < n; i++) {
            out[i] = (x[i] - mean) * inv_std;
        }
    }
}

static inline void normalize_rms_avx2(
    const float *x, const float *residual, float *out, size_t n, float inv_rms, const float *weight, float weight_offset
) {
    __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);
    size_t i = 0;

    if (residual != nullptr) {
        if (weight != nullptr) {
            __m256 offset_vec = _mm256_set1_ps(weight_offset);
            for (; i + 8 <= n; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 r_vec = _mm256_loadu_ps(residual + i);
                __m256 w_vec = _mm256_loadu_ps(weight + i);

                x_vec = _mm256_add_ps(x_vec, r_vec);
                w_vec = _mm256_add_ps(w_vec, offset_vec);
                __m256 out_vec = _mm256_mul_ps(_mm256_mul_ps(x_vec, inv_rms_vec), w_vec);

                _mm256_storeu_ps(out + i, out_vec);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = v * inv_rms * (weight[i] + weight_offset);
            }
        } else {
            for (; i + 8 <= n; i += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + i);
                __m256 r_vec = _mm256_loadu_ps(residual + i);

                x_vec = _mm256_add_ps(x_vec, r_vec);
                __m256 out_vec = _mm256_mul_ps(x_vec, inv_rms_vec);

                _mm256_storeu_ps(out + i, out_vec);
            }
            for (; i < n; i++) {
                float v = x[i] + residual[i];
                out[i] = v * inv_rms;
            }
        }
    } else if (weight != nullptr) {
        __m256 offset_vec = _mm256_set1_ps(weight_offset);
        for (; i + 8 <= n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 w_vec = _mm256_loadu_ps(weight + i);

            w_vec = _mm256_add_ps(w_vec, offset_vec);
            __m256 out_vec = _mm256_mul_ps(_mm256_mul_ps(x_vec, inv_rms_vec), w_vec);

            _mm256_storeu_ps(out + i, out_vec);
        }
        for (; i < n; i++) {
            out[i] = x[i] * inv_rms * (weight[i] + weight_offset);
        }
    } else {
        for (; i + 8 <= n; i += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + i);
            __m256 out_vec = _mm256_mul_ps(x_vec, inv_rms_vec);

            _mm256_storeu_ps(out + i, out_vec);
        }
        for (; i < n; i++) {
            out[i] = x[i] * inv_rms;
        }
    }
}
#endif
#if HAS_AVX2
static void layernorm_row_bf16_avx2(
    const marmot_bfloat16_t *x_row, const marmot_bfloat16_t *residual_row, const marmot_bfloat16_t *weight,
    const marmot_bfloat16_t *bias, size_t norm_size, float eps, marmot_bfloat16_t *out_row
) {
    __m256 sum_vec = _mm256_setzero_ps();
    __m256 sum_sq_vec = _mm256_setzero_ps();
    size_t i = 0;

    if (residual_row != nullptr) {
        for (; i + 8 <= norm_size; i += 8) {
            __m128i x_bits = _mm_loadu_si128((const __m128i *)(x_row + i));
            __m128i r_bits = _mm_loadu_si128((const __m128i *)(residual_row + i));
            __m256i x_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(x_bits), 16);
            __m256i r_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(r_bits), 16);
            __m256 x_f32 = _mm256_castsi256_ps(x_u32);
            __m256 r_f32 = _mm256_castsi256_ps(r_u32);
            x_f32 = _mm256_add_ps(x_f32, r_f32);
            sum_vec = _mm256_add_ps(sum_vec, x_f32);
#ifdef __FMA__
            sum_sq_vec = _mm256_fmadd_ps(x_f32, x_f32, sum_sq_vec);
#else
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, _mm256_mul_ps(x_f32, x_f32));
#endif
        }
    } else {
        for (; i + 8 <= norm_size; i += 8) {
            __m128i x_bits = _mm_loadu_si128((const __m128i *)(x_row + i));
            __m256i x_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(x_bits), 16);
            __m256 x_f32 = _mm256_castsi256_ps(x_u32);
            sum_vec = _mm256_add_ps(sum_vec, x_f32);
#ifdef __FMA__
            sum_sq_vec = _mm256_fmadd_ps(x_f32, x_f32, sum_sq_vec);
#else
            sum_sq_vec = _mm256_add_ps(sum_sq_vec, _mm256_mul_ps(x_f32, x_f32));
#endif
        }
    }

    float sum = horizontal_sum_m256(sum_vec);
    float sum_sq = horizontal_sum_m256(sum_sq_vec);
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
    __m256 inv_std_vec = _mm256_set1_ps(inv_std);
    __m256 mean_vec = _mm256_set1_ps(mean);

    size_t k = 0;
    if (residual_row != nullptr) {
        for (; k + 8 <= norm_size; k += 8) {
            __m128i x_bits = _mm_loadu_si128((const __m128i *)(x_row + k));
            __m128i r_bits = _mm_loadu_si128((const __m128i *)(residual_row + k));
            __m256i x_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(x_bits), 16);
            __m256i r_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(r_bits), 16);
            __m256 x_f32 = _mm256_castsi256_ps(x_u32);
            __m256 r_f32 = _mm256_castsi256_ps(r_u32);
            x_f32 = _mm256_add_ps(x_f32, r_f32);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x_f32, mean_vec), inv_std_vec);

            if (weight != nullptr) {
                __m128i w_bits = _mm_loadu_si128((const __m128i *)(weight + k));
                __m256i w_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w_bits), 16);
                __m256 w_f32 = _mm256_castsi256_ps(w_u32);
                norm = _mm256_mul_ps(norm, w_f32);
            }

            if (bias != nullptr) {
                __m128i b_bits = _mm_loadu_si128((const __m128i *)(bias + k));
                __m256i b_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b_bits), 16);
                __m256 b_f32 = _mm256_castsi256_ps(b_u32);
                norm = _mm256_add_ps(norm, b_f32);
            }

            __m256i norm_bits = _mm256_castps_si256(norm);
            __m256i bias_bits = _mm256_set1_epi32(0x7FFF);
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(norm_bits, 16), _mm256_set1_epi32(1));
            __m256i rounded = _mm256_add_epi32(norm_bits, _mm256_add_epi32(bias_bits, lsb));
            __m256i shifted = _mm256_srli_epi32(rounded, 16);
            __m128i lo = _mm256_castsi256_si128(shifted);
            __m128i hi = _mm256_extracti128_si256(shifted, 1);
            __m128i out_vals = _mm_packus_epi32(lo, hi);
            out_vals = _mm_permute4x64_epi64(out_vals, 0xD8);
            _mm_storeu_si128((__m128i *)(out_row + k), out_vals);
        }
    } else {
        for (; k + 8 <= norm_size; k += 8) {
            __m128i x_bits = _mm_loadu_si128((const __m128i *)(x_row + k));
            __m256i x_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(x_bits), 16);
            __m256 x_f32 = _mm256_castsi256_ps(x_u32);
            __m256 norm = _mm256_mul_ps(_mm256_sub_ps(x_f32, mean_vec), inv_std_vec);

            if (weight != nullptr) {
                __m128i w_bits = _mm_loadu_si128((const __m128i *)(weight + k));
                __m256i w_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w_bits), 16);
                __m256 w_f32 = _mm256_castsi256_ps(w_u32);
                norm = _mm256_mul_ps(norm, w_f32);
            }

            if (bias != nullptr) {
                __m128i b_bits = _mm_loadu_si128((const __m128i *)(bias + k));
                __m256i b_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(b_bits), 16);
                __m256 b_f32 = _mm256_castsi256_ps(b_u32);
                norm = _mm256_add_ps(norm, b_f32);
            }

            __m256i norm_bits = _mm256_castps_si256(norm);
            __m256i bias_bits = _mm256_set1_epi32(0x7FFF);
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(norm_bits, 16), _mm256_set1_epi32(1));
            __m256i rounded = _mm256_add_epi32(norm_bits, _mm256_add_epi32(bias_bits, lsb));
            __m256i shifted = _mm256_srli_epi32(rounded, 16);
            __m128i lo = _mm256_castsi256_si128(shifted);
            __m128i hi = _mm256_extracti128_si256(shifted, 1);
            __m128i out_vals = _mm_packus_epi32(lo, hi);
            out_vals = _mm_permute4x64_epi64(out_vals, 0xD8);
            _mm_storeu_si128((__m128i *)(out_row + k), out_vals);
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
#endif // HAS_AVX2
#if HAS_AVX2
static void rmsnorm_row_bf16_avx2(
    const marmot_bfloat16_t *x_row, const marmot_bfloat16_t *residual_row, const marmot_bfloat16_t *weight,
    size_t norm_size, float eps, float weight_offset, marmot_bfloat16_t *out_row
) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;
    if (residual_row != nullptr) {
        for (; i + 8 <= norm_size; i += 8) {
            __m128i x_bits = _mm_loadu_si128((const __m128i *)(x_row + i));
            __m128i r_bits = _mm_loadu_si128((const __m128i *)(residual_row + i));
            __m256i x_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(x_bits), 16);
            __m256i r_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(r_bits), 16);
            __m256 x_f32 = _mm256_castsi256_ps(x_u32);
            __m256 r_f32 = _mm256_castsi256_ps(r_u32);
            x_f32 = _mm256_add_ps(x_f32, r_f32);
#ifdef __FMA__
            sum_vec = _mm256_fmadd_ps(x_f32, x_f32, sum_vec);
#else
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x_f32, x_f32));
#endif
        }
    } else {
        for (; i + 8 <= norm_size; i += 8) {
            __m128i x_bits = _mm_loadu_si128((const __m128i *)(x_row + i));
            __m256i x_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(x_bits), 16);
            __m256 x_f32 = _mm256_castsi256_ps(x_u32);
#ifdef __FMA__
            sum_vec = _mm256_fmadd_ps(x_f32, x_f32, sum_vec);
#else
            sum_vec = _mm256_add_ps(sum_vec, _mm256_mul_ps(x_f32, x_f32));
#endif
        }
    }
    float sum = horizontal_sum_m256(sum_vec);
    for (; i < norm_size; i++) {
        float value = marmot_bf16_to_f32_ref(x_row[i]);
        if (residual_row != nullptr) {
            value += marmot_bf16_to_f32_ref(residual_row[i]);
        }
        sum += value * value;
    }

    float mean_sq = sum / (float)norm_size;
    float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + eps);
    __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);

    size_t k = 0;
    const __m256 offset_vec = _mm256_set1_ps(weight_offset);
    if (residual_row != nullptr) {
        for (; k + 8 <= norm_size; k += 8) {
            __m128i x_bits = _mm_loadu_si128((const __m128i *)(x_row + k));
            __m128i r_bits = _mm_loadu_si128((const __m128i *)(residual_row + k));
            __m256i x_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(x_bits), 16);
            __m256i r_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(r_bits), 16);
            __m256 x_f32 = _mm256_castsi256_ps(x_u32);
            __m256 r_f32 = _mm256_castsi256_ps(r_u32);
            x_f32 = _mm256_add_ps(x_f32, r_f32);
            __m256 out_vec = _mm256_mul_ps(x_f32, inv_rms_vec);
            if (weight != nullptr) {
                __m128i w_bits = _mm_loadu_si128((const __m128i *)(weight + k));
                __m256i w_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w_bits), 16);
                __m256 w_f32 = _mm256_castsi256_ps(w_u32);
                w_f32 = _mm256_add_ps(w_f32, offset_vec);
                out_vec = _mm256_mul_ps(out_vec, w_f32);
            }
            __m256i out_bits = _mm256_castps_si256(out_vec);
            __m256i bias_bits = _mm256_set1_epi32(0x7FFF);
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(out_bits, 16), _mm256_set1_epi32(1));
            __m256i rounded = _mm256_add_epi32(out_bits, _mm256_add_epi32(bias_bits, lsb));
            __m256i shifted = _mm256_srli_epi32(rounded, 16);
            __m128i lo = _mm256_castsi256_si128(shifted);
            __m128i hi = _mm256_extracti128_si256(shifted, 1);
            __m128i out_vals = _mm_packus_epi32(lo, hi);
            out_vals = _mm_permute4x64_epi64(out_vals, 0xD8);
            _mm_storeu_si128((__m128i *)(out_row + k), out_vals);
        }
    } else {
        for (; k + 8 <= norm_size; k += 8) {
            __m128i x_bits = _mm_loadu_si128((const __m128i *)(x_row + k));
            __m256i x_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(x_bits), 16);
            __m256 x_f32 = _mm256_castsi256_ps(x_u32);
            __m256 out_vec = _mm256_mul_ps(x_f32, inv_rms_vec);
            if (weight != nullptr) {
                __m128i w_bits = _mm_loadu_si128((const __m128i *)(weight + k));
                __m256i w_u32 = _mm256_slli_epi32(_mm256_cvtepu16_epi32(w_bits), 16);
                __m256 w_f32 = _mm256_castsi256_ps(w_u32);
                w_f32 = _mm256_add_ps(w_f32, offset_vec);
                out_vec = _mm256_mul_ps(out_vec, w_f32);
            }
            __m256i out_bits = _mm256_castps_si256(out_vec);
            __m256i bias_bits = _mm256_set1_epi32(0x7FFF);
            __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(out_bits, 16), _mm256_set1_epi32(1));
            __m256i rounded = _mm256_add_epi32(out_bits, _mm256_add_epi32(bias_bits, lsb));
            __m256i shifted = _mm256_srli_epi32(rounded, 16);
            __m128i lo = _mm256_castsi256_si128(shifted);
            __m128i hi = _mm256_extracti128_si256(shifted, 1);
            __m128i out_vals = _mm_packus_epi32(lo, hi);
            out_vals = _mm_permute4x64_epi64(out_vals, 0xD8);
            _mm_storeu_si128((__m128i *)(out_row + k), out_vals);
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
#endif // HAS_AVX2
#if HAS_AVX2
static void layernorm_f32_avx2_range(void *context, size_t start, size_t end) {
    layernorm_f32_context_t *ctx = (layernorm_f32_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const float *x_row = ctx->x + row * ctx->norm_size;
        const float *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        float *out_row = ctx->out + row * ctx->norm_size;
        float mean = 0.0f;
        float variance = 0.0f;
        compute_mean_variance_avx2(x_row, res_row, ctx->norm_size, &mean, &variance);
        float inv_std = cpu_norm_inv_sqrt_f32(variance + ctx->eps);
        normalize_avx2(x_row, res_row, out_row, ctx->norm_size, mean, inv_std, ctx->weight, ctx->bias);
    }
}
#endif
#if HAS_AVX2
static void rmsnorm_f32_avx2_range(void *context, size_t start, size_t end) {
    rmsnorm_f32_context_t *ctx = (rmsnorm_f32_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const float *x_row = ctx->x + row * ctx->norm_size;
        const float *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        float *out_row = ctx->out + row * ctx->norm_size;
        float mean_sq = compute_mean_square_avx2(x_row, res_row, ctx->norm_size);
        float inv_rms = cpu_norm_inv_sqrt_f32(mean_sq + ctx->eps);
        normalize_rms_avx2(x_row, res_row, out_row, ctx->norm_size, inv_rms, ctx->weight, ctx->weight_offset);
    }
}
#endif
#if HAS_AVX2
static marmot_error_t cpu_layernorm_f32_avx2(const void *device_ctx, const cpu_layernorm_params_t *params) {
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
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_f32_avx2_range);
    return MARMOT_SUCCESS;
}
#endif
#if HAS_AVX2
static marmot_error_t cpu_rmsnorm_f32_avx2(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
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
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_f32_avx2_range);
    return MARMOT_SUCCESS;
}
#endif
#if HAS_AVX2
static void layernorm_bf16_avx2_range(void *context, size_t start, size_t end) {
    layernorm_bf16_context_t *ctx = (layernorm_bf16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_bfloat16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_bfloat16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_bfloat16_t *out_row = ctx->out + row * ctx->norm_size;
        layernorm_row_bf16_avx2(x_row, res_row, ctx->weight, ctx->bias, ctx->norm_size, ctx->eps, out_row);
    }
}
#endif
#if HAS_AVX2
static void rmsnorm_bf16_avx2_range(void *context, size_t start, size_t end) {
    rmsnorm_bf16_context_t *ctx = (rmsnorm_bf16_context_t *)context;
    for (size_t row = start; row < end; ++row) {
        const marmot_bfloat16_t *x_row = ctx->x + row * ctx->norm_size;
        const marmot_bfloat16_t *res_row = ctx->residual != nullptr ? ctx->residual + row * ctx->norm_size : nullptr;
        marmot_bfloat16_t *out_row = ctx->out + row * ctx->norm_size;
        rmsnorm_row_bf16_avx2(x_row, res_row, ctx->weight, ctx->norm_size, ctx->eps, ctx->weight_offset, out_row);
    }
}
#endif
#if HAS_AVX2
static marmot_error_t cpu_layernorm_bf16_avx2(const void *device_ctx, const cpu_layernorm_params_t *params) {
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
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, layernorm_bf16_avx2_range);
    return MARMOT_SUCCESS;
}

static marmot_error_t cpu_rmsnorm_bf16_avx2(const void *device_ctx, const cpu_rmsnorm_params_t *params) {
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
    cpu_norm_dispatch_rows(params->outer_size, params->norm_size, &ctx, rmsnorm_bf16_avx2_range);
    return MARMOT_SUCCESS;
}
#endif
#if HAS_AVX2
const cpu_norm_traits_t cpu_norm_f32_avx2_traits = {
    .dtype = MARMOT_DTYPE_FLOAT32,
    .impl_kind = CPU_NORM_IMPL_AVX2,
    .ops = {
        .layernorm = cpu_layernorm_f32_avx2,
        .rmsnorm = cpu_rmsnorm_f32_avx2,
        .impl_name = "f32_avx2",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_f32_avx2_traits)
#endif
#if HAS_AVX2
const cpu_norm_traits_t cpu_norm_bf16_avx2_traits = {
    .dtype = MARMOT_DTYPE_BFLOAT16,
    .impl_kind = CPU_NORM_IMPL_AVX2,
    .ops = {
        .layernorm = cpu_layernorm_bf16_avx2,
        .rmsnorm = cpu_rmsnorm_bf16_avx2,
        .impl_name = "bf16_avx2",
    },
};
CPU_NORM_REGISTER_TRAITS(cpu_norm_bf16_avx2_traits)
#endif

#endif // MARMOT_ENABLE_AVX2
