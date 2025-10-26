#ifndef KQUANT_COMMON_H
#define KQUANT_COMMON_H

#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/quant_block.h"
#include "marmot/quant_traits.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#include "cpu_backend_internal.h"

#define QK_K 256
#define GROUP_MAX_EPS 1e-15f

static inline uint32_t qk_clamp_elems(uint32_t count) {
    return count > QK_K ? QK_K : count;
}

static inline void qk_copy_and_pad(const float *src, uint32_t elems, float *dst) {
    memcpy(dst, src, elems * sizeof(float));
    if (elems < QK_K) {
        memset(dst + elems, 0, (QK_K - elems) * sizeof(float));
    }
}

static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static inline float make_qkx2_quants(
    int n, int nmax, const float *x, const float *weights, uint8_t *L, float *the_min, uint8_t *Laux, float rmin,
    float rdelta, int nstep, bool use_mad
) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];

    for (int i = 1; i < n; ++i) {
        if (x[i] < min)
            min = x[i];
        if (x[i] > max)
            max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }

    if (min > 0)
        min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i)
            L[i] = 0;
        *the_min = -min;
        return 0.0f;
    }

    float iscale = nmax / (max - min);
    float scale = 1.0f / iscale;
    float best_error = 0;

    for (int i = 0; i < n; ++i) {
        int l = (int)nearbyintf(iscale * (x[i] - min));
        L[i] = (uint8_t)(l > nmax ? nmax : l < 0 ? 0 : l);
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }

    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }

    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;

        for (int i = 0; i < n; ++i) {
            int l = (int)nearbyintf(iscale * (x[i] - min));
            l = l > nmax ? nmax : l < 0 ? 0 : l;
            Laux[i] = (uint8_t)l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }

        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;

            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }

            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }

            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }

    *the_min = -min;
    return scale;
}

static inline int clamp_qx_value(int value, int nmax) {
    if (value > nmax - 1) {
        return nmax - 1;
    }
    if (value < -nmax) {
        return -nmax;
    }
    return value;
}

static inline int marmot_nearest_int(float value) {
    assert(fabsf(value) <= 4194303.0f);
    float val = value + 12582912.0f;
    int i;
    memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static inline float make_qx_quants(int n, int nmax, const float *x, int8_t *L, int rmse_type, const float *weights) {
    float max = 0.0f;
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < GROUP_MAX_EPS) {
        memset(L, 0, (size_t)n * sizeof(int8_t));
        return 0.0f;
    }

    float iscale = -((float)nmax) / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = marmot_nearest_int(iscale * x[i]);
            l = clamp_qx_value(l, nmax);
            L[i] = (int8_t)(l + nmax);
        }
        return 1.0f / iscale;
    }

    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }

    float sumlx = 0.0f;
    float suml2 = 0.0f;
#ifdef __APPLE__
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = marmot_nearest_int(iscale * x[i]);
        l = clamp_qx_value(l, nmax);
        L[i] = (int8_t)(l + nmax);
        float w = weights      ? weights[i]
            : (rmse_type == 1) ? x[i] * x[i]
            : (rmse_type == 2) ? 1.0f
            : (rmse_type == 3) ? fabsf(x[i])
                               : sqrtf(fabsf(x[i]));
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    float scale = (suml2 > 0.0f) ? (sumlx / suml2) : 0.0f;
    if (return_early) {
        return (suml2 > 0.0f) ? 0.5f * (scale + 1.0f / iscale) : 1.0f / iscale;
    }

    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        float trial_iscale = -(nmax + 0.1f * (float)is) / max;
        float trial_sumlx = 0.0f;
        float trial_suml2 = 0.0f;
        for (int i = 0; i < n; ++i) {
            int l = marmot_nearest_int(trial_iscale * x[i]);
            l = clamp_qx_value(l, nmax);
            float w = weights      ? weights[i]
                : (rmse_type == 1) ? x[i] * x[i]
                : (rmse_type == 2) ? 1.0f
                : (rmse_type == 3) ? fabsf(x[i])
                                   : sqrtf(fabsf(x[i]));
            trial_sumlx += w * x[i] * l;
            trial_suml2 += w * l * l;
        }
        if (trial_suml2 > 0.0f && trial_sumlx * trial_sumlx > best * trial_suml2) {
            for (int i = 0; i < n; ++i) {
                int l = marmot_nearest_int(trial_iscale * x[i]);
                l = clamp_qx_value(l, nmax);
                L[i] = (int8_t)(l + nmax);
            }
            scale = trial_sumlx / trial_suml2;
            best = scale * trial_sumlx;
        }
    }

    return scale;
}

static inline float make_q3_quants(int n, int nmax, const float *x, int8_t *L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }

    if (amax == 0) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.0f;
    }

    float iscale = -nmax / max;
    if (!do_rmse) {
        for (int i = 0; i < n; ++i) {
            int l = (int)nearbyintf(iscale * x[i]);
            l = l > (nmax - 1) ? (nmax - 1) : l < -nmax ? -nmax : l;
            L[i] = (int8_t)(l + nmax);
        }
        return 1.0f / iscale;
    }

    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = (int)nearbyintf(iscale * x[i]);
        l = l > (nmax - 1) ? (nmax - 1) : l < -nmax ? -nmax : l;
        L[i] = (int8_t)l;
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }

    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = x[i] * x[i];
            float slx = sumlx - w * x[i] * L[i];
            if (slx > 0) {
                float sl2 = suml2 - w * L[i] * L[i];
                int new_l = (int)nearbyintf(x[i] * sl2 / slx);
                new_l = new_l > (nmax - 1) ? (nmax - 1) : new_l < -nmax ? -nmax : new_l;
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = (int8_t)new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (n_changed == 0) {
            break;
        }
    }

    for (int i = 0; i < n; ++i) {
        L[i] += nmax;
    }

    return suml2 > 0 ? sumlx / suml2 : 0.0f;
}

static inline void quantize_row_q6_k_ref_single(const float *x, marmot_q6_k_block_t *y) {
    int8_t L[QK_K];
    float scales[QK_K / 16];

    float max_scale = 0.0f;
    float max_abs_scale = 0.0f;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
        float scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, nullptr);
        scales[ib] = scale;
        float abs_scale = fabsf(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if (max_abs_scale < GROUP_MAX_EPS) {
        memset(y, 0, sizeof(*y));
        y->d = marmot_native_to_float16((_Float16)0.0f);
        return;
    }

    float iscale = -128.0f / max_scale;
    y->d = marmot_native_to_float16((_Float16)(1.0f / iscale));
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        int sc = marmot_nearest_int(iscale * scales[ib]);
        sc = sc > 127 ? 127 : (sc < -128 ? -128 : sc);
        y->scales[ib] = (int8_t)sc;
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        float d = (float)marmot_float16_to_native(y->d) * y->scales[j];
        if (d == 0.0f) {
            continue;
        }
        for (int ii = 0; ii < 16; ++ii) {
            int l = marmot_nearest_int(x[16 * j + ii] / d);
            l = clamp_qx_value(l, 32);
            L[16 * j + ii] = (int8_t)(l + 32);
        }
    }

    uint8_t *ql = y->ql;
    uint8_t *qh = y->qh;
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = L[j + l + 0] & 0xF;
            const uint8_t q2 = L[j + l + 32] & 0xF;
            const uint8_t q3 = L[j + l + 64] & 0xF;
            const uint8_t q4 = L[j + l + 96] & 0xF;
            ql[l + 0] = q1 | (q3 << 4);
            ql[l + 32] = q2 | (q4 << 4);
            qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) |
                ((L[j + l + 96] >> 4) << 6);
        }
        ql += 64;
        qh += 32;
    }
}

static inline marmot_error_t
require_k_traits(marmot_quant_kind_t kind, const marmot_quant_traits_t **out_traits, const char *error_message) {
    const marmot_quant_traits_t *traits = marmot_get_quant_traits(kind);
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, error_message);
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    *out_traits = traits;
    return MARMOT_SUCCESS;
}

#endif // KQUANT_COMMON_H
