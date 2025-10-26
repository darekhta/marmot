#include "quant_utils.h"

#include <math.h>

marmot_quant_signed_scale_t marmot_quant_prepare_signed_scale(const float *values, size_t count, float quant_max) {
    marmot_quant_signed_scale_t result = {
        .scale = 0.0f,
        .inv_scale = 0.0f,
        .is_zero = true,
    };

    if (values == nullptr || count == 0 || quant_max <= 0.0f) {
        return result;
    }

    float best = 0.0f;
    float abs_best = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float value = values[i];
        const float abs_value = fabsf(value);
        if (abs_value > abs_best) {
            abs_best = abs_value;
            best = value;
        }
    }

    if (abs_best <= 0.0f) {
        return result;
    }

    const float inv_scale = -quant_max / best;
    result.inv_scale = inv_scale;
    result.scale = inv_scale != 0.0f ? (1.0f / inv_scale) : 0.0f;
    result.is_zero = false;
    return result;
}

void marmot_quant_store_symmetric_int8(const float *values, uint32_t count, float inv_scale, int8_t *dst) {
    if (values == nullptr || dst == nullptr || count == 0 || inv_scale == 0.0f) {
        return;
    }

    for (uint32_t i = 0; i < count; ++i) {
        const float scaled = values[i] * inv_scale;
        int32_t q = (int32_t)lrintf(scaled);
        if (q > 127) {
            q = 127;
        } else if (q < -128) {
            q = -128;
        }
        dst[i] = (int8_t)q;
    }
}

float marmot_quant_compute_positive_scale(const float *values, size_t count, float quant_max) {
    if (values == nullptr || count == 0 || quant_max <= 0.0f) {
        return 0.0f;
    }

    float max_abs = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        const float abs_val = fabsf(values[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }

    if (max_abs <= 0.0f) {
        return 0.0f;
    }

    return max_abs / quant_max;
}
