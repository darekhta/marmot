#ifndef MARMOT_CPU_QUANT_UTILS_H
#define MARMOT_CPU_QUANT_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float scale;
    float inv_scale;
    bool is_zero;
} marmot_quant_signed_scale_t;

/**
 * Finds the sample with the largest absolute magnitude and builds the signed
 * scaling pair used by the GGUF symmetric formats (e.g. Q8_K, Q8_0).
 *
 * When the input range is zero the returned structure sets `is_zero = true`
 * and both scale/iscale to 0 so callers can fast-path to a zero block.
 */
marmot_quant_signed_scale_t marmot_quant_prepare_signed_scale(const float *values, size_t count, float quant_max);

/**
 * Applies a symmetric signed quantization (clamped to [qmin, qmax]) using the
 * provided inverse scale. The helper mirrors the ggml reference kernels.
 */
void marmot_quant_store_symmetric_int8(const float *values, uint32_t count, float inv_scale, int8_t *dst);

/**
 * Computes a positive (unsigned) scale based on the absolute maximum element.
 * Returns 0 when the input range is empty so the caller can decide how to
 * handle the degenerate case (e.g., defaulting to scale=1.0f for legacy paths).
 */
float marmot_quant_compute_positive_scale(const float *values, size_t count, float quant_max);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CPU_QUANT_UTILS_H
