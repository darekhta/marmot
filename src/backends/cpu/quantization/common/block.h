#ifndef CPU_QUANTIZE_BLOCK_H
#define CPU_QUANTIZE_BLOCK_H

#include <stddef.h>
#include <stdint.h>

#include "cpu_backend_internal.h"

typedef struct {
    float min_val;
    float max_val;
} marmot_block_minmax_t;

float cpu_quant_block_max_abs(const float *data, size_t len);
marmot_block_minmax_t cpu_quant_block_minmax(const float *data, size_t len);

void cpu_quantize_q5_0_pack(const float *data, size_t len, float inv_scale, int8_t *out);
void cpu_quantize_q5_1_pack(const float *data, size_t len, float min_val, float inv_scale, uint8_t *out);
void cpu_quantize_q8_0_pack(const float *data, size_t len, float inv_scale, int8_t *out);

#endif
