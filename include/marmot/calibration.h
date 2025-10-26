#ifndef MARMOT_CALIBRATION_H
#define MARMOT_CALIBRATION_H

#include "marmot/device_caps.h"
#include "marmot/types.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float effective_tflops_fp32;
    float effective_tflops_fp16;
    float effective_gbps;
    float launch_overhead_us;
    float edge_penalty_alpha;
    float dequant_us_q4k;
    float dequant_us_q5k;
    float dequant_us_q6k;
    float dequant_us_q8_0;
    float epilogue_scale;
} marmot_calibration_t;

void marmot_calibration_defaults(marmot_calibration_t *calib);
MARMOT_NODISCARD bool marmot_calibration_load(const char *device_key, marmot_calibration_t *calib);
void marmot_calibration_apply(const marmot_calibration_t *calib, marmot_device_caps_t *caps);
MARMOT_NODISCARD bool
marmot_calibration_make_key(const char *prefix, const char *device_name, uint32_t cores, char *buf, size_t buf_sz);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CALIBRATION_H
