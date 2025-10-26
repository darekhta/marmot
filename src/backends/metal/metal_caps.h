#pragma once

#include "marmot/device_caps.h"

#include <Metal/Metal.h>

#ifdef __cplusplus
extern "C" {
#endif

marmot_device_caps_t marmot_metal_detect_caps(id<MTLDevice> device);
marmot_device_caps_t marmot_metal_detect_default_caps(void);
void marmot_metal_log_caps(id<MTLDevice> device, const marmot_device_caps_t *caps);

#ifdef __cplusplus
}
#endif
