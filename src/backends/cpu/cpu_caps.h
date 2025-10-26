#ifndef MARMOT_CPU_CAPS_H
#define MARMOT_CPU_CAPS_H

#include "marmot/device.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

marmot_profile_id_t marmot_cpu_detect_best_profile(void);

marmot_device_caps_t marmot_cpu_detect_capabilities(void);

// Returns optimal thread count based on CPU topology:
// - Hybrid CPU (P-cores + E-cores): returns p_cores - 1 to avoid E-core spillover
// - Homogeneous CPU: returns all cores
// Respects MARMOT_NUM_THREADS environment variable override.
size_t marmot_cpu_optimal_thread_count(const marmot_device_caps_t *caps);

marmot_cpu_microarch_t marmot_cpu_detect_microarch(void);

bool marmot_cpu_has_arm_bf16(void);

bool marmot_cpu_has_arm_dotprod(void);

bool marmot_cpu_has_arm_i8mm(void);

bool marmot_cpu_has_arm_sve(void);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CPU_CAPS_H
