#include "metal_caps.h"

#include "marmot/calibration.h"

#import <Metal/Metal.h>

#include <stdio.h>

#include <dispatch/dispatch.h>
#include <string.h>

static constexpr float DEFAULT_LAUNCH_OVERHEAD_US = 7.0f;
static constexpr float DEFAULT_EDGE_PENALTY_ALPHA = 0.5f;

typedef struct {
    double peak_fp32_tflops;
    double peak_fp16_tflops;
    double mem_bw_gbps;
} gpu_family_specs_t;

static int metal_detect_family(id<MTLDevice> device) {
#if defined(MTLGPUFamilyApple9)
    if ([device supportsFamily:MTLGPUFamilyApple9]) {
        return 9;
    }
#endif
    if ([device supportsFamily:MTLGPUFamilyApple8]) {
        return 8;
    }
    if ([device supportsFamily:MTLGPUFamilyApple7]) {
        return 7;
    }
    if ([device supportsFamily:MTLGPUFamilyApple6]) {
        return 6;
    }
    if ([device supportsFamily:MTLGPUFamilyApple5]) {
        return 5;
    }
    if ([device supportsFamily:MTLGPUFamilyApple4]) {
        return 4;
    }
    return 0;
}

// Base chip specs (non-Pro/Max/Ultra variants)
static gpu_family_specs_t specs_for_family(int family) {
    switch (family) {
    case 9: // M4
        return {.peak_fp32_tflops = 3.5, .peak_fp16_tflops = 7.0, .mem_bw_gbps = 120.0};
    case 8: // M3
        return {.peak_fp32_tflops = 3.0, .peak_fp16_tflops = 6.0, .mem_bw_gbps = 100.0};
    case 7: // M2 / M1 Pro/Max (different arch but same family)
        return {.peak_fp32_tflops = 3.5, .peak_fp16_tflops = 7.0, .mem_bw_gbps = 100.0};
    case 6: // M1
        return {.peak_fp32_tflops = 2.6, .peak_fp16_tflops = 5.2, .mem_bw_gbps = 68.0};
    case 5: // A14
        return {.peak_fp32_tflops = 1.5, .peak_fp16_tflops = 3.0, .mem_bw_gbps = 60.0};
    case 4: // A13
        return {.peak_fp32_tflops = 1.0, .peak_fp16_tflops = 2.0, .mem_bw_gbps = 50.0};
    default:
        return {.peak_fp32_tflops = 0.8, .peak_fp16_tflops = 1.6, .mem_bw_gbps = 40.0};
    }
}

typedef enum { GPU_VARIANT_BASE, GPU_VARIANT_PRO, GPU_VARIANT_MAX, GPU_VARIANT_ULTRA } gpu_variant_t;

static gpu_variant_t detect_gpu_variant(id<MTLDevice> device) {
    NSString *name = device.name.lowercaseString;
    if ([name containsString:@"ultra"]) {
        return GPU_VARIANT_ULTRA;
    }
    if ([name containsString:@"max"]) {
        return GPU_VARIANT_MAX;
    }
    if ([name containsString:@"pro"]) {
        return GPU_VARIANT_PRO;
    }
    return GPU_VARIANT_BASE;
}

// Variant scaling factors vary by generation
static gpu_family_specs_t apply_variant_scaling(gpu_family_specs_t base, int family, gpu_variant_t variant) {
    if (variant == GPU_VARIANT_BASE) {
        return base;
    }

    // Scaling factors based on actual Apple Silicon specs
    // Compute scales with GPU core count, memory BW varies by generation
    double compute_scale = 1.0;
    double mem_scale = 1.0;

    switch (variant) {
    case GPU_VARIANT_PRO:
        if (family >= 9) {
            // M4 Pro: ~2x GPU cores, ~2.3x memory BW vs base M4
            compute_scale = 2.0;
            mem_scale = 2.3;
        } else if (family == 8) {
            // M3 Pro: ~1.6x GPU cores, ~1.5x memory BW vs base M3
            compute_scale = 1.6;
            mem_scale = 1.5;
        } else {
            // M2 Pro, M1 Pro: ~1.6x compute, ~2x memory BW
            compute_scale = 1.6;
            mem_scale = 2.0;
        }
        break;
    case GPU_VARIANT_MAX:
        if (family >= 9) {
            // M4 Max: ~4x GPU cores, ~4.5x memory BW vs base M4
            compute_scale = 4.0;
            mem_scale = 4.5;
        } else if (family == 8) {
            // M3 Max: ~3.3x GPU cores, ~4x memory BW vs base M3
            compute_scale = 3.3;
            mem_scale = 4.0;
        } else {
            // M2 Max, M1 Max: ~3.2x compute, ~4x memory BW
            compute_scale = 3.2;
            mem_scale = 4.0;
        }
        break;
    case GPU_VARIANT_ULTRA:
        if (family >= 8) {
            // M3 Ultra, M4 Ultra (if exists): 2x Max
            compute_scale = (family >= 9) ? 8.0 : 6.6;
            mem_scale = (family >= 9) ? 9.0 : 8.0;
        } else {
            // M2 Ultra, M1 Ultra: 2x Max
            compute_scale = 6.4;
            mem_scale = 8.0;
        }
        break;
    default:
        break;
    }

    base.peak_fp32_tflops *= compute_scale;
    base.peak_fp16_tflops *= compute_scale;
    base.mem_bw_gbps *= mem_scale;
    return base;
}

static gpu_family_specs_t specs_for_device(id<MTLDevice> device) {
    const int family = metal_detect_family(device);
    gpu_family_specs_t base = specs_for_family(family);
    const gpu_variant_t variant = detect_gpu_variant(device);
    return apply_variant_scaling(base, family, variant);
}

static float env_float(const char *name, float fallback) {
    const char *val = getenv(name);
    if (val == nullptr || val[0] == '\0') {
        return fallback;
    }
    char *end = nullptr;
    float parsed = strtof(val, &end);
    return (end != val) ? parsed : fallback;
}

static bool env_bool(const char *name, bool fallback) {
    const char *val = getenv(name);
    if (val == nullptr || val[0] == '\0') {
        return fallback;
    }
    if (val[0] == '0') {
        return false;
    }
    if (val[0] == '1') {
        return true;
    }
    return fallback;
}

static bool metal_should_log_caps(void) {
    static dispatch_once_t once;
    static bool value = false;
    dispatch_once(&once, ^{
      const char *env = getenv("MARMOT_METAL_LOG_CAPS");
      value = (env != nullptr && env[0] != '\0' && env[0] != '0');
    });
    return value;
}

void marmot_metal_log_caps(id<MTLDevice> device, const marmot_device_caps_t *caps) {
    if (device == nil || caps == nullptr) {
        return;
    }
    if (!metal_should_log_caps()) {
        return;
    }
    const char *name = [[device name] UTF8String];
    const int family = metal_detect_family(device);
    fprintf(stderr, "[marmot] Metal Device: %s\n", name != nullptr ? name : "unknown");
    if (family > 0) {
        fprintf(stderr, "[marmot] Metal GPU Family: Apple%d\n", family);
    } else {
        fprintf(stderr, "[marmot] Metal GPU Family: unknown\n");
    }
    fprintf(stderr, "[marmot] simdgroup_mm: %s\n", caps->has_simdgroup_mm ? "enabled" : "disabled");
    fprintf(stderr, "[marmot] Peak FP32: %.1f TFLOPS\n", caps->peak_flops_tflops_fp32);
    fprintf(stderr, "[marmot] Peak FP16: %.1f TFLOPS\n", caps->peak_flops_tflops_fp16);
    fprintf(stderr, "[marmot] Memory BW: %.0f GB/s\n", caps->mem_bw_gbps);
}

static void apply_env_overrides(marmot_device_caps_t *caps) {
    caps->peak_flops_tflops_fp32 = env_float("MARMOT_METAL_PEAK_TFLOPS_FP32", caps->peak_flops_tflops_fp32);
    caps->peak_flops_tflops_fp16 = env_float("MARMOT_METAL_PEAK_TFLOPS_FP16", caps->peak_flops_tflops_fp16);
    caps->mem_bw_gbps = env_float("MARMOT_METAL_MEM_BW_GBPS", caps->mem_bw_gbps);
    caps->launch_overhead_us = env_float("MARMOT_METAL_LAUNCH_US", caps->launch_overhead_us);
    caps->edge_penalty_alpha = env_float("MARMOT_METAL_EDGE_ALPHA", caps->edge_penalty_alpha);
    caps->has_simdgroup_mm = env_bool("MARMOT_METAL_SIMDGROUP_MM", caps->has_simdgroup_mm);
}

static void apply_calibration(marmot_device_caps_t *caps, id<MTLDevice> device) {
    char key[256] = {0};
    const char *name = [[device name] UTF8String];
    if (!marmot_calibration_make_key("metal", name, 0, key, sizeof(key))) {
        return;
    }
    marmot_calibration_t calib;
    if (marmot_calibration_load(key, &calib)) {
        marmot_calibration_apply(&calib, caps);
    }
}

marmot_device_caps_t marmot_metal_detect_caps(id<MTLDevice> device) {
    if (device == nil) {
        marmot_device_caps_t empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }

    gpu_family_specs_t specs = specs_for_device(device);

    bool simdgroup_mm = false;
#if defined(MTLGPUFamilyApple9)
    simdgroup_mm = simdgroup_mm || [device supportsFamily:MTLGPUFamilyApple9];
#endif
    simdgroup_mm =
        simdgroup_mm || [device supportsFamily:MTLGPUFamilyApple8] || [device supportsFamily:MTLGPUFamilyApple7];

    marmot_device_caps_t caps = {};
    caps.peak_flops_tflops_fp32 = (float)specs.peak_fp32_tflops;
    caps.peak_flops_tflops_fp16 = (float)specs.peak_fp16_tflops;
    caps.mem_bw_gbps = (float)specs.mem_bw_gbps;
    caps.launch_overhead_us = DEFAULT_LAUNCH_OVERHEAD_US;
    caps.edge_penalty_alpha = DEFAULT_EDGE_PENALTY_ALPHA;
    caps.num_compute_units = 1;
    caps.simd_width = 0;
    caps.has_fma = true;
    caps.has_fp16_compute = true;
    caps.has_bf16_compute = false;
    caps.has_tensor_cores = false;
    caps.has_simdgroup_mm = simdgroup_mm;

    apply_calibration(&caps, device);
    apply_env_overrides(&caps);
    marmot_metal_log_caps(device, &caps);
    return caps;
}

marmot_device_caps_t marmot_metal_detect_default_caps(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    marmot_device_caps_t caps = marmot_metal_detect_caps(device);
    [device release];
    return caps;
}
