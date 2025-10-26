#ifndef MARMOT_DEVICE_CAPS_H
#define MARMOT_DEVICE_CAPS_H

#include "marmot/traits_ids.gen.h"
#include "marmot/types.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float peak_flops_tflops_fp32;
    float peak_flops_tflops_fp16;
    float mem_bw_gbps;
    float launch_overhead_us;
    float edge_penalty_alpha;

    uint32_t l1_cache_kb;
    uint32_t l2_cache_kb;
    uint32_t l3_cache_mb;

    uint32_t num_compute_units;
    uint32_t simd_width;

    bool has_fma;
    bool has_fp16_compute;
    bool has_bf16_compute;
    bool has_tensor_cores;
    bool has_simdgroup_mm;

    float h2d_bw_gbps;
    float d2h_bw_gbps;
    float d2d_bw_gbps;
    float d2d_lat_us;
    uint32_t copy_engines;
    uint32_t max_streams;

    // Calibration overrides (optional; 0 means unspecified)
    float calib_dequant_us_q4k;
    float calib_dequant_us_q5k;
    float calib_dequant_us_q6k;
    float calib_dequant_us_q8_0;
    float calib_epilogue_scale;

    union {
        struct {
            bool has_amx;
            uint32_t neural_engine_tops;
            bool has_arm_bf16;
            bool has_arm_dotprod;
            bool has_arm_sve;
            marmot_cpu_microarch_t cpu_microarch;
        } apple;
        struct {
            uint32_t sm_count;
            uint32_t shared_mem_kb;
            uint32_t compute_capability;
        } cuda;
    } backend;

    // CPU topology for hybrid architectures (P-cores + E-cores)
    struct {
        uint32_t total_cores; // All logical cores
        uint32_t p_cores;     // Performance cores (0 if unknown)
        uint32_t e_cores;     // Efficiency cores (0 if homogeneous)
        bool is_hybrid;       // true if heterogeneous (e_cores > 0)
    } topology;
} marmot_device_caps_t;

#ifdef __cplusplus
}
#endif

#endif
