#ifndef MARMOT_NEON_MATMUL_PARAMS_H
#define MARMOT_NEON_MATMUL_PARAMS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "cpu_caps.h"

// Tunable tiling parameters for NEON SGEMM (Apple Silicon M4 Pro defaults)
// Benchmarked on M4 Pro: block_k=768 gives ~5% improvement over 512 for large matrices
#define MARMOT_NEON_F32_TILE_M 8
#define MARMOT_NEON_F32_TILE_N 8
#define MARMOT_NEON_F32_BLOCK_K_DEFAULT 768
#define MARMOT_NEON_F32_BLOCK_M_DEFAULT 256
#define MARMOT_NEON_F32_BLOCK_N_DEFAULT 256
#define MARMOT_NEON_F32_MIN_DIM 8
#define MARMOT_NEON_F32_PREFETCH_K_AHEAD_DEFAULT 8
#define MARMOT_NEON_F32_DOUBLE_BUFFER_PACK_DEFAULT 1

#define MARMOT_NEON_F32_SMALL_THRESHOLD_M 32
#define MARMOT_NEON_F32_SMALL_THRESHOLD_N 32
#define MARMOT_NEON_F32_SMALL_THRESHOLD_K 64

typedef struct {
    size_t block_m;
    size_t block_n;
    size_t block_k;
    size_t prefetch_k_ahead;
    bool double_buffer_pack;
} marmot_neon_f32_params_t;

typedef struct {
    marmot_cpu_microarch_t cpu;
    size_t block_m;
    size_t block_n;
    size_t block_k;
    size_t prefetch_k_ahead;
} marmot_neon_f32_cpu_preset_t;

static const marmot_neon_f32_cpu_preset_t MARMOT_NEON_F32_CPU_PRESETS[] = {
    {MARMOT_CPU_APPLE_M1, 256, 256, 768, 8},     {MARMOT_CPU_APPLE_M2, 256, 256, 768, 8},
    {MARMOT_CPU_APPLE_M3, 256, 256, 768, 8},     {MARMOT_CPU_APPLE_M4, 256, 256, 768, 8},
    {MARMOT_CPU_CORTEX_A53, 128, 128, 256, 4},   {MARMOT_CPU_CORTEX_A55, 128, 128, 256, 4},
    {MARMOT_CPU_CORTEX_A57, 128, 2048, 256, 4},  {MARMOT_CPU_CORTEX_A72, 128, 2048, 256, 4},
    {MARMOT_CPU_CORTEX_A76, 256, 2048, 384, 6},  {MARMOT_CPU_CORTEX_X1, 256, 2048, 320, 6},
    {MARMOT_CPU_CORTEX_X2, 256, 2048, 384, 6},   {MARMOT_CPU_NEOVERSE_N1, 320, 2048, 256, 6},
    {MARMOT_CPU_NEOVERSE_N2, 320, 2048, 384, 6}, {MARMOT_CPU_NEOVERSE_V1, 256, 2048, 512, 8},
    {MARMOT_CPU_UNKNOWN, 256, 256, 512, 6},
};

static inline const marmot_neon_f32_cpu_preset_t *marmot_neon_f32_lookup_preset(marmot_cpu_microarch_t cpu) {
    for (size_t i = 0; i < sizeof(MARMOT_NEON_F32_CPU_PRESETS) / sizeof(MARMOT_NEON_F32_CPU_PRESETS[0]); ++i) {
        if (MARMOT_NEON_F32_CPU_PRESETS[i].cpu == cpu) {
            return &MARMOT_NEON_F32_CPU_PRESETS[i];
        }
    }
    return &MARMOT_NEON_F32_CPU_PRESETS
        [sizeof(MARMOT_NEON_F32_CPU_PRESETS) / sizeof(MARMOT_NEON_F32_CPU_PRESETS[0]) - 1];
}

static inline bool marmot_neon_f32_use_small_kernel(size_t M, size_t N, size_t K) {
    return M <= MARMOT_NEON_F32_SMALL_THRESHOLD_M && N <= MARMOT_NEON_F32_SMALL_THRESHOLD_N &&
        K <= MARMOT_NEON_F32_SMALL_THRESHOLD_K;
}

static inline size_t marmot_neon_min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static inline size_t marmot_neon_parse_env_size(const char *name, size_t fallback, size_t min, size_t max) {
    const char *env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    char *endptr = nullptr;
    const unsigned long parsed = strtoul(env, &endptr, 10);
    if (endptr == env || parsed < min) {
        return fallback;
    }
    if (parsed > max) {
        return max;
    }
    return (size_t)parsed;
}

static inline bool marmot_neon_parse_env_bool(const char *name, bool fallback) {
    const char *env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return fallback;
    }
    if (env[0] == '0') {
        return false;
    }
    return true;
}

static inline const marmot_neon_f32_params_t *marmot_neon_f32_get_params(void) {
    static marmot_neon_f32_params_t params = {0};
    static bool initialized = false;
    if (initialized) {
        return &params;
    }
    marmot_cpu_microarch_t cpu = marmot_cpu_detect_microarch();
    const marmot_neon_f32_cpu_preset_t *preset = marmot_neon_f32_lookup_preset(cpu);
    params.block_m =
        marmot_neon_parse_env_size("MARMOT_NEON_F32_BLOCK_M", preset->block_m, MARMOT_NEON_F32_TILE_M, 1024);
    params.block_n =
        marmot_neon_parse_env_size("MARMOT_NEON_F32_BLOCK_N", preset->block_n, MARMOT_NEON_F32_TILE_N, 4096);
    params.block_k =
        marmot_neon_parse_env_size("MARMOT_NEON_F32_BLOCK_K", preset->block_k, MARMOT_NEON_F32_TILE_N, 2048);
    params.prefetch_k_ahead =
        marmot_neon_parse_env_size("MARMOT_NEON_F32_PREFETCH_K_AHEAD", preset->prefetch_k_ahead, 1, 32);
    params.double_buffer_pack =
        marmot_neon_parse_env_bool("MARMOT_NEON_F32_DOUBLE_BUFFER_PACK", MARMOT_NEON_F32_DOUBLE_BUFFER_PACK_DEFAULT);
    initialized = true;
    return &params;
}

#endif
