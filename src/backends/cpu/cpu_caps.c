#include "cpu_caps.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include <string.h>
#include <unistd.h>

#if defined(__APPLE__)
#include <stdio.h>

#include <sys/sysctl.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

#if defined(__linux__)
#include <stdio.h>
#include <stdlib.h>
#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 (1 << 14)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#endif
#endif

#include "marmot/calibration.h"

// ============================================================================
// Constants
// ============================================================================

static constexpr double DEFAULT_FREQ_GHZ = 2.5;
static constexpr double DEFAULT_MEM_BW_BASE_GBPS = 15.0;
static constexpr double DEFAULT_MEM_BW_PER_CORE_GBPS = 5.0;
static constexpr float DEFAULT_LAUNCH_OVERHEAD_US = 0.05f;
static constexpr float DEFAULT_EDGE_PENALTY_ALPHA = 0.5f;

// ============================================================================
// Apple Silicon Frequency Table
// ============================================================================
// Sources:
//   - https://en.wikipedia.org/wiki/Apple_M1
//   - https://nanoreview.net/en/cpu/apple-m4-max-16-core
//   - https://www.notebookcheck.net/Apple-M4-9-cores-Processor-Benchmarks-and-Specs.836006.0.html
//   - https://eclecticlight.co/2025/01/20/what-are-cpu-core-frequencies-in-apple-silicon-macs/
// ============================================================================

typedef struct {
    const char *chip_prefix;  // Prefix to match in brand string (e.g., "Apple M4")
    const char *chip_variant; // Optional variant (e.g., "Pro", "Max", "Ultra")
    double p_core_freq_ghz;   // Performance core frequency
    double e_core_freq_ghz;   // Efficiency core frequency
} apple_silicon_spec_t;

// Order matters: more specific matches (with variant) must come before generic ones
static const apple_silicon_spec_t APPLE_SILICON_SPECS[] = {
    // M4 Series (2024) - 3nm second-generation
    {"Apple M4", "Ultra", 4.51, 2.59},
    {"Apple M4", "Max", 4.51, 2.59},
    {"Apple M4", "Pro", 4.51, 2.59},
    {"Apple M4", nullptr, 4.40, 2.85}, // Base M4 (slightly lower P-core)

    // M3 Series (2023-2024) - 3nm
    {"Apple M3", "Ultra", 4.05, 2.75},
    {"Apple M3", "Max", 4.05, 2.75},
    {"Apple M3", "Pro", 4.05, 2.75},
    {"Apple M3", nullptr, 4.05, 2.75},

    // M2 Series (2022-2023) - 5nm enhanced
    {"Apple M2", "Ultra", 3.49, 2.42},
    {"Apple M2", "Max", 3.49, 2.42},
    {"Apple M2", "Pro", 3.49, 2.42},
    {"Apple M2", nullptr, 3.49, 2.42},

    // M1 Series (2020-2022) - 5nm
    {"Apple M1", "Ultra", 3.23, 2.06},
    {"Apple M1", "Max", 3.23, 2.06},
    {"Apple M1", "Pro", 3.23, 2.06},
    {"Apple M1", nullptr, 3.23, 2.06},

    // Fallback for future/unknown Apple Silicon
    {"Apple", nullptr, 4.40, 2.75},
};

static constexpr size_t APPLE_SILICON_SPECS_COUNT = sizeof(APPLE_SILICON_SPECS) / sizeof(APPLE_SILICON_SPECS[0]);

// ============================================================================
// x86 CPUID Detection
// ============================================================================

[[maybe_unused]] static bool cpu_has_avx2(void) {
#if MARMOT_ENABLE_AVX2 && (defined(__x86_64__) || defined(_M_X64))
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;
    }
#endif
    return false;
}

[[maybe_unused]] static bool cpu_has_avx512(void) {
#if MARMOT_ENABLE_AVX512 && (defined(__x86_64__) || defined(_M_X64))
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 16)) != 0;
    }
#endif
    return false;
}

static uint32_t cpu_simd_width_bits(void) {
#if defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_avx512()) {
        return 512;
    }
    if (cpu_has_avx2()) {
        return 256;
    }
    return 128;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return 128;
#else
    return 128;
#endif
}

// ============================================================================
// Capability Computation
// ============================================================================

static marmot_device_caps_t
caps_from_hw(uint32_t cores, double freq_ghz, uint32_t simd_width_bits, bool is_apple_silicon) {
    const double lanes = (double)simd_width_bits / 32.0;
    // Apple Silicon has 2 FMA units per P-core for NEON
    // x86 AVX2/AVX512 also typically have 2 FMA units
    const double fma_factor = (is_apple_silicon || simd_width_bits >= 256) ? 2.0 : 1.0;
    const double tflops = (double)cores * lanes * fma_factor * freq_ghz / 1e3;
    const double mem_bw = DEFAULT_MEM_BW_BASE_GBPS + DEFAULT_MEM_BW_PER_CORE_GBPS * (double)cores;

    return (marmot_device_caps_t){
        .peak_flops_tflops_fp32 = (float)tflops,
        .peak_flops_tflops_fp16 = (float)tflops,
        .mem_bw_gbps = (float)mem_bw,
        .num_compute_units = cores,
        .simd_width = simd_width_bits,
        .has_fma = is_apple_silicon || simd_width_bits >= 256,
        .has_fp16_compute = true,
        .has_bf16_compute = true,
        .launch_overhead_us = DEFAULT_LAUNCH_OVERHEAD_US,
        .edge_penalty_alpha = DEFAULT_EDGE_PENALTY_ALPHA,
    };
}

static void apply_calibration(marmot_device_caps_t *caps, const char *brand, uint32_t cores) {
    char key[256] = {0};
    if (!marmot_calibration_make_key("cpu", brand, cores, key, sizeof(key))) {
        return;
    }
    marmot_calibration_t calib;
    if (marmot_calibration_load(key, &calib)) {
        marmot_calibration_apply(&calib, caps);
    }
}

// ============================================================================
// Apple Platform Detection
// ============================================================================

#if defined(__APPLE__)

static bool sysctl_u32(const char *name, uint32_t *out) {
    size_t sz = sizeof(*out);
    return sysctlbyname(name, out, &sz, nullptr, 0) == 0;
}

static bool sysctl_u64(const char *name, uint64_t *out) {
    size_t sz = sizeof(*out);
    return sysctlbyname(name, out, &sz, nullptr, 0) == 0;
}

static bool sysctl_str(const char *name, char *out, size_t out_sz) {
    size_t sz = out_sz;
    return sysctlbyname(name, out, &sz, nullptr, 0) == 0;
}

static bool is_apple_silicon_cpu(const char *brand) {
    return strncmp(brand, "Apple", 5) == 0;
}

static const apple_silicon_spec_t *lookup_apple_silicon_spec(const char *brand) {
    if (brand == nullptr || !is_apple_silicon_cpu(brand)) {
        return nullptr;
    }

    for (size_t i = 0; i < APPLE_SILICON_SPECS_COUNT; ++i) {
        const apple_silicon_spec_t *spec = &APPLE_SILICON_SPECS[i];

        // Check if brand starts with the chip prefix
        size_t prefix_len = strlen(spec->chip_prefix);
        if (strncmp(brand, spec->chip_prefix, prefix_len) != 0) {
            continue;
        }

        // If variant is specified, check for it after the prefix
        if (spec->chip_variant != nullptr) {
            const char *after_prefix = brand + prefix_len;
            // Skip whitespace
            while (*after_prefix == ' ') {
                after_prefix++;
            }
            if (strstr(after_prefix, spec->chip_variant) != nullptr) {
                return spec;
            }
        } else {
            // No variant required - this is the base model match
            // Make sure it's not actually a Pro/Max/Ultra variant
            const char *after_prefix = brand + prefix_len;
            if (strstr(after_prefix, "Pro") == nullptr && strstr(after_prefix, "Max") == nullptr &&
                strstr(after_prefix, "Ultra") == nullptr) {
                return spec;
            }
        }
    }

    // Fallback to last entry (generic Apple Silicon)
    return &APPLE_SILICON_SPECS[APPLE_SILICON_SPECS_COUNT - 1];
}

static marmot_cpu_microarch_t detect_apple_silicon_microarch(const char *brand) {
    if (brand == nullptr || !is_apple_silicon_cpu(brand)) {
        return MARMOT_CPU_UNKNOWN;
    }
    if (strstr(brand, "Apple M4") != nullptr) {
        return MARMOT_CPU_APPLE_M4;
    }
    if (strstr(brand, "Apple M3") != nullptr) {
        return MARMOT_CPU_APPLE_M3;
    }
    if (strstr(brand, "Apple M2") != nullptr) {
        return MARMOT_CPU_APPLE_M2;
    }
    if (strstr(brand, "Apple M1") != nullptr) {
        return MARMOT_CPU_APPLE_M1;
    }
    return MARMOT_CPU_UNKNOWN;
}

static bool detect_arm_feature_macos(const char *feature) {
    int32_t val = 0;
    size_t sz = sizeof(val);
    char name[64];
    snprintf(name, sizeof(name), "hw.optional.arm.%s", feature);
    return sysctlbyname(name, &val, &sz, nullptr, 0) == 0 && val != 0;
}

static marmot_device_caps_t detect_apple(void) {
    uint32_t cores = 0;
    uint32_t p_cores = 0, e_cores = 0;
    uint64_t freq_hz = 0;
    uint32_t l1 = 0, l2 = 0, l3 = 0;
    char brand[128] = "apple-cpu";

    // Query system information
    sysctl_u32("hw.logicalcpu", &cores);
    sysctl_u32("hw.perflevel0.logicalcpu", &p_cores); // P-cores (performance)
    sysctl_u32("hw.perflevel1.logicalcpu", &e_cores); // E-cores (efficiency)
    sysctl_u64("hw.cpufrequency", &freq_hz);
    sysctl_u32("hw.l1dcachesize", &l1);
    sysctl_u32("hw.l2cachesize", &l2);
    sysctl_u32("hw.l3cachesize", &l3);
    sysctl_str("machdep.cpu.brand_string", brand, sizeof(brand));

    if (cores == 0) {
        cores = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    }

    const bool apple_silicon = is_apple_silicon_cpu(brand);
    double freq_ghz;
    double effective_cores;

    if (freq_hz > 0) {
        // Intel Mac - use reported frequency and all cores
        freq_ghz = (double)freq_hz / 1e9;
        effective_cores = (double)cores;
    } else if (apple_silicon) {
        // Apple Silicon - look up chip specifications
        const apple_silicon_spec_t *spec = lookup_apple_silicon_spec(brand);

        if (spec != nullptr && p_cores > 0) {
            // Calculate weighted effective cores: P-cores + E-cores scaled by frequency ratio
            // This gives accurate theoretical peak TFLOPS
            double e_core_weight = spec->e_core_freq_ghz / spec->p_core_freq_ghz;
            effective_cores = (double)p_cores + (double)e_cores * e_core_weight;
            freq_ghz = spec->p_core_freq_ghz;
        } else if (spec != nullptr) {
            // Perflevel sysctls not available (older macOS), use all cores at P-core freq
            freq_ghz = spec->p_core_freq_ghz;
            effective_cores = (double)cores;
        } else {
            // Unknown Apple Silicon - use conservative defaults
            freq_ghz = DEFAULT_FREQ_GHZ;
            effective_cores = (double)cores;
        }
    } else {
        // Unknown platform
        freq_ghz = DEFAULT_FREQ_GHZ;
        effective_cores = (double)cores;
    }

    marmot_device_caps_t caps = caps_from_hw((uint32_t)effective_cores, freq_ghz, cpu_simd_width_bits(), apple_silicon);
    caps.num_compute_units = cores; // Report actual core count, not effective
    caps.l1_cache_kb = l1 / 1024U;
    caps.l2_cache_kb = l2 / 1024U;
    caps.l3_cache_mb = l3 / (1024U * 1024U);

    if (apple_silicon) {
        caps.backend.apple.cpu_microarch = detect_apple_silicon_microarch(brand);
        caps.backend.apple.has_arm_bf16 = detect_arm_feature_macos("FEAT_BF16");
        caps.backend.apple.has_arm_dotprod = detect_arm_feature_macos("FEAT_DotProd");
        caps.backend.apple.has_arm_sve = detect_arm_feature_macos("FEAT_SVE");
        caps.has_bf16_compute = caps.backend.apple.has_arm_bf16;
    }

    // Populate topology for hybrid CPU detection
    caps.topology.total_cores = cores;
    caps.topology.p_cores = p_cores > 0 ? p_cores : cores;
    caps.topology.e_cores = e_cores;
    caps.topology.is_hybrid = (e_cores > 0);

    apply_calibration(&caps, brand, cores);
    return caps;
}

#endif // __APPLE__

// ============================================================================
// Linux Platform Detection
// ============================================================================

#if defined(__linux__)

#if defined(__x86_64__) || defined(_M_X64)
#include <sched.h>

static bool cpu_has_hybrid_flag(void) {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (edx >> 15) & 1; // Hybrid bit in EDX[15]
    }
    return false;
}

static void detect_topology_intel_linux(marmot_device_caps_t *caps) {
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    caps->topology.total_cores = (uint32_t)num_cpus;

    if (!cpu_has_hybrid_flag()) {
        // Homogeneous CPU - use all cores
        caps->topology.p_cores = caps->topology.total_cores;
        caps->topology.e_cores = 0;
        caps->topology.is_hybrid = false;
        return;
    }

    // Count P-cores and E-cores by querying CPUID 0x1A on each core
    // Core type: 0x20 = Atom (E-core), 0x40 = Core (P-core)
    uint32_t p_count = 0, e_count = 0;
    cpu_set_t original_affinity;
    if (sched_getaffinity(0, sizeof(original_affinity), &original_affinity) != 0) {
        // Can't get affinity, assume all P-cores
        caps->topology.p_cores = caps->topology.total_cores;
        caps->topology.e_cores = 0;
        caps->topology.is_hybrid = false;
        return;
    }

    for (int cpu = 0; cpu < num_cpus; cpu++) {
        cpu_set_t single_cpu;
        CPU_ZERO(&single_cpu);
        CPU_SET(cpu, &single_cpu);
        if (sched_setaffinity(0, sizeof(single_cpu), &single_cpu) != 0) {
            continue;
        }

        // Query CPUID leaf 0x1A for core type
        unsigned int eax, ebx, ecx, edx;
        __cpuid(0x1A, eax, ebx, ecx, edx);
        uint8_t core_type = (eax >> 24) & 0xFF;

        if (core_type == 0x40) { // Intel Core (P-core)
            p_count++;
        } else if (core_type == 0x20) { // Intel Atom (E-core)
            e_count++;
        } else {
            p_count++; // Unknown type, assume P-core
        }
    }

    // Restore original affinity
    sched_setaffinity(0, sizeof(original_affinity), &original_affinity);

    caps->topology.p_cores = p_count > 0 ? p_count : caps->topology.total_cores;
    caps->topology.e_cores = e_count;
    caps->topology.is_hybrid = (e_count > 0);
}
#endif // __x86_64__

#if defined(__aarch64__)
static void detect_topology_arm_linux(marmot_device_caps_t *caps) {
    // Detect big.LITTLE by checking max frequencies of each CPU
    // CPUs with highest max_freq are "big" cores (P-cores)
    long num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    caps->topology.total_cores = (uint32_t)num_cpus;

    uint64_t max_freq_overall = 0;
    uint64_t *freqs = (uint64_t *)calloc((size_t)num_cpus, sizeof(uint64_t));
    if (freqs == nullptr) {
        // Fallback: assume homogeneous
        caps->topology.p_cores = caps->topology.total_cores;
        caps->topology.e_cores = 0;
        caps->topology.is_hybrid = false;
        return;
    }

    // Read max frequency for each CPU
    for (int cpu = 0; cpu < num_cpus; cpu++) {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", cpu);
        FILE *f = fopen(path, "r");
        if (f != nullptr) {
            if (fscanf(f, "%lu", &freqs[cpu]) == 1) {
                if (freqs[cpu] > max_freq_overall) {
                    max_freq_overall = freqs[cpu];
                }
            }
            fclose(f);
        }
    }

    if (max_freq_overall == 0) {
        // Couldn't read frequencies, assume homogeneous
        free(freqs);
        caps->topology.p_cores = caps->topology.total_cores;
        caps->topology.e_cores = 0;
        caps->topology.is_hybrid = false;
        return;
    }

    // Count cores at max frequency (P-cores) vs below (E-cores)
    // Use 90% threshold to account for frequency rounding
    uint64_t threshold = max_freq_overall * 9 / 10;
    uint32_t p_count = 0, e_count = 0;

    for (int cpu = 0; cpu < num_cpus; cpu++) {
        if (freqs[cpu] >= threshold) {
            p_count++;
        } else if (freqs[cpu] > 0) {
            e_count++;
        } else {
            p_count++; // No freq data, assume P-core
        }
    }

    free(freqs);

    caps->topology.p_cores = p_count > 0 ? p_count : caps->topology.total_cores;
    caps->topology.e_cores = e_count;
    caps->topology.is_hybrid = (e_count > 0);
}

static marmot_cpu_microarch_t detect_arm_microarch_linux(void) {
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f == nullptr) {
        return MARMOT_CPU_UNKNOWN;
    }

    char line[256];
    uint32_t part_num = 0;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (strncmp(line, "CPU part", 8) == 0) {
            const char *val = strchr(line, ':');
            if (val != nullptr) {
                part_num = (uint32_t)strtoul(val + 1, nullptr, 16);
                break;
            }
        }
    }
    fclose(f);

    switch (part_num) {
    case 0xd03:
        return MARMOT_CPU_CORTEX_A53;
    case 0xd05:
        return MARMOT_CPU_CORTEX_A55;
    case 0xd07:
        return MARMOT_CPU_CORTEX_A57;
    case 0xd08:
        return MARMOT_CPU_CORTEX_A72;
    case 0xd0a:
        return MARMOT_CPU_CORTEX_A76;
    case 0xd0c:
        return MARMOT_CPU_NEOVERSE_N1;
    case 0xd40:
        return MARMOT_CPU_NEOVERSE_V1;
    case 0xd49:
        return MARMOT_CPU_NEOVERSE_N2;
    case 0xd44:
        return MARMOT_CPU_CORTEX_X1;
    case 0xd4c:
        return MARMOT_CPU_CORTEX_X2;
    default:
        return MARMOT_CPU_UNKNOWN;
    }
}
#endif

static marmot_device_caps_t detect_linux(void) {
    uint32_t cores = (uint32_t)sysconf(_SC_NPROCESSORS_ONLN);
    double freq_ghz = 0.0;
    char brand[128] = "linux-cpu";

    FILE *f = fopen("/proc/cpuinfo", "r");
    if (f != nullptr) {
        char line[256];
        while (fgets(line, sizeof(line), f) != nullptr) {
            if (strncmp(line, "model name", 10) == 0) {
                const char *val = strchr(line, ':');
                if (val != nullptr) {
                    val++;
                    while (*val == ' ')
                        val++;
                    strncpy(brand, val, sizeof(brand) - 1);
                    brand[sizeof(brand) - 1] = '\0';
                    char *nl = strchr(brand, '\n');
                    if (nl != nullptr)
                        *nl = '\0';
                }
            } else if (strncmp(line, "cpu MHz", 7) == 0) {
                const char *val = strchr(line, ':');
                if (val != nullptr) {
                    freq_ghz = atof(val + 1) / 1000.0;
                }
            }
        }
        fclose(f);
    }

    if (freq_ghz <= 0.0) {
        freq_ghz = DEFAULT_FREQ_GHZ;
    }

    marmot_device_caps_t caps = caps_from_hw(cores, freq_ghz, cpu_simd_width_bits(), false);

    // Detect CPU topology (hybrid vs homogeneous)
#if defined(__x86_64__) || defined(_M_X64)
    detect_topology_intel_linux(&caps);
#elif defined(__aarch64__)
    detect_topology_arm_linux(&caps);
    caps.backend.apple.cpu_microarch = detect_arm_microarch_linux();
    caps.backend.apple.has_arm_bf16 = (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
    caps.backend.apple.has_arm_dotprod = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
    caps.backend.apple.has_arm_sve = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
    caps.has_bf16_compute = caps.backend.apple.has_arm_bf16;
#else
    // Unknown architecture - assume homogeneous
    caps.topology.total_cores = cores;
    caps.topology.p_cores = cores;
    caps.topology.e_cores = 0;
    caps.topology.is_hybrid = false;
#endif

    apply_calibration(&caps, brand, cores);
    return caps;
}

#endif // __linux__

// ============================================================================
// Public API
// ============================================================================

marmot_profile_id_t marmot_cpu_detect_best_profile(void) {
#if defined(__APPLE__) && MARMOT_ENABLE_ACCELERATE
    return MARMOT_PROFILE_ACCELERATE;
#elif defined(__x86_64__) || defined(_M_X64)
    if (cpu_has_avx512()) {
        return MARMOT_PROFILE_AVX512;
    }
    if (cpu_has_avx2()) {
        return MARMOT_PROFILE_AVX2;
    }
    return MARMOT_PROFILE_SCALAR;
#elif defined(__aarch64__) || defined(_M_ARM64)
#if MARMOT_ENABLE_NEON
    return MARMOT_PROFILE_NEON;
#else
    return MARMOT_PROFILE_SCALAR;
#endif
#else
    return MARMOT_PROFILE_SCALAR;
#endif
}

marmot_device_caps_t marmot_cpu_detect_capabilities(void) {
#if defined(__APPLE__)
    return detect_apple();
#elif defined(__linux__)
    return detect_linux();
#else
    uint32_t cores = 1;
#if defined(_SC_NPROCESSORS_ONLN)
    long hw = sysconf(_SC_NPROCESSORS_ONLN);
    if (hw > 0)
        cores = (uint32_t)hw;
#endif
    return (marmot_device_caps_t){
        .peak_flops_tflops_fp32 = 0.5f,
        .peak_flops_tflops_fp16 = 0.5f,
        .mem_bw_gbps = 25.0f,
        .simd_width = cpu_simd_width_bits(),
        .launch_overhead_us = 0.1f,
        .edge_penalty_alpha = DEFAULT_EDGE_PENALTY_ALPHA,
        .topology = {
            .total_cores = cores,
            .p_cores = cores,
            .e_cores = 0,
            .is_hybrid = false,
        },
    };
#endif
}

size_t marmot_cpu_optimal_thread_count(const marmot_device_caps_t *caps) {
    // Environment variable override
    const char *env = getenv("MARMOT_NUM_THREADS");
    if (env != nullptr && env[0] != '\0') {
        long n = strtol(env, nullptr, 10);
        if (n > 0 && n <= 64) {
            return (size_t)n;
        }
    }

    // Hybrid CPU: use P-cores - 1 to avoid scheduler spillover to E-cores.
    // Even with libdispatch + QOS_CLASS_USER_INTERACTIVE, using exactly
    // P-core count causes performance degradation. Empirically on M4 Pro:
    // 9 threads = 0.78 TFLOPS, 10 threads = 0.47 TFLOPS.
    if (caps->topology.is_hybrid && caps->topology.p_cores > 1) {
        return caps->topology.p_cores - 1;
    }

    // Homogeneous CPU: use all cores
    if (caps->topology.total_cores > 0) {
        return caps->topology.total_cores;
    }

    // Fallback
    return 1;
}

marmot_cpu_microarch_t marmot_cpu_detect_microarch(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    char brand[128] = "apple-cpu";
    sysctl_str("machdep.cpu.brand_string", brand, sizeof(brand));
    return detect_apple_silicon_microarch(brand);
#elif defined(__linux__) && defined(__aarch64__)
    return detect_arm_microarch_linux();
#else
    return MARMOT_CPU_UNKNOWN;
#endif
}

bool marmot_cpu_has_arm_bf16(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    return detect_arm_feature_macos("FEAT_BF16");
#elif defined(__linux__) && defined(__aarch64__)
    return (getauxval(AT_HWCAP2) & HWCAP2_BF16) != 0;
#else
    return false;
#endif
}

bool marmot_cpu_has_arm_dotprod(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    return detect_arm_feature_macos("FEAT_DotProd");
#elif defined(__linux__) && defined(__aarch64__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
    return false;
#endif
}

bool marmot_cpu_has_arm_i8mm(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    return detect_arm_feature_macos("FEAT_I8MM");
#elif defined(__linux__) && defined(__aarch64__)
    return (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
    return false;
#endif
}

bool marmot_cpu_has_arm_sve(void) {
#if defined(__APPLE__) && defined(__aarch64__)
    return detect_arm_feature_macos("FEAT_SVE");
#elif defined(__linux__) && defined(__aarch64__)
    return (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
#else
    return false;
#endif
}
