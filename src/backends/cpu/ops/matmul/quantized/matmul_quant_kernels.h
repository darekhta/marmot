#ifndef CPU_MATMUL_QUANT_KERNELS_H
#define CPU_MATMUL_QUANT_KERNELS_H

#include "cpu_backend_internal.h"
#include "matmul_quant_table.h"

const cpu_matmul_quant_kernel_t *cpu_matmul_quant_kernels_scalar(void);
const cpu_matmul_quant_kernel_t *cpu_matmul_quant_kernels_neon(void);
const cpu_matmul_quant_kernel_t *cpu_matmul_quant_kernels_avx2(void);

static inline const cpu_matmul_quant_kernel_t *cpu_matmul_quant_kernels_best(const void *device_ctx) {
#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        return cpu_matmul_quant_kernels_avx2();
    }
#endif

#if HAS_NEON
    if (has_neon(device_ctx)) {
        return cpu_matmul_quant_kernels_neon();
    }
#endif

    return cpu_matmul_quant_kernels_scalar();
}

static inline const cpu_matmul_quant_kernel_t *
cpu_matmul_quant_select_kernel(const void *device_ctx, marmot_quant_kind_t kind) {
    if (kind >= MARMOT_QUANT_KIND_COUNT) {
        return nullptr;
    }
    const cpu_matmul_quant_kernel_t *kernels = cpu_matmul_quant_kernels_best(device_ctx);
    if (kernels == nullptr) {
        return nullptr;
    }
    return kernels + kind;
}

#endif // CPU_MATMUL_QUANT_KERNELS_H
