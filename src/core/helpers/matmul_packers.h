#ifndef MARMOT_CORE_HELPERS_MATMUL_PACKERS_H
#define MARMOT_CORE_HELPERS_MATMUL_PACKERS_H

#include "marmot/types.h"

typedef enum {
    MARMOT_MATMUL_ACTIVATION_PACKER_Q8_0 = 0,
    MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K = 1,
} marmot_matmul_activation_packer_kind_t;

#define MARMOT_MATMUL_ACTIVATION_PACKER_TABLE(APPLY)                                                                   \
    APPLY(MARMOT_QUANT_KIND_Q4_0, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_0)                                                \
    APPLY(MARMOT_QUANT_KIND_Q4_1, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_0)                                                \
    APPLY(MARMOT_QUANT_KIND_Q5_0, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_0)                                                \
    APPLY(MARMOT_QUANT_KIND_Q5_1, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_0)                                                \
    APPLY(MARMOT_QUANT_KIND_Q8_0, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_0)                                                \
    APPLY(MARMOT_QUANT_KIND_Q8_1, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_0)                                                \
    APPLY(MARMOT_QUANT_KIND_Q2_K, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K)                                                \
    APPLY(MARMOT_QUANT_KIND_Q3_K, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K)                                                \
    APPLY(MARMOT_QUANT_KIND_Q4_K, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K)                                                \
    APPLY(MARMOT_QUANT_KIND_Q5_K, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K)                                                \
    APPLY(MARMOT_QUANT_KIND_Q6_K, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K)                                                \
    APPLY(MARMOT_QUANT_KIND_Q8_K, MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K)

#define MARMOT_DECLARE_PACKER_KIND_ENUM(kind, packer) enum { MARMOT_MATP_PACKER_KIND_##kind = packer };
MARMOT_MATMUL_ACTIVATION_PACKER_TABLE(MARMOT_DECLARE_PACKER_KIND_ENUM)
#undef MARMOT_DECLARE_PACKER_KIND_ENUM

#define MARMOT_MATP_PACKER_KIND(kind) ((marmot_matmul_activation_packer_kind_t)MARMOT_MATP_PACKER_KIND_##kind)

#endif // MARMOT_CORE_HELPERS_MATMUL_PACKERS_H
