#ifndef CPU_MATMUL_ACTIVATION_PACKERS_CPU_H
#define CPU_MATMUL_ACTIVATION_PACKERS_CPU_H

#include "core/helpers/matmul_packers.h"
#include "matmul_quant_activation.h"

#define CPU_ACT_PACK_KIND(kind_enum) MARMOT_MATP_PACKER_KIND(kind_enum)
#define CPU_ACT_PACK_USES_Q8K(kind_enum) (CPU_ACT_PACK_KIND(kind_enum) == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K)
#define CPU_ACT_PACK_F32(kind_enum)                                                                                    \
    (CPU_ACT_PACK_USES_Q8K(kind_enum) ? cpu_matmul_quant_pack_q8_k_f32 : cpu_matmul_quant_pack_q8_0_f32)
#define CPU_ACT_PACK_F16(kind_enum)                                                                                    \
    (CPU_ACT_PACK_USES_Q8K(kind_enum) ? cpu_matmul_quant_pack_q8_k_f16 : cpu_matmul_quant_pack_q8_0_f16)

#endif // CPU_MATMUL_ACTIVATION_PACKERS_CPU_H
