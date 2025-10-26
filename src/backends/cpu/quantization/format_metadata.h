#ifndef CPU_QUANT_FORMAT_METADATA_H
#define CPU_QUANT_FORMAT_METADATA_H

#include "marmot/quant_block.h"
#include "marmot/types.h"

#include <stddef.h>

#include "core/helpers/matmul_packers.h"

typedef struct cpu_quant_format_info {
    marmot_quant_kind_t kind;
    size_t block_bytes;
    size_t block_values;
    marmot_quant_layout_t layout;
    marmot_matmul_activation_packer_kind_t activation_packer;
} cpu_quant_format_info_t;

#define CPU_QUANT_FORMAT_TABLE(APPLY)                                                                                  \
    APPLY(MARMOT_QUANT_KIND_Q4_0, MARMOT_QUANT_BLOCK_SIZE, sizeof(marmot_q4_0_block_t), MARMOT_QUANT_LAYOUT_GGUF)      \
    APPLY(MARMOT_QUANT_KIND_Q4_1, MARMOT_QUANT_BLOCK_SIZE, sizeof(marmot_q4_1_block_t), MARMOT_QUANT_LAYOUT_GGUF)      \
    APPLY(MARMOT_QUANT_KIND_Q5_0, MARMOT_QUANT_BLOCK_SIZE, sizeof(marmot_q5_0_block_t), MARMOT_QUANT_LAYOUT_GGUF)      \
    APPLY(MARMOT_QUANT_KIND_Q5_1, MARMOT_QUANT_BLOCK_SIZE, sizeof(marmot_q5_1_block_t), MARMOT_QUANT_LAYOUT_GGUF)      \
    APPLY(MARMOT_QUANT_KIND_Q8_0, MARMOT_QUANT_BLOCK_SIZE, sizeof(marmot_q8_0_block_t), MARMOT_QUANT_LAYOUT_GGUF)      \
    APPLY(MARMOT_QUANT_KIND_Q8_1, MARMOT_QUANT_BLOCK_SIZE, sizeof(marmot_q8_1_block_t), MARMOT_QUANT_LAYOUT_GGUF)      \
    APPLY(MARMOT_QUANT_KIND_Q2_K, MARMOT_QK_K_VALUES, sizeof(marmot_q2_k_block_t), MARMOT_QUANT_LAYOUT_GGUF)           \
    APPLY(MARMOT_QUANT_KIND_Q3_K, MARMOT_QK_K_VALUES, sizeof(marmot_q3_k_block_t), MARMOT_QUANT_LAYOUT_GGUF)           \
    APPLY(MARMOT_QUANT_KIND_Q4_K, MARMOT_QK_K_VALUES, sizeof(marmot_q4_k_block_t), MARMOT_QUANT_LAYOUT_GGUF)           \
    APPLY(MARMOT_QUANT_KIND_Q5_K, MARMOT_QK_K_VALUES, sizeof(marmot_q5_k_block_t), MARMOT_QUANT_LAYOUT_GGUF)           \
    APPLY(MARMOT_QUANT_KIND_Q6_K, MARMOT_QK_K_VALUES, sizeof(marmot_q6_k_block_t), MARMOT_QUANT_LAYOUT_GGUF)           \
    APPLY(MARMOT_QUANT_KIND_Q8_K, MARMOT_QK_K_VALUES, sizeof(marmot_q8_k_block_t), MARMOT_QUANT_LAYOUT_GGUF)

#define CPU_QUANT_FORMAT_ENUM(qkind, values, bytes, qlayout)                                                           \
    enum { CPU_QUANT_FORMAT_BLOCK_VALUES_##qkind = (values) };                                                         \
    enum { CPU_QUANT_FORMAT_BLOCK_BYTES_##qkind = (bytes) };                                                           \
    enum { CPU_QUANT_FORMAT_LAYOUT_##qkind = (qlayout) };

CPU_QUANT_FORMAT_TABLE(CPU_QUANT_FORMAT_ENUM)

#undef CPU_QUANT_FORMAT_ENUM

#define CPU_QUANT_FORMAT_BLOCK_VALUES(kind) ((size_t)CPU_QUANT_FORMAT_BLOCK_VALUES_##kind)
#define CPU_QUANT_FORMAT_BLOCK_BYTES(kind) ((size_t)CPU_QUANT_FORMAT_BLOCK_BYTES_##kind)
#define CPU_QUANT_FORMAT_LAYOUT(kind) ((marmot_quant_layout_t)CPU_QUANT_FORMAT_LAYOUT_##kind)
#define CPU_QUANT_FORMAT_ACT_PACKER(kind) ((marmot_matmul_activation_packer_kind_t)MARMOT_MATP_PACKER_KIND(kind))
#define CPU_QUANT_FORMAT_INFO(kind) (&k_cpu_quant_format_table[(kind)])
#define CPU_QUANT_FORMAT_USES_Q8_K(kind) (CPU_QUANT_FORMAT_ACT_PACKER(kind) == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K)

[[maybe_unused]] static const cpu_quant_format_info_t k_cpu_quant_format_table[MARMOT_QUANT_KIND_COUNT] = {
#define CPU_QUANT_FORMAT_ENTRY(qkind, values, bytes, qlayout)                                                          \
    [qkind] = {                                                                                                        \
        .kind = qkind,                                                                                                 \
        .block_bytes = (bytes),                                                                                        \
        .block_values = (values),                                                                                      \
        .layout = (qlayout),                                                                                           \
        .activation_packer = CPU_QUANT_FORMAT_ACT_PACKER(qkind),                                                       \
    },
    CPU_QUANT_FORMAT_TABLE(CPU_QUANT_FORMAT_ENTRY)
#undef CPU_QUANT_FORMAT_ENTRY
};

static inline const cpu_quant_format_info_t *cpu_quant_format_info(marmot_quant_kind_t kind) {
    if (kind >= MARMOT_QUANT_KIND_COUNT) {
        return nullptr;
    }
    const cpu_quant_format_info_t *info = &k_cpu_quant_format_table[kind];
    if (info->block_values == 0 || info->block_bytes == 0) {
        return nullptr;
    }
    return info;
}

static inline size_t cpu_quant_format_block_values(marmot_quant_kind_t kind) {
    const cpu_quant_format_info_t *info = cpu_quant_format_info(kind);
    return info != nullptr ? info->block_values : 0;
}

static inline size_t cpu_quant_format_block_bytes(marmot_quant_kind_t kind) {
    const cpu_quant_format_info_t *info = cpu_quant_format_info(kind);
    return info != nullptr ? info->block_bytes : 0;
}

static inline marmot_quant_layout_t cpu_quant_format_layout(marmot_quant_kind_t kind) {
    const cpu_quant_format_info_t *info = cpu_quant_format_info(kind);
    return info != nullptr ? info->layout : MARMOT_QUANT_LAYOUT_GENERIC;
}

static inline bool cpu_quant_format_uses_q8_k(marmot_quant_kind_t kind) {
    const cpu_quant_format_info_t *info = cpu_quant_format_info(kind);
    return info != nullptr && info->activation_packer == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K;
}

static inline marmot_quant_kind_t cpu_quant_format_activation_kind(marmot_quant_kind_t kind) {
    const cpu_quant_format_info_t *info = cpu_quant_format_info(kind);
    if (info == nullptr) {
        return MARMOT_QUANT_KIND_COUNT;
    }
    return info->activation_packer == MARMOT_MATMUL_ACTIVATION_PACKER_Q8_K ? MARMOT_QUANT_KIND_Q8_K
                                                                           : MARMOT_QUANT_KIND_Q8_0;
}

#endif // CPU_QUANT_FORMAT_METADATA_H
