#ifndef MARMOT_CORE_DISPATCH_FUSION_DETECTION_H
#define MARMOT_CORE_DISPATCH_FUSION_DETECTION_H

#include "marmot/tensor.h"
#include "marmot/traits_ids.gen.h"

#include "core/dispatch/fusion_flags.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MARMOT_FUSION_PATTERN_NONE = 0,
    MARMOT_FUSION_PATTERN_ADD_RELU = 1u << 0,
    MARMOT_FUSION_PATTERN_ADD_GELU = 1u << 1,
    MARMOT_FUSION_PATTERN_ADD_SILU = 1u << 2,
    MARMOT_FUSION_PATTERN_MUL_ADD = 1u << 3,
    MARMOT_FUSION_PATTERN_MATMUL_BIAS = 1u << 4,
    MARMOT_FUSION_PATTERN_MATMUL_BIAS_RELU = 1u << 5,
    MARMOT_FUSION_PATTERN_MATMUL_BIAS_GELU = 1u << 6,
    MARMOT_FUSION_PATTERN_MATMUL_BIAS_SILU = 1u << 7,
} marmot_fusion_pattern_t;

typedef struct {
    marmot_op_id_t prev_op;
    marmot_op_id_t current_op;
    marmot_op_id_t next_op;
    marmot_op_id_t next_next_op;
    const marmot_tensor_t *intermediate;
    bool intermediate_is_temporary;
    bool next_intermediate_is_temporary;
    marmot_fusion_pattern_t detected_pattern;
} marmot_fusion_context_t;

marmot_fusion_pattern_t marmot_detect_fusion_pattern(const marmot_fusion_context_t *ctx);
marmot_op_id_t marmot_detect_fused_op_id(const marmot_fusion_context_t *ctx);

bool marmot_backend_supports_fusion(marmot_backend_type_t backend_type, marmot_fusion_flags_t flags);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_DISPATCH_FUSION_DETECTION_H
