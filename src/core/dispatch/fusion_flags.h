#ifndef MARMOT_CORE_DISPATCH_FUSION_FLAGS_H
#define MARMOT_CORE_DISPATCH_FUSION_FLAGS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MARMOT_FUSION_NONE = 0,
    MARMOT_FUSION_RESIDUAL_ADD = 1u << 14,
    MARMOT_FUSION_CUSTOM = 1u << 30,
} marmot_fusion_flags_t;

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_DISPATCH_FUSION_FLAGS_H
