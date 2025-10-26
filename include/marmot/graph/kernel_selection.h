#ifndef MARMOT_GRAPH_KERNEL_SELECTION_H
#define MARMOT_GRAPH_KERNEL_SELECTION_H

#include "marmot/traits_ids.gen.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    bool supported;
    marmot_kernel_id_t kernel_id;
    uint16_t op_index;
    double estimated_us;
    double est_comm_us;
    double est_workspace_mb;
    float confidence;
    const char *fallback_reason;
    uint32_t shardable_axes;
    uint32_t device_affinity;
} marmot_kernel_selection_t;

#define MARMOT_KERNEL_OP_INDEX_INVALID UINT16_MAX

#ifdef __cplusplus
}
#endif

#endif // MARMOT_GRAPH_KERNEL_SELECTION_H
