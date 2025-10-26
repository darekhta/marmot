#ifndef MARMOT_CORE_TENSOR_DEBUG_H
#define MARMOT_CORE_TENSOR_DEBUG_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t live_bytes;
    size_t peak_bytes;
    size_t alloc_bytes;
    size_t free_bytes;
    size_t allocs;
    size_t frees;
} marmot_tensor_debug_stats_t;

void marmot_tensor_debug_record_alloc(size_t bytes);
void marmot_tensor_debug_record_free(size_t bytes);
void marmot_tensor_debug_snapshot(marmot_tensor_debug_stats_t *out);

#ifdef __cplusplus
}
#endif

#endif
