#ifndef MARMOT_ALLOCATOR_H
#define MARMOT_ALLOCATOR_H

#include "error.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MARMOT_ALLOC_HEAP = 0,
    MARMOT_ALLOC_ALIGNED = 1,
    MARMOT_ALLOC_HUGE_PAGES = 2,
    MARMOT_ALLOC_GPU_SHARED = 3,
    MARMOT_ALLOC_GPU_PRIVATE = 4,
} marmot_alloc_type_t;

typedef struct {
    void *ptr;
    size_t size;
    size_t alignment;
    marmot_alloc_type_t type;
    uint64_t alloc_id;
} marmot_allocation_t;

typedef struct {
    size_t current_bytes;
    size_t peak_bytes;
    size_t pooled_bytes;
    size_t active_allocations;
    size_t peak_allocations;
    uint64_t total_allocations;
    uint64_t pool_hits;
    uint64_t pool_misses;
} marmot_allocator_usage_t;

typedef struct {
    marmot_error_t (*alloc)(
        void *allocator_ctx, size_t size, size_t alignment, marmot_alloc_type_t type, marmot_allocation_t *out
    );
    void (*free)(void *allocator_ctx, marmot_allocation_t *alloc);
    marmot_error_t (*realloc)(void *allocator_ctx, marmot_allocation_t *alloc, size_t new_size);
    size_t (*get_peak_usage)(void *allocator_ctx);
    size_t (*get_current_usage)(void *allocator_ctx);
} marmot_allocator_ops_t;

const marmot_allocator_ops_t *marmot_get_allocator(marmot_backend_type_t backend);
marmot_error_t marmot_allocator_get_usage(const marmot_context_t *ctx, marmot_allocator_usage_t *usage);

#ifdef __cplusplus
}
#endif

#endif
