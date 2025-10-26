#include "marmot/allocator.h"
#include "marmot/config.h"
#include "marmot/error.h"

#include <stddef.h>

extern const marmot_allocator_ops_t cpu_allocator_ops;

#if defined(__APPLE__) && MARMOT_ENABLE_METAL
extern const marmot_allocator_ops_t metal_allocator_ops;
#endif

const marmot_allocator_ops_t *marmot_get_allocator(marmot_backend_type_t backend) {
    switch (backend) {
    case MARMOT_BACKEND_CPU:
        return &cpu_allocator_ops;
#if defined(__APPLE__) && MARMOT_ENABLE_METAL
    case MARMOT_BACKEND_METAL:
        return &metal_allocator_ops;
#endif
    default:
        return &cpu_allocator_ops;
    }
}
