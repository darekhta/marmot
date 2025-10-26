#ifndef MARMOT_DISPATCH_H
#define MARMOT_DISPATCH_H

#include "marmot/error.h"

#include <stddef.h>

typedef enum {
    MARMOT_DISPATCH_PRIORITY_HIGH,
    MARMOT_DISPATCH_PRIORITY_NORMAL,
    MARMOT_DISPATCH_PRIORITY_LOW,
} marmot_dispatch_priority_t;

typedef void (*marmot_dispatch_work_fn)(void *context, size_t index);
typedef void (*marmot_dispatch_range_fn)(void *context, size_t start, size_t end);
typedef marmot_error_t (*marmot_dispatch_range_error_fn)(void *context, size_t start, size_t end);
typedef void (*marmot_dispatch_async_fn)(void *context);

void marmot_dispatch_init(void);
void marmot_dispatch_destroy(void);

void marmot_dispatch_parallel_for(
    marmot_dispatch_priority_t priority, size_t count, void *context, marmot_dispatch_work_fn work
);

void marmot_dispatch_parallel_for_range(
    marmot_dispatch_priority_t priority, size_t count, size_t min_chunk_size, void *context,
    marmot_dispatch_range_fn work
);

marmot_error_t marmot_dispatch_parallel_for_range_with_error(
    marmot_dispatch_priority_t priority, size_t count, size_t min_chunk_size, void *context,
    marmot_dispatch_range_error_fn work
);

void marmot_dispatch_async(marmot_dispatch_priority_t priority, void *context, marmot_dispatch_async_fn work);

void marmot_dispatch_barrier_sync(marmot_dispatch_priority_t priority);

const char *marmot_dispatch_priority_name(marmot_dispatch_priority_t priority);

#endif
