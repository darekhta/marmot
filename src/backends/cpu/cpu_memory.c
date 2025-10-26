#include "cpu_backend_internal.h"

static marmot_error_t cpu_allocator_tracker_add(cpu_context_t *ctx, const marmot_allocation_t *info) {
    if (ctx == nullptr || info == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_allocation_entry_t *entry = (cpu_allocation_entry_t *)malloc(sizeof(cpu_allocation_entry_t));
    if (entry == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }
    entry->ptr = info->ptr;
    entry->info = *info;
    pthread_mutex_lock(&ctx->allocator_tracker.mutex);
    entry->next = ctx->allocator_tracker.head;
    ctx->allocator_tracker.head = entry;
    pthread_mutex_unlock(&ctx->allocator_tracker.mutex);
    return MARMOT_SUCCESS;
}

static cpu_allocation_entry_t *cpu_allocator_tracker_take(cpu_context_t *ctx, void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return nullptr;
    }
    pthread_mutex_lock(&ctx->allocator_tracker.mutex);
    cpu_allocation_entry_t *prev = nullptr;
    cpu_allocation_entry_t *node = ctx->allocator_tracker.head;
    while (node != nullptr) {
        if (node->ptr == ptr) {
            if (prev == nullptr) {
                ctx->allocator_tracker.head = node->next;
            } else {
                prev->next = node->next;
            }
            pthread_mutex_unlock(&ctx->allocator_tracker.mutex);
            return node;
        }
        prev = node;
        node = node->next;
    }
    pthread_mutex_unlock(&ctx->allocator_tracker.mutex);
    return nullptr;
}

// ==================================================================
// CPU Backend Memory Operations
// ==================================================================
// For CPU backend, memory operations are simple wrappers around
// standard C memory functions since CPU has unified memory.
// ==================================================================

marmot_error_t cpu_alloc(const void *device_ctx, size_t size, void **ptr) {
    if (ptr == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (ctx == nullptr || ctx->allocator_ops == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "CPU allocator requires a valid backend context");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    marmot_allocation_t allocation = {0};
    marmot_error_t err = ctx->allocator_ops->alloc(ctx, size, 0, MARMOT_ALLOC_HEAP, &allocation);
    if (err != MARMOT_SUCCESS) {
        return err;
    }
    marmot_error_t track_err = cpu_allocator_tracker_add(ctx, &allocation);
    if (track_err != MARMOT_SUCCESS) {
        ctx->allocator_ops->free(ctx, &allocation);
        return track_err;
    }
    *ptr = allocation.ptr;
    return MARMOT_SUCCESS;
}

void cpu_free(const void *device_ctx, void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (ctx == nullptr || ctx->allocator_ops == nullptr) {
        free(ptr);
        return;
    }
    cpu_allocation_entry_t *entry = cpu_allocator_tracker_take(ctx, ptr);
    if (entry == nullptr) {
        free(ptr);
        return;
    }
    ctx->allocator_ops->free(ctx, &entry->info);
    free(entry);
}

marmot_error_t cpu_memcpy_to_device([[maybe_unused]] const void *device_ctx, void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
    return MARMOT_SUCCESS;
}

marmot_error_t
cpu_memcpy_from_device([[maybe_unused]] const void *device_ctx, void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_allocator_usage([[maybe_unused]] const void *device_ctx, marmot_allocator_usage_t *usage) {
    if (usage == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_allocator_collect_usage(usage);
    return MARMOT_SUCCESS;
}
