#include "marmot/allocator.h"

#include "marmot/error.h"

#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <errno.h>
#include <pthread.h>
#include <string.h>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(__linux__) && defined(MAP_HUGE_SHIFT) && !defined(MAP_HUGE_2MB)
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

typedef struct {
    pthread_mutex_t mutex;
    size_t current_usage;
    size_t peak_usage;
    uint64_t next_id;
    size_t pooled_bytes;
    size_t pool_capacity_bytes;
    size_t active_allocations;
    size_t peak_allocations;
    uint64_t total_allocations;
    uint64_t pool_hits;
    uint64_t pool_misses;
} cpu_allocator_state_t;

#define CPU_ALLOCATOR_POOL_BUCKET_COUNT 12
static const size_t cpu_allocator_pool_bucket_sizes[CPU_ALLOCATOR_POOL_BUCKET_COUNT] = {256,   512,    1024,   2048,
                                                                                        4096,  8192,   16384,  32768,
                                                                                        65536, 131072, 262144, 524288};
static const size_t cpu_allocator_pool_limit_bytes = 64ULL * 1024ULL * 1024ULL;

typedef struct cpu_pool_entry {
    void *ptr;
    size_t size;
    struct cpu_pool_entry *next;
} cpu_pool_entry_t;

typedef struct {
    cpu_pool_entry_t *head;
    size_t count;
} cpu_pool_bucket_t;

static cpu_allocator_state_t cpu_allocator_state = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .current_usage = 0,
    .peak_usage = 0,
    .next_id = 1,
    .pooled_bytes = 0,
    .pool_capacity_bytes = cpu_allocator_pool_limit_bytes,
    .active_allocations = 0,
    .peak_allocations = 0,
    .total_allocations = 0,
    .pool_hits = 0,
    .pool_misses = 0,
};
static cpu_pool_bucket_t cpu_allocator_pool[CPU_ALLOCATOR_POOL_BUCKET_COUNT] = {0};

static size_t cpu_allocator_pool_bucket_for(size_t size) {
    for (size_t i = 0; i < CPU_ALLOCATOR_POOL_BUCKET_COUNT; ++i) {
        if (size <= cpu_allocator_pool_bucket_sizes[i]) {
            return i;
        }
    }
    return CPU_ALLOCATOR_POOL_BUCKET_COUNT;
}

static size_t cpu_allocator_pool_bucket_size(size_t bucket) {
    if (bucket >= CPU_ALLOCATOR_POOL_BUCKET_COUNT) {
        return 0;
    }
    return cpu_allocator_pool_bucket_sizes[bucket];
}

static bool cpu_allocator_pool_acquire(size_t request_size, void **out_ptr, size_t *out_size) {
    if (out_ptr == nullptr || out_size == nullptr) {
        return false;
    }
    size_t bucket = cpu_allocator_pool_bucket_for(request_size);
    if (bucket >= CPU_ALLOCATOR_POOL_BUCKET_COUNT) {
        return false;
    }

    pthread_mutex_lock(&cpu_allocator_state.mutex);
    cpu_pool_entry_t *entry = cpu_allocator_pool[bucket].head;
    if (entry == nullptr) {
        cpu_allocator_state.pool_misses++;
        pthread_mutex_unlock(&cpu_allocator_state.mutex);
        return false;
    }
    cpu_allocator_pool[bucket].head = entry->next;
    if (cpu_allocator_state.pooled_bytes >= entry->size) {
        cpu_allocator_state.pooled_bytes -= entry->size;
    } else {
        cpu_allocator_state.pooled_bytes = 0;
    }
    cpu_allocator_state.pool_hits++;
    pthread_mutex_unlock(&cpu_allocator_state.mutex);

    *out_ptr = entry->ptr;
    *out_size = entry->size;
    free(entry);
    return true;
}

static bool cpu_allocator_pool_release(void *ptr, size_t size) {
    if (ptr == nullptr || size == 0) {
        return false;
    }
    size_t bucket = cpu_allocator_pool_bucket_for(size);
    if (bucket >= CPU_ALLOCATOR_POOL_BUCKET_COUNT) {
        return false;
    }

    cpu_pool_entry_t *entry = (cpu_pool_entry_t *)malloc(sizeof(cpu_pool_entry_t));
    if (entry == nullptr) {
        return false;
    }
    entry->ptr = ptr;
    entry->size = cpu_allocator_pool_bucket_size(bucket);

    pthread_mutex_lock(&cpu_allocator_state.mutex);
    if (cpu_allocator_state.pooled_bytes + entry->size > cpu_allocator_state.pool_capacity_bytes) {
        pthread_mutex_unlock(&cpu_allocator_state.mutex);
        free(entry);
        return false;
    }
    entry->next = cpu_allocator_pool[bucket].head;
    cpu_allocator_pool[bucket].head = entry;
    cpu_allocator_pool[bucket].count++;
    cpu_allocator_state.pooled_bytes += entry->size;
    pthread_mutex_unlock(&cpu_allocator_state.mutex);
    return true;
}

static size_t cpu_allocator_adjust_request_size(size_t size, marmot_alloc_type_t type) {
    if (type != MARMOT_ALLOC_HEAP) {
        return size;
    }
    size_t bucket = cpu_allocator_pool_bucket_for(size);
    if (bucket >= CPU_ALLOCATOR_POOL_BUCKET_COUNT) {
        return size;
    }
    return cpu_allocator_pool_bucket_size(bucket);
}

static size_t normalize_alignment(size_t alignment) {
    size_t min_alignment = alignof(max_align_t);
    if (alignment < min_alignment) {
        alignment = min_alignment;
    }
    if ((alignment & (alignment - 1)) == 0) {
        return alignment;
    }
    size_t value = min_alignment;
    while (value < alignment && value < (SIZE_MAX >> 1)) {
        value <<= 1;
    }
    return value;
}

static void cpu_allocator_record_alloc(size_t bytes, marmot_allocation_t *alloc) {
    pthread_mutex_lock(&cpu_allocator_state.mutex);
    cpu_allocator_state.current_usage += bytes;
    if (cpu_allocator_state.current_usage > cpu_allocator_state.peak_usage) {
        cpu_allocator_state.peak_usage = cpu_allocator_state.current_usage;
    }
    cpu_allocator_state.total_allocations++;
    cpu_allocator_state.active_allocations++;
    if (cpu_allocator_state.active_allocations > cpu_allocator_state.peak_allocations) {
        cpu_allocator_state.peak_allocations = cpu_allocator_state.active_allocations;
    }
    alloc->alloc_id = cpu_allocator_state.next_id++;
    pthread_mutex_unlock(&cpu_allocator_state.mutex);
}

static void cpu_allocator_record_free(size_t bytes) {
    pthread_mutex_lock(&cpu_allocator_state.mutex);
    if (cpu_allocator_state.current_usage >= bytes) {
        cpu_allocator_state.current_usage -= bytes;
    } else {
        cpu_allocator_state.current_usage = 0;
    }
    if (cpu_allocator_state.active_allocations > 0) {
        cpu_allocator_state.active_allocations--;
    }
    pthread_mutex_unlock(&cpu_allocator_state.mutex);
}

static marmot_error_t cpu_allocator_alloc(
    void *allocator_ctx, size_t size, size_t alignment, marmot_alloc_type_t type, marmot_allocation_t *out
) {
    (void)allocator_ctx;
    if (out == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t request_size = (size == 0) ? 1 : size;
    size_t request_alignment = alignment;
    marmot_alloc_type_t effective_type = type;
    void *ptr = nullptr;
    bool satisfied = false;

    if (type == MARMOT_ALLOC_HEAP) {
        if (cpu_allocator_pool_acquire(request_size, &ptr, &request_size)) {
            request_alignment = alignof(max_align_t);
            satisfied = true;
        } else {
            request_size = cpu_allocator_adjust_request_size(request_size, type);
        }
    }

    if (!satisfied) {
        switch (type) {
        case MARMOT_ALLOC_HEAP:
            ptr = malloc(request_size);
            request_alignment = alignof(max_align_t);
            break;
        case MARMOT_ALLOC_ALIGNED: {
            request_alignment = normalize_alignment(alignment == 0 ? alignof(max_align_t) : alignment);
            int rc = posix_memalign(&ptr, request_alignment, request_size);
            if (rc != 0) {
                ptr = nullptr;
            }
            break;
        }
        case MARMOT_ALLOC_HUGE_PAGES: {
#if defined(__linux__) && defined(MAP_HUGETLB)
            const size_t huge_alignment = (size_t)2 * 1024 * 1024;
            size_t map_size = (request_size + huge_alignment - 1) / huge_alignment * huge_alignment;
            int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;
#ifdef MAP_HUGE_2MB
            flags |= MAP_HUGE_2MB;
#endif
            void *mapped = mmap(nullptr, map_size, PROT_READ | PROT_WRITE, flags, -1, 0);
            if (mapped != MAP_FAILED) {
                ptr = mapped;
                request_size = map_size;
                request_alignment = huge_alignment;
                break;
            }
#endif
            const size_t huge_alignment = (size_t)2 * 1024 * 1024;
            request_alignment = normalize_alignment(alignment == 0 ? huge_alignment : alignment);
            int rc = posix_memalign(&ptr, request_alignment, request_size);
            if (rc != 0) {
                ptr = nullptr;
            }
            effective_type = MARMOT_ALLOC_ALIGNED;
            break;
        }
        case MARMOT_ALLOC_GPU_SHARED:
        case MARMOT_ALLOC_GPU_PRIVATE: {
            request_alignment = normalize_alignment(alignment == 0 ? alignof(max_align_t) : alignment);
            int rc = posix_memalign(&ptr, request_alignment, request_size);
            if (rc != 0) {
                ptr = nullptr;
            }
            effective_type = MARMOT_ALLOC_ALIGNED;
            break;
        }
        default:
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Unknown allocation type");
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
    }

    if (ptr == nullptr) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "CPU allocator failed to allocate memory");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    out->ptr = ptr;
    out->size = request_size;
    out->alignment = request_alignment;
    out->type = effective_type;
    cpu_allocator_record_alloc(request_size, out);
    return MARMOT_SUCCESS;
}

static void cpu_allocator_free(void *allocator_ctx, marmot_allocation_t *alloc) {
    (void)allocator_ctx;
    if (alloc == nullptr || alloc->ptr == nullptr) {
        return;
    }
    cpu_allocator_record_free(alloc->size);
#if defined(__linux__)
    if (alloc->type == MARMOT_ALLOC_HUGE_PAGES) {
        if (alloc->ptr != nullptr) {
            munmap(alloc->ptr, alloc->size);
        }
        alloc->ptr = nullptr;
        alloc->size = 0;
        return;
    }
#endif
    if (alloc->type == MARMOT_ALLOC_HEAP && cpu_allocator_pool_release(alloc->ptr, alloc->size)) {
        alloc->ptr = nullptr;
        alloc->size = 0;
        return;
    }
    free(alloc->ptr);
    alloc->ptr = nullptr;
    alloc->size = 0;
}

static marmot_error_t cpu_allocator_realloc(void *allocator_ctx, marmot_allocation_t *alloc, size_t new_size) {
    (void)allocator_ctx;
    if (alloc == nullptr || alloc->ptr == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    size_t target_size = (new_size == 0) ? 1 : new_size;
    target_size = cpu_allocator_adjust_request_size(target_size, alloc->type);

    if (alloc->type == MARMOT_ALLOC_HEAP) {
        void *new_ptr = realloc(alloc->ptr, target_size);
        if (new_ptr == nullptr) {
            marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "CPU allocator failed to grow allocation");
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
        cpu_allocator_record_free(alloc->size);
        alloc->ptr = new_ptr;
        alloc->size = target_size;
        cpu_allocator_record_alloc(target_size, alloc);
        return MARMOT_SUCCESS;
    }

    marmot_allocation_t replacement = {0};
    marmot_error_t err = cpu_allocator_alloc(allocator_ctx, target_size, alloc->alignment, alloc->type, &replacement);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    size_t to_copy = (target_size < alloc->size) ? target_size : alloc->size;
    memcpy(replacement.ptr, alloc->ptr, to_copy);
    cpu_allocator_free(allocator_ctx, alloc);
    *alloc = replacement;
    return MARMOT_SUCCESS;
}

static size_t cpu_allocator_get_peak_usage(void *allocator_ctx) {
    (void)allocator_ctx;
    pthread_mutex_lock(&cpu_allocator_state.mutex);
    size_t value = cpu_allocator_state.peak_usage;
    pthread_mutex_unlock(&cpu_allocator_state.mutex);
    return value;
}

static size_t cpu_allocator_get_current_usage(void *allocator_ctx) {
    (void)allocator_ctx;
    pthread_mutex_lock(&cpu_allocator_state.mutex);
    size_t value = cpu_allocator_state.current_usage;
    pthread_mutex_unlock(&cpu_allocator_state.mutex);
    return value;
}

const marmot_allocator_ops_t cpu_allocator_ops = {
    .alloc = cpu_allocator_alloc,
    .free = cpu_allocator_free,
    .realloc = cpu_allocator_realloc,
    .get_peak_usage = cpu_allocator_get_peak_usage,
    .get_current_usage = cpu_allocator_get_current_usage,
};

void cpu_allocator_collect_usage(marmot_allocator_usage_t *usage) {
    if (usage == nullptr) {
        return;
    }
    pthread_mutex_lock(&cpu_allocator_state.mutex);
    usage->current_bytes = cpu_allocator_state.current_usage;
    usage->peak_bytes = cpu_allocator_state.peak_usage;
    usage->pooled_bytes = cpu_allocator_state.pooled_bytes;
    usage->active_allocations = cpu_allocator_state.active_allocations;
    usage->peak_allocations = cpu_allocator_state.peak_allocations;
    usage->total_allocations = cpu_allocator_state.total_allocations;
    usage->pool_hits = cpu_allocator_state.pool_hits;
    usage->pool_misses = cpu_allocator_state.pool_misses;
    pthread_mutex_unlock(&cpu_allocator_state.mutex);
}

size_t cpu_allocator_current_usage(void) {
    pthread_mutex_lock(&cpu_allocator_state.mutex);
    size_t value = cpu_allocator_state.current_usage;
    pthread_mutex_unlock(&cpu_allocator_state.mutex);
    return value;
}

size_t cpu_allocator_peak_usage(void) {
    pthread_mutex_lock(&cpu_allocator_state.mutex);
    size_t value = cpu_allocator_state.peak_usage;
    pthread_mutex_unlock(&cpu_allocator_state.mutex);
    return value;
}
