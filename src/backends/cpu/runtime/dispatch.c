#include "marmot/dispatch.h"

#include <stdatomic.h>
#include <stdlib.h>

#if MARMOT_ENABLE_LIBDISPATCH

#include <dispatch/dispatch.h>

extern size_t cpu_default_thread_count(void);

static dispatch_queue_t g_high_queue = nullptr;
static dispatch_queue_t g_normal_queue = nullptr;
static dispatch_queue_t g_low_queue = nullptr;
static atomic_bool g_initialized = false;
static atomic_size_t g_thread_limit = 0;

void cpu_dispatch_set_thread_limit(size_t thread_limit) {
    atomic_store(&g_thread_limit, thread_limit > 0 ? thread_limit : 1);
}

size_t cpu_dispatch_get_thread_limit(void) {
    size_t thread_limit = atomic_load(&g_thread_limit);
    return thread_limit > 0 ? thread_limit : 1;
}

void marmot_dispatch_init(void) {
    if (atomic_exchange(&g_initialized, true)) {
        return;
    }

    if (atomic_load(&g_thread_limit) == 0) {
        cpu_dispatch_set_thread_limit(cpu_default_thread_count());
    }

    dispatch_queue_attr_t high_attr =
        dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INTERACTIVE, 0);
    dispatch_queue_attr_t normal_attr =
        dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0);
    dispatch_queue_attr_t low_attr =
        dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_UTILITY, 0);

    g_high_queue = dispatch_queue_create("com.marmot.compute.high", high_attr);
    g_normal_queue = dispatch_queue_create("com.marmot.compute.normal", normal_attr);
    g_low_queue = dispatch_queue_create("com.marmot.compute.low", low_attr);
}

void marmot_dispatch_destroy(void) {
    if (!atomic_exchange(&g_initialized, false)) {
        return;
    }

    if (g_high_queue != nullptr) {
        dispatch_release(g_high_queue);
        g_high_queue = nullptr;
    }
    if (g_normal_queue != nullptr) {
        dispatch_release(g_normal_queue);
        g_normal_queue = nullptr;
    }
    if (g_low_queue != nullptr) {
        dispatch_release(g_low_queue);
        g_low_queue = nullptr;
    }
}

static dispatch_queue_t marmot_dispatch_get_queue(marmot_dispatch_priority_t priority) {
    switch (priority) {
    case MARMOT_DISPATCH_PRIORITY_HIGH:
        return g_high_queue;
    case MARMOT_DISPATCH_PRIORITY_NORMAL:
        return g_normal_queue;
    case MARMOT_DISPATCH_PRIORITY_LOW:
        return g_low_queue;
    default:
        return g_normal_queue;
    }
}

typedef struct {
    void *context;
    marmot_dispatch_work_fn work;
    size_t total_count;
    size_t num_chunks;
} parallel_for_context_t;

void marmot_dispatch_parallel_for(
    marmot_dispatch_priority_t priority, size_t count, void *context, marmot_dispatch_work_fn work
) {
    if (count == 0) {
        return;
    }

    if (count == 1) {
        work(context, 0);
        return;
    }

    dispatch_queue_t queue = marmot_dispatch_get_queue(priority);
    if (queue == nullptr) {
        marmot_dispatch_init();
        queue = marmot_dispatch_get_queue(priority);
    }

    size_t num_chunks = cpu_dispatch_get_thread_limit();
    if (num_chunks > count) {
        num_chunks = count;
    }

    parallel_for_context_t pctx = {
        .context = context,
        .work = work,
        .total_count = count,
        .num_chunks = num_chunks,
    };

    dispatch_apply(num_chunks, queue, ^(size_t chunk_idx) {
      size_t chunk_size = (pctx.total_count + pctx.num_chunks - 1) / pctx.num_chunks;
      size_t start = chunk_idx * chunk_size;
      size_t end = start + chunk_size;
      if (end > pctx.total_count) {
          end = pctx.total_count;
      }
      for (size_t i = start; i < end; i++) {
          pctx.work(pctx.context, i);
      }
    });
}

typedef struct {
    void *context;
    marmot_dispatch_range_fn work;
    size_t count;
    size_t num_chunks;
} parallel_for_range_context_t;

void marmot_dispatch_parallel_for_range(
    marmot_dispatch_priority_t priority, size_t count, size_t min_chunk_size, void *context,
    marmot_dispatch_range_fn work
) {
    if (count == 0) {
        return;
    }

    dispatch_queue_t queue = marmot_dispatch_get_queue(priority);
    if (queue == nullptr) {
        marmot_dispatch_init();
        queue = marmot_dispatch_get_queue(priority);
    }

    size_t num_chunks = cpu_dispatch_get_thread_limit();
    if (min_chunk_size > 0 && count / min_chunk_size < num_chunks) {
        num_chunks = count / min_chunk_size;
    }
    if (num_chunks > count) {
        num_chunks = count;
    }
    if (num_chunks == 0) {
        num_chunks = 1;
    }

    if (num_chunks == 1) {
        work(context, 0, count);
        return;
    }

    parallel_for_range_context_t pctx = {
        .context = context,
        .work = work,
        .count = count,
        .num_chunks = num_chunks,
    };

    dispatch_apply(num_chunks, queue, ^(size_t chunk_idx) {
      size_t chunk_size = (pctx.count + pctx.num_chunks - 1) / pctx.num_chunks;
      size_t start = chunk_idx * chunk_size;
      size_t end = start + chunk_size;
      if (end > pctx.count) {
          end = pctx.count;
      }
      if (start < end) {
          pctx.work(pctx.context, start, end);
      }
    });
}

typedef struct {
    void *context;
    marmot_dispatch_range_error_fn work;
    size_t count;
    size_t num_chunks;
    _Atomic marmot_error_t first_error;
} parallel_for_range_error_context_t;

marmot_error_t marmot_dispatch_parallel_for_range_with_error(
    marmot_dispatch_priority_t priority, size_t count, size_t min_chunk_size, void *context,
    marmot_dispatch_range_error_fn work
) {
    if (count == 0) {
        return MARMOT_SUCCESS;
    }

    dispatch_queue_t queue = marmot_dispatch_get_queue(priority);
    if (queue == nullptr) {
        marmot_dispatch_init();
        queue = marmot_dispatch_get_queue(priority);
    }

    size_t num_chunks = cpu_dispatch_get_thread_limit();
    if (min_chunk_size > 0 && count / min_chunk_size < num_chunks) {
        num_chunks = count / min_chunk_size;
    }
    if (num_chunks > count) {
        num_chunks = count;
    }
    if (num_chunks == 0) {
        num_chunks = 1;
    }

    if (num_chunks == 1) {
        return work(context, 0, count);
    }

    __block parallel_for_range_error_context_t pctx = {
        .context = context,
        .work = work,
        .count = count,
        .num_chunks = num_chunks,
        .first_error = MARMOT_SUCCESS,
    };

    dispatch_apply(num_chunks, queue, ^(size_t chunk_idx) {
      size_t chunk_size = (pctx.count + pctx.num_chunks - 1) / pctx.num_chunks;
      size_t start = chunk_idx * chunk_size;
      size_t end = start + chunk_size;
      if (end > pctx.count) {
          end = pctx.count;
      }
      if (start < end) {
          marmot_error_t status = pctx.work(pctx.context, start, end);
          if (status != MARMOT_SUCCESS) {
              marmot_error_t expected = MARMOT_SUCCESS;
              atomic_compare_exchange_strong(&pctx.first_error, &expected, status);
          }
      }
    });

    return pctx.first_error;
}

typedef struct {
    void *context;
    marmot_dispatch_async_fn work;
} async_context_t;

void marmot_dispatch_async(marmot_dispatch_priority_t priority, void *context, marmot_dispatch_async_fn work) {
    dispatch_queue_t queue = marmot_dispatch_get_queue(priority);

    async_context_t *actx = malloc(sizeof(async_context_t));
    if (actx == nullptr) {
        work(context);
        return;
    }
    actx->context = context;
    actx->work = work;

    dispatch_async(queue, ^{
      actx->work(actx->context);
      free(actx);
    });
}

void marmot_dispatch_barrier_sync(marmot_dispatch_priority_t priority) {
    dispatch_queue_t queue = marmot_dispatch_get_queue(priority);
    dispatch_barrier_sync(queue, ^{});
}

#else

#include <pthread.h>

extern size_t cpu_default_thread_count(void);

static atomic_size_t g_thread_limit = 0;

void cpu_dispatch_set_thread_limit(size_t thread_limit) {
    atomic_store(&g_thread_limit, thread_limit > 0 ? thread_limit : 1);
}

size_t cpu_dispatch_get_thread_limit(void) {
    size_t thread_limit = atomic_load(&g_thread_limit);
    if (thread_limit == 0) {
        thread_limit = cpu_default_thread_count();
        cpu_dispatch_set_thread_limit(thread_limit);
    }
    return thread_limit;
}

typedef struct {
    void *context;
    marmot_dispatch_work_fn work;
    size_t start;
    size_t end;
} thread_work_t;

static void *thread_worker(void *arg) {
    thread_work_t *tw = (thread_work_t *)arg;
    for (size_t i = tw->start; i < tw->end; ++i) {
        tw->work(tw->context, i);
    }
    return nullptr;
}

void marmot_dispatch_init(void) {}

void marmot_dispatch_destroy(void) {}

void marmot_dispatch_parallel_for(
    marmot_dispatch_priority_t priority, size_t count, void *context, marmot_dispatch_work_fn work
) {
    (void)priority;

    if (count == 0) {
        return;
    }

    if (count == 1) {
        work(context, 0);
        return;
    }

    size_t thread_count = cpu_dispatch_get_thread_limit();
    if (thread_count > count) {
        thread_count = count;
    }

    if (thread_count <= 1) {
        for (size_t i = 0; i < count; ++i) {
            work(context, i);
        }
        return;
    }

    pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    thread_work_t *works = malloc(thread_count * sizeof(thread_work_t));
    if (threads == nullptr || works == nullptr) {
        free(threads);
        free(works);
        for (size_t i = 0; i < count; ++i) {
            work(context, i);
        }
        return;
    }

    size_t per_thread = (count + thread_count - 1) / thread_count;
    size_t launched = 0;

    for (size_t t = 0; t < thread_count; ++t) {
        size_t start = t * per_thread;
        if (start >= count) {
            break;
        }
        size_t end = start + per_thread;
        if (end > count) {
            end = count;
        }

        works[t].context = context;
        works[t].work = work;
        works[t].start = start;
        works[t].end = end;

        if (pthread_create(&threads[t], nullptr, thread_worker, &works[t]) != 0) {
            for (size_t i = start; i < end; ++i) {
                work(context, i);
            }
        } else {
            launched++;
        }
    }

    for (size_t t = 0; t < launched; ++t) {
        pthread_join(threads[t], nullptr);
    }

    free(threads);
    free(works);
}

typedef struct {
    void *context;
    marmot_dispatch_range_fn work;
    size_t start;
    size_t end;
} thread_range_work_t;

static void *thread_range_worker(void *arg) {
    thread_range_work_t *tw = (thread_range_work_t *)arg;
    tw->work(tw->context, tw->start, tw->end);
    return nullptr;
}

void marmot_dispatch_parallel_for_range(
    marmot_dispatch_priority_t priority, size_t count, size_t min_chunk_size, void *context,
    marmot_dispatch_range_fn work
) {
    (void)priority;

    if (count == 0) {
        return;
    }

    size_t thread_count = cpu_dispatch_get_thread_limit();
    if (min_chunk_size > 0 && count / min_chunk_size < thread_count) {
        thread_count = count / min_chunk_size;
    }
    if (thread_count > count) {
        thread_count = count;
    }
    if (thread_count == 0) {
        thread_count = 1;
    }

    if (thread_count <= 1) {
        work(context, 0, count);
        return;
    }

    pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    thread_range_work_t *works = malloc(thread_count * sizeof(thread_range_work_t));
    if (threads == nullptr || works == nullptr) {
        free(threads);
        free(works);
        work(context, 0, count);
        return;
    }

    size_t per_thread = (count + thread_count - 1) / thread_count;
    size_t launched = 0;

    for (size_t t = 0; t < thread_count; ++t) {
        size_t start = t * per_thread;
        if (start >= count) {
            break;
        }
        size_t end = start + per_thread;
        if (end > count) {
            end = count;
        }

        works[t].context = context;
        works[t].work = work;
        works[t].start = start;
        works[t].end = end;

        if (pthread_create(&threads[t], nullptr, thread_range_worker, &works[t]) != 0) {
            work(context, start, end);
        } else {
            launched++;
        }
    }

    for (size_t t = 0; t < launched; ++t) {
        pthread_join(threads[t], nullptr);
    }

    free(threads);
    free(works);
}

typedef struct {
    void *context;
    marmot_dispatch_range_error_fn work;
    size_t start;
    size_t end;
    marmot_error_t status;
} thread_range_error_work_t;

static void *thread_range_error_worker(void *arg) {
    thread_range_error_work_t *tw = (thread_range_error_work_t *)arg;
    tw->status = tw->work(tw->context, tw->start, tw->end);
    return nullptr;
}

marmot_error_t marmot_dispatch_parallel_for_range_with_error(
    marmot_dispatch_priority_t priority, size_t count, size_t min_chunk_size, void *context,
    marmot_dispatch_range_error_fn work
) {
    (void)priority;

    if (count == 0) {
        return MARMOT_SUCCESS;
    }

    size_t thread_count = cpu_dispatch_get_thread_limit();
    if (min_chunk_size > 0 && count / min_chunk_size < thread_count) {
        thread_count = count / min_chunk_size;
    }
    if (thread_count > count) {
        thread_count = count;
    }
    if (thread_count == 0) {
        thread_count = 1;
    }

    if (thread_count <= 1) {
        return work(context, 0, count);
    }

    pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    thread_range_error_work_t *works = malloc(thread_count * sizeof(thread_range_error_work_t));
    if (threads == nullptr || works == nullptr) {
        free(threads);
        free(works);
        return work(context, 0, count);
    }

    size_t per_thread = (count + thread_count - 1) / thread_count;
    size_t launched = 0;

    for (size_t t = 0; t < thread_count; ++t) {
        size_t start = t * per_thread;
        if (start >= count) {
            break;
        }
        size_t end = start + per_thread;
        if (end > count) {
            end = count;
        }

        works[t].context = context;
        works[t].work = work;
        works[t].start = start;
        works[t].end = end;
        works[t].status = MARMOT_SUCCESS;

        if (pthread_create(&threads[t], nullptr, thread_range_error_worker, &works[t]) != 0) {
            works[t].status = work(context, start, end);
        } else {
            launched++;
        }
    }

    for (size_t t = 0; t < launched; ++t) {
        pthread_join(threads[t], nullptr);
    }

    marmot_error_t first_error = MARMOT_SUCCESS;
    for (size_t t = 0; t < thread_count; ++t) {
        if (works[t].status != MARMOT_SUCCESS && first_error == MARMOT_SUCCESS) {
            first_error = works[t].status;
        }
    }

    free(threads);
    free(works);

    return first_error;
}

void marmot_dispatch_async(marmot_dispatch_priority_t priority, void *context, marmot_dispatch_async_fn work) {
    (void)priority;
    work(context);
}

void marmot_dispatch_barrier_sync(marmot_dispatch_priority_t priority) {
    (void)priority;
}

#endif

const char *marmot_dispatch_priority_name(marmot_dispatch_priority_t priority) {
    switch (priority) {
    case MARMOT_DISPATCH_PRIORITY_HIGH:
        return "high";
    case MARMOT_DISPATCH_PRIORITY_NORMAL:
        return "normal";
    case MARMOT_DISPATCH_PRIORITY_LOW:
        return "low";
    default:
        return "unknown";
    }
}
