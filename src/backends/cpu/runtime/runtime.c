#include "marmot/quant_block.h"

#include <stdio.h>

#include <unistd.h>

#include "core/dispatch/kernel_query.h"
#include "cpu_backend_internal.h"
#include "cpu_caps.h"

typedef struct {
    float d;
    float dmin;
    uint8_t scales[8];
    uint8_t mins[8];
    uint8_t qs[MARMOT_QK_K_QS_BYTES];
} cpu_q4_k_row_panel_decoded_block_t;

static_assert(sizeof(cpu_q4_k_row_panel_decoded_block_t) == 152, "decoded Q4_K row-panel block size mismatch");

typedef struct {
    float d;
    int8_t scales[MARMOT_QK_K_VALUES / 16];
    int8_t qs[MARMOT_QK_K_VALUES];
} cpu_q6_k_row_panel_decoded_block_t;

static_assert(sizeof(cpu_q6_k_row_panel_decoded_block_t) == 276, "decoded Q6_K row-panel block size mismatch");

static inline void cpu_q4_k_decode_scales_and_mins(
    const marmot_q4_k_block_t *src, uint8_t *scales_out, uint8_t *mins_out, float *d_out, float *dmin_out
) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];
    memcpy(utmp, src->scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
    memcpy(scales_out, &utmp[0], 8);
    memcpy(mins_out, &utmp[2], 8);
    *d_out = (float)marmot_float16_to_native(src->d);
    *dmin_out = (float)marmot_float16_to_native(src->dmin);
}

void cpu_packed_weight_cache_init(cpu_packed_weight_cache_t *cache) {
    if (cache == nullptr) {
        return;
    }
    memset(cache, 0, sizeof(*cache));
    pthread_mutex_init(&cache->mutex, nullptr);
}

void cpu_packed_weight_cache_destroy(cpu_packed_weight_cache_t *cache) {
    if (cache == nullptr) {
        return;
    }
    pthread_mutex_lock(&cache->mutex);
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        free(cache->entries[i].packed);
        cache->entries[i].packed = nullptr;
        cache->entries[i].capacity_bytes = 0;
        cache->entries[i].valid = false;
        cache->entries[i].sticky = false;
    }
    pthread_mutex_unlock(&cache->mutex);
    pthread_mutex_destroy(&cache->mutex);
}

void cpu_prepacked_weight_store_init(cpu_prepacked_weight_store_t *store) {
    if (store == nullptr) {
        return;
    }
    memset(store, 0, sizeof(*store));
    pthread_mutex_init(&store->mutex, nullptr);
}

void cpu_prepacked_weight_store_destroy(cpu_prepacked_weight_store_t *store) {
    if (store == nullptr) {
        return;
    }
    pthread_mutex_lock(&store->mutex);
    for (size_t i = 0; i < store->count; ++i) {
        free(store->entries[i].packed);
    }
    free(store->entries);
    store->entries = nullptr;
    store->count = 0;
    store->capacity = 0;
    pthread_mutex_unlock(&store->mutex);
    pthread_mutex_destroy(&store->mutex);
}

void cpu_quant_workspace_pool_init(cpu_quant_workspace_pool_t *pool) {
    if (pool == nullptr) {
        return;
    }
    memset(pool, 0, sizeof(*pool));
    for (size_t i = 0; i < CPU_QUANT_WORKSPACE_SLOTS; ++i) {
        pthread_mutex_init(&pool->slots[i].mutex, nullptr);
    }
}

void cpu_quant_workspace_pool_destroy(cpu_quant_workspace_pool_t *pool) {
    if (pool == nullptr) {
        return;
    }
    for (size_t i = 0; i < CPU_QUANT_WORKSPACE_SLOTS; ++i) {
        cpu_quant_workspace_slot_t *slot = &pool->slots[i];
        pthread_mutex_lock(&slot->mutex);
        free(slot->activation_blocks);
        slot->activation_blocks = nullptr;
        slot->activation_blocks_capacity = 0;
        free(slot->activation_panel);
        slot->activation_panel = nullptr;
        slot->activation_panel_capacity = 0;
        pthread_mutex_unlock(&slot->mutex);
        pthread_mutex_destroy(&slot->mutex);
    }
}

cpu_quant_workspace_slot_t *cpu_quant_workspace_acquire(cpu_context_t *ctx) {
    if (ctx == nullptr) {
        return nullptr;
    }
    cpu_quant_workspace_pool_t *pool = &ctx->quant_workspace_pool;
    for (size_t i = 0; i < CPU_QUANT_WORKSPACE_SLOTS; ++i) {
        cpu_quant_workspace_slot_t *slot = &pool->slots[i];
        if (pthread_mutex_trylock(&slot->mutex) == 0) {
            return slot;
        }
    }
    cpu_quant_workspace_slot_t *fallback = &pool->slots[0];
    pthread_mutex_lock(&fallback->mutex);
    return fallback;
}

void cpu_quant_workspace_release(cpu_quant_workspace_slot_t *slot) {
    if (slot == nullptr) {
        return;
    }
    pthread_mutex_unlock(&slot->mutex);
}

bool cpu_quant_workspace_ensure_buffers(
    cpu_quant_workspace_slot_t *slot, size_t activation_blocks_bytes, size_t activation_panel_bytes
) {
    if (slot == nullptr) {
        return false;
    }

    if (slot->activation_blocks_capacity < activation_blocks_bytes) {
        uint8_t *blocks = nullptr;
        if (activation_blocks_bytes != 0) {
            blocks = (uint8_t *)marmot_aligned_alloc(64, activation_blocks_bytes);
            if (blocks == nullptr) {
                return false;
            }
        }
        free(slot->activation_blocks);
        slot->activation_blocks = blocks;
        slot->activation_blocks_capacity = activation_blocks_bytes;
    }

    if (slot->activation_panel_capacity < activation_panel_bytes) {
        uint8_t *panel = nullptr;
        if (activation_panel_bytes != 0) {
            panel = (uint8_t *)marmot_aligned_alloc(64, activation_panel_bytes);
            if (panel == nullptr) {
                return false;
            }
        }
        free(slot->activation_panel);
        slot->activation_panel = panel;
        slot->activation_panel_capacity = activation_panel_bytes;
    }

    return true;
}

static bool cpu_packed_weight_cache_matches(
    const cpu_packed_weight_cache_entry_t *entry, const void *src, size_t bytes, size_t row_bytes, size_t rows,
    cpu_packed_weight_layout_t layout, size_t block_bytes, size_t blocks_per_row, size_t panel_rows
) {
    return entry != nullptr && entry->valid && entry->src == src && entry->bytes == bytes &&
        entry->row_bytes == row_bytes && entry->rows == rows && entry->layout == layout &&
        entry->block_bytes == block_bytes && entry->blocks_per_row == blocks_per_row && entry->panel_rows == panel_rows;
}

static cpu_packed_weight_cache_entry_t *cpu_packed_weight_cache_find_locked(
    cpu_packed_weight_cache_t *cache, const void *src, size_t bytes, size_t row_bytes, size_t rows,
    cpu_packed_weight_layout_t layout, size_t block_bytes, size_t blocks_per_row, size_t panel_rows
) {
    if (cache == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (cpu_packed_weight_cache_matches(
                entry, src, bytes, row_bytes, rows, layout, block_bytes, blocks_per_row, panel_rows
            )) {
            return entry;
        }
    }
    return nullptr;
}

static cpu_packed_weight_cache_entry_t *cpu_packed_weight_cache_find_free_locked(cpu_packed_weight_cache_t *cache) {
    if (cache == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (!entry->valid) {
            return entry;
        }
    }
    return nullptr;
}

static cpu_packed_weight_cache_entry_t *cpu_packed_weight_cache_select_victim_locked(cpu_packed_weight_cache_t *cache) {
    if (cache == nullptr) {
        return nullptr;
    }
    cpu_packed_weight_cache_entry_t *slot = cpu_packed_weight_cache_find_free_locked(cache);
    if (slot != nullptr) {
        return slot;
    }
    uint64_t oldest_stamp = UINT64_MAX;
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (!entry->valid || entry->sticky) {
            continue;
        }
        if (entry->stamp < oldest_stamp) {
            oldest_stamp = entry->stamp;
            slot = entry;
        }
    }
    return slot;
}

static cpu_packed_weight_cache_entry_t *cpu_prepacked_weight_store_find_locked(
    cpu_prepacked_weight_store_t *store, const void *src, size_t bytes, size_t row_bytes, size_t rows,
    cpu_packed_weight_layout_t layout, size_t block_bytes, size_t blocks_per_row, size_t panel_rows
) {
    if (store == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < store->count; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &store->entries[i];
        if (cpu_packed_weight_cache_matches(
                entry, src, bytes, row_bytes, rows, layout, block_bytes, blocks_per_row, panel_rows
            )) {
            return entry;
        }
    }
    return nullptr;
}

static cpu_packed_weight_cache_entry_t *
cpu_prepacked_weight_store_append_locked(cpu_prepacked_weight_store_t *store, size_t packed_bytes) {
    if (store == nullptr || packed_bytes == 0) {
        return nullptr;
    }
    if (store->count == store->capacity) {
        size_t next_capacity = store->capacity == 0 ? 16 : store->capacity * 2;
        cpu_packed_weight_cache_entry_t *entries =
            (cpu_packed_weight_cache_entry_t *)realloc(store->entries, next_capacity * sizeof(*entries));
        if (entries == nullptr) {
            return nullptr;
        }
        memset(entries + store->capacity, 0, (next_capacity - store->capacity) * sizeof(*entries));
        store->entries = entries;
        store->capacity = next_capacity;
    }
    cpu_packed_weight_cache_entry_t *entry = &store->entries[store->count++];
    memset(entry, 0, sizeof(*entry));
    entry->packed = (uint8_t *)marmot_aligned_alloc(64, packed_bytes);
    if (entry->packed == nullptr) {
        store->count--;
        return nullptr;
    }
    entry->capacity_bytes = packed_bytes;
    entry->valid = true;
    entry->sticky = true;
    store->stats.inserts++;
    return entry;
}

static bool cpu_env_profile_packed_weight_cache(void) {
    const char *value = getenv("MARMOT_CPU_PACKED_CACHE_PROFILE");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return value[0] != '0';
}

static void cpu_packed_weight_cache_print_stats(const char *label, const cpu_packed_weight_cache_stats_t *stats) {
    if (label == nullptr || stats == nullptr) {
        return;
    }
    fprintf(
        stderr, "[cpu packed cache] %s exact=%llu/%llu range=%llu/%llu inserts=%llu evictions=%llu full_sticky=%llu\n",
        label, (unsigned long long)stats->exact_hits, (unsigned long long)stats->exact_lookups,
        (unsigned long long)stats->range_hits, (unsigned long long)stats->range_lookups,
        (unsigned long long)stats->inserts, (unsigned long long)stats->evictions,
        (unsigned long long)stats->full_sticky_misses
    );
}

static void cpu_packed_weight_cache_invalidate_ptr_locked(cpu_packed_weight_cache_t *cache, const void *ptr) {
    if (cache == nullptr || ptr == nullptr) {
        return;
    }
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (!entry->valid || entry->src != ptr) {
            continue;
        }
        free(entry->packed);
        memset(entry, 0, sizeof(*entry));
    }
}

static void
cpu_packed_weight_cache_invalidate_range_locked(cpu_packed_weight_cache_t *cache, uintptr_t begin, uintptr_t end) {
    if (cache == nullptr || begin >= end) {
        return;
    }
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (!entry->valid) {
            continue;
        }
        uintptr_t src = (uintptr_t)entry->src;
        if (src < begin || src >= end) {
            continue;
        }
        free(entry->packed);
        memset(entry, 0, sizeof(*entry));
    }
}

static void cpu_prepacked_weight_store_invalidate_ptr_locked(cpu_prepacked_weight_store_t *store, const void *ptr) {
    if (store == nullptr || ptr == nullptr) {
        return;
    }
    size_t dst = 0;
    for (size_t i = 0; i < store->count; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &store->entries[i];
        if (entry->valid && entry->src == ptr) {
            free(entry->packed);
            continue;
        }
        if (dst != i) {
            store->entries[dst] = store->entries[i];
        }
        dst++;
    }
    store->count = dst;
}

static void cpu_prepacked_weight_store_invalidate_range_locked(
    cpu_prepacked_weight_store_t *store, uintptr_t begin, uintptr_t end
) {
    if (store == nullptr || begin >= end) {
        return;
    }
    size_t dst = 0;
    for (size_t i = 0; i < store->count; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &store->entries[i];
        const uintptr_t src = (uintptr_t)entry->src;
        if (entry->valid && src >= begin && src < end) {
            free(entry->packed);
            continue;
        }
        if (dst != i) {
            store->entries[dst] = store->entries[i];
        }
        dst++;
    }
    store->count = dst;
}

static size_t cpu_packed_weight_panel_bytes(size_t block_bytes, size_t blocks_per_row, size_t panel_rows) {
    return block_bytes * blocks_per_row * panel_rows;
}

static size_t
cpu_packed_weight_panel_packed_bytes(size_t rows, size_t block_bytes, size_t blocks_per_row, size_t panel_rows) {
    if (rows == 0 || block_bytes == 0 || blocks_per_row == 0 || panel_rows == 0) {
        return 0;
    }
    const size_t num_panels = (rows + panel_rows - 1) / panel_rows;
    return num_panels * cpu_packed_weight_panel_bytes(block_bytes, blocks_per_row, panel_rows);
}

static void cpu_packed_weight_pack_row_panel(
    uint8_t *dst, const uint8_t *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows
) {
    const size_t panel_bytes = cpu_packed_weight_panel_bytes(block_bytes, blocks_per_row, panel_rows);
    const size_t num_panels = (rows + panel_rows - 1) / panel_rows;
    memset(dst, 0, num_panels * panel_bytes);

    for (size_t panel = 0; panel < num_panels; ++panel) {
        const size_t row0 = panel * panel_rows;
        const size_t rows_this = (row0 + panel_rows <= rows) ? panel_rows : (rows - row0);
        uint8_t *panel_dst = dst + panel * panel_bytes;
        for (size_t block = 0; block < blocks_per_row; ++block) {
            uint8_t *block_dst = panel_dst + block * panel_rows * block_bytes;
            for (size_t row = 0; row < rows_this; ++row) {
                const uint8_t *block_src = src + (row0 + row) * row_bytes + block * block_bytes;
                memcpy(block_dst + row * block_bytes, block_src, block_bytes);
            }
        }
    }
}

static void cpu_packed_weight_pack_q4_k_row_panel_decoded(
    uint8_t *dst, const uint8_t *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
) {
    const size_t block_bytes = sizeof(cpu_q4_k_row_panel_decoded_block_t);
    const size_t panel_bytes = cpu_packed_weight_panel_bytes(block_bytes, blocks_per_row, panel_rows);
    const size_t num_panels = (rows + panel_rows - 1) / panel_rows;
    memset(dst, 0, num_panels * panel_bytes);

    for (size_t panel = 0; panel < num_panels; ++panel) {
        const size_t row0 = panel * panel_rows;
        const size_t rows_this = (row0 + panel_rows <= rows) ? panel_rows : (rows - row0);
        uint8_t *panel_dst = dst + panel * panel_bytes;
        for (size_t block = 0; block < blocks_per_row; ++block) {
            cpu_q4_k_row_panel_decoded_block_t *block_dst =
                (cpu_q4_k_row_panel_decoded_block_t *)(panel_dst + block * panel_rows * block_bytes);
            for (size_t row = 0; row < rows_this; ++row) {
                const marmot_q4_k_block_t *block_src =
                    (const marmot_q4_k_block_t *)(src + (row0 + row) * row_bytes + block * sizeof(marmot_q4_k_block_t));
                cpu_q4_k_row_panel_decoded_block_t *decoded = block_dst + row;
                cpu_q4_k_decode_scales_and_mins(block_src, decoded->scales, decoded->mins, &decoded->d, &decoded->dmin);
                memcpy(decoded->qs, block_src->qs, sizeof(decoded->qs));
            }
        }
    }
}

static void cpu_packed_weight_pack_q6_k_row_panel_decoded(
    uint8_t *dst, const uint8_t *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
) {
    const size_t block_bytes = sizeof(cpu_q6_k_row_panel_decoded_block_t);
    const size_t panel_bytes = cpu_packed_weight_panel_bytes(block_bytes, blocks_per_row, panel_rows);
    const size_t num_panels = (rows + panel_rows - 1) / panel_rows;
    memset(dst, 0, num_panels * panel_bytes);

    for (size_t panel = 0; panel < num_panels; ++panel) {
        const size_t row0 = panel * panel_rows;
        const size_t rows_this = (row0 + panel_rows <= rows) ? panel_rows : (rows - row0);
        uint8_t *panel_dst = dst + panel * panel_bytes;
        for (size_t block = 0; block < blocks_per_row; ++block) {
            cpu_q6_k_row_panel_decoded_block_t *block_dst =
                (cpu_q6_k_row_panel_decoded_block_t *)(panel_dst + block * panel_rows * block_bytes);
            for (size_t row = 0; row < rows_this; ++row) {
                const marmot_q6_k_block_t *block_src =
                    (const marmot_q6_k_block_t *)(src + (row0 + row) * row_bytes + block * sizeof(marmot_q6_k_block_t));
                cpu_q6_k_row_panel_decoded_block_t *decoded = block_dst + row;
                decoded->d = (float)marmot_float16_to_native(block_src->d);
                const uint8_t *ql = block_src->ql;
                const uint8_t *qh = block_src->qh;
                const int8_t *scales = block_src->scales;
                for (size_t n = 0; n < MARMOT_QK_K_VALUES; n += 128) {
                    memcpy(decoded->scales + n / 16, scales, 8);
                    for (size_t l = 0; l < 32; ++l) {
                        const size_t idx = n + l;
                        decoded->qs[idx + 0] = (int8_t)(((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 0x03) << 4))) - 32;
                        decoded->qs[idx + 32] = (int8_t)(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4))) - 32;
                        decoded->qs[idx + 64] = (int8_t)(((ql[l + 0] >> 4) | (((qh[l] >> 4) & 0x03) << 4))) - 32;
                        decoded->qs[idx + 96] = (int8_t)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4))) - 32;
                    }
                    ql += 64;
                    qh += 32;
                    scales += 8;
                }
            }
        }
    }
}

static void cpu_packed_weight_pack_row_panel_wrapper(
    uint8_t *dst, const uint8_t *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows
) {
    cpu_packed_weight_pack_row_panel(dst, src, rows, row_bytes, block_bytes, blocks_per_row, panel_rows);
}

static void cpu_packed_weight_pack_q4_k_row_panel_decoded_wrapper(
    uint8_t *dst, const uint8_t *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows
) {
    (void)block_bytes;
    cpu_packed_weight_pack_q4_k_row_panel_decoded(dst, src, rows, row_bytes, blocks_per_row, panel_rows);
}

static void cpu_packed_weight_pack_q6_k_row_panel_decoded_wrapper(
    uint8_t *dst, const uint8_t *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows
) {
    (void)block_bytes;
    cpu_packed_weight_pack_q6_k_row_panel_decoded(dst, src, rows, row_bytes, blocks_per_row, panel_rows);
}

const uint8_t *
cpu_prepacked_weight_lookup(cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows) {
    if (ctx == nullptr || src == nullptr || bytes == 0) {
        return nullptr;
    }

    cpu_prepacked_weight_store_t *store = &ctx->prepacked_weight_store;
    pthread_mutex_lock(&store->mutex);
    store->stats.exact_lookups++;
    cpu_packed_weight_cache_entry_t *entry = cpu_prepacked_weight_store_find_locked(
        store, src, bytes, row_bytes, rows, CPU_PACKED_WEIGHT_LAYOUT_RAW, 0, 0, 0
    );
    if (entry != nullptr) {
        store->stats.exact_hits++;
    }
    const uint8_t *packed = entry != nullptr ? entry->packed : nullptr;
    pthread_mutex_unlock(&store->mutex);
    return packed;
}

cpu_packed_weight_view_t cpu_prepacked_weight_lookup_packed_range(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows, cpu_packed_weight_layout_t layout
) {
    cpu_packed_weight_view_t view = {
        .data = (const uint8_t *)src,
        .row_bytes = row_bytes,
        .rows = rows,
        .block_bytes = block_bytes,
        .blocks_per_row = blocks_per_row,
        .panel_rows = panel_rows,
        .packed_bytes = row_bytes * rows,
        .layout = CPU_PACKED_WEIGHT_LAYOUT_RAW,
    };
    if (ctx == nullptr || src == nullptr || rows == 0 || row_bytes == 0) {
        return view;
    }

    const size_t source_bytes = row_bytes * rows;
    const uintptr_t begin = (uintptr_t)src;
    if (source_bytes > UINTPTR_MAX - begin) {
        return view;
    }
    const uintptr_t end = begin + source_bytes;

    cpu_prepacked_weight_store_t *store = &ctx->prepacked_weight_store;
    pthread_mutex_lock(&store->mutex);
    store->stats.range_lookups++;
    for (size_t i = 0; i < store->count; ++i) {
        const cpu_packed_weight_cache_entry_t *entry = &store->entries[i];
        if (!entry->valid || entry->layout != layout || entry->src == nullptr || entry->packed == nullptr ||
            entry->row_bytes != row_bytes || (blocks_per_row != 0 && entry->blocks_per_row != blocks_per_row) ||
            (panel_rows != 0 && entry->panel_rows != panel_rows) ||
            (block_bytes != 0 && entry->block_bytes != block_bytes)) {
            continue;
        }
        const uintptr_t entry_begin = (uintptr_t)entry->src;
        if (entry->bytes > UINTPTR_MAX - entry_begin) {
            continue;
        }
        const uintptr_t entry_end = entry_begin + entry->bytes;
        if (begin < entry_begin || end > entry_end) {
            continue;
        }
        const size_t offset_bytes = (size_t)(begin - entry_begin);
        if (offset_bytes % row_bytes != 0) {
            continue;
        }
        const size_t row_offset = offset_bytes / row_bytes;
        if (row_offset + rows > entry->rows) {
            continue;
        }
        if (layout == CPU_PACKED_WEIGHT_LAYOUT_RAW) {
            view.data = entry->packed + offset_bytes;
            view.packed_bytes = source_bytes;
            view.layout = entry->layout;
            store->stats.range_hits++;
            pthread_mutex_unlock(&store->mutex);
            return view;
        }
        if (panel_rows == 0 || row_offset % panel_rows != 0) {
            continue;
        }
        const size_t panel_bytes = cpu_packed_weight_panel_bytes(block_bytes, blocks_per_row, panel_rows);
        const size_t panel_offset = row_offset / panel_rows;
        view.data = entry->packed + panel_offset * panel_bytes;
        view.packed_bytes = cpu_packed_weight_panel_packed_bytes(rows, block_bytes, blocks_per_row, panel_rows);
        view.layout = entry->layout;
        store->stats.range_hits++;
        pthread_mutex_unlock(&store->mutex);
        return view;
    }
    pthread_mutex_unlock(&store->mutex);
    return view;
}

static cpu_packed_weight_view_t cpu_prepacked_weight_store_insert(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows, cpu_packed_weight_layout_t layout,
    void (*pack_fn)(
        uint8_t *dst, const uint8_t *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
        size_t panel_rows
    )
) {
    cpu_packed_weight_view_t view = {
        .data = (const uint8_t *)src,
        .row_bytes = row_bytes,
        .rows = rows,
        .block_bytes = block_bytes,
        .blocks_per_row = blocks_per_row,
        .panel_rows = panel_rows,
        .packed_bytes = row_bytes * rows,
        .layout = CPU_PACKED_WEIGHT_LAYOUT_RAW,
    };
    if (ctx == nullptr || src == nullptr || rows == 0 || row_bytes == 0) {
        return view;
    }

    const size_t source_bytes = row_bytes * rows;
    const size_t packed_bytes = layout == CPU_PACKED_WEIGHT_LAYOUT_RAW
        ? source_bytes
        : cpu_packed_weight_panel_packed_bytes(rows, block_bytes, blocks_per_row, panel_rows);
    if (packed_bytes == 0) {
        return view;
    }

    cpu_prepacked_weight_store_t *store = &ctx->prepacked_weight_store;
    pthread_mutex_lock(&store->mutex);
    store->stats.exact_lookups++;
    cpu_packed_weight_cache_entry_t *entry = cpu_prepacked_weight_store_find_locked(
        store, src, source_bytes, row_bytes, rows, layout, block_bytes, blocks_per_row, panel_rows
    );
    if (entry == nullptr) {
        entry = cpu_prepacked_weight_store_append_locked(store, packed_bytes);
        if (entry != nullptr) {
            entry->src = src;
            entry->bytes = source_bytes;
            entry->row_bytes = row_bytes;
            entry->rows = rows;
            entry->block_bytes = block_bytes;
            entry->blocks_per_row = blocks_per_row;
            entry->panel_rows = panel_rows;
            entry->packed_bytes = packed_bytes;
            entry->layout = layout;
            if (layout == CPU_PACKED_WEIGHT_LAYOUT_RAW) {
                memcpy(entry->packed, src, packed_bytes);
            } else if (pack_fn != nullptr) {
                pack_fn(entry->packed, (const uint8_t *)src, rows, row_bytes, block_bytes, blocks_per_row, panel_rows);
            }
        }
    } else {
        store->stats.exact_hits++;
    }

    if (entry != nullptr && entry->packed != nullptr) {
        view.data = entry->packed;
        view.packed_bytes = entry->packed_bytes;
        view.layout = entry->layout;
    }
    pthread_mutex_unlock(&store->mutex);
    return view;
}

bool cpu_prepacked_weight_store_put_raw(
    cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows
) {
    (void)bytes;
    const cpu_packed_weight_view_t view =
        cpu_prepacked_weight_store_insert(ctx, src, rows, row_bytes, 0, 0, 0, CPU_PACKED_WEIGHT_LAYOUT_RAW, nullptr);
    return view.data != nullptr && view.data != (const uint8_t *)src;
}

cpu_packed_weight_view_t cpu_prepacked_weight_store_put_row_panel(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows
) {
    return cpu_prepacked_weight_store_insert(
        ctx, src, rows, row_bytes, block_bytes, blocks_per_row, panel_rows, CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL,
        cpu_packed_weight_pack_row_panel_wrapper
    );
}

cpu_packed_weight_view_t cpu_prepacked_weight_store_put_q4_k_row_panel_decoded(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
) {
    return cpu_prepacked_weight_store_insert(
        ctx, src, rows, row_bytes, sizeof(cpu_q4_k_row_panel_decoded_block_t), blocks_per_row, panel_rows,
        CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED, cpu_packed_weight_pack_q4_k_row_panel_decoded_wrapper
    );
}

cpu_packed_weight_view_t cpu_prepacked_weight_store_put_q6_k_row_panel_decoded(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
) {
    return cpu_prepacked_weight_store_insert(
        ctx, src, rows, row_bytes, sizeof(cpu_q6_k_row_panel_decoded_block_t), blocks_per_row, panel_rows,
        CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED, cpu_packed_weight_pack_q6_k_row_panel_decoded_wrapper
    );
}

const uint8_t *
cpu_packed_weight_cache_get(cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows) {
    if (ctx == nullptr || src == nullptr || bytes == 0) {
        return (const uint8_t *)src;
    }

    cpu_packed_weight_cache_t *cache = &ctx->packed_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cache->stats.exact_lookups++;

    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (cpu_packed_weight_cache_matches(
                entry, src, bytes, row_bytes, rows, CPU_PACKED_WEIGHT_LAYOUT_RAW, 0, 0, 0
            )) {
            cache->stats.exact_hits++;
            entry->stamp = ++cache->stamp;
            const uint8_t *packed = entry->packed;
            pthread_mutex_unlock(&cache->mutex);
            return packed != nullptr ? packed : (const uint8_t *)src;
        }
    }

    cpu_packed_weight_cache_entry_t *slot = cpu_packed_weight_cache_select_victim_locked(cache);

    if (slot == nullptr) {
        cache->stats.full_sticky_misses++;
        pthread_mutex_unlock(&cache->mutex);
        return (const uint8_t *)src;
    }
    const bool evicting = slot->valid;

    if (slot->capacity_bytes < bytes) {
        uint8_t *packed = (uint8_t *)marmot_aligned_alloc(64, bytes);
        if (packed == nullptr) {
            pthread_mutex_unlock(&cache->mutex);
            return (const uint8_t *)src;
        }
        free(slot->packed);
        slot->packed = packed;
        slot->capacity_bytes = bytes;
    }

    memcpy(slot->packed, src, bytes);
    slot->src = src;
    slot->bytes = bytes;
    slot->row_bytes = row_bytes;
    slot->rows = rows;
    slot->block_bytes = 0;
    slot->blocks_per_row = 0;
    slot->panel_rows = 0;
    slot->packed_bytes = bytes;
    slot->layout = CPU_PACKED_WEIGHT_LAYOUT_RAW;
    slot->stamp = ++cache->stamp;
    slot->valid = true;
    slot->sticky = false;
    cache->stats.inserts++;
    if (evicting) {
        cache->stats.evictions++;
    }
    const uint8_t *packed = slot->packed;
    pthread_mutex_unlock(&cache->mutex);
    return packed != nullptr ? packed : (const uint8_t *)src;
}

const uint8_t *
cpu_pinned_weight_cache_lookup(cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows) {
    if (ctx == nullptr || src == nullptr || bytes == 0) {
        return nullptr;
    }

    cpu_packed_weight_cache_t *cache = &ctx->pinned_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cache->stats.exact_lookups++;
    cpu_packed_weight_cache_entry_t *entry =
        cpu_packed_weight_cache_find_locked(cache, src, bytes, row_bytes, rows, CPU_PACKED_WEIGHT_LAYOUT_RAW, 0, 0, 0);
    if (entry != nullptr) {
        cache->stats.exact_hits++;
    }
    const uint8_t *packed = entry != nullptr ? entry->packed : nullptr;
    pthread_mutex_unlock(&cache->mutex);
    return packed;
}

const uint8_t *cpu_pinned_weight_cache_lookup_range(cpu_context_t *ctx, const void *src, size_t bytes) {
    if (ctx == nullptr || src == nullptr || bytes == 0) {
        return nullptr;
    }

    const uintptr_t begin = (uintptr_t)src;
    if (bytes > UINTPTR_MAX - begin) {
        return nullptr;
    }
    const uintptr_t end = begin + bytes;

    cpu_packed_weight_cache_t *cache = &ctx->pinned_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cache->stats.range_lookups++;
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        const cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (!entry->valid || entry->layout != CPU_PACKED_WEIGHT_LAYOUT_RAW || entry->src == nullptr ||
            entry->packed == nullptr) {
            continue;
        }
        const uintptr_t entry_begin = (uintptr_t)entry->src;
        if (entry->bytes > UINTPTR_MAX - entry_begin) {
            continue;
        }
        const uintptr_t entry_end = entry_begin + entry->bytes;
        if (begin < entry_begin || end > entry_end) {
            continue;
        }
        const uint8_t *packed = entry->packed + (begin - entry_begin);
        cache->stats.range_hits++;
        pthread_mutex_unlock(&cache->mutex);
        return packed;
    }
    pthread_mutex_unlock(&cache->mutex);
    return nullptr;
}

bool cpu_pinned_weight_cache_pin(cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows) {
    if (ctx == nullptr || src == nullptr || bytes == 0) {
        return false;
    }

    cpu_packed_weight_cache_t *cache = &ctx->pinned_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cache->stats.exact_lookups++;

    cpu_packed_weight_cache_entry_t *slot =
        cpu_packed_weight_cache_find_locked(cache, src, bytes, row_bytes, rows, CPU_PACKED_WEIGHT_LAYOUT_RAW, 0, 0, 0);
    if (slot != nullptr) {
        cache->stats.exact_hits++;
        pthread_mutex_unlock(&cache->mutex);
        return true;
    }
    slot = cpu_packed_weight_cache_find_free_locked(cache);
    if (slot == nullptr) {
        pthread_mutex_unlock(&cache->mutex);
        return false;
    }

    if (slot->capacity_bytes < bytes) {
        uint8_t *packed = (uint8_t *)marmot_aligned_alloc(64, bytes);
        if (packed == nullptr) {
            pthread_mutex_unlock(&cache->mutex);
            return false;
        }
        free(slot->packed);
        slot->packed = packed;
        slot->capacity_bytes = bytes;
    }

    memcpy(slot->packed, src, bytes);
    slot->src = src;
    slot->bytes = bytes;
    slot->row_bytes = row_bytes;
    slot->rows = rows;
    slot->block_bytes = 0;
    slot->blocks_per_row = 0;
    slot->panel_rows = 0;
    slot->packed_bytes = bytes;
    slot->layout = CPU_PACKED_WEIGHT_LAYOUT_RAW;
    slot->stamp = 0;
    slot->valid = true;
    slot->sticky = true;
    cache->stats.inserts++;
    pthread_mutex_unlock(&cache->mutex);
    return true;
}

cpu_packed_weight_view_t cpu_packed_weight_cache_get_row_panel(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows
) {
    cpu_packed_weight_view_t view = (cpu_packed_weight_view_t){
        .data = (const uint8_t *)src,
        .row_bytes = row_bytes,
        .rows = rows,
        .block_bytes = block_bytes,
        .blocks_per_row = blocks_per_row,
        .panel_rows = panel_rows,
        .packed_bytes = row_bytes * rows,
        .layout = CPU_PACKED_WEIGHT_LAYOUT_RAW,
    };
    if (ctx == nullptr || src == nullptr || rows == 0 || row_bytes == 0 || block_bytes == 0 || blocks_per_row == 0 ||
        panel_rows == 0) {
        return view;
    }

    const size_t source_bytes = row_bytes * rows;
    const size_t packed_bytes = cpu_packed_weight_panel_packed_bytes(rows, block_bytes, blocks_per_row, panel_rows);
    if (packed_bytes == 0) {
        return view;
    }

    cpu_packed_weight_cache_t *cache = &ctx->packed_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cache->stats.exact_lookups++;

    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (cpu_packed_weight_cache_matches(
                entry, src, source_bytes, row_bytes, rows, CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL, block_bytes,
                blocks_per_row, panel_rows
            )) {
            cache->stats.exact_hits++;
            entry->stamp = ++cache->stamp;
            view.data = entry->packed;
            view.packed_bytes = entry->packed_bytes;
            view.layout = entry->layout;
            pthread_mutex_unlock(&cache->mutex);
            return view;
        }
    }

    cpu_packed_weight_cache_entry_t *slot = cpu_packed_weight_cache_select_victim_locked(cache);

    if (slot == nullptr) {
        cache->stats.full_sticky_misses++;
        pthread_mutex_unlock(&cache->mutex);
        return view;
    }
    const bool evicting = slot->valid;

    if (slot->capacity_bytes < packed_bytes) {
        uint8_t *packed = (uint8_t *)marmot_aligned_alloc(64, packed_bytes);
        if (packed == nullptr) {
            pthread_mutex_unlock(&cache->mutex);
            return view;
        }
        free(slot->packed);
        slot->packed = packed;
        slot->capacity_bytes = packed_bytes;
    }

    cpu_packed_weight_pack_row_panel(
        slot->packed, (const uint8_t *)src, rows, row_bytes, block_bytes, blocks_per_row, panel_rows
    );
    slot->src = src;
    slot->bytes = source_bytes;
    slot->row_bytes = row_bytes;
    slot->rows = rows;
    slot->block_bytes = block_bytes;
    slot->blocks_per_row = blocks_per_row;
    slot->panel_rows = panel_rows;
    slot->packed_bytes = packed_bytes;
    slot->layout = CPU_PACKED_WEIGHT_LAYOUT_ROW_PANEL;
    slot->stamp = ++cache->stamp;
    slot->valid = true;
    slot->sticky = false;
    cache->stats.inserts++;
    if (evicting) {
        cache->stats.evictions++;
    }

    view.data = slot->packed;
    view.packed_bytes = packed_bytes;
    view.layout = slot->layout;
    pthread_mutex_unlock(&cache->mutex);
    return view;
}

cpu_packed_weight_view_t cpu_packed_weight_cache_get_q4_k_row_panel_decoded(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
) {
    cpu_packed_weight_view_t view = (cpu_packed_weight_view_t){
        .data = (const uint8_t *)src,
        .row_bytes = row_bytes,
        .rows = rows,
        .block_bytes = sizeof(cpu_q4_k_row_panel_decoded_block_t),
        .blocks_per_row = blocks_per_row,
        .panel_rows = panel_rows,
        .packed_bytes = sizeof(cpu_q4_k_row_panel_decoded_block_t) * blocks_per_row * rows,
        .layout = CPU_PACKED_WEIGHT_LAYOUT_RAW,
    };
    if (ctx == nullptr || src == nullptr || rows == 0 || row_bytes == 0 || blocks_per_row == 0 || panel_rows == 0) {
        return view;
    }

    const size_t source_bytes = row_bytes * rows;
    const size_t block_bytes = sizeof(cpu_q4_k_row_panel_decoded_block_t);
    const size_t packed_bytes = cpu_packed_weight_panel_packed_bytes(rows, block_bytes, blocks_per_row, panel_rows);
    if (packed_bytes == 0) {
        return view;
    }

    cpu_packed_weight_cache_t *cache = &ctx->packed_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cache->stats.exact_lookups++;

    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (cpu_packed_weight_cache_matches(
                entry, src, source_bytes, row_bytes, rows, CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED, block_bytes,
                blocks_per_row, panel_rows
            )) {
            cache->stats.exact_hits++;
            entry->stamp = ++cache->stamp;
            view.data = entry->packed;
            view.packed_bytes = entry->packed_bytes;
            view.layout = entry->layout;
            pthread_mutex_unlock(&cache->mutex);
            return view;
        }
    }

    cpu_packed_weight_cache_entry_t *slot = cpu_packed_weight_cache_select_victim_locked(cache);

    if (slot == nullptr) {
        cache->stats.full_sticky_misses++;
        pthread_mutex_unlock(&cache->mutex);
        return view;
    }
    const bool evicting = slot->valid;

    if (slot->capacity_bytes < packed_bytes) {
        uint8_t *packed = (uint8_t *)marmot_aligned_alloc(64, packed_bytes);
        if (packed == nullptr) {
            pthread_mutex_unlock(&cache->mutex);
            return view;
        }
        free(slot->packed);
        slot->packed = packed;
        slot->capacity_bytes = packed_bytes;
    }

    cpu_packed_weight_pack_q4_k_row_panel_decoded(
        slot->packed, (const uint8_t *)src, rows, row_bytes, blocks_per_row, panel_rows
    );
    slot->src = src;
    slot->bytes = source_bytes;
    slot->row_bytes = row_bytes;
    slot->rows = rows;
    slot->block_bytes = block_bytes;
    slot->blocks_per_row = blocks_per_row;
    slot->panel_rows = panel_rows;
    slot->packed_bytes = packed_bytes;
    slot->layout = CPU_PACKED_WEIGHT_LAYOUT_Q4_K_ROW_PANEL_DECODED;
    slot->stamp = ++cache->stamp;
    slot->valid = true;
    slot->sticky = false;
    cache->stats.inserts++;
    if (evicting) {
        cache->stats.evictions++;
    }

    view.data = slot->packed;
    view.packed_bytes = packed_bytes;
    view.layout = slot->layout;
    pthread_mutex_unlock(&cache->mutex);
    return view;
}

cpu_packed_weight_view_t cpu_packed_weight_cache_get_q6_k_row_panel_decoded(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t blocks_per_row, size_t panel_rows
) {
    cpu_packed_weight_view_t view = (cpu_packed_weight_view_t){
        .data = (const uint8_t *)src,
        .row_bytes = row_bytes,
        .rows = rows,
        .block_bytes = sizeof(cpu_q6_k_row_panel_decoded_block_t),
        .blocks_per_row = blocks_per_row,
        .panel_rows = panel_rows,
        .packed_bytes = sizeof(cpu_q6_k_row_panel_decoded_block_t) * blocks_per_row * rows,
        .layout = CPU_PACKED_WEIGHT_LAYOUT_RAW,
    };
    if (ctx == nullptr || src == nullptr || rows == 0 || row_bytes == 0 || blocks_per_row == 0 || panel_rows == 0) {
        return view;
    }

    const size_t source_bytes = row_bytes * rows;
    const size_t block_bytes = sizeof(cpu_q6_k_row_panel_decoded_block_t);
    const size_t packed_bytes = cpu_packed_weight_panel_packed_bytes(rows, block_bytes, blocks_per_row, panel_rows);
    if (packed_bytes == 0) {
        return view;
    }

    cpu_packed_weight_cache_t *cache = &ctx->packed_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cache->stats.exact_lookups++;

    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (cpu_packed_weight_cache_matches(
                entry, src, source_bytes, row_bytes, rows, CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED, block_bytes,
                blocks_per_row, panel_rows
            )) {
            cache->stats.exact_hits++;
            entry->stamp = ++cache->stamp;
            view.data = entry->packed;
            view.packed_bytes = entry->packed_bytes;
            view.layout = entry->layout;
            pthread_mutex_unlock(&cache->mutex);
            return view;
        }
    }

    cpu_packed_weight_cache_entry_t *slot = cpu_packed_weight_cache_select_victim_locked(cache);

    if (slot == nullptr) {
        cache->stats.full_sticky_misses++;
        pthread_mutex_unlock(&cache->mutex);
        return view;
    }
    const bool evicting = slot->valid;

    if (slot->capacity_bytes < packed_bytes) {
        uint8_t *packed = (uint8_t *)marmot_aligned_alloc(64, packed_bytes);
        if (packed == nullptr) {
            pthread_mutex_unlock(&cache->mutex);
            return view;
        }
        free(slot->packed);
        slot->packed = packed;
        slot->capacity_bytes = packed_bytes;
    }

    cpu_packed_weight_pack_q6_k_row_panel_decoded(
        slot->packed, (const uint8_t *)src, rows, row_bytes, blocks_per_row, panel_rows
    );
    slot->src = src;
    slot->bytes = source_bytes;
    slot->row_bytes = row_bytes;
    slot->rows = rows;
    slot->block_bytes = block_bytes;
    slot->blocks_per_row = blocks_per_row;
    slot->panel_rows = panel_rows;
    slot->packed_bytes = packed_bytes;
    slot->layout = CPU_PACKED_WEIGHT_LAYOUT_Q6_K_ROW_PANEL_DECODED;
    slot->stamp = ++cache->stamp;
    slot->valid = true;
    slot->sticky = false;
    cache->stats.inserts++;
    if (evicting) {
        cache->stats.evictions++;
    }

    view.data = slot->packed;
    view.packed_bytes = packed_bytes;
    view.layout = slot->layout;
    pthread_mutex_unlock(&cache->mutex);
    return view;
}

cpu_packed_weight_view_t cpu_packed_weight_cache_lookup_packed_range(
    cpu_context_t *ctx, const void *src, size_t rows, size_t row_bytes, size_t block_bytes, size_t blocks_per_row,
    size_t panel_rows, cpu_packed_weight_layout_t layout
) {
    cpu_packed_weight_view_t view = {
        .data = (const uint8_t *)src,
        .row_bytes = row_bytes,
        .rows = rows,
        .block_bytes = block_bytes,
        .blocks_per_row = blocks_per_row,
        .panel_rows = panel_rows,
        .packed_bytes = row_bytes * rows,
        .layout = CPU_PACKED_WEIGHT_LAYOUT_RAW,
    };
    if (ctx == nullptr || src == nullptr || rows == 0 || row_bytes == 0) {
        return view;
    }

    const size_t source_bytes = row_bytes * rows;
    const uintptr_t begin = (uintptr_t)src;
    if (source_bytes > UINTPTR_MAX - begin) {
        return view;
    }
    const uintptr_t end = begin + source_bytes;

    cpu_packed_weight_cache_t *cache = &ctx->packed_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cache->stats.range_lookups++;
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        const cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (!entry->valid || entry->layout != layout || entry->src == nullptr || entry->packed == nullptr ||
            entry->row_bytes != row_bytes || (blocks_per_row != 0 && entry->blocks_per_row != blocks_per_row) ||
            (panel_rows != 0 && entry->panel_rows != panel_rows) ||
            (block_bytes != 0 && entry->block_bytes != block_bytes)) {
            continue;
        }
        const uintptr_t entry_begin = (uintptr_t)entry->src;
        if (entry->bytes > UINTPTR_MAX - entry_begin) {
            continue;
        }
        const uintptr_t entry_end = entry_begin + entry->bytes;
        if (begin < entry_begin || end > entry_end) {
            continue;
        }
        const size_t offset_bytes = (size_t)(begin - entry_begin);
        if (offset_bytes % row_bytes != 0) {
            continue;
        }
        const size_t row_offset = offset_bytes / row_bytes;
        if (row_offset + rows > entry->rows) {
            continue;
        }
        if (layout == CPU_PACKED_WEIGHT_LAYOUT_RAW) {
            view.data = entry->packed + offset_bytes;
            view.packed_bytes = source_bytes;
            view.layout = entry->layout;
            pthread_mutex_unlock(&cache->mutex);
            return view;
        }

        if (panel_rows == 0 || row_offset % panel_rows != 0) {
            continue;
        }
        const size_t panel_bytes = cpu_packed_weight_panel_bytes(block_bytes, blocks_per_row, panel_rows);
        const size_t panel_offset = row_offset / panel_rows;
        view.data = entry->packed + panel_offset * panel_bytes;
        view.packed_bytes = cpu_packed_weight_panel_packed_bytes(rows, block_bytes, blocks_per_row, panel_rows);
        view.layout = entry->layout;
        cache->stats.range_hits++;
        pthread_mutex_unlock(&cache->mutex);
        return view;
    }
    pthread_mutex_unlock(&cache->mutex);
    return view;
}

void cpu_packed_weight_cache_mark_sticky(
    cpu_context_t *ctx, const void *src, size_t bytes, size_t row_bytes, size_t rows, cpu_packed_weight_layout_t layout,
    size_t block_bytes, size_t blocks_per_row, size_t panel_rows
) {
    if (ctx == nullptr || src == nullptr || bytes == 0) {
        return;
    }
    cpu_packed_weight_cache_t *cache = &ctx->packed_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    for (size_t i = 0; i < CPU_PACKED_WEIGHT_CACHE_SLOTS; ++i) {
        cpu_packed_weight_cache_entry_t *entry = &cache->entries[i];
        if (!entry->valid || entry->src != src || entry->bytes != bytes || entry->row_bytes != row_bytes ||
            entry->rows != rows || entry->layout != layout) {
            continue;
        }
        if ((block_bytes != 0 && entry->block_bytes != block_bytes) ||
            (blocks_per_row != 0 && entry->blocks_per_row != blocks_per_row) ||
            (panel_rows != 0 && entry->panel_rows != panel_rows)) {
            continue;
        }
        entry->sticky = true;
        break;
    }
    pthread_mutex_unlock(&cache->mutex);
}

void cpu_packed_weight_cache_invalidate_ptr(cpu_context_t *ctx, const void *ptr) {
    if (ctx == nullptr || ptr == nullptr) {
        return;
    }
    cpu_prepacked_weight_store_t *store = &ctx->prepacked_weight_store;
    pthread_mutex_lock(&store->mutex);
    cpu_prepacked_weight_store_invalidate_ptr_locked(store, ptr);
    pthread_mutex_unlock(&store->mutex);

    cpu_packed_weight_cache_t *cache = &ctx->packed_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cpu_packed_weight_cache_invalidate_ptr_locked(cache, ptr);
    pthread_mutex_unlock(&cache->mutex);

    cache = &ctx->pinned_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cpu_packed_weight_cache_invalidate_ptr_locked(cache, ptr);
    pthread_mutex_unlock(&cache->mutex);
}

void cpu_packed_weight_cache_invalidate_range(cpu_context_t *ctx, const void *start, size_t length) {
    if (ctx == nullptr || start == nullptr || length == 0) {
        return;
    }
    uintptr_t begin = (uintptr_t)start;
    if (length > UINTPTR_MAX - begin) {
        return;
    }
    uintptr_t end = begin + length;

    cpu_prepacked_weight_store_t *store = &ctx->prepacked_weight_store;
    pthread_mutex_lock(&store->mutex);
    cpu_prepacked_weight_store_invalidate_range_locked(store, begin, end);
    pthread_mutex_unlock(&store->mutex);

    cpu_packed_weight_cache_t *cache = &ctx->packed_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cpu_packed_weight_cache_invalidate_range_locked(cache, begin, end);
    pthread_mutex_unlock(&cache->mutex);

    cache = &ctx->pinned_weight_cache;
    pthread_mutex_lock(&cache->mutex);
    cpu_packed_weight_cache_invalidate_range_locked(cache, begin, end);
    pthread_mutex_unlock(&cache->mutex);
}

void cpu_on_host_ptr_freed(void *device_ctx, const void *ptr) {
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    cpu_packed_weight_cache_invalidate_ptr(ctx, ptr);
}

void cpu_on_host_range_freed(void *device_ctx, const void *start, size_t length) {
    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    cpu_packed_weight_cache_invalidate_range(ctx, start, length);
}

size_t cpu_default_thread_count(void) {
    // Cache the result since CPU topology detection can be expensive
    // (especially Intel hybrid detection which requires thread affinity)
    static size_t cached_count = 0;
    if (cached_count == 0) {
        marmot_device_caps_t caps = marmot_cpu_detect_capabilities();
        cached_count = marmot_cpu_optimal_thread_count(&caps);
    }
    return cached_count;
}

static cpu_capabilities_t cpu_capabilities_clamp(const cpu_capabilities_t *requested) {
    const cpu_capabilities_t *compiled = cpu_compiled_capabilities();
    cpu_capabilities_t effective = *compiled;
    if (requested != nullptr) {
        effective.has_neon = requested->has_neon && compiled->has_neon;
        effective.has_avx2 = requested->has_avx2 && compiled->has_avx2;
        effective.has_f16c = requested->has_f16c && compiled->has_f16c;
        effective.has_accelerate = requested->has_accelerate && compiled->has_accelerate;
    }
    return effective;
}

static void cpu_context_apply_capabilities(cpu_context_t *ctx, const cpu_capabilities_t *requested) {
    if (ctx == nullptr) {
        return;
    }
    cpu_capabilities_t effective = cpu_capabilities_clamp(requested);
    ctx->runtime_caps = effective;
}

void cpu_context_use_compiled_capabilities(const void *device_ctx) {
    cpu_context_apply_capabilities((cpu_context_t *)device_ctx, cpu_compiled_capabilities());
}

void cpu_context_force_scalar(const void *device_ctx) {
    cpu_capabilities_t scalar_caps = {
        .has_neon = false,
        .has_avx2 = false,
        .has_f16c = false,
        .has_accelerate = false,
    };
    cpu_context_apply_capabilities((cpu_context_t *)device_ctx, &scalar_caps);
}

static bool cpu_quant_env_force_q8(void) {
    const char *env = getenv("MARMOT_QUANT_CPU_ACT");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }
    return env[0] == 'q' || env[0] == 'Q';
}

static bool cpu_env_thread_count_override(size_t *out_thread_count) {
    const char *env = getenv("MARMOT_NUM_THREADS");
    if (env == nullptr || env[0] == '\0') {
        return false;
    }

    long value = strtol(env, nullptr, 10);
    if (value <= 0 || value > 64) {
        return false;
    }

    if (out_thread_count != nullptr) {
        *out_thread_count = (size_t)value;
    }
    return true;
}

bool marmot_cpu_default_preferences(const marmot_device_caps_t *caps, marmot_backend_preferences_t *out) {
    if (out == nullptr) {
        return false;
    }
    (void)caps;
    *out = (marmot_backend_preferences_t){
        .policy =
            {
                .embedding_quant_output_dtype = MARMOT_DTYPE_FLOAT32,
                .quant_activation_mode = MARMOT_QUANT_ACTIVATION_AUTO,
                .variant_flags_mask = UINT32_MAX,
                .embedding_prefer_gpu_private = false,
                .embedding_allow_quant_decode_on_the_fly = true,
                .matmul_requires_temp_tensors = false,
                .matmul_prefer_packed_weights = false,
            },
        .routing_policy = MARMOT_ROUTING_ALWAYS_CPU,
    };
    return true;
}

marmot_error_t cpu_init(void **device_ctx) {
    cpu_context_t *ctx = calloc(1, sizeof(cpu_context_t));
    if (ctx == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    size_t thread_count = cpu_default_thread_count();
    ctx->thread_count_explicit = cpu_env_thread_count_override(&thread_count);
    ctx->num_threads = thread_count;
    ctx->quant_activation_mode = MARMOT_QUANT_ACTIVATION_AUTO;
    ctx->force_q8_activations = cpu_quant_env_force_q8();
    ctx->profile_packed_weight_cache = cpu_env_profile_packed_weight_cache();
    marmot_rope_freq_cache_init(&ctx->rope_cache);
    cpu_rope_sincos_cache_init(&ctx->rope_sincos_cache);

    cpu_context_use_compiled_capabilities(ctx);
    ctx->allocator_ops = marmot_get_allocator(MARMOT_BACKEND_CPU);
    pthread_mutex_init(&ctx->allocator_tracker.mutex, nullptr);
    ctx->allocator_tracker.head = nullptr;
    cpu_packed_weight_cache_init(&ctx->pinned_weight_cache);
    cpu_prepacked_weight_store_init(&ctx->prepacked_weight_store);
    cpu_packed_weight_cache_init(&ctx->packed_weight_cache);
    cpu_quant_workspace_pool_init(&ctx->quant_workspace_pool);
    cpu_dispatch_set_thread_limit(ctx->num_threads);

    // Initialize GEMM scratch buffer pool
    marmot_neon_scratch_pool_init(&ctx->neon_scratch_pool, ctx->num_threads);

    void *probe = nullptr;
    if (cpu_alloc(ctx, 1, &probe) == MARMOT_SUCCESS) {
        cpu_free(ctx, probe);
    }

    ctx->initialized = 1;
    *device_ctx = ctx;
    return MARMOT_SUCCESS;
}

void cpu_destroy(const void *device_ctx) {
    if (device_ctx == nullptr) {
        return;
    }

    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (ctx->allocator_ops == nullptr) {
        ctx->allocator_ops = marmot_get_allocator(MARMOT_BACKEND_CPU);
    }

    pthread_mutex_lock(&ctx->allocator_tracker.mutex);
    cpu_allocation_entry_t *entry = ctx->allocator_tracker.head;
    while (entry != nullptr) {
        cpu_allocation_entry_t *next = entry->next;
        if (ctx->allocator_ops != nullptr) {
            ctx->allocator_ops->free(ctx, &entry->info);
        } else {
            free(entry->info.ptr);
        }
        free(entry);
        entry = next;
    }
    pthread_mutex_unlock(&ctx->allocator_tracker.mutex);
    pthread_mutex_destroy(&ctx->allocator_tracker.mutex);

    marmot_rope_freq_cache_destroy(&ctx->rope_cache);
    cpu_rope_sincos_cache_destroy(&ctx->rope_sincos_cache);

    // Free RoPE scratch buffer
    free(ctx->rope_positions_scratch);
    ctx->rope_positions_scratch = nullptr;
    ctx->rope_positions_capacity = 0;

    // Free GEMM scratch buffer pool
    marmot_neon_scratch_pool_destroy(&ctx->neon_scratch_pool);
    if (ctx->profile_packed_weight_cache) {
        cpu_packed_weight_cache_print_stats("pinned", &ctx->pinned_weight_cache.stats);
        cpu_packed_weight_cache_print_stats("prepacked", &ctx->prepacked_weight_store.stats);
        cpu_packed_weight_cache_print_stats("packed", &ctx->packed_weight_cache.stats);
    }
    cpu_packed_weight_cache_destroy(&ctx->pinned_weight_cache);
    cpu_prepacked_weight_store_destroy(&ctx->prepacked_weight_store);
    cpu_packed_weight_cache_destroy(&ctx->packed_weight_cache);
    cpu_quant_workspace_pool_destroy(&ctx->quant_workspace_pool);

    free(ctx);
}

marmot_error_t cpu_context_set_num_threads(void *device_ctx, size_t num_threads, bool explicit_override) {
    if (device_ctx == nullptr || num_threads == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    cpu_context_t *ctx = (cpu_context_t *)device_ctx;
    if (ctx->num_threads != num_threads) {
        marmot_neon_scratch_pool_destroy(&ctx->neon_scratch_pool);
        marmot_neon_scratch_pool_init(&ctx->neon_scratch_pool, num_threads);
        ctx->num_threads = num_threads;
    }
    if (explicit_override) {
        ctx->thread_count_explicit = true;
    }
    cpu_dispatch_set_thread_limit(ctx->num_threads);
    return MARMOT_SUCCESS;
}

size_t cpu_context_get_num_threads(const void *device_ctx) {
    const cpu_context_t *ctx = (const cpu_context_t *)device_ctx;
    return ctx != nullptr ? ctx->num_threads : 0;
}

bool cpu_context_thread_count_is_explicit(const void *device_ctx) {
    const cpu_context_t *ctx = (const cpu_context_t *)device_ctx;
    return ctx != nullptr && ctx->thread_count_explicit;
}

marmot_error_t cpu_configure(const void *device_ctx, const marmot_context_t *ctx) {
    if (device_ctx == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    cpu_context_t *cpu_ctx = (cpu_context_t *)device_ctx;
    cpu_ctx->quant_activation_mode = ctx != nullptr ? ctx->policy.quant_activation_mode : MARMOT_QUANT_ACTIVATION_AUTO;
    return MARMOT_SUCCESS;
}

marmot_error_t cpu_synchronize(const void *device_ctx) {
    (void)device_ctx;
    return MARMOT_SUCCESS;
}

int32_t *cpu_get_rope_positions_scratch(cpu_context_t *ctx, size_t seq_len) {
    if (ctx == nullptr || seq_len == 0) {
        return nullptr;
    }

    // Grow buffer if needed
    if (ctx->rope_positions_capacity < seq_len) {
        free(ctx->rope_positions_scratch);
        ctx->rope_positions_scratch = malloc(seq_len * sizeof(int32_t));
        if (ctx->rope_positions_scratch == nullptr) {
            ctx->rope_positions_capacity = 0;
            return nullptr;
        }
        ctx->rope_positions_capacity = seq_len;
    }

    return ctx->rope_positions_scratch;
}
