#ifndef MARMOT_INFERENCE_KV_POOL_H
#define MARMOT_INFERENCE_KV_POOL_H

#include <stddef.h>
#include <stdint.h>

#include "../error.h"
#include "../macros.h"
#include "../tensor.h"
#include "../types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct marmot_kv_pool marmot_kv_pool_t;
typedef uint32_t marmot_block_id_t;
typedef uint32_t marmot_seq_slot_t;
typedef uint32_t marmot_kv_slot_t;

#define MARMOT_KV_POOL_OPTIONS_VERSION 1
#define MARMOT_BLOCK_ID_INVALID ((marmot_block_id_t)0xFFFFFFFFu)

typedef enum {
    MARMOT_KV_POOL_FLAG_NONE = 0,
    MARMOT_KV_POOL_FLAG_DETERMINISTIC_ALLOC = 1u << 0,
} marmot_kv_pool_flags_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;

    marmot_backend_type_t backend;

    size_t max_seqs;
    size_t max_seq_len;
    size_t block_size;

    size_t num_blocks;
    size_t num_swap_blocks;
    size_t num_layers;
    size_t num_kv_heads;
    size_t head_dim;

    marmot_dtype_t kv_dtype;

    const void *pnext;
    uint64_t reserved[4];
} marmot_kv_pool_options_t;

typedef struct {
    marmot_seq_slot_t seq;
    size_t token_count;
    uint64_t cookie;
} marmot_kv_append_plan_t;

typedef struct {
    marmot_seq_slot_t seq;
    size_t num_blocks;
    size_t prefix_len;
    uint64_t cookie;
} marmot_kv_prefix_plan_t;

MARMOT_NODISCARD marmot_error_t marmot_kv_pool_options_init(marmot_kv_pool_options_t *opts);

MARMOT_NODISCARD marmot_error_t
marmot_kv_pool_create(const marmot_context_t *ctx, const marmot_kv_pool_options_t *opts, marmot_kv_pool_t **out_pool);
void marmot_kv_pool_destroy(marmot_kv_pool_t *pool);

MARMOT_NODISCARD marmot_error_t marmot_kv_pool_acquire_seq(marmot_kv_pool_t *pool, marmot_seq_slot_t *out_seq);
MARMOT_NODISCARD marmot_error_t marmot_kv_pool_release_seq(marmot_kv_pool_t *pool, marmot_seq_slot_t seq);

MARMOT_NODISCARD marmot_error_t marmot_kv_pool_prepare_append(
    marmot_kv_pool_t *pool, marmot_seq_slot_t seq, size_t token_count, marmot_kv_append_plan_t *out_plan,
    marmot_kv_slot_t *out_slots, size_t *out_start_pos
);
MARMOT_NODISCARD marmot_error_t
marmot_kv_pool_commit_append(marmot_kv_pool_t *pool, const marmot_kv_append_plan_t *plan);
void marmot_kv_pool_abort_append(marmot_kv_pool_t *pool, const marmot_kv_append_plan_t *plan);

MARMOT_NODISCARD marmot_error_t marmot_kv_pool_prepare_prefix_attach(
    marmot_kv_pool_t *pool, marmot_seq_slot_t seq, const marmot_block_id_t *block_ids, size_t num_blocks,
    size_t prefix_len, marmot_kv_prefix_plan_t *out_plan
);
MARMOT_NODISCARD marmot_error_t
marmot_kv_pool_commit_prefix_attach(marmot_kv_pool_t *pool, const marmot_kv_prefix_plan_t *plan);
void marmot_kv_pool_abort_prefix_attach(marmot_kv_pool_t *pool, const marmot_kv_prefix_plan_t *plan);

MARMOT_NODISCARD marmot_error_t marmot_kv_pool_get_tensors(
    marmot_kv_pool_t *pool, marmot_tensor_t **out_k, marmot_tensor_t **out_v, marmot_tensor_t **out_block_table
);
MARMOT_NODISCARD marmot_error_t
marmot_kv_pool_get_scale_tensors(marmot_kv_pool_t *pool, marmot_tensor_t **out_k_scale, marmot_tensor_t **out_v_scale);

MARMOT_NODISCARD size_t marmot_kv_pool_free_block_count(const marmot_kv_pool_t *pool);
MARMOT_NODISCARD size_t marmot_kv_pool_total_block_count(const marmot_kv_pool_t *pool);

MARMOT_NODISCARD marmot_error_t marmot_kv_pool_swap_out_seq(marmot_kv_pool_t *pool, marmot_seq_slot_t seq);
MARMOT_NODISCARD marmot_error_t marmot_kv_pool_swap_in_seq(marmot_kv_pool_t *pool, marmot_seq_slot_t seq);
MARMOT_NODISCARD marmot_error_t
marmot_kv_pool_is_seq_swapped(const marmot_kv_pool_t *pool, marmot_seq_slot_t seq, bool *out_swapped);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_INFERENCE_KV_POOL_H
