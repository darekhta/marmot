#ifndef MARMOT_INFERENCE_ENGINE_H
#define MARMOT_INFERENCE_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#include "../error.h"
#include "../macros.h"
#include "../types.h"
#include "llm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct marmot_serving_engine marmot_serving_engine_t;
typedef uint64_t marmot_request_id_t;

// Thread-safety: marmot_serving_engine_t is single-threaded; callers must serialize
// submit/step/cancel/release/last_batch.

typedef struct {
    size_t token_count;
    size_t sample_count;
    const marmot_token_id_t *token_ids;
    const uint32_t *token_meta;
    const uint32_t *sample_indices;
    const marmot_request_id_t *sample_request_ids;
} marmot_serving_engine_batch_view_t;

#define MARMOT_SERVING_ENGINE_OPTIONS_VERSION 2
#define MARMOT_SERVING_REQUEST_EXT_VERSION 0

typedef enum {
    MARMOT_SERVING_ENGINE_FLAG_NONE = 0,
    MARMOT_SERVING_ENGINE_FLAG_ENABLE_PREFIX_CACHE = 1u << 0,
    MARMOT_SERVING_ENGINE_FLAG_ENABLE_SWAP = 1u << 1,
    MARMOT_SERVING_ENGINE_FLAG_DETERMINISTIC_KV = 1u << 2,
} marmot_serving_engine_flags_t;

typedef enum {
    MARMOT_SERVING_REQUEST_FLAG_NONE = 0,
    MARMOT_SERVING_REQUEST_FLAG_DISABLE_PREFIX_CACHE = 1u << 0,
} marmot_serving_request_flags_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;

    size_t max_seqs;
    size_t max_batch_seqs;
    size_t max_num_tokens;
    size_t max_seq_len;

    size_t block_size;
    size_t num_kv_blocks;
    size_t num_swap_blocks;
    marmot_dtype_t kv_dtype;
    float kv_block_watermark;

    size_t prefill_chunk_size;

    const void *pnext;
    uint64_t reserved[4];
} marmot_serving_engine_options_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;

    int32_t priority;
    const char *cache_salt;
    size_t cache_salt_len;
    size_t retention_blocks;
    size_t num_samples;
    void *const *sample_user_data;
    size_t sample_user_data_len;
    marmot_request_id_t *out_request_ids;
    size_t out_request_ids_capacity;

    const void *pnext;
    uint64_t reserved[4];
} marmot_serving_request_ext_t;

MARMOT_NODISCARD marmot_error_t marmot_serving_engine_options_init(marmot_serving_engine_options_t *opts);

MARMOT_NODISCARD marmot_error_t marmot_serving_engine_create(
    const marmot_context_t *ctx, const marmot_model_t *model, const marmot_serving_engine_options_t *opts,
    marmot_serving_engine_t **out_engine
);
void marmot_serving_engine_destroy(marmot_serving_engine_t *engine);

MARMOT_NODISCARD marmot_error_t marmot_serving_engine_submit(
    marmot_serving_engine_t *engine, const marmot_token_id_t *prompt_tokens, size_t prompt_len,
    const marmot_llm_generate_options_t *gen_opts, const marmot_llm_sampling_options_t *sampling_opts,
    marmot_request_id_t *out_request_id
);

MARMOT_NODISCARD marmot_error_t
marmot_serving_engine_step(marmot_serving_engine_t *engine, size_t max_steps, size_t *out_steps_done);

MARMOT_NODISCARD marmot_llm_request_state_t
marmot_serving_engine_request_state(const marmot_serving_engine_t *engine, marmot_request_id_t request_id);

MARMOT_NODISCARD marmot_error_t
marmot_serving_engine_request_cancel(marmot_serving_engine_t *engine, marmot_request_id_t request_id);

MARMOT_NODISCARD marmot_error_t
marmot_serving_engine_request_release(marmot_serving_engine_t *engine, marmot_request_id_t request_id);

MARMOT_NODISCARD marmot_error_t
marmot_serving_engine_last_batch(const marmot_serving_engine_t *engine, marmot_serving_engine_batch_view_t *out_batch);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_INFERENCE_ENGINE_H
