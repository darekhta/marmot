#ifndef MARMOT_INFERENCE_LLM_H
#define MARMOT_INFERENCE_LLM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "../error.h"
#include "../macros.h"
#include "../tokenizer.h"
#include "../types.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MARMOT_LLM_GENERATE_OPTIONS_VERSION 1
#define MARMOT_LLM_SAMPLING_OPTIONS_VERSION 1

typedef enum {
    MARMOT_LLM_SAMPLING_FLAG_NONE = 0,
    MARMOT_LLM_SAMPLING_FLAG_SUPPRESS_SPECIAL_TOKENS = 1u << 0,
} marmot_llm_sampling_flags_t;

typedef void (*marmot_llm_progress_callback_t)(void *user_data, size_t done, size_t total);
typedef void (*marmot_llm_token_callback_t)(void *user_data, marmot_token_id_t token_id);

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;
    size_t max_new_tokens;
    bool stop_on_eos;
    const marmot_token_id_t *stop_tokens;
    size_t stop_tokens_len;
    marmot_llm_progress_callback_t prefill_progress;
    marmot_llm_token_callback_t on_token;
    void *user_data;
    const void *pnext;
    uint64_t reserved[4];
} marmot_llm_generate_options_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;
    uint64_t seed;
    float temperature;
    size_t top_k;
    float top_p;
    float min_p;
    float repetition_penalty;
    const void *pnext;
    uint64_t reserved[4];
} marmot_llm_sampling_options_t;

typedef enum {
    MARMOT_LLM_REQUEST_STATE_INVALID = 0,
    MARMOT_LLM_REQUEST_STATE_PENDING = 1,
    MARMOT_LLM_REQUEST_STATE_PREFILL = 2,
    MARMOT_LLM_REQUEST_STATE_DECODING = 3,
    MARMOT_LLM_REQUEST_STATE_DONE = 4,
    MARMOT_LLM_REQUEST_STATE_FAILED = 5,
    MARMOT_LLM_REQUEST_STATE_CANCELED = 6,
} marmot_llm_request_state_t;
MARMOT_NODISCARD marmot_error_t marmot_llm_generate_options_init(marmot_llm_generate_options_t *opts);
MARMOT_NODISCARD marmot_error_t marmot_llm_sampling_options_init(marmot_llm_sampling_options_t *opts);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_INFERENCE_LLM_H
