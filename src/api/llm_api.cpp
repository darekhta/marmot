#include "marmot/error.h"
#include "marmot/inference/llm.h"

#include <cstring>

extern "C" {

marmot_error_t marmot_llm_generate_options_init(marmot_llm_generate_options_t *opts) {
    try {
        if (opts == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(marmot_llm_generate_options_t);
        opts->struct_version = MARMOT_LLM_GENERATE_OPTIONS_VERSION;
        opts->flags = 0;
        opts->max_new_tokens = 0;
        opts->stop_on_eos = true;
        opts->stop_tokens = nullptr;
        opts->stop_tokens_len = 0;
        opts->prefill_progress = nullptr;
        opts->on_token = nullptr;
        opts->user_data = nullptr;
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_llm_generate_options_init threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

marmot_error_t marmot_llm_sampling_options_init(marmot_llm_sampling_options_t *opts) {
    try {
        if (opts == nullptr) {
            return MARMOT_ERROR_INVALID_ARGUMENT;
        }
        std::memset(opts, 0, sizeof(*opts));
        opts->struct_size = sizeof(marmot_llm_sampling_options_t);
        opts->struct_version = MARMOT_LLM_SAMPLING_OPTIONS_VERSION;
        opts->flags = 0;
        opts->seed = 0;
        opts->temperature = 0.0f;
        opts->top_k = 0;
        opts->top_p = 1.0f;
        opts->min_p = 0.0f;
        opts->repetition_penalty = 1.0f;
        return MARMOT_SUCCESS;
    } catch (...) {
        marmot_set_error(MARMOT_ERROR_INVALID_OPERATION, "marmot_llm_sampling_options_init threw");
        return MARMOT_ERROR_INVALID_OPERATION;
    }
}

} // extern "C"
