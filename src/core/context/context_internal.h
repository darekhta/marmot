#ifndef MARMOT_CORE_CONTEXT_INTERNAL_H
#define MARMOT_CORE_CONTEXT_INTERNAL_H

#include "marmot/device.h"

#ifdef __cplusplus
extern "C" {
#endif

void marmot_context_apply_default_policy(marmot_context_t *ctx);
[[nodiscard]] marmot_error_t marmot_context_set_thread_count(marmot_context_t *ctx, size_t num_threads);
[[nodiscard]] marmot_error_t marmot_context_set_thread_count_auto(marmot_context_t *ctx, size_t num_threads);
[[nodiscard]] size_t marmot_context_get_thread_count(const marmot_context_t *ctx);
[[nodiscard]] bool marmot_context_thread_count_is_explicit(const marmot_context_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_CONTEXT_INTERNAL_H
