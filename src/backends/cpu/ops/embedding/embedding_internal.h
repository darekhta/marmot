#ifndef CPU_EMBEDDING_INTERNAL_H
#define CPU_EMBEDDING_INTERNAL_H

#include "cpu_backend_internal.h"

marmot_error_t cpu_embedding_gather_scalar(const void *device_ctx, const marmot_embedding_gather_desc_t *desc);
#if HAS_NEON
marmot_error_t cpu_embedding_gather_neon(const void *device_ctx, const marmot_embedding_gather_desc_t *desc);
#endif
#if HAS_AVX2
marmot_error_t cpu_embedding_gather_avx2(const void *device_ctx, const marmot_embedding_gather_desc_t *desc);
#endif

#endif
