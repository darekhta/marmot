#include "cpu_backend_internal.h"
#include "ops/embedding/embedding_internal.h"

marmot_error_t cpu_embedding_gather(const void *device_ctx, const marmot_embedding_gather_desc_t *desc) {
#if HAS_AVX2
    if (has_avx2(device_ctx)) {
        return cpu_embedding_gather_avx2(device_ctx, desc);
    }
#endif
#if HAS_NEON
    if (has_neon(device_ctx)) {
        return cpu_embedding_gather_neon(device_ctx, desc);
    }
#endif
    return cpu_embedding_gather_scalar(device_ctx, desc);
}
