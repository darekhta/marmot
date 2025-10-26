#include "marmot/allocator.h"
#include "marmot/device.h"
#include "marmot/error.h"

marmot_error_t marmot_allocator_get_usage(const marmot_context_t *ctx, marmot_allocator_usage_t *usage) {
    if (ctx == nullptr || usage == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (ctx->ops == nullptr || ctx->ops->allocator_usage == nullptr) {
        *usage = (marmot_allocator_usage_t){0};
        return MARMOT_ERROR_NOT_IMPLEMENTED;
    }
    if (ctx->device_ctx == nullptr) {
        *usage = (marmot_allocator_usage_t){0};
        return MARMOT_SUCCESS;
    }
    return ctx->ops->allocator_usage(ctx->device_ctx, usage);
}
