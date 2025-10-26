#include "marmot/error.h"
#include "marmot/ops/paged_attention.h"

#include "core/dispatch/dispatch_build.h"
#include "core/dispatch/dispatch_execute.h"
#include "graph/kernel_dispatch_args.gen.h"

marmot_error_t marmot_paged_attention_impl(const marmot_context_t *ctx, const marmot_paged_attention_desc_t *desc) {
    if (ctx == nullptr || desc == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "paged_attention requires context and descriptor");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (desc->token_count == 0) {
        return MARMOT_SUCCESS;
    }
    marmot_op_signature_t sig = {0};
    marmot_kernel_args_paged_attention_t packed = {0};
    marmot_error_t build_status = marmot_paged_attention_build(ctx, desc, &sig, &packed);
    if (build_status != MARMOT_SUCCESS) {
        return build_status;
    }
    return marmot_execute_signature(ctx, &sig, &packed, "paged_attention");
}
