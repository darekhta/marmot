#ifndef MARMOT_CORE_DISPATCH_EXECUTE_H
#define MARMOT_CORE_DISPATCH_EXECUTE_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/op_signature.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_error_t marmot_execute_signature(
    const marmot_context_t *ctx, const marmot_op_signature_t *sig, const void *args, const char *op_name
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_DISPATCH_EXECUTE_H
