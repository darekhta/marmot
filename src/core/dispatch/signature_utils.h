#ifndef MARMOT_CORE_DISPATCH_SIGNATURE_UTILS_H
#define MARMOT_CORE_DISPATCH_SIGNATURE_UTILS_H

#include "marmot/types.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_dtype_t marmot_elementwise_accum_dtype(marmot_dtype_t dtype);
marmot_dtype_t marmot_matmul_accum_dtype(marmot_dtype_t dtype);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_DISPATCH_SIGNATURE_UTILS_H
