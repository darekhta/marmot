#ifndef MARMOT_METAL_KERNEL_QUERY_GEN_H
#define MARMOT_METAL_KERNEL_QUERY_GEN_H

#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/graph/kernel_selection.h"
#include "marmot/graph/op_signature.h"
#include "marmot/types.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_kernel_selection_t marmot_metal_query_kernel(const marmot_op_signature_t *sig, const marmot_device_caps_t *caps);

#ifdef __cplusplus
}
#endif

#endif
