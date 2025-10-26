#ifndef CPU_CONVERT_REGISTRY_H
#define CPU_CONVERT_REGISTRY_H

#include "marmot/error.h"

#include "convert_table.h"

#ifdef __cplusplus
extern "C" {
#endif

struct cpu_context;
cpu_convert_fn
cpu_convert_resolve_fn(const struct cpu_context *ctx, marmot_dtype_t dst, marmot_dtype_t src, const char **impl_name);

#define CPU_CONVERT_REGISTER_TRAITS(symbol)

#ifdef __cplusplus
}
#endif

#endif
