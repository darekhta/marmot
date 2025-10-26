#ifndef CPU_CONVERT_TABLE_H
#define CPU_CONVERT_TABLE_H

#include "marmot/types.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    CPU_CONVERT_IMPL_SCALAR = 0,
    CPU_CONVERT_IMPL_NEON = 1,
    CPU_CONVERT_IMPL_AVX2 = 2,
    CPU_CONVERT_IMPL_ACCELERATE = 3,
    CPU_CONVERT_IMPL_CUSTOM = 4,
} cpu_convert_impl_kind_t;

typedef void (*cpu_convert_fn)(const void *device_ctx, void *dst, const void *src, size_t n);

typedef struct {
    cpu_convert_fn fn;
    const char *impl_name;
} cpu_convert_slot_t;

typedef struct {
    cpu_convert_slot_t table[MARMOT_DTYPE_COUNT][MARMOT_DTYPE_COUNT];
} cpu_convert_table_t;

typedef struct cpu_convert_traits {
    marmot_dtype_t src;
    marmot_dtype_t dst;
    cpu_convert_impl_kind_t impl_kind;
    cpu_convert_fn fn;
    const char *impl_name;
} cpu_convert_traits_t;

#ifdef __cplusplus
}
#endif

#endif
