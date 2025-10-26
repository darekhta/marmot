#ifndef CPU_ROPE_KERNELS_H
#define CPU_ROPE_KERNELS_H

#include "marmot/error.h"
#include "marmot/tensor.h"
#include "marmot/types.h"

#include <stddef.h>

typedef struct cpu_context cpu_context_t;

typedef enum {
    CPU_ROPE_IMPL_SCALAR = 0,
    CPU_ROPE_IMPL_NEON = 1,
    CPU_ROPE_IMPL_AVX2 = 2,
} cpu_rope_impl_kind_t;

typedef marmot_error_t (*cpu_rope_kernel_fn)(
    cpu_context_t *ctx, const marmot_tensor_t *x, const marmot_tensor_t *positions, const float *freqs,
    float attn_scale, marmot_rope_type_t rope_type, size_t seq_len, size_t dim, size_t total_seqs, marmot_tensor_t *out
);

typedef struct {
    marmot_dtype_t dtype;
    cpu_rope_impl_kind_t impl_kind;
    size_t min_dim;
    cpu_rope_kernel_fn kernel;
} cpu_rope_traits_t;

#endif // CPU_ROPE_KERNELS_H
