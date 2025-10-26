#include "ops/matmul/quantized/internal/vec_dot.h"

#define VEC_DOT_NAME(base) cpu_vec_dot_##base##_neon
#include "ops/matmul/quantized/vec_dot_impl.h"
#undef VEC_DOT_NAME
