#include "ops/matmul/quantized/internal/vec_dot.h"

#undef HAS_NEON
#undef HAS_AVX2
#define HAS_NEON 0
#define HAS_AVX2 0

#define VEC_DOT_NAME(base) cpu_vec_dot_##base##_scalar
#include "ops/matmul/quantized/vec_dot_impl.h"
#undef VEC_DOT_NAME
