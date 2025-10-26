#include "signature_utils.h"

marmot_dtype_t marmot_elementwise_accum_dtype(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
    case MARMOT_DTYPE_FLOAT8_E5M2:
#endif
    case MARMOT_DTYPE_FLOAT32:
        return MARMOT_DTYPE_FLOAT32;
    case MARMOT_DTYPE_FLOAT64:
        return MARMOT_DTYPE_FLOAT64;
    default:
        return dtype;
    }
}

marmot_dtype_t marmot_matmul_accum_dtype(marmot_dtype_t dtype) {
    switch (dtype) {
    case MARMOT_DTYPE_FLOAT16:
    case MARMOT_DTYPE_BFLOAT16:
#if MARMOT_ENABLE_FP8
    case MARMOT_DTYPE_FLOAT8_E4M3:
    case MARMOT_DTYPE_FLOAT8_E5M2:
#endif
        return MARMOT_DTYPE_FLOAT32;
    case MARMOT_DTYPE_FLOAT64:
        return MARMOT_DTYPE_FLOAT64;
    default:
        return MARMOT_DTYPE_FLOAT32;
    }
}
