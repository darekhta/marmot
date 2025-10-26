#ifndef MARMOT_MATMUL_TYPES_H
#define MARMOT_MATMUL_TYPES_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    size_t N;
    size_t K;
    size_t M;
} marmot_matmul_dims_t;

typedef enum {
    MARMOT_MATMUL_LAYOUT_INVALID = 0,
    MARMOT_MATMUL_LAYOUT_NN = 1,
    MARMOT_MATMUL_LAYOUT_NT = 2,
    MARMOT_MATMUL_LAYOUT_TN = 3,
    MARMOT_MATMUL_LAYOUT_TT = 4,
} marmot_matmul_layout_t;

#ifdef __cplusplus
}
#endif

#endif // MARMOT_MATMUL_TYPES_H
