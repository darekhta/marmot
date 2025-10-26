#ifndef MARMOT_GRAPH_GRAPH_TYPES_H
#define MARMOT_GRAPH_GRAPH_TYPES_H

#include "marmot/types.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t marmot_value_id_t;
#define MARMOT_VALUE_ID_INVALID ((marmot_value_id_t)UINT32_MAX)

typedef struct {
    marmot_dtype_t dtype;
    uint32_t ndim;
    size_t shape[MARMOT_MAX_DIMS];
    size_t strides[MARMOT_MAX_DIMS];
} marmot_graph_tensor_desc_t;

typedef enum {
    MARMOT_GRAPH_ATTR_INT = 0,
    MARMOT_GRAPH_ATTR_FLOAT = 1,
    MARMOT_GRAPH_ATTR_STRING = 2,
    MARMOT_GRAPH_ATTR_TENSOR = 3,
} marmot_graph_attr_type_t;

typedef struct {
    const char *key;
    marmot_graph_attr_type_t type;
    union {
        int64_t i64;
        double f64;
        const char *string_value;
        const marmot_tensor_t *tensor;
    } value;
} marmot_graph_attr_t;

#ifdef __cplusplus
}
#endif

#endif // MARMOT_GRAPH_GRAPH_TYPES_H
