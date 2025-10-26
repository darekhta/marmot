#ifndef MARMOT_GRAPH_OP_SIGNATURE_H
#define MARMOT_GRAPH_OP_SIGNATURE_H

#include "marmot/matmul_types.h"
#include "marmot/ops_types.h"
#include "marmot/traits_ids.gen.h"
#include "marmot/types.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MARMOT_WEIGHT_LAYOUT_INVALID = 0,
    MARMOT_WEIGHT_LAYOUT_SEPARATE = 1,
    MARMOT_WEIGHT_LAYOUT_PACKED_3MK = 2,
} marmot_weight_layout_t;

typedef enum {
    MARMOT_STRIDE_MODE_ANY = 0,
    MARMOT_STRIDE_MODE_CONTIGUOUS = 1,
    MARMOT_STRIDE_MODE_ROW_STRIDED = 2,
    MARMOT_STRIDE_MODE_STRIDED = 3,
} marmot_stride_mode_t;

typedef enum {
    MARMOT_EPILOGUE_NONE = 0,
    MARMOT_EPILOGUE_BIAS = 1u << 0,
    MARMOT_EPILOGUE_ACTIVATION = 1u << 1,
    MARMOT_EPILOGUE_RESIDUAL = 1u << 2,
    MARMOT_EPILOGUE_ROPE = 1u << 3,
} marmot_epilogue_flags_t;

typedef struct {
    uint32_t block_size;
    uint32_t group_size;
    marmot_dtype_t scale_dtype;
    marmot_dtype_t zero_point_dtype;
} marmot_quant_block_t;

typedef struct {
    uint32_t n;
    uint32_t d;
} marmot_layernorm_dims_t;

typedef struct {
    uint32_t seq_len;
    uint32_t n_rot;
} marmot_rope_dims_t;

typedef struct {
    uint32_t n_elems;
} marmot_elementwise_dims_t;

typedef struct {
    uint32_t axis_size;
    uint32_t inner_stride;
    uint32_t outer_size;
    uint32_t row_count;
} marmot_softmax_dims_t;

typedef union {
    marmot_matmul_dims_t matmul;
    marmot_layernorm_dims_t layernorm;
    marmot_rope_dims_t rope;
    marmot_elementwise_dims_t elementwise;
    marmot_softmax_dims_t softmax;
} marmot_op_signature_dims_t;

typedef struct {
    marmot_op_id_t op_id;
    marmot_profile_id_t profile_id;
    marmot_matmul_layout_t matmul_layout;

    marmot_dtype_t input_dtype;
    marmot_dtype_t weight_dtype;
    marmot_dtype_t output_dtype;
    marmot_dtype_t accum_dtype;

    marmot_qscheme_id_t qscheme_id;
    marmot_quant_block_t quant_block;

    marmot_weight_layout_t weight_layout;
    marmot_stride_mode_t stride_mode;
    uint32_t epilogue_flags;
    marmot_device_unary_op_t activation;
    uint32_t variant_flags;

    marmot_op_signature_dims_t dims;
} marmot_op_signature_t;

#ifdef __cplusplus
}
#endif

#endif // MARMOT_GRAPH_OP_SIGNATURE_H
