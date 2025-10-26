#ifndef MARMOT_OPS_TYPES_H
#define MARMOT_OPS_TYPES_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// Operation identifiers
//------------------------------------------------------------------------------

typedef enum {
    MARMOT_DEVICE_BINARY_ADD = 0,
    MARMOT_DEVICE_BINARY_SUB = 1,
    MARMOT_DEVICE_BINARY_MUL = 2,
    MARMOT_DEVICE_BINARY_DIV = 3,
    MARMOT_DEVICE_BINARY_MIN = 4,
    MARMOT_DEVICE_BINARY_MAX = 5,
    MARMOT_DEVICE_BINARY_POW = 6,
    MARMOT_DEVICE_BINARY_MOD = 7,
    MARMOT_DEVICE_BINARY_BITWISE_AND = 8,
    MARMOT_DEVICE_BINARY_BITWISE_OR = 9,
    MARMOT_DEVICE_BINARY_BITWISE_XOR = 10,
    MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT = 11,
    MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT = 12,
    MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL = 13,
    MARMOT_DEVICE_BINARY_COMPARE_EQ = 14,
    MARMOT_DEVICE_BINARY_COMPARE_NE = 15,
    MARMOT_DEVICE_BINARY_COMPARE_LT = 16,
    MARMOT_DEVICE_BINARY_COMPARE_LE = 17,
    MARMOT_DEVICE_BINARY_COMPARE_GT = 18,
    MARMOT_DEVICE_BINARY_COMPARE_GE = 19,
    MARMOT_DEVICE_BINARY_SWIGLU = 20,
    MARMOT_DEVICE_BINARY_GEGLU = 21,
    MARMOT_DEVICE_BINARY_COUNT,
} marmot_device_binary_op_t;

typedef enum {
    MARMOT_DEVICE_UNARY_IDENTITY = 0, // Must be 0 for zero-initialized structs
    MARMOT_DEVICE_UNARY_RELU = 1,
    MARMOT_DEVICE_UNARY_GELU = 2,
    MARMOT_DEVICE_UNARY_GELU_TANH = 3,
    MARMOT_DEVICE_UNARY_SILU = 4,
    MARMOT_DEVICE_UNARY_SIGMOID = 5,
    MARMOT_DEVICE_UNARY_TANH = 6,
    MARMOT_DEVICE_UNARY_MISH = 7,
    MARMOT_DEVICE_UNARY_ELU = 8,
    MARMOT_DEVICE_UNARY_SELU = 9,
    MARMOT_DEVICE_UNARY_LEAKY_RELU = 10,
    MARMOT_DEVICE_UNARY_PRELU = 11,
    MARMOT_DEVICE_UNARY_ABS = 12,
    MARMOT_DEVICE_UNARY_NEG = 13,
    MARMOT_DEVICE_UNARY_SIGN = 14,
    MARMOT_DEVICE_UNARY_SQRT = 15,
    MARMOT_DEVICE_UNARY_EXP = 16,
    MARMOT_DEVICE_UNARY_LOG = 17,
    MARMOT_DEVICE_UNARY_BITWISE_NOT = 18,
    MARMOT_DEVICE_UNARY_COUNT,
} marmot_device_unary_op_t;

typedef enum {
    MARMOT_DEVICE_TERNARY_FMA = 0,
    MARMOT_DEVICE_TERNARY_WHERE = 1,
    MARMOT_DEVICE_TERNARY_COUNT,
} marmot_device_ternary_op_t;

typedef enum {
    MARMOT_DEVICE_REDUCTION_SUM = 0,
    MARMOT_DEVICE_REDUCTION_MEAN = 1,
    MARMOT_DEVICE_REDUCTION_PROD = 2,
    MARMOT_DEVICE_REDUCTION_MAX = 3,
    MARMOT_DEVICE_REDUCTION_MIN = 4,
    MARMOT_DEVICE_REDUCTION_ARGMAX = 5,
    MARMOT_DEVICE_REDUCTION_ARGMIN = 6,
    MARMOT_DEVICE_REDUCTION_ANY = 7,
    MARMOT_DEVICE_REDUCTION_ALL = 8,
    MARMOT_DEVICE_REDUCTION_VARIANCE = 9,
    MARMOT_DEVICE_REDUCTION_STD = 10,
    MARMOT_DEVICE_REDUCTION_NORM_L1 = 11,
    MARMOT_DEVICE_REDUCTION_NORM_L2 = 12,
    MARMOT_DEVICE_REDUCTION_COUNT,
} marmot_device_reduction_op_t;

//------------------------------------------------------------------------------
// Operation descriptors
//------------------------------------------------------------------------------

struct marmot_activation_params {
    const marmot_tensor_t *parameter_tensor;
    const marmot_tensor_t *bias;
    float alpha;
    float beta;
    float gamma;
};

typedef struct marmot_rope_params {
    const marmot_tensor_t *positions;
    marmot_rope_scaling_type_t scaling_type;
    marmot_rope_type_t rope_type;
    float theta;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    uint32_t orig_ctx_len;
    uint32_t head_dim;
    bool apply_to_q;
    bool apply_to_k;
} marmot_rope_params_t;

struct marmot_matmul_epilogue {
    const marmot_tensor_t *bias;
    bool enable_output_cast;
    marmot_dtype_t output_dtype;
};

struct marmot_reduction_params {
    const int32_t *axes;
    size_t num_axes;
    bool keepdims;
    bool unbiased;
    float epsilon;
};

// Quantized vector dot-product descriptor
// Naming note (PyTorch convention):
// - "activations" corresponds to PyTorch "input"
// - "weights" corresponds to PyTorch "weight"
// The dot computes: dot(input, weight) for 1D vectors formed by concatenating
// `num_blocks` GGUF blocks of size 32. This descriptor is used as a building
// block for quantized matmul. A convenience constructor from tensors is
// provided in tensor.h.
struct marmot_vec_dot_descriptor {
    const void *weights;
    const void *activations;
    size_t num_blocks;
    marmot_quant_kind_t weight_kind;
    marmot_quant_kind_t activation_kind;
    marmot_quant_layout_t layout;
};

struct marmot_embedding_desc {
    const marmot_tensor_t *weights;
    const marmot_tensor_t *token_ids;
    marmot_tensor_t *out;
    marmot_dtype_t dtype_out;
    float scale;
    int32_t padding_id;
    bool bounds_check;
    bool ragged;
    const int32_t *row_offsets;
    size_t num_row_offsets;
    marmot_preference_t prefer_gpu_private;
    marmot_preference_t allow_quant_decode_on_the_fly;
};

struct marmot_embedding_gather_desc {
    const marmot_tensor_t *weights;
    const marmot_tensor_t *token_ids;
    marmot_tensor_t *out;
    marmot_dtype_t dtype_out;
    float scale;
    int32_t padding_id;
    bool bounds_check;
    marmot_preference_t prefer_gpu_private;
    marmot_preference_t allow_quant_decode_on_the_fly;
};

struct marmot_layernorm_desc {
    const marmot_tensor_t *x;
    const marmot_tensor_t *residual;
    const marmot_tensor_t *weight;
    const marmot_tensor_t *bias;
    marmot_tensor_t *out;
    float eps;
};

struct marmot_rmsnorm_desc {
    const marmot_tensor_t *x;
    const marmot_tensor_t *residual;
    const marmot_tensor_t *weight;
    marmot_tensor_t *out;
    float eps;
};

struct marmot_softmax_desc {
    const marmot_tensor_t *x;
    marmot_tensor_t *out;
    int32_t axis;
};

struct marmot_reduction_desc {
    const marmot_tensor_t *input;
    marmot_tensor_t *out;
    marmot_tensor_t *indices_out; // For argmax/argmin only, otherwise nullptr
    const int32_t *axes;
    size_t num_axes;
    bool keepdims;
    bool unbiased; // For variance/std only
    float epsilon; // For variance/std only
};

#ifdef __cplusplus
}
#endif

#endif
