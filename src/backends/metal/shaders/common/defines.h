#pragma once

struct ActivationParams {
    float alpha;
    float beta;
    float gamma;
    float delta;
};

enum UnaryActivationKind : uint {
    UnaryActivationIdentity = 0u,
    UnaryActivationRelu = 1u,
    UnaryActivationGelu = 2u,
    UnaryActivationGeluTanh = 3u,
    UnaryActivationSilu = 4u,
    UnaryActivationSigmoid = 5u,
    UnaryActivationTanh = 6u,
    UnaryActivationMish = 7u,
    UnaryActivationElu = 8u,
    UnaryActivationSelu = 9u,
    UnaryActivationLeakyRelu = 10u,
    UnaryActivationPrelu = 11u,
};

struct FusedBiasActivationUniforms {
    uint total_elements;
    uint bias_length;
    uint activation;
    uint flags;
    ActivationParams params;
    uint rope_dim;
    uint rope_pairs;
    uint rope_rows;
    uint rope_flags;
    uint rope_type;
    float rope_attn_scale;
};

constant uint FusedBiasFlagScalarBias = 1u << 0;
constant uint FusedBiasFlagHasBias = 1u << 1;
constant uint FusedBiasFlagHasResidual = 1u << 2;
constant uint FusedBiasFlagHasRope = 1u << 3;

enum MarmotDeviceBinaryOp : uint {
    MARMOT_DEVICE_BINARY_ADD = 0u,
    MARMOT_DEVICE_BINARY_SUB = 1u,
    MARMOT_DEVICE_BINARY_MUL = 2u,
    MARMOT_DEVICE_BINARY_DIV = 3u,
    MARMOT_DEVICE_BINARY_MIN = 4u,
    MARMOT_DEVICE_BINARY_MAX = 5u,
    MARMOT_DEVICE_BINARY_POW = 6u,
    MARMOT_DEVICE_BINARY_MOD = 7u,
    MARMOT_DEVICE_BINARY_BITWISE_AND = 8u,
    MARMOT_DEVICE_BINARY_BITWISE_OR = 9u,
    MARMOT_DEVICE_BINARY_BITWISE_XOR = 10u,
    MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT = 11u,
    MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT = 12u,
    MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL = 13u,
    MARMOT_DEVICE_BINARY_COMPARE_EQ = 14u,
    MARMOT_DEVICE_BINARY_COMPARE_NE = 15u,
    MARMOT_DEVICE_BINARY_COMPARE_LT = 16u,
    MARMOT_DEVICE_BINARY_COMPARE_LE = 17u,
    MARMOT_DEVICE_BINARY_COMPARE_GT = 18u,
    MARMOT_DEVICE_BINARY_COMPARE_GE = 19u,
    MARMOT_DEVICE_BINARY_SWIGLU = 20u,
    MARMOT_DEVICE_BINARY_GEGLU = 21u,
};

enum MarmotDeviceReductionOp : uint {
    MARMOT_DEVICE_REDUCTION_SUM = 0u,
    MARMOT_DEVICE_REDUCTION_MEAN = 1u,
    MARMOT_DEVICE_REDUCTION_PROD = 2u,
    MARMOT_DEVICE_REDUCTION_MAX = 3u,
    MARMOT_DEVICE_REDUCTION_MIN = 4u,
    MARMOT_DEVICE_REDUCTION_ARGMAX = 5u,
    MARMOT_DEVICE_REDUCTION_ARGMIN = 6u,
    MARMOT_DEVICE_REDUCTION_ANY = 7u,
    MARMOT_DEVICE_REDUCTION_ALL = 8u,
    MARMOT_DEVICE_REDUCTION_VARIANCE = 9u,
    MARMOT_DEVICE_REDUCTION_STD = 10u,
    MARMOT_DEVICE_REDUCTION_NORM_L1 = 11u,
    MARMOT_DEVICE_REDUCTION_NORM_L2 = 12u,
};
