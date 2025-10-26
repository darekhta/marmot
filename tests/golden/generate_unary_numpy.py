#!/usr/bin/env python3
"""Generate unary activation golden data using NumPy."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "backend" / "golden_unary_numpy.h"

INPUT_VALUES = np.array(
    [
        -2.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        -3.0,
        -1.5,
        0.25,
        0.75,
        1.25,
        1.75,
    ],
    dtype=np.float64,
)

ELU_ALPHA = np.float64(1.1)
SELU_ALPHA = np.float64(1.6732632423543772)
SELU_LAMBDA = np.float64(1.0507009873554805)
LEAKY_SLOPE = np.float64(0.02)
PRELU_SLOPE = np.float64(0.25)
BIAS_SCALAR = np.float64(0.375)
BIAS_VECTOR = np.linspace(-0.3, 0.45, INPUT_VALUES.size, dtype=np.float64)
SQRT_2_OVER_PI = np.float64(math.sqrt(2.0 / math.pi))
INV_SQRT2 = np.float64(1.0 / math.sqrt(2.0))
COEFF = np.float64(0.044715)


def sigmoid_vals(x: np.ndarray) -> np.ndarray:
    pos = x >= 0
    z = np.empty_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    z[~pos] = exp_x / (1.0 + exp_x)
    return z


def mish_vals(x: np.ndarray) -> np.ndarray:
    softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
    return x * np.tanh(softplus)


def gelu_erf(x: np.ndarray) -> np.ndarray:
    erfs = np.array([math.erf(val * INV_SQRT2) for val in x], dtype=np.float64)
    return 0.5 * x * (1.0 + erfs)


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x)
    return 0.5 * x * (1.0 + np.tanh(inner))


def elu_vals(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1.0))


def selu_vals(x: np.ndarray, alpha: float, lambd: float) -> np.ndarray:
    return lambd * elu_vals(x, alpha)


def leaky_vals(x: np.ndarray, slope: float) -> np.ndarray:
    return np.where(x >= 0.0, x, slope * x)


def compute_activation_outputs(x: np.ndarray) -> dict[str, np.ndarray]:
    sigmoid = sigmoid_vals(x)
    outputs = {
        "relu": np.maximum(x, 0.0),
        "gelu": gelu_erf(x),
        "gelu_tanh": gelu_tanh(x),
        "silu": x * sigmoid,
        "sigmoid": sigmoid,
        "tanh": np.tanh(x),
        "mish": mish_vals(x),
        "elu": elu_vals(x, ELU_ALPHA),
        "selu": selu_vals(x, SELU_ALPHA, SELU_LAMBDA),
        "leaky": leaky_vals(x, LEAKY_SLOPE),
        "prelu": leaky_vals(x, PRELU_SLOPE),
    }
    return outputs


def generate() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    x64 = INPUT_VALUES
    base = compute_activation_outputs(x64)
    scalar_bias = compute_activation_outputs(x64 + BIAS_SCALAR)
    vector_bias = compute_activation_outputs(x64 + BIAS_VECTOR)

    data64 = {"input": x64}
    data64.update(base)
    data32 = {name: arr.astype(np.float32) for name, arr in data64.items()}
    scalar32 = {name: arr.astype(np.float32) for name, arr in scalar_bias.items()}
    vector32 = {name: arr.astype(np.float32) for name, arr in vector_bias.items()}

    return data32, data64, scalar32, vector32, scalar_bias, vector_bias


def format_array(name: str, values: np.ndarray, ctype: str, fmt: str, per_line: int = 8) -> str:
    parts = [f"static const {ctype} {name}[{values.size}] = {{"]
    line = "    "
    for i, val in enumerate(values):
        line += fmt.format(val)
        if i + 1 < values.size:
            line += ", "
        if (i + 1) % per_line == 0 and i + 1 < values.size:
            parts.append(line)
            line = "    "
    if line.strip():
        parts.append(line)
    parts.append("};\n")
    return "\n".join(parts)


def write_header(
    data32: dict[str, np.ndarray],
    data64: dict[str, np.ndarray],
    bias_scalar32: dict[str, np.ndarray],
    bias_vector32: dict[str, np.ndarray],
    bias_scalar64: dict[str, np.ndarray],
    bias_vector64: dict[str, np.ndarray],
) -> None:
    header = [
        "// Generated via tests/golden/generate_unary_numpy.py using NumPy",
        "#pragma once",
        "#include <stddef.h>",
        "",
        "typedef struct {",
        "    size_t length;",
        "    float elu_alpha;",
        "    float selu_alpha;",
        "    float selu_lambda;",
        "    float leaky_slope;",
        "    float prelu_slope;",
        "    float bias_scalar;",
        "    double bias_scalar_f64;",
        "    const float *bias_vector;",
        "    const double *bias_vector_f64;",
        "    const float *input;",
        "    const float *relu;",
        "    const float *gelu;",
        "    const float *gelu_tanh;",
        "    const float *silu;",
        "    const float *sigmoid;",
        "    const float *tanh_v;",
        "    const float *mish;",
        "    const float *elu;",
        "    const float *selu;",
        "    const float *leaky_relu;",
        "    const float *prelu;",
        "    const float *relu_bias_scalar;",
        "    const float *relu_bias_vector;",
        "    const float *gelu_bias_scalar;",
        "    const float *gelu_bias_vector;",
        "    const float *gelu_tanh_bias_scalar;",
        "    const float *gelu_tanh_bias_vector;",
        "    const float *silu_bias_scalar;",
        "    const float *silu_bias_vector;",
        "    const float *sigmoid_bias_scalar;",
        "    const float *sigmoid_bias_vector;",
        "    const float *tanh_v_bias_scalar;",
        "    const float *tanh_v_bias_vector;",
        "    const float *mish_bias_scalar;",
        "    const float *mish_bias_vector;",
        "    const float *elu_bias_scalar;",
        "    const float *elu_bias_vector;",
        "    const float *selu_bias_scalar;",
        "    const float *selu_bias_vector;",
        "    const float *leaky_relu_bias_scalar;",
        "    const float *leaky_relu_bias_vector;",
        "    const float *prelu_bias_scalar;",
        "    const float *prelu_bias_vector;",
        "    const double *input_f64;",
        "    const double *relu_f64;",
        "    const double *gelu_f64;",
        "    const double *gelu_tanh_f64;",
        "    const double *silu_f64;",
        "    const double *sigmoid_f64;",
        "    const double *tanh_v_f64;",
        "    const double *mish_f64;",
        "    const double *elu_f64;",
        "    const double *selu_f64;",
        "    const double *leaky_relu_f64;",
        "    const double *prelu_f64;",
        "    const double *relu_bias_scalar_f64;",
        "    const double *relu_bias_vector_f64;",
        "    const double *gelu_bias_scalar_f64;",
        "    const double *gelu_bias_vector_f64;",
        "    const double *gelu_tanh_bias_scalar_f64;",
        "    const double *gelu_tanh_bias_vector_f64;",
        "    const double *silu_bias_scalar_f64;",
        "    const double *silu_bias_vector_f64;",
        "    const double *sigmoid_bias_scalar_f64;",
        "    const double *sigmoid_bias_vector_f64;",
        "    const double *tanh_v_bias_scalar_f64;",
        "    const double *tanh_v_bias_vector_f64;",
        "    const double *mish_bias_scalar_f64;",
        "    const double *mish_bias_vector_f64;",
        "    const double *elu_bias_scalar_f64;",
        "    const double *elu_bias_vector_f64;",
        "    const double *selu_bias_scalar_f64;",
        "    const double *selu_bias_vector_f64;",
        "    const double *leaky_relu_bias_scalar_f64;",
        "    const double *leaky_relu_bias_vector_f64;",
        "    const double *prelu_bias_scalar_f64;",
        "    const double *prelu_bias_vector_f64;",
        "} numpy_activation_golden_t;",
        "",
    ]

    base_order = [
        ("input", "input"),
        ("relu", "relu"),
        ("gelu", "gelu"),
        ("gelu_tanh", "gelu_tanh"),
        ("silu", "silu"),
        ("sigmoid", "sigmoid"),
        ("tanh", "tanh_v"),
        ("mish", "mish"),
        ("elu", "elu"),
        ("selu", "selu"),
        ("leaky", "leaky_relu"),
        ("prelu", "prelu"),
    ]

    bias_order = [
        ("relu", "relu"),
        ("gelu", "gelu"),
        ("gelu_tanh", "gelu_tanh"),
        ("silu", "silu"),
        ("sigmoid", "sigmoid"),
        ("tanh", "tanh_v"),
        ("mish", "mish"),
        ("elu", "elu"),
        ("selu", "selu"),
        ("leaky", "leaky_relu"),
        ("prelu", "prelu"),
    ]

    header.append(format_array("g_numpy_activation_bias_vector", BIAS_VECTOR.astype(np.float32), "float", "{:.9f}f"))
    header.append(format_array("g_numpy_activation_bias_vector_f64", BIAS_VECTOR, "double", "{:.17f}", per_line=6))

    for key, suffix in base_order:
        header.append(format_array(f"g_numpy_activation_{suffix}", data32[key], "float", "{:.9f}f", per_line=8))
    for key, suffix in base_order:
        header.append(
            format_array(f"g_numpy_activation_{suffix}_f64", data64[key], "double", "{:.17f}", per_line=6)
        )
    for key, suffix in bias_order:
        header.append(
            format_array(
                f"g_numpy_activation_{suffix}_bias_scalar",
                bias_scalar32[key],
                "float",
                "{:.9f}f",
                per_line=8,
            )
        )
        header.append(
            format_array(
                f"g_numpy_activation_{suffix}_bias_vector",
                bias_vector32[key],
                "float",
                "{:.9f}f",
                per_line=8,
            )
        )
    for key, suffix in bias_order:
        header.append(
            format_array(
                f"g_numpy_activation_{suffix}_bias_scalar_f64",
                bias_scalar64[key],
                "double",
                "{:.17f}",
                per_line=6,
            )
        )
        header.append(
            format_array(
                f"g_numpy_activation_{suffix}_bias_vector_f64",
                bias_vector64[key],
                "double",
                "{:.17f}",
                per_line=6,
            )
        )

    header.append("static const numpy_activation_golden_t g_numpy_activation_golden = {")
    header.extend(
        [
            f"    .length = {data32['input'].size},",
            f"    .elu_alpha = {float(ELU_ALPHA):.9f}f,",
            f"    .selu_alpha = {float(SELU_ALPHA):.9f}f,",
            f"    .selu_lambda = {float(SELU_LAMBDA):.9f}f,",
            f"    .leaky_slope = {float(LEAKY_SLOPE):.9f}f,",
            f"    .prelu_slope = {float(PRELU_SLOPE):.9f}f,",
            f"    .bias_scalar = {float(BIAS_SCALAR):.9f}f,",
            f"    .bias_scalar_f64 = {float(BIAS_SCALAR):.17f},",
            "    .bias_vector = g_numpy_activation_bias_vector,",
            "    .bias_vector_f64 = g_numpy_activation_bias_vector_f64,",
        ]
    )
    for key, suffix in base_order:
        header.append(f"    .{suffix} = g_numpy_activation_{suffix},")
    for key, suffix in base_order:
        header.append(f"    .{suffix}_f64 = g_numpy_activation_{suffix}_f64,")
    for key, suffix in bias_order:
        header.append(f"    .{suffix}_bias_scalar = g_numpy_activation_{suffix}_bias_scalar,")
        header.append(f"    .{suffix}_bias_vector = g_numpy_activation_{suffix}_bias_vector,")
    for key, suffix in bias_order:
        header.append(f"    .{suffix}_bias_scalar_f64 = g_numpy_activation_{suffix}_bias_scalar_f64,")
        header.append(f"    .{suffix}_bias_vector_f64 = g_numpy_activation_{suffix}_bias_vector_f64,")
    header.append("};\n")

    OUTPUT_PATH.write_text("\n".join(header))


def main() -> None:
    data32, data64, scalar32, vector32, scalar64, vector64 = generate()
    write_header(data32, data64, scalar32, vector32, scalar64, vector64)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
