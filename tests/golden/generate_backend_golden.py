#!/usr/bin/env python3
"""Generate golden data for backend tests using NumPy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER_PATH = ROOT / "backend" / "golden_data.h"


def round_half_away_from_zero(values: np.ndarray) -> np.ndarray:
    """Replicates C roundf() semantics (ties away from zero)."""
    return np.sign(values) * np.floor(np.abs(values) + 0.5)


def round_positive_nearest(value: float) -> int:
    """Round non-negative float to nearest integer, ties to nearest even as lrintf."""
    floor_val = math.floor(value)
    frac = value - floor_val
    if frac > 0.5:
        return floor_val + 1
    if frac < 0.5:
        return floor_val
    # tie: round to even
    return floor_val + (floor_val & 1)


def fp8_quantize_scalar(
    value: float, exp_bits: int, mant_bits: int, bias: int, has_infinity: bool, max_finite: float
) -> int:
    if math.isnan(value):
        payload = 1
        exp_mask = (1 << exp_bits) - 1
        return (0 << 7) | (exp_mask << mant_bits) | payload

    sign = 1 if value < 0 else 0
    abs_value = abs(value)
    sign_bit = sign << 7
    exp_mask = (1 << exp_bits) - 1
    max_exponent = exp_mask - 1
    mant_mask = (1 << mant_bits) - 1

    if math.isinf(value) or abs_value > max_finite:
        if has_infinity:
            return sign_bit | (exp_mask << mant_bits)
        return sign_bit | (max_exponent << mant_bits) | mant_mask

    if abs_value == 0.0:
        return sign_bit

    mantissa, exponent = math.frexp(abs_value)  # mantissa in [0.5, 1)
    mantissa *= 2.0
    exponent -= 1

    fp_exp = exponent + bias
    if 0 < fp_exp < (max_exponent + 1):
        scaled = (mantissa - 1.0) * (1 << mant_bits)
        rounded = round_positive_nearest(scaled)
        if rounded == (1 << mant_bits):
            rounded = 0
            fp_exp += 1
            if fp_exp >= (max_exponent + 1):
                if has_infinity:
                    return sign_bit | (exp_mask << mant_bits)
                return sign_bit | (max_exponent << mant_bits) | mant_mask
        return sign_bit | ((fp_exp & exp_mask) << mant_bits) | (rounded & mant_mask)

    emin = 1 - bias
    scaled = math.ldexp(abs_value, -emin + mant_bits)
    rounded = round_positive_nearest(scaled)
    if rounded == 0:
        return sign_bit
    if rounded > mant_mask:
        rounded = mant_mask
    return sign_bit | (rounded & mant_mask)


def fp8_dequantize_scalar(bits: int, exp_bits: int, mant_bits: int, bias: int, has_infinity: bool) -> float:
    sign = (bits >> 7) & 0x1
    exp = (bits >> mant_bits) & ((1 << exp_bits) - 1)
    mant = bits & ((1 << mant_bits) - 1)
    exp_mask = (1 << exp_bits) - 1
    sign_scale = -1.0 if sign else 1.0

    if exp == exp_mask:
        if has_infinity and mant == 0:
            return -math.inf if sign else math.inf
        return math.nan

    if exp == 0:
        if mant == 0:
            return -0.0 if sign else 0.0
        fraction = mant / float(1 << mant_bits)
        emin = 1 - bias
        return sign_scale * math.ldexp(fraction, emin)

    fraction = 1.0 + mant / float(1 << mant_bits)
    real_exp = exp - bias
    return sign_scale * math.ldexp(fraction, real_exp)


def fp8_quantize_array(
    values: np.ndarray, exp_bits: int, mant_bits: int, bias: int, has_infinity: bool, max_finite: float
) -> np.ndarray:
    return np.array(
        [fp8_quantize_scalar(float(v), exp_bits, mant_bits, bias, has_infinity, max_finite) for v in values],
        dtype=np.uint8,
    )


def fp8_dequantize_array(bits: np.ndarray, exp_bits: int, mant_bits: int, bias: int, has_infinity: bool) -> np.ndarray:
    return np.array(
        [fp8_dequantize_scalar(int(b), exp_bits, mant_bits, bias, has_infinity) for b in bits], dtype=np.float64
    )


def c_mod_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = []
    for ai, bi in zip(a, b):
        trunc = math.trunc(ai / bi)
        result.append(ai - trunc * bi)
    return np.array(result, dtype=a.dtype)


def c_div_array(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    values = []
    for ai, bi in zip(a, b):
        values.append(type(ai)(int(ai / bi)))
    return np.array(values, dtype=a.dtype)


def int_pow_array(a: np.ndarray, e: np.ndarray) -> np.ndarray:
    values = []
    for base, exp in zip(a, e):
        if exp < 0:
            if base == 1:
                values.append(type(base)(1))
            elif base == -1:
                values.append(type(base)(-1 if (exp & 1) else 1))
            else:
                values.append(type(base)(0))
            continue
        result = type(base)(1)
        factor = type(base)(base)
        exponent = int(exp)
        while exponent > 0:
            if exponent & 1:
                result = type(base)(result * factor)
            exponent >>= 1
            if exponent:
                factor = type(base)(factor * factor)
        values.append(result)
    return np.array(values, dtype=a.dtype)


def shift_left_signed(values: np.ndarray, amounts: np.ndarray, bits: int) -> np.ndarray:
    out = []
    for v, amt in zip(values, amounts):
        if amt < 0 or amt >= bits:
            out.append(type(v)(0))
        else:
            out.append(type(v)(v << amt))
    return np.array(out, dtype=values.dtype)


def shift_right_signed(values: np.ndarray, amounts: np.ndarray, bits: int) -> np.ndarray:
    out = []
    for v, amt in zip(values, amounts):
        if amt < 0 or amt >= bits:
            out.append(type(v)(-1 if v < 0 else 0))
        else:
            out.append(type(v)(v >> amt))
    return np.array(out, dtype=values.dtype)


def shift_right_logical_signed(values: np.ndarray, amounts: np.ndarray, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    vals = values.astype(np.int64, copy=False)
    result = np.zeros(vals.shape, dtype=np.uint64)
    valid = (amounts >= 0) & (amounts < bits)
    if np.any(valid):
        masked = np.bitwise_and(vals, mask)
        shifted = masked[valid].astype(np.uint64) >> amounts[valid].astype(np.uint64)
        result[valid] = shifted
    return result


def shift_left_unsigned(values: np.ndarray, amounts: np.ndarray, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    out = []
    for v, amt in zip(values, amounts):
        if amt < 0 or amt >= bits:
            out.append(0)
        else:
            out.append(((int(v) << amt) & mask))
    dtype = np.uint64 if bits == 64 else np.uint32
    return np.array(out, dtype=dtype)


def shift_right_unsigned(values: np.ndarray, amounts: np.ndarray, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    out = []
    for v, amt in zip(values, amounts):
        if amt < 0 or amt >= bits:
            out.append(0)
        else:
            out.append((int(v) & mask) >> amt)
    dtype = np.uint64 if bits == 64 else np.uint32
    return np.array(out, dtype=dtype)


def abs_int_array(values: np.ndarray, bits: int) -> np.ndarray:
    min_val = -(1 << (bits - 1))
    result = []
    for v in values:
        if v == min_val:
            result.append(v)
        else:
            result.append(-v if v < 0 else v)
    return np.array(result, dtype=values.dtype)


def neg_int_array(values: np.ndarray, bits: int) -> np.ndarray:
    min_val = -(1 << (bits - 1))
    result = []
    for v in values:
        if v == min_val:
            result.append(v)
        else:
            result.append(-v)
    return np.array(result, dtype=values.dtype)


def sign_int_array(values: np.ndarray) -> np.ndarray:
    return np.array([(1 if v > 0 else (-1 if v < 0 else 0)) for v in values], dtype=values.dtype)


def sign_uint_array(values: np.ndarray) -> np.ndarray:
    return np.array([1 if v > 0 else 0 for v in values], dtype=values.dtype)


def float_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    f32 = np.asarray(values, dtype=np.float32)
    bits = f32.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + 0x7FFF + lsb
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    bits32 = (np.asarray(bits, dtype=np.uint32) << 16).astype(np.uint32)
    return bits32.view(np.float32).astype(np.float64)


def fmt_floats(values: Sequence[float]) -> str:
    return ", ".join(f"{float(v):.10g}" for v in values)


def fmt_doubles(values: Sequence[float]) -> str:
    return ", ".join(f"{float(v):.17g}" for v in values)


def fmt_ints(values: Sequence[int]) -> str:
    return ", ".join(str(int(v)) for v in values)


def fmt_uints(values: Sequence[int]) -> str:
    return ", ".join(str(int(v)) + "u" for v in values)


def fmt_u64s(values: Sequence[int]) -> str:
    return ", ".join(f"UINT64_C({int(v)})" for v in values)


def make_elementwise_section() -> str:
    lines: list[str] = []

    # Float32 dataset
    a_fp32 = np.array([-2.0, -0.5, 1.0, 3.5], dtype=np.float64)
    b_fp32 = np.array([4.0, -1.5, 2.0, -0.5], dtype=np.float64)
    add = a_fp32 + b_fp32
    sub = a_fp32 - b_fp32
    mul = a_fp32 * b_fp32
    div = a_fp32 / b_fp32
    minimum = np.minimum(a_fp32, b_fp32)
    maximum = np.maximum(a_fp32, b_fp32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float a[4];")
    lines.append("    float b[4];")
    lines.append("    float add[4];")
    lines.append("    float sub[4];")
    lines.append("    float mul[4];")
    lines.append("    float div[4];")
    lines.append("    float minv[4];")
    lines.append("    float maxv[4];")
    lines.append("    double a_f64[4];")
    lines.append("    double b_f64[4];")
    lines.append("    double add_f64[4];")
    lines.append("    double sub_f64[4];")
    lines.append("    double mul_f64[4];")
    lines.append("    double div_f64[4];")
    lines.append("    double minv_f64[4];")
    lines.append("    double maxv_f64[4];")
    lines.append("} g_elementwise_fp32 = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(a_fp32)}}},")
    lines.append(f"    {{{fmt_floats(b_fp32)}}},")
    lines.append(f"    {{{fmt_floats(add)}}},")
    lines.append(f"    {{{fmt_floats(sub)}}},")
    lines.append(f"    {{{fmt_floats(mul)}}},")
    lines.append(f"    {{{fmt_floats(div)}}},")
    lines.append(f"    {{{fmt_floats(minimum)}}},")
    lines.append(f"    {{{fmt_floats(maximum)}}},")
    lines.append(f"    {{{fmt_doubles(a_fp32)}}},")
    lines.append(f"    {{{fmt_doubles(b_fp32)}}},")
    lines.append(f"    {{{fmt_doubles(add)}}},")
    lines.append(f"    {{{fmt_doubles(sub)}}},")
    lines.append(f"    {{{fmt_doubles(mul)}}},")
    lines.append(f"    {{{fmt_doubles(div)}}},")
    lines.append(f"    {{{fmt_doubles(minimum)}}},")
    lines.append(f"    {{{fmt_doubles(maximum)}}},")
    lines.append("};\n")

    a_fp32_wide = np.linspace(-8.0, 8.0, 17, dtype=np.float64)
    b_fp32_wide = np.linspace(6.75, -5.75, 17, dtype=np.float64)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float a[17];")
    lines.append("    float b[17];")
    lines.append("    float add[17];")
    lines.append("    float sub[17];")
    lines.append("    float mul[17];")
    lines.append("    float div[17];")
    lines.append("    float minv[17];")
    lines.append("    float maxv[17];")
    lines.append("} g_elementwise_fp32_wide = {")
    lines.append("    17,")
    lines.append(f"    {{{fmt_floats(a_fp32_wide)}}},")
    lines.append(f"    {{{fmt_floats(b_fp32_wide)}}},")
    lines.append(f"    {{{fmt_floats(a_fp32_wide + b_fp32_wide)}}},")
    lines.append(f"    {{{fmt_floats(a_fp32_wide - b_fp32_wide)}}},")
    lines.append(f"    {{{fmt_floats(a_fp32_wide * b_fp32_wide)}}},")
    lines.append(f"    {{{fmt_floats(a_fp32_wide / b_fp32_wide)}}},")
    lines.append(f"    {{{fmt_floats(np.minimum(a_fp32_wide, b_fp32_wide))}}},")
    lines.append(f"    {{{fmt_floats(np.maximum(a_fp32_wide, b_fp32_wide))}}},")
    lines.append("};\n")

    pow_base_fp32 = np.array([1.5, 2.0, 3.25, 4.5], dtype=np.float64)
    pow_exponent_fp32 = np.array([2.0, 3.0, 0.5, -1.0], dtype=np.float64)
    mod_div_fp32 = np.array([0.75, 2.5, 1.25, 2.0], dtype=np.float64)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float base[4];")
    lines.append("    float exponent[4];")
    lines.append("    float powv[4];")
    lines.append("    float divisor[4];")
    lines.append("    float modv[4];")
    lines.append("    double base_f64[4];")
    lines.append("    double exponent_f64[4];")
    lines.append("    double powv_f64[4];")
    lines.append("    double divisor_f64[4];")
    lines.append("    double modv_f64[4];")
    lines.append("} g_elementwise_powmod_fp32 = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(pow_base_fp32)}}},")
    lines.append(f"    {{{fmt_floats(pow_exponent_fp32)}}},")
    lines.append(f"    {{{fmt_floats(np.power(pow_base_fp32, pow_exponent_fp32))}}},")
    lines.append(f"    {{{fmt_floats(mod_div_fp32)}}},")
    lines.append(f"    {{{fmt_floats(np.fmod(pow_base_fp32, mod_div_fp32))}}},")
    lines.append(f"    {{{fmt_doubles(pow_base_fp32)}}},")
    lines.append(f"    {{{fmt_doubles(pow_exponent_fp32)}}},")
    lines.append(f"    {{{fmt_doubles(np.power(pow_base_fp32, pow_exponent_fp32))}}},")
    lines.append(f"    {{{fmt_doubles(mod_div_fp32)}}},")
    lines.append(f"    {{{fmt_doubles(np.fmod(pow_base_fp32, mod_div_fp32))}}},")
    lines.append("};\n")

    # Float16 dataset (store expected in float32 for comparison)
    a_fp16 = np.array([-3.0, -1.0, 0.0, 0.5, 1.75], dtype=np.float64)
    b_fp16 = np.array([2.0, -0.5, 4.0, 1.5, -2.0], dtype=np.float64)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float a[5];")
    lines.append("    float b[5];")
    lines.append("    float add[5];")
    lines.append("    float sub[5];")
    lines.append("    float mul[5];")
    lines.append("    float div[5];")
    lines.append("    float minv[5];")
    lines.append("    float maxv[5];")
    lines.append("} g_elementwise_fp16 = {")
    lines.append("    5,")
    lines.append(f"    {{{fmt_floats(a_fp16)}}},")
    lines.append(f"    {{{fmt_floats(b_fp16)}}},")
    lines.append(f"    {{{fmt_floats(a_fp16 + b_fp16)}}},")
    lines.append(f"    {{{fmt_floats(a_fp16 - b_fp16)}}},")
    lines.append(f"    {{{fmt_floats(a_fp16 * b_fp16)}}},")
    lines.append(f"    {{{fmt_floats(a_fp16 / b_fp16)}}},")
    lines.append(f"    {{{fmt_floats(np.minimum(a_fp16, b_fp16))}}},")
    lines.append(f"    {{{fmt_floats(np.maximum(a_fp16, b_fp16))}}},")
    lines.append("};\n")

    pow_base_fp16 = np.array([0.75, 1.5, 2.0, 2.5], dtype=np.float64)
    pow_exponent_fp16 = np.array([2.0, 1.5, 0.5, -1.0], dtype=np.float64)
    mod_div_fp16 = np.array([0.5, 0.75, 1.0, 1.25], dtype=np.float64)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float base[4];")
    lines.append("    float exponent[4];")
    lines.append("    float powv[4];")
    lines.append("    float divisor[4];")
    lines.append("    float modv[4];")
    lines.append("} g_elementwise_powmod_fp16 = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(pow_base_fp16)}}},")
    lines.append(f"    {{{fmt_floats(pow_exponent_fp16)}}},")
    lines.append(f"    {{{fmt_floats(np.power(pow_base_fp16, pow_exponent_fp16))}}},")
    lines.append(f"    {{{fmt_floats(mod_div_fp16)}}},")
    lines.append(f"    {{{fmt_floats(np.fmod(pow_base_fp16, mod_div_fp16))}}},")
    lines.append("};\n")

    # BF16 dataset
    a_bf16 = np.array([0.125, -0.25, 1.5], dtype=np.float64)
    b_bf16 = np.array([2.5, -4.0, -0.75], dtype=np.float64)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float a[3];")
    lines.append("    float b[3];")
    lines.append("    float add[3];")
    lines.append("    float sub[3];")
    lines.append("    float mul[3];")
    lines.append("    float div[3];")
    lines.append("    float minv[3];")
    lines.append("    float maxv[3];")
    lines.append("} g_elementwise_bf16 = {")
    lines.append("    3,")
    lines.append(f"    {{{fmt_floats(a_bf16)}}},")
    lines.append(f"    {{{fmt_floats(b_bf16)}}},")
    lines.append(f"    {{{fmt_floats(a_bf16 + b_bf16)}}},")
    lines.append(f"    {{{fmt_floats(a_bf16 - b_bf16)}}},")
    lines.append(f"    {{{fmt_floats(a_bf16 * b_bf16)}}},")
    lines.append(f"    {{{fmt_floats(a_bf16 / b_bf16)}}},")
    lines.append(f"    {{{fmt_floats(np.minimum(a_bf16, b_bf16))}}},")
    lines.append(f"    {{{fmt_floats(np.maximum(a_bf16, b_bf16))}}},")
    lines.append("};\n")

    pow_base_bf16 = np.array([0.625, 1.25, 2.5], dtype=np.float64)
    pow_exponent_bf16 = np.array([3.0, 0.5, -1.0], dtype=np.float64)
    mod_div_bf16 = np.array([0.5, 0.75, 1.5], dtype=np.float64)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float base[3];")
    lines.append("    float exponent[3];")
    lines.append("    float powv[3];")
    lines.append("    float divisor[3];")
    lines.append("    float modv[3];")
    lines.append("} g_elementwise_powmod_bf16 = {")
    lines.append("    3,")
    lines.append(f"    {{{fmt_floats(pow_base_bf16)}}},")
    lines.append(f"    {{{fmt_floats(pow_exponent_bf16)}}},")
    lines.append(f"    {{{fmt_floats(np.power(pow_base_bf16, pow_exponent_bf16))}}},")
    lines.append(f"    {{{fmt_floats(mod_div_bf16)}}},")
    lines.append(f"    {{{fmt_floats(np.fmod(pow_base_bf16, mod_div_bf16))}}},")
    lines.append("};\n")

    # Int32 dataset
    i32_a = np.array([9, -8, 7, -6], dtype=np.int32)
    i32_b = np.array([3, -2, 5, -4], dtype=np.int32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    int32_t a[4];")
    lines.append("    int32_t b[4];")
    lines.append("    int32_t add[4];")
    lines.append("    int32_t sub[4];")
    lines.append("    int32_t mul[4];")
    lines.append("    int32_t div[4];")
    lines.append("    int32_t minv[4];")
    lines.append("    int32_t maxv[4];")
    lines.append("} g_elementwise_i32 = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_ints(i32_a)}}},")
    lines.append(f"    {{{fmt_ints(i32_b)}}},")
    lines.append(f"    {{{fmt_ints(i32_a + i32_b)}}},")
    lines.append(f"    {{{fmt_ints(i32_a - i32_b)}}},")
    lines.append(f"    {{{fmt_ints(i32_a * i32_b)}}},")
    lines.append(f"    {{{fmt_ints(c_div_array(i32_a, i32_b))}}},")
    lines.append(f"    {{{fmt_ints(np.minimum(i32_a, i32_b))}}},")
    lines.append(f"    {{{fmt_ints(np.maximum(i32_a, i32_b))}}},")
    lines.append("};\n")

    i32_wide_len = 24
    i32_a_wide = (np.arange(i32_wide_len, dtype=np.int32) * 3 - 34).astype(np.int32)
    i32_b_wide = (np.arange(i32_wide_len, dtype=np.int32) * 2 + 5).astype(np.int32)
    i32_b_wide[1::2] *= -1
    shift_amt_wide = ((np.arange(i32_wide_len, dtype=np.int32) * 5 + 3) % 32).astype(np.uint32)
    shift_left_wide = shift_left_signed(i32_a_wide, shift_amt_wide, 32)
    shift_right_wide = shift_right_signed(i32_a_wide, shift_amt_wide, 32)
    shift_right_log_wide = shift_right_unsigned(i32_a_wide.astype(np.int64), shift_amt_wide, 32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    int32_t a[24];")
    lines.append("    int32_t b[24];")
    lines.append("    int32_t add[24];")
    lines.append("    int32_t sub[24];")
    lines.append("    int32_t mul[24];")
    lines.append("    int32_t div[24];")
    lines.append("    int32_t minv[24];")
    lines.append("    int32_t maxv[24];")
    lines.append("    uint32_t shift_amt[24];")
    lines.append("    int32_t shift_left[24];")
    lines.append("    int32_t shift_right[24];")
    lines.append("    uint32_t shift_right_logical[24];")
    lines.append("} g_elementwise_i32_wide = {")
    lines.append("    24,")
    lines.append(f"    {{{fmt_ints(i32_a_wide)}}},")
    lines.append(f"    {{{fmt_ints(i32_b_wide)}}},")
    lines.append(f"    {{{fmt_ints(i32_a_wide + i32_b_wide)}}},")
    lines.append(f"    {{{fmt_ints(i32_a_wide - i32_b_wide)}}},")
    lines.append(f"    {{{fmt_ints(i32_a_wide * i32_b_wide)}}},")
    lines.append(f"    {{{fmt_ints(c_div_array(i32_a_wide, i32_b_wide))}}},")
    lines.append(f"    {{{fmt_ints(np.minimum(i32_a_wide, i32_b_wide))}}},")
    lines.append(f"    {{{fmt_ints(np.maximum(i32_a_wide, i32_b_wide))}}},")
    lines.append(f"    {{{fmt_uints(shift_amt_wide)}}},")
    lines.append(f"    {{{fmt_ints(shift_left_wide)}}},")
    lines.append(f"    {{{fmt_ints(shift_right_wide)}}},")
    lines.append(f"    {{{fmt_uints(shift_right_log_wide)}}},")
    lines.append("};\n")

    pow_base_i32 = np.array([2, -3, 4, -5], dtype=np.int32)
    pow_exponent_i32 = np.array([3, 2, 0, 1], dtype=np.int32)
    pow_values_i32 = int_pow_array(pow_base_i32, pow_exponent_i32)
    mod_lhs_i32 = np.array([17, -19, 23, -25], dtype=np.int32)
    mod_rhs_i32 = np.array([5, 4, 7, 6], dtype=np.int32)
    mod_values_i32 = c_mod_array(mod_lhs_i32, mod_rhs_i32)
    shift_lhs_i32 = np.array([64, -64, 7, -9], dtype=np.int32)
    shift_amounts_i32 = np.array([1, 2, 3, 4], dtype=np.int32)
    shift_left_i32 = shift_left_signed(shift_lhs_i32, shift_amounts_i32, 32)
    shift_right_i32 = shift_right_signed(shift_lhs_i32, shift_amounts_i32, 32)
    shift_right_log_i32 = shift_right_logical_signed(shift_lhs_i32, shift_amounts_i32, 32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    int32_t pow_base[4];")
    lines.append("    int32_t pow_exp[4];")
    lines.append("    int32_t powv[4];")
    lines.append("    int32_t mod_lhs[4];")
    lines.append("    int32_t mod_rhs[4];")
    lines.append("    int32_t modv[4];")
    lines.append("    int32_t shift_lhs[4];")
    lines.append("    uint32_t shift_amt[4];")
    lines.append("    int32_t shift_left[4];")
    lines.append("    int32_t shift_right[4];")
    lines.append("    uint32_t shift_right_logical[4];")
    lines.append("} g_elementwise_i32_ext = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_ints(pow_base_i32)}}},")
    lines.append(f"    {{{fmt_ints(pow_exponent_i32)}}},")
    lines.append(f"    {{{fmt_ints(pow_values_i32)}}},")
    lines.append(f"    {{{fmt_ints(mod_lhs_i32)}}},")
    lines.append(f"    {{{fmt_ints(mod_rhs_i32)}}},")
    lines.append(f"    {{{fmt_ints(mod_values_i32)}}},")
    lines.append(f"    {{{fmt_ints(shift_lhs_i32)}}},")
    lines.append(f"    {{{fmt_uints(shift_amounts_i32)}}},")
    lines.append(f"    {{{fmt_ints(shift_left_i32)}}},")
    lines.append(f"    {{{fmt_ints(shift_right_i32)}}},")
    lines.append(f"    {{{fmt_uints(shift_right_log_i32)}}},")
    lines.append("};\n")

    # Unsigned 32-bit dataset for bitwise
    u32_a = np.array([0x0F, 0xF0, 0xAAAA, 0x3333], dtype=np.uint32)
    u32_b = np.array([0x33, 0x0F, 0x5555, 0xCCCC], dtype=np.uint32)
    shift_amounts_u32 = np.array([1, 2, 3, 4], dtype=np.uint32)
    shift_left_u32 = shift_left_unsigned(u32_a, shift_amounts_u32, 32)
    shift_right_u32 = shift_right_unsigned(u32_a, shift_amounts_u32, 32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    uint32_t a[4];")
    lines.append("    uint32_t b[4];")
    lines.append("    uint32_t band[4];")
    lines.append("    uint32_t bor[4];")
    lines.append("    uint32_t bxor[4];")
    lines.append("    uint32_t shift_amt[4];")
    lines.append("    uint32_t shl[4];")
    lines.append("    uint32_t shr[4];")
    lines.append("} g_elementwise_bitwise_u32 = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_uints(u32_a)}}},")
    lines.append(f"    {{{fmt_uints(u32_b)}}},")
    lines.append(f"    {{{fmt_uints(np.bitwise_and(u32_a, u32_b))}}},")
    lines.append(f"    {{{fmt_uints(np.bitwise_or(u32_a, u32_b))}}},")
    lines.append(f"    {{{fmt_uints(np.bitwise_xor(u32_a, u32_b))}}},")
    lines.append(f"    {{{fmt_uints(shift_amounts_u32)}}},")
    lines.append(f"    {{{fmt_uints(shift_left_u32)}}},")
    lines.append(f"    {{{fmt_uints(shift_right_u32)}}},")
    lines.append("};\n")

    u32_len_wide = 20
    u32_a_wide = np.array(
        [
            0x13579BDF,
            0x2468ACE0,
            0xFEDCBA98,
            0x89ABCDEF,
            0x01234567,
            0xA5A5A5A5,
            0x5A5A5A5A,
            0xFF00FF00,
            0x00FF00FF,
            0x0F0F0F0F,
            0xF0F0F0F0,
            0x33333333,
            0xCCCCCCCC,
            0x12345678,
            0xDEADBEEF,
            0xBEADF00D,
            0x7FFFFFFF,
            0x80000001,
            0x6DB6DB6D,
            0xDB6DB6DB,
        ],
        dtype=np.uint32,
    )
    u32_b_wide = np.roll(u32_a_wide, 3) ^ 0xAAAAAAAA
    shift_amt_u32_wide = (np.arange(u32_len_wide, dtype=np.uint32) * 3) % 32
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    uint32_t a[20];")
    lines.append("    uint32_t b[20];")
    lines.append("    uint32_t band[20];")
    lines.append("    uint32_t bor[20];")
    lines.append("    uint32_t bxor[20];")
    lines.append("    uint32_t shift_amt[20];")
    lines.append("    uint32_t shl[20];")
    lines.append("    uint32_t shr[20];")
    lines.append("} g_elementwise_bitwise_u32_wide = {")
    lines.append("    20,")
    lines.append(f"    {{{fmt_uints(u32_a_wide)}}},")
    lines.append(f"    {{{fmt_uints(u32_b_wide)}}},")
    lines.append(f"    {{{fmt_uints(np.bitwise_and(u32_a_wide, u32_b_wide))}}},")
    lines.append(f"    {{{fmt_uints(np.bitwise_or(u32_a_wide, u32_b_wide))}}},")
    lines.append(f"    {{{fmt_uints(np.bitwise_xor(u32_a_wide, u32_b_wide))}}},")
    lines.append(f"    {{{fmt_uints(shift_amt_u32_wide)}}},")
    lines.append(f"    {{{fmt_uints(shift_left_unsigned(u32_a_wide, shift_amt_u32_wide, 32))}}},")
    lines.append(f"    {{{fmt_uints(shift_right_unsigned(u32_a_wide, shift_amt_u32_wide, 32))}}},")
    lines.append("};\n")

    unary_affine = np.array([-3.5, -1.25, 0.0, 0.75, 2.5], dtype=np.float64)
    unary_math = np.array([0.25, 1.0, 2.0, 4.0], dtype=np.float64)
    lines.append(
        "typedef struct {\n"
        "    size_t len_affine;\n"
        "    float values_affine[5];\n"
        "    float absv[5];\n"
        "    float negv[5];\n"
        "    float signv[5];\n"
        "    size_t len_math;\n"
        "    float values_math[4];\n"
        "    float sqrtv[4];\n"
        "    float expv[4];\n"
        "    float logv[4];\n"
        "} llama_unary_float_golden_t;\n\n"
    )
    lines.append("static const llama_unary_float_golden_t g_unary_fp32 = {")
    lines.append("    5,")
    lines.append(f"    {{{fmt_floats(unary_affine)}}},")
    lines.append(f"    {{{fmt_floats(np.abs(unary_affine))}}},")
    lines.append(f"    {{{fmt_floats(-unary_affine)}}},")
    lines.append(f"    {{{fmt_floats(np.sign(unary_affine))}}},")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(unary_math)}}},")
    lines.append(f"    {{{fmt_floats(np.sqrt(unary_math))}}},")
    lines.append(f"    {{{fmt_floats(np.exp(unary_math))}}},")
    lines.append(f"    {{{fmt_floats(np.log(unary_math))}}},")
    lines.append("};\n")

#if MARMOT_ENABLE_FP8 equivalent dataset emitted regardless
    lines.append("static const llama_unary_float_golden_t g_unary_fp8 = {")
    lines.append("    5,")
    lines.append(f"    {{{fmt_floats(unary_affine)}}},")
    lines.append(f"    {{{fmt_floats(np.abs(unary_affine))}}},")
    lines.append(f"    {{{fmt_floats(-unary_affine)}}},")
    lines.append(f"    {{{fmt_floats(np.sign(unary_affine))}}},")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(unary_math)}}},")
    lines.append(f"    {{{fmt_floats(np.sqrt(unary_math))}}},")
    lines.append(f"    {{{fmt_floats(np.exp(unary_math))}}},")
    lines.append(f"    {{{fmt_floats(np.log(unary_math))}}},")
    lines.append("};\n")

    lines.append("static const llama_unary_float_golden_t g_unary_fp16 = {")
    lines.append("    5,")
    lines.append(f"    {{{fmt_floats(unary_affine)}}},")
    lines.append(f"    {{{fmt_floats(np.abs(unary_affine))}}},")
    lines.append(f"    {{{fmt_floats(-unary_affine)}}},")
    lines.append(f"    {{{fmt_floats(np.sign(unary_affine))}}},")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(unary_math)}}},")
    lines.append(f"    {{{fmt_floats(np.sqrt(unary_math))}}},")
    lines.append(f"    {{{fmt_floats(np.exp(unary_math))}}},")
    lines.append(f"    {{{fmt_floats(np.log(unary_math))}}},")
    lines.append("};\n")

    lines.append("static const llama_unary_float_golden_t g_unary_bf16 = {")
    lines.append("    5,")
    lines.append(f"    {{{fmt_floats(unary_affine)}}},")
    lines.append(f"    {{{fmt_floats(np.abs(unary_affine))}}},")
    lines.append(f"    {{{fmt_floats(-unary_affine)}}},")
    lines.append(f"    {{{fmt_floats(np.sign(unary_affine))}}},")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(unary_math)}}},")
    lines.append(f"    {{{fmt_floats(np.sqrt(unary_math))}}},")
    lines.append(f"    {{{fmt_floats(np.exp(unary_math))}}},")
    lines.append(f"    {{{fmt_floats(np.log(unary_math))}}},")
    lines.append("};\n")

    int_unary = np.array([-9, -1, 0, 7, 15], dtype=np.int32)
    abs_int_vals = abs_int_array(int_unary, 32)
    neg_int_vals = neg_int_array(int_unary, 32)
    sign_int_vals = sign_int_array(int_unary)
    not_int_vals = np.bitwise_not(int_unary).astype(np.int32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    int32_t values[5];")
    lines.append("    int32_t absv[5];")
    lines.append("    int32_t negv[5];")
    lines.append("    int32_t signv[5];")
    lines.append("    int32_t bitwise_not[5];")
    lines.append("} g_unary_i32 = {")
    lines.append("    5,")
    lines.append(f"    {{{fmt_ints(int_unary)}}},")
    lines.append(f"    {{{fmt_ints(abs_int_vals)}}},")
    lines.append(f"    {{{fmt_ints(neg_int_vals)}}},")
    lines.append(f"    {{{fmt_ints(sign_int_vals)}}},")
    lines.append(f"    {{{fmt_ints(not_int_vals)}}},")
    lines.append("};\n")

    uint_unary = np.array([0, 1, 0x80000000, 0xFFFFFFFF], dtype=np.uint32)
    sign_uint_vals = sign_uint_array(uint_unary)
    not_uint_vals = np.bitwise_not(uint_unary).astype(np.uint32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    uint32_t values[4];")
    lines.append("    uint32_t signv[4];")
    lines.append("    uint32_t bitwise_not[4];")
    lines.append("} g_unary_u32 = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_uints(uint_unary)}}},")
    lines.append(f"    {{{fmt_uints(sign_uint_vals)}}},")
    lines.append(f"    {{{fmt_uints(not_uint_vals)}}},")
    lines.append("};\n")

    # Unsigned 16-bit dataset for wrap-around arithmetic
    u16_a = np.array([0xFF00, 0x7FF0, 0x0FF0, 0x00F0], dtype=np.uint32)
    u16_b = np.array([0x0200, 0x000F, 0x000A, 0x0001], dtype=np.uint32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    uint16_t a[4];")
    lines.append("    uint16_t b[4];")
    lines.append("    uint16_t add[4];")
    lines.append("    uint16_t sub[4];")
    lines.append("    uint16_t div[4];")
    lines.append("} g_elementwise_u16 = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_uints(u16_a)}}},")
    lines.append(f"    {{{fmt_uints(u16_b)}}},")
    lines.append(f"    {{{fmt_uints((u16_a + u16_b) & 0xFFFF)}}},")
    lines.append(f"    {{{fmt_uints((u16_a - u16_b) & 0xFFFF)}}},")
    lines.append(f"    {{{fmt_uints(u16_a // u16_b)}}},")
    lines.append("};\n")

    # Comparisons (float)
    cf_a = np.array([1.0, -2.0, 3.5, 0.0], dtype=np.float64)
    cf_b = np.array([1.0, -3.0, 2.0, 0.5], dtype=np.float64)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float a[4];")
    lines.append("    float b[4];")
    lines.append("    uint8_t eq[4];")
    lines.append("    uint8_t lt[4];")
    lines.append("    uint8_t ge[4];")
    lines.append("} g_elementwise_cmp_float = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(cf_a)}}},")
    lines.append(f"    {{{fmt_floats(cf_b)}}},")
    lines.append(f"    {{{fmt_uints(np.equal(cf_a, cf_b).astype(np.uint8))}}},")
    lines.append(f"    {{{fmt_uints((cf_a < cf_b).astype(np.uint8))}}},")
    lines.append(f"    {{{fmt_uints((cf_a >= cf_b).astype(np.uint8))}}},")
    lines.append("};\n")

    # Comparisons (int)
    ci_a = np.array([5, -3, 7, -9], dtype=np.int32)
    ci_b = np.array([4, -3, 10, -10], dtype=np.int32)
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    int32_t a[4];")
    lines.append("    int32_t b[4];")
    lines.append("    uint8_t gt[4];")
    lines.append("    uint8_t ne[4];")
    lines.append("} g_elementwise_cmp_int = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_ints(ci_a)}}},")
    lines.append(f"    {{{fmt_ints(ci_b)}}},")
    lines.append(f"    {{{fmt_uints((ci_a > ci_b).astype(np.uint8))}}},")
    lines.append(f"    {{{fmt_uints((ci_a != ci_b).astype(np.uint8))}}},")
    lines.append("};\n")

    # FMA dataset
    fma_a = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    fma_b = np.array([2.0, 4.0, -1.5, 0.25], dtype=np.float64)
    fma_c = np.array([-0.5, 1.0, 0.25, -2.0], dtype=np.float64)
    fma_expected = fma_a * fma_b + fma_c
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float a[4];")
    lines.append("    float b[4];")
    lines.append("    float c[4];")
    lines.append("    float expected[4];")
    lines.append("} g_elementwise_fma = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(fma_a)}}},")
    lines.append(f"    {{{fmt_floats(fma_b)}}},")
    lines.append(f"    {{{fmt_floats(fma_c)}}},")
    lines.append(f"    {{{fmt_floats(fma_expected)}}},")
    lines.append("};\n")

    return "\n".join(lines)


def make_activation_section() -> str:
    lines: list[str] = []
    inputs = np.array([-6.0, -2.5, -0.25, 0.0, 0.5, 2.0], dtype=np.float64)
    relu = np.maximum(inputs, 0.0)
    erf_inputs = np.array([math.erf(float(x) / math.sqrt(2.0)) for x in inputs], dtype=np.float64)
    gelu = inputs * 0.5 * (1.0 + erf_inputs)
    silu = inputs / (1.0 + np.exp(-inputs))
    gelu_tanh_c = 0.7978845608028654
    gelu_tanh_coeff = 0.044715
    gelu_tanh = 0.5 * inputs * (1.0 + np.tanh(gelu_tanh_c * (inputs + gelu_tanh_coeff * inputs ** 3)))
    sigmoid = 1.0 / (1.0 + np.exp(-inputs))
    tanh = np.tanh(inputs)
    softplus = np.log1p(np.exp(-np.abs(inputs))) + np.maximum(inputs, 0.0)
    mish = inputs * np.tanh(softplus)
    elu_alpha = 1.1
    elu = np.where(inputs > 0.0, inputs, elu_alpha * (np.exp(inputs) - 1.0))
    selu_alpha = 1.6732632423543772
    selu_lambda = 1.0507009873554804
    selu_inner = np.where(inputs > 0.0, inputs, selu_alpha * (np.exp(inputs) - 1.0))
    selu = selu_lambda * selu_inner
    leaky_slope = 0.02
    leaky = np.where(inputs >= 0.0, inputs, leaky_slope * inputs)
    prelu_slope = 0.25
    prelu = np.where(inputs >= 0.0, inputs, prelu_slope * inputs)

    lines.append("static const struct {")
    lines.append("    size_t rows;")
    lines.append("    size_t cols;")
    lines.append("    float input[6];")
    lines.append("    float relu[6];")
    lines.append("    float gelu[6];")
    lines.append("    float silu[6];")
    lines.append("    float gelu_tanh[6];")
    lines.append("    float sigmoid[6];")
    lines.append("    float tanh_v[6];")
    lines.append("    float mish[6];")
    lines.append("    float elu[6];")
    lines.append("    float selu[6];")
    lines.append("    float leaky[6];")
    lines.append("    float prelu[6];")
    lines.append("} g_activation = {")
    lines.append("    2,")
    lines.append("    3,")
    lines.append(f"    {{{fmt_floats(inputs)}}},")
    lines.append(f"    {{{fmt_floats(relu)}}},")
    lines.append(f"    {{{fmt_floats(gelu)}}},")
    lines.append(f"    {{{fmt_floats(silu)}}},")
    lines.append(f"    {{{fmt_floats(gelu_tanh)}}},")
    lines.append(f"    {{{fmt_floats(sigmoid)}}},")
    lines.append(f"    {{{fmt_floats(tanh)}}},")
    lines.append(f"    {{{fmt_floats(mish)}}},")
    lines.append(f"    {{{fmt_floats(elu)}}},")
    lines.append(f"    {{{fmt_floats(selu)}}},")
    lines.append(f"    {{{fmt_floats(leaky)}}},")
    lines.append(f"    {{{fmt_floats(prelu)}}},")
    lines.append("};\n")
    return "\n".join(lines)


def make_normalization_section() -> str:
    lines: list[str] = []
    shape = (2, 3)
    x_vals = np.array([1.0, 2.0, 3.0, -2.0, -4.0, -6.0], dtype=np.float64).reshape(shape)
    weight_vals = np.array([1.0, 1.5, 0.5], dtype=np.float64)
    bias_vals = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    eps = 1e-5
    mean = np.mean(x_vals, axis=1, keepdims=True)
    var = np.mean((x_vals - mean) ** 2, axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    normalized = (x_vals - mean) * inv_std
    ln_no_affine = normalized
    ln_affine = normalized * weight_vals + bias_vals

    lines.append("static const struct {")
    lines.append("    size_t rows;")
    lines.append("    size_t cols;")
    lines.append("    float input[6];")
    lines.append("    float weight[3];")
    lines.append("    float bias[3];")
    lines.append("    float eps;")
    lines.append("    float expected_affine[6];")
    lines.append("    float expected_no_affine[6];")
    lines.append("} g_layernorm = {")
    lines.append("    2,")
    lines.append("    3,")
    lines.append(f"    {{{fmt_floats(x_vals.ravel())}}},")
    lines.append(f"    {{{fmt_floats(weight_vals)}}},")
    lines.append(f"    {{{fmt_floats(bias_vals)}}},")
    lines.append(f"    {eps:.6g},")
    lines.append(f"    {{{fmt_floats(ln_affine.ravel())}}},")
    lines.append(f"    {{{fmt_floats(ln_no_affine.ravel())}}},")
    lines.append("};\n")

    rms_shape = (2, 4)
    rms_x = np.array(
        [0.5, -1.0, 2.0, -4.0, 1.5, -3.0, 6.0, -2.0],
        dtype=np.float64,
    ).reshape(rms_shape)
    rms_weight = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    rms_eps = 1e-5
    rms_norm = rms_x / np.sqrt(np.mean(rms_x ** 2, axis=1, keepdims=True) + rms_eps)
    rms_expected = rms_norm * rms_weight

    lines.append("static const struct {")
    lines.append("    size_t rows;")
    lines.append("    size_t cols;")
    lines.append("    float input[8];")
    lines.append("    float weight[4];")
    lines.append("    float eps;")
    lines.append("    float expected[8];")
    lines.append("} g_rmsnorm = {")
    lines.append("    2,")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(rms_x.ravel())}}},")
    lines.append(f"    {{{fmt_floats(rms_weight)}}},")
    lines.append(f"    {rms_eps:.6g},")
    lines.append(f"    {{{fmt_floats(rms_expected.ravel())}}},")
    lines.append("};\n")

    return "\n".join(lines)


def make_softmax_section() -> str:
    lines: list[str] = []
    shape = (2, 3)
    data = np.array([1.0, 5.0, 2.0, -1.0, -2.0, -3.0], dtype=np.float64).reshape(shape)

    def softmax(arr: np.ndarray, axis: int) -> np.ndarray:
        shifted = arr - np.max(arr, axis=axis, keepdims=True)
        expv = np.exp(shifted)
        return expv / np.sum(expv, axis=axis, keepdims=True)

    soft_last = softmax(data, axis=-1)
    soft_axis0 = softmax(data, axis=0)

    lines.append("static const struct {")
    lines.append("    size_t ndim;")
    lines.append("    size_t shape[2];")
    lines.append("    float input[6];")
    lines.append("    float expected_axis_last[6];")
    lines.append("    float expected_axis0[6];")
    lines.append("    double input_f64[6];")
    lines.append("    double expected_axis_last_f64[6];")
    lines.append("    double expected_axis0_f64[6];")
    lines.append("} g_softmax = {")
    lines.append("    2,")
    lines.append("    {2, 3},")
    lines.append(f"    {{{fmt_floats(data.ravel())}}},")
    lines.append(f"    {{{fmt_floats(soft_last.ravel())}}},")
    lines.append(f"    {{{fmt_floats(soft_axis0.ravel())}}},")
    lines.append(f"    {{{fmt_doubles(data.ravel())}}},")
    lines.append(f"    {{{fmt_doubles(soft_last.ravel())}}},")
    lines.append(f"    {{{fmt_doubles(soft_axis0.ravel())}}},")
    lines.append("};\n")
    return "\n".join(lines)


def make_reduction_section() -> str:
    lines: list[str] = []

    fp32_data = np.array(
        [1.5, -2.0, 3.0, -4.5, 0.25, 2.5],
        dtype=np.float64,
    ).reshape(2, 3)
    fp32_sum_axis0 = fp32_data.sum(axis=0)
    fp32_sum_axis1 = fp32_data.sum(axis=1)
    fp32_mean_axis0 = fp32_data.mean(axis=0)
    fp32_mean_axis1 = fp32_data.mean(axis=1)
    fp32_max_axis0 = fp32_data.max(axis=0)
    fp32_max_axis1 = fp32_data.max(axis=1)
    fp32_min_axis0 = fp32_data.min(axis=0)
    fp32_min_axis1 = fp32_data.min(axis=1)
    fp32_argmax_axis0 = np.argmax(fp32_data, axis=0).astype(np.int64)
    fp32_argmax_axis1 = np.argmax(fp32_data, axis=1).astype(np.int64)
    fp32_argmin_axis0 = np.argmin(fp32_data, axis=0).astype(np.int64)
    fp32_argmin_axis1 = np.argmin(fp32_data, axis=1).astype(np.int64)
    fp32_any_axis0 = np.any(fp32_data != 0.0, axis=0).astype(np.float64)
    fp32_all_axis1 = np.all(fp32_data != 0.0, axis=1).astype(np.float64)

    lines.append("static const struct {")
    lines.append("    size_t ndim;")
    lines.append("    size_t shape[2];")
    lines.append("    float input[6];")
    lines.append("    float sum_all;")
    lines.append("    float mean_all;")
    lines.append("    float prod_all;")
    lines.append("    float max_all;")
    lines.append("    float min_all;")
    lines.append("    uint64_t argmax_all;")
    lines.append("    uint64_t argmin_all;")
    lines.append("    float any_all;")
    lines.append("    float all_all;")
    lines.append("    float variance_all;")
    lines.append("    float std_all;")
    lines.append("    float norm_l1_all;")
    lines.append("    float norm_l2_all;")
    lines.append("    float sum_axis0[3];")
    lines.append("    float sum_axis1[2];")
    lines.append("    float mean_axis0[3];")
    lines.append("    float mean_axis1[2];")
    lines.append("    float max_axis0[3];")
    lines.append("    float max_axis1[2];")
    lines.append("    float min_axis0[3];")
    lines.append("    float min_axis1[2];")
    lines.append("    uint64_t argmax_axis0[3];")
    lines.append("    uint64_t argmax_axis1[2];")
    lines.append("    uint64_t argmin_axis0[3];")
    lines.append("    uint64_t argmin_axis1[2];")
    lines.append("    float any_axis0[3];")
    lines.append("    float all_axis1[2];")
    lines.append("    double input_f64[6];")
    lines.append("    double sum_all_f64;")
    lines.append("    double mean_all_f64;")
    lines.append("    double prod_all_f64;")
    lines.append("    double max_all_f64;")
    lines.append("    double min_all_f64;")
    lines.append("    double any_all_f64;")
    lines.append("    double all_all_f64;")
    lines.append("    double variance_all_f64;")
    lines.append("    double std_all_f64;")
    lines.append("    double norm_l1_all_f64;")
    lines.append("    double norm_l2_all_f64;")
    lines.append("    double sum_axis0_f64[3];")
    lines.append("    double sum_axis1_f64[2];")
    lines.append("    double mean_axis0_f64[3];")
    lines.append("    double mean_axis1_f64[2];")
    lines.append("    double max_axis0_f64[3];")
    lines.append("    double max_axis1_f64[2];")
    lines.append("    double min_axis0_f64[3];")
    lines.append("    double min_axis1_f64[2];")
    lines.append("    double any_axis0_f64[3];")
    lines.append("    double all_axis1_f64[2];")
    lines.append("} g_reduction_fp32 = {")
    lines.append("    2,")
    lines.append("    {2, 3},")
    lines.append(f"    {{{fmt_floats(fp32_data.ravel())}}},")
    lines.append(f"    {float(fp32_data.sum()):.10g},")
    lines.append(f"    {float(fp32_data.mean()):.10g},")
    lines.append(f"    {float(fp32_data.prod()):.10g},")
    lines.append(f"    {float(fp32_data.max()):.10g},")
    lines.append(f"    {float(fp32_data.min()):.10g},")
    lines.append(f"    UINT64_C({int(np.argmax(fp32_data))}),")
    lines.append(f"    UINT64_C({int(np.argmin(fp32_data))}),")
    lines.append(f"    {float(np.any(fp32_data != 0.0)):.10g},")
    lines.append(f"    {float(np.all(fp32_data != 0.0)):.10g},")
    lines.append(f"    {float(np.var(fp32_data, ddof=0)):.10g},")
    lines.append(f"    {float(np.std(fp32_data, ddof=0)):.10g},")
    lines.append(f"    {float(np.sum(np.abs(fp32_data))):.10g},")
    lines.append(f"    {float(np.sqrt(np.sum(fp32_data ** 2))):.10g},")
    lines.append(f"    {{{fmt_floats(fp32_sum_axis0)}}},")
    lines.append(f"    {{{fmt_floats(fp32_sum_axis1)}}},")
    lines.append(f"    {{{fmt_floats(fp32_mean_axis0)}}},")
    lines.append(f"    {{{fmt_floats(fp32_mean_axis1)}}},")
    lines.append(f"    {{{fmt_floats(fp32_max_axis0)}}},")
    lines.append(f"    {{{fmt_floats(fp32_max_axis1)}}},")
    lines.append(f"    {{{fmt_floats(fp32_min_axis0)}}},")
    lines.append(f"    {{{fmt_floats(fp32_min_axis1)}}},")
    lines.append(f"    {{{fmt_u64s(fp32_argmax_axis0)}}},")
    lines.append(f"    {{{fmt_u64s(fp32_argmax_axis1)}}},")
    lines.append(f"    {{{fmt_u64s(fp32_argmin_axis0)}}},")
    lines.append(f"    {{{fmt_u64s(fp32_argmin_axis1)}}},")
    lines.append(f"    {{{fmt_floats(fp32_any_axis0)}}},")
    lines.append(f"    {{{fmt_floats(fp32_all_axis1)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_data.ravel())}}},")
    lines.append(f"    {fp32_data.sum():.17g},")
    lines.append(f"    {fp32_data.mean():.17g},")
    lines.append(f"    {fp32_data.prod():.17g},")
    lines.append(f"    {fp32_data.max():.17g},")
    lines.append(f"    {fp32_data.min():.17g},")
    lines.append(f"    {float(np.any(fp32_data != 0.0)):.17g},")
    lines.append(f"    {float(np.all(fp32_data != 0.0)):.17g},")
    lines.append(f"    {np.var(fp32_data, ddof=0):.17g},")
    lines.append(f"    {np.std(fp32_data, ddof=0):.17g},")
    lines.append(f"    {np.sum(np.abs(fp32_data)):.17g},")
    lines.append(f"    {np.sqrt(np.sum(fp32_data ** 2)):.17g},")
    lines.append(f"    {{{fmt_doubles(fp32_sum_axis0)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_sum_axis1)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_mean_axis0)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_mean_axis1)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_max_axis0)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_max_axis1)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_min_axis0)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_min_axis1)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_any_axis0)}}},")
    lines.append(f"    {{{fmt_doubles(fp32_all_axis1)}}},")
    lines.append("};\n")

    int_data = np.array(
        [12, 0, 3, -2, 5, -9],
        dtype=np.int64,
    ).reshape(2, 3)
    int_sum_axis0 = int_data.sum(axis=0)
    int_sum_axis1 = int_data.sum(axis=1)
    int_max_axis0 = int_data.max(axis=0)
    int_max_axis1 = int_data.max(axis=1)
    int_min_axis0 = int_data.min(axis=0)
    int_min_axis1 = int_data.min(axis=1)
    int_argmax_axis0 = np.argmax(int_data, axis=0).astype(np.int64)
    int_argmax_axis1 = np.argmax(int_data, axis=1).astype(np.int64)
    int_argmin_axis0 = np.argmin(int_data, axis=0).astype(np.int64)
    int_argmin_axis1 = np.argmin(int_data, axis=1).astype(np.int64)
    int_any_axis0 = np.any(int_data != 0, axis=0).astype(np.int64)
    int_all_axis1 = np.all(int_data != 0, axis=1).astype(np.int64)

    def append_signed_reduction_struct(name: str, ctype: str) -> None:
        lines.append("static const struct {")
        lines.append("    size_t ndim;")
        lines.append("    size_t shape[2];")
        lines.append(f"    {ctype} input[6];")
        lines.append(f"    {ctype} sum_all;")
        lines.append(f"    {ctype} prod_all;")
        lines.append(f"    {ctype} max_all;")
        lines.append(f"    {ctype} min_all;")
        lines.append("    uint64_t argmax_all;")
        lines.append("    uint64_t argmin_all;")
        lines.append(f"    {ctype} any_all;")
        lines.append(f"    {ctype} all_all;")
        lines.append(f"    {ctype} sum_axis0[3];")
        lines.append(f"    {ctype} sum_axis1[2];")
        lines.append(f"    {ctype} max_axis0[3];")
        lines.append(f"    {ctype} max_axis1[2];")
        lines.append(f"    {ctype} min_axis0[3];")
        lines.append(f"    {ctype} min_axis1[2];")
        lines.append("    uint64_t argmax_axis0[3];")
        lines.append("    uint64_t argmax_axis1[2];")
        lines.append("    uint64_t argmin_axis0[3];")
        lines.append("    uint64_t argmin_axis1[2];")
        lines.append(f"    {ctype} any_axis0[3];")
        lines.append(f"    {ctype} all_axis1[2];")
        lines.append(f"}} {name} = {{")
        lines.append("    2,")
        lines.append("    {2, 3},")
        lines.append(f"    {{{fmt_ints(int_data.ravel())}}},")
        lines.append(f"    {int(int_data.sum())},")
        lines.append(f"    {int(np.prod(int_data))},")
        lines.append(f"    {int(int_data.max())},")
        lines.append(f"    {int(int_data.min())},")
        lines.append(f"    UINT64_C({int(np.argmax(int_data))}),")
        lines.append(f"    UINT64_C({int(np.argmin(int_data))}),")
        lines.append(f"    {1 if np.any(int_data != 0) else 0},")
        lines.append(f"    {1 if np.all(int_data != 0) else 0},")
        lines.append(f"    {{{fmt_ints(int_sum_axis0)}}},")
        lines.append(f"    {{{fmt_ints(int_sum_axis1)}}},")
        lines.append(f"    {{{fmt_ints(int_max_axis0)}}},")
        lines.append(f"    {{{fmt_ints(int_max_axis1)}}},")
        lines.append(f"    {{{fmt_ints(int_min_axis0)}}},")
        lines.append(f"    {{{fmt_ints(int_min_axis1)}}},")
        lines.append(f"    {{{fmt_u64s(int_argmax_axis0)}}},")
        lines.append(f"    {{{fmt_u64s(int_argmax_axis1)}}},")
        lines.append(f"    {{{fmt_u64s(int_argmin_axis0)}}},")
        lines.append(f"    {{{fmt_u64s(int_argmin_axis1)}}},")
        lines.append(f"    {{{fmt_ints(int_any_axis0)}}},")
        lines.append(f"    {{{fmt_ints(int_all_axis1)}}},")
        lines.append("};\n")

    append_signed_reduction_struct("g_reduction_i32", "int32_t")
    append_signed_reduction_struct("g_reduction_i16", "int16_t")
    append_signed_reduction_struct("g_reduction_i8", "int8_t")

    uint_data = np.array(
        [7, 0, 14, 9, 3, 5],
        dtype=np.uint64,
    ).reshape(2, 3)
    uint_sum_axis0 = uint_data.sum(axis=0)
    uint_sum_axis1 = uint_data.sum(axis=1)
    uint_max_axis0 = uint_data.max(axis=0)
    uint_max_axis1 = uint_data.max(axis=1)
    uint_min_axis0 = uint_data.min(axis=0)
    uint_min_axis1 = uint_data.min(axis=1)
    uint_argmax_axis0 = np.argmax(uint_data, axis=0).astype(np.int64)
    uint_argmax_axis1 = np.argmax(uint_data, axis=1).astype(np.int64)
    uint_argmin_axis0 = np.argmin(uint_data, axis=0).astype(np.int64)
    uint_argmin_axis1 = np.argmin(uint_data, axis=1).astype(np.int64)
    uint_any_axis0 = np.any(uint_data != 0, axis=0).astype(np.int64)
    uint_all_axis1 = np.all(uint_data != 0, axis=1).astype(np.int64)

    def append_unsigned_reduction_struct(name: str, ctype: str) -> None:
        lines.append("static const struct {")
        lines.append("    size_t ndim;")
        lines.append("    size_t shape[2];")
        lines.append(f"    {ctype} input[6];")
        lines.append(f"    {ctype} sum_all;")
        lines.append(f"    {ctype} prod_all;")
        lines.append(f"    {ctype} max_all;")
        lines.append(f"    {ctype} min_all;")
        lines.append("    uint64_t argmax_all;")
        lines.append("    uint64_t argmin_all;")
        lines.append(f"    {ctype} any_all;")
        lines.append(f"    {ctype} all_all;")
        lines.append(f"    {ctype} sum_axis0[3];")
        lines.append(f"    {ctype} sum_axis1[2];")
        lines.append(f"    {ctype} max_axis0[3];")
        lines.append(f"    {ctype} max_axis1[2];")
        lines.append(f"    {ctype} min_axis0[3];")
        lines.append(f"    {ctype} min_axis1[2];")
        lines.append("    uint64_t argmax_axis0[3];")
        lines.append("    uint64_t argmax_axis1[2];")
        lines.append("    uint64_t argmin_axis0[3];")
        lines.append("    uint64_t argmin_axis1[2];")
        lines.append(f"    {ctype} any_axis0[3];")
        lines.append(f"    {ctype} all_axis1[2];")
        lines.append(f"}} {name} = {{")
        lines.append("    2,")
        lines.append("    {2, 3},")
        lines.append(f"    {{{fmt_uints(uint_data.ravel())}}},")
        lines.append(f"    {int(uint_data.sum())}u,")
        lines.append(f"    {int(np.prod(uint_data))}u,")
        lines.append(f"    {int(uint_data.max())}u,")
        lines.append(f"    {int(uint_data.min())}u,")
        lines.append(f"    UINT64_C({int(np.argmax(uint_data))}),")
        lines.append(f"    UINT64_C({int(np.argmin(uint_data))}),")
        lines.append(f"    {1 if np.any(uint_data != 0) else 0}u,")
        lines.append(f"    {1 if np.all(uint_data != 0) else 0}u,")
        lines.append(f"    {{{fmt_uints(uint_sum_axis0)}}},")
        lines.append(f"    {{{fmt_uints(uint_sum_axis1)}}},")
        lines.append(f"    {{{fmt_uints(uint_max_axis0)}}},")
        lines.append(f"    {{{fmt_uints(uint_max_axis1)}}},")
        lines.append(f"    {{{fmt_uints(uint_min_axis0)}}},")
        lines.append(f"    {{{fmt_uints(uint_min_axis1)}}},")
        lines.append(f"    {{{fmt_u64s(uint_argmax_axis0)}}},")
        lines.append(f"    {{{fmt_u64s(uint_argmax_axis1)}}},")
        lines.append(f"    {{{fmt_u64s(uint_argmin_axis0)}}},")
        lines.append(f"    {{{fmt_u64s(uint_argmin_axis1)}}},")
        lines.append(f"    {{{fmt_uints(uint_any_axis0)}}},")
        lines.append(f"    {{{fmt_uints(uint_all_axis1)}}},")
        lines.append("};\n")

    append_unsigned_reduction_struct("g_reduction_u32", "uint32_t")
    append_unsigned_reduction_struct("g_reduction_u16", "uint16_t")
    append_unsigned_reduction_struct("g_reduction_u8", "uint8_t")

    return "\n".join(lines)


def make_tensor_ops_section() -> str:
    lines: list[str] = []

    base = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64).reshape(2, 3)
    transpose_default = base.transpose((1, 0))
    transpose_identity = base.copy()

    lines.append("static const struct {")
    lines.append("    size_t shape_src[2];")
    lines.append("    size_t shape_dst[2];")
    lines.append("    float src[6];")
    lines.append("    float transpose_default[6];")
    lines.append("    float transpose_identity[6];")
    lines.append("} g_tensor_2d = {")
    lines.append("    {2, 3},")
    lines.append("    {3, 2},")
    lines.append(f"    {{{fmt_floats(base.ravel())}}},")
    lines.append(f"    {{{fmt_floats(transpose_default.ravel())}}},")
    lines.append(f"    {{{fmt_floats(transpose_identity.ravel())}}},")
    lines.append("};\n")

    left = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    right = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    concat_axis0 = np.concatenate([left, right])
    slice_1d = concat_axis0[2:5]

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float left[3];")
    lines.append("    float right[3];")
    lines.append("    float concat_axis0[6];")
    lines.append("    float slice_result[3];")
    lines.append("} g_tensor_concat1d = {")
    lines.append("    3,")
    lines.append(f"    {{{fmt_floats(left)}}},")
    lines.append(f"    {{{fmt_floats(right)}}},")
    lines.append(f"    {{{fmt_floats(concat_axis0)}}},")
    lines.append(f"    {{{fmt_floats(slice_1d)}}},")
    lines.append("};\n")

    tensor3 = np.arange(24, dtype=np.float64) * 0.25
    tensor3 = tensor3.reshape(2, 3, 4)
    perm = (2, 0, 1)
    transpose3 = np.transpose(tensor3, perm)

    lines.append("static const struct {")
    lines.append("    size_t shape_in[3];")
    lines.append("    size_t perm[3];")
    lines.append("    float input[24];")
    lines.append("    float expected[24];")
    lines.append("} g_tensor_transpose3d = {")
    lines.append("    {2, 3, 4},")
    lines.append("    {2, 0, 1},")
    lines.append(f"    {{{fmt_floats(tensor3.ravel())}}},")
    lines.append(f"    {{{fmt_floats(transpose3.ravel())}}},")
    lines.append("};\n")

    concat_a_values = np.array([i * 0.1 for i in range(12)], dtype=np.float64)
    concat_a = concat_a_values.reshape(2, 2, 3)
    concat_b = (2.0 + np.arange(6, dtype=np.float64) * 0.05).reshape(2, 1, 3)
    concat_axis1 = np.concatenate([concat_a, concat_b], axis=1)

    lines.append("static const struct {")
    lines.append("    size_t shape_a[3];")
    lines.append("    size_t shape_b[3];")
    lines.append("    float a[18];")
    lines.append("    float b[6];")
    lines.append("    float expected[18];")
    lines.append("} g_tensor_concat3d = {")
    lines.append("    {2, 2, 3},")
    lines.append("    {2, 1, 3},")
    lines.append(f"    {{{fmt_floats(concat_a.ravel())}}},")
    lines.append(f"    {{{fmt_floats(concat_b.ravel())}}},")
    lines.append(f"    {{{fmt_floats(concat_axis1.ravel())}}},")
    lines.append("};\n")

    slice_src = np.sin(np.arange(24, dtype=np.float64) * 0.3).reshape(2, 3, 4)
    slice_result = slice_src[1:2, 0:2, 1:3]

    lines.append("static const struct {")
    lines.append("    size_t shape_src[3];")
    lines.append("    size_t starts[3];")
    lines.append("    size_t sizes[3];")
    lines.append("    float src[24];")
    lines.append("    float expected[4];")
    lines.append("} g_tensor_slice3d = {")
    lines.append("    {2, 3, 4},")
    lines.append("    {1, 0, 1},")
    lines.append("    {1, 2, 2},")
    lines.append(f"    {{{fmt_floats(slice_src.ravel())}}},")
    lines.append(f"    {{{fmt_floats(slice_result.ravel())}}},")
    lines.append("};\n")

    return "\n".join(lines)


def make_matmul_section() -> str:
    lines: list[str] = []

    def matmul_case(m: int, k: int, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = (np.arange(m * k, dtype=np.float64) - (m * k) / 3.0) * 0.25
        b = (np.arange(k * n, dtype=np.float64) - (k * n) / 2.0) * 0.15
        a = a.reshape(m, k)
        b = b.reshape(k, n)
        expected = a @ b
        return a, b, expected

    cases = [
        (2, 3, 2),
        (3, 4, 2),
        (2, 4, 3),
        (2, 4, 2),
    ]

    for idx, (m, k, n) in enumerate(cases):
        a, b, expected = matmul_case(m, k, n)
        lines.append("static const struct {")
        lines.append("    size_t m;")
        lines.append("    size_t k;")
        lines.append("    size_t n;")
        lines.append(f"    float a[{m * k}];")
        lines.append(f"    float b[{k * n}];")
        lines.append(f"    float expected[{m * n}];")
        lines.append(f"}} g_matmul_case_{idx} = {{")
        lines.append(f"    {m},")
        lines.append(f"    {k},")
        lines.append(f"    {n},")
        lines.append(f"    {{{fmt_floats(a.ravel())}}},")
        lines.append(f"    {{{fmt_floats(b.ravel())}}},")
        lines.append(f"    {{{fmt_floats(expected.ravel())}}},")
        lines.append("};\n")

    return "\n".join(lines)


def make_matmul_fp8_section() -> str:
    """Generate FP8 matmul golden data.

    FP8 matmul: input(N×K) @ weight(M×K).T = output(N×M)
    - Inputs are FP8 (E4M3 or E5M2) stored as uint8
    - Output is always F32 (industry best practice)
    - Golden computation: quantize to FP8, dequantize to F64, matmul in F64, cast to F32
    """
    lines: list[str] = []

    # E4M3: 4-bit exponent, 3-bit mantissa, bias=7, no infinity, max=448
    # E5M2: 5-bit exponent, 2-bit mantissa, bias=15, has infinity, max=57344
    fp8_configs = [
        ("e4m3", 4, 3, 7, False, 448.0),
        ("e5m2", 5, 2, 15, True, 57344.0),
    ]

    # Test cases: (N, K, M) - using linear convention
    cases = [
        (2, 4, 3),   # Small case
        (3, 8, 4),   # Medium case
        (4, 16, 5),  # Larger K for better coverage
    ]

    lines.append("#if MARMOT_ENABLE_FP8\n")

    for fmt_name, exp_bits, mant_bits, bias, has_inf, max_finite in fp8_configs:
        for case_idx, (n, k, m) in enumerate(cases):
            # Generate input data in a range suitable for FP8
            # Use smaller values to avoid saturation
            input_f64 = (np.arange(n * k, dtype=np.float64) - (n * k) / 2.0) * 0.1
            weight_f64 = (np.arange(m * k, dtype=np.float64) - (m * k) / 2.0) * 0.08

            # Quantize to FP8
            input_fp8 = fp8_quantize_array(input_f64, exp_bits, mant_bits, bias, has_inf, max_finite)
            weight_fp8 = fp8_quantize_array(weight_f64, exp_bits, mant_bits, bias, has_inf, max_finite)

            # Dequantize back to F64 for golden computation
            input_dequant = fp8_dequantize_array(input_fp8, exp_bits, mant_bits, bias, has_inf)
            weight_dequant = fp8_dequantize_array(weight_fp8, exp_bits, mant_bits, bias, has_inf)

            # Compute matmul: input(N×K) @ weight(M×K).T = output(N×M)
            input_mat = input_dequant.reshape(n, k)
            weight_mat = weight_dequant.reshape(m, k)
            expected_f64 = input_mat @ weight_mat.T
            expected_f32 = expected_f64.astype(np.float32)

            lines.append(f"static const struct {{")
            lines.append(f"    size_t n;")
            lines.append(f"    size_t k;")
            lines.append(f"    size_t m;")
            lines.append(f"    uint8_t input[{n * k}];")
            lines.append(f"    uint8_t weight[{m * k}];")
            lines.append(f"    float expected[{n * m}];")
            lines.append(f"}} g_matmul_fp8_{fmt_name}_case{case_idx} = {{")
            lines.append(f"    {n},")
            lines.append(f"    {k},")
            lines.append(f"    {m},")
            lines.append(f"    {{{fmt_uints(input_fp8)}}},")
            lines.append(f"    {{{fmt_uints(weight_fp8)}}},")
            lines.append(f"    {{{fmt_floats(expected_f32.ravel())}}},")
            lines.append(f"}};\n")

    # Generate case arrays for iteration
    for fmt_name, _, _, _, _, _ in fp8_configs:
        lines.append(f"typedef struct {{")
        lines.append(f"    size_t n;")
        lines.append(f"    size_t k;")
        lines.append(f"    size_t m;")
        lines.append(f"    const uint8_t *input;")
        lines.append(f"    const uint8_t *weight;")
        lines.append(f"    const float *expected;")
        lines.append(f"}} matmul_fp8_{fmt_name}_case_t;\n")

        lines.append(f"static const matmul_fp8_{fmt_name}_case_t g_matmul_fp8_{fmt_name}_cases[] = {{")
        for case_idx in range(len(cases)):
            lines.append(f"    {{")
            lines.append(f"        g_matmul_fp8_{fmt_name}_case{case_idx}.n,")
            lines.append(f"        g_matmul_fp8_{fmt_name}_case{case_idx}.k,")
            lines.append(f"        g_matmul_fp8_{fmt_name}_case{case_idx}.m,")
            lines.append(f"        g_matmul_fp8_{fmt_name}_case{case_idx}.input,")
            lines.append(f"        g_matmul_fp8_{fmt_name}_case{case_idx}.weight,")
            lines.append(f"        g_matmul_fp8_{fmt_name}_case{case_idx}.expected,")
            lines.append(f"    }},")
        lines.append(f"}};\n")
        lines.append(f"static const size_t g_matmul_fp8_{fmt_name}_case_count = "
                     f"sizeof(g_matmul_fp8_{fmt_name}_cases) / sizeof(g_matmul_fp8_{fmt_name}_cases[0]);\n")

    lines.append("#endif // MARMOT_ENABLE_FP8\n")

    return "\n".join(lines)


def apply_activation(values: np.ndarray, activation: str) -> np.ndarray:
    """Apply activation function to values."""
    if activation == "identity":
        return values
    elif activation == "relu":
        return np.maximum(0.0, values)
    elif activation == "gelu":
        # GELU using erf approximation: 0.5 * x * (1 + erf(x / sqrt(2)))
        return 0.5 * values * (1.0 + np.vectorize(math.erf)(values / math.sqrt(2.0)))
    elif activation == "gelu_tanh":
        # GELU tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        k0 = math.sqrt(2.0 / math.pi)
        k1 = 0.044715
        inner = k0 * (values + k1 * values ** 3)
        return 0.5 * values * (1.0 + np.tanh(inner))
    elif activation == "silu":
        # SiLU (Swish): x / (1 + exp(-x))
        return values / (1.0 + np.exp(-values))
    else:
        raise ValueError(f"Unknown activation: {activation}")


def make_matmul_fused_section() -> str:
    cases = [
        # Original cases (no activation)
        {
            "name": "bias_residual",
            "n": 2,
            "k": 3,
            "m": 4,
            "input": np.array([0.5, -1.0, 2.0, -0.5, 1.5, 3.0], dtype=np.float64),
            "weight": np.array(
                [
                    1.25,
                    -0.75,
                    0.5,
                    0.0,
                    1.5,
                    -0.25,
                    -1.0,
                    0.75,
                    0.25,
                    0.5,
                    -0.5,
                    0.25,
                ],
                dtype=np.float64,
            ),
            "bias": np.array([0.15, -0.05, 0.35, -0.2], dtype=np.float64),
            "residual": np.array(
                [-0.2, 0.1, 0.05, -0.15, 0.3, -0.25, 0.2, -0.1],
                dtype=np.float64,
            ),
            "activation": "identity",
        },
        {
            "name": "residual_only",
            "n": 1,
            "k": 4,
            "m": 3,
            "input": np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64),
            "weight": np.array(
                [
                    -0.5,
                    0.25,
                    1.0,
                    -0.75,
                    0.5,
                    -1.25,
                    0.75,
                    0.0,
                    -0.25,
                    0.5,
                    0.2,
                    -0.4,
                ],
                dtype=np.float64,
            ),
            "bias": None,
            "residual": np.array([0.05, -0.02, 0.1], dtype=np.float64),
            "activation": "identity",
        },
        # New cases with activations
        {
            "name": "bias_relu",
            "n": 2,
            "k": 3,
            "m": 4,
            "input": np.array([0.5, -1.0, 2.0, -0.5, 1.5, 3.0], dtype=np.float64),
            "weight": np.array(
                [1.0, -0.5, 0.25, 0.75, -1.0, 0.5, -0.25, 1.25, -0.75, 0.5, -1.5, 0.75],
                dtype=np.float64,
            ),
            "bias": np.array([0.1, -0.15, 0.2, -0.05], dtype=np.float64),
            "residual": None,
            "activation": "relu",
        },
        {
            "name": "bias_gelu",
            "n": 2,
            "k": 3,
            "m": 3,
            "input": np.array([1.0, -0.5, 0.75, -1.25, 0.5, 2.0], dtype=np.float64),
            "weight": np.array(
                [0.5, -1.0, 0.25, 1.5, -0.75, 0.5, -0.5, 1.0, -1.25],
                dtype=np.float64,
            ),
            "bias": np.array([0.05, -0.1, 0.15], dtype=np.float64),
            "residual": None,
            "activation": "gelu",
        },
        {
            "name": "bias_gelu_tanh",
            "n": 1,
            "k": 4,
            "m": 3,
            "input": np.array([0.75, -1.0, 1.5, -0.25], dtype=np.float64),
            "weight": np.array(
                [1.0, -0.5, 0.75, -1.25, 0.5, -1.0, 0.25, 1.5, -0.75, 0.5, -0.25, 1.0],
                dtype=np.float64,
            ),
            "bias": np.array([0.2, -0.1, 0.05], dtype=np.float64),
            "residual": None,
            "activation": "gelu_tanh",
        },
        {
            "name": "bias_silu",
            "n": 2,
            "k": 2,
            "m": 3,
            "input": np.array([1.5, -0.75, 0.25, 2.0], dtype=np.float64),
            "weight": np.array([0.5, -1.0, 1.25, -0.5, 0.75, -1.5], dtype=np.float64),
            "bias": np.array([0.1, -0.2, 0.15], dtype=np.float64),
            "residual": None,
            "activation": "silu",
        },
        # Combined bias + activation + residual
        {
            "name": "bias_relu_residual",
            "n": 2,
            "k": 3,
            "m": 3,
            "input": np.array([0.5, -1.0, 1.5, -0.5, 2.0, 0.75], dtype=np.float64),
            "weight": np.array(
                [1.0, -0.75, 0.5, 0.5, -1.25, 0.75, -0.5, 1.0, -0.25],
                dtype=np.float64,
            ),
            "bias": np.array([0.15, -0.1, 0.25], dtype=np.float64),
            "residual": np.array([-0.1, 0.05, -0.15, 0.2, -0.25, 0.1], dtype=np.float64),
            "activation": "relu",
        },
        {
            "name": "bias_gelu_residual",
            "n": 1,
            "k": 4,
            "m": 4,
            "input": np.array([1.0, -0.5, 1.5, -1.0], dtype=np.float64),
            "weight": np.array(
                [
                    0.75,
                    -1.0,
                    0.5,
                    -0.25,
                    1.25,
                    -0.5,
                    0.75,
                    -1.5,
                    0.5,
                    -1.0,
                    0.25,
                    1.0,
                    -0.75,
                    1.25,
                    -0.5,
                    0.5,
                ],
                dtype=np.float64,
            ),
            "bias": np.array([0.1, -0.15, 0.2, -0.05], dtype=np.float64),
            "residual": np.array([0.05, -0.1, 0.15, -0.2], dtype=np.float64),
            "activation": "gelu",
        },
    ]

    lines: list[str] = []
    for case in cases:
        name = case["name"]
        n = case["n"]
        k = case["k"]
        m = case["m"]
        input_vals = case["input"]
        weight_vals = case["weight"]
        bias_vals = case["bias"]
        residual_vals = case["residual"]
        activation = case["activation"]

        input_mat = input_vals.reshape(n, k)
        weight_mat = weight_vals.reshape(m, k)
        expected = input_mat @ weight_mat.T
        if bias_vals is not None:
            expected = expected + bias_vals.reshape(1, m)
        # Apply activation BEFORE residual (matches implementation)
        expected = apply_activation(expected, activation)
        if residual_vals is not None:
            expected = expected + residual_vals.reshape(n, m)

        case["expected"] = expected.reshape(-1)

        lines.append(f"static const float g_matmul_fused_{name}_input[{n * k}] = {{{fmt_floats(input_vals)}}};")
        lines.append(f"static const double g_matmul_fused_{name}_input_f64[{n * k}] = {{{fmt_doubles(input_vals)}}};")
        lines.append(f"static const float g_matmul_fused_{name}_weight[{m * k}] = {{{fmt_floats(weight_vals)}}};")
        lines.append(
            f"static const double g_matmul_fused_{name}_weight_f64[{m * k}] = {{{fmt_doubles(weight_vals)}}};"
        )

        if bias_vals is not None:
            lines.append(f"static const float g_matmul_fused_{name}_bias[{m}] = {{{fmt_floats(bias_vals)}}};")
            lines.append(f"static const double g_matmul_fused_{name}_bias_f64[{m}] = {{{fmt_doubles(bias_vals)}}};")
        else:
            lines.append(f"static const float *g_matmul_fused_{name}_bias = nullptr;")
            lines.append(f"static const double *g_matmul_fused_{name}_bias_f64 = nullptr;")

        if residual_vals is not None:
            lines.append(
                f"static const float g_matmul_fused_{name}_residual[{n * m}] = {{{fmt_floats(residual_vals)}}};"
            )
            lines.append(
                f"static const double g_matmul_fused_{name}_residual_f64[{n * m}] = "
                f"{{{fmt_doubles(residual_vals)}}};"
            )
        else:
            lines.append(f"static const float *g_matmul_fused_{name}_residual = nullptr;")
            lines.append(f"static const double *g_matmul_fused_{name}_residual_f64 = nullptr;")

        expected_vals = case["expected"]
        lines.append(f"static const float g_matmul_fused_{name}_expected[{n * m}] = {{{fmt_floats(expected_vals)}}};")
        lines.append(
            f"static const double g_matmul_fused_{name}_expected_f64[{n * m}] = {{{fmt_doubles(expected_vals)}}};"
        )
        lines.append("")

    # Map activation names to enum values
    activation_map = {
        "identity": "MARMOT_DEVICE_UNARY_IDENTITY",
        "relu": "MARMOT_DEVICE_UNARY_RELU",
        "gelu": "MARMOT_DEVICE_UNARY_GELU",
        "gelu_tanh": "MARMOT_DEVICE_UNARY_GELU_TANH",
        "silu": "MARMOT_DEVICE_UNARY_SILU",
    }

    lines.append("static const struct {")
    lines.append("    const char *name;")
    lines.append("    size_t n;")
    lines.append("    size_t k;")
    lines.append("    size_t m;")
    lines.append("    int has_bias;")
    lines.append("    int has_residual;")
    lines.append("    marmot_device_unary_op_t activation;")
    lines.append("    const float *input;")
    lines.append("    const float *weight;")
    lines.append("    const float *bias;")
    lines.append("    const float *residual;")
    lines.append("    const float *expected;")
    lines.append("    const double *input_f64;")
    lines.append("    const double *weight_f64;")
    lines.append("    const double *bias_f64;")
    lines.append("    const double *residual_f64;")
    lines.append("    const double *expected_f64;")
    lines.append("} g_matmul_fused_cases[] = {")

    for case in cases:
        name = case["name"]
        n = case["n"]
        k = case["k"]
        m = case["m"]
        has_bias = 1 if case["bias"] is not None else 0
        has_residual = 1 if case["residual"] is not None else 0
        activation_enum = activation_map[case["activation"]]
        bias_ptr = f"g_matmul_fused_{name}_bias" if has_bias else "nullptr"
        bias_f64_ptr = f"g_matmul_fused_{name}_bias_f64" if has_bias else "nullptr"
        residual_ptr = f"g_matmul_fused_{name}_residual" if has_residual else "nullptr"
        residual_f64_ptr = f"g_matmul_fused_{name}_residual_f64" if has_residual else "nullptr"

        lines.append("    {")
        lines.append(f"        \"{name}\",")
        lines.append(f"        {n},")
        lines.append(f"        {k},")
        lines.append(f"        {m},")
        lines.append(f"        {has_bias},")
        lines.append(f"        {has_residual},")
        lines.append(f"        {activation_enum},")
        lines.append(f"        g_matmul_fused_{name}_input,")
        lines.append(f"        g_matmul_fused_{name}_weight,")
        lines.append(f"        {bias_ptr},")
        lines.append(f"        {residual_ptr},")
        lines.append(f"        g_matmul_fused_{name}_expected,")
        lines.append(f"        g_matmul_fused_{name}_input_f64,")
        lines.append(f"        g_matmul_fused_{name}_weight_f64,")
        lines.append(f"        {bias_f64_ptr},")
        lines.append(f"        {residual_f64_ptr},")
        lines.append(f"        g_matmul_fused_{name}_expected_f64,")
        lines.append("    },")
    lines.append("};")
    lines.append(
        "static const size_t g_matmul_fused_case_count = sizeof(g_matmul_fused_cases) / "
        "sizeof(g_matmul_fused_cases[0]);\n"
    )

    return "\n".join(lines)


def make_matmul_qkv_section() -> str:
    cases = [
        {
            "name": "bias",
            "n": 2,
            "k": 3,
            "m": 2,
            "input": np.array([0.25, -0.5, 1.0, -1.5, 2.0, 1.25], dtype=np.float64),
            "weights": {
                "q": np.array([[0.5, -0.25, 0.75], [-0.5, 0.1, 0.2]], dtype=np.float64),
                "k": np.array([[0.1, 0.4, -0.3], [0.6, -0.2, 0.5]], dtype=np.float64),
                "v": np.array([[-0.25, 0.5, 0.1], [0.35, -0.15, 0.2]], dtype=np.float64),
            },
            "bias": {
                "q": np.array([0.05, -0.02], dtype=np.float64),
                "k": np.array([-0.01, 0.03], dtype=np.float64),
                "v": np.array([0.02, -0.04], dtype=np.float64),
            },
        },
        {
            "name": "no_bias",
            "n": 1,
            "k": 4,
            "m": 3,
            "input": np.array([1.5, -2.0, 0.75, 3.25], dtype=np.float64),
            "weights": {
                "q": np.array([[0.2, -0.5, 0.1, 0.3], [-0.4, 0.25, 0.35, -0.15], [0.6, -0.2, 0.5, 0.1]], dtype=np.float64),
                "k": np.array([[0.1, 0.05, -0.3, 0.4], [0.2, -0.1, 0.25, -0.35], [-0.45, 0.15, -0.05, 0.2]],
                              dtype=np.float64),
                "v": np.array([[0.3, -0.25, 0.4, -0.1], [0.15, 0.05, -0.2, 0.35], [-0.05, 0.25, 0.1, -0.3]],
                              dtype=np.float64),
            },
            "bias": None,
        },
    ]

    lines: list[str] = []
    for case in cases:
        name = case["name"]
        n = case["n"]
        k = case["k"]
        m = case["m"]
        input_vals = case["input"]
        weight_q = case["weights"]["q"]
        weight_k = case["weights"]["k"]
        weight_v = case["weights"]["v"]
        fused_weight = np.vstack([weight_q, weight_k, weight_v])
        bias = case["bias"]
        if bias is not None:
            fused_bias = np.concatenate([bias["q"], bias["k"], bias["v"]])
        else:
            fused_bias = None

        expected_q = input_vals.reshape(n, k) @ weight_q.T
        expected_k = input_vals.reshape(n, k) @ weight_k.T
        expected_v = input_vals.reshape(n, k) @ weight_v.T
        if fused_bias is not None:
            expected_q += bias["q"].reshape(1, m)
            expected_k += bias["k"].reshape(1, m)
            expected_v += bias["v"].reshape(1, m)

        lines.append(f"static const float g_matmul_qkv_{name}_input[{n * k}] = {{{fmt_floats(input_vals)}}};")
        lines.append(f"static const double g_matmul_qkv_{name}_input_f64[{n * k}] = {{{fmt_doubles(input_vals)}}};")
        lines.append(f"static const float g_matmul_qkv_{name}_weight[{3 * m * k}] = {{{fmt_floats(fused_weight.ravel())}}};")
        lines.append(f"static const double g_matmul_qkv_{name}_weight_f64[{3 * m * k}] = "
                     f"{{{fmt_doubles(fused_weight.ravel())}}};")

        if fused_bias is not None:
            lines.append(f"static const float g_matmul_qkv_{name}_bias[{3 * m}] = {{{fmt_floats(fused_bias)}}};")
            lines.append(f"static const double g_matmul_qkv_{name}_bias_f64[{3 * m}] = "
                         f"{{{fmt_doubles(fused_bias)}}};")
        else:
            lines.append(f"static const float *g_matmul_qkv_{name}_bias = nullptr;")
            lines.append(f"static const double *g_matmul_qkv_{name}_bias_f64 = nullptr;")

        lines.append(f"static const float g_matmul_qkv_{name}_expected_q[{n * m}] = "
                     f"{{{fmt_floats(expected_q.ravel())}}};")
        lines.append(f"static const float g_matmul_qkv_{name}_expected_k[{n * m}] = "
                     f"{{{fmt_floats(expected_k.ravel())}}};")
        lines.append(f"static const float g_matmul_qkv_{name}_expected_v[{n * m}] = "
                     f"{{{fmt_floats(expected_v.ravel())}}};")
        lines.append(f"static const double g_matmul_qkv_{name}_expected_q_f64[{n * m}] = "
                     f"{{{fmt_doubles(expected_q.ravel())}}};")
        lines.append(f"static const double g_matmul_qkv_{name}_expected_k_f64[{n * m}] = "
                     f"{{{fmt_doubles(expected_k.ravel())}}};")
        lines.append(f"static const double g_matmul_qkv_{name}_expected_v_f64[{n * m}] = "
                     f"{{{fmt_doubles(expected_v.ravel())}}};")
        lines.append("")

    lines.append("static const struct {")
    lines.append("    const char *name;")
    lines.append("    size_t n;")
    lines.append("    size_t k;")
    lines.append("    size_t m;")
    lines.append("    int has_bias;")
    lines.append("    const float *input;")
    lines.append("    const float *weight;")
    lines.append("    const float *bias;")
    lines.append("    const float *expected_q;")
    lines.append("    const float *expected_k;")
    lines.append("    const float *expected_v;")
    lines.append("    const double *input_f64;")
    lines.append("    const double *weight_f64;")
    lines.append("    const double *bias_f64;")
    lines.append("    const double *expected_q_f64;")
    lines.append("    const double *expected_k_f64;")
    lines.append("    const double *expected_v_f64;")
    lines.append("} g_matmul_qkv_cases[] = {")

    for case in cases:
        name = case["name"]
        n = case["n"]
        k = case["k"]
        m = case["m"]
        has_bias = 1 if case["bias"] is not None else 0
        bias_ptr = f"g_matmul_qkv_{name}_bias" if has_bias else "nullptr"
        bias_f64_ptr = f"g_matmul_qkv_{name}_bias_f64" if has_bias else "nullptr"
        lines.append("    {")
        lines.append(f"        \"{name}\",")
        lines.append(f"        {n},")
        lines.append(f"        {k},")
        lines.append(f"        {m},")
        lines.append(f"        {has_bias},")
        lines.append(f"        g_matmul_qkv_{name}_input,")
        lines.append(f"        g_matmul_qkv_{name}_weight,")
        lines.append(f"        {bias_ptr},")
        lines.append(f"        g_matmul_qkv_{name}_expected_q,")
        lines.append(f"        g_matmul_qkv_{name}_expected_k,")
        lines.append(f"        g_matmul_qkv_{name}_expected_v,")
        lines.append(f"        g_matmul_qkv_{name}_input_f64,")
        lines.append(f"        g_matmul_qkv_{name}_weight_f64,")
        lines.append(f"        {bias_f64_ptr},")
        lines.append(f"        g_matmul_qkv_{name}_expected_q_f64,")
        lines.append(f"        g_matmul_qkv_{name}_expected_k_f64,")
        lines.append(f"        g_matmul_qkv_{name}_expected_v_f64,")
        lines.append("    },")
    lines.append("};")
    lines.append(
        "static const size_t g_matmul_qkv_case_count = sizeof(g_matmul_qkv_cases) / "
        "sizeof(g_matmul_qkv_cases[0]);\n"
    )

    return "\n".join(lines)


def pack_float32(value: float) -> bytes:
    return np.float32(value).tobytes()




def make_quantization_section() -> str:
    lines: list[str] = []

    params_values = np.array([-10.0, -5.0, 5.0, 10.0], dtype=np.float64)
    min_val = float(np.min(params_values))
    max_val = float(np.max(params_values))
    scale_int8 = (max_val - min_val) / (127.0 - (-128.0))
    zero_int8 = -min_val / scale_int8 - 128.0
    scale_uint8 = (max_val - min_val) / (255.0 - 0.0)
    zero_uint8 = -min_val / scale_uint8

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float values[4];")
    lines.append("    float scale_int8;")
    lines.append("    float zero_int8;")
    lines.append("    float scale_uint8;")
    lines.append("    float zero_uint8;")
    lines.append("} g_quant_params = {")
    lines.append("    4,")
    lines.append(f"    {{{fmt_floats(params_values)}}},")
    lines.append(f"    {scale_int8:.10g},")
    lines.append(f"    {zero_int8:.10g},")
    lines.append(f"    {scale_uint8:.10g},")
    lines.append(f"    {zero_uint8:.10g},")
    lines.append("};\n")

    values_int8 = np.array([-7.5, -3.0, -1.0, 0.0, 1.0, 3.5, 6.0, 8.0], dtype=np.float64)
    min_i8 = float(np.min(values_int8))
    max_i8 = float(np.max(values_int8))
    scale_i8 = (max_i8 - min_i8) / (127.0 - (-128.0))
    if scale_i8 < 1e-8:
        scale_i8 = 1.0
    zero_i8 = -min_i8 / scale_i8 - 128.0
    zero_i8 = max(-128.0, min(127.0, zero_i8))
    quant_i8 = round_half_away_from_zero(values_int8 / scale_i8 + zero_i8)
    quant_i8 = np.clip(quant_i8, -128, 127).astype(np.int8)
    dequant_i8 = (quant_i8.astype(np.float64) - zero_i8) * scale_i8

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float input[8];")
    lines.append("    float scale;")
    lines.append("    float zero_point;")
    lines.append("    int8_t quantized[8];")
    lines.append("    float dequantized[8];")
    lines.append("} g_quant_int8 = {")
    lines.append("    8,")
    lines.append(f"    {{{fmt_floats(values_int8)}}},")
    lines.append(f"    {scale_i8:.10g},")
    lines.append(f"    {zero_i8:.10g},")
    lines.append(f"    {{{fmt_ints(quant_i8)}}},")
    lines.append(f"    {{{fmt_floats(dequant_i8)}}},")
    lines.append("};\n")

    values_u8 = np.array([0.0, 0.5, 1.0, 2.25, 3.5, 6.0, 8.0, 10.0], dtype=np.float64)
    min_u8 = float(np.min(values_u8))
    max_u8 = float(np.max(values_u8))
    scale_u8 = (max_u8 - min_u8) / (255.0 - 0.0)
    if scale_u8 < 1e-8:
        scale_u8 = 1.0
    zero_u8 = -min_u8 / scale_u8
    zero_u8 = max(0.0, min(255.0, zero_u8))
    quant_u8 = round_half_away_from_zero(values_u8 / scale_u8 + zero_u8)
    quant_u8 = np.clip(quant_u8, 0, 255).astype(np.uint8)
    dequant_u8 = (quant_u8.astype(np.float64) - zero_u8) * scale_u8

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float input[8];")
    lines.append("    float scale;")
    lines.append("    float zero_point;")
    lines.append("    uint8_t quantized[8];")
    lines.append("    float dequantized[8];")
    lines.append("} g_quant_uint8 = {")
    lines.append("    8,")
    lines.append(f"    {{{fmt_floats(values_u8)}}},")
    lines.append(f"    {scale_u8:.10g},")
    lines.append(f"    {zero_u8:.10g},")
    lines.append(f"    {{{fmt_uints(quant_u8)}}},")
    lines.append(f"    {{{fmt_floats(dequant_u8)}}},")
    lines.append("};\n")


    return "\n".join(lines)


def make_conversion_section() -> str:
    lines: list[str] = []

    # F16 roundtrip
    f16_src = np.array([(i - 3) * 0.75 for i in range(8)], dtype=np.float64)
    f16_bits = np.asarray(f16_src, dtype=np.float16).view(np.uint16)
    f16_roundtrip = np.asarray(f16_bits, dtype=np.uint16).view(np.float16).astype(np.float64)

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float src[8];")
    lines.append("    uint16_t f16_bits[8];")
    lines.append("    float roundtrip[8];")
    lines.append("} g_conv_f16 = {")
    lines.append("    8,")
    lines.append(f"    {{{fmt_floats(f16_src)}}},")
    lines.append(f"    {{{fmt_uints(f16_bits)}}},")
    lines.append(f"    {{{fmt_floats(f16_roundtrip)}}},")
    lines.append("};\n")

    # BF16 roundtrip
    bf16_src = np.array([math.sin(i) for i in range(8)], dtype=np.float64)
    bf16_bits = float_to_bf16_bits(bf16_src)
    bf16_roundtrip = bf16_bits_to_float(bf16_bits)

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float src[8];")
    lines.append("    uint16_t bf16_bits[8];")
    lines.append("    float roundtrip[8];")
    lines.append("} g_conv_bf16 = {")
    lines.append("    8,")
    lines.append(f"    {{{fmt_floats(bf16_src)}}},")
    lines.append(f"    {{{fmt_uints(bf16_bits)}}},")
    lines.append(f"    {{{fmt_floats(bf16_roundtrip)}}},")
    lines.append("};\n")

    # F16 <-> BF16 bridge
    bridge_src = np.array([i * 0.5 - 1.0 for i in range(6)], dtype=np.float64)
    bridge_f16_bits = np.asarray(bridge_src, dtype=np.float16).view(np.uint16)
    bridge_bf16_bits = float_to_bf16_bits(bridge_src)
    bridge_bf16_from_f16 = float_to_bf16_bits(np.asarray(bridge_f16_bits, dtype=np.uint16).view(np.float16).astype(np.float64))
    bridge_f16_from_bf16 = np.asarray(bf16_bits_to_float(bridge_bf16_bits), dtype=np.float16).view(np.uint16)
    bridge_roundtrip_f32 = bf16_bits_to_float(bridge_bf16_from_f16)
    bridge_roundtrip_f32_alt = np.asarray(bridge_f16_from_bf16, dtype=np.uint16).view(np.float16).astype(np.float64)

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float src[6];")
    lines.append("    uint16_t f16_bits[6];")
    lines.append("    uint16_t bf16_bits[6];")
    lines.append("    uint16_t bf16_from_f16_bits[6];")
    lines.append("    float bf16_from_f16_roundtrip[6];")
    lines.append("    uint16_t f16_from_bf16_bits[6];")
    lines.append("    float f16_from_bf16_roundtrip[6];")
    lines.append("} g_conv_f16_bf16_bridge = {")
    lines.append("    6,")
    lines.append(f"    {{{fmt_floats(bridge_src)}}},")
    lines.append(f"    {{{fmt_uints(bridge_f16_bits)}}},")
    lines.append(f"    {{{fmt_uints(bridge_bf16_bits)}}},")
    lines.append(f"    {{{fmt_uints(bridge_bf16_from_f16)}}},")
    lines.append(f"    {{{fmt_floats(bridge_roundtrip_f32)}}},")
    lines.append(f"    {{{fmt_uints(bridge_f16_from_bf16)}}},")
    lines.append(f"    {{{fmt_floats(bridge_roundtrip_f32_alt)}}},")
    lines.append("};\n")

    fp8_src = np.array([(i - 8) * 0.4 for i in range(16)], dtype=np.float64)
    fp8_e4_bits = fp8_quantize_array(fp8_src, 4, 3, 7, False, 240.0)
    fp8_e5_bits = fp8_quantize_array(fp8_src, 5, 2, 15, True, 57344.0)
    fp8_e4_roundtrip = fp8_dequantize_array(fp8_e4_bits, 4, 3, 7, False)
    fp8_e5_roundtrip = fp8_dequantize_array(fp8_e5_bits, 5, 2, 15, True)

    lines.append("#if MARMOT_ENABLE_FP8\n")
    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float src[16];")
    lines.append("    uint8_t e4m3_bits[16];")
    lines.append("    float e4m3_roundtrip[16];")
    lines.append("    uint8_t e5m2_bits[16];")
    lines.append("    float e5m2_roundtrip[16];")
    lines.append("} g_conv_fp8 = {")
    lines.append("    16,")
    lines.append(f"    {{{fmt_floats(fp8_src)}}},")
    lines.append(f"    {{{fmt_uints(fp8_e4_bits)}}},")
    lines.append(f"    {{{fmt_floats(fp8_e4_roundtrip)}}},")
    lines.append(f"    {{{fmt_uints(fp8_e5_bits)}}},")
    lines.append(f"    {{{fmt_floats(fp8_e5_roundtrip)}}},")
    lines.append("};\n")

    fp8_half_src = np.array([0.25 * (i - 4.0) for i in range(12)], dtype=np.float64)
    fp8_half_f16_bits = np.asarray(fp8_half_src, dtype=np.float16).view(np.uint16)
    fp8_half_fp8_bits = fp8_quantize_array(fp8_half_src, 4, 3, 7, False, 240.0)
    fp8_half_roundtrip = fp8_dequantize_array(fp8_half_fp8_bits, 4, 3, 7, False)

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float src[12];")
    lines.append("    uint16_t f16_bits[12];")
    lines.append("    uint8_t fp8_bits[12];")
    lines.append("    float roundtrip[12];")
    lines.append("} g_conv_fp8_half_bridge = {")
    lines.append("    12,")
    lines.append(f"    {{{fmt_floats(fp8_half_src)}}},")
    lines.append(f"    {{{fmt_uints(fp8_half_f16_bits)}}},")
    lines.append(f"    {{{fmt_uints(fp8_half_fp8_bits)}}},")
    lines.append(f"    {{{fmt_floats(fp8_half_roundtrip)}}},")
    lines.append("};\n")

    fp8_bf16_src = np.array([math.sin(0.35 * i) for i in range(12)], dtype=np.float64)
    fp8_bf16_bits = float_to_bf16_bits(fp8_bf16_src)
    fp8_bf16_fp8_bits = fp8_quantize_array(fp8_bf16_src, 5, 2, 15, True, 57344.0)
    fp8_bf16_roundtrip = fp8_dequantize_array(fp8_bf16_fp8_bits, 5, 2, 15, True)

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float src[12];")
    lines.append("    uint16_t bf16_bits[12];")
    lines.append("    uint8_t fp8_bits[12];")
    lines.append("    float roundtrip[12];")
    lines.append("} g_conv_fp8_bf16_bridge = {")
    lines.append("    12,")
    lines.append(f"    {{{fmt_floats(fp8_bf16_src)}}},")
    lines.append(f"    {{{fmt_uints(fp8_bf16_bits)}}},")
    lines.append(f"    {{{fmt_uints(fp8_bf16_fp8_bits)}}},")
    lines.append(f"    {{{fmt_floats(fp8_bf16_roundtrip)}}},")
    lines.append("};\n")
    lines.append("#endif // MARMOT_ENABLE_FP8\n")

    # F64 paths
    f64_vals = np.array(
        [-7.75, -3.9375, -2.5, -1.0625, -0.1875, 0.4375, 1.8125, 3.625, 6.75, 11.5], dtype=np.float64
    )
    f64_to_f16 = np.asarray(f64_vals, dtype=np.float16).view(np.uint16)
    f64_to_bf16 = float_to_bf16_bits(f64_vals)
    f16_roundtrip = np.asarray(f64_to_f16, dtype=np.uint16).view(np.float16).astype(np.float64)
    bf16_roundtrip = bf16_bits_to_float(f64_to_bf16)
    f64_to_i64 = f64_vals.astype(np.int64)

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    double src[10];")
    lines.append("    uint16_t f16_bits[10];")
    lines.append("    uint16_t bf16_bits[10];")
    lines.append("    double f16_roundtrip[10];")
    lines.append("    double bf16_roundtrip[10];")
    lines.append("    int64_t trunc_i64[10];")
    lines.append("} g_conv_f64_paths = {")
    lines.append("    10,")
    lines.append(f"    {{{fmt_doubles(f64_vals)}}},")
    lines.append(f"    {{{fmt_uints(f64_to_f16)}}},")
    lines.append(f"    {{{fmt_uints(f64_to_bf16)}}},")
    lines.append(f"    {{{fmt_doubles(f16_roundtrip)}}},")
    lines.append(f"    {{{fmt_doubles(bf16_roundtrip)}}},")
    lines.append(f"    {{{fmt_ints(f64_to_i64)}}},")
    lines.append("};\n")

    # Integer bridge data
    f32_vals = np.array([-9.75, -6.5, -3.25, -1.5, -0.25, 0.75, 2.5, 4.75, 7.5, 12.25], dtype=np.float32)
    f64_vals_int = np.array([-8.875, -5.5, -3.125, -0.875, 0.375, 1.625, 3.875, 6.25, 10.5, 15.75], dtype=np.float64)
    f32_to_i64 = f32_vals.astype(np.int64)
    f64_to_i64 = f64_vals_int.astype(np.int64)
    i64_vals = np.array([-17, -9, -4, -1, 0, 1, 3, 8, 21, 37], dtype=np.int64)
    i64_to_f32 = i64_vals.astype(np.float32)
    i64_to_f64 = i64_vals.astype(np.float64)

    lines.append("static const struct {")
    lines.append("    size_t length;")
    lines.append("    float f32_src[10];")
    lines.append("    int64_t f32_to_i64[10];")
    lines.append("    double f64_src[10];")
    lines.append("    int64_t f64_to_i64[10];")
    lines.append("    int64_t i64_src[10];")
    lines.append("    float i64_to_f32[10];")
    lines.append("    double i64_to_f64[10];")
    lines.append("} g_conv_i64_bridge = {")
    lines.append("    10,")
    lines.append(f"    {{{fmt_floats(f32_vals)}}},")
    lines.append(f"    {{{fmt_ints(f32_to_i64)}}},")
    lines.append(f"    {{{fmt_doubles(f64_vals_int)}}},")
    lines.append(f"    {{{fmt_ints(f64_to_i64)}}},")
    lines.append(f"    {{{fmt_ints(i64_vals)}}},")
    lines.append(f"    {{{fmt_floats(i64_to_f32)}}},")
    lines.append(f"    {{{fmt_doubles(i64_to_f64)}}},")
    lines.append("};")

    return "\n".join(lines)


def main() -> None:
    header_lines: list[str] = []
    header_lines.append("// Auto-generated by tests/golden/generate_backend_golden.py. Do not edit manually.")
    header_lines.append("#pragma once")
    header_lines.append("#include <stddef.h>")
    header_lines.append("#include <stdint.h>")
    header_lines.append("#include \"marmot/types.h\"")
    header_lines.append("#include \"marmot/ops_types.h\"")
    header_lines.append("")
    header_lines.append("// Elementwise operation golden data\n")
    header_lines.append(make_elementwise_section())
    header_lines.append("\n// Activation function golden data\n")
    header_lines.append(make_activation_section())
    header_lines.append("\n// Normalization golden data\n")
    header_lines.append(make_normalization_section())
    header_lines.append("\n// Softmax golden data\n")
    header_lines.append(make_softmax_section())
    header_lines.append("\n// Reduction golden data\n")
    header_lines.append(make_reduction_section())
    header_lines.append("\n// Tensor operation golden data\n")
    header_lines.append(make_tensor_ops_section())
    header_lines.append("\n// Matmul golden data\n")
    header_lines.append(make_matmul_section())
    header_lines.append("\n// FP8 matmul golden data\n")
    header_lines.append(make_matmul_fp8_section())
    header_lines.append("\n// Fused matmul epilogue golden data\n")
    header_lines.append(make_matmul_fused_section())
    header_lines.append("\n// Fused QKV matmul golden data\n")
    header_lines.append(make_matmul_qkv_section())
    header_lines.append("\n// Quantization golden data\n")
    header_lines.append(make_quantization_section())
    header_lines.append("\n// Conversion golden data\n")
    header_lines.append(make_conversion_section())

    HEADER_PATH.write_text("\n".join(header_lines) + "\n")


if __name__ == "__main__":
    main()
