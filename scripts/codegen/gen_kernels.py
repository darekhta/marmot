#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jinja2

from backend_config import BackendConfig, get_backend_config
from codegen_base import make_jinja_env, write_output, write_stamp
from def_parser import (
    KernelDescriptor,
    QuantSchemeDescriptor,
    collect_descriptors,
    parse_quant_schemes,
    validate_descriptor,
)


DTYPE_MAP: Dict[str, str] = {
    "FLOAT16": "MARMOT_DTYPE_FLOAT16",
    "FLOAT32": "MARMOT_DTYPE_FLOAT32",
    "FLOAT64": "MARMOT_DTYPE_FLOAT64",
    "BFLOAT16": "MARMOT_DTYPE_BFLOAT16",
    "FLOAT8_E4M3": "MARMOT_DTYPE_FLOAT8_E4M3",
    "FLOAT8_E5M2": "MARMOT_DTYPE_FLOAT8_E5M2",
    "INT8": "MARMOT_DTYPE_INT8",
    "INT16": "MARMOT_DTYPE_INT16",
    "INT32": "MARMOT_DTYPE_INT32",
    "INT64": "MARMOT_DTYPE_INT64",
    "UINT8": "MARMOT_DTYPE_UINT8",
    "UINT16": "MARMOT_DTYPE_UINT16",
    "UINT32": "MARMOT_DTYPE_UINT32",
    "UINT64": "MARMOT_DTYPE_UINT64",
    "ANY": "MARMOT_DTYPE_COUNT",
}

_DTYPE_RANK = {
    "MARMOT_DTYPE_FLOAT32": 0,
    "MARMOT_DTYPE_FLOAT16": 1,
    "MARMOT_DTYPE_BFLOAT16": 2,
    "MARMOT_DTYPE_INT32": 3,
    "MARMOT_DTYPE_INT16": 4,
    "MARMOT_DTYPE_INT8": 5,
    "MARMOT_DTYPE_UINT8": 6,
    "MARMOT_DTYPE_UINT16": 7,
    "MARMOT_DTYPE_UINT32": 8,
    "MARMOT_DTYPE_UINT64": 9,
    "MARMOT_DTYPE_FLOAT64": 10,
    "MARMOT_DTYPE_INT64": 11,
    "MARMOT_DTYPE_INT4": 12,
    "MARMOT_DTYPE_UINT4": 13,
    "MARMOT_DTYPE_FLOAT8_E4M3": 14,
    "MARMOT_DTYPE_FLOAT8_E5M2": 15,
}

PROFILE_MAP = {
    # CPU implementation variants (ordered by preference: Accelerate > SIMD > Scalar)
    "ACCELERATE": "MARMOT_PROFILE_ACCELERATE",
    "NEON": "MARMOT_PROFILE_NEON",
    "AVX2": "MARMOT_PROFILE_AVX2",
    "AVX512": "MARMOT_PROFILE_AVX512",
    "SCALAR": "MARMOT_PROFILE_SCALAR",
    # Generic/None
    "NONE": "MARMOT_PROFILE_INVALID",
}

MATMUL_LAYOUT_MAP = {
    "NN": "MARMOT_MATMUL_LAYOUT_NN",
    "NT": "MARMOT_MATMUL_LAYOUT_NT",
    "TN": "MARMOT_MATMUL_LAYOUT_TN",
    "TT": "MARMOT_MATMUL_LAYOUT_TT",
    "INVALID": "MARMOT_MATMUL_LAYOUT_INVALID",
}

WEIGHT_LAYOUT_MAP = {
    "SEPARATE": "MARMOT_WEIGHT_LAYOUT_SEPARATE",
    "PACKED_3MK": "MARMOT_WEIGHT_LAYOUT_PACKED_3MK",
}

STRIDE_MODE_MAP = {
    "ANY": "MARMOT_STRIDE_MODE_ANY",
    "CONTIGUOUS": "MARMOT_STRIDE_MODE_CONTIGUOUS",
    "ROW_STRIDED": "MARMOT_STRIDE_MODE_ROW_STRIDED",
    "STRIDED": "MARMOT_STRIDE_MODE_STRIDED",
}

PLATFORM_GUARD_MAP = {
    "ACCELERATE": "MARMOT_ENABLE_ACCELERATE",
    "NEON": "MARMOT_ENABLE_NEON",
    "AVX2": "MARMOT_ENABLE_AVX2",
    "AVX512": "MARMOT_ENABLE_AVX512",
    "ANY": "1",  # Always enabled (no guard needed)
}

# Maps PROFILE to default PLATFORM (if PLATFORM not explicitly specified)
PROFILE_TO_PLATFORM_MAP = {
    "SCALAR": "ANY",
    "ACCELERATE": "ACCELERATE",
    "NEON": "NEON",
    "AVX2": "AVX2",
    "AVX512": "AVX512",
    # Default
    "NONE": "ANY",
    "INVALID": "ANY",
}

EPILOGUE_MAP = {
    "BIAS": "MARMOT_EPILOGUE_BIAS",
    "ACTIVATION": "MARMOT_EPILOGUE_ACTIVATION",
    "RESIDUAL": "MARMOT_EPILOGUE_RESIDUAL",
    "ROPE": "MARMOT_EPILOGUE_ROPE",
}

FUSION_MAP = {
    "QKV_SHARED_INPUT": "MARMOT_FUSION_QKV_SHARED_INPUT",
    "ADD_RELU": "MARMOT_FUSION_ADD_RELU",
    "ADD_GELU": "MARMOT_FUSION_ADD_GELU",
    "ADD_SILU": "MARMOT_FUSION_ADD_SILU",
    "MUL_ADD": "MARMOT_FUSION_MUL_ADD",
    "MATMUL_BIAS": "MARMOT_FUSION_MATMUL_BIAS",
    "MATMUL_BIAS_RELU": "MARMOT_FUSION_MATMUL_BIAS_RELU",
    "MATMUL_BIAS_GELU": "MARMOT_FUSION_MATMUL_BIAS_GELU",
    "MATMUL_BIAS_SILU": "MARMOT_FUSION_MATMUL_BIAS_SILU",
    "LAYERNORM_AFFINE": "MARMOT_FUSION_LAYERNORM_AFFINE",
    "RMSNORM_SCALE": "MARMOT_FUSION_RMSNORM_SCALE",
    "QKV_PROJECTION": "MARMOT_FUSION_QKV_PROJECTION",
    "QKV_ROPE": "MARMOT_FUSION_QKV_ROPE",
    "ATTENTION_BLOCK": "MARMOT_FUSION_ATTENTION_BLOCK",
    "RESIDUAL_ADD": "MARMOT_FUSION_RESIDUAL_ADD",
    "DROPOUT": "MARMOT_FUSION_DROPOUT",
    "CUSTOM": "MARMOT_FUSION_CUSTOM",
}

FUSED_BINARY_OPS = {
    "add_relu": "ADD_RELU",
    "add_gelu": "ADD_GELU",
    "add_silu": "ADD_SILU",
}

ACTIVATION_MAP = {
    "IDENTITY": "MARMOT_DEVICE_UNARY_IDENTITY",
    "RELU": "MARMOT_DEVICE_UNARY_RELU",
    "GELU": "MARMOT_DEVICE_UNARY_GELU",
    "GELU_TANH": "MARMOT_DEVICE_UNARY_GELU_TANH",
    "SILU": "MARMOT_DEVICE_UNARY_SILU",
}

DTYPE_SHORT_MAP: Dict[str, str] = {
    "FLOAT16": "f16",
    "FLOAT32": "f32",
    "FLOAT64": "f64",
    "BFLOAT16": "bf16",
    "FLOAT8_E4M3": "fp8_e4m3",
    "FLOAT8_E5M2": "fp8_e5m2",
    "INT8": "i8",
    "INT16": "i16",
    "INT32": "i32",
    "INT64": "i64",
    "UINT8": "u8",
    "UINT16": "u16",
    "UINT32": "u32",
    "UINT64": "u64",
}

ELEMENTWISE_BINARY_OPS = {
    "add",
    "add_gelu",
    "add_relu",
    "add_silu",
    "mul",
    "sub",
    "div",
    "min",
    "max",
    "pow",
    "mod",
    "swiglu",
    "geglu",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "bitwise_shl",
    "bitwise_shr",
    "bitwise_shr_logical",
    "compare_eq",
    "compare_ne",
    "compare_lt",
    "compare_le",
    "compare_gt",
    "compare_ge",
}

UNARY_OPS = {
    "abs",
    "neg",
    "sign",
    "sqrt",
    "exp",
    "log",
    "bitwise_not",
    "relu",
    "gelu",
    "gelu_tanh",
    "silu",
    "sigmoid",
    "tanh",
    "mish",
    "elu",
    "selu",
    "leaky_relu",
    "prelu",
}

REDUCTION_OPS = {
    "reduction_sum",
    "reduction_mean",
    "reduction_max",
    "reduction_min",
    "reduction_variance",
    "reduction_std",
    "reduction_norm_l1",
    "reduction_norm_l2",
    "reduction_prod",
    "reduction_argmax",
    "reduction_argmin",
    "reduction_any",
    "reduction_all",
}

TERNARY_OPS = {
    "fma",
    "where",
}

QUANTIZATION_OPS = {
    "quantize",
    "dequantize",
    "compute_quant_params",
}

REDUCTION_OP_ENUM: Dict[str, str] = {
    "reduction_sum": "MARMOT_DEVICE_REDUCTION_SUM",
    "reduction_mean": "MARMOT_DEVICE_REDUCTION_MEAN",
    "reduction_max": "MARMOT_DEVICE_REDUCTION_MAX",
    "reduction_min": "MARMOT_DEVICE_REDUCTION_MIN",
    "reduction_variance": "MARMOT_DEVICE_REDUCTION_VARIANCE",
    "reduction_std": "MARMOT_DEVICE_REDUCTION_STD",
    "reduction_norm_l1": "MARMOT_DEVICE_REDUCTION_NORM_L1",
    "reduction_norm_l2": "MARMOT_DEVICE_REDUCTION_NORM_L2",
    "reduction_prod": "MARMOT_DEVICE_REDUCTION_PROD",
    "reduction_argmax": "MARMOT_DEVICE_REDUCTION_ARGMAX",
    "reduction_argmin": "MARMOT_DEVICE_REDUCTION_ARGMIN",
    "reduction_any": "MARMOT_DEVICE_REDUCTION_ANY",
    "reduction_all": "MARMOT_DEVICE_REDUCTION_ALL",
}

TERNARY_OP_ENUM: Dict[str, str] = {
    "fma": "MARMOT_DEVICE_TERNARY_FMA",
    "where": "MARMOT_DEVICE_TERNARY_WHERE",
}

QUANT_KIND_MAP: Dict[str, str] = {
    "GENERIC": "MARMOT_QUANT_KIND_GENERIC",
    "Q4_0": "MARMOT_QUANT_KIND_Q4_0",
    "Q4_1": "MARMOT_QUANT_KIND_Q4_1",
    "Q5_0": "MARMOT_QUANT_KIND_Q5_0",
    "Q5_1": "MARMOT_QUANT_KIND_Q5_1",
    "Q8_0": "MARMOT_QUANT_KIND_Q8_0",
    "Q8_1": "MARMOT_QUANT_KIND_Q8_1",
    "Q2_K": "MARMOT_QUANT_KIND_Q2_K",
    "Q3_K": "MARMOT_QUANT_KIND_Q3_K",
    "Q4_K": "MARMOT_QUANT_KIND_Q4_K",
    "Q5_K": "MARMOT_QUANT_KIND_Q5_K",
    "Q6_K": "MARMOT_QUANT_KIND_Q6_K",
    "Q8_K": "MARMOT_QUANT_KIND_Q8_K",
}

EMBEDDING_OPS = {
    "embedding",
}

TENSOR_OPS = {
    "contiguous",
    "reshape",
    "transpose",
    "concat",
    "view",
    "slice",
    "gather_rows",
    "scatter_u64_to_i32",
}

CONVERSION_OPS = {
    "convert",
    "cast",
}

PAGED_ATTENTION_OPS = {
    "paged_attention",
}

TOPK_OPS = {
    "topk",
}

MOE_OPS = {
    "moe_experts",
}

VEC_DOT_OPS = {
    "vec_dot",
}

BINARY_OP_ENUM: Dict[str, str] = {
    "add": "MARMOT_DEVICE_BINARY_ADD",
    "mul": "MARMOT_DEVICE_BINARY_MUL",
    "sub": "MARMOT_DEVICE_BINARY_SUB",
    "div": "MARMOT_DEVICE_BINARY_DIV",
    "min": "MARMOT_DEVICE_BINARY_MIN",
    "max": "MARMOT_DEVICE_BINARY_MAX",
    "pow": "MARMOT_DEVICE_BINARY_POW",
    "mod": "MARMOT_DEVICE_BINARY_MOD",
    "swiglu": "MARMOT_DEVICE_BINARY_SWIGLU",
    "geglu": "MARMOT_DEVICE_BINARY_GEGLU",
    "bitwise_and": "MARMOT_DEVICE_BINARY_BITWISE_AND",
    "bitwise_or": "MARMOT_DEVICE_BINARY_BITWISE_OR",
    "bitwise_xor": "MARMOT_DEVICE_BINARY_BITWISE_XOR",
    "bitwise_shl": "MARMOT_DEVICE_BINARY_BITWISE_SHIFT_LEFT",
    "bitwise_shr": "MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT",
    "bitwise_shr_logical": "MARMOT_DEVICE_BINARY_BITWISE_SHIFT_RIGHT_LOGICAL",
    "compare_eq": "MARMOT_DEVICE_BINARY_COMPARE_EQ",
    "compare_ne": "MARMOT_DEVICE_BINARY_COMPARE_NE",
    "compare_lt": "MARMOT_DEVICE_BINARY_COMPARE_LT",
    "compare_le": "MARMOT_DEVICE_BINARY_COMPARE_LE",
    "compare_gt": "MARMOT_DEVICE_BINARY_COMPARE_GT",
    "compare_ge": "MARMOT_DEVICE_BINARY_COMPARE_GE",
}

UNARY_OP_ENUM: Dict[str, str] = {
    "abs": "MARMOT_DEVICE_UNARY_ABS",
    "neg": "MARMOT_DEVICE_UNARY_NEG",
    "sign": "MARMOT_DEVICE_UNARY_SIGN",
    "sqrt": "MARMOT_DEVICE_UNARY_SQRT",
    "exp": "MARMOT_DEVICE_UNARY_EXP",
    "log": "MARMOT_DEVICE_UNARY_LOG",
    "bitwise_not": "MARMOT_DEVICE_UNARY_BITWISE_NOT",
    "relu": "MARMOT_DEVICE_UNARY_RELU",
    "gelu": "MARMOT_DEVICE_UNARY_GELU",
    "gelu_tanh": "MARMOT_DEVICE_UNARY_GELU_TANH",
    "silu": "MARMOT_DEVICE_UNARY_SILU",
    "sigmoid": "MARMOT_DEVICE_UNARY_SIGMOID",
    "tanh": "MARMOT_DEVICE_UNARY_TANH",
    "mish": "MARMOT_DEVICE_UNARY_MISH",
    "elu": "MARMOT_DEVICE_UNARY_ELU",
    "selu": "MARMOT_DEVICE_UNARY_SELU",
    "leaky_relu": "MARMOT_DEVICE_UNARY_LEAKY_RELU",
    "prelu": "MARMOT_DEVICE_UNARY_PRELU",
}


def stable_hash(value: str, *, start: int, end: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    span = end - start
    return start + (int.from_bytes(digest[:4], "big") % span)


def stable_op_id(name: str) -> int:
    return stable_hash(f"op:{name}", start=100, end=999)


def allocate_op_ids(ops: List[str]) -> Dict[str, int]:
    start, end = (100, 999)
    span = end - start
    used: Dict[int, str] = {}
    mapping: Dict[str, int] = {}
    for op in sorted(set(ops)):
        candidate = stable_op_id(op)
        if candidate in used:
            for offset in range(1, span):
                probe = start + ((candidate - start + offset) % span)
                if probe not in used:
                    candidate = probe
                    break
            else:
                raise ValueError(f"Op ID range exhausted while allocating '{op}'")
        used[candidate] = op
        mapping[op] = candidate
    return mapping


def stable_kernel_id(name: str, backend: str) -> int:
    ranges = {
        "metal": (1000, 9999),
        "cpu": (10000, 59999),
        "cuda": (60000, 99999),
    }
    if backend not in ranges:
        raise ValueError(f"Unsupported backend '{backend}' for kernel '{name}'")
    return stable_hash(f"{backend}:{name}", start=ranges[backend][0], end=ranges[backend][1])


def allocate_kernel_ids(backend: str, descriptors: List[KernelDescriptor]) -> Dict[str, int]:
    start, end = {
        "metal": (1000, 9999),
        "cpu": (10000, 59999),
        "cuda": (60000, 99999),
    }[backend]
    span = end - start
    used: Dict[int, str] = {}
    mapping: Dict[str, int] = {}
    for desc in sorted(descriptors, key=lambda d: d.name):
        candidate = stable_kernel_id(desc.name, backend)
        if candidate in used:
            # Linear probe to avoid collisions while keeping deterministic order.
            for offset in range(1, span):
                probe = start + ((candidate - start + offset) % span)
                if probe not in used:
                    candidate = probe
                    break
            else:
                raise ValueError(f"Kernel ID range exhausted while allocating '{desc.name}'")
        used[candidate] = desc.name
        mapping[desc.name] = candidate
    return mapping


def expand_profiles(descriptors: List[KernelDescriptor]) -> List[KernelDescriptor]:
    expanded: List[KernelDescriptor] = []
    for desc in descriptors:
        profiles = desc.core_fields.get("PROFILES")
        if profiles is None or "PROFILE" in desc.core_fields:
            expanded.append(desc)
            continue

        if not isinstance(profiles, dict):
            expanded.append(desc)
            continue

        normalized: Dict[str, str] = {}
        for key, value in profiles.items():
            key_str = str(key).strip().upper()
            value_str = str(value).strip()
            if not key_str or not value_str:
                continue
            normalized[key_str] = value_str

        for profile_key, impl_symbol in normalized.items():
            core_fields = dict(desc.core_fields)
            core_fields.pop("PROFILES", None)
            core_fields["PROFILE"] = profile_key
            core_fields["IMPL_FUNCTION"] = impl_symbol

            expanded.append(
                KernelDescriptor(
                    name=f"{desc.name}_{profile_key.lower()}",
                    core_fields=core_fields,
                    extensions=dict(desc.extensions),
                    path=desc.path,
                    source=desc.source,
                    locations=dict(desc.locations),
                )
            )
    return expanded


def _normalize_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value:
        return [str(value).strip()]
    return []


def _map_dtypes(names: List[str]) -> List[str]:
    mapped = []
    for name in names:
        key = name.upper()
        if key not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype '{name}' in descriptor")
        mapped.append(DTYPE_MAP[key])
    return mapped


def render_traits_header(
    env: jinja2.Environment,
    ops: List[str],
    kernels: List[Tuple[str, str, int]],
    qschemes: List[str],
    profiles: List[str],
) -> str:
    template = env.get_template("traits_ids.h.j2")
    op_ids = allocate_op_ids(ops)
    op_entries = [(op, op_ids[op]) for op in sorted(set(ops))]
    kernel_entries = [(backend, name, kid) for backend, name, kid in kernels]

    # Generate profile entries with stable IDs (start from 1, 0 is INVALID)
    profile_entries = []
    profile_set = set(profiles)
    # Ensure INVALID is first
    profile_entries.append(("INVALID", 0))
    # Add all unique profiles in sorted order
    for idx, profile in enumerate(sorted(profile_set), start=1):
        profile_entries.append((profile, idx))

    return template.render(
        op_entries=op_entries,
        kernel_entries=kernel_entries,
        qscheme_entries=[(q, idx) for idx, q in enumerate(qschemes)],
        profile_entries=profile_entries,
    )


def _mask_expr(values: List[str]) -> str:
    terms = [f"(UINT64_C(1) << {value})" for value in values]
    if not terms:
        return "0"
    return " | ".join(terms)


def _flags_expr(entries: List[str], table: Dict[str, str]) -> str:
    flags: List[str] = []
    for entry in entries:
        key = entry.upper()
        if key == "NONE":
            continue
        if key not in table:
            raise ValueError(f"Unsupported flag '{entry}'")
        flags.append(table[key])
    if not flags:
        return "0"
    return " | ".join(flags)


def _dispatch_kind(op: str) -> str:
    if op in ELEMENTWISE_BINARY_OPS:
        return "binary"
    if op in UNARY_OPS:
        return "unary"
    if op in REDUCTION_OPS:
        return "reduction"
    if op in EMBEDDING_OPS:
        return "embedding"
    if op in TENSOR_OPS:
        return "tensor_ops"
    if op in CONVERSION_OPS:
        return "conversion"
    if op in TERNARY_OPS:
        return "ternary"
    if op in QUANTIZATION_OPS:
        return "quantization"
    if op in PAGED_ATTENTION_OPS:
        return "paged_attention"
    if op in TOPK_OPS:
        return "topk"
    if op in MOE_OPS:
        return "moe_experts"
    if op in VEC_DOT_OPS:
        return "vec_dot"
    if op in {"rms_norm", "rms_norm_gemma"}:
        return "rms_norm"
    if op == "layernorm":
        return "layernorm"
    if op == "softmax":
        return "softmax"
    if op == "rope":
        return "rope"
    if op == "rms_norm_gemma":
        return "rms_norm"
    if op in {"matmul", "matmul_bias", "matmul_bias_relu", "matmul_bias_gelu", "matmul_bias_silu"}:
        return "matmul"
    if op in {"qkv_rope", "qkv_shared_input", "qkv_projection"}:
        return "qkv"
    raise ValueError(f"Unsupported op '{op}'")


def _args_base(op: str) -> str:
    if op in UNARY_OPS:
        return "unary"
    if op in REDUCTION_OPS:
        return "reduction"
    if op in PAGED_ATTENTION_OPS:
        return "paged_attention"
    if op in TOPK_OPS:
        return "topk"
    if op in MOE_OPS:
        return "moe_experts"
    if op in EMBEDDING_OPS:
        return "embedding"
    if op in TENSOR_OPS:
        return "tensor_op"
    if op in CONVERSION_OPS:
        return "convert"
    if op in TERNARY_OPS:
        return "ternary"
    if op in QUANTIZATION_OPS:
        return op
    if op in VEC_DOT_OPS:
        return "vec_dot"
    if op in {"matmul", "matmul_bias", "matmul_bias_relu", "matmul_bias_gelu", "matmul_bias_silu"}:
        return "matmul"
    if op in {"qkv_rope", "qkv_shared_input", "qkv_projection"}:
        return "qkv"
    if op == "rope":
        return "rope"
    return op


def _platform_guard(desc: KernelDescriptor) -> str:
    if "PLATFORM" in desc.core_fields:
        platform = str(desc.core_fields.get("PLATFORM")).strip().upper()
    else:
        profile_key = str(desc.core_fields.get("PROFILE", "")).upper()
        platform = PROFILE_TO_PLATFORM_MAP.get(profile_key, "ANY")
    return PLATFORM_GUARD_MAP.get(platform, "1")


def render_query_source(
    env: jinja2.Environment,
    backend: str,
    descriptors: List[KernelDescriptor],
    kernel_ids: Dict[str, int],
) -> str:
    template = env.get_template("backend_query.c.j2")
    op_index_by_name = {desc.name: idx + 1 for idx, desc in enumerate(sorted(descriptors, key=lambda d: d.name))}
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for desc in descriptors:
        op = desc.core_fields["OP"]
        grouped.setdefault(op, [])
        profile = PROFILE_MAP.get(str(desc.core_fields.get("PROFILE", "")).upper(), "MARMOT_PROFILE_INVALID")
        matmul_layout = MATMUL_LAYOUT_MAP.get(
            str(desc.core_fields.get("MATMUL_LAYOUT", "INVALID")).upper(),
            "MARMOT_MATMUL_LAYOUT_INVALID",
        )
        input_dtypes = _map_dtypes(_normalize_list(desc.core_fields.get("INPUT_DTYPE")))
        weight_dtypes = _map_dtypes(_normalize_list(desc.core_fields.get("WEIGHT_DTYPE")))
        output_dtypes = _map_dtypes(_normalize_list(desc.core_fields.get("OUTPUT_DTYPE")))
        accum_dtype = DTYPE_MAP.get(str(desc.core_fields.get("ACCUM_DTYPE", "")).upper(), "MARMOT_DTYPE_COUNT")
        qschemes = [f"MARMOT_QSCHEME_{q.upper()}" for q in _normalize_list(desc.core_fields.get("WEIGHT_QUANT"))]
        if not qschemes:
            qschemes = ["MARMOT_QSCHEME_NONE"]
        weight_layout = WEIGHT_LAYOUT_MAP.get(
            str(desc.core_fields.get("WEIGHT_LAYOUT", "INVALID")).upper(),
            "MARMOT_WEIGHT_LAYOUT_INVALID",
        )
        stride_mode_entry = str(desc.core_fields.get("STRIDE_MODE", "CONTIGUOUS")).strip().upper()
        stride_mode = STRIDE_MODE_MAP.get(stride_mode_entry, "MARMOT_STRIDE_MODE_CONTIGUOUS")
        epilogue_entries = _normalize_list(desc.core_fields.get("EPILOGUE"))
        fusion_entries = _normalize_list(desc.core_fields.get("FUSION"))
        activation_entry = str(desc.core_fields.get("ACTIVATION", "")).strip().upper()
        activation = ACTIVATION_MAP.get(activation_entry, "MARMOT_DEVICE_UNARY_COUNT")
        data_dtypes = input_dtypes + weight_dtypes + output_dtypes
        requires_fp8 = any("FLOAT8" in dtype for dtype in data_dtypes)
        platform_guard = _platform_guard(desc)

        grouped[op].append(
            {
                "profile": profile,
                "matmul_layout": matmul_layout,
                "input_mask": _mask_expr(input_dtypes),
                "weight_mask": _mask_expr(weight_dtypes),
                "output_mask": _mask_expr(output_dtypes),
                "accum_dtype": accum_dtype,
                "qscheme_mask": _mask_expr(qschemes),
                "weight_layout": weight_layout,
                "stride_mode": stride_mode,
                "epilogue_mask": _flags_expr(epilogue_entries, EPILOGUE_MAP),
                "fusion_mask": _flags_expr(fusion_entries, FUSION_MAP),
                "activation": activation,
                "kernel_id": f"MARMOT_KERNEL_{backend.upper()}_{desc.name.upper()}",
                "op_index": op_index_by_name[desc.name],
                "requires_fp8": requires_fp8,
                "platform_guard": platform_guard,
            }
        )
    return template.render(backend=backend, grouped=grouped)


# Kernel category classification
MATMUL_KINDS = {"matmul", "qkv"}
ELEMENTWISE_KINDS = {"binary", "unary", "ternary"}
REDUCTION_KINDS = {"reduction"}
NEURAL_KINDS = {"softmax", "paged_attention", "rms_norm", "layernorm", "rope", "topk", "moe_experts"}
MISC_KINDS = {"quantization", "conversion", "embedding", "vec_dot", "tensor_ops"}


def _categorize_kernel(kind: str) -> str:
    """Categorize a kernel kind into a dispatch category."""
    if kind in MATMUL_KINDS:
        return "matmul"
    if kind in ELEMENTWISE_KINDS:
        return "elementwise"
    if kind in REDUCTION_KINDS:
        return "reduction"
    if kind in NEURAL_KINDS:
        return "neural"
    if kind in MISC_KINDS:
        return "misc"
    return "misc"  # Default fallback


def _build_kernel_entry(backend: str, desc: KernelDescriptor) -> Dict:
    """Build a kernel entry dict for template rendering."""
    op = str(desc.core_fields.get("OP", "")).lower()
    kind = _dispatch_kind(op)
    kernel_id_enum = f"MARMOT_KERNEL_{backend.upper()}_{desc.name.upper()}"
    platform_guard = _platform_guard(desc)

    impl_override = str(desc.core_fields.get("IMPL_FUNCTION", "")).strip()

    dtype_fields: List[str] = []
    dtype_fields.extend(_normalize_list(desc.core_fields.get("INPUT_DTYPE")))
    dtype_fields.extend(_normalize_list(desc.core_fields.get("WEIGHT_DTYPE")))
    dtype_fields.extend(_normalize_list(desc.core_fields.get("OUTPUT_DTYPE")))
    requires_fp8 = any("FLOAT8" in str(dtype).upper() for dtype in dtype_fields)

    entry = {
        "name": desc.name,
        "impl_name": f"marmot_{backend}_{desc.name}_impl",
        "kernel_id": kernel_id_enum,
        "platform_guard": platform_guard,
        "requires_fp8": requires_fp8,
        "kind": kind,
        "op": op,
        "args_base": _args_base(op),
        "category": _categorize_kernel(kind),
    }

    if impl_override:
        entry["kernel_func"] = impl_override

    # Add op_enum for binary/unary/reduction/ternary operations
    if kind == "binary":
        op_enum = BINARY_OP_ENUM.get(op)
        if op_enum:
            entry["op_enum"] = op_enum
    elif kind == "unary":
        op_enum = UNARY_OP_ENUM.get(op)
        if op_enum:
            entry["op_enum"] = op_enum
    elif kind == "reduction":
        op_enum = REDUCTION_OP_ENUM.get(op)
        if op_enum:
            entry["op_enum"] = op_enum
    elif kind == "ternary":
        op_enum = TERNARY_OP_ENUM.get(op)
        if op_enum:
            entry["op_enum"] = op_enum
    elif kind == "quantization":
        quant = str(desc.core_fields.get("WEIGHT_QUANT", "")).upper()
        entry["quant_kind"] = QUANT_KIND_MAP.get(quant, "MARMOT_QUANT_KIND_GENERIC")
    elif kind == "matmul":
        # Derive direct kernel function for non-quantized matmul
        weight_quant = str(desc.core_fields.get("WEIGHT_QUANT", "")).upper()
        if not weight_quant:  # Non-quantized matmul
            input_dtype = str(desc.core_fields.get("INPUT_DTYPE", "")).upper()
            layout = str(desc.core_fields.get("MATMUL_LAYOUT", "NT")).upper()
            dtype_short = DTYPE_SHORT_MAP.get(input_dtype)
            if dtype_short:
                if "kernel_func" not in entry:
                    if backend == "cpu":
                        layout_suffix = "_nn" if layout == "NN" else ""
                        entry["kernel_func"] = f"cpu_matmul_{dtype_short}_scalar{layout_suffix}"
                    elif backend == "metal":
                        layout_suffix = "_nn" if layout == "NN" else "_nt"
                        entry["kernel_func"] = f"metal_matmul_{dtype_short}{layout_suffix}"
                entry["layout"] = layout
                entry["is_quantized"] = False
        else:
            entry["is_quantized"] = True
            entry["weight_quant"] = weight_quant.lower()
    elif kind == "qkv":
        # QKV dispatch - quantized or dense
        weight_quant = str(desc.core_fields.get("WEIGHT_QUANT", "")).upper()
        input_dtype = str(desc.core_fields.get("INPUT_DTYPE", "")).upper()
        if weight_quant:
            entry["is_quantized"] = True
            entry["weight_quant"] = weight_quant.lower()
        else:
            entry["is_quantized"] = False
            dtype_short = DTYPE_SHORT_MAP.get(input_dtype, input_dtype.lower())
            entry["input_dtype"] = dtype_short

    if backend == "cpu" and kind == "binary" and "kernel_func" not in entry:
        entry["kernel_func"] = f"cpu_{desc.name}"

    return entry


def render_metal_unary_tables(env: jinja2.Environment, descriptors: List[KernelDescriptor]) -> str:
    template = env.get_template("metal_unary_tables.gen.h.j2")

    unary_entries: List[Dict[str, object]] = []
    for desc in descriptors:
        op = str(desc.core_fields.get("OP", "")).lower()
        if op not in UNARY_OP_ENUM:
            continue
        if _dispatch_kind(op) != "unary":
            continue

        entry = _build_kernel_entry("metal", desc)
        kernel_info = {
            "platform_guard": entry["platform_guard"],
            "requires_fp8": entry["requires_fp8"],
        }
        kernel_name = desc.name

        for dtype_name in _normalize_list(desc.core_fields.get("INPUT_DTYPE")):
            dtype_enum = DTYPE_MAP.get(str(dtype_name).upper())
            if dtype_enum is None:
                continue
            unary_entries.append(
                {
                    "dtype_enum": dtype_enum,
                    "op_enum": UNARY_OP_ENUM[op],
                    "kernel_name": kernel_name,
                    "kernel": kernel_info,
                    "op": op,
                }
            )

    unary_entries.sort(key=lambda e: (_DTYPE_RANK.get(str(e["dtype_enum"]), 999), str(e["op_enum"]), str(e["kernel_name"])))

    unary_by_dtype: Dict[str, List[Dict[str, object]]] = {}
    for entry in unary_entries:
        unary_by_dtype.setdefault(str(entry["dtype_enum"]), []).append(entry)

    unary_groups: List[Dict[str, object]] = []
    for dtype_enum in sorted(unary_by_dtype, key=lambda d: _DTYPE_RANK.get(d, 999)):
        entries = unary_by_dtype[dtype_enum]
        group_kernel = entries[0]["kernel"]
        group_entries = [
            {
                "op_enum": entry["op_enum"],
                "kernel_name": entry["kernel_name"],
            }
            for entry in sorted(entries, key=lambda e: str(e["op_enum"]))
        ]
        unary_groups.append(
            {
                "dtype_enum": dtype_enum,
                "kernel": group_kernel,
                "entries": group_entries,
            }
        )

    vec4_ops = {"relu", "gelu", "silu"}
    vec4_dtypes = [
        "MARMOT_DTYPE_FLOAT32",
        "MARMOT_DTYPE_FLOAT16",
        "MARMOT_DTYPE_BFLOAT16",
    ]
    vec4_dtype_set = set(vec4_dtypes)
    vec4_entries: List[Dict[str, object]] = []
    for entry in unary_entries:
        op = str(entry["op"])
        dtype_enum = str(entry["dtype_enum"])
        if op not in vec4_ops or dtype_enum not in vec4_dtype_set:
            continue
        kernel_name = str(entry["kernel_name"])
        vec4_entries.append(
            {
                "dtype_enum": dtype_enum,
                "op_enum": entry["op_enum"],
                "kernel_name": f"{kernel_name}_vec4",
            }
        )

    vec4_by_dtype: Dict[str, List[Dict[str, object]]] = {}
    for entry in vec4_entries:
        vec4_by_dtype.setdefault(str(entry["dtype_enum"]), []).append(entry)

    vec4_groups: List[Dict[str, object]] = []
    for dtype_enum in sorted(vec4_by_dtype, key=lambda d: _DTYPE_RANK.get(d, 999)):
        entries = vec4_by_dtype[dtype_enum]
        group_entries = [
            {
                "op_enum": entry["op_enum"],
                "kernel_name": entry["kernel_name"],
            }
            for entry in sorted(entries, key=lambda e: str(e["op_enum"]))
        ]
        vec4_groups.append(
            {
                "dtype_enum": dtype_enum,
                "entries": group_entries,
            }
        )

    dtype_enum_to_short = {
        DTYPE_MAP[name]: short
        for name, short in DTYPE_SHORT_MAP.items()
        if name in DTYPE_MAP and short is not None
    }

    fused_bias_entries: List[Dict[str, object]] = []
    for dtype_enum in vec4_dtypes:
        dtype_short = dtype_enum_to_short.get(dtype_enum)
        if dtype_short is None:
            continue
        fused_bias_entries.append(
            {
                "dtype_enum": dtype_enum,
                "kernel_name": f"fused_bias_activation_{dtype_short}",
            }
        )

    needs_params_ops = {"elu", "selu", "leaky_relu", "prelu"}
    needs_params_entries = [
        {"op_enum": UNARY_OP_ENUM[op]}
        for op in sorted(needs_params_ops)
        if op in UNARY_OP_ENUM
    ]

    return template.render(
        unary_groups=unary_groups,
        vec4_groups=vec4_groups,
        fused_bias_entries=fused_bias_entries,
        needs_params_entries=needs_params_entries,
    )


def render_split_dispatch_sources(
    env: jinja2.Environment,
    backend: str,
    descriptors: List[KernelDescriptor],
    kernel_ids: Dict[str, int],
    output_dir: Path,
) -> Dict[str, Path]:
    """Render category-specific dispatch files and return paths to generated files."""
    # Build all kernel entries
    all_kernels = [_build_kernel_entry(backend, desc) for desc in descriptors]

    # Categorize kernels
    categories = {
        "matmul": [],
        "elementwise": [],
        "reduction": [],
        "neural": [],
        "misc": [],
    }
    for kernel in all_kernels:
        cat = kernel["category"]
        if cat in categories:
            categories[cat].append(kernel)

    # Calculate ID ranges for each category
    def calc_range(kernels):
        if not kernels:
            return None
        ids = [kernel_ids[k["name"]] for k in kernels]
        return {"first": min(ids), "last": max(ids)}

    ranges = {cat: calc_range(kernels) for cat, kernels in categories.items()}

    generated_files = {}
    ext = ".mm" if backend == "metal" else ".c"

    # Render category-specific dispatch files
    category_templates = {
        "matmul": f"matmul_dispatch_{backend}.c.j2",
        "elementwise": f"elementwise_dispatch_{backend}.c.j2",
        "reduction": f"reduction_dispatch_{backend}.c.j2",
        "neural": f"neural_dispatch_{backend}.c.j2",
        "misc": f"misc_dispatch_{backend}.c.j2",
    }

    for cat, template_name in category_templates.items():
        if not categories[cat]:
            continue
        try:
            template = env.get_template(template_name)
        except jinja2.TemplateNotFound:
            # Skip categories without templates
            continue

        # Render with category-specific kernels
        var_name = f"{cat}_kernels"
        content = template.render(**{var_name: categories[cat], "backend": backend})

        output_path = output_dir / f"{cat}_dispatch_{backend}.gen{ext}"
        write_output(output_path, content)
        generated_files[cat] = output_path

    # Render main dispatch router
    try:
        main_template = env.get_template(f"kernel_dispatch_{backend}.c.j2")
        main_content = main_template.render(
            backend=backend,
            matmul_kernels=categories["matmul"],
            elementwise_kernels=categories["elementwise"],
            reduction_kernels=categories["reduction"],
            neural_kernels=categories["neural"],
            misc_kernels=categories["misc"],
        )
        main_path = output_dir / f"kernel_dispatch_{backend}.gen{ext}"
        write_output(main_path, main_content)
        generated_files["main"] = main_path
    except jinja2.TemplateNotFound:
        pass

    return generated_files


def render_op_signature_hash_header(env: jinja2.Environment) -> str:
    """Render the shared op signature hash header."""
    template = env.get_template("op_signature_hash.h.j2")
    return template.render()


def render_metal_quant_dispatch(
    env: jinja2.Environment, schemes: List[QuantSchemeDescriptor]
) -> str:
    """Render Metal quantized matmul dispatch from QUANT_SCHEME definitions."""
    template = env.get_template("metal_quant_dispatch.gen.mm.j2")
    return template.render(schemes=schemes)


def render_unified_dispatch(
    env: jinja2.Environment,
    backend: str,
    descriptors: List[KernelDescriptor],
    kernel_ids: Dict[str, int],
) -> str:
    """Render monolithic dispatch source using the unified template."""
    cfg = get_backend_config(backend)
    template = env.get_template("backend_dispatch.j2")

    kernels = []
    has_rope_kernels = False

    for desc in descriptors:
        entry = _build_kernel_entry(backend, desc)

        # Check for fused operations (Metal-specific)
        fusion = str(desc.core_fields.get("FUSION", "")).upper()
        if fusion:
            entry["is_fused"] = True
            entry["fusion"] = fusion
        elif entry["kind"] == "binary":
            fused_tag = FUSED_BINARY_OPS.get(entry["op"])
            if fused_tag:
                entry["is_fused"] = True
                entry["fusion"] = fused_tag

        # Track if we have rope kernels
        if entry["kind"] == "rope":
            has_rope_kernels = True

        kernels.append(entry)

    return template.render(
        cfg=cfg,
        kernels=kernels,
        has_rope_kernels=has_rope_kernels,
    )


def render_bytecode_tables(
    env: jinja2.Environment,
    backend: str,
    descriptors: List[KernelDescriptor],
) -> Tuple[str, str]:
    template_h = env.get_template("bytecode_tables.h.j2")
    template_c = env.get_template("bytecode_tables.c.j2")

    op_count = len(descriptors) + 1

    header = template_h.render(backend=backend, op_count=op_count)
    source = template_c.render(backend=backend)
    return header, source


def render_bytecode_exec(
    env: jinja2.Environment,
    backend: str,
    descriptors: List[KernelDescriptor],
) -> Tuple[str, str]:
    template_h = env.get_template("bytecode_exec.h.j2")
    template_c = env.get_template("bytecode_exec.c.j2")

    def exec_guard(entry: Dict[str, Any]) -> str:
        guard = entry.get("platform_guard") or "1"
        if guard == "1":
            guard = "1"
        if entry.get("requires_fp8"):
            if guard == "1":
                guard = "MARMOT_ENABLE_FP8"
            else:
                guard = f"({guard}) && MARMOT_ENABLE_FP8"
        return guard

    kernels = []
    for desc in sorted(descriptors, key=lambda d: d.name):
        entry = _build_kernel_entry(backend, desc)
        entry["exec_guard"] = exec_guard(entry)
        kernels.append(entry)

    op_count = len(kernels) + 1
    header = template_h.render(backend=backend, op_count=op_count)
    source = template_c.render(backend=backend, op_count=op_count, kernels=kernels)
    return header, source


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate kernel traits/dispatch from .def descriptors")
    parser.add_argument("--backends", required=True, nargs='+', help="Backend names (cpu, metal, cuda)")
    parser.add_argument("--backend-dirs", required=True, nargs='+', type=Path, help="Directories containing backend .def files (one per backend)")
    parser.add_argument("--traits-header", required=True, type=Path, help="Output path for traits_ids.gen.h")
    parser.add_argument("--query-sources", required=True, nargs='+', type=Path, help="Output paths for backend query sources (one per backend)")
    parser.add_argument(
        "--dispatch-sources",
        nargs='+',
        type=Path,
        help="Output paths for monolithic backend dispatch sources (one per unsplit backend, or one per backend for legacy usage)",
    )
    parser.add_argument("--dispatch-dirs", nargs='+', type=Path, help="Output directories for split dispatch files. Can specify for individual backends using backend:path format (e.g., cpu:/path/to/dir).")
    parser.add_argument(
        "--op-signature-hash-header",
        type=Path,
        help="Output path for op_signature_hash.gen.h (replaces legacy dispatch_cache.gen.h)",
    )
    parser.add_argument("--bytecode-table-headers", nargs='+', type=Path, help="Output paths for bytecode table headers (one per backend)")
    parser.add_argument("--bytecode-table-sources", nargs='+', type=Path, help="Output paths for bytecode table sources (one per backend)")
    parser.add_argument("--bytecode-exec-headers", nargs='+', type=Path, help="Output paths for bytecode exec headers (one per backend)")
    parser.add_argument("--bytecode-exec-sources", nargs='+', type=Path, help="Output paths for bytecode exec sources (one per backend)")
    parser.add_argument("--quant-scheme-def", type=Path, help="Path to quant_schemes.def for Metal quant dispatch")
    parser.add_argument("--metal-quant-dispatch-output", type=Path, help="Output path for Metal quant dispatch")
    parser.add_argument("--metal-unary-tables-header", type=Path, help="Output path for Metal unary tables header")
    parser.add_argument("--template-dir", required=True, type=Path, help="Directory containing Jinja2 templates")
    parser.add_argument("--stamp-output", type=Path, help="Optional stamp file to write after generation")
    args = parser.parse_args()

    # Validate that all lists have the same length
    num_backends = len(args.backends)
    if len(args.backend_dirs) != num_backends:
        raise ValueError(f"Number of backend-dirs ({len(args.backend_dirs)}) must match number of backends ({num_backends})")
    if len(args.query_sources) != num_backends:
        raise ValueError(f"Number of query-sources ({len(args.query_sources)}) must match number of backends ({num_backends})")

    if args.bytecode_table_headers or args.bytecode_table_sources:
        if args.bytecode_table_headers is None or args.bytecode_table_sources is None:
            raise ValueError("Both bytecode-table-headers and bytecode-table-sources must be provided")
        if len(args.bytecode_table_headers) != num_backends:
            raise ValueError(
                f"Number of bytecode-table-headers ({len(args.bytecode_table_headers)}) must match number of backends ({num_backends})"
            )
        if len(args.bytecode_table_sources) != num_backends:
            raise ValueError(
                f"Number of bytecode-table-sources ({len(args.bytecode_table_sources)}) must match number of backends ({num_backends})"
            )

    if args.bytecode_exec_headers or args.bytecode_exec_sources:
        if args.bytecode_exec_headers is None or args.bytecode_exec_sources is None:
            raise ValueError("Both bytecode-exec-headers and bytecode-exec-sources must be provided")
        if len(args.bytecode_exec_headers) != num_backends:
            raise ValueError(
                f"Number of bytecode-exec-headers ({len(args.bytecode_exec_headers)}) must match number of backends ({num_backends})"
            )
        if len(args.bytecode_exec_sources) != num_backends:
            raise ValueError(
                f"Number of bytecode-exec-sources ({len(args.bytecode_exec_sources)}) must match number of backends ({num_backends})"
            )

    emit_dispatch = args.dispatch_dirs is not None or args.dispatch_sources is not None

    # Parse dispatch-dirs which can be in format "backend:path" or just "path" (for all backends)
    split_dispatch_dirs: dict[str, Path] = {}
    if emit_dispatch and args.dispatch_dirs:
        for entry in args.dispatch_dirs:
            entry_str = str(entry)
            if ':' in entry_str and not entry_str.startswith('/'):
                # Format: backend:path
                backend_name, path_str = entry_str.split(':', 1)
                split_dispatch_dirs[backend_name] = Path(path_str)
            else:
                # Format: path (applies to all backends in order)
                # Find first backend without a dispatch dir
                for backend in args.backends:
                    if backend not in split_dispatch_dirs:
                        split_dispatch_dirs[backend] = entry
                        break

    dispatch_sources: dict[str, Path] = {}
    if emit_dispatch:
        unsplit_backends = [backend for backend in args.backends if backend not in split_dispatch_dirs]
        if unsplit_backends:
            if args.dispatch_sources is None:
                raise ValueError(f"dispatch-sources is required for unsplit backends: {', '.join(unsplit_backends)}")
            if len(args.dispatch_sources) == num_backends:
                dispatch_sources = dict(zip(args.backends, args.dispatch_sources))
            elif len(args.dispatch_sources) == len(unsplit_backends):
                dispatch_sources = dict(zip(unsplit_backends, args.dispatch_sources))
            else:
                raise ValueError(
                    f"Number of dispatch-sources ({len(args.dispatch_sources)}) must match number of backends ({num_backends}) "
                    f"or unsplit backends ({len(unsplit_backends)})"
                )

    env = make_jinja_env(args.template_dir)

    # Collect data from all backends
    all_ops = []
    all_kernels = []
    all_qschemes = ["NONE"]
    all_profiles = []

    backend_data = []  # List of (backend, descriptors, kernel_ids) tuples

    extra_ops = [
        "linear",
        "add_relu",
        "add_gelu",
        "add_silu",
        "matmul_bias",
        "matmul_bias_relu",
        "matmul_bias_gelu",
        "matmul_bias_silu",
        "qkv_rope",
        "qkv_shared_input",
        "qkv_projection",
    ]
    for backend, backend_dir in zip(args.backends, args.backend_dirs):
        descriptors = collect_descriptors([backend_dir])
        for desc in descriptors:
            validate_descriptor(desc)

        descriptors = expand_profiles(descriptors)
        for desc in descriptors:
            validate_descriptor(desc)

        kernel_ids = allocate_kernel_ids(backend, descriptors)
        backend_data.append((backend, descriptors, kernel_ids))

        # Accumulate operations
        for desc in descriptors:
            op = desc.core_fields["OP"]
            if op not in all_ops:
                all_ops.append(op)

        # Accumulate kernels
        for desc in descriptors:
            all_kernels.append((backend, desc.name, kernel_ids[desc.name]))

        # Accumulate qschemes
        for desc in descriptors:
            quant = desc.core_fields.get("WEIGHT_QUANT")
            for entry in _normalize_list(quant):
                upper = entry.upper()
                if upper not in all_qschemes:
                    all_qschemes.append(upper)

        # Accumulate profiles
        for desc in descriptors:
            profile = desc.core_fields.get("PROFILE", "NONE")
            if profile and profile != "NONE":
                for entry in _normalize_list(profile):
                    upper = entry.upper()
                    if upper not in all_profiles and upper in PROFILE_MAP:
                        all_profiles.append(upper)

    for op in extra_ops:
        if op not in all_ops:
            all_ops.append(op)

    # Generate shared traits header with ALL backends
    header_text = render_traits_header(env, all_ops, all_kernels, all_qschemes, all_profiles)
    write_output(args.traits_header, header_text)

    bytecode_headers = {}
    bytecode_sources = {}
    if args.bytecode_table_headers and args.bytecode_table_sources:
        bytecode_headers = dict(zip(args.backends, args.bytecode_table_headers))
        bytecode_sources = dict(zip(args.backends, args.bytecode_table_sources))

    bytecode_exec_headers = {}
    bytecode_exec_sources = {}
    if args.bytecode_exec_headers and args.bytecode_exec_sources:
        bytecode_exec_headers = dict(zip(args.backends, args.bytecode_exec_headers))
        bytecode_exec_sources = dict(zip(args.backends, args.bytecode_exec_sources))

    # Generate per-backend query/dispatch files
    for i, (backend, descriptors, kernel_ids) in enumerate(backend_data):
        query_source = args.query_sources[i]
        dispatch_source = dispatch_sources.get(backend) if emit_dispatch else None

        query_text = render_query_source(env, backend, descriptors, kernel_ids)
        write_output(query_source, query_text)

        if emit_dispatch:
            if backend in split_dispatch_dirs:
                # Generate category-specific dispatch files for this backend
                dispatch_dir = split_dispatch_dirs[backend]
                render_split_dispatch_sources(env, backend, descriptors, kernel_ids, dispatch_dir)
            else:
                # Generate monolithic dispatch using unified template
                if dispatch_source is None:
                    raise ValueError(f"Missing dispatch source for backend '{backend}'")
                dispatch_text = render_unified_dispatch(env, backend, descriptors, kernel_ids)
                write_output(dispatch_source, dispatch_text)

        if backend == "metal":
            unary_tables_text = render_metal_unary_tables(env, descriptors)
            if args.metal_unary_tables_header is not None:
                write_output(args.metal_unary_tables_header, unary_tables_text)
            elif dispatch_source is not None:
                write_output(dispatch_source.parent / "metal_unary_tables.gen.h", unary_tables_text)

        if backend in bytecode_headers:
            header_text, source_text = render_bytecode_tables(env, backend, descriptors)
            write_output(bytecode_headers[backend], header_text)
            write_output(bytecode_sources[backend], source_text)

        if backend in bytecode_exec_headers:
            header_text, source_text = render_bytecode_exec(env, backend, descriptors)
            write_output(bytecode_exec_headers[backend], header_text)
            write_output(bytecode_exec_sources[backend], source_text)

    # Generate op signature hash header (shared by all backends)
    if args.op_signature_hash_header:
        header_text = render_op_signature_hash_header(env)
        write_output(args.op_signature_hash_header, header_text)

    # Generate Metal quantized matmul dispatch from QUANT_SCHEME definitions
    if args.quant_scheme_def and args.metal_quant_dispatch_output:
        schemes = parse_quant_schemes(args.quant_scheme_def)
        quant_dispatch_text = render_metal_quant_dispatch(env, schemes)
        write_output(args.metal_quant_dispatch_output, quant_dispatch_text)

    if args.stamp_output is not None:
        write_stamp(args.stamp_output)


if __name__ == "__main__":
    main()
