#!/usr/bin/env python3
"""
Op schema definitions for kernel dispatch args generation.

This module defines the args structure for each op category.
Codegen uses this to generate kernel_dispatch_args.h and graph executor dispatch.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set

try:
    from ops_schema_gen import (
        BINARY_OPS,
        COMPUTE_QPARAMS_OPS,
        CONVERT_OPS,
        DEQUANTIZE_OPS,
        QUANTIZE_OPS,
        REDUCTION_OPS,
        TERNARY_OPS,
        UNARY_OPS,
    )
except ModuleNotFoundError as exc:
    raise RuntimeError("Missing ops_schema_gen.py; run codegen") from exc


@dataclass
class ArgField:
    name: str
    c_type: str
    is_output: bool = False  # True for output tensors/params
    enum_name: str | None = None  # Override for enum name if different from struct field

    @property
    def get_enum_name(self) -> str:
        return self.enum_name if self.enum_name else self.name


@dataclass
class OpSchema:
    name: str
    args: List[ArgField]
    ops: Set[str]  # Op names that use this schema
    enum_prefix_override: str | None = None  # Override enum prefix if different from name

    @property
    def struct_name(self) -> str:
        return f"marmot_kernel_args_{self.name}_t"

    @property
    def enum_prefix(self) -> str:
        prefix = self.enum_prefix_override if self.enum_prefix_override else self.name
        return f"MARMOT_KERNEL_ARGS_{prefix.upper()}"


# =============================================================================
# Schema Definitions
# =============================================================================

SCHEMAS: List[OpSchema] = [
    # Binary ops: add, sub, mul, div, etc.
    OpSchema(
        name="binary",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input_a", "const marmot_tensor_t *"),
            ArgField("input_b", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
        ],
        ops=BINARY_OPS,
    ),

    # Unary ops: relu, gelu, abs, neg, etc.
    OpSchema(
        name="unary",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("params", "const marmot_activation_params_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
        ],
        ops=UNARY_OPS,
    ),

    # Ternary ops: fma, where
    OpSchema(
        name="ternary",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input_a", "const marmot_tensor_t *"),
            ArgField("input_b", "const marmot_tensor_t *"),
            ArgField("input_c", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
        ],
        ops=TERNARY_OPS,
    ),

    # Reduction ops
    OpSchema(
        name="reduction",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("out_values", "marmot_tensor_t *", is_output=True),
            ArgField("out_indices", "marmot_tensor_t *", is_output=True),
            ArgField("params", "const marmot_reduction_params_t *"),
        ],
        ops=REDUCTION_OPS,
    ),

    # Softmax
    OpSchema(
        name="softmax",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("axis", "int32_t"),
        ],
        ops={"softmax"},
    ),

    # Layer normalization
    OpSchema(
        name="layernorm",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("weight", "const marmot_tensor_t *"),
            ArgField("bias", "const marmot_tensor_t *"),
            ArgField("residual", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("eps", "float"),
        ],
        ops={"layernorm"},
    ),

    # RMS normalization
    OpSchema(
        name="rms_norm",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("weight", "const marmot_tensor_t *"),
            ArgField("residual", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("eps", "float"),
        ],
        ops={"rms_norm", "rms_norm_gemma"},
    ),

    # Paged attention
    OpSchema(
        name="paged_attention",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("token_meta", "const marmot_tensor_t *"),
            ArgField("q", "const marmot_tensor_t *"),
            ArgField("k_new", "const marmot_tensor_t *"),
            ArgField("v_new", "const marmot_tensor_t *"),
            ArgField("kv_k", "marmot_tensor_t *"),
            ArgField("kv_v", "marmot_tensor_t *"),
            ArgField("block_table", "const marmot_tensor_t *"),
            ArgField("kv_k_scale", "marmot_tensor_t *"),
            ArgField("kv_v_scale", "marmot_tensor_t *"),
            ArgField("out", "marmot_tensor_t *", is_output=True),
            ArgField("token_count", "uint32_t"),
            ArgField("layer_idx", "uint32_t"),
            ArgField("num_q_heads", "uint32_t"),
            ArgField("num_kv_heads", "uint32_t"),
            ArgField("head_dim", "uint32_t"),
            ArgField("block_size", "uint32_t"),
            ArgField("scale", "float"),
        ],
        ops={"paged_attention"},
    ),

    # RoPE
    OpSchema(
        name="rope",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("rope_params", "const marmot_rope_params_t *"),
            ArgField("n_past", "uint32_t"),
            ArgField("n_rot", "uint32_t"),
        ],
        ops={"rope"},
    ),

    # Matmul
    OpSchema(
        name="matmul",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("weight", "const marmot_tensor_t *"),
            ArgField("epilogue", "const marmot_matmul_epilogue_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
        ],
        ops={"matmul", "matmul_bias", "matmul_bias_relu", "matmul_bias_gelu", "matmul_bias_silu"},
    ),

    # QKV projection
    OpSchema(
        name="qkv",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("wq", "const marmot_tensor_t *"),
            ArgField("wk", "const marmot_tensor_t *"),
            ArgField("wv", "const marmot_tensor_t *"),
            ArgField("bq", "const marmot_tensor_t *"),
            ArgField("bk", "const marmot_tensor_t *"),
            ArgField("bv", "const marmot_tensor_t *"),
            ArgField("epilogue", "const marmot_matmul_epilogue_t *"),
            ArgField("rope_params", "const marmot_rope_params_t *"),
            ArgField("out_q", "marmot_tensor_t *", is_output=True),
            ArgField("out_k", "marmot_tensor_t *", is_output=True),
            ArgField("out_v", "marmot_tensor_t *", is_output=True),
        ],
        ops={"qkv_rope", "qkv_shared_input", "qkv_projection"},
    ),

    # Reshape
    OpSchema(
        name="reshape",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("new_shape", "const size_t *"),
            ArgField("new_ndim", "size_t"),
        ],
        ops={"reshape"},
    ),

    # View (zero-copy view with byte offset - handled directly in executor)
    OpSchema(
        name="view",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("byte_offset", "size_t"),
        ],
        ops={"view"},
    ),

    # Contiguous (explicit layout copy)
    OpSchema(
        name="contiguous",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
        ],
        ops={"contiguous"},
    ),

    # Transpose
    OpSchema(
        name="transpose",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("perm", "const int *"),
        ],
        ops={"transpose"},
    ),

    # Concat
    OpSchema(
        name="concat",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("inputs", "const marmot_tensor_t *const *"),
            ArgField("num_inputs", "size_t"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("axis", "int"),
        ],
        ops={"concat"},
    ),

    # Slice
    OpSchema(
        name="slice",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("starts", "const size_t *"),
            ArgField("sizes", "const size_t *"),
        ],
        ops={"slice"},
    ),

    # Gather rows
    OpSchema(
        name="gather_rows",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("indices", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
        ],
        ops={"gather_rows"},
    ),

    # Quantize
    OpSchema(
        name="quantize",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("quant_params", "const marmot_quant_params_t *", enum_name="params"),  # struct: quant_params, enum: PARAMS
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("kind", "marmot_quant_kind_t"),
            ArgField("layout", "marmot_quant_layout_t"),
        ],
        ops=QUANTIZE_OPS,
    ),

    # Dequantize
    OpSchema(
        name="dequantize",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("input", "const marmot_tensor_t *"),
            ArgField("output", "marmot_tensor_t *", is_output=True),
            ArgField("kind", "marmot_quant_kind_t"),
            ArgField("layout", "marmot_quant_layout_t"),
        ],
        ops=DEQUANTIZE_OPS,
    ),

    # Compute quant params
    OpSchema(
        name="compute_qparams",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("tensor", "const marmot_tensor_t *"),
            ArgField("target_dtype", "marmot_dtype_t"),
            ArgField("block_size", "size_t"),
            ArgField("out_params", "marmot_quant_params_t *", is_output=True, enum_name="output"),  # struct: out_params, enum: OUTPUT
        ],
        ops=COMPUTE_QPARAMS_OPS,
    ),

    # Embedding - struct is "embedding_t" but enum prefix is "EMBED"
    OpSchema(
        name="embedding",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("weights", "const marmot_tensor_t *"),
            ArgField("token_ids", "const marmot_tensor_t *"),
            ArgField("out", "marmot_tensor_t *", is_output=True),
            ArgField("dtype_out", "marmot_dtype_t"),
            ArgField("scale", "float"),
            ArgField("padding_id", "int32_t"),
            ArgField("bounds_check", "bool"),
            ArgField("prefer_gpu_private", "bool", enum_name="prefer_gpu"),  # struct uses full name
            ArgField("allow_quant_decode_on_the_fly", "bool", enum_name="allow_decode"),  # struct uses full name
        ],
        ops={"embedding"},
        enum_prefix_override="embed",  # uses MARMOT_KERNEL_ARGS_EMBED_* for backward compat
    ),

    # Convert
    OpSchema(
        name="convert",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("dst", "void *", is_output=True),
            ArgField("src", "const void *"),
            ArgField("n", "size_t"),
            ArgField("dst_dtype", "marmot_dtype_t"),
            ArgField("src_dtype", "marmot_dtype_t"),
        ],
        ops=CONVERT_OPS,
    ),

    # Vec dot
    OpSchema(
        name="vec_dot",
        args=[
            ArgField("ctx", "const marmot_context_t *"),
            ArgField("desc", "const marmot_vec_dot_descriptor_t *"),
            ArgField("result", "float *", is_output=True),
        ],
        ops={"vec_dot"},
    ),
]


def get_schema_for_op(op_name: str) -> OpSchema | None:
    """Find the schema that handles a given op."""
    for schema in SCHEMAS:
        if op_name in schema.ops:
            return schema
    return None


def get_all_ops() -> Set[str]:
    """Get all ops covered by schemas."""
    all_ops: Set[str] = set()
    for schema in SCHEMAS:
        all_ops |= schema.ops
    return all_ops


# Build lookup table
OP_TO_SCHEMA: Dict[str, OpSchema] = {}
for schema in SCHEMAS:
    for op in schema.ops:
        OP_TO_SCHEMA[op] = schema
