#!/usr/bin/env python3
"""
Backend configuration for unified code generation.

This module defines the per-backend traits and function mappings that allow
a single template to generate dispatch code for any backend.

When adding a new backend (e.g., CUDA):
1. Add a new BackendConfig instance to BACKENDS dict
2. Implement the required kernel functions in src/backends/<backend>/
3. Run codegen - the unified template handles the rest
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BackendConfig:
    """Configuration for a single backend's code generation."""

    # Basic identity
    name: str  # "cpu", "metal", "cuda"
    file_extension: str  # ".c", ".mm", ".cu"

    # Include configuration
    internal_header: str  # "cpu_backend_internal.h"
    extra_includes: List[str] = field(default_factory=list)
    needs_extern_c: bool = False  # Objective-C++ / C++ files need this

    # Function name mappings (operation -> backend function)
    # These are the "default" implementations when kernel.kernel_func is not set
    handlers: Dict[str, str] = field(default_factory=dict)

    # Backend capabilities/traits
    supports_fusion: bool = False  # Can fuse binary+activation
    fusion_functions: Dict[str, str] = field(default_factory=dict)  # fusion_type -> function

    # Matmul specifics
    matmul_needs_dimension_extraction: bool = False  # CPU needs N,K,M extraction
    matmul_needs_epilogue_apply: bool = False  # CPU calls cpu_matmul_apply_epilogue

    # Quantization dispatch strategy
    quantize_uses_unified_dispatcher: bool = False  # Metal: single dispatcher function
    # CPU: inline branching for generic vs kind-specific

    # Reduction dispatch strategy
    reduction_uses_generic_handler: bool = False  # CPU: cpu_reduction(op_enum)
    # Metal: per-kernel functions

    # Tensor ops
    split_and_slice_unified: bool = False  # Metal treats them as one op

    # Architecture variant selection
    allows_kernel_func_override: bool = False  # Metal allows per-kernel function overrides

    # Platform guard style
    uses_platform_guards: bool = False  # CPU uses #if guards for SIMD
    platform_guard_wrapper: Optional[str] = None  # e.g., "#ifdef __APPLE__" for Metal

    # Context type for backend exec context
    context_type: str = "void"  # "cpu_context_t", "metal_context_t"


# =============================================================================
# Backend Definitions
# =============================================================================

CPU_CONFIG = BackendConfig(
    name="cpu",
    file_extension=".c",
    internal_header="cpu_backend_internal.h",
    extra_includes=[
        "kernel_dispatch.h",
        "graph/kernel_dispatch_args.gen.h",
        "ops/elementwise/elementwise_dispatch.h",
        "ops/matmul/matmul_kernels.h",
    ],
    needs_extern_c=False,
    handlers={
        "binary": "cpu_elementwise_binary_impl",
        "unary": "cpu_unary_apply",
        "ternary": "cpu_elementwise_ternary_impl",
        "matmul": "cpu_matmul",
        "matmul_quantized": "cpu_matmul_quantized",
        "qkv": "cpu_matmul_qkv",  # suffixed with dtype/quant
        "softmax": "cpu_softmax_impl",
        "paged_attention": "cpu_paged_attention_impl",
        "rms_norm": "cpu_rmsnorm",
        "layernorm": "cpu_layernorm",
        "rope": "cpu_rope",
        "reduction": "cpu_reduction",
        "vec_dot": "cpu_vec_dot",
        "embedding": "cpu_embedding_gather",
        "reshape": "cpu_reshape",
        "contiguous": "cpu_contiguous",
        "view": "cpu_view",
        "transpose": "cpu_transpose",
        "concat": "cpu_concat",
        "slice": "cpu_slice",
        "gather_rows": "cpu_gather_rows",
        "scatter_u64_to_i32": "cpu_scatter_u64_to_i32",
        "convert": "cpu_convert_dispatch",
        "quantize": "cpu_quantize",
        "quantize_with_kind": "cpu_quantize_with_kind",
        "dequantize": "cpu_dequantize",
        "dequantize_with_kind": "cpu_dequantize_with_kind",
        "compute_quant_params": "cpu_compute_quant_params",
    },
    supports_fusion=False,
    matmul_needs_dimension_extraction=True,
    matmul_needs_epilogue_apply=True,
    quantize_uses_unified_dispatcher=False,
    reduction_uses_generic_handler=True,
    split_and_slice_unified=False,
    allows_kernel_func_override=False,
    uses_platform_guards=True,
    context_type="cpu_context_t",
)

METAL_CONFIG = BackendConfig(
    name="metal",
    file_extension=".mm",
    internal_header="../metal_backend_internal.h",
    extra_includes=[
        "marmot/ops/matmul.h",
        "marmot/ops/reduction.h",
        "marmot/ops_types.h",
        "marmot/tensor.h",
        "marmot/traits_ids.gen.h",
        "graph/kernel_dispatch_args.gen.h",
        "../metal_fusion.h",
        "../ops/matmul_kernels.h",
    ],
    needs_extern_c=True,
    handlers={
        "binary": "metal_elementwise_binary_impl",
        "unary": "metal_elementwise_unary_impl",
        "ternary": "metal_elementwise_ternary_impl",
        "matmul": "metal_matmul",
        "matmul_quantized": "metal_matmul_quantized",
        "qkv": "metal_matmul_qkv",  # uses kernel.kernel_func directly
        "softmax": "metal_softmax_impl",
        "paged_attention": "metal_paged_attention_impl",
        "rms_norm": "metal_rmsnorm",
        "layernorm": "metal_layernorm",
        "rope": "metal_rope",
        "reduction": None,  # per-kernel functions only
        "vec_dot": "metal_vec_dot",
        "embedding": "metal_embedding_gather",
        "reshape": "metal_reshape",
        "contiguous": "metal_contiguous",
        "view": "metal_view",
        "transpose": "metal_transpose",
        "concat": "metal_concat",
        "slice": "metal_slice",
        "gather_rows": "metal_gather_rows",
        "scatter_u64_to_i32": "metal_scatter_u64_to_i32",
        "convert": "metal_convert_dispatch",
        "quantize": "metal_quantize_dispatch",
        "dequantize": "metal_dequantize_dispatch",
        "compute_quant_params": "metal_compute_quant_params",
    },
    supports_fusion=True,
    fusion_functions={
        "ADD_RELU": "metal_add_relu_fused",
        "ADD_GELU": "metal_add_gelu_fused",
        "ADD_SILU": "metal_add_silu_fused",
    },
    matmul_needs_dimension_extraction=False,
    matmul_needs_epilogue_apply=False,
    quantize_uses_unified_dispatcher=True,
    reduction_uses_generic_handler=False,
    split_and_slice_unified=True,
    allows_kernel_func_override=True,
    uses_platform_guards=False,
    platform_guard_wrapper="#ifdef __APPLE__",
    context_type="metal_context_t",
)

# Registry of all backends
BACKENDS: Dict[str, BackendConfig] = {
    "cpu": CPU_CONFIG,
    "metal": METAL_CONFIG,
}


def get_backend_config(name: str) -> BackendConfig:
    """Get configuration for a backend by name."""
    if name not in BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS.keys())}")
    return BACKENDS[name]
