#!/usr/bin/env python3
"""
Generate API wrapper includes and op metadata from ops.def.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Any, Dict, Iterable, List

import def_parser
from codegen_base import make_jinja_env, write_output, write_stamp


@dataclasses.dataclass(frozen=True)
class Operation:
    name: str
    category: str
    api_function: str
    api_signature: str
    desc_type: str | None
    build_fn: str | None
    impl_function: str | None
    alias_of: str | None
    op_enum: str | None
    op_id: str | None
    label: str | None
    has_indices: bool
    supports_bias: bool
    kernel_op: str | None
    kernel_enum_prefix: str | None
    allow_bool_out: bool
    stride_policy: str | None
    lookup_dtype: str | None
    params: Dict[str, str]
    has_default_alpha: bool
    has_default_beta: bool
    default_alpha: str
    default_beta: str
    default_alpha_if_zero: bool
    default_beta_if_zero: bool
    params_required: bool
    allow_param_tensor: bool
    src_dtype: str | None
    dst_dtype: str | None
    quant_kind: str | None
    path: Path
    offset: int


def _raise_missing(op_def: def_parser.BlockDef, key: str) -> None:
    offset = op_def.field_offsets.get(key, op_def.offset)
    def_parser._raise_with_location(
        f"Operation '{op_def.name}' missing required field '{key}'",
        path=op_def.path,
        content=op_def.source,
        offset=offset,
    )


def _require(fields: Dict[str, Any], op_def: def_parser.BlockDef, key: str) -> Any:
    if key not in fields:
        _raise_missing(op_def, key)
    return fields[key]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise ValueError(f"Expected boolean, got {value!r}")


def _escape_c_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', "\\\"")


def _format_c_float(value: Any) -> str:
    if value is None:
        return "0.0f"
    if isinstance(value, bool):
        raise ValueError("DEFAULT_* values must be numeric")
    if isinstance(value, (int, float)):
        text = repr(float(value))
        if "e" not in text and "." not in text:
            text += ".0"
        return f"{text}f"
    return str(value)


def _eval_ops(paths: Iterable[Path]) -> List[Operation]:
    mod = def_parser.parse_def_files(paths)
    sets: Dict[str, List[Any]] = {}
    for name, s in mod.sets.items():
        sets[name] = def_parser._expand_set(mod.sets, name, stack=[], path=s.path, source=s.source)
    fragments = def_parser._resolve_fragments(mod, sets)

    ops: List[Operation] = []
    for name, op_def in mod.operations.items():
        fields_expr = def_parser._resolve_block_fields(mod.operations, name, kind="operation", stack=[])
        ctx = def_parser._ResolvedContext(
            path=op_def.path,
            source=op_def.source,
            sets=sets,
            fragments=fragments,
            quant_schemes={},
        )
        fields: Dict[str, Any] = {}
        for key, expr in fields_expr.items():
            fields[key] = def_parser._eval_expr(expr, {}, ctx)

        category = str(_require(fields, op_def, "CATEGORY")).lower()
        api_function = str(_require(fields, op_def, "API_FUNCTION"))
        api_signature = str(_require(fields, op_def, "API_SIGNATURE")).lower()
        desc_type = fields.get("DESC_TYPE")
        build_fn = fields.get("BUILD_FN")
        impl_function = fields.get("IMPL_FUNCTION")
        alias_of = fields.get("ALIAS_OF")
        op_enum = fields.get("OP_ENUM")
        op_id = fields.get("OP_ID")
        label = fields.get("LABEL")
        kernel_op = fields.get("KERNEL_OP")
        kernel_enum_prefix = fields.get("KERNEL_ENUM_PREFIX")
        allow_bool_out = _as_bool(fields.get("ALLOW_BOOL_OUT", False))
        stride_policy = fields.get("STRIDE_POLICY")
        lookup_dtype = fields.get("LOOKUP_DTYPE")
        params = fields.get("PARAMS")
        if params is None:
            params_map: Dict[str, str] = {}
        elif isinstance(params, dict):
            params_map = {str(k): str(v) for k, v in params.items()}
        else:
            raise ValueError(f"PARAMS must be a struct for {name}")

        default_alpha = fields.get("DEFAULT_ALPHA")
        default_beta = fields.get("DEFAULT_BETA")
        has_default_alpha = default_alpha is not None
        has_default_beta = default_beta is not None
        default_alpha_literal = _format_c_float(default_alpha)
        default_beta_literal = _format_c_float(default_beta)
        default_alpha_if_zero = _as_bool(fields.get("DEFAULT_ALPHA_IF_ZERO", False))
        default_beta_if_zero = _as_bool(fields.get("DEFAULT_BETA_IF_ZERO", False))
        params_required = _as_bool(fields.get("PARAMS_REQUIRED", False))
        allow_param_tensor = _as_bool(fields.get("ALLOW_PARAM_TENSOR", True))

        has_indices = _as_bool(fields.get("HAS_INDICES", False))
        supports_bias = _as_bool(fields.get("SUPPORTS_BIAS", False))
        src_dtype = fields.get("SRC_DTYPE")
        dst_dtype = fields.get("DST_DTYPE")
        quant_kind = fields.get("QUANT_KIND")

        ops.append(
            Operation(
                name=name,
                category=category,
                api_function=api_function,
                api_signature=api_signature,
                desc_type=str(desc_type) if desc_type is not None else None,
                build_fn=str(build_fn) if build_fn is not None else None,
                impl_function=str(impl_function) if impl_function is not None else None,
                alias_of=str(alias_of) if alias_of is not None else None,
                op_enum=str(op_enum) if op_enum is not None else None,
                op_id=str(op_id) if op_id is not None else None,
                label=str(label) if label is not None else None,
                has_indices=has_indices,
                supports_bias=supports_bias,
                kernel_op=str(kernel_op) if kernel_op is not None else None,
                kernel_enum_prefix=str(kernel_enum_prefix) if kernel_enum_prefix is not None else None,
                allow_bool_out=allow_bool_out,
                stride_policy=str(stride_policy) if stride_policy is not None else None,
                lookup_dtype=str(lookup_dtype) if lookup_dtype is not None else None,
                params=params_map,
                has_default_alpha=has_default_alpha,
                has_default_beta=has_default_beta,
                default_alpha=default_alpha_literal,
                default_beta=default_beta_literal,
                default_alpha_if_zero=default_alpha_if_zero,
                default_beta_if_zero=default_beta_if_zero,
                params_required=params_required,
                allow_param_tensor=allow_param_tensor,
                src_dtype=str(src_dtype) if src_dtype is not None else None,
                dst_dtype=str(dst_dtype) if dst_dtype is not None else None,
                quant_kind=str(quant_kind) if quant_kind is not None else None,
                path=op_def.path,
                offset=op_def.offset,
            )
        )
    return ops


def _alias_target(ops_by_name: Dict[str, Operation], op: Operation) -> str:
    if op.alias_of is None:
        raise ValueError("alias_of missing")
    if op.alias_of in ops_by_name:
        return ops_by_name[op.alias_of].api_function
    return op.alias_of


def _render_reduction(op: Operation) -> str:
    if op.op_enum is None:
        raise ValueError(f"Reduction {op.name} missing OP_ENUM")
    label = _escape_c_string(op.label or op.name)
    has_indices = "true" if op.has_indices else "false"
    lines = [
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_reduction_desc_t *desc) {{",
        "    if (ctx == nullptr || desc == nullptr || desc->input == nullptr || desc->out == nullptr) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"Reduce {label} requires valid context and descriptor\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
        "    marmot_reduction_params_t params = marmot_desc_to_params(desc);",
        f"    marmot_tensor_t *indices = {has_indices} ? desc->indices_out : nullptr;",
        f"    return marmot_dispatch_reduction(ctx, {op.op_enum}, desc->input, desc->out, indices, &params, \"{label}\");",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _render_unary_basic(op: Operation) -> str:
    if op.op_enum is None or op.op_id is None:
        raise ValueError(f"Unary {op.name} missing OP_ENUM or OP_ID")
    label = _escape_c_string(op.label or op.name)
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out) {{\n"
        f"    return marmot_dispatch_unary_uniform(ctx, {op.op_enum}, {op.op_id}, x, nullptr, out, \"{label}\");\n"
        "}\n"
    )


def _render_unary_alpha(op: Operation) -> str:
    if op.op_enum is None or op.op_id is None:
        raise ValueError(f"Unary {op.name} missing OP_ENUM or OP_ID")
    label = _escape_c_string(op.label or op.name)
    arg_name = op.params.get("alpha", "alpha")
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, float {arg_name}) {{\n"
        "    marmot_activation_params_t params = {\n"
        "        .parameter_tensor = nullptr,\n"
        "        .bias = nullptr,\n"
        f"        .alpha = {arg_name},\n"
        "        .beta = 0.0f,\n"
        "        .gamma = 0.0f,\n"
        "    };\n"
        f"    return marmot_dispatch_unary_uniform(ctx, {op.op_enum}, {op.op_id}, x, &params, out, \"{label}\");\n"
        "}\n"
    )


def _render_unary_alpha_beta(op: Operation) -> str:
    if op.op_enum is None or op.op_id is None:
        raise ValueError(f"Unary {op.name} missing OP_ENUM or OP_ID")
    label = _escape_c_string(op.label or op.name)
    alpha_arg = op.params.get("alpha", "alpha")
    beta_arg = op.params.get("beta", "beta")
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, float {alpha_arg}, float {beta_arg}) {{\n"
        "    marmot_activation_params_t params = {\n"
        "        .parameter_tensor = nullptr,\n"
        "        .bias = nullptr,\n"
        f"        .alpha = {alpha_arg},\n"
        f"        .beta = {beta_arg},\n"
        "        .gamma = 0.0f,\n"
        "    };\n"
        f"    return marmot_dispatch_unary_uniform(ctx, {op.op_enum}, {op.op_id}, x, &params, out, \"{label}\");\n"
        "}\n"
    )


def _render_unary_alias(op: Operation, target: str) -> str:
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out) {{\n"
        f"    return {target}(ctx, x, out);\n"
        "}\n"
    )


def _stride_expr(op: Operation) -> str:
    if op.stride_policy is None:
        return "elementwise_stride_mode_2d(a, b, out)"
    if op.stride_policy == "elementwise_stride_mode_2d":
        return "elementwise_stride_mode_2d(a, b, out)"
    return op.stride_policy


def _render_binary(op: Operation) -> str:
    if op.op_id is None:
        raise ValueError(f"Binary {op.name} missing OP_ID")
    if op.kernel_op is None:
        raise ValueError(f"Binary {op.name} missing KERNEL_OP")
    label = _escape_c_string(op.label or op.name)
    allow_bool = "true" if op.allow_bool_out else "false"
    use_stride_2d = "true" if op.stride_policy is None or op.stride_policy == "elementwise_stride_mode_2d" else "false"
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, marmot_tensor_t *out) {{\n"
        f"    return marmot_dispatch_binary(ctx, {op.op_id}, a, b, out, {allow_bool}, {use_stride_2d}, \"{label}\");\n"
        "}\n"
    )


def _lookup_expr(op: Operation, first_arg: str, out_arg: str) -> str:
    if op.lookup_dtype == "a":
        return f"{first_arg}->dtype"
    if op.lookup_dtype == "out":
        return f"{out_arg}->dtype"
    raise ValueError(f"Unsupported LOOKUP_DTYPE for {op.name}")


def _render_ternary_fma(op: Operation) -> str:
    if op.op_enum is None or op.op_id is None:
        raise ValueError(f"Ternary {op.name} missing OP_ENUM or OP_ID")
    label = _escape_c_string(op.label or op.name)
    lookup = _lookup_expr(op, "a", "out")
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *a, const marmot_tensor_t *b, const marmot_tensor_t *c, marmot_tensor_t *out) {{\n"
        f"    return marmot_dispatch_ternary(ctx, {op.op_enum}, {op.op_id}, a, b, c, out, {lookup}, \"{label}\");\n"
        "}\n"
    )


def _render_ternary_where(op: Operation) -> str:
    if op.op_enum is None or op.op_id is None:
        raise ValueError(f"Ternary {op.name} missing OP_ENUM or OP_ID")
    label = _escape_c_string(op.label or op.name)
    lookup = _lookup_expr(op, "mask", "out")
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *mask, const marmot_tensor_t *true_value, const marmot_tensor_t *false_value, marmot_tensor_t *out) {{\n"
        f"    return marmot_dispatch_ternary(ctx, {op.op_enum}, {op.op_id}, mask, true_value, false_value, out, {lookup}, \"{label}\");\n"
        "}\n"
    )


def _render_convert_generic(op: Operation) -> str:
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, marmot_dtype_t dst_dtype, void *dst, marmot_dtype_t src_dtype, const void *src, size_t n) {{\n"
        "    return marmot_convert_dispatch(ctx, dst_dtype, dst, src_dtype, src, n);\n"
        "}\n"
    )


def _render_convert_typed(op: Operation, dtype_map: Dict[str, str]) -> str:
    if op.src_dtype is None or op.dst_dtype is None:
        raise ValueError(f"Conversion {op.name} missing SRC_DTYPE or DST_DTYPE")
    if op.src_dtype not in dtype_map or op.dst_dtype not in dtype_map:
        raise ValueError(f"Unknown dtype in conversion {op.name}")
    dst_type = dtype_map[op.dst_dtype]
    src_type = dtype_map[op.src_dtype]
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, {dst_type} *dst, const {src_type} *src, size_t n) {{\n"
        f"    return marmot_convert_dispatch(ctx, {op.dst_dtype}, dst, {op.src_dtype}, src, n);\n"
        "}\n"
    )


def _render_compute_quant_params(op: Operation) -> str:
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *tensor, marmot_dtype_t target_dtype, size_t block_size, marmot_quant_params_t *out_params) {{\n"
        "    return marmot_compute_quant_params_dispatch(ctx, tensor, target_dtype, block_size, out_params);\n"
        "}\n"
    )


def _render_quantize(op: Operation, *, generic: bool) -> str:
    if op.quant_kind is None:
        raise ValueError(f"Quantize {op.name} missing QUANT_KIND")
    if generic:
        return (
            f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_quant_params_t *quant_params, marmot_tensor_t *output) {{\n"
            f"    return marmot_quantize_dispatch(ctx, {op.quant_kind}, input, quant_params, output);\n"
            "}\n"
        )
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {{\n"
        f"    return marmot_quantize_dispatch(ctx, {op.quant_kind}, input, nullptr, output);\n"
        "}\n"
    )


def _dequantize_function_name(op: Operation) -> str:
    if op.api_function.startswith("marmot_quantize"):
        return op.api_function.replace("marmot_quantize", "marmot_dequantize", 1)
    return f"{op.api_function}_dequantize"


def _render_dequantize(op: Operation) -> str:
    if op.quant_kind is None:
        raise ValueError(f"Quantize {op.name} missing QUANT_KIND")
    fn_name = _dequantize_function_name(op)
    return (
        f"marmot_error_t {fn_name}(const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output) {{\n"
        f"    return marmot_dequantize_dispatch(ctx, {op.quant_kind}, input, output);\n"
        "}\n"
    )


def _quant_kind_to_qscheme(kind: str) -> str:
    if kind == "MARMOT_QUANT_KIND_GENERIC":
        return "MARMOT_QSCHEME_NONE"
    prefix = "MARMOT_QUANT_KIND_"
    if not kind.startswith(prefix):
        raise ValueError(f"Unknown quant kind '{kind}'")
    return "MARMOT_QSCHEME_" + kind[len(prefix) :]


def _kernel_args_count(op: Operation) -> str:
    prefix = op.kernel_enum_prefix or op.kernel_op
    if prefix is None:
        raise ValueError(f"Operation {op.name} missing KERNEL_OP")
    return f"MARMOT_KERNEL_ARGS_{prefix.upper()}_COUNT"


def _kernel_args_struct(op: Operation) -> str:
    prefix = op.kernel_enum_prefix or op.kernel_op
    if prefix is None:
        raise ValueError(f"Operation {op.name} missing KERNEL_OP")
    name = str(prefix).lower()
    if name == "embed":
        name = "embedding"
    return f"marmot_kernel_args_{name}_t"


def _humanize_op_name(name: str) -> str:
    if not name:
        return name
    text = name.replace("_", " ")
    return text[0].upper() + text[1:]


def _join_lines(lines: List[str]) -> str:
    return "\n".join(lines) + "\n"


def _render_forward_desc_impl(op: Operation) -> str:
    if op.desc_type is None or op.impl_function is None:
        raise ValueError(f"Forwarder {op.name} missing DESC_TYPE or IMPL_FUNCTION")
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const {op.desc_type} *desc) {{\n"
        f"    return {op.impl_function}(ctx, desc);\n"
        "}\n"
    )


def _render_forward_matmul_impl(op: Operation) -> str:
    if op.impl_function is None:
        raise ValueError(f"Forwarder {op.name} missing IMPL_FUNCTION")
    return (
        f"marmot_error_t {op.api_function}(\n"
        "    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *weight,\n"
        "    const marmot_matmul_epilogue_t *epilogue, marmot_tensor_t *out\n"
        ") {\n"
        f"    return {op.impl_function}(ctx, input, weight, epilogue, out);\n"
        "}\n"
    )


def _render_forward_ctx_tensor_impl(op: Operation) -> str:
    if op.impl_function is None:
        raise ValueError(f"Forwarder {op.name} missing IMPL_FUNCTION")
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *weight) {{\n"
        f"    return {op.impl_function}(ctx, weight);\n"
        "}\n"
    )


def _render_layernorm_desc(op: Operation) -> str:
    if op.desc_type is None:
        raise ValueError(f"Layernorm {op.name} missing DESC_TYPE")
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const {op.desc_type} *desc) {{\n"
        "    return marmot_layernorm_dispatch(ctx, desc);\n"
        "}\n"
    )


def _render_rmsnorm_desc(op: Operation) -> str:
    if op.desc_type is None:
        raise ValueError(f"RMSNorm {op.name} missing DESC_TYPE")
    dispatch_fn = "marmot_rmsnorm_gemma_dispatch" if op.name == "rmsnorm_gemma" else "marmot_rmsnorm_dispatch"
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const {op.desc_type} *desc) {{\n"
        f"    return {dispatch_fn}(ctx, desc);\n"
        "}\n"
    )


def _render_softmax_desc(op: Operation) -> str:
    if op.desc_type is None:
        raise ValueError(f"Softmax {op.name} missing DESC_TYPE")
    return (
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const {op.desc_type} *desc) {{\n"
        "    return marmot_softmax_dispatch(ctx, desc);\n"
        "}\n"
    )


def _render_embedding_gather_desc(op: Operation) -> str:
    if op.desc_type is None or op.build_fn is None:
        raise ValueError(f"Embedding {op.name} missing DESC_TYPE or BUILD_FN")
    label = _escape_c_string(op.label or op.name)
    error_label = _humanize_op_name(op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const {op.desc_type} *desc) {{",
        "    if (ctx == nullptr || desc == nullptr) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"{error_label} requires non-null context and descriptor\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
        "    marmot_embedding_gather_desc_t resolved_desc;",
        "    marmot_embedding_resolve_gather_desc(ctx, desc, &resolved_desc);",
        "    const marmot_embedding_gather_desc_t *resolved_ptr = &resolved_desc;",
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        "    marmot_dtype_t resolved_dtype = (marmot_dtype_t)MARMOT_DTYPE_COUNT;",
        f"    marmot_error_t build_status = {op.build_fn}(ctx, resolved_ptr, &sig, &packed, &resolved_dtype);",
        "    if (build_status != MARMOT_SUCCESS) {",
            "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_vec_dot_desc(op: Operation) -> str:
    if op.desc_type is None:
        raise ValueError(f"Vec dot {op.name} missing DESC_TYPE")
    lines = [
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const {op.desc_type} *desc, float *result) {{",
        f"    return marmot_vec_dot_dispatch(ctx, desc, result);",
        "}",
    ]
    return _join_lines(lines)


def _render_tensor_reshape(op: Operation) -> str:
    if op.build_fn is None:
        raise ValueError(f"Tensor op {op.name} missing BUILD_FN")
    label = _escape_c_string(op.label or op.name)
    error_label = _humanize_op_name(op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(",
        "    const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *new_shape,",
        "    size_t new_ndim",
        ") {",
        "    if (ctx == nullptr || x == nullptr || out == nullptr || new_shape == nullptr) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"{error_label} requires non-null context, tensors, and shape\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
    ]
    lines += [
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        f"    marmot_error_t build_status = {op.build_fn}(ctx, x, out, new_shape, new_ndim, &sig, &packed);",
        "    if (build_status != MARMOT_SUCCESS) {",
        "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_tensor_view(op: Operation) -> str:
    if op.build_fn is None:
        raise ValueError(f"Tensor op {op.name} missing BUILD_FN")
    label = _escape_c_string(op.label or op.name)
    error_label = _humanize_op_name(op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, size_t byte_offset) {{",
        "    if (ctx == nullptr || x == nullptr || out == nullptr) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"{error_label} requires non-null context and tensors\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
    ]
    lines += [
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        f"    marmot_error_t build_status = {op.build_fn}(ctx, x, out, byte_offset, &sig, &packed);",
        "    if (build_status != MARMOT_SUCCESS) {",
        "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_tensor_contiguous(op: Operation) -> str:
    if op.build_fn is None:
        raise ValueError(f"Tensor op {op.name} missing BUILD_FN")
    label = _escape_c_string(op.label or op.name)
    error_label = _humanize_op_name(op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out) {{",
        "    if (ctx == nullptr || x == nullptr || out == nullptr) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"{error_label} requires non-null context and tensors\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        f"    marmot_error_t build_status = {op.build_fn}(ctx, x, out, &sig, &packed);",
        "    if (build_status != MARMOT_SUCCESS) {",
        "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_tensor_transpose(op: Operation) -> str:
    if op.build_fn is None:
        raise ValueError(f"Tensor op {op.name} missing BUILD_FN")
    label = _escape_c_string(op.label or op.name)
    error_label = _humanize_op_name(op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const int *perm) {{",
        "    if (ctx == nullptr || x == nullptr || out == nullptr) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"{error_label} requires non-null context and tensors\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
    ]
    lines += [
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        f"    marmot_error_t build_status = {op.build_fn}(ctx, x, out, perm, &sig, &packed);",
        "    if (build_status != MARMOT_SUCCESS) {",
        "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_tensor_concat(op: Operation) -> str:
    if op.build_fn is None:
        raise ValueError(f"Tensor op {op.name} missing BUILD_FN")
    label = _escape_c_string(op.label or op.name)
    error_label = _humanize_op_name(op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(",
        "    const marmot_context_t *ctx, const marmot_tensor_t *const *tensors, size_t num_tensors, marmot_tensor_t *out,",
        "    int axis",
        ") {",
        "    if (ctx == nullptr || tensors == nullptr || out == nullptr || num_tensors == 0) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"{error_label} requires non-null context, tensors, and outputs\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
    ]
    lines += [
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        f"    marmot_error_t build_status = {op.build_fn}(ctx, tensors, num_tensors, out, axis, &sig, &packed);",
        "    if (build_status != MARMOT_SUCCESS) {",
        "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_tensor_slice(op: Operation) -> str:
    if op.build_fn is None:
        raise ValueError(f"Tensor op {op.name} missing BUILD_FN")
    label = _escape_c_string(op.label or op.name)
    error_label = _humanize_op_name(op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(",
        "    const marmot_context_t *ctx, const marmot_tensor_t *x, marmot_tensor_t *out, const size_t *starts,",
        "    const size_t *sizes",
        ") {",
        "    if (ctx == nullptr || x == nullptr || out == nullptr || starts == nullptr || sizes == nullptr) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"{error_label} requires non-null context, tensors, and slice metadata\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
    ]
    lines += [
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        f"    marmot_error_t build_status = {op.build_fn}(ctx, x, out, starts, sizes, &sig, &packed);",
        "    if (build_status != MARMOT_SUCCESS) {",
        "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_tensor_gather_rows(op: Operation) -> str:
    if op.build_fn is None:
        raise ValueError(f"Tensor op {op.name} missing BUILD_FN")
    label = _escape_c_string(op.label or op.name)
    error_label = _humanize_op_name(op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(",
        "    const marmot_context_t *ctx, const marmot_tensor_t *input, const marmot_tensor_t *indices,",
        "    marmot_tensor_t *output",
        ") {",
        "    if (ctx == nullptr || input == nullptr || indices == nullptr || output == nullptr) {",
        f"        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, \"{error_label} requires non-null context and tensors\");",
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        f"    marmot_error_t build_status = {op.build_fn}(ctx, input, indices, output, &sig, &packed);",
        "    if (build_status != MARMOT_SUCCESS) {",
        "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_rope_args(op: Operation) -> str:
    if op.kernel_op is None or op.op_id is None:
        raise ValueError(f"RoPE {op.name} missing KERNEL_OP or OP_ID")
    label = _escape_c_string(op.label or op.name)
    args_count = _kernel_args_count(op)
    args_struct = _kernel_args_struct(op)
    lines = [
        f"marmot_error_t {op.api_function}(",
        "    const marmot_context_t *ctx, const marmot_tensor_t *x, const marmot_rope_params_t *params, marmot_tensor_t *out",
        ") {",
        "    if (ctx == nullptr) {",
        '        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE operation requires non-null context");',
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
        "    if (x == nullptr || out == nullptr || params == nullptr) {",
        '        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE operation requires non-null tensors");',
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
        "    if (params->positions == nullptr) {",
        '        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "RoPE operation requires positions tensor");',
        "        return MARMOT_ERROR_INVALID_ARGUMENT;",
        "    }",
        "    marmot_op_signature_t sig = {0};",
        f"    {args_struct} packed = {{0}};",
        "    marmot_error_t build_status = marmot_rope_build(ctx, x, params, out, &sig, &packed);",
        "    if (build_status != MARMOT_SUCCESS) {",
        "        return build_status;",
        "    }",
        f"    return marmot_execute_signature(ctx, &sig, &packed, \"{label}\");",
        "}",
    ]
    return _join_lines(lines)


def _render_entries(template_dir: Path, out_path: Path, entries: List[str]) -> None:
    env = make_jinja_env(template_dir)
    template = env.get_template("api_dispatch.inc.j2")
    content = template.render(entries=entries)
    write_output(out_path, content)


def _render_metadata(
    template_dir: Path,
    out_path: Path,
    reductions: List[Operation],
    unary_ops: List[Operation],
    binary_ops: List[Operation],
    ternary_ops: List[Operation],
    unary_params: List[Operation],
    unary_bias: List[Operation],
    quant_kinds: List[tuple[str, str]],
) -> None:
    env = make_jinja_env(template_dir)
    template = env.get_template("op_metadata.h.j2")
    content = template.render(
        reductions=reductions,
        unary_ops=unary_ops,
        binary_ops=binary_ops,
        ternary_ops=ternary_ops,
        unary_params=unary_params,
        unary_bias=unary_bias,
        quant_kinds=quant_kinds,
    )
    write_output(out_path, content)


def _render_docs(template_dir: Path, out_path: Path, ops_by_category: Dict[str, List[Operation]]) -> None:
    sections = [
        ("Reductions", ops_by_category.get("reduction", [])),
        ("Unary", ops_by_category.get("unary", [])),
        ("Binary Elementwise", ops_by_category.get("binary", [])),
        ("Ternary Elementwise", ops_by_category.get("ternary", [])),
        ("Conversions", ops_by_category.get("convert", [])),
        ("Quantization", ops_by_category.get("quantize", [])),
        ("Normalization", ops_by_category.get("normalization", [])),
        ("Embedding", ops_by_category.get("embedding", [])),
        ("Tensor Manipulation", ops_by_category.get("manipulation", [])),
        ("RoPE", ops_by_category.get("rope", [])),
        ("Matmul", ops_by_category.get("matmul", [])),
        ("Attention", ops_by_category.get("attention", [])),
    ]
    env = make_jinja_env(template_dir)
    env.trim_blocks = False
    env.lstrip_blocks = False
    template = env.get_template("api_ops_md.j2")
    content = template.render(sections=sections)
    write_output(out_path, content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate API dispatch wrappers and metadata")
    parser.add_argument("--ops-def", required=True, type=Path, help="ops.def path")
    parser.add_argument("--metadata-def", type=Path, help="Optional ops metadata .def path")
    parser.add_argument("--output-reductions-inc", required=True, type=Path)
    parser.add_argument("--output-unary-inc", required=True, type=Path)
    parser.add_argument("--output-elementwise-inc", required=True, type=Path)
    parser.add_argument("--output-conversion-inc", required=True, type=Path)
    parser.add_argument("--output-quantization-inc", required=True, type=Path)
    parser.add_argument("--output-normalization-inc", required=True, type=Path)
    parser.add_argument("--output-embedding-inc", required=True, type=Path)
    parser.add_argument("--output-tensor-ops-inc", required=True, type=Path)
    parser.add_argument("--output-rope-inc", required=True, type=Path)
    parser.add_argument("--output-matmul-inc", required=True, type=Path)
    parser.add_argument("--output-attention-inc", required=True, type=Path)
    parser.add_argument("--output-metadata", required=True, type=Path)
    parser.add_argument("--output-docs", type=Path)
    parser.add_argument("--template-dir", required=True, type=Path)
    parser.add_argument("--stamp-output", type=Path)
    args = parser.parse_args()

    paths = [args.ops_def]
    if args.metadata_def is not None:
        paths.append(args.metadata_def)

    ops = _eval_ops(paths)
    ops_by_name = {op.name: op for op in ops}

    reductions: List[Operation] = []
    unary: List[Operation] = []
    elementwise: List[Operation] = []
    conversions: List[Operation] = []
    quantization: List[Operation] = []
    normalization: List[Operation] = []
    embedding: List[Operation] = []
    manipulation: List[Operation] = []
    rope: List[Operation] = []
    matmul: List[Operation] = []
    attention: List[Operation] = []
    ops_by_category: Dict[str, List[Operation]] = {}

    for op in ops:
        ops_by_category.setdefault(op.category, []).append(op)
        if op.category == "reduction":
            reductions.append(op)
        elif op.category == "unary":
            unary.append(op)
        elif op.category in {"binary", "ternary"}:
            elementwise.append(op)
        elif op.category == "convert":
            conversions.append(op)
        elif op.category == "quantize":
            quantization.append(op)
        elif op.category == "normalization":
            normalization.append(op)
        elif op.category == "embedding":
            embedding.append(op)
        elif op.category == "manipulation":
            manipulation.append(op)
        elif op.category == "rope":
            rope.append(op)
        elif op.category == "matmul":
            matmul.append(op)
        elif op.category == "attention":
            attention.append(op)

    reduction_entries = [_render_reduction(op) for op in reductions]

    unary_entries: List[str] = []
    for op in unary:
        if op.alias_of is not None:
            target = _alias_target(ops_by_name, op)
            unary_entries.append(_render_unary_alias(op, target))
        elif op.api_signature == "unary_basic":
            unary_entries.append(_render_unary_basic(op))
        elif op.api_signature == "unary_alpha":
            unary_entries.append(_render_unary_alpha(op))
        elif op.api_signature == "unary_alpha_beta":
            unary_entries.append(_render_unary_alpha_beta(op))
        else:
            raise ValueError(f"Unsupported unary signature {op.api_signature} for {op.name}")

    elementwise_entries: List[str] = []
    for op in elementwise:
        if op.api_signature == "binary_basic":
            elementwise_entries.append(_render_binary(op))
        elif op.api_signature == "ternary_fma":
            elementwise_entries.append(_render_ternary_fma(op))
        elif op.api_signature == "ternary_where":
            elementwise_entries.append(_render_ternary_where(op))
        else:
            raise ValueError(f"Unsupported elementwise signature {op.api_signature} for {op.name}")

    dtype_map = {
        "MARMOT_DTYPE_FLOAT16": "marmot_float16_t",
        "MARMOT_DTYPE_BFLOAT16": "marmot_bfloat16_t",
        "MARMOT_DTYPE_FLOAT32": "float",
        "MARMOT_DTYPE_FLOAT64": "double",
        "MARMOT_DTYPE_INT8": "int8_t",
        "MARMOT_DTYPE_UINT8": "uint8_t",
        "MARMOT_DTYPE_INT16": "int16_t",
        "MARMOT_DTYPE_UINT16": "uint16_t",
        "MARMOT_DTYPE_INT32": "int32_t",
        "MARMOT_DTYPE_UINT32": "uint32_t",
        "MARMOT_DTYPE_INT64": "int64_t",
        "MARMOT_DTYPE_UINT64": "uint64_t",
    }

    conversion_entries: List[str] = []
    for op in conversions:
        if op.api_signature == "convert_generic":
            conversion_entries.append(_render_convert_generic(op))
        elif op.api_signature == "convert_typed":
            conversion_entries.append(_render_convert_typed(op, dtype_map))
        else:
            raise ValueError(f"Unsupported conversion signature {op.api_signature} for {op.name}")

    quant_entries: List[str] = []
    for op in quantization:
        if op.api_signature == "compute_quant_params":
            quant_entries.append(_render_compute_quant_params(op))
        elif op.api_signature == "quantize_generic":
            quant_entries.append(_render_quantize(op, generic=True))
            quant_entries.append(_render_dequantize(op))
        elif op.api_signature == "quantize_fixed_kind":
            quant_entries.append(_render_quantize(op, generic=False))
            quant_entries.append(_render_dequantize(op))
        elif op.api_signature == "vec_dot_desc":
            quant_entries.append(_render_vec_dot_desc(op))
        else:
            raise ValueError(f"Unsupported quant signature {op.api_signature} for {op.name}")

    normalization_entries: List[str] = []
    for op in normalization:
        if op.api_signature == "layernorm_desc":
            normalization_entries.append(_render_layernorm_desc(op))
        elif op.api_signature == "rmsnorm_desc":
            normalization_entries.append(_render_rmsnorm_desc(op))
        elif op.api_signature == "softmax_desc":
            normalization_entries.append(_render_softmax_desc(op))
        else:
            raise ValueError(f"Unsupported normalization signature {op.api_signature} for {op.name}")

    embedding_entries: List[str] = []
    for op in embedding:
        if op.api_signature == "embedding_gather_desc":
            embedding_entries.append(_render_embedding_gather_desc(op))
        elif op.api_signature == "forward_desc_impl":
            embedding_entries.append(_render_forward_desc_impl(op))
        else:
            raise ValueError(f"Unsupported embedding signature {op.api_signature} for {op.name}")

    manipulation_entries: List[str] = []
    for op in manipulation:
        if op.api_signature == "tensor_reshape":
            manipulation_entries.append(_render_tensor_reshape(op))
        elif op.api_signature == "tensor_view":
            manipulation_entries.append(_render_tensor_view(op))
        elif op.api_signature == "tensor_contiguous":
            manipulation_entries.append(_render_tensor_contiguous(op))
        elif op.api_signature == "tensor_transpose":
            manipulation_entries.append(_render_tensor_transpose(op))
        elif op.api_signature == "tensor_concat":
            manipulation_entries.append(_render_tensor_concat(op))
        elif op.api_signature == "tensor_slice":
            manipulation_entries.append(_render_tensor_slice(op))
        elif op.api_signature == "tensor_gather_rows":
            manipulation_entries.append(_render_tensor_gather_rows(op))
        else:
            raise ValueError(f"Unsupported manipulation signature {op.api_signature} for {op.name}")

    rope_entries: List[str] = []
    for op in rope:
        if op.api_signature == "rope_args":
            rope_entries.append(_render_rope_args(op))
        else:
            raise ValueError(f"Unsupported rope signature {op.api_signature} for {op.name}")

    matmul_entries: List[str] = []
    for op in matmul:
        if op.api_signature == "forward_matmul_impl":
            matmul_entries.append(_render_forward_matmul_impl(op))
        elif op.api_signature == "forward_desc_impl":
            matmul_entries.append(_render_forward_desc_impl(op))
        elif op.api_signature == "forward_ctx_tensor_impl":
            matmul_entries.append(_render_forward_ctx_tensor_impl(op))
        else:
            raise ValueError(f"Unsupported matmul signature {op.api_signature} for {op.name}")

    attention_entries: List[str] = []
    for op in attention:
        if op.api_signature == "forward_desc_impl":
            attention_entries.append(_render_forward_desc_impl(op))
        else:
            raise ValueError(f"Unsupported attention signature {op.api_signature} for {op.name}")

    _render_entries(args.template_dir, args.output_reductions_inc, reduction_entries)
    _render_entries(args.template_dir, args.output_unary_inc, unary_entries)
    _render_entries(args.template_dir, args.output_elementwise_inc, elementwise_entries)
    _render_entries(args.template_dir, args.output_conversion_inc, conversion_entries)
    _render_entries(args.template_dir, args.output_quantization_inc, quant_entries)
    _render_entries(args.template_dir, args.output_normalization_inc, normalization_entries)
    _render_entries(args.template_dir, args.output_embedding_inc, embedding_entries)
    _render_entries(args.template_dir, args.output_tensor_ops_inc, manipulation_entries)
    _render_entries(args.template_dir, args.output_rope_inc, rope_entries)
    _render_entries(args.template_dir, args.output_matmul_inc, matmul_entries)
    _render_entries(args.template_dir, args.output_attention_inc, attention_entries)

    unary_params = [
        op
        for op in unary
        if op.alias_of is None and op.api_signature in {"unary_alpha", "unary_alpha_beta"} and op.op_enum is not None
    ]
    unary_bias = [op for op in unary if op.supports_bias and op.op_enum is not None]
    unary_ops = [op for op in unary if op.op_enum is not None and op.op_id is not None]
    binary_ops = [
        op for op in ops_by_category.get("binary", []) if op.op_enum is not None and op.op_id is not None
    ]
    ternary_ops = [
        op for op in ops_by_category.get("ternary", []) if op.op_enum is not None and op.op_id is not None
    ]
    quant_kinds = []
    seen_kinds = set()
    for op in quantization:
        if op.quant_kind is None:
            continue
        if op.quant_kind in seen_kinds:
            continue
        seen_kinds.add(op.quant_kind)
        quant_kinds.append((op.quant_kind, _quant_kind_to_qscheme(op.quant_kind)))
    _render_metadata(
        args.template_dir,
        args.output_metadata,
        reductions,
        unary_ops,
        binary_ops,
        ternary_ops,
        unary_params,
        unary_bias,
        quant_kinds,
    )

    if args.output_docs is not None:
        _render_docs(args.template_dir, args.output_docs, ops_by_category)

    if args.stamp_output:
        write_stamp(args.stamp_output)


if __name__ == "__main__":
    main()
