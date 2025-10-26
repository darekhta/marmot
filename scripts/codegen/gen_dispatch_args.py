#!/usr/bin/env python3
"""
Generate kernel_dispatch_args.h from ops_schema.py definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import def_parser
from codegen_base import make_jinja_env, write_output, write_stamp
from ops_schema import SCHEMAS


def _op_id_to_name(op_id: str) -> str:
    prefix = "MARMOT_OP_"
    if not op_id.startswith(prefix):
        raise ValueError(f"Unexpected op id '{op_id}'")
    return op_id[len(prefix) :].lower()


def _load_ops_from_def(paths: list[Path]) -> dict[str, set[str]]:
    if not paths:
        return {}
    mod = def_parser.parse_def_files(paths)
    sets: dict[str, list[object]] = {}
    for name, s in mod.sets.items():
        sets[name] = def_parser._expand_set(mod.sets, name, stack=[], path=s.path, source=s.source)
    fragments = def_parser._resolve_fragments(mod, sets)
    ctx = def_parser._ResolvedContext(path=Path("<ops.def>"), source="", sets=sets, fragments=fragments, quant_schemes={})

    ops_by_category: dict[str, set[str]] = {}
    for name, op_def in mod.operations.items():
        fields_expr = def_parser._resolve_block_fields(mod.operations, name, kind="operation", stack=[])
        fields: dict[str, object] = {}
        for key, expr in fields_expr.items():
            fields[key] = def_parser._eval_expr(expr, {}, ctx)
        category = str(fields.get("CATEGORY", "")).lower()
        if not category:
            continue
        if fields.get("ALIAS_OF") is not None:
            continue
        op_id = fields.get("OP_ID")
        if op_id is None:
            continue
        op_name = _op_id_to_name(str(op_id))
        ops_by_category.setdefault(category, set()).add(op_name)
    return ops_by_category


_SCHEMA_OPS_DEF_CATEGORIES = {"binary", "convert", "reduction", "ternary", "unary"}


def _apply_ops_def(ops_by_category: dict[str, set[str]]) -> None:
    for schema in SCHEMAS:
        if schema.name not in ops_by_category or schema.name not in _SCHEMA_OPS_DEF_CATEGORIES:
            continue
        ops_from_def = ops_by_category[schema.name]
        if schema.ops and schema.ops != ops_from_def:
            missing = sorted(ops_from_def - schema.ops)
            extra = sorted(schema.ops - ops_from_def)
            parts = []
            if missing:
                parts.append(f"missing in ops_schema: {', '.join(missing)}")
            if extra:
                parts.append(f"extra in ops_schema: {', '.join(extra)}")
            detail = "; ".join(parts) if parts else "unknown mismatch"
            raise SystemExit(f"ops_schema mismatch for category '{schema.name}': {detail}")
        schema.ops = ops_from_def


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate kernel dispatch args header")
    parser.add_argument("--output", required=True, type=Path, help="Output header file path")
    parser.add_argument("--template-dir", required=True, type=Path, help="Template directory")
    parser.add_argument("--ops-def", type=Path, help="Optional ops.def path for schema validation")
    parser.add_argument("--metadata-def", type=Path, help="Optional ops metadata .def path")
    parser.add_argument("--stamp-output", type=Path, help="Optional stamp file for build system")
    args = parser.parse_args()

    ops_def_paths: list[Path] = []
    if args.ops_def is not None:
        ops_def_paths.append(args.ops_def)
    if args.metadata_def is not None:
        ops_def_paths.append(args.metadata_def)
    if ops_def_paths:
        ops_by_category = _load_ops_from_def(ops_def_paths)
        _apply_ops_def(ops_by_category)

    env = make_jinja_env(args.template_dir)
    template = env.get_template("kernel_dispatch_args.h.j2")
    output = template.render(schemas=SCHEMAS, enumerate=enumerate)

    write_output(args.output, output)

    if args.stamp_output:
        write_stamp(args.stamp_output)


if __name__ == "__main__":
    main()
