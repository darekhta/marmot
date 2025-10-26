#!/usr/bin/env python3
"""Generate numpy-based golden data for graph-layer tests."""

from __future__ import annotations

from pathlib import Path

import random

try:  # pragma: no cover - numpy optional
    import numpy as np  # type: ignore
    if not hasattr(np, "random"):
        np = None
except Exception:  # pragma: no cover - numpy optional
    np = None

if np is not None:
    _np_rng = np.random.default_rng(seed=42)
else:
    _fallback_rng = random.Random(42)

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "graph" / "golden_graph_numpy.h"


def _flatten(values):
    if np is not None:
        return values.flatten()
    flat = []
    for row in values:
        if isinstance(row, (list, tuple)):
            flat.extend(row)
        else:
            flat.append(row)
    return flat


def _matmul(lhs, rhs):
    if np is not None:
        return lhs @ rhs.T
    rows = len(lhs)
    k = len(lhs[0])
    m = len(rhs)
    result = []
    for i in range(rows):
        row = []
        for j in range(m):
            acc = 0.0
            for p in range(k):
                acc += lhs[i][p] * rhs[j][p]
            row.append(acc)
        result.append(row)
    return result


def _standard_normal(rows: int, cols: int):
    if np is not None:
        return np.array(_np_rng.standard_normal((rows, cols)), dtype=np.float32)
    return [[_fallback_rng.gauss(0.0, 1.0) for _ in range(cols)] for _ in range(rows)]


def format_case(struct_name: str, rows: int, k: int, cols: int, input_vals: np.ndarray,
                weight_vals: np.ndarray, expected: np.ndarray) -> str:
    input_str = ", ".join(f"{val:.9f}f" for val in _flatten(input_vals))
    weight_str = ", ".join(f"{val:.9f}f" for val in _flatten(weight_vals))
    expected_str = ", ".join(f"{val:.9f}f" for val in _flatten(expected))
    parts = [
        f"static const graph_matmul_case_t {struct_name} = {{",
        f"    .rows = {rows},",
        f"    .k = {k},",
        f"    .cols = {cols},",
        f"    .input = {{{input_str}}},",
        f"    .weight = {{{weight_str}}},",
        f"    .expected = {{{expected_str}}},",
        "};",
    ]
    return "\n".join(parts)


def format_chain_case(struct_name: str, rows: int, k1: int, m1: int, m2: int,
                      input_vals: np.ndarray, weight1: np.ndarray, weight2: np.ndarray,
                      expected: np.ndarray) -> str:
    input_str = ", ".join(f"{val:.9f}f" for val in _flatten(input_vals))
    weight1_str = ", ".join(f"{val:.9f}f" for val in _flatten(weight1))
    weight2_str = ", ".join(f"{val:.9f}f" for val in _flatten(weight2))
    expected_str = ", ".join(f"{val:.9f}f" for val in _flatten(expected))
    parts = [
        f"static const graph_matmul_chain_case_t {struct_name} = {{",
        f"    .rows = {rows},",
        f"    .k1 = {k1},",
        f"    .m1 = {m1},",
        f"    .m2 = {m2},",
        f"    .input = {{{input_str}}},",
        f"    .weight1 = {{{weight1_str}}},",
        f"    .weight2 = {{{weight2_str}}},",
        f"    .expected = {{{expected_str}}},",
        "};",
    ]
    return "\n".join(parts)


def generate() -> None:
    # Single matmul case
    rows, k, cols = 2, 3, 4
    single_input = _standard_normal(rows, k)
    single_weight = _standard_normal(cols, k)
    single_expected = _matmul(single_input, single_weight)

    # Two-stage case (stage1: rows x k1 times m1 x k1, stage2: rows x m1 times m2 x m1)
    rows_chain, k1, m1, m2 = 3, 4, 5, 6
    chain_input = _standard_normal(rows_chain, k1)
    chain_weight1 = _standard_normal(m1, k1)
    chain_weight2 = _standard_normal(m2, m1)
    chain_stage1 = _matmul(chain_input, chain_weight1)
    chain_expected = _matmul(chain_stage1, chain_weight2)

    single_input_count = rows * k
    single_weight_count = cols * k
    single_expected_count = rows * cols
    chain_input_count = rows_chain * k1
    chain_weight1_count = m1 * k1
    chain_weight2_count = m2 * m1
    chain_expected_count = rows_chain * m2

    header_lines = [
        "// Generated via tests/golden/generate_graph_numpy.py using NumPy",
        "#pragma once",
        "#include <stddef.h>",
        "",
        "typedef struct {",
        "    size_t rows;",
        "    size_t k;",
        "    size_t cols;",
        f"    float input[{single_input_count}];",
        f"    float weight[{single_weight_count}];",
        f"    float expected[{single_expected_count}];",
        "} graph_matmul_case_t;",
        "",
        "typedef struct {",
        "    size_t rows;",
        "    size_t k1;",
        "    size_t m1;",
        "    size_t m2;",
        f"    float input[{chain_input_count}];",
        f"    float weight1[{chain_weight1_count}];",
        f"    float weight2[{chain_weight2_count}];",
        f"    float expected[{chain_expected_count}];",
        "} graph_matmul_chain_case_t;",
        "",
    ]

    header_lines.append(format_case("g_graph_matmul_single_case", rows, k, cols,
                                    single_input, single_weight, single_expected))
    header_lines.append("")
    header_lines.append(format_chain_case("g_graph_matmul_chain_case", rows_chain, k1, m1, m2,
                                          chain_input, chain_weight1, chain_weight2, chain_expected))
    header_lines.append("")

    OUTPUT_PATH.write_text("\n".join(header_lines), encoding="utf-8")


if __name__ == "__main__":
    generate()
