#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    root = repo_root()
    sys.path.insert(0, str(root / "scripts/codegen"))

    from gen_kernels import (
        allocate_op_ids,
        allocate_kernel_ids,
        stable_kernel_id,
        stable_op_id,
    )  # pylint: disable=import-error

    op_range = (100, 999)
    for op_name in ("matmul", "qkv_rope", "layernorm"):
        first = stable_op_id(op_name)
        second = stable_op_id(op_name)
        if first != second:
            raise SystemExit(f"stable_op_id('{op_name}') is not deterministic")
        if not (op_range[0] <= first < op_range[1]):
            raise SystemExit(f"stable_op_id('{op_name}')={first} outside range {op_range[0]}-{op_range[1] - 1}")

    backend_ranges = {
        "metal": (1000, 9999),
        "cpu": (10000, 59999),
        "cuda": (60000, 99999),
    }
    for backend, (start, end) in backend_ranges.items():
        for kernel_name in ("matmul_f16_nt", "matmul_f32_nt"):
            first = stable_kernel_id(kernel_name, backend)
            second = stable_kernel_id(kernel_name, backend)
            if first != second:
                raise SystemExit(f"stable_kernel_id('{kernel_name}', '{backend}') is not deterministic")
            if not (start <= first < end):
                raise SystemExit(
                    f"stable_kernel_id('{kernel_name}', '{backend}')={first} outside range {start}-{end - 1}"
                )

    ops = ["matmul", "qkv_rope", "rope", "matmul", "add"]
    forward = allocate_op_ids(ops)
    backward = allocate_op_ids(list(reversed(ops)))
    if forward != backward:
        raise SystemExit("allocate_op_ids is not deterministic across input order")
    if len(set(forward.values())) != len(forward):
        raise SystemExit("allocate_op_ids produced duplicate IDs")
    for op_name, op_id in forward.items():
        if not (op_range[0] <= op_id < op_range[1]):
            raise SystemExit(f"allocate_op_ids produced out-of-range ID for '{op_name}': {op_id}")

    from def_parser import KernelDescriptor  # pylint: disable=import-error

    def make_desc(name: str) -> KernelDescriptor:
        return KernelDescriptor(name=name, core_fields={}, extensions={}, path=Path(), source="")

    descs = [make_desc("cpu_matmul_f16_nt"), make_desc("cpu_matmul_f32_nt"), make_desc("cpu_matmul_bf16_nt")]
    forward_ids = allocate_kernel_ids("cpu", descs)
    reverse_ids = allocate_kernel_ids("cpu", list(reversed(descs)))
    if forward_ids != reverse_ids:
        raise SystemExit("allocate_kernel_ids is not deterministic across input order")
    if len(set(forward_ids.values())) != len(forward_ids):
        raise SystemExit("allocate_kernel_ids produced duplicate IDs")

    cpu_start, cpu_end = backend_ranges["cpu"]
    for kernel_name, kernel_id in forward_ids.items():
        if not (cpu_start <= kernel_id < cpu_end):
            raise SystemExit(f"allocate_kernel_ids produced out-of-range ID for '{kernel_name}': {kernel_id}")


if __name__ == "__main__":
    main()
