#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def add_codegen_path() -> None:
    sys.path.insert(0, str(repo_root() / "scripts/codegen"))


def expect_error(content: str, *, expected_line: int, expected_col: int, expected_substring: str) -> None:
    add_codegen_path()
    from def_parser import DefParseError, validate_paths  # pylint: disable=import-error

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.def"
        path.write_text(content, encoding="utf-8")
        try:
            validate_paths([path])
        except DefParseError as exc:
            message = str(exc)
            if path.name not in message:
                raise SystemExit(f"expected path '{path.name}' in error message: {message}")
            location = f":{expected_line}:{expected_col}"
            if location not in message:
                raise SystemExit(f"expected location {location} in error message: {message}")
            if expected_substring not in message:
                raise SystemExit(f"expected substring '{expected_substring}' in error message: {message}")
            return
        raise SystemExit("expected DefParseError but validation succeeded")


def main() -> None:
    # Test missing required field (ACCUM_DTYPE is still required)
    expect_error(
        """KERNEL_FAMILY(missing_accum) {
    NAME_PATTERN: "missing_accum",
    OP: matmul,
    PROFILE: SCALAR,
    MATMUL_LAYOUT: NT,
    INPUT_DTYPE: FLOAT16,
    OUTPUT_DTYPE: FLOAT16,
    WEIGHT_DTYPE: FLOAT16
}
""",
        expected_line=1,
        expected_col=1,
        expected_substring="missing required field 'ACCUM_DTYPE'",
    )

    expect_error(
        """KERNEL_FAMILY(missing_stride) {
    NAME_PATTERN: "missing_stride",
    OP: softmax,
    INPUT_DTYPE: FLOAT16,
    OUTPUT_DTYPE: FLOAT16,
    ACCUM_DTYPE: FLOAT32
}
""",
        expected_line=1,
        expected_col=1,
        expected_substring="missing required field 'STRIDE_MODE'",
    )

    expect_error(
        """KERNEL_FAMILY(bad_quant_block) {
    NAME_PATTERN: "bad_quant_block",
    OP: matmul,
    PROFILE: SCALAR,
    MATMUL_LAYOUT: NT,
    INPUT_DTYPE: FLOAT16,
    OUTPUT_DTYPE: FLOAT16,
    ACCUM_DTYPE: FLOAT32,
    STRIDE_MODE: CONTIGUOUS,
    WEIGHT_QUANT: Q4_K,
    QUANT_BLOCK: {
        block_size: 64,
        group_size: 32,
        scale_dtype: FLOAT16
    }
}
""",
        expected_line=11,
        expected_col=5,
        expected_substring="QUANT_BLOCK missing required subfields",
    )


if __name__ == "__main__":
    main()
