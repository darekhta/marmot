#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import List

from def_parser import DefParseError, collect_descriptors, validate_descriptor


def validate_stdin() -> None:
    content = sys.stdin.read()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "stdin.def"
        path.write_text(content, encoding="utf-8")
        validate_path(path)


def validate_path(path: Path) -> None:
    descriptors = collect_descriptors([path])
    for desc in descriptors:
        validate_descriptor(desc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Marmot kernel .def files")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more .def files or directories containing .def files (use '-' for stdin)",
    )
    args = parser.parse_args()
    try:
        real_paths: List[Path] = []
        stdin_requested = False
        for raw in args.paths:
            if raw == "-":
                stdin_requested = True
            else:
                real_paths.append(Path(raw))
        if stdin_requested:
            validate_stdin()
        for path in real_paths:
            validate_path(path)
    except DefParseError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
