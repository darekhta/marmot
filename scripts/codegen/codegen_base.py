#!/usr/bin/env python3
"""
Shared utilities for Marmot code generation scripts.

This module provides common functionality used across all codegen scripts:
- Jinja2 environment setup
- Output file writing with consistent formatting
- Stamp file handling
- Comment stripping for .def file parsing
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import jinja2


def make_jinja_env(template_dir: Path) -> jinja2.Environment:
    """Create a standardized Jinja2 environment for codegen templates."""
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def _should_format(path: Path) -> bool:
    """Check if a file should be formatted with clang-format."""
    return path.suffix in {".c", ".h", ".cpp", ".hpp", ".mm", ".metal"}


def _run_clang_format(path: Path) -> None:
    """Run clang-format on a file if available."""
    clang_format = shutil.which("clang-format")
    if clang_format and _should_format(path):
        subprocess.run([clang_format, "-i", str(path)], check=False)


def write_output(path: Path, content: str) -> None:
    """Write generated content to a file, ensuring proper formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")
    _run_clang_format(path)


def write_stamp(path: Path) -> None:
    """Write an empty stamp file for build system tracking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def strip_comments(text: str) -> str:
    """Strip // comments from text (used for .def file parsing)."""
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.split("//", 1)[0]
        lines.append(line)
    return "\n".join(lines)
