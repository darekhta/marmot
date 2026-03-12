#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "benchmark_llm_matrix.py"
    spec = importlib.util.spec_from_file_location("benchmark_llm_matrix", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_json_lines() -> None:
    mod = load_module()

    rows = mod.parse_json_lines(
        '{"model":"fixture.gguf","backend":"cpu","llm_mode":"direct","ctx_size":4096,"batch_size":512,'
        '"threads":9,"n_prompt":64,"n_gen":0,"n_depth":0,"n_seqs":1,"max_seqs":1,"max_batch_seqs":1,'
        '"pp_tokens_sec":1234.5,"tg_tokens_sec":0.0,"ttft_ns":123456.0,"pp_mean_us":51.0,"tg_mean_us":0.0}\n'
    )

    assert len(rows) == 1
    assert rows[0]["backend"] == "cpu"
    assert rows[0]["threads"] == 9
    assert rows[0]["ttft_ns"] == 123456.0


def test_build_direct_cases() -> None:
    mod = load_module()

    cases = mod.build_direct_cases([64], [1, 4], [128, 1024])

    assert cases == [
        mod.MatrixCase("direct", 64, 0, 0, 1, 1, 1),
        mod.MatrixCase("direct", 0, 1, 128, 1, 1, 1),
        mod.MatrixCase("direct", 0, 4, 128, 1, 1, 1),
        mod.MatrixCase("direct", 0, 1, 1024, 1, 1, 1),
        mod.MatrixCase("direct", 0, 4, 1024, 1, 1, 1),
    ]


def test_build_serving_cases_defaults_to_concurrency() -> None:
    mod = load_module()

    cases = mod.build_serving_cases([64], [16], [0], [2, 4], [], [])

    assert cases == [
        mod.MatrixCase("serving", 64, 16, 0, 2, 2, 2),
        mod.MatrixCase("serving", 64, 16, 0, 4, 4, 4),
    ]


def test_build_command_serving_includes_scheduler_knobs() -> None:
    mod = load_module()

    cmd = mod.build_command(
        Path("/tmp/marmot-bench"),
        Path("/tmp/model.gguf"),
        "cpu",
        4096,
        512,
        9,
        3,
        1,
        False,
        mod.MatrixCase("serving", 64, 16, 0, 4, 8, 8),
    )

    assert cmd[cmd.index("--llm-mode") + 1] == "serving"
    assert cmd[cmd.index("--concurrency") + 1] == "4"
    assert cmd[cmd.index("--max-seqs") + 1] == "8"
    assert cmd[cmd.index("--max-batch-seqs") + 1] == "8"


def test_render_markdown_splits_sections() -> None:
    mod = load_module()

    results = [
        mod.MatrixResult(
            "fixture.gguf", "direct", "cpu", 9, 64, 0, 0, 1, 1, 1, 2000.0, 0.0, 1_000_000.0, 32_000.0, 0.0, "direct"
        ),
        mod.MatrixResult(
            "fixture.gguf", "serving", "cpu", 9, 64, 16, 0, 4, 8, 8, 40.0, 38.0, 12_000_000.0, 3000000.0, 800000.0,
            "serving"
        ),
    ]

    text = mod.render_markdown(results, "cpu", 9, 3, 1)

    assert "## `fixture.gguf`" in text
    assert "### Direct" in text
    assert "### Serving" in text
    assert "| prefill | 64 | 0 | 0 | 2000.00 | 0.00 | 1.000 |" in text
    assert "| 64 | 16 | 0 | 4 | 8 | 8 | 40.00 | 38.00 | 12.000 |" in text


def test_render_markdown_groups_multiple_models() -> None:
    mod = load_module()

    results = [
        mod.MatrixResult(
            "one.gguf", "direct", "cpu", 9, 64, 0, 0, 1, 1, 1, 1000.0, 0.0, 1_000_000.0, 64_000.0, 0.0, "one"
        ),
        mod.MatrixResult(
            "two.gguf", "direct", "cpu", 9, 64, 0, 0, 1, 1, 1, 2000.0, 0.0, 2_000_000.0, 32_000.0, 0.0, "two"
        ),
    ]

    text = mod.render_markdown(results, "cpu", 9, 3, 1)

    assert "## `one.gguf`" in text
    assert "## `two.gguf`" in text
    assert text.index("## `one.gguf`") < text.index("## `two.gguf`")


if __name__ == "__main__":
    test_parse_json_lines()
    test_build_direct_cases()
    test_build_serving_cases_defaults_to_concurrency()
    test_build_command_serving_includes_scheduler_knobs()
    test_render_markdown_splits_sections()
