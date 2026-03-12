#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "compare_moe_cpu.py"
    spec = importlib.util.spec_from_file_location("compare_moe_cpu", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_and_extract() -> None:
    mod = load_module()

    marmot_rows = mod.parse_json_lines(
        '{"model":"fixture.gguf","ctx_size":4096,"batch_size":512,"n_prompt":64,"n_gen":0,'
        '"n_depth":0,"pp_tokens_sec":321.5,"tg_tokens_sec":0.0,"pp_mean_us":0.0,"tg_mean_us":0.0}\n'
        '{"model":"fixture.gguf","ctx_size":4096,"batch_size":512,"n_prompt":128,"n_gen":4,'
        '"n_depth":0,"pp_tokens_sec":0.0,"tg_tokens_sec":91.25,"pp_mean_us":0.0,"tg_mean_us":0.0}\n'
    )
    llama_rows = mod.parse_json_lines(
        '{"n_prompt":64,"n_gen":0,"avg_ts":300.0}\n'
        '{"n_prompt":128,"n_gen":4,"avg_ts":88.0}\n'
    )

    pp_case = mod.BenchCase("prefill", 64, 64)
    tg_case = mod.BenchCase("decode", 4, 128)
    pp_marmot = mod.find_record(marmot_rows, pp_case)
    tg_marmot = mod.find_record(marmot_rows, tg_case)
    pp_llama = mod.find_record(llama_rows, pp_case)
    tg_llama = mod.find_record(llama_rows, tg_case)

    assert mod.marmot_tokens_per_sec(pp_marmot, "prefill") == 321.5
    assert mod.marmot_tokens_per_sec(tg_marmot, "decode") == 91.25
    assert mod.llama_tokens_per_sec(pp_llama) == 300.0
    assert mod.llama_tokens_per_sec(tg_llama) == 88.0


def test_render_markdown() -> None:
    mod = load_module()

    result = mod.BenchResult(
        mode="cold",
        phase="decode",
        tokens=4,
        prompt_tokens=128,
        depth_tokens=128,
        marmot_tokens_per_sec=91.25,
        llama_tokens_per_sec=88.0,
        speedup_vs_llama=1.0369318181818181,
        marmot_command="marmot",
        llama_command="llama",
    )

    text = mod.render_markdown([result], Path("fixture.gguf"), 10, 3)

    assert "| cold | decode | 128 | 4 | 91.25 | 88.00 | 1.037x |" in text
    assert "Decode KV depth: `128`" in text
    assert "fixture.gguf" in text


def test_build_cases_decode_prompt() -> None:
    mod = load_module()

    cases = mod.build_cases([64], [1, 4], 128)

    assert cases == [
        mod.BenchCase("prefill", 64, 64),
        mod.BenchCase("decode", 1, 0, 128),
        mod.BenchCase("decode", 4, 0, 128),
    ]


def test_build_marmot_command_uses_direct_depth_decode() -> None:
    mod = load_module()

    cmd = mod.build_marmot_command(
        Path("/tmp/marmot-bench"),
        Path("/tmp/model.gguf"),
        mod.BenchCase("decode", 4, 0, 128),
        4096,
        512,
        8,
        3,
        "warm",
    )

    assert "--llm-mode" in cmd
    assert "direct" in cmd
    assert cmd[cmd.index("-p") + 1] == "0"
    assert cmd[cmd.index("-d") + 1] == "128"


def test_find_record_matches_depth_decode() -> None:
    mod = load_module()

    rows = mod.parse_json_lines('{"n_prompt":0,"n_gen":1,"n_depth":128,"avg_ts":500.0}\n')

    row = mod.find_record(rows, mod.BenchCase("decode", 1, 0, 128))

    assert mod.llama_tokens_per_sec(row) == 500.0


if __name__ == "__main__":
    test_parse_and_extract()
    test_render_markdown()
