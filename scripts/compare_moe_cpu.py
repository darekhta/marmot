#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchCase:
    phase: str
    tokens: int
    prompt_tokens: int
    depth_tokens: int = 0


@dataclass(frozen=True)
class BenchResult:
    mode: str
    phase: str
    tokens: int
    prompt_tokens: int
    depth_tokens: int
    marmot_tokens_per_sec: float
    llama_tokens_per_sec: float
    speedup_vs_llama: float
    marmot_command: str
    llama_command: str


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_marmot_bench() -> Path:
    return repo_root() / "build-release" / "tools" / "marmot-bench" / "marmot-bench"


def default_model() -> Path:
    return repo_root() / "tests" / "fixtures" / "gguf" / "multiarch" / "qwen3moe-30b-a3b-1layer-q4km.gguf"


def default_threads() -> int:
    return max(1, os.cpu_count() or 1)


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        value = int(part.strip())
        if value < 0:
            raise ValueError(f"negative token count is invalid: {value}")
        values.append(value)
    return values


def parse_json_lines(stdout: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        rows.append(json.loads(line))
    return rows


def find_record(rows: list[dict[str, object]], case: BenchCase, allow_promptless_decode: bool = False) -> dict[str, object]:
    for row in rows:
        n_prompt = int(row.get("n_prompt", 0))
        n_gen = int(row.get("n_gen", 0))
        n_depth = int(row.get("n_depth", 0))
        if case.phase == "prefill" and n_prompt == case.prompt_tokens and n_gen == 0 and n_depth == case.depth_tokens:
            return row
        if (
            case.phase == "decode" and
            (n_prompt == case.prompt_tokens or (allow_promptless_decode and n_prompt == 0)) and n_gen == case.tokens and
            n_depth == case.depth_tokens
        ):
            return row
    raise ValueError(
        f"missing {case.phase} result for prompt={case.prompt_tokens} tokens={case.tokens} depth={case.depth_tokens}"
    )


def marmot_tokens_per_sec(row: dict[str, object], phase: str) -> float:
    key = "pp_tokens_sec" if phase == "prefill" else "tg_tokens_sec"
    return float(row[key])


def llama_tokens_per_sec(row: dict[str, object]) -> float:
    return float(row["avg_ts"])


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: list[str], timeout: int, verbose: bool) -> tuple[list[dict[str, object]], str]:
    if verbose:
        print(f"$ {format_command(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout or f"exit code {proc.returncode}"
        raise RuntimeError(f"command failed: {format_command(cmd)}\n{detail}")
    return parse_json_lines(proc.stdout), proc.stderr


def build_marmot_command(
    marmot_bench: Path, model: Path, case: BenchCase, ctx_size: int, batch_size: int, threads: int, repetitions: int,
    mode: str
) -> list[str]:
    prompt = case.prompt_tokens if case.phase == "prefill" else 0
    gen = 0 if case.phase == "prefill" else case.tokens
    cmd = [
        str(marmot_bench),
        "-m",
        str(model),
        "-b",
        "cpu",
        "--llm-mode",
        "direct",
        "-C",
        str(ctx_size),
        "-B",
        str(batch_size),
        "-t",
        str(threads),
        "--concurrency",
        "1",
        "--max-seqs",
        "1",
        "--max-batch-seqs",
        "1",
        "--flash-attn",
        "0",
        "-p",
        str(prompt),
        "-g",
        str(gen),
        "-d",
        str(case.depth_tokens),
        "-r",
        str(repetitions),
        "-F",
        "jsonl",
    ]
    if mode == "cold":
        cmd.append("--no-warmup")
    return cmd


def build_llama_command(
    llama_bench: str, model: Path, case: BenchCase, batch_size: int, ubatch_size: int, threads: int, repetitions: int,
    mode: str
) -> list[str]:
    prompt = case.prompt_tokens if case.phase == "prefill" else 0
    gen = 0 if case.phase == "prefill" else case.tokens
    cmd = [
        llama_bench,
        "-m",
        str(model),
        "-p",
        str(prompt),
        "-n",
        str(gen),
        "-d",
        str(case.depth_tokens),
        "-b",
        str(batch_size),
        "-ub",
        str(ubatch_size),
        "-t",
        str(threads),
        "-r",
        str(repetitions),
        "-ngl",
        "0",
        "-ncmoe",
        "999",
        "-fa",
        "0",
        "-o",
        "jsonl",
    ]
    if mode == "cold":
        cmd.append("--no-warmup")
    return cmd


def benchmark_case(
    case: BenchCase, marmot_bench: Path, llama_bench: str, model: Path, ctx_size: int, batch_size: int, ubatch_size: int,
    threads: int, repetitions: int, timeout: int, verbose: bool, mode: str
) -> BenchResult:
    marmot_cmd = build_marmot_command(
        marmot_bench, model, case, ctx_size, batch_size, threads, repetitions, mode
    )
    llama_cmd = build_llama_command(
        llama_bench, model, case, batch_size, ubatch_size, threads, repetitions, mode
    )

    marmot_rows, _ = run_command(marmot_cmd, timeout, verbose)
    llama_rows, _ = run_command(llama_cmd, timeout, verbose)

    marmot_row = find_record(marmot_rows, case)
    llama_row = find_record(llama_rows, case)

    marmot_tps = marmot_tokens_per_sec(marmot_row, case.phase)
    llama_tps = llama_tokens_per_sec(llama_row)
    speedup = marmot_tps / llama_tps if llama_tps > 0.0 else 0.0

    return BenchResult(
        mode=mode,
        phase=case.phase,
        tokens=case.tokens,
        prompt_tokens=case.prompt_tokens,
        depth_tokens=case.depth_tokens,
        marmot_tokens_per_sec=marmot_tps,
        llama_tokens_per_sec=llama_tps,
        speedup_vs_llama=speedup,
        marmot_command=format_command(marmot_cmd),
        llama_command=format_command(llama_cmd),
    )


def render_markdown(results: list[BenchResult], model: Path, threads: int, repetitions: int) -> str:
    decode_depths = sorted({result.depth_tokens for result in results if result.phase == "decode"})
    lines = [
        "# CPU MoE benchmark comparison",
        "",
        f"- Model: `{model}`",
        f"- Threads: `{threads}`",
        f"- Repetitions: `{repetitions}`",
        f"- Decode KV depth: `{','.join(str(depth) for depth in decode_depths) if decode_depths else '0'}`",
        "",
        "| mode | phase | depth | tokens | Marmot t/s | llama.cpp t/s | Marmot / llama.cpp |",
        "|------|-------|-------|--------|------------|----------------|--------------------|",
    ]
    for result in results:
        depth = result.depth_tokens if result.phase == "decode" else result.prompt_tokens
        lines.append(
            f"| {result.mode} | {result.phase} | {depth} | {result.tokens} | "
            f"{result.marmot_tokens_per_sec:.2f} | "
            f"{result.llama_tokens_per_sec:.2f} | {result.speedup_vs_llama:.3f}x |"
        )
    return "\n".join(lines) + "\n"


def build_cases(prefill_tokens: list[int], decode_tokens: list[int], decode_prompt: int) -> list[BenchCase]:
    cases: list[BenchCase] = []
    cases.extend(BenchCase("prefill", tokens, tokens) for tokens in prefill_tokens)
    cases.extend(BenchCase("decode", tokens, 0, decode_prompt) for tokens in decode_tokens)
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Marmot CPU MoE throughput against llama.cpp.")
    parser.add_argument("--model", type=Path, default=default_model())
    parser.add_argument("--marmot-bench", type=Path, default=default_marmot_bench())
    parser.add_argument("--llama-bench", default=shutil.which("llama-bench") or "/opt/homebrew/bin/llama-bench")
    parser.add_argument("--prefill", default="64")
    parser.add_argument("--decode", default="1,4")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--ubatch-size", type=int, default=512)
    parser.add_argument("--decode-depth", "--decode-prompt", dest="decode_depth", type=int, default=0)
    parser.add_argument("--threads", type=int, default=default_threads())
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--mode", choices=["cold", "warm", "both"], default="both")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise FileNotFoundError(f"model not found: {args.model}")
    if not args.marmot_bench.is_file():
        raise FileNotFoundError(f"marmot-bench not found: {args.marmot_bench}")
    if shutil.which(args.llama_bench) is None and not Path(args.llama_bench).is_file():
        raise FileNotFoundError(f"llama-bench not found: {args.llama_bench}")
    if args.threads <= 0:
        raise ValueError("threads must be > 0")
    if args.repetitions <= 0:
        raise ValueError("repetitions must be > 0")
    if args.batch_size <= 0 or args.ubatch_size <= 0 or args.ctx_size <= 0:
        raise ValueError("ctx-size, batch-size, and ubatch-size must be > 0")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    validate_args(args)

    prefill_tokens = parse_int_list(args.prefill)
    decode_tokens = parse_int_list(args.decode)
    cases = build_cases(prefill_tokens, decode_tokens, args.decode_depth)
    modes = [args.mode] if args.mode != "both" else ["cold", "warm"]

    results: list[BenchResult] = []
    for mode in modes:
        for case in cases:
            results.append(
                benchmark_case(
                    case,
                    args.marmot_bench,
                    args.llama_bench,
                    args.model,
                    args.ctx_size,
                    args.batch_size,
                    args.ubatch_size,
                    args.threads,
                    args.repetitions,
                    args.timeout,
                    args.verbose,
                    mode,
                )
            )

    markdown = render_markdown(results, args.model, args.threads, args.repetitions)
    if args.markdown_out is not None:
        write_text(args.markdown_out, markdown)
    else:
        sys.stdout.write(markdown)

    if args.json_out is not None:
        payload = {
            "model": str(args.model),
            "threads": args.threads,
            "repetitions": args.repetitions,
            "results": [asdict(result) for result in results],
        }
        write_text(args.json_out, json.dumps(payload, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
