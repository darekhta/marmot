#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class MatrixCase:
    llm_mode: str
    n_prompt: int
    n_gen: int
    n_depth: int
    concurrency: int
    max_seqs: int
    max_batch_seqs: int


@dataclass(frozen=True)
class MatrixResult:
    model: str
    llm_mode: str
    backend: str
    threads: int
    n_prompt: int
    n_gen: int
    n_depth: int
    concurrency: int
    max_seqs: int
    max_batch_seqs: int
    pp_tokens_per_sec: float
    tg_tokens_per_sec: float
    ttft_ns: float
    pp_mean_us: float
    tg_mean_us: float
    command: str


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_marmot_bench() -> Path:
    return repo_root() / "build-release" / "tools" / "marmot-bench" / "marmot-bench"


def default_model() -> Path:
    return repo_root() / "tests" / "fixtures" / "gguf" / "multiarch" / "qwen3moe-30b-a3b-1layer-q4km.gguf"


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0:
            raise ValueError(f"negative value is invalid: {value}")
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


def format_command(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: list[str], timeout: int, verbose: bool) -> dict[str, object]:
    if verbose:
        print(f"$ {format_command(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"exit code {proc.returncode}"
        raise RuntimeError(f"command failed: {format_command(cmd)}\n{detail}")
    rows = parse_json_lines(proc.stdout)
    if len(rows) != 1:
        raise RuntimeError(f"expected exactly one JSONL row from: {format_command(cmd)}")
    return rows[0]


def build_command(
    marmot_bench: Path,
    model: Path,
    backend: str,
    ctx_size: int,
    batch_size: int,
    threads: int,
    repetitions: int,
    warmup: int,
    flash_attn: bool,
    case: MatrixCase,
) -> list[str]:
    cmd = [
        str(marmot_bench),
        "-m",
        str(model),
        "--llm-mode",
        case.llm_mode,
        "-b",
        backend,
        "-C",
        str(ctx_size),
        "-B",
        str(batch_size),
        "-t",
        str(threads),
        "--concurrency",
        str(case.concurrency),
        "--max-seqs",
        str(case.max_seqs),
        "--max-batch-seqs",
        str(case.max_batch_seqs),
        "--flash-attn",
        "1" if flash_attn else "0",
        "-p",
        str(case.n_prompt),
        "-g",
        str(case.n_gen),
        "-d",
        str(case.n_depth),
        "-r",
        str(repetitions),
        "-w",
        str(warmup),
        "-F",
        "jsonl",
    ]
    return cmd


def benchmark_case(
    marmot_bench: Path,
    model: Path,
    backend: str,
    ctx_size: int,
    batch_size: int,
    threads: int,
    repetitions: int,
    warmup: int,
    flash_attn: bool,
    timeout: int,
    verbose: bool,
    case: MatrixCase,
) -> MatrixResult:
    cmd = build_command(
        marmot_bench, model, backend, ctx_size, batch_size, threads, repetitions, warmup, flash_attn, case
    )
    row = run_command(cmd, timeout, verbose)
    return MatrixResult(
        model=str(model),
        llm_mode=str(row["llm_mode"]),
        backend=str(row["backend"]),
        threads=int(row["threads"]),
        n_prompt=int(row["n_prompt"]),
        n_gen=int(row["n_gen"]),
        n_depth=int(row["n_depth"]),
        concurrency=int(row["n_seqs"]),
        max_seqs=int(row["max_seqs"]),
        max_batch_seqs=int(row["max_batch_seqs"]),
        pp_tokens_per_sec=float(row["pp_tokens_sec"]),
        tg_tokens_per_sec=float(row["tg_tokens_sec"]),
        ttft_ns=float(row.get("ttft_ns", 0.0)),
        pp_mean_us=float(row["pp_mean_us"]),
        tg_mean_us=float(row["tg_mean_us"]),
        command=format_command(cmd),
    )


def build_direct_cases(prefill: list[int], decode: list[int], depths: list[int]) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for prompt in prefill:
        cases.append(
            MatrixCase(
                llm_mode="direct",
                n_prompt=prompt,
                n_gen=0,
                n_depth=0,
                concurrency=1,
                max_seqs=1,
                max_batch_seqs=1,
            )
        )
    for depth in depths:
        for tokens in decode:
            cases.append(
                MatrixCase(
                    llm_mode="direct",
                    n_prompt=0,
                    n_gen=tokens,
                    n_depth=depth,
                    concurrency=1,
                    max_seqs=1,
                    max_batch_seqs=1,
                )
            )
    return cases


def build_serving_cases(
    prompts: list[int],
    gens: list[int],
    depths: list[int],
    concurrencies: list[int],
    max_seqs_list: list[int],
    max_batch_seqs_list: list[int],
) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for concurrency in concurrencies:
        max_seqs_candidates = max_seqs_list if max_seqs_list else [concurrency]
        max_batch_candidates = max_batch_seqs_list if max_batch_seqs_list else [concurrency]
        for prompt in prompts:
            for gen in gens:
                for depth in depths:
                    for max_seqs in max_seqs_candidates:
                        for max_batch in max_batch_candidates:
                            if max_seqs < concurrency or max_batch < concurrency:
                                continue
                            cases.append(
                                MatrixCase(
                                    llm_mode="serving",
                                    n_prompt=prompt,
                                    n_gen=gen,
                                    n_depth=depth,
                                    concurrency=concurrency,
                                    max_seqs=max_seqs,
                                    max_batch_seqs=max_batch,
                                )
                            )
    return cases


def render_markdown(
    results: list[MatrixResult],
    backend: str,
    threads: int,
    repetitions: int,
    warmup: int,
) -> str:
    lines = [
        "# LLM Benchmark Matrix",
        "",
        f"- Backend: `{backend}`",
        f"- Threads: `{threads}`",
        f"- Repetitions: `{repetitions}`",
        f"- Warmup runs: `{warmup}`",
        "",
    ]

    model_order: list[str] = []
    for result in results:
        if result.model not in model_order:
            model_order.append(result.model)

    for model in model_order:
        lines.extend([f"## `{Path(model).name}`", ""])

        direct = [r for r in results if r.model == model and r.llm_mode == "direct"]
        serving = [r for r in results if r.model == model and r.llm_mode == "serving"]

        if direct:
            lines.extend(
                [
                    "### Direct",
                    "",
                    "| phase | prompt | gen | depth | pp t/s | tg t/s | ttft ms |",
                    "|-------|--------|-----|-------|--------|--------|---------|",
                ]
            )
            for result in direct:
                phase = "prefill" if result.n_prompt > 0 and result.n_gen == 0 else "decode"
                lines.append(
                    f"| {phase} | {result.n_prompt} | {result.n_gen} | {result.n_depth} | "
                    f"{result.pp_tokens_per_sec:.2f} | {result.tg_tokens_per_sec:.2f} | "
                    f"{result.ttft_ns / 1e6:.3f} |"
                )
            lines.append("")

        if serving:
            lines.extend(
                [
                    "### Serving",
                    "",
                    "| prompt | gen | depth | concurrency | max seqs | max batch seqs | pp t/s | tg t/s | ttft ms |",
                    "|--------|-----|-------|-------------|----------|----------------|--------|--------|---------|",
                ]
            )
            for result in serving:
                lines.append(
                    f"| {result.n_prompt} | {result.n_gen} | {result.n_depth} | {result.concurrency} | "
                    f"{result.max_seqs} | {result.max_batch_seqs} | {result.pp_tokens_per_sec:.2f} | "
                    f"{result.tg_tokens_per_sec:.2f} | {result.ttft_ns / 1e6:.3f} |"
                )
            lines.append("")

        lines.append("")

    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct and serving LLM benchmark sweeps with marmot-bench.")
    parser.add_argument("--model", type=Path, action="append", dest="models")
    parser.add_argument("--marmot-bench", type=Path, default=default_marmot_bench())
    parser.add_argument("--backend", choices=["cpu", "metal"], default="cpu")
    parser.add_argument("--modes", choices=["direct", "serving", "both"], default="both")
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=9)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--flash-attn", action="store_true")
    parser.add_argument("--direct-prefill", default="64,256,512")
    parser.add_argument("--direct-decode", default="1,4,16")
    parser.add_argument("--direct-depths", default="128")
    parser.add_argument("--serving-prompts", default="64")
    parser.add_argument("--serving-gen", default="16")
    parser.add_argument("--serving-depths", default="0")
    parser.add_argument("--serving-concurrency", default="1,2,4")
    parser.add_argument("--serving-max-seqs", default="")
    parser.add_argument("--serving-max-batch-seqs", default="")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.models:
        args.models = [default_model()]
    for model in args.models:
        if not model.is_file():
            raise FileNotFoundError(f"model not found: {model}")
    if not args.marmot_bench.is_file():
        raise FileNotFoundError(f"marmot-bench not found: {args.marmot_bench}")
    if args.ctx_size <= 0 or args.batch_size <= 0 or args.threads <= 0 or args.repetitions <= 0:
        raise ValueError("ctx-size, batch-size, threads, and repetitions must be > 0")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")


def main() -> int:
    args = parse_args()
    validate_args(args)

    results: list[MatrixResult] = []
    for model in args.models:
        if args.modes in ("direct", "both"):
            for case in build_direct_cases(
                parse_int_list(args.direct_prefill), parse_int_list(args.direct_decode), parse_int_list(args.direct_depths)
            ):
                results.append(
                    benchmark_case(
                        args.marmot_bench,
                        model,
                        args.backend,
                        args.ctx_size,
                        args.batch_size,
                        args.threads,
                        args.repetitions,
                        args.warmup,
                        args.flash_attn,
                        args.timeout,
                        args.verbose,
                        case,
                    )
                )

        if args.modes in ("serving", "both"):
            for case in build_serving_cases(
                parse_int_list(args.serving_prompts),
                parse_int_list(args.serving_gen),
                parse_int_list(args.serving_depths),
                parse_int_list(args.serving_concurrency),
                parse_int_list(args.serving_max_seqs),
                parse_int_list(args.serving_max_batch_seqs),
            ):
                results.append(
                    benchmark_case(
                        args.marmot_bench,
                        model,
                        args.backend,
                        args.ctx_size,
                        args.batch_size,
                        args.threads,
                        args.repetitions,
                        args.warmup,
                        args.flash_attn,
                        args.timeout,
                        args.verbose,
                        case,
                    )
                )

    markdown = render_markdown(results, args.backend, args.threads, args.repetitions, args.warmup)
    if args.markdown_out is not None:
        write_text(args.markdown_out, markdown)
    else:
        sys.stdout.write(markdown)

    if args.json_out is not None:
        payload = {
            "models": [str(model) for model in args.models],
            "backend": args.backend,
            "threads": args.threads,
            "repetitions": args.repetitions,
            "warmup": args.warmup,
            "results": [asdict(result) for result in results],
        }
        write_text(args.json_out, json.dumps(payload, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
