# Benchmarking (`marmot-bench`)

Marmot ships a benchmark runner under `tools/marmot-bench/` for measuring kernel and graph performance on CPU and Metal.

The binaries are built automatically by `make build`:
- `build-debug/tools/marmot-bench/marmot-bench`
- `build-debug/tools/marmot-bench/marmot-bench-tile` (tile-focused variants)

---

## Basic Usage

```bash
build-debug/tools/marmot-bench/marmot-bench --help
```

### Common Options

| Flag | Description |
|------|-------------|
| `-b, --backend <cpu\|metal\|compare>` | Run on one backend or compare both |
| `-c, --category <micro\|composite\|all>` | Workload category |
| `-f, --filter <pattern>` | Substring filter on workload name |
| `-r, --repetitions <n>` | Number of repetitions per workload |
| `-w, --warmup <n>` | Warmup iterations before measurement |
| `-n, --iterations <n>` | Iterations per repetition |
| `-T, --min-time <sec>` | Minimum time per measurement |
| `-F, --format <fmt>` | Output format (json, csv, md, jsonl, sql) |
| `-o, --output <path>` | Write results to file |

### LLM Modes

`marmot-bench -m ...` supports two distinct LLM benchmark modes:

- `--llm-mode direct`  
  Lower-level packed-graph benchmarking. This is the default and is intended to match `llama.cpp`'s `llama-bench` methodology as closely as possible:
  - random token sequences
  - empty-context starts unless `-d/--n-depth` is used
  - warmup run(s) discarded from measurement
  - generation excludes sampling and measures model execution only

- `--llm-mode serving`  
  End-to-end serving-engine benchmarking. This includes request submission, scheduler behavior, KV-pool orchestration, and sampling callbacks. Use this to measure Marmot serving/runtime overhead, not to compare directly against `llama-bench`.

For LLM mode:
- `-w/--warmup` sets warmup runs
- `--no-warmup` forces zero warmup runs
- steady-state decode should be modeled with `-d/--n-depth`, not `-p/--n-prompt`

### Metal Calibration Overrides

| Flag | Description |
|------|-------------|
| `--metal-peak-fp32 <tflops>` | Override Metal FP32 peak TFLOPS |
| `--metal-peak-fp16 <tflops>` | Override Metal FP16 peak TFLOPS |
| `--metal-mem-bw <gbps>` | Override Metal memory bandwidth (GB/s) |
| `--metal-launch-us <usec>` | Override Metal kernel launch overhead |
| `--metal-edge-alpha <alpha>` | Override Metal edge alpha parameter |

---

## Workload Categories

### Micro Workloads

Individual kernel benchmarks:
- **matmul** -- Dense matrix multiplication at various sizes
- **matmul_quant** -- Quantized matmul (Q4K, Q6K, Q8)
- **fusion** -- Fused vs unfused operation pairs (e.g., mul+add vs fma)
- **rope** -- Rotary position embedding
- **reductions** -- Sum, mean, max across dimensions
- **elementwise** -- Binary and unary operations

### Composite Workloads

Multi-operation sequences:
- **ffn** -- Feed-forward network block (matmul + activation + matmul)
- **layer** -- Full transformer layer

---

## Examples

```bash
# CPU micro benchmarks, FMA operations only
build-debug/tools/marmot-bench/marmot-bench -b cpu -c micro -f fma_ -r 3 -w 3 -n 50 -T 0.2

# Unfused baseline for comparison
build-debug/tools/marmot-bench/marmot-bench -b cpu -c micro -f mul_add_ -r 3 -w 3 -n 50 -T 0.2

# Compare CPU vs Metal
build-debug/tools/marmot-bench/marmot-bench -b compare -c micro -f fma_

# Save results as JSON
build-debug/tools/marmot-bench/marmot-bench -b cpu -c micro -f fma_ -F json -o /tmp/bench.json

# llama-bench-style prefill throughput
build-release/tools/marmot-bench/marmot-bench \
  -m model.gguf \
  --llm-mode direct \
  -b cpu \
  -p 512 \
  -g 0 \
  -d 0 \
  -t 8 \
  -r 5 \
  -F jsonl

# steady-state decode throughput at KV depth 128
build-release/tools/marmot-bench/marmot-bench \
  -m model.gguf \
  --llm-mode direct \
  -b cpu \
  -p 0 \
  -g 4 \
  -d 128 \
  -t 8 \
  -r 5 \
  -F jsonl

# serving-engine benchmark for the same model
build-release/tools/marmot-bench/marmot-bench \
  -m model.gguf \
  --llm-mode serving \
  -b cpu \
  -p 512 \
  -g 128 \
  -t 8 \
  -r 5 \
  -F jsonl
```

## CPU MoE Comparison

For CPU MoE tuning, Marmot also ships a comparison script against `llama.cpp`'s `llama-bench`.

Prerequisites:
- `make build-release`
- `llama-bench` installed and on `PATH`
- a real MoE GGUF fixture, for example `tests/fixtures/gguf/multiarch/qwen3moe-30b-a3b-1layer-q4km.gguf`

Example:

```bash
python3 scripts/compare_moe_cpu.py \
  --model tests/fixtures/gguf/multiarch/qwen3moe-30b-a3b-1layer-q4km.gguf \
  --threads 10 \
  --prefill 64 \
  --decode 1,4 \
  --decode-depth 128 \
  --repetitions 3 \
  --json-out /tmp/marmot-moe-cpu.json
```

The script runs:
- `marmot-bench --llm-mode direct` in CPU-only LLM mode for prompt-only and depth-conditioned decode-only cases
- `llama-bench` in CPU-only mode on the same GGUF

The markdown output reports:
- prompt processing tokens/sec
- steady-state decode tokens/sec at the requested KV depth
- Marmot / `llama.cpp` speed ratio per case

Use this when tuning the CPU MoE path so every optimization has a reproducible external baseline, not only Marmot-internal microbenchmarks.

## LLM Matrix Sweeps

For broader LLM benchmarking, Marmot also ships a matrix runner for `marmot-bench`:

```bash
python3 scripts/benchmark_llm_matrix.py \
  --model tests/fixtures/gguf/multiarch/qwen3moe-30b-a3b-1layer-q4km.gguf \
  --model tests/fixtures/gguf/multiarch/qwen3moe-30b-a3b-3layer-q4km.gguf \
  --backend cpu \
  --threads 9 \
  --repetitions 3 \
  --modes both \
  --direct-prefill 64,256,512 \
  --direct-decode 1,4,16 \
  --direct-depths 128,1024 \
  --serving-prompts 64 \
  --serving-gen 16 \
  --serving-concurrency 1,2,4 \
  --markdown-out /tmp/llm-matrix.md \
  --json-out /tmp/llm-matrix.json
```

This is intentionally split into two tracks:

- `direct`
  Use this for kernel/runtime parity and architecture work. It runs `marmot-bench --llm-mode direct` and is the right layer to compare against `llama-bench`.
- `serving`
  Use this for scheduler, paged-KV, and continuous-batching behavior. This is where Marmot’s serving architecture can matter; it should not be treated as directly comparable to `llama-bench`.

The matrix runner records:
- backend
- llm mode
- prompt/gen/depth
- concurrency
- `max_seqs`
- `max_batch_seqs`
- prompt throughput
- generation throughput
- TTFT

Recommended usage:
- tune backend kernels with `compare_moe_cpu.py` and the `direct` matrix
- evaluate paged attention and batching with the `serving` matrix
- do not mix the two when reasoning about parity with `llama.cpp`
- pass `--model` more than once to compare fixtures with different layer counts in one run

---

## Tips for Reproducible Results

- Use a fixed `-T/--min-time` for consistent measurement duration
- Pin thread counts when comparing runs (`-t/--threads` for LLM mode)
- Keep the build type consistent (debug vs release) across comparisons
- Use `make build-release` for performance-representative numbers
- Close other applications to reduce noise on shared hardware

---

## Fusion Micro-Baseline

The `micro` category includes paired workloads for fusion analysis:

| Unfused Workload | Fused Workload | Pattern |
|-----------------|----------------|---------|
| `mul_add_f32_1M` | `fma_f32_1M` | mul + add -> fma |
| `mul_add_f32_16M` | `fma_f32_16M` | mul + add -> fma |

---

## See Also

- [Cost Model](../kernels/COST_MODEL.md) -- How benchmark data can feed kernel cost estimates
- [Metal Performance](../METAL_PERFORMANCE_OPTIMIZATION.md) -- GPU-specific optimization
