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
```

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
