# Kernel Cost Modeling

Marmot has the plumbing for per-kernel cost estimates (`marmot_kernel_selection_t.estimated_us`), but the generated backend queries do not currently compute them.

---

## Current Status

- Generated backend kernel queries always return `estimated_us = 0.0`:
  - CPU: `marmot_cpu_query_kernel` in `src/backends/cpu/dispatch/cpu_kernel_query.gen.c`
  - Metal: `marmot_metal_query_kernel` in `src/backends/metal/internal/metal_kernel_query.gen.mm`
- `.def` files can include `COST_MODEL`, and the parser accepts it, but `gen_kernels.py` does not use it to compute or rank costs
- CPU reports `supports_cost_modeling = true`; Metal reports `supports_cost_modeling = false`. In practice, both return `estimated_us = 0.0`

### Implications

- **Auto backend selection**: behaves as "first backend that supports all kernels wins" (totals tie at 0)
- **Graph fusion profitability**: effectively disabled, because the fusion pass requires `fused_estimated_us < unfused_estimated_us_sum` (see `src/graph/fusion_pass.cpp`). With all estimates at 0, fusion always fires when a pattern matches.

---

## What Cost Modeling Enables

Once estimates are populated, they drive:
1. **Backend selection** -- `finalize_auto` chooses the backend with lowest total estimated cost
2. **Kernel ranking** -- When multiple kernels match a signature within a backend, pick the cheapest
3. **Fusion profitability** -- Only fuse operations when the fused kernel is cheaper than the sum of unfused kernels

The inputs to a cost model are already threaded through the query API:
- `marmot_op_signature_t` (dimensions, dtypes) -- `include/marmot/graph/op_signature.h`
- `marmot_device_caps_t` (hardware capabilities, calibration) -- `include/marmot/device_caps.h`

Default capabilities are detected per backend via `marmot_backend_detect_default_caps()` in `src/core/dispatch/kernel_query.c`.

---

## Where to Implement

The codegen template initializes estimates to 0 and never updates them:
- `scripts/codegen/templates/backend_query.c.j2`

Kernel selection happens via `marmot_select_kernel_case()` in the same template. That function receives `(sig, caps, cases, case_count)`, providing two integration paths:

### Option A: Named Cost Functions

- Keep `COST_MODEL: <name>` in `.def` records
- Extend codegen to translate `<name>` into a function pointer or enum
- Compute `estimated_us` in the generated query using a backend-implemented function
- Most shape-aware approach; scales to matmul, quantized matmul, and bandwidth-bound elementwise ops

### Option B: Per-Kernel Constants

- Add a scalar field like `ESTIMATED_US` or `COST_WEIGHT` to the DSL
- Codegen emits it into the match table as a tie-breaker
- Easy to implement but not shape-aware (estimate does not change with M/N/K or element counts)

---

## Suggested Minimal Models

The goal is not perfect prediction but stable ordering.

### Elementwise (bandwidth-bound)

```
bytes = n_elements * (n_inputs + n_outputs) * sizeof(dtype)
time  = bytes / caps.mem_bw_gbps + caps.launch_overhead_us
```

### Matmul (compute-bound)

```
flops = 2 * M * N * K
time  = max(flops / peak_flops, bytes / mem_bw) + launch_overhead
```

### Quantized Matmul

Incorporate dequantization calibration fields from `marmot_device_caps_t` (e.g., `caps.calib_dequant_us_q4k`).

---

## Calibration

For measuring actual kernel costs, use `tools/marmot-bench/` (see [Benchmarking](../getting-started/BENCHMARKING.md)).

---

## See Also

- [Fusion](../graph/FUSION.md) -- How cost estimates affect fusion decisions
- [Benchmarking](../getting-started/BENCHMARKING.md) -- Measurement infrastructure
