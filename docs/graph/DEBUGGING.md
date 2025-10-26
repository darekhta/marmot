# Debugging Graph Execution

This document covers diagnostic tools, common issues, step-by-step troubleshooting workflows, and sanitizer usage for Marmot graph execution.

---

## Diagnostic Tools

### JSON Graph Dump

`marmot_graph_dump_json(graph, path)` writes a JSON representation of the graph containing:
- Backend type.
- Node list with op names, signatures, selected `kernel_id`, and `kernel_name`.
- Value list with tensor descriptors.

The `kernel_id` in JSON output is diagnostic only. Runtime execution dispatches through the compiled bytecode using `bc_op_index`, not `kernel_id`.

This is the most useful tool for verifying what the builder produced and what finalization selected.

```c
marmot_graph_dump_json(graph, "/tmp/graph.json");
```

### Execution Tracing

Set `MARMOT_GRAPH_TRACE=1` to print each executed node (op_name, op_id) and allocator deltas after execution.

```bash
MARMOT_GRAPH_TRACE=1 ./my_app
```

### NaN/Inf Detection

Set `MARMOT_GRAPH_NAN_CHECK=1` to check outputs for NaN/Inf after each node. Execution stops on first detection, printing min/max values and NaN/Inf counts to help isolate the divergence point.

```bash
MARMOT_GRAPH_NAN_CHECK=1 ./my_app
```

### Routing Diagnostics

Set `MARMOT_DEBUG_ROUTING=1` to print backend selection decisions. Force a specific backend with `MARMOT_ROUTING`:

```bash
MARMOT_ROUTING=cpu ./my_app    # Force CPU
MARMOT_ROUTING=gpu ./my_app    # Force GPU (Metal)
```

### Debug Environment Summary

| Variable | Purpose |
|----------|---------|
| `MARMOT_GRAPH_TRACE=1` | Print each executed node |
| `MARMOT_GRAPH_NAN_CHECK=1` | Detect NaN/Inf in outputs |
| `MARMOT_ROUTING=cpu` | Force CPU backend |
| `MARMOT_ROUTING=gpu` | Force GPU backend |
| `MARMOT_DEBUG_ROUTING=1` | Print routing decisions |
| `MARMOT_DEBUG_ALLOCATOR=1` | Print allocation events |

See [ENVIRONMENT.md](ENVIRONMENT.md) for the complete list.

---

## Common Issues

### "Kernel not supported" Error

**Symptoms**: `marmot_graph_finalize()` returns `MARMOT_ERROR_NOT_IMPLEMENTED` with a detail message like "Kernel missing for op ...".

**Diagnosis**:

1. Print the error detail:
   ```c
   fprintf(stderr, "detail: %s\n", marmot_get_last_error_detail());
   ```

2. Dump the graph to inspect shapes and dtypes:
   ```c
   marmot_graph_dump_json(graph, "/tmp/debug.json");
   ```

3. Check the signature fields in the JSON output:
   - Is `op_id` correct for the intended operation?
   - Do `input_dtype` and `output_dtype` match supported types?
   - Is `stride_mode` compatible with the tensor layout?
   - For quantized ops, is `qscheme_id` supported by the target backend?

4. Consult the coverage matrix in `docs/kernels/COVERAGE.md` for supported op/dtype/backend combinations.

**Common causes**:
- Using an unsupported dtype (e.g., FP8 for operations that only support F32/F16).
- Quantized matmul with an unsupported qscheme on the target backend.
- Attempting to use the Metal backend on non-Apple hardware.

---

### NaN in Graph Output

**Symptoms**: Output tensors contain NaN or Inf values.

**Step-by-step workflow**:

1. Enable NaN checking to find the first divergent node:
   ```bash
   MARMOT_GRAPH_NAN_CHECK=1 ./my_app
   ```

2. Combine with tracing for execution context:
   ```bash
   MARMOT_GRAPH_TRACE=1 MARMOT_GRAPH_NAN_CHECK=1 ./my_app
   ```

3. Dump the graph to examine the failing node's signature:
   ```c
   marmot_graph_dump_json(graph, "/tmp/nan_debug.json");
   ```

4. Isolate the problematic node by building a minimal graph with just that operation and testing with known-good inputs.

**Common causes**:
- Division by zero in div or mod operations.
- `exp()` with large positive inputs producing Inf.
- `log()` with zero or negative inputs producing NaN.
- Uninitialized input tensors.
- Accumulator overflow in reductions (check `accum_dtype`).

---

### Wrong Output Values

**Symptoms**: Results are numerically incorrect but not NaN.

**Step-by-step workflow**:

1. Compare CPU and GPU results to isolate backend-specific issues:
   ```bash
   MARMOT_ROUTING=cpu ./my_app > cpu_output.txt
   MARMOT_ROUTING=gpu ./my_app > gpu_output.txt
   diff cpu_output.txt gpu_output.txt
   ```

2. Verify tensor layouts:
   - Are input tensors contiguous when the kernel requires it?
   - Do shapes match the expected dimensions?
   - Are strides correct?

3. Check dtype consistency:
   - Mixed precision can cause unexpected truncation.
   - Verify `accum_dtype` for reductions and matmul.

4. Test with simple inputs (e.g., all 1.0) for easy manual verification.

**Common causes**:
- Incorrect tensor dimensions (M, N, K for matmul).
- Transposition mismatch (NT vs. NN layout).
- Broadcasting issues.
- Quantization scale or zero-point errors.

---

### Performance Regression

**Symptoms**: Execution is slower than expected.

**Step-by-step workflow**:

1. Enable routing diagnostics to verify backend selection:
   ```bash
   MARMOT_DEBUG_ROUTING=1 ./my_app
   ```

2. Dump graph JSON to inspect selected kernels:
   ```c
   marmot_graph_dump_json(graph, "/tmp/perf_debug.json");
   ```
   Note: `estimated_us` is currently `0.0` for generated matches (cost modeling is not yet wired).

3. Check for missing fusion. Graph fusion is guarded by profitability checks using `estimated_us`. If all costs are `0.0`, fusions will not apply automatically, and intermediate tensors will be materialized unnecessarily.

4. Profile individual operations with stable timing (multiple iterations). Use actual wall-clock measurements rather than `estimated_us`.

**Common causes**:
- CPU selected when GPU would be faster (or vice versa).
- Missing fusion causing intermediate tensor materialization.
- Contiguous copy overhead from non-contiguous input tensors.
- Memory allocation overhead (consider tensor caching).

---

## Sanitizer Usage

Marmot provides build targets for three sanitizers. Use these when investigating memory corruption, undefined behavior, or threading issues.

### AddressSanitizer (ASan)

Detects out-of-bounds access, use-after-free, double-free, and memory leaks.

```bash
make build-asan
make test-asan
```

### UndefinedBehaviorSanitizer (UBSan)

Detects signed integer overflow, null pointer dereference, misaligned access, and other undefined behavior.

```bash
make build-ubsan
make test-ubsan
```

### ThreadSanitizer (TSan)

Detects data races and other threading issues.

```bash
make build-tsan
make test-tsan
```

### Recommended Workflow

For memory-sensitive changes (allocator, buffer management, tensor lifecycle):
1. Build and test with ASan first (`make build-asan && make test-asan`).
2. Follow up with UBSan for undefined behavior checks.
3. Use TSan when modifying concurrent code paths.

---

## See Also

- [ENVIRONMENT.md](ENVIRONMENT.md) -- Complete environment variable reference
- [ARCHITECTURE.md](ARCHITECTURE.md) -- Graph internals and execution model
- [Kernel Coverage](../kernels/COVERAGE.md) -- Supported operations by backend and dtype
