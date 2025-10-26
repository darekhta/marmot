# Debugging Kernel Selection and Dispatch

Tools and techniques for inspecting kernel selection, tracing execution, and diagnosing dispatch issues.

---

## Inspecting Kernel IDs

Convert numeric IDs to human-readable strings using helpers from `include/marmot/traits_ids.gen.h`:

```c
const char *op_name      = marmot_op_id_to_string(sig.op_id);
const char *profile_name = marmot_profile_id_to_string(kernel.profile_id);
const char *qscheme_name = marmot_qscheme_id_to_string(sig.qscheme_id);
const char *kernel_name  = marmot_kernel_id_to_string(kernel.kernel_id);
```

These functions return `"UNKNOWN"` for invalid IDs.

---

## Graph JSON Dump

Dump the finalized graph to JSON to see per-node signatures and selected kernels:

```c
marmot_graph_dump_json(graph, "/tmp/graph.json");
```

Or set the environment variable to auto-dump on finalization:

```bash
MARMOT_GRAPH_DUMP_JSON=1 ./my_program
```

The JSON includes:
- Node operation names and types
- Full `marmot_op_signature_t` fields per node
- Selected `kernel_id` per node
- Input/output value IDs and shapes

---

## Execution Tracing

Enable per-node execution trace output:

```bash
MARMOT_GRAPH_TRACE=1 ./my_program
```

This prints each node as it executes, including timing and backend information.

### NaN Detection

Stop execution on NaN/Inf values in outputs:

```bash
MARMOT_GRAPH_NAN_CHECK=1 ./my_program
```

---

## Routing and Backend Selection

Control which backend is used:

| Variable | Values | Effect |
|----------|--------|--------|
| `MARMOT_ROUTE` | `cpu`, `metal`, `auto` | Force backend selection |
| `MARMOT_DEBUG_ROUTING` | `1` | Log Metal routing decisions |

Example:

```bash
# Force CPU backend for all operations
MARMOT_ROUTE=cpu ./my_program

# Enable routing decision logging
MARMOT_DEBUG_ROUTING=1 ./my_program
```

---

## Metal Quantized Matmul Knobs

Additional runtime controls for quantized matmul dispatch on Metal:

| Variable | Effect |
|----------|--------|
| `MARMOT_METAL_LOG_MATMUL_QUANT=1` | Log selected dispatch routes |
| `MARMOT_METAL_FORCE_MM=1` | Force matrix-matrix path |
| `MARMOT_METAL_FORCE_MV=1` | Force matrix-vector path |

---

## Common Issues

### "Unsupported kernel" error

1. Dump the graph JSON and inspect the failing node's signature
2. Check if the operation + dtype combination exists in the `.def` files:
   - CPU: `src/backends/cpu/kernels/*.def`
   - Metal: `src/backends/metal/kernels/*.def`
3. Check [Coverage Matrix](COVERAGE.md) for known gaps
4. If the kernel should exist, verify codegen ran: `make build` (codegen is automatic)

### Wrong output values

1. Compare against golden tests: `tests/golden/`
2. Use `MARMOT_GRAPH_TRACE=1` to identify which node diverges
3. Dump intermediate tensors by inserting debug outputs in the graph
4. Check dtype mismatches (e.g., F16 accumulation losing precision)

### NaN in output

1. Enable `MARMOT_GRAPH_NAN_CHECK=1` to find the first NaN-producing node
2. Common causes: division by zero, log of negative, normalization with epsilon=0
3. Check input data validity
4. Run with UBSAN: `make build-ubsan && make test-ubsan`

### Performance regression

1. Check backend selection: `MARMOT_GRAPH_TRACE=1` shows which backend each node uses
2. Verify fusion is happening: check graph JSON for fused op names
3. Check thread count: `MARMOT_CPU_THREADS` environment variable
4. Profile with Instruments (macOS) for Metal backend

---

## Tests for Kernel Development

When modifying kernels, run these tests:

| Test | Purpose |
|------|---------|
| `test_kernel_query` | Kernel query table correctness |
| `test_bytecode_tables` | Bytecode opcode tables |
| `test_bytecode_exec_cpu` | CPU bytecode execution |
| `test_bytecode_execute` | End-to-end bytecode execution |
| `test_graph_basic` | Graph construction and execution |
| `test_graph_fusion` | Fusion pass correctness |
| `test_graph_layout_legalization` | Layout handling |
| `test_cross_backend_fusion` | Cross-backend fusion |
| `test_metal_compat` | Metal backend smoke test |
| `test_metal_quant_optimized` | Metal quantized matmul |

---

## See Also

- [Graph Debugging](../graph/DEBUGGING.md) -- Graph-level troubleshooting
- [Environment Variables](../graph/ENVIRONMENT.md) -- Full variable reference
- [Coverage Matrix](COVERAGE.md) -- Supported operations per backend
