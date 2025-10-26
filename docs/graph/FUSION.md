# Operation Fusion

This document describes Marmot's graph-level fusion pass: a rewrite that replaces certain adjacent operation sequences with a single fused operation when the backend has a matching kernel and the fusion is estimated to be profitable.

---

## Where Fusion Runs

Fusion runs as the second pass of graph finalization:
- `Graph::finalize_impl()` (`src/graph/graph.cpp`) calls `Graph::apply_fusion_pass()`.
- The pass implementation lives in `src/graph/passes/fusion_pass.cpp`.

The pass is intentionally local:
- Only considers adjacent nodes in build order.
- Only fuses through intermediates with a single use (to avoid changing sharing semantics).
- Uses backend kernel query and cost estimate to decide whether to apply the rewrite.

---

## Profitability

The pass queries the backend for cost estimates of both the unfused sequence and the candidate fused operation. Fusion is applied only when the fused kernel is supported and estimated to be faster.

**Current status**: Generated backend queries return `estimated_us = 0.0` for all matches. Because the profitability check compares fused cost against unfused cost, and `0.0 >= 0.0` does not pass the "strictly cheaper" threshold, automatic fusion is effectively disabled. This will change once cost modeling is wired in. See `docs/kernels/COST_MODEL.md` for planned work.

---

## Supported Fusion Patterns

### Elementwise + Activation

| Pattern | Fused Op ID |
|---------|-------------|
| `add` then `relu` | `MARMOT_OP_ADD_RELU` |
| `add` then `gelu` | `MARMOT_OP_ADD_GELU` |
| `add` then `silu` | `MARMOT_OP_ADD_SILU` |
| `mul` then `add` | `MARMOT_OP_FMA` |

### Gated Activations

| Pattern | Fused Op ID |
|---------|-------------|
| `mul(silu(x), y)` | `MARMOT_OP_SWIGLU` (built explicitly by model builders) |
| `mul(gelu(x), y)` | `MARMOT_OP_GEGLU` (built explicitly by model builders) |

### Matmul + Bias

| Pattern | Fused Op ID |
|---------|-------------|
| `matmul` then `add` | `MARMOT_OP_MATMUL_BIAS` |

### Matmul + Bias + Activation

| Pattern | Fused Op ID |
|---------|-------------|
| `matmul` then `add` then `relu` | `MARMOT_OP_MATMUL_BIAS_RELU` |
| `matmul` then `add` then `gelu` | `MARMOT_OP_MATMUL_BIAS_GELU` |
| `matmul` then `add` then `silu` | `MARMOT_OP_MATMUL_BIAS_SILU` |

---

## Automatic vs. Explicit Fusion

**Automatic fusion** is performed by the fusion pass during finalization. It detects patterns from generic node sequences and rewrites them using the mechanism described above.

**Explicit fusion** is used by model builders (such as the GGUF graph builder) that construct fused operations directly rather than relying on pattern detection. Examples include QKV projection variants and gated activation patterns like SwiGLU and GeGLU.

---

## Interaction with Bytecode

Graph fusion rewrites the `marmot_op_signature_t.op_id` on the affected nodes. After fusion, the bytecode compilation pass (pass 4) compiles the fused signature into a single `bc_op_index`. At runtime, the bytecode executor dispatches the fused operation as one step rather than multiple.

### `variant_flags`

`marmot_op_signature_t.variant_flags` is a separate mechanism from graph fusion. It is a bitmask of `MARMOT_FUSION_*` flags used to select backend kernel variants (e.g., normalization kernels that include a residual add). It does not control `add` then `relu` or `matmul` then `add` style fusions, which are represented via fused `op_id` values.

---

## How the Pass Works

The fusion pass uses two components:

1. **Pattern identification**: `marmot_detect_fused_op_id()` in `src/core/dispatch/fusion_detection.c` maps a small op neighborhood to a candidate fused `op_id`.

2. **Profitability check**: The pass queries the backend for the cost of the unfused nodes (summed) and the candidate fused node. If the fused kernel is unsupported or not estimated to be cheaper, the rewrite is skipped.

---

## Debugging Fusion

1. **Dump graph JSON** to inspect which operations were fused:
   ```c
   marmot_graph_dump_json(graph, "/tmp/graph.json");
   ```
   Look for fused `op_id` values and `kernel_id` assignments.

2. **Trace execution** to see whether fused operations appear:
   ```bash
   MARMOT_GRAPH_TRACE=1 ./my_program
   ```
   Fused nodes appear as a single execution step (e.g., `matmul_bias_relu`) instead of multiple separate nodes.

If all cost estimates are `0.0`, fusion will not apply automatically and you will see unfused nodes in the trace output.

---

## Adding a New Fusion Pattern

1. Define a fused op (new `op_id`) and add kernels for it in the CPU and/or Metal `.def` files with implementations.
2. Teach `marmot_detect_fused_op_id()` how to map the unfused sequence to the fused op ID.
3. Update `Graph::apply_fusion_pass()` if the rewrite needs to splice extra inputs (e.g., bias tensor) or adjust signature fields.
4. Add tests to verify that the fused op is selected when profitable and that correctness matches the unfused sequence.

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) -- Finalization pipeline overview
- [SIGNATURES.md](SIGNATURES.md) -- `marmot_op_signature_t` field reference
- [DEBUGGING.md](DEBUGGING.md) -- Debug tools and environment variables
