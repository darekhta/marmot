# Graph Architecture

The graph module lives in `src/graph/` (C++ implementation) and is exposed through a C API in `include/marmot/graph/`. This document covers the internal representation, the finalization pipeline, and the execution model.

---

## Overview

```
                         Graph Lifecycle
  ================================================================

  BUILD PHASE                          COMPILE PHASE
  ─────────────                        ──────────────
  ┌──────────┐   ┌──────────────┐     ┌──────────────────────────┐
  │  Create   │──>│  Add Inputs  │──>  │  Finalize (4 passes)     │
  │  Graph    │   │  & Constants │     │                          │
  └──────────┘   └──────┬───────┘     │  1. Populate signatures  │
                        │             │  2. Fusion detection      │
                        v             │  3. Kernel selection      │
                 ┌──────────────┐     │  4. Bytecode compilation  │
                 │  Add Ops     │──>  └────────────┬─────────────┘
                 │  (build DAG) │                   │
                 └──────────────┘                   v

  EXECUTION PHASE
  ───────────────
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ Bind Inputs  │──>│  Execute     │──>│ Read Outputs │
  │ & Outputs    │   │  (bytecode)  │   │              │
  └──────────────┘   └──────────────┘   └──────┬───────┘
                                                │
                                                v
                                        ┌──────────────┐
                                        │  Destroy     │
                                        │  Graph       │
                                        └──────────────┘
```

---

## Graph Representation (SSA IR)

The internal storage uses a simple SSA-style intermediate representation with three core structures:

**GraphValue** (`src/graph/graph_value.hpp`)
- Tensor descriptor (dtype, shape, strides) plus def/use metadata.
- Optionally holds a constant tensor for weight data.

**GraphNode** (`src/graph/graph_node.hpp`)
- Operation name, `marmot_op_signature_t`, input/output value IDs.
- After finalization: `kernel_id` (diagnostic), `bc_op_index` (bytecode dispatch index), `estimated_us` (cost estimate).

**Graph::Impl** (`src/graph/graph_impl.hpp`)
- `values[]` -- all graph values, appended during construction.
- `nodes[]` -- all graph nodes, appended during construction.
- `plan[]` -- linear list of `ExecutionCommand` entries after finalization.

Construction rules enforced in `Graph::add_op` (`src/graph/graph.cpp`):
- Values are append-only; each output value has exactly one defining node.
- Inputs must already exist and must be defined before use.
- Construction order determines execution order (no topological reordering).

---

## Finalization Pipeline

Finalization transforms an unfinalized graph into an executable bytecode program. It runs four passes in sequence.

### Pass 1: Populate Signatures

Implementation: `populate_signature` in `src/graph/graph_signature.cpp`.

Fills missing fields in each node's `marmot_op_signature_t`:
- Infers `input_dtype`, `output_dtype`, `accum_dtype` from tensor descriptors.
- Computes dimension fields from tensor shapes (e.g., `dims.matmul.{N,K,M}` for matmul, `dims.elementwise.n_elems` for elementwise ops).
- Infers `matmul_layout` (NN/NT) from tensor shapes for matmul and linear operations.

### Pass 2: Fusion Detection

Implementation: `Graph::apply_fusion_pass` in `src/graph/passes/fusion_pass.cpp`.

Scans adjacent nodes for fusible patterns and rewrites them into single fused operations:
- Only considers adjacent nodes in build order.
- Only fuses through intermediates with a single use (preserves sharing semantics).
- Uses backend kernel query and cost estimate for profitability.

Current status: generated backend queries return `estimated_us = 0.0`, so profitability checks effectively disable automatic fusion. See [FUSION.md](FUSION.md) for supported patterns and details.

### Pass 3: Kernel Selection

Implementation: `query_backend_for_node` in `src/graph/kernel_query.cpp`.

Queries the chosen backend for each node:
- Calls `marmot_backend_query_kernel(backend, sig, caps)` (`src/core/dispatch/kernel_query.c`).
- Dispatches to backend-specific generated queries (e.g., `marmot_cpu_query_kernel`, `marmot_metal_query_kernel`).
- If a kernel is not found and `variant_flags != 0`, retries with `variant_flags = 0` as a fallback.
- Records `kernel_id` on the node for diagnostic purposes.

### Pass 4: Bytecode Compilation

Compiles the finalized node list into a bytecode program:
- Translates each node's signature into a `bc_op_index` via `marmot_bc_compile_signature_with_caps(...)`.
- Encodes immediates and a constant pool into `marmot_bc_program_t`.
- Stores `bc_op_index` and `estimated_us` on each node as metadata.
- `kernel_id` is retained only for JSON dumps and debugging; execution uses `bc_op_index`.

**Auto backend selection** (`Graph::finalize_auto_with_policy` in `src/graph/graph.cpp`) tries candidate backends and selects the first that can finalize the entire graph. Because cost estimates are currently all zero, the selection is order-based rather than cost-based.

---

## Bytecode Execution

Execution uses a cached session that owns persistent intermediates and KV caches.

**Session management**: `Graph::execute` (`src/graph/graph_executor.cpp`) creates or reuses an `ExecutionSession` (`src/graph/execution_session.cpp`). The session:
- Binds constant tensors from the graph.
- Allocates intermediate tensors.
- Uploads buffers for non-CPU backends.

**Dispatch**: The `Executor` (`src/graph/graph_executor.cpp`) runs the compiled bytecode program via `marmot_bc_execute(...)`. When tracing, NaN checking, or RoPE fixups are enabled, it uses `marmot_bc_execute_with_hooks(...)` instead.

### Paged Attention and KV Pool

`paged_attention` nodes consume packed token metadata and KV pool tensors:
- The serving layer owns the `marmot_kv_pool_t` and provides `kv_k`, `kv_v`, and `block_table` inputs.
- Per-step KV writes happen inside paged attention kernels using the supplied block table.

### View and Reshape Caching

`reshape` and `view` operations execute as metadata updates without kernel launches when possible:
- `reshape` shares the input's data pointer after verifying element counts match.
- `view` adds a byte offset into the input buffer (`GraphNode.view_byte_offset`).

The GGUF builder uses these for splitting fused tensors. The public C API does not currently expose per-node view/slice parameters.

---

## Multi-Backend Support

The graph module supports multiple backends through the vtable-based device interface:
- **CPU backend**: SIMD-optimized kernels (NEON, AVX2, Accelerate) with scalar fallbacks.
- **Metal backend**: GPU compute shaders for Apple Silicon.

Backend selection happens at finalization time. The `MARMOT_ROUTING` environment variable or explicit `marmot_backend_type_t` parameter controls which backend is used. See [ENVIRONMENT.md](ENVIRONMENT.md) for routing options.

---

## See Also

- [FUSION.md](FUSION.md) -- Operation fusion patterns and detection
- [API.md](API.md) -- Public C API reference
- [DEBUGGING.md](DEBUGGING.md) -- Debug tools and environment variables
- [Kernel Dispatch](../kernels/DISPATCH.md) -- How kernels are selected and dispatched
