# Metal Backend Performance Optimization Plan

## Current State (2026-01-25)

Note: graph execution now runs a compiled bytecode program (see `docs/BYTECODE_DISPATCH.md`). The measurements below were
captured before/around that migration and should be re-measured on the current code.

### Update: Sampling Sync Fix Landed

We reduced per-token sampling overhead by avoiding global residency sync + full queue drains on small readbacks.

- `metal_memcpy_from_device()` now syncs only the residency entry covering the requested `src` range and only blocks when
  that range is dirty. This avoids syncing unrelated private residency buffers when reading a single scalar.
- For shared-buffer reads, the Metal backend now tracks per-command-buffer serial numbers and per-pointer last-write
  serials, so a readback can wait only for the command buffer that wrote that pointer (instead of inserting a barrier that
  waits for all queued GPU work).

### Benchmark Comparison

| Metric | Marmot | llama.cpp | Gap |
|--------|--------|-----------|-----|
| Decode throughput | ~43 tok/s | ~214 tok/s | 5x |
| Time per token | ~23ms | ~4.7ms | 5x |
| Graph dispatch | ~17ms | ~2ms | 8.5x |
| Sampling | ~5-6ms | <1ms | 6x |

**Model**: TinyLlama 1.1B Q4_K_M (22 layers)
**Hardware**: Apple Silicon (Metal backend)

### Current Performance Breakdown

```
[profile step] decode total=23ms
  ├── batch=0.002ms     (batch building)
  ├── sync=0.01ms       (host sync)
  ├── embed=0.03ms      (embedding lookup)
  ├── graph=17ms        (graph dispatch)  ← Main bottleneck
  └── sample=6ms        (sampling)        ← Secondary bottleneck
```

### What's Working Well

- **Command buffer batching**: All 572 nodes commit in 1 Metal command buffer
- **Kernel selection**: Finalization selects `kernel_id` (diagnostic) and dense `op_index`; execution uses `op_index` and does no runtime kernel selection
- **Device argmax**: Works with `--temperature 0` for greedy decoding
- **Kernel performance**: Individual kernels (matmul_quant_k) match llama.cpp

---

## Root Cause Analysis

### 1. High Graph Node Count

**Current**: 572 nodes per forward pass (26 nodes/layer × 22 layers)
**llama.cpp**: ~150-200 nodes per forward pass

Per-layer breakdown:
```
rms_norm (1)
linear × 3 (Q, K, V projections)
reshape × 3
contiguous × 2
rope × 2
contiguous × 2
reshape × 3
paged_attention (1)
reshape (1)
linear (output projection)
add (residual)
rms_norm
linear × 2 (MLP gate/up)
swiglu
linear (MLP down)
add (residual)
─────────────────
Total: 26 nodes/layer
```

Note: this breakdown assumes unfused QKV + standalone RoPE/contiguous steps; if QKV projection/rope fusion is enabled the
node count can be materially lower. Re-measure and update after confirming which kernels the current graph builder emits.

**Problem**: Many reshape/contiguous nodes are metadata-only but still incur C++ dispatch overhead.

### 2. Per-Node Dispatch Overhead

**Measured**: ~30µs per node average
**Breakdown**:
- Function call chain: `execute_bound` → `marmot_bc_execute` (bytecode interpreter) → `exec_table[op_index]`
- Register accesses: `regs[u16]` for each input/output operand
- Dispatch table: bytecode `exec_table[op_index]` (switch dispatch removed)
- Metal command encoding: pipeline selection, buffer binding

Note: the large C++ `execute_node()` if/else chain still exists for node-at-a-time execution/debugging, but finalized graphs
execute the bytecode program directly.

**Impact**: 572 nodes × 30µs = 17.2ms (matches observed graph time)

### 3. GPU Sync Per Token

**Current flow**:
1. Graph execution (async) → 17ms dispatch
2. Argmax kernel (async) → <1ms
3. `marmot_tensor_data_u64()` → sync + wait to read argmax result

**Update**: the readback path no longer performs a global "sync all dirty residency buffers" when it only needs a single
result. The remaining sync cost is dominated by the actual dependency (waiting for graph + argmax completion).

**Problem**: Cannot proceed to next token until argmax result is read from GPU.

---

## Optimization Proposals

### Priority 1: Graph Fusion (Impact: ~10-12ms)

#### 1.1 Eliminate Redundant View/Reshape Nodes

**Current**: View/reshape nodes are now skipped during graph execution; alias metadata is applied in-session with caching so
it only recomputes when inputs change (pointer/shape/metadata).

**Status**: Implemented.

**Implementation (current)**:
```cpp
// Finalize: skip view/reshape nodes in bytecode
for (auto& node : impl.nodes) {
    if (node.signature.op_id == MARMOT_OP_RESHAPE ||
        node.signature.op_id == MARMOT_OP_VIEW) {
        node.skip = true;
    }
}

// Session: cache + apply aliases before execution
build_view_aliases();
apply_view_aliases(ctx);
```

**Expected savings**: ~200 nodes eliminated → 6ms saved

#### 1.2 Fuse Contiguous into Following Operations

**Current**: Explicit contiguous copy before ops that need contiguous input.

**Proposal**: Let operations handle strided inputs natively where possible:
- RoPE can work on strided tensors
- Softmax can work on strided tensors
- Matmul can handle strided inputs (already does for weight)

**Implementation**: Add strided input support to Metal kernels:
```metal
// Current: assumes contiguous
float val = input[idx];

// Proposed: handle strides
float val = input[compute_strided_offset(idx, shape, strides)];
```

**Expected savings**: ~88 contiguous nodes eliminated → 3ms saved

#### 1.3 Fused QKV Projection

**Current**: 3 separate linear ops for Q, K, V projections.

**Status**: Graph builder now selects QKV kernels when compatible (separate weights, shared `qscheme`, MHA with
`n_head == n_head_kv`, and all-or-none biases). RoPE fusion is enabled when the RoPE dimension matches `head_dim`
(even), with multi-head support via head-dim-aware QKV kernels; otherwise QKV projection runs with standalone RoPE ops.
Fused QKV weights (Phi-3) still take the matmul + split path.

**Proposal**: Single fused matmul that outputs [Q, K, V] (fused or separate outputs depending on layout):
```
input @ [Wq | Wk | Wv] → [Q | K | V]
```

**Implementation**:
- Graph builder pattern matching for QKV (prefill + decode)
- Kernel selection: `MARMOT_OP_QKV_PROJECTION` by default, `MARMOT_OP_QKV_SHARED_INPUT` when RoPE fuse is safe
- Runtime patches QKV RoPE params (positions pointer) in bytecode const pool

**Expected savings**: 2 nodes per layer × 22 layers = 44 nodes → 1.5ms saved

### Priority 2: Async Sampling (Impact: ~3-5ms)

#### 2.1 GPU-Driven Next-Token Embedding (Avoid CPU Readback on Critical Path)

**Current**: Wait for argmax → lookup embedding → proceed.

**Proposal**: Keep argmax indices on the GPU, and run embedding gather on the GPU using those indices as input. CPU can
read the chosen token ID later for output, but the next-token compute path no longer has to stall on a tiny readback.

**Implementation sketch**:
- Produce `argmax_indices` as a GPU-resident tensor.
- Add an embedding-gather path that accepts GPU-resident indices (or add a small cast kernel: `u64 -> i32`).
- Execute `argmax -> (optional cast) -> embedding_gather` in the same command-buffer batch.

**Expected impact**: removes the CPU readback stall from the critical path (especially valuable for streaming decode).

**Status**: added a dedicated `scatter_u64_to_i32` op (CPU + Metal) and a Metal kernel
(`tensor_scatter_u64_to_i32_generic`) to write argmax indices directly into `token_ids_` on the GPU for decode-only
device-argmax batches. This avoids the host token-id upload on the next step.

The Metal reduction path now also writes small argmax index outputs directly into shared buffers and uses targeted serial
waits for readback. This makes it safe to read argmax indices after submitting additional command buffers without forcing a
full queue drain.

**Related quick win**: avoid per-step token-id buffer allocations in embedding gather (today we build a fresh
`MTLBuffer` via `newBufferWithBytes` for token IDs). Reuse a small shared staging buffer per context and update it in
place, or use `setBytes` for `n == 1`.

**Status**: implemented via a `setBytes` fast path for `n == 1` plus a reusable staging buffer for `n > 1`.

#### 2.2 Double-Buffered Token Processing

**Current**: Sequential token processing with sync barrier.

**Proposal**: Use double buffering to overlap token N+1 graph with token N sampling:

```
Token N:   [====graph====][sync][sample]
Token N+1:                [====graph====][sync][sample]

Becomes:

Token N:   [====graph====][sync][sample]
Token N+1:        [====graph====]     [sync][sample]
```

**Implementation**: Requires careful buffer management and graph execution ordering.

**Expected savings**: 3-5ms overlap per token

**Status**: Implemented for Metal greedy decode (temperature 0, top-k <= 1) with 1-token lookahead. The serving engine
submits token N+1 before reading token N, double-buffers argmax outputs to avoid overwrite hazards, and drains only the
relevant command buffer when reading argmax.

### Priority 3: Reduce Interpreter Overhead (Impact: varies)

#### 3.1 Dense `op_index` exec table (done)

**Status**: implemented via bytecode.

Finalized graphs execute a bytecode program; each instruction dispatches via `exec_table[op_index]` and does not go through
the C++ per-op dispatch chain.

#### 3.2 Tighten the interpreter loop

**Current**: Every bytecode instruction decodes immediates and calls an exec function.

**Proposal**: Inline the most common operations:
- `MARMOT_OP_MATMUL` (7 per layer)
- `MARMOT_OP_ADD` (2 per layer)
- `MARMOT_OP_RMS_NORM` (2 per layer)

**Expected savings**: ~3µs per hot-path node → 1.5ms total

#### 3.3 Reduce register/binding overhead

**Current**: Bytecode exec wrappers load registers (`regs[u16]`) and may consult const-pool data for parameters.

**Proposal**: Minimize per-instruction overhead:
- keep the no-hooks fast path hot (avoid hook checks when tracing/NaN-check is disabled)
- avoid dynamic allocations inside exec wrappers
- encode frequently-used parameters in immediates instead of const-pool lookups where it reduces work

### Priority 4: Precompiled Execution (Impact: ~3-5ms)

#### 4.1 Cached Execution Plans for Decode

**Observation**: Decode always uses the same graph with same tensor shapes.

**Proposal**: Cache the entire execution plan including:
- Tensor allocations
- Binding table
- Precomputed per-op binding metadata (so the exec loop does less work per node)

**Implementation**:
```cpp
struct CachedExecutionPlan {
    std::vector<marmot_tensor_t*> bindings;
    // Note: you cannot reuse a command encoder across command buffers.
    // For Metal-side pre-encoding, prefer Indirect Command Buffers (ICB) and/or argument buffers.
    bool valid;
};

// First decode: build and cache
// Subsequent decodes: reuse cached plan
```

**Expected savings**: Skip allocation/binding → 2-3ms saved

#### 4.2 Indirect Command Buffers

**Metal feature**: ICBs allow pre-encoding commands, updating only buffer pointers.

**Proposal**: Pre-encode graph into ICB, update input/output buffers each step.

**Challenges**:
- Requires Metal 2.0+
- Buffer pointer updates still need CPU
- May not work with all kernel types

**Expected savings**: 3-5ms if applicable

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- [x] Skip view/reshape nodes in finalize + cached aliasing in session
- [x] Dense `op_index` bytecode exec table
- [x] Avoid global residency sync on small readbacks
- [x] Avoid per-step token-id buffer allocations (setBytes for `n == 1`, reuse staging buffer for `n > 1`)
- [x] Avoid RoPE position copy in fused QKV when positions are `float32`
- [ ] Profile individual node types to identify hot spots

### Phase 2: Graph Optimization (3-5 days)
- [ ] Coalesce view/reshape chains in graph builder (optional)
- [ ] Add strided input support to RoPE kernel
- [x] Enable QKV projection in graph builder (RoPE fused when `rope_dim == head_dim`, multi-head supported)

### Phase 3: Async Sampling (2-3 days)
- [ ] GPU-driven next-token embedding (argmax → embedding on GPU; scatter op added, CPU readback still on path)
- [ ] Measure end-to-end decode impact (streaming vs batched)

### Phase 4: Advanced Optimizations (5-7 days)
- [ ] Double-buffered token processing
- [ ] Cached execution plans
- [ ] Investigate indirect command buffers

---

## Success Metrics

| Milestone | Target | Current |
|-----------|--------|---------|
| Decode throughput | >100 tok/s | 43 tok/s |
| Time per token | <10ms | 23ms |
| Graph dispatch | <5ms | 17ms |
| Sampling | <2ms | 6ms |
| Parity with llama.cpp | ~200 tok/s | 43 tok/s |

---

## Appendix: Profiling Commands

```bash
# Step profiling
MARMOT_PROFILE_STEP=1 marmot-lm run model.gguf --prompt "Hello" --temperature 0

# GPU kernel profiling
MARMOT_PROFILE_GPU=1 marmot-lm run model.gguf --prompt "Hello"

# Graph trace (node-level)
MARMOT_GRAPH_TRACE=1 marmot-lm run model.gguf --prompt "Hello"

# Metal command buffer trace
MARMOT_METAL_TRACE_BATCH=1 marmot-lm run model.gguf --prompt "Hello"
```

## References

- llama.cpp Metal backend: `ggml/src/ggml-metal/`
- Apple Metal Best Practices Guide
- Metal Performance Shaders documentation
