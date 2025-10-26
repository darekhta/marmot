# Environment Variables Reference

This document lists the environment variables read by Marmot at runtime, organized by category.

To discover new variables after code changes:
```bash
rg -n 'getenv\("MARMOT_' src
rg -n 'std::getenv\("MARMOT_' src
```

---

## Graph Tracing

### `MARMOT_GRAPH_TRACE`

Enables execution tracing for debugging graph runs. Source: `src/graph/graph_executor.cpp`.

| Setting | Behavior |
|---------|----------|
| `1` | Print each executed node (op_name, op_id) and allocator deltas |
| unset | No tracing |

### `MARMOT_GRAPH_NAN_CHECK`

Enables runtime NaN/Inf checks during graph execution. Source: `src/graph/graph_executor.cpp`.

| Setting | Behavior |
|---------|----------|
| `1` | Check outputs for NaN/Inf after each node; stop on detection |
| unset | No checking |

---

## Routing and Backend Selection

### `MARMOT_ROUTING`

Controls backend selection policy for graph finalization and GGUF auto-backend builders. Source: `src/core/context/routing.c`.

| Value | Behavior |
|-------|----------|
| `auto` (default) | Try preferred backends in order |
| `cpu` or `always_cpu` | Force CPU backend |
| `gpu` or `always_gpu` | Force GPU backend (Metal when available) |

### `MARMOT_QUANT_ACT`

Controls activation quantization mode for quantized matmul. Source: `src/core/context/routing.c`.

| Value | Behavior |
|-------|----------|
| `auto` (default) | Backend chooses |
| `direct` or `fp` | Force direct (non-packed) activations |

Values `packed` and `q8` are currently treated as unsupported and fall back to `auto`.

### `MARMOT_DEBUG_ROUTING`

Enables verbose routing decision logging for the Metal backend. Source: `src/backends/metal/metal_backend.mm`.

| Setting | Behavior |
|---------|----------|
| set (any value) | Print routing decisions |
| unset | Silent |

---

## CPU Backend Tuning

### `MARMOT_NUM_THREADS`

Overrides the CPU thread count used by compute kernels. Source: `src/backends/cpu/cpu_caps.c`.

| Setting | Behavior |
|---------|----------|
| integer > 0 | Use that many threads (clamped to system limits) |
| unset | Auto-detect a suitable default |

### `MARMOT_QUANT_CPU_ACT`

CPU-specific knob to bias quantized matmul toward Q8 activation packing when `MARMOT_QUANT_ACT=auto`. Source: `src/backends/cpu/runtime/runtime.c`.

| Setting | Behavior |
|---------|----------|
| value starting with `q` or `Q` | Enable the hint |
| unset | Disabled |

### `MARMOT_EMBEDDING_PREFETCH_DISTANCE`

Controls how far ahead the CPU embedding kernel prefetches quantized weight blocks. Source: `src/backends/cpu/ops/embedding/*`.

| Setting | Behavior |
|---------|----------|
| integer 0..16 | Prefetch that many blocks ahead (clamped) |
| unset | 0 (no prefetch) |

---

## Metal Backend Tuning

These variables apply only when Marmot is built with Metal enabled (`MARMOT_ENABLE_METAL`).

### General

#### `MARMOT_PROFILE_GPU`

Enables Metal timestamp-based profiling support (when supported by the device). Source: `src/backends/metal/metal_backend.mm`.

| Setting | Behavior |
|---------|----------|
| set (any value) | Enable profiling |
| unset | Disabled |

#### `MARMOT_METAL_TRACE_BATCH`

Enables batch execution tracing for Metal command submission. Source: `src/backends/metal/metal_backend.mm`.

| Setting | Behavior |
|---------|----------|
| set (any non-`0` value) | Trace batch commits |
| unset or `0` | Disabled |

#### `MARMOT_METAL_FORCE_F32`

Disables half-precision storage optimizations in Metal. Source: `src/backends/metal/metal_backend.mm`.

| Setting | Behavior |
|---------|----------|
| set (any value) | Prefer FLOAT32 intermediate storage |
| unset | Allow FLOAT16 where beneficial |

### Packed Weights (Quantized Matmul)

#### `MARMOT_METAL_ENABLE_PACKED_WEIGHTS`

Controls whether Metal uses packed-weight caches for quantized matmul. Source: `src/backends/metal/metal_backend.mm`.

| Setting | Behavior |
|---------|----------|
| unset | Enabled (default) |
| `0` | Disabled |

#### `MARMOT_METAL_PACKED_MIN_DIM`

Minimum matrix dimension to consider packing weights.

#### `MARMOT_METAL_PACKED_MIN_ELEMENTS`

Minimum element count to consider packing weights.

#### `MARMOT_METAL_PACKED_TILE_COLS` / `MARMOT_METAL_PACKED_TILE_K`

Override packing tile parameters.

#### `MARMOT_METAL_PACKED_CACHE_LIMIT_MB`

Cache size limit for packed weights, in MiB.

| Setting | Behavior |
|---------|----------|
| `0` | Disable caching |
| integer > 0 | Set cache limit |

#### `MARMOT_METAL_QKV_FUSED_CACHE_LIMIT_MB`

Cache size limit for fused QKV weights, in MiB.

### Quantized Matmul Dispatch

#### `MARMOT_METAL_LOG_MATMUL_QUANT`

Enables logging for quantized matmul dispatch decisions. Source: `src/backends/metal/ops/matmul_quant_dispatch.mm`.

| Setting | Behavior |
|---------|----------|
| set (any non-`0` value) | Enable logging |
| unset | Silent |

#### `MARMOT_METAL_FORCE_MM` / `MARMOT_METAL_FORCE_MV`

Forces matrix-matrix or matrix-vector dispatch paths in quantized matmul. Source: `src/backends/metal/ops/matmul_quant_dispatch.mm`.

| Setting | Behavior |
|---------|----------|
| `1` | Force the specified path |
| unset | Auto-select based on dimensions |

### Attention Tuning

#### `MARMOT_METAL_DECODE_SIMD_GROUPS`

Overrides SIMD group count for decode attention. Source: `src/backends/metal/ops/attention.mm`.

| Setting | Behavior |
|---------|----------|
| `4` or `8` | Force that value |
| unset | Auto-select |

#### `MARMOT_METAL_PAGED_FLASH_VARIANT`

Selects the paged-flash attention kernel tuning mode. Source: `src/backends/metal/ops/attention.mm`.

| Setting | Behavior |
|---------|----------|
| `auto` (default) | Heuristic selection |
| `default` or `0` | Force default variant |
| `narrow` or `1` | Force narrow variant |

---

## Allocator Debugging

### `MARMOT_DEBUG_ALLOCATOR`

Enables debug logging for memory allocations. Source: `src/core/context/context.c`.

| Setting | Behavior |
|---------|----------|
| set (any value) | Log allocation and deallocation events |
| unset | Silent |

---

## Quick Reference

| Variable | Category | Purpose |
|----------|----------|---------|
| `MARMOT_GRAPH_TRACE` | Graph | Trace graph execution |
| `MARMOT_GRAPH_NAN_CHECK` | Graph | NaN/Inf detection |
| `MARMOT_ROUTING` | Routing | Backend selection policy |
| `MARMOT_QUANT_ACT` | Routing | Activation quantization mode |
| `MARMOT_DEBUG_ROUTING` | Routing | Verbose routing logs (Metal) |
| `MARMOT_NUM_THREADS` | CPU | Thread count override |
| `MARMOT_QUANT_CPU_ACT` | CPU | Quant activation packing hint |
| `MARMOT_EMBEDDING_PREFETCH_DISTANCE` | CPU | Embedding prefetch distance |
| `MARMOT_PROFILE_GPU` | Metal | GPU profiling support |
| `MARMOT_METAL_TRACE_BATCH` | Metal | Batch commit tracing |
| `MARMOT_METAL_FORCE_F32` | Metal | Force FLOAT32 storage |
| `MARMOT_METAL_ENABLE_PACKED_WEIGHTS` | Metal | Packed weight caches |
| `MARMOT_METAL_PACKED_*` | Metal | Packed weight thresholds/tiling/cache |
| `MARMOT_METAL_QKV_FUSED_CACHE_LIMIT_MB` | Metal | Fused QKV cache limit |
| `MARMOT_METAL_LOG_MATMUL_QUANT` | Metal | Quant matmul dispatch logging |
| `MARMOT_METAL_FORCE_MM` / `_MV` | Metal | Force quant matmul dispatch path |
| `MARMOT_METAL_DECODE_SIMD_GROUPS` | Metal | Decode attention SIMD tuning |
| `MARMOT_METAL_PAGED_FLASH_VARIANT` | Metal | Paged-flash attention variant |
| `MARMOT_DEBUG_ALLOCATOR` | Allocator | Allocation event logging |
