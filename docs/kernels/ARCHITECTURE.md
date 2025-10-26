# Kernel Architecture

Marmot uses a unified bytecode pipeline for both graph execution and ad-hoc C API calls. Kernel selection happens at compile time (graph finalization or first C API invocation); execution is a dense opcode dispatch via per-backend exec tables.

---

## 1. What Is a Kernel

A **kernel** is a backend-specific implementation of an operation for a specific combination of dtypes, stride mode, profile, and optional features (epilogues, quantization, fusion flags). Each kernel is declared in a `.def` file and expanded by codegen into a concrete record with a unique `marmot_kernel_id_t`.

For example, the CPU backend might have these kernels for `add` on `FLOAT32`:

- `add_f32_accelerate` -- Apple Accelerate implementation
- `add_f32_neon` -- ARM NEON SIMD implementation
- `add_f32_avx2` -- Intel AVX2 implementation
- `add_f32_scalar` -- Portable scalar fallback

All four share the same operation semantics but differ in their implementation profile and platform requirements.

---

## 2. Core Types

### marmot_op_signature_t

Defined in `include/marmot/graph/op_signature.h`. This is the universal operation descriptor -- a normalized struct describing "what we want to run":

```
marmot_op_signature_t
    op_id            : marmot_op_id_t          -- which operation (add, matmul, layernorm, ...)
    profile_id       : marmot_profile_id_t     -- CPU profile preference (or INVALID for "any")
    matmul_layout    : marmot_matmul_layout_t  -- NN/NT/TN/TT (matmul ops only)
    input_dtype      : marmot_dtype_t          -- input tensor dtype
    weight_dtype     : marmot_dtype_t          -- weight tensor dtype
    output_dtype     : marmot_dtype_t          -- output tensor dtype
    accum_dtype      : marmot_dtype_t          -- accumulation dtype
    qscheme_id       : marmot_qscheme_id_t     -- quantization scheme (if quantized)
    quant_block      : marmot_quant_block_t    -- block_size, group_size, scale/zp dtypes
    weight_layout    : marmot_weight_layout_t  -- SEPARATE or PACKED_3MK
    stride_mode      : marmot_stride_mode_t    -- ANY, CONTIGUOUS, ROW_STRIDED, STRIDED
    epilogue_flags   : uint32_t                -- bitmask: BIAS, ACTIVATION, RESIDUAL, ROPE
    activation       : marmot_device_unary_op_t -- activation type (if epilogue includes ACTIVATION)
    variant_flags    : uint32_t                -- variant bitmask (e.g., RESIDUAL_ADD)
    dims             : marmot_op_signature_dims_t -- union of shape metadata per op category
```

### marmot_kernel_selection_t

Defined in `include/marmot/graph/kernel_selection.h`. The result of querying a backend for a kernel:

```
marmot_kernel_selection_t
    supported        : bool                    -- whether a matching kernel was found
    kernel_id        : marmot_kernel_id_t      -- stable identifier for the chosen kernel
    op_index         : uint16_t                -- dense opcode for bytecode dispatch
    estimated_us     : double                  -- estimated execution time (0.0 today)
    est_comm_us      : double                  -- estimated communication cost
    est_workspace_mb : double                  -- estimated workspace memory
    confidence       : float                   -- estimate confidence [0..1]
    fallback_reason  : const char *            -- human-readable reason if unsupported
    shardable_axes   : uint32_t                -- axes that can be sharded
    device_affinity  : uint32_t                -- device preference hint
```

### op_index (dense uint16_t)

A backend-local dense opcode assigned by codegen. Each unique kernel gets a sequential `op_index` starting from 0. This is used directly as an array index into the bytecode exec table, enabling O(1) dispatch with no hash lookups or switch statements.

`MARMOT_KERNEL_OP_INDEX_INVALID` (`UINT16_MAX`) signals "no kernel found."

### marmot_bc_selection_t

Defined in `src/core/bytecode/bytecode_compile.h`. The ad-hoc compile result for C API dispatch:

```
marmot_bc_selection_t
    supported        : bool
    op_index         : uint16_t
    resolved_sig     : marmot_op_signature_t   -- signature after fallback resolution
    reason           : const char *            -- error reason if unsupported
```

---

## 3. Where Kernels Come From

Source of truth:

- CPU `.def` files: `src/backends/cpu/kernels/*.def`
- Metal `.def` files: `src/backends/metal/kernels/*.def`
- Shared metadata: `src/backends/*/kernels/metadata/*.def`

Codegen (`scripts/codegen/gen_kernels.py`) expands these into:

- **Kernel query functions**: `marmot_cpu_query_kernel(sig, caps)` and `marmot_metal_query_kernel(sig, caps)` -- generated decision trees that match a signature to a `kernel_id` + dense `op_index`.
- **Bytecode opcode counts**: `bytecode_tables_*.gen.h` -- total number of dense opcodes per backend.
- **Bytecode exec tables**: dense `op_index` to exec function pointer, immediate sizes, and schema IDs.
- **Traits header**: `include/marmot/traits_ids.gen.h` -- all op IDs, profile IDs, qscheme IDs, and kernel IDs as enums with `*_id_to_string()` helpers.

---

## 4. Kernel Selection Flow

The flow from "I want to run an operation" to "execute a kernel" has three phases:

```
+---------------------+     +---------------------+     +---------------------+
|   Build Signature   | --> |   Kernel Selection  | --> | Bytecode Execution  |
|                     |     |                     |     |                     |
| Fill op_id, dtypes, |     | Query backend with  |     | Use op_index as     |
| stride_mode, quant, |     | signature + caps.   |     | array index into    |
| epilogue flags,     |     | Get kernel_id +     |     | exec_table[].       |
| variant flags, dims |     | dense op_index.     |     | Call fn(exec_ctx,   |
|                     |     |                     |     |   packed_args).     |
+---------------------+     +---------------------+     +---------------------+
```

### Phase 1: Build Signature

Both graph finalization and C API calls construct a `marmot_op_signature_t`. The signature normalizes the operation request into a fixed-size struct that the kernel query can match against.

### Phase 2: Kernel Selection (Query)

The signature is passed to a generated backend query function:

```
                       marmot_op_signature_t
                              |
                              v
                 +---------------------------+
                 | marmot_backend_query_     |
                 | kernel_with_fallback()    |
                 | (src/core/dispatch/       |
                 |  kernel_query.c)          |
                 +---------------------------+
                    |                    |
                    v                    v
        +-----------------+   +-----------------+
        | marmot_cpu_     |   | marmot_metal_   |
        | query_kernel()  |   | query_kernel()  |
        | (generated)     |   | (generated)     |
        +-----------------+   +-----------------+
                    |                    |
                    v                    v
            marmot_kernel_selection_t
            { supported, kernel_id, op_index, ... }
```

The generated query functions are decision trees over the signature fields: op_id, input_dtype, weight_dtype, output_dtype, stride_mode, profile_id, qscheme_id, epilogue_flags, and variant_flags. They return a `marmot_kernel_selection_t` with a dense `op_index` for bytecode dispatch and a `kernel_id` for diagnostics.

The `_with_fallback` wrapper implements variant flag fallback: if `variant_flags != 0` and no kernel matches, it retries with `variant_flags = 0`.

### Phase 3: Bytecode Execution

The `op_index` is used as a direct array index into per-backend tables:

```
marmot_bc_tables_t (from marmot_bc_get_tables(backend))
    exec_table[op_index]  -->  function pointer
    imm_size[op_index]    -->  immediate argument size in bytes
    schema_id[op_index]   -->  argument schema for unpacking
```

The bytecode interpreter (`src/core/bytecode/bytecode.c`) reads opcodes from a program buffer, indexes into `exec_table`, and calls the function pointer with the execution context and packed arguments.

---

## 5. The Dense op_index Scheme

The `op_index` is a `uint16_t` assigned sequentially by codegen to every kernel in a backend. This gives O(1) dispatch with no runtime hashing or branching:

```
op_index:  0     1     2     3     ...   N-1
           |     |     |     |           |
           v     v     v     v           v
exec:    [fn0] [fn1] [fn2] [fn3] ... [fnN-1]   (function pointers)
imm:     [s0]  [s1]  [s2]  [s3]  ... [sN-1]    (immediate sizes)
schema:  [id0] [id1] [id2] [id3] ... [idN-1]   (schema IDs)
```

Each backend has its own independent `op_index` space. The total count is available in `bytecode_tables_*.gen.h`.

The `kernel_id` (a `marmot_kernel_id_t` enum) is a separate, globally stable identifier used for logging, debugging, and human-readable output via `marmot_kernel_id_to_string()`. It is not used in the hot dispatch path.

---

## 6. Generated Entry Points

Codegen produces these stable function names:

### Kernel Query

- `marmot_cpu_query_kernel(const marmot_op_signature_t *sig, const marmot_device_caps_t *caps)`
  - Source: `src/backends/cpu/dispatch/cpu_kernel_query.gen.c`
- `marmot_metal_query_kernel(const marmot_op_signature_t *sig, const marmot_device_caps_t *caps)`
  - Source: `src/backends/metal/internal/metal_kernel_query.gen.mm`

### Backend Query Router

- `marmot_backend_query_kernel_with_fallback(backend_type, sig, caps, resolved_sig_out)`
  - Source: `src/core/dispatch/kernel_query.c`
  - Routes to the correct backend query and implements variant flag fallback.

### Bytecode Tables

- `marmot_bc_get_tables(marmot_backend_type_t backend, marmot_bc_tables_t *out)`
  - Returns the dense exec/imm/schema arrays for a backend.
  - Source: `src/core/bytecode/bytecode_compile.c`

### Bytecode Interpreter

- `marmot_bc_execute(const marmot_bc_program_t *program, const marmot_bc_exec_ctx_t *ctx, marmot_tensor_t **regs)`
  - Source: `src/core/bytecode/bytecode.c`

### Ad-hoc Dispatch

- `marmot_bc_try_execute_signature(const marmot_context_t *ctx, const marmot_op_signature_t *sig, const void *args)`
  - Source: `src/core/dispatch/dispatch_execute.c`
  - Compiles (with cache) and executes a 1-op bytecode program.

---

## 7. Backend Notes

### CPU

- Uses `PROFILES` expansion in `.def` files for per-ISA variants (ACCELERATE, NEON, AVX2, SCALAR).
- Platform guards (`MARMOT_ENABLE_NEON`, etc.) skip unavailable profiles at compile time.
- Registers `MARMOT_FUSION_RESIDUAL_ADD` as a supported `variant_flags` bit.
- Advertises `supports_cost_modeling = true`, but generated queries currently return `estimated_us = 0.0`.

### Metal

- Uses `PROFILE: NONE` for all kernels (no ISA variants; Metal shaders are uniform).
- Registers `MARMOT_FUSION_RESIDUAL_ADD` as a supported `variant_flags` bit.
- Advertises `supports_cost_modeling = false`; generated queries return `estimated_us = 0.0`.

---

## See Also

- `docs/kernels/DSL.md` -- Kernel definition language
- `docs/kernels/DISPATCH.md` -- Dispatch flow details
- `docs/kernels/CODEGEN.md` -- Code generation pipeline
- `docs/kernels/COST_MODEL.md` -- Cost estimation (future)
