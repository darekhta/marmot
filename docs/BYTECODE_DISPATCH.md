# Bytecode Dispatch

Marmot uses a bytecode-driven dispatch system for all tensor operation execution. This document is the definitive specification of the architecture, encoding format, compilation flow, and per-backend interpreters.

## Motivation

Traditional dispatch approaches -- vtable calls, signature-based runtime matching, giant `switch` statements on kernel IDs -- impose per-operation CPU overhead that dominates execution time for graphs with hundreds of small operations. In a 500-600 operation LLM graph, the dispatch overhead alone could reach 15-20ms per forward pass.

Bytecode dispatch eliminates this overhead by moving all kernel selection and argument resolution to compile time and reducing runtime to a single, predictable loop: decode opcode, call function pointer.

## Architecture

The system has three stages: compilation, encoding, and execution.

```
                    .def files
                        |
                        v
                  scripts/codegen
                   /          \
                  v            v
         Backend compiler    Backend executor
         (sig -> op_index)   (exec_table[op_index])
                  \            /
                   v          v
           marmot_bc_program_t
            (bytecode + const pool)
                      |
                      v
            Bytecode interpreter
```

### Design principles

1. **Dense opcodes.** Execution uses `op_index` in a dense range `[0, N)`. The opcode space is not indexed by sparse `marmot_kernel_id_t` values.
2. **Single runtime dispatch.** The only per-operation dynamic dispatch on the hot path is `exec_table[op_index](...)`.
3. **No per-operation argument packing.** Tensor operands are referenced by register slots. Scalar parameters and metadata are encoded as immediates in the bytecode stream.
4. **Shared by graph and ad-hoc calls.** Both graph execution (many ops compiled once, executed many times) and non-graph C API calls (single op, compile-on-cache-miss) use the same machinery.

## Glossary

| Term | Definition |
|------|------------|
| **Signature** | `marmot_op_signature_t` -- op ID plus traits (dtypes, layouts, flags, quantization scheme) |
| **Compile** | Map a signature to a dense `op_index` and encode immediates into the program |
| **Execute** | Run bytecode by decoding opcodes and invoking `exec_table[op_index]` |
| **Register / slot** | `uint16_t` index into a `marmot_tensor_t *regs[]` array |
| **Constant pool** | Byte array storing variable-sized constants (arrays, descriptors) referenced by offset |
| **Schema** | `marmot_bc_schema_id_t` -- describes the immediate layout for a class of operations |

## Program Structure

### marmot_bc_program_t

Every compiled unit (graph or single-op call) is represented as a `marmot_bc_program_t`:

```c
typedef struct marmot_bc_program {
    uint8_t *code;              // bytecode stream (little-endian)
    size_t code_size;
    uint8_t *const_pool;        // constants blob
    size_t const_pool_size;
    uint16_t reg_count;         // number of tensor register slots
    uint16_t op_count;          // total dense opcode count for the backend
    const uint16_t *imm_size;   // imm_size[op_index] -> immediate byte count
    const marmot_bc_exec_fn *exec_table;  // exec_table[op_index] -> function pointer
} marmot_bc_program_t;
```

The `code` field contains a sequence of instructions terminated by `MARMOT_BC_END` (opcode 0). The `const_pool` holds variable-length data (permutation arrays, shape descriptors) referenced by offset from the instruction immediates.

### Instruction encoding

Each instruction is:

```
u16 op_index
u8  imm[imm_size[op_index]]
```

The `op_index` selects both the execution function and the immediate byte count. The immediate payload is schema-specific and generated from `.def` files and op schemas.

### Operand conventions

- Tensor operands are `u16` slot indices into the `regs[]` array.
- Small scalar parameters (`f32 eps`, `i32 axis`, `u32 head_dim`) are embedded inline as immediates.
- Variable-length data (permutation arrays, start/size arrays, larger descriptors) is stored in the constant pool. The instruction embeds a `u32 const_offset` (byte offset into `const_pool`) and a length/count where needed.

### Schemas

Each operation class has a schema ID (`marmot_bc_schema_id_t`) that describes its immediate layout. Current schemas include:

- `MARMOT_BC_SCHEMA_UNARY` -- one input register, one output register
- `MARMOT_BC_SCHEMA_BINARY` -- two input registers, one output register
- `MARMOT_BC_SCHEMA_TERNARY` -- three input registers, one output register
- `MARMOT_BC_SCHEMA_MATMUL` -- input/weight/output registers plus dimension parameters
- `MARMOT_BC_SCHEMA_QKV` -- fused QKV projection registers and parameters
- `MARMOT_BC_SCHEMA_PAGED_ATTENTION` -- registers for Q/K/V caches, block tables, parameters
- `MARMOT_BC_SCHEMA_ROPE` -- registers plus RoPE configuration parameters
- `MARMOT_BC_SCHEMA_REDUCTION` -- input/output registers plus axis and element count
- `MARMOT_BC_SCHEMA_SOFTMAX` -- input/output registers plus softmax parameters
- `MARMOT_BC_SCHEMA_LAYERNORM` -- input/output/weight/bias registers plus epsilon
- `MARMOT_BC_SCHEMA_RMS_NORM` -- input/output/weight registers plus epsilon
- `MARMOT_BC_SCHEMA_RESHAPE`, `MARMOT_BC_SCHEMA_VIEW`, `MARMOT_BC_SCHEMA_CONTIGUOUS`, `MARMOT_BC_SCHEMA_TRANSPOSE`, `MARMOT_BC_SCHEMA_CONCAT`, `MARMOT_BC_SCHEMA_SLICE`, `MARMOT_BC_SCHEMA_GATHER_ROWS` -- tensor manipulation schemas
- `MARMOT_BC_SCHEMA_QUANTIZE`, `MARMOT_BC_SCHEMA_DEQUANTIZE`, `MARMOT_BC_SCHEMA_COMPUTE_QPARAMS` -- quantization schemas
- `MARMOT_BC_SCHEMA_EMBEDDING` -- embedding gather registers and parameters
- `MARMOT_BC_SCHEMA_CONVERT` -- type conversion registers
- `MARMOT_BC_SCHEMA_VEC_DOT` -- vector dot product registers

### Endianness and alignment

Bytecode is little-endian. The decoder must not assume alignment; all multi-byte reads use `memcpy` to avoid undefined behavior:

```c
static inline uint16_t bc_read_u16(const uint8_t **pc) {
    uint16_t v;
    memcpy(&v, *pc, sizeof(v));
    *pc += sizeof(v);
    return v;
}
```

## Compilation Flow

Compilation maps operation signatures to dense opcodes and encodes immediates. It happens at two points:

1. **Graph finalization.** All nodes in the graph are compiled into a single multi-instruction program.
2. **Ad-hoc API calls.** On cache miss, a single-instruction program is compiled and cached for the signature.

### Graph finalization

During `marmot_graph_finalize` (or `marmot_graph_finalize_auto`):

1. Assign `u16` register slots for all tensor values.
2. For each node in execution order:
   - Build or retrieve the operation signature.
   - Call the backend compiler to obtain a dense `op_index`.
   - Emit the instruction: `emit_u16(op_index)` followed by schema-specific immediates (register slots, scalar parameters, constant pool references).
3. Emit `MARMOT_BC_END`.

### Ad-hoc API calls

Each public C API function (`marmot_matmul`, `marmot_relu`, etc.) compiles and executes through the same bytecode machinery:

1. Build a signature from the call arguments.
2. Look up the compile cache (key: `hash(signature)` + backend + device capabilities).
3. On cache miss: compile the signature to obtain `op_index`, then cache the result.
4. Create a minimal program with one instruction plus `MARMOT_BC_END`.
5. Populate `regs[]` with the actual tensor pointers.
6. Execute via the interpreter.

This is how the old C-API vtable dispatch and universal dispatch were replaced.

### Backend compiler interface

The compiler returns a selection result that includes the dense opcode:

```c
typedef struct {
    bool supported;
    uint16_t op_index;
    marmot_op_signature_t resolved_sig;
    const char *reason;
} marmot_bc_selection_t;
```

The `op_index` is consumed directly by the bytecode encoder. The `kernel_id` is retained only for diagnostic purposes (graph JSON dumps, error messages).

## Execution Model

### Exec function signature

Each entry in the backend exec table has this signature:

```c
typedef marmot_error_t (*marmot_bc_exec_fn)(
    const void *backend_exec_ctx,
    marmot_tensor_t **regs,
    const uint8_t *imm,
    const uint8_t *const_pool
);
```

The function reads its tensor operands from `regs[]` using slot indices decoded from `imm`, reads any scalar parameters from `imm`, and reads variable-length data from `const_pool` at offsets decoded from `imm`.

### Interpreter loop

The core interpreter is a tight loop with no kernel selection, no vtable indirection, and no argument packing:

```c
marmot_error_t marmot_bc_execute(
    const marmot_bc_program_t *p,
    const void *backend_exec_ctx,
    marmot_tensor_t **regs
) {
    const uint8_t *pc = p->code;
    for (;;) {
        uint16_t op = bc_read_u16(&pc);
        if (op == MARMOT_BC_END) return MARMOT_SUCCESS;
        const uint8_t *imm = pc;
        pc += p->imm_size[op];
        marmot_error_t st = p->exec_table[op](backend_exec_ctx, regs, imm, p->const_pool);
        if (st != MARMOT_SUCCESS) return st;
    }
}
```

### Hooks

The interpreter supports optional hooks for debugging and instrumentation via `marmot_bc_execute_with_hooks`. Hook callbacks are provided through a `marmot_bc_hooks_t` structure:

- `on_start` -- called before the first instruction
- `before_op` -- called before each instruction with opcode, immediates, and register state
- `after_op` -- called after each instruction with the return status
- `on_finish` -- called after the last instruction or on error

This is used for graph tracing (`MARMOT_GRAPH_TRACE=1`), NaN checking (`MARMOT_GRAPH_NAN_CHECK=1`), and RoPE binding fixups.

## Per-Backend Interpreters

### CPU

The CPU backend uses the shared interpreter directly. The `backend_exec_ctx` is the CPU device context. Each exec table entry calls the appropriate kernel function (NEON, AVX2, Accelerate, or scalar) that was selected at compile time. No additional dispatch occurs at runtime.

### Metal

The Metal backend also uses the shared interpreter, but the `backend_exec_ctx` wraps a `metal_context_t` that includes the active command buffer and compute encoder. Metal exec functions encode GPU work into the current command buffer without per-node commits.

Graph execution uses command buffer batching:
- `marmot_graph_batch_begin(ctx)` starts a command batch.
- All bytecode instructions encode into the same command buffer.
- `marmot_graph_batch_end(ctx, commit)` commits the batch.

This ensures one command buffer commit per graph execution rather than one per operation.

## Generated Outputs

Code generation from `.def` files produces the following per-backend artifacts:

| Backend | File | Contents |
|---------|------|----------|
| CPU | `src/backends/cpu/dispatch/cpu_kernel_query.gen.c` | Signature-to-kernel selection (returns `kernel_id` + `op_index`) |
| CPU | `src/backends/cpu/dispatch/bytecode_exec_cpu.gen.c` | `exec_table[]`, `imm_size[]`, `schema_id[]` |
| CPU | `src/backends/cpu/dispatch/bytecode_tables_cpu.gen.h` | Opcode counts, invalid markers |
| Metal | `src/backends/metal/internal/metal_kernel_query.gen.mm` | Signature-to-kernel selection |
| Metal | `src/backends/metal/ops/bytecode_exec_metal.gen.mm` | `exec_table[]`, `imm_size[]`, `schema_id[]` |
| Metal | `src/backends/metal/ops/bytecode_tables_metal.gen.h` | Opcode counts, invalid markers |

## Implementation Status

All five implementation phases are complete.

### Phase 1: Backend op_index + exec tables (DONE)

Introduced dense opcodes per backend. Generated `exec_table[]` arrays. The compiler returns `op_index` directly from kernel query.

### Phase 2: Bytecode runtime (DONE)

Implemented `marmot_bc_program_t`, the builder API, constant pool, and the interpreter. Added tests for bytecode tables and execution.

### Phase 3: Graph integration (DONE)

Graph finalization compiles all nodes to a bytecode program. Graph execution runs the program through the interpreter. Correctness validated for both CPU and Metal graphs in CI.

### Phase 4: Ad-hoc API integration (DONE)

All C API functions (`marmot_matmul`, `marmot_relu`, etc.) route through bytecode selection and execution. Universal dispatch and per-backend query-and-dispatch removed.

### Phase 5: Cleanup (DONE)

Legacy dispatch files and generated artifacts removed. Build glue for the old dispatch path removed. The vtable remains only for lifecycle/memory/batching operations; all op dispatch is bytecode-driven.

## Performance Characteristics

The bytecode approach reduces per-operation dispatch overhead from approximately 30 microseconds (legacy path with argument packing, signature matching, and kernel-ID switching) to low single-digit microseconds (opcode decode plus indexed function call).

For a 500-600 operation LLM graph, this translates from approximately 17ms of CPU dispatch overhead to approximately 1ms, leaving the GPU or SIMD kernel execution as the dominant cost.

Exact gains depend on:
- How many metadata-only operations (reshape, view) are eliminated during compilation.
- How much work moves from execution to compilation.
- How small each per-operation backend executor becomes.

## Optional Future Work: Metal ICB Capture

Indirect Command Buffers (ICBs) can be layered on top of bytecode execution as a second-stage optimization. Prerequisites include stable resource bindings (or argument buffers), stable dispatch sizes (or indirect dispatch), and isolation of per-token varying work outside the captured ICB. This is not currently implemented.

## Source Files

| File | Purpose |
|------|---------|
| `src/core/bytecode/bytecode.h` | Program, builder, interpreter, and hook type definitions |
| `src/core/bytecode/bytecode.c` | Builder and interpreter implementation |
| `src/core/bytecode/bytecode_compile.c` | Signature-to-opcode compilation |
| `scripts/codegen/gen_kernels.py` | Code generation from `.def` files |
| `scripts/codegen/templates/bytecode_exec.c.j2` | Jinja2 template for exec table generation |
