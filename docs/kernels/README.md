# Kernel System

Marmot's kernel system translates declarative `.def` definitions through a code generation pipeline into dispatch tables and bytecode interpreters. Kernel families are specified as cartesian products over operations, dtypes, and profiles using a purpose-built DSL. The codegen (`gen_kernels.py`) expands these definitions into per-backend kernel query functions, dense opcode tables, and bytecode exec functions -- enabling O(1) dispatch from an operation signature to a concrete kernel implementation at runtime.

---

## Learning Path

Start here if you are new to the kernel system:

1. **DSL.md** -- The `.def` file authoring language. Covers sets, axes, kernel families, profile expansion, template strings, and validation rules. Read this first to understand how kernels are declared.
2. **ARCHITECTURE.md** -- How signatures, kernel selection, and bytecode execution fit together. Covers `marmot_op_signature_t`, `marmot_kernel_selection_t`, the dense `op_index` scheme, and generated entry points.
3. **DISPATCH.md** -- The unified dispatch flow for both graph execution and ad-hoc C API calls. Covers the bytecode interpreter, fallback behavior, and backend-specific interpreters.
4. **CODEGEN.md** -- Code generation pipeline: entry points, inputs, outputs, Meson integration, and Jinja2 templates. Read this to understand what gets generated and how to regenerate it.
5. **docs/tutorials/ADD_KERNEL.md** -- Step-by-step tutorial for adding a new kernel end-to-end.

---

## Key Concepts Quick Reference

| Concept | Definition |
|---------|------------|
| **Kernel family** | A `KERNEL_FAMILY` block in a `.def` file that declares one or more kernels via cartesian product expansion over axes (op, dtype, profile, etc.). |
| **Profile** | A CPU implementation variant (ACCELERATE, NEON, AVX2, SCALAR). Profile expansion generates one kernel record per variant. SCALAR is the required fallback. |
| **Trait** | A generated identifier (op ID, profile ID, qscheme ID, kernel ID) in `traits_ids.gen.h`. Used for matching signatures to kernels at runtime. |
| **Signature** | `marmot_op_signature_t` -- the normalized descriptor of "what to run": op ID, dtypes, quantization, stride mode, epilogue flags, and shape metadata. |
| **Kernel selection** | `marmot_kernel_selection_t` -- the result of querying a backend: `supported`, `kernel_id`, dense `op_index`, cost estimates, and `fallback_reason`. |
| **op_index** | A dense `uint16_t` opcode, backend-local, used as an index into the bytecode exec table for O(1) dispatch. |
| **Bytecode opcode** | The encoded form of an operation in a bytecode program: `op_index` plus packed immediate arguments. Executed by the per-backend bytecode interpreter. |

---

## Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| `DSL.md` | `.def` syntax, expansion semantics, field schema, validation rules |
| `ARCHITECTURE.md` | Signatures, kernel selection, bytecode execution, generated entry points |
| `DISPATCH.md` | Unified dispatch flow, graph and C API paths, fallback behavior |
| `CODEGEN.md` | Code generation pipeline, inputs/outputs, Meson integration |
| `COST_MODEL.md` | Cost estimation for kernel selection (future) |
| `COVERAGE.md` | Op x dtype x backend support tables |
| `OPS.md` | Complete operation catalog |
| `DEBUGGING.md` | Tracing and inspection tools |

### Related Documentation

| Document | Description |
|----------|-------------|
| `docs/graph/SIGNATURES.md` | Operation signature fields |
| `docs/graph/FUSION.md` | Operation fusion system |
| `docs/graph/ENVIRONMENT.md` | Debug and tuning variables |
| `docs/tutorials/ADD_KERNEL.md` | Step-by-step kernel addition tutorial |

---

## File Structure

### Kernel Definitions (`.def` files)

```
src/backends/cpu/kernels/
    metadata/
        common.def               # Shared dtype sets, fragments
        quant_schemes.def        # Quantization scheme definitions
    elementwise.def              # Binary elementwise (add, mul, sub, ...)
    unary.def                    # Unary operations (abs, neg, exp, ...)
    ternary.def                  # Ternary operations (where, clamp, ...)
    matmul_scalar.def            # Scalar matmul kernels
    matmul_neon.def              # NEON matmul kernels
    matmul_avx2.def              # AVX2 matmul kernels
    matmul_accelerate.def        # Accelerate matmul kernels
    matmul_quantized_scalar.def  # Quantized matmul kernels
    matmul_qkv.def               # QKV projection/fusion kernels
    normalization.def            # LayerNorm, RMSNorm
    softmax.def                  # Softmax kernels
    reductions.def               # Sum, mean, max, min reductions
    embedding.def                # Embedding gather
    rope.def                     # Rotary position embeddings
    conversions.def              # Dtype conversions
    quantization.def             # Quantization/dequantization
    tensor_ops.def               # Layout ops (reshape, transpose, concat, ...)
    vec_dot.def                  # Vector dot product
    paged_attention.def          # Paged attention kernels

src/backends/metal/kernels/
    elementwise.def              # Metal elementwise kernels
    matmul.def                   # Metal matmul kernels
    matmul_quantized.def         # Metal quantized matmul
    matmul_qkv.def               # Metal QKV kernels
    normalization.def            # Metal normalization
    softmax.def                  # Metal softmax
    reductions.def               # Metal reductions
    embedding.def                # Metal embedding
    rope.def                     # Metal RoPE
    conversions.def              # Metal conversions
    quantization.def             # Metal quantization
    ternary.def                  # Metal ternary ops
    tensor_ops.def               # Metal layout ops
    activations.def              # Metal fused activations
    vec_dot.def                  # Metal vector dot product
    paged_attention.def          # Metal paged attention
```

### Generated Files

```
include/marmot/
    traits_ids.gen.h             # Op IDs, profile IDs, qscheme IDs, kernel IDs
    op_metadata.gen.h            # Operation metadata tables
    op_signature_hash.gen.h      # Signature hashing

src/core/defs/
    *_api.gen.inc                # 11 per-category API dispatch includes

src/backends/cpu/dispatch/
    cpu_kernel_query.gen.c       # Signature -> kernel selection
    bytecode_tables_cpu.gen.h    # Opcode counts, imm sizes
    bytecode_tables_cpu.gen.c    # Dense opcode tables
    bytecode_exec_cpu.gen.h      # Exec function declarations
    bytecode_exec_cpu.gen.c      # op_index -> exec function table
    elementwise_dispatch_cpu.gen.c
    matmul_dispatch_cpu.gen.c
    neural_dispatch_cpu.gen.c
    reduction_dispatch_cpu.gen.c
    misc_dispatch_cpu.gen.c

src/backends/metal/
    internal/
        metal_kernel_query.gen.h
        metal_kernel_query.gen.mm    # Metal signature -> kernel selection
    ops/
        bytecode_tables_metal.gen.h
        bytecode_tables_metal.gen.mm
        bytecode_exec_metal.gen.h
        bytecode_exec_metal.gen.mm   # Metal bytecode exec
        metal_kernel_dispatch.gen.mm # Metal kernel dispatch wrappers
        metal_matmul_quant_dispatch.gen.mm
        metal_unary_tables.gen.h

docs/
    API_OPS.gen.md               # Auto-generated operation reference
```

### Codegen Scripts

```
scripts/codegen/
    gen_kernels.py               # Main codegen: .def -> dispatch tables + bytecode
    gen_api_dispatch.py          # ops.def -> API wrapper includes + metadata
    gen_ops_schema.py            # Operation schema helpers
    gen_dispatch_args.py         # Bytecode argument structs
    def_parser.py                # .def file parser
    backend_config.py            # Per-backend codegen configuration
    codegen_base.py              # Shared codegen utilities
    templates/                   # Jinja2 templates for all generated files
```
