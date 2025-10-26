# Kernel Codegen

Kernel codegen transforms backend `.def` catalogs and operation schemas into generated C, Objective-C++, and header files that power the kernel query, bytecode dispatch, and API routing systems.

The DSL syntax itself is documented in `docs/kernels/DSL.md`.

---

## 1. Codegen Entry Points

Four Python scripts drive code generation. All live under `scripts/codegen/`.

### gen_kernels.py

The main codegen script. Parses all `.def` files for each backend and produces:
- The shared traits ID header (op IDs, profile IDs, qscheme IDs, kernel IDs)
- Per-backend kernel query sources (signature matching decision trees)
- Per-backend bytecode tables and exec tables (dense opcode dispatch)
- Per-backend dispatch wrapper sources (impl-level routing)
- Signature hash and operation metadata headers

### gen_api_dispatch.py

Generates API wrapper includes and operation metadata from `src/core/defs/ops.def`. Produces the 11 per-category `*_api.gen.inc` files that route public C API calls into the dispatch system, plus the auto-generated `API_OPS.gen.md` operation reference.

### gen_ops_schema.py

Generates operation schema helpers used by bytecode exec wrappers to unpack arguments.

### gen_dispatch_args.py

Generates `src/graph/kernel_dispatch_args.gen.h` from operation schema definitions. This header defines the argument structs passed through the bytecode interpreter to exec functions.

---

## 2. Inputs

### Kernel Definitions (`.def` files)

Each backend has its own set of `.def` files plus shared metadata:

- CPU kernel catalogs: `src/backends/cpu/kernels/*.def` (19 files)
- Metal kernel catalogs: `src/backends/metal/kernels/*.def` (16 files)
- CPU shared metadata: `src/backends/cpu/kernels/metadata/common.def`, `metadata/quant_schemes.def`

### Operation Schema

- `src/core/defs/ops.def` -- operation definitions consumed by `gen_api_dispatch.py`
- `scripts/codegen/ops_schema.py` -- shared operation schema helpers

### Templates

- `scripts/codegen/templates/*.j2` -- Jinja2 templates (20 files)

---

## 3. Outputs

### Shared Headers (include/marmot/)

| File | Description |
|------|-------------|
| `traits_ids.gen.h` | Op ID enums, profile ID enums, qscheme ID enums, kernel ID enums, and `*_id_to_string()` functions. Large file (~277KB) shared by all backends. |
| `op_metadata.gen.h` | Operation metadata tables (category, label, field counts). |
| `op_signature_hash.gen.h` | Signature hashing function for cache lookups. |

### API Dispatch Includes (src/core/defs/)

11 per-category generated includes that route public C API calls:

| File | Category |
|------|----------|
| `attention_api.gen.inc` | Attention operations |
| `conversion_api.gen.inc` | Dtype conversion |
| `elementwise_api.gen.inc` | Binary elementwise (add, mul, etc.) |
| `embedding_api.gen.inc` | Embedding gather |
| `matmul_api.gen.inc` | Matrix multiplication |
| `normalization_api.gen.inc` | LayerNorm, RMSNorm |
| `quantization_api.gen.inc` | Quantization/dequantization |
| `reductions_api.gen.inc` | Sum, mean, max reductions |
| `rope_api.gen.inc` | Rotary position embeddings |
| `tensor_ops_api.gen.inc` | Layout operations (reshape, transpose, etc.) |
| `unary_api.gen.inc` | Unary operations (abs, exp, etc.) |

### CPU Backend (src/backends/cpu/dispatch/)

| File | Description |
|------|-------------|
| `cpu_kernel_query.gen.c` | Signature -> kernel selection decision tree |
| `bytecode_tables_cpu.gen.h` | Opcode count, immediate sizes, schema IDs |
| `bytecode_tables_cpu.gen.c` | Dense opcode table data |
| `bytecode_exec_cpu.gen.h` | Exec function declarations |
| `bytecode_exec_cpu.gen.c` | op_index -> exec function pointer table |
| `elementwise_dispatch_cpu.gen.c` | Elementwise dispatch wrappers |
| `matmul_dispatch_cpu.gen.c` | Matmul dispatch wrappers |
| `neural_dispatch_cpu.gen.c` | Neural op dispatch wrappers |
| `reduction_dispatch_cpu.gen.c` | Reduction dispatch wrappers |
| `misc_dispatch_cpu.gen.c` | Misc op dispatch wrappers |

### Metal Backend (src/backends/metal/)

| File | Description |
|------|-------------|
| `internal/metal_kernel_query.gen.h` | Metal query function declaration |
| `internal/metal_kernel_query.gen.mm` | Signature -> kernel selection for Metal |
| `ops/bytecode_tables_metal.gen.h` | Metal opcode count, immediate sizes |
| `ops/bytecode_tables_metal.gen.mm` | Metal dense opcode table data |
| `ops/bytecode_exec_metal.gen.h` | Metal exec function declarations |
| `ops/bytecode_exec_metal.gen.mm` | Metal op_index -> exec function table (large, ~1.25MB) |
| `ops/metal_kernel_dispatch.gen.mm` | Metal kernel dispatch wrappers (large, ~390KB) |
| `ops/metal_matmul_quant_dispatch.gen.mm` | Quantized matmul dispatch for Metal |
| `ops/metal_unary_tables.gen.h` | Unary operation tables for Metal |

### Documentation

| File | Description |
|------|-------------|
| `docs/API_OPS.gen.md` | Auto-generated operation reference |

### Graph Dispatch Args

| File | Description |
|------|-------------|
| `src/graph/kernel_dispatch_args.gen.h` | Bytecode argument structs per op schema |

---

## 4. Template System

All generated files are produced via Jinja2 templates in `scripts/codegen/templates/`. Key templates:

| Template | Produces |
|----------|----------|
| `traits_ids.h.j2` | `traits_ids.gen.h` |
| `op_metadata.h.j2` | `op_metadata.gen.h` |
| `op_signature_hash.h.j2` | `op_signature_hash.gen.h` |
| `backend_query.c.j2` | `cpu_kernel_query.gen.c`, `metal_kernel_query.gen.mm` |
| `bytecode_tables.h.j2` | `bytecode_tables_*.gen.h` |
| `bytecode_tables.c.j2` | `bytecode_tables_*.gen.c/mm` |
| `bytecode_exec.h.j2` | `bytecode_exec_*.gen.h` |
| `bytecode_exec.c.j2` | `bytecode_exec_*.gen.c/mm` |
| `backend_dispatch.j2` | `metal_kernel_dispatch.gen.mm` |
| `elementwise_dispatch_cpu.c.j2` | `elementwise_dispatch_cpu.gen.c` |
| `matmul_dispatch_cpu.c.j2` | `matmul_dispatch_cpu.gen.c` |
| `neural_dispatch_cpu.c.j2` | `neural_dispatch_cpu.gen.c` |
| `reduction_dispatch_cpu.c.j2` | `reduction_dispatch_cpu.gen.c` |
| `misc_dispatch_cpu.c.j2` | `misc_dispatch_cpu.gen.c` |
| `api_dispatch.inc.j2` | `*_api.gen.inc` files |
| `api_ops_md.j2` | `API_OPS.gen.md` |
| `kernel_dispatch_args.h.j2` | `kernel_dispatch_args.gen.h` |
| `metal_quant_dispatch.gen.mm.j2` | `metal_matmul_quant_dispatch.gen.mm` |
| `metal_unary_tables.gen.h.j2` | `metal_unary_tables.gen.h` |
| `_macros.j2` | Shared Jinja2 macros |

---

## 5. Meson Build Integration

Meson runs codegen as `custom_target()` blocks in `meson.build`. The repository uses `uv` to run Python in a reproducible environment, so the build invokes codegen through `uv run python3 ...`.

There are four codegen targets:

1. **kernel_codegen** -- Runs `gen_kernels.py`. Inputs: all `.def` files + templates. Outputs: traits IDs, kernel query sources, bytecode tables/exec, dispatch wrappers.

2. **dispatch_args_codegen** -- Runs `gen_dispatch_args.py`. Output: `src/graph/kernel_dispatch_args.gen.h`.

3. **api_dispatch_codegen** -- Runs `gen_api_dispatch.py`. Inputs: `ops.def` + templates. Outputs: `*_api.gen.inc` files, `op_metadata.gen.h`, `API_OPS.gen.md`.

4. **ops_schema_codegen** -- Runs `gen_ops_schema.py`. Output: operation schema helpers.

---

## 6. How to Regenerate

Running `make build` automatically runs all codegen steps via Meson's `custom_target` dependency tracking. If a `.def` file or template changes, the affected codegen targets re-run automatically.

To force a full regeneration:

```sh
make clean-all && make build
```

When adding a new `.def` file, update `meson.build` so the codegen target includes it as an input dependency.

---

## 7. Adding or Updating a Kernel

1. Add the kernel record(s) to the appropriate backend `.def` file(s).
2. Implement the referenced symbol in the backend code (CPU C / Metal Objective-C++).
3. Ensure the op's schema helpers exist in `src/graph/kernel_dispatch_args.gen.h` (update `scripts/codegen/ops_schema.py` if needed).
4. Run a normal build (`make build`) to regenerate outputs.
5. Add or extend tests (kernel query, dispatch correctness, backend-specific tests).

---

## See Also

- `docs/kernels/DSL.md` -- `.def` file syntax
- `docs/kernels/ARCHITECTURE.md` -- Kernel selection and bytecode execution
- `docs/kernels/DISPATCH.md` -- Dispatch flow
- `docs/tutorials/ADD_KERNEL.md` -- Step-by-step kernel addition
