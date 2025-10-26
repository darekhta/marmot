# Tutorial: Adding a CPU Kernel

This tutorial covers two scenarios: adding a new architecture-specific profile for an existing operation, and adding a completely new operation. Both workflows use Marmot's kernel definition system (`.def` files) and code generation pipeline.

## Background

Marmot's kernel system works as follows:

1. **Kernel definitions** in `.def` files describe kernel families, their axes (op, dtype), and available profiles (ACCELERATE, NEON, AVX2, SCALAR).
2. **Code generation** (`scripts/codegen/gen_kernels.py`) reads the `.def` files and produces dispatch tables, bytecode exec tables, and kernel query tables.
3. **Profile selection** happens at compile time (graph finalization or ad-hoc call caching). The runtime picks the best available profile for the current hardware.

`.def` files live in `src/backends/cpu/kernels/` (CPU) and `src/backends/metal/kernels/` (Metal).

## Part 1: Adding a NEON Profile for an Existing Op

This walkthrough adds a NEON-optimized variant for `pow` in the `elementwise_binary_scalar` kernel family.

### Step 1: Edit the kernel definition

Open `src/backends/cpu/kernels/elementwise.def` and find the `PROFILES` block for `pow` within its kernel family. Add a `NEON` entry:

```diff
 pow: {
     ACCELERATE: "cpu_{op}_{dtype:short}_accelerate",
+    NEON: "cpu_{op}_{dtype:short}_neon",
     SCALAR: "cpu_{op}_{dtype:short}_scalar"
 },
```

The symbol name pattern (`cpu_{op}_{dtype:short}_neon`) must match the C function you implement in the next step.

For reference, here is what a complete kernel family looks like in the `.def` DSL (from `unary.def`):

```
KERNEL_FAMILY(unary_float) {
    AXIS(op): @unary_float_ops,
    AXIS(dtype): @float_all,
    NAME_PATTERN: "{op}_{dtype:short}",
    OP: $op,
    INPUT_DTYPE: $dtype,
    OUTPUT_DTYPE: $dtype,
    ACCUM_DTYPE: select($dtype, { FLOAT64: FLOAT64, *: FLOAT32 }),
    STRIDE_MODE: STRIDED,
    PROFILES: {
        ACCELERATE: "cpu_unary_apply",
        NEON: "cpu_unary_apply",
        AVX2: "cpu_unary_apply",
        SCALAR: "cpu_unary_apply"
    },
}
```

Key concepts:
- `AXIS` declares a dimension that the kernel family is expanded over (one kernel per combination of axis values).
- `PROFILES` maps hardware profile names to implementation symbol names. The codegen expands these into dispatch tables.
- `select(...)` chooses values conditionally based on the axis variable.

### Step 2: Implement the kernel function

Create or edit the appropriate source file. For float32 NEON elementwise kernels, implementations live in `src/backends/cpu/ops/elementwise/neon/`.

The function must match the binary elementwise signature used by the kernel family:

```c
marmot_error_t
cpu_pow_f32_neon(const void *device_ctx,
                 const marmot_tensor_t *a,
                 const marmot_tensor_t *b,
                 marmot_tensor_t *out)
{
    (void)device_ctx;
    const float *lhs = (const float *)a->data;
    const float *rhs = (const float *)b->data;
    float *dst = (float *)out->data;
    const size_t n = marmot_tensor_num_elements(a);
    for (size_t i = 0; i < n; ++i) {
        dst[i] = powf(lhs[i], rhs[i]);
    }
    return MARMOT_SUCCESS;
}
```

If you add a new `.c` file, update `src/backends/cpu/meson.build` to include it.

### Step 3: Rebuild

```bash
make build
```

This triggers code generation and rebuilds the project. Among the regenerated files:

| Generated file | Contents |
|----------------|----------|
| `src/backends/cpu/dispatch/cpu_kernel_query.gen.c` | Signature-to-kernel selection tables |
| `src/backends/cpu/dispatch/bytecode_exec_cpu.gen.c` | Bytecode exec table (op_index to function pointer) |
| `src/backends/cpu/dispatch/bytecode_tables_cpu.gen.h` | Opcode counts and invalid markers |
| `include/marmot/traits_ids.gen.h` | Op, kernel, and profile identifiers |

### Step 4: Add a correctness test

Write a test that exercises the new kernel through the graph API:

```c
#include <marmot/marmot.h>
#include <math.h>

int main(void) {
    marmot_graph_t *graph = marmot_graph_create();
    marmot_graph_tensor_desc_t desc = {
        .ndim = 1, .shape = {8}, .dtype = MARMOT_DTYPE_FLOAT32
    };

    marmot_value_id_t a_id, b_id, out_id;
    marmot_graph_add_input(graph, &desc, &a_id);
    marmot_graph_add_input(graph, &desc, &b_id);

    marmot_op_signature_t sig = {
        .op_id = MARMOT_OP_POW,
        .input_dtype = MARMOT_DTYPE_FLOAT32,
        .weight_dtype = MARMOT_DTYPE_FLOAT32,
        .output_dtype = MARMOT_DTYPE_FLOAT32,
        .accum_dtype = MARMOT_DTYPE_FLOAT32,
        .stride_mode = MARMOT_STRIDE_MODE_CONTIGUOUS,
        .dims.elementwise = {.n_elems = 8},
    };
    marmot_value_id_t inputs[] = {a_id, b_id};
    marmot_graph_add_op(graph, "pow", &sig, inputs, 2, &desc, 1, &out_id);
    marmot_graph_finalize(graph, MARMOT_BACKEND_CPU);

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    const size_t shape[] = {8};
    marmot_tensor_t *a = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);

    float *a_data = marmot_tensor_data_f32_mut(ctx, a);
    float *b_data = marmot_tensor_data_f32_mut(ctx, b);
    for (size_t i = 0; i < 8; ++i) {
        a_data[i] = (float)i;
        b_data[i] = 2.0f;
    }

    const marmot_tensor_t *graph_inputs[] = {a, b};
    marmot_tensor_t *graph_outputs[] = {out};
    marmot_graph_execute(graph, ctx, graph_inputs, 2, graph_outputs, 1);

    const float *out_data = marmot_tensor_data_f32(ctx, out);
    for (size_t i = 0; i < 8; ++i) {
        float expected = powf((float)i, 2.0f);
        if (fabsf(out_data[i] - expected) > 1e-6f) {
            return 1;
        }
    }

    marmot_tensor_destroy(a);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(out);
    marmot_destroy(ctx);
    marmot_graph_destroy(graph);
    return 0;
}
```

Register the test in `tests/meson.build` and run:

```bash
make test
```

For memory-sensitive changes, also run with sanitizers:

```bash
make test-asan
```

## Part 2: Adding a Completely New Operation

If you are adding an operation that does not yet exist in the `.def` files, the workflow has additional steps.

### Step 1: Define the operation in ops.def

Add the new operation to `src/core/defs/ops.def`. This gives it a `MARMOT_OP_*` identifier and defines its category (unary, binary, reduction, etc.).

### Step 2: Create a kernel family in the .def file

Add a `KERNEL_FAMILY(...)` block to the appropriate `.def` file in `src/backends/cpu/kernels/`. Choose the file that matches the operation category (e.g., `unary.def`, `elementwise.def`, `reduction.def`).

Define:
- The axes the kernel is expanded over (typically `op` and `dtype`).
- The op signature fields (`OP`, `INPUT_DTYPE`, `OUTPUT_DTYPE`, `ACCUM_DTYPE`, `STRIDE_MODE`).
- The `PROFILES` mapping for each hardware variant you want to support.

### Step 3: Implement the kernel functions

Write the actual C implementations under `src/backends/cpu/ops/<category>/`. Each profile that appears in the `.def` `PROFILES` block must have a corresponding symbol.

### Step 4: Add Metal support (optional)

If the operation should also run on Metal:
1. Add a kernel family to `src/backends/metal/kernels/<category>.def`.
2. Implement the Metal shader in `src/backends/metal/shaders/`.
3. Implement the Objective-C++ dispatch in `src/backends/metal/ops/`.

### Step 5: Build, test, verify

```bash
make build && make test
```

The build will regenerate all dispatch tables, bytecode tables, and kernel query files for both backends.

## Summary

| Task | Files to edit |
|------|---------------|
| Add profile for existing op | `.def` file + implementation `.c` file |
| Add new operation | `ops.def` + `.def` kernel file + implementation `.c` file(s) |
| Metal support | Metal `.def` + shader `.metal` + dispatch `.mm` |
| Register new `.c` file | `meson.build` in the backend directory |

The code generation pipeline ensures that changes to `.def` files are automatically propagated to all dispatch tables, bytecode exec tables, and kernel query tables on the next build.
