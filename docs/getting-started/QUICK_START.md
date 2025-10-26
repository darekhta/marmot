# Quick Start

Get up and running with Marmot in 5 minutes.

---

## Prerequisites

| Dependency | Minimum Version | Notes |
|------------|----------------|-------|
| C compiler | Clang 17+ or GCC 14+ | Must support `-std=c2x` (C23 draft) |
| C++ compiler | Clang 17+ or GCC 14+ | Must support `-std=c++23` |
| Meson | 1.2+ | Build system |
| Ninja | 1.10+ | Build backend |
| Python | 3.10+ | Codegen scripts |
| Jinja2 | 3.1+ | Template engine for codegen (`pip install jinja2`) |

**Optional:**
- Apple Accelerate framework (auto-detected on macOS)
- Metal SDK (auto-detected on macOS for GPU backend)

---

## Build

```bash
# Debug build (auto-formats code first)
make build

# Optimized release build
make build-release
```

`make build` produces a debug build in `build-debug/`, including `build-debug/libmarmot.dylib`.

### All Build Variants

| Command | Description |
|---------|-------------|
| `make build` | Debug build (default) |
| `make build-release` | Optimized release build |
| `make build-asan` | Debug + AddressSanitizer |
| `make build-ubsan` | Debug + UndefinedBehaviorSanitizer |
| `make build-tsan` | Debug + ThreadSanitizer |
| `make build-release-asan` | Release + AddressSanitizer |
| `make build-release-ubsan` | Release + UndefinedBehaviorSanitizer |
| `make build-lm` | Build marmot-lm CLI (debug) |
| `make build-lm-release` | Build marmot-lm CLI (release) |
| `make install` | Install libmarmot + marmot-lm to PREFIX |

---

## Run Tests

```bash
# Fast tests (default suite, for development)
make test

# All tests including slow/LLM tests (CI suite)
make test-ci

# With sanitizers
make test-asan
make test-ubsan
make test-tsan
```

| Command | Description |
|---------|-------------|
| `make test` | Fast default suite |
| `make test-verbose` | Default suite with verbose output |
| `make test-ci` | Full CI suite (includes slow tests) |
| `make test-ci-verbose` | Full CI suite, verbose |
| `make test-asan` | Default suite with AddressSanitizer |
| `make test-ubsan` | Default suite with UBSanitizer |
| `make test-tsan` | Default suite with ThreadSanitizer |
| `make test-ci-metal-matrix` | CI suite with Metal environment matrix |

---

## Minimal Graph Example

```c
#include <marmot/marmot.h>
#include <stdio.h>

int main(void) {
    // 1. Create a graph
    marmot_graph_t *graph = marmot_graph_create();

    // 2. Add an input: 64x256 float32 tensor
    marmot_graph_tensor_desc_t desc = {
        .ndim = 2, .shape = {64, 256}, .dtype = MARMOT_DTYPE_FLOAT32
    };
    marmot_value_id_t input_id;
    marmot_graph_add_input(graph, &desc, &input_id);

    // 3. Add a ReLU operation
    marmot_value_id_t output_id;
    marmot_graph_add_op(graph, "relu", nullptr, &input_id, 1, &desc, 1, &output_id);

    // 4. Finalize (compiles to bytecode, selects backend)
    marmot_backend_type_t backend;
    marmot_graph_finalize_auto(graph, &backend);

    // 5. Create context and tensors
    marmot_context_t *ctx = marmot_init(backend);
    const size_t shape[] = {64, 256};
    marmot_tensor_t *input  = marmot_tensor_create(ctx, shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output = marmot_tensor_create(ctx, shape, 2, MARMOT_DTYPE_FLOAT32);

    // 6. Fill input data
    float *in = marmot_tensor_data_f32_mut(ctx, input);
    for (size_t i = 0; i < 64 * 256; ++i) in[i] = 1.0f;

    // 7. Execute
    const marmot_tensor_t *inputs[]  = {input};
    marmot_tensor_t       *outputs[] = {output};
    marmot_graph_execute(graph, ctx, inputs, 1, outputs, 1);

    // 8. Read result
    const float *out = marmot_tensor_data_f32(ctx, output);
    printf("out[0] = %f\n", out ? out[0] : 0.0f);

    // Cleanup
    marmot_tensor_destroy(input);
    marmot_tensor_destroy(output);
    marmot_destroy(ctx);
    marmot_graph_destroy(graph);
    return 0;
}
```

Compile and run:

```bash
cc -std=c2x -I include main.c -L build-debug -lmarmot -Wl,-rpath,build-debug -o main
./main
```

---

## Inspect Kernel Selection

After finalization, dump the graph to see which kernels were selected per node:

```c
marmot_graph_dump_json(graph, "/tmp/graph.json");
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `MARMOT_ROUTE=cpu` | Force CPU backend |
| `MARMOT_ROUTE=metal` | Force Metal (GPU) backend |
| `MARMOT_GRAPH_TRACE=1` | Print per-node execution trace |
| `MARMOT_GRAPH_DUMP_JSON=1` | Auto-dump graph JSON on finalization |
| `MARMOT_GRAPH_NAN_CHECK=1` | Detect NaN/Inf in outputs |

See [Environment Variables](../graph/ENVIRONMENT.md) for the full reference.

---

## Project Structure

```
marmot/
├── include/marmot/        Public headers
│   ├── graph/             Graph API
│   ├── ops/               Operation headers
│   └── inference/         Inference headers
├── src/
│   ├── core/              Dispatch, bytecode, tensor, context
│   ├── backends/          CPU and Metal implementations
│   ├── graph/             Graph execution (C++)
│   ├── inference/         Serving engine, KV pool
│   └── tokenizer/         BPE/WordPiece/Unigram
├── tests/                 Test suites (backend, graph, inference, golden)
├── scripts/codegen/       Kernel code generation (.def → .gen.c/.gen.mm)
├── tools/marmot-bench/    Benchmarking suite
└── apps/marmot-lm/        Rust-based LLM serving application
```

---

## Next Steps

- **Tutorials**
  - [Simple Graph](../tutorials/SIMPLE_GRAPH.md) -- Detailed graph walkthrough
  - [Add a Kernel](../tutorials/ADD_KERNEL.md) -- Adding custom kernel variants

- **Reference**
  - [Operations Catalog](../kernels/OPS.md) -- All supported operations
  - [Signatures](../graph/SIGNATURES.md) -- Signature field details
  - [Operations Utilities](OPS_UTILS.md) -- Shape inference and validation helpers
  - [Benchmarking](BENCHMARKING.md) -- Running `marmot-bench`
  - [Environment Variables](../graph/ENVIRONMENT.md) -- All environment variables

- **Architecture**
  - [Graph Architecture](../graph/ARCHITECTURE.md) -- Graph system internals
  - [Bytecode Dispatch](../BYTECODE_DISPATCH.md) -- Execution model
  - [Kernel Dispatch](../kernels/DISPATCH.md) -- Kernel dispatch paths
  - [Kernel DSL](../kernels/DSL.md) -- Kernel definition language
