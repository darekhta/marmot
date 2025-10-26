# Tutorial: Building a Simple Computation Graph

This tutorial walks through building, finalizing, and executing a computation graph using Marmot's graph API. By the end, you will have a working program that performs a matrix multiplication followed by a ReLU activation on the GPU or CPU.

## Prerequisites

- Marmot built and installed (`make build`)
- A C compiler with C23 (c2x) support
- Basic understanding of tensor operations

## Step 1: Create the Graph

A graph is a container for inputs, operations, and their connections. Start by creating an empty graph with `marmot_graph_create`:

```c
#include <marmot/marmot.h>
#include <marmot/graph/graph.h>
#include <stdio.h>

int main(void) {
    marmot_graph_t *graph = marmot_graph_create();
    if (graph == nullptr) {
        fprintf(stderr, "Failed to create graph\n");
        return 1;
    }

    // ... build the graph ...

    marmot_graph_destroy(graph);
    return 0;
}
```

## Step 2: Add Input Tensors

Define each input tensor with a descriptor specifying its shape and data type, then register it with the graph. Each call to `marmot_graph_add_input` returns a `marmot_value_id_t` that you use to wire operations together.

```c
marmot_graph_tensor_desc_t desc_a = {
    .ndim = 2,
    .shape = {32, 64},
    .dtype = MARMOT_DTYPE_FLOAT32,
};
marmot_value_id_t input_a;
marmot_error_t err = marmot_graph_add_input(graph, &desc_a, &input_a);
if (err != MARMOT_SUCCESS) {
    fprintf(stderr, "Failed to add input A: %s\n", marmot_error_string(err));
    return 1;
}

marmot_graph_tensor_desc_t desc_b = {
    .ndim = 2,
    .shape = {64, 128},
    .dtype = MARMOT_DTYPE_FLOAT32,
};
marmot_value_id_t input_b;
err = marmot_graph_add_input(graph, &desc_b, &input_b);
if (err != MARMOT_SUCCESS) {
    fprintf(stderr, "Failed to add input B: %s\n", marmot_error_string(err));
    return 1;
}
```

## Step 3: Add a Matmul Operation

Add a matmul node that computes `C = A * B`. The operation is described by a `marmot_op_signature_t` that specifies the op type, data types, layout, and dimensions. The output value ID is returned through the last parameter.

```c
marmot_op_signature_t matmul_sig = {
    .op_id = MARMOT_OP_MATMUL,
    .matmul_layout = MARMOT_MATMUL_LAYOUT_NN,
    .input_dtype = MARMOT_DTYPE_FLOAT32,
    .weight_dtype = MARMOT_DTYPE_FLOAT32,
    .output_dtype = MARMOT_DTYPE_FLOAT32,
    .accum_dtype = MARMOT_DTYPE_FLOAT32,
    .stride_mode = MARMOT_STRIDE_MODE_CONTIGUOUS,
    .dims.matmul = { .N = 32, .M = 128, .K = 64 },
};

marmot_graph_tensor_desc_t desc_c = {
    .ndim = 2,
    .shape = {32, 128},
    .dtype = MARMOT_DTYPE_FLOAT32,
};

marmot_value_id_t matmul_inputs[] = {input_a, input_b};
marmot_value_id_t output_c;

err = marmot_graph_add_op(
    graph, "matmul", &matmul_sig,
    matmul_inputs, 2,
    &desc_c, 1,
    &output_c
);
if (err != MARMOT_SUCCESS) {
    fprintf(stderr, "Failed to add matmul: %s\n", marmot_error_string(err));
    return 1;
}
```

## Step 4: Add a ReLU Activation

Chain operations by using one node's output value ID as another node's input. Here we apply ReLU to the matmul result.

```c
marmot_op_signature_t relu_sig = {
    .op_id = MARMOT_OP_RELU,
    .input_dtype = MARMOT_DTYPE_FLOAT32,
    .output_dtype = MARMOT_DTYPE_FLOAT32,
    .accum_dtype = MARMOT_DTYPE_FLOAT32,
    .stride_mode = MARMOT_STRIDE_MODE_CONTIGUOUS,
    .dims.elementwise = { .n_elems = 32 * 128 },
};

marmot_graph_tensor_desc_t desc_d = {
    .ndim = 2,
    .shape = {32, 128},
    .dtype = MARMOT_DTYPE_FLOAT32,
};

marmot_value_id_t relu_inputs[] = {output_c};
marmot_value_id_t output_d;

err = marmot_graph_add_op(
    graph, "relu", &relu_sig,
    relu_inputs, 1,
    &desc_d, 1,
    &output_d
);
if (err != MARMOT_SUCCESS) {
    fprintf(stderr, "Failed to add relu: %s\n", marmot_error_string(err));
    return 1;
}
```

## Step 5: Finalize the Graph

Finalization selects a backend, resolves kernel variants for each node, detects operation fusion opportunities, and compiles the graph into a bytecode program for efficient execution.

There are three ways to finalize.

**Auto-select backend** (recommended). Marmot picks the best backend based on cost estimation:

```c
marmot_backend_type_t backend;
err = marmot_graph_finalize_auto(graph, &backend);
if (err != MARMOT_SUCCESS) {
    fprintf(stderr, "Finalize failed: %s\n", marmot_error_string(err));
    return 1;
}
printf("Selected backend: %s\n",
    backend == MARMOT_BACKEND_METAL ? "Metal" : "CPU");
```

**Force a specific backend**:

```c
err = marmot_graph_finalize(graph, MARMOT_BACKEND_CPU);
```

**Use finalize options** for fine-grained control (routing policy, flags):

```c
marmot_graph_finalize_options_t opts;
marmot_graph_finalize_options_init(&opts);
opts.routing_policy = MARMOT_ROUTING_ALWAYS_GPU;

marmot_backend_type_t backend;
err = marmot_graph_finalize_with_options(graph, &opts, &backend);
```

### What happens during finalization

1. Missing signature fields are populated from context.
2. Adjacent operations are checked for fusion patterns (for example, `matmul + add` becomes `matmul_bias`, `add + relu` becomes `add_relu`).
3. Each node's signature is compiled to a dense bytecode opcode via the backend kernel query tables.
4. A linear bytecode program is emitted for the entire graph.

## Step 6: Execute the Graph

Create a context that matches the selected backend, allocate tensors, and run the graph.

```c
marmot_context_t *ctx = marmot_init(backend);
if (ctx == nullptr) {
    fprintf(stderr, "marmot_init failed: %s\n", marmot_get_last_error_detail());
    return 1;
}

const size_t shape_a[] = {32, 64};
const size_t shape_b[] = {64, 128};
const size_t shape_d[] = {32, 128};

marmot_tensor_t *tensor_a = marmot_tensor_create(ctx, shape_a, 2, MARMOT_DTYPE_FLOAT32);
marmot_tensor_t *tensor_b = marmot_tensor_create(ctx, shape_b, 2, MARMOT_DTYPE_FLOAT32);
marmot_tensor_t *tensor_d = marmot_tensor_create(ctx, shape_d, 2, MARMOT_DTYPE_FLOAT32);

float *da = marmot_tensor_data_f32_mut(ctx, tensor_a);
float *db = marmot_tensor_data_f32_mut(ctx, tensor_b);
for (size_t i = 0; i < 32 * 64; i++) da[i] = 1.0f;
for (size_t i = 0; i < 64 * 128; i++) db[i] = 1.0f;

const marmot_tensor_t *inputs[] = {tensor_a, tensor_b};
marmot_tensor_t *outputs[] = {tensor_d};

err = marmot_graph_execute(graph, ctx, inputs, 2, outputs, 1);
if (err != MARMOT_SUCCESS) {
    fprintf(stderr, "Execution failed: %s\n", marmot_error_string(err));
    return 1;
}

const float *result = marmot_tensor_data_f32(ctx, tensor_d);
printf("Result[0] = %f (expected: %f)\n", result[0], 64.0f);
```

The input and output arrays must be ordered to match the order in which inputs were added to the graph and outputs were produced by the final operations.

## Step 7: Debug with JSON Dump

Export the finalized graph to JSON for inspection:

```c
err = marmot_graph_dump_json(graph, "/tmp/my_graph.json");
if (err == MARMOT_SUCCESS) {
    printf("Graph dumped to /tmp/my_graph.json\n");
}
```

The JSON file contains the node list with operation signatures and selected kernel IDs, the value list with tensor descriptors, and the backend that was selected.

### Environment variables for debugging

| Variable | Purpose |
|----------|---------|
| `MARMOT_GRAPH_TRACE=1` | Print each bytecode instruction as it executes |
| `MARMOT_GRAPH_NAN_CHECK=1` | Check for NaN/Inf after every operation |
| `MARMOT_ROUTING=cpu` | Force CPU backend regardless of finalization |

## Cleanup

Always destroy tensors, the context, and the graph when done:

```c
marmot_tensor_destroy(tensor_a);
marmot_tensor_destroy(tensor_b);
marmot_tensor_destroy(tensor_d);
marmot_destroy(ctx);
marmot_graph_destroy(graph);
```

## Complete Example

A complete working program combining all steps above is available in the graph test suite under `tests/graph/`. The pattern is the same: create graph, add inputs, add ops, finalize, execute, verify, destroy.

## Next Steps

- **Adding custom kernels**: `docs/tutorials/ADD_KERNEL.md`
- **Operation fusion**: `docs/graph/FUSION.md`
- **Available operations**: `docs/API_OPS.gen.md`
- **Signature field reference**: `docs/graph/SIGNATURES.md`
- **Bytecode dispatch architecture**: `docs/BYTECODE_DISPATCH.md`

## API Reference

The graph API is declared in `include/marmot/graph/graph.h`. The key functions used in this tutorial are:

| Function | Purpose |
|----------|---------|
| `marmot_graph_create` | Create an empty graph |
| `marmot_graph_add_input` | Register an input tensor descriptor |
| `marmot_graph_add_op` | Add an operation node |
| `marmot_graph_finalize` | Finalize with a specific backend |
| `marmot_graph_finalize_auto` | Finalize with automatic backend selection |
| `marmot_graph_finalize_with_options` | Finalize with detailed options |
| `marmot_graph_execute` | Execute the finalized graph |
| `marmot_graph_dump_json` | Export graph to JSON for debugging |
| `marmot_graph_destroy` | Destroy the graph |
