# Graph C API Reference

Header: `include/marmot/graph/graph.h`

The graph API provides lifecycle management, construction, finalization, execution, and debug facilities for computational graphs. All functions that can fail return `marmot_error_t` and are marked `[[nodiscard]]`.

---

## Key Types

### `marmot_graph_t`

Opaque graph handle. Created with `marmot_graph_create`, freed with `marmot_graph_destroy`.

### `marmot_value_id_t`

Header: `include/marmot/graph/graph_types.h`

An unsigned 32-bit identifier for SSA values (inputs, outputs, and intermediates). The sentinel `MARMOT_VALUE_ID_INVALID` (`UINT32_MAX`) indicates an uninitialized or invalid value.

### `marmot_graph_tensor_desc_t`

Header: `include/marmot/graph/graph_types.h`

Describes a tensor's metadata within the graph:

```c
typedef struct {
    marmot_dtype_t dtype;
    uint32_t ndim;
    size_t shape[MARMOT_MAX_DIMS];
    size_t strides[MARMOT_MAX_DIMS];
} marmot_graph_tensor_desc_t;
```

Stride conventions:
- If all strides are zero, the tensor is treated as contiguous and row-major strides are computed automatically.
- If some strides are zero and some are non-zero, the descriptor is rejected.

### `marmot_op_signature_t`

Header: `include/marmot/graph/op_signature.h`

Operation signature used for kernel selection. Captures op type, data types, memory layout, quantization parameters, epilogues, and dimension information. See [SIGNATURES.md](SIGNATURES.md) for the complete field reference.

### `marmot_graph_attr_t`

Header: `include/marmot/graph/graph_types.h`

Key-value attribute for graph nodes. Supports integer, float, string, and tensor value types.

---

## Lifecycle

### `marmot_graph_create`

```c
marmot_graph_t *marmot_graph_create(void);
```

Creates a new empty graph. Returns `nullptr` on allocation failure.

### `marmot_graph_destroy`

```c
void marmot_graph_destroy(marmot_graph_t *graph);
```

Destroys a graph and frees all associated resources. Safe to call with `nullptr`.

### `marmot_graph_get_backend`

```c
marmot_backend_type_t marmot_graph_get_backend(const marmot_graph_t *graph);
```

Returns the backend assigned during finalization. Undefined before finalization.

---

## Build Phase

### `marmot_graph_add_input`

```c
marmot_error_t marmot_graph_add_input(
    marmot_graph_t *graph,
    const marmot_graph_tensor_desc_t *desc,
    marmot_value_id_t *out_value_id
);
```

Adds a graph input with the given tensor descriptor. Writes the new value ID to `out_value_id`.

### `marmot_graph_add_op`

```c
marmot_error_t marmot_graph_add_op(
    marmot_graph_t *graph,
    const char *op_name,
    const marmot_op_signature_t *signature,
    const marmot_value_id_t *input_ids,
    size_t num_inputs,
    const marmot_graph_tensor_desc_t *output_descs,
    size_t num_outputs,
    marmot_value_id_t *out_value_ids
);
```

Adds an operation node to the graph.

Parameters:
- `op_name` -- Operation name string (e.g., `"matmul"`, `"add"`, `"layernorm"`).
- `signature` -- Optional. Pass `nullptr` for simple ops; Marmot infers `op_id` from `op_name` and fills signature fields during finalization. Provide an explicit signature when the operation needs constraints (quantization scheme, epilogues, stride mode) or when automatic inference is insufficient.
- `input_ids` -- Array of value IDs for the operation's inputs. All referenced values must already exist.
- `output_descs` -- Array of tensor descriptors for the operation's outputs.
- `out_value_ids` -- Receives the newly created output value IDs.

Signature inference (implemented in `src/graph/graph_signature.cpp`) handles:
- `matmul` / `linear`: infers `matmul_layout` (NN/NT) and `dims.matmul.{N,K,M}` from tensor shapes.
- Elementwise, unary, and reduction ops: infers dtypes and `dims.elementwise.n_elems`.
- Other ops generally require explicit signature fields if kernel selection depends on them.

---

## Finalization

Finalization compiles the graph for a specific backend. A graph must be finalized before execution.

### `marmot_graph_finalize`

```c
marmot_error_t marmot_graph_finalize(marmot_graph_t *graph, marmot_backend_type_t backend);
```

Finalizes the graph for the specified backend. Runs the full 4-pass pipeline: signature population, fusion detection, kernel selection, and bytecode compilation.

### `marmot_graph_finalize_auto`

```c
marmot_error_t marmot_graph_finalize_auto(marmot_graph_t *graph, marmot_backend_type_t *out_backend);
```

Automatically selects a backend and finalizes the graph. Writes the chosen backend to `out_backend`. Tries candidate backends in order and selects the first that can finalize the entire graph.

### `marmot_graph_finalize_with_options`

```c
marmot_error_t marmot_graph_finalize_with_options(
    marmot_graph_t *graph,
    const marmot_graph_finalize_options_t *opts,
    marmot_backend_type_t *out_backend
);
```

Finalizes with extended options.

### `marmot_graph_finalize_options_init`

```c
marmot_error_t marmot_graph_finalize_options_init(marmot_graph_finalize_options_t *opts);
```

Initializes finalize options to defaults. Must be called before modifying individual fields.

### Finalize Options Structure

```c
typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;                      // MARMOT_GRAPH_FINALIZE_FLAG_*
    marmot_routing_policy_t routing_policy;
    marmot_backend_type_t backend;
    const void *pnext;
    uint64_t reserved[4];
} marmot_graph_finalize_options_t;
```

Flags:
- `MARMOT_GRAPH_FINALIZE_FLAG_AUTO_BACKEND` -- Enable automatic backend selection.

---

## Execution

### `marmot_graph_execute`

```c
marmot_error_t marmot_graph_execute(
    marmot_graph_t *graph,
    const marmot_context_t *ctx,
    const marmot_tensor_t **inputs,
    size_t num_inputs,
    marmot_tensor_t **outputs,
    size_t num_outputs
);
```

Executes the finalized graph. The graph must have been finalized before calling this function.

Parameters:
- `ctx` -- Marmot context providing backend resources and allocator.
- `inputs` -- Array of input tensors, matching the order of `marmot_graph_add_input` calls.
- `outputs` -- Array of output tensors to receive results.

Execution creates or reuses a cached `ExecutionSession` that persists intermediate buffers across calls.

---

## Debug

### `marmot_graph_dump_json`

```c
marmot_error_t marmot_graph_dump_json(const marmot_graph_t *graph, const char *path);
```

Writes a JSON representation of the graph to the given file path. The dump includes:
- Backend type.
- Node list with op names, signatures, and selected `kernel_id` / `kernel_name`.
- Value list with tensor descriptors.

The `kernel_id` in JSON is diagnostic only; runtime execution uses the compiled bytecode `bc_op_index`.

---

## Current Constraints

- Graph construction order is the execution order. There is no global scheduler or topological reordering.
- Shapes are static at build and finalize time.
- Some internal operations (`view`, `slice` metadata) require per-node parameters not yet exposed in the C API.
- Constant binding and inference hints (max sequence length, RoPE parameters, KV-cache options) are available through the internal C++ API but not yet fully exposed in the public C API.

---

## See Also

- [SIGNATURES.md](SIGNATURES.md) -- Complete `marmot_op_signature_t` field reference
- [ARCHITECTURE.md](ARCHITECTURE.md) -- Graph internals and finalization pipeline
- [GGUF.md](GGUF.md) -- Building graphs from GGUF model files
- [DEBUGGING.md](DEBUGGING.md) -- Troubleshooting and diagnostic tools
