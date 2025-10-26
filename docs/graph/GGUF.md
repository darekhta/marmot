# GGUF Loading and Graph Building

Marmot's GGUF importer reads GGUF model files and builds finalized execution graphs for LLM inference. It supports two entry points: an extendable loader API and direct builder helpers.

---

## Supported Architectures

Architecture traits are defined in `src/graph/gguf/architecture.cpp`. The following architectures are currently supported:

| Architecture | Notes |
|-------------|-------|
| Llama / Mistral | Includes Llama 2, Llama 3, Mistral, and compatible variants |
| Qwen2 | Qwen2 family |
| Qwen3 | Qwen3 family |
| Phi-3 | Microsoft Phi-3 |
| Gemma | Google Gemma |

---

## Supported Quantization Formats

The GGUF importer handles the following GGML quantization types:

| Format | Description |
|--------|-------------|
| Q4_0 | 4-bit, no zero-point |
| Q4_1 | 4-bit with zero-point |
| Q5_0 | 5-bit, no zero-point |
| Q5_1 | 5-bit with zero-point |
| Q8_0 | 8-bit, no zero-point |
| Q8_1 | 8-bit with zero-point |
| Q2_K | K-quant 2-bit |
| Q3_K | K-quant 3-bit |
| Q4_K | K-quant 4-bit |
| Q5_K | K-quant 5-bit |
| Q6_K | K-quant 6-bit |
| Q8_K | K-quant 8-bit |
| F16 | IEEE 754 half-precision (unquantized) |
| F32 | Single-precision float (unquantized) |

---

## Loader API (Recommended)

Header: `include/marmot/graph/gguf_loader.h`

The loader API provides a structured workflow for loading GGUF files with configurable options.

### Types

- `marmot_gguf_options_t` -- Loader configuration. Initialize with `marmot_gguf_options_init()`.
- `marmot_gguf_loader_t` -- Opaque loader handle.
- `marmot_gguf_caps_t` -- Loader capability descriptor.

### Options Structure

```c
typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;                              // MARMOT_GGUF_FLAG_*
    marmot_backend_type_t backend;
    const struct marmot_allocator *allocator;
    marmot_context_t *ctx;
    const marmot_packed_graph_options_t *packed_opts;  // Required
    const void *pnext;
    uint64_t reserved[2];
} marmot_gguf_options_t;
```

The `packed_opts` field is required and controls token count, KV-cache sizing, and block configuration.

### Loader Flags

| Flag | Description |
|------|-------------|
| `MARMOT_GGUF_FLAG_STRICT_VALIDATION` | Enable strict validation of GGUF metadata |
| `MARMOT_GGUF_FLAG_ALLOW_UNKNOWN_OPS` | Allow unknown operations (skip instead of fail) |
| `MARMOT_GGUF_FLAG_AUTO_BACKEND` | Automatically select the best backend |

### Functions

```c
marmot_error_t marmot_gguf_options_init(marmot_gguf_options_t *opts);
marmot_error_t marmot_gguf_loader_create(const marmot_gguf_options_t *opts,
                                          marmot_gguf_loader_t **out_loader);
void           marmot_gguf_loader_destroy(marmot_gguf_loader_t *loader);
marmot_error_t marmot_gguf_loader_load_file(marmot_gguf_loader_t *loader,
                                             const char *path,
                                             marmot_graph_t **out_graph);
marmot_error_t marmot_gguf_loader_load_memory(marmot_gguf_loader_t *loader,
                                               const void *data, size_t len,
                                               marmot_graph_t **out_graph);
marmot_error_t marmot_gguf_loader_query_capabilities(const marmot_gguf_loader_t *loader,
                                                      marmot_gguf_caps_t *out_caps);
const marmot_error_info_t *marmot_gguf_loader_last_error(const marmot_gguf_loader_t *loader);
```

Notes:
- Graphs returned by `marmot_gguf_loader_load_file` are already finalized for the selected backend.
- `marmot_gguf_loader_load_memory` currently returns `MARMOT_ERROR_NOT_IMPLEMENTED`.

---

## Direct Builder API

Header: `include/marmot/graph/gguf_model.h`

For cases where you need access to the model object or want more control over the loading process.

### Model Functions

```c
marmot_error_t marmot_gguf_model_load(const char *path,
                                       marmot_backend_type_t backend,
                                       marmot_gguf_model_t **model);
marmot_error_t marmot_gguf_model_metadata(const marmot_gguf_model_t *model,
                                           marmot_gguf_model_meta_t *meta);
marmot_tensor_t *marmot_gguf_model_tensor(const marmot_gguf_model_t *model,
                                           const char *name);
void marmot_gguf_model_destroy(marmot_gguf_model_t *model);
```

### Graph Builder Functions

```c
marmot_error_t marmot_graph_from_gguf_packed(const char *path,
                                              marmot_backend_type_t backend,
                                              const marmot_packed_graph_options_t *packed_opts,
                                              marmot_graph_t **graph);
marmot_error_t marmot_graph_from_model_packed(marmot_gguf_model_t *model,
                                               marmot_backend_type_t backend,
                                               const marmot_packed_graph_options_t *packed_opts,
                                               marmot_graph_t **graph);
```

- `marmot_graph_from_gguf_packed` loads a model internally and attaches it to the graph for automatic cleanup.
- `marmot_graph_from_model_packed` builds from an existing model; the caller must keep the model alive for the graph's lifetime.

Both builders finalize the graph before returning.

---

## Packed Graph Options

The GGUF importer builds a "packed token graph" using paged attention and token metadata. Options are configured via `marmot_packed_graph_options_t`:

| Field | Description | Constraints |
|-------|-------------|-------------|
| `token_count` | Number of tokens per execution step | |
| `sample_count` | Number of tokens to sample | Must be <= `token_count` |
| `max_seqs` | Maximum concurrent sequences | |
| `max_seq_len` | Maximum sequence length | |
| `block_size` | KV-cache block size | Must be a power of two and > 1 |
| `num_kv_blocks` | Number of KV-cache blocks | |
| `kv_dtype` | Data type for KV-cache entries | |
| `flags` | Feature flags | See `MARMOT_PACKED_GRAPH_FLAG_*` |

Use `marmot_packed_graph_options_init()` to fill defaults, then set the required sizes.

---

## Low-Level GGUF Access

Header: `include/marmot/graph/gguf_loader.h`

For direct access to GGUF file contents without graph building:

```c
marmot_gguf_t *marmot_gguf_load(const char *path);
void marmot_gguf_unload(marmot_gguf_t *gguf);
const marmot_gguf_kv_t *marmot_gguf_find_kv(const marmot_gguf_t *gguf, const char *key);
const marmot_gguf_tensor_t *marmot_gguf_find_tensor(const marmot_gguf_t *gguf, const char *name);
```

The `marmot_gguf_t` structure provides access to version, alignment, key-value metadata, and tensor descriptors with quantization scheme information.

---

## See Also

- [API.md](API.md) -- Graph C API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) -- Graph finalization and execution pipeline
- [SIGNATURES.md](SIGNATURES.md) -- Operation signature fields (quantization, layout)
