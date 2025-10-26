# Marmot Development Guide for AI Assistants

## Project Overview

**Marmot** is a high-performance ML inference framework with a modern C23 core, a C++ graph runtime, and bytecode-compiled execution. Graphs are compiled into bytecode programs that execute through generated per-backend dispatch tables.

**Key Principles:**
- Pure C23 (no backward compatibility)
- Type-generic macros via `_Generic`
- Bytecode dispatch as the primary execution model
- Consolidated, clean codebase
- Full sanitizer support
- Zero-copy unified memory on Apple Silicon

### API naming rules

- Cheap metadata queries (`O(1)`, no allocation/device work) must use `marmot_get_*`, `marmot_is_*`, `marmot_has_*`, `marmot_num_*`, or similar predicate prefixes. Bare nouns are not allowed for new APIs.
- Trait/metadata lookups (dtype info, quantization traits, registries) always use `marmot_get_*`.
- Functions that may allocate, touch device backends, or perform meaningful computation use `marmot_compute_*` or action verbs such as `marmot_quantize`, `marmot_matmul`, `marmot_softmax`.
- Reserve the `marmot_compute_*` prefix for APIs whose implementation can dispatch into backend vtables or otherwise be "expensive".

## Project Structure

```
marmot/
├── include/marmot/        # Public headers (core, ops, graph, inference, tokenizer)
├── src/
│   ├── api/              # C API glue for graph/gguf/etc
│   ├── backends/         # CPU/Metal backends + generated dispatch
│   ├── core/             # Core C23 implementation (tensor/device/dtype/memory/bytecode)
│   ├── graph/            # C++ graph executor, kernel selection, GGUF loader
│   ├── inference/        # Inference runtime (serving engine, KV pool, model state)
│   ├── tokenizer/        # Tokenization (BPE, WordPiece, Unigram)
│   └── utils/            # Shared helpers
├── apps/
│   └── marmot-lm/        # Rust-based LLM serving app (Cargo project)
├── benchmarks/           # Benchmarks
├── examples/             # Example programs
├── scripts/              # Build/test/format/codegen scripts
├── tests/                # Backend, graph, inference, codegen tests
└── docs/                 # Documentation
```

## Quick Commands

### Build & Test
```bash
make build              # Debug build (auto-formats first)
make build-release      # Optimized release build
make test               # Run all tests
make format             # Format code with clang-format
make clean              # Clean debug build
make clean-all          # Remove all build directories
```

### With Sanitizers
```bash
make build-asan         # AddressSanitizer
make test-asan          # Test with ASAN
make build-ubsan        # UndefinedBehaviorSanitizer
make test-ubsan         # Test with UBSan
make build-tsan         # ThreadSanitizer
make test-tsan          # Test with TSan
```

### Manual Control
```bash
./scripts/build.sh --type debug --sanitizer asan --clean
./scripts/test.sh --sanitizer asan --verbose
```

## Build Directories

Build directories are created on demand:
- `build-debug` - Debug build
- `build-release` - Optimized release
- `build-debug-asan` - Debug + AddressSanitizer
- `build-debug-ubsan` - Debug + UBSanitizer
- `build-debug-tsan` - Debug + ThreadSanitizer

## Language Standards

### C Code
- **Standard**: C2x (C23 draft)
- **Compiler**: clang/gcc with `-std=c2x`
- **Features**: Pure C23 only, no backward compatibility
  - `nullptr` - Type-safe null pointer
  - `constexpr` - Compile-time constants
  - `thread_local` - Thread-local storage
  - `static_assert` - Compile-time assertions
  - `[[nodiscard]]` - Compiler-enforced error checking
  - `[[maybe_unused]]` - Clean unused parameter handling
  - `_Generic` - Type-generic macro dispatch
  - `typeof` - Type-safe macros
  - Designated initializers
- **Style**: Self-documenting code, minimal comments
- **No macros for compatibility** - Pure C23 features only

### C23 Type-Generic Macros

The tensor API uses `_Generic` for type-safe operations:

```c
// User code (type-generic macros - USE THESE)
marmot_tensor_set(tensor, indices, 42.0f);     // Dispatches to _f32
marmot_tensor_get(tensor, indices, 0.0f);      // Returns float
marmot_tensor_fill(tensor, 3.14f);             // Fills with float

// Internal implementation (DO NOT call directly)
float marmot_tensor_get_f32(...);   // Backing function for _Generic
void marmot_tensor_set_f32(...);    // Backing function for _Generic
// ... etc for i32, i16, i8, u8
```

**Why internal functions are in headers:**
- `_Generic` macros expand at compile-time in user code
- Compiler needs function declarations visible
- Clearly documented as "INTERNAL - DO NOT USE"

### Go Bindings (Planned)
- Go bindings are not in the tree yet.

## File Organization

### Consolidated Tensor API

The tensor implementation is split across:

1. **include/marmot/tensor.h**:
   - Public struct definition
   - Creation/destruction functions
   - `_Generic` macros for type-safe access

2. **src/core/tensor/**:
   - `create.c`, `copy.c`, `fill.c`, `transfer.c`
   - `data_access.c`, `metadata.c`, `descriptor.c`
   - `tensor_internal.h` for shared helpers

### Bytecode Dispatch

The primary execution path compiles graphs into bytecode programs:

1. **src/core/bytecode/** - Bytecode format and compilation (`bytecode_compile.h`, `bytecode.h`)
2. **src/backends/cpu/dispatch/bytecode_exec_cpu.gen.c** - Generated CPU bytecode executor
3. **src/backends/metal/ops/bytecode_exec_metal.gen.mm** - Generated Metal bytecode executor
4. **Bytecode tables** - Generated per-backend op tables (`bytecode_tables_cpu.gen.h`, `bytecode_tables_metal.gen.h`)

The codegen pipeline (`scripts/codegen/`) generates bytecode dispatch from `.def` kernel definitions using Jinja2 templates.

### Inference Runtime

The inference module (`src/inference/`) provides:

1. **Serving engine** (`frontends/serving_engine.cpp`) - Continuous batching, request scheduling
2. **KV cache pool** (`kv_pool/kv_pool.cpp`) - Paged attention with pool-level management and swap
3. **Model state** (`model/model.cpp`) - Model loading and weight management
4. **Common utilities** (`common/`) - Shared storage, tensor pointer helpers

### Tokenizer

The tokenizer module (`src/tokenizer/`) implements:
- **BPE** - Byte-pair encoding (`model_bpe.cpp`)
- **WordPiece** - WordPiece tokenization (`model_wordpiece.cpp`)
- **Unigram** - SentencePiece-style unigram (`model_unigram.cpp`)
- **Vocabulary** - Token-to-id mapping and special token handling (`vocab.cpp`)

Public API: `include/marmot/tokenizer.h`

### Header Organization

**types.h** - Foundation types
- No other marmot headers included
- Only stdlib: `<stdbool.h>`, `<stddef.h>`, `<stdint.h>`
- Contains: error enums, backend enums, dtype enums, shape struct, quantization params
- `constexpr` and `static_assert` usage

**error.h** - Error handling
- Includes: `types.h` only
- Thread-safe error state
- `[[nodiscard]]` on all getters

**macros.h** - General utilities
- Includes: `<stddef.h>` only
- Type-safe macros: `MIN`, `MAX`, `SWAP`
- Alignment helpers
- `likely()`/`unlikely()` branch hints
- **NO tensor-specific code**

**tensor.h** - Tensor API
- Includes: `types.h` only
- PUBLIC API section (macros users should call)
- INTERNAL section (type-specific implementations)

**device.h** - Backend abstraction
- Includes: `tensor.h`, `types.h`
- Vtable ops structure
- Context management

**inference.h** - Inference umbrella header
- Pulls in `inference/engine.h`, `inference/llm.h`, `inference/kv_pool.h`, `inference/model.h`

**tokenizer.h** - Tokenizer API
- BPE, WordPiece, Unigram tokenizer creation and encoding

## Adding New C Code

1. **Headers go in**: `include/marmot/`
2. **Implementation**: `src/core/` or `src/backends/`
3. **Update**: `meson.build` to include new source files
4. **Rebuild**: `make build`

Example:
```c
// include/marmot/new_feature.h
#ifndef MARMOT_NEW_FEATURE_H
#define MARMOT_NEW_FEATURE_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

[[nodiscard]] marmot_error_t marmot_new_function(marmot_tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif
```

```c
// src/core/new_feature.c
#include "marmot/new_feature.h"
#include "marmot/error.h"

marmot_error_t marmot_new_function(marmot_tensor_t *tensor) {
    if (unlikely(tensor == nullptr)) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Tensor is null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    // Implementation
    return MARMOT_SUCCESS;
}
```

Update `meson.build` by adding new files to the appropriate `*_sources` list (see existing layout in `meson.build`).

## Testing

### C Tests
- **Location**: `tests/backend`, `tests/graph`, `tests/inference`, `tests/codegen`
- **Run**: `meson test -C build-debug` or `make test`

### Go Tests
- Go bindings are planned but not in the tree yet.

### Sanitizer Testing

```bash
# AddressSanitizer (memory errors)
make build-asan && make test-asan

# UndefinedBehaviorSanitizer (UB detection)
make build-ubsan && make test-ubsan

# ThreadSanitizer (data races)
make build-tsan && make test-tsan
```

**ASAN Build Fix:** The build script now forces system compilers to avoid version mismatches:
```bash
export CC="/usr/bin/clang"
export CXX="/usr/bin/clang++"
export OBJCXX="/usr/bin/clang++"
```

## Code Formatting

All C/C++/ObjC++ code must be formatted with clang-format before building:

```bash
make format              # Format all code
./scripts/format.sh      # Same

make format-check        # Check if formatting needed
./scripts/format-check.sh
```

Configuration: `.clang-format` (LLVM-based)
- IndentWidth: 4
- PointerAlignment: Right (`void *ptr`)
- BinPackParameters: false
- ColumnLimit: 100

**The Makefile auto-formats before every build.**

## Backend Architecture

### Execution Pipeline

1. **Graph construction** - Operations are added to a DAG via the graph API or loaded from GGUF
2. **Bytecode compilation** - The graph is compiled into a linear bytecode program (`src/core/bytecode/`)
3. **Kernel selection** - Each bytecode op is matched to a backend kernel via generated query tables
4. **Bytecode execution** - The compiled program runs through generated dispatch tables per backend

This compile-once execute-many model avoids repeated graph traversal and dispatch overhead during inference.

### Vtable Pattern
```c
struct marmot_device_ops {
    marmot_error_t (*init)(void **device_ctx);
    void (*destroy)(const void *device_ctx);
    marmot_error_t (*matmul)(...);
    // ... more ops
};
```

### Supported Model Architectures
GGUF models are loaded via `src/graph/gguf/`. Currently supported:
- Llama, Mistral, Qwen2, Qwen3, Phi-3, Gemma

Architecture definitions live in `src/graph/gguf/architecture.cpp`.

### Adding Backend Operation

1. **C Header** (`include/marmot/device.h`):
```c
marmot_error_t (*new_op)(const void *device_ctx,
                         const marmot_tensor_t *input,
                         marmot_tensor_t *output);
```

2. **CPU Backend** (`src/backends/cpu/cpu_backend.c`):
```c
static marmot_error_t cpu_new_op([[maybe_unused]] const void *device_ctx,
                                 const marmot_tensor_t *input,
                                 marmot_tensor_t *output) {
    // CPU implementation
    return MARMOT_SUCCESS;
}

static const marmot_device_ops_t cpu_ops = {
    // ... existing ops
    .new_op = cpu_new_op,
};
```

3. **Metal Backend** (`src/backends/metal/metal_backend.mm`):
```objc
static marmot_error_t metal_new_op([[maybe_unused]] const void *device_ctx,
                                   const marmot_tensor_t *input,
                                   marmot_tensor_t *output) {
    // Metal/MPS implementation
    return MARMOT_SUCCESS;
}

static const marmot_device_ops_t metal_ops = {
    // ... existing ops
    .new_op = metal_new_op,
};
```

## Error Handling

### C Code
```c
if (unlikely(condition_failed)) {
    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Details here");
    return MARMOT_ERROR_INVALID_ARGUMENT;
}

// Thread-safe per-thread error state
extern thread_local marmot_error_t marmot_last_error;
extern thread_local char marmot_last_error_detail[256];
```

### Bindings (Planned)
Go bindings are not in the tree yet. For now, error handling is via `marmot_error_t` and `marmot_error_info_t`.

## Debugging

### C Debugging
```bash
# Build with debug symbols
make build-debug

# Run with lldb
lldb build-debug/test_tensor
```

### With Sanitizers
```bash
# Memory errors (leaks, overflows, use-after-free)
make build-asan && make test-asan

# Undefined behavior (null deref, overflow, misaligned access)
make build-ubsan && make test-ubsan

# Thread issues (data races, deadlocks)
make build-tsan && make test-tsan
```


## Performance Tips

### API Overhead
- **Minimize calls**: Batch operations in C
- **Coarse-grained**: Each call should do significant work

### Memory
- **Large buffers**: Allocate in C (via `marmot_tensor_create`)
- **Unified memory**: Use on Apple Silicon when available

### Bytecode
- **Compile once**: Reuse compiled bytecode programs across invocations
- **Avoid recompilation**: Only recompile when the graph structure changes

## CI/CD Integration

Recommended pipeline:
```bash
#!/bin/bash
set -e
make build && make test
make test-ci
make build-release && make test
make build-asan && make test-asan
make build-ubsan && make test-ubsan
```

## Documentation

- **Docs index**: `docs/README.md`
- **Build & codegen**: `docs/getting-started/QUICK_START.md`, `docs/kernels/CODEGEN.md`
- **Graph architecture**: `docs/graph/ARCHITECTURE.md`
- **Dispatch architecture**: `docs/kernels/ARCHITECTURE.md`, `docs/kernels/DISPATCH.md`
- **Graph examples**: `docs/tutorials/SIMPLE_GRAPH.md`
- **Benchmarks**: `docs/getting-started/BENCHMARKING.md`

## When Things Break

1. **Build fails**: Check `meson.build` syntax
2. **Link fails**: Verify all backends implement vtable ops
3. **Sanitizer errors**: Read stack trace carefully
4. **ASAN linker errors**: Script forces system compilers now (fixed)
5. **Missing includes**: Check header dependencies (no circular deps)

## Key Design Principles

1. **No backward compatibility** - Pure C23 only
2. **Type-generic macros** - Use `_Generic` for type-safe APIs
3. **Bytecode-first execution** - Graphs compile to bytecode, not interpreted
4. **Consolidated files** - Minimize file count, clear organization
5. **Coarse-grained APIs** - Big batches, not tiny calls
6. **Backend isolation** - CPU/Metal independent
7. **Zero-copy** - Unified memory, minimal data movement
8. **Self-documenting code** - Minimal comments, clear names
9. **Sanitizer targets** - ASAN/UBSan/TSan available

## Current State

Treat `docs/README.md` as the canonical index for up-to-date architecture and workflow documentation.
