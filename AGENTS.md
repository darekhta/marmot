# Marmot Project Rules for AI Assistants

## Code Style
- C23 (c2x) standard required
- No unnecessary comments - code should be self-documenting
- Use nullptr, static_assert, thread_local
- Clean, minimal code
- Format with clang-format: `make format`
- Pointer alignment: `void *ptr` (not `void* ptr`)

## C23 Features in Use
Marmot uses modern C23 features for safer, clearer code:
- **nullptr**: Type-safe null pointer (not NULL macro)
- **constexpr**: Compile-time constants for variables (e.g., `MARMOT_CACHE_LINE_SIZE`)
- **static_assert**: Compile-time checks for type sizes and invariants
- **thread_local**: Thread-safe error state (`marmot_last_error`)
- **typeof / typeof_unqual**: Type deduction in generic macros
- **_BitInt(N)**: Bit-precise integers for 4-bit quantization types
- **[[nodiscard]]**: Warn on ignored return values (errors, allocations)
- **[[maybe_unused]]**: Suppress warnings for platform-specific code
- **[[likely]] / [[unlikely]]**: Branch prediction hints for optimization
- **_Generic**: Type-safe function dispatch (e.g., `marmot_tensor_full`)

Note: We use `c2x` (C23 draft) as the `c23` keyword is not yet supported by Clang 21/Meson 1.9.

## Build Commands
- `make build` - debug build (auto-formats first)
- `make build-release` - optimized release build
- `make build-lm` - build marmot-lm CLI (debug, requires libmarmot)
- `make build-lm-release` - build marmot-lm CLI (release)
- `make test` - run all tests
- `make test-lm` - run marmot-lm e2e tests
- `make build-asan` - build with AddressSanitizer
- `make build-ubsan` - build with UndefinedBehaviorSanitizer
- `make build-tsan` - build with ThreadSanitizer
- `make test-asan` / `make test-ubsan` / `make test-tsan` - test with sanitizers
- `make install` - install libmarmot + marmot-lm to PREFIX (default: /usr/local)
- `make clean-all` - clean everything (including marmot-lm cargo artifacts)

## File Locations

### Headers (`include/marmot/`)
- **types.h** - Foundation types, enums, constexpr definitions
- **error.h** - Thread-safe error handling
- **macros.h** - General utilities (MIN, MAX, SWAP, alignment)
- **tensor.h** - Tensor API with `_Generic` macros
- **device.h** - Backend vtable interface (~30+ operations)
- **allocator.h** - Memory allocator interface
- **config.h** - Configuration settings
- **marmot.h** - Umbrella header
- **ops/*.h** - Operation headers (attention, matmul, elementwise, reduction, neural, etc.)
- **graph/*.h** - Graph execution headers (graph.h, kernel_selection.h, gguf_loader.h, etc.)
- **inference/*.h** - Inference headers (engine.h, llm.h, kv_pool.h, model.h)
- **inference.h** - Top-level inference umbrella header
- **tokenizer.h** - Tokenizer API (BPE, WordPiece, Unigram)
- **quant_*.h** - Quantization trait headers

### Core Implementation (`src/core/`)
- **context/** - Context init/destroy, routing policy, fast ops selection
- **dispatch/** - Universal dispatch entry points, fusion detection, dispatch helpers
- **bytecode/** - Bytecode compilation and execution (compile-once execute-many)
- **ops/** - Operation DSL (`defs/`), generated wrappers (`generated/`), public APIs (`api/`), internal implementations (`impl/`), shared helpers (`common/`)
- **tensor/** - Tensor helpers (layout, contiguity, utilities)
- **error.c** - Thread-local error state
- **tensor.c** - Consolidated tensor operations

### Backends (`src/backends/`)

**CPU Backend** (`src/backends/cpu/`):
- **dispatch/** - Device ops implementation, bytecode execution (`bytecode_exec_cpu.gen.c`)
- **ops/** - SIMD-optimized operations (18 subdirectories)
  - Per-op directories with architecture variants: `accelerate/`, `avx2/`, `neon/`, `scalar/`
- **kernels/** - Kernel definition files (.def format)
- **quantization/** - GGUF quantization schemes (Q4K, Q5K, Q6K, Q8, etc.)
- **generated/** - Codegen outputs (kernel query/dispatch)

**Metal Backend** (`src/backends/metal/`):
- **metal_backend.mm** - Main Metal backend
- **metal_memory.mm** - Unified memory management
- **ops/** - Metal operation implementations, bytecode execution (`bytecode_exec_metal.gen.mm`)
- **kernels/** - Metal kernel definitions (.def format)
- **shaders/** - Metal shader code
- **generated/** - Codegen outputs (Objective-C++ implementations)
- **quantization/** - Metal quantization implementations

**Common Backend Code** (`src/core/ops/common/`):
- Shared patterns for elementwise, matmul, normalization, reduction, unary ops

### Graph Execution (`src/graph/`)
C++ implementation for computational graphs:
- **graph_impl.hpp**, **graph_executor.hpp** - Core graph execution
- **kernel_query.hpp** - Kernel selection system
- **execution_session.hpp** - Session management for compiled graphs
- **gguf/** - GGUF model loading, graph building, architecture definitions (Llama, Mistral, Qwen2, Qwen3, Phi-3, Gemma)

### Inference (`src/inference/`)
C++ inference runtime:
- **model/** - Model loading and state management
- **frontends/** - Serving engine with continuous batching
- **kv_pool/** - Paged KV cache pool with swap support
- **common/** - Shared storage and tensor pointer utilities

### Tokenizer (`src/tokenizer/`)
C++ tokenizer implementations:
- **tokenizer.hpp/cpp** - Core tokenizer interface
- **model_bpe.hpp/cpp** - Byte-pair encoding
- **model_wordpiece.hpp/cpp** - WordPiece tokenization
- **model_unigram.hpp/cpp** - Unigram (SentencePiece) tokenization
- **vocab.hpp/cpp** - Vocabulary management
- **utf8.hpp/cpp** - UTF-8 utilities

### Applications (`apps/`)
- **marmot-lm/** - Rust-based LLM serving application (Cargo project, builds against C library)

### Tests (`tests/`)
- **backend/** - Backend operation tests (29+ test files)
- **graph/** - Graph execution tests (8 test files)
- **inference/** - Inference tests (KV pool, LLM generation, serving engine scheduler)
- **golden/** - Golden tests with NumPy baselines
- **codegen/** - Code generation tests
- **fixtures/**, **utils/** - Test infrastructure

### Scripts (`scripts/`)
- **build.sh**, **test.sh** - Main build/test scripts
- **format.sh**, **format-check.sh** - Code formatting
- **codegen/** - Kernel code generation
  - **gen_kernels.py** - Main codegen from .def files
  - **def_parser.py** - Parser for kernel definition language
  - **templates/** - Jinja2 templates for C, Objective-C++, Metal

### Documentation (`docs/`)
- **BUILD_SYSTEM.md** - Build configuration details
- **API.md** - API reference
- **kernels/** - Kernel system docs (DSL, dispatch, codegen, coverage)
- **graph/** - Graph module docs (architecture, fusion, signatures, environment)
- **tutorials/** - Step-by-step guides (SIMPLE_GRAPH.md, ADD_KERNEL.md)
- **getting-started/** - Quick start guide
- **awq/** - AWQ quantization planning documents

## When Modifying
1. Update `meson.build` for new C files
2. Format code: `make format`
3. Run `make build && make test` before committing. **All suites must pass**---do not claim success (or move on to new work) if any test fails, even if the failure seems unrelated to your change.
4. Use sanitizers for memory-sensitive code: `make build-asan && make test-asan`

## Architecture

### Core Design
- C core for performance (BLAS/Metal/MPS)
- Bytecode dispatch as primary execution model (graphs compile to bytecode programs, then execute via generated per-backend dispatch)
- Vtable-based backends (CPU, Metal, future: CUDA)
- Paged KV cache with pool-level management and swap support
- Serving engine with continuous batching for concurrent request handling
- Tokenizer supporting BPE, WordPiece, and Unigram models
- Go bindings planned (no Go code in tree yet)

### Execution Pipeline
1. **Graph construction** - operations added to a DAG via the graph API or GGUF loader
2. **Bytecode compilation** - graph is compiled into a linear bytecode program (`src/core/bytecode/`)
3. **Kernel selection** - each bytecode op is matched to a backend kernel via generated query tables
4. **Bytecode execution** - the compiled program runs through generated dispatch (`bytecode_exec_cpu.gen.c`, `bytecode_exec_metal.gen.mm`)

### Backend Vtable Pattern
The `marmot_device_ops` structure defines the backend interface:
- Memory: alloc, free, memcpy_to_device, memcpy_from_device, sync
- Tensor: matmul, matmul_quantized, matmul_qkv, embedding_gather, vec_dot
- Neural: layernorm, rmsnorm, softmax, attention (including paged attention)
- Element-wise: binary, ternary, unary, reduction

### Kernel Definition System
- `.def` files define kernel signatures in a domain-specific language
- `gen_kernels.py` generates dispatch tables, bytecode execution, trait maps, and implementations
- Supports multiple backends from single definition

### Supported Model Architectures
- Llama, Mistral, Qwen2, Qwen3, Phi-3, Gemma (via GGUF loader in `src/graph/gguf/architecture.cpp`)

### API Naming Conventions
- `marmot_get_*`, `marmot_is_*`, `marmot_has_*` - Cheap metadata queries (O(1))
- `marmot_compute_*` - Expensive operations (may allocate, dispatch to backend)
- Action verbs (`marmot_matmul`, `marmot_quantize`) - Direct operations

## Read First
- `docs/README.md` - Documentation index
- `.ai/DEVELOPMENT.md` - Full development guide
- `docs/getting-started/QUICK_START.md` - Build/test basics
- `docs/kernels/CODEGEN.md` - Codegen + build integration
- `docs/kernels/DSL.md` - Kernel DSL (authoring + expansion)
- `docs/graph/README.md` - Graph module documentation
- `docs/kernels/README.md` - Kernel system documentation
