# GitHub Copilot Instructions for Marmot

## Tech Stack

- **C23 (c2x)** -- Core tensor library (compiled with `-std=c2x`)
- **C++23** -- Graph execution engine (`src/graph/`)
- **Objective-C++** -- Metal/MPS GPU backend (`src/backends/metal/`)
- **Meson** -- Build system
- **Python** -- Code generation (`scripts/codegen/`)
- **Rust** -- `marmot-lm` LLM serving application (`apps/marmot-lm/`)
- **Go** -- Optional API bindings (`pkg/marmot/`)

## Code Conventions

- Standard: C2x (`-std=c2x`), no comments unless critical
- Pointer alignment: `void *ptr` (not `void* ptr`)
- Use `nullptr` (never `NULL`)
- Use `constexpr` for compile-time constants
- Use `static_assert` for compile-time invariants
- Use `thread_local` for thread-safe state
- Use `_Generic` for type-safe function dispatch macros
- Use `[[nodiscard]]`, `[[maybe_unused]]`, `[[likely]]`/`[[unlikely]]` where appropriate
- Format with clang-format: `make format`

## Build and Test

```bash
make build              # Debug build (auto-formats first)
make build-release      # Optimized release build
make test               # Run all tests
make format             # Format code
make format-check       # Check formatting without modifying
make build-asan         # Build with AddressSanitizer
make build-ubsan        # Build with UndefinedBehaviorSanitizer
make test-asan          # Test with AddressSanitizer
make test-ubsan         # Test with UndefinedBehaviorSanitizer
```

## Critical Paths

```
include/marmot/         Public C23 headers (types, tensor, device vtable, ops)
src/core/               Core implementation
  dispatch/             Universal dispatch entry points, fusion detection
  tensor/               Tensor helpers (layout, contiguity)
  ops/                  Operation definitions, generated wrappers, APIs
  defs/                 Operation DSL definitions (.def)
src/backends/
  cpu/                  CPU backend (SIMD: AVX2, NEON, Accelerate)
    ops/                Per-operation directories with arch variants
    kernels/            Kernel definitions (.def)
    quantization/       GGUF quantization (Q4K, Q5K, Q6K, Q8)
  metal/                Metal/MPS GPU backend
    ops/                Metal operation implementations
    shaders/            Metal shader code (.metal)
    kernels/            Kernel definitions (.def)
src/graph/              C++ graph execution engine
src/inference/          LLM serving engine (C++)
src/tokenizer/          Tokenizer implementations (BPE, WordPiece, Unigram)
apps/marmot-lm/         Rust-based LLM serving application
tests/                  C test suites (backend, graph, golden, codegen)
scripts/codegen/        Python code generation tooling
```

## Code Generation

Kernel definitions use a `.def` DSL processed by `scripts/codegen/gen_kernels.py`:

- Input: `.def` files in `src/backends/cpu/kernels/` and `src/backends/metal/kernels/`
- Output: `.gen.c`, `.gen.mm`, `.gen.h` files (dispatch tables, trait maps, kernel queries)
- Templates: Jinja2 templates in `scripts/codegen/templates/`

**Do NOT edit files with `.gen.` in their name.** These are generated artifacts. Modify the `.def` source or the codegen templates instead.

## Common Tasks

**Add a new C function:**
1. Declare in `include/marmot/*.h`
2. Implement in `src/core/`
3. Update `meson.build` if adding a new source file
4. Run `make build && make test`

**Add a backend operation:**
1. Add to the vtable in `include/marmot/device.h`
2. Implement in both `src/backends/cpu/` and `src/backends/metal/`
3. Add tests in `tests/backend/`
4. Run `make build && make test`

**Run with sanitizers for memory-sensitive changes:**
```bash
make build-asan && make test-asan
make build-ubsan && make test-ubsan
```

## References

- `.ai/DEVELOPMENT.md` -- Full development guide
- `docs/BUILD_SYSTEM.md` -- Build configuration details
- `docs/kernels/CODEGEN.md` -- Codegen and build integration
- `docs/kernels/DSL.md` -- Kernel DSL authoring and expansion
- `docs/graph/README.md` -- Graph module documentation
