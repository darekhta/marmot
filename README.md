<p align="center">
  <img src="assets/logo/marmot-logo-256.png" alt="Marmot" width="200">
</p>

<h1 align="center">Marmot</h1>

<p align="center">
  <strong>LLM inference engine written in C23. CPU and Metal backends. GGUF models out of the box.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License"></a>
  <a href="#supported-models"><img src="https://img.shields.io/badge/models-Llama%20%7C%20Mistral%20%7C%20Qwen%20%7C%20Phi--3%20%7C%20Gemma-green" alt="Supported Models"></a>
  <a href="#quantization-formats"><img src="https://img.shields.io/badge/quantization-12%20GGUF%20formats-orange" alt="Quantization"></a>
</p>

<p align="center">
  <a href="#get-started">Get Started</a> &middot;
  <a href="#c-api">C API</a> &middot;
  <a href="#supported-models">Models</a> &middot;
  <a href="#documentation">Docs</a> &middot;
  <a href="#contributing">Contributing</a>
</p>

---

## Get Started

### Install

```bash
# macOS (Homebrew)
brew tap darekhta/marmot https://github.com/darekhta/marmot
brew install marmot-lm

# From source
brew install meson rust       # or: apt install meson + rustup
git clone https://github.com/darekhta/marmot.git && cd marmot
make build-release && make build-lm-release
make install                  # installs to /usr/local
```

### Run a Model

```bash
marmot-lm pull bartowski/TinyLlama-1.1B-Chat-v1.0-GGUF
marmot-lm run TinyLlama-1.1B-Chat-v1.0 -p "What is the capital of France?"
```

### Start a Server

```bash
marmot-lm serve --port 1234
marmot-lm pull bartowski/Qwen3-0.6B-GGUF --quantization Q4_K_M
marmot-lm run Qwen3-0.6B -p "Hello!" --temperature 0.7
```

### Use the C Library Directly

```bash
make build-release
meson compile -C build-release llama_generate
./build-release/llama_generate --metal /path/to/model.gguf "Once upon a time" 64
```

---

## Why Marmot

- **Modern C23 core** -- `_Generic` type dispatch, `constexpr`, `nullptr`, `_BitInt(N)` for 4-bit quantization. No legacy baggage.
- **Two backends, one API** -- CPU (Accelerate/AVX2/NEON) and Metal (Apple Silicon GPU with simdgroup matmul). Backend is a single enum switch.
- **Bytecode execution** -- Graphs compile once to bytecode programs, then execute many times. No per-token interpretation overhead.
- **GGUF native** -- Load Llama, Mistral, Qwen2, Qwen3, Phi-3, and Gemma models directly. Built-in BPE/WordPiece/Unigram tokenizers from the same file.
- **Production serving** -- Paged KV cache with pool-level management and swap. Continuous batching serving engine. Chat templates from GGUF metadata.
- **107 operations** -- Matmul, attention, RoPE, normalization, 12 GGUF quantization formats, and more. All defined in a kernel DSL with codegen to both backends.

---

## C API

Load a GGUF model, size a packed graph, tokenize, run inference, decode output:

```c
#include "marmot/marmot.h"
#include "marmot/graph/gguf_model.h"
#include "marmot/graph/graph.h"
#include "marmot/tokenizer.h"

int main(void) {
    // 1. Initialize backend
    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_METAL);

    // 2. Load model weights + build a packed execution graph
    marmot_gguf_model_t *model = nullptr;
    marmot_gguf_model_load("model.gguf", MARMOT_BACKEND_METAL, &model);

    marmot_packed_graph_options_t graph_opts;
    marmot_packed_graph_options_init(&graph_opts);
    graph_opts.token_count = 1;
    graph_opts.sample_count = 1;
    graph_opts.max_seqs = 1;
    graph_opts.max_seq_len = 256;
    graph_opts.block_size = 16;
    graph_opts.num_kv_blocks = 16;

    marmot_graph_t *graph = nullptr;
    marmot_graph_from_model_packed(model, MARMOT_BACKEND_METAL, &graph_opts, &graph);

    // 3. Load tokenizer from the same GGUF file
    marmot_tokenizer_options_t tok_opts;
    marmot_tokenizer_options_init(&tok_opts);
    marmot_tokenizer_t *tokenizer = nullptr;
    marmot_tokenizer_create_from_gguf_file("model.gguf", &tok_opts, &tokenizer);

    // 4. Tokenize -> embed -> execute packed graph -> argmax -> decode
    //    See examples/llama_generate.c for the complete working loop.

    // 5. Cleanup
    marmot_graph_destroy(graph);
    marmot_gguf_model_destroy(model);
    marmot_tokenizer_destroy(tokenizer);
    marmot_destroy(ctx);
}
```

See [`examples/llama_generate.c`](examples/llama_generate.c) for the full working implementation.

---

## Architecture

```
                          +------------------+
                          |    Application   |
                          |  (C / Rust CLI)  |
                          +--------+---------+
                                   |
                    +--------------+--------------+
                    |      C API (marmot.h)       |
                    +--------------+--------------+
                                   |
              +--------------------+--------------------+
              |                                         |
    +---------+---------+                  +------------+----------+
    |  Direct Dispatch  |                  |   Graph Execution     |
    |  (single ops)     |                  |   (GGUF load,         |
    +-------------------+                  |    bytecode compile,   |
              |                            |    fusion passes)      |
              |                            +------------+----------+
              |                                         |
              +--------------------+--------------------+
                                   |
                    +--------------+--------------+
                    |   Kernel DSL (.def files)   |
                    |   Codegen -> dispatch tables |
                    +--------------+--------------+
                                   |
              +--------------------+--------------------+
              |                                         |
    +---------+---------+                  +------------+----------+
    |   CPU Backend     |                  |   Metal Backend       |
    |  Accelerate/BLAS  |                  |  Compute shaders      |
    |  AVX2 / NEON      |                  |  simdgroup matmul     |
    |  Scalar fallback  |                  |  Unified memory       |
    +-------------------+                  +-----------------------+
```

**Execution pipeline:** Graph construction -> Bytecode compilation -> Kernel selection via generated tables -> Per-backend bytecode dispatch

---

## Supported Models

| Architecture | Status | Notes |
|---|---|---|
| **Llama** | Working | TinyLlama, Llama 2, Llama 3 |
| **Mistral** | Working | Mistral 7B and compatible GGUF variants |
| **Qwen2** | Working | Qwen 2 family |
| **Qwen3** | Working | QK normalization support |
| **Phi-3** | Working | Phi-3 Mini / Small |
| **Gemma** | Partial | 2B (known numerical stability issues) |

All models load from standard GGUF files. Use `marmot-lm pull` to download from HuggingFace.

### Quantization Formats

12 GGUF block-quantization schemes on both CPU and Metal:

| Format | Bits | Block | | Format | Bits | Block |
|--------|------|-------|-|--------|------|-------|
| Q4_0   | 4    | 32    | | Q2_K   | 2-4  | 256   |
| Q4_1   | 4    | 32    | | Q3_K   | 3-4  | 256   |
| Q5_0   | 5    | 32    | | Q4_K   | 4    | 256   |
| Q5_1   | 5    | 32    | | Q5_K   | 5    | 256   |
| Q8_0   | 8    | 32    | | Q6_K   | 6    | 256   |
| Q8_1   | 8    | 32    | | Q8_K   | 8    | 256   |

---

## Backends

| Backend | Platforms | SIMD / Acceleration |
|---|---|---|
| **CPU** | macOS, Linux (x86, ARM) | Apple Accelerate, AVX2/AVX512, NEON, scalar fallback |
| **Metal** | macOS (Apple Silicon) | Compute shaders, simdgroup matmul, unified memory |

### Apple Silicon

| GPU | Status | Notes |
|---|---|---|
| M4 / M4 Pro / M4 Max | Validated | Optimized simdgroup matmul |
| M3 / M3 Pro / M3 Max | Expected | simdgroup matmul enabled |
| M2 Pro / M2 Max | Expected | simdgroup matmul enabled |
| M2 | Expected | Fallback path |
| M1 / M1 Pro / M1 Max / M1 Ultra | Expected | Fallback path |

---

## marmot-lm CLI

Rust-based application for model management and inference, built against the C library.

| Command | Description |
|---|---|
| `marmot-lm run <model> -p "..."` | Generate text from a prompt |
| `marmot-lm serve [--port 1234]` | Start WebSocket inference server |
| `marmot-lm pull <repo> [--quantization Q4_K_M]` | Download model from HuggingFace |
| `marmot-lm list` | Show locally available models |
| `marmot-lm info <model>` | Display model metadata |
| `marmot-lm ps` | List models loaded on server |
| `marmot-lm unload <model>` | Unload model from server memory |
| `marmot-lm rm <model>` | Delete local model files |
| `marmot-lm stop` | Stop running server |

---

## Operations

107 operations defined in [`src/core/defs/ops.def`](src/core/defs/ops.def):

<details>
<summary>Full operation table</summary>

| Category | Count | Examples |
|---|---|---|
| Binary | 25 | add, sub, mul, div, pow, min, max, logical ops |
| Unary | 19 | abs, neg, exp, log, sqrt, gelu, silu, sigmoid, tanh |
| Quantization | 15 | quantize/dequantize for each GGUF format |
| Reduction | 13 | sum, mean, prod, min, max, argmax, argmin, norm |
| Matmul | 9 | matmul, matmul_transposed, quantized variants, QKV |
| Manipulation | 8 | permute, reshape, slice, concat, pad, gather |
| Conversion | 7 | f32/f16/bf16 dtype casting |
| Normalization | 3 | layernorm, rmsnorm, softmax |
| Ternary | 2 | where, fma |
| Embedding | 2 | embedding_gather, embedding_gather_quantized |
| RoPE | 1 | rotary positional encoding |
| Attention | 1 | scaled dot-product attention (paged) |

</details>

---

## Build

Requires: Meson 1.9+, Clang 19+ (C23 / `c2x`), Rust (for marmot-lm)

| Command | Description |
|---|---|
| `make build` | Debug build (auto-formats) |
| `make build-release` | Optimized release build |
| `make build-lm` | Build marmot-lm CLI (debug) |
| `make build-lm-release` | Build marmot-lm CLI (release) |
| `make test` | Run test suite (56 tests) |
| `make test-lm` | Run marmot-lm e2e tests (7 checks) |
| `make install` | Install to `PREFIX` (default: `/usr/local`) |
| `make format` | Format all C/C++/ObjC++ |
| `make clean-all` | Remove all build artifacts |

<details>
<summary>Sanitizer builds</summary>

| Command | Description |
|---|---|
| `make build-asan` | AddressSanitizer build |
| `make build-ubsan` | UndefinedBehaviorSanitizer build |
| `make build-tsan` | ThreadSanitizer build |
| `make test-asan` | Test under ASAN |
| `make test-ubsan` | Test under UBSan |
| `make test-tsan` | Test under TSan |

</details>

---

## Project Structure

```
marmot/
├── include/marmot/           Public C headers
│   ├── marmot.h              Umbrella header
│   ├── types.h               Dtypes, enums, constants
│   ├── tensor.h              Tensor API with _Generic macros
│   ├── device.h              Backend vtable (~30+ ops)
│   ├── ops/                  Operation headers
│   ├── graph/                Graph execution, GGUF loader
│   └── inference/            Serving engine, KV pool
├── src/
│   ├── core/                 Dispatch, bytecode, tensor helpers
│   │   └── defs/ops.def      107 operation definitions
│   ├── backends/
│   │   ├── cpu/              Accelerate / AVX2 / NEON / scalar
│   │   └── metal/            Metal shaders, compute pipelines
│   ├── graph/                C++ graph runtime, GGUF builder
│   ├── inference/            Serving engine, paged KV cache
│   └── tokenizer/            BPE, WordPiece, Unigram
├── apps/marmot-lm/           Rust CLI and server
├── examples/                 C examples
├── tests/                    90 tests (C + Rust)
├── scripts/codegen/          Kernel DSL -> codegen pipeline
├── tools/marmot-bench/       Benchmarking suite
└── docs/                     Documentation and guides
```

---

## Documentation

| Document | Description |
|---|---|
| [Installation](docs/getting-started/INSTALL.md) | Homebrew, from source, development setup |
| [Quick Start](docs/getting-started/QUICK_START.md) | Build, test, first run |
| [Inference Quality](docs/INFERENCE_QUALITY_GUIDE.md) | Templates, stopping, sampling guidance |
| [Graph Architecture](docs/graph/README.md) | SSA IR, bytecode, fusion passes |
| [Kernel System](docs/kernels/README.md) | DSL, dispatch, codegen pipeline |
| [Kernel DSL Reference](docs/kernels/DSL.md) | `.def` file authoring |
| [API Operations](docs/API_OPS.gen.md) | Generated op reference |
| [Adding a Kernel](docs/tutorials/ADD_KERNEL.md) | Step-by-step tutorial |
| [Building a Graph](docs/tutorials/SIMPLE_GRAPH.md) | Graph construction walkthrough |
| [Benchmarking](docs/getting-started/BENCHMARKING.md) | Performance measurement |

---

## Contributing

```bash
make format                   # Format code
make build && make test       # All 56 C tests must pass
make build-lm && cd apps/marmot-lm && cargo test  # 34 Rust tests
make build-asan && make test-asan   # For memory-sensitive changes
```

- C23 standard (`c2x`) -- use `nullptr`, `constexpr`, `static_assert`, `_Generic`
- No unnecessary comments; self-documenting code
- Pointer alignment: `void *ptr` (not `void* ptr`)
- Update `meson.build` when adding new source files

---

## License

MIT -- see [LICENSE](LICENSE).
