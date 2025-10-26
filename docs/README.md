# Marmot Documentation

Marmot is a high-performance tensor computation and LLM inference framework built in C23 with dual-backend execution (CPU and Metal). This documentation covers everything from getting started to deep architectural internals.

---

## Quick Navigation

| I want to...                    | Start here                                      |
|---------------------------------|-------------------------------------------------|
| Install Marmot / marmot-lm      | [Installation](getting-started/INSTALL.md)       |
| Build and run Marmot            | [Quick Start](getting-started/QUICK_START.md)   |
| Improve LLM output quality      | [Inference Quality Guide](INFERENCE_QUALITY_GUIDE.md) |
| Build a computational graph     | [Simple Graph Tutorial](tutorials/SIMPLE_GRAPH.md) |
| Load and run a GGUF model       | [GGUF Loading](graph/GGUF.md)                   |
| Add a new kernel implementation | [Add Kernel Tutorial](tutorials/ADD_KERNEL.md)   |
| Understand the kernel DSL       | [Kernel DSL Reference](kernels/DSL.md)           |
| Debug graph execution           | [Graph Debugging](graph/DEBUGGING.md)            |
| Benchmark operations            | [Benchmarking Guide](getting-started/BENCHMARKING.md) |
| Browse all operations           | [Operations Reference](API_OPS.gen.md)           |

---

## Documentation Map

### Getting Started

- **[Installation](getting-started/INSTALL.md)** -- Homebrew, from source, system install, development setup
- **[Quick Start](getting-started/QUICK_START.md)** -- Build, test, and run your first graph in 5 minutes
- **[Operations Utilities](getting-started/OPS_UTILS.md)** -- Shape inference and validation helpers
- **[Benchmarking](getting-started/BENCHMARKING.md)** -- Micro and composite benchmarks with marmot-bench

### Architecture and Design

- **[Bytecode Dispatch](BYTECODE_DISPATCH.md)** -- The compile-once execute-many execution model
- **[Metal Performance](METAL_PERFORMANCE_OPTIMIZATION.md)** -- GPU optimization strategies and profiling

### Kernel System

- **[Kernel Overview](kernels/README.md)** -- Entry point to the kernel subsystem
- **[Architecture](kernels/ARCHITECTURE.md)** -- Signatures, kernel selection, and dispatch flow
- **[DSL Reference](kernels/DSL.md)** -- The `.def` kernel definition language
- **[Code Generation](kernels/CODEGEN.md)** -- How codegen produces dispatch tables and bytecode interpreters
- **[Dispatch](kernels/DISPATCH.md)** -- Unified bytecode dispatch for graphs and the C API
- **[Coverage Matrix](kernels/COVERAGE.md)** -- Operation x dtype x backend support
- **[Operations Catalog](kernels/OPS.md)** -- Complete operation reference with signatures
- **[Debugging](kernels/DEBUGGING.md)** -- Kernel inspection and troubleshooting
- **[Cost Model](kernels/COST_MODEL.md)** -- Backend selection and fusion profitability

### Graph Execution

- **[Graph Overview](graph/README.md)** -- Entry point to the graph subsystem
- **[Architecture](graph/ARCHITECTURE.md)** -- SSA IR, finalization pipeline, bytecode compilation
- **[C API Reference](graph/API.md)** -- Graph lifecycle, build, finalize, execute
- **[Fusion](graph/FUSION.md)** -- Operation fusion patterns and profitability
- **[GGUF Loading](graph/GGUF.md)** -- Model loading for Llama, Mistral, Qwen2, Qwen3, Phi-3, Gemma
- **[Signatures](graph/SIGNATURES.md)** -- `marmot_op_signature_t` field reference
- **[Environment Variables](graph/ENVIRONMENT.md)** -- Runtime tuning and debug knobs
- **[Debugging](graph/DEBUGGING.md)** -- Troubleshooting graph execution issues

### Tutorials

- **[Simple Graph](tutorials/SIMPLE_GRAPH.md)** -- Build and execute a matmul + relu graph step by step
- **[Add a Kernel](tutorials/ADD_KERNEL.md)** -- Add a new kernel variant to the CPU backend

### Reference

- **[Operations Reference](API_OPS.gen.md)** -- Auto-generated catalog of all public C API operations (regenerated on each build)

### Planning and Validation

- **[Inference Quality Guide](INFERENCE_QUALITY_GUIDE.md)** -- Practical defaults and failure modes for reducing garbage output and AI slop
- **[Inference Validation](INFERENCE_VALIDATION_PLAN.md)** -- Systematic LLM correctness validation methodology
- **[Adoption Strategy](DEVELOPER_ADOPTION_STRATEGY.md)** -- Developer positioning and feature prioritization

---

## Recommended Reading Order

**New user:** Quick Start -> Simple Graph Tutorial -> Graph API -> GGUF Loading

**Kernel developer:** Kernel Overview -> DSL Reference -> Code Generation -> Add Kernel Tutorial -> Coverage Matrix

**Graph user:** Graph Overview -> Architecture -> Signatures -> Fusion -> Environment Variables

**Contributor:** Quick Start -> Bytecode Dispatch -> Kernel Architecture -> Graph Architecture -> all Roadmaps
