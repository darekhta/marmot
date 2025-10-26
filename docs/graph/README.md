# Graph Module

The graph module is Marmot's execution-oriented layer for building, compiling, and running computational graphs. Implemented in C++ (`src/graph/`) with a stable C API (`include/marmot/graph/`), it provides an SSA-style intermediate representation, a 4-pass finalization pipeline (signature population, fusion detection, kernel selection, and bytecode compilation), and a bytecode executor with cached sessions. The module includes a GGUF loader that supports five model architectures out of the box, making it the primary entry point for LLM inference.

---

## Learning Path

1. [Quick Start](../getting-started/QUICK_START.md) -- 5-minute onboarding, build and run.
2. [Simple Graph Tutorial](../tutorials/SIMPLE_GRAPH.md) -- Build a graph step by step.
3. [Architecture](ARCHITECTURE.md) -- Internal representation and execution pipeline.
4. [C API Reference](API.md) -- Full public API documentation.

---

## Documentation Index

### Core

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | SSA IR, finalization pipeline, bytecode execution, view caching |
| [API.md](API.md) | Public C API: lifecycle, build, finalize, execute, debug |
| [FUSION.md](FUSION.md) | Operation fusion patterns, profitability, and bytecode interaction |
| [GGUF.md](GGUF.md) | GGUF loader, supported architectures, quantization formats |
| [SIGNATURES.md](SIGNATURES.md) | `marmot_op_signature_t` field reference with examples |

### Reference

| Document | Description |
|----------|-------------|
| [ENVIRONMENT.md](ENVIRONMENT.md) | Runtime environment variables for tracing, routing, and tuning |
| [DEBUGGING.md](DEBUGGING.md) | Troubleshooting guide: diagnostics, common issues, sanitizers |

### Related

| Document | Description |
|----------|-------------|
| [Kernel Operations](../kernels/OPS.md) | All supported operations across backends |
| [Kernel Dispatch](../kernels/DISPATCH.md) | Kernel selection and dispatch mechanics |
| [Kernel Coverage](../kernels/COVERAGE.md) | Op x dtype x backend support matrix |
