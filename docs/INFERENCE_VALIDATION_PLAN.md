# Inference Validation Methodology

This document describes Marmot's systematic approach to validating LLM inference correctness. It covers the validation phases, debug tooling, golden test infrastructure, and lessons learned from past debugging.

## Purpose

LLM inference correctness is subtle. Unit tests for individual kernels can pass while end-to-end output is incoherent due to integration bugs in batching, KV cache management, position tracking, or buffer synchronization. This methodology provides a structured way to detect, localize, and fix such issues.

## Ground Rules

1. **Pick a valid reference.** For quantized GGUF models, HuggingFace FP32 is not an exact reference. Prefer a reference runtime that consumes the same GGUF (e.g., `llama.cpp`), or use Marmot self-consistency comparisons (single-seq vs. multi-seq, 1-block vs. multi-block).

2. **Make runs deterministic.** Use greedy decoding with fixed seeds: `temperature=0`, `top_k=0`, `top_p=1`, fixed `seed`. Use `MARMOT_SERVING_ENGINE_FLAG_DETERMINISTIC_KV` when comparing runs.

3. **Bisect by feature toggles.** Narrow bugs by answering yes/no questions: Does it reproduce with `max_seqs=1`? Does it reproduce before crossing a KV `block_size` boundary? Does prefill match but decode diverge?

4. **Stop at first divergence.** Compare tokens (or top-k logits) step by step. Do not debug full text output.

## Validation Phases

### Phase 0: Fast Reproduction and Isolation

Run existing tests first:

- `tests/backend/test_rope.c` -- RoPE kernel correctness
- `tests/backend/test_backend_paged_attention.c` -- paged attention kernel
- `tests/graph/test_graph_paged_attention.c` -- graph-level paged attention
- `tests/inference/test_llm_generate_tokens.c` -- end-to-end deterministic generation
- `tests/inference/test_serving_engine_scheduler.c` -- scheduler logic

If these pass but real workloads are incoherent, the likely culprits are multi-request batching, token_meta/block_table wiring, position/sequence bookkeeping, or prompt/template/sampling.

Run invariance checks that must hold without any external reference:

- **Batch invariance.** Generate tokens for prompt A alone, then generate A while prompt B is present in the same batch. A's tokens must be identical.
- **Block-boundary invariance.** Choose a prompt length that crosses `block_size` (e.g., `block_size=16`, prompt lengths 15 vs. 17). Tokens should not diverge at the boundary.
- **Token-capacity invariance.** Changing `max_num_tokens` must not change the output token stream for a given prompt.

### Phase 1: Golden Data Generation

For GGUF end-to-end correctness (recommended), record greedy tokens from `llama.cpp` for each fixture GGUF and prompt:

```bash
llama-cli -m tinyllama.gguf -p "Hello" -n 10 --temp 0 --log-disable 2>/dev/null
```

Optionally record top-k logits per step for finer-grained comparison.

For algorithm-level debugging (RoPE, attention math), HuggingFace FP32 intermediate tensors can be useful as a math reference, but they are not token-exact for quantized GGUF models.

### Phase 2: Tokenizer Validation

Compare token IDs against a reference tokenizer for the same vocabulary. Verify exact match including BOS/EOS handling. Verify roundtrip: `decode(encode(text)) == text`.

### Phase 3: Embedding Layer Validation

After embedding gather, verify:
- Shape is `[batch, seq_len, hidden_dim]`.
- No NaN or Inf values.
- Values match a reference that uses the same weights and quantization.

### Phase 4: Single Forward Pass (Prefill) Validation

Compare each transformer layer's outputs against a reference, layer by layer:

1. Pre-attention hidden state
2. Q, K, V projections
3. Attention output
4. Post-attention hidden state (after residual)
5. MLP output
6. Post-MLP hidden state (after residual)

The first layer whose output diverges from the reference indicates the bug location.

### Phase 5: Attention Mechanism Validation

For a simple input, manually verify:
- `Q @ K^T / sqrt(d_k)` produces correct attention scores.
- `softmax(scores)` produces attention weights that sum to 1.0 per query.
- Causal mask is enforced (weights are zero for future positions).
- RoPE-encoded Q and K match the reference implementation.

For paged attention specifically:
- Single sequence in one block matches non-paged attention exactly.
- Single sequence spanning multiple blocks reads from correct physical blocks.
- Tokens at block boundaries (e.g., positions 15, 16, 17 for `block_size=16`) have no off-by-one errors.

### Phase 6: KV Cache Validation

After each forward pass, verify:
- KV entries are written to the correct block and offset.
- On decode step N, attention reads all N-1 previous KV entries.
- Block table entries map logical blocks to allocated physical blocks correctly.

### Phase 7: Decode Phase Validation

For each decode step, verify:
- Input is a single new token.
- Position is `[prefill_len + step]`.
- KV cache reads include all previous entries.
- Output is logits for a single new position.

Generate multiple tokens and compare step by step against the reference.

### Phase 8: Sampling Validation

- Greedy sampling (`temperature=0`): the generated token must be `argmax(logits)`.
- Top-K filtering: only top-K tokens have non-zero probability after filtering.
- Top-P (nucleus) filtering: cumulative probability of selected tokens meets the threshold.

### Phase 9: End-to-End Comparison

Compare Marmot's token stream against `llama.cpp` on the same GGUF with deterministic settings:

```bash
# llama.cpp reference
llama-cli -m tinyllama.gguf -p "Hello" -n 10 --temp 0

# marmot
marmot-lm run --raw -p "Hello" -n 10 tinyllama.gguf
```

The first token that differs pinpoints where the divergence starts.

## Debug Environment Variables

### Currently available

| Variable | Purpose |
|----------|---------|
| `MARMOT_GGUF_FIXTURE_DIR` | Override GGUF fixture directory for tests |
| `MARMOT_METAL_SIMDGROUP_MM` | Metal matmul kernel toggle (0 or 1) |
| `MARMOT_METAL_DECODE_SIMD_GROUPS` | Metal decode tuning |
| `MARMOT_GRAPH_TRACE=1` | Print each bytecode instruction during execution |
| `MARMOT_GRAPH_NAN_CHECK=1` | Check for NaN/Inf after every operation |
| `MARMOT_ROUTING=cpu` | Force CPU backend |
| `MARMOT_PROFILE_STEP=1` | Print per-step timing breakdown in serving engine |
| `MARMOT_METAL_TRACE_BATCH=1` | Trace Metal command buffer commits |

### Debugging-specific (enable as needed)

| Variable | Purpose |
|----------|---------|
| `MARMOT_DEBUG_TOKENS=1` | Print tokenized input and special token handling |
| `MARMOT_DEBUG_POSITIONS=1` | Print per-token positions (critical for RoPE debugging) |
| `MARMOT_DEBUG_LAYER=N` | Dump layer N intermediates around QKV/RoPE/attention |
| `MARMOT_DEBUG_LAYER_DUMP` | Select which layer outputs to dump (`attn_in`, `attn_proj`, `post_mlp`, `all`) |
| `MARMOT_DEBUG_LAYER_QKV=1` | Include Q/K/V rows in layer dumps |
| `MARMOT_DEBUG_LAYER_HASH=all` | Hash layer outputs for quick comparison across runs |
| `MARMOT_DEBUG_KV_READ=1` | Log KV block/offset reads for each decode step |
| `MARMOT_DEBUG_LOGITS=1` | Print top-k logits per decode step |
| `MARMOT_DEBUG_LOGITS_TOPK=N` | Number of top logits to print (default: 10) |
| `MARMOT_DEBUG_LOGITS_ROW=N` | Which row to print logits for |
| `MARMOT_CPU_FORCE_SCALAR_MATMUL=1` | Force scalar matmul profile for kernel isolation |
| `MARMOT_DEBUG_ROPE_CONTIGUOUS=1` | Force contiguous RoPE inputs on CPU |

## Golden Test Infrastructure

Marmot maintains golden data from two sources:

1. **llama.cpp** -- quantization parameters, embedding vectors, vec_dot results, and matmul outputs generated by linking against ggml. See `tests/golden/README.md`.

2. **NumPy** -- unary activation functions, fused-bias variants, and graph matmul fixtures. See the generation scripts in `tests/golden/`.

Golden tests compare Marmot's output against these reference values at appropriate tolerances for the data type and quantization scheme.

## Debugging Workflow Summary

1. **Identify the failure level.** Run each phase's tests. The first failure indicates the problem area.

| Phase | Failure indicates |
|-------|-------------------|
| 2 (Tokenizer) | Encoding or decoding bug |
| 3 (Embedding) | Embedding table or gather operation |
| 4 (Prefill) | Transformer layer computation |
| 5 (Attention) | Attention kernel or RoPE |
| 6 (KV Cache) | Block table or cache write/read |
| 7 (Decode) | Position handling or KV read |
| 8 (Sampling) | Logit processing or RNG |

2. **Bisect within the phase.** If prefill fails at layer 5, layers 0-4 match but layer 5 diverges. The bug is in layer 5.

3. **Isolate the component.** Within the failing layer, compare sub-component outputs (pre-attention, Q/K/V, attention, post-attention, MLP, final) to find the first point of divergence.

## Lessons Learned

Several root causes have been identified and fixed through this methodology:

- **Reshape kernel stride mode.** The reshape kernel query only allowed `CONTIGUOUS`, forcing graph layout legalization to insert contiguous copies. This made head layout depend on `seq_len` (i.e., `max_num_tokens`), causing token-capacity invariance failures. Fix: allow reshape to be `STRIDED` in CPU and Metal kernel definitions.

- **Metal buffer synchronization.** The serving engine updated host buffers (token_ids, positions, token_meta, block_table) but did not sync them to the Metal device before graph execution. Metal kernels consumed stale data, producing repeated logits. Fix: explicit `marmot_tensor_to_device` calls for these buffers on GPU backends.

- **Metal command buffer batching.** Some elementwise kernel helpers called `metal_command_stream_flush(ctx, true)`, bypassing the batch depth check and causing 67 command buffer commits per graph execution instead of 1. Fix: changed to `false` so the batch mechanism handles synchronization.

- **Decode graph sizing.** The serving engine always built the packed graph at `max_num_tokens`, so decode-only steps ran a full-size graph with padding. Fix: use a compact graph at the actual `token_count` for decode-only batches.

## Success Criteria

- Deterministic token stream matches `llama.cpp` on a curated prompt set for each supported GGUF fixture.
- Batch invariance: request output is identical when run alone vs. alongside other requests.
- Block-boundary invariance: no first-divergence at `block_size` transitions.
- Decode throughput is stable with increasing context (does not scale like full prefill each step).
