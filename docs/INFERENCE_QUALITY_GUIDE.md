# Inference Quality Guide

This guide is based on the current serving engine and `marmot-lm` frontend code, not on generic LLM advice. Its purpose is simple: reduce garbage output, reduce "AI slop", and make it obvious when the problem is prompt formatting or stopping instead of model quality.

Relevant code paths:

- `src/inference/frontends/serving_engine.cpp`
- `src/api/llm_api.cpp`
- `apps/marmot-lm/src/embedded/runtime.rs`
- `apps/marmot-lm/src/server/channels/llm.rs`
- `apps/marmot-lm/src/server/rpc/format.rs`
- `apps/marmot-lm/src/template/source.rs`
- `apps/marmot-lm/src/template/stop_strings.rs`

## What Matters Most

For Marmot, low-quality output usually comes from one of four places:

1. Wrong chat template for the model
2. Missing or weak stop conditions
3. Sampling that is too loose for the task
4. Treating a correctness bug like a tuning problem

The scheduler, prefix cache, and batching logic should not change deterministic output. If greedy decoding with a correct template is still incoherent, stop tuning and move to [`INFERENCE_VALIDATION_PLAN.md`](INFERENCE_VALIDATION_PLAN.md).

## What The Code Actually Does

### Prompt formatting

`marmot-lm` formats messages in this order:

1. GGUF chat template, if the model provides one
2. Architecture fallback template, but only if the model name looks instruct-tuned
3. Base-model default template

The default template is intentionally minimal. It emits only the last user message as a completion prompt. That is correct for a base completion model and wrong for most chat/instruct models.

Current fallback families:

- Llama 3 style: Llama 3, generic Llama, modern Mistral fallback
- ChatML: Qwen, Qwen2, Qwen3, Yi, DeepSeek, StarCoder, InternLM
- Gemma: Gemma, Gemma 2
- Phi 3 style: Phi, Phi 2, Phi 3
- Mistral `[INST]` style: Llama 2 family

If a chat model is missing GGUF template metadata and its name does not contain markers like `chat`, `instruct`, `-it`, or `assistant`, Marmot will conservatively treat it like a base model. That is a common route to slop.

### Stopping

The template layer derives stop strings from the active template family. Examples:

- Llama 3: `<|eot_id|>`, `<|end_of_text|>`
- ChatML: `<|im_end|>`
- Gemma: `<end_of_turn>`
- Phi 3: `<|end|>`, `<|user|>`
- Mistral: `</s>`, `[/INST]`

The embedded runtime is stricter than the websocket server:

- Embedded mode converts stop strings to engine `stop_tokens` when they map to a single token, and also adds EOS when available.
- The websocket server currently keeps `GenerateOptions.stop_tokens` empty and stops on decoded text after fragments are emitted.

That means token-level stopping is stronger and cleaner than string-only stopping. When possible, prefer single-token stop markers that match the model template.

### Sampling

Raw C API defaults are conservative:

- `temperature = 0.0`
- `top_k = 0`
- `top_p = 1.0`
- `min_p = 0.0`
- `repetition_penalty = 1.0`

That is effectively greedy decoding.

The engine sampling order is:

1. Build history from prompt plus generated tokens
2. Apply repetition penalty
3. Optionally suppress special tokens
4. Apply temperature
5. Apply `top_k`
6. Apply `top_p`
7. Apply `min_p`
8. Sample

Special-token suppression is quality-critical for non-greedy runs. The engine already exempts EOS and explicit stop tokens, so you can keep suppression enabled without breaking normal end-of-turn behavior.

## Recommended Quality Profiles

### Debugging profile

Use this first when you are not sure whether the problem is quality tuning or engine correctness.

| Setting | Value |
|---------|-------|
| `temperature` | `0.0` |
| `top_k` | `0` |
| `top_p` | `1.0` |
| `min_p` | `0.0` |
| `repetition_penalty` | `1.0` |
| `suppress_special_tokens` | `true` |
| `seed` | fixed, e.g. `42` |

If this profile still produces junk, the next suspects are template selection, tokenizer mismatch, model incompatibility, or an inference bug.

### Low-slop assistant profile

Use this for factual Q&A, coding help, and summarization.

| Setting | Value |
|---------|-------|
| `temperature` | `0.2` to `0.4` |
| `top_k` | `20` to `40` |
| `top_p` | `0.85` to `0.92` |
| `min_p` | `0.0` to `0.05` |
| `repetition_penalty` | `1.05` to `1.10` |
| `suppress_special_tokens` | `true` |
| `max_new_tokens` | keep tight |

This is a better default for "useful answers with less fluff" than the current websocket server defaults.

### Avoid these combinations

- High `temperature` with high `top_p` and non-zero `min_p`
- Aggressive repetition penalty above `1.15` unless you have measured benefit
- Large `max_new_tokens` with weak stops
- Base-model default template for instruct/chat checkpoints

## Frontend-Specific Notes

### Raw C API

The C API starts greedy. You only get loose sampling if you explicitly opt into it.

```c
marmot_llm_sampling_options_t sampling;
marmot_llm_sampling_options_init(&sampling);
sampling.flags |= MARMOT_LLM_SAMPLING_FLAG_SUPPRESS_SPECIAL_TOKENS;
sampling.seed = 42;
sampling.temperature = 0.25f;
sampling.top_k = 40;
sampling.top_p = 0.9f;
sampling.min_p = 0.0f;
sampling.repetition_penalty = 1.05f;
```

### Embedded `marmot-lm`

`SamplingOptions::default()` is looser than the C API:

- `temperature = 0.7`
- `top_k = 40`
- `top_p = 0.9`
- `repetition_penalty = 1.0`
- `suppress_special_tokens = false`

The interactive embedded runtime overrides `suppress_special_tokens` to `true`, but callers using the wrapper directly should do the same for non-greedy runs.

```rust
let sampling = SamplingOptions {
    seed: 42,
    temperature: 0.25,
    top_k: 40,
    top_p: 0.9,
    min_p: 0.0,
    repetition_penalty: 1.05,
    suppress_special_tokens: true,
};
```

### Websocket server

The current server base prediction config is more exploratory:

- `temperature = 0.8`
- `top_k = 40`
- `top_p = 0.95`
- `min_p = 0.05`
- `repeatPenalty = 1.1`

That profile is more likely to produce filler, hedging, and drift on small or mid-size local models. For lower slop, move it closer to the assistant profile above.

Also note that server `stopStrings` are enforced after decode. That can still emit part of a stop marker before cancellation. If you want cleaner truncation, convert single-token stop strings into engine `stop_tokens` before submit.

## Practical Checklist

### Before changing sampling

- Confirm the model has the correct GGUF chat template
- If not, confirm the fallback template family matches the model architecture
- Check whether the model name causes Marmot to choose the instruct fallback or the base-model default template
- Make sure stop strings match the active template

### When changing sampling

- Lower `temperature` first
- Keep `suppress_special_tokens = true`
- Use small `min_p`, or disable it before raising `temperature`
- Add only a mild repetition penalty
- Reduce `max_new_tokens` before adding more penalties

### When output is still wrong

- Run greedy with fixed seed
- Compare solo generation vs batched generation
- Compare prompt formatted with GGUF template vs fallback template
- If greedy output is still incoherent, use [`INFERENCE_VALIDATION_PLAN.md`](INFERENCE_VALIDATION_PLAN.md)

## Recommended Repo Follow-Ups

If the goal is to reduce slop across Marmot itself, these are the highest-leverage changes:

1. Unify frontend defaults around a low-slop assistant profile instead of shipping different sampling personalities.
2. Warn loudly when a known chat architecture falls back to the base-model default template.
3. In the websocket server, convert single-token stop strings into engine `stop_tokens` before submit.
4. Expose explicit quality presets such as `debug`, `assistant`, and `creative`.

Those changes will do more for output quality than scheduler tweaks.
