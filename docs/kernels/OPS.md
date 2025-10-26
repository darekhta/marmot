# Operation Catalog

Complete reference for all operations supported by Marmot's kernel system. Each operation includes its signature requirements, supported data types, and backend availability.

---

## Table of Contents

1. [Matmul Operations](#matmul-operations)
2. [Elementwise Binary Operations](#elementwise-binary-operations)
3. [Fused Elementwise Operations](#fused-elementwise-operations)
4. [Elementwise Unary Operations](#elementwise-unary-operations)
5. [Ternary Operations](#ternary-operations)
6. [Reduction Operations](#reduction-operations)
7. [Normalization Operations](#normalization-operations)
8. [Attention Operations](#attention-operations)
9. [Tensor Manipulation Operations](#tensor-manipulation-operations)
10. [Conversion Operations](#conversion-operations)
11. [Quantization Operations](#quantization-operations)
12. [Embedding Operations](#embedding-operations)
13. [Positional Encoding](#positional-encoding)

---

## Matmul Operations

### `matmul`

General matrix multiplication: C = A x B (with layout variants).

| Field | Requirement |
|-------|-------------|
| `op_id` | `MARMOT_OP_MATMUL` |
| `matmul_layout` | `NN`, `NT`, `TN`, or `TT` |
| `input_dtype` | Float types |
| `weight_dtype` | Float types or quantized (requires `qscheme_id`) |
| `output_dtype` | Float types |
| `accum_dtype` | `FLOAT32` or `FLOAT64` |
| `dims.matmul` | N, K, M dimensions |

**Supported dtypes**: FLOAT32, FLOAT16, BFLOAT16, FLOAT64

**Backend support**:
| Backend | Dense | Quantized |
|---------|-------|-----------|
| CPU | ACCELERATE, NEON, AVX2, SCALAR | Q4_0 through Q8_K (all 12 schemes) |
| Metal | Custom shaders | Q4_0 through Q8_K (all 12 schemes) |

### `matmul_bias`

Fused matmul + bias: C = A x B + bias

| Field | Requirement |
|-------|-------------|
| `op_id` | `MARMOT_OP_MATMUL_BIAS` |
| `epilogue_flags` | `MARMOT_EPILOGUE_BIAS` |

Same dtype support as `matmul`.

### `matmul_bias_relu` / `matmul_bias_gelu` / `matmul_bias_silu`

Fused matmul + bias + activation.

| Variant | Formula |
|---------|---------|
| `matmul_bias_relu` | ReLU(A x B + bias) |
| `matmul_bias_gelu` | GELU(A x B + bias) |
| `matmul_bias_silu` | SiLU(A x B + bias) |

### `matmul_qkv`

Fused Q/K/V projection for multi-head attention. Supports both separate and fused weight matrices, with optional quantized weights.

---

## Elementwise Binary Operations

All binary ops: `output[i] = op(a[i], b[i])` with broadcasting support.

### Arithmetic

| Operation | `op_id` | Formula | Dtypes |
|-----------|---------|---------|--------|
| `add` | `MARMOT_OP_ADD` | a + b | All numeric |
| `sub` | `MARMOT_OP_SUB` | a - b | All numeric |
| `mul` | `MARMOT_OP_MUL` | a * b | All numeric |
| `div` | `MARMOT_OP_DIV` | a / b | All numeric |
| `min` | `MARMOT_OP_MIN` | min(a, b) | All numeric |
| `max` | `MARMOT_OP_MAX` | max(a, b) | All numeric |
| `pow` | `MARMOT_OP_POW` | a^b | Float types, I32 (scalar) |
| `mod` | `MARMOT_OP_MOD` | a mod b | All numeric |

### Bitwise

| Operation | `op_id` | Formula | Dtypes |
|-----------|---------|---------|--------|
| `bitwise_and` | `MARMOT_OP_BITWISE_AND` | a & b | Integer types |
| `bitwise_or` | `MARMOT_OP_BITWISE_OR` | a \| b | Integer types |
| `bitwise_xor` | `MARMOT_OP_BITWISE_XOR` | a ^ b | Integer types |
| `bitwise_shl` | `MARMOT_OP_BITWISE_SHL` | a << b | Integer types |
| `bitwise_shr` | `MARMOT_OP_BITWISE_SHR` | a >> b (arithmetic) | Integer types |
| `bitwise_shr_logical` | `MARMOT_OP_BITWISE_SHR_LOGICAL` | a >>> b (logical) | Integer types |

### Comparison

All comparison ops output `UINT8` (0 or 1). Input dtypes: all numeric types.

| Operation | `op_id` | Formula |
|-----------|---------|---------|
| `compare_eq` | `MARMOT_OP_COMPARE_EQ` | a == b |
| `compare_ne` | `MARMOT_OP_COMPARE_NE` | a != b |
| `compare_lt` | `MARMOT_OP_COMPARE_LT` | a < b |
| `compare_le` | `MARMOT_OP_COMPARE_LE` | a <= b |
| `compare_gt` | `MARMOT_OP_COMPARE_GT` | a > b |
| `compare_ge` | `MARMOT_OP_COMPARE_GE` | a >= b |

### GLU Variants

Gated Linear Units for transformer FFN blocks.

| Operation | `op_id` | Formula | Dtypes |
|-----------|---------|---------|--------|
| `swiglu` | `MARMOT_OP_SWIGLU` | a * SiLU(b) | F32, F16, BF16, F64 |
| `geglu` | `MARMOT_OP_GEGLU` | a * GELU(b) | F32, F16, BF16, F64 |

---

## Fused Elementwise Operations

Automatically detected during graph finalization when adjacent ops can be fused.

| Operation | `op_id` | Pattern | Formula |
|-----------|---------|---------|---------|
| `add_relu` | `MARMOT_OP_ADD_RELU` | add -> relu | ReLU(a + b) |
| `add_gelu` | `MARMOT_OP_ADD_GELU` | add -> gelu | GELU(a + b) |
| `add_silu` | `MARMOT_OP_ADD_SILU` | add -> silu | SiLU(a + b) |

Fusion fires when: (1) intermediate tensor has a single consumer, (2) both ops target the same backend, (3) a fused kernel exists for the dtype. See [Fusion](../graph/FUSION.md).

---

## Elementwise Unary Operations

All unary ops: `output[i] = op(input[i])`

### Activation Functions

| Operation | `op_id` | Formula | Notes |
|-----------|---------|---------|-------|
| `relu` | `MARMOT_OP_RELU` | max(0, x) | |
| `gelu` | `MARMOT_OP_GELU` | x * Phi(x) | Exact Gaussian CDF |
| `gelu_tanh` | `MARMOT_OP_GELU_TANH` | 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))) | Fast approximation |
| `silu` | `MARMOT_OP_SILU` | x * sigmoid(x) | Also called Swish |
| `sigmoid` | `MARMOT_OP_SIGMOID` | 1 / (1 + e^(-x)) | |
| `tanh` | `MARMOT_OP_TANH` | (e^x - e^(-x)) / (e^x + e^(-x)) | |
| `mish` | `MARMOT_OP_MISH` | x * tanh(softplus(x)) | |
| `elu` | `MARMOT_OP_ELU` | x if x > 0, else alpha(e^x - 1) | Requires alpha |
| `selu` | `MARMOT_OP_SELU` | lambda * ELU(x, alpha) | Self-normalizing |
| `leaky_relu` | `MARMOT_OP_LEAKY_RELU` | x if x > 0, else alpha*x | Requires alpha |
| `prelu` | `MARMOT_OP_PRELU` | x if x > 0, else alpha*x | Learnable alpha per channel |

**Supported dtypes**: FLOAT32, FLOAT16, BFLOAT16, FLOAT64

### Math Functions

| Operation | `op_id` | Formula | Dtypes |
|-----------|---------|---------|--------|
| `abs` | `MARMOT_OP_ABS` | \|x\| | Float + integer |
| `neg` | `MARMOT_OP_NEG` | -x | Float + integer |
| `sign` | `MARMOT_OP_SIGN` | -1, 0, or 1 | Float + integer |
| `sqrt` | `MARMOT_OP_SQRT` | sqrt(x) | Float only |
| `exp` | `MARMOT_OP_EXP` | e^x | Float only |
| `log` | `MARMOT_OP_LOG` | ln(x) | Float only |
| `bitwise_not` | `MARMOT_OP_BITWISE_NOT` | ~x | Integer only |

---

## Ternary Operations

| Operation | `op_id` | Formula | Notes |
|-----------|---------|---------|-------|
| `fma` | `MARMOT_OP_FMA` | a * b + c | Auto-detected from mul -> add |
| `where` | `MARMOT_OP_WHERE` | cond ? a : b | Condition is UINT8 |

---

## Reduction Operations

All reductions operate along one or more axes. Set `keepdims` to preserve reduced dimensions.

| Operation | `op_id` | Formula | Output dtype |
|-----------|---------|---------|--------------|
| `reduce_sum` | `MARMOT_OP_REDUCTION_SUM` | Sum(x) | Same as input |
| `reduce_mean` | `MARMOT_OP_REDUCTION_MEAN` | Sum(x) / n | Same as input |
| `reduce_max` | `MARMOT_OP_REDUCTION_MAX` | max(x) | Same as input |
| `reduce_min` | `MARMOT_OP_REDUCTION_MIN` | min(x) | Same as input |
| `reduce_prod` | `MARMOT_OP_REDUCTION_PROD` | Product(x) | Same as input |
| `reduce_variance` | `MARMOT_OP_REDUCTION_VARIANCE` | Var(x) | Same as input |
| `reduce_std` | `MARMOT_OP_REDUCTION_STD` | StdDev(x) | Same as input |
| `reduce_norm_l1` | `MARMOT_OP_REDUCTION_NORM_L1` | Sum(\|x\|) | Same as input |
| `reduce_norm_l2` | `MARMOT_OP_REDUCTION_NORM_L2` | sqrt(Sum(x^2)) | Same as input |
| `reduce_argmax` | `MARMOT_OP_REDUCTION_ARGMAX` | argmax(x) | INT64 |
| `reduce_argmin` | `MARMOT_OP_REDUCTION_ARGMIN` | argmin(x) | INT64 |
| `reduce_any` | `MARMOT_OP_REDUCTION_ANY` | Any(x != 0) | UINT8 |
| `reduce_all` | `MARMOT_OP_REDUCTION_ALL` | All(x != 0) | UINT8 |

**Supported input dtypes**: FLOAT32, FLOAT16, BFLOAT16, FLOAT64

---

## Normalization Operations

### `layernorm`

Layer normalization with optional affine transform and residual.

**Formula**: y = gamma * (x - mean) / sqrt(var + epsilon) + beta

**Inputs**: input, gamma (scale), optional beta (bias), optional residual

### `rmsnorm`

Root Mean Square normalization (used in Llama, Qwen, etc.).

**Formula**: y = gamma * x / sqrt(mean(x^2) + epsilon)

**Inputs**: input, gamma (scale), optional residual

### `rmsnorm_gemma`

RMSNorm variant with weight offset (used in Gemma).

**Formula**: y = (1 + gamma) * x / sqrt(mean(x^2) + epsilon)

### `softmax`

Numerically stable softmax along a specified axis.

**Formula**: y[i] = exp(x[i] - max(x)) / Sum(exp(x - max(x)))

**Supported dtypes**: FLOAT32, FLOAT16, BFLOAT16

---

## Attention Operations

### `paged_attention`

Paged attention for efficient KV cache management during LLM inference.

**Inputs**: query, KV cache keys, KV cache values, block table (page indices), token metadata

**Backend support**: Metal (flash attention), CPU (scalar fallback)

---

## Tensor Manipulation Operations

| Operation | `op_id` | Description | Kernel? |
|-----------|---------|-------------|---------|
| `contiguous` | `MARMOT_OP_CONTIGUOUS` | Make tensor contiguous in memory | Yes (data copy) |
| `reshape` | `MARMOT_OP_RESHAPE` | Change shape (same total elements) | No (metadata only) |
| `view` | `MARMOT_OP_VIEW` | Create view with byte offset | No (metadata only) |
| `transpose` | `MARMOT_OP_TRANSPOSE` | Permute dimensions | Yes |
| `concat` | `MARMOT_OP_CONCAT` | Concatenate along axis | Yes |
| `slice` | `MARMOT_OP_SLICE` | Extract subtensor | Yes |
| `gather_rows` | `MARMOT_OP_GATHER_ROWS` | Gather rows by index | Yes |
| `scatter_u64_to_i32` | `MARMOT_OP_SCATTER_U64_TO_I32` | Index conversion scatter | Yes |

---

## Conversion Operations

| Operation | Description |
|-----------|-------------|
| `convert` | Generic dtype conversion (any numeric -> any numeric) |
| `convert_f32_to_f16` | Optimized F32 -> F16 |
| `convert_f16_to_f32` | Optimized F16 -> F32 |
| `convert_f32_to_bf16` | Optimized F32 -> BF16 |
| `convert_bf16_to_f32` | Optimized BF16 -> F32 |
| `convert_f16_to_bf16` | Optimized F16 -> BF16 |
| `convert_bf16_to_f16` | Optimized BF16 -> F16 |

---

## Quantization Operations

| Operation | Description |
|-----------|-------------|
| `quantize` | Quantize float tensor (per scheme) |
| `dequantize` | Dequantize to float |
| `compute_quant_params` | Compute scale and zero-point |
| `vec_dot` | Quantized vector dot product |

**Supported schemes**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K

---

## Embedding Operations

### `embedding_lookup`

Token embedding lookup with optional scaling.

**Inputs**: token indices (INT32/INT64), embedding weight matrix

**Options**: output dtype, scale factor, padding token ID, ragged batch support

### `embedding_gather`

Simplified embedding gather (gather rows from weight matrix by index).

---

## Positional Encoding

### `rope`

Rotary Position Embeddings with configurable parameters.

**Parameters** (`marmot_rope_params_t`):
- `theta` -- Base frequency (default 10000.0)
- `scaling_type` -- none, linear, yarn, longrope
- `rope_type` -- norm, neox
- `freq_scale`, `ext_factor`, `attn_factor` -- Scaling parameters
- `head_dim` -- Head dimension for frequency computation

---

## Backend Summary

| Category | CPU | Metal |
|----------|-----|-------|
| Matmul (dense) | ACCELERATE, NEON, AVX2, SCALAR | Custom shaders |
| Matmul (quantized) | All 12 GGUF schemes | All 12 GGUF schemes |
| Elementwise (binary) | SIMD + scalar | Compute shaders |
| Elementwise (unary) | SIMD + scalar | Compute shaders |
| Reductions | Optimized | Compute shaders |
| Normalization | Optimized | Compute shaders |
| Attention | Scalar | Flash attention |
| RoPE | Optimized | Compute shaders |

---

## See Also

- [Signatures](../graph/SIGNATURES.md) -- Signature field reference for kernel selection
- [Coverage Matrix](COVERAGE.md) -- Detailed dtype x backend matrix
- [Fusion](../graph/FUSION.md) -- Operation fusion system
- `include/marmot/traits_ids.gen.h` -- Generated op IDs
