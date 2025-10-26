# Kernel Coverage Matrix

Detailed operation x dtype x backend support matrices for all Marmot kernel families.

---

## Legend

- **Y** = Optimized kernel available
- **S** = Scalar fallback only
- **--** = Not supported

---

## Matmul Operations

### Dense Matmul

| Dtype | CPU Accelerate | CPU NEON | CPU AVX2 | CPU Scalar | Metal |
|-------|---------------|----------|----------|------------|-------|
| FLOAT32 | Y | Y | Y | Y | Y |
| FLOAT16 | Y | Y | -- | Y | Y |
| BFLOAT16 | Y | -- | -- | Y | Y |
| FLOAT64 | Y | Y | Y | Y | -- |

Both NN and NT layouts are supported on all backends where the dtype is available.

### Quantized Matmul

| Quant Scheme | CPU (Scalar) | CPU (NEON) | CPU (AVX2) | Metal |
|--------------|-------------|------------|------------|-------|
| Q4_0 | Y | Y | Y | Y |
| Q4_1 | Y | Y | Y | Y |
| Q5_0 | Y | Y | Y | Y |
| Q5_1 | Y | Y | Y | Y |
| Q8_0 | Y | Y | Y | Y |
| Q8_1 | Y | Y | Y | Y |
| Q2_K | Y | Y | Y | Y |
| Q3_K | Y | Y | Y | Y |
| Q4_K | Y | Y | Y | Y |
| Q5_K | Y | Y | Y | Y |
| Q6_K | Y | Y | Y | Y |
| Q8_K | Y | Y | Y | Y |

### Fused Matmul Variants

| Operation | CPU | Metal |
|-----------|-----|-------|
| matmul_bias | Y | Y |
| matmul_bias_relu | Y | Y |
| matmul_bias_gelu | Y | Y |
| matmul_bias_silu | Y | Y |
| matmul_qkv | Y | Y |
| matmul_qkv (quantized) | Y | Y |

---

## Elementwise Binary Operations

### Arithmetic

| Operation | F32 CPU | F32 Metal | F16 CPU | F16 Metal | I32 CPU | I32 Metal |
|-----------|---------|-----------|---------|-----------|---------|-----------|
| add | Y | Y | Y | Y | Y | Y |
| sub | Y | Y | Y | Y | Y | Y |
| mul | Y | Y | Y | Y | Y | Y |
| div | Y | Y | Y | Y | Y | Y |
| min | Y | Y | Y | Y | Y | Y |
| max | Y | Y | Y | Y | Y | Y |
| pow | Y | Y | Y | Y | S | S |
| mod | Y | Y | Y | Y | Y | Y |

CPU profiles: Accelerate (float types), NEON (ARM), AVX2 (x86), Scalar (all platforms).

### Bitwise

| Operation | I32 CPU | I32 Metal | I64 CPU | I64 Metal |
|-----------|---------|-----------|---------|-----------|
| bitwise_and | Y | Y | Y | Y |
| bitwise_or | Y | Y | Y | Y |
| bitwise_xor | Y | Y | Y | Y |
| bitwise_shl | Y | Y | Y | Y |
| bitwise_shr | Y | Y | Y | Y |
| bitwise_shr_logical | Y | Y | Y | Y |

### Comparison

Output dtype is always UINT8 (0 or 1).

| Operation | F32 | F16 | I32 | I64 |
|-----------|-----|-----|-----|-----|
| compare_eq | Y | Y | Y | Y |
| compare_ne | Y | Y | Y | Y |
| compare_lt | Y | Y | Y | Y |
| compare_le | Y | Y | Y | Y |
| compare_gt | Y | Y | Y | Y |
| compare_ge | Y | Y | Y | Y |

Both CPU and Metal support all comparison operations.

### GLU Operations

| Operation | F32 CPU | F32 Metal | F16 CPU | F16 Metal |
|-----------|---------|-----------|---------|-----------|
| swiglu | Y | Y | Y | Y |
| geglu | Y | Y | Y | Y |

---

## Fused Elementwise Operations

| Operation | F32 CPU | F32 Metal | F16 CPU | F16 Metal |
|-----------|---------|-----------|---------|-----------|
| add_relu | Y | Y | Y | Y |
| add_gelu | Y | Y | Y | Y |
| add_silu | Y | Y | Y | Y |
| fma | Y | Y | Y | Y |

---

## Unary Operations

### Activation Functions

| Operation | F32 CPU SIMD | F32 Metal | F16 CPU | F16 Metal |
|-----------|-------------|-----------|---------|-----------|
| relu | Y | Y | Y | Y |
| gelu | Y | Y | Y | Y |
| gelu_tanh | Y | Y | Y | Y |
| silu | Y | Y | Y | Y |
| sigmoid | Y | Y | Y | Y |
| tanh | Y | Y | Y | Y |
| mish | S | Y | S | Y |
| elu | S | Y | S | Y |
| selu | S | Y | S | Y |
| leaky_relu | S | Y | S | Y |
| prelu | S | Y | S | Y |

### Math Functions

| Operation | F32 CPU | F32 Metal | F16 CPU | F16 Metal | I32 CPU |
|-----------|---------|-----------|---------|-----------|---------|
| abs | Y | Y | Y | Y | Y |
| neg | Y | Y | Y | Y | Y |
| sign | Y | Y | Y | Y | Y |
| sqrt | Y | Y | Y | Y | -- |
| exp | Y | Y | Y | Y | -- |
| log | Y | Y | Y | Y | -- |
| bitwise_not | -- | -- | -- | -- | Y |

---

## Reduction Operations

| Operation | F32 CPU | F32 Metal | F16 CPU | F16 Metal |
|-----------|---------|-----------|---------|-----------|
| reduce_sum | Y | Y | Y | Y |
| reduce_mean | Y | Y | Y | Y |
| reduce_max | Y | Y | Y | Y |
| reduce_min | Y | Y | Y | Y |
| reduce_prod | Y | Y | Y | Y |
| reduce_variance | Y | Y | S | S |
| reduce_std | Y | Y | S | S |
| reduce_norm_l1 | Y | Y | Y | Y |
| reduce_norm_l2 | Y | Y | Y | Y |
| reduce_argmax | Y | Y | Y | Y |
| reduce_argmin | Y | Y | Y | Y |
| reduce_any | Y | Y | Y | Y |
| reduce_all | Y | Y | Y | Y |

---

## Normalization Operations

| Operation | F32 CPU | F32 Metal | F16 CPU | F16 Metal |
|-----------|---------|-----------|---------|-----------|
| layernorm | Y | Y | Y | Y |
| rmsnorm | Y | Y | Y | Y |
| rmsnorm_gemma | Y | Y | Y | Y |
| softmax | Y | Y | Y | Y |

All normalization operations support optional residual input on both backends.

---

## Attention Operations

| Operation | CPU | Metal |
|-----------|-----|-------|
| paged_attention | S | Y (flash) |

---

## Embedding Operations

| Operation | F32 CPU | F32 Metal | F16 CPU | F16 Metal |
|-----------|---------|-----------|---------|-----------|
| embedding_lookup | Y | Y | Y | Y |
| embedding_gather | Y | Y | Y | Y |

---

## Positional Encoding

| Operation | F32 CPU | F32 Metal | F16 CPU | F16 Metal |
|-----------|---------|-----------|---------|-----------|
| rope | Y | Y | Y | Y |

Supports scaling modes: none, linear, yarn, longrope. Supports rope types: norm, neox.

---

## Tensor Manipulation

| Operation | CPU | Metal | Notes |
|-----------|-----|-------|-------|
| contiguous | Y | Y | Dense memory copy |
| reshape | Y | Y | Metadata-only (no kernel launch) |
| view | Y | Y | Metadata-only (no kernel launch) |
| transpose | Y | Y | |
| concat | Y | Y | Along specified axis |
| slice | Y | Y | Multi-dimensional |
| gather_rows | Y | Y | Row gather by index |
| scatter_u64_to_i32 | Y | Y | Index conversion scatter |

---

## Conversion Operations

| From -> To | CPU | Metal |
|-----------|-----|-------|
| F32 -> F16 | Y | Y |
| F16 -> F32 | Y | Y |
| F32 -> BF16 | Y | Y |
| BF16 -> F32 | Y | Y |
| F16 -> BF16 | Y | Y |
| BF16 -> F16 | Y | Y |
| F32 -> I32 | Y | Y |
| I32 -> F32 | Y | Y |
| Other combinations | S | S |

---

## Quantization Operations

| Operation | CPU | Metal |
|-----------|-----|-------|
| quantize (all schemes) | Y | Y |
| dequantize (all schemes) | Y | Y |
| vec_dot (quantized) | Y | Y |
| compute_quant_params | Y | -- |

---

## Platform Notes

### FP8 Support
FP8 (FLOAT8_E4M3, FLOAT8_E5M2) is conditional on `MARMOT_ENABLE_FP8`. Primarily used for storage and conversion; compute is done in FP16/FP32.

### BF16 Support
- CPU: Full support via Apple Accelerate on Apple Silicon; scalar fallback elsewhere
- Metal: Native support on Apple Silicon M1+

### INT64 Limitations
Some SIMD operations are limited to INT32. INT64 falls back to scalar on platforms without native 64-bit SIMD.

---

## Adding Missing Coverage

1. Check if kernel exists: `src/backends/<backend>/kernels/*.def`
2. Add definition: add KERNEL_FAMILY entry with the dtype
3. Implement: add implementation in `src/backends/<backend>/ops/`
4. Rebuild: `make build` (codegen runs automatically)
5. Test: add test case in `tests/backend/`

See [Add Kernel Tutorial](../tutorials/ADD_KERNEL.md) for detailed instructions.

---

## See Also

- [Operations Catalog](OPS.md) -- Operation descriptions and signatures
- [Add Kernel Tutorial](../tutorials/ADD_KERNEL.md) -- Adding new coverage
- `src/backends/*/kernels/*.def` -- Kernel definitions (source of truth)
