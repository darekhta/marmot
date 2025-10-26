# Operations Utilities (`ops_utils.h`)

`include/marmot/ops_utils.h` provides small, reusable helpers for validating operation inputs and inferring output shapes. These are useful when building graphs or allocating output tensors manually.

All functions return `marmot_error_t` and set the global error detail on failure (`marmot_get_last_error_detail()`).

---

## Shape Inference

### Matmul

`marmot_infer_matmul_output_shape()` implements standard matmul shape rules:
- A: `[..., M, K]`
- B: `[..., K, N]`
- Out: `[..., M, N]` (with broadcastable batch dimensions)

```c
#include <marmot/marmot.h>

int main(void) {
    marmot_shape_t a = {.ndim = 2, .shape = {32, 64}};
    marmot_shape_t b = {.ndim = 2, .shape = {64, 128}};
    marmot_shape_t out = {0};

    marmot_error_t err = marmot_infer_matmul_output_shape(&a, &b, &out);
    if (err != MARMOT_SUCCESS) return 1;

    // out.shape == [32, 128]
    return 0;
}
```

### Linear

`marmot_infer_linear_output_shape()` follows GGUF / `nn.Linear` conventions:
- input: `[N, K]`
- weight: `[M, K]` (stored transposed relative to PyTorch's `[out, in]`)
- output: `[N, M]`

---

## Validation Helpers

Some operations require more than shape math:

### Normalization

`marmot_norm_validate()` checks:
- Feature dimension consistency between input, weight, and optional bias
- Optional residual tensor compatibility
- Epsilon value validity

### Softmax

`marmot_softmax_prepare()` validates and computes:
- Axis bounds checking
- Derived launch parameters (axis_size, inner_stride, outer_size, row_count)

---

## See Also

- [Operations Catalog](../kernels/OPS.md) -- Full operation reference
- [Signatures](../graph/SIGNATURES.md) -- How ops are described to the kernel system
