#ifndef MARMOT_QUANT_TRAITS_H
#define MARMOT_QUANT_TRAITS_H

#include "macros.h"
#include "types.h"

struct marmot_tensor;

#ifdef __cplusplus
extern "C" {
#endif

typedef marmot_error_t (*marmot_quant_compute_params_fn)(
    const marmot_tensor_t *tensor, marmot_quant_params_t *params_out
);

typedef marmot_error_t (*marmot_quantize_block_fn)(
    const float *values, uint32_t count, void *block_out, const void *params
);

typedef marmot_error_t (*marmot_dequantize_block_fn)(
    const void *block, uint32_t count, float *values_out, const void *params
);

typedef marmot_error_t (*marmot_vec_dot_block_fn)(
    const void *lhs_block, const void *rhs_block, uint32_t count, float *result_out
);

typedef struct marmot_quant_traits {
    marmot_quant_kind_t kind;
    const char *name;
    uint32_t block_size;
    uint32_t block_bytes;
    uint32_t weight_bits;
    bool has_zero_point;
    bool requires_calibration;
    marmot_quant_layout_t layout;
    marmot_quant_compute_params_fn compute_params;
    marmot_quantize_block_fn quantize_block;
    marmot_dequantize_block_fn dequantize_block;
    marmot_vec_dot_block_fn vec_dot_block;
} marmot_quant_traits_t;

MARMOT_NODISCARD const marmot_quant_traits_t *marmot_get_quant_traits(marmot_quant_kind_t kind);
MARMOT_NODISCARD marmot_error_t marmot_quant_register_scheme(const marmot_quant_traits_t *traits);

static inline uint32_t marmot_get_quant_num_blocks(const marmot_quant_traits_t *traits, uint32_t num_values) {
    if (traits == nullptr || traits->block_size == 0) {
        return 0;
    }
    return (num_values + traits->block_size - 1U) / traits->block_size;
}

#define MARMOT_REGISTER_QUANT_SCHEME(traits_name)                                                                      \
    static void marmot_register_##traits_name(void) __attribute__((constructor));                                      \
    static void marmot_register_##traits_name(void) {                                                                  \
        (void)marmot_quant_register_scheme(&(traits_name));                                                            \
    }

#ifdef __cplusplus
}
#endif

#endif // MARMOT_QUANT_TRAITS_H
