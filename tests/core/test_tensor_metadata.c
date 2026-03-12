#include "marmot/error.h"
#include "marmot/tensor.h"

// clang-format off
#include <setjmp.h>  // Must be before cmocka.h for jmp_buf
#include <stdarg.h>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
// clang-format on

static marmot_tensor_t
make_quant_tensor(marmot_quant_kind_t kind, const size_t *shape, size_t ndim, marmot_quant_layout_t layout) {
    marmot_tensor_t tensor = {0};
    tensor.shape.ndim = ndim;
    for (size_t i = 0; i < ndim; ++i) {
        tensor.shape.shape[i] = shape[i];
    }
    tensor.quant_kind = kind;
    tensor.quant_layout = layout;
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    assert_non_null(traits);
    tensor.dtype = traits->storage_dtype;
    return tensor;
}

static size_t expected_quant_bytes(marmot_quant_kind_t kind, const size_t *shape, size_t ndim) {
    const marmot_quant_kind_traits_t *traits = marmot_get_quant_kind_traits(kind);
    assert_non_null(traits);
    const size_t block_bytes = traits->header_bytes + traits->payload_bytes;
    size_t inner_dim = 0;
    size_t row_count = 0;
    if (ndim == 2) {
        row_count = shape[0];
        inner_dim = shape[1];
    } else {
        inner_dim = shape[0];
        row_count = 1;
        for (size_t i = 1; i < ndim; ++i) {
            row_count *= shape[i];
        }
    }
    const size_t blocks_per_row = (inner_dim + traits->block_values - 1) / traits->block_values;
    return row_count * blocks_per_row * block_bytes;
}

static void test_quant_storage_bytes_2d_round_up(void **state) {
    (void)state;

    const marmot_quant_kind_t kinds[] = {
        MARMOT_QUANT_KIND_Q4_K,
        MARMOT_QUANT_KIND_Q5_K,
        MARMOT_QUANT_KIND_Q6_K,
        MARMOT_QUANT_KIND_Q8_0,
    };
    const size_t shape[] = {257, 3};

    for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
        marmot_tensor_t tensor = make_quant_tensor(kinds[i], shape, 2, MARMOT_QUANT_LAYOUT_GGUF);
        assert_true(marmot_tensor_is_logical_quant_weight(&tensor));
        const size_t expected = expected_quant_bytes(kinds[i], shape, 2);
        assert_int_equal(marmot_tensor_quant_storage_bytes(&tensor), expected);
        assert_int_equal(marmot_tensor_size_bytes(&tensor), expected);
    }
}

static void test_quant_storage_bytes_3d_uses_inner_dimension(void **state) {
    (void)state;

    const marmot_quant_kind_t kinds[] = {
        MARMOT_QUANT_KIND_Q4_K,
        MARMOT_QUANT_KIND_Q5_K,
        MARMOT_QUANT_KIND_Q6_K,
        MARMOT_QUANT_KIND_Q8_0,
    };
    const size_t shape[] = {257, 3, 5};

    for (size_t i = 0; i < sizeof(kinds) / sizeof(kinds[0]); ++i) {
        marmot_tensor_t tensor = make_quant_tensor(kinds[i], shape, 3, MARMOT_QUANT_LAYOUT_GGUF);
        assert_true(marmot_tensor_is_logical_quant_weight(&tensor));
        const size_t expected = expected_quant_bytes(kinds[i], shape, 3);
        assert_int_equal(marmot_tensor_quant_storage_bytes(&tensor), expected);
        assert_int_equal(marmot_tensor_size_bytes(&tensor), expected);
    }
}

static void test_quant_storage_requires_matching_layout(void **state) {
    (void)state;

    const size_t shape[] = {33, 2, 4};
    marmot_tensor_t tensor = make_quant_tensor(MARMOT_QUANT_KIND_Q8_0, shape, 3, MARMOT_QUANT_LAYOUT_GENERIC);
    assert_false(marmot_tensor_is_logical_quant_weight(&tensor));
    assert_int_equal(marmot_tensor_quant_storage_bytes(&tensor), 0);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_quant_storage_bytes_2d_round_up),
        cmocka_unit_test(test_quant_storage_bytes_3d_uses_inner_dimension),
        cmocka_unit_test(test_quant_storage_requires_matching_layout),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
