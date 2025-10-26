#include "marmot/error.h"
#include "marmot/quant_traits.h"

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

static const marmot_quant_kind_t kTestQuantKind = (marmot_quant_kind_t)(MARMOT_QUANT_KIND_COUNT + 1);

static const marmot_quant_traits_t kTestTraits = {
    .kind = kTestQuantKind,
    .name = "test",
    .block_size = 32,
    .block_bytes = 16,
    .weight_bits = 4,
    .has_zero_point = false,
    .requires_calibration = false,
    .layout = MARMOT_QUANT_LAYOUT_GENERIC,
    .compute_params = nullptr,
    .quantize_block = nullptr,
    .dequantize_block = nullptr,
    .vec_dot_block = nullptr,
};

static void test_register_null_fails(void **state) {
    (void)state;

    marmot_clear_error();
    marmot_error_t err = marmot_quant_register_scheme(nullptr);
    assert_int_equal(err, MARMOT_ERROR_INVALID_ARGUMENT);
    assert_int_equal(marmot_get_last_error(), MARMOT_ERROR_INVALID_ARGUMENT);
}

static void test_register_scheme_succeeds(void **state) {
    (void)state;

    assert_null(marmot_get_quant_traits(kTestQuantKind));

    marmot_clear_error();
    marmot_error_t err = marmot_quant_register_scheme(&kTestTraits);
    assert_int_equal(err, MARMOT_SUCCESS);

    const marmot_quant_traits_t *found = marmot_get_quant_traits(kTestQuantKind);
    assert_ptr_equal(found, &kTestTraits);
}

static void test_register_duplicate_fails(void **state) {
    (void)state;

    marmot_clear_error();
    marmot_error_t err = marmot_quant_register_scheme(&kTestTraits);
    assert_int_equal(err, MARMOT_ERROR_INVALID_ARGUMENT);
    assert_int_equal(marmot_get_last_error(), MARMOT_ERROR_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_register_null_fails),
        cmocka_unit_test(test_register_scheme_succeeds),
        cmocka_unit_test(test_register_duplicate_fails),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
