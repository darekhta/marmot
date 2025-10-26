#include "core/helpers/bitpack.h"

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

static void test_sign_extend5_values(void **state) {
    (void)state;

    assert_int_equal(marmot_sign_extend5(0x00), 0);
    assert_int_equal(marmot_sign_extend5(0x0F), 15);
    assert_int_equal(marmot_sign_extend5(0x10), -16);
    assert_int_equal(marmot_sign_extend5(0x1F), -1);
}

static void test_uint5_round_trip(void **state) {
    (void)state;

    enum { kCount = 13, kPackedBytes = MARMOT_PACKED_5BIT_BYTES(kCount) };
    uint8_t values[kCount] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 31};
    uint8_t packed[kPackedBytes];
    uint8_t decoded[kCount] = {0};

    marmot_pack_uint5_block(values, kCount, packed);
    marmot_unpack_uint5_block(packed, kCount, decoded);

    for (size_t i = 0; i < kCount; ++i) {
        assert_int_equal(decoded[i], values[i]);
    }
}

static void test_int5_round_trip(void **state) {
    (void)state;

    enum { kCount = 13, kPackedBytes = MARMOT_PACKED_5BIT_BYTES(kCount) };
    int8_t values[kCount] = {-16, -8, -1, 0, 1, 7, 15, -3, 12, -15, 8, 5, -6};
    uint8_t packed[kPackedBytes];
    int8_t decoded[kCount] = {0};

    marmot_pack_int5_block(values, kCount, packed);
    marmot_unpack_int5_block(packed, kCount, decoded);

    for (size_t i = 0; i < kCount; ++i) {
        assert_int_equal(decoded[i], values[i]);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_sign_extend5_values),
        cmocka_unit_test(test_uint5_round_trip),
        cmocka_unit_test(test_int5_round_trip),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
