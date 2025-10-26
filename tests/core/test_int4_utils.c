#include "core/helpers/int4.h"

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

static void test_pack_unpack_int4_pair(void **state) {
    (void)state;

    int8_t x = -7;
    int8_t y = 7;
    uint8_t packed = marmot_pack_int4(x, y);

    int8_t out_x = 0;
    int8_t out_y = 0;
    marmot_unpack_int4(packed, &out_x, &out_y);

    assert_int_equal(out_x, x);
    assert_int_equal(out_y, y);
}

static void test_pack_unpack_uint4_pair(void **state) {
    (void)state;

    uint8_t x = 0;
    uint8_t y = 15;
    uint8_t packed = marmot_pack_uint4(x, y);

    uint8_t out_x = 0;
    uint8_t out_y = 0;
    marmot_unpack_uint4(packed, &out_x, &out_y);

    assert_int_equal(out_x, x);
    assert_int_equal(out_y, y);
}

static void test_int4_block_round_trip(void **state) {
    (void)state;

    enum { kCount = 9, kPackedBytes = (kCount + 1) / 2 };
    int8_t values[kCount] = {-7, -1, 0, 1, 7, 3, -3, 5, -5};
    uint8_t packed[kPackedBytes];
    int8_t decoded[kCount] = {0};

    marmot_pack_int4_block(values, kCount, packed);
    marmot_unpack_int4_block(packed, kCount, decoded);

    for (size_t i = 0; i < kCount; ++i) {
        assert_int_equal(decoded[i], values[i]);
    }
}

static void test_uint4_block_round_trip(void **state) {
    (void)state;

    enum { kCount = 9, kPackedBytes = (kCount + 1) / 2 };
    uint8_t values[kCount] = {0, 1, 7, 15, 3, 5, 9, 2, 12};
    uint8_t packed[kPackedBytes];
    uint8_t decoded[kCount] = {0};

    marmot_pack_uint4_block(values, kCount, packed);
    marmot_unpack_uint4_block(packed, kCount, decoded);

    for (size_t i = 0; i < kCount; ++i) {
        assert_int_equal(decoded[i], values[i]);
    }
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_pack_unpack_int4_pair),
        cmocka_unit_test(test_pack_unpack_uint4_pair),
        cmocka_unit_test(test_int4_block_round_trip),
        cmocka_unit_test(test_uint4_block_round_trip),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
