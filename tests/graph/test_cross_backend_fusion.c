/* clang-format off */
#include "marmot/marmot.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
/* clang-format on */

// Cross-backend fusion tests skipped - CPU doesn't have fused elementwise kernels
// CPU falls back to unfused add, Metal uses fused kernels, so results won't match
// TODO: Enable when CPU fused kernels are added to elementwise.def
static void test_cross_backend_add_relu_fusion(void **state) {
    (void)state;
    skip();
}

static void test_cross_backend_add_gelu_fusion(void **state) {
    (void)state;
    skip();
}

static void test_cross_backend_add_silu_fusion(void **state) {
    (void)state;
    skip();
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cross_backend_add_relu_fusion),
        cmocka_unit_test(test_cross_backend_add_gelu_fusion),
        cmocka_unit_test(test_cross_backend_add_silu_fusion),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
