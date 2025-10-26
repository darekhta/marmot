#include "marmot/allocator.h"
#include "marmot/config.h"

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

static void test_allocator_cpu_default(void **state) {
    (void)state;

    const marmot_allocator_ops_t *cpu = marmot_get_allocator(MARMOT_BACKEND_CPU);
    assert_non_null(cpu);

    const marmot_allocator_ops_t *fallback = marmot_get_allocator((marmot_backend_type_t)MARMOT_BACKEND_COUNT);
    assert_ptr_equal(fallback, cpu);
}

#if defined(__APPLE__) && MARMOT_ENABLE_METAL
static void test_allocator_metal_selected(void **state) {
    (void)state;

    const marmot_allocator_ops_t *cpu = marmot_get_allocator(MARMOT_BACKEND_CPU);
    const marmot_allocator_ops_t *metal = marmot_get_allocator(MARMOT_BACKEND_METAL);
    assert_non_null(metal);
    assert_true(metal != cpu);
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_allocator_cpu_default),
#if defined(__APPLE__) && MARMOT_ENABLE_METAL
        cmocka_unit_test(test_allocator_metal_selected),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
