#include "marmot/device.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>

#include <cmocka.h>
#include <setjmp.h>
#include <string.h>

#include "core/context/context_internal.h"

static void restore_env(const char *saved) {
    if (saved != nullptr) {
        assert_int_equal(setenv("MARMOT_NUM_THREADS", saved, 1), 0);
        return;
    }
    assert_int_equal(unsetenv("MARMOT_NUM_THREADS"), 0);
}

static void test_context_thread_count_control(void **state) {
    (void)state;

    const char *saved_env = getenv("MARMOT_NUM_THREADS");
    char *saved_copy = saved_env != nullptr ? strdup(saved_env) : nullptr;
    assert_true(saved_env == nullptr || saved_copy != nullptr);

    assert_int_equal(unsetenv("MARMOT_NUM_THREADS"), 0);

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    const size_t initial_threads = marmot_context_get_thread_count(ctx);
    assert_true(initial_threads > 0);
    assert_false(marmot_context_thread_count_is_explicit(ctx));
    assert_int_equal(marmot_context_set_thread_count(ctx, 0), MARMOT_ERROR_INVALID_ARGUMENT);

    const size_t auto_threads = initial_threads > 1 ? initial_threads - 1 : initial_threads + 1;
    assert_int_equal(marmot_context_set_thread_count_auto(ctx, auto_threads), MARMOT_SUCCESS);
    assert_int_equal(marmot_context_get_thread_count(ctx), auto_threads);
    assert_false(marmot_context_thread_count_is_explicit(ctx));

    const size_t explicit_threads = auto_threads > 1 ? auto_threads - 1 : auto_threads + 1;
    assert_int_equal(marmot_context_set_thread_count(ctx, explicit_threads), MARMOT_SUCCESS);
    assert_int_equal(marmot_context_get_thread_count(ctx), explicit_threads);
    assert_true(marmot_context_thread_count_is_explicit(ctx));

    marmot_destroy(ctx);
    restore_env(saved_copy);
    free(saved_copy);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_context_thread_count_control),
    };

    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
