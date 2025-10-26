#include "marmot/error.h"
#include "marmot/ops_types.h"

#include <stdlib.h>

#include "core/helpers/rope.h"

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

static marmot_rope_params_t make_default_params(void) {
    marmot_rope_params_t params = {0};
    params.theta = 10000.0f;
    params.scaling_type = MARMOT_ROPE_SCALING_NONE;
    return params;
}

static void test_rope_freq_cache_init_sets_defaults(void **state) {
    (void)state;

    marmot_rope_freq_cache_t cache;
    marmot_rope_freq_cache_init(&cache);

    assert_null(cache.freqs);
    assert_int_equal(cache.capacity_pairs, 0);
    assert_int_equal(cache.dim, 0);
    assert_int_equal(cache.scaling_type, MARMOT_ROPE_SCALING_NONE);
    assert_float_equal(cache.freq_scale, 1.0f, 0.0f);
    assert_float_equal(cache.attn_factor, 1.0f, 0.0f);
    assert_float_equal(cache.attn_scale, 1.0f, 0.0f);
    assert_false(cache.owns_storage);
}

static void test_rope_freq_cache_reset_clears_state(void **state) {
    (void)state;

    marmot_rope_freq_cache_t cache;
    marmot_rope_freq_cache_init(&cache);

    marmot_rope_params_t params = make_default_params();
    marmot_rope_freq_span_t span;
    marmot_error_t err = marmot_rope_freq_cache_ensure(&cache, 64, &params, &span);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_non_null(cache.freqs);
    assert_true(cache.owns_storage);

    marmot_rope_freq_cache_reset(&cache);

    assert_null(cache.freqs);
    assert_int_equal(cache.capacity_pairs, 0);
    assert_int_equal(cache.dim, 0);
    assert_false(cache.owns_storage);
    assert_float_equal(cache.freq_scale, 1.0f, 0.0f);
    assert_float_equal(cache.attn_factor, 1.0f, 0.0f);
    assert_float_equal(cache.attn_scale, 1.0f, 0.0f);
}

static void test_rope_freq_cache_ensure_with_cache_returns_borrowed_span(void **state) {
    (void)state;

    marmot_rope_freq_cache_t cache;
    marmot_rope_freq_cache_init(&cache);

    marmot_rope_params_t params = make_default_params();
    marmot_rope_freq_span_t span;
    marmot_error_t err = marmot_rope_freq_cache_ensure(&cache, 64, &params, &span);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_non_null(span.freqs);
    assert_int_equal(span.dim, 64);
    assert_false(span.owns_buffer);

    marmot_rope_freq_cache_destroy(&cache);
}

static void test_rope_freq_cache_ensure_with_null_cache_returns_owned_span(void **state) {
    (void)state;

    marmot_rope_params_t params = make_default_params();
    marmot_rope_freq_span_t span;
    marmot_error_t err = marmot_rope_freq_cache_ensure(nullptr, 64, &params, &span);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_non_null(span.freqs);
    assert_true(span.owns_buffer);

    free((void *)span.freqs);
}

static void test_rope_freq_cache_ensure_rejects_odd_dim(void **state) {
    (void)state;

    marmot_rope_params_t params = make_default_params();
    marmot_rope_freq_span_t span;

    marmot_clear_error();
    marmot_error_t err = marmot_rope_freq_cache_ensure(nullptr, 63, &params, &span);
    assert_int_equal(err, MARMOT_ERROR_INVALID_ARGUMENT);
    assert_int_equal(marmot_get_last_error(), MARMOT_ERROR_INVALID_ARGUMENT);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_rope_freq_cache_init_sets_defaults),
        cmocka_unit_test(test_rope_freq_cache_reset_clears_state),
        cmocka_unit_test(test_rope_freq_cache_ensure_with_cache_returns_borrowed_span),
        cmocka_unit_test(test_rope_freq_cache_ensure_with_null_cache_returns_owned_span),
        cmocka_unit_test(test_rope_freq_cache_ensure_rejects_odd_dim),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
