#include "marmot/allocator.h"
#include "marmot/config.h"

#include <math.h>

#include "backend/test_backend_utils.h"

static void run_memory_checks(marmot_test_env_t *env) {
    assert_non_null(env);
    assert_non_null(env->ctx);
    assert_non_null(env->ctx->ops);
    assert_non_null(env->ctx->ops->alloc);
    assert_non_null(env->ctx->ops->free);
    assert_non_null(env->ctx->ops->memcpy_to_device);
    assert_non_null(env->ctx->ops->memcpy_from_device);

    // Basic allocate/copy/free
    void *device_ptr = nullptr;
    marmot_error_t err = env->ctx->ops->alloc(env->ctx->device_ctx, 128, &device_ptr);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_non_null(device_ptr);

    uint8_t host_src[128];
    uint8_t host_dst[128];
    for (size_t i = 0; i < sizeof(host_src); ++i) {
        host_src[i] = (uint8_t)(i ^ 0x5A);
        host_dst[i] = 0U;
    }

    err = env->ctx->ops->memcpy_to_device(env->ctx->device_ctx, device_ptr, host_src, sizeof(host_src));
    assert_int_equal(err, MARMOT_SUCCESS);

    err = env->ctx->ops->memcpy_from_device(env->ctx->device_ctx, host_dst, device_ptr, sizeof(host_dst));
    assert_int_equal(err, MARMOT_SUCCESS);

    for (size_t i = 0; i < sizeof(host_src); ++i) {
        assert_int_equal(host_dst[i], host_src[i]);
    }

    env->ctx->ops->free(env->ctx->device_ctx, device_ptr);

    // High-level helpers should round-trip through the backend without exposing memcpy details.
    const size_t helper_shape[] = {8};
    marmot_tensor_t *helper_tensor = marmot_tensor_create(env->ctx, helper_shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(helper_tensor);

    float helper_src[8];
    for (size_t i = 0; i < ARRAY_LENGTH(helper_src); ++i) {
        helper_src[i] = (float)i * 1.25f - 3.0f;
    }
    assert_int_equal(
        marmot_tensor_copy_from_host_buffer(env->ctx, helper_tensor, helper_src, sizeof(helper_src)), MARMOT_SUCCESS
    );
    assert_int_equal(marmot_tensor_to_device(env->ctx, helper_tensor), MARMOT_SUCCESS);

    float helper_dst[8] = {0.f};
    assert_int_equal(
        marmot_tensor_copy_to_host_buffer(env->ctx, helper_tensor, helper_dst, sizeof(helper_dst)), MARMOT_SUCCESS
    );
    for (size_t i = 0; i < ARRAY_LENGTH(helper_dst); ++i) {
        assert_true(fabsf(helper_src[i] - helper_dst[i]) <= 1e-6f);
    }

    marmot_tensor_destroy(helper_tensor);

    // Zero-size allocation should succeed and be freeable.
    void *zero_ptr = nullptr;
    err = env->ctx->ops->alloc(env->ctx->device_ctx, 0, &zero_ptr);
    assert_int_equal(err, MARMOT_SUCCESS);
    env->ctx->ops->free(env->ctx->device_ctx, zero_ptr);

    // Multiple sequential allocations to ensure we don't leak bookkeeping.
    void *blocks[4] = {nullptr, nullptr, nullptr, nullptr};
    for (size_t i = 0; i < 4; ++i) {
        err = env->ctx->ops->alloc(env->ctx->device_ctx, 64 + i * 16, &blocks[i]);
        assert_int_equal(err, MARMOT_SUCCESS);
        assert_non_null(blocks[i]);
    }
    for (size_t i = 0; i < 4; ++i) {
        env->ctx->ops->free(env->ctx->device_ctx, blocks[i]);
    }

    marmot_allocator_usage_t usage_before = {0};
    assert_int_equal(marmot_allocator_get_usage(env->ctx, &usage_before), MARMOT_SUCCESS);

    void *usage_probe = nullptr;
    err = env->ctx->ops->alloc(env->ctx->device_ctx, 256, &usage_probe);
    assert_int_equal(err, MARMOT_SUCCESS);
    assert_non_null(usage_probe);

    marmot_allocator_usage_t usage_after = {0};
    assert_int_equal(marmot_allocator_get_usage(env->ctx, &usage_after), MARMOT_SUCCESS);
    assert_true(usage_after.current_bytes >= usage_before.current_bytes);
    assert_true(usage_after.peak_bytes >= usage_before.peak_bytes);
    assert_true(usage_after.active_allocations >= 1);
    assert_true(usage_after.peak_allocations >= usage_after.active_allocations);

    uint64_t after_pool_events = usage_after.pool_hits + usage_after.pool_misses;
    uint64_t before_pool_events = usage_before.pool_hits + usage_before.pool_misses;
    assert_true(after_pool_events >= before_pool_events);

    env->ctx->ops->free(env->ctx->device_ctx, usage_probe);

    marmot_allocator_usage_t usage_final = {0};
    assert_int_equal(marmot_allocator_get_usage(env->ctx, &usage_final), MARMOT_SUCCESS);
    assert_true(usage_final.active_allocations <= usage_after.active_allocations);

#if defined(__linux__)
    const bool is_cpu_backend = (env->backend == MARMOT_BACKEND_CPU);
#else
    const bool is_cpu_backend = (env->backend == MARMOT_BACKEND_CPU);
#endif
    if (is_cpu_backend) {
        assert_true(usage_final.pooled_bytes >= usage_before.pooled_bytes);
    }

#if defined(__linux__)
    if (env->backend == MARMOT_BACKEND_CPU) {
        const marmot_allocator_ops_t *cpu_alloc = marmot_get_allocator(MARMOT_BACKEND_CPU);
        assert_non_null(cpu_alloc);
        marmot_allocation_t huge = {0};
        marmot_error_t huge_err =
            cpu_alloc->alloc(env->ctx->device_ctx, 2 * 1024 * 1024, 0, MARMOT_ALLOC_HUGE_PAGES, &huge);
        assert_int_equal(huge_err, MARMOT_SUCCESS);
        assert_true(huge.size >= 2 * 1024 * 1024);
        assert_int_equal(huge.type, MARMOT_ALLOC_HUGE_PAGES);
        cpu_alloc->free(env->ctx->device_ctx, &huge);
    }
#endif

#if defined(__APPLE__) && MARMOT_ENABLE_METAL
    if (env->backend == MARMOT_BACKEND_METAL) {
        const marmot_allocator_ops_t *metal_alloc = marmot_get_allocator(MARMOT_BACKEND_METAL);
        assert_non_null(metal_alloc);
        marmot_allocation_t private_block = {0};
        marmot_error_t priv_err =
            metal_alloc->alloc(env->ctx->device_ctx, 4096, 0, MARMOT_ALLOC_GPU_PRIVATE, &private_block);
        assert_int_equal(priv_err, MARMOT_SUCCESS);
        assert_int_equal(private_block.type, MARMOT_ALLOC_GPU_PRIVATE);
        metal_alloc->free(env->ctx->device_ctx, &private_block);
    }
#endif
}

static void test_backend_memory_default(void **state) {
    run_memory_checks((marmot_test_env_t *)(*state));
}

#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
static void test_backend_memory_scalar(void **state) {
    run_memory_checks((marmot_test_env_t *)(*state));
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_backend_memory_default, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
#if MARMOT_TEST_HAS_CPU_SCALAR_SUITE
        cmocka_unit_test_setup_teardown(
            test_backend_memory_scalar, marmot_test_backend_setup_scalar, marmot_test_backend_teardown
        ),
#endif
    };

    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
