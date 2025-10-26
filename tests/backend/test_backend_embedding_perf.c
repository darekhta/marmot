#include <stdio.h>
#include <stdlib.h>

#include <time.h>

#include "backend/test_backend_utils.h"

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void test_embedding_perf_optional(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);

    const char *run = getenv("MARMOT_RUN_EMB_PERF");
    if (run == nullptr || run[0] == '\0') {
        skip();
        return;
    }

    const size_t vocab = 32768; // 32k rows
    const size_t dim = 2048;    // reasonably large D
    const size_t N = 8192;      // flat token count

    size_t wshape[] = {vocab, dim};
    marmot_tensor_t *weights = marmot_tensor_create(env->ctx, wshape, 2, MARMOT_DTYPE_FLOAT16);
    assert_non_null(weights);

    // Initialize weights deterministically
    marmot_float16_t *w = marmot_tensor_data_f16_mut(env->ctx, weights);
    assert_non_null(w);
    for (size_t r = 0; r < vocab; ++r) {
        for (size_t c = 0; c < dim; ++c) {
            float v = (float)((r * 1315423911u + c * 2654435761u) & 0xFF) * (1.0f / 255.0f) - 0.5f;
            w[r * dim + c] = marmot_f32_to_f16_ref(v);
        }
    }

    size_t tshape[] = {N};
    marmot_tensor_t *ids = marmot_tensor_create(env->ctx, tshape, 1, MARMOT_DTYPE_INT32);
    assert_non_null(ids);
    int32_t *tid = (int32_t *)marmot_tensor_data_i32_mut(env->ctx, ids);
    assert_non_null(tid);
    for (size_t i = 0; i < N; ++i) {
        uint32_t r = (uint32_t)(i * 1664525u + 1013904223u);
        int32_t id = (int32_t)(r % (uint32_t)vocab);
        // 20% padding
        if ((r & 0xF) < 3)
            id = -1;
        tid[i] = id;
    }

    size_t oshape[] = {N, dim};
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, oshape, 2, MARMOT_DTYPE_FLOAT16);
    assert_non_null(out);

    marmot_embedding_desc_t desc = marmot_embedding_desc_default();
    desc.weights = weights;
    desc.token_ids = ids;
    desc.out = out;
    desc.dtype_out = MARMOT_DTYPE_FLOAT16;
    desc.padding_id = -1;
    desc.bounds_check = false;
    desc.ragged = false;
    desc.prefer_gpu_private = MARMOT_PREFERENCE_ENABLE; // exercise staging path on Metal

    double t0 = now_seconds();
    marmot_error_t err = marmot_embedding_lookup(env->ctx, &desc);
    double t1 = now_seconds();
    assert_int_equal(err, MARMOT_SUCCESS);

    double seconds = t1 - t0;
    double bytes = (double)N * (double)dim * sizeof(marmot_float16_t);
    double gbps = (bytes / seconds) / 1e9;
    printf(
        "\n[embedding perf %s] N=%zu D=%zu time=%.3fs out=%.2f GB/s\n",
        env->backend == MARMOT_BACKEND_CPU ? "cpu" : "metal", N, dim, seconds, gbps
    );

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(ids);
    marmot_tensor_destroy(weights);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_embedding_perf_optional, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
