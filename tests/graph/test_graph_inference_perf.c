/* clang-format off */
#include "marmot/marmot.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
/* clang-format on */

#include "backend/test_backend_utils.h"

static void init_desc_2d(marmot_graph_tensor_desc_t *desc, size_t dim0, size_t dim1, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 2;
    desc->shape[0] = dim0;
    desc->shape[1] = dim1;
    desc->strides[1] = 1;
    desc->strides[0] = dim1;
}

static void init_desc_1d(marmot_graph_tensor_desc_t *desc, size_t dim0, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = 1;
    desc->shape[0] = dim0;
    desc->strides[0] = 1;
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000ull) + (uint64_t)ts.tv_nsec;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    if (va < vb) {
        return -1;
    }
    if (va > vb) {
        return 1;
    }
    return 0;
}

static size_t parse_size_env(const char *name, size_t fallback) {
    const char *value = getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char *endptr = nullptr;
    unsigned long parsed = strtoul(value, &endptr, 10);
    if (endptr == value) {
        return fallback;
    }
    if (parsed > (unsigned long)SIZE_MAX) {
        return fallback;
    }
    return (size_t)parsed;
}

static void test_graph_inference_perf_optional(void **state) {
    marmot_test_env_t *env = (marmot_test_env_t *)(*state);
    assert_non_null(env);
    assert_non_null(env->ctx);

    const char *run_env = getenv("MARMOT_RUN_GRAPH_INFERENCE_PERF");
    if (run_env == nullptr || run_env[0] == '\0') {
        skip();
    }

    const marmot_dtype_t dtype = (env->backend == MARMOT_BACKEND_METAL) ? MARMOT_DTYPE_FLOAT16 : MARMOT_DTYPE_FLOAT32;
    const size_t d = parse_size_env("MARMOT_GRAPH_PERF_D", 512);
    const size_t ff = parse_size_env("MARMOT_GRAPH_PERF_FF", 2048);
    const size_t iters = parse_size_env("MARMOT_GRAPH_PERF_ITERS", 1000);
    const size_t warmup = parse_size_env("MARMOT_GRAPH_PERF_WARMUP", 10);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    marmot_graph_tensor_desc_t x_desc;
    init_desc_2d(&x_desc, 1, d, dtype);
    marmot_graph_tensor_desc_t rms_w_desc;
    init_desc_1d(&rms_w_desc, d, dtype);
    marmot_graph_tensor_desc_t w1_desc;
    init_desc_2d(&w1_desc, ff, d, dtype);
    marmot_graph_tensor_desc_t w2_desc;
    init_desc_2d(&w2_desc, d, ff, dtype);
    marmot_graph_tensor_desc_t ff_desc;
    init_desc_2d(&ff_desc, 1, ff, dtype);

    marmot_value_id_t x_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t rms_w_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t w1_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t w2_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_input(graph, &x_desc, &x_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &rms_w_desc, &rms_w_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &w1_desc, &w1_id), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &w2_desc, &w2_id), MARMOT_SUCCESS);

    marmot_value_id_t norm_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t norm_inputs[2] = {x_id, rms_w_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "rms_norm", nullptr, norm_inputs, 2, &x_desc, 1, &norm_id), MARMOT_SUCCESS
    );

    marmot_value_id_t mm1_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t mm1_inputs[2] = {norm_id, w1_id};
    assert_int_equal(
        marmot_graph_add_op(graph, "matmul", nullptr, mm1_inputs, 2, &ff_desc, 1, &mm1_id), MARMOT_SUCCESS
    );

    marmot_value_id_t silu_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(marmot_graph_add_op(graph, "silu", nullptr, &mm1_id, 1, &ff_desc, 1, &silu_id), MARMOT_SUCCESS);

    marmot_value_id_t mm2_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t mm2_inputs[2] = {silu_id, w2_id};
    assert_int_equal(marmot_graph_add_op(graph, "matmul", nullptr, mm2_inputs, 2, &x_desc, 1, &mm2_id), MARMOT_SUCCESS);

    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t add_inputs[2] = {mm2_id, x_id};
    assert_int_equal(marmot_graph_add_op(graph, "add", nullptr, add_inputs, 2, &x_desc, 1, &out_id), MARMOT_SUCCESS);

    assert_int_equal(marmot_graph_finalize(graph, env->backend), MARMOT_SUCCESS);

    size_t x_shape[2] = {1, d};
    size_t w1_shape[2] = {ff, d};
    size_t w2_shape[2] = {d, ff};
    size_t rms_shape[1] = {d};

    marmot_tensor_t *x = marmot_tensor_create(env->ctx, x_shape, 2, dtype);
    marmot_tensor_t *rms_w = marmot_tensor_create(env->ctx, rms_shape, 1, dtype);
    marmot_tensor_t *w1 = marmot_tensor_create(env->ctx, w1_shape, 2, dtype);
    marmot_tensor_t *w2 = marmot_tensor_create(env->ctx, w2_shape, 2, dtype);
    marmot_tensor_t *out = marmot_tensor_create(env->ctx, x_shape, 2, dtype);
    assert_non_null(x);
    assert_non_null(rms_w);
    assert_non_null(w1);
    assert_non_null(w2);
    assert_non_null(out);

    const marmot_tensor_t *inputs[] = {x, rms_w, w1, w2};
    marmot_tensor_t *outputs[] = {out};

    for (size_t i = 0; i < warmup; ++i) {
        assert_int_equal(marmot_graph_execute(graph, env->ctx, inputs, 4, outputs, 1), MARMOT_SUCCESS);
    }

    uint64_t *samples = (uint64_t *)calloc(iters, sizeof(*samples));
    assert_non_null(samples);

    for (size_t i = 0; i < iters; ++i) {
        uint64_t start = now_ns();
        assert_int_equal(marmot_graph_execute(graph, env->ctx, inputs, 4, outputs, 1), MARMOT_SUCCESS);
        uint64_t end = now_ns();
        samples[i] = end - start;
    }

    qsort(samples, iters, sizeof(*samples), cmp_u64);
    uint64_t median_ns = samples[iters / 2];

    fprintf(
        stderr, "[graph perf] backend=%s d=%zu ff=%zu iters=%zu median=%.3f us\n",
        env->backend == MARMOT_BACKEND_METAL ? "metal" : "cpu", d, ff, iters, (double)median_ns / 1000.0
    );

    free(samples);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(w2);
    marmot_tensor_destroy(w1);
    marmot_tensor_destroy(rms_w);
    marmot_tensor_destroy(x);
    marmot_graph_destroy(graph);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(
            test_graph_inference_perf_optional, marmot_test_backend_setup, marmot_test_backend_teardown
        ),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
