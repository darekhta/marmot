#ifndef MARMOT_TESTS_QUANT_TEST_UTILS_H
#define MARMOT_TESTS_QUANT_TEST_UTILS_H

#include "backend/test_backend_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline void marmot_quant_expect_quant_bytes(
    const marmot_test_env_t *env, const marmot_tensor_t *tensor, const void *golden, size_t golden_bytes,
    const char *label
) {
    assert_non_null(env);
    assert_non_null(tensor);
    assert_non_null(golden);
    const size_t tensor_bytes = marmot_tensor_size_bytes(tensor);
    assert_true(tensor_bytes >= golden_bytes);
    uint8_t *host = (uint8_t *)malloc(tensor_bytes);
    assert_non_null(host);
    marmot_error_t err = marmot_tensor_copy_to_host_buffer(env->ctx, tensor, host, tensor_bytes);
    assert_int_equal(err, MARMOT_SUCCESS);
    const uint8_t *ref = (const uint8_t *)golden;
    for (size_t i = 0; i < golden_bytes; ++i) {
        if (host[i] != ref[i]) {
            fprintf(
                stderr, "%s quant mismatch @%zu expected 0x%02x got 0x%02x\n", label != nullptr ? label : "(unnamed)",
                i, ref[i], host[i]
            );
            size_t dump_start = (i >= 8) ? i - 8 : 0;
            size_t dump_end = (i + 8 < golden_bytes) ? i + 8 : golden_bytes - 1;
            fprintf(stderr, "expected bytes: ");
            for (size_t j = dump_start; j <= dump_end; ++j) {
                fprintf(stderr, "%02x ", ref[j]);
            }
            fprintf(stderr, "\nactual bytes:   ");
            for (size_t j = dump_start; j <= dump_end; ++j) {
                fprintf(stderr, "%02x ", host[j]);
            }
            fprintf(stderr, "\n");
            fail_msg(
                "%s quant bytes mismatch at %zu: expected 0x%02x got 0x%02x", label != nullptr ? label : "(unnamed)", i,
                ref[i], host[i]
            );
        }
    }
    free(host);
}

typedef marmot_error_t (*marmot_quant_dequant_fn)(
    const marmot_context_t *ctx, const marmot_tensor_t *input, marmot_tensor_t *output
);

static inline void marmot_quant_expect_dequant_golden(
    marmot_test_env_t *env, marmot_tensor_t *quant, const void *golden_data, size_t golden_bytes,
    marmot_tensor_t *dequant, marmot_quant_dequant_fn fn, const float *expected, size_t expected_len, float tol,
    marmot_quant_kind_t kind
) {
    assert_non_null(env);
    assert_non_null(env->ctx);
    assert_non_null(quant);
    assert_non_null(golden_data);
    assert_true(marmot_tensor_size_bytes(quant) >= golden_bytes);
    assert_int_equal(marmot_tensor_copy_from_host_buffer(env->ctx, quant, golden_data, golden_bytes), MARMOT_SUCCESS);
    quant->quant_kind = kind;
    quant->quant_layout = MARMOT_QUANT_LAYOUT_GGUF;

    marmot_error_t err = fn(env->ctx, quant, dequant);
    if (err != MARMOT_SUCCESS) {
        const char *detail = marmot_get_last_error_detail();
        fail_msg(
            "Dequant %d failed: %s%s%s", kind, marmot_error_string(err),
            detail != nullptr && detail[0] != '\0' ? " - " : "", detail != nullptr ? detail : ""
        );
    }

    float *actual = (float *)malloc(expected_len * sizeof(float));
    assert_non_null(actual);
    marmot_test_fetch_f32_span(env, actual, dequant, expected_len);
    marmot_test_expect_close_array(actual, expected, expected_len, tol);
    free(actual);
}

#ifdef __cplusplus
}
#endif

#endif // MARMOT_TESTS_QUANT_TEST_UTILS_H
