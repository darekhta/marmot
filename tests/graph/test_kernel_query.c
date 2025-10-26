/* clang-format off */
#include "core/dispatch/kernel_query.h"
#include "marmot/graph/op_signature.h"
#include "core/dispatch/fusion_flags.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
/* clang-format on */

static void init_matmul_signature(marmot_op_signature_t *sig, marmot_dtype_t input_dtype, marmot_dtype_t output_dtype) {
    memset(sig, 0, sizeof(*sig));
    sig->op_id = MARMOT_OP_MATMUL;
    sig->matmul_layout = MARMOT_MATMUL_LAYOUT_NT;
    sig->input_dtype = input_dtype;
    sig->weight_dtype = input_dtype;
    sig->output_dtype = output_dtype;
    sig->accum_dtype = MARMOT_DTYPE_FLOAT32;
    sig->qscheme_id = MARMOT_QSCHEME_NONE;
    sig->weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;
    sig->epilogue_flags = MARMOT_EPILOGUE_NONE;
    sig->variant_flags = MARMOT_FUSION_NONE;
    sig->dims.matmul.N = 2;
    sig->dims.matmul.M = 4;
    sig->dims.matmul.K = 3;
}

static marmot_device_caps_t default_caps(void) {
    marmot_device_caps_t caps = {
        .peak_flops_tflops_fp32 = 4.0f,
        .peak_flops_tflops_fp16 = 4.0f,
        .mem_bw_gbps = 100.0f,
        .launch_overhead_us = 0.1f,
        .edge_penalty_alpha = 0.5f,
        .has_fma = true,
    };
    return caps;
}

static void test_cpu_matmul_bias_is_matched(void **state) {
    (void)state;
    marmot_op_signature_t sig;
    init_matmul_signature(&sig, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32);
    sig.epilogue_flags = MARMOT_EPILOGUE_BIAS;

    marmot_device_caps_t caps = default_caps();
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    // Kernel selection now returns platform-specific kernel (scalar, neon, or accelerate)
    // just check it's a valid CPU matmul kernel
    assert_true(sel.kernel_id > 0);
}

static void test_cpu_matmul_bias_residual_rejected(void **state) {
    (void)state;
    marmot_op_signature_t sig;
    init_matmul_signature(&sig, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32);
    sig.epilogue_flags = MARMOT_EPILOGUE_BIAS | MARMOT_EPILOGUE_RESIDUAL;

    marmot_device_caps_t caps = default_caps();
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_false(sel.supported);
}

static void test_cpu_matmul_quantized_q4k_matches(void **state) {
    (void)state;
    marmot_op_signature_t sig;
    init_matmul_signature(&sig, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT16);
    sig.qscheme_id = MARMOT_QSCHEME_Q4_K;
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE;
    sig.epilogue_flags = MARMOT_EPILOGUE_BIAS;

    marmot_device_caps_t caps = default_caps();
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_true(sel.kernel_id > 0); // Kernel ID varies by platform
}

static void test_cpu_matmul_quantized_q4_0_matches(void **state) {
    (void)state;
    marmot_op_signature_t sig;
    init_matmul_signature(&sig, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT16);
    sig.qscheme_id = MARMOT_QSCHEME_Q4_0;
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE;
    sig.epilogue_flags = MARMOT_EPILOGUE_BIAS;

    marmot_device_caps_t caps = default_caps();
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_true(sel.kernel_id > 0); // Kernel ID varies by platform
}

static void test_cpu_matmul_epilogue_bias_selection(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_caps();

    marmot_op_signature_t base;
    init_matmul_signature(&base, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32);
    marmot_kernel_selection_t base_sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &base, &caps);
    assert_true(base_sel.supported);

    marmot_op_signature_t bias = base;
    bias.epilogue_flags = MARMOT_EPILOGUE_BIAS;
    marmot_kernel_selection_t bias_sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &bias, &caps);
    assert_true(bias_sel.supported);
    // Both should select valid kernels
    assert_true(base_sel.kernel_id > 0);
    assert_true(bias_sel.kernel_id > 0);
}

static void test_cpu_matmul_bias_activation_matches(void **state) {
    (void)state;
    marmot_op_signature_t sig;
    init_matmul_signature(&sig, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32);
    sig.op_id = MARMOT_OP_MATMUL_BIAS_RELU;
    sig.epilogue_flags = MARMOT_EPILOGUE_BIAS;
    sig.activation = MARMOT_DEVICE_UNARY_RELU;

    marmot_device_caps_t caps = default_caps();
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    // CPU matmul kernels don't fuse activation.
    assert_false(sel.supported);
}

static void test_cpu_matmul_different_sizes(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_caps();

    marmot_op_signature_t aligned;
    init_matmul_signature(&aligned, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32);
    aligned.dims.matmul.M = 64;
    aligned.dims.matmul.N = 64;
    aligned.dims.matmul.K = 64;
    marmot_kernel_selection_t aligned_sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &aligned, &caps);
    assert_true(aligned_sel.supported);
    assert_true(aligned_sel.kernel_id > 0);

    marmot_op_signature_t ragged = aligned;
    ragged.dims.matmul.M = 65;
    ragged.dims.matmul.N = 67;
    ragged.dims.matmul.K = 70;
    marmot_kernel_selection_t ragged_sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &ragged, &caps);
    assert_true(ragged_sel.supported);
    assert_true(ragged_sel.kernel_id > 0);
}

static void
init_elementwise_signature(marmot_op_signature_t *sig, marmot_op_id_t op, marmot_dtype_t dtype, uint32_t n_elems) {
    memset(sig, 0, sizeof(*sig));
    sig->op_id = op;
    sig->profile_id = MARMOT_PROFILE_SCALAR;
    sig->input_dtype = dtype;
    sig->weight_dtype = dtype;
    sig->output_dtype = dtype;
    sig->accum_dtype = dtype;
    sig->qscheme_id = MARMOT_QSCHEME_NONE;
    sig->epilogue_flags = MARMOT_EPILOGUE_NONE;
    sig->variant_flags = MARMOT_FUSION_NONE;
    sig->dims.elementwise.n_elems = n_elems;
}

static void test_cpu_elementwise_selection(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_caps();

    marmot_op_signature_t sig;
    init_elementwise_signature(&sig, MARMOT_OP_ADD, MARMOT_DTYPE_FLOAT32, 1024 * 1024);
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_true(sel.kernel_id > 0);
}

static void test_cpu_elementwise_row_strided_selection(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_caps();

    marmot_op_signature_t sig;
    init_elementwise_signature(&sig, MARMOT_OP_ADD, MARMOT_DTYPE_FLOAT32, 1024 * 1024);
    sig.stride_mode = MARMOT_STRIDE_MODE_ROW_STRIDED;
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_true(sel.kernel_id > 0);
}

static void test_cpu_unary_selection(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_caps();

    marmot_op_signature_t sig;
    init_elementwise_signature(&sig, MARMOT_OP_RELU, MARMOT_DTYPE_FLOAT32, 1024 * 1024);
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_true(sel.kernel_id > 0);
}

static void test_cpu_fma_selection(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_caps();

    marmot_op_signature_t sig;
    init_elementwise_signature(&sig, MARMOT_OP_FMA, MARMOT_DTYPE_FLOAT32, 1024 * 1024);
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_true(sel.kernel_id > 0);
}

static void test_cpu_confidence_bounded(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_caps();

    marmot_op_signature_t sig;
    init_matmul_signature(&sig, MARMOT_DTYPE_FLOAT32, MARMOT_DTYPE_FLOAT32);
    sig.dims.matmul.M = 512;
    sig.dims.matmul.N = 512;
    sig.dims.matmul.K = 512;
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_true(sel.confidence > 0.0f);
    assert_true(sel.confidence <= 1.0f);
}

#if MARMOT_ENABLE_METAL
static marmot_device_caps_t default_metal_caps(void) {
    marmot_device_caps_t caps = {
        .peak_flops_tflops_fp32 = 4.0f,
        .peak_flops_tflops_fp16 = 8.0f,
        .mem_bw_gbps = 120.0f,
        .launch_overhead_us = 10.0f,
        .edge_penalty_alpha = 0.5f,
        .has_fma = true,
        .has_fp16_compute = true,
    };
    return caps;
}

static void test_metal_matmul_basic_selection(void **state) {
    (void)state;
    marmot_op_signature_t sig;
    init_matmul_signature(&sig, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT16);

    marmot_device_caps_t caps = default_metal_caps();
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_METAL, &sig, &caps);
    assert_true(sel.supported);
    assert_int_not_equal(sel.kernel_id, MARMOT_KERNEL_INVALID);
}

static void test_metal_add_relu_selection(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_metal_caps();

    marmot_op_signature_t sig;
    init_elementwise_signature(&sig, MARMOT_OP_ADD_RELU, MARMOT_DTYPE_FLOAT32, 1024 * 1024);
    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_METAL, &sig, &caps);
    assert_true(sel.supported);
    assert_int_not_equal(sel.kernel_id, MARMOT_KERNEL_INVALID);
}

static void test_metal_matmul_bias_relu_selection(void **state) {
    (void)state;
    marmot_device_caps_t caps = default_metal_caps();

    marmot_op_signature_t sig;
    init_matmul_signature(&sig, MARMOT_DTYPE_FLOAT16, MARMOT_DTYPE_FLOAT16);
    sig.op_id = MARMOT_OP_MATMUL_BIAS_RELU;
    sig.epilogue_flags = MARMOT_EPILOGUE_BIAS | MARMOT_EPILOGUE_ACTIVATION;
    sig.activation = MARMOT_DEVICE_UNARY_RELU;

    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_METAL, &sig, &caps);
    assert_true(sel.supported);
    assert_int_not_equal(sel.kernel_id, MARMOT_KERNEL_INVALID);
}
#endif

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cpu_matmul_bias_is_matched),
        cmocka_unit_test(test_cpu_matmul_bias_residual_rejected),
        cmocka_unit_test(test_cpu_matmul_quantized_q4k_matches),
        cmocka_unit_test(test_cpu_matmul_quantized_q4_0_matches),
        cmocka_unit_test(test_cpu_matmul_epilogue_bias_selection),
        cmocka_unit_test(test_cpu_matmul_bias_activation_matches),
        cmocka_unit_test(test_cpu_matmul_different_sizes),
        cmocka_unit_test(test_cpu_elementwise_selection),
        cmocka_unit_test(test_cpu_elementwise_row_strided_selection),
        cmocka_unit_test(test_cpu_unary_selection),
        cmocka_unit_test(test_cpu_fma_selection),
        cmocka_unit_test(test_cpu_confidence_bounded),
#if MARMOT_ENABLE_METAL
        cmocka_unit_test(test_metal_matmul_basic_selection),
        cmocka_unit_test(test_metal_add_relu_selection),
        cmocka_unit_test(test_metal_matmul_bias_relu_selection),
#endif
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
