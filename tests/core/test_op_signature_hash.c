#include "marmot/op_signature_hash.gen.h"

#include "core/dispatch/fusion_flags.h"

// clang-format off
#include <setjmp.h>
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

static marmot_op_signature_t make_base_sig(void) {
    marmot_op_signature_t sig = {0};
    sig.op_id = MARMOT_OP_QKV_SHARED_INPUT;
    sig.profile_id = MARMOT_PROFILE_INVALID;
    sig.matmul_layout = MARMOT_MATMUL_LAYOUT_NT;
    sig.input_dtype = MARMOT_DTYPE_FLOAT16;
    sig.weight_dtype = MARMOT_DTYPE_FLOAT16;
    sig.output_dtype = MARMOT_DTYPE_FLOAT16;
    sig.accum_dtype = MARMOT_DTYPE_FLOAT32;
    sig.qscheme_id = MARMOT_QSCHEME_NONE;
    sig.quant_block = (marmot_quant_block_t){
        .block_size = 32,
        .group_size = 32,
        .scale_dtype = MARMOT_DTYPE_FLOAT16,
        .zero_point_dtype = MARMOT_DTYPE_UINT8,
    };
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_SEPARATE;
    sig.epilogue_flags = MARMOT_EPILOGUE_NONE;
    sig.activation = MARMOT_DEVICE_UNARY_COUNT;
    sig.variant_flags = MARMOT_FUSION_NONE;
    sig.dims.matmul.N = 128;
    sig.dims.matmul.K = 64;
    sig.dims.matmul.M = 192;
    return sig;
}

static void test_hash_includes_weight_layout(void **state) {
    (void)state;

    marmot_op_signature_t sig = make_base_sig();
    uint64_t base = marmot_hash_op_signature(&sig);
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_PACKED_3MK;
    uint64_t changed = marmot_hash_op_signature(&sig);
    assert_true(base != changed);
}

static void test_hash_includes_activation(void **state) {
    (void)state;

    marmot_op_signature_t sig = make_base_sig();
    uint64_t base = marmot_hash_op_signature(&sig);
    sig.activation = MARMOT_DEVICE_UNARY_RELU;
    uint64_t changed = marmot_hash_op_signature(&sig);
    assert_true(base != changed);
}

static void test_hash_includes_quant_block(void **state) {
    (void)state;

    marmot_op_signature_t sig = make_base_sig();
    uint64_t base = marmot_hash_op_signature(&sig);
    sig.quant_block.block_size = 64;
    uint64_t changed = marmot_hash_op_signature(&sig);
    assert_true(base != changed);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_hash_includes_weight_layout),
        cmocka_unit_test(test_hash_includes_activation),
        cmocka_unit_test(test_hash_includes_quant_block),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
