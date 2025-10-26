#include "backends/cpu/dispatch/bytecode_tables_cpu.gen.h"

#if MARMOT_ENABLE_METAL
#include "backends/metal/ops/bytecode_tables_metal.gen.h"
#endif

#include "core/bytecode/bytecode_compile.h"
#include "core/dispatch/fusion_flags.h"
#include "core/dispatch/kernel_query.h"

// clang-format off
#include <setjmp.h>  // Must be before cmocka.h for jmp_buf
#include <stdarg.h>
#include <stddef.h>
#include <string.h>
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
// clang-format on

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

static void test_cpu_bytecode_table_mapping(void **state) {
    (void)state;

    marmot_device_caps_t caps;
    assert_true(marmot_backend_detect_default_caps(MARMOT_BACKEND_CPU, &caps));

    marmot_op_signature_t sig;
    init_elementwise_signature(&sig, MARMOT_OP_ADD, MARMOT_DTYPE_FLOAT32, 1024);

    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_int_not_equal(sel.op_index, MARMOT_KERNEL_OP_INDEX_INVALID);
    assert_true(sel.op_index < MARMOT_CPU_BC_OP_COUNT);
    assert_true(marmot_bc_exec_supported(MARMOT_BACKEND_CPU, sel.op_index));
}

static void test_metal_bytecode_table_mapping(void **state) {
    (void)state;

#if MARMOT_ENABLE_METAL
    marmot_device_caps_t caps;
    assert_true(marmot_backend_detect_default_caps(MARMOT_BACKEND_METAL, &caps));

    marmot_op_signature_t sig;
    init_elementwise_signature(&sig, MARMOT_OP_ADD, MARMOT_DTYPE_FLOAT32, 1024);

    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_METAL, &sig, &caps);
    assert_true(sel.supported);
    assert_int_not_equal(sel.op_index, MARMOT_KERNEL_OP_INDEX_INVALID);
    assert_true(sel.op_index < MARMOT_METAL_BC_OP_COUNT);
    assert_true(marmot_bc_exec_supported(MARMOT_BACKEND_METAL, sel.op_index));
#else
    (void)state;
#endif
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_cpu_bytecode_table_mapping),
        cmocka_unit_test(test_metal_bytecode_table_mapping),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
