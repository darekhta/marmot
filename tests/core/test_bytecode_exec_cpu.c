#include "marmot/device.h"
#include "marmot/error.h"
#include "marmot/tensor.h"
#include "marmot/traits_ids.gen.h"

#include "backends/cpu/dispatch/bytecode_exec_cpu.gen.h"
#include "core/bytecode/bytecode.h"
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

static void test_bytecode_exec_cpu_add_f32(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    const size_t shape[] = {4};
    marmot_tensor_t *a = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *b = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, shape, 1, MARMOT_DTYPE_FLOAT32);
    assert_non_null(a);
    assert_non_null(b);
    assert_non_null(out);

    float *a_data = marmot_tensor_data_f32_mut(ctx, a);
    float *b_data = marmot_tensor_data_f32_mut(ctx, b);
    assert_non_null(a_data);
    assert_non_null(b_data);
    for (size_t i = 0; i < 4; ++i) {
        a_data[i] = (float)(i + 1);
        b_data[i] = (float)(10 - i);
    }

    marmot_device_caps_t caps;
    assert_true(marmot_backend_detect_default_caps(MARMOT_BACKEND_CPU, &caps));

    marmot_op_signature_t sig;
    init_elementwise_signature(&sig, MARMOT_OP_ADD, MARMOT_DTYPE_FLOAT32, 4);

    marmot_kernel_selection_t sel = marmot_backend_query_kernel(MARMOT_BACKEND_CPU, &sig, &caps);
    assert_true(sel.supported);
    assert_int_not_equal(sel.op_index, MARMOT_KERNEL_OP_INDEX_INVALID);
    uint16_t op_index = sel.op_index;

    marmot_bc_builder_t builder;
    assert_true(marmot_bc_builder_init(&builder));

    assert_true(marmot_bc_builder_emit_u16(&builder, op_index));
    assert_true(marmot_bc_builder_emit_u16(&builder, 0));
    assert_true(marmot_bc_builder_emit_u16(&builder, 1));
    assert_true(marmot_bc_builder_emit_u16(&builder, 2));
    assert_true(marmot_bc_builder_emit_u16(&builder, MARMOT_BC_END));

    marmot_bc_program_t program;
    assert_true(marmot_bc_builder_finish(
        &builder, &program, marmot_cpu_bc_imm_size, marmot_cpu_bc_exec_table, 3, MARMOT_CPU_BC_OP_COUNT
    ));

    marmot_tensor_t *regs[] = {a, b, out};
    marmot_bc_exec_ctx_t exec_ctx = {
        .ctx = ctx,
        .device_ctx = ctx->device_ctx,
    };
    marmot_error_t err = marmot_bc_execute(&program, &exec_ctx, regs);
    assert_int_equal(err, MARMOT_SUCCESS);

    const float *out_data = marmot_tensor_data_f32(ctx, out);
    assert_non_null(out_data);
    for (size_t i = 0; i < 4; ++i) {
        float expected = a_data[i] + b_data[i];
        assert_true(out_data[i] == expected);
    }

    marmot_bc_program_destroy(&program);
    marmot_bc_builder_reset(&builder);
    marmot_tensor_destroy(out);
    marmot_tensor_destroy(b);
    marmot_tensor_destroy(a);
    marmot_destroy(ctx);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_bytecode_exec_cpu_add_f32),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
