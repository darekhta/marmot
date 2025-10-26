#include "core/bytecode/bytecode.h"

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

#include <string.h>

typedef struct {
    uint32_t sum;
} test_exec_ctx_t;

static marmot_error_t
exec_add_imm(const void *backend_exec_ctx, marmot_tensor_t **regs, const uint8_t *imm, const uint8_t *const_pool) {
    (void)regs;
    (void)const_pool;

    if (backend_exec_ctx == nullptr || imm == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    test_exec_ctx_t *exec_ctx = (test_exec_ctx_t *)backend_exec_ctx;
    uint32_t value = 0;
    memcpy(&value, imm, sizeof(value));
    exec_ctx->sum += value;
    return MARMOT_SUCCESS;
}

static marmot_error_t
exec_add_const(const void *backend_exec_ctx, marmot_tensor_t **regs, const uint8_t *imm, const uint8_t *const_pool) {
    (void)regs;

    if (backend_exec_ctx == nullptr || imm == nullptr || const_pool == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    test_exec_ctx_t *exec_ctx = (test_exec_ctx_t *)backend_exec_ctx;
    uint32_t offset = 0;
    uint32_t value = 0;
    memcpy(&offset, imm, sizeof(offset));
    memcpy(&value, const_pool + offset, sizeof(value));
    exec_ctx->sum += value;
    return MARMOT_SUCCESS;
}

static void test_bytecode_execute_basic(void **state) {
    (void)state;

    marmot_bc_builder_t builder;
    assert_true(marmot_bc_builder_init(&builder));

    const uint32_t const_value = 33;
    uint32_t const_offset = marmot_bc_builder_add_const(&builder, &const_value, sizeof(const_value), 4);
    assert_int_not_equal(const_offset, MARMOT_BC_INVALID_OFFSET);

    assert_true(marmot_bc_builder_emit_u16(&builder, 1));
    assert_true(marmot_bc_builder_emit_u32(&builder, 5));
    assert_true(marmot_bc_builder_emit_u16(&builder, 2));
    assert_true(marmot_bc_builder_emit_u32(&builder, const_offset));
    assert_true(marmot_bc_builder_emit_u16(&builder, MARMOT_BC_END));

    const uint16_t imm_sizes[] = {
        0,
        sizeof(uint32_t),
        sizeof(uint32_t),
    };
    const marmot_bc_exec_fn exec_table[] = {
        nullptr,
        exec_add_imm,
        exec_add_const,
    };

    marmot_bc_program_t program;
    assert_true(marmot_bc_builder_finish(&builder, &program, imm_sizes, exec_table, 0, 3));

    test_exec_ctx_t ctx = {.sum = 0};
    assert_int_equal(marmot_bc_execute(&program, &ctx, nullptr), MARMOT_SUCCESS);
    assert_int_equal(ctx.sum, 38);

    marmot_bc_program_destroy(&program);
    marmot_bc_builder_reset(&builder);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_bytecode_execute_basic),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
