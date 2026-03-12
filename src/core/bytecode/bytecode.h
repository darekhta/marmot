#ifndef MARMOT_CORE_BYTECODE_H
#define MARMOT_CORE_BYTECODE_H

#include "marmot/error.h"
#include "marmot/macros.h"
#include "marmot/tensor.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MARMOT_BC_END = 0,
};

#define MARMOT_BC_INVALID_OFFSET UINT32_MAX
#define MARMOT_BC_U32_RUNTIME UINT32_MAX
#define MARMOT_BC_OP_INVALID UINT16_MAX
#define MARMOT_BC_REG_INVALID UINT16_MAX

typedef enum {
    MARMOT_BC_SCHEMA_INVALID = 0,
    MARMOT_BC_SCHEMA_UNARY,
    MARMOT_BC_SCHEMA_BINARY,
    MARMOT_BC_SCHEMA_TERNARY,
    MARMOT_BC_SCHEMA_REDUCTION,
    MARMOT_BC_SCHEMA_SOFTMAX,
    MARMOT_BC_SCHEMA_TOPK,
    MARMOT_BC_SCHEMA_MOE_EXPERTS,
    MARMOT_BC_SCHEMA_LAYERNORM,
    MARMOT_BC_SCHEMA_RMS_NORM,
    MARMOT_BC_SCHEMA_PAGED_ATTENTION,
    MARMOT_BC_SCHEMA_ROPE,
    MARMOT_BC_SCHEMA_MATMUL,
    MARMOT_BC_SCHEMA_QKV,
    MARMOT_BC_SCHEMA_RESHAPE,
    MARMOT_BC_SCHEMA_VIEW,
    MARMOT_BC_SCHEMA_CONTIGUOUS,
    MARMOT_BC_SCHEMA_TRANSPOSE,
    MARMOT_BC_SCHEMA_CONCAT,
    MARMOT_BC_SCHEMA_SLICE,
    MARMOT_BC_SCHEMA_GATHER_ROWS,
    MARMOT_BC_SCHEMA_QUANTIZE,
    MARMOT_BC_SCHEMA_DEQUANTIZE,
    MARMOT_BC_SCHEMA_COMPUTE_QPARAMS,
    MARMOT_BC_SCHEMA_EMBEDDING,
    MARMOT_BC_SCHEMA_CONVERT,
    MARMOT_BC_SCHEMA_VEC_DOT,
} marmot_bc_schema_id_t;

typedef struct marmot_bc_exec_ctx {
    const marmot_context_t *ctx;
    const void *device_ctx;
} marmot_bc_exec_ctx_t;

typedef marmot_error_t (*marmot_bc_exec_fn)(
    const void *backend_exec_ctx, marmot_tensor_t **regs, const uint8_t *imm, const uint8_t *const_pool
);

typedef struct marmot_bc_program {
    uint8_t *code;
    size_t code_size;
    uint8_t *const_pool;
    size_t const_pool_size;
    uint16_t reg_count;
    uint16_t op_count;
    const uint16_t *imm_size;
    const marmot_bc_exec_fn *exec_table;
} marmot_bc_program_t;

typedef struct marmot_bc_hook_info {
    const marmot_bc_program_t *program;
    const void *backend_exec_ctx;
    marmot_tensor_t **regs;
    size_t instr_index;
    uint16_t op;
    const uint8_t *imm;
    uint16_t imm_size;
} marmot_bc_hook_info_t;

typedef marmot_error_t (*marmot_bc_before_op_fn)(const marmot_bc_hook_info_t *info, void *user_data, void *op_state);

typedef marmot_error_t (*marmot_bc_after_op_fn)(
    const marmot_bc_hook_info_t *info, marmot_error_t status, void *user_data, void *op_state
);

typedef void (*marmot_bc_on_start_fn)(
    const marmot_bc_program_t *program, const void *backend_exec_ctx, marmot_tensor_t **regs, void *user_data
);

typedef void (*marmot_bc_on_finish_fn)(
    const marmot_bc_program_t *program, const void *backend_exec_ctx, marmot_tensor_t **regs, marmot_error_t status,
    void *user_data
);

typedef struct marmot_bc_hooks {
    void *user_data;
    void *op_state;
    size_t op_state_size;
    marmot_bc_on_start_fn on_start;
    marmot_bc_before_op_fn before_op;
    marmot_bc_after_op_fn after_op;
    marmot_bc_on_finish_fn on_finish;
} marmot_bc_hooks_t;

typedef struct marmot_bc_builder {
    uint8_t *code;
    size_t code_size;
    size_t code_capacity;
    uint8_t *const_pool;
    size_t const_pool_size;
    size_t const_pool_capacity;
    bool ok;
} marmot_bc_builder_t;

MARMOT_NODISCARD bool marmot_bc_builder_init(marmot_bc_builder_t *builder);
void marmot_bc_builder_reset(marmot_bc_builder_t *builder);

MARMOT_NODISCARD bool marmot_bc_builder_emit_bytes(marmot_bc_builder_t *builder, const void *data, size_t size);
MARMOT_NODISCARD bool marmot_bc_builder_emit_u8(marmot_bc_builder_t *builder, uint8_t value);
MARMOT_NODISCARD bool marmot_bc_builder_emit_u16(marmot_bc_builder_t *builder, uint16_t value);
MARMOT_NODISCARD bool marmot_bc_builder_emit_u32(marmot_bc_builder_t *builder, uint32_t value);
MARMOT_NODISCARD bool marmot_bc_builder_emit_u64(marmot_bc_builder_t *builder, uint64_t value);
MARMOT_NODISCARD bool marmot_bc_builder_emit_f32(marmot_bc_builder_t *builder, float value);

MARMOT_NODISCARD uint32_t
marmot_bc_builder_add_const(marmot_bc_builder_t *builder, const void *data, size_t size, size_t alignment);

MARMOT_NODISCARD bool marmot_bc_builder_finish(
    marmot_bc_builder_t *builder, marmot_bc_program_t *program, const uint16_t *imm_size,
    const marmot_bc_exec_fn *exec_table, uint16_t reg_count, uint16_t op_count
);

void marmot_bc_program_destroy(marmot_bc_program_t *program);

MARMOT_NODISCARD marmot_error_t
marmot_bc_execute(const marmot_bc_program_t *program, const void *backend_exec_ctx, marmot_tensor_t **regs);

MARMOT_NODISCARD marmot_error_t marmot_bc_execute_with_hooks(
    const marmot_bc_program_t *program, const void *backend_exec_ctx, marmot_tensor_t **regs,
    const marmot_bc_hooks_t *hooks
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_BYTECODE_H
