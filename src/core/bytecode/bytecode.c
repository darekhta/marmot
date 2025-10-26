#include "core/bytecode/bytecode.h"

#include <stdlib.h>

#include <limits.h>
#include <string.h>

static bool marmot_bc_reserve(uint8_t **buffer, size_t *capacity, size_t required) {
    if (buffer == nullptr || capacity == nullptr) {
        return false;
    }
    if (required <= *capacity) {
        return true;
    }
    size_t new_capacity = *capacity == 0 ? 64 : *capacity;
    while (new_capacity < required) {
        size_t next = new_capacity * 2;
        if (next <= new_capacity) {
            return false;
        }
        new_capacity = next;
    }
    void *ptr = realloc(*buffer, new_capacity);
    if (ptr == nullptr) {
        return false;
    }
    *buffer = ptr;
    *capacity = new_capacity;
    return true;
}

static bool marmot_bc_align_up(size_t value, size_t alignment, size_t *out) {
    if (out == nullptr) {
        return false;
    }
    if (alignment == 0) {
        alignment = 1;
    }
    size_t padded = value + alignment - 1;
    if (padded < value) {
        return false;
    }
    *out = (padded / alignment) * alignment;
    return true;
}

bool marmot_bc_builder_init(marmot_bc_builder_t *builder) {
    if (builder == nullptr) {
        return false;
    }
    *builder = (marmot_bc_builder_t){0};
    builder->ok = true;
    return true;
}

void marmot_bc_builder_reset(marmot_bc_builder_t *builder) {
    if (builder == nullptr) {
        return;
    }
    free(builder->code);
    free(builder->const_pool);
    *builder = (marmot_bc_builder_t){0};
}

bool marmot_bc_builder_emit_bytes(marmot_bc_builder_t *builder, const void *data, size_t size) {
    if (builder == nullptr || !builder->ok) {
        return false;
    }
    if (size == 0) {
        return true;
    }
    if (data == nullptr) {
        builder->ok = false;
        return false;
    }
    if (!marmot_bc_reserve(&builder->code, &builder->code_capacity, builder->code_size + size)) {
        builder->ok = false;
        return false;
    }
    memcpy(builder->code + builder->code_size, data, size);
    builder->code_size += size;
    return true;
}

bool marmot_bc_builder_emit_u8(marmot_bc_builder_t *builder, uint8_t value) {
    return marmot_bc_builder_emit_bytes(builder, &value, sizeof(value));
}

bool marmot_bc_builder_emit_u16(marmot_bc_builder_t *builder, uint16_t value) {
    return marmot_bc_builder_emit_bytes(builder, &value, sizeof(value));
}

bool marmot_bc_builder_emit_u32(marmot_bc_builder_t *builder, uint32_t value) {
    return marmot_bc_builder_emit_bytes(builder, &value, sizeof(value));
}

bool marmot_bc_builder_emit_u64(marmot_bc_builder_t *builder, uint64_t value) {
    return marmot_bc_builder_emit_bytes(builder, &value, sizeof(value));
}

bool marmot_bc_builder_emit_f32(marmot_bc_builder_t *builder, float value) {
    return marmot_bc_builder_emit_bytes(builder, &value, sizeof(value));
}

uint32_t marmot_bc_builder_add_const(marmot_bc_builder_t *builder, const void *data, size_t size, size_t alignment) {
    if (builder == nullptr || !builder->ok) {
        return MARMOT_BC_INVALID_OFFSET;
    }
    if (size == 0) {
        return (uint32_t)builder->const_pool_size;
    }
    if (data == nullptr) {
        builder->ok = false;
        return MARMOT_BC_INVALID_OFFSET;
    }

    size_t aligned = 0;
    if (!marmot_bc_align_up(builder->const_pool_size, alignment, &aligned)) {
        builder->ok = false;
        return MARMOT_BC_INVALID_OFFSET;
    }
    if (aligned > UINT32_MAX || size > UINT32_MAX || aligned + size > UINT32_MAX) {
        builder->ok = false;
        return MARMOT_BC_INVALID_OFFSET;
    }

    if (!marmot_bc_reserve(&builder->const_pool, &builder->const_pool_capacity, aligned + size)) {
        builder->ok = false;
        return MARMOT_BC_INVALID_OFFSET;
    }
    if (aligned > builder->const_pool_size) {
        memset(builder->const_pool + builder->const_pool_size, 0, aligned - builder->const_pool_size);
    }
    memcpy(builder->const_pool + aligned, data, size);
    builder->const_pool_size = aligned + size;
    return (uint32_t)aligned;
}

bool marmot_bc_builder_finish(
    marmot_bc_builder_t *builder, marmot_bc_program_t *program, const uint16_t *imm_size,
    const marmot_bc_exec_fn *exec_table, uint16_t reg_count, uint16_t op_count
) {
    if (builder == nullptr || program == nullptr) {
        return false;
    }
    if (!builder->ok) {
        return false;
    }
    if (imm_size == nullptr || exec_table == nullptr || op_count == 0) {
        return false;
    }

    *program = (marmot_bc_program_t){
        .code = builder->code,
        .code_size = builder->code_size,
        .const_pool = builder->const_pool,
        .const_pool_size = builder->const_pool_size,
        .reg_count = reg_count,
        .op_count = op_count,
        .imm_size = imm_size,
        .exec_table = exec_table,
    };

    builder->code = nullptr;
    builder->code_size = 0;
    builder->code_capacity = 0;
    builder->const_pool = nullptr;
    builder->const_pool_size = 0;
    builder->const_pool_capacity = 0;
    builder->ok = true;

    return true;
}

void marmot_bc_program_destroy(marmot_bc_program_t *program) {
    if (program == nullptr) {
        return;
    }
    free(program->code);
    free(program->const_pool);
    *program = (marmot_bc_program_t){0};
}

static bool marmot_bc_read_u16(const uint8_t **pc, const uint8_t *end, uint16_t *out) {
    if (pc == nullptr || out == nullptr || *pc == nullptr) {
        return false;
    }
    if (*pc + sizeof(uint16_t) > end) {
        return false;
    }
    memcpy(out, *pc, sizeof(uint16_t));
    *pc += sizeof(uint16_t);
    return true;
}

marmot_error_t marmot_bc_execute_with_hooks(
    const marmot_bc_program_t *program, const void *backend_exec_ctx, marmot_tensor_t **regs,
    const marmot_bc_hooks_t *hooks
) {
    if (program == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Null bytecode program");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (program->exec_table == nullptr || program->imm_size == nullptr || program->op_count == 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid bytecode tables");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (program->code == nullptr && program->code_size != 0) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Invalid bytecode buffer");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    const uint8_t *pc = program->code;
    const uint8_t *end = program->code + program->code_size;
    if (hooks != nullptr && hooks->on_start != nullptr) {
        hooks->on_start(program, backend_exec_ctx, regs, hooks->user_data);
    }
    size_t instr_index = 0;
    marmot_error_t status = MARMOT_SUCCESS;

    for (;;) {
        uint16_t op = 0;
        if (!marmot_bc_read_u16(&pc, end, &op)) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Truncated bytecode stream");
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            break;
        }
        if (op == MARMOT_BC_END) {
            status = MARMOT_SUCCESS;
            break;
        }
        if (op >= program->op_count) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Bytecode opcode out of range");
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            break;
        }
        uint16_t imm_size = program->imm_size[op];
        if (pc + imm_size > end) {
            marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Bytecode immediate overrun");
            status = MARMOT_ERROR_INVALID_ARGUMENT;
            break;
        }
        const uint8_t *imm = pc;
        pc += imm_size;

        marmot_bc_hook_info_t info = {
            .program = program,
            .backend_exec_ctx = backend_exec_ctx,
            .regs = regs,
            .instr_index = instr_index,
            .op = op,
            .imm = imm,
            .imm_size = imm_size,
        };
        void *op_state = nullptr;
        if (hooks != nullptr && hooks->op_state != nullptr && hooks->op_state_size > 0) {
            op_state = hooks->op_state;
            memset(op_state, 0, hooks->op_state_size);
        }

        marmot_error_t step_status = MARMOT_SUCCESS;
        if (hooks != nullptr && hooks->before_op != nullptr) {
            step_status = hooks->before_op(&info, hooks->user_data, op_state);
        }
        if (step_status == MARMOT_SUCCESS) {
            marmot_bc_exec_fn exec_fn = program->exec_table[op];
            if (exec_fn == nullptr) {
                marmot_set_error(MARMOT_ERROR_NOT_IMPLEMENTED, "Bytecode executor missing");
                step_status = MARMOT_ERROR_NOT_IMPLEMENTED;
            } else {
                step_status = exec_fn(backend_exec_ctx, regs, imm, program->const_pool);
            }
        }
        if (hooks != nullptr && hooks->after_op != nullptr) {
            step_status = hooks->after_op(&info, step_status, hooks->user_data, op_state);
        }
        if (step_status != MARMOT_SUCCESS) {
            status = step_status;
            break;
        }
        instr_index++;
    }
    if (hooks != nullptr && hooks->on_finish != nullptr) {
        hooks->on_finish(program, backend_exec_ctx, regs, status, hooks->user_data);
    }
    return status;
}

marmot_error_t
marmot_bc_execute(const marmot_bc_program_t *program, const void *backend_exec_ctx, marmot_tensor_t **regs) {
    return marmot_bc_execute_with_hooks(program, backend_exec_ctx, regs, nullptr);
}
