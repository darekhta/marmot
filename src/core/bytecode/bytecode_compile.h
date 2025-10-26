#ifndef MARMOT_CORE_BYTECODE_COMPILE_H
#define MARMOT_CORE_BYTECODE_COMPILE_H

#include "marmot/device.h"
#include "marmot/graph/op_signature.h"

#include <stdbool.h>
#include <stdint.h>

#include "core/bytecode/bytecode.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct marmot_bc_selection {
    bool supported;
    uint16_t op_index;
    marmot_op_signature_t resolved_sig;
    const char *reason;
} marmot_bc_selection_t;

typedef struct marmot_bc_tables {
    const uint16_t *imm_size;
    const marmot_bc_exec_fn *exec_table;
    const marmot_bc_schema_id_t *schema_id;
    uint16_t op_count;
} marmot_bc_tables_t;

MARMOT_NODISCARD marmot_bc_selection_t
marmot_bc_compile_signature(const marmot_context_t *ctx, const marmot_op_signature_t *sig);

MARMOT_NODISCARD marmot_bc_selection_t marmot_bc_compile_signature_with_caps(
    marmot_backend_type_t backend, const marmot_device_caps_t *caps, const marmot_op_signature_t *sig,
    bool allow_fallback
);

MARMOT_NODISCARD bool marmot_bc_get_tables(marmot_backend_type_t backend, marmot_bc_tables_t *out);

MARMOT_NODISCARD bool marmot_bc_exec_supported(marmot_backend_type_t backend, uint16_t op_index);

MARMOT_NODISCARD marmot_error_t
marmot_bc_try_execute_signature(const marmot_context_t *ctx, const marmot_op_signature_t *sig, const void *args);

MARMOT_NODISCARD marmot_error_t marmot_bc_execute_op(
    marmot_backend_type_t backend, uint16_t op_index, const marmot_bc_exec_ctx_t *exec_ctx, const void *args
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_CORE_BYTECODE_COMPILE_H
