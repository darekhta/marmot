#pragma once

#include "marmot/graph/kernel_selection.h"
#include "marmot/graph/op_signature.h"
#include "marmot/types.h"

#include "internal/kernel_selection.h"

namespace marmot::graph {

[[nodiscard]] marmot_kernel_selection_t
query_backend_for_node(marmot_backend_type_t backend, const marmot_op_signature_t *signature);

} // namespace marmot::graph
