#include "binding_table.hpp"

namespace marmot::graph {

void TensorDeleter::operator()(marmot_tensor_t *ptr) const {
    if (ptr)
        marmot_tensor_destroy(ptr);
}

BindingTable::BindingTable(size_t size) : bindings_(size, nullptr), owned_(size) {}

std::span<marmot_tensor_t *> BindingTable::bindings() {
    return bindings_;
}

void BindingTable::set(size_t index, marmot_tensor_t *ptr) {
    bindings_[index] = ptr;
}

marmot_error_t BindingTable::emplace_owned(size_t index, marmot_tensor_t *ptr) {
    if (!ptr)
        return MARMOT_ERROR_OUT_OF_MEMORY;
    owned_[index].reset(ptr);
    bindings_[index] = ptr;
    return MARMOT_SUCCESS;
}

} // namespace marmot::graph
