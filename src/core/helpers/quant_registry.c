#include "marmot/error.h"
#include "marmot/quant_traits.h"

#define MARMOT_MAX_QUANT_SCHEMES 64

static const marmot_quant_traits_t *g_quant_registry[MARMOT_MAX_QUANT_SCHEMES];
static size_t g_quant_registry_size = 0;

const marmot_quant_traits_t *marmot_get_quant_traits(marmot_quant_kind_t kind) {
    for (size_t i = 0; i < g_quant_registry_size; i++) {
        if (g_quant_registry[i]->kind == kind) {
            return g_quant_registry[i];
        }
    }
    return nullptr;
}

marmot_error_t marmot_quant_register_scheme(const marmot_quant_traits_t *traits) {
    if (traits == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quant traits pointer cannot be null");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (marmot_get_quant_traits(traits->kind) != nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Quant kind already registered");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (g_quant_registry_size >= MARMOT_MAX_QUANT_SCHEMES) {
        marmot_set_error(MARMOT_ERROR_OUT_OF_MEMORY, "Quant trait registry is full");
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    g_quant_registry[g_quant_registry_size++] = traits;
    return MARMOT_SUCCESS;
}
