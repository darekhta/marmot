#ifndef MARMOT_GRAPH_GGUF_LOADER_H
#define MARMOT_GRAPH_GGUF_LOADER_H

#include "marmot/error.h"
#include "marmot/graph/graph.h"
#include "marmot/macros.h"
#include "marmot/tensor.h"
#include "marmot/traits_ids.gen.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//==============================================================================
// Extendable loader API (C-facing)
//==============================================================================

#define MARMOT_GGUF_OPTIONS_VERSION 2
#define MARMOT_GGUF_CAPS_VERSION 1

typedef enum {
    MARMOT_GGUF_FLAG_STRICT_VALIDATION = 1u << 0,
    MARMOT_GGUF_FLAG_ALLOW_UNKNOWN_OPS = 1u << 1,
    MARMOT_GGUF_FLAG_AUTO_BACKEND = 1u << 2,
} marmot_gguf_loader_flags_t;

typedef struct marmot_gguf_loader marmot_gguf_loader_t;
struct marmot_allocator;
typedef struct marmot_packed_graph_options marmot_packed_graph_options_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;
    marmot_backend_type_t backend;
    const struct marmot_allocator *allocator;
    marmot_context_t *ctx;
    const marmot_packed_graph_options_t *packed_opts;
    const void *pnext;
    uint64_t reserved[2];
} marmot_gguf_options_t;

typedef struct {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t supported_flags;
    uint32_t supported_version_min;
    uint32_t supported_version_max;
    uint64_t reserved[4];
} marmot_gguf_caps_t;

MARMOT_NODISCARD marmot_error_t marmot_gguf_options_init(marmot_gguf_options_t *opts);
MARMOT_NODISCARD marmot_error_t
marmot_gguf_loader_create(const marmot_gguf_options_t *opts, marmot_gguf_loader_t **out_loader);
void marmot_gguf_loader_destroy(marmot_gguf_loader_t *loader);
MARMOT_NODISCARD marmot_error_t
marmot_gguf_loader_load_file(marmot_gguf_loader_t *loader, const char *path, marmot_graph_t **out_graph);
MARMOT_NODISCARD marmot_error_t
marmot_gguf_loader_load_memory(marmot_gguf_loader_t *loader, const void *data, size_t len, marmot_graph_t **out_graph);
MARMOT_NODISCARD marmot_error_t
marmot_gguf_loader_query_capabilities(const marmot_gguf_loader_t *loader, marmot_gguf_caps_t *out_caps);
MARMOT_NODISCARD const marmot_error_info_t *marmot_gguf_loader_last_error(const marmot_gguf_loader_t *loader);

//==============================================================================
// Legacy GGUF file structures (still exposed for compatibility)
//==============================================================================

typedef enum {
    MARMOT_GGUF_TYPE_UINT8 = 0,
    MARMOT_GGUF_TYPE_INT8 = 1,
    MARMOT_GGUF_TYPE_UINT16 = 2,
    MARMOT_GGUF_TYPE_INT16 = 3,
    MARMOT_GGUF_TYPE_UINT32 = 4,
    MARMOT_GGUF_TYPE_INT32 = 5,
    MARMOT_GGUF_TYPE_FLOAT32 = 6,
    MARMOT_GGUF_TYPE_BOOL = 7,
    MARMOT_GGUF_TYPE_STRING = 8,
    MARMOT_GGUF_TYPE_ARRAY = 9,
    MARMOT_GGUF_TYPE_UINT64 = 10,
    MARMOT_GGUF_TYPE_INT64 = 11,
    MARMOT_GGUF_TYPE_FLOAT64 = 12,
} marmot_gguf_value_type_t;

typedef struct {
    char *data;
    size_t length;
} marmot_gguf_string_t;

typedef union {
    uint8_t *uint8_values;
    int8_t *int8_values;
    uint16_t *uint16_values;
    int16_t *int16_values;
    uint32_t *uint32_values;
    int32_t *int32_values;
    uint64_t *uint64_values;
    int64_t *int64_values;
    float *float32_values;
    double *float64_values;
    bool *bool_values;
    marmot_gguf_string_t *string_values;
    void *raw;
} marmot_gguf_array_data_t;

typedef struct {
    marmot_gguf_value_type_t type;
    size_t length;
    marmot_gguf_array_data_t data;
} marmot_gguf_array_t;

typedef struct {
    marmot_gguf_value_type_t type;
    union {
        uint8_t uint8_value;
        int8_t int8_value;
        uint16_t uint16_value;
        int16_t int16_value;
        uint32_t uint32_value;
        int32_t int32_value;
        uint64_t uint64_value;
        int64_t int64_value;
        float float32_value;
        double float64_value;
        bool bool_value;
        marmot_gguf_string_t string_value;
        marmot_gguf_array_t array_value;
    } data;
} marmot_gguf_value_t;

typedef struct {
    char *key;
    marmot_gguf_value_t value;
} marmot_gguf_kv_t;

typedef struct {
    char *name;
    marmot_tensor_t *tensor;
    marmot_qscheme_id_t qscheme_id;
    uint32_t ggml_type;
    uint64_t data_offset;
    size_t byte_length;
} marmot_gguf_tensor_t;

typedef struct marmot_gguf {
    uint32_t version;
    size_t alignment;
    size_t tensor_data_offset;
    size_t kv_count;
    size_t tensor_count;
    marmot_gguf_kv_t *kv;
    marmot_gguf_tensor_t *tensors;
    void *data;
    size_t size;
    int fd;
    marmot_context_t *ctx;
} marmot_gguf_t;

MARMOT_NODISCARD marmot_gguf_t *marmot_gguf_load(const char *path);
void marmot_gguf_unload(marmot_gguf_t *gguf);

const marmot_gguf_kv_t *marmot_gguf_find_kv(const marmot_gguf_t *gguf, const char *key);
const marmot_gguf_tensor_t *marmot_gguf_find_tensor(const marmot_gguf_t *gguf, const char *name);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_GRAPH_GGUF_LOADER_H
