#ifndef MARMOT_GRAPH_GGUF_MODEL_H
#define MARMOT_GRAPH_GGUF_MODEL_H

#include "marmot/macros.h"
#include "marmot/types.h"

#include "gguf_loader.h"
#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct marmot_gguf_model marmot_gguf_model_t;

typedef struct {
    marmot_architecture_t architecture;
    size_t context_length;
    size_t n_embd;
    size_t n_layer;
    size_t n_head;
    size_t n_head_kv;
    size_t n_vocab;
    size_t ff_length;
    size_t rope_dimension;
    size_t head_dim; // explicit head dimension (for GQA models like Qwen3)
    float rope_freq_base;
    marmot_rope_type_t rope_type;
    marmot_rope_scaling_type_t rope_scaling_type;
    float rope_freq_scale;
    float rope_ext_factor;
    float rope_attn_factor;
    float rope_beta_fast;
    float rope_beta_slow;
    uint32_t rope_orig_ctx_len;
    float rms_norm_eps;
    float embedding_scale;
} marmot_gguf_model_meta_t;

// Packed token graph build options (paged attention + token metadata)
#define MARMOT_PACKED_GRAPH_OPTIONS_VERSION 1

typedef enum {
    MARMOT_PACKED_GRAPH_FLAG_NONE = 0,
    // Select kv_dtype automatically per backend/architecture (vLLM-style default).
    // When set, graph builders ignore kv_dtype and use the backend activation dtype instead.
    MARMOT_PACKED_GRAPH_FLAG_KV_DTYPE_AUTO = 1u << 0,
} marmot_packed_graph_flags_t;

struct marmot_packed_graph_options {
    uint32_t struct_size;
    uint32_t struct_version;
    uint64_t flags;

    size_t token_count;
    size_t sample_count;

    size_t max_seqs;
    size_t max_seq_len;
    size_t block_size;
    size_t num_kv_blocks;

    marmot_dtype_t kv_dtype;

    const void *pnext;
    uint64_t reserved[4];
};

MARMOT_NODISCARD marmot_error_t marmot_packed_graph_options_init(marmot_packed_graph_options_t *opts);

MARMOT_NODISCARD marmot_error_t
marmot_gguf_model_load(const char *path, marmot_backend_type_t backend, marmot_gguf_model_t **out_model);
void marmot_gguf_model_destroy(marmot_gguf_model_t *model);

const marmot_gguf_t *marmot_gguf_model_file(const marmot_gguf_model_t *model);
const marmot_tensor_t *marmot_gguf_model_tensor(const marmot_gguf_model_t *model, const char *name);
const marmot_gguf_tensor_t *marmot_gguf_model_tensor_info(const marmot_gguf_model_t *model, size_t index);
size_t marmot_gguf_model_tensor_count(const marmot_gguf_model_t *model);
bool marmot_gguf_model_metadata(const marmot_gguf_model_t *model, marmot_gguf_model_meta_t *out);

MARMOT_NODISCARD marmot_error_t marmot_graph_from_model_packed(
    const marmot_gguf_model_t *model, marmot_backend_type_t backend, const marmot_packed_graph_options_t *opts,
    marmot_graph_t **out_graph
);

MARMOT_NODISCARD marmot_error_t marmot_graph_from_gguf_packed(
    const char *path, marmot_backend_type_t backend, const marmot_packed_graph_options_t *opts,
    marmot_graph_t **out_graph
);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_GRAPH_GGUF_MODEL_H
