#ifndef MARMOT_BENCH_MODEL_H
#define MARMOT_BENCH_MODEL_H

#include "marmot/error.h"
#include "marmot/inference/model.h"
#include "marmot/types.h"

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Model configuration for LLM benchmarks
typedef struct {
    const char *model_path;        // Path to GGUF model file
    size_t ctx_size;               // Context size (-c)
    size_t batch_size;             // Batch size (-b)
    size_t ubatch_size;            // Micro-batch size (-ub)
    size_t max_seqs;               // Serving engine max sequences
    size_t max_batch_seqs;         // Serving engine max sequences per step
    marmot_dtype_t kv_type_k;      // KV cache K type (-ctk)
    marmot_dtype_t kv_type_v;      // KV cache V type (-ctv)
    size_t gpu_layers;             // GPU layers to offload (-ngl)
    bool flash_attn;               // Use flash attention (-fa)
    bool create_engine;            // Create serving engine sidecar
    size_t n_threads;              // Number of threads (-t)
    marmot_backend_type_t backend; // Backend to use
} marmot_bench_model_config_t;

// Loaded model for benchmarking
typedef struct {
    void *model;  // marmot_model_t*
    void *engine; // marmot_serving_engine_t*
    marmot_context_t *ctx;
    marmot_bench_model_config_t config;
    marmot_model_info_t info;
    bool loaded;
} marmot_bench_model_t;

// Initialize model config with defaults
void marmot_bench_model_config_init(marmot_bench_model_config_t *config);

// Load model from GGUF file
[[nodiscard]] marmot_error_t marmot_bench_model_load(const marmot_bench_model_config_t *config, marmot_bench_model_t *out
);

// Free model resources
void marmot_bench_model_free(marmot_bench_model_t *model);

// Get model info string for output
const char *marmot_bench_model_info(const marmot_bench_model_t *model);

// Parse KV cache type from string (f16, f32, q8_0)
[[nodiscard]] marmot_error_t marmot_bench_parse_kv_type(const char *str, marmot_dtype_t *out);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_BENCH_MODEL_H
