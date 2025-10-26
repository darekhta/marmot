#ifndef MARMOT_BENCH_LLM_H
#define MARMOT_BENCH_LLM_H

#include "bench_model.h"
#include "bench_stats.h"
#include "marmot/error.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// LLM benchmark parameters
typedef struct {
    size_t n_prompt;    // -p: Number of prompt tokens to process
    size_t n_gen;       // -n: Number of tokens to generate
    size_t n_depth;     // -d: Pre-filled KV cache depth
    size_t n_seqs;      // --concurrency: Concurrent requests
} marmot_bench_llm_params_t;

// LLM benchmark results
typedef struct {
    // Throughput metrics
    double pp_tokens_per_sec;   // Prompt processing throughput
    double tg_tokens_per_sec;   // Token generation throughput

    // Latency metrics (nanoseconds)
    double pp_total_ns;         // Total prompt processing time
    double tg_total_ns;         // Total token generation time
    double ttft_ns;             // Time to first token

    // Detailed statistics
    marmot_bench_stats_t pp_stats;  // Per-token prompt processing stats
    marmot_bench_stats_t tg_stats;  // Per-token generation stats

    // Config info (for output)
    size_t n_prompt;
    size_t n_gen;
    size_t n_depth;
    size_t n_seqs;
} marmot_bench_llm_result_t;

// Initialize LLM params with defaults
void marmot_bench_llm_params_init(marmot_bench_llm_params_t *params);

// Run LLM benchmark (prompt processing + token generation)
[[nodiscard]] marmot_error_t marmot_bench_llm_run(
    const marmot_bench_model_t *model,
    const marmot_bench_llm_params_t *params,
    size_t repetitions,
    marmot_bench_llm_result_t *result
);

// Run prompt processing only
[[nodiscard]] marmot_error_t marmot_bench_llm_run_pp(
    const marmot_bench_model_t *model,
    const marmot_bench_llm_params_t *params,
    size_t repetitions,
    marmot_bench_llm_result_t *result
);

// Run token generation only
[[nodiscard]] marmot_error_t marmot_bench_llm_run_tg(
    const marmot_bench_model_t *model,
    const marmot_bench_llm_params_t *params,
    size_t repetitions,
    marmot_bench_llm_result_t *result
);

// Format result as string (for console output)
const char *marmot_bench_llm_result_str(const marmot_bench_llm_result_t *result);

#ifdef __cplusplus
}
#endif

#endif // MARMOT_BENCH_LLM_H
