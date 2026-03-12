#include "bench_llm.h"
#include "bench_stats.h"

#include "marmot/inference/engine.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void marmot_bench_llm_params_init(marmot_bench_llm_params_t *params) {
    params->n_prompt = 512;
    params->n_gen = 128;
    params->n_depth = 0;
    params->n_seqs = 1;
    params->warmup_runs = 1;
    params->mode = MARMOT_BENCH_LLM_MODE_DIRECT;
}

static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// Generate random token IDs for benchmarking (not semantically meaningful)
static void generate_random_tokens(marmot_token_id_t *tokens, size_t count, size_t vocab_size) {
    for (size_t i = 0; i < count; i++) {
        tokens[i] = (marmot_token_id_t)(rand() % vocab_size);
    }
}

typedef struct {
    marmot_token_id_t *tokens;
    size_t capacity;
    size_t count;
} token_capture_t;

typedef struct {
    marmot_llm_token_callback_t inner_cb;
    void *inner_user_data;
    size_t total_tokens;
    size_t step_tokens;
} bench_token_observer_t;

static void capture_token(void *user_data, marmot_token_id_t token_id) {
    token_capture_t *capture = (token_capture_t *)user_data;
    if (capture == nullptr || capture->tokens == nullptr) {
        return;
    }
    if (capture->count < capture->capacity) {
        capture->tokens[capture->count++] = token_id;
    }
}

static void observe_token(void *user_data, marmot_token_id_t token_id) {
    bench_token_observer_t *observer = (bench_token_observer_t *)user_data;
    if (observer == nullptr) {
        return;
    }
    observer->total_tokens++;
    observer->step_tokens++;
    if (observer->inner_cb != nullptr) {
        observer->inner_cb(observer->inner_user_data, token_id);
    }
}

static void prepare_bench_generate_options(
    const marmot_llm_generate_options_t *src, marmot_llm_generate_options_t *dst, bench_token_observer_t *observer
) {
    *dst = *src;
    *observer = (bench_token_observer_t){
        .inner_cb = src->on_token,
        .inner_user_data = src->user_data,
    };
    if (src->max_new_tokens > 0) {
        dst->on_token = observe_token;
        dst->user_data = observer;
    }
}

typedef struct {
    uint64_t start_ns;
    uint64_t decode_start_ns;
    uint64_t end_ns;
} request_timing_t;

static marmot_error_t run_serving_request(
    marmot_serving_engine_t *engine, const marmot_token_id_t *prompt_tokens, size_t prompt_len,
    marmot_llm_generate_options_t *gen_opts, marmot_llm_sampling_options_t *sampling_opts, request_timing_t *timing
) {
    if (engine == nullptr || gen_opts == nullptr || sampling_opts == nullptr || timing == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    timing->start_ns = get_time_ns();
    timing->decode_start_ns = 0;
    timing->end_ns = timing->start_ns;

    marmot_llm_generate_options_t bench_gen_opts;
    bench_token_observer_t token_observer = {0};
    prepare_bench_generate_options(gen_opts, &bench_gen_opts, &token_observer);

    marmot_request_id_t request_id = 0;
    marmot_error_t err =
        marmot_serving_engine_submit(engine, prompt_tokens, prompt_len, &bench_gen_opts, sampling_opts, &request_id);
    if (err != MARMOT_SUCCESS) {
        return err;
    }

    size_t token_budget = prompt_len + gen_opts->max_new_tokens;
    if (token_budget == 0) {
        token_budget = 1;
    }
    const size_t max_steps = token_budget + 16;

    for (size_t step = 0; step < max_steps; ++step) {
        size_t steps_done = 0;
        token_observer.step_tokens = 0;
        const uint64_t step_start_ns = get_time_ns();
        err = marmot_serving_engine_step(engine, 1, &steps_done);
        if (err != MARMOT_SUCCESS) {
            (void)marmot_serving_engine_request_release(engine, request_id);
            return err;
        }

        marmot_llm_request_state_t state = marmot_serving_engine_request_state(engine, request_id);
        if (timing->decode_start_ns == 0 &&
            (state == MARMOT_LLM_REQUEST_STATE_DECODING || token_observer.step_tokens > 0)) {
            timing->decode_start_ns = step_start_ns;
        }
        if (state == MARMOT_LLM_REQUEST_STATE_FAILED || state == MARMOT_LLM_REQUEST_STATE_CANCELED) {
            (void)marmot_serving_engine_request_release(engine, request_id);
            return MARMOT_ERROR_INVALID_OPERATION;
        }
        if (state == MARMOT_LLM_REQUEST_STATE_DONE || state == MARMOT_LLM_REQUEST_STATE_INVALID) {
            timing->end_ns = get_time_ns();
            if (timing->decode_start_ns == 0) {
                timing->decode_start_ns = (prompt_len == 0 && gen_opts->max_new_tokens > 0) ? timing->start_ns : timing->end_ns;
            }
            return MARMOT_SUCCESS;
        }
    }

    (void)marmot_serving_engine_request_release(engine, request_id);
    return MARMOT_ERROR_INVALID_OPERATION;
}

static marmot_error_t run_serving_requests(
    marmot_serving_engine_t *engine, const marmot_token_id_t *prompt_tokens, size_t prompt_len,
    marmot_llm_generate_options_t *gen_opts, marmot_llm_sampling_options_t *sampling_opts, size_t num_requests,
    request_timing_t *timing
) {
    if (engine == nullptr || gen_opts == nullptr || sampling_opts == nullptr || timing == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }
    if (num_requests == 0) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    timing->start_ns = get_time_ns();
    timing->decode_start_ns = 0;
    timing->end_ns = timing->start_ns;

    marmot_llm_generate_options_t bench_gen_opts;
    bench_token_observer_t token_observer = {0};
    prepare_bench_generate_options(gen_opts, &bench_gen_opts, &token_observer);

    marmot_request_id_t *request_ids = malloc(num_requests * sizeof(*request_ids));
    if (request_ids == nullptr) {
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    size_t submitted = 0;
    marmot_error_t err = MARMOT_SUCCESS;
    for (; submitted < num_requests; ++submitted) {
        marmot_request_id_t request_id = 0;
        err = marmot_serving_engine_submit(engine, prompt_tokens, prompt_len, &bench_gen_opts, sampling_opts, &request_id);
        if (err != MARMOT_SUCCESS) {
            break;
        }
        request_ids[submitted] = request_id;
    }
    if (err != MARMOT_SUCCESS) {
        for (size_t i = 0; i < submitted; ++i) {
            (void)marmot_serving_engine_request_release(engine, request_ids[i]);
        }
        free(request_ids);
        return err;
    }

    size_t token_budget = prompt_len + gen_opts->max_new_tokens;
    if (token_budget == 0) {
        token_budget = 1;
    }
    const size_t max_steps = num_requests * (token_budget + 16);

    for (size_t step = 0; step < max_steps; ++step) {
        size_t steps_done = 0;
        token_observer.step_tokens = 0;
        const uint64_t step_start_ns = get_time_ns();
        err = marmot_serving_engine_step(engine, 1, &steps_done);
        if (err != MARMOT_SUCCESS) {
            for (size_t i = 0; i < num_requests; ++i) {
                (void)marmot_serving_engine_request_release(engine, request_ids[i]);
            }
            free(request_ids);
            return err;
        }

        bool any_decoding = false;
        bool all_done = true;
        for (size_t i = 0; i < num_requests; ++i) {
            const marmot_request_id_t request_id = request_ids[i];
            const marmot_llm_request_state_t state = marmot_serving_engine_request_state(engine, request_id);
            if (state == MARMOT_LLM_REQUEST_STATE_DECODING) {
                any_decoding = true;
            }
            if (state == MARMOT_LLM_REQUEST_STATE_FAILED || state == MARMOT_LLM_REQUEST_STATE_CANCELED) {
                for (size_t j = 0; j < num_requests; ++j) {
                    (void)marmot_serving_engine_request_release(engine, request_ids[j]);
                }
                free(request_ids);
                return MARMOT_ERROR_INVALID_OPERATION;
            }
            if (state != MARMOT_LLM_REQUEST_STATE_DONE && state != MARMOT_LLM_REQUEST_STATE_INVALID) {
                all_done = false;
            }
        }

        if (timing->decode_start_ns == 0 && (any_decoding || token_observer.step_tokens > 0)) {
            timing->decode_start_ns = step_start_ns;
        }
        if (all_done) {
            timing->end_ns = get_time_ns();
            if (timing->decode_start_ns == 0) {
                timing->decode_start_ns = (prompt_len == 0 && gen_opts->max_new_tokens > 0) ? timing->start_ns : timing->end_ns;
            }
            for (size_t i = 0; i < num_requests; ++i) {
                (void)marmot_serving_engine_request_release(engine, request_ids[i]);
            }
            free(request_ids);
            return MARMOT_SUCCESS;
        }
    }

    for (size_t i = 0; i < num_requests; ++i) {
        (void)marmot_serving_engine_request_release(engine, request_ids[i]);
    }
    free(request_ids);
    return MARMOT_ERROR_INVALID_OPERATION;
}

static marmot_error_t prefill_depth(
    marmot_serving_engine_t *engine, const marmot_token_id_t *depth_tokens, size_t depth_len,
    marmot_llm_generate_options_t *gen_opts, marmot_llm_sampling_options_t *sampling_opts
) {
    if (depth_len == 0) {
        return MARMOT_SUCCESS;
    }
    gen_opts->max_new_tokens = 0;
    gen_opts->on_token = nullptr;
    gen_opts->user_data = nullptr;

    request_timing_t timing = {0};
    return run_serving_request(engine, depth_tokens, depth_len, gen_opts, sampling_opts, &timing);
}

marmot_error_t marmot_bench_llm_run_serving(
    const marmot_bench_model_t *model, const marmot_bench_llm_params_t *params, size_t repetitions,
    marmot_bench_llm_result_t *result
) {
    if (model == nullptr || params == nullptr || result == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    if (!model->loaded || model->engine == nullptr) {
        marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "Model not loaded");
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    // Initialize result
    memset(result, 0, sizeof(*result));
    result->n_prompt = params->n_prompt;
    result->n_gen = params->n_gen;
    result->n_depth = params->n_depth;
    result->n_seqs = params->n_seqs;
    result->mode = params->mode;

    marmot_serving_engine_t *engine = (marmot_serving_engine_t *)model->engine;

    // Allocate timing arrays
    double *pp_times_us = nullptr;
    double *tg_times_us = nullptr;

    if (params->n_prompt > 0) {
        pp_times_us = malloc(repetitions * sizeof(double));
        if (pp_times_us == nullptr) {
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    if (params->n_gen > 0) {
        tg_times_us = malloc(repetitions * sizeof(double));
        if (tg_times_us == nullptr) {
            free(pp_times_us);
            return MARMOT_ERROR_OUT_OF_MEMORY;
        }
    }

    // Allocate token buffers
    size_t vocab_size = model->info.n_vocab > 0 ? model->info.n_vocab : 32000;
    size_t total_prompt = params->n_prompt + params->n_depth;
    size_t output_capacity = params->n_gen;

    marmot_token_id_t *prompt_tokens = nullptr;
    marmot_token_id_t *output_tokens = nullptr;

    if (total_prompt > 0) {
        prompt_tokens = malloc(total_prompt * sizeof(marmot_token_id_t));
    }
    if (params->n_seqs <= 1 && output_capacity > 0) {
        output_tokens = malloc(output_capacity * sizeof(marmot_token_id_t));
    }

    if ((total_prompt > 0 && prompt_tokens == nullptr) ||
        (params->n_seqs <= 1 && output_capacity > 0 && output_tokens == nullptr)) {
        free(pp_times_us);
        free(tg_times_us);
        free(prompt_tokens);
        free(output_tokens);
        return MARMOT_ERROR_OUT_OF_MEMORY;
    }

    // Initialize options
    marmot_llm_generate_options_t gen_opts;
    marmot_llm_sampling_options_t sampling_opts;
    (void)marmot_llm_generate_options_init(&gen_opts);
    (void)marmot_llm_sampling_options_init(&sampling_opts);

    // Disable sampling randomness for deterministic benchmarking
    sampling_opts.temperature = 0.0f;
    sampling_opts.top_k = 1;

    marmot_error_t err = MARMOT_SUCCESS;

    const size_t total_runs = params->warmup_runs + repetitions;
    for (size_t run = 0; run < total_runs; run++) {
        // Benchmark prompt processing (prefill).
        if (params->n_prompt > 0 && pp_times_us != nullptr) {
            if (params->n_depth > 0 && prompt_tokens != nullptr) {
                generate_random_tokens(prompt_tokens, params->n_depth, vocab_size);
            }
            if (params->n_prompt > 0 && prompt_tokens != nullptr) {
                generate_random_tokens(prompt_tokens + params->n_depth, params->n_prompt, vocab_size);
            }

            err = prefill_depth(engine, prompt_tokens, params->n_depth, &gen_opts, &sampling_opts);
            if (err != MARMOT_SUCCESS) {
                goto cleanup;
            }

            gen_opts.max_new_tokens = 0;
            gen_opts.on_token = nullptr;
            gen_opts.user_data = nullptr;

            request_timing_t timing = {0};
            if (params->n_seqs > 1) {
                err = run_serving_requests(
                    engine, prompt_tokens, total_prompt, &gen_opts, &sampling_opts, params->n_seqs, &timing
                );
            } else {
                err = run_serving_request(engine, prompt_tokens, total_prompt, &gen_opts, &sampling_opts, &timing);
            }
            if (err != MARMOT_SUCCESS) {
                goto cleanup;
            }

            if (run >= params->warmup_runs) {
                const size_t rep = run - params->warmup_runs;
                pp_times_us[rep] = (double)(timing.end_ns - timing.start_ns) / 1000.0;
            }
            if (run == params->warmup_runs && params->n_gen == 0) {
                result->ttft_ns = (double)(timing.decode_start_ns - timing.start_ns);
            }
        }

        // Benchmark token generation.
        if (params->n_gen > 0 && tg_times_us != nullptr) {
            if (params->n_depth > 0 && prompt_tokens != nullptr) {
                generate_random_tokens(prompt_tokens, params->n_depth, vocab_size);
            }
            if (params->n_prompt > 0 && prompt_tokens != nullptr) {
                generate_random_tokens(prompt_tokens + params->n_depth, params->n_prompt, vocab_size);
            }

            err = prefill_depth(engine, prompt_tokens, params->n_depth, &gen_opts, &sampling_opts);
            if (err != MARMOT_SUCCESS) {
                goto cleanup;
            }

            gen_opts.max_new_tokens = params->n_gen;
            gen_opts.stop_on_eos = false;

            token_capture_t capture = {0};
            if (params->n_seqs <= 1) {
                capture = (token_capture_t){
                    .tokens = output_tokens,
                    .capacity = output_capacity,
                    .count = 0,
                };
                gen_opts.on_token = capture_token;
                gen_opts.user_data = &capture;
            } else {
                gen_opts.on_token = nullptr;
                gen_opts.user_data = nullptr;
            }

            request_timing_t timing = {0};
            if (params->n_seqs > 1) {
                err = run_serving_requests(
                    engine, prompt_tokens, total_prompt, &gen_opts, &sampling_opts, params->n_seqs, &timing
                );
            } else {
                err = run_serving_request(engine, prompt_tokens, total_prompt, &gen_opts, &sampling_opts, &timing);
            }
            if (err != MARMOT_SUCCESS) {
                goto cleanup;
            }

            if (run >= params->warmup_runs) {
                const size_t rep = run - params->warmup_runs;
                tg_times_us[rep] = (double)(timing.end_ns - timing.decode_start_ns) / 1000.0;
            }
            if (run == params->warmup_runs) {
                result->ttft_ns = (double)(timing.decode_start_ns - timing.start_ns);
            }
        }
    }

    // Compute statistics
    if (params->n_prompt > 0 && pp_times_us != nullptr) {
        marmot_bench_compute_stats(pp_times_us, repetitions, 0.95, &result->pp_stats);
        result->pp_total_ns = result->pp_stats.mean_us * 1000.0;
        if (result->pp_total_ns > 0) {
            const size_t tokens = params->n_prompt * (params->n_seqs == 0 ? 1 : params->n_seqs);
            result->pp_tokens_per_sec = (double)tokens / (result->pp_total_ns / 1e9);
        }
    }

    if (params->n_gen > 0 && tg_times_us != nullptr) {
        marmot_bench_compute_stats(tg_times_us, repetitions, 0.95, &result->tg_stats);
        result->tg_total_ns = result->tg_stats.mean_us * 1000.0;
        if (result->tg_total_ns > 0) {
            const size_t tokens = params->n_gen * (params->n_seqs == 0 ? 1 : params->n_seqs);
            result->tg_tokens_per_sec = (double)tokens / (result->tg_total_ns / 1e9);
        }
    }

cleanup:
    free(pp_times_us);
    free(tg_times_us);
    free(prompt_tokens);
    free(output_tokens);

    return err;
}

marmot_error_t marmot_bench_llm_run(
    const marmot_bench_model_t *model, const marmot_bench_llm_params_t *params, size_t repetitions,
    marmot_bench_llm_result_t *result
) {
    if (params == nullptr) {
        return MARMOT_ERROR_INVALID_ARGUMENT;
    }

    switch (params->mode) {
    case MARMOT_BENCH_LLM_MODE_DIRECT:
        return marmot_bench_llm_run_direct(model, params, repetitions, result);
    case MARMOT_BENCH_LLM_MODE_SERVING:
        return marmot_bench_llm_run_serving(model, params, repetitions, result);
    }

    marmot_set_error(MARMOT_ERROR_INVALID_ARGUMENT, "unknown LLM benchmark mode");
    return MARMOT_ERROR_INVALID_ARGUMENT;
}

marmot_error_t marmot_bench_llm_run_pp(
    const marmot_bench_model_t *model, const marmot_bench_llm_params_t *params, size_t repetitions,
    marmot_bench_llm_result_t *result
) {
    marmot_bench_llm_params_t pp_only = *params;
    pp_only.n_gen = 0;
    return marmot_bench_llm_run(model, &pp_only, repetitions, result);
}

marmot_error_t marmot_bench_llm_run_tg(
    const marmot_bench_model_t *model, const marmot_bench_llm_params_t *params, size_t repetitions,
    marmot_bench_llm_result_t *result
) {
    marmot_bench_llm_params_t tg_only = *params;
    tg_only.n_prompt = 0;
    return marmot_bench_llm_run(model, &tg_only, repetitions, result);
}

static char llm_result_buffer[256];

const char *marmot_bench_llm_result_str(const marmot_bench_llm_result_t *result) {
    if (result == nullptr) {
        return "null";
    }

    if (result->n_prompt > 0 && result->n_gen > 0) {
        snprintf(
            llm_result_buffer, sizeof(llm_result_buffer), "%s seqs%zu pp%zu @ %.1f t/s, tg%zu @ %.1f t/s",
            result->mode == MARMOT_BENCH_LLM_MODE_DIRECT ? "direct" : "serving", result->n_seqs, result->n_prompt,
            result->pp_tokens_per_sec, result->n_gen, result->tg_tokens_per_sec
        );
    } else if (result->n_prompt > 0) {
        snprintf(
            llm_result_buffer, sizeof(llm_result_buffer), "%s seqs%zu pp%zu @ %.1f t/s",
            result->mode == MARMOT_BENCH_LLM_MODE_DIRECT ? "direct" : "serving", result->n_seqs, result->n_prompt,
            result->pp_tokens_per_sec
        );
    } else if (result->n_gen > 0) {
        snprintf(
            llm_result_buffer, sizeof(llm_result_buffer), "%s seqs%zu tg%zu @ %.1f t/s",
            result->mode == MARMOT_BENCH_LLM_MODE_DIRECT ? "direct" : "serving", result->n_seqs, result->n_gen,
            result->tg_tokens_per_sec
        );
    } else {
        return "no benchmark";
    }

    return llm_result_buffer;
}
