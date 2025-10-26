#include "bench_core.h"
#include "bench_grid.h"
#include "bench_llm.h"
#include "bench_model.h"
#include "bench_output.h"
#include "bench_param_sweep.h"
#include "bench_workloads.h"

#include <getopt.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    // Kernel benchmark config
    marmot_bench_config_t bench_config;

    // LLM benchmark config
    const char *model_path;
    marmot_bench_param_range_t ctx_sizes;
    marmot_bench_param_range_t n_prompts;
    marmot_bench_param_range_t n_gens;
    marmot_bench_param_range_t n_depths;
    marmot_bench_param_range_t batch_sizes;
    marmot_bench_param_range_t n_threads;
    size_t max_seqs;
    size_t max_batch_seqs;
    size_t concurrency;
    marmot_dtype_t kv_type_k;
    marmot_dtype_t kv_type_v;
    bool flash_attn;
    size_t repetitions;

    // Flags
    bool progress;
    bool no_warmup;
    bool list_devices;
} cli_config_t;

static void cli_config_init(cli_config_t *cfg) {
    marmot_bench_config_defaults(&cfg->bench_config);

    cfg->model_path = nullptr;
    marmot_bench_range_init(&cfg->ctx_sizes);
    marmot_bench_range_init(&cfg->n_prompts);
    marmot_bench_range_init(&cfg->n_gens);
    marmot_bench_range_init(&cfg->n_depths);
    marmot_bench_range_init(&cfg->batch_sizes);
    marmot_bench_range_init(&cfg->n_threads);
    cfg->max_seqs = 0;
    cfg->max_batch_seqs = 0;
    cfg->concurrency = 1;
    cfg->kv_type_k = MARMOT_DTYPE_FLOAT16;
    cfg->kv_type_v = MARMOT_DTYPE_FLOAT16;
    cfg->flash_attn = true;
    cfg->repetitions = 5;
    cfg->progress = false;
    cfg->no_warmup = false;
    cfg->list_devices = false;
}

static void cli_config_free(cli_config_t *cfg) {
    marmot_bench_range_free(&cfg->ctx_sizes);
    marmot_bench_range_free(&cfg->n_prompts);
    marmot_bench_range_free(&cfg->n_gens);
    marmot_bench_range_free(&cfg->n_depths);
    marmot_bench_range_free(&cfg->batch_sizes);
    marmot_bench_range_free(&cfg->n_threads);
}

static int set_env_override(const char *name, const char *value) {
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    if (setenv(name, value, 1) != 0) {
        fprintf(stderr, "Failed to set %s=%s: %s\n", name, value, strerror(errno));
        return 1;
    }
    return 0;
}

static int parse_size_t_arg(const char *arg, size_t *out) {
    if (arg == nullptr || out == nullptr) {
        return 1;
    }
    errno = 0;
    char *end = nullptr;
    unsigned long long value = strtoull(arg, &end, 10);
    if (errno != 0 || end == arg || (end != nullptr && *end != '\0')) {
        return 1;
    }
    *out = (size_t)value;
    return 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "\nMarmot Benchmark Suite\n\n");
    fprintf(stderr, "Kernel Benchmark Options:\n");
    fprintf(stderr, "  -b, --backend <cpu|metal|compare>  Backend mode (default: compare)\n");
    fprintf(stderr, "  -c, --category <micro|composite|all>  Category filter (default: all)\n");
    fprintf(stderr, "  -f, --filter <pattern>      Workload name filter\n");
    fprintf(stderr, "\nLLM Benchmark Options (requires -m):\n");
    fprintf(stderr, "  -m, --model <path>          GGUF model file (enables LLM mode)\n");
    fprintf(stderr, "  -p, --n-prompt <n,...>      Prompt sizes to test (default: 512)\n");
    fprintf(stderr, "  -g, --n-gen <n,...>         Generation lengths (default: 128)\n");
    fprintf(stderr, "  -d, --n-depth <n,...>       Pre-filled KV cache depth (default: 0)\n");
    fprintf(stderr, "  -C, --ctx-size <n,...>      Context sizes (default: 4096)\n");
    fprintf(stderr, "  -B, --batch-size <n,...>    Batch sizes (default: 512)\n");
    fprintf(stderr, "  -t, --threads <n,...>       Thread counts (default: 4)\n");
    fprintf(stderr, "  --concurrency <n>           Concurrent requests per run (default: 1)\n");
    fprintf(stderr, "  --max-seqs <n>              Serving engine max_seqs (default: concurrency)\n");
    fprintf(stderr, "  --max-batch-seqs <n>        Serving engine max_batch_seqs (default: concurrency)\n");
    fprintf(stderr, "  -ctk <f16|f32|q8_0>         K cache type (default: f16)\n");
    fprintf(stderr, "  -ctv <f16|f32|q8_0>         V cache type (default: f16)\n");
    fprintf(stderr, "  -fa, --flash-attn <0|1>     Flash attention (default: 1)\n");
    fprintf(stderr, "\nExecution Options:\n");
    fprintf(stderr, "  -r, --repetitions <n>       Repetitions per config (default: 5)\n");
    fprintf(stderr, "  -w, --warmup <n>            Warmup iterations (default: 5)\n");
    fprintf(stderr, "  -n, --iterations <n>        Measure iterations (default: 100)\n");
    fprintf(stderr, "  -T, --min-time <sec>        Min time per workload (default: 1.0)\n");
    fprintf(stderr, "  --no-warmup                 Skip warmup iterations\n");
    fprintf(stderr, "  --progress                  Show progress during run\n");
    fprintf(stderr, "\nMetal Calibration Overrides:\n");
    fprintf(stderr, "  --metal-peak-fp32 <tflops>  Override Metal FP32 TFLOPS\n");
    fprintf(stderr, "  --metal-peak-fp16 <tflops>  Override Metal FP16 TFLOPS\n");
    fprintf(stderr, "  --metal-mem-bw <gbps>       Override Metal memory bandwidth\n");
    fprintf(stderr, "  --metal-launch-us <usec>    Override Metal launch overhead\n");
    fprintf(stderr, "  --metal-edge-alpha <alpha>  Override Metal edge penalty alpha\n");
    fprintf(stderr, "\nOutput Options:\n");
    fprintf(stderr, "  -o, --output <path>         Output file path (default: stdout)\n");
    fprintf(stderr, "  -F, --format <fmt>          Output format (default: console)\n");
    fprintf(stderr, "                              Formats: console, json, jsonl, md, csv, sql\n");
    fprintf(stderr, "\nInformation:\n");
    fprintf(stderr, "  --list-devices              List available devices and exit\n");
    fprintf(stderr, "  -v, --verbose               Verbose output\n");
    fprintf(stderr, "  -h, --help                  Show this help\n");
    fprintf(stderr, "\nParameter Ranges:\n");
    fprintf(stderr, "  Multiple values can be specified with commas: -p 256,512,1024\n");
    fprintf(stderr, "  Ranges: -p 256-1024 (doubling), -p 100-1000+100 (additive), -p 1-8*2 (multiplicative)\n");
    fprintf(stderr, "\nExamples:\n");
    fprintf(stderr, "  %s                                    Run kernel benchmarks, CPU vs Metal\n", prog);
    fprintf(stderr, "  %s -b metal -c micro                  Run micro benchmarks on Metal only\n", prog);
    fprintf(stderr, "  %s -m model.gguf -p 512 -g 128        Run LLM benchmark\n", prog);
    fprintf(stderr, "  %s -m model.gguf -p 256,512,1024      Sweep prompt sizes\n", prog);
    fprintf(stderr, "  %s -F csv -o results.csv              Export results to CSV\n", prog);
}

static void list_devices(void) {
    fprintf(stdout, "Available devices:\n");

    // CPU is always available
    marmot_device_caps_t cpu_caps;
    marmot_bench_get_device_caps(MARMOT_BACKEND_CPU, &cpu_caps);
    fprintf(stdout, "  cpu: CPU backend (%.1f TFLOPS fp16, %.1f GB/s)\n", cpu_caps.peak_flops_tflops_fp16, cpu_caps.mem_bw_gbps);

#ifdef __APPLE__
    // Try to initialize Metal to check availability
    marmot_context_t *metal_ctx = marmot_init(MARMOT_BACKEND_METAL);
    if (metal_ctx != nullptr) {
        marmot_device_caps_t caps;
        marmot_bench_get_device_caps(MARMOT_BACKEND_METAL, &caps);
        fprintf(stdout, "  metal: Metal backend (%.1f TFLOPS fp16, %.1f GB/s)\n", caps.peak_flops_tflops_fp16, caps.mem_bw_gbps);
        marmot_destroy(metal_ctx);
    } else {
        fprintf(stdout, "  metal: Metal backend (not available)\n");
    }
#endif
}

static int run_kernel_benchmarks(cli_config_t *cfg) {
    marmot_bench_suite_t *suite = marmot_bench_create_full_suite();
    if (suite == nullptr) {
        fprintf(stderr, "Failed to create benchmark suite\n");
        return 1;
    }

    if (cfg->bench_config.verbose) {
        fprintf(stderr, "Running %zu workloads...\n", suite->num_workloads);
    }

    if (cfg->no_warmup) {
        cfg->bench_config.warmup_iterations = 0;
    }

    marmot_bench_result_t *results = nullptr;
    size_t num_results = 0;

    marmot_error_t err = marmot_bench_run(&cfg->bench_config, suite, &results, &num_results);
    if (err != MARMOT_SUCCESS) {
        fprintf(stderr, "Benchmark failed: %s\n", marmot_error_string(err));
        marmot_bench_suite_destroy(suite);
        return 1;
    }

    FILE *out = stdout;
    if (cfg->bench_config.output_path != nullptr) {
        out = fopen(cfg->bench_config.output_path, "w");
        if (out == nullptr) {
            fprintf(stderr, "Failed to open output file: %s\n", cfg->bench_config.output_path);
            marmot_bench_results_free(results, num_results);
            marmot_bench_suite_destroy(suite);
            return 1;
        }
    }

    switch (cfg->bench_config.output_format) {
    case MARMOT_BENCH_OUTPUT_CONSOLE:
        marmot_bench_output_console(out, &cfg->bench_config, results, num_results);
        break;
    case MARMOT_BENCH_OUTPUT_JSON:
        marmot_bench_output_json(out, &cfg->bench_config, results, num_results);
        break;
    case MARMOT_BENCH_OUTPUT_MARKDOWN:
        marmot_bench_output_markdown(out, &cfg->bench_config, results, num_results);
        break;
    case MARMOT_BENCH_OUTPUT_CSV:
        marmot_bench_output_csv(out, &cfg->bench_config, results, num_results);
        break;
    case MARMOT_BENCH_OUTPUT_SQL:
        marmot_bench_output_sql(out, &cfg->bench_config, results, num_results);
        break;
    case MARMOT_BENCH_OUTPUT_JSONL:
        marmot_bench_output_jsonl(out, &cfg->bench_config, results, num_results);
        break;
    }

    if (cfg->bench_config.output_path != nullptr) {
        fclose(out);
    }

    marmot_bench_results_free(results, num_results);
    marmot_bench_suite_destroy(suite);
    return 0;
}

static int run_llm_benchmarks(cli_config_t *cfg) {
    // Set defaults if not specified
    if (cfg->ctx_sizes.count == 0) {
        (void)marmot_bench_parse_range_or_default(nullptr, 4096, &cfg->ctx_sizes);
    }
    if (cfg->n_prompts.count == 0) {
        (void)marmot_bench_parse_range_or_default(nullptr, 512, &cfg->n_prompts);
    }
    if (cfg->n_gens.count == 0) {
        (void)marmot_bench_parse_range_or_default(nullptr, 128, &cfg->n_gens);
    }
    if (cfg->n_depths.count == 0) {
        (void)marmot_bench_parse_range_or_default(nullptr, 0, &cfg->n_depths);
    }
    if (cfg->batch_sizes.count == 0) {
        (void)marmot_bench_parse_range_or_default(nullptr, 512, &cfg->batch_sizes);
    }
    if (cfg->n_threads.count == 0) {
        (void)marmot_bench_parse_range_or_default(nullptr, 4, &cfg->n_threads);
    }

    // Build grid of configurations
    marmot_bench_grid_t grid;
    marmot_bench_grid_init(&grid);

    marmot_error_t err = marmot_bench_grid_build(
        &grid, &cfg->ctx_sizes, &cfg->n_prompts, &cfg->n_gens, &cfg->n_depths, &cfg->batch_sizes, &cfg->n_threads
    );
    if (err != MARMOT_SUCCESS) {
        fprintf(stderr, "Failed to build parameter grid: %s\n", marmot_error_string(err));
        return 1;
    }

    if (cfg->bench_config.verbose || cfg->progress) {
        fprintf(stderr, "Running %zu LLM configurations with %zu repetitions each...\n", grid.count, cfg->repetitions);
    }

    FILE *out = stdout;
    if (cfg->bench_config.output_path != nullptr) {
        out = fopen(cfg->bench_config.output_path, "w");
        if (out == nullptr) {
            fprintf(stderr, "Failed to open output file: %s\n", cfg->bench_config.output_path);
            marmot_bench_grid_free(&grid);
            return 1;
        }
    }

    // Print header for console/markdown output
    if (cfg->bench_config.output_format == MARMOT_BENCH_OUTPUT_CONSOLE) {
        fprintf(out, "%-40s %8s %8s %8s %12s %12s\n", "model", "ctx", "pp", "tg", "pp t/s", "tg t/s");
        fprintf(out, "%-40s %8s %8s %8s %12s %12s\n", "----------------------------------------", "--------", "--------",
                "--------", "------------", "------------");
    } else if (cfg->bench_config.output_format == MARMOT_BENCH_OUTPUT_MARKDOWN) {
        fprintf(out, "| model | ctx | pp | tg | pp t/s | tg t/s |\n");
        fprintf(out, "|-------|-----|----|----|--------|--------|\n");
    } else if (cfg->bench_config.output_format == MARMOT_BENCH_OUTPUT_CSV) {
        fprintf(out, "model,ctx_size,batch_size,threads,n_prompt,n_gen,n_depth,pp_tokens_sec,tg_tokens_sec,pp_mean_us,tg_mean_us\n"
        );
    }

    int ret = 0;

    for (size_t i = 0; i < grid.count; i++) {
        marmot_bench_grid_config_t *gc = &grid.configs[i];

        if (cfg->progress) {
            fprintf(stderr, "marmot-bench: config %zu/%zu: ctx=%zu, pp=%zu, tg=%zu\n", i + 1, grid.count, gc->ctx_size,
                    gc->n_prompt, gc->n_gen);
        }

        // Load model with this configuration
        marmot_bench_model_config_t model_cfg;
        marmot_bench_model_config_init(&model_cfg);
        model_cfg.model_path = cfg->model_path;
        model_cfg.ctx_size = gc->ctx_size;
        model_cfg.batch_size = gc->batch_size;
        model_cfg.n_threads = gc->n_threads;
        model_cfg.max_seqs = cfg->max_seqs != 0 ? cfg->max_seqs : cfg->concurrency;
        model_cfg.max_batch_seqs = cfg->max_batch_seqs != 0 ? cfg->max_batch_seqs : cfg->concurrency;
        model_cfg.kv_type_k = cfg->kv_type_k;
        model_cfg.kv_type_v = cfg->kv_type_v;
        model_cfg.flash_attn = cfg->flash_attn;

        if (cfg->concurrency == 0) {
            fprintf(stderr, "Invalid concurrency: must be > 0\n");
            ret = 1;
            continue;
        }
        if (cfg->concurrency > model_cfg.max_seqs) {
            fprintf(stderr, "Invalid config: concurrency (%zu) exceeds max_seqs (%zu)\n", cfg->concurrency,
                    model_cfg.max_seqs);
            ret = 1;
            continue;
        }

        // Determine backend
        if (cfg->bench_config.backend_mode == MARMOT_BENCH_BACKEND_METAL) {
            model_cfg.backend = MARMOT_BACKEND_METAL;
        } else {
            model_cfg.backend = MARMOT_BACKEND_CPU;
        }

        marmot_bench_model_t model;
        err = marmot_bench_model_load(&model_cfg, &model);
        if (err != MARMOT_SUCCESS) {
            fprintf(stderr, "Failed to load model: %s\n", marmot_error_string(err));
            ret = 1;
            continue;
        }

        // Run benchmark
        marmot_bench_llm_params_t llm_params;
        marmot_bench_llm_params_init(&llm_params);
        llm_params.n_prompt = gc->n_prompt;
        llm_params.n_gen = gc->n_gen;
        llm_params.n_depth = gc->n_depth;
        llm_params.n_seqs = cfg->concurrency;

        marmot_bench_llm_result_t result;
        err = marmot_bench_llm_run(&model, &llm_params, cfg->repetitions, &result);

        if (err == MARMOT_SUCCESS) {
            // Output result based on format
            const char *model_name = marmot_bench_model_info(&model);

            switch (cfg->bench_config.output_format) {
            case MARMOT_BENCH_OUTPUT_CONSOLE:
                fprintf(out, "%-40s %8zu %8zu %8zu %12.1f %12.1f\n", model_name, gc->ctx_size, gc->n_prompt, gc->n_gen,
                        result.pp_tokens_per_sec, result.tg_tokens_per_sec);
                break;
            case MARMOT_BENCH_OUTPUT_MARKDOWN:
                fprintf(out, "| %s | %zu | %zu | %zu | %.1f | %.1f |\n", model_name, gc->ctx_size, gc->n_prompt,
                        gc->n_gen, result.pp_tokens_per_sec, result.tg_tokens_per_sec);
                break;
            case MARMOT_BENCH_OUTPUT_CSV:
                fprintf(out, "%s,%zu,%zu,%zu,%zu,%zu,%zu,%.2f,%.2f,%.3f,%.3f\n", cfg->model_path, gc->ctx_size,
                        gc->batch_size, gc->n_threads, gc->n_prompt, gc->n_gen, gc->n_depth, result.pp_tokens_per_sec,
                        result.tg_tokens_per_sec, result.pp_stats.mean_us, result.tg_stats.mean_us);
                break;
            case MARMOT_BENCH_OUTPUT_JSON:
            case MARMOT_BENCH_OUTPUT_JSONL:
                fprintf(out,
                        "{\"model\":\"%s\",\"ctx_size\":%zu,\"batch_size\":%zu,\"n_prompt\":%zu,"
                        "\"n_gen\":%zu,\"n_depth\":%zu,\"pp_tokens_sec\":%.2f,\"tg_tokens_sec\":%.2f,"
                        "\"pp_mean_us\":%.3f,\"tg_mean_us\":%.3f}\n",
                        cfg->model_path, gc->ctx_size, gc->batch_size, gc->n_prompt, gc->n_gen, gc->n_depth,
                        result.pp_tokens_per_sec, result.tg_tokens_per_sec, result.pp_stats.mean_us,
                        result.tg_stats.mean_us);
                break;
            case MARMOT_BENCH_OUTPUT_SQL:
                fprintf(out,
                        "INSERT INTO llm_bench (model, ctx_size, batch_size, n_prompt, n_gen, n_depth, "
                        "pp_tokens_sec, tg_tokens_sec) VALUES ('%s', %zu, %zu, %zu, %zu, %zu, %.2f, %.2f);\n",
                        cfg->model_path, gc->ctx_size, gc->batch_size, gc->n_prompt, gc->n_gen, gc->n_depth,
                        result.pp_tokens_per_sec, result.tg_tokens_per_sec);
                break;
            }
            fflush(out);
        } else {
            fprintf(stderr, "Benchmark failed for config %zu: %s\n", i, marmot_error_string(err));
            ret = 1;
        }

        marmot_bench_model_free(&model);
    }

    if (cfg->bench_config.output_path != nullptr) {
        fclose(out);
    }

    marmot_bench_grid_free(&grid);
    return ret;
}

int main(int argc, char *argv[]) {
    cli_config_t cfg;
    cli_config_init(&cfg);

    static struct option long_options[] = {
        {"backend", required_argument, 0, 'b'},
        {"category", required_argument, 0, 'c'},
        {"filter", required_argument, 0, 'f'},
        {"model", required_argument, 0, 'm'},
        {"n-prompt", required_argument, 0, 'p'},
        {"n-gen", required_argument, 0, 'g'},
        {"n-depth", required_argument, 0, 'd'},
        {"ctx-size", required_argument, 0, 'C'},
        {"batch-size", required_argument, 0, 'B'},
        {"threads", required_argument, 0, 't'},
        {"flash-attn", required_argument, 0, 'a'},
        {"concurrency", required_argument, 0, 1200},
        {"max-seqs", required_argument, 0, 1201},
        {"max-batch-seqs", required_argument, 0, 1202},
        {"repetitions", required_argument, 0, 'r'},
        {"output", required_argument, 0, 'o'},
        {"format", required_argument, 0, 'F'},
        {"warmup", required_argument, 0, 'w'},
        {"iterations", required_argument, 0, 'n'},
        {"min-time", required_argument, 0, 'T'},
        {"no-warmup", no_argument, 0, 1002},
        {"progress", no_argument, 0, 1003},
        {"list-devices", no_argument, 0, 1004},
        {"metal-peak-fp32", required_argument, 0, 1100},
        {"metal-peak-fp16", required_argument, 0, 1101},
        {"metal-mem-bw", required_argument, 0, 1102},
        {"metal-launch-us", required_argument, 0, 1103},
        {"metal-edge-alpha", required_argument, 0, 1104},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "b:c:f:m:p:g:d:C:B:t:a:r:o:F:w:n:T:vh", long_options, nullptr)) != -1) {
        switch (opt) {
        case 'b':
            if (strcmp(optarg, "cpu") == 0) {
                cfg.bench_config.backend_mode = MARMOT_BENCH_BACKEND_CPU;
            } else if (strcmp(optarg, "metal") == 0) {
                cfg.bench_config.backend_mode = MARMOT_BENCH_BACKEND_METAL;
            } else if (strcmp(optarg, "compare") == 0) {
                cfg.bench_config.backend_mode = MARMOT_BENCH_BACKEND_COMPARE;
            } else {
                fprintf(stderr, "Unknown backend: %s\n", optarg);
                return 1;
            }
            break;
        case 'c':
            if (strcmp(optarg, "micro") == 0) {
                cfg.bench_config.category_mask = MARMOT_BENCH_CATEGORY_MICRO;
            } else if (strcmp(optarg, "composite") == 0) {
                cfg.bench_config.category_mask = MARMOT_BENCH_CATEGORY_COMPOSITE;
            } else if (strcmp(optarg, "all") == 0) {
                cfg.bench_config.category_mask = MARMOT_BENCH_CATEGORY_ALL;
            } else {
                fprintf(stderr, "Unknown category: %s\n", optarg);
                return 1;
            }
            break;
        case 'f':
            cfg.bench_config.workload_filter = optarg;
            break;
        case 'm':
            cfg.model_path = optarg;
            break;
        case 'p':
            if (marmot_bench_parse_range(optarg, &cfg.n_prompts) != MARMOT_SUCCESS) {
                fprintf(stderr, "Invalid prompt range: %s\n", optarg);
                return 1;
            }
            break;
        case 'g':
            if (marmot_bench_parse_range(optarg, &cfg.n_gens) != MARMOT_SUCCESS) {
                fprintf(stderr, "Invalid generation range: %s\n", optarg);
                return 1;
            }
            break;
        case 'd':
            if (marmot_bench_parse_range(optarg, &cfg.n_depths) != MARMOT_SUCCESS) {
                fprintf(stderr, "Invalid depth range: %s\n", optarg);
                return 1;
            }
            break;
        case 'C':
            if (marmot_bench_parse_range(optarg, &cfg.ctx_sizes) != MARMOT_SUCCESS) {
                fprintf(stderr, "Invalid context size range: %s\n", optarg);
                return 1;
            }
            break;
        case 'B':
            if (marmot_bench_parse_range(optarg, &cfg.batch_sizes) != MARMOT_SUCCESS) {
                fprintf(stderr, "Invalid batch size range: %s\n", optarg);
                return 1;
            }
            break;
        case 't':
            if (marmot_bench_parse_range(optarg, &cfg.n_threads) != MARMOT_SUCCESS) {
                fprintf(stderr, "Invalid thread count range: %s\n", optarg);
                return 1;
            }
            break;
        case 'a':
            cfg.flash_attn = (strcmp(optarg, "1") == 0 || strcmp(optarg, "true") == 0 || strcmp(optarg, "on") == 0);
            break;
        case 1200: // --concurrency
            if (parse_size_t_arg(optarg, &cfg.concurrency) != 0) {
                fprintf(stderr, "Invalid concurrency: %s\n", optarg);
                return 1;
            }
            break;
        case 1201: // --max-seqs
            if (parse_size_t_arg(optarg, &cfg.max_seqs) != 0) {
                fprintf(stderr, "Invalid max_seqs: %s\n", optarg);
                return 1;
            }
            break;
        case 1202: // --max-batch-seqs
            if (parse_size_t_arg(optarg, &cfg.max_batch_seqs) != 0) {
                fprintf(stderr, "Invalid max_batch_seqs: %s\n", optarg);
                return 1;
            }
            break;
        case 'r':
            cfg.repetitions = (size_t)atoi(optarg);
            break;
        case 'F':
            if (strcmp(optarg, "json") == 0) {
                cfg.bench_config.output_format = MARMOT_BENCH_OUTPUT_JSON;
            } else if (strcmp(optarg, "jsonl") == 0) {
                cfg.bench_config.output_format = MARMOT_BENCH_OUTPUT_JSONL;
            } else if (strcmp(optarg, "md") == 0 || strcmp(optarg, "markdown") == 0) {
                cfg.bench_config.output_format = MARMOT_BENCH_OUTPUT_MARKDOWN;
            } else if (strcmp(optarg, "csv") == 0) {
                cfg.bench_config.output_format = MARMOT_BENCH_OUTPUT_CSV;
            } else if (strcmp(optarg, "sql") == 0) {
                cfg.bench_config.output_format = MARMOT_BENCH_OUTPUT_SQL;
            } else if (strcmp(optarg, "console") == 0) {
                cfg.bench_config.output_format = MARMOT_BENCH_OUTPUT_CONSOLE;
            } else {
                fprintf(stderr, "Unknown format: %s\n", optarg);
                return 1;
            }
            break;
        case 'w':
            cfg.bench_config.warmup_iterations = (uint32_t)atoi(optarg);
            break;
        case 'n':
            cfg.bench_config.measure_iterations = (uint32_t)atoi(optarg);
            break;
        case 'T':
            cfg.bench_config.min_time_seconds = atof(optarg);
            break;
        case 'o':
            cfg.bench_config.output_path = optarg;
            break;
        case 1002: // --no-warmup
            cfg.no_warmup = true;
            break;
        case 1003: // --progress
            cfg.progress = true;
            break;
        case 1004: // --list-devices
            cfg.list_devices = true;
            break;
        case 1100: // --metal-peak-fp32
            if (set_env_override("MARMOT_METAL_PEAK_TFLOPS_FP32", optarg) != 0) {
                return 1;
            }
            break;
        case 1101: // --metal-peak-fp16
            if (set_env_override("MARMOT_METAL_PEAK_TFLOPS_FP16", optarg) != 0) {
                return 1;
            }
            break;
        case 1102: // --metal-mem-bw
            if (set_env_override("MARMOT_METAL_MEM_BW_GBPS", optarg) != 0) {
                return 1;
            }
            break;
        case 1103: // --metal-launch-us
            if (set_env_override("MARMOT_METAL_LAUNCH_US", optarg) != 0) {
                return 1;
            }
            break;
        case 1104: // --metal-edge-alpha
            if (set_env_override("MARMOT_METAL_EDGE_ALPHA", optarg) != 0) {
                return 1;
            }
            break;
        case 'v':
            cfg.bench_config.verbose = true;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    // Handle -ctk and -ctv (need to be parsed specially since they have non-standard names)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-ctk") == 0 && i + 1 < argc) {
            if (marmot_bench_parse_kv_type(argv[i + 1], &cfg.kv_type_k) != MARMOT_SUCCESS) {
                fprintf(stderr, "Invalid KV cache K type: %s\n", argv[i + 1]);
                return 1;
            }
        } else if (strcmp(argv[i], "-ctv") == 0 && i + 1 < argc) {
            if (marmot_bench_parse_kv_type(argv[i + 1], &cfg.kv_type_v) != MARMOT_SUCCESS) {
                fprintf(stderr, "Invalid KV cache V type: %s\n", argv[i + 1]);
                return 1;
            }
        }
    }

    if (cfg.list_devices) {
        list_devices();
        cli_config_free(&cfg);
        return 0;
    }

    int ret;
    if (cfg.model_path != nullptr) {
        // LLM benchmark mode
        ret = run_llm_benchmarks(&cfg);
    } else {
        // Kernel benchmark mode
        ret = run_kernel_benchmarks(&cfg);
    }

    cli_config_free(&cfg);
    return ret;
}
