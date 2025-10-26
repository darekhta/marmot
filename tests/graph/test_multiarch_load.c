#include "marmot/graph/architecture.h"
#include "marmot/graph/gguf_loader.h"
#include "marmot/graph/gguf_model.h"
#include "marmot/graph/graph.h"
#include "marmot/types.h"

#include <stdio.h>

#include <assert.h>
#include <string.h>

static void test_load_model(const char *path, marmot_architecture_t expected_arch) {
    printf("  Loading: %s\n", path);

    // Load model metadata
    marmot_gguf_model_t *model = nullptr;
    marmot_error_t err = marmot_gguf_model_load(path, MARMOT_BACKEND_CPU, &model);
    if (err != MARMOT_SUCCESS) {
        printf("    FAILED: marmot_gguf_model_load returned %d\n", err);
        printf("    Error: %s\n", marmot_get_last_error_detail());
        return;
    }
    assert(model != nullptr);

    marmot_gguf_model_meta_t meta = {};
    bool ok = marmot_gguf_model_metadata(model, &meta);
    if (!ok) {
        printf("    FAILED: marmot_gguf_model_metadata returned false\n");
        marmot_gguf_model_destroy(model);
        return;
    }

    printf(
        "    Architecture: %s (expected: %s)\n", marmot_architecture_to_string(meta.architecture),
        marmot_architecture_to_string(expected_arch)
    );

    if (expected_arch != MARMOT_ARCH_UNKNOWN && meta.architecture != expected_arch) {
        printf("    FAILED: Architecture mismatch!\n");
        marmot_gguf_model_destroy(model);
        return;
    }

    const marmot_architecture_traits_t *traits = marmot_get_architecture_traits(meta.architecture);
    if (traits == nullptr) {
        printf("    FAILED: Could not get architecture traits\n");
        marmot_gguf_model_destroy(model);
        return;
    }

    const char *ffn_name = "GELU";
    if (traits->ffn_type == MARMOT_FFN_SWIGLU) {
        ffn_name = "SwiGLU";
    } else if (traits->ffn_type == MARMOT_FFN_GEGLU) {
        ffn_name = "GeGLU";
    }
    printf("    FFN type: %s\n", ffn_name);
    printf("    Has attention bias: %s\n", traits->has_attention_bias ? "yes" : "no");
    printf("    Has Q/K norm: %s\n", traits->has_qk_norm ? "yes" : "no");
    printf("    Context length: %zu\n", meta.context_length);
    printf("    Embedding dim: %zu\n", meta.n_embd);
    printf("    Layers: %zu\n", meta.n_layer);
    printf("    Heads: %zu (KV: %zu)\n", meta.n_head, meta.n_head_kv);
    printf("    Vocab size: %zu\n", meta.n_vocab);

    marmot_gguf_model_destroy(model);

    // Now try to build a graph using the loader API
    printf("    Building graph...\n");

    marmot_gguf_options_t opts = {};
    err = marmot_gguf_options_init(&opts);
    if (err != MARMOT_SUCCESS) {
        printf("    FAILED: marmot_gguf_options_init returned %d\n", err);
        return;
    }
    opts.backend = MARMOT_BACKEND_CPU;
    marmot_packed_graph_options_t packed_opts = {};
    marmot_packed_graph_options_init(&packed_opts);
    packed_opts.token_count = 8;
    packed_opts.sample_count = 1;
    packed_opts.max_seqs = 1;
    packed_opts.max_seq_len = 128;
    packed_opts.block_size = 16;
    const size_t max_blocks_per_seq = (packed_opts.max_seq_len + packed_opts.block_size - 1) / packed_opts.block_size;
    packed_opts.num_kv_blocks = packed_opts.max_seqs * max_blocks_per_seq;
    packed_opts.kv_dtype = MARMOT_DTYPE_FLOAT32;
    opts.packed_opts = &packed_opts;

    marmot_gguf_loader_t *loader = nullptr;
    err = marmot_gguf_loader_create(&opts, &loader);
    if (err != MARMOT_SUCCESS) {
        printf("    FAILED: marmot_gguf_loader_create returned %d\n", err);
        return;
    }

    marmot_graph_t *graph = nullptr;
    err = marmot_gguf_loader_load_file(loader, path, &graph);
    if (err != MARMOT_SUCCESS) {
        const marmot_error_info_t *info = marmot_gguf_loader_last_error(loader);
        printf("    FAILED: marmot_gguf_loader_load_file returned %d\n", err);
        if (info != nullptr && info->message[0] != '\0') {
            printf("    Loader error: %s\n", info->message);
        }
        const char *detail = marmot_get_last_error_detail();
        if (detail && detail[0] != '\0') {
            printf("    Last error: %s\n", detail);
        }
        marmot_gguf_loader_destroy(loader);
        return;
    }

    printf("    Graph built successfully!\n");

    marmot_graph_destroy(graph);
    marmot_gguf_loader_destroy(loader);
    printf("    OK\n");
}

int main(int argc, char **argv) {
    printf("Multi-architecture GGUF load tests:\n\n");

    if (argc < 2) {
        printf("Usage: %s <model.gguf> [expected_arch]\n", argv[0]);
        printf("\nSupported architectures: llama, qwen2, qwen3, phi3, gemma\n");
        return 1;
    }

    const char *path = argv[1];
    marmot_architecture_t expected = MARMOT_ARCH_UNKNOWN;

    if (argc >= 3) {
        expected = marmot_architecture_from_string(argv[2]);
        if (expected == MARMOT_ARCH_UNKNOWN && strcmp(argv[2], "unknown") != 0) {
            printf("Unknown architecture: %s\n", argv[2]);
            return 1;
        }
    }

    test_load_model(path, expected);

    printf("\nDone!\n");
    return 0;
}
