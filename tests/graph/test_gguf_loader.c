/* clang-format off */
#include "marmot/graph/architecture.h"
#include "marmot/graph/gguf_loader.h"
#include "marmot/graph/gguf_model.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#include "yyjson.h"
#include "test_fixture_utils.h"
/* clang-format on */

// GGUF loader test uses tinyllama fixture (single fixture as it tests specific metadata)
#define GGUF_LOADER_TEST_FIXTURE "tinyllama-q4_k_m.gguf"

// Thread-local fixture path buffer
static _Thread_local char g_fixture_path[MARMOT_TEST_PATH_MAX];

static const char *get_fixture_path(void) {
    return marmot_test_get_fixture_path(GGUF_LOADER_TEST_FIXTURE, g_fixture_path, sizeof(g_fixture_path));
}

static marmot_packed_graph_options_t
make_packed_opts(marmot_backend_type_t backend, const marmot_gguf_model_meta_t *meta) {
    marmot_packed_graph_options_t opts;
    marmot_packed_graph_options_init(&opts);
    opts.token_count = 8;
    opts.sample_count = 1;
    opts.max_seqs = 1;
    opts.max_seq_len = 128;
    opts.block_size = 16;
    const size_t max_blocks_per_seq = (opts.max_seq_len + opts.block_size - 1) / opts.block_size;
    opts.num_kv_blocks = opts.max_seqs * max_blocks_per_seq;
    if (backend == MARMOT_BACKEND_CPU) {
        opts.kv_dtype = MARMOT_DTYPE_FLOAT32;
    } else if (meta != nullptr) {
        opts.kv_dtype = marmot_activation_dtype_for_architecture(meta->architecture, backend);
    } else {
        opts.kv_dtype = MARMOT_DTYPE_FLOAT16;
    }
    return opts;
}

static int setup_gguf(void **state) {
    const char *path = get_fixture_path();
    if (!marmot_test_fixture_exists(path)) {
        *state = nullptr;
        return 0;
    }
    marmot_gguf_t *gguf = marmot_gguf_load(path);
    if (gguf == nullptr) {
        return -1;
    }
    *state = gguf;
    return 0;
}

static int teardown_gguf(void **state) {
    if (*state != nullptr) {
        marmot_gguf_unload((marmot_gguf_t *)(*state));
    }
    return 0;
}

static void test_header_fields(void **state) {
    if (*state == nullptr) {
        skip();
    }
    marmot_gguf_t *gguf = (marmot_gguf_t *)(*state);
    assert_int_equal(gguf->version, 3);
    assert_int_equal((int)gguf->alignment, 32);
    assert_int_equal((int)gguf->kv_count, 23);
    assert_int_equal((int)gguf->tensor_count, 201);

    const marmot_gguf_kv_t *arch = marmot_gguf_find_kv(gguf, "general.architecture");
    assert_non_null(arch);
    assert_int_equal(arch->value.type, MARMOT_GGUF_TYPE_STRING);
    assert_string_equal(arch->value.data.string_value.data, "llama");
}

static void test_float_tensor_view(void **state) {
    if (*state == nullptr) {
        skip();
    }
    marmot_gguf_t *gguf = (marmot_gguf_t *)(*state);
    const marmot_gguf_tensor_t *tensor_info = marmot_gguf_find_tensor(gguf, "blk.0.attn_norm.weight");
    assert_non_null(tensor_info);
    assert_non_null(tensor_info->tensor);
    const marmot_tensor_t *tensor = tensor_info->tensor;
    assert_int_equal(tensor->shape.ndim, 1);
    assert_int_equal(tensor->shape.shape[0], 2048);
    assert_int_equal(tensor->dtype, MARMOT_DTYPE_FLOAT32);
    assert_int_equal(tensor->quant_kind, MARMOT_QUANT_KIND_GENERIC);
    assert_true(tensor_info->byte_length == marmot_tensor_size_bytes(tensor));
    assert_non_null(tensor->data);
}

static void test_quantized_tensor_views(void **state) {
    if (*state == nullptr) {
        skip();
    }
    marmot_gguf_t *gguf = (marmot_gguf_t *)(*state);

    const marmot_gguf_tensor_t *q4 = marmot_gguf_find_tensor(gguf, "blk.0.ffn_gate.weight");
    assert_non_null(q4);
    assert_non_null(q4->tensor);
    assert_int_equal(q4->tensor->quant_kind, MARMOT_QUANT_KIND_Q4_K);
    assert_int_equal(q4->tensor->quant_layout, MARMOT_QUANT_LAYOUT_GGUF);
    assert_int_equal(q4->qscheme_id, MARMOT_QSCHEME_Q4_K);
    const size_t q4_expected = marmot_tensor_size_bytes(q4->tensor);
    assert_true(q4_expected > 0);
    assert_true(q4->byte_length >= q4_expected);
    assert_int_equal(q4->tensor->capacity_bytes, q4_expected);

    const marmot_gguf_tensor_t *q6 = marmot_gguf_find_tensor(gguf, "output.weight");
    assert_non_null(q6);
    assert_non_null(q6->tensor);
    assert_int_equal(q6->tensor->quant_kind, MARMOT_QUANT_KIND_Q6_K);
    assert_int_equal(q6->tensor->quant_layout, MARMOT_QUANT_LAYOUT_GGUF);
    assert_int_equal(q6->qscheme_id, MARMOT_QSCHEME_Q6_K);
    const size_t q6_expected = marmot_tensor_size_bytes(q6->tensor);
    assert_true(q6_expected > 0);
    assert_true(q6->byte_length >= q6_expected);
    assert_int_equal(q6->tensor->capacity_bytes, q6_expected);
}

static void test_model_wrapper(void **state) {
    (void)state;

    const char *path = get_fixture_path();
    if (!marmot_test_fixture_exists(path)) {
        skip();
    }

    marmot_gguf_model_t *model = nullptr;
    assert_int_equal(marmot_gguf_model_load(path, MARMOT_BACKEND_CPU, &model), MARMOT_SUCCESS);
    assert_non_null(model);

    assert_int_equal((int)marmot_gguf_model_tensor_count(model), 201);
    const marmot_tensor_t *embed = marmot_gguf_model_tensor(model, "token_embd.weight");
    assert_non_null(embed);
    assert_int_equal(embed->shape.ndim, 2);

    const marmot_gguf_t *file = marmot_gguf_model_file(model);
    assert_non_null(file);
    assert_int_equal((int)file->tensor_count, (int)marmot_gguf_model_tensor_count(model));

    marmot_gguf_model_destroy(model);
}

static void test_model_metadata(void **state) {
    (void)state;

    const char *path = get_fixture_path();
    if (!marmot_test_fixture_exists(path)) {
        skip();
    }

    marmot_gguf_model_t *model = nullptr;
    assert_int_equal(marmot_gguf_model_load(path, MARMOT_BACKEND_CPU, &model), MARMOT_SUCCESS);

    marmot_gguf_model_meta_t meta;
    assert_true(marmot_gguf_model_metadata(model, &meta));
    assert_int_equal((int)meta.context_length, 2048);
    assert_int_equal((int)meta.n_embd, 2048);
    assert_int_equal((int)meta.n_layer, 22);
    assert_int_equal((int)meta.n_head, 32);
    assert_int_equal((int)meta.n_head_kv, 4);
    assert_int_equal((int)meta.n_vocab, 32000);
    assert_int_equal((int)meta.ff_length, 5632);
    assert_int_equal((int)meta.rope_dimension, 64);
    assert_true(meta.rms_norm_eps > 0.0f);
    assert_int_equal((int)meta.rope_freq_base, 10000);

    marmot_gguf_model_destroy(model);
}

static void test_graph_from_gguf(void **state) {
    (void)state;

    const char *path = get_fixture_path();
    if (!marmot_test_fixture_exists(path)) {
        skip();
    }

    marmot_packed_graph_options_t opts = make_packed_opts(MARMOT_BACKEND_CPU, nullptr);
    marmot_graph_t *graph = nullptr;
    assert_int_equal(marmot_graph_from_gguf_packed(path, MARMOT_BACKEND_CPU, &opts, &graph), MARMOT_SUCCESS);
    assert_non_null(graph);
    assert_int_equal(marmot_graph_get_backend(graph), MARMOT_BACKEND_CPU);
    marmot_graph_destroy(graph);
}

static void test_graph_from_gguf_quant_roundtrip(void **state) {
    (void)state;

    const char *path = get_fixture_path();
    if (!marmot_test_fixture_exists(path)) {
        skip();
    }

    marmot_gguf_model_t *model = nullptr;
    assert_int_equal(marmot_gguf_model_load(path, MARMOT_BACKEND_CPU, &model), MARMOT_SUCCESS);
    marmot_gguf_model_meta_t meta;
    assert_true(marmot_gguf_model_metadata(model, &meta));

    marmot_packed_graph_options_t opts = make_packed_opts(MARMOT_BACKEND_CPU, &meta);
    marmot_graph_t *graph = nullptr;
    assert_int_equal(marmot_graph_from_gguf_packed(path, MARMOT_BACKEND_CPU, &opts, &graph), MARMOT_SUCCESS);
    assert_non_null(graph);

    char tmp_path[] = "/tmp/marmot_graph_builderXXXXXX";
    int fd = mkstemp(tmp_path);
    assert_true(fd >= 0);
    close(fd);

    assert_int_equal(marmot_graph_dump_json(graph, tmp_path), MARMOT_SUCCESS);

    FILE *f = fopen(tmp_path, "rb");
    assert_non_null(f);
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    assert_true(len > 0);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len);
    assert_non_null(buf);
    assert_int_equal((size_t)len, fread(buf, 1, (size_t)len, f));
    fclose(f);

    yyjson_doc *doc = yyjson_read(buf, (size_t)len, 0);
    assert_non_null(doc);
    yyjson_val *root = yyjson_doc_get_root(doc);
    assert_non_null(root);

    yyjson_val *nodes = yyjson_obj_get(root, "nodes");
    assert_non_null(nodes);
    assert_true(yyjson_is_arr(nodes));

    assert_true(yyjson_arr_size(nodes) > meta.n_layer);

    size_t quantized = 0;
    size_t paged_attention_nodes = 0;
    size_t scaled_attention_nodes = 0;
    const size_t node_count = yyjson_arr_size(nodes);
    for (size_t i = 0; i < node_count; ++i) {
        yyjson_val *node = yyjson_arr_get(nodes, i);
        yyjson_val *op = yyjson_obj_get(node, "op");
        const char *op_name = op != nullptr ? yyjson_get_str(op) : nullptr;
        if (op_name != nullptr) {
            if (strcmp(op_name, "paged_attention") == 0) {
                ++paged_attention_nodes;
            } else if (strcmp(op_name, "scaled_dot_product_attention") == 0) {
                ++scaled_attention_nodes;
            }
        }
        yyjson_val *sig = yyjson_obj_get(node, "signature");
        assert_non_null(sig);
        yyjson_val *qscheme = yyjson_obj_get(sig, "qscheme");
        assert_non_null(qscheme);
        const char *qscheme_str = yyjson_get_str(qscheme);
        if (qscheme_str != nullptr && strcmp(qscheme_str, "none") != 0) {
            ++quantized;
        }
    }
    assert_true(quantized > 0);
    assert_int_equal(paged_attention_nodes, meta.n_layer);
    assert_int_equal(scaled_attention_nodes, 0);

    yyjson_doc_free(doc);
    free(buf);
    unlink(tmp_path);
    marmot_graph_destroy(graph);
    marmot_gguf_model_destroy(model);
}

static void test_gguf_loader_caps_include_auto_backend(void **state) {
    (void)state;

    marmot_gguf_options_t opts;
    assert_int_equal(marmot_gguf_options_init(&opts), MARMOT_SUCCESS);
    marmot_packed_graph_options_t packed_opts = make_packed_opts(MARMOT_BACKEND_CPU, nullptr);
    opts.packed_opts = &packed_opts;

    marmot_gguf_loader_t *loader = nullptr;
    assert_int_equal(marmot_gguf_loader_create(&opts, &loader), MARMOT_SUCCESS);
    assert_non_null(loader);

    marmot_gguf_caps_t caps;
    assert_int_equal(marmot_gguf_loader_query_capabilities(loader, &caps), MARMOT_SUCCESS);
    assert_true((caps.supported_flags & MARMOT_GGUF_FLAG_AUTO_BACKEND) != 0);

    marmot_gguf_loader_destroy(loader);
}

static void test_gguf_loader_auto_backend_prefers_gpu(void **state) {
    (void)state;

    const char *path = get_fixture_path();
    if (!marmot_test_fixture_exists(path)) {
        skip();
    }

    marmot_gguf_model_t *model = nullptr;
    marmot_gguf_model_meta_t meta;
    bool meta_ok = false;
    if (marmot_gguf_model_load(path, MARMOT_BACKEND_CPU, &model) == MARMOT_SUCCESS) {
        meta_ok = marmot_gguf_model_metadata(model, &meta);
    }

    marmot_packed_graph_options_t opts_auto = make_packed_opts(MARMOT_BACKEND_CPU, meta_ok ? &meta : nullptr);
    // Check if CPU backend works
    marmot_graph_t *cpu_graph = nullptr;
    bool cpu_ok = (marmot_graph_from_gguf_packed(path, MARMOT_BACKEND_CPU, &opts_auto, &cpu_graph) == MARMOT_SUCCESS);
    if (cpu_graph) {
        marmot_graph_destroy(cpu_graph);
    }

    // Check if Metal backend works
    bool metal_ok = false;
#if MARMOT_ENABLE_METAL
    marmot_graph_t *metal_graph = nullptr;
    if (marmot_graph_from_gguf_packed(path, MARMOT_BACKEND_METAL, &opts_auto, &metal_graph) == MARMOT_SUCCESS) {
        metal_ok = true;
        marmot_graph_destroy(metal_graph);
    }
#endif
    if (model) {
        marmot_gguf_model_destroy(model);
    }

    marmot_gguf_options_t opts;
    assert_int_equal(marmot_gguf_options_init(&opts), MARMOT_SUCCESS);
    opts.flags |= MARMOT_GGUF_FLAG_AUTO_BACKEND;
    opts.packed_opts = &opts_auto;

    marmot_gguf_loader_t *loader = nullptr;
    assert_int_equal(marmot_gguf_loader_create(&opts, &loader), MARMOT_SUCCESS);
    assert_non_null(loader);

    marmot_graph_t *graph = nullptr;
    assert_int_equal(marmot_gguf_loader_load_file(loader, path, &graph), MARMOT_SUCCESS);
    assert_non_null(graph);

    // Auto selection should prefer Metal over CPU when both work
    const marmot_backend_type_t chosen_backend = marmot_graph_get_backend(graph);
    if (metal_ok) {
        assert_int_equal(chosen_backend, MARMOT_BACKEND_METAL);
    } else if (cpu_ok) {
        assert_int_equal(chosen_backend, MARMOT_BACKEND_CPU);
    }

    marmot_graph_destroy(graph);
    marmot_gguf_loader_destroy(loader);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_header_fields, setup_gguf, teardown_gguf),
        cmocka_unit_test_setup_teardown(test_float_tensor_view, setup_gguf, teardown_gguf),
        cmocka_unit_test_setup_teardown(test_quantized_tensor_views, setup_gguf, teardown_gguf),
        cmocka_unit_test(test_model_wrapper),
        cmocka_unit_test(test_model_metadata),
        cmocka_unit_test(test_graph_from_gguf),
        cmocka_unit_test(test_graph_from_gguf_quant_roundtrip),
        cmocka_unit_test(test_gguf_loader_caps_include_auto_backend),
        cmocka_unit_test(test_gguf_loader_auto_backend_prefers_gpu),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
