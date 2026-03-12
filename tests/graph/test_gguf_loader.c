/* clang-format off */
#include "marmot/graph/architecture.h"
#include "marmot/graph/gguf_loader.h"
#include "marmot/graph/gguf_model.h"

#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
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
    assert_int_equal(marmot_packed_graph_options_init(&opts), MARMOT_SUCCESS);
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

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
} synthetic_gguf_buffer_t;

typedef struct {
    const char *name;
    uint32_t ndim;
    uint64_t dims[3];
} synthetic_gguf_tensor_t;

static void synthetic_gguf_reserve(synthetic_gguf_buffer_t *buf, size_t extra) {
    assert_non_null(buf);
    assert_true(buf->len <= SIZE_MAX - extra);
    const size_t needed = buf->len + extra;
    if (needed <= buf->cap) {
        return;
    }

    size_t new_cap = buf->cap == 0 ? 1024 : buf->cap;
    while (new_cap < needed) {
        assert_true(new_cap <= SIZE_MAX / 2);
        new_cap *= 2;
    }

    uint8_t *new_data = realloc(buf->data, new_cap);
    assert_non_null(new_data);
    buf->data = new_data;
    buf->cap = new_cap;
}

static void synthetic_gguf_append_bytes(synthetic_gguf_buffer_t *buf, const void *src, size_t size) {
    synthetic_gguf_reserve(buf, size);
    memcpy(buf->data + buf->len, src, size);
    buf->len += size;
}

static void synthetic_gguf_append_zeroes(synthetic_gguf_buffer_t *buf, size_t size) {
    synthetic_gguf_reserve(buf, size);
    memset(buf->data + buf->len, 0, size);
    buf->len += size;
}

static void synthetic_gguf_append_u32(synthetic_gguf_buffer_t *buf, uint32_t value) {
    synthetic_gguf_append_bytes(buf, &value, sizeof(value));
}

static void synthetic_gguf_append_u64(synthetic_gguf_buffer_t *buf, uint64_t value) {
    synthetic_gguf_append_bytes(buf, &value, sizeof(value));
}

static void synthetic_gguf_append_f32(synthetic_gguf_buffer_t *buf, float value) {
    synthetic_gguf_append_bytes(buf, &value, sizeof(value));
}

static void synthetic_gguf_append_string_data(synthetic_gguf_buffer_t *buf, const char *value) {
    assert_non_null(value);
    const uint64_t len = (uint64_t)strlen(value);
    synthetic_gguf_append_u64(buf, len);
    synthetic_gguf_append_bytes(buf, value, (size_t)len);
}

static void synthetic_gguf_append_kv_string(synthetic_gguf_buffer_t *buf, const char *key, const char *value) {
    synthetic_gguf_append_string_data(buf, key);
    synthetic_gguf_append_u32(buf, MARMOT_GGUF_TYPE_STRING);
    synthetic_gguf_append_string_data(buf, value);
}

static void synthetic_gguf_append_kv_u32(synthetic_gguf_buffer_t *buf, const char *key, uint32_t value) {
    synthetic_gguf_append_string_data(buf, key);
    synthetic_gguf_append_u32(buf, MARMOT_GGUF_TYPE_UINT32);
    synthetic_gguf_append_u32(buf, value);
}

static void synthetic_gguf_append_kv_f32(synthetic_gguf_buffer_t *buf, const char *key, float value) {
    synthetic_gguf_append_string_data(buf, key);
    synthetic_gguf_append_u32(buf, MARMOT_GGUF_TYPE_FLOAT32);
    synthetic_gguf_append_f32(buf, value);
}

static size_t synthetic_gguf_tensor_data_size(const synthetic_gguf_tensor_t *tensor) {
    assert_non_null(tensor);
    size_t elements = 1;
    for (uint32_t i = 0; i < tensor->ndim; ++i) {
        assert_true(tensor->dims[i] <= SIZE_MAX / elements);
        elements *= (size_t)tensor->dims[i];
    }
    assert_true(elements <= SIZE_MAX / sizeof(float));
    return elements * sizeof(float);
}

static void synthetic_gguf_append_tensor_info(
    synthetic_gguf_buffer_t *buf, const synthetic_gguf_tensor_t *tensor, uint32_t ggml_type, uint64_t data_offset
) {
    assert_non_null(buf);
    assert_non_null(tensor);
    synthetic_gguf_append_string_data(buf, tensor->name);
    synthetic_gguf_append_u32(buf, tensor->ndim);
    for (uint32_t i = 0; i < tensor->ndim; ++i) {
        synthetic_gguf_append_u64(buf, tensor->dims[i]);
    }
    synthetic_gguf_append_u32(buf, ggml_type);
    synthetic_gguf_append_u64(buf, data_offset);
}

static void synthetic_gguf_pad_to_alignment(synthetic_gguf_buffer_t *buf, size_t alignment) {
    assert_non_null(buf);
    assert_true(alignment != 0);
    const size_t aligned = (buf->len + alignment - 1) / alignment * alignment;
    synthetic_gguf_append_zeroes(buf, aligned - buf->len);
}

static int write_synthetic_qwen3moe_gguf(char *out_path, size_t out_path_size) {
    static const synthetic_gguf_tensor_t tensors[] = {
        {.name = "output_norm.weight", .ndim = 1, .dims = {8}},
        {.name = "output.weight", .ndim = 2, .dims = {8, 16}},
        {.name = "blk.0.attn_norm.weight", .ndim = 1, .dims = {8}},
        {.name = "blk.0.attn_q.weight", .ndim = 2, .dims = {8, 8}},
        {.name = "blk.0.attn_k.weight", .ndim = 2, .dims = {8, 4}},
        {.name = "blk.0.attn_v.weight", .ndim = 2, .dims = {8, 4}},
        {.name = "blk.0.attn_output.weight", .ndim = 2, .dims = {8, 8}},
        {.name = "blk.0.attn_q_norm.weight", .ndim = 1, .dims = {4}},
        {.name = "blk.0.attn_k_norm.weight", .ndim = 1, .dims = {4}},
        {.name = "blk.0.ffn_norm.weight", .ndim = 1, .dims = {8}},
        {.name = "blk.0.ffn_gate_inp.weight", .ndim = 2, .dims = {8, 2}},
        {.name = "blk.0.ffn_gate_exps.weight", .ndim = 3, .dims = {8, 6, 2}},
        {.name = "blk.0.ffn_up_exps.weight", .ndim = 3, .dims = {8, 6, 2}},
        {.name = "blk.0.ffn_down_exps.weight", .ndim = 3, .dims = {6, 8, 2}},
        {.name = "blk.0.ffn_gate_shexp.weight", .ndim = 2, .dims = {8, 6}},
        {.name = "blk.0.ffn_up_shexp.weight", .ndim = 2, .dims = {8, 6}},
        {.name = "blk.0.ffn_down_shexp.weight", .ndim = 2, .dims = {6, 8}},
    };

    synthetic_gguf_buffer_t buf = {0};
    uint64_t tensor_offsets[sizeof(tensors) / sizeof(tensors[0])] = {0};
    uint64_t next_offset = 0;
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        tensor_offsets[i] = next_offset;
        next_offset += (uint64_t)synthetic_gguf_tensor_data_size(&tensors[i]);
    }

    synthetic_gguf_append_u32(&buf, 0x46554747u);
    synthetic_gguf_append_u32(&buf, 3u);
    synthetic_gguf_append_u64(&buf, (uint64_t)(sizeof(tensors) / sizeof(tensors[0])));
    synthetic_gguf_append_u64(&buf, 16u);

    synthetic_gguf_append_kv_string(&buf, "general.architecture", "qwen3moe");
    synthetic_gguf_append_kv_u32(&buf, "general.alignment", 32u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.context_length", 32u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.embedding_length", 8u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.block_count", 1u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.attention.head_count", 2u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.attention.head_count_kv", 1u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.feed_forward_length", 6u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.rope.dimension_count", 4u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.attention.key_length", 4u);
    synthetic_gguf_append_kv_f32(&buf, "qwen3moe.attention.layer_norm_rms_epsilon", 1.0e-5f);
    synthetic_gguf_append_kv_f32(&buf, "qwen3moe.rope.freq_base", 10000.0f);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.expert_count", 2u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.expert_used_count", 1u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.expert_shared_count", 1u);
    synthetic_gguf_append_kv_f32(&buf, "qwen3moe.expert_weights_scale", 1.25f);

    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        synthetic_gguf_append_tensor_info(&buf, &tensors[i], 0u, tensor_offsets[i]);
    }

    synthetic_gguf_pad_to_alignment(&buf, 32u);
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        synthetic_gguf_append_zeroes(&buf, synthetic_gguf_tensor_data_size(&tensors[i]));
    }

    char tmp_path[] = "/tmp/marmot_qwen3moe_XXXXXX";
    int fd = mkstemp(tmp_path);
    if (fd < 0) {
        free(buf.data);
        return -1;
    }
    ssize_t written = write(fd, buf.data, buf.len);
    free(buf.data);
    close(fd);
    if (written < 0 || (size_t)written != buf.len) {
        unlink(tmp_path);
        return -1;
    }

    int copied = snprintf(out_path, out_path_size, "%s", tmp_path);
    if (copied <= 0 || (size_t)copied >= out_path_size) {
        unlink(tmp_path);
        return -1;
    }

    return 0;
}

static yyjson_val *find_value_by_name(yyjson_val *values, const char *name) {
    if (values == nullptr || name == nullptr) {
        return nullptr;
    }
    const size_t count = yyjson_arr_size(values);
    for (size_t i = 0; i < count; ++i) {
        yyjson_val *value = yyjson_arr_get(values, i);
        yyjson_val *value_name = yyjson_obj_get(value, "name");
        const char *actual = value_name != nullptr ? yyjson_get_str(value_name) : nullptr;
        if (actual != nullptr && strcmp(actual, name) == 0) {
            return value;
        }
    }
    return nullptr;
}

static size_t count_nodes_by_op(yyjson_val *nodes, const char *op_name) {
    if (nodes == nullptr || op_name == nullptr) {
        return 0;
    }
    size_t count = 0;
    const size_t node_count = yyjson_arr_size(nodes);
    for (size_t i = 0; i < node_count; ++i) {
        yyjson_val *node = yyjson_arr_get(nodes, i);
        yyjson_val *op = yyjson_obj_get(node, "op");
        const char *actual = op != nullptr ? yyjson_get_str(op) : nullptr;
        if (actual != nullptr && strcmp(actual, op_name) == 0) {
            ++count;
        }
    }
    return count;
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

// ---------------------------------------------------------------------------
// Configurable synthetic MoE GGUF writer for metadata variant tests
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t expert_count;
    uint32_t expert_used_count;
    bool include_expert_shared_count;
    uint32_t expert_shared_count;
    bool include_expert_weights_scale;
    float expert_weights_scale;
    bool include_shared_expert_tensors;
} synthetic_moe_config_t;

static int
write_synthetic_moe_gguf_configurable(const synthetic_moe_config_t *cfg, char *out_path, size_t out_path_size) {
    // Base tensors (always present)
    static const synthetic_gguf_tensor_t base_tensors[] = {
        {.name = "output_norm.weight", .ndim = 1, .dims = {8}},
        {.name = "output.weight", .ndim = 2, .dims = {8, 16}},
        {.name = "blk.0.attn_norm.weight", .ndim = 1, .dims = {8}},
        {.name = "blk.0.attn_q.weight", .ndim = 2, .dims = {8, 8}},
        {.name = "blk.0.attn_k.weight", .ndim = 2, .dims = {8, 4}},
        {.name = "blk.0.attn_v.weight", .ndim = 2, .dims = {8, 4}},
        {.name = "blk.0.attn_output.weight", .ndim = 2, .dims = {8, 8}},
        {.name = "blk.0.attn_q_norm.weight", .ndim = 1, .dims = {4}},
        {.name = "blk.0.attn_k_norm.weight", .ndim = 1, .dims = {4}},
        {.name = "blk.0.ffn_norm.weight", .ndim = 1, .dims = {8}},
        {.name = "blk.0.ffn_gate_inp.weight", .ndim = 2, .dims = {8, 2}},
        {.name = "blk.0.ffn_gate_exps.weight", .ndim = 3, .dims = {8, 6, 2}},
        {.name = "blk.0.ffn_up_exps.weight", .ndim = 3, .dims = {8, 6, 2}},
        {.name = "blk.0.ffn_down_exps.weight", .ndim = 3, .dims = {6, 8, 2}},
    };
    static const synthetic_gguf_tensor_t shared_tensors[] = {
        {.name = "blk.0.ffn_gate_shexp.weight", .ndim = 2, .dims = {8, 6}},
        {.name = "blk.0.ffn_up_shexp.weight", .ndim = 2, .dims = {8, 6}},
        {.name = "blk.0.ffn_down_shexp.weight", .ndim = 2, .dims = {6, 8}},
    };

    const size_t n_base = sizeof(base_tensors) / sizeof(base_tensors[0]);
    const size_t n_shared = cfg->include_shared_expert_tensors ? sizeof(shared_tensors) / sizeof(shared_tensors[0]) : 0;
    const size_t n_tensors = n_base + n_shared;

    // Count KV pairs
    uint64_t n_kv = 14; // base metadata count
    if (cfg->include_expert_shared_count) {
        n_kv++;
    }
    if (cfg->include_expert_weights_scale) {
        n_kv++;
    }

    synthetic_gguf_buffer_t buf = {0};

    // Compute tensor offsets
    uint64_t next_offset = 0;
    uint64_t base_offsets[14];
    for (size_t i = 0; i < n_base; ++i) {
        base_offsets[i] = next_offset;
        next_offset += (uint64_t)synthetic_gguf_tensor_data_size(&base_tensors[i]);
    }
    uint64_t shared_offsets[3] = {0};
    for (size_t i = 0; i < n_shared; ++i) {
        shared_offsets[i] = next_offset;
        next_offset += (uint64_t)synthetic_gguf_tensor_data_size(&shared_tensors[i]);
    }

    // Header
    synthetic_gguf_append_u32(&buf, 0x46554747u);
    synthetic_gguf_append_u32(&buf, 3u);
    synthetic_gguf_append_u64(&buf, (uint64_t)n_tensors);
    synthetic_gguf_append_u64(&buf, n_kv);

    // Metadata
    synthetic_gguf_append_kv_string(&buf, "general.architecture", "qwen3moe");
    synthetic_gguf_append_kv_u32(&buf, "general.alignment", 32u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.context_length", 32u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.embedding_length", 8u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.block_count", 1u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.attention.head_count", 2u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.attention.head_count_kv", 1u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.feed_forward_length", 6u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.rope.dimension_count", 4u);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.attention.key_length", 4u);
    synthetic_gguf_append_kv_f32(&buf, "qwen3moe.attention.layer_norm_rms_epsilon", 1.0e-5f);
    synthetic_gguf_append_kv_f32(&buf, "qwen3moe.rope.freq_base", 10000.0f);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.expert_count", cfg->expert_count);
    synthetic_gguf_append_kv_u32(&buf, "qwen3moe.expert_used_count", cfg->expert_used_count);
    if (cfg->include_expert_shared_count) {
        synthetic_gguf_append_kv_u32(&buf, "qwen3moe.expert_shared_count", cfg->expert_shared_count);
    }
    if (cfg->include_expert_weights_scale) {
        synthetic_gguf_append_kv_f32(&buf, "qwen3moe.expert_weights_scale", cfg->expert_weights_scale);
    }

    // Tensor info
    for (size_t i = 0; i < n_base; ++i) {
        synthetic_gguf_append_tensor_info(&buf, &base_tensors[i], 0u, base_offsets[i]);
    }
    for (size_t i = 0; i < n_shared; ++i) {
        synthetic_gguf_append_tensor_info(&buf, &shared_tensors[i], 0u, shared_offsets[i]);
    }

    // Padding + tensor data
    synthetic_gguf_pad_to_alignment(&buf, 32u);
    for (size_t i = 0; i < n_base; ++i) {
        synthetic_gguf_append_zeroes(&buf, synthetic_gguf_tensor_data_size(&base_tensors[i]));
    }
    for (size_t i = 0; i < n_shared; ++i) {
        synthetic_gguf_append_zeroes(&buf, synthetic_gguf_tensor_data_size(&shared_tensors[i]));
    }

    // Write to temp file
    char tmp_path[] = "/tmp/marmot_moe_variant_XXXXXX";
    int fd = mkstemp(tmp_path);
    if (fd < 0) {
        free(buf.data);
        return -1;
    }
    ssize_t written = write(fd, buf.data, buf.len);
    free(buf.data);
    close(fd);
    if (written < 0 || (size_t)written != buf.len) {
        unlink(tmp_path);
        return -1;
    }

    int copied = snprintf(out_path, out_path_size, "%s", tmp_path);
    if (copied <= 0 || (size_t)copied >= out_path_size) {
        unlink(tmp_path);
        return -1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Test: Minimal metadata — matches real Qwen3-30B-A3B (no optional keys)
// ---------------------------------------------------------------------------
static void test_moe_minimal_metadata(void **state) {
    (void)state;

    synthetic_moe_config_t cfg = {
        .expert_count = 2,
        .expert_used_count = 1,
        .include_expert_shared_count = false,
        .include_expert_weights_scale = false,
        .include_shared_expert_tensors = false,
    };

    char gguf_path[MARMOT_TEST_PATH_MAX];
    assert_int_equal(write_synthetic_moe_gguf_configurable(&cfg, gguf_path, sizeof(gguf_path)), 0);

    marmot_gguf_model_t *model = nullptr;
    assert_int_equal(marmot_gguf_model_load(gguf_path, MARMOT_BACKEND_CPU, &model), MARMOT_SUCCESS);
    assert_non_null(model);

    marmot_gguf_model_meta_t meta;
    assert_true(marmot_gguf_model_metadata(model, &meta));
    assert_int_equal(meta.architecture, MARMOT_ARCH_QWEN3MOE);
    assert_true(meta.is_moe);
    assert_int_equal((int)meta.n_experts, 2);
    assert_int_equal((int)meta.n_experts_used, 1);
    assert_int_equal((int)meta.n_shared_experts, 0);
    assert_true(fabsf(meta.expert_weights_scale - 1.0f) <= 1.0e-6f);

    marmot_gguf_model_destroy(model);
    unlink(gguf_path);
}

// ---------------------------------------------------------------------------
// Test: Minimal metadata — graph builds successfully without optional keys
// ---------------------------------------------------------------------------
static void test_moe_minimal_metadata_graph_builds(void **state) {
    (void)state;

    synthetic_moe_config_t cfg = {
        .expert_count = 2,
        .expert_used_count = 1,
        .include_expert_shared_count = false,
        .include_expert_weights_scale = false,
        .include_shared_expert_tensors = false,
    };

    char gguf_path[MARMOT_TEST_PATH_MAX];
    assert_int_equal(write_synthetic_moe_gguf_configurable(&cfg, gguf_path, sizeof(gguf_path)), 0);

    marmot_packed_graph_options_t opts;
    assert_int_equal(marmot_packed_graph_options_init(&opts), MARMOT_SUCCESS);
    opts.token_count = 2;
    opts.sample_count = 0;
    opts.max_seqs = 1;
    opts.max_seq_len = 8;
    opts.block_size = 8;
    opts.num_kv_blocks = 1;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_graph_t *graph = nullptr;
    assert_int_equal(marmot_graph_from_gguf_packed(gguf_path, MARMOT_BACKEND_CPU, &opts, &graph), MARMOT_SUCCESS);
    assert_non_null(graph);

    // Dump and verify graph structure — should have moe_experts but no shared expert
    char json_path[] = "/tmp/marmot_moe_minimal_graphXXXXXX";
    int fd = mkstemp(json_path);
    assert_true(fd >= 0);
    close(fd);

    assert_int_equal(marmot_graph_dump_json(graph, json_path), MARMOT_SUCCESS);

    FILE *f = fopen(json_path, "rb");
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
    yyjson_val *nodes = yyjson_obj_get(root, "nodes");
    yyjson_val *values = yyjson_obj_get(root, "values");
    assert_non_null(nodes);
    assert_non_null(values);

    assert_int_equal(count_nodes_by_op(nodes, "moe_experts"), 1);
    assert_int_equal(count_nodes_by_op(nodes, "topk"), 1);
    assert_null(find_value_by_name(values, "layer.0.shared_expert_out"));

    yyjson_doc_free(doc);
    free(buf);
    unlink(json_path);
    marmot_graph_destroy(graph);
    unlink(gguf_path);
}

// ---------------------------------------------------------------------------
// Test: expert_used_count == expert_count (degenerate MoE — all experts active)
// ---------------------------------------------------------------------------
static void test_moe_all_experts_used(void **state) {
    (void)state;

    synthetic_moe_config_t cfg = {
        .expert_count = 2,
        .expert_used_count = 2,
        .include_expert_shared_count = false,
        .include_expert_weights_scale = false,
        .include_shared_expert_tensors = false,
    };

    char gguf_path[MARMOT_TEST_PATH_MAX];
    assert_int_equal(write_synthetic_moe_gguf_configurable(&cfg, gguf_path, sizeof(gguf_path)), 0);

    marmot_gguf_model_t *model = nullptr;
    assert_int_equal(marmot_gguf_model_load(gguf_path, MARMOT_BACKEND_CPU, &model), MARMOT_SUCCESS);
    assert_non_null(model);

    marmot_gguf_model_meta_t meta;
    assert_true(marmot_gguf_model_metadata(model, &meta));
    assert_true(meta.is_moe);
    assert_int_equal((int)meta.n_experts, 2);
    assert_int_equal((int)meta.n_experts_used, 2);

    marmot_packed_graph_options_t opts;
    assert_int_equal(marmot_packed_graph_options_init(&opts), MARMOT_SUCCESS);
    opts.token_count = 2;
    opts.sample_count = 0;
    opts.max_seqs = 1;
    opts.max_seq_len = 8;
    opts.block_size = 8;
    opts.num_kv_blocks = 1;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_graph_t *graph = nullptr;
    assert_int_equal(marmot_graph_from_gguf_packed(gguf_path, MARMOT_BACKEND_CPU, &opts, &graph), MARMOT_SUCCESS);
    assert_non_null(graph);

    marmot_graph_destroy(graph);
    marmot_gguf_model_destroy(model);
    unlink(gguf_path);
}

static void test_synthetic_qwen3moe_graph_from_gguf(void **state) {
    (void)state;

    char gguf_path[MARMOT_TEST_PATH_MAX];
    assert_int_equal(write_synthetic_qwen3moe_gguf(gguf_path, sizeof(gguf_path)), 0);

    marmot_gguf_model_t *model = nullptr;
    assert_int_equal(marmot_gguf_model_load(gguf_path, MARMOT_BACKEND_CPU, &model), MARMOT_SUCCESS);
    assert_non_null(model);

    marmot_gguf_model_meta_t meta;
    assert_true(marmot_gguf_model_metadata(model, &meta));
    assert_int_equal(meta.architecture, MARMOT_ARCH_QWEN3MOE);
    assert_true(meta.is_moe);
    assert_int_equal((int)meta.n_layer, 1);
    assert_int_equal((int)meta.n_embd, 8);
    assert_int_equal((int)meta.n_head, 2);
    assert_int_equal((int)meta.n_head_kv, 1);
    assert_int_equal((int)meta.head_dim, 4);
    assert_int_equal((int)meta.ff_length, 6);
    assert_int_equal((int)meta.n_experts, 2);
    assert_int_equal((int)meta.n_experts_used, 1);
    assert_int_equal((int)meta.n_shared_experts, 1);
    assert_int_equal((int)meta.n_vocab, 16);
    assert_int_equal(meta.router_weight_policy, MARMOT_ROUTER_WEIGHT_POLICY_SOFTMAX_SELECTED_SCALED);
    assert_true(fabsf(meta.expert_weights_scale - 1.25f) <= 1.0e-6f);

    marmot_packed_graph_options_t opts;
    assert_int_equal(marmot_packed_graph_options_init(&opts), MARMOT_SUCCESS);
    opts.token_count = 2;
    opts.sample_count = 0;
    opts.max_seqs = 1;
    opts.max_seq_len = 8;
    opts.block_size = 8;
    opts.num_kv_blocks = 1;
    opts.kv_dtype = MARMOT_DTYPE_FLOAT32;

    marmot_graph_t *graph = nullptr;
    assert_int_equal(marmot_graph_from_gguf_packed(gguf_path, MARMOT_BACKEND_CPU, &opts, &graph), MARMOT_SUCCESS);
    assert_non_null(graph);

    char json_path[] = "/tmp/marmot_qwen3moe_graphXXXXXX";
    int fd = mkstemp(json_path);
    assert_true(fd >= 0);
    close(fd);

    assert_int_equal(marmot_graph_dump_json(graph, json_path), MARMOT_SUCCESS);

    FILE *f = fopen(json_path, "rb");
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
    yyjson_val *values = yyjson_obj_get(root, "values");
    assert_non_null(nodes);
    assert_non_null(values);
    assert_int_equal(count_nodes_by_op(nodes, "moe_experts"), 1);
    assert_int_equal(count_nodes_by_op(nodes, "topk"), 1);
    assert_non_null(find_value_by_name(values, "layer.0.router_logits"));
    assert_non_null(find_value_by_name(values, "layer.0.router_topk_values"));
    assert_non_null(find_value_by_name(values, "layer.0.router_weights"));
    assert_null(find_value_by_name(values, "layer.0.router_probs"));
    assert_non_null(find_value_by_name(values, "layer.0.routed_moe_out"));
    assert_non_null(find_value_by_name(values, "layer.0.shared_expert_out"));
    assert_non_null(find_value_by_name(values, "layer.0.moe_out"));

    yyjson_doc_free(doc);
    free(buf);
    unlink(json_path);
    marmot_graph_destroy(graph);
    marmot_gguf_model_destroy(model);
    unlink(gguf_path);
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
        cmocka_unit_test(test_moe_minimal_metadata),
        cmocka_unit_test(test_moe_minimal_metadata_graph_builds),
        cmocka_unit_test(test_moe_all_experts_used),
        cmocka_unit_test(test_synthetic_qwen3moe_graph_from_gguf),
    };
    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
