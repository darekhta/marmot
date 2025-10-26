/* clang-format off */
#include "marmot/marmot.h"
#include "tests/test_dtype_helpers.h"
#include "core/dispatch/fusion_flags.h"

#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <cmocka.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
/* clang-format on */

static uint32_t log2_u32(uint32_t value) {
    uint32_t shift = 0;
    while (value > 1u) {
        value >>= 1u;
        shift++;
    }
    return shift;
}

static void init_desc(marmot_graph_tensor_desc_t *desc, const size_t *shape, size_t ndim, marmot_dtype_t dtype) {
    memset(desc, 0, sizeof(*desc));
    desc->dtype = dtype;
    desc->ndim = ndim;
    for (size_t i = 0; i < ndim; ++i) {
        desc->shape[i] = shape[i];
    }
    if (ndim > 0) {
        desc->strides[ndim - 1] = 1;
        for (size_t i = ndim - 1; i-- > 0;) {
            desc->strides[i] = desc->strides[i + 1] * desc->shape[i + 1];
        }
    }
}

static void fill_tensor_f16(marmot_context_t *ctx, marmot_tensor_t *tensor, const float *src, size_t count) {
    marmot_float16_t *dst = marmot_tensor_data_f16_mut(ctx, tensor);
    assert_non_null(dst);
    assert_int_equal(marmot_convert_f32_to_f16(ctx, dst, src, count), MARMOT_SUCCESS);
}

static void read_tensor_f16(marmot_context_t *ctx, marmot_tensor_t *tensor, float *dst, size_t count) {
    const marmot_float16_t *src = marmot_tensor_data_f16(ctx, tensor);
    assert_non_null(src);
    assert_int_equal(marmot_convert_f16_to_f32(ctx, dst, src, count), MARMOT_SUCCESS);
}

static void test_graph_paged_attention_exec(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    const size_t token_meta_shape[2] = {2, 4};
    const size_t q_shape[3] = {2, 1, 2};
    const size_t kv_shape[5] = {1, 1, 1, 4, 2};
    const size_t block_table_shape[2] = {1, 1};

    marmot_graph_tensor_desc_t desc_meta;
    marmot_graph_tensor_desc_t desc_q;
    marmot_graph_tensor_desc_t desc_kv;
    marmot_graph_tensor_desc_t desc_block;
    init_desc(&desc_meta, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    init_desc(&desc_q, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    init_desc(&desc_kv, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    init_desc(&desc_block, block_table_shape, 2, MARMOT_DTYPE_UINT32);

    marmot_value_id_t ids[7] = {0};
    assert_int_equal(marmot_graph_add_input(graph, &desc_meta, &ids[0]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q, &ids[1]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q, &ids[2]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q, &ids[3]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_kv, &ids[4]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_kv, &ids[5]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_block, &ids[6]), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {0};
    sig.op_id = MARMOT_OP_PAGED_ATTENTION;
    sig.profile_id = MARMOT_PROFILE_INVALID;
    sig.input_dtype = MARMOT_DTYPE_COUNT;
    sig.weight_dtype = MARMOT_DTYPE_COUNT;
    sig.output_dtype = MARMOT_DTYPE_COUNT;
    sig.accum_dtype = MARMOT_DTYPE_COUNT;
    sig.qscheme_id = MARMOT_QSCHEME_NONE;
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;
    sig.epilogue_flags = MARMOT_EPILOGUE_NONE;
    sig.activation = MARMOT_DEVICE_UNARY_IDENTITY;
    sig.variant_flags = MARMOT_FUSION_NONE;

    marmot_graph_tensor_desc_t out_desc;
    init_desc(&out_desc, q_shape, 3, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(
        marmot_graph_add_op(graph, "paged_attention", &sig, ids, 7, &out_desc, 1, &out_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    marmot_tensor_t *token_meta = marmot_tensor_create(ctx, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *q = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *k_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *v_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *kv_k = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *kv_v = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *block_table = marmot_tensor_create(ctx, block_table_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);

    assert_non_null(token_meta);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);
    assert_non_null(out);

    marmot_uint32_t *meta_data = marmot_tensor_data_u32_mut(ctx, token_meta);
    marmot_uint32_t *block_data = marmot_tensor_data_u32_mut(ctx, block_table);
    assert_non_null(meta_data);
    assert_non_null(block_data);
    block_data[0].value = 0;

    const uint32_t shift = log2_u32(4);
    meta_data[0].value = 0;
    meta_data[1].value = 0;
    meta_data[2].value = (0u << shift) | 0u;
    meta_data[3].value = 0;
    meta_data[4].value = 0;
    meta_data[5].value = 1;
    meta_data[6].value = (0u << shift) | 1u;
    meta_data[7].value = 0;

    float q_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    memcpy(q->data, q_vals, sizeof(q_vals));
    memcpy(k_new->data, q_vals, sizeof(q_vals));
    memcpy(v_new->data, q_vals, sizeof(q_vals));

    const marmot_tensor_t *inputs[7] = {token_meta, q, k_new, v_new, kv_k, kv_v, block_table};
    marmot_tensor_t *outputs[1] = {out};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 7, outputs, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_data = marmot_tensor_data_f32(ctx, out);
    assert_non_null(out_data);
    assert_true(fabsf(out_data[0] - 1.0f) < 1e-4f);
    assert_true(fabsf(out_data[1] - 0.0f) < 1e-4f);

    const float scale = 1.0f / sqrtf((float)q_shape[2]);
    const float exp1 = expf(scale);
    const float w0 = 1.0f / (1.0f + exp1);
    const float w1 = exp1 / (1.0f + exp1);
    assert_true(fabsf(out_data[2] - w0) < 1e-4f);
    assert_true(fabsf(out_data[3] - w1) < 1e-4f);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(block_table);
    marmot_tensor_destroy(kv_v);
    marmot_tensor_destroy(kv_k);
    marmot_tensor_destroy(v_new);
    marmot_tensor_destroy(k_new);
    marmot_tensor_destroy(q);
    marmot_tensor_destroy(token_meta);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_paged_attention_fp8_kv(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    if (ctx == nullptr) {
        skip();
    }
#if !MARMOT_ENABLE_FP8
    marmot_destroy(ctx);
    skip();
#else
    marmot_graph_t *graph_f16 = marmot_graph_create();
    marmot_graph_t *graph_fp8 = marmot_graph_create();
    assert_non_null(graph_f16);
    assert_non_null(graph_fp8);

    const size_t token_meta_shape[2] = {2, 4};
    const size_t q_shape[3] = {2, 1, 2};
    const size_t kv_shape[5] = {1, 1, 1, 4, 2};
    const size_t block_table_shape[2] = {1, 1};
    const size_t kv_scale_shape[3] = {1, 1, 1};

    marmot_graph_tensor_desc_t desc_meta;
    marmot_graph_tensor_desc_t desc_q;
    marmot_graph_tensor_desc_t desc_kv_f16;
    marmot_graph_tensor_desc_t desc_kv_fp8;
    marmot_graph_tensor_desc_t desc_block;
    marmot_graph_tensor_desc_t desc_scale;
    init_desc(&desc_meta, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    init_desc(&desc_q, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    init_desc(&desc_kv_f16, kv_shape, 5, MARMOT_DTYPE_FLOAT16);
    init_desc(&desc_kv_fp8, kv_shape, 5, MARMOT_DTYPE_FLOAT8_E4M3);
    init_desc(&desc_block, block_table_shape, 2, MARMOT_DTYPE_UINT32);
    init_desc(&desc_scale, kv_scale_shape, 3, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t ids_f16[7] = {0};
    assert_int_equal(marmot_graph_add_input(graph_f16, &desc_meta, &ids_f16[0]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_f16, &desc_q, &ids_f16[1]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_f16, &desc_q, &ids_f16[2]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_f16, &desc_q, &ids_f16[3]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_f16, &desc_kv_f16, &ids_f16[4]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_f16, &desc_kv_f16, &ids_f16[5]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_f16, &desc_block, &ids_f16[6]), MARMOT_SUCCESS);

    marmot_value_id_t ids_fp8[9] = {0};
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_meta, &ids_fp8[0]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_q, &ids_fp8[1]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_q, &ids_fp8[2]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_q, &ids_fp8[3]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_kv_fp8, &ids_fp8[4]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_kv_fp8, &ids_fp8[5]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_block, &ids_fp8[6]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_scale, &ids_fp8[7]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph_fp8, &desc_scale, &ids_fp8[8]), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {0};
    sig.op_id = MARMOT_OP_PAGED_ATTENTION;
    sig.profile_id = MARMOT_PROFILE_INVALID;
    sig.input_dtype = MARMOT_DTYPE_COUNT;
    sig.weight_dtype = MARMOT_DTYPE_COUNT;
    sig.output_dtype = MARMOT_DTYPE_COUNT;
    sig.accum_dtype = MARMOT_DTYPE_COUNT;
    sig.qscheme_id = MARMOT_QSCHEME_NONE;
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;
    sig.epilogue_flags = MARMOT_EPILOGUE_NONE;
    sig.activation = MARMOT_DEVICE_UNARY_IDENTITY;
    sig.variant_flags = MARMOT_FUSION_NONE;

    marmot_graph_tensor_desc_t out_desc;
    init_desc(&out_desc, q_shape, 3, MARMOT_DTYPE_FLOAT16);

    marmot_value_id_t out_id_f16 = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(
        marmot_graph_add_op(graph_f16, "paged_attention", &sig, ids_f16, 7, &out_desc, 1, &out_id_f16), MARMOT_SUCCESS
    );

    marmot_value_id_t out_id_fp8 = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(
        marmot_graph_add_op(graph_fp8, "paged_attention", &sig, ids_fp8, 9, &out_desc, 1, &out_id_fp8), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph_f16, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_finalize(graph_fp8, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    marmot_tensor_t *token_meta = marmot_tensor_create(ctx, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *q = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *k_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *v_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *kv_k_f16 = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *kv_v_f16 = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *kv_k_fp8 = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT8_E4M3);
    marmot_tensor_t *kv_v_fp8 = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT8_E4M3);
    marmot_tensor_t *block_table = marmot_tensor_create(ctx, block_table_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *kv_k_scale = marmot_tensor_create(ctx, kv_scale_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *kv_v_scale = marmot_tensor_create(ctx, kv_scale_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *out_f16 = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);
    marmot_tensor_t *out_fp8 = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT16);

    assert_non_null(token_meta);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(kv_k_f16);
    assert_non_null(kv_v_f16);
    assert_non_null(kv_k_fp8);
    assert_non_null(kv_v_fp8);
    assert_non_null(block_table);
    assert_non_null(kv_k_scale);
    assert_non_null(kv_v_scale);
    assert_non_null(out_f16);
    assert_non_null(out_fp8);

    marmot_uint32_t *meta_data = marmot_tensor_data_u32_mut(ctx, token_meta);
    marmot_uint32_t *block_data = marmot_tensor_data_u32_mut(ctx, block_table);
    assert_non_null(meta_data);
    assert_non_null(block_data);
    block_data[0].value = 0;

    const uint32_t shift = log2_u32(4);
    meta_data[0].value = 0;
    meta_data[1].value = 0;
    meta_data[2].value = (0u << shift) | 0u;
    meta_data[3].value = 0;
    meta_data[4].value = 0;
    meta_data[5].value = 1;
    meta_data[6].value = (0u << shift) | 1u;
    meta_data[7].value = 0;

    float q_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    fill_tensor_f16(ctx, q, q_vals, 4);
    fill_tensor_f16(ctx, k_new, q_vals, 4);
    fill_tensor_f16(ctx, v_new, q_vals, 4);

    const marmot_tensor_t *inputs_f16[7] = {token_meta, q, k_new, v_new, kv_k_f16, kv_v_f16, block_table};
    marmot_tensor_t *outputs_f16[1] = {out_f16};
    marmot_error_t exec_err = marmot_graph_execute(graph_f16, ctx, inputs_f16, 7, outputs_f16, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute f16 failed: %s", marmot_get_last_error_detail());
    }

    const marmot_tensor_t *inputs_fp8[9] = {token_meta, q,           k_new,      v_new,     kv_k_fp8,
                                            kv_v_fp8,   block_table, kv_k_scale, kv_v_scale};
    marmot_tensor_t *outputs_fp8[1] = {out_fp8};
    exec_err = marmot_graph_execute(graph_fp8, ctx, inputs_fp8, 9, outputs_fp8, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute fp8 failed: %s", marmot_get_last_error_detail());
    }

    float out_f16_data[4] = {0};
    float out_fp8_data[4] = {0};
    read_tensor_f16(ctx, out_f16, out_f16_data, 4);
    read_tensor_f16(ctx, out_fp8, out_fp8_data, 4);

    for (size_t i = 0; i < 4; ++i) {
        assert_true(fabsf(out_fp8_data[i] - out_f16_data[i]) < 0.1f);
    }

    const float *k_scale_data = marmot_tensor_data_f32(ctx, kv_k_scale);
    const float *v_scale_data = marmot_tensor_data_f32(ctx, kv_v_scale);
    assert_non_null(k_scale_data);
    assert_non_null(v_scale_data);
    assert_true(k_scale_data[0] > 0.0f);
    assert_true(v_scale_data[0] > 0.0f);

    marmot_tensor_destroy(out_fp8);
    marmot_tensor_destroy(out_f16);
    marmot_tensor_destroy(kv_v_scale);
    marmot_tensor_destroy(kv_k_scale);
    marmot_tensor_destroy(block_table);
    marmot_tensor_destroy(kv_v_fp8);
    marmot_tensor_destroy(kv_k_fp8);
    marmot_tensor_destroy(kv_v_f16);
    marmot_tensor_destroy(kv_k_f16);
    marmot_tensor_destroy(v_new);
    marmot_tensor_destroy(k_new);
    marmot_tensor_destroy(q);
    marmot_tensor_destroy(token_meta);
    marmot_graph_destroy(graph_fp8);
    marmot_graph_destroy(graph_f16);
    marmot_destroy(ctx);
#endif
}

static void test_graph_paged_attention_exec_strided_q(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    const size_t token_meta_shape[2] = {2, 4};
    const size_t q_shape[3] = {2, 1, 2};
    const size_t kv_shape[5] = {1, 1, 1, 4, 2};
    const size_t block_table_shape[2] = {1, 1};

    marmot_graph_tensor_desc_t desc_meta;
    marmot_graph_tensor_desc_t desc_q_strided;
    marmot_graph_tensor_desc_t desc_q_contig;
    marmot_graph_tensor_desc_t desc_kv;
    marmot_graph_tensor_desc_t desc_block;
    init_desc(&desc_meta, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    init_desc(&desc_q_strided, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    desc_q_strided.strides[2] = 1;
    desc_q_strided.strides[1] = 2;
    desc_q_strided.strides[0] = 4;
    init_desc(&desc_q_contig, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    init_desc(&desc_kv, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    init_desc(&desc_block, block_table_shape, 2, MARMOT_DTYPE_UINT32);

    marmot_value_id_t ids[7] = {0};
    assert_int_equal(marmot_graph_add_input(graph, &desc_meta, &ids[0]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q_strided, &ids[1]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q_contig, &ids[2]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q_contig, &ids[3]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_kv, &ids[4]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_kv, &ids[5]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_block, &ids[6]), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {0};
    sig.op_id = MARMOT_OP_PAGED_ATTENTION;
    sig.profile_id = MARMOT_PROFILE_INVALID;
    sig.input_dtype = MARMOT_DTYPE_COUNT;
    sig.weight_dtype = MARMOT_DTYPE_COUNT;
    sig.output_dtype = MARMOT_DTYPE_COUNT;
    sig.accum_dtype = MARMOT_DTYPE_COUNT;
    sig.qscheme_id = MARMOT_QSCHEME_NONE;
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;
    sig.epilogue_flags = MARMOT_EPILOGUE_NONE;
    sig.activation = MARMOT_DEVICE_UNARY_IDENTITY;
    sig.variant_flags = MARMOT_FUSION_NONE;

    marmot_graph_tensor_desc_t out_desc;
    init_desc(&out_desc, q_shape, 3, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(
        marmot_graph_add_op(graph, "paged_attention", &sig, ids, 7, &out_desc, 1, &out_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    marmot_tensor_t *token_meta = marmot_tensor_create(ctx, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *q_base = marmot_tensor_create(ctx, (size_t[]){2, 2, 2}, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *q = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *k_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *v_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *kv_k = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *kv_v = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *block_table = marmot_tensor_create(ctx, block_table_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);

    assert_non_null(token_meta);
    assert_non_null(q_base);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);
    assert_non_null(out);

    assert_int_equal(marmot_view(ctx, q_base, q, 0), MARMOT_SUCCESS);
    q->shape.strides[0] = q_base->shape.strides[0];
    q->shape.strides[1] = q_base->shape.strides[1];
    q->shape.strides[2] = q_base->shape.strides[2];

    marmot_uint32_t *meta_data = marmot_tensor_data_u32_mut(ctx, token_meta);
    marmot_uint32_t *block_data = marmot_tensor_data_u32_mut(ctx, block_table);
    assert_non_null(meta_data);
    assert_non_null(block_data);
    block_data[0].value = 0;

    const uint32_t shift = log2_u32(4);
    meta_data[0].value = 0;
    meta_data[1].value = 0;
    meta_data[2].value = (0u << shift) | 0u;
    meta_data[3].value = 0;
    meta_data[4].value = 0;
    meta_data[5].value = 1;
    meta_data[6].value = (0u << shift) | 1u;
    meta_data[7].value = 0;

    float q_base_vals[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    memcpy(q_base->data, q_base_vals, sizeof(q_base_vals));

    float q_vals[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    memcpy(k_new->data, q_vals, sizeof(q_vals));
    memcpy(v_new->data, q_vals, sizeof(q_vals));

    const marmot_tensor_t *inputs[7] = {token_meta, q, k_new, v_new, kv_k, kv_v, block_table};
    marmot_tensor_t *outputs[1] = {out};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 7, outputs, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_data = marmot_tensor_data_f32(ctx, out);
    assert_non_null(out_data);
    assert_true(fabsf(out_data[0] - 1.0f) < 1e-4f);
    assert_true(fabsf(out_data[1] - 0.0f) < 1e-4f);

    const float scale = 1.0f / sqrtf((float)q_shape[2]);
    const float exp1 = expf(scale);
    const float w0 = 1.0f / (1.0f + exp1);
    const float w1 = exp1 / (1.0f + exp1);
    assert_true(fabsf(out_data[2] - w0) < 1e-4f);
    assert_true(fabsf(out_data[3] - w1) < 1e-4f);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(block_table);
    marmot_tensor_destroy(kv_v);
    marmot_tensor_destroy(kv_k);
    marmot_tensor_destroy(v_new);
    marmot_tensor_destroy(k_new);
    marmot_tensor_destroy(q);
    marmot_tensor_destroy(q_base);
    marmot_tensor_destroy(token_meta);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

static void test_graph_paged_attention_exec_capacity(void **state) {
    (void)state;

    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert_non_null(ctx);

    marmot_graph_t *graph = marmot_graph_create();
    assert_non_null(graph);

    const size_t token_meta_shape[2] = {4, 4};
    const size_t token_meta_runtime_shape[2] = {2, 4};
    const size_t q_shape[3] = {4, 1, 2};
    const size_t kv_shape[5] = {1, 1, 1, 4, 2};
    const size_t block_table_shape[2] = {1, 1};

    marmot_graph_tensor_desc_t desc_meta;
    marmot_graph_tensor_desc_t desc_q;
    marmot_graph_tensor_desc_t desc_kv;
    marmot_graph_tensor_desc_t desc_block;
    init_desc(&desc_meta, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    init_desc(&desc_q, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    init_desc(&desc_kv, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    init_desc(&desc_block, block_table_shape, 2, MARMOT_DTYPE_UINT32);

    marmot_value_id_t ids[7] = {0};
    assert_int_equal(marmot_graph_add_input(graph, &desc_meta, &ids[0]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q, &ids[1]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q, &ids[2]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_q, &ids[3]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_kv, &ids[4]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_kv, &ids[5]), MARMOT_SUCCESS);
    assert_int_equal(marmot_graph_add_input(graph, &desc_block, &ids[6]), MARMOT_SUCCESS);

    marmot_op_signature_t sig = {0};
    sig.op_id = MARMOT_OP_PAGED_ATTENTION;
    sig.profile_id = MARMOT_PROFILE_INVALID;
    sig.input_dtype = MARMOT_DTYPE_COUNT;
    sig.weight_dtype = MARMOT_DTYPE_COUNT;
    sig.output_dtype = MARMOT_DTYPE_COUNT;
    sig.accum_dtype = MARMOT_DTYPE_COUNT;
    sig.qscheme_id = MARMOT_QSCHEME_NONE;
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;
    sig.epilogue_flags = MARMOT_EPILOGUE_NONE;
    sig.activation = MARMOT_DEVICE_UNARY_IDENTITY;
    sig.variant_flags = MARMOT_FUSION_NONE;

    marmot_graph_tensor_desc_t out_desc;
    init_desc(&out_desc, q_shape, 3, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t out_id = MARMOT_VALUE_ID_INVALID;
    assert_int_equal(
        marmot_graph_add_op(graph, "paged_attention", &sig, ids, 7, &out_desc, 1, &out_id), MARMOT_SUCCESS
    );

    assert_int_equal(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU), MARMOT_SUCCESS);

    marmot_tensor_t *token_meta = marmot_tensor_create(ctx, token_meta_runtime_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *q = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *k_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *v_new = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *kv_k = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *kv_v = marmot_tensor_create(ctx, kv_shape, 5, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *block_table = marmot_tensor_create(ctx, block_table_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *out = marmot_tensor_create(ctx, q_shape, 3, MARMOT_DTYPE_FLOAT32);

    assert_non_null(token_meta);
    assert_non_null(q);
    assert_non_null(k_new);
    assert_non_null(v_new);
    assert_non_null(kv_k);
    assert_non_null(kv_v);
    assert_non_null(block_table);
    assert_non_null(out);

    marmot_uint32_t *meta_data = marmot_tensor_data_u32_mut(ctx, token_meta);
    marmot_uint32_t *block_data = marmot_tensor_data_u32_mut(ctx, block_table);
    assert_non_null(meta_data);
    assert_non_null(block_data);
    block_data[0].value = 0;

    const uint32_t shift = log2_u32(4);
    meta_data[0].value = 0;
    meta_data[1].value = 0;
    meta_data[2].value = (0u << shift) | 0u;
    meta_data[3].value = 0;
    meta_data[4].value = 0;
    meta_data[5].value = 1;
    meta_data[6].value = (0u << shift) | 1u;
    meta_data[7].value = 0;

    float q_vals[8] = {1.0f, 0.0f, 0.0f, 1.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    memcpy(q->data, q_vals, sizeof(q_vals));
    memcpy(k_new->data, q_vals, sizeof(q_vals));
    memcpy(v_new->data, q_vals, sizeof(q_vals));

    float *out_data_mut = marmot_tensor_data_f32_mut(ctx, out);
    assert_non_null(out_data_mut);
    for (size_t i = 0; i < 8; ++i) {
        out_data_mut[i] = 123.0f;
    }

    const marmot_tensor_t *inputs[7] = {token_meta, q, k_new, v_new, kv_k, kv_v, block_table};
    marmot_tensor_t *outputs[1] = {out};
    marmot_error_t exec_err = marmot_graph_execute(graph, ctx, inputs, 7, outputs, 1);
    if (exec_err != MARMOT_SUCCESS) {
        fail_msg("graph_execute failed: %s", marmot_get_last_error_detail());
    }

    const float *out_data = marmot_tensor_data_f32(ctx, out);
    assert_non_null(out_data);
    assert_true(fabsf(out_data[0] - 1.0f) < 1e-4f);
    assert_true(fabsf(out_data[1] - 0.0f) < 1e-4f);

    const float scale = 1.0f / sqrtf((float)q_shape[2]);
    const float exp1 = expf(scale);
    const float w0 = 1.0f / (1.0f + exp1);
    const float w1 = exp1 / (1.0f + exp1);
    assert_true(fabsf(out_data[2] - w0) < 1e-4f);
    assert_true(fabsf(out_data[3] - w1) < 1e-4f);

    assert_true(fabsf(out_data[4] - 123.0f) < 1e-4f);
    assert_true(fabsf(out_data[5] - 123.0f) < 1e-4f);
    assert_true(fabsf(out_data[6] - 123.0f) < 1e-4f);
    assert_true(fabsf(out_data[7] - 123.0f) < 1e-4f);

    marmot_tensor_destroy(out);
    marmot_tensor_destroy(block_table);
    marmot_tensor_destroy(kv_v);
    marmot_tensor_destroy(kv_k);
    marmot_tensor_destroy(v_new);
    marmot_tensor_destroy(k_new);
    marmot_tensor_destroy(q);
    marmot_tensor_destroy(token_meta);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
}

int main(void) {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_graph_paged_attention_exec),
        cmocka_unit_test(test_graph_paged_attention_fp8_kv),
        cmocka_unit_test(test_graph_paged_attention_exec_strided_q),
        cmocka_unit_test(test_graph_paged_attention_exec_capacity),
    };

    return cmocka_run_group_tests(tests, nullptr, nullptr);
}
