#include "marmot/graph/graph.h"
#include "marmot/marmot.h"
#include "marmot/op_metadata.gen.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <variant>

#include "graph/fast_executor.hpp"
#include "graph/fast_plan.hpp"
#include "graph/graph_handle.hpp"

namespace {

void init_desc_2d(marmot_graph_tensor_desc_t &desc, size_t dim0, size_t dim1, marmot_dtype_t dtype) {
    std::memset(&desc, 0, sizeof(desc));
    desc.dtype = dtype;
    desc.ndim = 2;
    desc.shape[0] = dim0;
    desc.shape[1] = dim1;
    desc.strides[1] = 1;
    desc.strides[0] = dim1;
}

void init_desc_1d(marmot_graph_tensor_desc_t &desc, size_t dim0, marmot_dtype_t dtype) {
    std::memset(&desc, 0, sizeof(desc));
    desc.dtype = dtype;
    desc.ndim = 1;
    desc.shape[0] = dim0;
    desc.strides[0] = 1;
}

marmot_op_signature_t make_sig(marmot_op_id_t op_id) {
    marmot_op_signature_t sig{};
    sig.op_id = op_id;
    sig.profile_id = MARMOT_PROFILE_INVALID;
    sig.input_dtype = MARMOT_DTYPE_FLOAT32;
    sig.weight_dtype = MARMOT_DTYPE_FLOAT32;
    sig.output_dtype = MARMOT_DTYPE_FLOAT32;
    sig.accum_dtype = MARMOT_DTYPE_FLOAT32;
    sig.qscheme_id = MARMOT_QSCHEME_NONE;
    sig.weight_layout = MARMOT_WEIGHT_LAYOUT_INVALID;
    sig.stride_mode = MARMOT_STRIDE_MODE_CONTIGUOUS;
    return sig;
}

void set_last_fast_hint(
    marmot_graph_t *graph, marmot_fast_stage_hint_t stage_hint, marmot_fast_node_role_t role, uint32_t block_id
) {
    assert(graph != nullptr);
    assert(graph->inner.set_last_node_fast_hint_checked(stage_hint, role, block_id) == MARMOT_SUCCESS);
}

void test_compile_fast_plan_generic() {
    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t desc{};
    init_desc_2d(desc, 2, 4, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &desc, &input) == MARMOT_SUCCESS);

    const marmot_op_signature_t sig = make_sig(MARMOT_OP_RELU);
    marmot_value_id_t out = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_op(graph, "relu", &sig, &input, 1, &desc, 1, &out) == MARMOT_SUCCESS);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 16,
        .sample_count = 0,
        .emit_logits = false,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->phase() == marmot::graph::FastPlanPhase::Prefill);
    assert(plan->node_count() == 1);
    assert(plan->stages().size() == 1);
    assert(plan->stages()[0].kind == marmot::graph::FastStageKind::GenericFallback);
    assert(plan->fast_stage_count() == 0);

    marmot_graph_destroy(graph);
    std::printf("  ok compile_fast_plan_generic\n");
}

void test_compile_fast_plan_boundaries() {
    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t desc{};
    init_desc_2d(desc, 2, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t gather_idx_desc{};
    init_desc_1d(gather_idx_desc, 1, MARMOT_DTYPE_INT32);
    marmot_graph_tensor_desc_t gather_desc{};
    init_desc_2d(gather_desc, 1, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t logits_weight_desc{};
    init_desc_2d(logits_weight_desc, 4, 8, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t logits_desc{};
    init_desc_2d(logits_desc, 1, 8, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t gather_idx = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t logits_weight = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &desc, &input) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &gather_idx_desc, &gather_idx) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &logits_weight_desc, &logits_weight) == MARMOT_SUCCESS);

    const marmot_op_signature_t relu = make_sig(MARMOT_OP_RELU);
    const marmot_op_signature_t attn = make_sig(MARMOT_OP_PAGED_ATTENTION);
    const marmot_op_signature_t gather = make_sig(MARMOT_OP_GATHER_ROWS);
    const marmot_op_signature_t matmul = make_sig(MARMOT_OP_MATMUL);

    marmot_value_id_t v1 = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v2 = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v3 = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v4 = MARMOT_VALUE_ID_INVALID;

    assert(marmot_graph_add_op(graph, "relu", &relu, &input, 1, &desc, 1, &v1) == MARMOT_SUCCESS);
    assert(marmot_graph_add_op(graph, "paged_attention", &attn, &v1, 1, &desc, 1, &v2) == MARMOT_SUCCESS);
    const marmot_value_id_t gather_inputs[2] = {v2, gather_idx};
    assert(
        marmot_graph_add_op(graph, "gather_rows", &gather, gather_inputs, 2, &gather_desc, 1, &v3) == MARMOT_SUCCESS
    );
    const marmot_value_id_t matmul_inputs[2] = {v3, logits_weight};
    assert(marmot_graph_add_op(graph, "matmul", &matmul, matmul_inputs, 2, &logits_desc, 1, &v4) == MARMOT_SUCCESS);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 1,
        .sample_count = 1,
        .emit_logits = true,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->phase() == marmot::graph::FastPlanPhase::Decode);
    assert(plan->stages().size() == 3);
    assert(plan->stages()[0].kind == marmot::graph::FastStageKind::GenericFallback);
    assert(plan->stages()[1].kind == marmot::graph::FastStageKind::Attention);
    assert(plan->stages()[2].kind == marmot::graph::FastStageKind::LogitsTail);
    assert(plan->stages()[2].lowering == marmot::graph::FastStageLowering::LogitsGatherMatmul);
    assert(std::holds_alternative<marmot::graph::FastLogitsSupernode>(plan->stages()[2].payload));
    assert(plan->fast_stage_count() == 2);

    marmot_graph_destroy(graph);
    std::printf("  ok compile_fast_plan_boundaries\n");
}

void test_compile_fast_plan_logits_tail_with_norm() {
    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t hidden_desc{};
    init_desc_2d(hidden_desc, 2, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t norm_desc{};
    init_desc_1d(norm_desc, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t gather_idx_desc{};
    init_desc_1d(gather_idx_desc, 1, MARMOT_DTYPE_INT32);
    marmot_graph_tensor_desc_t gather_desc{};
    init_desc_2d(gather_desc, 1, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t logits_weight_desc{};
    init_desc_2d(logits_weight_desc, 4, 8, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t logits_desc{};
    init_desc_2d(logits_desc, 1, 8, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t hidden = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t norm_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t gather_idx = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t logits_weight = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &hidden_desc, &hidden) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &norm_desc, &norm_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &gather_idx_desc, &gather_idx) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &logits_weight_desc, &logits_weight) == MARMOT_SUCCESS);

    const marmot_op_signature_t rms_norm = make_sig(MARMOT_OP_RMS_NORM);
    const marmot_op_signature_t gather = make_sig(MARMOT_OP_GATHER_ROWS);
    const marmot_op_signature_t matmul = make_sig(MARMOT_OP_MATMUL);

    marmot_value_id_t norm = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t gathered = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t logits = MARMOT_VALUE_ID_INVALID;
    const marmot_value_id_t norm_inputs[2] = {hidden, norm_weight};
    assert(marmot_graph_add_op(graph, "rms_norm", &rms_norm, norm_inputs, 2, &hidden_desc, 1, &norm) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_LOGITS_TAIL, MARMOT_FAST_NODE_ROLE_LOGITS_NORM, 0);
    const marmot_value_id_t gather_inputs[2] = {norm, gather_idx};
    assert(
        marmot_graph_add_op(graph, "gather_rows", &gather, gather_inputs, 2, &gather_desc, 1, &gathered) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_LOGITS_TAIL, MARMOT_FAST_NODE_ROLE_LOGITS_GATHER, 0);
    const marmot_value_id_t logits_inputs[2] = {gathered, logits_weight};
    assert(marmot_graph_add_op(graph, "logits", &matmul, logits_inputs, 2, &logits_desc, 1, &logits) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_LOGITS_TAIL, MARMOT_FAST_NODE_ROLE_LOGITS_PROJECTION, 0);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 1,
        .sample_count = 1,
        .emit_logits = true,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->stages().size() == 1);
    assert(plan->stages()[0].kind == marmot::graph::FastStageKind::LogitsTail);
    assert(plan->stages()[0].lowering == marmot::graph::FastStageLowering::LogitsGatherMatmul);
    const auto *payload = std::get_if<marmot::graph::FastLogitsSupernode>(&plan->stages()[0].payload);
    assert(payload != nullptr);
    assert(payload->final_norm.has_value());
    assert(payload->gather.has_value());
    assert(payload->matmul.has_value());
    assert(!payload->vec_dot.has_value());

    marmot_graph_destroy(graph);
    std::printf("  ok compile_fast_plan_logits_tail_with_norm\n");
}

void test_compile_fast_plan_dense_ffn() {
    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t hidden_desc{};
    init_desc_2d(hidden_desc, 2, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t norm_desc{};
    init_desc_1d(norm_desc, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t up_weight_desc{};
    init_desc_2d(up_weight_desc, 4, 8, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t ffn_desc{};
    init_desc_2d(ffn_desc, 2, 8, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t down_weight_desc{};
    init_desc_2d(down_weight_desc, 8, 4, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t hidden = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t norm_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t up_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t down_weight = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &hidden_desc, &hidden) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &norm_desc, &norm_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &up_weight_desc, &up_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &down_weight_desc, &down_weight) == MARMOT_SUCCESS);

    const marmot_op_signature_t rms_norm = make_sig(MARMOT_OP_RMS_NORM);
    const marmot_op_signature_t matmul = make_sig(MARMOT_OP_MATMUL);
    const marmot_op_signature_t gelu = make_sig(MARMOT_OP_GELU);
    const marmot_op_signature_t add = make_sig(MARMOT_OP_ADD);

    marmot_value_id_t norm = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t up = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t act = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t down = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output = MARMOT_VALUE_ID_INVALID;
    const marmot_value_id_t rms_inputs[2] = {hidden, norm_weight};

    assert(marmot_graph_add_op(graph, "rms_norm", &rms_norm, rms_inputs, 2, &hidden_desc, 1, &norm) == MARMOT_SUCCESS);
    const marmot_value_id_t up_inputs[2] = {norm, up_weight};
    assert(marmot_graph_add_op(graph, "up_proj", &matmul, up_inputs, 2, &ffn_desc, 1, &up) == MARMOT_SUCCESS);
    assert(marmot_graph_add_op(graph, "gelu", &gelu, &up, 1, &ffn_desc, 1, &act) == MARMOT_SUCCESS);
    const marmot_value_id_t down_inputs[2] = {act, down_weight};
    assert(marmot_graph_add_op(graph, "down_proj", &matmul, down_inputs, 2, &hidden_desc, 1, &down) == MARMOT_SUCCESS);
    const marmot_value_id_t add_inputs[2] = {hidden, down};
    assert(marmot_graph_add_op(graph, "add", &add, add_inputs, 2, &hidden_desc, 1, &output) == MARMOT_SUCCESS);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 1,
        .sample_count = 1,
        .emit_logits = true,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->stages().size() == 1);
    assert(plan->stages()[0].kind == marmot::graph::FastStageKind::DenseFfn);
    assert(plan->stages()[0].lowering == marmot::graph::FastStageLowering::DenseFfnGelu);
    assert(std::holds_alternative<marmot::graph::FastDenseFfnGeluSupernode>(plan->stages()[0].payload));
    assert(plan->stages()[0].node_count == 5);
    assert(plan->fast_stage_count() == 1);

    marmot_graph_destroy(graph);
    std::printf("  ok compile_fast_plan_dense_ffn\n");
}

void test_compile_fast_plan_moe_ffn() {
    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t hidden_desc{};
    init_desc_2d(hidden_desc, 2, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t norm_desc{};
    init_desc_1d(norm_desc, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t router_weight_desc{};
    init_desc_2d(router_weight_desc, 4, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t router_desc{};
    init_desc_2d(router_desc, 2, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t topk_ids_desc{};
    init_desc_2d(topk_ids_desc, 2, 2, MARMOT_DTYPE_INT32);
    marmot_graph_tensor_desc_t topk_weights_desc{};
    init_desc_2d(topk_weights_desc, 2, 2, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t expert_desc{};
    std::memset(&expert_desc, 0, sizeof(expert_desc));
    expert_desc.dtype = MARMOT_DTYPE_FLOAT32;
    expert_desc.ndim = 3;
    expert_desc.shape[0] = 2;
    expert_desc.shape[1] = 4;
    expert_desc.shape[2] = 4;
    expert_desc.strides[2] = 1;
    expert_desc.strides[1] = expert_desc.shape[2];
    expert_desc.strides[0] = expert_desc.shape[1] * expert_desc.shape[2];

    marmot_value_id_t hidden = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t norm_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t router_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t gate = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t up = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t down = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t expert_ids = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t expert_weights = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &hidden_desc, &hidden) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &norm_desc, &norm_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &router_weight_desc, &router_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &expert_desc, &gate) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &expert_desc, &up) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &expert_desc, &down) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &topk_ids_desc, &expert_ids) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &topk_weights_desc, &expert_weights) == MARMOT_SUCCESS);

    const marmot_op_signature_t rms_norm = make_sig(MARMOT_OP_RMS_NORM);
    const marmot_op_signature_t matmul = make_sig(MARMOT_OP_MATMUL);
    const marmot_op_signature_t topk = make_sig(MARMOT_OP_TOPK);
    const marmot_op_signature_t softmax = make_sig(MARMOT_OP_SOFTMAX);
    const marmot_op_signature_t moe = make_sig(MARMOT_OP_MOE_EXPERTS);
    const marmot_op_signature_t add = make_sig(MARMOT_OP_ADD);

    marmot_value_id_t norm = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t router = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t topk_outputs[2] = {MARMOT_VALUE_ID_INVALID, MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t weights = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t moe_out = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output = MARMOT_VALUE_ID_INVALID;
    const marmot_value_id_t rms_inputs[2] = {hidden, norm_weight};

    assert(marmot_graph_add_op(graph, "rms_norm", &rms_norm, rms_inputs, 2, &hidden_desc, 1, &norm) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_MOE_FFN, MARMOT_FAST_NODE_ROLE_FFN_NORM, 0);
    const marmot_value_id_t router_inputs[2] = {norm, router_weight};
    assert(marmot_graph_add_op(graph, "router", &matmul, router_inputs, 2, &router_desc, 1, &router) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_MOE_FFN, MARMOT_FAST_NODE_ROLE_FFN_ROUTER, 0);
    const marmot_graph_tensor_desc_t topk_outputs_desc[2] = {topk_weights_desc, topk_ids_desc};
    assert(marmot_graph_add_op(graph, "topk", &topk, &router, 1, topk_outputs_desc, 2, topk_outputs) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_MOE_FFN, MARMOT_FAST_NODE_ROLE_FFN_TOPK, 0);
    assert(
        marmot_graph_add_op(graph, "softmax", &softmax, &topk_outputs[0], 1, &topk_weights_desc, 1, &weights) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_MOE_FFN, MARMOT_FAST_NODE_ROLE_FFN_ROUTER_WEIGHTS, 0);
    const marmot_value_id_t moe_inputs[6] = {norm, gate, up, down, topk_outputs[1], weights};
    assert(marmot_graph_add_op(graph, "moe_experts", &moe, moe_inputs, 6, &hidden_desc, 1, &moe_out) == MARMOT_SUCCESS);
    assert(marmot_graph_set_last_node_moe_params(graph, MARMOT_FFN_SWIGLU, 1.0f) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_MOE_FFN, MARMOT_FAST_NODE_ROLE_FFN_MOE_EXPERTS, 0);
    const marmot_value_id_t add_inputs[2] = {hidden, moe_out};
    assert(marmot_graph_add_op(graph, "add", &add, add_inputs, 2, &hidden_desc, 1, &output) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_MOE_FFN, MARMOT_FAST_NODE_ROLE_FFN_RESIDUAL, 0);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 1,
        .sample_count = 1,
        .emit_logits = true,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->stages().size() == 1);
    assert(plan->stages()[0].kind == marmot::graph::FastStageKind::MoeFfn);
    assert(plan->stages()[0].lowering == marmot::graph::FastStageLowering::MoeFfnBasic);
    assert(std::holds_alternative<marmot::graph::FastMoeFfnBasicSupernode>(plan->stages()[0].payload));
    assert(plan->stages()[0].node_count == 6);
    assert(plan->fast_stage_count() == 1);

    marmot_graph_destroy(graph);
    std::printf("  ok compile_fast_plan_moe_ffn\n");
}

void test_fast_executor_moe_ffn_cpu() {
    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert(ctx != nullptr);

    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t hidden_desc{};
    init_desc_2d(hidden_desc, 2, 2, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t norm_desc{};
    init_desc_1d(norm_desc, 2, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t router_weight_desc{};
    init_desc_2d(router_weight_desc, 2, 2, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t router_desc{};
    init_desc_2d(router_desc, 2, 2, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t topk_values_desc{};
    init_desc_2d(topk_values_desc, 2, 2, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t topk_ids_desc{};
    init_desc_2d(topk_ids_desc, 2, 2, MARMOT_DTYPE_INT32);
    marmot_graph_tensor_desc_t expert_desc{};
    std::memset(&expert_desc, 0, sizeof(expert_desc));
    expert_desc.dtype = MARMOT_DTYPE_FLOAT32;
    expert_desc.ndim = 3;
    expert_desc.shape[0] = 2;
    expert_desc.shape[1] = 2;
    expert_desc.shape[2] = 2;
    expert_desc.strides[2] = 1;
    expert_desc.strides[1] = 2;
    expert_desc.strides[0] = 4;

    marmot_value_id_t hidden = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t norm_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t router_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t gate = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t up = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t down = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &hidden_desc, &hidden) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &norm_desc, &norm_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &router_weight_desc, &router_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &expert_desc, &gate) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &expert_desc, &up) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &expert_desc, &down) == MARMOT_SUCCESS);

    const marmot_op_signature_t rms_norm = make_sig(MARMOT_OP_RMS_NORM);
    const marmot_op_signature_t matmul = make_sig(MARMOT_OP_MATMUL);
    const marmot_op_signature_t topk = make_sig(MARMOT_OP_TOPK);
    const marmot_op_signature_t softmax = make_sig(MARMOT_OP_SOFTMAX);
    const marmot_op_signature_t moe = make_sig(MARMOT_OP_MOE_EXPERTS);
    const marmot_op_signature_t add = make_sig(MARMOT_OP_ADD);

    marmot_value_id_t norm = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t router = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t topk_outputs[2] = {MARMOT_VALUE_ID_INVALID, MARMOT_VALUE_ID_INVALID};
    marmot_value_id_t router_weights = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t moe_out = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output = MARMOT_VALUE_ID_INVALID;

    const marmot_value_id_t rms_inputs[2] = {hidden, norm_weight};
    assert(marmot_graph_add_op(graph, "rms_norm", &rms_norm, rms_inputs, 2, &hidden_desc, 1, &norm) == MARMOT_SUCCESS);
    const marmot_value_id_t router_inputs[2] = {norm, router_weight};
    assert(marmot_graph_add_op(graph, "router", &matmul, router_inputs, 2, &router_desc, 1, &router) == MARMOT_SUCCESS);
    const marmot_graph_tensor_desc_t topk_outputs_desc[2] = {topk_values_desc, topk_ids_desc};
    assert(marmot_graph_add_op(graph, "topk", &topk, &router, 1, topk_outputs_desc, 2, topk_outputs) == MARMOT_SUCCESS);
    assert(
        marmot_graph_add_op(graph, "softmax", &softmax, &topk_outputs[0], 1, &topk_values_desc, 1, &router_weights) ==
        MARMOT_SUCCESS
    );
    const marmot_value_id_t moe_inputs[6] = {norm, gate, up, down, topk_outputs[1], router_weights};
    assert(marmot_graph_add_op(graph, "moe_experts", &moe, moe_inputs, 6, &hidden_desc, 1, &moe_out) == MARMOT_SUCCESS);
    assert(marmot_graph_set_last_node_moe_params(graph, MARMOT_FFN_SWIGLU, 1.0f) == MARMOT_SUCCESS);
    const marmot_value_id_t add_inputs[2] = {hidden, moe_out};
    assert(marmot_graph_add_op(graph, "add", &add, add_inputs, 2, &hidden_desc, 1, &output) == MARMOT_SUCCESS);
    assert(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU) == MARMOT_SUCCESS);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 1,
        .sample_count = 1,
        .emit_logits = true,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->stages().size() == 1);
    assert(plan->stages()[0].lowering == marmot::graph::FastStageLowering::MoeFfnBasic);

    const size_t hidden_shape[2] = {2, 2};
    const size_t norm_shape[1] = {2};
    const size_t router_shape[2] = {2, 2};
    const size_t expert_shape[3] = {2, 2, 2};
    const float hidden_data[] = {1.0f, -0.5f, 0.25f, 0.75f};
    const float norm_weight_data[] = {1.0f, 1.0f};
    const float router_weight_data[] = {1.5f, -0.25f, -0.5f, 1.25f};
    const float gate_data[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.2f, -0.4f, 0.7f, 0.1f,
    };
    const float up_data[] = {
        0.5f, 0.0f, 0.0f, 2.0f, 1.2f, 0.3f, -0.6f, 0.8f,
    };
    const float down_data[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 0.4f, -0.2f, 0.3f, 0.6f,
    };

    marmot_tensor_t *hidden_tensor = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *norm_weight_tensor = marmot_tensor_create(ctx, norm_shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *router_weight_tensor = marmot_tensor_create(ctx, router_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *gate_tensor = marmot_tensor_create(ctx, expert_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *up_tensor = marmot_tensor_create(ctx, expert_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *down_tensor = marmot_tensor_create(ctx, expert_shape, 3, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_ref = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_fast = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert(hidden_tensor != nullptr);
    assert(norm_weight_tensor != nullptr);
    assert(router_weight_tensor != nullptr);
    assert(gate_tensor != nullptr);
    assert(up_tensor != nullptr);
    assert(down_tensor != nullptr);
    assert(output_ref != nullptr);
    assert(output_fast != nullptr);

    std::memcpy(hidden_tensor->data, hidden_data, sizeof(hidden_data));
    std::memcpy(norm_weight_tensor->data, norm_weight_data, sizeof(norm_weight_data));
    std::memcpy(router_weight_tensor->data, router_weight_data, sizeof(router_weight_data));
    std::memcpy(gate_tensor->data, gate_data, sizeof(gate_data));
    std::memcpy(up_tensor->data, up_data, sizeof(up_data));
    std::memcpy(down_tensor->data, down_data, sizeof(down_data));

    const marmot_tensor_t *inputs[] = {
        hidden_tensor, norm_weight_tensor, router_weight_tensor, gate_tensor, up_tensor, down_tensor,
    };
    marmot_tensor_t *outputs_ref[] = {output_ref};
    assert(
        marmot_graph_execute(graph, ctx, inputs, std::size(inputs), outputs_ref, std::size(outputs_ref)) ==
        MARMOT_SUCCESS
    );

    marmot_tensor_t *outputs_fast[] = {output_fast};
    marmot::graph::FastExecProfile profile{};
    assert(
        marmot::graph::FastExecutor::execute(
            graph, &plan.value(), ctx, std::span<const marmot_tensor_t *const>(inputs, std::size(inputs)),
            std::span<marmot_tensor_t *const>(outputs_fast, std::size(outputs_fast)), &profile
        ) == MARMOT_SUCCESS
    );

    const float *ref_data = marmot_tensor_data_f32(ctx, output_ref);
    const float *fast_data = marmot_tensor_data_f32(ctx, output_fast);
    assert(ref_data != nullptr);
    assert(fast_data != nullptr);
    for (size_t i = 0; i < 4; ++i) {
        assert(std::fabs(ref_data[i] - fast_data[i]) < 1e-6f);
    }

    assert(profile.stages.size() == 1);
    assert(profile.stages[0].kind == marmot::graph::FastStageKind::MoeFfn);
    assert(profile.stages[0].executed_ops == 6);

    marmot_tensor_destroy(output_fast);
    marmot_tensor_destroy(output_ref);
    marmot_tensor_destroy(down_tensor);
    marmot_tensor_destroy(up_tensor);
    marmot_tensor_destroy(gate_tensor);
    marmot_tensor_destroy(router_weight_tensor);
    marmot_tensor_destroy(norm_weight_tensor);
    marmot_tensor_destroy(hidden_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
    std::printf("  ok fast_executor_moe_ffn_cpu\n");
}

void test_compile_fast_plan_attention_decode() {
    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t hidden_desc{};
    init_desc_2d(hidden_desc, 1, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t norm_desc{};
    init_desc_1d(norm_desc, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t proj_weight_desc{};
    init_desc_2d(proj_weight_desc, 4, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t qkv_desc{};
    init_desc_2d(qkv_desc, 1, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t q_heads_desc{};
    std::memset(&q_heads_desc, 0, sizeof(q_heads_desc));
    q_heads_desc.dtype = MARMOT_DTYPE_FLOAT32;
    q_heads_desc.ndim = 3;
    q_heads_desc.shape[0] = 2;
    q_heads_desc.shape[1] = 1;
    q_heads_desc.shape[2] = 2;
    q_heads_desc.strides[2] = 1;
    q_heads_desc.strides[1] = 4;
    q_heads_desc.strides[0] = 2;
    marmot_graph_tensor_desc_t q_tokens_desc{};
    std::memset(&q_tokens_desc, 0, sizeof(q_tokens_desc));
    q_tokens_desc.dtype = MARMOT_DTYPE_FLOAT32;
    q_tokens_desc.ndim = 3;
    q_tokens_desc.shape[0] = 1;
    q_tokens_desc.shape[1] = 2;
    q_tokens_desc.shape[2] = 2;
    q_tokens_desc.strides[2] = 1;
    q_tokens_desc.strides[1] = 2;
    q_tokens_desc.strides[0] = 4;
    marmot_graph_tensor_desc_t token_meta_desc{};
    init_desc_2d(token_meta_desc, 1, 4, MARMOT_DTYPE_UINT32);
    marmot_graph_tensor_desc_t kv_desc{};
    std::memset(&kv_desc, 0, sizeof(kv_desc));
    kv_desc.dtype = MARMOT_DTYPE_FLOAT32;
    kv_desc.ndim = 5;
    kv_desc.shape[0] = 1;
    kv_desc.shape[1] = 1;
    kv_desc.shape[2] = 2;
    kv_desc.shape[3] = 1;
    kv_desc.shape[4] = 2;
    kv_desc.strides[4] = 1;
    kv_desc.strides[3] = 2;
    kv_desc.strides[2] = 2;
    kv_desc.strides[1] = 4;
    kv_desc.strides[0] = 4;
    marmot_graph_tensor_desc_t block_table_desc{};
    init_desc_2d(block_table_desc, 1, 1, MARMOT_DTYPE_UINT32);
    marmot_graph_tensor_desc_t positions_desc{};
    init_desc_1d(positions_desc, 1, MARMOT_DTYPE_INT32);

    marmot_value_id_t hidden = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t norm_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t attn_out_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t token_meta = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t kv_k = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t kv_v = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t block_table = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t positions = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &hidden_desc, &hidden) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &norm_desc, &norm_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &proj_weight_desc, &q_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &proj_weight_desc, &k_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &proj_weight_desc, &v_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &proj_weight_desc, &attn_out_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &token_meta_desc, &token_meta) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &kv_desc, &kv_k) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &kv_desc, &kv_v) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &block_table_desc, &block_table) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &positions_desc, &positions) == MARMOT_SUCCESS);

    const marmot_op_signature_t rms_norm = make_sig(MARMOT_OP_RMS_NORM);
    const marmot_op_signature_t matmul = make_sig(MARMOT_OP_MATMUL);
    const marmot_op_signature_t reshape = make_sig(MARMOT_OP_RESHAPE);
    const marmot_op_signature_t rope = make_sig(MARMOT_OP_ROPE);
    const marmot_op_signature_t paged_attention = make_sig(MARMOT_OP_PAGED_ATTENTION);
    const marmot_op_signature_t add = make_sig(MARMOT_OP_ADD);

    marmot_value_id_t norm = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q_heads = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k_heads = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v_heads = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q_rope = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k_rope = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q_tokens = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k_tokens = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v_tokens = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t attn_heads = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t attn_flat = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t attn_proj = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output = MARMOT_VALUE_ID_INVALID;

    const marmot_value_id_t norm_inputs[2] = {hidden, norm_weight};
    assert(marmot_graph_add_op(graph, "rms_norm", &rms_norm, norm_inputs, 2, &hidden_desc, 1, &norm) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_NORM, 0);
    const marmot_value_id_t q_inputs[2] = {norm, q_weight};
    const marmot_value_id_t k_inputs[2] = {norm, k_weight};
    const marmot_value_id_t v_inputs[2] = {norm, v_weight};
    assert(marmot_graph_add_op(graph, "q_proj", &matmul, q_inputs, 2, &qkv_desc, 1, &q) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_Q_PROJ, 0);
    assert(marmot_graph_add_op(graph, "k_proj", &matmul, k_inputs, 2, &qkv_desc, 1, &k) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_K_PROJ, 0);
    assert(marmot_graph_add_op(graph, "v_proj", &matmul, v_inputs, 2, &qkv_desc, 1, &v) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_V_PROJ, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_q_heads", &reshape, &q, 1, &q_heads_desc, 1, &q_heads) == MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_Q_HEADS, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_k_heads", &reshape, &k, 1, &q_heads_desc, 1, &k_heads) == MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_K_HEADS, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_v_heads", &reshape, &v, 1, &q_heads_desc, 1, &v_heads) == MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_V_HEADS, 0);
    const marmot_value_id_t q_rope_inputs[2] = {q_heads, positions};
    const marmot_value_id_t k_rope_inputs[2] = {k_heads, positions};
    assert(marmot_graph_add_op(graph, "q_rope", &rope, q_rope_inputs, 2, &q_heads_desc, 1, &q_rope) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_Q_ROPE, 0);
    assert(marmot_graph_add_op(graph, "k_rope", &rope, k_rope_inputs, 2, &q_heads_desc, 1, &k_rope) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_K_ROPE, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_q_tokens", &reshape, &q_rope, 1, &q_tokens_desc, 1, &q_tokens) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_Q_TOKENS, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_k_tokens", &reshape, &k_rope, 1, &q_tokens_desc, 1, &k_tokens) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_K_TOKENS, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_v_tokens", &reshape, &v_heads, 1, &q_tokens_desc, 1, &v_tokens) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_V_TOKENS, 0);
    const marmot_value_id_t attn_inputs[7] = {token_meta, q_tokens, k_tokens, v_tokens, kv_k, kv_v, block_table};
    assert(
        marmot_graph_add_op(
            graph, "paged_attention", &paged_attention, attn_inputs, 7, &q_tokens_desc, 1, &attn_heads
        ) == MARMOT_SUCCESS
    );
    graph->inner.set_last_node_paged_attention_layer(0);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_PAGED, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_attn_flat", &reshape, &attn_heads, 1, &hidden_desc, 1, &attn_flat) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_OUT_RESHAPE, 0);
    const marmot_value_id_t attn_proj_inputs[2] = {attn_flat, attn_out_weight};
    assert(
        marmot_graph_add_op(graph, "attn_out", &matmul, attn_proj_inputs, 2, &hidden_desc, 1, &attn_proj) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_OUT_PROJ, 0);
    const marmot_value_id_t add_inputs[2] = {hidden, attn_proj};
    assert(marmot_graph_add_op(graph, "add", &add, add_inputs, 2, &hidden_desc, 1, &output) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_RESIDUAL, 0);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 1,
        .sample_count = 1,
        .emit_logits = true,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->stages().size() == 1);
    assert(plan->stages()[0].kind == marmot::graph::FastStageKind::Attention);
    assert(plan->stages()[0].lowering == marmot::graph::FastStageLowering::AttentionDecodePaged);
    const auto *payload = std::get_if<marmot::graph::FastAttentionDecodeSupernode>(&plan->stages()[0].payload);
    assert(payload != nullptr);
    assert(!payload->q_norm.has_value());
    assert(!payload->k_norm.has_value());
    assert(payload->paged_attention.layer_idx == 0);

    marmot_graph_destroy(graph);
    std::printf("  ok compile_fast_plan_attention_decode\n");
}

void test_fast_executor_attention_decode_cpu() {
    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert(ctx != nullptr);

    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t hidden_desc{};
    init_desc_2d(hidden_desc, 1, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t norm_desc{};
    init_desc_1d(norm_desc, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t proj_weight_desc{};
    init_desc_2d(proj_weight_desc, 4, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t qkv_desc{};
    init_desc_2d(qkv_desc, 1, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t q_heads_desc{};
    std::memset(&q_heads_desc, 0, sizeof(q_heads_desc));
    q_heads_desc.dtype = MARMOT_DTYPE_FLOAT32;
    q_heads_desc.ndim = 3;
    q_heads_desc.shape[0] = 2;
    q_heads_desc.shape[1] = 1;
    q_heads_desc.shape[2] = 2;
    q_heads_desc.strides[2] = 1;
    q_heads_desc.strides[1] = 4;
    q_heads_desc.strides[0] = 2;
    marmot_graph_tensor_desc_t q_tokens_desc{};
    std::memset(&q_tokens_desc, 0, sizeof(q_tokens_desc));
    q_tokens_desc.dtype = MARMOT_DTYPE_FLOAT32;
    q_tokens_desc.ndim = 3;
    q_tokens_desc.shape[0] = 1;
    q_tokens_desc.shape[1] = 2;
    q_tokens_desc.shape[2] = 2;
    q_tokens_desc.strides[2] = 1;
    q_tokens_desc.strides[1] = 2;
    q_tokens_desc.strides[0] = 4;
    marmot_graph_tensor_desc_t token_meta_desc{};
    init_desc_2d(token_meta_desc, 1, 4, MARMOT_DTYPE_UINT32);
    marmot_graph_tensor_desc_t kv_desc{};
    std::memset(&kv_desc, 0, sizeof(kv_desc));
    kv_desc.dtype = MARMOT_DTYPE_FLOAT32;
    kv_desc.ndim = 5;
    kv_desc.shape[0] = 1;
    kv_desc.shape[1] = 1;
    kv_desc.shape[2] = 2;
    kv_desc.shape[3] = 1;
    kv_desc.shape[4] = 2;
    kv_desc.strides[4] = 1;
    kv_desc.strides[3] = 2;
    kv_desc.strides[2] = 2;
    kv_desc.strides[1] = 4;
    kv_desc.strides[0] = 4;
    marmot_graph_tensor_desc_t block_table_desc{};
    init_desc_2d(block_table_desc, 1, 1, MARMOT_DTYPE_UINT32);
    marmot_graph_tensor_desc_t positions_desc{};
    init_desc_1d(positions_desc, 1, MARMOT_DTYPE_INT32);

    marmot_value_id_t hidden = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t norm_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t attn_out_weight = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t token_meta = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t kv_k = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t kv_v = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t block_table = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t positions = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &hidden_desc, &hidden) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &norm_desc, &norm_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &proj_weight_desc, &q_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &proj_weight_desc, &k_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &proj_weight_desc, &v_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &proj_weight_desc, &attn_out_weight) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &token_meta_desc, &token_meta) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &kv_desc, &kv_k) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &kv_desc, &kv_v) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &block_table_desc, &block_table) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &positions_desc, &positions) == MARMOT_SUCCESS);

    const marmot_op_signature_t rms_norm = make_sig(MARMOT_OP_RMS_NORM);
    const marmot_op_signature_t matmul = make_sig(MARMOT_OP_MATMUL);
    const marmot_op_signature_t reshape = make_sig(MARMOT_OP_RESHAPE);
    const marmot_op_signature_t rope = make_sig(MARMOT_OP_ROPE);
    const marmot_op_signature_t paged_attention = make_sig(MARMOT_OP_PAGED_ATTENTION);
    const marmot_op_signature_t add = make_sig(MARMOT_OP_ADD);

    marmot_value_id_t norm = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q_heads = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k_heads = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v_heads = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q_rope = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k_rope = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t q_tokens = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t k_tokens = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t v_tokens = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t attn_heads = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t attn_flat = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t attn_proj = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output = MARMOT_VALUE_ID_INVALID;

    const marmot_value_id_t norm_inputs[2] = {hidden, norm_weight};
    assert(marmot_graph_add_op(graph, "rms_norm", &rms_norm, norm_inputs, 2, &hidden_desc, 1, &norm) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_NORM, 0);
    const marmot_value_id_t q_inputs[2] = {norm, q_weight};
    const marmot_value_id_t k_inputs[2] = {norm, k_weight};
    const marmot_value_id_t v_inputs[2] = {norm, v_weight};
    assert(marmot_graph_add_op(graph, "q_proj", &matmul, q_inputs, 2, &qkv_desc, 1, &q) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_Q_PROJ, 0);
    assert(marmot_graph_add_op(graph, "k_proj", &matmul, k_inputs, 2, &qkv_desc, 1, &k) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_K_PROJ, 0);
    assert(marmot_graph_add_op(graph, "v_proj", &matmul, v_inputs, 2, &qkv_desc, 1, &v) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_V_PROJ, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_q_heads", &reshape, &q, 1, &q_heads_desc, 1, &q_heads) == MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_Q_HEADS, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_k_heads", &reshape, &k, 1, &q_heads_desc, 1, &k_heads) == MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_K_HEADS, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_v_heads", &reshape, &v, 1, &q_heads_desc, 1, &v_heads) == MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_V_HEADS, 0);
    const marmot_value_id_t q_rope_inputs[2] = {q_heads, positions};
    const marmot_value_id_t k_rope_inputs[2] = {k_heads, positions};
    assert(marmot_graph_add_op(graph, "q_rope", &rope, q_rope_inputs, 2, &q_heads_desc, 1, &q_rope) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_Q_ROPE, 0);
    assert(marmot_graph_add_op(graph, "k_rope", &rope, k_rope_inputs, 2, &q_heads_desc, 1, &k_rope) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_K_ROPE, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_q_tokens", &reshape, &q_rope, 1, &q_tokens_desc, 1, &q_tokens) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_Q_TOKENS, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_k_tokens", &reshape, &k_rope, 1, &q_tokens_desc, 1, &k_tokens) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_K_TOKENS, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_v_tokens", &reshape, &v_heads, 1, &q_tokens_desc, 1, &v_tokens) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_V_TOKENS, 0);
    const marmot_value_id_t attn_inputs[7] = {token_meta, q_tokens, k_tokens, v_tokens, kv_k, kv_v, block_table};
    assert(
        marmot_graph_add_op(
            graph, "paged_attention", &paged_attention, attn_inputs, 7, &q_tokens_desc, 1, &attn_heads
        ) == MARMOT_SUCCESS
    );
    graph->inner.set_last_node_paged_attention_layer(0);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_PAGED, 0);
    assert(
        marmot_graph_add_op(graph, "reshape_attn_flat", &reshape, &attn_heads, 1, &hidden_desc, 1, &attn_flat) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_OUT_RESHAPE, 0);
    const marmot_value_id_t attn_proj_inputs[2] = {attn_flat, attn_out_weight};
    assert(
        marmot_graph_add_op(graph, "attn_out", &matmul, attn_proj_inputs, 2, &hidden_desc, 1, &attn_proj) ==
        MARMOT_SUCCESS
    );
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_OUT_PROJ, 0);
    const marmot_value_id_t add_inputs[2] = {hidden, attn_proj};
    assert(marmot_graph_add_op(graph, "add", &add, add_inputs, 2, &hidden_desc, 1, &output) == MARMOT_SUCCESS);
    set_last_fast_hint(graph, MARMOT_FAST_STAGE_HINT_ATTENTION, MARMOT_FAST_NODE_ROLE_ATTN_RESIDUAL, 0);
    assert(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU) == MARMOT_SUCCESS);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 1,
        .sample_count = 1,
        .emit_logits = true,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->stages().size() == 1);
    assert(plan->stages()[0].lowering == marmot::graph::FastStageLowering::AttentionDecodePaged);

    const size_t hidden_shape[2] = {1, 4};
    const size_t norm_shape[1] = {4};
    const size_t proj_weight_shape[2] = {4, 4};
    const size_t qkv_shape[5] = {1, 1, 2, 1, 2};
    const size_t token_meta_shape[2] = {1, 4};
    const size_t block_table_shape[2] = {1, 1};
    const size_t positions_shape[1] = {1};

    marmot_tensor_t *hidden_tensor = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *norm_weight_tensor = marmot_tensor_create(ctx, norm_shape, 1, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *q_weight_tensor = marmot_tensor_create(ctx, proj_weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *k_weight_tensor = marmot_tensor_create(ctx, proj_weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *v_weight_tensor = marmot_tensor_create(ctx, proj_weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *attn_out_weight_tensor = marmot_tensor_create(ctx, proj_weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *token_meta_tensor = marmot_tensor_create(ctx, token_meta_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *kv_k_tensor = marmot_tensor_create(ctx, qkv_shape, 5, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *kv_v_tensor = marmot_tensor_create(ctx, qkv_shape, 5, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *block_table_tensor = marmot_tensor_create(ctx, block_table_shape, 2, MARMOT_DTYPE_UINT32);
    marmot_tensor_t *positions_tensor = marmot_tensor_create(ctx, positions_shape, 1, MARMOT_DTYPE_INT32);
    marmot_tensor_t *output_ref = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_fast = marmot_tensor_create(ctx, hidden_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert(hidden_tensor != nullptr);
    assert(norm_weight_tensor != nullptr);
    assert(q_weight_tensor != nullptr);
    assert(k_weight_tensor != nullptr);
    assert(v_weight_tensor != nullptr);
    assert(attn_out_weight_tensor != nullptr);
    assert(token_meta_tensor != nullptr);
    assert(kv_k_tensor != nullptr);
    assert(kv_v_tensor != nullptr);
    assert(block_table_tensor != nullptr);
    assert(positions_tensor != nullptr);
    assert(output_ref != nullptr);
    assert(output_fast != nullptr);

    const float hidden_data[] = {0.5f, -1.0f, 1.5f, 0.25f};
    const float norm_weight_data[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float q_weight_data[] = {
        1.0f, 0.0f, 0.2f, -0.1f, 0.0f, 1.0f, -0.3f, 0.4f, 0.1f, -0.2f, 1.0f, 0.0f, 0.3f, 0.1f, -0.2f, 1.0f,
    };
    const float k_weight_data[] = {
        0.8f, 0.0f, 0.1f, 0.0f, 0.1f, 0.9f, 0.0f, 0.2f, -0.2f, 0.1f, 0.7f, 0.0f, 0.0f, -0.1f, 0.3f, 1.1f,
    };
    const float v_weight_data[] = {
        0.9f, 0.1f, 0.0f, 0.0f, -0.1f, 1.0f, 0.2f, 0.0f, 0.0f, 0.1f, 0.8f, 0.1f, 0.2f, 0.0f, -0.1f, 0.9f,
    };
    const float attn_out_weight_data[] = {
        1.0f, 0.0f, 0.1f, 0.0f, 0.0f, 1.0f, 0.0f, -0.1f, 0.1f, 0.0f, 1.0f, 0.0f, 0.0f, -0.1f, 0.0f, 1.0f,
    };

    std::memcpy(hidden_tensor->data, hidden_data, sizeof(hidden_data));
    std::memcpy(norm_weight_tensor->data, norm_weight_data, sizeof(norm_weight_data));
    std::memcpy(q_weight_tensor->data, q_weight_data, sizeof(q_weight_data));
    std::memcpy(k_weight_tensor->data, k_weight_data, sizeof(k_weight_data));
    std::memcpy(v_weight_tensor->data, v_weight_data, sizeof(v_weight_data));
    std::memcpy(attn_out_weight_tensor->data, attn_out_weight_data, sizeof(attn_out_weight_data));
    std::memset(kv_k_tensor->data, 0, kv_k_tensor->capacity_bytes);
    std::memset(kv_v_tensor->data, 0, kv_v_tensor->capacity_bytes);
    marmot_uint32_t *token_meta_data = marmot_tensor_data_u32_mut(ctx, token_meta_tensor);
    marmot_uint32_t *block_table_data = marmot_tensor_data_u32_mut(ctx, block_table_tensor);
    marmot_int32_t *positions_data = marmot_tensor_data_i32_mut(ctx, positions_tensor);
    assert(token_meta_data != nullptr);
    assert(block_table_data != nullptr);
    assert(positions_data != nullptr);
    token_meta_data[0].value = 0;
    token_meta_data[1].value = 0;
    token_meta_data[2].value = 0;
    token_meta_data[3].value = 0;
    block_table_data[0].value = 0;
    positions_data[0].value = 0;

    const marmot_tensor_t *inputs[] = {
        hidden_tensor,   norm_weight_tensor,     q_weight_tensor,   k_weight_tensor,
        v_weight_tensor, attn_out_weight_tensor, token_meta_tensor, kv_k_tensor,
        kv_v_tensor,     block_table_tensor,     positions_tensor,
    };
    marmot_tensor_t *outputs_ref[] = {output_ref};
    assert(
        marmot_graph_execute(graph, ctx, inputs, std::size(inputs), outputs_ref, std::size(outputs_ref)) ==
        MARMOT_SUCCESS
    );

    marmot_tensor_t *outputs_fast[] = {output_fast};
    marmot::graph::FastExecProfile profile{};
    assert(
        marmot::graph::FastExecutor::execute(
            graph, &plan.value(), ctx, std::span<const marmot_tensor_t *const>(inputs, std::size(inputs)),
            std::span<marmot_tensor_t *const>(outputs_fast, std::size(outputs_fast)), &profile
        ) == MARMOT_SUCCESS
    );

    const float *ref_data = marmot_tensor_data_f32(ctx, output_ref);
    const float *fast_data = marmot_tensor_data_f32(ctx, output_fast);
    assert(ref_data != nullptr);
    assert(fast_data != nullptr);
    for (size_t i = 0; i < 4; ++i) {
        assert(std::fabs(ref_data[i] - fast_data[i]) < 1e-6f);
    }

    assert(profile.stages.size() == 1);
    assert(profile.stages[0].kind == marmot::graph::FastStageKind::Attention);
    assert(profile.stages[0].executed_ops == 16);

    marmot_tensor_destroy(output_fast);
    marmot_tensor_destroy(output_ref);
    marmot_tensor_destroy(positions_tensor);
    marmot_tensor_destroy(block_table_tensor);
    marmot_tensor_destroy(kv_v_tensor);
    marmot_tensor_destroy(kv_k_tensor);
    marmot_tensor_destroy(token_meta_tensor);
    marmot_tensor_destroy(attn_out_weight_tensor);
    marmot_tensor_destroy(v_weight_tensor);
    marmot_tensor_destroy(k_weight_tensor);
    marmot_tensor_destroy(q_weight_tensor);
    marmot_tensor_destroy(norm_weight_tensor);
    marmot_tensor_destroy(hidden_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
    std::printf("  ok fast_executor_attention_decode_cpu\n");
}

void test_fast_executor_profile_cpu() {
    marmot_context_t *ctx = marmot_init(MARMOT_BACKEND_CPU);
    assert(ctx != nullptr);

    marmot_graph_t *graph = marmot_graph_create();
    assert(graph != nullptr);

    marmot_graph_tensor_desc_t desc{};
    init_desc_2d(desc, 2, 4, MARMOT_DTYPE_FLOAT32);
    marmot_graph_tensor_desc_t weight_desc{};
    init_desc_2d(weight_desc, 4, 4, MARMOT_DTYPE_FLOAT32);

    marmot_value_id_t input = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t weight = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_input(graph, &desc, &input) == MARMOT_SUCCESS);
    assert(marmot_graph_add_input(graph, &weight_desc, &weight) == MARMOT_SUCCESS);

    const marmot_op_signature_t relu = make_sig(MARMOT_OP_RELU);
    const marmot_op_signature_t matmul = make_sig(MARMOT_OP_MATMUL);

    marmot_value_id_t hidden = MARMOT_VALUE_ID_INVALID;
    marmot_value_id_t output = MARMOT_VALUE_ID_INVALID;
    assert(marmot_graph_add_op(graph, "relu", &relu, &input, 1, &desc, 1, &hidden) == MARMOT_SUCCESS);
    const marmot_value_id_t matmul_inputs[2] = {hidden, weight};
    assert(marmot_graph_add_op(graph, "matmul", &matmul, matmul_inputs, 2, &desc, 1, &output) == MARMOT_SUCCESS);
    assert(marmot_graph_finalize(graph, MARMOT_BACKEND_CPU) == MARMOT_SUCCESS);

    const marmot::graph::FastPlanBucket bucket{
        .token_count = 1,
        .sample_count = 1,
        .emit_logits = true,
    };
    auto plan = marmot::graph::compile_fast_plan(graph, bucket);
    assert(plan.has_value());
    assert(plan->stages().size() == 2);
    assert(plan->stages()[1].lowering == marmot::graph::FastStageLowering::LogitsMatmul);

    size_t input_shape[2] = {2, 4};
    size_t weight_shape[2] = {4, 4};
    marmot_tensor_t *input_tensor = marmot_tensor_create(ctx, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_ref = marmot_tensor_create(ctx, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    marmot_tensor_t *output_fast = marmot_tensor_create(ctx, input_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert(input_tensor != nullptr);
    assert(output_ref != nullptr);
    assert(output_fast != nullptr);

    const float input_vals[8] = {-1.0f, 0.5f, 2.0f, -0.25f, 0.0f, 1.0f, 3.0f, 2.5f};
    const float weight_vals[16] = {
        1.0f, 0.0f, 0.5f, -0.5f, 0.25f, 1.0f, -0.25f, 0.75f, -0.5f, 0.25f, 1.0f, 0.0f, 0.5f, -0.75f, 0.25f, 1.0f,
    };
    marmot_tensor_t *weight_tensor = marmot_tensor_create(ctx, weight_shape, 2, MARMOT_DTYPE_FLOAT32);
    assert(weight_tensor != nullptr);
    std::memcpy(input_tensor->data, input_vals, sizeof(input_vals));
    std::memcpy(weight_tensor->data, weight_vals, sizeof(weight_vals));

    const marmot_tensor_t *inputs[] = {input_tensor, weight_tensor};
    marmot_tensor_t *outputs_ref[] = {output_ref};
    assert(
        marmot_graph_execute(graph, ctx, inputs, std::size(inputs), outputs_ref, std::size(outputs_ref)) ==
        MARMOT_SUCCESS
    );

    marmot_tensor_t *outputs_fast[] = {output_fast};
    marmot::graph::FastExecProfile profile{};
    assert(
        marmot::graph::FastExecutor::execute(
            graph, &plan.value(), ctx, std::span<const marmot_tensor_t *const>(inputs, std::size(inputs)),
            std::span<marmot_tensor_t *const>(outputs_fast, std::size(outputs_fast)), &profile
        ) == MARMOT_SUCCESS
    );

    const float *ref_data = marmot_tensor_data_f32(ctx, output_ref);
    const float *fast_data = marmot_tensor_data_f32(ctx, output_fast);
    assert(ref_data != nullptr);
    assert(fast_data != nullptr);
    for (size_t i = 0; i < 8; ++i) {
        assert(std::fabs(ref_data[i] - fast_data[i]) < 1e-6f);
    }

    assert(profile.backend == MARMOT_BACKEND_CPU);
    assert(profile.phase == marmot::graph::FastPlanPhase::Decode);
    assert(profile.stages.size() == 2);
    assert(profile.stages[0].kind == marmot::graph::FastStageKind::GenericFallback);
    assert(profile.stages[0].executed_ops == 1);
    assert(profile.stages[1].kind == marmot::graph::FastStageKind::LogitsTail);
    assert(profile.stages[1].executed_ops == 1);
    const uint64_t stage_total = profile.stages[0].duration_ns + profile.stages[1].duration_ns;
    assert(profile.total_ns >= stage_total);

    marmot_tensor_destroy(output_fast);
    marmot_tensor_destroy(output_ref);
    marmot_tensor_destroy(weight_tensor);
    marmot_tensor_destroy(input_tensor);
    marmot_graph_destroy(graph);
    marmot_destroy(ctx);
    std::printf("  ok fast_executor_profile_cpu\n");
}

} // namespace

int main() {
    std::printf("Fast plan tests:\n");
    test_compile_fast_plan_generic();
    test_compile_fast_plan_boundaries();
    test_compile_fast_plan_logits_tail_with_norm();
    test_compile_fast_plan_dense_ffn();
    test_compile_fast_plan_moe_ffn();
    test_compile_fast_plan_attention_decode();
    test_fast_executor_moe_ffn_cpu();
    test_fast_executor_attention_decode_cpu();
    test_fast_executor_profile_cpu();
    std::printf("\nAll fast plan tests passed!\n");
    return 0;
}
