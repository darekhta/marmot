#include "bench_workloads.h"

marmot_bench_suite_t *marmot_bench_create_full_suite(void) {
    marmot_bench_suite_t *suite = marmot_bench_suite_create("marmot-bench");
    if (suite == nullptr)
        return nullptr;

    marmot_bench_register_matmul_workloads(suite);
    marmot_bench_register_matmul_quant_workloads(suite);
    marmot_bench_register_moe_expert_workloads(suite);
    marmot_bench_register_logits_workloads(suite);
    marmot_bench_register_elementwise_workloads(suite);
    marmot_bench_register_fusion_workloads(suite);
    marmot_bench_register_reduction_workloads(suite);
    marmot_bench_register_rope_workloads(suite);
    marmot_bench_register_ffn_workloads(suite);
    marmot_bench_register_moe_workloads(suite);
    marmot_bench_register_layer_workloads(suite);
    marmot_bench_register_tokenizer_workloads(suite);

    return suite;
}
