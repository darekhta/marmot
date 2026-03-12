#ifndef MARMOT_BENCH_WORKLOADS_H
#define MARMOT_BENCH_WORKLOADS_H

#include "bench_core.h"

#ifdef __cplusplus
extern "C" {
#endif

marmot_bench_suite_t *marmot_bench_create_full_suite(void);

void marmot_bench_register_matmul_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_matmul_quant_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_moe_expert_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_logits_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_elementwise_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_fusion_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_reduction_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_rope_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_ffn_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_moe_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_layer_workloads(marmot_bench_suite_t *suite);
void marmot_bench_register_tokenizer_workloads(marmot_bench_suite_t *suite);

#ifdef __cplusplus
}
#endif

#endif
