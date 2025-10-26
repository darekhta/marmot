bench_sources = files(
    'main.c',
    'bench_core.c',
    'bench_stats.c',
    'bench_workloads.c',
    'bench_param_sweep.c',
    'bench_model.c',
    'bench_llm.c',
    'bench_grid.c',
    'bench_output_console.c',
    'bench_output_csv.c',
    'bench_output_json.c',
    'bench_output_markdown.c',
    'bench_output_sql.c',
    'bench_output_jsonl.c',
    'workloads/micro_matmul.c',
    'workloads/micro_matmul_quant.c',
    'workloads/micro_elementwise.c',
    'workloads/micro_fusion.c',
    'workloads/micro_reduction.c',
    'workloads/micro_rope.c',
    'workloads/micro_tokenizer.c',
    'workloads/composite_ffn.c',
    'workloads/composite_layer.c',
)

marmot_bench = executable(
    'marmot-bench',
    bench_sources,
    include_directories: [
        include_directories('.'),
        include_directories('../../include'),
        include_directories('../../src'),
    ],
    c_args: c_args,
    link_with: libmarmot,
    dependencies: deps,
    install: true,
)

marmot_bench_tile = executable(
    'marmot-bench-tile',
    files('workloads/micro_matmul_quant_tile.c'),
    include_directories: [
        include_directories('.'),
        include_directories('../../include'),
        include_directories('../../src'),
    ],
    c_args: c_args,
    link_with: libmarmot,
    dependencies: deps,
    install: false,
)
