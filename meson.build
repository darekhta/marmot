project('marmot', ['c', 'cpp'],
  version: '0.1.0',
  meson_version: '>=1.9.0',
  default_options: [
    'c_std=c2x',
    'cpp_std=c++23',
    'warning_level=3',
    'buildtype=debugoptimized',
  ],
)

cc = meson.get_compiler('c')

host_system = host_machine.system()
host_cpu = host_machine.cpu_family()
is_darwin = host_system == 'darwin'
is_arm = host_cpu in ['aarch64', 'arm']
is_x86 = host_cpu in ['x86_64', 'x86']

enable_simd = get_option('enable_simd')
enable_apple_accelerate_opt = get_option('enable_apple_accelerate')

simd_enabled = enable_simd
neon_enabled = simd_enabled and is_arm
avx2_enabled = simd_enabled and is_x86

if enable_apple_accelerate_opt and not is_darwin
  warning('enable_apple_accelerate requested on non-macOS host; disabling')
endif

enable_apple_accelerate_actual = enable_apple_accelerate_opt and is_darwin and simd_enabled

if enable_apple_accelerate_opt and is_darwin and not simd_enabled
  warning('Apple Accelerate requires SIMD support, disabling (enable_simd=false)')
endif

c_args = [
  '-Wall',
  '-Wextra',
  '-Wpedantic',
]

if cc.get_id() == 'clang' or cc.get_id() == 'gcc'
  c_args += ['-Wno-unused-parameter']
endif

# Objective-C++ args for Metal backend
objcpp_args = [
  '-std=c++23',
]

deps = []
accelerate_dep = dependency('', required: false)
enable_metal = false
enable_metal_opt = get_option('enable_metal')

# SIMD configuration and defines
if neon_enabled
  c_args += ['-DMARMOT_ENABLE_NEON=1']
  objcpp_args += ['-DMARMOT_ENABLE_NEON=1']
  message('ARM NEON: enabled')

  # ARM BF16 support detection
  bf16_test = '''
  #include <arm_neon.h>
  int main(void) {
    bfloat16x8_t a, b;
    float32x4_t c = vbfdotq_f32(vdupq_n_f32(0), a, b);
    return 0;
  }
  '''
  # Use full feature set including fp16, bf16, dotprod to avoid conflicts with other code
  bf16_args = ['-march=armv8.2-a+fp16+bf16+dotprod']
  if cc.compiles(bf16_test, args: bf16_args)
    c_args += bf16_args
    objcpp_args += bf16_args
    message('ARM BF16: enabled (vbfdotq_f32)')
  else
    message('ARM BF16: disabled (compiler does not support)')
  endif
else
  c_args += ['-DMARMOT_ENABLE_NEON=0']
  objcpp_args += ['-DMARMOT_ENABLE_NEON=0']
  if is_arm
    message('ARM NEON: disabled (scalar build)')
  endif
endif

if avx2_enabled
  c_args += ['-mavx2', '-mfma', '-mf16c', '-DMARMOT_ENABLE_AVX2=1']
  objcpp_args += ['-mavx2', '-mfma', '-mf16c', '-DMARMOT_ENABLE_AVX2=1']
  message('x86 AVX2/F16C: enabled')
else
  c_args += ['-DMARMOT_ENABLE_AVX2=0']
  objcpp_args += ['-DMARMOT_ENABLE_AVX2=0']
  if is_x86
    message('x86 AVX2/F16C: disabled (scalar build)')
  endif
endif

# x86 AVX512
avx512_enabled = simd_enabled and is_x86
if avx512_enabled and cc.has_argument('-mavx512f')
  c_args += ['-mavx512f', '-DMARMOT_ENABLE_AVX512=1']
  objcpp_args += ['-mavx512f', '-DMARMOT_ENABLE_AVX512=1']
  message('x86 AVX512: enabled')
else
  c_args += ['-DMARMOT_ENABLE_AVX512=0']
  objcpp_args += ['-DMARMOT_ENABLE_AVX512=0']
  if is_x86 and simd_enabled
    message('x86 AVX512: disabled (not supported or scalar build)')
  endif
endif

# Common feature defines
c_args += ['-DMARMOT_ENABLE_SCALAR=1']
objcpp_args += ['-DMARMOT_ENABLE_SCALAR=1']
c_args += ['-DMARMOT_ENABLE_FP8=1']
objcpp_args += ['-DMARMOT_ENABLE_FP8=1']

# Accelerate framework configuration
if enable_apple_accelerate_actual
  c_args += ['-DMARMOT_ENABLE_ACCELERATE=1', '-DACCELERATE_NEW_LAPACK']
  objcpp_args += ['-DMARMOT_ENABLE_ACCELERATE=1']
  accelerate_dep = dependency('Accelerate', required: true)
  deps += [accelerate_dep]
  message('Accelerate framework: enabled')
else
  c_args += ['-DMARMOT_ENABLE_ACCELERATE=0']
  objcpp_args += ['-DMARMOT_ENABLE_ACCELERATE=0']
  if enable_apple_accelerate_opt and not simd_enabled
    message('Accelerate framework disabled because enable_simd=false')
  elif enable_apple_accelerate_opt and not is_darwin
    message('Accelerate framework disabled (non-macOS)')
  else
    message('Accelerate framework: disabled')
  endif
endif

# libdispatch (Grand Central Dispatch) for thread pool management
if is_darwin
  c_args += ['-DMARMOT_ENABLE_LIBDISPATCH=1', '-fblocks']
  objcpp_args += ['-DMARMOT_ENABLE_LIBDISPATCH=1', '-fblocks']
  message('libdispatch: enabled (system)')
else
  dispatch_dep = dependency('dispatch', required: false)
  blocksruntime_dep = cc.find_library('BlocksRuntime', required: false)
  if dispatch_dep.found() and blocksruntime_dep.found()
    deps += [dispatch_dep, blocksruntime_dep]
    c_args += ['-DMARMOT_ENABLE_LIBDISPATCH=1', '-fblocks']
    message('libdispatch: enabled (Linux)')
  else
    c_args += ['-DMARMOT_ENABLE_LIBDISPATCH=0']
    message('libdispatch: disabled')
  endif
endif

py = import('python').find_installation('python3', required: true)
fs = import('fs')
uv_prog = find_program('uv', required: true)
uv_python = [uv_prog, 'run', 'python3']

yyjson_inc = include_directories('third_party/yyjson')
inc_dirs = include_directories('include', 'src', 'src/graph', 'src/backends/cpu', 'src/backends/metal', 'third_party/yyjson')
root_inc = include_directories('.')
all_inc_dirs = [inc_dirs, root_inc]

# Suppress noisy linker warnings on macOS (duplicate -lc++ and missing Xcode SDK framework paths)
if host_machine.system() == 'darwin'
  link_silence = ['-Wl,-no_warn_duplicate_libraries']
  sdk_path_cmd = run_command('xcrun', '--show-sdk-path', check: false)
  if sdk_path_cmd.returncode() == 0
    sdk_path = sdk_path_cmd.stdout().strip()
    link_silence += ['-F' + sdk_path + '/System/Library/Frameworks']
  endif
  add_project_link_arguments(link_silence, language: ['c', 'cpp', 'objcpp'])
endif

traits_template_dir = meson.current_source_dir() / 'scripts/codegen/templates'

# ops_schema codegen - generates ops_schema_gen.py from ops.def
ops_schema_codegen = custom_target(
  'ops_schema_codegen',
  output: ['ops_schema_codegen.stamp'],
  command: [
    uv_prog, 'run', 'python3',
    meson.current_source_dir() / 'scripts/codegen/gen_ops_schema.py',
    '--ops-def', meson.current_source_dir() / 'src/core/defs/ops.def',
    '--metadata-def', meson.current_source_dir() / 'src/core/defs/ops_metadata.def',
    '--output', meson.current_source_dir() / 'scripts/codegen/ops_schema_gen.py',
    '--stamp-output', '@OUTPUT@',
  ],
  depend_files: [
    meson.current_source_dir() / 'scripts/codegen/gen_ops_schema.py',
    meson.current_source_dir() / 'scripts/codegen/def_parser.py',
    meson.current_source_dir() / 'scripts/codegen/codegen_base.py',
    meson.current_source_dir() / 'src/core/defs/ops.def',
    meson.current_source_dir() / 'src/core/defs/ops_metadata.def',
  ],
  build_by_default: true,
)

# Dispatch args codegen - generates kernel_dispatch_args.h from ops_schema.py
dispatch_args_codegen = custom_target(
  'dispatch_args_codegen',
  output: ['dispatch_args_codegen.stamp'],
  depends: [ops_schema_codegen],
  command: [
    uv_prog, 'run', 'python3',
    meson.current_source_dir() / 'scripts/codegen/gen_dispatch_args.py',
    '--output', meson.current_source_dir() / 'src/graph/kernel_dispatch_args.gen.h',
    '--template-dir', traits_template_dir,
    '--ops-def', meson.current_source_dir() / 'src/core/defs/ops.def',
    '--metadata-def', meson.current_source_dir() / 'src/core/defs/ops_metadata.def',
    '--stamp-output', '@OUTPUT@',
  ],
  depend_files: [
    meson.current_source_dir() / 'scripts/codegen/gen_dispatch_args.py',
    meson.current_source_dir() / 'scripts/codegen/ops_schema.py',
    meson.current_source_dir() / 'scripts/codegen/def_parser.py',
    meson.current_source_dir() / 'scripts/codegen/codegen_base.py',
    meson.current_source_dir() / 'src/core/defs/ops.def',
    meson.current_source_dir() / 'src/core/defs/ops_metadata.def',
    traits_template_dir / 'kernel_dispatch_args.h.j2',
  ],
  build_by_default: true,
)

# API dispatch codegen - generates per-module API wrappers and op metadata.
api_dispatch_codegen = custom_target(
  'api_dispatch_codegen',
  output: ['api_dispatch_codegen.stamp'],
  command: [
    uv_prog, 'run', 'python3',
    meson.current_source_dir() / 'scripts/codegen/gen_api_dispatch.py',
    '--ops-def', meson.current_source_dir() / 'src/core/defs/ops.def',
    '--metadata-def', meson.current_source_dir() / 'src/core/defs/ops_metadata.def',
    '--output-reductions-inc', meson.current_source_dir() / 'src/core/defs/reductions_api.gen.inc',
    '--output-unary-inc', meson.current_source_dir() / 'src/core/defs/unary_api.gen.inc',
    '--output-elementwise-inc', meson.current_source_dir() / 'src/core/defs/elementwise_api.gen.inc',
    '--output-conversion-inc', meson.current_source_dir() / 'src/core/defs/conversion_api.gen.inc',
    '--output-quantization-inc', meson.current_source_dir() / 'src/core/defs/quantization_api.gen.inc',
    '--output-normalization-inc', meson.current_source_dir() / 'src/core/defs/normalization_api.gen.inc',
    '--output-embedding-inc', meson.current_source_dir() / 'src/core/defs/embedding_api.gen.inc',
    '--output-tensor-ops-inc', meson.current_source_dir() / 'src/core/defs/tensor_ops_api.gen.inc',
    '--output-rope-inc', meson.current_source_dir() / 'src/core/defs/rope_api.gen.inc',
    '--output-matmul-inc', meson.current_source_dir() / 'src/core/defs/matmul_api.gen.inc',
    '--output-attention-inc', meson.current_source_dir() / 'src/core/defs/attention_api.gen.inc',
    '--output-metadata', meson.current_source_dir() / 'include/marmot/op_metadata.gen.h',
    '--output-docs', meson.current_source_dir() / 'docs/API_OPS.gen.md',
    '--template-dir', traits_template_dir,
    '--stamp-output', '@OUTPUT@',
  ],
  depend_files: [
    meson.current_source_dir() / 'scripts/codegen/gen_api_dispatch.py',
    meson.current_source_dir() / 'scripts/codegen/def_parser.py',
    meson.current_source_dir() / 'scripts/codegen/codegen_base.py',
    meson.current_source_dir() / 'src/core/defs/ops.def',
    meson.current_source_dir() / 'src/core/defs/ops_metadata.def',
    traits_template_dir / 'api_dispatch.inc.j2',
    traits_template_dir / 'op_metadata.h.j2',
    traits_template_dir / 'api_ops_md.j2',
  ],
  build_by_default: true,
)

# Multi-backend kernel codegen - generates traits header, per-backend dispatch/query,
# and Metal quantized matmul dispatch. All codegen consolidated into gen_kernels.py.
kernel_codegen = custom_target(
  'kernel_codegen',
  output: ['kernel_codegen.stamp'],
  command: [
    uv_prog, 'run', 'python3',
    meson.current_source_dir() / 'scripts/codegen/gen_kernels.py',
    '--backends', 'cpu', 'metal',
    '--backend-dirs',
      meson.current_source_dir() / 'src/backends/cpu/kernels',
      meson.current_source_dir() / 'src/backends/metal/kernels',
    '--traits-header', meson.current_source_dir() / 'include/marmot/traits_ids.gen.h',
    '--query-sources',
      meson.current_source_dir() / 'src/backends/cpu/dispatch/cpu_kernel_query.gen.c',
      meson.current_source_dir() / 'src/backends/metal/internal/metal_kernel_query.gen.mm',
    '--dispatch-sources',
      meson.current_source_dir() / 'src/backends/metal/ops/metal_kernel_dispatch.gen.mm',
    '--dispatch-dirs',
      'cpu:' + meson.current_source_dir() / 'src/backends/cpu/dispatch',
    '--op-signature-hash-header', meson.current_source_dir() / 'include/marmot/op_signature_hash.gen.h',
    '--bytecode-table-headers',
      meson.current_source_dir() / 'src/backends/cpu/dispatch/bytecode_tables_cpu.gen.h',
      meson.current_source_dir() / 'src/backends/metal/ops/bytecode_tables_metal.gen.h',
    '--bytecode-table-sources',
      meson.current_source_dir() / 'src/backends/cpu/dispatch/bytecode_tables_cpu.gen.c',
      meson.current_source_dir() / 'src/backends/metal/ops/bytecode_tables_metal.gen.mm',
    '--bytecode-exec-headers',
      meson.current_source_dir() / 'src/backends/cpu/dispatch/bytecode_exec_cpu.gen.h',
      meson.current_source_dir() / 'src/backends/metal/ops/bytecode_exec_metal.gen.h',
    '--bytecode-exec-sources',
      meson.current_source_dir() / 'src/backends/cpu/dispatch/bytecode_exec_cpu.gen.c',
      meson.current_source_dir() / 'src/backends/metal/ops/bytecode_exec_metal.gen.mm',
    '--quant-scheme-def', meson.current_source_dir() / 'src/backends/metal/kernels/metadata/quant_schemes.def',
    '--metal-quant-dispatch-output', meson.current_source_dir() / 'src/backends/metal/ops/metal_matmul_quant_dispatch.gen.mm',
    '--metal-unary-tables-header', meson.current_source_dir() / 'src/backends/metal/ops/metal_unary_tables.gen.h',
    '--template-dir', traits_template_dir,
    '--stamp-output', '@OUTPUT@',
  ],
  depend_files: [
    meson.current_source_dir() / 'scripts/codegen/gen_kernels.py',
    meson.current_source_dir() / 'scripts/codegen/def_parser.py',
    meson.current_source_dir() / 'scripts/codegen/codegen_base.py',
    meson.current_source_dir() / 'scripts/codegen/backend_config.py',
    meson.current_source_dir() / 'src/backends/cpu/kernels/metadata/common.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/metadata/quant_schemes.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/conversions.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/elementwise.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/embedding.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/matmul_accelerate.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/matmul_avx2.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/matmul_neon.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/matmul_qkv.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/matmul_quantized_scalar.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/matmul_scalar.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/normalization.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/paged_attention.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/quantization.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/reductions.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/rope.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/softmax.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/tensor_ops.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/ternary.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/unary.def',
    meson.current_source_dir() / 'src/backends/cpu/kernels/vec_dot.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/metadata/common.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/metadata/quant_schemes.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/activations.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/conversions.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/elementwise.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/embedding.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/matmul.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/matmul_qkv.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/matmul_quantized.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/normalization.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/paged_attention.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/quantization.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/reductions.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/rope.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/softmax.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/tensor_ops.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/ternary.def',
    meson.current_source_dir() / 'src/backends/metal/kernels/vec_dot.def',
    traits_template_dir / 'traits_ids.h.j2',
    traits_template_dir / 'backend_query.c.j2',
    traits_template_dir / 'backend_dispatch.j2',
    traits_template_dir / 'bytecode_tables.h.j2',
    traits_template_dir / 'bytecode_tables.c.j2',
    traits_template_dir / 'bytecode_exec.h.j2',
    traits_template_dir / 'bytecode_exec.c.j2',
    traits_template_dir / '_macros.j2',
    traits_template_dir / 'matmul_dispatch_cpu.c.j2',
    traits_template_dir / 'elementwise_dispatch_cpu.c.j2',
    traits_template_dir / 'reduction_dispatch_cpu.c.j2',
    traits_template_dir / 'neural_dispatch_cpu.c.j2',
    traits_template_dir / 'misc_dispatch_cpu.c.j2',
    traits_template_dir / 'metal_unary_tables.gen.h.j2',
    traits_template_dir / 'op_signature_hash.h.j2',
    traits_template_dir / 'metal_quant_dispatch.gen.mm.j2',
  ],
  build_by_default: true,
)

core_sources = files(
  'src/core/error.c',
  'src/core/bytecode/bytecode.c',
  'src/core/bytecode/bytecode_compile.c',
  'src/core/helpers/calibration.c',
  'src/core/helpers/elementwise.c',
  'src/core/helpers/quant.c',
  'src/core/helpers/norm.c',
  'src/core/helpers/embedding.c',
  'src/core/ops_utils.c',
  'src/core/tensor/metadata.c',
  'src/core/tensor/create.c',
  'src/core/tensor/copy.c',
  'src/core/tensor/transfer.c',
  'src/core/tensor/host_io.c',
  'src/core/tensor/data_access.c',
  'src/core/tensor/fill.c',
  'src/core/tensor/descriptor.c',
  'src/core/helpers/dtype.c',
  'src/core/context/context.c',
  'src/core/context/routing.c',
  'src/core/dispatch/dispatch.c',
  'src/core/dispatch/dispatch_execute.c',
  'src/core/dispatch/signature_utils.c',
  'src/core/dispatch/signature_infer.c',
  'src/core/dispatch/elementwise_dispatch.c',
  'src/core/dispatch/normalization_dispatch.c',
  'src/core/dispatch/conversion_dispatch.c',
  'src/core/dispatch/quantization_dispatch.c',
  'src/core/dispatch/kernel_query.c',
  'src/core/dispatch/fusion_detection.c',
  'src/core/dispatch/quantization_build.c',
  'src/core/dispatch/rope_dispatch.c',
  'src/core/dispatch/attention_build.c',
  'src/core/dispatch/matmul_build.c',
  'src/core/dispatch/embedding_build.c',
  'src/core/dispatch/tensor_ops_build.c',
  'src/core/ops/embedding_impl.c',
  'src/core/ops/attention_impl.c',
  'src/core/ops/matmul_impl.c',
  'src/core/tensor/tensor_utils.c',
  'src/core/api/elementwise.c',
  'src/core/api/unary.c',
  'src/core/api/reductions.c',
  'src/core/api/quantization.c',
  'src/core/api/conversion.c',
  'src/core/api/normalization.c',
  'src/core/api/embedding.c',
  'src/core/api/tensor_ops.c',
  'src/core/api/rope.c',
  'src/core/api/matmul.c',
  'src/core/api/attention.c',
  'src/inference/model/model.cpp',
  'src/inference/frontends/serving_engine.cpp',
  'src/core/memory/allocator_registry.c',
  'src/core/memory/allocator_usage.c',
  'src/core/helpers/int4.c',
  'src/core/helpers/bitpack.c',
  'src/inference/common/storage.cpp',
  'src/inference/kv_pool/kv_pool.cpp',
  'src/core/helpers/quant_registry.c',
  'src/core/helpers/rope.c',
  'third_party/yyjson/yyjson.c',
  'src/graph/graph.cpp',
  'src/graph/passes/fusion_pass.cpp',
  'src/graph/graph_executor.cpp',
  'src/graph/graph_validator.cpp',
  'src/graph/graph_signature.cpp',
  'src/graph/graph_naming.cpp',
  'src/graph/binding_table.cpp',
  'src/graph/execution_session.cpp',
  'src/graph/kernel_query.cpp',
  'src/graph/gguf/loader.cpp',
  'src/graph/gguf/model.cpp',
  'src/graph/gguf/model_c_api.cpp',
  'src/graph/gguf/graph_builder.cpp',
  'src/graph/gguf/architecture.cpp',
  'src/graph/gguf/error.cpp',
  'src/graph/gguf/graph_loader.cpp',
  'src/graph/gguf/gguf_reader.cpp',
  'src/graph/gguf/gguf_validator.cpp',
  'src/graph/gguf/tensor_resolver.cpp',
  'src/graph/gguf/op_registry.cpp',
  'src/graph/graph_api_impl.cpp',
  'src/tokenizer/utf8.cpp',
  'src/tokenizer/vocab.cpp',
  'src/tokenizer/model_bpe.cpp',
  'src/tokenizer/model_unigram.cpp',
  'src/tokenizer/model_wordpiece.cpp',
  'src/tokenizer/tokenizer.cpp',
  'src/api/error_capture.c',
  'src/api/inference_model_api.cpp',
  'src/api/llm_api.cpp',
  'src/api/serving_engine_api.cpp',
  'src/api/kv_pool_api.cpp',
  'src/api/gguf_loader.cpp',
  'src/api/graph_api.cpp',
  'src/api/tokenizer_api.cpp',
  'src/backends/cpu/dispatch/cpu_kernel_query.gen.c',
  'src/backends/cpu/dispatch/matmul_dispatch_cpu.gen.c',
  'src/backends/cpu/dispatch/elementwise_dispatch_cpu.gen.c',
  'src/backends/cpu/dispatch/reduction_dispatch_cpu.gen.c',
  'src/backends/cpu/dispatch/neural_dispatch_cpu.gen.c',
  'src/backends/cpu/dispatch/misc_dispatch_cpu.gen.c',
)

utils_sources = files(
  'src/utils/dtype_ref.c',
)

cpu_backend_sources = files(
  'src/backends/cpu/runtime/runtime.c',
  'src/backends/cpu/runtime/dispatch.c',
  'src/backends/cpu/dispatch/device_ops.c',
  'src/backends/cpu/dispatch/bytecode_tables_cpu.gen.c',
  'src/backends/cpu/dispatch/bytecode_exec_cpu.gen.c',
  'src/backends/cpu/cpu_backend_helpers.c',
  'src/backends/cpu/cpu_caps.c',
  'src/backends/cpu/cpu_memory.c',
  'src/backends/cpu/memory/allocator.c',
  'src/backends/cpu/ops/tensor_ops/tensor_ops.c',
  'src/backends/cpu/ops/paged_attention/paged_attention_ref.c',
  'src/backends/cpu/ops/paged_attention/scalar/paged_attention_scalar.c',
  'src/backends/cpu/ops/paged_attention/neon/paged_attention_neon.c',
  'src/backends/cpu/ops/paged_attention/avx2/paged_attention_avx2.c',
  'src/backends/cpu/ops/convert/dtype_convert.c',
  'src/backends/cpu/ops/convert/convert_registry.c',
  'src/backends/cpu/quantization/common/block.c',
  'src/backends/cpu/quantization/common/quant_utils.c',
  'src/backends/cpu/quantization/formats/q4_gguf.c',
  'src/backends/cpu/quantization/formats/q5_gguf.c',
  'src/backends/cpu/quantization/formats/q8_gguf.c',
  'src/backends/cpu/quantization/ops/quantize_core.c',
  'src/backends/cpu/quantization/ops/quant_dispatch.c',
  'src/backends/cpu/quantization/formats/q2k_gguf.c',
  'src/backends/cpu/quantization/formats/q3k_gguf.c',
  'src/backends/cpu/quantization/formats/q4k_gguf.c',
  'src/backends/cpu/quantization/formats/q5k_gguf.c',
  'src/backends/cpu/quantization/formats/q6k_gguf.c',
  'src/backends/cpu/quantization/formats/q8k_gguf.c',
  'src/backends/cpu/ops/embedding/embedding_dispatch.c',
  'src/backends/cpu/ops/embedding/scalar/embedding_scalar.c',
  'src/backends/cpu/ops/embedding/neon/embedding_neon.c',
  'src/backends/cpu/ops/embedding/avx2/embedding_avx2.c',
  'src/backends/cpu/ops/vec_dot/vec_dot.c',
  'src/backends/cpu/ops/elementwise/accelerate/elementwise_accelerate.c',
  'src/backends/cpu/ops/elementwise/avx2/elementwise_f32_avx2.c',
  'src/backends/cpu/ops/elementwise/avx2/elementwise_f64_avx2.c',
  'src/backends/cpu/ops/elementwise/avx2/elementwise_i32_avx2.c',
  'src/backends/cpu/ops/elementwise/avx2/elementwise_i64_avx2.c',
  'src/backends/cpu/ops/elementwise/neon/elementwise_f32_neon.c',
  'src/backends/cpu/ops/elementwise/neon/elementwise_f64_neon.c',
  'src/backends/cpu/ops/elementwise/neon/elementwise_i32_neon.c',
  'src/backends/cpu/ops/elementwise/neon/elementwise_i64_neon.c',
  'src/backends/cpu/ops/elementwise/scalar/elementwise_f32_scalar.c',
  'src/backends/cpu/ops/elementwise/scalar/elementwise_f64_scalar.c',
  'src/backends/cpu/ops/elementwise/scalar/elementwise_via_f32_scalar.c',
  'src/backends/cpu/ops/elementwise/scalar/elementwise_int_scalar.c',
  'src/backends/cpu/ops/elementwise/scalar/elementwise_ternary_scalar.c',
  'src/core/helpers/matmul.c',
  'src/backends/cpu/ops/unary/unary_dispatch.c',
  'src/backends/cpu/ops/unary/neon/unary_neon.c',
  'src/backends/cpu/ops/unary/avx2/unary_avx2.c',
  'src/backends/cpu/ops/unary/accelerate/unary_accelerate.c',
  'src/backends/cpu/ops/unary/scalar/unary_scalar.c',
  'src/backends/cpu/ops/normalization/normalization_dispatch.c',
  'src/backends/cpu/ops/normalization/scalar/normalization_scalar.c',
  'src/backends/cpu/ops/normalization/neon/normalization_neon.c',
  'src/backends/cpu/ops/normalization/avx2/normalization_avx2.c',
  'src/backends/cpu/ops/softmax/softmax_dispatch.c',
  'src/backends/cpu/ops/softmax/scalar/softmax_f32_scalar.c',
  'src/backends/cpu/ops/softmax/scalar/softmax_f64_scalar.c',
  'src/backends/cpu/ops/softmax/scalar/softmax_misc_scalar.c',
  'src/backends/cpu/ops/softmax/neon/softmax_f32_neon.c',
  'src/backends/cpu/ops/softmax/neon/softmax_f16_neon.c',
  'src/backends/cpu/ops/softmax/avx2/softmax_f32_avx2.c',
  'src/backends/cpu/ops/matmul/matmul_epilogue.c',
  'src/backends/cpu/ops/matmul/qkv/matmul_qkv.c',
  'src/backends/cpu/ops/matmul/qkv/matmul_qkv_neon.c',
  'src/backends/cpu/ops/matmul/qkv/matmul_qkv_avx2.c',
  'src/backends/cpu/ops/matmul/qkv/matmul_qkv_scalar.c',
  'src/backends/cpu/ops/matmul/scalar/matmul_scalar.c',
  'src/backends/cpu/ops/matmul/neon/matmul_neon.c',
  'src/backends/cpu/ops/matmul/neon/neon_matmul_scratch.c',
  'src/backends/cpu/ops/matmul/avx2/matmul_avx2.c',
  'src/backends/cpu/ops/matmul/accelerate/matmul_accelerate.c',
  'src/backends/cpu/ops/matmul/quantized/scalar/vec_dot_scalar.c',
  'src/backends/cpu/ops/matmul/quantized/scalar/matmul_quant_scalar.c',
  'src/backends/cpu/ops/matmul/quantized/neon/vec_dot_neon.c',
  'src/backends/cpu/ops/matmul/quantized/neon/vec_dot_dotprod.c',
  'src/backends/cpu/ops/matmul/quantized/neon/matmul_quant_neon.c',
  'src/backends/cpu/ops/matmul/quantized/avx2/vec_dot_avx2.c',
  'src/backends/cpu/ops/matmul/quantized/avx2/matmul_quant_avx2.c',
  'src/backends/cpu/ops/matmul/quantized/matmul_quant_compute.c',
  'src/backends/cpu/ops/matmul/quantized/matmul_quant_dispatch.c',
  'src/backends/cpu/ops/matmul/quantized/matmul_quant_pack.c',
  'src/backends/cpu/ops/rope/rope_sincos_cache.c',
  'src/backends/cpu/ops/rope/rope_dispatch.c',
  'src/backends/cpu/ops/rope/rope_neon.c',
  'src/backends/cpu/ops/rope/rope_avx.c',
  'src/backends/cpu/ops/reduction/reduction_dispatch.c',
  'src/backends/cpu/ops/reduction/scalar/f32.c',
  'src/backends/cpu/ops/reduction/scalar/int.c',
  'src/backends/cpu/ops/reduction/neon/f32.c',
  'src/backends/cpu/ops/reduction/neon/int32.c',
  'src/backends/cpu/ops/reduction/neon/f16.c',
  'src/backends/cpu/ops/reduction/neon/bf16.c',
  'src/backends/cpu/ops/reduction/avx2/int.c',
)

backend_sources = cpu_backend_sources

if host_machine.system() == 'darwin' and enable_metal_opt
  add_languages('objcpp', native: false, required: false)

  metal_dep = dependency('appleframeworks', modules: ['Foundation', 'Metal'], required: false)
  if metal_dep.found()
    enable_metal = true
    deps += [metal_dep]
    xcrun = find_program('xcrun', required: true)

    metal_def_dir = 'src/backends/metal/kernels'
    metal_shader_dir = 'src/backends/metal/shaders'
    metal_common_headers = files(
      metal_shader_dir + '/common/defines.h',
      metal_shader_dir + '/common/math_utils.h',
      metal_shader_dir + '/common/dtype_utils.h',
      metal_shader_dir + '/common/quant_blocks.h',
    )
    metal_shaders = files(
      metal_shader_dir + '/elementwise.metal',
      metal_shader_dir + '/activations.metal',
      metal_shader_dir + '/normalization.metal',
      metal_shader_dir + '/matmul.metal',
      metal_shader_dir + '/marmot_matmul.metal',
      metal_shader_dir + '/matmul_qkv.metal',
      metal_shader_dir + '/matmul_quantized.metal',
      metal_shader_dir + '/embedding.metal',
      metal_shader_dir + '/vec_dot.metal',
      metal_shader_dir + '/fusion_kernels.metal',
      metal_shader_dir + '/tensor_ops.metal',
      metal_shader_dir + '/attention.metal',
      metal_shader_dir + '/conversions.metal',
      metal_shader_dir + '/quantization.metal',
      metal_shader_dir + '/reductions.metal',
    )

    metal_include_dir = join_paths(meson.current_source_dir(), metal_shader_dir)
    metal_compile_args = ['-I', metal_include_dir]
    if get_option('buildtype') != 'debug'
      metal_compile_args += ['-O3']
    endif

    metal_air_files = []
    foreach shader : metal_shaders
      shader_name = fs.stem(shader.full_path())
      air = custom_target('marmot_metal_air_' + shader_name,
        input: shader,
        output: shader_name + '.air',
        command: [xcrun, '-sdk', 'macosx', 'metal'] + metal_compile_args + ['-c', '@INPUT@', '-o', '@OUTPUT@'],
        depend_files: metal_common_headers,
      )
      metal_air_files += air
    endforeach

    metal_lib = custom_target('marmot_metal_lib',
      input: metal_air_files,
      output: 'kernels.metallib',
      command: [xcrun, '-sdk', 'macosx', 'metallib', '@INPUT@', '-o', '@OUTPUT@']
    )

    metal_c = custom_target('marmot_metal_c',
      input: metal_lib,
      output: 'metal_kernels_data.c',
      command: [py, files('scripts/embed_binary.py'), '@INPUT@', '@OUTPUT@', 'marmot_metal_lib']
    )

    backend_sources += [
      files(
        'src/backends/metal/metal_backend.mm',
        'src/backends/metal/metal_memory.mm',
        'src/backends/metal/ops/elementwise.mm',
        'src/backends/metal/internal/runtime.mm',
        'src/backends/metal/ops/normalization.mm',
        'src/backends/metal/ops/conversions.mm',
        'src/backends/metal/ops/matmul.mm',
        'src/backends/metal/metal_packed_weight.mm',
        'src/backends/metal/metal_caps.mm',
        'src/backends/metal/ops/fusion.mm',
        'src/backends/metal/ops/metal_kernel_dispatch.gen.mm',
        'src/backends/metal/ops/bytecode_tables_metal.gen.mm',
        'src/backends/metal/ops/bytecode_exec_metal.gen.mm',
        'src/backends/metal/internal/metal_kernel_query.gen.mm',
        'src/backends/metal/ops/matmul_quant_buffers.mm',
        'src/backends/metal/ops/matmul_quant_dispatch.mm',
        'src/backends/metal/ops/metal_matmul_quant_dispatch.gen.mm',
        'src/backends/metal/ops/matmul_kernels.mm',
        'src/backends/metal/ops/matmul_qkv_pack.mm',
        'src/backends/metal/ops/matmul_qkv_kernel.mm',
        'src/backends/metal/ops/matmul_qkv.mm',
        'src/backends/metal/ops/matmul_qkv_quant.mm',
        'src/backends/metal/ops/vec_dot.mm',
        'src/backends/metal/quantization/metal_quant_utils.mm',
        'src/backends/metal/quantization/q_gguf.mm',
        'src/backends/metal/quantization/qk_gguf.mm',
        'src/backends/metal/quantization/quantization.mm',
        'src/backends/metal/ops/tensor_ops.mm',
        'src/backends/metal/ops/attention.mm',
        'src/backends/metal/ops/reduction.mm',
        'src/backends/metal/ops/embedding.mm'
      ),
      metal_c,
    ]
  endif
elif host_machine.system() == 'darwin' and not enable_metal_opt
  message('Metal backend explicitly disabled via -Denable_metal=false')
endif

c_args += ['-DMARMOT_ENABLE_METAL=' + (enable_metal ? '1' : '0')]
objcpp_args += ['-DMARMOT_ENABLE_METAL=' + (enable_metal ? '1' : '0')]
cpp_args = c_args + ['-std=c++23']

libmarmot = shared_library('marmot',
  core_sources + utils_sources + backend_sources,
  include_directories: all_inc_dirs,
  c_args: c_args,
  cpp_args: cpp_args,
  objcpp_args: objcpp_args,
  dependencies: deps,
  install: true,
  version: meson.project_version(),
  soversion: '0',
)

libmarmot_static = static_library('marmot_static',
  core_sources + utils_sources + backend_sources,
  include_directories: all_inc_dirs,
  c_args: c_args + ['-fPIC'],
  cpp_args: cpp_args + ['-fPIC'],
  objcpp_args: objcpp_args + ['-fPIC'],
  dependencies: deps,
  install: true,
  install_dir: get_option('libdir'),
)

install_subdir('include/marmot', install_dir: 'include')

pkg = import('pkgconfig')
pkg.generate(
  libmarmot,
  name: 'marmot',
  description: 'High-performance ML inference framework',
)

# Test dependencies
cmocka_dep = dependency('cmocka', required: false)
test_inc_dirs = [inc_dirs, include_directories('tests'), root_inc]

llama_generate_example = executable('llama_generate',
  'examples/llama_generate.c',
  include_directories: all_inc_dirs,
  c_args: c_args,
  link_with: libmarmot,
  dependencies: deps,
  build_by_default: false,
)

backend_tests = [
  {'target': 'test_backend_memory', 'label': 'Backend memory ops', 'source': 'tests/backend/test_backend_memory.c'},
  {'target': 'test_backend_tensor_ops', 'label': 'Backend tensor ops', 'source': 'tests/backend/test_backend_tensor_ops.c'},
  {'target': 'test_backend_elementwise', 'label': 'Backend elementwise ops', 'source': 'tests/backend/test_backend_elementwise.c'},
  {'target': 'test_backend_activations', 'label': 'Backend activations', 'source': 'tests/backend/test_backend_activations.c'},
  {'target': 'test_backend_normalization', 'label': 'Backend normalization', 'source': 'tests/backend/test_backend_normalization.c'},
  {'target': 'test_backend_softmax', 'label': 'Backend softmax', 'source': 'tests/backend/test_backend_softmax.c'},
  {'target': 'test_backend_reduction', 'label': 'Backend reductions', 'source': 'tests/backend/test_backend_reduction.c'},
  {'target': 'test_backend_matmul', 'label': 'Backend matmul', 'source': 'tests/backend/test_backend_matmul.c'},
  {'target': 'test_backend_paged_attention', 'label': 'Backend paged attention', 'source': 'tests/backend/test_backend_paged_attention.c'},
  {'target': 'test_backend_paged_attention_perf', 'label': 'Backend paged attention prefill perf (optional)', 'source': 'tests/backend/test_backend_paged_attention_perf.c'},
  {'target': 'test_backend_embedding', 'label': 'Backend embedding', 'source': 'tests/backend/test_backend_embedding.c'},
  {'target': 'test_backend_embedding_golden', 'label': 'Backend embedding golden', 'source': 'tests/backend/test_backend_embedding_golden.c'},
  {'target': 'test_backend_embedding_perf', 'label': 'Backend embedding perf (optional)', 'source': 'tests/backend/test_backend_embedding_perf.c'},
  {'target': 'test_graph_inference_perf', 'label': 'Graph inference perf (optional)', 'source': 'tests/graph/test_graph_inference_perf.c'},
  {'target': 'test_rope', 'label': 'RoPE', 'source': 'tests/backend/test_rope.c'},
  {'target': 'test_backend_quantization', 'label': 'Backend quantization', 'source': 'tests/backend/test_backend_quantization.c'},
  {'target': 'test_vec_dot', 'label': 'Vec dot kernels', 'source': 'tests/backend/test_vec_dot.c'},
  {'target': 'test_matmul_quantized', 'label': 'Quantized matmul', 'source': 'tests/backend/test_matmul_quantized.c'},
  {'target': 'test_metal_compat', 'label': 'Metal compatibility', 'source': 'tests/backend/test_metal_compat.c'},
  {'target': 'test_backend_conversions', 'label': 'Backend dtype conversions', 'source': 'tests/backend/test_backend_conversions.c'},
  {'target': 'test_metal_quant_optimized', 'label': 'Metal K-quant optimization harness', 'source': 'tests/backend/test_metal_quant_optimized.c'},
]

# CI-only tests (performance benchmarks requiring env vars or heavy fixtures)
ci_only_tests = [
  'test_backend_embedding_perf',
  'test_graph_inference_perf',
  'test_metal_quant_optimized',
]
perf_only_tests = [
  'test_backend_paged_attention_perf',
]

foreach t : backend_tests
  exe = executable(t['target'],
    t['source'],
    include_directories: test_inc_dirs,
    c_args: c_args,
    link_with: libmarmot,
    dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
  )

  backend_variants = ['cpu']
  if enable_metal
    backend_variants += ['metal']
  endif

  # Determine test suite based on whether it's a CI-only or perf-only test
  if perf_only_tests.contains(t['target'])
    test_suite = ['perf']
  else
    test_suite = ci_only_tests.contains(t['target']) ? ['ci'] : ['default', 'ci']
  endif

  foreach backend_name : backend_variants
      test_env = environment()
      test_env.set('MARMOT_TEST_BACKEND', backend_name)
      test_env.set('MARMOT_TEST_MODE', '1')
      if backend_name == 'metal'
        test_env.set('MALLOC_PERTURB_', '0')
      endif
      test(t['label'] + ' [' + backend_name + ']', exe,
        env: test_env,
        is_parallel: false,
        suite: test_suite,
      )
  endforeach
endforeach

core_tests = [
  {'target': 'test_op_signature_hash', 'label': 'Op signature hashing', 'source': 'tests/core/test_op_signature_hash.c'},
  {'target': 'test_bytecode_execute', 'label': 'Bytecode execution', 'source': 'tests/core/test_bytecode_execute.c'},
  {'target': 'test_bytecode_tables', 'label': 'Bytecode tables', 'source': 'tests/core/test_bytecode_tables.c'},
  {'target': 'test_bytecode_exec_cpu', 'label': 'Bytecode exec cpu', 'source': 'tests/core/test_bytecode_exec_cpu.c'},
  {'target': 'test_ops_utils', 'label': 'Ops utils shape inference', 'source': 'tests/core/test_ops_utils.c'},
  {'target': 'test_rope_freq_cache', 'label': 'RoPE frequency cache', 'source': 'tests/core/test_rope_freq_cache.c'},
  {'target': 'test_quant_registry', 'label': 'Quant registry', 'source': 'tests/core/test_quant_registry.c'},
  {'target': 'test_bitpack_utils', 'label': 'Bitpack utils', 'source': 'tests/core/test_bitpack_utils.c'},
  {'target': 'test_int4_utils', 'label': 'Int4 utils', 'source': 'tests/core/test_int4_utils.c'},
  {'target': 'test_allocator_registry', 'label': 'Allocator registry', 'source': 'tests/core/test_allocator_registry.c'},
]

foreach t : core_tests
  exe = executable(t['target'],
    t['source'],
    include_directories: test_inc_dirs,
    c_args: c_args,
    link_with: libmarmot,
    dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
  )
  test(t['label'], exe, is_parallel: false, suite: ['default', 'ci'])
endforeach

graph_tests = [
  {'target': 'test_graph_basic', 'label': 'Graph basic construction', 'source': 'tests/graph/test_graph_basic.c'},
  {'target': 'test_kernel_query', 'label': 'Kernel query selection', 'source': 'tests/graph/test_kernel_query.c'},
  {'target': 'test_graph_api_errors', 'label': 'Graph API error handling', 'source': 'tests/graph/test_graph_api_errors.c'},
  {'target': 'test_graph_paged_attention', 'label': 'Graph paged attention', 'source': 'tests/graph/test_graph_paged_attention.c'},
  {'target': 'test_graph_layout_legalization', 'label': 'Graph layout legalization', 'source': 'tests/graph/test_graph_layout_legalization.c'},
  {'target': 'test_graph_dump', 'label': 'Graph JSON dump', 'source': 'tests/graph/test_graph_dump.c'},
  {'target': 'test_graph_fusion', 'label': 'Graph fusion', 'source': 'tests/graph/test_graph_fusion.c'},
  {'target': 'test_cross_backend_fusion', 'label': 'Cross-backend fusion', 'source': 'tests/graph/test_cross_backend_fusion.c'},
  {'target': 'test_architecture_traits', 'label': 'Architecture traits', 'source': 'tests/graph/test_architecture_traits.c'},
]

# Manual test tool for multi-architecture GGUF loading (not part of test suite)
executable('test_multiarch_load',
  'tests/graph/test_multiarch_load.c',
  include_directories: test_inc_dirs,
  c_args: c_args,
  link_with: libmarmot,
  dependencies: deps,
)

foreach t : graph_tests
  exe = executable(t['target'],
    t['source'],
    include_directories: test_inc_dirs,
    c_args: c_args,
    link_with: libmarmot,
    dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
  )

  test(t['label'], exe, is_parallel: false, suite: ['default', 'ci'])
endforeach

# GGUF fixture directory - can be overridden at runtime via MARMOT_GGUF_FIXTURE_DIR env var
gguf_fixture_dir = meson.project_source_root() / 'tests/fixtures/gguf/multiarch'
gguf_fixture_c_arg = '-DMARMOT_GGUF_FIXTURE_DIR_DEFAULT="' + gguf_fixture_dir + '"'

gguf_loader_test = executable('test_gguf_loader',
  'tests/graph/test_gguf_loader.c',
  include_directories: test_inc_dirs,
  c_args: c_args + [gguf_fixture_c_arg],
  link_with: libmarmot,
  dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
)

test('GGUF loader', gguf_loader_test, is_parallel: false, suite: ['ci'])

tokenizer_test = executable('test_tokenizer_gguf',
  'tests/tokenizer/test_tokenizer_gguf.c',
  include_directories: test_inc_dirs,
  c_args: c_args + [gguf_fixture_c_arg],
  link_with: libmarmot,
  dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
)

test('Tokenizer (GGUF)', tokenizer_test, is_parallel: false, suite: ['ci'])

tokenizer_qwen2_test = executable('test_tokenizer_qwen2',
  'tests/tokenizer/test_tokenizer_qwen2.c',
  include_directories: test_inc_dirs,
  c_args: c_args + [gguf_fixture_c_arg],
  link_with: libmarmot,
  dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
)

test('Tokenizer (Qwen2)', tokenizer_qwen2_test, is_parallel: false, suite: ['ci'])

tokenizer_qwen3_test = executable('test_tokenizer_qwen3',
  'tests/tokenizer/test_tokenizer_qwen3.c',
  include_directories: test_inc_dirs,
  c_args: c_args + [gguf_fixture_c_arg],
  link_with: libmarmot,
  dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
)

test('Tokenizer (Qwen3)', tokenizer_qwen3_test, is_parallel: false, suite: ['ci'])

llm_generate_test = executable('test_llm_generate_tokens',
  'tests/inference/test_llm_generate_tokens.c',
  include_directories: test_inc_dirs,
  c_args: c_args + [gguf_fixture_c_arg],
  link_with: libmarmot,
  dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
)

test('LLM generate tokens', llm_generate_test, is_parallel: false, suite: ['ci'], timeout: 600)

serving_engine_scheduler_test = executable('test_serving_engine_scheduler',
  'tests/inference/test_serving_engine_scheduler.c',
  include_directories: test_inc_dirs,
  c_args: c_args + [gguf_fixture_c_arg],
  link_with: libmarmot,
  dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
)

test('Serving engine scheduler', serving_engine_scheduler_test, is_parallel: false, suite: ['ci'], timeout: 600)

kv_pool_swap_test = executable('test_kv_pool_swap',
  'tests/inference/test_kv_pool_swap.c',
  include_directories: test_inc_dirs,
  c_args: c_args,
  link_with: libmarmot,
  dependencies: deps + (cmocka_dep.found() ? [cmocka_dep] : []),
)

test('KV pool swap', kv_pool_swap_test, is_parallel: false, suite: ['ci'])

test('Kernel def validation (fixtures)',
  py,
  args: [
    meson.current_source_dir() / 'scripts/codegen/validate_def.py',
    meson.current_source_dir() / 'tests/fixtures/codegen',
    meson.current_source_dir() / 'src/backends/cpu/kernels',
    meson.current_source_dir() / 'src/backends/metal/kernels',
  ],
  is_parallel: false,
  suite: ['default', 'ci'],
)

test('Kernel codegen ID determinism',
  py,
  args: [
    meson.current_source_dir() / 'tests/codegen/test_codegen_ids.py',
  ],
  is_parallel: false,
  suite: ['default', 'ci'],
)

test('Kernel def parser errors',
  py,
  args: [
    meson.current_source_dir() / 'tests/codegen/test_def_parser_errors.py',
  ],
  is_parallel: false,
  suite: ['default', 'ci'],
)

subdir('tools')
