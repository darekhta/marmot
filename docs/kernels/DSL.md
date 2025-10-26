# Marmot Kernel DSL

This document specifies the `.def` authoring language used by Marmot's kernel codegen.
The language is optimized for kernel catalogs that are cartesian products (dtype x op x profile x epilogue x scheme).

The DSL is parsed and expanded by `scripts/codegen/def_parser.py` and consumed by `scripts/codegen/gen_kernels.py`.

---

## 1. Processing Model

- Codegen discovers all `*.def` files under a backend kernel directory (recursively), e.g. `src/backends/metal/kernels/**.def`.
- All discovered files are parsed and merged into a single module:
  - Duplicate `SET` / `DTYPE_SET` / `FRAGMENT` / `QUANT_SCHEME_TRAITS` / `QUANT_SCHEME` names are errors.
- `KERNEL_FAMILY` declarations are expanded into concrete kernel records.
- Kernel names must match the regex `[A-Za-z0-9_]+` and must be unique within a backend.

There is no `include` directive. Share common declarations by placing `.def` files in
`src/backends/<backend>/kernels/metadata/` (for example `metadata/common.def` and `metadata/quant_schemes.def`).

---

## 2. Lexical Rules

### 2.1 Whitespace

Whitespace is ignored except inside string literals.

### 2.2 Comments

Line comments start with `//` and continue to end-of-line.

Limitation: comment stripping is a pre-pass that does not respect strings; avoid `//` inside string literals.

### 2.3 Identifiers

Identifiers match `[A-Za-z_][A-Za-z0-9_]*`.

Conventions:
- Field names are written in `UPPER_SNAKE_CASE` (e.g. `INPUT_DTYPE`).
- Ops and symbolic values are written in `lower_snake_case` (e.g. `matmul`, `relu`).
- Dtypes are written in `UPPER_SNAKE_CASE` (e.g. `FLOAT16`).

All identifiers are case-sensitive.

### 2.4 Numbers

Numbers begin with a digit and may include:
- integers: `32`
- floats: `7.0`
- base-2 size suffixes: `4KB`, `8MB`, `1GB` (multipliers: 1024, 1024^2, 1024^3)

### 2.5 Strings

Strings are double-quoted (`"..."`) and support escapes: `\\n`, `\\t`, `\\r`, `\\"`, `\\\\`.

Strings are also *template strings* (see section 4.9).

---

## 3. Top-Level Declarations

### 3.1 `SET(name) { ... }` and `DTYPE_SET(name) { ... }`

Defines an ordered set. `DTYPE_SET` is a semantic alias of `SET` used for readability.

Set items can be:
- identifiers (e.g. `FLOAT16`, `matmul`)
- strings (e.g. `"bias_relu"`)
- set splices: `@other_set`

Example from `src/backends/cpu/kernels/elementwise.def`:

```c
SET(elementwise_binary_ops) {
    add, mul, sub, div, min, max, pow, mod,
    bitwise_and, bitwise_or, bitwise_xor,
    bitwise_shl, bitwise_shr, bitwise_shr_logical,
    compare_eq, compare_ne, compare_lt, compare_le, compare_gt, compare_ge
}

SET(elementwise_arith_ops) { add, mul, sub, div, min, max, pow, mod }
SET(compare_ops) { compare_eq, compare_ne, compare_lt, compare_le, compare_gt, compare_ge }
```

Example from `src/backends/cpu/kernels/metadata/common.def` (typical shared sets):

```c
DTYPE_SET(float_types) { FLOAT32, FLOAT16, BFLOAT16, FLOAT8_E4M3, FLOAT8_E5M2 }
DTYPE_SET(int_types) { INT64, INT32, INT16, INT8 }
DTYPE_SET(uint_types) { UINT8, UINT16, UINT32, UINT64 }
DTYPE_SET(all_numeric) { @float_types, @int_types, @uint_types }
```

Notes:
- `@set_name` splices another set (recursively).
- Order is preserved and used for deterministic expansion.
- Trailing commas are allowed in set bodies.

### 3.2 `FRAGMENT(name) [EXTENDS parent] { ... }`

A fragment is a reusable bundle of fields.
Fragments can inherit from a single parent; the parent is applied first, then the child overrides.

Example:

```c
FRAGMENT(epilogue_bias) {
    EPILOGUE: BIAS
}

FRAGMENT(epilogue_bias_relu) EXTENDS epilogue_bias {
    EPILOGUE: [BIAS, ACTIVATION],
    ACTIVATION: RELU
}
```

### 3.3 `QUANT_SCHEME_TRAITS(name) [EXTENDS parent] { ... }`

Reusable quant scheme traits. Recommended: define canonical `QUANT_BLOCK` here so kernels can reference it via
`QUANT_SCHEME($scheme).QUANT_BLOCK`.

Example:

```c
QUANT_SCHEME_TRAITS(legacy_zp_none) {
    QUANT_BLOCK: { block_size: 32, group_size: 32, scale_dtype: FLOAT16, zp_dtype: NONE }
}
```

### 3.4 `QUANT_SCHEME(NAME) [EXTENDS traits] { ... }`

Defines a quant scheme. Schemes inherit from a `QUANT_SCHEME_TRAITS` block and may override fields.

Example:

```c
QUANT_SCHEME(Q4_0) EXTENDS legacy_zp_none { }
```

### 3.5 `KERNEL_FAMILY(name) { ... }`

Declares a cartesian-product expansion that generates one or more concrete kernel records.

Inside a family:
- `AXIS(name): expr` declares an expansion axis (evaluated in source order).
- `NAME_PATTERN: expr` declares the kernel name template (must render to `[A-Za-z0-9_]+`).
- `WITH: expr` (optional) applies one fragment name or a list of fragment names.
- All other entries are kernel fields (e.g. `OP`, `INPUT_DTYPE`, ...).

Reserved entries (not emitted as kernel fields):
- `AXIS(...)`
- `NAME_PATTERN`
- `WITH`

**Example -- single-axis expansion** (from `src/backends/cpu/kernels/matmul_scalar.def`):

```c
KERNEL_FAMILY(matmul_scalar) {
    AXIS(layout): @matmul_layouts,
    AXIS(dtype): @matmul_types_scalar,
    NAME_PATTERN: "matmul_{dtype:short}_scalar_{layout:lower}",
    OP: matmul,
    PROFILE: SCALAR,
    MATMUL_LAYOUT: $layout,
    PLATFORM: SCALAR,
    INPUT_DTYPE: $dtype,
    WEIGHT_DTYPE: $dtype,
    OUTPUT_DTYPE: select($dtype, { @fp8_types: FLOAT32, *: $dtype }),
    ACCUM_DTYPE: select($dtype, { FLOAT64: FLOAT64, *: FLOAT32 }),
    STRIDE_MODE: CONTIGUOUS,
    EPILOGUE: [BIAS],
    IMPL_FUNCTION: select($layout, {
        NT: "cpu_matmul_{dtype:short}_scalar",
        NN: "cpu_matmul_{dtype:short}_scalar_nn",
        *: "cpu_matmul_{dtype:short}_scalar"
    }),
}
```

**Example -- multi-axis with conditional profiles** (from `src/backends/cpu/kernels/elementwise.def`):

```c
KERNEL_FAMILY(elementwise_binary_scalar) {
    AXIS(op): @elementwise_binary_ops,
    AXIS(dtype): select($op, {
        @elementwise_bitwise_ops: [@int_types, @uint_types],
        *: @numeric_all
    }),
    NAME_PATTERN: "{op}_{dtype:short}",
    OP: $op,
    INPUT_DTYPE: $dtype,
    WEIGHT_DTYPE: $dtype,
    OUTPUT_DTYPE: select($op, { @compare_ops: UINT8, *: $dtype }),
    ACCUM_DTYPE: select($dtype, { FLOAT64: FLOAT64, @float_types: FLOAT32, *: $dtype }),
    STRIDE_MODE: CONTIGUOUS,
    PROFILES: select($dtype, {
        FLOAT32: select($op, {
            @elementwise_float_simd_ops: {
                ACCELERATE: "cpu_{op}_{dtype:short}_accelerate",
                NEON: "cpu_{op}_{dtype:short}_neon",
                AVX2: "cpu_{op}_{dtype:short}_avx2",
                SCALAR: "cpu_{op}_{dtype:short}_scalar"
            },
            *: { SCALAR: "cpu_{op}_{dtype:short}_scalar" }
        }),
        INT32: select($op, {
            @elementwise_int_simd_ops: {
                NEON: "cpu_{op}_{dtype:short}_neon",
                AVX2: "cpu_{op}_{dtype:short}_avx2",
                SCALAR: "cpu_{op}_{dtype:short}_scalar"
            },
            *: { SCALAR: "cpu_{op}_{dtype:short}_scalar" }
        }),
        *: { SCALAR: "cpu_{op}_{dtype:short}_scalar" }
    }),
}
```

**Example -- tensor layout ops** (from `src/backends/cpu/kernels/tensor_ops.def`):

```c
KERNEL_FAMILY(tensor_ops) {
    AXIS(op): [contiguous, reshape, transpose, concat, slice, view,
               gather_rows, scatter_u64_to_i32],
    AXIS(kind): [generic, fp8],
    NAME_PATTERN: "tensor_{op}_{kind}",
    OP: $op,
    PROFILE: NONE,
    INPUT_DTYPE: select($kind, { fp8: @fp8_types, *: @convert_types_no_fp8 }),
    OUTPUT_DTYPE: select($kind, { fp8: @fp8_types, *: @convert_types_no_fp8 }),
    ACCUM_DTYPE: ANY,
    STRIDE_MODE: select($op, {
        contiguous: STRIDED,
        reshape: STRIDED,
        view: STRIDED,
        gather_rows: STRIDED,
        scatter_u64_to_i32: STRIDED,
        *: CONTIGUOUS
    }),
}
```

**Example -- Metal kernels** (from `src/backends/metal/kernels/elementwise.def`):

```c
KERNEL_FAMILY(arithmetic_binary_float) {
    AXIS(op): @arithmetic_ops,
    AXIS(dtype): @float_types,
    NAME_PATTERN: "{op}_{dtype:short}",
    OP: $op,
    PROFILE: NONE,
    INPUT_DTYPE: $dtype,
    OUTPUT_DTYPE: $dtype,
    ACCUM_DTYPE: FLOAT32,
    STRIDE_MODE: ROW_STRIDED,
}
```

Metal kernels use `PROFILE: NONE` since there are no ISA variants -- Metal shaders are uniform across devices.

### 3.6 `PROFILES: { ... }` (CPU-only profile expansion)

`PROFILES` is an optional struct field that lets a single logical kernel record expand into multiple **CPU profile** variants,
each bound to an implementation symbol (used by codegen to build kernel query tables and bytecode exec tables).

Example:

```c
KERNEL_FAMILY(add_binary) {
    AXIS(dtype): [FLOAT16, FLOAT32, BFLOAT16],
    NAME_PATTERN: "add_{dtype:short}",
    OP: add,
    INPUT_DTYPE: $dtype,
    WEIGHT_DTYPE: $dtype,
    OUTPUT_DTYPE: $dtype,
    ACCUM_DTYPE: select($dtype, { @float_types: FLOAT32, *: $dtype }),
    STRIDE_MODE: CONTIGUOUS,

    PROFILES: {
        ACCELERATE: "cpu_add_{dtype:short}_accelerate",
        NEON:       "cpu_add_{dtype:short}_neon",
        AVX2:       "cpu_add_{dtype:short}_avx2",
        SCALAR:     "cpu_add_{dtype:short}_scalar",
    },
}
```

Semantics:
- `PROFILES` is consumed by codegen and is not emitted as a kernel field.
- When `PROFILES` is present and the record does not already define `PROFILE`, codegen expands it into one record per key:
  - `name` becomes `"{base_name}_{profile:lower}"` (e.g. `add_f32_neon`)
  - `PROFILE` is set to the profile key (e.g. `NEON`)
  - `IMPL_FUNCTION` is set to the corresponding symbol template
- `SCALAR` is required (guaranteed fallback).
- Values must be non-empty strings; they are template strings and can reference axes (e.g. `{dtype:short}`).

Notes:
- `PROFILE` is the CPU SIMD/profile selector (`ACCELERATE`, `NEON`, `AVX2`, `SCALAR`). Matmul transpose modes use
  `MATMUL_LAYOUT` (`NN/NT/TN/TT`).

---

## 4. Values and Expressions

### 4.1 Scalars

Supported scalar literals:
- identifiers
- strings
- numbers
- booleans: `true` / `false`
- nulls: `none` / `null`

### 4.2 Set References: `@name`

`@name` evaluates to the expanded list of items in `SET(name)` / `DTYPE_SET(name)`.

### 4.3 Variables: `$name`

Each `AXIS(name)` binds `$name` during expansion.

Example:

```c
AXIS(dtype): @float_types,
INPUT_DTYPE: $dtype
```

### 4.4 Lists: `[ ... ]`

Lists are written as `[expr, expr, ...]`.

Evaluation rule: nested lists are flattened. This makes unions ergonomic:

```c
AXIS(dtype): [@int_types, @uint_types]  // becomes one flat list
```

### 4.5 Structs: `{ key: value, ... }`

Structs are written as `{ key: expr, ... }`. Keys can be identifiers or strings.

Example:

```c
QUANT_BLOCK: { block_size: 64, group_size: 32, scale_dtype: FLOAT16, zp_dtype: NONE }
```

Struct keys `@set` and `*` are reserved for `select(...)` mapping structs (see section 4.6).

### 4.6 `select(key, { ... })`

`select` chooses a value based on a key. The second argument is a mapping struct whose keys are tested in order:

- `@set_name`: matches if `key` is a member of the set
- `IDENT`: matches if `str(key) == "IDENT"`
- `"string"`: matches if `str(key) == "string"`
- `*`: required default

Example:

```c
ACCUM_DTYPE: select($dtype, {
    @float_types: FLOAT32,
    FLOAT64: FLOAT64,
    *: $dtype
})
```

Nested selects are supported. From `src/backends/cpu/kernels/elementwise.def`:

```c
PROFILES: select($dtype, {
    FLOAT32: select($op, {
        @elementwise_float_simd_ops: {
            ACCELERATE: "cpu_{op}_{dtype:short}_accelerate",
            NEON: "cpu_{op}_{dtype:short}_neon",
            AVX2: "cpu_{op}_{dtype:short}_avx2",
            SCALAR: "cpu_{op}_{dtype:short}_scalar"
        },
        pow: {
            ACCELERATE: "cpu_{op}_{dtype:short}_accelerate",
            SCALAR: "cpu_{op}_{dtype:short}_scalar"
        },
        *: { SCALAR: "cpu_{op}_{dtype:short}_scalar" }
    }),
    *: { SCALAR: "cpu_{op}_{dtype:short}_scalar" }
}),
```

### 4.7 Member Access: `expr.FIELD`

Member access reads a field from a dict value (struct or resolved quant scheme).

Lookup rules:
- First tries `FIELD` exactly.
- Then tries `FIELD` uppercased.

Example:

```c
QUANT_BLOCK: QUANT_SCHEME($scheme).QUANT_BLOCK
```

### 4.8 Builtins

#### `QUANT_SCHEME(name)`

Returns the resolved quant scheme record (traits + overrides) as a dict.

Example:

```c
QUANT_BLOCK: QUANT_SCHEME($scheme).QUANT_BLOCK
```

### 4.9 Template Strings

All string literals are template strings. Placeholders have the form `{var}` or `{var:formatter}` and are substituted
using the current axis environment.

Supported formatters:
- `lower`, `upper`
- `short` (dtype short-name, e.g. `FLOAT16` -> `f16`, `FLOAT8_E4M3` -> `fp8_e4m3`)
- `opt(prefix)`: emits `prefix + value` if value is non-empty, otherwise `""`
- `or(default)`: emits `default` if value is empty, otherwise the value

Examples:

```c
NAME_PATTERN: "matmul_{dtype:short}{epi:opt(_)}_{profile:lower}"
IMPL_FUNCTION: "cpu_matmul_{dtype:short}_neon_blocked_{profile:lower}"
```

Notes:
- `none` is treated as an empty string in templates.
- Formatters are not chainable.

---

## 5. Expansion Semantics

For each `KERNEL_FAMILY`:

1. Expand axes in source order (top to bottom).
2. For each expanded environment:
   - Render `NAME_PATTERN` to produce a kernel name.
   - Evaluate all family fields to concrete values.
   - If `WITH` is present, evaluate it to a fragment name or list of names, then merge fragment fields left-to-right.
     Fragment fields are applied after family fields and can override them.
3. Emit one kernel record per expanded name.

Errors include:
- Unknown references (`$var`, `@set`, fragment, quant scheme)
- Duplicate names (generated by different families or expansions)
- Invalid kernel name (does not match `[A-Za-z0-9_]+`)
- `select(...)` missing `*` default
- Cycles in set expansion or inheritance

---

## 6. Kernel Record Schema (Fields)

After expansion, each kernel record is a name plus a field map.
Fields are stored as either **core fields** (recognized by codegen) or **extensions** (passed through).

### 6.1 Required Fields (All Kernels)

Every kernel record must define:
- `OP`
- `INPUT_DTYPE`
- `OUTPUT_DTYPE`
- `ACCUM_DTYPE`
- `STRIDE_MODE`

`INPUT_DTYPE` and `OUTPUT_DTYPE` may be a scalar dtype or a list of dtypes.

### 6.2 Matmul/QKV Rules

For `OP: matmul`, `OP: qkv_rope`, `OP: qkv_shared_input`, and `OP: qkv_projection`:
- `MATMUL_LAYOUT` is required and must be one of `NN`, `NT`, `TN`, `TT`
- One of `WEIGHT_DTYPE` or `WEIGHT_QUANT` must be present

### 6.3 Quantized Weight Rules

If `WEIGHT_QUANT` is present:
- `QUANT_BLOCK` is required and must be a struct with subfields:
  - `block_size`, `group_size`, `scale_dtype`, `zp_dtype`

`WEIGHT_DTYPE` and `WEIGHT_QUANT` are mutually exclusive.

### 6.4 ACCUM_DTYPE Rules

Marmot distinguishes:
- Arithmetic ops: must not use `ACCUM_DTYPE: ANY`
- Layout + reductions: should use `ACCUM_DTYPE: ANY`

These are validated by `scripts/codegen/def_parser.py` during codegen.

### 6.5 EPILOGUE / ACTIVATION Consistency

- If `EPILOGUE` includes `ACTIVATION`, the `ACTIVATION` field is required.
- If the `ACTIVATION` field is present, `EPILOGUE` must include `ACTIVATION`.

### 6.6 Core Field List

Core fields currently recognized by the parser:

`ACTIVATION`, `OP`, `PROFILE`, `MATMUL_LAYOUT`, `PROFILES`, `PLATFORM`, `IMPL_FUNCTION`, `INPUT_DTYPE`, `WEIGHT_DTYPE`,
`OUTPUT_DTYPE`, `ACCUM_DTYPE`, `WEIGHT_QUANT`, `QUANT_BLOCK`, `WEIGHT_LAYOUT`, `STRIDE_MODE`, `EPILOGUE`, `BIAS`,
`FUSION`, `TILING`, `PIPELINE`, `COST_MODEL`, `SHARDABLE`, `DEVICE_AFFINITY`, `KERNEL_FUNCTION`, `THREADGROUP_SIZE`,
`SIMD_GROUP`, `MPS_FALLBACK`.

Any field not in the core set is stored as an extension field and preserved for backend-specific use.

---

## 7. Authoring Checklist

- Put reusable sets/fragments/schemes under `src/backends/<backend>/kernels/metadata/`.
- Use `SET` / `DTYPE_SET` + `AXIS(...)` to express cartesian products instead of copy/paste.
- Use `select(...)` for per-dtype or per-scheme rules.
- Keep `NAME_PATTERN` stable; kernel IDs are derived from names.
- Avoid `//` inside strings.
- Avoid trailing commas inside `[...]` lists and `{...}` structs.

## 8. Core Field Semantics

This section documents the complete semantics of each core field.

### STRIDE_MODE

Controls memory layout requirements for kernel dispatch. Affects which tensor layouts a kernel can accept.

| Value | Enum | Meaning |
|-------|------|---------|
| `ANY` | `MARMOT_STRIDE_MODE_ANY` | No layout constraint; accepts any tensor |
| `CONTIGUOUS` | `MARMOT_STRIDE_MODE_CONTIGUOUS` | Requires fully contiguous memory (best for SIMD) |
| `ROW_STRIDED` | `MARMOT_STRIDE_MODE_ROW_STRIDED` | Requires contiguous rows (allows batch striding) |
| `STRIDED` | `MARMOT_STRIDE_MODE_STRIDED` | Arbitrary strides (flexible but slower) |

**Usage**: Set based on kernel implementation requirements.
- SIMD-vectorized kernels typically need `CONTIGUOUS`
- Batch operations often work with `ROW_STRIDED`
- Generic scalar fallbacks use `ANY` or `STRIDED`

```c
// Fast NEON kernel requires contiguous data
STRIDE_MODE: CONTIGUOUS,

// Row-strided scalar fallback
STRIDE_MODE: ROW_STRIDED,
```

### COST_MODEL

Optional name for a cost estimation function. The parser accepts and preserves the field, but **cost-based kernel ranking is
not wired into the generated backend queries today**, so `marmot_kernel_selection_t.estimated_us` is currently `0.0` for all
generated matches.

See `docs/kernels/COST_MODEL.md` for current status and a roadmap for enabling estimates.

### FUSION

Declares that this kernel matches a **variant flag mask** (`marmot_op_signature_t.variant_flags`). This is **not** the same as
graph-level fusion (`add->relu`, `matmul->add`, etc), which is represented by a different fused `op_id`.

Today, the only shipped use is:
- `RESIDUAL_ADD` -- selects normalization kernels that add a residual input while normalizing.

**Example (normalization residual variant)**:
```c
KERNEL_FAMILY(layernorm_residual) {
    OP: layernorm,
    FUSION: RESIDUAL_ADD,
    // ...
}
```

### DEVICE_AFFINITY

Optional backend preference hint. This field is currently **not consumed by generated queries** and is reserved for future
cross-backend routing logic.

### PLATFORM

Guards kernel availability based on hardware features. If the platform is unavailable at runtime, the kernel is skipped during selection.

| Value | Guard Macro | Meaning |
|-------|-------------|---------|
| `ACCELERATE` | `MARMOT_ENABLE_ACCELERATE` | Apple Accelerate framework |
| `NEON` | `MARMOT_ENABLE_NEON` | ARM NEON SIMD |
| `AVX2` | `MARMOT_ENABLE_AVX2` | Intel AVX2 |
| `AVX512` | `MARMOT_ENABLE_AVX512` | Intel AVX-512 |
| `ANY` | `1` | Always available |

**Note**: When `PLATFORM` is not specified, it is inferred from `PROFILE`:
- `PROFILE: NEON` -> `PLATFORM: NEON`
- `PROFILE: SCALAR` -> `PLATFORM: ANY`

### IMPL_FUNCTION

Specifies the C function symbol that implements this kernel. Used by codegen to generate dispatch tables.

**For CPU kernels** (set automatically by `PROFILES` expansion):
```c
IMPL_FUNCTION: "cpu_add_f32_neon"
```

**For Metal kernels** (typically not needed; Metal uses shader names).

### Metal-Specific Fields

| Field | Description |
|-------|-------------|
| `KERNEL_FUNCTION` | Metal shader function name |
| `THREADGROUP_SIZE` | Threadgroup dimensions `{ x: N, y: M, z: 1 }` |
| `SIMD_GROUP` | SIMD group configuration |
| `MPS_FALLBACK` | Use MPS framework as fallback |

---

## 9. Validation Rules

The DSL parser (`scripts/codegen/def_parser.py`) enforces these rules during codegen.

### Structural Rules

| Rule | Error |
|------|-------|
| Kernel names must match `[A-Za-z0-9_]+` | `Invalid kernel name` |
| Kernel names must be unique per backend | `Duplicate kernel name` |
| `select()` must have `*` default | `Missing default in select()` |
| Set references must exist | `Unknown set reference` |
| Variable references must be bound by AXIS | `Unbound variable` |
| Fragment inheritance must not cycle | `Cycle in fragment inheritance` |

### Field Rules

| Rule | Error |
|------|-------|
| `OP` is required | `Kernel '<name>' missing required field 'OP'` |
| `INPUT_DTYPE` is required | `Kernel '<name>' missing required field 'INPUT_DTYPE'` |
| `OUTPUT_DTYPE` is required | `Kernel '<name>' missing required field 'OUTPUT_DTYPE'` |
| `ACCUM_DTYPE` is required | `Kernel '<name>' missing required field 'ACCUM_DTYPE'` |
| `STRIDE_MODE` is required | `Kernel '<name>' missing required field 'STRIDE_MODE'` |

### Matmul/QKV Rules

| Rule | Error |
|------|-------|
| Matmul/QKV requires `MATMUL_LAYOUT` in {NN, NT, TN, TT} | `Invalid MATMUL_LAYOUT` |
| Matmul/QKV requires `WEIGHT_DTYPE` or `WEIGHT_QUANT` | `Requires WEIGHT_DTYPE or WEIGHT_QUANT` |
| `WEIGHT_DTYPE` and `WEIGHT_QUANT` are exclusive | `Mutually exclusive` |

### Quantization Rules

| Rule | Error |
|------|-------|
| `WEIGHT_QUANT` requires `QUANT_BLOCK` | `Missing QUANT_BLOCK` |
| `QUANT_BLOCK` requires `block_size` | `Invalid quant block` |
| `QUANT_BLOCK` requires `group_size` | `Invalid quant block` |
| `QUANT_BLOCK` requires `scale_dtype` | `Invalid quant block` |
| `QUANT_BLOCK` requires `zp_dtype` | `Invalid quant block` |

### Epilogue Rules

| Rule | Error |
|------|-------|
| `EPILOGUE: ACTIVATION` requires `ACTIVATION` field | `Missing ACTIVATION field` |
| `ACTIVATION` field requires `EPILOGUE: ACTIVATION` | `Orphaned ACTIVATION` |

### PROFILES Rules (CPU only)

| Rule | Error |
|------|-------|
| `PROFILES` must include `SCALAR` | `Missing SCALAR fallback` |
| `PROFILES` values must be non-empty strings | `Invalid profile symbol` |

---

## See Also

- `docs/graph/SIGNATURES.md` -- Runtime signature field reference
- `docs/kernels/OPS.md` -- Complete operation catalog
- `docs/kernels/CODEGEN.md` -- How codegen works
- `scripts/codegen/def_parser.py` -- Parser implementation
- `scripts/codegen/gen_kernels.py` -- Code generator
