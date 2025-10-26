# Marmot Pre-Release Audit — LLM Handover Prompt

You are auditing the **Marmot** repository for correctness before its first public release. The repo is at `/Users/darekhta/Development/marmot` (or the current working directory). It has a single commit on `main` (`eb9d9e1`), 747 tracked files.

Marmot is an LLM inference engine written in C23 with CPU and Metal backends.

## Your Task

Perform a comprehensive correctness audit of the entire codebase and website. Report all issues found. Do NOT fix anything — only report. Organize findings by severity (critical, moderate, minor, cosmetic).

---

## 1. Documentation Link Integrity

Check every markdown file for broken internal links. There are 37 markdown files (listed below). For each `[text](path)` link:
- Verify the target file exists on disk
- Verify anchor links (`#section-name`) match actual headings in the target
- Flag any links to deleted files (ROADMAP.md was recently removed — check for stragglers)
- Flag any links referencing `anthropics/marmot` (should be `darekhta/marmot`)

**Markdown files to check:**
```
.ai/DEVELOPMENT.md
.github/copilot-instructions.md
AGENTS.md
CLAUDE.md (symlink — resolve and check target)
README.md
docs/README.md
docs/API_OPS.gen.md
docs/BYTECODE_DISPATCH.md
docs/DEVELOPER_ADOPTION_STRATEGY.md
docs/INFERENCE_QUALITY_GUIDE.md
docs/INFERENCE_VALIDATION_PLAN.md
docs/METAL_PERFORMANCE_OPTIMIZATION.md
docs/getting-started/BENCHMARKING.md
docs/getting-started/INSTALL.md
docs/getting-started/OPS_UTILS.md
docs/getting-started/QUICK_START.md
docs/graph/API.md
docs/graph/ARCHITECTURE.md
docs/graph/DEBUGGING.md
docs/graph/ENVIRONMENT.md
docs/graph/FUSION.md
docs/graph/GGUF.md
docs/graph/README.md
docs/graph/SIGNATURES.md
docs/kernels/ARCHITECTURE.md
docs/kernels/CODEGEN.md
docs/kernels/COST_MODEL.md
docs/kernels/COVERAGE.md
docs/kernels/DEBUGGING.md
docs/kernels/DISPATCH.md
docs/kernels/DSL.md
docs/kernels/OPS.md
docs/kernels/README.md
docs/tutorials/ADD_KERNEL.md
docs/tutorials/SIMPLE_GRAPH.md
tests/golden/README.md
apps/marmot-lm/docs/LM_STUDIO_COMPATIBILITY_GUIDE.md
```

## 2. README.md Accuracy

Cross-reference every claim in `README.md` against the actual codebase:
- "107 operations" — count operations in `src/core/defs/ops.def`
- "12 GGUF quantization formats" — verify in quantization source files
- "5 model architectures" — check `src/graph/gguf/architecture.cpp`
- "90 tests (56 C, 34 Rust)" — run `make test` and `cd apps/marmot-lm && cargo test` or count test files/functions
- "18 data types" — verify against `include/marmot/types.h`
- Model table (Llama, Qwen2, Qwen3, Phi-3, Gemma statuses) — verify against architecture code
- Quantization format table (formats, bit widths, block sizes) — verify against quant headers
- Build commands table — verify each `make` target exists in `Makefile`
- C API example — verify function signatures match actual headers (`marmot_init`, `marmot_gguf_model_load`, `marmot_graph_from_gguf`, `marmot_tokenizer_create_from_gguf_file`, etc.)
- Install commands — verify brew formula files exist at `Formula/`
- All GitHub URLs point to `github.com/darekhta/marmot`

## 3. Website Audit (`website/index.html`)

- All external links are valid (GitHub URLs, docs links)
- All GitHub URLs use `darekhta/marmot` (not `anthropics/marmot`)
- brew tap command uses `darekhta/marmot`
- Claims match README (operation count, model count, quant format count, test count)
- Logo image loads (`assets/logo.png` exists in `website/assets/`)
- OG image path is correct
- No broken anchor links in nav
- Code example syntax matches actual C API
- Copy button JS works (review the `copyText` and `copyCode` functions)
- No ROADMAP references remain
- Meta tags (title, description, og:*) are accurate
- Tailwind CDN script loads correctly (check URL)
- Google Fonts URLs are valid

## 4. CLAUDE.md / AGENTS.md Accuracy

CLAUDE.md is a symlink — verify what it points to (should be AGENTS.md). Then verify the content:
- File paths listed under "File Locations" — spot-check that listed files/directories exist
- Build commands — verify they work (`make build`, `make test`, etc.)
- Architecture descriptions match current code structure
- No references to deleted features (embeddings.h, diffusion.h, diffusion pipeline were removed — verify they're gone)
- No ROADMAP references
- "Supported Model Architectures" list matches `src/graph/gguf/architecture.cpp`

## 5. Cross-File Consistency

Check that these facts are consistent across README.md, website/index.html, CLAUDE.md/AGENTS.md, and docs/:
- Operation count (107)
- Model architecture list and statuses
- Quantization format count (12) and table
- Test count
- Backend descriptions (CPU: Accelerate/AVX2/NEON/scalar; Metal: simdgroup matmul, unified memory)
- GitHub URL (`darekhta/marmot` everywhere)
- License (MIT)

## 6. Stale References Scan

Search the entire codebase (all 747 files) for:
- `ROADMAP` — should have zero matches (all three ROADMAP.md files were deleted)
- `anthropics/marmot` — should have zero matches (replaced with `darekhta/marmot`)
- `embeddings.h` or `diffusion.h` — removed in prior cleanup, verify no references remain
- `diffusion_pipeline` or `diffusion pipeline` — same
- `pkg/marmot` — deleted Go bindings directory reference, should be gone
- `marmot_embeddings_` or `marmot_diffusion_` — dead API prefixes that were removed

## 7. Build & Test Verification

Run and report results:
```bash
make build && make test                    # C library + 56 tests
make build-lm                             # Rust CLI
cd apps/marmot-lm && cargo test           # 34 Rust tests
make format-check                         # Formatting compliance
```

Report any warnings, failures, or unexpected output.

## 8. .gitignore Coverage

Verify nothing that should be ignored is tracked:
- No `.DS_Store` files
- No `__pycache__` or `.pyc` files
- No build artifacts (`.o`, `.a`, `.so`, `.dylib`)
- No IDE files (`.vscode/`, `.idea/`)
- `.wrangler/` is gitignored (recently added)
- `website/` is intentionally tracked (not gitignored)

## 9. License and Legal

- `LICENSE` file exists and contains MIT license text
- Copyright line is appropriate
- README footer references MIT license
- Website footer references MIT license

## 10. Homebrew Formulas

Check `Formula/libmarmot.rb` and `Formula/marmot-lm.rb`:
- URLs point to `darekhta/marmot` (not `anthropics/marmot`)
- Version/dependency info looks reasonable
- No references to deleted files or features

---

## Output Format

```
## Critical (blocks release)
- [file:line] description

## Moderate (should fix before release)
- [file:line] description

## Minor (nice to fix)
- [file:line] description

## Cosmetic (optional)
- [file:line] description

## Verified OK
- Brief summary of what passed checks
```
