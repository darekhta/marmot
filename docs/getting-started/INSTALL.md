# Installation

How to install Marmot (the C library) and marmot-lm (the CLI/server).

---

## Homebrew (macOS)

The fastest way to get marmot-lm running on macOS:

```bash
# Install from the Marmot tap
brew tap darekhta/marmot https://github.com/darekhta/marmot
brew install marmot-lm
```

This installs both `libmarmot` (the shared library) and the `marmot-lm` binary.

To install only the library (for C/C++ development):

```bash
brew install libmarmot
```

### Verify

```bash
marmot-lm --help
pkg-config --modversion marmot   # 0.1.0
```

---

## From Source

### Prerequisites

| Tool | macOS | Linux |
|------|-------|-------|
| Meson >= 1.9 | `brew install meson` | `pip install meson` |
| Ninja | `brew install ninja` | `apt install ninja-build` |
| Clang (C23) | Xcode Command Line Tools | `apt install clang` |
| Python 3 + uv | `brew install uv` | `pip install uv` |
| Rust (for marmot-lm) | `brew install rust` | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |

### Build Everything

```bash
git clone https://github.com/darekhta/marmot.git
cd marmot

# Build libmarmot (debug)
make build

# Build marmot-lm (debug, links to libmarmot in build-debug/)
make build-lm

# Or build both in release mode
make build-release
make build-lm-release
```

### Install to System

```bash
# Install libmarmot + marmot-lm to /usr/local (default)
make install

# Or specify a custom prefix
PREFIX=/opt/marmot make install

# Install components separately
make install-lib              # libmarmot only
make install-lm               # marmot-lm only (requires libmarmot installed)
```

The install target:
1. Builds libmarmot in release mode
2. Runs `meson install` (library, headers, pkg-config)
3. Builds marmot-lm in release mode (finds libmarmot via pkg-config)
4. Copies the binary to `$PREFIX/bin/`

### What Gets Installed

| Component | Path |
|-----------|------|
| Shared library | `$PREFIX/lib/libmarmot.dylib` (macOS) or `.so` (Linux) |
| Headers | `$PREFIX/include/marmot/` |
| pkg-config | `$PREFIX/lib/pkgconfig/marmot.pc` |
| CLI binary | `$PREFIX/bin/marmot-lm` |
| Benchmark tool | `$PREFIX/bin/marmot-bench` |

### Using the Installed Library

After installation, C/C++ projects can find libmarmot via pkg-config:

```bash
# Compile flags
pkg-config --cflags marmot    # -I/usr/local/include

# Link flags
pkg-config --libs marmot      # -L/usr/local/lib -lmarmot
```

In a Meson project:

```meson
marmot_dep = dependency('marmot')
executable('myapp', 'main.c', dependencies: marmot_dep)
```

In a CMake project:

```cmake
find_package(PkgConfig REQUIRED)
pkg_check_modules(MARMOT REQUIRED marmot)
target_link_libraries(myapp ${MARMOT_LIBRARIES})
target_include_directories(myapp PRIVATE ${MARMOT_INCLUDE_DIRS})
```

---

## Development Setup

For active development on marmot itself, skip the install step and use the build directories directly:

```bash
make build                # libmarmot in build-debug/
make build-lm             # marmot-lm links to build-debug/libmarmot.dylib

# Run marmot-lm from the build tree
./apps/marmot-lm/target/debug/marmot-lm --help

# Run with release builds
make build-release
make build-lm-release
./apps/marmot-lm/target/release/marmot-lm --help
```

The `build.rs` in marmot-lm automatically finds libmarmot in the local build directories and sets the runtime library path (rpath) so the binary works without installation.

### Running Tests

```bash
make test                 # libmarmot fast tests
make test-lm              # marmot-lm e2e tests
make test-ci              # full CI suite (slow, includes LLM tests)
```

---

## Uninstall

### Homebrew

```bash
brew uninstall marmot-lm
brew uninstall libmarmot
```

### Manual

If installed with `make install`:

```bash
# Remove the binary
rm $PREFIX/bin/marmot-lm

# Remove the library and headers
rm $PREFIX/lib/libmarmot.*
rm -rf $PREFIX/include/marmot/
rm $PREFIX/lib/pkgconfig/marmot.pc
```

---

## See Also

- [Quick Start](QUICK_START.md) -- Build commands and first run
- [Benchmarking](BENCHMARKING.md) -- Performance measurement
