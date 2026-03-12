#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PREFIX="/usr/local"
LIB_ONLY=0
LM_ONLY=0

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Install libmarmot and/or marmot-lm.

OPTIONS:
    --prefix DIR        Installation prefix (default: /usr/local)
    --lib-only          Install libmarmot only (library, headers, pkg-config)
    --lm-only           Install marmot-lm only (requires libmarmot already installed)
    -h, --help          Show this help message

EXAMPLES:
    # Install everything to /usr/local
    $0

    # Install to /opt/marmot
    $0 --prefix /opt/marmot

    # Install just the library (for development)
    $0 --lib-only

    # Install just the CLI (after library is installed)
    $0 --lm-only

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --prefix)
            PREFIX="$2"
            shift 2
            ;;
        --lib-only)
            LIB_ONLY=1
            shift
            ;;
        --lm-only)
            LM_ONLY=1
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [[ $LIB_ONLY -eq 1 ]] && [[ $LM_ONLY -eq 1 ]]; then
    echo "Error: --lib-only and --lm-only are mutually exclusive"
    exit 1
fi

INSTALL_LIB=1
INSTALL_LM=1
if [[ $LIB_ONLY -eq 1 ]]; then INSTALL_LM=0; fi
if [[ $LM_ONLY -eq 1 ]]; then INSTALL_LIB=0; fi

echo "========================================="
echo "Marmot Installation"
echo "========================================="
echo "Prefix:         $PREFIX"
echo "Install lib:    $([ $INSTALL_LIB -eq 1 ] && echo yes || echo no)"
echo "Install CLI:    $([ $INSTALL_LM -eq 1 ] && echo yes || echo no)"
echo "========================================="
echo

# Force system compilers for consistent builds
export CC="/usr/bin/clang"
export CXX="/usr/bin/clang++"
export OBJCXX="/usr/bin/clang++"

BUILD_DIR="build-release"

ACCELERATE_OPT="-Denable_apple_accelerate=false"
if [[ "$(uname)" == "Darwin" ]]; then
    ACCELERATE_OPT="-Denable_apple_accelerate=true"
fi

# --- Install libmarmot ---
if [[ $INSTALL_LIB -eq 1 ]]; then
    echo "--- Building libmarmot (release) ---"

    MESON_OPTS=(
        "-Dbuildtype=release"
        "$ACCELERATE_OPT"
        "--prefix=$PREFIX"
    )

    if [[ ! -d "$BUILD_DIR" ]]; then
        meson setup "$BUILD_DIR" "${MESON_OPTS[@]}"
    else
        meson configure "$BUILD_DIR" "${MESON_OPTS[@]}"
    fi

    meson compile -C "$BUILD_DIR"

    echo
    echo "--- Installing libmarmot to $PREFIX ---"
    meson install -C "$BUILD_DIR"
    echo
    echo "libmarmot installed:"
    echo "  Library:    $PREFIX/lib/libmarmot.*"
    echo "  Headers:    $PREFIX/include/marmot/"
    echo "  pkg-config: $PREFIX/lib/pkgconfig/marmot.pc"
    echo
fi

# --- Install marmot-lm ---
if [[ $INSTALL_LM -eq 1 ]]; then
    echo "--- Building marmot-lm (release) ---"

    # Ensure pkg-config can find libmarmot at the install prefix
    export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig:${PKG_CONFIG_PATH:-}"

    if ! pkg-config --exists marmot 2>/dev/null; then
        echo "Warning: pkg-config cannot find 'marmot'."
        echo "  If libmarmot is not installed, run: make install-lib PREFIX=$PREFIX"
        echo "  Falling back to local build directory detection."
    fi

    cd apps/marmot-lm
    cargo build --release

    echo
    echo "--- Installing marmot-lm to $PREFIX/bin ---"
    install -d "$PREFIX/bin"
    install -m 755 target/release/marmot-lm "$PREFIX/bin/marmot-lm"
    cd "$PROJECT_ROOT"

    echo
    echo "marmot-lm installed:"
    echo "  Binary: $PREFIX/bin/marmot-lm"
    echo
fi

echo "========================================="
echo "Installation complete!"
echo
echo "Verify with:"
if [[ $INSTALL_LIB -eq 1 ]]; then
    echo "  pkg-config --modversion marmot"
fi
if [[ $INSTALL_LM -eq 1 ]]; then
    echo "  marmot-lm --help"
fi
echo "========================================="
