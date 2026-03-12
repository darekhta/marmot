#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

BUILD_TYPE="${BUILD_TYPE:-debug}"
SANITIZER="${SANITIZER:-none}"
BUILD_DIR="build"

show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Marmot ML framework with various configurations.

OPTIONS:
    -t, --type TYPE         Build type: debug, release, debugoptimized (default: debug)
    -s, --sanitizer SAN     Enable sanitizer: asan, ubsan, tsan, none (default: none)
    -c, --clean             Clean build directory before building
    -h, --help              Show this help message

ENVIRONMENT VARIABLES:
    BUILD_TYPE              Same as --type
    SANITIZER               Same as --sanitizer
    ENABLE_KQUANT           Enable experimental K-Quant support (0/1, default: 0)
    ENABLE_ACCELERATE       Enable Apple Accelerate (0/1, default: 0; pass 1 to override)

EXAMPLES:
    # Debug build
    $0 --type debug

    # Release build
    $0 --type release

    # Debug with address sanitizer
    $0 --type debug --sanitizer asan

    # Clean release build with undefined behavior sanitizer
    $0 --clean --type release --sanitizer ubsan

    # Using environment variables
    BUILD_TYPE=release SANITIZER=asan $0

EOF
}

CLEAN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -s|--sanitizer)
            SANITIZER="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN=1
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

case "$BUILD_TYPE" in
    debug|release|debugoptimized)
        ;;
    *)
        echo "Error: Invalid build type '$BUILD_TYPE'"
        echo "Valid options: debug, release, debugoptimized"
        exit 1
        ;;
esac

case "$SANITIZER" in
    none|asan|ubsan|tsan)
        ;;
    *)
        echo "Error: Invalid sanitizer '$SANITIZER'"
        echo "Valid options: none, asan, ubsan, tsan"
        exit 1
        ;;
esac

if [[ "$SANITIZER" != "none" ]]; then
    BUILD_DIR="build-${BUILD_TYPE}-${SANITIZER}"
else
    BUILD_DIR="build-${BUILD_TYPE}"
fi

echo "========================================="
echo "Marmot Build Configuration"
echo "========================================="
echo "Build Type:  $BUILD_TYPE"
echo "Sanitizer:   $SANITIZER"
echo "Build Dir:   $BUILD_DIR"
echo "========================================="
echo

reset_build_dir() {
    local dir="$1"
    if [[ ! -e "$dir" ]]; then
        return 0
    fi
    if [[ "$(uname)" == "Darwin" ]]; then
        /bin/chmod -RN "$dir" 2>/dev/null || true
        find "$dir" -exec /bin/chmod -h -N {} + 2>/dev/null || true
    fi
    rm -rf "$dir"
}

if [[ $CLEAN -eq 1 ]] && [[ -d "$BUILD_DIR" ]]; then
    echo "Cleaning build directory..."
    reset_build_dir "$BUILD_DIR"
fi

# Force use of system compilers to avoid ASAN version mismatches
export CC="/usr/bin/clang"
export CXX="/usr/bin/clang++"
export OBJCXX="/usr/bin/clang++"

MESON_OPTS=()
MESON_OPTS+=("-Dbuildtype=$BUILD_TYPE")

if [[ "$SANITIZER" != "none" ]]; then
    SAN_TYPE="$SANITIZER"
    case "$SANITIZER" in
        asan) SAN_TYPE="address" ;;
        ubsan) SAN_TYPE="undefined,float-divide-by-zero,float-cast-overflow" ;;
        tsan) SAN_TYPE="thread" ;;
    esac

    SAN_FLAGS="-fsanitize=$SAN_TYPE -fno-omit-frame-pointer -g"
    MESON_OPTS+=("-Dc_args=$SAN_FLAGS")
    MESON_OPTS+=("-Dcpp_args=$SAN_FLAGS")
    MESON_OPTS+=("-Dobjcpp_args=$SAN_FLAGS")

    SAN_LINK_FLAGS="-fsanitize=$SAN_TYPE"
    MESON_OPTS+=("-Dc_link_args=$SAN_LINK_FLAGS")
    MESON_OPTS+=("-Dcpp_link_args=$SAN_LINK_FLAGS")
    MESON_OPTS+=("-Dobjcpp_link_args=$SAN_LINK_FLAGS")
fi

flag_enabled() {
    case "${1:-0}" in
        1|true|TRUE|on|ON|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

accelerate_default=0
if [[ "$(uname)" == "Darwin" ]]; then
    accelerate_default=1
fi

if flag_enabled "${ENABLE_ACCELERATE:-$accelerate_default}"; then
    MESON_OPTS+=("-Denable_apple_accelerate=true")
else
    MESON_OPTS+=("-Denable_apple_accelerate=false")
fi

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "Setting up build directory..."
    meson setup "$BUILD_DIR" "${MESON_OPTS[@]}"
    echo
elif [[ ! -f "$BUILD_DIR/meson-private/coredata.dat" ]]; then
    echo "Resetting invalid build directory..."
    reset_build_dir "$BUILD_DIR"
    meson setup "$BUILD_DIR" "${MESON_OPTS[@]}"
    echo
else
    echo "Reconfiguring existing build..."
    meson configure "$BUILD_DIR" "${MESON_OPTS[@]}"
    echo
fi

echo "Building..."
meson compile -C "$BUILD_DIR"
echo

echo "========================================="
echo "Build complete!"
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Library: $BUILD_DIR/libmarmot.dylib"
else
    echo "Library: $BUILD_DIR/libmarmot.so"
fi
echo "========================================="
