#!/usr/bin/env bash
set -euo pipefail

VERSION="0.1.0"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"

# Detect platform
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "${OS}" in
  darwin) PLATFORM="darwin" ;;
  linux)  PLATFORM="linux" ;;
  *)      echo "Unsupported OS: ${OS}"; exit 1 ;;
esac

case "${ARCH}" in
  arm64|aarch64) ARCH_LABEL="arm64" ;;
  x86_64)        ARCH_LABEL="x86_64" ;;
  *)             echo "Unsupported arch: ${ARCH}"; exit 1 ;;
esac

TRIPLE="${PLATFORM}-${ARCH_LABEL}"
echo "==> Packaging for ${TRIPLE}"

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}"

# --- Build libmarmot (release) ---
echo "==> Building libmarmot (release)..."
cd "${ROOT_DIR}"
if [ ! -d build-release ]; then
  meson setup build-release --buildtype=release
fi
meson compile -C build-release

# --- Build marmot-lm (release) ---
echo "==> Building marmot-lm (release)..."
cd "${ROOT_DIR}/apps/marmot-lm"
cargo build --release

# --- Locate artifacts ---
MARMOT_LM_BIN="${ROOT_DIR}/apps/marmot-lm/target/release/marmot-lm"

if [ "${PLATFORM}" = "darwin" ]; then
  DYLIB_NAME="libmarmot.dylib"
  STATIC_NAME="libmarmot_static.a"
else
  DYLIB_NAME="libmarmot.so"
  STATIC_NAME="libmarmot_static.a"
fi

DYLIB_PATH="${ROOT_DIR}/build-release/${DYLIB_NAME}"
STATIC_PATH="${ROOT_DIR}/build-release/${STATIC_NAME}"

if [ ! -f "${MARMOT_LM_BIN}" ]; then
  echo "ERROR: marmot-lm binary not found at ${MARMOT_LM_BIN}"
  exit 1
fi

if [ ! -f "${DYLIB_PATH}" ]; then
  echo "ERROR: ${DYLIB_NAME} not found at ${DYLIB_PATH}"
  exit 1
fi

# --- Package marmot-lm tarball ---
echo "==> Packaging marmot-lm tarball..."
LM_STAGE="${DIST_DIR}/marmot-lm-${VERSION}"
mkdir -p "${LM_STAGE}/bin" "${LM_STAGE}/lib"

cp "${MARMOT_LM_BIN}" "${LM_STAGE}/bin/"
cp "${DYLIB_PATH}" "${LM_STAGE}/lib/"

if [ "${PLATFORM}" = "darwin" ]; then
  # Fix dylib install_name to be rpath-relative
  install_name_tool -id "@rpath/libmarmot.dylib" "${LM_STAGE}/lib/libmarmot.dylib"

  # Fix marmot-lm to find dylib via @rpath, with rpath pointing to ../lib
  # First, replace any absolute path reference to the dylib
  CURRENT_DYLIB_REF=$(otool -L "${LM_STAGE}/bin/marmot-lm" | grep libmarmot | head -1 | awk '{print $1}')
  if [ -n "${CURRENT_DYLIB_REF}" ] && [ "${CURRENT_DYLIB_REF}" != "@rpath/libmarmot.dylib" ]; then
    install_name_tool -change "${CURRENT_DYLIB_REF}" "@rpath/libmarmot.dylib" "${LM_STAGE}/bin/marmot-lm"
  fi

  # Add rpath for relative lib directory
  install_name_tool -add_rpath "@executable_path/../lib" "${LM_STAGE}/bin/marmot-lm" 2>/dev/null || true

  # Strip ad-hoc signature and re-sign
  codesign --remove-signature "${LM_STAGE}/bin/marmot-lm" 2>/dev/null || true
  codesign --remove-signature "${LM_STAGE}/lib/libmarmot.dylib" 2>/dev/null || true
  codesign -s - "${LM_STAGE}/lib/libmarmot.dylib"
  codesign -s - "${LM_STAGE}/bin/marmot-lm"
fi

LM_TARBALL="marmot-lm-${VERSION}-${TRIPLE}.tar.gz"
cd "${DIST_DIR}"
tar czf "${LM_TARBALL}" "marmot-lm-${VERSION}/"
echo "  -> ${DIST_DIR}/${LM_TARBALL}"

# --- Package libmarmot-dev tarball ---
echo "==> Packaging libmarmot-dev tarball..."
DEV_STAGE="${DIST_DIR}/libmarmot-dev-${VERSION}"
mkdir -p "${DEV_STAGE}/lib/pkgconfig" "${DEV_STAGE}/include"

cp "${DYLIB_PATH}" "${DEV_STAGE}/lib/"
if [ -f "${STATIC_PATH}" ]; then
  cp "${STATIC_PATH}" "${DEV_STAGE}/lib/"
fi

# Copy headers
cp -R "${ROOT_DIR}/include/marmot" "${DEV_STAGE}/include/"

# Generate a relocatable pkg-config file
cat > "${DEV_STAGE}/lib/pkgconfig/marmot.pc" <<PCEOF
prefix=\${pcfiledir}/../..
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: marmot
Description: High-performance ML inference framework
Version: ${VERSION}
Libs: -L\${libdir} -lmarmot
Cflags: -I\${includedir}
PCEOF

if [ "${PLATFORM}" = "darwin" ]; then
  install_name_tool -id "@rpath/libmarmot.dylib" "${DEV_STAGE}/lib/libmarmot.dylib"
  codesign --remove-signature "${DEV_STAGE}/lib/libmarmot.dylib" 2>/dev/null || true
  codesign -s - "${DEV_STAGE}/lib/libmarmot.dylib"
fi

DEV_TARBALL="libmarmot-dev-${VERSION}-${TRIPLE}.tar.gz"
cd "${DIST_DIR}"
tar czf "${DEV_TARBALL}" "libmarmot-dev-${VERSION}/"
echo "  -> ${DIST_DIR}/${DEV_TARBALL}"

# --- Summary ---
echo ""
echo "==> Done! Tarballs in ${DIST_DIR}:"
ls -lh "${DIST_DIR}"/*.tar.gz

echo ""
echo "==> Verification hints:"
echo "  tar xzf ${DIST_DIR}/${LM_TARBALL}"
echo "  otool -L marmot-lm-${VERSION}/bin/marmot-lm"
echo "  ./marmot-lm-${VERSION}/bin/marmot-lm --help"
