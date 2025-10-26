#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format not found"
    echo "Install with: brew install clang-format"
    exit 1
fi

echo "Checking code formatting..."

FILES=$(find include src tests -type f \( -name "*.c" -o -name "*.h" -o -name "*.mm" -o -name "*.metal" \) 2>/dev/null || true)

if [[ -z "$FILES" ]]; then
    echo "No C/C++/Metal files found"
    exit 0
fi

NEED_FORMAT=0

for file in $FILES; do
    if ! clang-format --dry-run --Werror "$file" &>/dev/null; then
        echo "  ✗ $file needs formatting"
        NEED_FORMAT=1
    fi
done

if [[ $NEED_FORMAT -eq 0 ]]; then
    echo "✓ All files are properly formatted"
    exit 0
else
    echo ""
    echo "Some files need formatting. Run:"
    echo "  make format"
    echo "  ./scripts/format.sh"
    exit 1
fi
