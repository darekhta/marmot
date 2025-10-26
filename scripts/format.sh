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

echo "Formatting C/C++/ObjC++/Metal code..."

{ find include src -type f \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hh" -o -name "*.hpp" -o -name "*.mm" -o -name "*.metal" \) -print0 2>/dev/null; \
  find tests -type f \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) -not -path "*/llama.cpp/*" -print0 2>/dev/null; } | \
    while IFS= read -r -d '' file; do
        echo "  $file"
        clang-format -i "$file"
    done

echo "✓ Done"
