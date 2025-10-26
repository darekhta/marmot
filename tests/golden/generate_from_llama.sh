#!/bin/bash
set -e

echo "Cloning llama.cpp..."
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi

echo "Building llama.cpp..."
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=OFF
cmake --build build -j8 --target ggml

cd ..

echo "Ensuring Marmot library..."
ninja -C ../../build libmarmot.dylib

echo "Compiling generator..."
c++ -o gen_golden_from_llama gen_golden_from_llama.c \
   -I llama.cpp/ggml/include \
   -I llama.cpp/ggml/src \
   -I ../../include \
   -L../../build -lmarmot -Wl,-rpath,@executable_path/../../build \
   llama.cpp/build/ggml/src/libggml-base.a \
   llama.cpp/build/ggml/src/libggml-cpu.a \
   -framework Accelerate -framework Metal -framework Foundation -lpthread -lm -lc++

echo "Generating golden quantization, matmul, and vec_dot data..."
./gen_golden_from_llama

echo "Cleaning up llama.cpp..."
rm -rf llama.cpp

echo "Done!"
echo "  - tests/backend/golden_quant_llama.h"
echo "  - tests/backend/golden_vec_dot_llama.h"
echo "  - tests/backend/golden_matmul_llama.h"
echo "  - tests/backend/golden_float_ops_llama.h"
echo "  - tests/golden/embedding_q4_0.txt"
echo "  - tests/golden/embedding_q4_1.txt"
echo "  - tests/golden/embedding_q5_0.txt"
echo "  - tests/golden/embedding_q5_1.txt"
echo "  - tests/golden/embedding_q8_0.txt"
echo "  - tests/golden/embedding_q8_1.txt"
echo "  - tests/golden/embedding_q4_0_ragged.txt"
