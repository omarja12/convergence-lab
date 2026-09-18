#!/usr/bin/env sh
# Build without CMake. g++ or clang++, nothing else required.
set -e
CXX="${CXX:-g++}"
mkdir -p build
$CXX -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude src/main.cpp   -o build/convergence_demo
$CXX -std=c++20 -O2 -Wall -Wextra -Wpedantic -Iinclude tests/test_convergence.cpp -o build/convergence_tests
echo "built build/convergence_demo and build/convergence_tests"
