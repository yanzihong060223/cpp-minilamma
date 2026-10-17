#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/../.." && pwd)"
build_dir="$project_dir/build-tiny-model"
mkdir -p "$build_dir"
cd "$project_dir"

# These inference sources exist but are not all listed in the root CMake target.
# Compile the current sources directly, so a stale library cannot hide a change.
"${CXX:-g++}" -std=c++17 -O2 -I "$project_dir" \
    "$script_dir/smoke_test.cc" \
    src/tensor.cc src/load.cc src/quantiezd_tensor.cc \
    src/kvcache.cc src/context.cc src/batch.cc src/forward.cc src/sampler.cc \
    src/ops.cc src/tonizer.cc src/jsonvocab.cc src/bpevocab.cc \
    matumal/matuml_ops.cc \
    -o "$build_dir/tiny-model-smoke"

"$build_dir/tiny-model-smoke" "$script_dir" "${1-hello}" "${2-16}"
