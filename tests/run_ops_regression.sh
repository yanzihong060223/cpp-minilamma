#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$(mktemp -d /tmp/yan-minillama-ops-test.XXXXXX)"
trap 'rm -rf -- "$build_dir"' EXIT

sources=(src/tensor.cc src/ops.cc matumal/matuml_ops.cc tests/ops_regression.cc)
common_flags=(-std=c++17 -Wall -Wextra -Wpedantic -Werror -I "$project_root")

for compiler in g++ clang++; do
    if ! command -v "$compiler" >/dev/null 2>&1; then
        printf 'Required compiler is unavailable: %s\n' "$compiler" >&2
        exit 1
    fi
    for variant in regular sanitizers; do
        printf '\n%s: %s\n' "$compiler" "$variant"
        flags=(-O2)
        if [[ "$variant" == sanitizers ]]; then
            flags=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
        fi
        objects=()
        for source in "${sources[@]}"; do
            object="$build_dir/${compiler}-${variant}-$(basename -- "$source").o"
            "$compiler" "${common_flags[@]}" "${flags[@]}" \
                -c "$project_root/$source" -o "$object"
            objects+=("$object")
        done
        binary="$build_dir/${compiler}-${variant}"
        "$compiler" "${common_flags[@]}" "${flags[@]}" "${objects[@]}" -o "$binary"
        ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
            UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 "$binary"
    done
done
