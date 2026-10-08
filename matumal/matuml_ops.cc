#include "matumal/matuml_ops.h"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// 本次编译启用了 ARM NEON
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MINI_LLAMA_USE_NEON 1
#endif

// 本次编译启用了 AVX2 和 FMA
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define MINI_LLAMA_USE_AVX2 1
#endif

namespace yan_lamma {
namespace {

std::size_t CheckedNumel(const std::vector<int>& shape, const char* caller) {
    std::size_t total = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument(std::string(caller) +
                                        ": dimensions must be positive");
        }
        const std::size_t dimension = static_cast<std::size_t>(dim);
        if (total > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error(std::string(caller) +
                                      ": shape element count overflow");
        }
        total *= dimension;
    }
    return total;
}

void CheckStorage(const Tensor& tensor, const char* caller) {
    if (CheckedNumel(tensor.shape, caller) != tensor.data.size()) {
        throw std::invalid_argument(std::string(caller) +
                                    ": data size does not match shape");
    }
}

void CheckLinearInputs(const Tensor& a, const Tensor& b) {
    if ((a.GetNumDims() != 1 && a.GetNumDims() != 2) ||
        b.GetNumDims() != 2) {
        throw std::invalid_argument(
            "LinearDispatch: input must be [in] or [1, in], "
            "and weight must be [out, in]");
    }
    if (a.GetNumDims() == 2 && a.shape[0] != 1) {
        throw std::invalid_argument(
            "LinearDispatch: 2D input must contain exactly one row");
    }
    CheckStorage(a, "LinearDispatch input");
    CheckStorage(b, "LinearDispatch weight");
    if (a.shape.back() != b.shape[1]) {
        throw std::invalid_argument(
            "LinearDispatch: input size does not match weight input size");
    }
}

void CheckMatmulInputs(const Tensor& a, const Tensor& b) {
    if (a.GetNumDims() != 2 || b.GetNumDims() != 2) {
        throw std::invalid_argument("MatmulDispatch: inputs must be 2D");
    }
    CheckStorage(a, "MatmulDispatch left input");
    CheckStorage(b, "MatmulDispatch right input");
    if (a.shape[1] != b.shape[0]) {
        throw std::invalid_argument(
            "MatmulDispatch: left columns must equal right rows");
    }
}

Tensor LinearNative(const Tensor& a, const Tensor& b) {
    const bool is_vector = a.GetNumDims() == 1;
    const std::size_t in_size = static_cast<std::size_t>(b.shape[1]);
    const std::size_t out_size = static_cast<std::size_t>(b.shape[0]);
    const std::vector<int> output_shape =
        is_vector ? std::vector<int>{b.shape[0]}
                  : std::vector<int>{1, b.shape[0]};
    CheckedNumel(output_shape, "LinearNative output");
    Tensor y(output_shape);
    for (std::size_t i = 0; i < out_size; ++i) {
        float sum = 0.0f;
        for (std::size_t k = 0; k < in_size; ++k) {
            sum += a.data[k] * b.data[i * in_size + k];
        }
        y.data[i] = sum;
    }
    return y;
}

Tensor MatmulNative(const Tensor& a, const Tensor& b) {
    const std::size_t m = static_cast<std::size_t>(a.shape[0]);
    const std::size_t inner_size = static_cast<std::size_t>(a.shape[1]);
    const std::size_t n = static_cast<std::size_t>(b.shape[1]);
    const std::vector<int> output_shape{a.shape[0], b.shape[1]};
    CheckedNumel(output_shape, "MatmulNative output");
    Tensor y(output_shape);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (std::size_t k = 0; k < inner_size; ++k) {
                sum += a.data[i * inner_size + k] * b.data[k * n + j];
            }
            y.data[i * n + j] = sum;
        }
    }
    return y;
}

}  // namespace

MatulMode DefaultMatulMode() {
    return MatulMode::Native;
}

Tensor LinearDispatch(const Tensor& a, const Tensor& b, MatulMode mode) {
    CheckLinearInputs(a, b);
    switch (mode) {
        case MatulMode::Native:
            return LinearNative(a, b);
        case MatulMode::Threaded:
        case MatulMode::SIMD:
        case MatulMode::ThreadedSIMD:
            throw std::runtime_error(
                "LinearDispatch: Threaded, SIMD and ThreadedSIMD are not implemented");
        default:
            throw std::invalid_argument("LinearDispatch: invalid MatulMode");
    }
}

Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatulMode mode) {
    CheckMatmulInputs(a, b);
    switch (mode) {
        case MatulMode::Native:
            return MatmulNative(a, b);
        case MatulMode::Threaded:
        case MatulMode::SIMD:
        case MatulMode::ThreadedSIMD:
            throw std::runtime_error(
                "MatmulDispatch: Threaded, SIMD and ThreadedSIMD are not implemented");
        default:
            throw std::invalid_argument("MatmulDispatch: invalid MatulMode");
    }
}

}  // namespace yan_lamma
