#pragma once
#include "include/tensor.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace yan_lamma::test {
constexpr float kAbsTol = 2e-3f;
constexpr float kRelTol = 2e-3f;
inline void Require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}
inline bool CloseEnough(float actual, float expected) {
    if (!std::isfinite(actual) || !std::isfinite(expected)) return false;
    const float abs_err = std::abs(actual - expected);
    return abs_err <= kAbsTol || abs_err / std::max(1.0f, std::abs(expected)) <= kRelTol;
}
inline void RequireCloseTensor(const Tensor& actual, const Tensor& expected, const std::string& label) {
    Require(actual.shape == expected.shape, label + ": shape actual=" + actual.ShapeToString() +
            " expected=" + expected.ShapeToString());
    Require(actual.data.size() == actual.GetDataSize() && actual.data.size() == expected.data.size(),
            label + ": storage length mismatch");
    size_t elements = 1;
    for (int dim : expected.shape) {
        Require(dim > 0 && elements <= std::numeric_limits<size_t>::max() / size_t(dim), label + ": invalid shape");
        elements *= size_t(dim);
    }
    Require(expected.data.size() == elements, label + ": shape/storage mismatch");
    for (size_t i = 0; i < actual.data.size(); ++i) {
        if (!CloseEnough(actual.data[i], expected.data[i])) {
            std::ostringstream message;
            message << std::setprecision(std::numeric_limits<float>::max_digits10)
                    << label << ": index=" << i << " actual=" << actual.data[i]
                    << " expected=" << expected.data[i];
            throw std::runtime_error(message.str());
        }
    }
}
template<class F> void RequireError(F operation, const std::string& text) {
    bool caught = false;
    try { operation(); }
    catch (const std::runtime_error& error) {
        caught = true;
        Require(std::string(error.what()).find(text) != std::string::npos,
                "wrong error: " + std::string(error.what()) + ", expected " + text);
    }
    Require(caught, "expected error containing: " + text);
}
}
