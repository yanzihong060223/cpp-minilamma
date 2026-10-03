#include "include/ops.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using yan_lamma::MatulMode;
using yan_lamma::Tensor;

std::size_t assertions = 0;

void Require(bool condition, const std::string& message) {
    ++assertions;
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void Near(double actual, double expected, const std::string& message,
          double tolerance = 2e-5) {
    Require(std::isfinite(actual) &&
                std::abs(actual - expected) <=
                    tolerance * std::max(1.0, std::abs(expected)),
            message + ": expected " + std::to_string(expected) +
                ", got " + std::to_string(actual));
}

template <typename Function>
void Throws(Function&& function, const std::string& message) {
    ++assertions;
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message + ": expected an exception");
}

Tensor MakeTensor(std::vector<int> shape, std::vector<float> values) {
    Tensor result(shape);
    Require(result.data.size() == values.size(), "test data length");
    result.data = std::move(values);
    return result;
}

// Public fields can be corrupted after construction. These cases must be
// rejected by the operator before it reads or writes an element.
Tensor Malformed(std::vector<int> shape, std::vector<float> values = {}) {
    Tensor result;
    result.shape = std::move(shape);
    result.data = std::move(values);
    return result;
}

void Values(const Tensor& result, const std::vector<int>& shape,
            const std::vector<double>& expected, const std::string& name) {
    Require(result.shape == shape, name + " shape");
    Require(result.data.size() == expected.size(), name + " data length");
    for (std::size_t i = 0; i < expected.size(); ++i) {
        Near(result.data[i], expected[i], name + " element " + std::to_string(i));
    }
}

void TestArgMax() {
    Require(yan_lamma::ArgMax(MakeTensor({3}, {1, 5, 3})) == 1,
            "ArgMax updates the running maximum");
    Require(yan_lamma::ArgMax(MakeTensor({4}, {1, 5, 5, 3})) == 1,
            "ArgMax keeps the first maximum");
    Require(yan_lamma::ArgMax(MakeTensor({3}, {-9, -2, -4})) == 1,
            "ArgMax handles negative values");
    Throws([] { yan_lamma::ArgMax(Tensor{}); }, "ArgMax rejects empty input");
}

void CheckRmsReference(const std::vector<float>& values,
                       const std::vector<float>& weights, float eps,
                       const std::string& name) {
    const int dim = static_cast<int>(values.size());
    const Tensor result = yan_lamma::RmsNorm(MakeTensor({dim}, values),
                                            MakeTensor({dim}, weights), eps);
    double squared_sum = 0.0;
    for (float value : values) {
        const double promoted = value;
        squared_sum += promoted * promoted;
    }
    const double rms = std::sqrt(squared_sum / values.size() + eps);
    Require(result.shape == std::vector<int>{dim}, name + " shape");
    for (std::size_t i = 0; i < values.size(); ++i) {
        const double expected = static_cast<double>(values[i]) / rms * weights[i];
        Near(result.data[i], expected, name + " element " + std::to_string(i));
    }
}

void TestRmsNorm() {
    Values(yan_lamma::RmsNorm(MakeTensor({2}, {3, 4}),
                              MakeTensor({2}, {1, 1}), 0.0f),
           {2}, {0.8485281374, 1.1313708499}, "RmsNorm known example");
    CheckRmsReference({-2, 0.25f, 3.5f}, {0.5f, 2, -1}, 1e-5f,
                      "RmsNorm double reference");
    const float largest = std::numeric_limits<float>::max();
    CheckRmsReference({largest, -largest}, {1, 1}, 0,
                      "RmsNorm avoids float square overflow");
    const float smallest = std::numeric_limits<float>::denorm_min();
    CheckRmsReference({smallest, 2 * smallest}, {1, 1}, 0,
                      "RmsNorm avoids float square underflow");
    Values(yan_lamma::RmsNorm(MakeTensor({2}, {0, 0}),
                              MakeTensor({2}, {1, 1}), 1e-5f),
           {2}, {0, 0}, "RmsNorm zero input with positive eps");
    const Tensor x = MakeTensor({2}, {3, 4});
    const Tensor weight = MakeTensor({2}, {1, 1});
    Throws([&] { yan_lamma::RmsNorm(MakeTensor({2}, {0, 0}), weight, 0); },
           "RmsNorm rejects a zero denominator");
    for (float eps : {-1.0f, std::numeric_limits<float>::quiet_NaN(),
                      std::numeric_limits<float>::infinity()}) {
        Throws([&] { yan_lamma::RmsNorm(x, weight, eps); },
               "RmsNorm rejects invalid eps");
    }
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
        Throws([&] { yan_lamma::RmsNorm(MakeTensor({2}, {value, 1}), weight, 0); },
               "RmsNorm rejects a nonfinite denominator");
    }
    const std::vector<Tensor> invalid = {
        Tensor{}, Malformed({1, 2}, {1, 2}), Malformed({0}),
        Malformed({-2}), Malformed({2}, {1}), Malformed({2}, {1, 2, 3}),
        MakeTensor({3}, {1, 2, 3})};
    for (std::size_t i = 0; i < invalid.size(); ++i) {
        Throws([&] { yan_lamma::RmsNorm(invalid[i], weight, 1e-5f); },
               "RmsNorm rejects invalid input " + std::to_string(i));
        Throws([&] { yan_lamma::RmsNorm(x, invalid[i], 1e-5f); },
               "RmsNorm rejects invalid weight " + std::to_string(i));
    }
}

void TestLinearAndMatmul() {
    const Tensor x = MakeTensor({3}, {1, 2, 3});
    const Tensor weight = MakeTensor({2, 3}, {1, 0, 1, 0, 1, 1});
    Values(yan_lamma::Linear(x, weight), {2}, {4, 5}, "Linear default Native");
    Values(yan_lamma::LinearDispatch(x, weight, MatulMode::Native),
           {2}, {4, 5}, "LinearDispatch Native");
    Values(yan_lamma::Linear(MakeTensor({1, 3}, {1, 2, 3}), weight),
           {1, 2}, {4, 5}, "Linear one-row input");
    Values(yan_lamma::LinearDispatch(MakeTensor({1, 3}, {1, 2, 3}), weight,
                                    MatulMode::Native),
           {1, 2}, {4, 5}, "LinearDispatch one-row input");
    const Tensor matrix = MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6});
    const Tensor b = MakeTensor({3, 2}, {1, 0, 0, 1, 1, 1});
    Values(yan_lamma::Matmul(matrix, b), {2, 2}, {4, 5, 10, 11},
           "Matmul non-square inputs");
    Values(yan_lamma::MatmulDispatch(matrix, b, MatulMode::Native),
           {2, 2}, {4, 5, 10, 11}, "MatmulDispatch Native");
    Require(yan_lamma::DefaultMatulMode() == MatulMode::Native,
            "DefaultMatulMode stays Native");
}

void TestInvalidMatrices() {
    const Tensor matrix = MakeTensor({2, 2}, {1, 2, 3, 4});
    const std::vector<Tensor> invalid_matrices = {
        Tensor{}, Malformed({}, {1}), MakeTensor({2}, {1, 2}),
        MakeTensor({1, 1, 2}, {1, 2}), Malformed({0, 2}), Malformed({2, 0}),
        Malformed({-1, 2}), Malformed({2, -1}),
        Malformed({2, 2}, {1, 2, 3}), Malformed({2, 2}, {1, 2, 3, 4, 5})};
    for (std::size_t i = 0; i < invalid_matrices.size(); ++i) {
        const Tensor& bad = invalid_matrices[i];
        for (bool dispatch : {false, true}) {
            const auto multiply = [dispatch](const Tensor& a, const Tensor& b) {
                return dispatch ? yan_lamma::MatmulDispatch(a, b, MatulMode::Native)
                                : yan_lamma::Matmul(a, b);
            };
            Throws([&] { multiply(bad, matrix); }, "Matmul rejects invalid lhs");
            Throws([&] { multiply(matrix, bad); }, "Matmul rejects invalid rhs");
        }
        Throws([&] { yan_lamma::Linear(MakeTensor({2}, {1, 2}), bad); },
               "Linear rejects invalid weights");
        Throws([&] { yan_lamma::LinearDispatch(MakeTensor({2}, {1, 2}), bad,
                                                MatulMode::Native); },
               "LinearDispatch rejects invalid weights");
    }
    const std::vector<Tensor> invalid_inputs = {
        Tensor{}, Malformed({}, {1}), MakeTensor({1, 1, 2}, {1, 2}),
        Malformed({0}), Malformed({-2}), Malformed({0, 2}), Malformed({2, 0}),
        Malformed({-1, 2}), Malformed({2, -1}),
        Malformed({2}, {1}), Malformed({2}, {1, 2, 3}),
        Malformed({2, 2}, {1, 2, 3}), Malformed({2, 2}, {1, 2, 3, 4, 5}),
        MakeTensor({3}, {1, 2, 3}), MakeTensor({2, 3}, {1, 2, 3, 4, 5, 6}),
        MakeTensor({2, 2}, {1, 2, 3, 4})};
    for (const Tensor& bad : invalid_inputs) {
        Throws([&] { yan_lamma::Linear(bad, matrix); },
               "Linear rejects invalid input");
        Throws([&] { yan_lamma::LinearDispatch(bad, matrix, MatulMode::Native); },
               "LinearDispatch rejects invalid input");
    }
    const Tensor mismatch = MakeTensor({3, 2}, {1, 2, 3, 4, 5, 6});
    Throws([&] { yan_lamma::Matmul(matrix, mismatch); },
           "Matmul rejects inner-dimension mismatch");
    Throws([&] { yan_lamma::MatmulDispatch(matrix, mismatch, MatulMode::Native); },
           "MatmulDispatch rejects inner-dimension mismatch");
    const Tensor vector = MakeTensor({2}, {1, 2});
    for (MatulMode mode : {MatulMode::Threaded, MatulMode::SIMD,
                           MatulMode::ThreadedSIMD, static_cast<MatulMode>(999)}) {
        Throws([&] { yan_lamma::Linear(vector, matrix, mode); },
               "Linear rejects unsupported/invalid mode");
        Throws([&] { yan_lamma::LinearDispatch(vector, matrix, mode); },
               "LinearDispatch rejects unsupported/invalid mode");
        Throws([&] { yan_lamma::Matmul(matrix, matrix, mode); },
               "Matmul rejects unsupported/invalid mode");
        Throws([&] { yan_lamma::MatmulDispatch(matrix, matrix, mode); },
               "MatmulDispatch rejects unsupported/invalid mode");
    }
}

void TestRandomReferences() {
    std::mt19937 random(20261003);
    std::uniform_int_distribution<int> dimension(1, 8);
    std::uniform_real_distribution<float> value(-3.0f, 3.0f);
    for (int trial = 0; trial < 60; ++trial) {
        const int m = dimension(random);
        const int k = dimension(random);
        const int n = dimension(random);
        Tensor a({m, k});
        Tensor b({k, n});
        for (float& element : a.data) { element = value(random); }
        for (float& element : b.data) { element = value(random); }
        const Tensor result = yan_lamma::Matmul(a, b);
        Require(result.shape == std::vector<int>({m, n}), "random Matmul shape");
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                double expected = 0;
                for (int inner = 0; inner < k; ++inner) {
                    expected += static_cast<double>(a.At2(row, inner)) *
                                b.At2(inner, col);
                }
                Near(result.At2(row, col), expected, "random Matmul double reference");
            }
        }
        Tensor weights({n, k});
        for (float& element : weights.data) { element = value(random); }
        Tensor row_input({1, k});
        Tensor vector_input({k});
        for (int inner = 0; inner < k; ++inner) {
            row_input.At2(0, inner) = a.At2(0, inner);
            vector_input.At1(inner) = a.At2(0, inner);
        }
        const Tensor row_result = yan_lamma::Linear(row_input, weights);
        const Tensor vector_result = yan_lamma::Linear(vector_input, weights);
        Require(row_result.shape == std::vector<int>({1, n}), "random 2D Linear shape");
        Require(vector_result.shape == std::vector<int>{n}, "random 1D Linear shape");
        for (int output = 0; output < n; ++output) {
            double expected = 0;
            for (int inner = 0; inner < k; ++inner) {
                expected += static_cast<double>(row_input.At2(0, inner)) *
                            weights.At2(output, inner);
            }
            Near(row_result.At2(0, output), expected,
                 "random one-row Linear double reference");
            Near(row_result.At2(0, output), vector_result.At1(output),
                 "one-row Linear matches one-dimensional Linear");
        }
    }
}

void TestUnchangedOperators() {
    Values(yan_lamma::ElementwiseMul(MakeTensor({2}, {2, -3}),
                                     MakeTensor({2}, {4, 5})),
           {2}, {8, -15}, "ElementwiseMul smoke");
    const Tensor softmax = yan_lamma::SoftMax(MakeTensor({3}, {2, 1, 0}));
    const double sum = std::exp(2.0) + std::exp(1.0) + 1.0;
    Values(softmax, {3}, {std::exp(2.0) / sum, std::exp(1.0) / sum, 1.0 / sum},
           "SoftMax smoke");
    const Tensor gate = MakeTensor({3}, {-2, 0, 2});
    Values(yan_lamma::Silu(gate), {3},
           {-2.0 / (1.0 + std::exp(2.0)), 0, 2.0 / (1.0 + std::exp(-2.0))},
           "Silu smoke");
    Values(yan_lamma::SwiGlu(gate, MakeTensor({3}, {3, 4, 5})), {3},
           {-6.0 / (1.0 + std::exp(2.0)), 0, 10.0 / (1.0 + std::exp(-2.0))},
           "SwiGlu smoke");
}
}  // namespace

int main() {
    const std::vector<std::pair<const char*, std::function<void()>>> tests = {
        {"ArgMax", TestArgMax}, {"RmsNorm", TestRmsNorm},
        {"Linear and Matmul", TestLinearAndMatmul},
        {"Invalid inputs and modes", TestInvalidMatrices},
        {"Fixed-seed random references", TestRandomReferences},
        {"Unchanged operators", TestUnchangedOperators}};
    int failures = 0;
    for (const auto& test : tests) {
        try {
            test.second();
            std::cout << "PASS " << test.first << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << test.first << ": " << error.what() << '\n';
        }
    }
    std::cout << assertions << " assertions, " << failures << " failed groups\n";
    return failures == 0 ? 0 : 1;
}
