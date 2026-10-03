#include "include/ops.h"
#include "include/tensor.h"

#include <stdexcept>
#include <cmath>
namespace yan_lamma {
namespace {
    const char* CallerName(const char* call) {
        return call == nullptr?  "Tensor" : call;
    }
    void CheckDims(const Tensor& T, std::size_t expected, const char* caller) {
        if (T.GetNumDims() != expected) {
              throw std::runtime_error(std::string(CallerName(caller)) + ": expected " +
                             std::to_string(expected) + "D tensor, got " + T. ShapeToString()
                            );
        }
    }
}
Tensor ElementwiseMul(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("No Match Shape");
    }
    Tensor y (a.shape);
    int n = a.GetDataSize();
    for (int i = 0; i < n; i++) {
        y.data[i] = a.data[i] * b.data[i];
    }
    return y;
}
size_t ArgMax(const Tensor& a) {
    const std::size_t n = a.GetDataSize();
    if (n == 0) {
        throw std::invalid_argument("ArgMax: input must not be empty");
    }
    float flag = a.data[0];
    std::size_t res = 0;
    for (std::size_t i = 1; i < n; i++) {
        if (a.data[i] > flag) {
            flag = a.data[i];
            res = i;
        }
    }
    return res;
}
Tensor Matmul(const Tensor& a, const Tensor& b,  MatulMode mode) {
    CheckDims(a, 2, "Matmul");
    CheckDims(b, 2, "Matmul");
    return MatmulDispatch(a, b, mode);
}

Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps) {
    CheckDims(x, 1, "RmsNorm");
    CheckDims(weight, 1, "RmsNorm");
    if (x.shape[0] <= 0 || x.shape[0] != weight.shape[0]) {
        throw std::invalid_argument("RmsNorm: expected matching positive dimensions");
    }
    const std::size_t dim = static_cast<std::size_t>(x.shape[0]);
    if (x.data.size() != dim || weight.data.size() != dim) {
        throw std::invalid_argument("RmsNorm: data size does not match shape");
    }
    if (!std::isfinite(eps) || eps < 0.0f) {
        throw std::invalid_argument("RmsNorm: eps must be finite and non-negative");
    }
    double ss = 0.0;
    for (std::size_t i = 0; i < dim; i++) {
        const double value = static_cast<double>(x.data[i]);
        ss += value * value;
    }
    const double denominator = std::sqrt(
        ss / static_cast<double>(dim) + static_cast<double>(eps));
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        throw std::invalid_argument("RmsNorm: normalization denominator must be finite and positive");
    }
    const double scale = 1.0 / denominator;
    Tensor y(x.shape);
    for (std::size_t i = 0; i < dim; i++) {
        y.data[i] = static_cast<float>(static_cast<double>(x.data[i]) *
                                       static_cast<double>(weight.data[i]) * scale);
    }
    return y;
}
Tensor SoftMax(const Tensor& a) {
    if (a.GetNumDims() != 1) {
        throw std::runtime_error("SoftMax Rand Not Match");
    }
    int n = a.shape[0];
    float max_data = a.data[0];
    for (int i = 1; i < n; i++) {
        if (a.data[i] > max_data) {
            max_data = a.data[i];
        }
    }
    float sum = 0.0f;
    Tensor y({n});
    for (int i = 0; i < n; i++) {
        y.data[i] = std::exp(a.data[i] - max_data);
        sum += y.data[i];
    }
    for (int i = 0; i < n; i++) {
        y.data[i] /= sum;
    }
    return y;
}
Tensor Silu(const Tensor& a) {
    size_t total = a.GetDataSize();
    Tensor y(a.shape);
    for (size_t i = 0; i < total; i++) {
        float xv = a.data[i];
        float sig = 1.0f /(1.0f + std::exp(-xv));
        y.data[i] = xv * sig;
    }
    return y;
}
Tensor SwiGlu(const Tensor& gate, const Tensor& up) {
    if (gate.shape != up.shape) {
        throw std::runtime_error("Gate/Up Not Equal");
    }
    Tensor gate_silu = Silu(gate);
    return  ElementwiseMul(gate_silu, up);
}
Tensor Linear(const Tensor& a, const Tensor& b, MatulMode mode) {
    return LinearDispatch(a, b, mode);
}
}//namespace yan_lamma
