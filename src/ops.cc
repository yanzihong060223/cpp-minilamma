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

void Rope(Tensor& q, Tensor& k, int pos, float theta, RopeType rope_type) {
    // q: [n_heads, head_dim]，k: [n_kv_heads, head_dim]。
    CheckDims(q, 2, "Rope q");
    CheckDims(k, 2, "Rope k");
    if (pos < 0) {
        throw std::out_of_range("Rope: position must be non-negative");
    }
    if (!std::isfinite(theta) || theta <= 0.0f) {
        throw std::runtime_error("Rope: theta must be finite and positive");
    }
    const int head_dim = q.shape[1];
    if (q.shape[0] <= 0 || k.shape[0] <= 0 || head_dim <= 0 ||
        head_dim % 2 != 0 || k.shape[1] != head_dim) {
        throw std::runtime_error("Rope: expected positive heads and matching positive even head_dim");
    }
    const size_t dim = static_cast<size_t>(head_dim);
    if (q.data.size() % dim != 0 ||
        q.data.size() / dim != static_cast<size_t>(q.shape[0]) ||
        k.data.size() % dim != 0 ||
        k.data.size() / dim != static_cast<size_t>(k.shape[0])) {
        throw std::runtime_error("Rope: data size does not match shape");
    }

    const int half_dim = head_dim / 2;
    for (int i = 0; i < half_dim; ++i) {
        const double angle = static_cast<double>(pos) *
            std::pow(static_cast<double>(theta), -2.0 * i / head_dim);
        const double cos_value = std::cos(angle);
        const double sin_value = std::sin(angle);
        // Normal 旋转相邻两个元素；NeoX 旋转前、后半区的对应元素。
        const size_t first = rope_type == RopeType::kNormal
            ? static_cast<size_t>(i) * 2 : static_cast<size_t>(i);
        const size_t second = rope_type == RopeType::kNormal
            ? first + 1 : first + static_cast<size_t>(half_dim);
        for (int tensor_index = 0; tensor_index < 2; ++tensor_index) {
            Tensor& tensor = tensor_index == 0 ? q : k;
            for (int head = 0; head < tensor.shape[0]; ++head) {
                const size_t base = static_cast<size_t>(head) * dim;
                const double x0 = tensor.data[base + first];
                const double x1 = tensor.data[base + second];
                tensor.data[base + first] =
                    static_cast<float>(x0 * cos_value - x1 * sin_value);
                tensor.data[base + second] =
                    static_cast<float>(x0 * sin_value + x1 * cos_value);
            }
        }
    }
}
}//namespace yan_lamma
