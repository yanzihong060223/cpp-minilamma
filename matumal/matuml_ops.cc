#include "matumal/matuml_ops.h"
#include "thread_pool/thread_pool.h"

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
MatulMode default_matul_mode = MatulMode::Native;

#ifdef MINI_LLAMA_USE_NEON
float DotMindNeon(const float* a, const float* b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i <= n - 4; i += 4) {
        const float32x4_t af = vld1q_f32(a + i);
        const float32x4_t bf = vld1q_f32(b + i);
        sum = vmlaq_f32(sum, af, bf);
    }
    const float32x2_t r =
        vadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    float result = vget_lane_f32(r, 0) + vget_lane_f32(r, 1);
    // 剩余不足四个的元素使用普通循环。
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif
/*| 函数 | 在这段代码里的作用 |
|---|---|
| `_mm256_setzero_ps()` | 生成 `[0,0,0,0,0,0,0,0]` |
| `_mm256_loadu_ps(p)` | 从 `p` 加载连续 8 个 float |
| `_mm256_fmadd_ps(a,b,c)` | 对应位置计算 `a*b+c` |
| `_mm256_castps256_ps128(v)` | 取得低半区的 4 个 float |
| `_mm256_extractf128_ps(v,1)` | 取得高半区的 4 个 float |
| `_mm_add_ps(a,b)` | 两个四元素向量，对应位置相加 |
| `_mm_hadd_ps(a,b)` | 分别把两个向量内部的相邻元素成对相加 |
| `_mm_cvtss_f32(v)` | 取出第 0 个位置，得到普通 `float` |*/
#ifdef MINI_LLAMA_USE_AVX2
float DotMindAvx2(const float* a, const float* b, int n) {
    __m256 sum = _mm256_setzero_ps();
    int i = 0;
    for (; i <= n - 8; i += 8) {
        const __m256 af = _mm256_loadu_ps(a + i);
        const __m256 bf = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(af, bf, sum);
    }
    __m128 low = _mm256_castps256_ps128(sum);
    const __m128 high = _mm256_extractf128_ps(sum, 1);
    low = _mm_add_ps(low, high);
    low = _mm_hadd_ps(low, low);
    low = _mm_hadd_ps(low, low);
    float result = _mm_cvtss_f32(low);
    // 剩余不足八个的元素使用普通循环。
    for (; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}
#endif
//按照寄存器分发
float DotMind(const float* a, const float* b, int n) {
#if defined(MINI_LLAMA_USE_NEON)
    return DotMindNeon(a, b, n);
#elif defined(MINI_LLAMA_USE_AVX2)
    return DotMindAvx2(a, b, n);
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}
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

Tensor LinearThread(const Tensor& a, const Tensor& b) {
    const bool is_vector = a.GetNumDims() == 1;
    const std::size_t in_size = static_cast<std::size_t>(b.shape[1]);
    const int out_size = b.shape[0];
    const std::vector<int> output_shape =
        is_vector ? std::vector<int>{out_size}
                  : std::vector<int>{1, out_size};
    CheckedNumel(output_shape, "LinearThread output");
    Tensor y(output_shape);

    ParallelFor(out_size, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const std::size_t weight_base =
                static_cast<std::size_t>(i) * in_size;
            float sum = 0.0f;
            for (std::size_t k = 0; k < in_size; ++k) {
                sum += a.data[k] * b.data[weight_base + k];
            }
            y.data[i] = sum;
        }
    });

    return y;
}
Tensor LinearSIMD(const Tensor& a, const Tensor& b) {
    const bool is_vector = a.GetNumDims() == 1;
    const int in_size = b.shape[1];
    const int out_size = b.shape[0];
    const std::vector<int> output_shape =
        is_vector ? std::vector<int>{out_size}
                  : std::vector<int>{1, out_size};
    CheckedNumel(output_shape, "LinearSIMD output");
    Tensor y(output_shape);
    for (int i = 0; i < out_size; ++i) {
        const std::size_t weight_base =
            static_cast<std::size_t>(i) * in_size;
        y.data[i] = DotMind(a.data.data(), b.data.data() + weight_base,
                            in_size);
    }
    return y;
}
Tensor LinearSIMDThread(const Tensor& a, const Tensor& b) {
    const bool is_vector = a.GetNumDims() == 1;
    const int in_size = b.shape[1];
    const int out_size = b.shape[0];
    const std::vector<int> output_shape =
        is_vector ? std::vector<int>{out_size}
                  : std::vector<int>{1, out_size};
    CheckedNumel(output_shape, "LinearSIMDThread output");
    Tensor y(output_shape);
    ParallelFor(out_size, [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            const std::size_t weight_base =
                static_cast<std::size_t>(i) * in_size;
            y.data[i] = DotMind(a.data.data(), b.data.data() + weight_base,
                                in_size);
        }
    });
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

Tensor MatmulThread(const Tensor& a, const Tensor& b) {
    const int m = a.shape[0];
    const std::size_t inner_size = static_cast<std::size_t>(a.shape[1]);
    const int n = b.shape[1];
    if (m > std::numeric_limits<int>::max() / n) {
        throw std::overflow_error("MatmulThread: too many output elements");
    }
    const int tasks = m * n;
    const std::vector<int> output_shape{m, n};
    CheckedNumel(output_shape, "MatmulThread output");
    Tensor y(output_shape);

    ParallelFor(tasks, [&](int begin, int end) {
        for (int index = begin; index < end; ++index) {
            const int row = index / n;
            const int col = index % n;
            const std::size_t row_base =
                static_cast<std::size_t>(row) * inner_size;
            float sum = 0.0f;
            for (std::size_t k = 0; k < inner_size; ++k) {
                sum += a.data[row_base + k] * b.data[k * n + col];
            }
            y.data[index] = sum;
        }
    });

    return y;
}

}  // namespace

MatulMode DefaultMatulMode() {
    return default_matul_mode;
}

void SetDefaultMatulMode(MatulMode mode) {
    switch (mode) {
        case MatulMode::Native:
        case MatulMode::Threaded:
        case MatulMode::SIMD:
        case MatulMode::ThreadedSIMD:
            default_matul_mode = mode;
            return;
        default:
            throw std::invalid_argument("SetDefaultMatulMode: invalid MatulMode");
    }
}

Tensor LinearDispatch(const Tensor& a, const Tensor& b, MatulMode mode) {
    CheckLinearInputs(a, b);
    switch (mode) {
        case MatulMode::Native:
            return LinearNative(a, b);
        case MatulMode::Threaded:
            return LinearThread(a, b);
        case MatulMode::SIMD:
            return LinearSIMD(a, b);
        case MatulMode::ThreadedSIMD:
            return LinearSIMDThread(a, b);
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
            return MatmulThread(a, b);
        case MatulMode::SIMD:
        case MatulMode::ThreadedSIMD:
            throw std::runtime_error(
                "MatmulDispatch: SIMD and ThreadedSIMD are not implemented");
        default:
            throw std::invalid_argument("MatmulDispatch: invalid MatulMode");
    }
}

}  // namespace yan_lamma
