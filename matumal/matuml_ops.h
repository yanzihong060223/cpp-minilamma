#pragma once
#include "include/tensor.h"

namespace yan_lamma {
enum class MatulMode {
    Native, //普通循环
    Threaded, //多个线程分别计算不同的输出元素
    SIMD, // CPU 一条指令同时处理多个元素
    ThreadedSIMD
};

MatulMode DefaultMatulMode();
// Configure before starting inference; the process starts in Native mode.
// Do not change the mode while inference is running.
void SetDefaultMatulMode(MatulMode mode);

// input [in] -> [out], or [1, in] -> [1, out]; weight [out, in].
// All modes are implemented. SIMD uses NEON or AVX2/FMA when enabled for
// the compilation target, and otherwise falls back to a scalar dot product.
Tensor LinearDispatch(const Tensor& a, const Tensor& b, MatulMode mode);

// [m, k] * [k, n] -> [m, n]. Native and Threaded are implemented.
// SIMD modes throw an exception.
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatulMode mode);

}  // namespace yan_lamma
