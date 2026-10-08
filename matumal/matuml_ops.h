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

// input [in] -> [out], or [1, in] -> [1, out]; weight [out, in].
// Only Native is implemented; other execution modes throw an exception.
Tensor LinearDispatch(const Tensor& a, const Tensor& b, MatulMode mode);

// [m, k] * [k, n] -> [m, n]. Only Native is implemented.
Tensor MatmulDispatch(const Tensor& a, const Tensor& b, MatulMode mode);

}  // namespace yan_lamma
