#pragma once
#include "include/tensor.h"

#include <vector>
#include <stdexcept>
#include <string>
#include <limits>
namespace yan_lamma {
enum class QuantType {
  kF32,
  kQ80,
  kQ40,
  kQ41,
};
struct QuanTensor {
//默认kf32
QuantType type = QuantType::kF32;
std::vector<int> shape;
std::vector<float> kF32_data;
size_t GetNumSize() const {
    size_t res = 1;
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::runtime_error("QuanTensor: dimensions must be positive");
        }
        const size_t d = static_cast<size_t>(dim);
        if (res > std::numeric_limits<size_t>::max() / d) {
            throw std::runtime_error("QuanTensor: element count overflow");
        }
        res *= d;
    }
    return res;
}
size_t GetDim() const {
    return shape.size();
}
std::string ShapeToString(const std::vector<int>& arr) const {
    std::string s = "[";
        for (std::size_t i = 0; i < arr.size(); i++) {
                if ( i > 0) {
                    s += ", ";
                }
                s += std::to_string(arr[i]);
        }
        s += "]";
        return s;
}
void AssertShape(const std::vector<int>& expected, const char* call) {
    if (shape != expected) {
         throw std::runtime_error(std::string(std::string(call)) +
                                 ": expected " + ShapeToString(expected) +
                                 ", got " + ShapeToString(shape));
    }
}
};
Tensor ToTensor(const  QuanTensor& q);
QuanTensor ToQuanTensor(const Tensor& t);
}