#pragma once
#include "include/tensor.h"

#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include <limits>
namespace yan_lamma {
constexpr size_t kQuantBlockSize   = 32;
constexpr size_t kBytesPerQ80Block = 34;  // fp16 scale + 32 int8
constexpr size_t kBytesPerQ40Block = 18;  // fp16 scale + 16 nibble bytes
constexpr size_t kBytesPerQ41Block = 20;  // fp16 scale + fp16 min + 16 nibble bytes
enum class QuantType {
  kF32,
  kF16,
  kQ80,
  kQ40,
  kQ41,
};
struct Block_Q80 {
    //scale
    uint16_t d;
    int8_t qs [32];
};
struct Block_Q40 {
    //scale
    uint16_t d;
    uint8_t qs[16];
};
struct Block_Q41 {
    uint16_t d;
    uint16_t m;
    uint8_t qs[16];
};
static_assert(sizeof(Block_Q80) == sizeof(uint16_t) + kQuantBlockSize, "invailed block q80 size" );
struct QuanTensor {
//默认kf32
QuantType type = QuantType::kF32;
std::vector<int> shape;
std::vector<float> kF32_data;
std::vector<uint8_t> kF16_data;
std::vector<uint8_t> kQ80_data;
std::vector<uint8_t> kQ40_data;
std::vector<uint8_t> kQ41_data;
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
Tensor Q80ToTensor(const QuanTensor& q);
Tensor Q40ToTensor(const QuanTensor& q);
Tensor Q41ToTensor(const QuanTensor& q);
}
