#include "include/quantized_tensor.h"

#include <stdexcept>
namespace yan_lamma {
Tensor ToTensor(const  QuanTensor& q) {
    if (q.type !=  QuantType::kF32) {
        throw std::runtime_error(" ToTensor: expected F32 QuantizedTensor, got quantized type");
    }
    if (q.shape.empty() && q.kF32_data.empty()) {
        return Tensor();
    }
    if (q.GetNumSize() != q.kF32_data.size()) {
        throw std::runtime_error("ToTensor: shape and F32 data size mismatch");
    }
    Tensor t (q.shape);
    t.data = q.kF32_data;
    return t;

}
QuanTensor ToQuanTensor(const Tensor& t) {
     QuanTensor q;
     q.type = QuantType::kF32;
     q.shape = t.shape;
     if ((!t.shape.empty() || !t.data.empty()) &&
         q.GetNumSize() != t.data.size()) {
         throw std::runtime_error("ToQuanTensor: shape and data size mismatch");
     }
     q.kF32_data = t.data;
     return q;
}


}// namespace yan_lamma