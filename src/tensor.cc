#include "include/tensor.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>
namespace yan_lamma {
namespace {
    const char* CallerName(const char* call) {
        return call == nullptr?  "Tensor" : call;
    }
    std::string ShapeOToString(const std::vector<int>& shape) {
        std::string s = "[";
        for (std::size_t i = 0; i < shape.size(); i++) {
                if ( i > 0) {
                    s += ", ";
                }
                s += std::to_string(shape[i]);
        }
        s += "]";
        return s;
    }
    std::size_t CheckedNumel (const std::vector<int>& shape, const char* caller) {
            std::size_t total = 1;
            for (std::size_t i = 0; i < shape.size(); i++) {
                int artis = shape[i];
                if (artis <= 0) {
                        throw std::runtime_error(std::string(CallerName(caller)) +
                               ": dimension at axis " + std::to_string(i) +
                               " must be positive, got " + std::to_string(artis) +
                               " in shape " + ShapeOToString(shape));
                }
                const std::size_t dim_size = static_cast<std::size_t>(artis);
                if (total > std::numeric_limits<std::size_t>::max() / dim_size) {
                    throw std::runtime_error(std::string(CallerName(caller)) +
                               ": shape element count overflow for " +
                               ShapeOToString(shape));
                }
                total *= dim_size;
            }
            return total;
    }
    void CheckDims(const Tensor& T, std::size_t expected, const char* caller) {
        if (T.GetNumDims() != expected) {
              throw std::runtime_error(std::string(CallerName(caller)) + ": expected " +
                             std::to_string(expected) + "D tensor, got " + T. ShapeToString()
                            );
        }
    }
    void CheckAxisIndex(const Tensor& T, int test_index, std::size_t dim, const char* caller) {
            if (dim >= T.GetNumDims()) {
                throw std::out_of_range(
                    std::string(CallerName(caller)) + ": axis " + std::to_string(dim) +
                    " out of range for tensor " + T.ShapeToString());
            }
            const int max_len = T.shape[dim];
            if (test_index < 0 || test_index >= max_len) {
                  throw std::out_of_range(
        std::string(CallerName(caller)) + ": index " + std::to_string(test_index) +
        " out of range for axis " + std::to_string(dim) + " with size " +
        std::to_string(max_len) + " in tensor " + T. ShapeToString());
         }
    }

}
Tensor::Tensor(const std::vector<int>& data_shape, float default_data) {
    shape = std::move( data_shape);
    std::size_t total = CheckedNumel(shape, "Tensor::Tensor");
    data.resize(total,  default_data);
}
std::size_t Tensor::SwitchIdex(const std::vector<int>& indices) const {
    if (indices.size() != GetNumDims()) {
         throw std::invalid_argument("Tensor::SwitchIdex expected " +
                             std::to_string(shape.size()) +
                             " indices for tensor " +  ShapeToString()+
                             ", got " + std::to_string(indices.size()));
    }
    std::size_t flat = 0;
    for (std::size_t axis = 0; axis < shape.size(); ++axis) {
        CheckAxisIndex(*this, indices[axis], axis, "Tensor::SwitchIdex");
        flat = flat * static_cast<std::size_t>(shape[axis]) +
               static_cast<std::size_t>(indices[axis]);
    }
    return flat;
}
float Tensor::At1(int i) const {
     CheckDims(*this, 1, "Tensor::At1");
     CheckAxisIndex(*this, i, 0, "Tensor::At1");
     return data[i];

}
float& Tensor::At1(int i) {
     CheckDims(*this, 1, "Tensor::At1");
     CheckAxisIndex(*this, i, 0, "Tensor::At1");
     return data[i];
}
float Tensor::At2(int i, int j) const {
     CheckDims(*this, 2, "Tensor::At2");
     CheckAxisIndex(*this, i, 0, "Tensor::At2");
     CheckAxisIndex(*this, j, 1, "Tensor::At2");
     return data[static_cast<std::size_t>(i) * shape[1] + j];
}
float& Tensor::At2 (int i, int j) {
     CheckDims(*this, 2, "Tensor::At2");
     CheckAxisIndex(*this, i, 0, "Tensor::At2");
     CheckAxisIndex(*this, j, 1, "Tensor::At2");
     return data[static_cast<std::size_t>(i) * shape[1] + j];
}
float Tensor::At3(int i, int j, int k) const {
    CheckDims(*this, 3, "Tensor::At3");
    CheckAxisIndex(*this, i, 0, "Tensor::At3");
    CheckAxisIndex(*this, j, 1, "Tensor::At3");
    CheckAxisIndex(*this, k, 2, "Tensor::At3");
    std::size_t offset = (static_cast<std::size_t>(i) * shape[1] + j) * shape[2] + k;
    return data[offset];
}
float& Tensor::At3(int i, int j, int k) {
    CheckDims(*this, 3, "Tensor::At3");
    CheckAxisIndex(*this, i, 0, "Tensor::At3");
    CheckAxisIndex(*this, j, 1, "Tensor::At3");
    CheckAxisIndex(*this, k, 2, "Tensor::At3");
    std::size_t offset = (static_cast<std::size_t>(i) * shape[1] + j) * shape[2] + k;
    return data[offset];
}
float Tensor:: At4(int i, int j, int k, int m) const {
    CheckDims(*this, 4, "Tensor::At4");
    CheckAxisIndex(*this, i, 0, "Tensor::At4");
    CheckAxisIndex(*this, j, 1, "Tensor::At4");
    CheckAxisIndex(*this, k, 2, "Tensor::At4");
    CheckAxisIndex(*this, m, 3, "Tensor::At4");
    std::size_t offset = ((static_cast<std::size_t>(i) * shape[1] + j) * shape[2] + k) * shape[3] + m;
    return data[offset];
}
float& Tensor:: At4(int i, int j, int k, int m) {
    CheckDims(*this, 4, "Tensor::At4");
    CheckAxisIndex(*this, i, 0, "Tensor::At4");
    CheckAxisIndex(*this, j, 1, "Tensor::At4");
    CheckAxisIndex(*this, k, 2, "Tensor::At4");
    CheckAxisIndex(*this, m, 3, "Tensor::At4");
    std::size_t offset = ((static_cast<std::size_t>(i) * shape[1] + j) * shape[2] + k) * shape[3] + m;
    return data[offset];
}
float* Tensor:: RowPtr(int row) {
    CheckDims(*this, 2, "Tensor::RowPtr");
    CheckAxisIndex(*this, row, 0, "Tensor::RowPtr");
    return data.data() + static_cast<std::size_t>(row) * shape[1];
}
const float* Tensor::RowPtr(int row) const {
     CheckDims(*this, 2, "Tensor::RowPtr");
    CheckAxisIndex(*this, row, 0, "Tensor::RowPtr");
    return data.data() + static_cast<std::size_t>(row) * shape[1];
}
void Tensor::AssertShape(const std::vector<int>& expected, const char* call) {
    if (shape != expected) {
        throw std::runtime_error(std::string(CallerName(call)) +
                                 ": expected " + ShapeOToString(expected) +
                                 ", got " + ShapeToString());
    }
}
Tensor Tensor::Reshape(const std::vector<int>& new_shape, const char* call) {
    std::size_t total = CheckedNumel(new_shape, call);
    if (total != data.size()) {
        throw std::runtime_error(std::string(CallerName(call)) +
                                 ": cannot reshape " + ShapeToString() +
                                 " with " + std::to_string(data.size()) +
                                 " elements to " + ShapeOToString(new_shape) +
                                 " with " + std::to_string(total) + " elements");
    }
    Tensor r =  *this;
    r.shape = new_shape;
    return r;
}
std::string Tensor::ShapeToString() const {
   return ShapeOToString(shape);
}
void Tensor::Print(const std::string& name, bool print_data) const {
    std::cout << name;
    std::cout << ShapeToString() << " " <<  GetNumDims() << "\n";
      if (print_data) {
    for (std::size_t i = 0; i < data.size(); ++i) {
      std::cout << std::fixed << std::setprecision(6) << data[i];
      if (i + 1 < data.size()) {
        std::cout << " ";
      }
    }
}
    std::cout << std::endl;
}

Tensor Tensor::MakeOneD(int i, float default_data) {
    return Tensor({i}, default_data);
}
Tensor Tensor::MakeSecondD(int i, int j, float default_data) {
    return Tensor({i, j}, default_data);
}
Tensor Tensor::MakeThirdD(int i, int j, int k, float default_data) {
    return Tensor({i, j, k}, default_data);
}
Tensor Tensor::MakeFouthD(int i, int j, int k, int m, float default_data) {
    return Tensor({i, j, k, m}, default_data);
}
}//namespace yan_lamma
