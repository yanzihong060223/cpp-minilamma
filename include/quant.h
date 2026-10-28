#pragma once
#include "include/tensor.h"
#include "include/quantized_tensor.h"
#include "include/ops.h"

#include <vector>

namespace yan_lamma {
    // 以下块数组接口按最后一维分行，每行独立分成 32 元素块，尾块补浮点零后量化。
    // 反量化与 LinearQxx/MatmulQxx 需要原始 shape，补位不参与输出或点积。
    // 此约定适用于 vector<Block_Qxx>，QuanTensor 原始字节入口仍使用原有布局。
    //Tensor To Q80 Tensor
    std::vector<Block_Q80> TensorToQ80(const Tensor& tensor);
    // Q80 to Tensor
    Tensor Q80ToTensor(const std::vector<Block_Q80>& Q80tensor, const std::vector<int>& shape);
    //Tensor To Q40 Tensor
    std::vector<Block_Q40> TensorToQ40(const Tensor& tensor);
    //Q40ToTensor
    Tensor Q40ToTensor(const std::vector<Block_Q40>& Q40tensor, const std::vector<int>& shape);
    std::vector<Block_Q41> TensorToQ41(const Tensor& tensor);
    Tensor Q41ToTensor(const std::vector<Block_Q41>& Q41tensor, const std::vector<int>& shape);
    Tensor LinearQ80(const Tensor& tensor, const std::vector<Block_Q80>& weight, const std::vector<int>& shape);
    Tensor LinearQ40(const Tensor& tensor, const std::vector<Block_Q40>& weight, const std::vector<int>& shape);
    Tensor LinearQ41(const Tensor& tensor, const std::vector<Block_Q41>& weight, const std::vector<int>& shape);
    Tensor MatmulQ80(const std::vector<Block_Q80>& weight, const Tensor& b, const std::vector<int>& shape);
    Tensor MatmulQ40(const std::vector<Block_Q40>& weight, const Tensor& b, const std::vector<int>& shape);
    Tensor MatmulQ41(const std::vector<Block_Q41>& weight, const Tensor& b, const std::vector<int>& shape);
    float CompareMatmulErrorQ80(const Tensor& tensor, const Tensor& input);
    float CompareMatmulErrorQ40(const Tensor& tensor, const Tensor& input);
    float CompareMatmulErrorQ41(const Tensor& tensor, const Tensor& input);
    float CompareLinearErrorQ80(const Tensor& tensor, const Tensor& input);
    float CompareLinearErrorQ40(const Tensor& tensor, const Tensor& input);
    float CompareLinearErrorQ41(const Tensor& tensor, const Tensor& input);
}// namespace yan_lamma
