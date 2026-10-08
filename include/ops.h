#pragma once
#include "include/tensor.h"
#include "matumal/matuml_ops.h"
#include <cmath>
namespace yan_lamma{

//对应元素相乘
Tensor ElementwiseMul(const Tensor& a, const Tensor& b);
//返回最大元素下标；相同最大值保留第一个下标，拒绝空输入
size_t ArgMax(const Tensor& a);
//输入 [in] 或 [1, in]，权重 [out, in]；输出 [out] 或 [1, out]
Tensor Linear(const Tensor& a, const Tensor& b, MatulMode mode = DefaultMatulMode());
//矩阵相乘：[m, k] x [k, n] -> [m, n]
Tensor Matmul(const Tensor& a, const Tensor& b, MatulMode mode = DefaultMatulMode());
//一维 RMS 归一化；x/weight 维度一致，eps 有限且非负
Tensor RmsNorm(const Tensor& x, const Tensor& weight, float eps);
// 把分数转换成和为1的关注权重
Tensor SoftMax(const Tensor& a);
//对每个元素应用激活函数
Tensor Silu(const Tensor& a);
//Silu(gate) 与 up 逐元素相乘
Tensor SwiGlu(const Tensor& gate, const Tensor& up);
//根据 token 位置旋转 Q、K
//void Rope(Tensor& q, Tensor& k, int pos, float theta,
 //         RopeType rope_type = RopeType::kNormal);




} //namespace yan_lamma
