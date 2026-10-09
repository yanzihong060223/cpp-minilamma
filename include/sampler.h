#pragma once
#include "include/tensor.h"

#include <random>

namespace yan_lamma {
struct SamplerParmas {
// temperature 为0 使用greedy
float temperature = 0.0f;
// top _k  = 1 //使用greedy
int top_k = 0;
//随机种子
int seed = 0; 
};
class Sampler {
public:
 explicit Sampler(const SamplerParmas& parmas);
 explicit Sampler(int seed = 0);
  //暴露在外的接口
 int SamplerO(const Tensor& logits, const SamplerParmas& parmas);
private:
 //贪心策略
 // - temperature = 0 调用greedy
 // - top_k == 1 调用greedy
 int SamplerGreedy(const Tensor& logits);
 int SamplerTemperature(const Tensor& logits, float temperature);
 int SamplerTopK(const Tensor& logits, float temperature, int top_k);
 std::mt19937 nrg_;
};



}