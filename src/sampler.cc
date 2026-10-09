#include "include/sampler.h"

#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>
constexpr float Ktemperatute = 1e-6f;
namespace yan_lamma {
namespace {
    void VailedLogits(const Tensor& logits) {
        if (logits.GetDataSize() == 0) {
            throw std::runtime_error("Logits Empty");
        }
        for (float value : logits.data) {
            if (!std::isfinite(value)) {
                throw std::runtime_error("Logits must be finite");
            }
        }
    }
    void VailedTemperature (float temperature) {
        if (!std::isfinite(temperature) || temperature < 0.0f) {
            throw std::runtime_error("No Vailed Temperature");
        }
    }
    void VailedTopK(int top_k) {
        if (top_k < 0) {
            throw std::runtime_error("No Vailed Top_K");
        }
    }
    // Greedy方法 暴力获取最大的token id
    int Asynx(const Tensor& logits) {
        int n = logits.GetDataSize();
        int max_res = 0;
        for (int i = 1; i < n; i++) {
            if (logits[i] > logits[max_res]) {
                max_res = i;
                continue;
            }
        }
        return max_res;
    }
    int SamplerCandidates(const std::vector<std::pair<float, int>>& candidates, float temperature, std::mt19937& nrg) {
        // 温度小于限定值 退化到greedy
        if (temperature < Ktemperatute) {
            int n = candidates.size();
            int best =  candidates[0].second;
            float max_val = candidates[0].first;
            for (int i = 1; i < n; i++) {
                if (candidates[i].first > max_val) {
                    best = candidates[i].second;
                    max_val =  candidates[i].first;
                }
            }
            return best;
        }
        int n = candidates.size();
        float max_val = candidates[0].first;
        for (int i = 1; i < n; i++) {
            if (candidates[i].first > max_val) {
                max_val =  candidates[i].first;
            }
        }
        std::vector<float> temp(n, 0.0f);
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            temp[i] = std::exp((candidates[i].first - max_val) / temperature);
            sum += temp[i];
        }
        std::uniform_real_distribution dist(0.0f, 1.0f);
        float r = dist(nrg);
        float cmulative = 0.0f;
        for (int i = 0; i < n; i++) {
            cmulative += temp[i] / sum;
            if (r <= cmulative) {
                return candidates[i].second;
            }
        }
        return  candidates.back().second;
    }
}//namespace
Sampler::Sampler(const SamplerParmas& parmas) : Sampler(parmas.seed){
     
}
Sampler::Sampler(int seed ) {
    if (seed != 0) {
        nrg_.seed (seed);
    } else {
        std::random_device rd;
        nrg_.seed(rd());
    }
}
int Sampler::SamplerGreedy(const Tensor& logits) {
     VailedLogits(logits);
     return Asynx(logits);
}
int Sampler:: SamplerTemperature(const Tensor& logits, float temperature) {
    VailedLogits(logits);
    VailedTemperature(temperature);
    // temperature == 0 路由到greedy
    if (temperature == 0.0f) {
    return Asynx(logits);
    }
    std::vector<std::pair<float, int>> candidates;
    int n = logits.GetDataSize();
    for (int i = 0; i < n; i++) {
        candidates.emplace_back(logits.data[i], i);
    }
    return SamplerCandidates(candidates, temperature,  nrg_);
}
int Sampler::SamplerTopK(const Tensor& logits, float temperature, int top_k) {
    VailedLogits(logits);
    VailedTemperature(temperature);
    VailedTopK(top_k);
    int n = logits.GetDataSize();
    if (top_k > n) {
        top_k = n;
    }
    
    if (temperature == 0.0f) {
        return SamplerGreedy(logits);
    }
    if (top_k == 1) {
        return SamplerGreedy(logits);
    }
    std::vector<std::pair<float, int>> candidates;
    for (int i = 0; i < n; i++) {
        candidates.emplace_back(logits.data[i], i);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {return a.first > b.first;});
    // 取k_top 
    candidates.resize(top_k);
    return SamplerCandidates(candidates, temperature, nrg_);

}
int Sampler::SamplerO(const Tensor& logits, const SamplerParmas& parmas) {
    VailedLogits(logits);
    VailedTemperature(parmas.temperature);
    VailedTopK(parmas.top_k);

    if (parmas.temperature < Ktemperatute || parmas.top_k == 1) {
        return SamplerGreedy(logits);
    }
    if (parmas.top_k > 1) {
        return SamplerTopK(logits, parmas.temperature, parmas.top_k);
    }
    return SamplerTemperature(logits, parmas.temperature);
}
}//namespace yan_lamma