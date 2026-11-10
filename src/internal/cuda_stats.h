#pragma once
#include "include/model.h"
namespace yan_lamma::internal {
// Thread-local attribution only; models retain their existing statistics storage.
extern thread_local CudaModelWeights* active_cuda_stats;
class CudaStatsScope {
 public:
    explicit CudaStatsScope(CudaModelWeights* stats)
        : previous_(active_cuda_stats) { active_cuda_stats = stats; }
    ~CudaStatsScope() { active_cuda_stats = previous_; }
    CudaStatsScope(const CudaStatsScope&) = delete;
    CudaStatsScope& operator=(const CudaStatsScope&) = delete;
 private:
    CudaModelWeights* previous_;
};
}
