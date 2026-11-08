#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <valarray>
#include "include/context.h"
#include "include/cuda_attention.h"
#include "include/cuda_kvcache.h"

namespace yan_lamma {
namespace {
void CheckCudaAttention(cudaError_t err, const char* expr) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA attention error in " + std::string(expr) +
                             ": " + cudaGetErrorString(err));
  }
}

void CheckLastAttentionKernel(const char* kernel_name) {
  CheckCudaAttention(cudaGetLastError(), kernel_name);
  CheckCudaAttention(cudaDeviceSynchronize(), kernel_name);
}
constexpr int kBlockSize = 128;
void VailedKvcache(const CudaCache& cache, const char* caller) {
    if (cache.KData() == nullptr || cache.vData() == nullptr) {
        throw std::runtime_error(std::string(caller) +
            ": key and value cache buffers must both be initialized");
    }
}
void VailedIndices (int index, int limit, const char* caller) {
    if (index < 0 || index >= limit) {
        throw std::runtime_error(std::string(caller) + ": index " +
            std::to_string(index) + " out of range [0, " +
            std::to_string(limit) + ")");
    }
}
void RequirePositive (int ele, const char* caller) {
    if (ele <= 0) {
        throw std::runtime_error(std::string(caller) +
            ": value must be positive, got " + std::to_string(ele));
    }
}
void CheckMul(size_t a, size_t b, const char* caller) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        throw std::runtime_error(std::string(caller) +
            ": size multiplication overflow for " + std::to_string(a) +
            " * " + std::to_string(b));
    }
}
void VailedGQA (int n_head, int n_kv_heads, const char* caller) {
    if (n_kv_heads == 0 || n_head % n_kv_heads != 0) {
        throw std::runtime_error(std::string(caller) +
            ": n_kv_heads must be non-zero and divide n_heads, got n_heads=" +
            std::to_string(n_head) + ", n_kv_heads=" +
            std::to_string(n_kv_heads));
    }
}
void CheckCacheArgs(const CudaCache& cache, int n_kv_head, int head_dim, int device_id, const char* caller) {
    if (cache.n_kv_heads() != n_kv_head) {
        throw std::runtime_error(std::string(caller) +
            ": cache n_kv_heads mismatch, expected " + std::to_string(n_kv_head) +
            ", got " + std::to_string(cache.n_kv_heads()));
    }
    if (cache.head_dim() != head_dim) {
        throw std::runtime_error(std::string(caller) +
            ": cache head_dim mismatch, expected " + std::to_string(head_dim) +
            ", got " + std::to_string(cache.head_dim()));
    }
    if (cache.device_id() != device_id) {
        throw std::runtime_error(std::string(caller) +
            ": cache device_id mismatch, expected " + std::to_string(device_id) +
            ", got " + std::to_string(cache.device_id()));
    }
}
__global__ void ScoresKernel(const float* q, const float* keys,
                                      float* scores, int layer, int max_seq_len,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      int pos, float scale) {
        int h =  blockIdx.x;
        int t =  blockIdx.y;
        int kv_group = n_heads / n_kv_heads;
        int kv_head = h / kv_group;
        size_t key_base =
         ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) +
            static_cast<size_t>(t)) *
            static_cast<size_t>(n_kv_heads) +
            static_cast<size_t>(kv_head)) *
            static_cast<size_t>(head_dim);
        size_t q_base = static_cast<size_t>(h) * static_cast<size_t>(head_dim);
        float sum = 0.0f;
        __shared__ float partial[kBlockSize];
        for (int i = threadIdx.x; i < head_dim; i+= blockDim.x) {
            sum += q[q_base + i] * keys[key_base + i];
        }
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] += partial[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            scores[static_cast<size_t>(h) * static_cast<size_t>(pos + 1) +
           static_cast<size_t>(t)] = partial[0] * scale;
        }
}
__global__ void AttentionReduceKernel(const float* scores, const float* values,
                                      float* out, int layer, int max_seq_len,
                                      int n_heads, int n_kv_heads, int head_dim,
                                      int pos) {
        int h = blockIdx.x;
        int kv_group = n_heads / n_kv_heads;
        int kv_head = h / kv_group;
        size_t score_base = static_cast<size_t>(h) * static_cast<size_t>(pos + 1);
        __shared__ float partial[kBlockSize];
        float local_max = -CUDART_INF_F;
        for(int i =  threadIdx.x; i <= pos; i += blockDim.x) {
             local_max = fmaxf(local_max, scores[score_base + i]);
        }
        partial[threadIdx.x] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                partial[threadIdx.x] =
                    fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        float max_score = partial[0];
        float local_sum = 0.0f;
        __shared__ float temp[kBlockSize];
        for (int i = threadIdx.x; i <= pos; i+= blockDim.x) {
            local_sum += expf(scores[score_base + i] - max_score);
        }
        temp[threadIdx.x] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                temp[threadIdx.x] += temp[threadIdx.x + stride];
            }
            __syncthreads();
        }
        float denom = temp[0];
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float value = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                float prob =
                    expf(scores[score_base + static_cast<size_t>(t)] - max_score) / denom;
                size_t value_idx =
                    ((static_cast<size_t>(layer) * static_cast<size_t>(max_seq_len) +
                      static_cast<size_t>(t)) *
                     static_cast<size_t>(n_kv_heads) +
                     static_cast<size_t>(kv_head)) *
                    static_cast<size_t>(head_dim) + static_cast<size_t>(d);
                value += prob * values[value_idx];
            }
            out[static_cast<size_t>(h) * static_cast<size_t>(head_dim) +
                static_cast<size_t>(d)] = value;
        }
}
}// namespace

Tensor CudaAttentionDecode (const Tensor& q, const CudaCache& cache, int layer, int pos, int n_heads, int n_kv_head, int head_dim, int device_id) {
    const char* caller = "CudaAttentionDecode";
    VailedKvcache(cache, caller);
    VailedIndices(layer, cache.n_layers(), caller);
    VailedIndices(pos, cache.max_seq_len(), caller);
    RequirePositive(n_heads, caller);
    RequirePositive(n_kv_head, caller);
    RequirePositive(head_dim, caller);
    VailedGQA(n_heads, n_kv_head, caller);
    CheckCacheArgs(cache, n_kv_head, head_dim, device_id, caller);
    CheckMul(static_cast<size_t>(n_heads), static_cast<size_t>(head_dim), caller);
    const size_t out_elements = static_cast<size_t>(n_heads) * head_dim;
    if (q.shape != std::vector<int>{n_heads, head_dim} ||
        q.GetDataSize() != out_elements) {
        throw std::runtime_error(std::string(caller) +
            ": q shape and element count must match [n_heads, head_dim]");
    }
    SetCudaService(device_id);
    const int seq_len = pos + 1;
    CheckMul(q.GetDataSize(), sizeof(float), caller);
    const size_t q_bytes = q.GetDataSize() * sizeof(float);
    CheckMul(static_cast<size_t>(n_heads), static_cast<size_t>(seq_len), caller);
    const size_t score_elements = static_cast<size_t>(n_heads) * seq_len;
    CheckMul(score_elements, sizeof(float), caller);
    const size_t score_bytes = score_elements * sizeof(float);

    CudaDeviceBuffer q_buffer(q_bytes, device_id);
    CudaDeviceBuffer scores(score_bytes, device_id);
    CudaDeviceBuffer out_buffer(q_bytes, device_id);
    q_buffer.upload(q.data.data(), q_bytes);
    ScoresKernel<<<dim3(n_heads, seq_len), kBlockSize>>>(
        static_cast<const float*>(q_buffer.GetData()),
        static_cast<const float*>(cache.KData()),
        static_cast<float*>(scores.GetData()), layer, cache.max_seq_len(),
        n_heads, n_kv_head, head_dim, pos,
        1.0f / std::sqrt(static_cast<float>(head_dim)));
    CheckLastAttentionKernel("ScoresKernel");
    AttentionReduceKernel<<<n_heads, kBlockSize>>>(
        static_cast<const float*>(scores.GetData()),
        static_cast<const float*>(cache.vData()),
        static_cast<float*>(out_buffer.GetData()), layer, cache.max_seq_len(),
        n_heads, n_kv_head, head_dim, pos);
    CheckLastAttentionKernel("AttentionReduceKernel");

    Tensor out({n_heads, head_dim});
    out_buffer.download(out.data.data(), q_bytes);
    return out;
}

CudaTensor CudaAttentionDevice(const CudaTensor& q, const CudaCache& cache,
                               int layer, int pos, int n_heads, int n_kv_head,
                               int head_dim, int device_id) {
    const char* caller = "CudaAttentionDevice";
    VailedKvcache(cache, caller);
    VailedIndices(layer, cache.n_layers(), caller);
    VailedIndices(pos, cache.max_seq_len(), caller);
    RequirePositive(n_heads, caller);
    RequirePositive(n_kv_head, caller);
    RequirePositive(head_dim, caller);
    VailedGQA(n_heads, n_kv_head, caller);
    CheckCacheArgs(cache, n_kv_head, head_dim, device_id, caller);
    CheckMul(static_cast<size_t>(n_heads), static_cast<size_t>(head_dim), caller);
    const size_t out_elements = static_cast<size_t>(n_heads) * head_dim;
    CheckMul(out_elements, sizeof(float), caller);
    if (q.shape() != std::vector<int>{n_heads, head_dim} ||
        q.NumEle() != out_elements) {
        throw std::runtime_error(std::string(caller) +
            ": q shape and element count must match [n_heads, head_dim]");
    }
    if (q.IsEmpty()) {
        throw std::runtime_error(std::string(caller) + ": q must not be empty");
    }
    if (q.DeviceId() != cache.device_id()) {
        throw std::runtime_error(std::string(caller) +
            ": q and cache must be on the same CUDA device, got q device_id=" +
            std::to_string(q.DeviceId()) + ", cache device_id=" +
            std::to_string(cache.device_id()));
    }
    SetCudaService(device_id);
    const int seq_len = pos + 1;
    CheckMul(static_cast<size_t>(n_heads), static_cast<size_t>(seq_len), caller);
    const size_t score_elements = static_cast<size_t>(n_heads) * seq_len;
    CheckMul(score_elements, sizeof(float), caller);
    CudaDeviceBuffer scores(score_elements * sizeof(float), device_id);
    CudaTensor out({n_heads, head_dim}, device_id);
    ScoresKernel<<<dim3(n_heads, seq_len), kBlockSize>>>(
        static_cast<const float*>(q.GetData()),
        static_cast<const float*>(cache.KData()),
        static_cast<float*>(scores.GetData()), layer, cache.max_seq_len(),
        n_heads, n_kv_head, head_dim, pos,
        1.0f / std::sqrt(static_cast<float>(head_dim)));
    CheckLastAttentionKernel("ScoresKernel");
    AttentionReduceKernel<<<n_heads, kBlockSize>>>(
        static_cast<const float*>(scores.GetData()),
        static_cast<const float*>(cache.vData()),
        static_cast<float*>(out.GetData()), layer, cache.max_seq_len(),
        n_heads, n_kv_head, head_dim, pos);
    CheckLastAttentionKernel("AttentionReduceKernel");
    return out;
}

}// namespace yan_lamma
