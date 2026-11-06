#include "include/cuda_runtime.h"
#include "include/cuda_tensor.h"
#include "include/cuda_ops.h"
#include "include/tensor.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace yan_lamma {
namespace {
constexpr int kBlockSize = 256;
void CheckOps(cudaError_t error, const char* caller) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(caller) + ": " + cudaGetErrorString(error));
    }
}
void CheckKernel(const char* caller) {
    CheckOps(cudaGetLastError(), caller);
}
//校验函数
void RequireSameShape(const Tensor& a, const Tensor& b, const char* caller) {
    if (a.shape != b.shape) {
        throw std::runtime_error(std::string(caller) + ": input shapes must match");
    }
}
void RequireSameShape(const CudaTensor& a, const CudaTensor& b, const char* caller) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error(std::string(caller) + ": input shapes must match");
    }
    if (a.DeviceId() != b.DeviceId()) {
        throw std::runtime_error(std::string(caller) + ": inputs must be on the same CUDA device");
    }
}
//输入一维且非空	满足当前 RMSNorm、Softmax 的输入约定
void Require1d(const Tensor& x, const char* caller) {
    if (x.GetNumDims() != 1) {
        throw std::runtime_error(std::string(caller) + ": expected a 1D input tensor");
    }
    if (x.GetDataSize() == 0) {
        throw std::runtime_error(std::string(caller) + ": input tensor must not be empty");
    }
}
void Require1d(const CudaTensor& x, const char* caller) {
    if (x.num_dims() != 1) {
        throw std::runtime_error(std::string(caller) + ": expected a 1D input tensor");
    }
    if (x.NumEle() == 0 || x.IsEmpty()) {
        throw std::runtime_error(std::string(caller) + ": input tensor must not be empty");
    }
}
void VailedRopeInput(const Tensor& q, const Tensor& k, int pos, float theta, const char* caller) {
    if (q.GetNumDims() != 2 || k.GetNumDims() != 2) {
        throw std::runtime_error(std::string(caller) + ": expected 2D q and k tensors");
    }
    if (pos < 0) {
        throw std::runtime_error(std::string(caller) + ": pos must be non-negative");
    }
    if (! std::isfinite(theta) || theta <= 0) {
        throw std::runtime_error(std::string(caller) + ": theta must be finite and positive");
    }
    if (q.shape[1] != k.shape[1]) {
        throw std::runtime_error(std::string(caller) + ": q and k head dimensions must match");
    }
    if (q.shape[1] % 2 != 0) {
        throw std::runtime_error(std::string(caller) + ": head dimension must be even");
    }
}
void VailedRopeInput(CudaTensor& q, CudaTensor& k, int pos,
       int n_heads, int n_kv_heads, int head_dim, float theta, int device_id, const char* caller) {
    if (q.DeviceId() != device_id || k.DeviceId() != device_id) {
        throw std::runtime_error(std::string(caller) + ": q and k devices must match device_id");
    }
    if (pos < 0) {
        throw std::runtime_error(std::string(caller) + ": pos must be non-negative");
    }
    if (! std::isfinite(theta) || theta <= 0) {
        throw std::runtime_error(std::string(caller) + ": theta must be finite and positive");
    }
    if (n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error(std::string(caller) + ": head counts must be positive and head_dim must be positive and even");
    }
    if (q.NumEle() != static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim)) {
        throw std::runtime_error(std::string(caller) + ": q element count must equal n_heads * head_dim");
    }
    if (k.NumEle() != static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim)) {
        throw std::runtime_error(std::string(caller) + ": k element count must equal n_kv_heads * head_dim");
    }
}
void ValidateEmbeddingDeviceWeightInputs(const void* embedding_data, const std::vector<int>& shape, int token_id) {
    if (embedding_data == nullptr) {
        throw std::runtime_error("CudaEmbeddingDevice: embedding weight pointer must not be null");
    }
    if (shape.size() != 2) {
        throw std::runtime_error("CudaEmbeddingDevice: expected weight shape [vocab_size, dim]");
    }
    const int vocab_size = shape[0];
    const int dim = shape[1];
    if (vocab_size <= 0 || dim <= 0) {
        throw std::runtime_error("CudaEmbeddingDevice: weight dimensions must be positive");
    }
    if (token_id < 0 || token_id >= vocab_size) {
        throw std::runtime_error("CudaEmbeddingDevice: token_id must be in [0, vocab_size)");
    }
}
void ValidateRmsNormDeviceWeightInputs(const CudaTensor& x, const void* weight_data,
                                      const std::vector<int>& shape, float eps,
                                      int device_id) {
    Require1d(x, "CudaRmsNromDevice");
    if (weight_data == nullptr) {
        throw std::runtime_error("CudaRmsNromDevice: weight pointer must not be null");
    }
    if (x.shape() != shape) {
        throw std::runtime_error("CudaRmsNromDevice: weight shape must match input shape");
    }
    if (x.DeviceId() != device_id) {
        throw std::runtime_error("CudaRmsNromDevice: input device must match device_id");
    }
    if (!std::isfinite(eps) || eps <= 0) {
        throw std::runtime_error("CudaRmsNromDevice: eps must be finite and positive");
    }
}

void ValidateAddDeviceWeightInputs(const CudaTensor& x, const void* weight_data,
                                  const std::vector<int>& shape, int device_id) {
    if (x.DeviceId() != device_id) {
        throw std::runtime_error("OCudaElementAddDevice: input device must match device_id");
    }
    if (weight_data == nullptr) {
        throw std::runtime_error("OCudaElementAddDevice: weight pointer must not be null");
    }
    if (x.shape() != shape) {
        throw std::runtime_error("OCudaElementAddDevice: weight shape must match input shape");
    }
}
//算所有元素平方和
__global__ void SumSquareKernel(const float* x, float * sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n) {
        atomicAdd(sum, x[i] * x[i]);
    }
}
__global__ void RmsNormKernel (const float* x, const float* weight, float eps, float * sum, float * y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n) {
    float scale = rsqrtf((*sum / static_cast<float>(n)) + eps);
    y[i] = x[i] * scale * weight[i];
    }
}
__global__ void SiluKernel(const float* x, float * y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }

}
//f[i] = silu_gate[i] * up[i];
__global__ void ElementWiseMulKernel(const float* a, const float* b, float* y, int n) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) {
        y[i] = a[i] * b[i];
     }
}
// 残差连接 或加上bias
__global__ void ElementWiseAddKernel(const float* a, const float* b, float* y, int n) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) {
        y[i] = a[i] + b[i];
     }
}
__global__ void EmbeddingKernel(const float* embeddding, float * y, int dim, int token_id) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) {
        y[i] = embeddding[token_id * dim + i];
    }
}
//得到max_out
__global__ void SoftmaxMaxKernel(const float * x, float* y, int n) {
    __shared__ float shared[kBlockSize]; //一个block内都能访问的数组
    int tid = threadIdx.x;
    float local_max = -3.402823466e+38F;
    //每个线程间隔读取数组
    for(int i = tid; i < n; i += blockDim.x) {
         local_max = fmaxf(local_max, x[i]);
    }
    //对应线程的最小值
    shared[tid] = local_max;
    //全部线程到达
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
     }
        __syncthreads();
    }
    if (tid == 0) {
        *y =  shared[tid];
    }

}
/* \[
e_i=e^{x_i-m},\qquad S=\sum_i e_i
\]
*/
__global__ void SoftmaxExpSumKernel (const float* x, float * y,
                                     const float* max_value, float* sum_out, int n) {
    __shared__ float shared[kBlockSize]; //一个block内都能访问的数组
    int tid = threadIdx.x;
    float local = 0.0f;
    float max_v = *max_value;
    //每个线程间隔读取数组
    for(int i = tid; i < n; i += blockDim.x) {
         float e = expf(x[i] - max_v);
         y[i] = e;
         local += e;
    }
    //对应线程的最小值
    shared[tid] = local;
    //全部线程到达
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
     }
        __syncthreads();
    }
    if (tid == 0) {
        *sum_out = shared[tid];
    }
}
__global__ void SoftmaxNormKernel(float * x, const float* sum, int n) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i < n) {
        x[i] /= *sum;
     }
}
//Rope
__global__ void RopeKernel(float* x, int n_heads, int head_dim, int pos, float theta) {
    // 每个线程负责一个 head 内的一对元素。
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    if (pair_idx >= n_heads * half_dim) {
        return;
    }
    int head = pair_idx / half_dim;
    int i = pair_idx % half_dim;

    // Normal：相邻元素配对 (0, 1)、(2, 3)……
    int first = head * head_dim + 2 * i;
    int second = first + 1;
    float angle = static_cast<float>(pos) *
                  powf(theta, -2.0f * i / head_dim);
    float cos_value = cosf(angle);
    float sin_value = sinf(angle);
    // 先读取两个原值，再原地写回，避免覆盖后续计算需要的数据。
    float a = x[first];
    float b = x[second];
    x[first] = a * cos_value - b * sin_value;
    x[second] = a * sin_value + b * cos_value;
}

__global__ void RopeNeoXKernel(float* x, int n_heads, int head_dim, int pos, float theta) {
    // 每个线程独占一对元素，不需要共享内存或线程同步。
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    if (pair_idx >= n_heads * half_dim) {
        return;
    }
    int head = pair_idx / half_dim;
    int i = pair_idx % half_dim;

    // NeoX：前后半区对应元素配对 (i, i + head_dim / 2)。
    int first = head * head_dim + i;
    int second = first + half_dim;
    float angle = static_cast<float>(pos) *
                  powf(theta, -2.0f * i / head_dim);
    float cos_value = cosf(angle);
    float sin_value = sinf(angle);
    float a = x[first];
    float b = x[second];
    x[first] = a * cos_value - b * sin_value;
    x[second] = a * sin_value + b * cos_value;
}
// 分配线程块
int GridFor(int n) { 
    return (n + kBlockSize - 1) / kBlockSize; }
} // namespace
Tensor CudaRmsNormHost(const Tensor& x, const Tensor& weight, float eps, int device_id) {
    const char* caller = "CudaRmsNormHost";
    Require1d(x, caller);
    Require1d(weight, caller);
    RequireSameShape(x, weight, caller);
    if (!std::isfinite(eps) || eps <= 0) {
        throw std::runtime_error(std::string(caller) + ": eps must be finite and positive");
    }
    Tensor y(x.shape);
    int n = static_cast<int>(x.GetDataSize());
    size_t bytes = x.GetDataSize() * sizeof(float);
    CudaDeviceBuffer x_buffer(bytes, device_id);
    CudaDeviceBuffer weight_buffer(bytes, device_id);
    CudaDeviceBuffer y_buffer(bytes, device_id);
    CudaDeviceBuffer sum_buffer(sizeof(float), device_id);
    x_buffer.upload(x.data.data(), bytes);
    weight_buffer.upload(weight.data.data(), bytes);
    float* sum = static_cast<float*>(sum_buffer.GetData());
    CheckOps(cudaMemset(sum, 0, sizeof(float)), "CudaRmsNormHost: clear sum");
    SumSquareKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x_buffer.GetData()), sum, n);
    CheckKernel("SumSquareKernel");
    RmsNormKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x_buffer.GetData()),
        static_cast<const float*>(weight_buffer.GetData()), eps, sum,
        static_cast<float*>(y_buffer.GetData()), n);
    CheckKernel("RmsNormKernel");
    y_buffer.download(y.data.data(), bytes);
    return y;
}

CudaTensor CudaRmsNromDevice(const CudaTensor& x, const void* weight_data,
                            const std::vector<int>& shape, float eps, int device_id) {
    ValidateRmsNormDeviceWeightInputs(x, weight_data, shape, eps, device_id);
    SetCudaService(device_id);
    CudaTensor y(x.shape(), device_id);
    CudaDeviceBuffer sum_buffer(sizeof(float), device_id);
    float* sum = static_cast<float*>(sum_buffer.GetData());
    int n = static_cast<int>(x.NumEle());
    // atomicAdd 需要从零开始累加平方和。
    CheckOps(cudaMemset(sum, 0, sizeof(float)), "CudaRmsNromDevice: clear sum");
    SumSquareKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x.GetData()), sum, n);
    CheckKernel("SumSquareKernel");
    RmsNormKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(x.GetData()), static_cast<const float*>(weight_data),
        eps, sum, static_cast<float*>(y.GetData()), n);
    CheckKernel("RmsNormKernel");
    // 确保归一化已读完临时平方和，再释放缓冲区。
    CheckOps(cudaStreamSynchronize(0), "CudaRmsNromDevice: synchronize");
    return y;
}

CudaTensor CudaEmbeddingDevice(const void* embedding_data,
                               const std::vector<int>& shape, int token_id, int device_id) {
    ValidateEmbeddingDeviceWeightInputs(embedding_data, shape, token_id);
    SetCudaService(device_id);
    int dim = shape[1];
    CudaTensor y({dim}, device_id);
    EmbeddingKernel<<<GridFor(dim), kBlockSize>>>(
        static_cast<const float*>(embedding_data), static_cast<float*>(y.GetData()),
        dim, token_id);
    CheckKernel("EmbeddingKernel");
    return y;
}

Tensor CudaSilu(const Tensor& a, int device_id) {
    Tensor y(a.shape);
    int n = static_cast<int>(a.GetDataSize());
    size_t bytes = a.GetDataSize() * sizeof(float);
    CudaDeviceBuffer a_buffer(bytes, device_id);
    CudaDeviceBuffer y_buffer(bytes, device_id);
    a_buffer.upload(a.data.data(), bytes);
    SiluKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(a_buffer.GetData()),
        static_cast<float*>(y_buffer.GetData()), n);
    CheckKernel("SiluKernel");
    y_buffer.download(y.data.data(), bytes);
    return y;
}

Tensor CudaElementwiseMul(const Tensor& a, const Tensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementwiseMul");
    Tensor y(a.shape);
    int n = static_cast<int>(a.GetDataSize());
    size_t bytes = a.GetDataSize() * sizeof(float);
    CudaDeviceBuffer a_buffer(bytes, device_id);
    CudaDeviceBuffer b_buffer(bytes, device_id);
    CudaDeviceBuffer y_buffer(bytes, device_id);
    a_buffer.upload(a.data.data(), bytes);
    b_buffer.upload(b.data.data(), bytes);
    ElementWiseMulKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(a_buffer.GetData()),
        static_cast<const float*>(b_buffer.GetData()),
        static_cast<float*>(y_buffer.GetData()), n);
    CheckKernel("ElementWiseMulKernel");
    y_buffer.download(y.data.data(), bytes);
    return y;
}

CudaTensor CudaElementAddDevice(const CudaTensor& a, const CudaTensor& b, int device_id) {
    RequireSameShape(a, b, "CudaElementAddDevice");
    return OCudaElementAddDevice(a, b.GetData(), b.shape(), device_id);
}

CudaTensor OCudaElementAddDevice(const CudaTensor& a, const void* b,
                                const std::vector<int>& shape, int device_id) {
    ValidateAddDeviceWeightInputs(a, b, shape, device_id);
    SetCudaService(device_id);
    CudaTensor y(a.shape(), device_id);
    int n = static_cast<int>(a.NumEle());
    ElementWiseAddKernel<<<GridFor(n), kBlockSize>>>(
        static_cast<const float*>(a.GetData()), static_cast<const float*>(b),
        static_cast<float*>(y.GetData()), n);
    CheckKernel("ElementWiseAddKernel");
    return y;
}

Tensor CudaSoftmax(Tensor& x, int device_id) {
    Require1d(x, "CudaSoftmax");
    Tensor y(x.shape);
    int n = static_cast<int>(x.GetDataSize());
    size_t bytes = x.GetDataSize() * sizeof(float);
    CudaDeviceBuffer x_buffer(bytes, device_id);
    CudaDeviceBuffer y_buffer(bytes, device_id);
    CudaDeviceBuffer max_buffer(sizeof(float), device_id);
    CudaDeviceBuffer sum_buffer(sizeof(float), device_id);
    x_buffer.upload(x.data.data(), bytes);
    float* max_value = static_cast<float*>(max_buffer.GetData());
    float* sum = static_cast<float*>(sum_buffer.GetData());
    float* output = static_cast<float*>(y_buffer.GetData());
    const float* input = static_cast<const float*>(x_buffer.GetData());
    // 两个归约 kernel 都只启动一个 block；默认流保证三个阶段依次执行。
    SoftmaxMaxKernel<<<1, kBlockSize>>>(input, max_value, n);
    CheckKernel("SoftmaxMaxKernel");
    SoftmaxExpSumKernel<<<1, kBlockSize>>>(input, output, max_value, sum, n);
    CheckKernel("SoftmaxExpSumKernel");
    SoftmaxNormKernel<<<GridFor(n), kBlockSize>>>(output, sum, n);
    CheckKernel("SoftmaxNormKernel");
    y_buffer.download(y.data.data(), bytes);
    return y;
}

void CudaRopeHost(Tensor& q, Tensor& k, int pos, float theta,
                  RopeType rope_type, int device_id) {
    VailedRopeInput(q, k, pos, theta, "CudaRopeHost");
    size_t q_bytes = q.GetDataSize() * sizeof(float);
    size_t k_bytes = k.GetDataSize() * sizeof(float);
    CudaDeviceBuffer q_buffer(q_bytes, device_id);
    CudaDeviceBuffer k_buffer(k_bytes, device_id);
    q_buffer.upload(q.data.data(), q_bytes);
    k_buffer.upload(k.data.data(), k_bytes);
    int head_dim = q.shape[1];
    int q_pairs = q.shape[0] * (head_dim / 2);
    int k_pairs = k.shape[0] * (head_dim / 2);
    if (rope_type == RopeType::kNormal) {
        RopeKernel<<<GridFor(q_pairs), kBlockSize>>>(
            static_cast<float*>(q_buffer.GetData()), q.shape[0], head_dim, pos, theta);
        CheckKernel("RopeKernel: q");
        RopeKernel<<<GridFor(k_pairs), kBlockSize>>>(
            static_cast<float*>(k_buffer.GetData()), k.shape[0], head_dim, pos, theta);
        CheckKernel("RopeKernel: k");
    } else {
        RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
            static_cast<float*>(q_buffer.GetData()), q.shape[0], head_dim, pos, theta);
        CheckKernel("RopeNeoXKernel: q");
        RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
            static_cast<float*>(k_buffer.GetData()), k.shape[0], head_dim, pos, theta);
        CheckKernel("RopeNeoXKernel: k");
    }
    q_buffer.download(q.data.data(), q_bytes);
    k_buffer.download(k.data.data(), k_bytes);
}

void CudaRopeDevice(CudaTensor& q, CudaTensor& k, int pos,
                    int n_heads, int n_kv_heads, int head_dim, float theta,
                    RopeType rope_type, int device_id) {
    VailedRopeInput(q, k, pos, n_heads, n_kv_heads, head_dim, theta,
                    device_id, "CudaRopeDevice");
    SetCudaService(device_id);
    int q_pairs = n_heads * (head_dim / 2);
    int k_pairs = n_kv_heads * (head_dim / 2);
    if (rope_type == RopeType::kNormal) {
        RopeKernel<<<GridFor(q_pairs), kBlockSize>>>(
            static_cast<float*>(q.GetData()), n_heads, head_dim, pos, theta);
        CheckKernel("RopeKernel: q");
        RopeKernel<<<GridFor(k_pairs), kBlockSize>>>(
            static_cast<float*>(k.GetData()), n_kv_heads, head_dim, pos, theta);
        CheckKernel("RopeKernel: k");
    } else {
        RopeNeoXKernel<<<GridFor(q_pairs), kBlockSize>>>(
            static_cast<float*>(q.GetData()), n_heads, head_dim, pos, theta);
        CheckKernel("RopeNeoXKernel: q");
        RopeNeoXKernel<<<GridFor(k_pairs), kBlockSize>>>(
            static_cast<float*>(k.GetData()), n_kv_heads, head_dim, pos, theta);
        CheckKernel("RopeNeoXKernel: k");
    }
}

} // namespace yan_lamma
