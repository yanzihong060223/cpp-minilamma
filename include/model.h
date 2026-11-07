#pragma once
#include "include/tensor.h"
#include "include/quantized_tensor.h"
#include "include/cuda_runtime.h"

#include <memory>
#include <string>
#include <vector>

namespace yan_lamma {
// 模型权重与算子的对应关系
//
// 输入：
//   token_embedding -> EmbedToken：根据 token ID 查表，取出输入向量。
//
// 每个 Transformer 层：
//   attention_norm  -> RmsNorm：Attention 前归一化，并按权重缩放。
//   wq              -> Linear：生成 Query。
//   wk              -> Linear：生成 Key。
//   wv              -> Linear：生成 Value。
//   bq / bk / bv    -> 逐元素加法：可选，分别加到 Q、K、V 上。
//   wo              -> Linear：将 Attention 结果投影回模型维度。
//
//   ffn_norm        -> RmsNorm：FFN 前归一化，并按权重缩放。
//   w_gate          -> Linear：生成门控向量，随后经过 SiLU 激活。
//   w_up            -> Linear：生成升维向量，与激活后的门控向量逐元素相乘。
//   w_down          -> Linear：将 FFN 中间结果投影回模型维度。
//
// 输出：
//   final_norm      -> RmsNorm：所有 Transformer 层结束后的归一化。
//   lm_head         -> Linear：生成各 token 的分数（logits）。
//
// Linear 的权重使用 QuanTensor，目前加载和转换只实现 F32。
// QuantType 中的 Q8 / Q4 类型留待后续实现。
// Embedding、Norm 和偏置使用普通 F32 Tensor。

enum RopeType {
    kNormal, //默认
    kNeoX
};
struct Layer {
    Tensor attention_norm;
    QuanTensor wq;
    QuanTensor wk;
    QuanTensor wv;
    QuanTensor wo;
    Tensor bq; // optional
    Tensor bk; // optional
    Tensor bv; // optional
    Tensor ffn_norm;
    QuanTensor w_gate;
    QuanTensor w_up;
    QuanTensor w_down;


};
using LayerWeights = Layer;
struct ModelConfig {
//词表大小
int vocab_size = 128;
// 模型维度
int dim = 32;
//中间维度
int hidden_dim = 86;
// 层数
int layers = 2;
// 注意力头
int n_heads = 4;
// kv头
int n_kv_heads = 4;
// 每个头的维度
int head_dim = 8;
// 最大序列长度
int max_seq_len = 128;
float rope_theta = 10000.0f;
float rms_norm_eps = 1e-5f;
RopeType rope_type = RopeType:: kNormal;



};
struct CudaloadWeight {
    std::string name;
    QuantType type = QuantType::kF32;
    bool linear = true;
    std::vector<int> shape;
    // 量化块总数
    size_t block_count = 0;
    CudaDeviceBuffer buffer;
};
struct CudaModelWeights {
  int device_id = 0;
  size_t uploaded_weight_count = 0;
  size_t uploaded_bytes = 0;
  size_t linear_calls = 0;
  size_t activation_calls = 0;
  size_t attention_calls = 0;
  size_t attention_cpu_fallbacks = 0;
  size_t kv_cache_write_bytes = 0;
  size_t kv_cache_read_bytes = 0;
  size_t host_to_device_copies = 0;
  size_t device_to_host_copies = 0;
  size_t host_to_device_bytes = 0;
  size_t device_to_host_bytes = 0;
  std::vector<CudaloadWeight> weights;
};
class Model {
public:
 Model() = default;
 bool is_loaded = false;
 Tensor embedding_data_;
 std::vector<Layer> n_layers_;
 ModelConfig config;
 Tensor  final_norm ;
 QuanTensor lm_head_;
 std::string load_error_;
 std::shared_ptr<CudaModelWeights> cuda_weights;
};
// 权重上传
void UploadModelWeightToCuda(Model& model, int device_id);
// 清理cuda权重
void ClearCudaWeight(Model& model);
void ModeloQuanTensorQ80(Model& model);
void ModeloQuanTensorQ40(Model& model);
size_t ModelCudaLinearCalls(const Model& model);                // Linear 调用次数
size_t ModelCudaActivationCalls(const Model& model);            // RMSNorm、加法等小算子调用次数
size_t ModelCudaAttentionCalls(const Model& model);             // GPU Attention 调用次数
size_t ModelCudaAttentionCpuFallbacks(const Model& model);      // Attention 回退 CPU 的次数
void ResetModelCudaRuntimeStats(Model& model);                  // 重置运行统计
size_t ModelCudaKvCacheWriteBytes(const Model& model);           // KV cache 写入量
size_t ModelCudaKvCacheReadBytes(const Model& model);            // KV cache 估算读取量

size_t ModelCudaHostToDeviceCopies(const Model& model);
size_t ModelCudaDeviceToHostCopies(const Model& model);
size_t ModelCudaHostToDeviceBytes(const Model& model);           // CPU → GPU 传输字节数
size_t ModelCudaDeviceToHostBytes(const Model& model);           // GPU → CPU 传输字节数

}//namespace lamma
