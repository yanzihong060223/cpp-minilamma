#pragma once
#include "include/tensor.h"
#include "include/quantized_tensor.h"

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
    Tensor bk; // optrional
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
};



}//namespace lamma