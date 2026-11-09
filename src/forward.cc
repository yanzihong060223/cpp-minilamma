#include "include/forward.h"
#include "include/cuda_kvcache.h"
#include "include/cuda_tensor.h"
#include "include/tensor.h"
#include "include/quantized_tensor.h"
#include "include/model.h"
#include "include/cuda_matmul.h"
#include "include/cuda_quant.h"
#include "include/cuda_ops.h"
#include "include/cuda_attention.h"

#include <vector>
#include <stdexcept>
#include <utility>
#include <cmath>
namespace yan_lamma {
namespace {

// forward 运行统计：仅累加 CPU 侧计数器，无 CUDA 状态时直接返回。
// 操作成功返回后记录；异步操作仅表示成功提交。
// 不包含权重上传、KV cache 初始化等准备阶段，底层封装不重复记录。
static void RecordCudaLinearCall(const Model& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->linear_calls += 1;
}

static void RecordCudaActivationCall(const Model& model, size_t calls = 1) {
    if (!model.cuda_weights) {
        return;
    }
    // 逻辑算子次数，不是 kernel launch 次数；SiLU + mul 组成的 SwiGLU 记 2 次。
    model.cuda_weights->activation_calls += calls;
}

static void RecordCudaAttentionCall(const Model& model) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->attention_calls += 1;
}

static void RecordCudaAttentionCpuFallback(const Model& model) {
    if (!model.cuda_weights) {
        return;
    }
    // 仅在真正执行 CPU attention 回退后调用，普通 CPU 路径不调用。
    model.cuda_weights->attention_cpu_fallbacks += 1;
}

static void RecordCudaHostToDeviceCopy(const Model& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->host_to_device_copies += 1;
    model.cuda_weights->host_to_device_bytes += bytes;
}

static void RecordCudaDeviceToHostCopy(const Model& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    model.cuda_weights->device_to_host_copies += 1;
    model.cuda_weights->device_to_host_bytes += bytes;
}

static void RecordCudaKvCacheWrite(const Model& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    // 当前 token 的逻辑写入量：(K 元素数 + V 元素数) * sizeof(float)。
    model.cuda_weights->kv_cache_write_bytes += bytes;
}

static void RecordCudaKvCacheRead(const Model& model, size_t bytes) {
    if (!model.cuda_weights) {
        return;
    }
    // 逻辑读取量由调用层按实际历史长度和 head 映射传入，不是硬件显存流量。
    // 单 token decode 每个 query head 读取映射 KV head 的 K、V 各一次：
    // 2 * (pos + 1) * n_heads * head_dim * sizeof(float)，GQA 共享 head 重复计入。
    model.cuda_weights->kv_cache_read_bytes += bytes;
}

// 仅用于一次输入上传、一次 Linear、一次输出下载，不再分别记录这三项。
// 不计权重上传或 CPU bias add。
static void RecordCudaLinearCopy(const Model& model, const Tensor& x,
                          const Tensor& y) {
    if (!model.cuda_weights) {
        return;
    }
    RecordCudaLinearCall(model);
    RecordCudaHostToDeviceCopy(model, x.GetDataSize() * sizeof(float));
    RecordCudaDeviceToHostCopy(model, y.GetDataSize() * sizeof(float));
}


CudaloadWeight* FindLoadWeight(const Model& model, const std::string& name, QuantType type) {
        auto arr = model.cuda_weights;
        if (!arr) {
            return nullptr;
        }
        for (size_t i = 0; i < (*arr).weights.size(); i++) {
            if ((*arr).weights[i].name == name && (*arr).weights[i].type == type) {
                return &((*arr).weights[i]);
            }
        }
       return nullptr;
}  
bool ForwaredLinearSupport (const QuanTensor& weight) {
    if (weight.type == QuantType::kF32 || weight.type == QuantType::kQ40 || 
        weight.type == QuantType::kQ41 || weight.type == QuantType::kQ80) {
            return true;
        }
    return false;
}
bool ForwardLinearReady(const Model& model, const std::string name, QuantType type) {
    if (model.cuda_weights != nullptr) { 
        auto p = FindLoadWeight(model, name, type);
        if ( p != nullptr) {
            return true;
        }
    }
    return false;
}
CudaTensor ForwardLinearDevice(const Model& model, const CudaTensor& x, const std::string name, const QuanTensor& tensor) {
    if (!model.cuda_weights) {
        throw std::runtime_error("ForwardLinearDevice: CUDA weights are not loaded: " + name);
    }
    if (! ForwaredLinearSupport(tensor)) {
        throw std::runtime_error("CUDA forward Linear unsupported weight type: " +
                             name);
    }
    CudaloadWeight* weight = FindLoadWeight(model, name, tensor.type);
    if (weight == nullptr) {
        throw std::runtime_error("ForwardLinearDevice: CUDA weight not found: " + name);
    }
    CudaTensor y;
    switch(tensor.type) {
        case yan_lamma::QuantType ::kF32 :
            y = CudaLinearDevice(x, weight->buffer.GetData(), weight->shape, model.cuda_weights->device_id);
            break;
        case yan_lamma::QuantType ::kQ40 :
            y = CudaQ40DeviceInputLinear(x, weight->buffer.GetData(), weight->shape, weight->block_count, model.cuda_weights->device_id);
            break;
        case yan_lamma::QuantType ::kQ41 : 
            y = CudaQ41DeviceInputLinear(x, weight->buffer.GetData(), weight->shape, weight->block_count, model.cuda_weights->device_id);
            break;
        case yan_lamma::QuantType ::kQ80 : 
            y = CudaQ80DeviceInputLinear(x, weight->buffer.GetData(), weight->shape, weight->block_count, model.cuda_weights->device_id); 
            break;
        default :
            throw std::runtime_error("ForwardLinearDevice: unsupported weight type: " + name);
    }
    RecordCudaLinearCall(model);
    return y;
}
static CudaTensor ForwardAddBiasDevice(const Model& model, CudaTensor x,
                                       const std::string& name, const Tensor& bias) {
    if (bias.data.empty()) {
        return x;
    }
    if (!model.cuda_weights) {
        throw std::runtime_error("ForwardAddBiasDevice: CUDA weights are not loaded: " + name);
    }
    const CudaloadWeight* weight = FindLoadWeight(model, name, QuantType::kF32);
    if (weight == nullptr) {
        throw std::runtime_error("ForwardAddBiasDevice: CUDA bias weight not found: " + name);
    }
    CudaTensor y = OCudaElementAddDevice(
        x, weight->buffer.GetData(), weight->shape, model.cuda_weights->device_id);
    RecordCudaActivationCall(model);
    return y;
}

void FowardQKVproject(const Model& model, CudaTensor& q, CudaTensor& k, CudaTensor& v, int layer, const CudaTensor& h, const Layer& layer_weight) {
    auto cuda_Load_weight = model.cuda_weights;
    if (cuda_Load_weight == nullptr) {
        throw std::runtime_error("FowardQKVproject: CUDA weights are not loaded");
    }
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    q = ForwardLinearDevice(model, h, prefix + "wq", layer_weight.wq);
    k = ForwardLinearDevice(model, h, prefix + "wk", layer_weight.wk);
    v = ForwardLinearDevice(model, h, prefix + "wv", layer_weight.wv);
    q = ForwardAddBiasDevice(model, std::move(q), prefix + "bq", layer_weight.bq);
    k = ForwardAddBiasDevice(model, std::move(k), prefix + "bk", layer_weight.bk);
    v = ForwardAddBiasDevice(model, std::move(v), prefix + "bv", layer_weight.bv);

}
CudaTensor ForwardRmsForm(const Model& model, CudaTensor& x, const std::string name, float eps) {
    auto cuda_weight = model.cuda_weights;
    if (cuda_weight == nullptr) {
        throw std::runtime_error("ForwardRmsForm: CUDA weights are not loaded: " + name);
    }
    CudaTensor y;
    CudaloadWeight * weight = FindLoadWeight(model, name, QuantType::kF32);
    if (weight == nullptr) {
        throw std::runtime_error("ForwardRmsForm: CUDA weight not found: " + name);
    }
    y = CudaRmsNromDevice(x, weight->buffer.GetData(), weight->shape, eps, cuda_weight->device_id);
    RecordCudaActivationCall(model);
    return y;
    
}
CudaTensor ForwardAddDevice(const Model& model, const CudaTensor& a, const CudaTensor& b) {
    if (!model.cuda_weights) {
        throw std::runtime_error("ForwardAddDevice: CUDA weights are not loaded");
    }
    CudaTensor y = CudaElementAddDevice(a, b, model.cuda_weights->device_id);
    RecordCudaActivationCall(model);
    return y;
}
Tensor ForwardSoftMaxDevice(const Model& model, Tensor& x) {
    Tensor y = CudaSoftmax(x, model.cuda_weights->device_id);
    return y;
}
void ForwardRopeDevice (const Model& model, CudaTensor& q, CudaTensor& k, int n_heads, int n_kv_heads, int head_dim, int pos, float theta, RopeType type) {
    if (!model.cuda_weights) {
        throw std::runtime_error("ForwardRopeDevice: CUDA weights are not loaded");
    }
    CudaRopeDevice(q, k, pos, n_heads, n_kv_heads, head_dim, theta, type, model.cuda_weights->device_id);
    RecordCudaActivationCall(model);
}
Tensor Embedding(const Model& model, int token_id) {
  int dim = model.config.dim;
  Tensor t({dim}, 0.0f);
  for(int i = 0; i < dim; i++) {
    t.data[i] = model.embedding_data_.data[token_id * dim + i];
  }
  return t;
}
CudaTensor EmbeddingDevice(const Model& model, int token_id) {
    CudaloadWeight* weight = FindLoadWeight(model, "embedding_data_", QuantType ::kF32);
    if (weight == nullptr) {
        throw std::runtime_error("EmbeddingDevice: CUDA weight not found: embedding_data_");
    }
    return CudaEmbeddingDevice(weight->buffer.GetData(), weight->shape, token_id, model.cuda_weights->device_id);
}
Tensor ForwardAdd(const Model& model, const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error(" Add Shape Not Matched");
    }
    Tensor t (a.shape);
    for (size_t i = 0; i < a.GetDataSize(); i++) {
        t.data[i] = a.data[i] + b.data[i];
    }
    return t;
}
Tensor AddOptionBais (const Tensor& t, const Tensor& bais) {
    if(bais.data.empty()) {
        return t;
    }
    if (t.GetNumDims() != 1 || bais.GetNumDims() != 1 || t.shape[0] != bais.shape[0]) {
        throw std::runtime_error("Tensor Bais Not Matched");
    }
    Tensor y =t;
    for (size_t i = 0; i < y.GetDataSize(); i++) {
        y.data[i] += bais.data[i];
    }
    return y;
}
//void Rope(Tensor& q, Tensor& k, int pos, float theta,
 //      RopeType rope_type = RopeType::kNormal);
//Q/K/V 投影和 RoPE
//用 x 计算 Q/K/V，拆成 head，再按 ctx.pos 对 Q、K 做 RoPE。
void ForwardRope(const Model& model, Tensor& q, Tensor& k, int pos, float theta,  RopeType rope_type) {
    // cpu
    Rope(q, k, pos, theta, rope_type);
}
CudaTensor ForwardAttentionDevice(const Model& model, const CudaTensor& q, const CudaTensor& k, const CudaTensor&v, int layer, int pos, 
int n_kv_heads, int n_heads, int head_dim, int device_id, CudaCache& cache) {
    cache.WriteDevice(layer, pos, k, v);
    RecordCudaKvCacheWrite(model, (k.NumEle() + v.NumEle()) * sizeof(float));
    CudaTensor out = CudaAttentionDevice(q, cache, layer, pos, n_heads, n_kv_heads, head_dim, device_id);
    RecordCudaAttentionCall(model);
    // 每个 query head 读取 [0, pos] 的映射 KV head 的 K/V 各一次。
    // GQA 共享 KV head 按 query head 重复计入；这是逻辑访问量，不是硬件流量。
    RecordCudaKvCacheRead(model, 2 * (static_cast<size_t>(pos) + 1) *
        static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim) * sizeof(float));
    return out;
}
CudaTensor ForwardffuDevice(const Model& model, const CudaTensor& h, const std::string layer_prefile, const LayerWeights& lw) {
   CudaTensor gate = ForwardLinearDevice(model, h, layer_prefile + "w_gate", lw.w_gate);
   CudaTensor up = ForwardLinearDevice(model, h, layer_prefile + "w_up", lw.w_up);
   CudaTensor silugate = CudaSiluDevice(gate, model.cuda_weights->device_id);
   RecordCudaActivationCall(model);
   CudaTensor ff = CudaElementwiseMulDevice(silugate, up, model.cuda_weights->device_id);
   RecordCudaActivationCall(model);
   return ForwardLinearDevice(model, ff, layer_prefile + "w_down", lw.w_down);
}
Tensor ForwardAttention(const Model& model, const Tensor& q, const Tensor& k, const Tensor& v, int layer, int pos,
KvCache& kvcache, int n_heads, int n_kv_heads, int head_dim) {
    // 先写入缓存
    kvcache.Write(layer, pos, k, v);
    Tensor out({n_heads, head_dim});
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int i = 0; i < n_heads; i++) {
        int kv_head = i / (n_heads / n_kv_heads);
        std::vector<float> temp(pos + 1, 0.0f);
        for (int j = 0; j <= pos; j++) {
            const float* ptr = kvcache.KeyPtr(layer, j, kv_head);
            float dot = 0.0f;
            //score[t] = (q0×k0 + q1×k1 + q2×k2 + q3×k3) / √4
            for (int d = 0; d < head_dim; d++) {
                dot += q.data[i * head_dim + d] * ptr[d];
            }
            temp[j] = dot * scale;
        }

        Tensor senor({pos + 1}, 0.0f);
        senor.data = std::move(temp);
        Tensor ts = SoftMax(senor);
        for (int h = 0; h < head_dim; h++) {
            float val = 0.0f;
            for (int n = 0; n <= pos; n++) {
                const float* vptr = kvcache.ValPtr(layer, n, kv_head);
                val += ts.data[n] * vptr[h];
            }
            out.data[i * head_dim + h] = val;
        }
    }
    return out;
}
Tensor ForwardLinear (const Model& model, const std::string& name, const Tensor& x, const QuanTensor& weight) {
    // cpu
    return Linear(x, weight);
}
//Tensor SwiGlu(const Tensor& gate, const Tensor& up)
Tensor ForwardSwiGlu(const Model& model, const Tensor& gate, const Tensor& up) {
    // cpu
    if (model.cuda_weights == nullptr) {
    return SwiGlu(gate, up);
    }
    Tensor silu_gate = CudaSilu(gate, model.cuda_weights->device_id);
    Tensor y = CudaElementwiseMul(silu_gate, up,  model.cuda_weights->device_id);
    return y;
}
Tensor FfnForward(const Model& model, const std::string& st, const Tensor& h, const LayerWeights& weight) {
    std::string gate_name = st + "w_gate";
    std::string up_name = st + "w_up";
    std::string down_name = st + "w_down";
    // cpu
    Tensor gate = ForwardLinear(model, gate_name, h, weight.w_gate);
    Tensor up = ForwardLinear(model, up_name, h, weight.w_up);
    Tensor ff = ForwardSwiGlu(model, gate, up);
    return ForwardLinear(model, down_name, ff, weight.w_down);
}
//ensor RmsNorm(const Tensor& x, const Tensor& weight, float eps)
Tensor ForwardRmsNorm(const Model& model, const Tensor& x, const Tensor& weight, float eps) {
    // cpu
    return RmsNorm(x, weight, eps);
}
void ForwardQKVProject(const Model& model, const std::string layer_prefix, const Tensor& h, const LayerWeights& weight, Tensor& q_float, Tensor& k_float, Tensor& v_float) {
    const std::string q_name = layer_prefix + "wq";
    const std::string k_name = layer_prefix + "wk";
    const std::string v_name = layer_prefix + "wv";
    q_float = AddOptionBais(ForwardLinear(model, q_name, h, weight.wq), weight.bq);
    k_float = AddOptionBais(ForwardLinear(model, k_name, h, weight.wk), weight.bk);
    v_float = AddOptionBais(ForwardLinear(model, v_name, h, weight.wv), weight.bv);

}
CudaTensor ForwardLayerDevice(const Model& model, CudaTensor& x, Context& context,
                              int layer, const LayerWeights& weight) {
    if (!model.cuda_weights) {
        throw std::runtime_error("ForwardLayerDevice: CUDA weights are not loaded");
    }
    const auto& config = model.config;
    const int device_id = model.cuda_weights->device_id;
    if (layer < 0 || layer >= config.layers) {
        throw std::out_of_range("ForwardLayerDevice: layer out of range");
    }
    if (config.max_seq_len <= 0 ||
        context.pos >= static_cast<size_t>(config.max_seq_len)) {
        throw std::out_of_range("ForwardLayerDevice: position out of range");
    }
    if (x.IsEmpty() || x.shape() != std::vector<int>{config.dim} ||
        x.DeviceId() != device_id) {
        throw std::runtime_error("ForwardLayerDevice: expected input [dim] on the model CUDA device");
    }
    if (context.cuda_cache_.device_id() != device_id) {
        throw std::runtime_error("ForwardLayerDevice: KV cache must be on the model CUDA device");
    }
    const int pos = static_cast<int>(context.pos);
    const std::string prefix = "blk." + std::to_string(layer) + ".";

    CudaTensor normed = ForwardRmsForm(
        model, x, prefix + "attention_norm", config.rms_norm_eps);
    CudaTensor q;
    CudaTensor k;
    CudaTensor v;
    FowardQKVproject(model, q, k, v, layer, normed, weight);
    q.Reshape({config.n_heads, config.head_dim});
    k.Reshape({config.n_kv_heads, config.head_dim});
    v.Reshape({config.n_kv_heads, config.head_dim});
    ForwardRopeDevice(model, q, k, config.n_heads, config.n_kv_heads,
                      config.head_dim, pos, config.rope_theta, config.rope_type);
    CudaTensor attention = ForwardAttentionDevice(
        model, q, k, v, layer, pos, config.n_kv_heads, config.n_heads,
        config.head_dim, device_id, context.cuda_cache_);
    attention.Reshape({config.dim});
    CudaTensor projected = ForwardLinearDevice(model, attention, prefix + "wo", weight.wo);
    CudaTensor residual = ForwardAddDevice(model, x, projected);

    CudaTensor ffn_normed = ForwardRmsForm(
        model, residual, prefix + "ffn_norm", config.rms_norm_eps);
    CudaTensor ff = ForwardffuDevice(model, ffn_normed, prefix, weight);
    return ForwardAddDevice(model, residual, ff);
}

Tensor ForwardLayer(const Model& model, Tensor& x, Context& context, int layer, const LayerWeights& weight) {
    auto config = model.config;
    int dim = config.dim;
    int n_heads = config.n_heads;
    int n_kv_heads = config.n_kv_heads;
    int head_dim = config.head_dim;
    int pos = context.pos;
    const std::string layer_prefix = "layers." + std::to_string(layer) + ".";
    Tensor rmnorm = ForwardRmsNorm(model, x, weight.attention_norm, config.rms_norm_eps);
    Tensor q_float;
    Tensor k_float;
    Tensor v_float;
    ForwardQKVProject(model, layer_prefix, rmnorm, weight, q_float, k_float, v_float);
    Tensor q = q_float.Reshape({n_heads, head_dim}, "forward_layer q");
    Tensor k = k_float.Reshape({n_kv_heads, head_dim}, "forward_layer k");
    Tensor v = v_float.Reshape({n_kv_heads, head_dim}, "forward_layer v");
    ForwardRope(model, q, k, pos, config.rope_theta, config.rope_type);
    Tensor out = ForwardAttention(model, q, k, v, layer, pos,
                                  context.cpu_cache_, n_heads, n_kv_heads, head_dim);
    Tensor out_flat = out.Reshape({1, n_heads * head_dim}, "forward out_flat");
    Tensor out_flat_obj = ForwardLinear(model, layer_prefix + "wo", out_flat, weight.wo);
    out_flat_obj.AssertShape({1, dim}, "forward out_flat");
    Tensor out_pattern = out_flat_obj.Reshape({dim}, "forward out_pattern");
    Tensor res = ForwardAdd(model, x, out_pattern);
    Tensor h2 = ForwardRmsNorm(model, res, weight.ffn_norm, config.rms_norm_eps);
    Tensor ff = FfnForward(model, layer_prefix, h2, weight);
    ff.AssertShape({dim}, "forward_layer ffn");

    return ForwardAdd(model, res, ff);
}
Tensor ComputeLogits(const Model& model, Tensor& x) {
    auto config = model.config;
    Tensor normed = ForwardRmsNorm(
        model, x, model.final_norm, config.rms_norm_eps);

    Tensor logits = ForwardLinear(
        model, "lm_head", normed, model.lm_head_);

    return logits.Reshape({config.vocab_size}, "logits");
}
Tensor ComputeLogitsDevice(const Model& model, CudaTensor& x) {
    CudaTensor normed = ForwardRmsForm(model, x, "final_norm", model.config.rms_norm_eps);
    CudaTensor logits = ForwardLinearDevice(model, normed, "lm_head", model.lm_head_);
    Tensor logits_host(logits.shape());
    logits.DownloadToHost(logits_host);
    RecordCudaDeviceToHostCopy(model, logits.Bytes());
    return logits_host.Reshape({model.config.vocab_size}, "logits tensor");
}
} // namespace
Tensor ForwardToken(const Model& model, Context& context, int token) {
    auto config = model.config;
    if (!model.is_loaded) {
        throw std::runtime_error("Model Load Failed");
    }
    if (token < 0 || token >= config.vocab_size) {
        throw std::out_of_range("ForwardToken: token out of range");
    }
    if (context.pos < 0 || context.pos >= config.max_seq_len) {
        throw std::out_of_range("ForwardToken position out of range");
    }
    if (model.n_layers_.size() != static_cast<size_t>(config.layers)) {
        throw std::runtime_error("Layer Match Failed");
    }
    if (model.embedding_data_.data.size() <
                static_cast<size_t>(config.vocab_size * config.dim)) {
                throw std::runtime_error(
                "ForwardToken token embedding tensor is smaller than config");
    }
    int layers = config.layers;
    if (model.cuda_weights) {
        if (context.cuda_cache_.Bytes_() == 0) {
            context.cuda_cache_.Reset(config.layers, config.max_seq_len, config.n_kv_heads,
                                      config.head_dim, model.cuda_weights->device_id);
        }
        CudaTensor emb_dev = EmbeddingDevice(model, token);
        for (int  i = 0; i < layers; i++) {
            emb_dev = ForwardLayerDevice(model, emb_dev, context, i, model.n_layers_[i]);
        }
        return ComputeLogitsDevice(model, emb_dev);
    }
    Tensor emb = Embedding(model, token);

    for (int i = 0; i < layers; ++i) {
        
        emb = ForwardLayer(model, emb, context, i, model.n_layers_[i]);
    }

    return ComputeLogits(model, emb);
}
Tensor ForwardBatch(const Model& model, Context& context, const Batch& batch) {
    if (batch.tokens.size() == 0) {
        throw std::runtime_error("Empty Batch");
    }
    if (batch.tokens.size() != batch.poss.size()) {
        throw std::runtime_error("Tokens Pos Not Matched");
    }
    int n = batch.tokens.size();
    Tensor logits;
    for (int i = 0; i < n; i++) {
        context.pos = batch.poss[i];
        logits =  ForwardToken(model, context, batch.tokens[i]);
        context.token_id_ .push_back(batch.tokens[i]);
    }
    return logits;
}
} // namespace yan_lamma
