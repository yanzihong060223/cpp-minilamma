#include "include/forward.h"
#include "include/tensor.h"

#include <vector>
#include <stdexcept>
#include <utility>
#include <cmath>
namespace yan_lamma {
    //void AssertShape(const std::vector<int>& expected, const char* call
    
namespace {
  Tensor Embedding(const Model& model, int token_id) {
  int dim = model.config.dim;
  Tensor t({dim}, 0.0f);
  for(int i = 0; i < dim; i++) {
    t.data[i] = model.token_embedding.data[token_id * dim + i];
  }
  return t;
}
Tensor ForwardAdd(const Model& model, const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error(" Add Shape Not Matched");
    }
    Tensor t (a.shape);
    for (int i = 0; i < a.GetDataSize(); i++) {
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
    for (int i = 0; i < y.GetDataSize(); i++) {
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
Tensor ForwardLinear (const Model& model, const std::string& name, const Tensor& x, const Tensor& weight) {
    // cpu
    return Linear(x, weight);
}
//Tensor SwiGlu(const Tensor& gate, const Tensor& up)
Tensor ForwardSwiGlu(const Model& model, const Tensor& gate, const Tensor& up) {
    // cpu
    return SwiGlu(gate, up);
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
    //Tensor ForwardLinear (const Model& model, const std::string& name, const Tensor& x, const Tensor& weight)
    q_float = AddOptionBais(ForwardLinear(model, q_name, h, weight.wq), weight.bq);
    k_float = AddOptionBais(ForwardLinear(model, k_name, h, weight.wk), weight.bk);
    v_float = AddOptionBais(ForwardLinear(model, v_name, h, weight.wv), weight.bv);

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
        model, "lm_head", normed, model.lm_head);

    return logits.Reshape({config.vocab_size}, "logits");
}
} // namespace
Tensor ForwardToken(const Model& model, Context& context, int token) {
    auto config = model.config;
    if (!model.loaded) {
        throw std::runtime_error("Model Load Failed");
    }
    if (token < 0 || token >= config.vocab_size) {
        throw std::out_of_range("ForwardToken: token out of range");
    }
    if (context.pos < 0 || context.pos >= config.max_seq_len) {
        throw std::out_of_range("ForwardToken position out of range");
    }
    if (model.n_layer != config.n_layers) {
        throw std::runtime_error("Layer Match Failed");
    }
    if (model.token_embedding.data.size() <
                static_cast<size_t>(config.vocab_size * config.dim)) {
                throw std::runtime_error(
                "ForwardToken token embedding tensor is smaller than config");
    }
    int layers = config.n_layers;
    Tensor emb = Embedding(model, token);

    for (int i = 0; i < layers; ++i) {
        
        emb = ForwardLayer(model, emb, context, i, model.layers[i]);
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
