#include "include/gguf.h"
#include "include/quantized_tensor.h"
#include "include/model.h"

#include<vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <unordered_map>
namespace yan_lamma {
namespace {
constexpr size_t kQuantBlockSize = 32;
constexpr size_t kBytesPerQ40Block = 18;  // fp16 scale(2) + 16 nibbles
constexpr size_t kBytesPerQ41Block = 20;  // fp16 scale + fp16 min + 16 nibbles
constexpr size_t kBytesPerQ80Block = 34;  // fp16 scale + 32 int8
constexpr size_t kBytesPerF16Elem  = 2;
}//namespace
std:: string MapTensorName(const std::string& gguf_name) {
    std::string res;
    if (gguf_name == "token_embd.weight") {
        res = "embedding_data_";
        return res;
    }
    if (gguf_name == "output_norm.weight") {
        res = "final_norm";
        return res;
    }
    if (gguf_name == "output.weight") {
        res = "lm_head";
        return res;
    }
    if(gguf_name.rfind("blk.", 0) == 0) {
        size_t dol;
        dol = gguf_name.find(".", 4);
        if (dol == std::string :: npos) {
            return "";
        }
        std::string layer = gguf_name.substr(4, dol - 4);
        if (layer == "") {
            return "";
        }
        size_t dol2 = gguf_name.find('.', dol + 1);
        std::string op = gguf_name.substr(dol + 1, dol2 - dol - 1);
        if (op == "") {
            return "";
        }
        std::string suffex = gguf_name.substr(dol2 + 1);
        std::string temp;
        if (suffex == "bias") {
            if (op == "attn_q") {
                temp = "bq";
            } else if (op == "attn_k") {
                temp = "bk";
            } else if(op == "attn_v") {
                temp = "bv";
            } else {
                return "";
            }
        } else if (suffex == "weight") {
            if (op ==  "attn_norm") {
                temp = "attention_norm";
            } else if(op == "attn_q") {
                temp = "wq";
            } else if(op == "attn_k") {
                temp = "wk";
            } else if (op == "attn_v") {
                temp = "wv";
            } else if(op == "attn_output") {
                temp = "wo";
            } else if(op == "ffn_norm") {
                temp = "ffn_norm";
            } else if(op == "ffn_gate") {
                temp = "w_gate";
            } else if(op == "ffn_up") {
                temp = "w_up";
            } else if(op == "ffn_down") {
                temp = "w_down";
            } else {
                return "";
            }
        } else {
            return "";
        }
        res = "blk." + layer + "." + temp;
    }
    return res;
}
// 转换张量shape
std::vector<int> MapShape( const std::vector<uint64_t>& gguf_shape) {
    // 二维转换
    std::vector<int> res;
    res.reserve(gguf_shape.size());
    if (gguf_shape.size() == 2) {
        res.push_back(gguf_shape[1]);
        res.push_back(gguf_shape[0]);
        return res;
    }
    //一维不用转换
    for( uint64_t dim : gguf_shape) {
        res.push_back(dim);
    }
    return res;
}
QuanTensor LoadQuantTensor(const std::vector<uint8_t>& raw, const std::vector<int> shape, int gguf_type) {
    size_t num_elents = 1;
    for (auto dim: shape) {
        num_elents *= dim;
    }
    if (num_elents == 0) {
        throw std::runtime_error("Num data is zero");
    }
    QuanTensor q;
    q.shape = shape;
    switch(gguf_type) {
        case static_cast<int>(Ggmltype::GgmltypeF32) : {
            const size_t expected = num_elents * sizeof(float);
            if (raw.size() != expected) {
                throw std::runtime_error("LoadQuantTensor: F32 raw size mismatch");
            }
            q.type = QuantType::kF32;
            q.kF32_data.resize(num_elents);
            std::memcpy(q.kF32_data.data(), raw.data(), expected);
            break;
        }
        case static_cast<int>(Ggmltype::GgmltypeF16) : {
            const size_t expected = num_elents * kBytesPerF16Elem;
            if (raw.size() != expected) {
                throw std::runtime_error(
                    "LoadQuantTensor: F16 raw size mismatch");
            }
            q.type = QuantType::kF16;
            q.kF16_data.assign(raw.begin(), raw.end());
            break;
        }
        case static_cast<int>(Ggmltype::GgmltypeQ40) : {
            if (num_elents % kQuantBlockSize != 0) {
                throw std::runtime_error(
                    "LoadQuantTensor: Q4_0 element count not block-aligned");
            }
            const size_t expected =
                (num_elents / kQuantBlockSize) * kBytesPerQ40Block;
            if (raw.size() != expected) {
                throw std::runtime_error(
                    "LoadQuantTensor: Q4_0 raw size mismatch");
            }
            q.type = QuantType::kQ40;
            q.kQ40_data.assign(raw.begin(), raw.end());
            break;
        }
        case static_cast<int>(Ggmltype::GgmltypeQ41) : {
            if (num_elents % kQuantBlockSize != 0) {
                throw std::runtime_error(
                    "LoadQuantTensor: Q4_1 element count not block-aligned");
            }
            const size_t expected =
                (num_elents / kQuantBlockSize) * kBytesPerQ41Block;
            if (raw.size() != expected) {
                throw std::runtime_error(
                    "LoadQuantTensor: Q4_1 raw size mismatch");
            }
            q.type = QuantType::kQ41;
            q.kQ41_data.assign(raw.begin(), raw.end());
            break;
        }
        case static_cast<int>(Ggmltype::GgmlytypeQ80) : {
            if (num_elents % kQuantBlockSize != 0) {
                throw std::runtime_error(
                    "LoadQuantTensor: Q8_0 element count not block-aligned");
            }
            const size_t expected =
                (num_elents / kQuantBlockSize) * kBytesPerQ80Block;
            if (raw.size() != expected) {
                throw std::runtime_error(
                    "LoadQuantTensor: Q8_0 raw size mismatch");
            }
            q.type = QuantType::kQ80;
            q.kQ80_data.assign(raw.begin(), raw.end());
            break;
        }
        default:
            throw std::runtime_error(
                "LoadQuantTensor: unsupported gguf_type " +
                std::to_string(gguf_type));
    }
    return q;
}
static void RequireShape(Tensor& tensor, const std::vector<int>& expected,
                         const std::string& name) {
  tensor.AssertShape(expected, name.c_str());
}

static void RequireShape(QuanTensor& tensor,
                         const std::vector<int>& expected,
                         const std::string& name) {
  tensor.AssertShape(expected, name.c_str());
}

static void RequireOptionalShape(Tensor& tensor,
                                 const std::vector<int>& expected,
                                 const std::string& name) {
  if (tensor.data.empty()) {
    return;
  }
  RequireShape(tensor, expected, name);
}

void VailedShape(Model& model) {
    const ModelConfig& c = model.config;
    const int q_proj  = c.n_heads    * c.head_dim;
    const int kv_proj = c.n_kv_heads * c.head_dim;

    if (model.n_layers_.size() != static_cast<size_t>(c.layers)) {
        throw std::runtime_error("VailedShape: layer count mismatch");
    }

    RequireShape(model.embedding_data_, {c.vocab_size, c.dim}, "embedding_data_");
    RequireShape(model.final_norm,      {c.dim},               "final_norm");
    RequireShape(model.lm_head_,        {c.vocab_size, c.dim}, "lm_head");

    for (size_t i = 0; i < model.n_layers_.size(); ++i) {
        Layer& L = model.n_layers_[i];
        const std::string pre = "blk." + std::to_string(i) + ".";
        RequireShape(L.attention_norm, {c.dim},                pre + "attention_norm");
        RequireShape(L.wq,             {q_proj,  c.dim},       pre + "wq");
        RequireShape(L.wk,             {kv_proj, c.dim},       pre + "wk");
        RequireShape(L.wv,             {kv_proj, c.dim},       pre + "wv");
        RequireShape(L.wo,             {c.dim,  q_proj},       pre + "wo");
        RequireOptionalShape(L.bq,     {q_proj},               pre + "bq");
        RequireOptionalShape(L.bk,     {kv_proj},              pre + "bk");
        RequireOptionalShape(L.bv,     {kv_proj},              pre + "bv");
        RequireShape(L.ffn_norm,       {c.dim},                pre + "ffn_norm");
        RequireShape(L.w_gate,         {c.hidden_dim, c.dim},  pre + "w_gate");
        RequireShape(L.w_up,           {c.hidden_dim, c.dim},  pre + "w_up");
        RequireShape(L.w_down,         {c.dim, c.hidden_dim},  pre + "w_down");
    }
}
ModelConfig GetModelConfig(const std::string& path) {
    ModelConfig config;
    auto getU32 = [&](const std::string& path, const std::string& key, int call) {
       int64_t v;
       if (GetGuffMetadaInt(path, key, v)) {
            return static_cast<int>(v);
       }
       return call;
    };
    auto getF32 = [&](const std::string& path, const std::string& key, float call ) {
        float v;
        if (GetGuffMetadaFloat(path, key, v)) {
                return v;
        }
        return call;
    };
    std::string arch;
    if (GetGuffMetadaString(path, "general.architecture", arch) && arch == "qwen2") {
        config.rope_type = RopeType::kNeoX;
    }
    config.layers = getU32(path, "qwen2.block_count", getU32(path, "llama.block_count", 0));
    config.dim = getU32(path, "qwen2.embedding_length", getU32(path, "llama.embedding_length", 0));
    config.n_heads = getU32(path, "qwen2.attention.head_count", getU32(path, "llama.attention.head_count", 0));
    config.n_kv_heads = getU32(path, "qwen2.attention.head_count_kv", getU32(path, "llama.attention.head_count_kv", 0));
    config.max_seq_len = getU32(path, "qwen2.context_length", getU32(path, "llama.context_length", 0));
    config.hidden_dim = getU32(path, "qwen2.feed_forward_length",
                               getU32(path, "llama.feed_forward_length", 0));
    config.rms_norm_eps = getF32(path, "qwen2.attention.layer_norm_rms_epsilon",
                                 getF32(path, "llama.attention.layer_norm_rms_epsilon", 0.0f));
    config.rope_theta = getF32(path, "qwen2.rope.freq_base",
                               getF32(path, "llama.rope.freq_base", 10000.0f));
    // head_dim 走派生,和 VaileConfig 的 head_dim == dim / n_heads 约束对齐
    if (config.n_heads > 0) {
        config.head_dim = config.dim / config.n_heads;
    }
    std::vector<std::string> tokens;
    if (GetGuffMetadaStringArry(path, "tokenizer.ggml.tokens", tokens)) {
        config.vocab_size = static_cast<int>(tokens.size());
    }
    // 校验,规则 + 错误串与 src/load.cc 的 VaileConfig 对齐
    if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
        config.layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 ||
        config.head_dim <= 0 || config.max_seq_len <= 0) {
        throw std::runtime_error("All model dimensions must be positive");
    }
    if (config.dim % config.n_heads != 0) {
        throw std::runtime_error("Dim n heads not match");
    }
    if (config.n_heads % config.n_kv_heads != 0) {
        throw std::runtime_error("n_heads n_kv_heads not match");
    }
    if (config.head_dim <= 0 || config.head_dim % 2 != 0) {
        throw std::runtime_error("head_ dim must be positive even number");
    }
    if (config.head_dim != config.dim / config.n_heads) {
        throw std::runtime_error("head_dim does not match dim / n_heads");
    }
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f) {
        throw std::runtime_error("rope_theta must be finite and positive");
    }
    if (!std::isfinite(config.rms_norm_eps) || config.rms_norm_eps <= 0.0f) {
        throw std::runtime_error(" rms_norm_eps must be finite and positive");
    }
    return config;
}
Model ModelLoad(const std::string& gguf) {
     GgufRead gguf_read;
     if (!gguf_read.load(gguf)) {
        throw std::runtime_error("gguf read load failed");
     }
     Model model;
     try {
        model.config =  GetModelConfig(gguf);
     } catch(std::exception& e) {
        std::cout << "Get model config failed" << e.what() << "\n";
        return model;
     }
     std::unordered_map<std::string, const GgufTensorInfo*> tensor_maps;
     for (const auto& t : gguf_read.tensorInfo_) {
        const std::string model_tensor_name = MapTensorName(t.name);
        if (model_tensor_name.empty()) {
            continue;  // 未知/不需要的张量,跳过
        }
        tensor_maps[model_tensor_name] = &t;
     }
     auto Load_Weight = [&](const std::string& tensor_name) ->QuanTensor {
        auto it = tensor_maps.find(tensor_name);
        if (it == tensor_maps.end()) {
            throw std::runtime_error("Missing Tensor: " + tensor_name);
        }
        std::vector<int> shape = MapShape(it->second->shape);
        if (shape.empty()) {
            throw std::runtime_error("No vailed shape");
        }
        std::string mapp_name = it->first;
        std::vector<uint8_t> out;
        if (!ReadTensorRaw(gguf, gguf_read.data_offset_, out, *(it->second))) {
            throw std::runtime_error("mssing data");
        }
        return LoadQuantTensor(out, shape, static_cast<int>(it->second->type));
     };
     //可为空的权重
     auto Load_Optional_Weight =  [&] (const std::string& tensor_name) ->QuanTensor {
        if (tensor_maps.find(tensor_name) == tensor_maps.end()) {
            return QuanTensor();
        } else {
            return Load_Weight(tensor_name);
        }
     };
     // 转为普通tensor
     auto Load_Weight_Tensor = [&](const std::string& tensor_name) ->Tensor {
        QuanTensor q = Load_Weight(tensor_name);
        switch(q.type) {
            case QuantType ::kF32 :{
                return ToTensor(q);
                break;
            }
            case QuantType ::kQ80 :{
                 return Q80ToTensor(q);
                 break;
            }
            case QuantType ::kQ40 : {
                 return Q40ToTensor(q);
            }
            case QuantType ::kQ41 : {
                return Q41ToTensor(q);
                break;
            }
            default : {
                return Tensor();
                break;
            }
        }
     };
     auto Load_Weight_Tensor_Optional = [&](const std::string& tensor_name)->Tensor {
        if (tensor_maps.find(tensor_name) == tensor_maps.end()) {
            return Tensor();
        }
        return Load_Weight_Tensor(tensor_name);
     };
     (void)Load_Optional_Weight;  // 未直接调用,保留签名避免 unused warning

     try {
        // 顶层
        model.embedding_data_ = Load_Weight_Tensor("embedding_data_");
        model.final_norm      = Load_Weight_Tensor("final_norm");
        // GGUF may omit output.weight when it shares token_embd.weight.
        // Load the original quantized tensor rather than duplicating the F32 embedding.
        model.lm_head_ = Load_Weight(tensor_maps.count("lm_head") ? "lm_head" : "embedding_data_");
        // 每一层
        model.n_layers_.resize(static_cast<size_t>(model.config.layers));
        for (int i = 0; i < model.config.layers; ++i) {
            Layer& L = model.n_layers_[static_cast<size_t>(i)];
            const std::string pre = "blk." + std::to_string(i) + ".";
            L.attention_norm = Load_Weight_Tensor         (pre + "attention_norm");
            L.wq             = Load_Weight                (pre + "wq");
            L.wk             = Load_Weight                (pre + "wk");
            L.wv             = Load_Weight                (pre + "wv");
            L.wo             = Load_Weight                (pre + "wo");
            L.bq             = Load_Weight_Tensor_Optional(pre + "bq");
            L.bk             = Load_Weight_Tensor_Optional(pre + "bk");
            L.bv             = Load_Weight_Tensor_Optional(pre + "bv");
            L.ffn_norm       = Load_Weight_Tensor         (pre + "ffn_norm");
            L.w_gate         = Load_Weight                (pre + "w_gate");
            L.w_up           = Load_Weight                (pre + "w_up");
            L.w_down         = Load_Weight                (pre + "w_down");
        }

        VailedShape(model);
        model.is_loaded = true;
     } catch (const std::exception& e) {
        model.load_error_ = e.what();
        std::cout << "ModelLoad failed: " << e.what() << "\n";
        return model;
     }
     return model;
}
}//namespace yan_lamma
