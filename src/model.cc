#include "include/cuda_runtime.h"
#include "include/model.h"
#include "include/quant.h"
#include "include/quantized_tensor.h"
#include "include/tensor.h"
#include <cstddef>
#include <limits>
#include <iostream>

#include <stdexcept>
#include <utility>
namespace yan_lamma {
namespace {
size_t TensorBytes(const Tensor& tensor) {
    return sizeof(float) * tensor.GetDataSize();
}
size_t QuanTensorBytes(const QuanTensor& tensor) {
    switch(tensor.type) {
        case yan_lamma::QuantType::kF16 :
        return tensor.kF16_data.size();
        case yan_lamma::QuantType::kF32 : 
        return sizeof(float) * tensor.kF32_data.size();
        case yan_lamma::QuantType::kQ40 :
        return sizeof(Block_Q40) * tensor.kQ40_data.size();
        case yan_lamma::QuantType::kQ41 :
        return sizeof(Block_Q41) * tensor.kQ41_data.size();
        case yan_lamma::QuantType::kQ80 :
        return sizeof(Block_Q80) * tensor.kQ80_data.size();
        default :
        return sizeof(float) * tensor.GetNumSize();
    }
}
size_t QuanTensorBytesF32(const QuanTensor& tensor) {
    return sizeof(float) * tensor.GetNumSize();
}

}// namespace
//量化情况所占字节
size_t ModelWeightBytes(const Model& model) {
    size_t bytes = 0;
    bytes += TensorBytes(model.embedding_data_);
    bytes += TensorBytes(model.final_norm);
    bytes +=  QuanTensorBytes(model.lm_head_);
    for (const auto& lw : model.n_layers_) {
        bytes += TensorBytes(lw.attention_norm);
        bytes += QuanTensorBytes(lw.wq);
        bytes += QuanTensorBytes(lw.wk);
        bytes += QuanTensorBytes(lw.wv);
        bytes += TensorBytes(lw.bq);
        bytes += TensorBytes(lw.bk);
        bytes += TensorBytes(lw.bv);
        bytes += QuanTensorBytes(lw.wo);
        bytes += TensorBytes(lw.ffn_norm);
        bytes += QuanTensorBytes(lw.w_gate);
        bytes += QuanTensorBytes(lw.w_up);
        bytes += QuanTensorBytes(lw.w_down);
  }
  return bytes;
}
//F32效果 用来观察量化效果
size_t ModelWeightBytesF32(const Model& model) {
     size_t bytes = 0;
    bytes += TensorBytes(model.embedding_data_);
    bytes += TensorBytes(model.final_norm);
    bytes +=  QuanTensorBytesF32(model.lm_head_);
    for (const auto& lw : model.n_layers_) {
        bytes += TensorBytes(lw.attention_norm);
        bytes += QuanTensorBytesF32(lw.wq);
        bytes += QuanTensorBytesF32(lw.wk);
        bytes += QuanTensorBytesF32(lw.wv);
        bytes += TensorBytes(lw.bq);
        bytes += TensorBytes(lw.bk);
        bytes += TensorBytes(lw.bv);
        bytes += QuanTensorBytesF32(lw.wo);
        bytes += TensorBytes(lw.ffn_norm);
        bytes += QuanTensorBytesF32(lw.w_gate);
        bytes += QuanTensorBytesF32(lw.w_up);
        bytes += QuanTensorBytesF32(lw.w_down);
    }
    return bytes;
}
void UploadLinearF32(CudaModelWeights& arry, int device_id, const std::string& name, const QuanTensor& tensor) {
    if (tensor.type != QuantType::kF32) {
        return;
    }
    if (tensor.kF32_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty F32 data for " + name);
    }
    if (tensor.GetNumSize() != tensor.kF32_data.size()) {
        throw std::runtime_error("CUDA weight upload: data size mismatch for " +
                             name + ", expected " +
                             std::to_string(tensor.GetNumSize()) + ", got " +
                             std::to_string(tensor.kF32_data.size()));
    }
    size_t byte_size =  QuanTensorBytesF32(tensor);
    CudaloadWeight weight;
    weight.linear = true;
    weight.block_count = 0;
    weight.name = name;
    weight.shape = tensor.shape;
    weight.type = tensor.type;
    CudaDeviceBuffer buffer;
    buffer.Reset(byte_size, device_id);
    buffer.upload(tensor.kF32_data.data(), byte_size);
    weight.buffer = std::move(buffer);
    arry.weights.push_back(std::move(weight));
    arry.device_id  = device_id;
    arry.uploaded_weight_count++;
    arry.uploaded_bytes += byte_size;
}
// 不参加线性投影
void UploadTensorF32(CudaModelWeights& arry, int device_id, const std::string& name, const Tensor& tensor) {
    if(tensor.data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty F32 data for " + name);
    }
    if (tensor.shape.empty()) {
        throw std::runtime_error("CUDA weight upload: empty shape for " + name);
    }
    size_t expected_elements = 1;
    for (int dimension : tensor.shape) {
        if (dimension <= 0) {
            throw std::runtime_error("CUDA weight upload: dimensions must be positive for " + name);
        }
        const size_t dim = static_cast<size_t>(dimension);
        if (expected_elements > std::numeric_limits<size_t>::max() / dim) {
            throw std::runtime_error("CUDA weight upload: element count overflow for " + name);
        }
        expected_elements *= dim;
    }
    if (expected_elements != tensor.data.size()) {
        throw std::runtime_error(
            "CUDA weight upload: F32 data size mismatch for " + name +
            ", expected elements " + std::to_string(expected_elements) +
            ", got " + std::to_string(tensor.data.size()));
    }
    size_t byte_size = sizeof(float) * tensor.data.size();
    CudaloadWeight weight;
    weight.linear = false;
    weight.block_count = 0;
    weight.name = name;
    weight.shape = tensor.shape;
    weight.type = QuantType::kF32;
    CudaDeviceBuffer buffer;
    buffer.Reset(byte_size, device_id);
    buffer.upload(tensor.data.data(), byte_size);
    weight.buffer = std::move(buffer);
    arry.device_id = device_id;
    arry.uploaded_weight_count++;
    arry.uploaded_bytes +=  byte_size;
    arry.weights.push_back(std::move(weight));
}
void UploadLinearQ80(CudaModelWeights& arry, int device_id, const std::string name,  const QuanTensor& tensor) {
    if (tensor.type != QuantType::kQ80) {
        return;
    }
    if (tensor.kQ80_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q80 data for " + name);
    }
    if (tensor.shape.size() != 2 || tensor.shape[0] <= 0 || tensor.shape[1] <= 0) {
        throw std::runtime_error("CUDA weight upload: expected positive 2D shape for " + name);
    }
    //行数
    size_t out_features = static_cast<size_t>(tensor.shape[0]);
    //每行元素数
    size_t in_features = static_cast<size_t>(tensor.shape[1]);
    //每行块数 不足补0
    size_t block_per_row = (in_features + kQuantBlockSize - 1) / kQuantBlockSize;
    if (out_features > std::numeric_limits<size_t>::max() / block_per_row) {
        throw std::runtime_error("CUDA weight upload: block count overflow for " + name);
    }
    size_t expexted_block = block_per_row  * out_features;
    if (expexted_block != tensor.kQ80_data.size()) {
        throw std::runtime_error(
            "CUDA weight upload: Q80 block count/data size mismatch for " + name +
            ", calculated blocks " + std::to_string(expexted_block) +
            ", actual blocks " + std::to_string(tensor.kQ80_data.size()));
    }
    size_t bytes =  kBytesPerQ80Block * tensor.kQ80_data.size();
    CudaloadWeight weight;
    weight.block_count = expexted_block;
    weight.linear = true;
    weight.name = name;
    weight.shape = tensor.shape;
    weight.type = tensor.type;
    CudaDeviceBuffer buffer;
    buffer.Reset(bytes, device_id);
    buffer.upload(tensor.kQ80_data.data(), bytes);
    weight.buffer = std::move(buffer);
    arry.device_id = device_id;
    arry.uploaded_bytes += bytes;
    arry.uploaded_weight_count++;
    arry.weights.push_back(std::move(weight));
}
void UploadLinearQ40(CudaModelWeights& arry, int device_id, const std::string name,  const QuanTensor& tensor) {
    if (tensor.type != QuantType::kQ40) {
        return;
    }
    if (tensor.kQ40_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q40 data for " + name);
    }
    if (tensor.shape.size() != 2 || tensor.shape[0] <= 0 || tensor.shape[1] <= 0) {
        throw std::runtime_error("CUDA weight upload: expected positive 2D shape for " + name);
    }
    //行数
    size_t out_features = static_cast<size_t>(tensor.shape[0]);
    //每行元素数
    size_t in_features = static_cast<size_t>(tensor.shape[1]);
    //每行块数 不足补0
    size_t block_per_row = (in_features + kQuantBlockSize - 1) / kQuantBlockSize;
    if (out_features > std::numeric_limits<size_t>::max() / block_per_row) {
        throw std::runtime_error("CUDA weight upload: block count overflow for " + name);
    }
    size_t expexted_block = block_per_row  * out_features;
    if (expexted_block != tensor.kQ40_data.size()) {
        throw std::runtime_error(
            "CUDA weight upload: Q40 block count/data size mismatch for " + name +
            ", calculated blocks " + std::to_string(expexted_block) +
            ", actual blocks " + std::to_string(tensor.kQ40_data.size()));
    }
    size_t bytes =  kBytesPerQ40Block * tensor.kQ40_data.size();
    CudaloadWeight weight;
    weight.block_count = expexted_block;
    weight.linear = true;
    weight.name = name;
    weight.shape = tensor.shape;
    weight.type = tensor.type;
    CudaDeviceBuffer buffer;
    buffer.Reset(bytes, device_id);
    buffer.upload(tensor.kQ40_data.data(), bytes);
    weight.buffer = std::move(buffer);
    arry.device_id = device_id;
    arry.uploaded_bytes += bytes;
    arry.uploaded_weight_count++;
    arry.weights.push_back(std::move(weight));
}
void UploadLinearQ41(CudaModelWeights& arry, int device_id, const std::string name,  const QuanTensor& tensor) {
    if (tensor.type != QuantType::kQ41) {
        return;
    }
    if (tensor.kQ41_data.empty()) {
        throw std::runtime_error("CUDA weight upload: empty Q41 data for " + name);
    }
    if (tensor.shape.size() != 2 || tensor.shape[0] <= 0 || tensor.shape[1] <= 0) {
        throw std::runtime_error("CUDA weight upload: expected positive 2D shape for " + name);
    }
    //行数
    size_t out_features = static_cast<size_t>(tensor.shape[0]);
    //每行元素数
    size_t in_features = static_cast<size_t>(tensor.shape[1]);
    //每行块数 不足补0
    size_t block_per_row = (in_features + kQuantBlockSize - 1) / kQuantBlockSize;
    if (out_features > std::numeric_limits<size_t>::max() / block_per_row) {
        throw std::runtime_error("CUDA weight upload: block count overflow for " + name);
    }
    size_t expexted_block = block_per_row  * out_features;
    if (expexted_block != tensor.kQ41_data.size()) {
        throw std::runtime_error(
            "CUDA weight upload: Q41 block count/data size mismatch for " + name +
            ", calculated blocks " + std::to_string(expexted_block) +
            ", actual blocks " + std::to_string(tensor.kQ41_data.size()));
    }
    size_t bytes =  kBytesPerQ41Block * tensor.kQ41_data.size();
    CudaloadWeight weight;
    weight.block_count = expexted_block;
    weight.linear = true;
    weight.name = name;
    weight.shape = tensor.shape;
    weight.type = tensor.type;
    CudaDeviceBuffer buffer;
    buffer.Reset(bytes, device_id);
    buffer.upload(tensor.kQ41_data.data(), bytes);
    weight.buffer = std::move(buffer);
    arry.device_id = device_id;
    arry.uploaded_bytes += bytes;
    arry.uploaded_weight_count++;
    arry.weights.push_back(std::move(weight));
}
static void UploadLinear(CudaModelWeights& arry, int device_id, const std::string name,  const QuanTensor& tensor) {
    UploadLinearF32(arry, device_id, name, tensor);
    if (tensor.type != QuantType::kF32 && tensor.type != QuantType::kQ80 &&
        tensor.type != QuantType::kQ40 && tensor.type != QuantType::kQ41) {
        std::cerr << "cuda fallback: unsupported or missing quantized linear weights use CPU path\n";
        return;
    }
    UploadLinearQ40(arry, device_id, name, tensor);
    UploadLinearQ41(arry, device_id, name, tensor);
    UploadLinearQ80(arry, device_id, name, tensor);
}
void UploadModelWeightToCuda(Model& model, int device_id) {
    if (!model.is_loaded) {
        throw std::runtime_error("UploadModelWeightToCuda: model is not loaded");
    }
    model.cuda_weights = std::make_shared<CudaModelWeights>();
    CudaModelWeights& weights = *model.cuda_weights;
    weights.device_id = device_id;

    UploadTensorF32(weights, device_id, "embedding_data_", model.embedding_data_);
    UploadTensorF32(weights, device_id, "final_norm", model.final_norm);
    UploadLinear(weights, device_id, "lm_head", model.lm_head_);

    for (size_t i = 0; i < model.n_layers_.size(); ++i) {
        const Layer& layer = model.n_layers_[i];
        const std::string prefix = "blk." + std::to_string(i) + ".";
        UploadTensorF32(weights, device_id, prefix + "attention_norm", layer.attention_norm);
        UploadLinear(weights, device_id, prefix + "wq", layer.wq);
        UploadLinear(weights, device_id, prefix + "wk", layer.wk);
        UploadLinear(weights, device_id, prefix + "wv", layer.wv);
        if (!layer.bq.data.empty()) {
            UploadTensorF32(weights, device_id, prefix + "bq", layer.bq);
        }
        if (!layer.bk.data.empty()) {
            UploadTensorF32(weights, device_id, prefix + "bk", layer.bk);
        }
        if (!layer.bv.data.empty()) {
            UploadTensorF32(weights, device_id, prefix + "bv", layer.bv);
        }
        UploadLinear(weights, device_id, prefix + "wo", layer.wo);
        UploadTensorF32(weights, device_id, prefix + "ffn_norm", layer.ffn_norm);
        UploadLinear(weights, device_id, prefix + "w_gate", layer.w_gate);
        UploadLinear(weights, device_id, prefix + "w_up", layer.w_up);
        UploadLinear(weights, device_id, prefix + "w_down", layer.w_down);
    }
}
void ClearCudaWeight(Model& model) {
    model.cuda_weights.reset();
}

size_t ModelCudaLinearCpuFallbacks(const Model& model) {
    if (!model.is_loaded) throw std::runtime_error("ModelCudaLinearCpuFallbacks: model is not loaded");
    return model.cuda_weights ? model.cuda_weights->linear_cpu_fallbacks : 0;
}
size_t ModelCudaLinearCalls(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaLinearCalls: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->linear_calls : 0;
}
size_t ModelCudaActivationCalls(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaActivationCalls: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->activation_calls : 0;
}
size_t ModelCudaAttentionCalls(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaAttentionCalls: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->attention_calls : 0;
}
size_t ModelCudaAttentionCpuFallbacks(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaAttentionCpuFallbacks: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->attention_cpu_fallbacks : 0;
}
void ResetModelCudaRuntimeStats(Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ResetModelCudaRuntimeStats: model is not loaded");
    }
    if (!model.cuda_weights) {
        return;
    }
    CudaModelWeights& weights = *model.cuda_weights;
    weights.linear_calls = 0;
    weights.linear_cpu_fallbacks = 0;
    weights.activation_calls = 0;
    weights.attention_calls = 0;
    weights.attention_cpu_fallbacks = 0;
    weights.kv_cache_write_bytes = 0;
    weights.kv_cache_read_bytes = 0;
    weights.host_to_device_copies = 0;
    weights.device_to_host_copies = 0;
    weights.host_to_device_bytes = 0;
    weights.device_to_host_bytes = 0;
}
size_t ModelCudaKvCacheWriteBytes(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaKvCacheWriteBytes: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->kv_cache_write_bytes : 0;
}
size_t ModelCudaKvCacheReadBytes(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaKvCacheReadBytes: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->kv_cache_read_bytes : 0;
}
size_t ModelCudaHostToDeviceCopies(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaHostToDeviceCopies: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->host_to_device_copies : 0;
}
size_t ModelCudaDeviceToHostCopies(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaDeviceToHostCopies: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->device_to_host_copies : 0;
}
size_t ModelCudaHostToDeviceBytes(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaHostToDeviceBytes: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->host_to_device_bytes : 0;
}
size_t ModelCudaDeviceToHostBytes(const Model& model) {
    if (!model.is_loaded) {
        throw std::runtime_error("ModelCudaDeviceToHostBytes: model is not loaded");
    }
    return model.cuda_weights ? model.cuda_weights->device_to_host_bytes : 0;
}
// 提供量化tensor到Q80和Q40的转换
Tensor QuanTensorToTensor(QuanTensor& tensor) {
    switch(tensor.type) {
        case yan_lamma::QuantType::kF32 :
        return ToTensor(tensor);
        case yan_lamma::QuantType::kQ40 :
        return  Q40ToTensor(tensor.kQ40_data, tensor.shape);
        case yan_lamma::QuantType::kQ41 :
        return  Q41ToTensor(tensor.kQ41_data, tensor.shape);
        case yan_lamma::QuantType::kQ80 :
        return  Q80ToTensor(tensor.kQ80_data, tensor.shape);
        case yan_lamma::QuantType::kF16 :
        throw std::runtime_error("QuanTensorToTensor: F16 conversion is not supported");
        default :
        throw std::runtime_error("QuanTensorToTensor: unsupported quantization type");
    }
}
void ToQuanTensorQ80(QuanTensor& tensor) {
    if (tensor.type == QuantType::kQ80) {
        return;
    }
    Tensor t = QuanTensorToTensor(tensor);
    tensor.kQ80_data = TensorToQ80(t);
    tensor.kF16_data.clear();
    tensor.kF16_data.shrink_to_fit();
    tensor.kF32_data.clear();
    tensor.kF32_data.shrink_to_fit();
    tensor.kQ40_data.clear();
    tensor.kQ40_data.shrink_to_fit();
    tensor.kQ41_data.clear();
    tensor.kQ41_data.shrink_to_fit();
    tensor.type = QuantType::kQ80;

}
void ModeloQuanTensorQ80 (Model& model) {
   ToQuanTensorQ80(model.lm_head_);
   for (auto& lw:model.n_layers_) {
        ToQuanTensorQ80(lw.wq);
        ToQuanTensorQ80(lw.wk);
        ToQuanTensorQ80(lw.wv);
        ToQuanTensorQ80(lw.wo);
        ToQuanTensorQ80(lw.w_gate);
        ToQuanTensorQ80(lw.w_up);
        ToQuanTensorQ80(lw.w_down);
   }
}
void ToQuanTensorQ40(QuanTensor& tensor) {
    if (tensor.type == QuantType::kQ40) {
        return;
    }
    Tensor t = QuanTensorToTensor(tensor);
    tensor.kQ40_data = TensorToQ40(t);
    tensor.kF16_data.clear();
    tensor.kF16_data.shrink_to_fit();
    tensor.kF32_data.clear();
    tensor.kF32_data.shrink_to_fit();
    tensor.kQ80_data.clear();
    tensor.kQ80_data.shrink_to_fit();
    tensor.kQ41_data.clear();
    tensor.kQ41_data.shrink_to_fit();
    tensor.type = QuantType::kQ40;
}
void ModeloQuanTensorQ40 (Model& model) {
   ToQuanTensorQ40(model.lm_head_);
   for (auto& lw:model.n_layers_) {
        ToQuanTensorQ40(lw.wq);
        ToQuanTensorQ40(lw.wk);
        ToQuanTensorQ40(lw.wv);
        ToQuanTensorQ40(lw.wo);
        ToQuanTensorQ40(lw.w_gate);
        ToQuanTensorQ40(lw.w_up);
        ToQuanTensorQ40(lw.w_down);
   }
}
}// namespace yan_lamma
