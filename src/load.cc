#include "include/load.h"

#include <cstring>
#include <string>
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <cctype>
#include <limits>
namespace yan_lamma {
namespace {
Model LoadError(const std::string& message) {
    Model model;
    model.is_loaded = false;
    model.load_error_ = message;
    std::cerr << " Model Load Failed" << message << "\n";
    return model;
}
size_t FindOptionalKey(const std::string& content, const std::string& optional) {
    const std::string temp = "\"" + optional + "\"";
    size_t pos = content.find(temp);
    if (pos == std::string::npos) {
        return pos;
    }
    pos += temp.size();
    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos >= content.size() || content[pos] != ':') {
        throw std::runtime_error("Find : Failed: " + optional);
    }
    ++pos;
    while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos >= content.size() || content[pos] == ',' || content[pos] == '}' || content[pos] == ']') {
        throw std::runtime_error("Find Value Failed: " + optional);
    }
    return pos;
}
size_t FindConfigKey(const std::string& content, const std::string& key) {
    size_t pos = FindOptionalKey(content, key);
    if (pos == std::string::npos) {
        throw std::runtime_error("missing required field: " + key);
    }
    return pos;
}
std::string FindTextKey(const std::string& content, const std::string& text) {
    size_t pos = FindOptionalKey(content, text);
    if (pos == std::string::npos) {
        return "";
    }
    if (content[pos] != '{') {
        throw std::runtime_error("expected object for field: " + text);
    }
    int depth = 0;
    for (size_t i = pos; i < content.size(); ++i) {
        if (content[i] == '{') {
            ++depth;
        } else if (content[i] == '}') {
            --depth;
            if (depth == 0) {
                return content.substr(pos, i - pos + 1);
            }
        }
    }
    throw std::runtime_error("unclosed object for field: " + text);
}
//去掉空白
std::string Trim(const std::string& s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) {
        a++;
    }
    while (b > a && std::isspace(static_cast<unsigned char>(s[b -1]))) {
        b--;
    }
    return s.substr(a, b - a);
}
// 判断字符是否能够作为边界
bool IsJsonValue(char c) {
    return std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == '}' || c == ']';
}
bool ValidNumber(const std::string& content,
                 const std::string& key,
                 size_t parsed) {
    size_t flag = FindConfigKey(content, key);
    if (parsed == 0 || parsed > content.size() - flag) {
        return false;
    }
    size_t pos = flag + parsed;
    while (pos < content.size() &&
           std::isspace(static_cast<unsigned char>(content[pos]))) {
        ++pos;
    }
    if (pos == content.size()) {
        return true;
    }
    return IsJsonValue(content[pos]);
}
int ParseRequiredInt(const std::string& content, const std::string& key) {
    size_t pos = FindConfigKey(content, key);
    size_t parse = 0;
    int value = std::stoi(content.substr(pos), &parse);
    if (parse == 0) {
        throw std::runtime_error("Parse Int Failed");
    }
    if ( !ValidNumber(content, key, parse)) {
        throw std::runtime_error ("No Such Current Number");
    }
    return value;
}
float ParseRequireFloat(const std::string& content, const std::string& key) {
    size_t pos = FindConfigKey(content, key);
    size_t parse = 0;
    float value = std::stof(content.substr(pos), &parse);
    if (parse == 0) {
        throw std::runtime_error("Parse Failed");
    } 
    if (!ValidNumber(content, key, parse) || !std::isfinite(value)) {
        throw std::runtime_error("No Such Current Numver");
    }
    return value;
}
int ParseRequireIntOption(const std::string& content, const std::string& key, int call) {
    if (FindOptionalKey(content, key) == std::string::npos) {
        return call;
    }
    return ParseRequiredInt(content, key);
}
std::string ParseStringField(const std::string& content,
                             const std::string& key, std::string call) {
    size_t pos = FindOptionalKey(content, key);
    if (pos == std::string::npos) {
        return call;
    }
    if (content[pos] != '"') {
        throw std::runtime_error("expected string for field: " + key);
    }
    size_t end = content.find('"', pos + 1);
    if (end == std::string::npos) {
        throw std::runtime_error("unclosed string for field: " + key);
    }
    return content.substr(pos + 1, end - pos - 1);
}
// parse Tonizer
TonizerInfo ParseTonizer(const std::string& content) {
    std::string temps = FindTextKey(content, "tokenizer");
    TonizerInfo res;
    if (temps.empty()) {
        return res;
    }
    res.type = ParseStringField(temps, "type", res.type);
    res.path_name = ParseStringField(temps, "pathname", res.path_name);
    if (res.type == "JsonVocab") {
        res.bos_id = 1;
        res.eos_id = 2;
        res.unk_id = 0;
    }
    res.bos_id = ParseRequireIntOption(temps, "bos_id", res.bos_id);
    res.eos_id = ParseRequireIntOption(temps, "eos_id", res.eos_id);
    res.unk_id = ParseRequireIntOption(temps, "unk_id", res.unk_id);
    //rule
    if (res.type != "ASCII" && res.type != "JsonVocab") {
        throw std::runtime_error("No current Tonizer Type: " +res.type);
    }
    //显示三个id 
    if (res.bos_id < 0 || res.eos_id < 0 || res.unk_id < 0) {
        throw std::runtime_error("No Current Tonizer Id");
    }
    if (res.bos_id == res.eos_id || res.bos_id == res.unk_id || res.eos_id == res.unk_id) {
         throw std::runtime_error("No Current Tonizer Id");
    }
    if (res.type == "JsonVocab" && res.path_name.empty()) {
        throw std::runtime_error("Type and file mismatch");
    }
    return res;
}

std::vector<TensorInfo> ParseTensor(const std::string& content) {
    std::vector<TensorInfo> tensor_map;
    size_t tensor_open = FindConfigKey(content, "tensors");
    if (content[tensor_open] != '[') {
        throw std::runtime_error("expected array for tensors");
    }
    size_t i = tensor_open + 1;
    // 沿用 depth，避免 shape 的 [] 影响外层数组。
    int depth = 1;
    while (i < content.size() && depth > 0) {
        while (i < content.size() && depth > 0 && content[i] != '{') {
            if (content[i] == '[') {
                ++depth;
            } else if (content[i] == ']') {
                --depth;
            }
            ++i;
        }
        if (i >= content.size() || depth == 0) {
            break;
        }
        size_t start = i;
        size_t end = content.find('}', start);
        if (end == std::string::npos) {
            throw std::runtime_error("No Such End");
        }
        std::string obj = content.substr(start, end - start + 1);
        TensorInfo tensor;
        bool has_name = false;
        bool has_shape = false;
        bool has_dtype = false;
        bool has_offset = false;
        bool has_byte_size = false;
        if (FindOptionalKey(obj, "name") != std::string::npos) {
            tensor.tensor_name = ParseStringField(obj, "name", "");
            has_name = true;
        }
        size_t shape_start = FindOptionalKey(obj, "shape");
        if (shape_start != std::string::npos) {
            if (obj[shape_start] != '[') {
                throw std::runtime_error("expected array for tensor shape");
            }
            size_t shape_end = obj.find(']', shape_start + 1);
            if (shape_end == std::string::npos) {
                throw std::runtime_error("unclosed tensor shape");
            }
            size_t after_shape = shape_end + 1;
            while (after_shape < obj.size() &&
                   std::isspace(static_cast<unsigned char>(obj[after_shape]))) {
                ++after_shape;
            }
            if (after_shape >= obj.size() ||
                (obj[after_shape] != ',' && obj[after_shape] != '}')) {
                throw std::runtime_error("invalid tensor shape boundary");
            }
            std::string shape_text = Trim(
                obj.substr(shape_start + 1, shape_end - shape_start - 1));
            if (shape_text.empty() || shape_text.back() == ',') {
                throw std::runtime_error("empty dimension in tensor shape");
            }
            std::stringstream ss(shape_text);
            std::string line;
            while (std::getline(ss, line, ',')) {
                line = Trim(line);
                if (line.empty()) {
                    throw std::runtime_error("empty dimension in tensor shape");
                }
                for (char digit : line) {
                    if (digit < '0' || digit > '9') {
                        throw std::runtime_error("invalid tensor dimension");
                    }
                }
                if (line.size() > 1 && line[0] == '0') {
                    throw std::runtime_error("invalid leading zero in tensor shape");
                }
                size_t parsed = 0;
                int dim = std::stoi(line, &parsed);
                if (parsed != line.size() || dim <= 0) {
                    throw std::runtime_error("tensor dimensions must be positive integers");
                }
                tensor.shape.push_back(dim);
            }
            has_shape = true;
        }
        if (FindOptionalKey(obj, "dtype") != std::string::npos) {
            tensor.dtype = ParseStringField(obj, "dtype", "");
            has_dtype = true;
        }
        size_t offset_pos = FindOptionalKey(obj, "offset");
        if (offset_pos != std::string::npos) {
            if (obj[offset_pos] == '-') {
                throw std::runtime_error("tensor offset must not be negative");
            }
            size_t parsed = 0;
            unsigned long long value = std::stoull(obj.substr(offset_pos), &parsed);
            if (!ValidNumber(obj, "offset", parsed) ||
                value > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("invalid tensor offset");
            }
            tensor.offset = static_cast<size_t>(value);
            has_offset = true;
        }
        size_t bytes_pos = FindOptionalKey(obj, "byte_size");
        if (bytes_pos != std::string::npos) {
            if (obj[bytes_pos] == '-') {
                throw std::runtime_error("tensor byte_size must not be negative");
            }
            size_t parsed = 0;
            unsigned long long value = std::stoull(obj.substr(bytes_pos), &parsed);
            if (!ValidNumber(obj, "byte_size", parsed) ||
                value > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("invalid tensor byte_size");
            }
            tensor.byte_size = static_cast<size_t>(value);
            has_byte_size = true;
        }
        if (!has_name || tensor.tensor_name.empty())
            throw std::runtime_error("Missing tensor name");
        if (!has_shape) throw std::runtime_error("Missing tensor shape");
        if (!has_dtype) throw std::runtime_error("Missing tensor dtype");
        if (!has_offset) throw std::runtime_error("Missing tensor offset");
        if (!has_byte_size) throw std::runtime_error("Missing tensor byte_size");
        tensor_map.push_back(tensor);
        i = end + 1;
    }
    if (depth != 0) {
        throw std::runtime_error("unclosed tensors array");
    }
    return tensor_map;
}
void VaileConfig (const ModelConfig& config) {
    /*struct ModelConfig {
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



};*/
    if (config.vocab_size <= 0 || config.dim <= 0 || config.hidden_dim <= 0 ||
        config.layers <= 0 || config.n_heads <= 0 || config.n_kv_heads <= 0 || config. head_dim <= 0 ||
        config. max_seq_len <= 0) {
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
}
// 检查字节数
size_t CheckByteSize(const TensorInfo& tensor) {
    size_t res = 1;
    const size_t limit = std::numeric_limits<size_t>::max();
    if (tensor.shape.empty()) {
        throw std::runtime_error("tensor shape must not be empty");
    }
    for (int dim : tensor.shape) {
        if (dim <= 0) {
            throw std::runtime_error("tensor dimensions must be positive");
        }
        const size_t d = static_cast<size_t>(dim);
        if (res > limit / d) {
            throw std::runtime_error("tensor element count overflow");
        }
        res *= d;
    }
    if (res > limit / sizeof(float)) {
        throw std::runtime_error("tensor byte_size overflow");
    }
    return res * sizeof(float);
}
std::vector<std::pair<std::string, std::vector<int>>> ExpectedTensor(const ModelConfig& config) {
    using expected_tensor = std::vector<std::pair<std::string, std::vector<int>>>;
    expected_tensor res = {
        {"token_embedding", {config.vocab_size, config.dim}},
        {"final_norm", {config.dim}},
        {"lm_head", {config.vocab_size, config.dim}}
    };
    for (int i = 0; i < config.layers; ++i) {
        const std::string prefix = "layers." + std::to_string(i) + ".";
        res.push_back({prefix + "attention_norm", {config.dim}});
        res.push_back({prefix + "wq", {config.n_heads * config.head_dim, config.dim}});
        res.push_back({prefix + "wk", {config.n_kv_heads * config.head_dim, config.dim}});
        res.push_back({prefix + "wv", {config.n_kv_heads * config.head_dim, config.dim}});
        res.push_back({prefix + "wo", {config.dim, config.n_heads * config.head_dim}});
        res.push_back({prefix + "ffn_norm", {config.dim}});
        res.push_back({prefix + "w_gate", {config.hidden_dim, config.dim}});
        res.push_back({prefix + "w_up", {config.hidden_dim, config.dim}});
        res.push_back({prefix + "w_down", {config.dim, config.hidden_dim}});
    }
    return res;
}
void VailedModelJson(const ModelJson& modeljson) {
    if (modeljson.tensor_json.empty()) {
        throw std::runtime_error("Missing Tensor");
    }
    // 先检查清单数量，避免按异常层数生成大量期望 shape。
    const size_t count = modeljson.tensor_json.size();
    if (count < 3 ||
        static_cast<size_t>(modeljson.config.layers) > (count - 3) / 9) {
        throw std::runtime_error("missing required tensors for configured layers");
    }
    // 检查是否重名
    std::unordered_map<std::string, TensorInfo>tensor_map;
    std::vector<TensorInfo> off_sets;
    for (const auto& t : modeljson.tensor_json) {
        if (t.dtype != "float32") {
            throw std::runtime_error("Dtype not match");
        }
        if (t.shape.empty()) {
            throw std::runtime_error(t.tensor_name + "missing shape");
        }
         size_t expected = CheckByteSize(t);
        if (expected != t.byte_size) {
            throw std::runtime_error(t.tensor_name + ": no current size");
        }
        if (t.offset > std::numeric_limits<size_t>::max() - t.byte_size) {
            throw std::runtime_error("byte_size overflow");
        }
        if (tensor_map.find(t.tensor_name) != tensor_map.end()) {
            throw std::runtime_error("duplicate objects");
        }
        tensor_map[t.tensor_name] = t;
        off_sets.push_back(t);
    }
    std::sort(off_sets.begin(), off_sets.end(), [](const auto& a, const auto& b) {return a.offset < b.offset;});
    for (size_t i = 1; i < off_sets.size(); i++) {
        const auto& previous = off_sets[i - 1];
        size_t previous_temp = previous.byte_size + previous.offset;
        if (previous_temp >  off_sets[i].offset) {
            throw std::runtime_error("Byte overlap");
        }
    }  
      auto temp =  ExpectedTensor(modeljson.config);
        for (const auto& it : temp) {
            auto ten = tensor_map.find(it.first);
            if (ten == tensor_map.end()) {
                throw std::runtime_error(" missing require");
            }
            if (ten->second.shape != it.second) {
                throw std::runtime_error("shape not shaped");
            }
            
    }
}
ModelConfig ParseModelConfig(const std::string& content) {
    const std::string config_object = FindTextKey(content, "config");
    const std::string& fields = config_object.empty() ? content : config_object;
    ModelConfig config;
    config.vocab_size = ParseRequiredInt(fields, "vocab_size");
    config.dim = ParseRequiredInt(fields, "dim");
    config.hidden_dim = ParseRequiredInt(fields, "hidden_dim");
    config.layers = ParseRequiredInt(fields, "layers");
    config.n_heads = ParseRequiredInt(fields, "n_heads");
    config.n_kv_heads = ParseRequiredInt(fields, "n_kv_heads");
    config.head_dim = ParseRequiredInt(fields, "head_dim");
    config.max_seq_len = ParseRequiredInt(fields, "max_seq_len");
    config.rope_theta = ParseRequireFloat(fields, "rope_theta");
    config.rms_norm_eps = ParseRequireFloat(fields, "rms_norm_eps");
    VaileConfig(config);
    return config;
}
Tensor LoadTensorByname(
    std::ifstream& file,
    const std::unordered_map<std::string, TensorInfo>& tensor_map,
    const std::string& name) {
    auto it = tensor_map.find(name);
    if (it == tensor_map.end()) {
        throw std::runtime_error("missing tensor: " + name);
    }
    const TensorInfo& info = it->second;
    if (info.offset >
        static_cast<size_t>(std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("tensor offset exceeds streamoff range: " + name);
    }
    if (info.byte_size >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("tensor byte_size exceeds streamsize range: " + name);
    }

    file.seekg(static_cast<std::streamoff>(info.offset), std::ios::beg);
    if (!file.good()) {
        throw std::runtime_error("failed to seek to tensor: " + name);
    }

    Tensor tensor(info.shape, 0.0f);
    file.read(reinterpret_cast<char*>(tensor.data.data()),
              static_cast<std::streamsize>(info.byte_size));
    if (file.bad() ||
        static_cast<size_t>(file.gcount()) != info.byte_size) {
        throw std::runtime_error("failed to read complete tensor: " + name);
    }
    return tensor;
}
}//namespace 
ModelJson ParseModelJson(const std::string& config_path) {
    if (config_path.empty()) {
        throw std::runtime_error("No config Path");
    }
    std::ifstream f(config_path);
    if (! f.is_open()) {
        throw std::runtime_error("Config Path Open Failed");

    }
    std::stringstream buffer;
    buffer << f.rdbuf();
    if (f.bad()) {
        throw std::runtime_error("Write failed");
    }
    ModelJson modeljson;

    std::string str = Trim(buffer.str());
    if (str.empty() || str.front() != '{' || str.back() != '}') {
        throw std::runtime_error("expected model JSON object");
    }
    modeljson.config = ParseModelConfig(str);
    modeljson.toninzer_json = ParseTonizer(str);
    modeljson.tensor_json  = ParseTensor(str);
    VailedModelJson(modeljson);
    return modeljson;
}
Model ParseModel(const std::string& config_path,
                 const std::string& weight_path) {
    try {
        ModelJson json = ParseModelJson(config_path);
        std::ifstream file(weight_path, std::ios::binary);
        if (!file.is_open()) {
            return LoadError("failed to open weights file: " + weight_path);
        }

        file.seekg(0, std::ios::end);
        std::streamoff end = file.tellg();
        if (!file.good() || end < 0) {
            return LoadError("failed to determine weights file size");
        }
        const size_t file_size = static_cast<size_t>(end);
        size_t expected_end = 0;
        std::unordered_map<std::string, TensorInfo> tensor_map;
        for (const auto& info : json.tensor_json) {
            if (info.offset > file_size ||
                info.byte_size > file_size - info.offset) {
                return LoadError("tensor exceeds weights file: " + info.tensor_name);
            }
            const size_t tensor_end = info.offset + info.byte_size;
            expected_end = std::max(expected_end, tensor_end);
            tensor_map.emplace(info.tensor_name, info);
        }
        if (file_size != expected_end) {
            return LoadError("weights file has trailing bytes: file size=" +
                             std::to_string(file_size) + ", expected=" +
                             std::to_string(expected_end));
        }
        file.seekg(0, std::ios::beg);
        if (!file.good()) {
            return LoadError("failed to seek to beginning of weights file");
        }

        Model model;
        model.config = json.config;
        model.embedding_data_ =
            LoadTensorByname(file, tensor_map, "token_embedding");
        model.n_layers_.resize(static_cast<size_t>(model.config.layers));
        for (int i = 0; i < model.config.layers; ++i) {
            Layer& layer = model.n_layers_[static_cast<size_t>(i)];
            const std::string prefix = "layers." + std::to_string(i) + ".";
            layer.attention_norm =
                LoadTensorByname(file, tensor_map, prefix + "attention_norm");
            layer.wq = ToQuanTensor(
                LoadTensorByname(file, tensor_map, prefix + "wq"));
            layer.wk = ToQuanTensor(
                LoadTensorByname(file, tensor_map, prefix + "wk"));
            layer.wv = ToQuanTensor(
                LoadTensorByname(file, tensor_map, prefix + "wv"));
            layer.wo = ToQuanTensor(
                LoadTensorByname(file, tensor_map, prefix + "wo"));
            layer.ffn_norm =
                LoadTensorByname(file, tensor_map, prefix + "ffn_norm");
            layer.w_gate = ToQuanTensor(
                LoadTensorByname(file, tensor_map, prefix + "w_gate"));
            layer.w_up = ToQuanTensor(
                LoadTensorByname(file, tensor_map, prefix + "w_up"));
            layer.w_down = ToQuanTensor(
                LoadTensorByname(file, tensor_map, prefix + "w_down"));
        }
        model.final_norm = LoadTensorByname(file, tensor_map, "final_norm");
        model.lm_head_ =
            ToQuanTensor(LoadTensorByname(file, tensor_map, "lm_head"));
        model.is_loaded = true;
        std::cout << "Model loaded successfully from " << weight_path << '\n';
        return model;
    } catch (const std::exception& error) {
        return LoadError(error.what());
    }
}

bool InspectModel(const std::string& config_path) {
    try {
        ModelJson json = ParseModelJson(config_path);

        const ModelConfig& config = json.config;
        std::cout << "=== Model Config ===\n";
        std::cout << "vocab_size: " << config.vocab_size << "\n";
        std::cout << "dim: " << config.dim << "\n";
        std::cout << "hidden_dim: " << config.hidden_dim << "\n";
        std::cout << "layers: " << config.layers << "\n";
        std::cout << "n_heads: " << config.n_heads << "\n";
        std::cout << "n_kv_heads: " << config.n_kv_heads << "\n";
        std::cout << "head_dim: " << config.head_dim << "\n";
        std::cout << "max_seq_len: " << config.max_seq_len << "\n";
        std::cout << "rope_theta: " << config.rope_theta << "\n";
        std::cout << "rms_norm_eps: " << config.rms_norm_eps << "\n";
        std::cout << "rope_type: "
                  << (config.rope_type == RopeType::kNeoX
                          ? "NeoX" : "Normal")
                  << "\n";

        std::cout << "\n=== Tensors ===\n";
        for (const auto& tensor : json.tensor_json) {
            std::cout << tensor.tensor_name << "\n";
            std::cout << "  shape: [";
            for (size_t i = 0; i < tensor.shape.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << tensor.shape[i];
            }
            std::cout << "]\n";
            std::cout << "  dtype: " << tensor.dtype << "\n";
            std::cout << "  offset: " << tensor.offset << "\n";
            std::cout << "  byte_size: " << tensor.byte_size << "\n";
        }

        const TonizerInfo& tokenizer = json.toninzer_json;
        std::cout << "\n=== Tokenizer ===\n";
        std::cout << "type: " << tokenizer.type << "\n";
        std::cout << "pathname: " << tokenizer.path_name << "\n";
        std::cout << "bos_id: " << tokenizer.bos_id << "\n";
        std::cout << "eos_id: " << tokenizer.eos_id << "\n";
        std::cout << "unk_id: " << tokenizer.unk_id << "\n";

        return true;
    } catch (const std::exception& error) {
        std::cerr << "Inspect failed: " << error.what() << "\n";
        return false;
    }
}
}//namespace yan_lamma 