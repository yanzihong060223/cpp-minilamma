#include "include/gguf.h"

#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace yan_lamma {

const char* GgmlTypeName(Ggmltype type) {
    switch (type) {
        case Ggmltype::GgmltypeF32:
            return "F32";
        case Ggmltype::GgmltypeF16:
            return "F16";
        case Ggmltype::GgmltypeQ40:
            return "Q4_0";
        case Ggmltype::GgmltypeQ41:
            return "Q4_1";
        case Ggmltype::GgmlytypeQ80:
            return "Q8_0";
        default:
            return "UNKNOWN";
    }
}

const char* GgufValueTypeName(GgufValueType type) {
    switch (type) {
        case GgufValueType::kGgufTypeUint8:
            return "uint8";
        case GgufValueType::kGgufTypeInt8:
            return "int8";
        case GgufValueType::kGgufTypeUint16:
            return "uint16";
        case GgufValueType::kGgufTypeInt16:
            return "int16";
        case GgufValueType::kGgufTypeUint32:
            return "uint32";
        case GgufValueType::kGgufTypeInt32:
            return "int32";
        case GgufValueType::kGgufTypeFloat32:
            return "float32";
        case GgufValueType::kGgufTypeBool:
            return "bool";
        case GgufValueType::kGgufTypeString:
            return "string";
        case GgufValueType::kGgufTypeArray:
            return "array";
        case GgufValueType::kGgufTypeUint64:
            return "uint64";
        case GgufValueType::kGgufTypeInt64:
            return "int64";
        case GgufValueType::kGgufTypeFloat64:
            return "float64";
        default:
            return "unknown";
    }
}

namespace {
constexpr uint64_t kDefaultAlignment = 32;
constexpr uint64_t kMaxAlignment = 1024 * 1024;
constexpr uint64_t kMaxStringBytes = 16 * 1024 * 1024;
constexpr uint64_t kMaxMetadataEntries = 1000000;
constexpr uint64_t kMaxTensorEntries = 1000000;
constexpr uint64_t kMaxArrayElements = 10000000;
constexpr uint32_t kMaxTensorDims = 8;
constexpr uint64_t kQuantBlockSize = 32;
template<typename T>
bool ReadDl(std::ifstream& f, T&out) {
    char buf [sizeof(T)];
    if (!f.read(buf, sizeof(T))) {
        return false;
    }
    std::memcpy(&out, buf, sizeof(T));
    return true;
}
bool ReadU32(std::ifstream& f, uint32_t& out) {
    return ReadDl(f, out);
}
bool ReadU64(std::ifstream& f, uint64_t& out) {
    return ReadDl(f, out);
}
bool ReadString(std::ifstream& f, std::string& out) {
    uint64_t ssize = 0;
    if (!ReadU64(f, ssize)) {
        return false;
    }
    if (ssize > kMaxStringBytes || ssize > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        return false;
    }
    out.resize(ssize);
    if (!f.read(out.data(), static_cast<std::streamsize>(ssize))) {
        return false;
    }
    return true;
}

bool SkipValue(std::ifstream& f,  GgufValueType type) {
  switch (type) {
    case  GgufValueType::kGgufTypeUint8:
    case  GgufValueType::kGgufTypeInt8: {
      char dummy;
      return f.read(&dummy, 1).good();
    }
    case GgufValueType::kGgufTypeUint16:
    case GgufValueType::kGgufTypeInt16: {
      char dummy[2];
      return f.read(dummy, 2).good();
    }
    case GgufValueType::kGgufTypeUint32:
    case GgufValueType::kGgufTypeInt32:
    case GgufValueType::kGgufTypeFloat32: {
      char dummy[4];
      return f.read(dummy, 4).good();
    }
    case GgufValueType::kGgufTypeBool: {
      char dummy;
      return f.read(&dummy, 1).good();
    }
    case GgufValueType::kGgufTypeString: {
      std::string dummy;
      return ReadString(f, dummy);
    }
    case GgufValueType::kGgufTypeUint64:
    case GgufValueType::kGgufTypeInt64:
    case GgufValueType::kGgufTypeFloat64: {
      char dummy[8];
      return f.read(dummy, 8).good();
    }
    case GgufValueType::kGgufTypeArray: {
      uint32_t elem_type = 0;
      uint64_t len = 0;
      if (!ReadU32(f, elem_type) || !ReadU64(f, len)) {
        return false;
      }
      if (len > kMaxArrayElements) {
        return false;
      }
      for (uint64_t i = 0; i < len; ++i) {
        if (!SkipValue(f, static_cast<GgufValueType>(elem_type))) {
          return false;
        }
      }
      return true;
    }
    default:
      return false;
  }
}
// 检查乘法是否溢出
bool MultiplyChecked(uint64_t a, uint64_t b, uint64_t& out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<uint64_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}
bool TensorNumel(const GgufTensorInfo& tensor, uint64_t& out) {
    if (tensor.shape.empty()) {
        return false;
    }
    out = 1;
   for (uint64_t dim : tensor.shape) {
        if (!MultiplyChecked(out, dim, out)) {
            return false;
        }
   }
   return true;
}
bool TensorDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
    uint64_t sum = 1;
    if (!TensorNumel(tensor,sum)) {
        return false;
    }
    switch(tensor.type) {
        case Ggmltype::GgmltypeF32 :
        return  MultiplyChecked(sum, 4, out);
        break;
        case Ggmltype::GgmltypeF16 :
        return  MultiplyChecked(sum, 2, out);
        break;
        case Ggmltype::GgmltypeQ40 :
        if (sum % kQuantBlockSize != 0) {
            return false;
        }
        return MultiplyChecked((sum / kQuantBlockSize ), 18, out );
        break;
        case Ggmltype:: GgmltypeQ41 :
         if (sum % kQuantBlockSize != 0) {
            return false;
        }
        return MultiplyChecked((sum / kQuantBlockSize ), 20, out );
         case Ggmltype:: GgmlytypeQ80 :
          if (sum % kQuantBlockSize != 0) {
            return false;
        }
        return MultiplyChecked((sum / kQuantBlockSize ), 34, out );
        break;
        default:
        out = 0;
        return false;
    }   
}
bool AddCheck(uint64_t a, uint64_t b, uint64_t& out) {
    if ( a > std::numeric_limits<uint64_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}
//读取标量值并将其转换为字符串表示形式。
bool ReadIntToString(std::ifstream& f, GgufValueType type, std::string& out) {
  std::ostringstream oss;
  switch(type) {
    case GgufValueType:: kGgufTypeUint8 : {
    uint8_t v= 0;
    if (!ReadDl(f, v)) {
        return false;
     }
     oss << v;
     break;
  }
   case GgufValueType:: kGgufTypeInt8 : {
    int8_t v = 0;
    if (!ReadDl(f, v)) {
      return false;
    }
    oss << v;
    break;
  }
  case GgufValueType::kGgufTypeUint16 : {
    uint16_t v = 0;
    if (!ReadDl(f, v)) {
       return false;
    }
    oss << v;
    break;
  }
  case GgufValueType:: kGgufTypeInt16 : {
    int16_t v = 0;
    if (!ReadDl(f, v)) {
        return false;
    }
    oss << v;
    break;
  }
  case GgufValueType::kGgufTypeUint32 : {
    uint32_t v = 0;
    if (!ReadDl(f, v)) {
       return false;
    }
    oss << v;
    break;
  }
  case GgufValueType:: kGgufTypeInt32 : {
    int32_t v = 0;
    if (!ReadDl(f, v)) {
      return false;
    }
    oss << v;
    break;
  }
  case GgufValueType::  kGgufTypeFloat32 : {
    float v = 0;
    if (!ReadDl(f, v)) {
      return false;
    }
    oss << v;
    break;
  }
  case GgufValueType:: kGgufTypeBool : {
    bool v = true;
    if (!ReadDl(f, v)) {
        return false;
    }
    oss << v;
    break;
  }
  case GgufValueType:: kGgufTypeString : {
    std::string v = "";
    if (!ReadString(f, v)) {
      return false;
    }
    oss << v;
    break;
  }
  case GgufValueType:: kGgufTypeArray : {
    uint32_t data_type = 0;
    uint64_t data_size = 0;
    if (!ReadU32(f, data_type) || ! ReadU64(f, data_size)) {
      return false;
    }
    if (data_size >  kMaxArrayElements) {
        return false;
    }
     oss << "[" << data_size  << " x " << GgufValueTypeName(static_cast<GgufValueType>(data_type)) << "]";
     //跳过数组元素
     for (uint64_t i = 0; i < data_size; i++) {
        if (!SkipValue(f, static_cast<GgufValueType>(data_type))) {
          return false;
        }
     }
     break;
  }
  case GgufValueType::kGgufTypeUint64 : {
    uint64_t v = 0;
    if (!ReadU64(f, v)) {
        return false;
    }
    oss << v;
    break;
  }
  case GgufValueType:: kGgufTypeInt64 : {
    int64_t v = 0;
    if (!ReadDl(f, v)) {
      return false;
    }
    oss << v;
    break;
  }
  case GgufValueType:: kGgufTypeFloat64 : {
    double v = 0;
    if (!ReadDl(f, v)) {
      return false;
    }
    oss << v;
    break;
  }
  default:
  return false;
  break;
}// switch
  out = oss.str();
  return true;
}
// 读出按多少字节对齐
uint32_t ParseAlignment(const std::vector<GgufMetadaKv>& metadata) {
  uint32_t res = kDefaultAlignment;
  for (auto t : metadata) {
    if (t.name !=  "general.alignment") {
      continue;
    }
    size_t parse = 0;
    auto Alignment = std::stoull(t.value_str, &parse);
    if (parse != t.value_str.size()) {
      return 0;
    }
    res = static_cast< uint32_t >(Alignment);
  }
  return res;
}
bool FindMatedaKey(std::ifstream& f, const std::string& path, const std::string key, GgufValueType& outtype) {
  f.open(path, std::ios::binary);
  if (!f.is_open()) {
      return false;
  }
  //跳过
  f.seekg(8, std::ios::beg);
  uint64_t n_tensors = 0;
  uint64_t n_metadata = 0;
  if (!ReadU64(f, n_tensors) || !ReadU64(f, n_metadata)) {
    return false;
  }
  for (uint64_t i = 0; i < n_metadata; i++) {
    std::string name = "";
    if (!ReadString(f, name)) {
      return false;
    }
    GgufValueType type;
    if (!ReadDl(f, type)) {
      return false;
    }
    if (key == name) {
      outtype = type;
      return true;
    }
    if (!SkipValue(f, type)) {
      return false;
    }
  }
  return false;
}
}//namespace 
bool GgufRead::load(const std::string& file) {
  std::ifstream f(file,  std::ios::binary);
  if (! f.is_open()) {
     std::cerr << "gguf file open failed" << "\n";
     return false;
  }
  // 校验魔数
  char magic[4];
  std::string src = "GGUF";
  if (!f.read(magic, 4) || std::memcmp(magic, src.data(), 4)) {
      std::cerr << "Magic number verification failed" << "\n";
      return false;
  }
  if (!ReadDl(f, version_)) {
    std::cerr << "Read version failed" << "\n";
      return false;
  }
  if (version_ != 3) {
    std::cerr << "No current version" << "\n";
      return false;
  }
  if (!ReadDl(f,  n_tensors_) || !ReadDl(f,  n_metadata_)) {
      return false;
  }
  if (n_metadata_ > kMaxMetadataEntries || n_tensors_ > kMaxTensorEntries) {
      return false;
  }
  //metadakv
  metadakv_.reserve(n_metadata_);
  std::unordered_set<std::string> meta_set;
  for (size_t i = 0; i < n_metadata_; i++) {
     GgufMetadaKv kv;
     if (!ReadString(f, kv.name)) {
        return false;
     }
     if (! meta_set.insert(kv.name).second) {
      return false;
     }
     if (!ReadDl(f, kv.type)) {
      return false;
     }
     if (!ReadIntToString(f, kv.type, kv.value_str)) {
        return false;
     }
      metadakv_.push_back(std::move(kv));
  }
  //tensorinfo
  std::unordered_set<std::string> tensor_set;
  tensorInfo_.reserve(n_tensors_);
  for (size_t i = 0; i < n_tensors_; i++) {
    GgufTensorInfo tensor;
    if (!ReadString(f, tensor.name)) {
        return false;
    }
    if (!tensor_set.insert(tensor.name).second) {
        return false;
    }
    uint32_t dim = 0;
    if (!ReadDl(f, dim)) {
        return false;
    }
    if (dim > kMaxTensorDims) {
        return false;
    }
    tensor.shape.resize(dim);
    for (uint32_t j = 0; j < dim; j++) {
      if (!ReadU64(f, tensor.shape[j])) {
          return false;
      }
      if (tensor.shape[j] == 0) {
        return false;
      }
    }
    if (!ReadDl(f, tensor.type)) {
        return false;
    }
    if (!ReadDl(f, tensor.offset)) {
        return false;
    }
    tensorInfo_.push_back(std::move(tensor));
  }
  // 找到光标位置
 std::streampos oss = f.tellg();
 uint32_t alignment = ParseAlignment(metadakv_);
 data_offset_ = static_cast<uint32_t>(oss);
 if (alignment == 0) {
  return false;
 }
 uint64_t padding = (alignment - (data_offset_ % alignment)) % alignment;
  data_offset_ += padding;
// 计算大小
 f.seekg(0, std::ios::end);
 std::streampos file_size = f.tellg();
 if (static_cast<uint32_t>(file_size) < data_offset_) {
  return false;
 }
   for (size_t i = 0; i < n_tensors_; i++) {
    uint64_t data_size = 0;  
    if (!TensorDataSize(tensorInfo_[i], data_size )) {
        return false;
    }
    uint64_t tensor_start = 0;
    uint64_t tensor_end = 0;
    if (!AddCheck(data_offset_, tensorInfo_[i].offset,  tensor_start)) {
      return false;
    }
    if (!AddCheck(tensor_start, data_size, tensor_end)) {
      return false;
    }
    if (tensor_end > static_cast<uint64_t>(file_size)){
      return false;
    }
   }
   return true;
}
  bool GetGuffDataSize(const GgufTensorInfo& tensor, uint64_t& out) {
    return TensorDataSize(tensor, out);
}
bool Inspect(const GgufRead& read) {
  // 头信息
  std::cout << "=== GGUF Header ===" << "\n";
  std::cout << "version     : " << read.version_ << "\n";
  std::cout << "n_metadata  : " << read.n_metadata_ << "\n";
  std::cout << "n_tensors   : " << read.n_tensors_ << "\n";
  std::cout << "data_offset : " << read.data_offset_ << "\n";
  // 元数据
  std::cout << "=== Metadata (" << read.metadakv_.size() << ") ===" << "\n";
  for (const auto& kv : read.metadakv_) {
    std::cout << kv.name << " [" << GgufValueTypeName(kv.type) << "] = " << kv.value_str << "\n";
  }
  // 张量信息
  std::cout << "=== Tensors (" << read.tensorInfo_.size() << ") ===" << "\n";
  uint64_t total_bytes = 0;
  for (const auto& tensor : read.tensorInfo_) {
    std::cout << tensor.name << " shape=[";
    for (size_t i = 0; i < tensor.shape.size(); i++) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << tensor.shape[i];
    }
    std::cout << "] type=" << GgmlTypeName(tensor.type) << " offset=" << tensor.offset;
    uint64_t size = 0;
    if (TensorDataSize(tensor, size)) {
      std::cout << " bytes=" << size;
      AddCheck(total_bytes, size, total_bytes);
    } else {
      std::cout << " bytes=invalid";
    }
    std::cout << "\n";
  }
  std::cout << "total tensor bytes: " << total_bytes << "\n";
  return true;
}
 bool GetGuffMetadaInt(const std::string& path, const std::string& key, int64_t& out) {
   std::ifstream f;
   GgufValueType type;
   if (!FindMatedaKey(f, path,  key, type)) {
      return false;
   }
        /* kGgufTypeUint8 = 0,
        kGgufTypeInt8 = 1,
        kGgufTypeUint16 = 2,
        kGgufTypeInt16 = 3,
        kGgufTypeUint32 = 4,
        kGgufTypeInt32 = 5,*/
   switch(type) {
      case GgufValueType::kGgufTypeUint8 : {
        uint8_t v =0;
        if (!ReadDl(f, v)) {
          return false;
        }
        out = static_cast<int64_t>(v);
        break;
      }
      case GgufValueType::kGgufTypeInt8 : {
        int8_t v = 0;
        if (!ReadDl(f, v)) {
            return false;
        }
         out = static_cast<int64_t>(v);
        break;
      }
      case GgufValueType:: kGgufTypeUint16 : {
        uint16_t v = 0;
        if (!ReadDl(f, v)) {
            return false;
        }
         out = static_cast<int64_t>(v);
        break;
      }
     case GgufValueType::kGgufTypeInt16 :{
        int16_t v = 0;
        if (!ReadDl(f, v)) {
            return false;
        }
         out = static_cast<int64_t>(v);
        break;
     }
    case GgufValueType::kGgufTypeUint32 : {
      uint32_t v = 0;
      if (!ReadU32(f, v)) {
          return false;
      }
       out = static_cast<int64_t>(v);
      break;
    }
    case  GgufValueType::kGgufTypeUint64 : {
      uint64_t v = 0;
      if (!ReadU64(f, v)) {
          return false;
      }
       out = static_cast<int64_t>(v);
      break;
    }
    case GgufValueType::kGgufTypeInt64 : {
      int64_t v = 0;
      if (!ReadDl(f, v)) {
          return false;
      }
       out = static_cast<int64_t>(v);
      break;
    }
    default:
    return false;
    break;
   }
  return true;
 }
bool GetGuffMetadaDoule(const std::string& path, const std::string& key, double& out) {
  std::ifstream f;
  GgufValueType type;
  if (!FindMatedaKey(f, path, key, type)) {
    return false;
  }
  switch(type) {
    case GgufValueType::kGgufTypeFloat32: {
      float v = 0;
      if (!ReadDl(f, v)) {
        return false;
      }
      out = static_cast<double>(v);
      break;
    }
    case GgufValueType::kGgufTypeFloat64: {
      double v = 0;
      if (!ReadDl(f, v)) {
        return false;
      }
      out = v;
      break;
    }
    default:
      return false;
  }
  return true;
}

bool GetGuffMetadaString(const std::string& path, const std::string& key, std::string& out) {
  std::ifstream f;
  GgufValueType type;
  if (!FindMatedaKey(f, path, key, type)) {
    return false;
  }
  switch(type) {
    case GgufValueType::kGgufTypeString: {
      if (!ReadString(f, out)) {
        return false;
      }
      break;
    }
    default:
      return false;
  }
  return true;
}
  bool ReadTensorRaw(const std::string& path, uint64_t offset, std::vector<uint8_t>& out, const  GgufTensorInfo& info) {
      std::ifstream f;
      f.open(path, std::ios::binary);
      if (!f.is_open()) {
        return false;
      }
      uint64_t tensor_size = 0;
      if (!TensorDataSize(info, tensor_size)){
          return false;
      }
      uint64_t tensor_start = 0;
      if (! AddCheck(offset, info.offset, tensor_start)) {
          return false;
      }
      if (tensor_start == 0) {
          return false;
      }
      if (tensor_start > static_cast<uint64_t>(std::numeric_limits<std::streamoff> :: max())) {
        return false;
      }
      f.seekg(tensor_start, std::ios::beg);
      if (! f.good()) {
        return false;
      }
      out.resize(tensor_size);
      f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamoff>(tensor_size));
      if (static_cast<uint64_t>(f.gcount()) != tensor_size) {
        out.clear();
        return false;
      }
      return true;
  }
bool GetGuffMetadaStringArry(const std::string& path, const std::string& key, std::vector<std::string>& out) {
  std::ifstream f;
  GgufValueType type;
  if (!FindMatedaKey(f, path, key, type)) {
      return false;
  }
  if (type != GgufValueType::kGgufTypeArray) {
      return false;
  }
  uint32_t ele_type_raw = 0;
  uint64_t len = 0;
  if (!ReadU32(f, ele_type_raw) || !ReadU64(f, len)) {
    return false;
  }
  GgufValueType ele_type = static_cast<GgufValueType>(ele_type_raw);
  if (len == 0 || len > kMaxArrayElements || ele_type != GgufValueType::kGgufTypeString) {
      return false;
  }
  out.clear();
  out.reserve(len);
  for (uint64_t i = 0; i < len; i++) {
      std::string s;
      if (!ReadString(f, s)) {
          return false;
      }
      out.push_back(s);
  }
  return true;
}
bool GetGuffMetadaIntArry(const std::string& path, const std::string& key, std::vector<int>& out) {
   std::ifstream f;
  GgufValueType type;
  if (!FindMatedaKey(f, path, key, type)) {
      return false;
  }
  if (type != GgufValueType::kGgufTypeArray) {
      return false;
  }
  uint32_t ele_type_raw = 0;
  uint64_t len = 0;
  if (!ReadU32(f, ele_type_raw) || !ReadU64(f, len)) {
    return false;
  }
  GgufValueType ele_type = static_cast<GgufValueType>(ele_type_raw);
  if (len == 0 || len > kMaxArrayElements || ele_type != GgufValueType::kGgufTypeInt32) {
      return false;
  }
  out.clear();
  out.reserve(len);
  for (uint64_t i = 0; i < len; i++) {
      int32_t t;
      if (!ReadDl(f, t)) {
          return false;
      }
      out.push_back(t);
  }
  return true;
}
}// namespace yan_lamma