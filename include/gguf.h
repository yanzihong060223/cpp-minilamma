#pragma once
#include <string>
#include <vector>
#include <cstdint>
namespace yan_lamma {
    //权重存储类型
    enum class Ggmltype {
        GgmltypeF32 = 0,
        GgmltypeF16 = 1,
        GgmltypeQ40 = 2,
        GgmltypeQ41 = 3,
        GgmlytypeQ80 = 8
    };
    // 元数据数据类型
    enum class GgufValueType {
        kGgufTypeUint8 = 0,
        kGgufTypeInt8 = 1,
        kGgufTypeUint16 = 2,
        kGgufTypeInt16 = 3,
        kGgufTypeUint32 = 4,
        kGgufTypeInt32 = 5,
        kGgufTypeFloat32 = 6,
        kGgufTypeBool = 7,
        kGgufTypeString = 8,
        kGgufTypeArray = 9,
        kGgufTypeUint64 = 10,
        kGgufTypeInt64 = 11,
        kGgufTypeFloat64 = 12,
    };
    struct GgufTensorInfo {
        std::string name;
        std::vector<uint64_t> shape;
        Ggmltype type;
        size_t offset = 0;
    };
    struct GgufMetadaKv {
        std::string name;
        GgufValueType type;
        std::string value_str;
    };
    class GgufRead {
    public:
        size_t n_tensors_ = 0;
        size_t n_metadata_ = 0;
        uint32_t version_ = 0;
        std::vector<GgufTensorInfo> tensorInfo_;
        std::vector<GgufMetadaKv> metadakv_;
        bool load(const std::string& file);
        std::string load_error_;
        uint64_t data_offset_ = 0;
    };
    bool ReadTensorRaw(const std::string& path, uint64_t offset, std::vector<uint8_t>& out, const GgufTensorInfo& info);
    bool Inspect(const GgufRead& read);
    // 将文件数据读到内存 
    //计算权重占用字节数 
    bool GetGuffDataSize(const GgufTensorInfo& tensor, uint64_t& out);
    //读取元数据字段
    bool GetGuffMetadaInt(const std::string& path, const std::string& key, int64_t& out);
    bool GetGuffMetadaDoule(const std::string& path, const std::string& key, double& out);
    bool GetGuffMetadaString(const std::string& path, const std::string& key, std::string& out);
    bool GetGuffMetadaStringArry(const std::string& path, const std::string& key, std::vector<std::string>& out);
    bool GetGuffMetadaIntArry(const std::string& path, const std::string& key, std::vector<int>& out);
}//namespace lamma