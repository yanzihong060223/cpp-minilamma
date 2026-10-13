#pragma once
#include "include/model.h"
#include "include/tonizer.h"
#include "include/tensor.h"

#include <string>
#include <vector>
namespace yan_lamma {
struct TensorInfo {
std::string tensor_name;
std::vector<int> shape;
std:: string dtype; //数据类型
size_t offset = 0;
size_t byte_size = 0;
};
struct TonizerInfo {
// 类型
std::string type = "ASCII";
std::string path_name;
int bos_id = 128;
int eos_id = 129;
int unk_id = 130;
};
struct ModelJson {
ModelConfig config;
TonizerInfo toninzer_json;
std::vector<TensorInfo> tensor_json;


};
ModelJson ParseModelJson(const std::string& config_path);
Model ParseModel(const std::string& config_path, const std::string& weight_path);
bool InspectModel(const std::string& config_path);

}