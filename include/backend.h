#pragma once
#include "include/cuda_runtime.h"

#include <string>
namespace yan_lamma {
enum class BackendKind {
    Kcpu = 0,
    Kgpu
};
struct BackendConfig {
// 针对多卡
    int device_id = 0;
    BackendKind kind = BackendKind::Kcpu;
    bool device_set = false;

};
// 校验失败时抛出异常。
void VailedBackend (const BackendConfig& config);
bool ParseBackendKind(const std::string& text, BackendKind& kind);
std::string BackendKindName(BackendKind kind);
//是否走cuda链路
bool CudaBackendBuilt();
// 检查设备编号是否合法；runtime 查询失败时返回 false。
bool CudaBackendAvailable(int device_id);
// 用于展示：成功返回设备摘要，查询失败返回异常消息。
std::string CudaDeviceSummary(int device_id);
std::string BackendActivePath(const BackendConfig& config);
}
