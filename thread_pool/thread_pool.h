#pragma once
#include <thread>
#include <functional>
namespace yan_lamma {
int GetThreadNum();
// 0 表示自动选择；须在任务启动前设置，不支持并发修改线程数。
void SetThreadNum(int n);
void ParallelFor (int n, const std::function<void(int begin, int end)>& f);

}// namespace yan_lamma
