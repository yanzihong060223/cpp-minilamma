#include "thread_pool/thread_pool.h"

#include <exception>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace yan_lamma {
namespace {
int thread_num = 0;
}  // namespace

int GetThreadNum() {
    if (thread_num != 0) {
        return thread_num;
    }
    int thread_o_num = std::thread::hardware_concurrency();
    if (thread_o_num == 0) {
        return 4;
    }
    return thread_o_num;
}

void SetThreadNum(int n) {
    if (n < 0) {
        throw std::invalid_argument("SetThreadNum: n must be non-negative");
    }
    thread_num = n;
}

void ParallelFor(int n, const std::function<void(int begin, int end)>& f) {
    if (n < 0) {
        throw std::invalid_argument("ParallelFor: n must be non-negative");
    }
    if (n == 0) {
        return;
    }

    constexpr int min_chunk = 16;
    const int num_threads = GetThreadNum();
    if (num_threads <= 1 || n / min_chunk < num_threads) {
        f(0, n);
        return;
    }

    std::exception_ptr first_exception = nullptr;
    std::mutex exception_mutex;
    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    const int chunk = n / num_threads;
    const int remain = n % num_threads;
    int start = 0;
    try {
        for (int i = 0; i < num_threads; ++i) {
            const int count = chunk + (i < remain ? 1 : 0);
            const int end = start + count;
            workers.emplace_back([start, end, &first_exception,
                                  &exception_mutex, &f]() {
                try {
                    f(start, end);
                } catch (...) {
                    // 只在记录任务异常时加锁。
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    if (!first_exception) {
                        first_exception = std::current_exception();
                    }
                }
            });
            start = end;
        }
    } catch (...) {
        // 创建线程失败时，先等待已启动的线程，再传回创建异常。
        for (auto& t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
        throw;
    }

    for (auto& t : workers) {
        if (t.joinable()) {
            t.join();
        }
    }
    // 全部线程结束后，再向调用者传回任务异常。
    if (first_exception) {
        std::rethrow_exception(first_exception);
    }
}
}  // namespace yan_lamma
