#include "thread_pool/thread_pool.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <string>

namespace yan_lamma {
namespace {

class ThreadPoolTest : public ::testing::Test {
protected:
    void SetUp() override { SetThreadNum(0); }
    void TearDown() override { SetThreadNum(0); }
};

TEST_F(ThreadPoolTest, AutomaticAndExplicitThreadCountsStayPositive) {
    EXPECT_GT(GetThreadNum(), 0);

    SetThreadNum(3);
    EXPECT_EQ(GetThreadNum(), 3);
    EXPECT_THROW(SetThreadNum(-1), std::invalid_argument);
    EXPECT_EQ(GetThreadNum(), 3);

    SetThreadNum(0);
    EXPECT_GT(GetThreadNum(), 0);
}

TEST_F(ThreadPoolTest, RejectsNegativeWorkAndSkipsEmptyWork) {
    SetThreadNum(2);
    std::atomic<int> calls{0};
    const auto callback = [&](int, int) { ++calls; };

    EXPECT_THROW(ParallelFor(-1, callback), std::invalid_argument);
    ParallelFor(0, callback);
    EXPECT_EQ(calls.load(), 0);
}

TEST_F(ThreadPoolTest, SerialCasesInvokeCallbackExactlyOnce) {
    struct SerialCase {
        int threads;
        int count;
    };
    for (const auto& test_case : {SerialCase{2, 8}, SerialCase{1, 96},
                                  SerialCase{std::numeric_limits<int>::max(), 8}}) {
        SCOPED_TRACE(test_case.threads);
        SetThreadNum(test_case.threads);
        std::atomic<int> calls{0};
        std::atomic<int> invalid_ranges{0};

        ParallelFor(test_case.count, [&](int begin, int end) {
            ++calls;
            if (begin != 0 || end != test_case.count) {
                ++invalid_ranges;
            }
        });

        EXPECT_EQ(calls.load(), 1);
        EXPECT_EQ(invalid_ranges.load(), 0);
    }
}

TEST_F(ThreadPoolTest, UnevenRangesVisitEveryIndexExactlyOnce) {
    SetThreadNum(3);
    constexpr int count = 65;
    std::array<std::atomic<int>, count> visits;
    for (auto& visit : visits) {
        visit.store(0);
    }
    std::atomic<int> calls{0};
    std::atomic<int> invalid_ranges{0};

    ParallelFor(count, [&](int begin, int end) {
        ++calls;
        if (begin < 0 || end > count || begin >= end) {
            ++invalid_ranges;
            return;
        }
        for (int i = begin; i < end; ++i) {
            ++visits[i];
        }
    });

    EXPECT_EQ(calls.load(), 3);
    EXPECT_EQ(invalid_ranges.load(), 0);
    for (int i = 0; i < count; ++i) {
        EXPECT_EQ(visits[i].load(), 1) << "index=" << i;
    }
}

TEST_F(ThreadPoolTest, JoinsAllWorkersBeforeRethrowingTaskException) {
    SetThreadNum(3);
    std::atomic<int> started{0};
    std::atomic<int> completed{0};

    try {
        ParallelFor(96, [&](int begin, int) {
            ++started;
            if (begin == 0) {
                throw std::runtime_error("worker failure");
            }
            ++completed;
        });
        FAIL() << "Expected the worker exception";
    } catch (const std::runtime_error& error) {
        EXPECT_EQ(std::string(error.what()), "worker failure");
    }

    EXPECT_EQ(started.load(), 3);
    EXPECT_EQ(completed.load(), 2);
}

TEST_F(ThreadPoolTest, PropagatesSerialTaskException) {
    SetThreadNum(2);
    std::atomic<int> calls{0};
    EXPECT_THROW(ParallelFor(8, [&](int, int) {
        ++calls;
        throw std::runtime_error("serial failure");
    }), std::runtime_error);
    EXPECT_EQ(calls.load(), 1);
}

}  // namespace
}  // namespace yan_lamma
