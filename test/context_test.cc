#include "include/context.h"

#include <gtest/gtest.h>

namespace yan_lamma {
namespace {

TEST(ContextTest, DefaultConstructionInitializesSessionState) {
    Context context;

    EXPECT_EQ(context.model_ptr_, nullptr);
    EXPECT_EQ(context.pos, 0u);
    EXPECT_TRUE(context.token_id_.empty());
    EXPECT_EQ(context.n_decode_size, 0);
    EXPECT_EQ(context.n_profill_size, 0);
}

TEST(ContextTest, DefaultConstructionLeavesCpuCacheEmpty) {
    Context context;

    EXPECT_TRUE(context.cpu_cache_.key_.shape.empty());
    EXPECT_TRUE(context.cpu_cache_.value_.shape.empty());
    EXPECT_TRUE(context.cpu_cache_.key_.data.empty());
    EXPECT_TRUE(context.cpu_cache_.value_.data.empty());
    EXPECT_EQ(context.cpu_cache_.key_.GetDataSize(), 0u);
    EXPECT_EQ(context.cpu_cache_.value_.GetDataSize(), 0u);
}

}  // namespace
}  // namespace yan_lamma
