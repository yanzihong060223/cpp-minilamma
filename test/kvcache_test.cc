#include "include/kvcache.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace yan_lamma {
namespace {

TEST(KvCacheTest, ConstructionCreatesFourDimensionalZeroStorage) {
    KvCache cache(2, 3, 2, 2);

    EXPECT_EQ(cache.key_.shape, (std::vector<int>{2, 3, 2, 2}));
    EXPECT_EQ(cache.value_.shape, (std::vector<int>{2, 3, 2, 2}));
    EXPECT_EQ(cache.key_.GetDataSize(), 24u);
    EXPECT_EQ(cache.value_.GetDataSize(), 24u);
    EXPECT_EQ(cache.key_.data, std::vector<float>(24, 0.0f));
    EXPECT_EQ(cache.value_.data, std::vector<float>(24, 0.0f));
}

TEST(KvCacheTest, WriteAndReadPreserveEveryLayerPositionAndHead) {
    KvCache cache(2, 3, 2, 2);

    for (int layer = 0; layer < 2; ++layer) {
        for (int pos = 0; pos < 3; ++pos) {
            Tensor k({2, 2});
            Tensor v({2, 2});
            for (int head = 0; head < 2; ++head) {
                for (int dim = 0; dim < 2; ++dim) {
                    const float expected =
                        static_cast<float>(100 * layer + 10 * pos + 2 * head + dim + 1);
                    k.At2(head, dim) = expected;
                    v.At2(head, dim) = -expected;
                }
            }
            cache.Write(layer, pos, k, v);
        }
    }

    for (int layer = 0; layer < 2; ++layer) {
        for (int pos = 0; pos < 3; ++pos) {
            for (int head = 0; head < 2; ++head) {
                const float* key = cache.KeyPtr(layer, pos, head);
                const float* value = cache.ValPtr(layer, pos, head);
                ASSERT_NE(key, nullptr);
                ASSERT_NE(value, nullptr);
                for (int dim = 0; dim < 2; ++dim) {
                    const float expected =
                        static_cast<float>(100 * layer + 10 * pos + 2 * head + dim + 1);
                    EXPECT_FLOAT_EQ(key[dim], expected);
                    EXPECT_FLOAT_EQ(value[dim], -expected);
                    EXPECT_FLOAT_EQ(cache.key_.At4(layer, pos, head, dim), expected);
                    EXPECT_FLOAT_EQ(cache.value_.At4(layer, pos, head, dim), -expected);
                }
            }
        }
    }
}

TEST(KvCacheTest, OverwritingOneSlotLeavesOtherLayersAndPositionsUnchanged) {
    KvCache cache(2, 3, 2, 2);
    Tensor initial_key({2, 2}, 1.0f);
    Tensor initial_value({2, 2}, 2.0f);
    cache.Write(0, 1, initial_key, initial_value);
    cache.Write(1, 1, initial_key, initial_value);

    Tensor replacement_key({2, 2});
    Tensor replacement_value({2, 2});
    replacement_key.data = {3.0f, 4.0f, 5.0f, 6.0f};
    replacement_value.data = {7.0f, 8.0f, 9.0f, 10.0f};
    cache.Write(1, 1, replacement_key, replacement_value);

    for (int layer = 0; layer < 2; ++layer) {
        for (int pos = 0; pos < 3; ++pos) {
            for (int head = 0; head < 2; ++head) {
                for (int dim = 0; dim < 2; ++dim) {
                    const float expected_key =
                        pos != 1 ? 0.0f
                                 : layer == 0 ? 1.0f
                                              : replacement_key.At2(head, dim);
                    const float expected_value =
                        pos != 1 ? 0.0f
                                 : layer == 0 ? 2.0f
                                              : replacement_value.At2(head, dim);
                    EXPECT_FLOAT_EQ(cache.key_.At4(layer, pos, head, dim), expected_key);
                    EXPECT_FLOAT_EQ(cache.value_.At4(layer, pos, head, dim), expected_value);
                }
            }
        }
    }
}

TEST(KvCacheTest, WriteRejectsNegativeAndUpperBoundLayerAndPosition) {
    KvCache cache(2, 3, 2, 2);
    Tensor k({2, 2}, 1.0f);
    Tensor v({2, 2}, 2.0f);

    EXPECT_THROW(cache.Write(-1, 0, k, v), std::out_of_range);
    EXPECT_THROW(cache.Write(2, 0, k, v), std::out_of_range);
    EXPECT_THROW(cache.Write(0, -1, k, v), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 3, k, v), std::out_of_range);
    EXPECT_EQ(cache.key_.data, std::vector<float>(24, 0.0f));
    EXPECT_EQ(cache.value_.data, std::vector<float>(24, 0.0f));
}

TEST(KvCacheTest, KeyPtrRejectsNegativeAndUpperBoundIndicesOnEveryAxis) {
    KvCache cache(2, 3, 2, 2);

    EXPECT_THROW(cache.KeyPtr(-1, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(2, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, -1, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, 3, 0), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, 0, -1), std::out_of_range);
    EXPECT_THROW(cache.KeyPtr(0, 0, 2), std::out_of_range);
}

TEST(KvCacheTest, ValPtrRejectsNegativeAndUpperBoundIndicesOnEveryAxis) {
    KvCache cache(2, 3, 2, 2);

    EXPECT_THROW(cache.ValPtr(-1, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.ValPtr(2, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.ValPtr(0, -1, 0), std::out_of_range);
    EXPECT_THROW(cache.ValPtr(0, 3, 0), std::out_of_range);
    EXPECT_THROW(cache.ValPtr(0, 0, -1), std::out_of_range);
    EXPECT_THROW(cache.ValPtr(0, 0, 2), std::out_of_range);
}

TEST(KvCacheTest, DefaultCacheRejectsReadsAndWrites) {
    KvCache cache;
    Tensor k({2, 2}, 1.0f);
    Tensor v({2, 2}, 2.0f);

    EXPECT_THROW(cache.KeyPtr(0, 0, 0), std::runtime_error);
    EXPECT_THROW(cache.ValPtr(0, 0, 0), std::runtime_error);
    EXPECT_THROW(cache.Write(0, 0, k, v), std::runtime_error);
    EXPECT_TRUE(cache.key_.data.empty());
    EXPECT_TRUE(cache.value_.data.empty());
}

TEST(KvCacheTest, ReadsRejectStorageThatIsNotFourDimensional) {
    KvCache cache(2, 3, 2, 2);
    cache.key_ = Tensor({2, 3, 2});
    cache.value_ = Tensor({2, 3});

    EXPECT_THROW(cache.KeyPtr(0, 0, 0), std::runtime_error);
    EXPECT_THROW(cache.ValPtr(0, 0, 0), std::runtime_error);
}

TEST(KvCacheTest, WriteRejectsStorageThatIsNotFourDimensional) {
    Tensor k({2, 2}, 1.0f);
    Tensor v({2, 2}, 2.0f);
    KvCache invalid_key(2, 3, 2, 2);
    invalid_key.key_ = Tensor({2, 3, 2});
    const std::vector<float> original_values = invalid_key.value_.data;

    EXPECT_THROW(invalid_key.Write(0, 0, k, v), std::runtime_error);
    EXPECT_EQ(invalid_key.value_.data, original_values);

    KvCache invalid_value(2, 3, 2, 2);
    invalid_value.value_ = Tensor({2, 3, 2});
    const std::vector<float> original_keys = invalid_value.key_.data;

    EXPECT_THROW(invalid_value.Write(0, 0, k, v), std::runtime_error);
    EXPECT_EQ(invalid_value.key_.data, original_keys);
}

TEST(KvCacheTest, WriteRejectsInputsThatAreNotTwoDimensional) {
    KvCache cache(2, 3, 2, 2);
    Tensor valid({2, 2}, 1.0f);
    Tensor rank_one({4}, 2.0f);
    Tensor rank_three({1, 2, 2}, 3.0f);

    EXPECT_THROW(cache.Write(0, 0, rank_one, valid), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 0, valid, rank_three), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 0, rank_one, rank_one), std::out_of_range);
    EXPECT_EQ(cache.key_.data, std::vector<float>(24, 0.0f));
    EXPECT_EQ(cache.value_.data, std::vector<float>(24, 0.0f));
}

TEST(KvCacheTest, WriteRejectsInputShapeMismatch) {
    KvCache cache(2, 3, 2, 2);
    Tensor valid({2, 2}, 1.0f);
    Tensor wrong_heads({1, 2}, 2.0f);
    Tensor wrong_dimension({2, 3}, 3.0f);

    EXPECT_THROW(cache.Write(0, 0, valid, wrong_heads), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 0, wrong_heads, wrong_heads), std::out_of_range);
    EXPECT_THROW(cache.Write(0, 0, wrong_dimension, wrong_dimension),
                 std::out_of_range);
    EXPECT_EQ(cache.key_.data, std::vector<float>(24, 0.0f));
    EXPECT_EQ(cache.value_.data, std::vector<float>(24, 0.0f));
}

TEST(KvCacheTest, WriteRejectsDifferentCacheShapesBeforeChangingEitherStorage) {
    KvCache cache(2, 3, 2, 2);
    cache.value_ = Tensor({2, 2, 2, 2}, -5.0f);
    const std::vector<float> original_keys = cache.key_.data;
    const std::vector<float> original_values = cache.value_.data;
    Tensor k({2, 2}, 1.0f);
    Tensor v({2, 2}, 2.0f);

    EXPECT_THROW(cache.Write(1, 1, k, v), std::runtime_error);
    EXPECT_EQ(cache.key_.data, original_keys);
    EXPECT_EQ(cache.value_.data, original_values);
}

TEST(KvCacheTest, ValPtrUsesValuesOwnShapeForBoundsAndOffset) {
    KvCache cache(1, 1, 1, 1);
    cache.value_ = Tensor({2, 3, 2, 2});
    for (size_t i = 0; i < cache.value_.GetDataSize(); ++i) {
        cache.value_.data[i] = static_cast<float>(i + 1);
    }

    const float* value = cache.ValPtr(1, 2, 1);
    ASSERT_EQ(value, cache.value_.data.data() + 22);
    EXPECT_FLOAT_EQ(value[0], 23.0f);
    EXPECT_FLOAT_EQ(value[1], 24.0f);
    EXPECT_THROW(cache.ValPtr(2, 0, 0), std::out_of_range);
    EXPECT_THROW(cache.ValPtr(0, 3, 0), std::out_of_range);
    EXPECT_THROW(cache.ValPtr(0, 0, 2), std::out_of_range);
    EXPECT_FLOAT_EQ(cache.key_.At4(0, 0, 0, 0), 0.0f);
}

}  // namespace
}  // namespace yan_lamma
