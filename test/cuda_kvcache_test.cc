#include "include/cuda_kvcache.h"

#include <gtest/gtest.h>

#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yan_lamma {
namespace {

template <typename Operation>
void ExpectCacheError(Operation operation,
                      std::initializer_list<const char*> fragments) {
    try {
        operation();
        ADD_FAILURE() << "Expected std::runtime_error";
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        for (const char* fragment : fragments) {
            EXPECT_NE(message.find(fragment), std::string::npos) << message;
        }
    }
}

void ExpectEmptyCache(const CudaCache& cache) {
    EXPECT_EQ(cache.n_layers(), 0);
    EXPECT_EQ(cache.max_seq_len(), 0);
    EXPECT_EQ(cache.n_kv_heads(), 0);
    EXPECT_EQ(cache.head_dim(), 0);
    EXPECT_EQ(cache.device_id(), 0);
    EXPECT_EQ(cache.Bytes_(), 0u);
    ExpectCacheError([&] { (void)cache.KData(); }, {"KData", "empty"});
    ExpectCacheError([&] { (void)cache.vData(); }, {"vData", "empty"});
}

Tensor Values(int layer, int pos, bool value) {
    Tensor result({2, 4});
    for (int head = 0; head < 2; ++head) {
        for (int dim = 0; dim < 4; ++dim) {
            const float number = 1 + layer * 100 + pos * 20 + head * 5 + dim;
            result.data[head * 4 + dim] = value ? -number - 0.5f : number;
        }
    }
    return result;
}

void ExpectTensor(const Tensor& actual, const Tensor& expected) {
    EXPECT_EQ(actual.shape, expected.shape);
    EXPECT_EQ(actual.data, expected.data);
}

void ExpectSlot(const CudaCache& cache, int layer, int pos,
                const Tensor& key, const Tensor& value) {
    ExpectTensor(cache.ReadKey(layer, pos), key);
    ExpectTensor(cache.ReadVal(layer, pos), value);
    for (int head = 0; head < cache.n_kv_heads(); ++head) {
        Tensor expected_key({cache.head_dim()});
        Tensor expected_value({cache.head_dim()});
        for (int dim = 0; dim < cache.head_dim(); ++dim) {
            expected_key.data[dim] = key.data[head * cache.head_dim() + dim];
            expected_value.data[dim] = value.data[head * cache.head_dim() + dim];
        }
        ExpectTensor(cache.ReadHeadKey(layer, pos, head), expected_key);
        ExpectTensor(cache.ReadHeadVal(layer, pos, head), expected_value);
    }
}

void ExpectZeroCache(const CudaCache& cache) {
    const Tensor zero({cache.n_kv_heads(), cache.head_dim()}, 0.0f);
    for (int layer = 0; layer < cache.n_layers(); ++layer) {
        for (int pos = 0; pos < cache.max_seq_len(); ++pos) {
            ExpectSlot(cache, layer, pos, zero, zero);
        }
    }
}

TEST(CudaCacheTest, DefaultStateAndRepeatedResetAreEmpty) {
    CudaCache cache;
    ExpectEmptyCache(cache);
    EXPECT_NO_THROW(cache.Clear());
    EXPECT_NO_THROW(cache.Reset());
    EXPECT_NO_THROW(cache.Reset());
    ExpectEmptyCache(cache);
    ExpectCacheError([&] { (void)cache.SlotOffsetBytes(0, 0); },
                     {"SlotOffsetBytes", "initialized"});
    ExpectCacheError([&] { (void)cache.SlotOffsetBytes(0, 0, 0); },
                     {"SlotOffsetBytes", "initialized"});
}

TEST(CudaCacheTest, BuildPredicateMatchesRuntime) {
    EXPECT_EQ(CudaCacheBuild(), CudaRuntimeBuilt());
}

TEST(CudaCacheTest, CpuBuildReportsUnavailableBackendForInitializationAndTransfers) {
    if (CudaRuntimeBuilt()) {
        GTEST_SKIP() << "This test checks the build without CUDA support";
    }
    CudaCache cache;
    const Tensor host = Values(0, 0, false);
    const CudaTensor device;
    const auto unavailable = [&](auto operation, const char* caller) {
        ExpectCacheError(operation, {caller, "not built", "MINI_LLAMA_CUDA=ON"});
    };
    unavailable([] { CudaCache allocated(2, 3, 2, 4); }, "Reset");
    unavailable([&] { cache.Reset(2, 3, 2, 4); }, "Reset");
    unavailable([&] { cache.WriteHost(0, 0, host, host); }, "WriteHost");
    unavailable([&] { cache.WriteDevice(0, 0, device, device); }, "WriteDevice");
    unavailable([&] { (void)cache.ReadKey(0, 0); }, "ReadKey");
    unavailable([&] { (void)cache.ReadVal(0, 0); }, "ReadVal");
    unavailable([&] { (void)cache.ReadHeadKey(0, 0, 0); }, "ReadHeadKey");
    unavailable([&] { (void)cache.ReadHeadVal(0, 0, 0); }, "ReadHeadVal");
    ExpectEmptyCache(cache);
}

class CudaCacheGpuTest : public ::testing::Test {
 protected:
    void SetUp() override {
        try {
            (void)GetCudaInfo(0);
        } catch (const std::exception& error) {
            GTEST_SKIP() << "CUDA device 0 is unavailable: " << error.what();
        }
    }
};

TEST_F(CudaCacheGpuTest, InitializationAndOffsetsCoverAllLayersPositionsAndHeads) {
    const CudaCache cache(2, 3, 2, 4);
    EXPECT_EQ(cache.n_layers(), 2);
    EXPECT_EQ(cache.max_seq_len(), 3);
    EXPECT_EQ(cache.n_kv_heads(), 2);
    EXPECT_EQ(cache.head_dim(), 4);
    EXPECT_EQ(cache.device_id(), 0);
    EXPECT_NE(cache.KData(), nullptr);
    EXPECT_NE(cache.vData(), nullptr);
    EXPECT_NE(cache.KData(), cache.vData());
    EXPECT_EQ(cache.Bytes_(), 2u * 2 * 3 * 2 * 4 * sizeof(float));
    EXPECT_EQ(cache.SlotOffsetBytes(0, 0), 0u);
    EXPECT_EQ(cache.SlotOffsetBytes(0, 1), 8u * sizeof(float));
    EXPECT_EQ(cache.SlotOffsetBytes(1, 0), 24u * sizeof(float));
    EXPECT_EQ(cache.SlotOffsetBytes(1, 2), 40u * sizeof(float));
    EXPECT_EQ(cache.SlotOffsetBytes(1, 2, 0), 40u * sizeof(float));
    EXPECT_EQ(cache.SlotOffsetBytes(1, 2, 1), 44u * sizeof(float));
    ExpectZeroCache(cache);
}

TEST_F(CudaCacheGpuTest, HostAndDeviceWritesRoundTripWithoutChangingOtherSlots) {
    for (bool device_write : {false, true}) {
        SCOPED_TRACE(device_write);
        CudaCache cache(2, 3, 2, 4);
        for (int slot = 0; slot < 6; ++slot) {
            const int layer = slot / 3;
            const int pos = slot % 3;
            const Tensor key = Values(layer, pos, false);
            const Tensor value = Values(layer, pos, true);
            if (device_write) {
                const CudaTensor device_key = TensorHostToDevice(key);
                const CudaTensor device_value = TensorHostToDevice(value);
                cache.WriteDevice(layer, pos, device_key, device_value);
            } else {
                cache.WriteHost(layer, pos, key, value);
            }
            for (int check = 0; check < 6; ++check) {
                SCOPED_TRACE(check);
                const Tensor zero({2, 4}, 0.0f);
                ExpectSlot(cache, check / 3, check % 3,
                           check <= slot ? Values(check / 3, check % 3, false) : zero,
                           check <= slot ? Values(check / 3, check % 3, true) : zero);
            }
        }
        // Replacing an existing slot must also preserve its neighbors.
        cache.WriteHost(0, 1, Values(1, 2, true), Values(1, 2, false));
        for (int slot = 0; slot < 6; ++slot) {
            ExpectSlot(cache, slot / 3, slot % 3,
                       slot == 1 ? Values(1, 2, true) : Values(slot / 3, slot % 3, false),
                       slot == 1 ? Values(1, 2, false) : Values(slot / 3, slot % 3, true));
        }
    }
}

TEST_F(CudaCacheGpuTest, ClearAndResetReleaseOrZeroTheExpectedState) {
    CudaCache cache(2, 3, 2, 4);
    cache.WriteHost(1, 2, Values(1, 2, false), Values(1, 2, true));
    const void* key_pointer = cache.KData();
    const void* value_pointer = cache.vData();
    cache.Clear();
    EXPECT_EQ(cache.KData(), key_pointer);
    EXPECT_EQ(cache.vData(), value_pointer);
    EXPECT_EQ(cache.n_layers(), 2);
    EXPECT_EQ(cache.max_seq_len(), 3);
    ExpectZeroCache(cache);
    cache.WriteHost(0, 0, Values(0, 0, false), Values(0, 0, true));
    cache.Reset(1, 2, 3, 5);
    EXPECT_EQ(cache.n_layers(), 1);
    EXPECT_EQ(cache.max_seq_len(), 2);
    EXPECT_EQ(cache.n_kv_heads(), 3);
    EXPECT_EQ(cache.head_dim(), 5);
    EXPECT_EQ(cache.Bytes_(), 2u * 1 * 2 * 3 * 5 * sizeof(float));
    ExpectZeroCache(cache);
    cache.Reset();
    cache.Reset();
    ExpectEmptyCache(cache);
    EXPECT_NO_THROW(cache.Clear());
    cache.Reset(2, 3, 2, 4);
    ExpectZeroCache(cache);
}

TEST_F(CudaCacheGpuTest, RejectsInvalidIndicesForAllAccessRoutes) {
    const CudaCache cache(2, 3, 2, 4);
    const Tensor key = Values(0, 0, false);
    const Tensor value = Values(0, 0, true);
    const CudaTensor device_key = TensorHostToDevice(key);
    const CudaTensor device_value = TensorHostToDevice(value);
    for (const auto& indices : std::vector<std::pair<int, int>>{
             {-1, 0}, {2, 0}, {0, -1}, {0, 3}}) {
        const int layer = indices.first;
        const int pos = indices.second;
        SCOPED_TRACE(::testing::PrintToString(indices));
        ExpectCacheError([&] { (void)cache.SlotOffsetBytes(layer, pos); }, {"out of range"});
        ExpectCacheError([&] { (void)cache.SlotOffsetBytes(layer, pos, 0); }, {"out of range"});
        ExpectCacheError([&] { cache.WriteHost(layer, pos, key, value); }, {"WriteHost", "out of range"});
        ExpectCacheError([&] { cache.WriteDevice(layer, pos, device_key, device_value); }, {"WriteDevice", "out of range"});
        ExpectCacheError([&] { (void)cache.ReadKey(layer, pos); }, {"ReadKey", "out of range"});
        ExpectCacheError([&] { (void)cache.ReadVal(layer, pos); }, {"ReadVal", "out of range"});
        ExpectCacheError([&] { (void)cache.ReadHeadKey(layer, pos, 0); }, {"ReadHeadKey", "out of range"});
        ExpectCacheError([&] { (void)cache.ReadHeadVal(layer, pos, 0); }, {"ReadHeadVal", "out of range"});
    }
    for (int head : {-1, 2}) {
        ExpectCacheError([&] { (void)cache.SlotOffsetBytes(0, 0, head); }, {"head", "out of range"});
        ExpectCacheError([&] { (void)cache.ReadHeadKey(0, 0, head); }, {"ReadHeadKey", "out of range"});
        ExpectCacheError([&] { (void)cache.ReadHeadVal(0, 0, head); }, {"ReadHeadVal", "out of range"});
    }
    ExpectZeroCache(cache);
}

TEST_F(CudaCacheGpuTest, RejectsMalformedInputsWithoutChangingStoredValues) {
    CudaCache cache(2, 3, 2, 4);
    const Tensor key = Values(0, 0, false);
    const Tensor value = Values(0, 0, true);
    cache.WriteHost(0, 0, key, value);
    const CudaTensor device_key = TensorHostToDevice(key);
    const CudaTensor device_value = TensorHostToDevice(value);
    for (const auto& shape : std::vector<std::vector<int>>{{8}, {1, 2, 4}, {1, 4}, {2, 5}, {4, 2}}) {
        SCOPED_TRACE(::testing::PrintToString(shape));
        const Tensor wrong(shape, 99.0f);
        const CudaTensor device_wrong = TensorHostToDevice(wrong);
        ExpectCacheError([&] { cache.WriteHost(0, 0, wrong, value); }, {"WriteHost k", "shape"});
        ExpectCacheError([&] { cache.WriteHost(0, 0, key, wrong); }, {"WriteHost v", "shape"});
        ExpectCacheError([&] { cache.WriteDevice(0, 0, device_wrong, device_value); }, {"WriteDevice k", "shape"});
        ExpectCacheError([&] { cache.WriteDevice(0, 0, device_key, device_wrong); }, {"WriteDevice v", "shape"});
        ExpectSlot(cache, 0, 0, key, value);
    }
    for (int size : {0, 7, 9}) {
        Tensor wrong({2, 4});
        wrong.data.resize(size);
        ExpectCacheError([&] { cache.WriteHost(0, 0, wrong, value); }, {"WriteHost", "expected 8", "got K"});
        ExpectCacheError([&] { cache.WriteHost(0, 0, key, wrong); }, {"WriteHost", "expected 8", "and V"});
        ExpectSlot(cache, 0, 0, key, value);
    }
}

TEST_F(CudaCacheGpuTest, InvalidResetParametersPreserveCacheBeforeReallocation) {
    CudaCache cache(2, 3, 2, 4);
    const Tensor key = Values(0, 0, false);
    const Tensor value = Values(0, 0, true);
    cache.WriteHost(0, 0, key, value);
    const void* pointer = cache.KData();
    for (int invalid : {0, -1}) {
        ExpectCacheError([&] { cache.Reset(invalid, 3, 2, 4); }, {"Reset n_layers", "positive"});
        ExpectCacheError([&] { cache.Reset(2, invalid, 2, 4); }, {"Reset max_seq_len", "positive"});
        ExpectCacheError([&] { cache.Reset(2, 3, invalid, 4); }, {"Reset n_kv_heads", "positive"});
        ExpectCacheError([&] { cache.Reset(2, 3, 2, invalid); }, {"Reset head_dim", "positive"});
    }
    ExpectCacheError([&] { cache.Reset(2, 3, 2, 4, -1); }, {"device_id", "non-negative"});
    const int large = std::numeric_limits<int>::max();
    ExpectCacheError([&] { cache.Reset(large, large, large, large); }, {"Reset", "overflow"});
    ExpectCacheError([&] { cache.Reset(large, large, 2, 1); }, {"Reset bytes", "overflow"});
    if (sizeof(size_t) == 8) {
        // One buffer fits size_t, but the combined K/V byte count does not.
        ExpectCacheError([&] { cache.Reset(large, large, 1, 1); }, {"total K/V bytes", "overflow"});
    }
    EXPECT_EQ(cache.KData(), pointer);
    EXPECT_EQ(cache.n_layers(), 2);
    EXPECT_EQ(cache.max_seq_len(), 3);
    ExpectSlot(cache, 0, 0, key, value);
}

TEST_F(CudaCacheGpuTest, MovesPreserveDataAndSourceCanBeReused) {
    CudaCache source(2, 3, 2, 4);
    const Tensor key = Values(1, 2, false);
    const Tensor value = Values(1, 2, true);
    source.WriteHost(1, 2, key, value);
    const void* original_pointer = source.KData();
    const auto reject_access = [&](const CudaCache& moved) {
        const CudaTensor empty;
        ExpectCacheError([&] { (void)moved.KData(); }, {"empty"});
        ExpectCacheError([&] { (void)moved.vData(); }, {"empty"});
        ExpectCacheError([&] { (void)moved.SlotOffsetBytes(1, 2); }, {"initialized"});
        ExpectCacheError([&] { (void)moved.SlotOffsetBytes(1, 2, 1); }, {"initialized"});
        ExpectCacheError([&] { moved.WriteHost(1, 2, key, value); }, {"WriteHost", "initialized"});
        ExpectCacheError([&] { moved.WriteDevice(1, 2, empty, empty); }, {"WriteDevice", "initialized"});
        ExpectCacheError([&] { (void)moved.ReadKey(1, 2); }, {"ReadKey", "initialized"});
        ExpectCacheError([&] { (void)moved.ReadVal(1, 2); }, {"ReadVal", "initialized"});
        ExpectCacheError([&] { (void)moved.ReadHeadKey(1, 2, 1); }, {"ReadHeadKey", "initialized"});
        ExpectCacheError([&] { (void)moved.ReadHeadVal(1, 2, 1); }, {"ReadHeadVal", "initialized"});
        EXPECT_EQ(moved.Bytes_(), 0u);
    };
    CudaCache destination(std::move(source));
    EXPECT_EQ(destination.KData(), original_pointer);
    ExpectSlot(destination, 1, 2, key, value);
    reject_access(source);
    source.Reset(2, 3, 2, 4);
    ExpectZeroCache(source);
    source = std::move(destination);
    EXPECT_EQ(source.KData(), original_pointer);
    ExpectSlot(source, 1, 2, key, value);
    reject_access(destination);
    destination.Reset(1, 1, 1, 1);
    ExpectZeroCache(destination);
}

TEST_F(CudaCacheGpuTest, MultipleDevicesRejectForeignInputsAndReadsSelectCacheDevice) {
    if (DeviceCount() < 2) {
        GTEST_SKIP() << "This test requires two CUDA devices";
    }
    try {
        (void)GetCudaInfo(1);
    } catch (const std::exception& error) {
        GTEST_SKIP() << "CUDA device 1 is unavailable: " << error.what();
    }
    struct RestoreDevice {
        ~RestoreDevice() {
            try { SetCudaService(0); } catch (...) {}
        }
    } restore;
    const Tensor key = Values(0, 0, false);
    const Tensor value = Values(0, 0, true);
    const CudaTensor local_key = TensorHostToDevice(key, 0);
    const CudaTensor local_value = TensorHostToDevice(value, 0);
    const CudaTensor foreign_key = TensorHostToDevice(key, 1);
    const CudaTensor foreign_value = TensorHostToDevice(value, 1);
    CudaCache cache(2, 3, 2, 4, 0);
    cache.WriteHost(0, 0, key, value);
    ExpectCacheError([&] { cache.WriteDevice(0, 0, foreign_key, local_value); }, {"WriteDevice", "cache device 0", "K device 1"});
    ExpectCacheError([&] { cache.WriteDevice(0, 0, local_key, foreign_value); }, {"WriteDevice", "cache device 0", "V device 1"});
    ExpectCacheError([&] { cache.WriteDevice(0, 0, foreign_key, foreign_value); }, {"WriteDevice", "cache device 0", "K device 1", "V device 1"});
    SetCudaService(1);
    ExpectTensor(cache.ReadKey(0, 0), key);
    SetCudaService(1);
    ExpectTensor(cache.ReadVal(0, 0), value);
    for (int head = 0; head < 2; ++head) {
        Tensor expected_key({4});
        Tensor expected_value({4});
        for (int dim = 0; dim < 4; ++dim) {
            expected_key.data[dim] = key.data[head * 4 + dim];
            expected_value.data[dim] = value.data[head * 4 + dim];
        }
        SetCudaService(1);
        ExpectTensor(cache.ReadHeadKey(0, 0, head), expected_key);
        SetCudaService(1);
        ExpectTensor(cache.ReadHeadVal(0, 0, head), expected_value);
    }
}

}  // namespace
}  // namespace yan_lamma
