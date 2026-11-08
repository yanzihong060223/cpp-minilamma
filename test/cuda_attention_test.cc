#include "include/cuda_attention.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {

TEST(CudaAttentionTest, BuildPredicateMatchesRuntime) {
    EXPECT_EQ(CudaAttentionBuilt(), CudaRuntimeBuilt());
}

TEST(CudaAttentionTest, CpuBuildReportsUnavailableBackend) {
    if (CudaAttentionBuilt()) {
        GTEST_SKIP() << "Requires a build without CUDA";
    }
    const Tensor q({2, 3});
    const CudaTensor device_q;
    const CudaCache cache;
    const auto unavailable = [](auto operation) {
        try {
            operation();
            FAIL() << "Expected an unavailable CUDA backend error";
        } catch (const std::runtime_error& error) {
            const std::string message = error.what();
            EXPECT_NE(message.find("not built"), std::string::npos);
            EXPECT_NE(message.find("MINI_LLAMA_CUDA=ON"), std::string::npos);
        }
    };
    unavailable([&] { (void)CudaAttentionDecode(q, cache, 0, 0, 2, 1, 3); });
    unavailable([&] { (void)CudaAttentionDevice(device_q, cache, 0, 0, 2, 1, 3); });
}

class CudaAttentionGpuTest : public ::testing::Test {
 protected:
    void SetUp() override {
        try {
            (void)GetCudaInfo(0);
        } catch (const std::exception& error) {
            GTEST_SKIP() << "CUDA device unavailable: " << error.what();
        }
    }
};

// Independent double-precision reference, including GQA and the causal prefix.
Tensor Reference(const Tensor& q, const std::vector<Tensor>& keys,
                 const std::vector<Tensor>& values, int pos) {
    const int heads = q.shape[0];
    const int dim = q.shape[1];
    const int group = heads / keys[0].shape[0];
    Tensor out(q.shape);
    for (int h = 0; h < heads; ++h) {
        std::vector<double> scores(pos + 1);
        for (int t = 0; t <= pos; ++t) {
            double dot = 0.0;
            for (int d = 0; d < dim; ++d) {
                dot += static_cast<double>(q.data[h * dim + d]) *
                       keys[t].data[(h / group) * dim + d];
            }
            scores[t] = dot / std::sqrt(static_cast<double>(dim));
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double denom = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denom += score;
        }
        for (int d = 0; d < dim; ++d) {
            double value = 0.0;
            for (int t = 0; t <= pos; ++t) {
                value += scores[t] / denom * values[t].data[(h / group) * dim + d];
            }
            out.data[h * dim + d] = static_cast<float>(value);
        }
    }
    return out;
}

TEST_F(CudaAttentionGpuTest, HostAndDeviceMatchReference) {
    struct Case { int heads; int kv_heads; int dim; int pos; int layer; };
    for (const Case c : {Case{2, 2, 7, 0, 0}, Case{4, 2, 9, 5, 1},
                         Case{4, 1, 133, 130, 1}}) {
        SCOPED_TRACE(::testing::Message() << "dim=" << c.dim << " pos=" << c.pos);
        CudaCache cache(2, c.pos + 2, c.kv_heads, c.dim);
        Tensor q({c.heads, c.dim});
        for (size_t i = 0; i < q.data.size(); ++i) {
            q.data[i] = std::sin(static_cast<float>(i) * 0.17f);
        }
        std::vector<Tensor> keys, values;
        // Also populate a future slot: it must not contribute to attention.
        for (int t = 0; t < cache.max_seq_len(); ++t) {
            Tensor k({c.kv_heads, c.dim}), v({c.kv_heads, c.dim});
            for (size_t i = 0; i < k.data.size(); ++i) {
                k.data[i] = std::cos(static_cast<float>(i) * 0.13f + t * 0.1f);
                v.data[i] = std::sin(static_cast<float>(i) * 0.11f - t * 0.2f);
            }
            cache.WriteHost(c.layer, t, k, v);
            keys.push_back(k);
            values.push_back(v);
        }
        const Tensor expected = Reference(q, keys, values, c.pos);
        const Tensor host = CudaAttentionDecode(q, cache, c.layer, c.pos,
                                                c.heads, c.kv_heads, c.dim);
        const CudaTensor device_q = TensorHostToDevice(q);
        const CudaTensor device = CudaAttentionDevice(device_q, cache, c.layer,
                                                      c.pos, c.heads, c.kv_heads, c.dim);
        EXPECT_EQ(device.DeviceId(), 0);
        const Tensor downloaded = device.Download();
        ASSERT_EQ(host.shape, expected.shape);
        ASSERT_EQ(downloaded.shape, expected.shape);
        for (size_t i = 0; i < expected.data.size(); ++i) {
            EXPECT_NEAR(host.data[i], expected.data[i], 2e-5f);
            EXPECT_NEAR(downloaded.data[i], expected.data[i], 2e-5f);
        }
    }
}

TEST_F(CudaAttentionGpuTest, RejectsInvalidInputsBeforeLaunchingKernels) {
    const CudaCache cache(2, 3, 2, 5);
    Tensor q({4, 5});
    const auto invalid = [&](int layer, int pos, int heads, int kv_heads, int dim,
                             int device = 0) {
        EXPECT_THROW((void)CudaAttentionDecode(q, cache, layer, pos, heads,
                                               kv_heads, dim, device), std::runtime_error);
    };
    invalid(-1, 0, 4, 2, 5);
    invalid(2, 0, 4, 2, 5);
    invalid(0, -1, 4, 2, 5);
    invalid(0, 3, 4, 2, 5);
    invalid(0, 0, 0, 2, 5);
    invalid(0, 0, 4, 0, 5);
    invalid(0, 0, 4, -2, 5);
    invalid(0, 0, 4, 2, 0);
    invalid(0, 0, 3, 2, 5);
    invalid(0, 0, 4, 1, 5);
    invalid(0, 0, 4, 2, 6);
    invalid(0, 0, 4, 2, 5, -1);
    q.shape = {20};
    invalid(0, 0, 4, 2, 5);
    q.shape = {4, 5};
    q.data.pop_back();
    invalid(0, 0, 4, 2, 5);
    const CudaTensor empty;
    EXPECT_THROW((void)CudaAttentionDevice(empty, cache, 0, 0, 4, 2, 5), std::runtime_error);
    const CudaTensor wrong_shape({20});
    EXPECT_THROW((void)CudaAttentionDevice(wrong_shape, cache, 0, 0, 4, 2, 5), std::runtime_error);
    const CudaCache empty_cache;
    EXPECT_THROW((void)CudaAttentionDecode(Tensor({4, 5}), empty_cache, 0, 0, 4, 2, 5),
                 std::runtime_error);
}

} // namespace
} // namespace yan_lamma
