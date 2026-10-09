#include "include/sampler.h"

#include <gtest/gtest.h>

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using yan_lamma::Sampler;
using yan_lamma::SamplerParmas;
using yan_lamma::Tensor;

Tensor MakeLogits(std::initializer_list<float> values) {
    Tensor logits({static_cast<int>(values.size())});
    logits.data = values;
    return logits;
}

SamplerParmas Params(float temperature, int top_k) {
    SamplerParmas params;
    params.temperature = temperature;
    params.top_k = top_k;
    return params;
}

TEST(SamplerTest, GreedyChoosesLargestIncludingNegativeFractionalValues) {
    Sampler sampler(42);
    const SamplerParmas params;
    EXPECT_EQ(sampler.SamplerO(MakeLogits({1.9f, 1.8f, 1.7f}), params), 0);
    EXPECT_EQ(sampler.SamplerO(MakeLogits({-3.9f, -1.2f, -1.8f}), params), 1);
    EXPECT_EQ(sampler.SamplerO(MakeLogits({0.2f, 3.9f, 3.8f}), params), 1);
}

TEST(SamplerTest, GreedyKeepsFirstIndexWhenMaximumIsRepeated) {
    Sampler sampler(42);
    const SamplerParmas params;
    EXPECT_EQ(sampler.SamplerO(MakeLogits({1.0f, 2.5f, 2.5f}), params), 1);
    EXPECT_EQ(sampler.SamplerO(MakeLogits({-0.2f, -0.2f, -0.2f}), params), 0);
}

TEST(SamplerTest, ZeroTemperatureUsesGreedyEvenWithTopKEnabled) {
    Sampler sampler(42);
    const Tensor logits = MakeLogits({1.9f, 3.8f, 2.6f});
    for (int top_k : {0, 2, 100}) {
        SCOPED_TRACE(top_k);
        EXPECT_EQ(sampler.SamplerO(logits, Params(0.0f, top_k)), 1);
    }
}

TEST(SamplerTest, TopKOneUsesGreedyWithPositiveTemperature) {
    Sampler sampler(42);
    const Tensor logits = MakeLogits({-1.9f, -0.2f, -1.1f});
    for (int iteration = 0; iteration < 32; ++iteration) {
        EXPECT_EQ(sampler.SamplerO(logits, Params(2.0f, 1)), 1);
    }
}

TEST(SamplerTest, TemperatureBelowThresholdUsesGreedyForEqualLogits) {
    Sampler sampler(42);
    const Tensor logits = MakeLogits({0.5f, 0.5f, 0.5f});
    for (int iteration = 0; iteration < 32; ++iteration) {
        EXPECT_EQ(sampler.SamplerO(logits, Params(1e-7f, 0)), 0);
    }
}

TEST(SamplerTest, LowTemperaturePreservesFractionalMaximumWithoutOverflow) {
    const Tensor logits = MakeLogits({3.9f, 3.8f});
    for (int top_k : {0, 2}) {
        SCOPED_TRACE(top_k);
        Sampler sampler(42);
        for (int iteration = 0; iteration < 32; ++iteration) {
            EXPECT_EQ(sampler.SamplerO(logits, Params(0.001f, top_k)), 0);
        }
    }
}

TEST(SamplerTest, TopKOnlySamplesFromHighestCandidates) {
    Sampler sampler(42);
    const Tensor logits = MakeLogits({-5.0f, 1.0f, 3.0f, 2.0f, -4.0f});
    for (int iteration = 0; iteration < 64; ++iteration) {
        const int token = sampler.SamplerO(logits, Params(1.0f, 2));
        EXPECT_TRUE(token == 2 || token == 3);
    }
}

TEST(SamplerTest, TopKLargerThanVocabularyMatchesClampedTopK) {
    Sampler clamped(42);
    Sampler oversized(42);
    const Tensor logits = MakeLogits({-3.0f, -2.0f, -1.0f});
    for (int iteration = 0; iteration < 64; ++iteration) {
        EXPECT_EQ(oversized.SamplerO(logits, Params(1.0f, 100)),
                  clamped.SamplerO(logits, Params(1.0f, 3)));
    }
}

TEST(SamplerTest, FixedSeedReproducesTemperatureAndTopKSequences) {
    const Tensor logits = MakeLogits({0.1f, 1.2f, 0.7f, -0.5f});
    for (int top_k : {0, 2}) {
        SCOPED_TRACE(top_k);
        Sampler first(1234);
        Sampler second(1234);
        const SamplerParmas params = Params(0.8f, top_k);
        for (int iteration = 0; iteration < 64; ++iteration) {
            EXPECT_EQ(first.SamplerO(logits, params),
                      second.SamplerO(logits, params));
        }
    }
}

TEST(SamplerTest, ParameterConstructorUsesSeedForReproducibleSequences) {
    const Tensor logits = MakeLogits({0.1f, 1.2f, 0.7f, -0.5f});
    for (int top_k : {0, 2}) {
        SCOPED_TRACE(top_k);
        SamplerParmas params = Params(0.8f, top_k);
        params.seed = 1234;
        Sampler from_params(params);
        Sampler from_seed(params.seed);
        for (int iteration = 0; iteration < 64; ++iteration) {
            EXPECT_EQ(from_params.SamplerO(logits, params),
                      from_seed.SamplerO(logits, params));
        }
    }
}

TEST(SamplerTest, RejectsEmptyLogitsOnEveryDispatchPath) {
    Sampler sampler(42);
    const std::vector<SamplerParmas> paths{
        Params(0.0f, 0), Params(0.0f, 2), Params(1.0f, 1),
        Params(1.0f, 0), Params(1.0f, 2)};
    for (const SamplerParmas& params : paths) {
        SCOPED_TRACE(params.temperature);
        SCOPED_TRACE(params.top_k);
        EXPECT_THROW(sampler.SamplerO(Tensor{}, params), std::runtime_error);
    }
}

TEST(SamplerTest, RejectsNonfiniteLogitsOnEveryDispatchPath) {
    Sampler sampler(42);
    const std::vector<SamplerParmas> paths{
        Params(0.0f, 0), Params(0.0f, 2), Params(1.0f, 1),
        Params(1.0f, 0), Params(1.0f, 2)};
    const std::vector<float> invalid{
        std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    for (float value : invalid) {
        const Tensor logits = MakeLogits({0.0f, value});
        for (const SamplerParmas& params : paths) {
            SCOPED_TRACE(value);
            SCOPED_TRACE(params.temperature);
            SCOPED_TRACE(params.top_k);
            EXPECT_THROW(sampler.SamplerO(logits, params), std::runtime_error);
        }
    }
}

TEST(SamplerTest, RejectsInvalidTemperatureEvenWhenTopKOneWouldUseGreedy) {
    Sampler sampler(42);
    const Tensor logits = MakeLogits({0.0f, 1.0f});
    const std::vector<float> invalid{
        -1.0f, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};
    for (float temperature : invalid) {
        for (int top_k : {0, 1, 2}) {
            SCOPED_TRACE(temperature);
            SCOPED_TRACE(top_k);
            EXPECT_THROW(sampler.SamplerO(logits, Params(temperature, top_k)),
                         std::runtime_error);
        }
    }
}

TEST(SamplerTest, RejectsNegativeTopKEvenWhenTemperatureWouldUseGreedy) {
    Sampler sampler(42);
    const Tensor logits = MakeLogits({0.0f, 1.0f});
    for (float temperature : {0.0f, 1e-7f, 1.0f}) {
        SCOPED_TRACE(temperature);
        EXPECT_THROW(sampler.SamplerO(logits, Params(temperature, -1)),
                     std::runtime_error);
    }
}

}  // namespace
