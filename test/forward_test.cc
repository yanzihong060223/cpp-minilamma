#include "include/batch.h"
#include "include/context.h"
#include "include/forward.h"
#include "include/model.h"
#include "include/ops.h"
#include "include/quantized_tensor.h"
#include "matumal/matuml_ops.h"
#include "thread_pool/thread_pool.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace yan_lamma {
namespace {

struct RestoreForwardExecution {
    const MatulMode mode = DefaultMatulMode();
    ~RestoreForwardExecution() {
        SetDefaultMatulMode(mode);
        SetThreadNum(0);
    }
};

Tensor ForwardFixtureTensor(const std::vector<int>& shape, int salt) {
    Tensor result(shape);
    for (std::size_t i = 0; i < result.data.size(); ++i) {
        const float phase = static_cast<float>(i + 1);
        result.data[i] = 0.025f * std::sin(phase * (0.07f + salt * 0.003f)) +
                         0.01f * std::cos(phase * 0.11f + salt);
    }
    return result;
}

Model ForwardFixtureModel() {
    Model model;
    auto& config = model.config;
    config.vocab_size = 131;
    config.dim = 32;
    config.hidden_dim = 86;
    config.layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.head_dim = 8;
    config.max_seq_len = 32;
    config.rms_norm_eps = 3e-4f;
    config.rope_theta = 37.0f;
    model.embedding_data_ = ForwardFixtureTensor({131, 32}, 1);
    model.final_norm = Tensor({32}, 1.0f);
    model.lm_head_ = ToQuanTensor(ForwardFixtureTensor({131, 32}, 2));
    model.n_layers_.resize(2);
    for (int index = 0; index < 2; ++index) {
        auto& layer = model.n_layers_[index];
        const int salt = index * 10;
        layer.attention_norm = Tensor({32}, 1.0f);
        layer.ffn_norm = Tensor({32}, 1.0f);
        layer.wq = ToQuanTensor(ForwardFixtureTensor({32, 32}, salt + 3));
        layer.wk = ToQuanTensor(ForwardFixtureTensor({16, 32}, salt + 4));
        layer.wv = ToQuanTensor(ForwardFixtureTensor({16, 32}, salt + 5));
        layer.wo = ToQuanTensor(ForwardFixtureTensor({32, 32}, salt + 6));
        layer.w_gate = ToQuanTensor(ForwardFixtureTensor({86, 32}, salt + 7));
        layer.w_up = ToQuanTensor(ForwardFixtureTensor({86, 32}, salt + 8));
        layer.w_down = ToQuanTensor(ForwardFixtureTensor({32, 86}, salt + 9));
    }
    model.is_loaded = true;
    return model;
}

void ExpectSameLogits(const Tensor& a, const Tensor& b) {
    ASSERT_EQ(a.shape, (std::vector<int>{131}));
    ASSERT_EQ(a.shape, b.shape);
    ASSERT_EQ(a.data.size(), b.data.size());
    EXPECT_EQ(a.data, b.data);
    for (float value : a.data) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

void ExpectSameCache(const Context& a, const Context& b) {
    EXPECT_EQ(a.cpu_cache_.key_.shape, b.cpu_cache_.key_.shape);
    EXPECT_EQ(a.cpu_cache_.value_.shape, b.cpu_cache_.value_.shape);
    EXPECT_EQ(a.cpu_cache_.key_.data, b.cpu_cache_.key_.data);
    EXPECT_EQ(a.cpu_cache_.value_.data, b.cpu_cache_.value_.data);
}

TEST(GoldenForwardTest, ExplicitLinearModesPreserveNativeDefault) {
    RestoreForwardExecution restore;
    ASSERT_EQ(DefaultMatulMode(), MatulMode::Native);
    const Tensor input = ForwardFixtureTensor({32}, 1);
    const Tensor weights = ForwardFixtureTensor({65, 32}, 2);
    SetThreadNum(4);
    for (const auto mode : {MatulMode::Native, MatulMode::Threaded,
                            MatulMode::SIMD, MatulMode::ThreadedSIMD}) {
        SCOPED_TRACE(static_cast<int>(mode));
        SetDefaultMatulMode(mode);
        EXPECT_EQ(DefaultMatulMode(), mode);
        const Tensor implicit = Linear(input, weights);
        const Tensor explicit_result = LinearDispatch(input, weights, mode);
        EXPECT_EQ(implicit.shape, explicit_result.shape);
        EXPECT_EQ(implicit.data, explicit_result.data);
    }
    EXPECT_THROW(SetDefaultMatulMode(static_cast<MatulMode>(-1)),
                 std::invalid_argument);
    EXPECT_EQ(DefaultMatulMode(), MatulMode::ThreadedSIMD);
    SetDefaultMatulMode(MatulMode::Native);
    EXPECT_EQ(DefaultMatulMode(), MatulMode::Native);
}

TEST(GoldenForwardTest, MultiTokenBatchMatchesSingleTokenPrefillAndDecode) {
    RestoreForwardExecution restore;
    const Model model = ForwardFixtureModel();
    std::vector<int> prompt{128, 'h', 'e', 'l', 'l', 'o'};

    for (const auto mode : {MatulMode::Native, MatulMode::Threaded,
                            MatulMode::SIMD, MatulMode::ThreadedSIMD}) {
        SCOPED_TRACE(static_cast<int>(mode));
        SetDefaultMatulMode(mode);
        for (const int threads : {1, 4}) {
            SCOPED_TRACE(threads);
            SetThreadNum(threads);
            Context batched(&model);
            Context singles(&model);
            Context direct(&model);
            Tensor batch_logits = ForwardBatch(
                model, batched, Batch::FromTokens(prompt, 0));
            Tensor single_logits;
            Tensor direct_logits;
            for (std::size_t pos = 0; pos < prompt.size(); ++pos) {
                single_logits = ForwardBatch(
                    model, singles, Batch::Single(prompt[pos], static_cast<int>(pos)));
                direct.pos = pos;
                direct_logits = ForwardToken(model, direct, prompt[pos]);
                EXPECT_EQ(direct.pos, pos);
            }

            ExpectSameLogits(batch_logits, single_logits);
            ExpectSameLogits(batch_logits, direct_logits);
            ExpectSameCache(batched, singles);
            ExpectSameCache(batched, direct);
            EXPECT_EQ(batched.pos, prompt.size() - 1);
            EXPECT_EQ(singles.pos, batched.pos);
            EXPECT_EQ(batched.token_id_, prompt);
            EXPECT_EQ(singles.token_id_, prompt);
            EXPECT_TRUE(direct.token_id_.empty());

            auto history = prompt;
            for (int step = 0; step < 5; ++step) {
                SCOPED_TRACE(step);
                const int next = static_cast<int>(ArgMax(batch_logits));
                EXPECT_EQ(ArgMax(single_logits), static_cast<std::size_t>(next));
                EXPECT_EQ(ArgMax(direct_logits), static_cast<std::size_t>(next));
                const int pos = static_cast<int>(prompt.size()) + step;
                batch_logits = ForwardBatch(model, batched, Batch::Single(next, pos));
                single_logits = ForwardBatch(model, singles, Batch::Single(next, pos));
                direct.pos = static_cast<std::size_t>(pos);
                direct_logits = ForwardToken(model, direct, next);
                history.push_back(next);

                ExpectSameLogits(batch_logits, single_logits);
                ExpectSameLogits(batch_logits, direct_logits);
                ExpectSameCache(batched, singles);
                ExpectSameCache(batched, direct);
                EXPECT_EQ(batched.pos, static_cast<std::size_t>(pos));
                EXPECT_EQ(singles.pos, batched.pos);
                EXPECT_EQ(direct.pos, batched.pos);
                EXPECT_EQ(batched.token_id_, history);
                EXPECT_EQ(singles.token_id_, history);
                EXPECT_TRUE(direct.token_id_.empty());
            }
        }
    }
}

}  // namespace
}  // namespace yan_lamma
