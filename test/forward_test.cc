#include "include/batch.h"
#include "include/context.h"
#include "include/cuda_attention.h"
#include "include/cuda_ops.h"
#include "include/forward.h"
#include "include/model.h"
#include "include/ops.h"
#include "include/quant.h"
#include "include/quantized_tensor.h"
#include "matumal/matuml_ops.h"
#include "thread_pool/thread_pool.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <string>
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

class QuantizedForwardTest : public ::testing::TestWithParam<QuantType> {};

TEST_P(QuantizedForwardTest, PrefillAndDecodeMatchRowDecodedWeights) {
    RestoreForwardExecution restore;
    SetDefaultMatulMode(MatulMode::Native);
    SetThreadNum(1);
    const auto type = GetParam();
    SCOPED_TRACE(static_cast<int>(type));
    Model model = ForwardFixtureModel();
    if (type == QuantType::kQ80) {
        ModeloQuanTensorQ80(model);
    } else {
        ModeloQuanTensorQ40(model);
    }
    // Each of the 32 rows needs three blocks for its 86 input features.
    const auto& down = model.n_layers_[0].w_down;
    ASSERT_EQ(down.shape, (std::vector<int>{32, 86}));
    ASSERT_EQ(type == QuantType::kQ80 ? down.kQ80_data.size()
                                     : down.kQ40_data.size(), 96u);

    Model reference = model;
    const auto decode = [](QuanTensor& weight) {
        const Tensor decoded = weight.type == QuantType::kQ80
            ? Q80ToTensor(weight.kQ80_data, weight.shape)
            : Q40ToTensor(weight.kQ40_data, weight.shape);
        weight = ToQuanTensor(decoded);
    };
    decode(reference.lm_head_);
    for (auto& layer : reference.n_layers_) {
        for (auto* weight : {&layer.wq, &layer.wk, &layer.wv, &layer.wo,
                             &layer.w_gate, &layer.w_up, &layer.w_down}) {
            decode(*weight);
        }
    }

    Context batched(&model);
    Context singles(&model);
    Context expected(&reference);
    std::vector<int> prompt{128, 'h', 'i'};
    Tensor actual = ForwardBatch(model, batched, Batch::FromTokens(prompt, 0));
    Tensor single;
    for (std::size_t pos = 0; pos < prompt.size(); ++pos) {
        singles.pos = pos;
        single = ForwardToken(model, singles, prompt[pos]);
    }
    Tensor golden = ForwardBatch(reference, expected, Batch::FromTokens(prompt, 0));
    const auto compare = [&] {
        ExpectSameLogits(actual, single);
        ExpectSameCache(batched, singles);
        ASSERT_EQ(actual.shape, golden.shape);
        ASSERT_EQ(actual.data.size(), golden.data.size());
        for (std::size_t i = 0; i < actual.data.size(); ++i) {
            EXPECT_TRUE(std::isfinite(golden.data[i]));
            EXPECT_NEAR(actual.data[i], golden.data[i], 1e-5f);
        }
    };
    compare();
    for (int step = 0; step < 3; ++step) {
        SCOPED_TRACE(step);
        const int next = static_cast<int>(ArgMax(actual));
        const int pos = static_cast<int>(prompt.size()) + step;
        actual = ForwardBatch(model, batched, Batch::Single(next, pos));
        singles.pos = pos;
        single = ForwardToken(model, singles, next);
        golden = ForwardBatch(reference, expected, Batch::Single(next, pos));
        compare();
    }
}

INSTANTIATE_TEST_SUITE_P(
    RowQuantization, QuantizedForwardTest,
    ::testing::Values(QuantType::kQ80, QuantType::kQ40));


// F32 CUDA coverage uses the same deterministic tiny fixture as the CPU tests.
Model CudaForwardFixture(int kv_heads, RopeType rope, bool biases) {
    Model model = ForwardFixtureModel();
    model.config.n_kv_heads = kv_heads;
    model.config.rope_type = rope;
    for (int index = 0; index < model.config.layers; ++index) {
        auto& layer = model.n_layers_[index];
        const int kv_dim = kv_heads * model.config.head_dim;
        layer.wk = ToQuanTensor(ForwardFixtureTensor({kv_dim, model.config.dim}, index * 10 + 4));
        layer.wv = ToQuanTensor(ForwardFixtureTensor({kv_dim, model.config.dim}, index * 10 + 5));
        if (biases) {
            layer.bq = ForwardFixtureTensor({model.config.dim}, index * 10 + 20);
            layer.bk = ForwardFixtureTensor({kv_dim}, index * 10 + 21);
            layer.bv = ForwardFixtureTensor({kv_dim}, index * 10 + 22);
        }
    }
    return model;
}

void ExpectCudaForwardNear(const Tensor& actual, const Tensor& expected) {
    ASSERT_EQ(actual.shape, expected.shape);
    ASSERT_EQ(actual.data.size(), expected.data.size());
    for (size_t i = 0; i < expected.data.size(); ++i) {
        ASSERT_TRUE(std::isfinite(expected.data[i])) << "reference element " << i;
        ASSERT_TRUE(std::isfinite(actual.data[i])) << "actual element " << i;
        EXPECT_NEAR(actual.data[i], expected.data[i],
                    1e-5f + 1e-4f * std::abs(expected.data[i])) << "element " << i;
    }
}

void ExpectDeviceStage(const CudaTensor& actual, const Tensor& expected,
                       int device, const std::string& stage) {
    SCOPED_TRACE(stage);
    EXPECT_EQ(actual.DeviceId(), device);
    EXPECT_EQ(actual.NumEle(), expected.GetDataSize());
    EXPECT_EQ(actual.Bytes(), expected.GetDataSize() * sizeof(float));
    // Diagnostic downloads are outside production forward and its counters.
    ExpectCudaForwardNear(actual.Download(), expected);
}

const CudaloadWeight& CudaFixtureWeight(const Model& model, std::string name) {
    if (model.cuda_weights) {
        for (const auto& weight : model.cuda_weights->weights) {
            if (weight.name == name && weight.type == QuantType::kF32) {
                return weight;
            }
        }
    }
    throw std::runtime_error("CUDA test fixture weight missing: " + name);
}

Tensor ReferenceAdd(const Tensor& a, const Tensor& b) {
    if (a.shape != b.shape) {
        throw std::runtime_error("ReferenceAdd: shape mismatch");
    }
    Tensor result = a;
    for (size_t i = 0; i < result.data.size(); ++i) {
        result.data[i] += b.data[i];
    }
    return result;
}

Tensor ReferenceEmbedding(const Model& model, int token) {
    Tensor result({model.config.dim});
    std::copy_n(model.embedding_data_.data.begin() + token * model.config.dim,
                model.config.dim, result.data.begin());
    return result;
}

// Independent attention loop: causal prefix, stable softmax, and explicit GQA mapping.
Tensor ReferenceCacheAttention(const Tensor& q, KvCache& cache,
                               int layer, int pos, const ModelConfig& config) {
    Tensor result({config.n_heads, config.head_dim});
    const int group = config.n_heads / config.n_kv_heads;
    for (int head = 0; head < config.n_heads; ++head) {
        std::vector<double> scores(pos + 1);
        for (int t = 0; t <= pos; ++t) {
            const float* key = cache.KeyPtr(layer, t, head / group);
            double dot = 0;
            for (int d = 0; d < config.head_dim; ++d) {
                dot += static_cast<double>(q.data[head * config.head_dim + d]) * key[d];
            }
            scores[t] = dot / std::sqrt(static_cast<double>(config.head_dim));
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double total = 0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            total += score;
        }
        for (int d = 0; d < config.head_dim; ++d) {
            double value = 0;
            for (int t = 0; t <= pos; ++t) {
                value += scores[t] / total * cache.ValPtr(layer, t, head / group)[d];
            }
            result.data[head * config.head_dim + d] = static_cast<float>(value);
        }
    }
    return result;
}

std::vector<unsigned char> HostWeightSnapshot(const Model& model) {
    std::vector<unsigned char> result;
    const auto append = [&](const auto& values) {
        if (!values.empty()) {
            const auto* bytes = reinterpret_cast<const unsigned char*>(values.data());
            result.insert(result.end(), bytes, bytes + values.size() * sizeof(values[0]));
        }
    };
    const auto tensor = [&](const Tensor& value) { append(value.data); };
    const auto quant = [&](const QuanTensor& value) {
        append(value.kF32_data);
        append(value.kF16_data);
        append(value.kQ40_data);
        append(value.kQ41_data);
        append(value.kQ80_data);
    };
    tensor(model.embedding_data_);
    tensor(model.final_norm);
    quant(model.lm_head_);
    for (const auto& layer : model.n_layers_) {
        tensor(layer.attention_norm);
        tensor(layer.ffn_norm);
        tensor(layer.bq);
        tensor(layer.bk);
        tensor(layer.bv);
        for (const auto* weight : {&layer.wq, &layer.wk, &layer.wv, &layer.wo,
                                   &layer.w_gate, &layer.w_up, &layer.w_down}) {
            quant(*weight);
        }
    }
    return result;
}

std::vector<std::vector<unsigned char>> DeviceWeightSnapshot(const Model& model) {
    std::vector<std::vector<unsigned char>> result;
    for (const auto& weight : model.cuda_weights->weights) {
        result.emplace_back(weight.buffer.GetByteSize());
        weight.buffer.download(result.back().data(), result.back().size());
    }
    return result;
}

void ExpectRuntimeStats(const Model& model, size_t tokens, size_t history_sum) {
    const size_t layers = model.config.layers;
    size_t biases = 0;
    for (const auto& layer : model.n_layers_) {
        biases += !layer.bq.data.empty();
        biases += !layer.bk.data.empty();
        biases += !layer.bv.data.empty();
    }
    if (!model.cuda_weights) {
        tokens = 0;
        history_sum = 0;
    }
    EXPECT_EQ(ModelCudaLinearCalls(model), tokens * (7 * layers + 1));
    EXPECT_EQ(ModelCudaActivationCalls(model), tokens * (7 * layers + 1 + biases));
    EXPECT_EQ(ModelCudaAttentionCalls(model), tokens * layers);
    EXPECT_EQ(ModelCudaAttentionCpuFallbacks(model), 0u);
    EXPECT_EQ(ModelCudaHostToDeviceCopies(model), 0u);
    EXPECT_EQ(ModelCudaHostToDeviceBytes(model), 0u);
    EXPECT_EQ(ModelCudaDeviceToHostCopies(model), tokens);
    EXPECT_EQ(ModelCudaDeviceToHostBytes(model),
              tokens * model.config.vocab_size * sizeof(float));
    EXPECT_EQ(ModelCudaKvCacheWriteBytes(model),
              tokens * layers * 2 * model.config.n_kv_heads * model.config.head_dim * sizeof(float));
    EXPECT_EQ(ModelCudaKvCacheReadBytes(model),
              history_sum * layers * 2 * model.config.n_heads * model.config.head_dim * sizeof(float));
}

TEST(CudaForwardTest, TinyModelMatchesCpu) {
    RestoreForwardExecution restore;
    SetDefaultMatulMode(MatulMode::Native);
    SetThreadNum(1);
    // Exercise CPU execution even when this build has no usable CUDA device.
    Model cpu_only = CudaForwardFixture(2, RopeType::kNormal, true);
    Context cpu_context(&cpu_only);
    const std::vector<int> tokens{128, 104, 105, 33, 7, 129};
    for (size_t pos = 0; pos < tokens.size(); ++pos) {
        cpu_context.pos = pos;
        const Tensor logits = ForwardToken(cpu_only, cpu_context, tokens[pos]);
        ASSERT_EQ(logits.shape, (std::vector<int>{131}));
        for (float value : logits.data) {
            ASSERT_TRUE(std::isfinite(value));
        }
        ExpectRuntimeStats(cpu_only, 0, 0);
    }
    try {
        (void)GetCudaInfo(0);
    } catch (const std::exception& error) {
        GTEST_SKIP() << "CUDA device unavailable: " << error.what();
    }
    for (int kv_heads : {4, 2}) {
        for (RopeType rope : {RopeType::kNormal, RopeType::kNeoX}) {
            for (bool biases : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "kv_heads=" << kv_heads
                             << " rope=" << static_cast<int>(rope) << " biases=" << biases);
                Model cpu = CudaForwardFixture(kv_heads, rope, biases);
                Model gpu = cpu;
                UploadModelWeightToCuda(gpu, 0);
                Context expected(&cpu), actual(&gpu), batched(&gpu);
                Tensor prefill_logits;
                std::vector<Tensor> golden_logits;
                for (size_t pos = 0; pos < tokens.size(); ++pos) {
                    SCOPED_TRACE(::testing::Message() << "pos=" << pos);
                    expected.pos = pos;
                    actual.pos = pos;
                    const Tensor golden = ForwardToken(cpu, expected, tokens[pos]);
                    golden_logits.push_back(golden);
                    const Tensor logits = ForwardToken(gpu, actual, tokens[pos]);
                    ASSERT_EQ(logits.shape, (std::vector<int>{131}));
                    ExpectCudaForwardNear(logits, golden);
                    if (pos == 2) {
                        prefill_logits = logits;
                    }
                }
                std::vector<int> prompt(tokens.begin(), tokens.begin() + 3);
                ExpectCudaForwardNear(
                    ForwardBatch(gpu, batched, Batch::FromTokens(prompt, 0)), prefill_logits);
                for (size_t pos = 3; pos < tokens.size(); ++pos) {
                    SCOPED_TRACE(::testing::Message() << "batched decode pos=" << pos);
                    const Tensor batch_logits = ForwardBatch(
                        gpu, batched, Batch::Single(tokens[pos], static_cast<int>(pos)));
                    ExpectCudaForwardNear(batch_logits, golden_logits[pos]);
                }
                ExpectRuntimeStats(cpu, 0, 0);
            }
        }
    }
}

TEST(CudaForwardTest, IntermediateDataAndRuntimeStats) {
    RestoreForwardExecution restore;
    SetDefaultMatulMode(MatulMode::Native);
    SetThreadNum(1);
    try {
        (void)GetCudaInfo(0);
    } catch (const std::exception& error) {
        GTEST_SKIP() << "CUDA device unavailable: " << error.what();
    }
    const std::vector<int> tokens{128, 104, 105, 33, 7, 129};
    for (int kv_heads : {4, 2}) {
        for (RopeType rope : {RopeType::kNormal, RopeType::kNeoX}) {
            SCOPED_TRACE(::testing::Message() << "kv_heads=" << kv_heads
                         << " rope=" << static_cast<int>(rope));
            Model cpu = CudaForwardFixture(kv_heads, rope, true);
            Model gpu = cpu;
            const auto cpu_before = HostWeightSnapshot(cpu);
            const auto gpu_host_before = HostWeightSnapshot(gpu);
            UploadModelWeightToCuda(gpu, 0);
            const auto gpu_before = DeviceWeightSnapshot(gpu);
            Context host(&cpu), device(&gpu);
            ExpectRuntimeStats(gpu, 0, 0);
            const auto& c = cpu.config;
            // Previous slots must remain bitwise unchanged as later tokens are written.
            std::vector<std::vector<Tensor>> old_keys(c.layers), old_values(c.layers);
            for (size_t pos = 0; pos < tokens.size(); ++pos) {
                SCOPED_TRACE(::testing::Message() << "pos=" << pos);
                Tensor x = ReferenceEmbedding(cpu, tokens[pos]);
                const auto& embedding = CudaFixtureWeight(gpu, "embedding_data_");
                CudaTensor dx = CudaEmbeddingDevice(
                    embedding.buffer.GetData(), embedding.shape, tokens[pos], 0);
                ExpectDeviceStage(dx, x, 0, "embedding");
                for (int layer_index = 0; layer_index < c.layers; ++layer_index) {
                    SCOPED_TRACE(::testing::Message() << "layer=" << layer_index);
                    const auto& layer = cpu.n_layers_[layer_index];
                    const std::string prefix = "blk." + std::to_string(layer_index) + ".";
                    const auto linear = [&](const Tensor& input, const CudaTensor& din,
                                            const QuanTensor& weight, const std::string& suffix,
                                            Tensor& output, CudaTensor& dout) {
                        output = Linear(input, weight);
                        const auto& w = CudaFixtureWeight(gpu, prefix + suffix);
                        dout = CudaLinearDevice(din, w.buffer.GetData(), w.shape, 0);
                        ExpectDeviceStage(dout, output, 0, suffix);
                    };
                    const auto norm = [&](const Tensor& input, const CudaTensor& din,
                                          const Tensor& weight, const std::string& suffix,
                                          Tensor& output, CudaTensor& dout) {
                        output = RmsNorm(input, weight, c.rms_norm_eps);
                        const auto& w = CudaFixtureWeight(gpu, prefix + suffix);
                        dout = CudaRmsNromDevice(din, w.buffer.GetData(), w.shape, c.rms_norm_eps, 0);
                        ExpectDeviceStage(dout, output, 0, suffix);
                    };
                    Tensor h, q, k, v;
                    CudaTensor dh, dq, dk, dv;
                    norm(x, dx, layer.attention_norm, "attention_norm", h, dh);
                    linear(h, dh, layer.wq, "wq", q, dq);
                    linear(h, dh, layer.wk, "wk", k, dk);
                    linear(h, dh, layer.wv, "wv", v, dv);
                    const auto bias = [&](Tensor& value, CudaTensor& dvalue,
                                          const Tensor& b, const std::string& suffix) {
                        value = ReferenceAdd(value, b);
                        const auto& w = CudaFixtureWeight(gpu, prefix + suffix);
                        dvalue = OCudaElementAddDevice(dvalue, w.buffer.GetData(), w.shape, 0);
                        ExpectDeviceStage(dvalue, value, 0, suffix);
                    };
                    bias(q, dq, layer.bq, "bq");
                    bias(k, dk, layer.bk, "bk");
                    bias(v, dv, layer.bv, "bv");
                    void* q_ptr = dq.GetData();
                    void* k_ptr = dk.GetData();
                    void* v_ptr = dv.GetData();
                    q = q.Reshape({c.n_heads, c.head_dim}, "test q");
                    k = k.Reshape({c.n_kv_heads, c.head_dim}, "test k");
                    v = v.Reshape({c.n_kv_heads, c.head_dim}, "test v");
                    dq.Reshape(q.shape);
                    dk.Reshape(k.shape);
                    dv.Reshape(v.shape);
                    EXPECT_EQ(dq.GetData(), q_ptr);
                    EXPECT_EQ(dk.GetData(), k_ptr);
                    EXPECT_EQ(dv.GetData(), v_ptr);
                    ExpectDeviceStage(dq, q, 0, "q reshape");
                    ExpectDeviceStage(dk, k, 0, "k reshape");
                    ExpectDeviceStage(dv, v, 0, "v reshape");
                    Rope(q, k, static_cast<int>(pos), c.rope_theta, c.rope_type);
                    CudaRopeDevice(dq, dk, static_cast<int>(pos), c.n_heads,
                        c.n_kv_heads, c.head_dim, c.rope_theta, c.rope_type, 0);
                    ExpectDeviceStage(dq, q, 0, "RoPE q");
                    ExpectDeviceStage(dk, k, 0, "RoPE k");
                    host.cpu_cache_.Write(layer_index, static_cast<int>(pos), k, v);
                    device.cuda_cache_.WriteDevice(layer_index, static_cast<int>(pos), dk, dv);
                    for (size_t t = 0; t <= pos; ++t) {
                        SCOPED_TRACE(::testing::Message() << "cache slot=" << t);
                        const Tensor key = device.cuda_cache_.ReadKey(layer_index, static_cast<int>(t));
                        const Tensor value = device.cuda_cache_.ReadVal(layer_index, static_cast<int>(t));
                        Tensor expected_key({c.n_kv_heads, c.head_dim});
                        Tensor expected_value(expected_key.shape);
                        for (int head = 0; head < c.n_kv_heads; ++head) {
                            std::copy_n(host.cpu_cache_.KeyPtr(layer_index, static_cast<int>(t), head),
                                c.head_dim, expected_key.data.begin() + head * c.head_dim);
                            std::copy_n(host.cpu_cache_.ValPtr(layer_index, static_cast<int>(t), head),
                                c.head_dim, expected_value.data.begin() + head * c.head_dim);
                        }
                        ExpectCudaForwardNear(key, expected_key);
                        ExpectCudaForwardNear(value, expected_value);
                        if (t < pos) {
                            EXPECT_EQ(key.data, old_keys[layer_index][t].data);
                            EXPECT_EQ(value.data, old_values[layer_index][t].data);
                        } else {
                            old_keys[layer_index].push_back(key);
                            old_values[layer_index].push_back(value);
                        }
                    }
                    Tensor attention = ReferenceCacheAttention(q, host.cpu_cache_, layer_index,
                                                               static_cast<int>(pos), c);
                    CudaTensor dattn = CudaAttentionDevice(dq, device.cuda_cache_, layer_index,
                        static_cast<int>(pos), c.n_heads, c.n_kv_heads, c.head_dim, 0);
                    ExpectDeviceStage(dattn, attention, 0, "attention");
                    void* attention_ptr = dattn.GetData();
                    attention = attention.Reshape({c.dim}, "test attention flatten");
                    dattn.Reshape({c.dim});
                    EXPECT_EQ(dattn.GetData(), attention_ptr);
                    ExpectDeviceStage(dattn, attention, 0, "attention flatten");
                    Tensor projected;
                    CudaTensor dprojected;
                    linear(attention, dattn, layer.wo, "wo", projected, dprojected);
                    Tensor residual = ReferenceAdd(x, projected);
                    CudaTensor dresidual = CudaElementAddDevice(dx, dprojected, 0);
                    ExpectDeviceStage(dresidual, residual, 0, "attention residual");
                    Tensor h2, gate, up;
                    CudaTensor dh2, dgate, dup;
                    norm(residual, dresidual, layer.ffn_norm, "ffn_norm", h2, dh2);
                    linear(h2, dh2, layer.w_gate, "w_gate", gate, dgate);
                    linear(h2, dh2, layer.w_up, "w_up", up, dup);
                    Tensor silu = Silu(gate);
                    CudaTensor dsilu = CudaSiluDevice(dgate, 0);
                    ExpectDeviceStage(dsilu, silu, 0, "SiLU");
                    Tensor ff = ElementwiseMul(silu, up);
                    CudaTensor dff = CudaElementwiseMulDevice(dsilu, dup, 0);
                    ExpectDeviceStage(dff, ff, 0, "SwiGLU mul");
                    Tensor down;
                    CudaTensor ddown;
                    linear(ff, dff, layer.w_down, "w_down", down, ddown);
                    x = ReferenceAdd(residual, down);
                    dx = CudaElementAddDevice(dresidual, ddown, 0);
                    ExpectDeviceStage(dx, x, 0, "FFN residual");
                }
                const auto& final_norm = CudaFixtureWeight(gpu, "final_norm");
                Tensor normed = RmsNorm(x, cpu.final_norm, c.rms_norm_eps);
                CudaTensor dnormed = CudaRmsNromDevice(dx, final_norm.buffer.GetData(),
                    final_norm.shape, c.rms_norm_eps, 0);
                ExpectDeviceStage(dnormed, normed, 0, "final_norm");
                const auto& lm_head = CudaFixtureWeight(gpu, "lm_head");
                CudaTensor logits = CudaLinearDevice(dnormed, lm_head.buffer.GetData(), lm_head.shape, 0);
                ExpectDeviceStage(logits, Linear(normed, cpu.lm_head_), 0, "logits");
            }
            // Direct operator calls and diagnostic downloads do not update model counters.
            ExpectRuntimeStats(gpu, 0, 0);
            EXPECT_EQ(HostWeightSnapshot(cpu), cpu_before);
            EXPECT_EQ(HostWeightSnapshot(gpu), gpu_host_before);
            EXPECT_EQ(DeviceWeightSnapshot(gpu), gpu_before);

            // Separate models/contexts exercise the actual production forward and its counters.
            Model runtime_cpu = CudaForwardFixture(kv_heads, rope, true);
            Model runtime_gpu = runtime_cpu;
            UploadModelWeightToCuda(runtime_gpu, 0);
            Context runtime_host(&runtime_cpu), runtime_device(&runtime_gpu);
            const auto runtime_host_before = HostWeightSnapshot(runtime_cpu);
            const auto runtime_gpu_host_before = HostWeightSnapshot(runtime_gpu);
            const auto runtime_weights = DeviceWeightSnapshot(runtime_gpu);
            const size_t weight_count = runtime_gpu.cuda_weights->uploaded_weight_count;
            const size_t weight_bytes = runtime_gpu.cuda_weights->uploaded_bytes;
            ExpectRuntimeStats(runtime_gpu, 0, 0);
            size_t history_sum = 0;
            for (size_t pos = 0; pos < tokens.size(); ++pos) {
                SCOPED_TRACE(::testing::Message() << "runtime pos=" << pos);
                runtime_host.pos = pos;
                runtime_device.pos = pos;
                const Tensor expected = ForwardToken(runtime_cpu, runtime_host, tokens[pos]);
                ExpectCudaForwardNear(ForwardToken(runtime_gpu, runtime_device, tokens[pos]), expected);
                history_sum += pos + 1;
                ExpectRuntimeStats(runtime_gpu, pos + 1, history_sum);
                ExpectRuntimeStats(runtime_cpu, 0, 0);
            }
            std::vector<Tensor> cache_before;
            for (int layer = 0; layer < c.layers; ++layer) {
                for (size_t pos = 0; pos < tokens.size(); ++pos) {
                    cache_before.push_back(runtime_device.cuda_cache_.ReadKey(layer, static_cast<int>(pos)));
                    cache_before.push_back(runtime_device.cuda_cache_.ReadVal(layer, static_cast<int>(pos)));
                }
            }
            ResetModelCudaRuntimeStats(runtime_gpu);
            ResetModelCudaRuntimeStats(runtime_cpu);
            ExpectRuntimeStats(runtime_gpu, 0, 0);
            ExpectRuntimeStats(runtime_cpu, 0, 0);
            EXPECT_EQ(runtime_gpu.cuda_weights->uploaded_weight_count, weight_count);
            EXPECT_EQ(runtime_gpu.cuda_weights->uploaded_bytes, weight_bytes);
            EXPECT_EQ(DeviceWeightSnapshot(runtime_gpu), runtime_weights);
            size_t slot = 0;
            for (int layer = 0; layer < c.layers; ++layer) {
                for (size_t pos = 0; pos < tokens.size(); ++pos) {
                    EXPECT_EQ(runtime_device.cuda_cache_.ReadKey(layer, static_cast<int>(pos)).data,
                              cache_before[slot++].data);
                    EXPECT_EQ(runtime_device.cuda_cache_.ReadVal(layer, static_cast<int>(pos)).data,
                              cache_before[slot++].data);
                }
            }
            runtime_host.pos = tokens.size();
            runtime_device.pos = tokens.size();
            const Tensor expected = ForwardToken(runtime_cpu, runtime_host, 42);
            ExpectCudaForwardNear(ForwardToken(runtime_gpu, runtime_device, 42), expected);
            ExpectRuntimeStats(runtime_gpu, 1, tokens.size() + 1);
            ExpectRuntimeStats(runtime_cpu, 0, 0);
            EXPECT_EQ(HostWeightSnapshot(runtime_cpu), runtime_host_before);
            EXPECT_EQ(HostWeightSnapshot(runtime_gpu), runtime_gpu_host_before);
            EXPECT_EQ(DeviceWeightSnapshot(runtime_gpu), runtime_weights);
        }
    }
}

}  // namespace
}  // namespace yan_lamma
