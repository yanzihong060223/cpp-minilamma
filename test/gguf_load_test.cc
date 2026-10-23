#include "include/gguf_load.h"
#include "include/model.h"
#include "include/quantized_tensor.h"
#include "include/tensor.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace yan_lamma {
namespace {

// Shape tests use in-memory weights; loader tests below write tiny GGUF files.

ModelConfig MakeTinyConfig() {
    ModelConfig c;
    c.vocab_size   = 8;
    c.dim          = 4;
    c.hidden_dim   = 12;
    c.layers       = 2;
    c.n_heads      = 2;
    c.n_kv_heads   = 2;
    c.head_dim     = 2;   // dim / n_heads
    c.max_seq_len  = 16;
    c.rope_theta   = 10000.0f;
    c.rms_norm_eps = 1e-5f;
    c.rope_type    = RopeType::kNormal;
    return c;
}

QuanTensor MakeF32Quan(const std::vector<int>& shape) {
    QuanTensor q;
    q.type = QuantType::kF32;
    q.shape = shape;
    std::size_t total = 1;
    for (int d : shape) total *= static_cast<std::size_t>(d);
    q.kF32_data.assign(total, 0.0f);
    return q;
}

Model MakeValidModel() {
    const ModelConfig c = MakeTinyConfig();
    const int q_proj  = c.n_heads    * c.head_dim;   // 4
    const int kv_proj = c.n_kv_heads * c.head_dim;   // 4

    Model m;
    m.config = c;
    m.embedding_data_ = Tensor({c.vocab_size, c.dim});
    m.final_norm      = Tensor({c.dim});
    m.lm_head_        = MakeF32Quan({c.vocab_size, c.dim});

    m.n_layers_.resize(static_cast<std::size_t>(c.layers));
    for (int i = 0; i < c.layers; ++i) {
        Layer& L = m.n_layers_[static_cast<std::size_t>(i)];
        L.attention_norm = Tensor({c.dim});
        L.wq             = MakeF32Quan({q_proj,  c.dim});
        L.wk             = MakeF32Quan({kv_proj, c.dim});
        L.wv             = MakeF32Quan({kv_proj, c.dim});
        L.wo             = MakeF32Quan({c.dim,  q_proj});
        // bq / bk / bv 保持默认空 Tensor —— 可选,VailedShape 应放过
        L.ffn_norm       = Tensor({c.dim});
        L.w_gate         = MakeF32Quan({c.hidden_dim, c.dim});
        L.w_up           = MakeF32Quan({c.hidden_dim, c.dim});
        L.w_down         = MakeF32Quan({c.dim, c.hidden_dim});
    }
    return m;
}

TEST(VailedShapeTest, AcceptsValidModelWithoutBias) {
    Model m = MakeValidModel();
    EXPECT_NO_THROW(VailedShape(m));
}

TEST(VailedShapeTest, AcceptsValidModelWithBias) {
    Model m = MakeValidModel();
    const ModelConfig& c = m.config;
    const int q_proj  = c.n_heads    * c.head_dim;
    const int kv_proj = c.n_kv_heads * c.head_dim;
    for (auto& L : m.n_layers_) {
        L.bq = Tensor({q_proj});
        L.bk = Tensor({kv_proj});
        L.bv = Tensor({kv_proj});
    }
    EXPECT_NO_THROW(VailedShape(m));
}

TEST(VailedShapeTest, RejectsLayerCountMismatch) {
    Model m = MakeValidModel();
    m.n_layers_.pop_back();  // 现在只剩 1 层,但 config.layers == 2
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsEmbeddingShape) {
    Model m = MakeValidModel();
    m.embedding_data_ = Tensor({m.config.vocab_size, m.config.dim + 1});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsFinalNormShape) {
    Model m = MakeValidModel();
    m.final_norm = Tensor({m.config.dim + 1});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsLmHeadShape) {
    Model m = MakeValidModel();
    m.lm_head_ = MakeF32Quan({m.config.vocab_size + 1, m.config.dim});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsAttentionNormShape) {
    Model m = MakeValidModel();
    m.n_layers_[0].attention_norm = Tensor({m.config.dim + 1});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsWqShape) {
    Model m = MakeValidModel();
    const int q_proj = m.config.n_heads * m.config.head_dim;
    m.n_layers_[0].wq = MakeF32Quan({q_proj + 1, m.config.dim});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsWkShape) {
    Model m = MakeValidModel();
    m.n_layers_[1].wk = MakeF32Quan({m.config.dim, m.config.dim + 1});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsWvShape) {
    Model m = MakeValidModel();
    m.n_layers_[0].wv = MakeF32Quan({m.config.dim});  // 一维,错
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsWoShape) {
    Model m = MakeValidModel();
    const int q_proj = m.config.n_heads * m.config.head_dim;
    m.n_layers_[1].wo = MakeF32Quan({m.config.dim, q_proj + 1});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsFfnNormShape) {
    Model m = MakeValidModel();
    m.n_layers_[1].ffn_norm = Tensor({m.config.hidden_dim});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsWGateShape) {
    Model m = MakeValidModel();
    m.n_layers_[0].w_gate = MakeF32Quan({m.config.hidden_dim, m.config.dim + 1});
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsWUpShape) {
    Model m = MakeValidModel();
    m.n_layers_[0].w_up = MakeF32Quan({m.config.dim, m.config.hidden_dim});  // 行列反
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsWDownShape) {
    Model m = MakeValidModel();
    m.n_layers_[1].w_down = MakeF32Quan({m.config.hidden_dim, m.config.dim});  // 行列反
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, RejectsNonEmptyBiasWithWrongShape) {
    Model m = MakeValidModel();
    const int q_proj = m.config.n_heads * m.config.head_dim;
    m.n_layers_[0].bq = Tensor({q_proj + 1});  // 不为空且形状错
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

TEST(VailedShapeTest, AcceptsAllBiasesAbsent) {
    // bq / bk / bv 的 data 都空,Optional 版应跳过
    Model m = MakeValidModel();
    for (auto& L : m.n_layers_) {
        EXPECT_TRUE(L.bq.data.empty());
        EXPECT_TRUE(L.bk.data.empty());
        EXPECT_TRUE(L.bv.data.empty());
    }
    EXPECT_NO_THROW(VailedShape(m));
}

TEST(VailedShapeTest, RejectsZeroLayerCountWithPopulatedLayers) {
    Model m = MakeValidModel();
    m.config.layers = 0;  // 但 n_layers_ 还有 2 层
    EXPECT_THROW(VailedShape(m), std::runtime_error);
}

class GgufModelLoadTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        directory_ = std::filesystem::temp_directory_path() /
            ("yan-gguf-model-load-" + std::to_string(stamp));
        ASSERT_TRUE(std::filesystem::create_directories(directory_));
    }

    void TearDown() override {
        std::error_code ec;
        std::filesystem::remove_all(directory_, ec);
    }

    void WriteModel(bool separate_output, const std::string& omit = "",
                    bool quantized_embedding = false) {
        std::vector<char> bytes;
        const auto integer = [&bytes](uint64_t value, size_t width) {
            for (size_t i = 0; i < width; ++i) {
                bytes.push_back(static_cast<char>((value >> (8 * i)) & 0xff));
            }
        };
        const auto string = [&bytes, &integer](const std::string& value) {
            integer(value.size(), 8);
            bytes.insert(bytes.end(), value.begin(), value.end());
        };
        const auto real = [&integer](float value) {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            integer(bits, 4);
        };
        struct Weight {
            std::string name;
            std::vector<uint64_t> shape;
            float value;
        };
        std::vector<Weight> weights{
            {"token_embd.weight", {4, 8}, 0.25f},
            {"output_norm.weight", {4}, 1.0f},
            {"blk.0.attn_norm.weight", {4}, 1.0f},
            {"blk.0.attn_q.weight", {4, 4}, 0.125f},
            {"blk.0.attn_k.weight", {4, 2}, 0.125f},
            {"blk.0.attn_v.weight", {4, 2}, 0.125f},
            {"blk.0.attn_output.weight", {4, 4}, 0.125f},
            {"blk.0.ffn_norm.weight", {4}, 1.0f},
            {"blk.0.ffn_gate.weight", {4, 8}, 0.125f},
            {"blk.0.ffn_up.weight", {4, 8}, 0.125f},
            {"blk.0.ffn_down.weight", {8, 4}, 0.125f}
        };
        if (separate_output) weights.push_back({"output.weight", {4, 8}, 0.75f});
        for (auto it = weights.begin(); it != weights.end();) {
            if (it->name == omit) it = weights.erase(it);
            else ++it;
        }

        bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
        integer(3, 4);
        integer(weights.size(), 8);
        integer(9, 8);  // architecture, 6 dimensions, RMS epsilon, vocabulary
        string("general.architecture");
        integer(8, 4);
        string("qwen2");
        const std::vector<std::pair<std::string, uint32_t>> dimensions{
            {"block_count", 1}, {"embedding_length", 4},
            {"attention.head_count", 2}, {"attention.head_count_kv", 1},
            {"context_length", 32}, {"feed_forward_length", 8}
        };
        for (const auto& field : dimensions) {
            string("qwen2." + field.first);
            integer(4, 4);
            integer(field.second, 4);
        }
        string("qwen2.attention.layer_norm_rms_epsilon");
        integer(6, 4);
        real(1e-5f);
        string("tokenizer.ggml.tokens");
        integer(9, 4);
        integer(8, 4);
        integer(8, 8);
        for (int i = 0; i < 8; ++i) string("token" + std::to_string(i));

        std::vector<size_t> sizes;
        uint64_t offset = 0;
        for (const auto& weight : weights) {
            string(weight.name);
            integer(weight.shape.size(), 4);
            size_t count = 1;
            for (uint64_t dim : weight.shape) {
                integer(dim, 8);
                count *= dim;
            }
            const bool q8 = quantized_embedding && weight.name == "token_embd.weight";
            integer(q8 ? 8 : 0, 4);  // GGML Q8_0 or F32
            integer(offset, 8);
            sizes.push_back(count);
            const size_t payload_size = q8 ? 34 : count * sizeof(float);
            offset += ((payload_size + 31) / 32) * 32;
        }
        while (bytes.size() % 32 != 0) bytes.push_back(0);
        for (size_t i = 0; i < weights.size(); ++i) {
            if (quantized_embedding && weights[i].name == "token_embd.weight") {
                integer(0x3800, 2);  // fp16 scale = 0.5
                for (int j = 1; j <= 32; ++j) integer(j, 1);
            } else {
                for (size_t j = 0; j < sizes[i]; ++j) real(weights[i].value);
            }
            while (bytes.size() % 32 != 0) bytes.push_back(0);
        }
        std::ofstream file(Path(), std::ios::binary);
        ASSERT_TRUE(file.is_open());
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        file.close();
        ASSERT_TRUE(file.good());
    }

    std::string Path() const { return (directory_ / "model.gguf").string(); }
    std::filesystem::path directory_;
};

TEST_F(GgufModelLoadTest, MissingOutputUsesTokenEmbeddingWeights) {
    ASSERT_NO_FATAL_FAILURE(WriteModel(false));
    const auto model = ModelLoad(Path());

    ASSERT_TRUE(model.is_loaded) << model.load_error_;
    EXPECT_EQ(model.lm_head_.shape, model.embedding_data_.shape);
    EXPECT_EQ(ToTensor(model.lm_head_).data, model.embedding_data_.data);
}

TEST_F(GgufModelLoadTest, ExplicitOutputTakesPrecedenceOverEmbedding) {
    ASSERT_NO_FATAL_FAILURE(WriteModel(true));
    const auto model = ModelLoad(Path());

    ASSERT_TRUE(model.is_loaded) << model.load_error_;
    EXPECT_EQ(ToTensor(model.lm_head_).data, std::vector<float>(32, 0.75f));
    EXPECT_EQ(model.embedding_data_.data, std::vector<float>(32, 0.25f));
}

TEST_F(GgufModelLoadTest, SharedOutputPreservesQ8Weights) {
    ASSERT_NO_FATAL_FAILURE(WriteModel(false, "", true));
    const auto model = ModelLoad(Path());

    ASSERT_TRUE(model.is_loaded) << model.load_error_;
    EXPECT_EQ(model.lm_head_.type, QuantType::kQ80);
    EXPECT_TRUE(model.lm_head_.kF32_data.empty());
    EXPECT_EQ(model.lm_head_.kQ80_data.size(), 34u);
    EXPECT_EQ(Q80ToTensor(model.lm_head_).data, model.embedding_data_.data);
    EXPECT_FLOAT_EQ(model.embedding_data_.data.front(), 0.5f);
    EXPECT_FLOAT_EQ(model.embedding_data_.data.back(), 16.0f);
}

TEST_F(GgufModelLoadTest, MissingRequiredLayerReportsItsName) {
    ASSERT_NO_FATAL_FAILURE(WriteModel(false, "blk.0.attn_q.weight"));
    const auto model = ModelLoad(Path());

    EXPECT_FALSE(model.is_loaded);
    EXPECT_NE(model.load_error_.find("blk.0.wq"), std::string::npos);
}

}  // namespace
}  // namespace yan_lamma
