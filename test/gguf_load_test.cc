#include "include/gguf_load.h"
#include "include/model.h"
#include "include/quantized_tensor.h"
#include "include/tensor.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace yan_lamma {
namespace {

// 本套测试只覆盖 VailedShape。
// ModelLoad 需要真实 GGUF 文件,留给后面的 e2e 测试补。

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

}  // namespace
}  // namespace yan_lamma
