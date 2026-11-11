#include "tests/cuda_test_support.h"
using namespace yan_lamma;
using namespace yan_lamma::test;
void Ops() {
    Tensor x = Values({86}), y = Values({86}, 1.2f), norm({86}, 1.0f);
    auto rms = RmsNorm(x, norm, 1e-5f), silu = Silu(x), mul = ElementwiseMul(x, y), softmax = SoftMax(x);
    auto model = Tiny();
    if (!CudaRuntimeBuilt()) {
        for (const auto& t : {rms, silu, mul, softmax}) RequireCloseTensor(t, t, "CPU ops finite");
        RequireError([&] { (void)CudaRmsNormHost(x, norm, 1e-5f); }, "not built");
        RequireError([&] { (void)CudaSilu(x); }, "not built");
        RequireError([&] { (void)CudaElementwiseMul(x, y); }, "not built");
        RequireError([&] { (void)CudaSoftmax(x); }, "not built");
    } else {
        auto dx = TensorHostToDevice(x), dy = TensorHostToDevice(y), dw = TensorHostToDevice(norm);
        RequireCloseTensor(CudaRmsNormHost(x, norm, 1e-5f), rms, "RMSNorm host");
        RequireCloseTensor(CudaRmsNromDevice(dx, dw.GetData(), norm.shape, 1e-5f).Download(), rms, "RMSNorm device");
        RequireCloseTensor(CudaSilu(x), silu, "SiLU host");
        RequireCloseTensor(CudaSiluDevice(dx).Download(), silu, "SiLU device");
        RequireCloseTensor(CudaElementwiseMul(x, y), mul, "mul host");
        RequireCloseTensor(CudaElementwiseMulDevice(dx, dy).Download(), mul, "mul device");
        RequireCloseTensor(CudaElementAddDevice(dx, dy).Download(), ForwardAdd(model, x, y), "add");
        RequireCloseTensor(CudaSoftmax(x), softmax, "softmax");
        Tensor extreme({5}); extreme.data = {-1000, 0, 1000, 999, -999};
        const auto reference = SoftMax(extreme);
        RequireCloseTensor(CudaSoftmax(extreme), reference, "stable softmax");
        auto embedding = TensorHostToDevice(model.embedding_data_);
        for (int token : {0, 1, 130}) RequireCloseTensor(
            CudaEmbeddingDevice(embedding.GetData(), model.embedding_data_.shape, token).Download(),
            Embedding(model, token), "embedding token=" + std::to_string(token));
    }
    for (RopeType type : {kNormal, kNeoX}) {
        for (int pos : {0, 7, 63}) {
            Tensor q = Values({4, 8}), k = Values({2, 8}, 0.3f), expected_q = q, expected_k = k;
            Rope(expected_q, expected_k, pos, 10000.0f, type);
            if (!CudaRuntimeBuilt()) {
                RequireCloseTensor(expected_q, expected_q, "CPU RoPE");
                RequireError([&] { CudaRopeHost(q, k, pos, 10000.0f, type); }, "not built");
            } else {
                auto dq = TensorHostToDevice(q), dk = TensorHostToDevice(k);
                CudaRopeDevice(dq, dk, pos, 4, 2, 8, 10000.0f, type);
                CudaRopeHost(q, k, pos, 10000.0f, type);
                RequireCloseTensor(q, expected_q, "RoPE q host");
                RequireCloseTensor(k, expected_k, "RoPE k host");
                RequireCloseTensor(dq.Download(), expected_q, "RoPE q device");
                RequireCloseTensor(dk.Download(), expected_k, "RoPE k device");
            }
        }
    }
}
int main(int argc, char** argv) { return Run(argc, argv, {{"ops", Ops}}); }
