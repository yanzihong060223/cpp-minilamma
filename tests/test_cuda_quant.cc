#include "tests/cuda_test_support.h"
using namespace yan_lamma;
using namespace yan_lamma::test;
void QuantLinear(QuantType type) {
    for (int columns : {32, 86}) {
        Tensor weight = Values({7, columns}, 0.8f), input = Values({columns}), bias = Values({7}, 0.4f);
        QuanTensor q; q.type = type; q.shape = weight.shape;
        if (type == QuantType::kQ80) q.kQ80_data = TensorToQ80(weight);
        else if (type == QuantType::kQ40) q.kQ40_data = TensorToQ40(weight);
        else q.kQ41_data = TensorToQ41(weight);
        const auto expected = Linear(input, q);
        if (!CudaRuntimeBuilt()) {
            RequireCloseTensor(expected, expected, "CPU quantized finite result");
            RequireError([&] { (void)CudaQ80Linear(input, {}, weight.shape); }, "not built");
            continue;
        }
        const size_t blocks = 7 * ((columns + 31) / 32);
        const void* host = nullptr; size_t bytes = 0;
        Tensor actual;
        if (type == QuantType::kQ80) {
            Require(q.kQ80_data.size() == blocks, "q8 row blocks");
            host = q.kQ80_data.data(); bytes = blocks*sizeof(Block_Q80);
            actual = CudaQ80Linear(input, q.kQ80_data, weight.shape, &bias);
        } else if (type == QuantType::kQ40) {
            Require(q.kQ40_data.size() == blocks, "q40 row blocks");
            host = q.kQ40_data.data(); bytes = blocks*sizeof(Block_Q40);
            actual = CudaQ40Linear(input, q.kQ40_data, weight.shape, &bias);
        } else {
            Require(q.kQ41_data.size() == blocks, "q41 row blocks");
            host = q.kQ41_data.data(); bytes = blocks*sizeof(Block_Q41);
            actual = CudaQ41Linear(input, q.kQ41_data, weight.shape, &bias);
        }
        Tensor with_bias = expected;
        for (size_t i = 0; i < bias.data.size(); ++i) with_bias.data[i] += bias.data[i];
        RequireCloseTensor(actual, with_bias, "quantized host linear with bias");
        CudaDeviceBuffer buffer(bytes); buffer.upload(host, bytes);
        const auto x = TensorHostToDevice(input);
        Tensor host_actual;
        if (type == QuantType::kQ80) {
            actual = CudaQ80DeviceInputLinear(x, buffer.GetData(), q.shape, blocks).Download();
            host_actual = CudaQ80DeviceLinear(input, buffer.GetData(), q.shape, blocks);
        } else if (type == QuantType::kQ40) {
            actual = CudaQ40DeviceInputLinear(x, buffer.GetData(), q.shape, blocks).Download();
            host_actual = CudaQ40DeviceLinear(input, buffer.GetData(), q.shape, blocks);
        } else {
            actual = CudaQ41DeviceInputLinear(x, buffer.GetData(), q.shape, blocks).Download();
            host_actual = CudaQ41DeviceLinear(input, buffer.GetData(), q.shape, blocks);
        }
        RequireCloseTensor(actual, expected, "quantized device input linear");
        RequireCloseTensor(host_actual, expected, "quantized resident weight linear");
    }
}
int main(int argc, char** argv) {
    return Run(argc, argv, {
        {"q8_0", [] { QuantLinear(QuantType::kQ80); }},
        {"q4", [] { QuantLinear(QuantType::kQ40); QuantLinear(QuantType::kQ41); }},
        {"forward", [] { ForwardCase(QuantType::kQ80); }},
        {"forward_q4", [] { ForwardCase(QuantType::kQ40); ForwardCase(QuantType::kQ41); }} });
}
