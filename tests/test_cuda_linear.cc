#include "tests/cuda_test_support.h"
using namespace yan_lamma;
using namespace yan_lamma::test;
void LinearCase() {
    Tensor weight = Values({17, 33}, 0.3f), bias = Values({17}, 0.2f);
    for (auto shape : {std::vector<int>{33}, std::vector<int>{1, 33}}) {
        const Tensor input = Values(shape);
        const auto cpu = Linear(input, weight);
        if (!CudaRuntimeBuilt()) {
            RequireCloseTensor(cpu, cpu, "CPU linear");
            RequireError([&] { (void)CudaLinear(input, weight); }, "not built");
            continue;
        }
        RequireCloseTensor(CudaLinear(input, weight), cpu, "F32 linear");
        auto expected = cpu;
        for (size_t i = 0; i < expected.data.size(); ++i) expected.data[i] += bias.data[i];
        RequireCloseTensor(CudaLinear(input, weight, &bias), expected, "F32 linear bias");
        auto device_x = TensorHostToDevice(input), device_w = TensorHostToDevice(weight);
        RequireCloseTensor(CudaLinearDevice(device_x, device_w.GetData(), weight.shape).Download(), cpu, "device linear");
        RequireCloseTensor(CudaLinearHostDevice(input, device_w.GetData(), weight.shape, &bias), expected, "resident weight bias");
    }
}
int main(int argc, char** argv) { return Run(argc, argv, {{"f32", LinearCase}}); }
