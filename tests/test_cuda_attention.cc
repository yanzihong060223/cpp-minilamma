#include "tests/cuda_test_support.h"
using namespace yan_lamma;
using namespace yan_lamma::test;
void Attention(bool history) {
    auto model = Tiny();
    for (int kv_heads : {1, 2, 4}) {
        const int last = history ? 5 : 0;
        KvCache cpu(2, 7, kv_heads, 8);
        CudaCache gpu;
        if (CudaRuntimeBuilt()) gpu.Reset(2, 7, kv_heads, 8);
        // Populate future positions with unrelated data; attention must ignore them.
        for (int pos = 0; pos < 7; ++pos) {
            Tensor k = Values({kv_heads,8}, float(pos)), v = Values({kv_heads,8}, float(pos)+0.4f);
            cpu.Write(1, pos, k, v);
            if (CudaRuntimeBuilt()) {
                if (history) {
                    auto dk = TensorHostToDevice(k), dv = TensorHostToDevice(v);
                    gpu.WriteDevice(1, pos, dk, dv);
                } else gpu.WriteHost(1, pos, k, v);
            }
        }
        for (int pos = 0; pos <= last; ++pos) {
            Tensor q = Values({4,8}, 0.2f), k = Values({kv_heads,8}, float(pos)), v = Values({kv_heads,8}, float(pos)+0.4f);
            Tensor expected = ForwardAttention(model, q, k, v, 1, pos, cpu, 4, kv_heads, 8);
            if (!CudaRuntimeBuilt()) {
                RequireCloseTensor(expected, expected, "CPU attention finite");
                RequireError([&] { (void)CudaAttentionDecode(q, gpu, 1, pos, 4, kv_heads, 8); }, "not built");
            } else {
                RequireCloseTensor(CudaAttentionDecode(q, gpu, 1, pos, 4, kv_heads, 8), expected, "attention host");
                auto dq = TensorHostToDevice(q);
                RequireCloseTensor(CudaAttentionDevice(dq, gpu, 1, pos, 4, kv_heads, 8).Download(), expected, "attention GPU KV");
            }
        }
    }
}
int main(int argc, char** argv) {
    return Run(argc, argv, {{"direct", [] { Attention(false); }}, {"kv", [] { Attention(true); }},
        {"forward", [] { ForwardCase(QuantType::kF32); }}});
}
