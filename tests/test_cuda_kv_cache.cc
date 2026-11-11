#include "tests/cuda_test_support.h"
using namespace yan_lamma;
using namespace yan_lamma::test;
void Cache() {
    KvCache cpu(2, 6, 2, 8);
    if (!CudaRuntimeBuilt()) {
        Tensor k = Values({2,8}), v = Values({2,8}, 1);
        cpu.Write(1, 5, k, v);
        Require(cpu.KeyPtr(1, 5, 1)[3] == k.data[11], "CPU KV slot");
        Require(cpu.ValPtr(1, 5, 0)[3] == v.data[3], "CPU KV value");
        RequireError([] { CudaCache cache(2, 6, 2, 8); }, "not built");
        return;
    }
    CudaCache gpu(2, 6, 2, 8);
    Require(gpu.Bytes_() == 2*2*6*2*8*sizeof(float), "KV allocation bytes");
    for (int layer : {0, 1}) for (int pos : {0, 2, 5}) {
        Tensor k = Values({2, 8}, float(layer*7+pos)), v = Values({2, 8}, float(layer+pos)+0.7f);
        cpu.Write(layer, pos, k, v);
        if (pos == 2) {
            auto dk = TensorHostToDevice(k), dv = TensorHostToDevice(v);
            gpu.WriteDevice(layer, pos, dk, dv);
        } else gpu.WriteHost(layer, pos, k, v);
    }
    for (int layer : {0, 1}) for (int pos : {0, 2, 5}) {
        Tensor k({2,8}), v({2,8});
        for (int h = 0; h < 2; ++h) for (int d = 0; d < 8; ++d) {
            k.data[h*8+d] = cpu.KeyPtr(layer, pos, h)[d];
            v.data[h*8+d] = cpu.ValPtr(layer, pos, h)[d];
        }
        RequireCloseTensor(gpu.ReadKey(layer, pos), k, "KV key slot");
        RequireCloseTensor(gpu.ReadVal(layer, pos), v, "KV value slot");
        Tensor head({8});
        for (int d = 0; d < 8; ++d) head.data[d] = k.data[8+d];
        RequireCloseTensor(gpu.ReadHeadKey(layer, pos, 1), head, "KV head key");
        for (int d = 0; d < 8; ++d) head.data[d] = v.data[8+d];
        RequireCloseTensor(gpu.ReadHeadVal(layer, pos, 1), head, "KV head value");
        Require(gpu.SlotOffsetBytes(layer, pos, 1) == size_t((layer*6+pos)*16+8)*4, "KV byte offset");
    }
    RequireError([&] { (void)gpu.ReadKey(2, 0); }, "out of range");
    RequireError([&] { (void)gpu.ReadVal(0, 6); }, "out of range");
    CudaCache moved(std::move(gpu));
    Require(gpu.Bytes_() == 0 && moved.Bytes_() > 0, "KV move ownership");
    moved.Clear(); RequireCloseTensor(moved.ReadKey(1,5), Tensor({2,8}, 0), "KV clear");
    RequireCloseTensor(moved.ReadVal(1,5), Tensor({2,8}, 0), "KV value clear");
    moved.Reset(); Require(moved.Bytes_() == 0 && moved.n_layers() == 0, "KV release");
    moved.Reset(1, 2, 1, 8);
    RequireCloseTensor(moved.ReadKey(0,1), Tensor({1,8}, 0), "KV reinitialize");
}
int main(int argc, char** argv) { return Run(argc, argv, {{"kv", Cache}}); }
