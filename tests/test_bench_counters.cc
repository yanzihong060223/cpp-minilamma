#include "tests/cuda_test_support.h"
#include <memory>
using namespace yan_lamma;
using namespace yan_lamma::test;
int main() {
    try {
        auto model = Tiny();
        Require(ModelLinearWeightTypeTotals(model, QuantType::kF32) == 1 + 7 * model.n_layers_.size(), "linear total");
        ModeloQuanTensorQ40(model);
        Require(ModelLinearWeightTypeTotals(model, QuantType::kF32) == 0, "quantized f32 total");
        Require(ModelLinearWeightTypeTotals(model, QuantType::kQ40) == 1 + 7 * model.n_layers_.size(), "q4 total");
        auto first = std::make_shared<CudaAllocationStats>();
        auto second = std::make_shared<CudaAllocationStats>();
        if (CudaRuntimeBuilt()) {
            CudaDeviceBuffer survivor;
            {
                CudaAllocationScope scope(first);
                CudaDeviceBuffer a(128), b(64);
                Require(first->current_bytes == 192 && first->peak_bytes == 192, "allocation accounting");
                CudaDeviceBuffer moved(std::move(a));
                Require(first->current_bytes == 192 && a.IsEmpty(), "move constructor ownership");
                b = std::move(moved);
                Require(first->current_bytes == 128, "move assignment releases destination");
                {
                    CudaAllocationScope nested(second);
                    CudaDeviceBuffer c(32);
                    survivor = std::move(c);
                    Require(second->current_bytes == 32, "nested allocation owner");
                    b.ReSet();
                    Require(first->current_bytes == 0, "release attributed to original owner");
                }
                CudaDeviceBuffer restored(16);
                Require(first->current_bytes == 16, "restored scope");
            }
            Require(first->current_bytes == 0 && first->peak_bytes == 192, "first released, peak retained");
            survivor.ReSet();
            Require(second->current_bytes == 0 && second->peak_bytes == 32, "owner survives scope");
            CudaAllocationScope scope(first);
            RequireError([&] { CudaDeviceBuffer invalid(16, DeviceCount()); }, "cudaSetDevice");
            Require(first->current_bytes == 0, "failed allocation must not count");
        } else {
            CudaAllocationScope scope(first);
            RequireError([&] { CudaDeviceBuffer unavailable(16); }, "not built");
            Require(first->current_bytes == 0 && first->peak_bytes == 0, "unbuilt allocation not counted");
        }
        std::cout << "PASS benchmark totals and allocation ownership "
                  << (CudaRuntimeBuilt() ? "CUDA" : "CPU branch; GPU ownership unverified") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
