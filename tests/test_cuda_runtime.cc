#include "tests/cuda_test_support.h"
#include <limits>
using namespace yan_lamma;
using namespace yan_lamma::test;
void Runtime() {
    Require(CloseEnough(0.002f, 0), "absolute tolerance inclusive");
    Require(!CloseEnough(0.00201f, 0), "absolute tolerance strict outside");
    Require(CloseEnough(100.1f, 100), "relative tolerance");
    Require(!CloseEnough(100.3f, 100), "relative outside");
    Require(!CloseEnough(NAN, 0) && !CloseEnough(INFINITY, INFINITY), "nonfinite rejected");
    Tensor a({2}); a.data = {0, 1}; Tensor b = a; b.data[1] = 2;
    RequireError([&] { RequireCloseTensor(a, b, "diagnostic"); }, "index=1 actual=1 expected=2");
    RequireError([&] { RequireCloseTensor(a, Tensor({1, 2}), "shape"); }, "shape actual=");
    Tensor broken = a; broken.data.pop_back();
    RequireError([&] { RequireCloseTensor(broken, a, "storage"); }, "storage");
    if (!CudaRuntimeBuilt()) {
        RequireError([] { (void)DeviceCount(); }, "not built");
        RequireError([] { SetCudaService(0); }, "not built");
        RequireError([] { CudaDeviceBuffer b(16); }, "not built");
        CudaDeviceBuffer empty; empty.ReSet(); Require(empty.IsEmpty(), "empty buffer");
        return;
    }
    const auto info = GetCudaInfo(0);
    Require(!info.name.empty() && info.total_bytes > 0 && info.compute_major > 0 &&
            info.runtime_version > 0 && info.driver_version > 0, "CUDA info");
    RequireError([] { (void)GetCudaInfo(DeviceCount()); }, "not available");
    Tensor input = Values({37}), output({37});
    CudaDeviceBuffer a_dev(input.GetDataSize()*sizeof(float));
    a_dev.upload(input.data.data(), a_dev.GetByteSize());
    CudaDeviceBuffer moved(std::move(a_dev));
    Require(a_dev.IsEmpty() && moved.GetDeviceId() == 0, "move ownership");
    CudaDeviceBuffer destination(moved.GetByteSize());
    CudaMemcpy(destination.GetData(), moved.GetData(), moved.GetByteSize(), CudaMemoryKind::kDeviceDevice);
    destination.download(output.data.data(), destination.GetByteSize());
    RequireCloseTensor(output, input, "buffer roundtrip");
    RequireError([&] { moved.upload(input.data.data(), moved.GetByteSize()+1); }, "exceeds");
    moved.ReSet(); destination.ReSet();
    Require(moved.IsEmpty() && destination.IsEmpty(), "explicit release");
}
int main(int argc, char** argv) { return Run(argc, argv, {{"runtime", Runtime}}); }
