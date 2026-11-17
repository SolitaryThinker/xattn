// Present only so the refusal probe is built as a CUDA extension.
//
// The preparation header reads the device's properties through ATen's CUDA
// context (at::cuda::getDeviceProperties), whose symbol lives in torch_cuda;
// torch's builder links that library for an extension that has at least one
// CUDA source and not otherwise. Without this translation unit the probe
// compiles and then fails at link, which looks like a probe bug and is not one.
//
// Nothing launches this kernel and nothing includes it. It exists so the
// linker sees a CUDA unit; the case table stays in prepare_refusal_check.cc.
__global__ void PrepareRefusalCheckStub() {}
