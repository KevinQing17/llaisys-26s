// CoreX does not ship NVIDIA's separate device runtime library. Xmake 2.8
// links it unconditionally for CUDA sources even when RDC is disabled.
extern "C" void llaisysCorexCudaDeviceRuntimeStub() {}
