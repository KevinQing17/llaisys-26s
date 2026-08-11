#include "swiglu_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
__global__ void swigluKernel(T *out, const T *gate, const T *up, size_t numel) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < numel;
         index += stride) {
        const float gate_value = llaisys::device::nvidia::toFloat(gate[index]);
        const float up_value = llaisys::device::nvidia::toFloat(up[index]);
        const float silu = gate_value / (1.0f + expf(-gate_value));
        out[index] = llaisys::device::nvidia::fromFloat<T>(up_value * silu);
    }
}

template <typename T>
void launchSwiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    size_t numel,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    constexpr size_t max_blocks = 65535;
    const size_t required_blocks = (numel - 1) / threads + 1;
    const auto blocks = static_cast<unsigned int>(
        required_blocks < max_blocks ? required_blocks : max_blocks);
    swigluKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
    llaisys::device::nvidia::checkCuda(
        cudaGetLastError(), "SwiGLU kernel launch failed");
}

} // namespace

namespace llaisys::ops::nvidia {

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }

    const cudaStream_t cuda_stream = device::nvidia::toCudaStream(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSwiglu<float>(out, gate, up, numel, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchSwiglu<__half>(out, gate, up, numel, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchSwiglu<__nv_bfloat16>(out, gate, up, numel, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
