#include "add_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

namespace {

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < numel;
         index += stride) {
        const float value = llaisys::device::nvidia::toFloat(a[index])
                          + llaisys::device::nvidia::toFloat(b[index]);
        c[index] = llaisys::device::nvidia::fromFloat<T>(value);
    }
}

template <typename T>
void launchAdd(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    size_t numel,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    constexpr size_t max_blocks = 65535;
    const size_t required_blocks = (numel - 1) / threads + 1;
    const auto blocks = static_cast<unsigned int>(
        required_blocks < max_blocks ? required_blocks : max_blocks);
    addKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);
    llaisys::device::nvidia::checkCuda(cudaGetLastError(), "Add kernel launch failed");
}

} // namespace

namespace llaisys::ops::nvidia {

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }

    const cudaStream_t cuda_stream = device::nvidia::toCudaStream(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchAdd<float>(c, a, b, numel, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchAdd<__half>(c, a, b, numel, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchAdd<__nv_bfloat16>(c, a, b, numel, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
