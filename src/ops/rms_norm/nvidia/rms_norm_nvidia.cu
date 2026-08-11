#include "rms_norm_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>

namespace {

template <typename T>
__global__ void rmsNormKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t hidden_size,
    float eps) {
    __shared__ float sums[256];
    for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const size_t row_offset = row * hidden_size;

        float sum_square = 0.0f;
        for (size_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
            const float value = llaisys::device::nvidia::toFloat(in[row_offset + column]);
            sum_square += value * value;
        }
        sums[threadIdx.x] = sum_square;
        __syncthreads();

        for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
            if (threadIdx.x < offset) {
                sums[threadIdx.x] += sums[threadIdx.x + offset];
            }
            __syncthreads();
        }

        const float inv_rms = rsqrtf(sums[0] / static_cast<float>(hidden_size) + eps);
        for (size_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
            const float input_value = llaisys::device::nvidia::toFloat(in[row_offset + column]);
            const float weight_value = llaisys::device::nvidia::toFloat(weight[column]);
            out[row_offset + column] = llaisys::device::nvidia::fromFloat<T>(
                input_value * inv_rms * weight_value);
        }
        __syncthreads();
    }
}

template <typename T>
void launchRmsNorm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t rows,
    size_t hidden_size,
    float eps,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    constexpr size_t max_blocks = 65535;
    const auto blocks = static_cast<unsigned int>(rows < max_blocks ? rows : max_blocks);
    rmsNormKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        rows,
        hidden_size,
        eps);
    llaisys::device::nvidia::checkCuda(
        cudaGetLastError(), "RMSNorm kernel launch failed");
}

} // namespace

namespace llaisys::ops::nvidia {

void rmsNorm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t rows,
    size_t hidden_size,
    float eps,
    llaisysStream_t stream) {
    if (rows == 0) {
        return;
    }

    const cudaStream_t cuda_stream = device::nvidia::toCudaStream(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(
            out, in, weight, rows, hidden_size, eps, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(
            out, in, weight, rows, hidden_size, eps, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__nv_bfloat16>(
            out, in, weight, rows, hidden_size, eps, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
