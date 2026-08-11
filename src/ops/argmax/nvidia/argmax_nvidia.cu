#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

namespace {

__device__ bool isBetter(float candidate, size_t candidate_idx, float best, size_t best_idx) {
    const bool candidate_nan = isnan(candidate);
    const bool best_nan = isnan(best);
    if (candidate_nan != best_nan) {
        return candidate_nan;
    }
    if (candidate_nan) {
        return candidate_idx < best_idx;
    }
    return candidate > best || (candidate == best && candidate_idx < best_idx);
}

template <typename T>
__global__ void argmaxKernel(int64_t *max_idx, T *max_val, const T *vals, size_t numel) {
    __shared__ float shared_values[256];
    __shared__ size_t shared_indices[256];

    size_t best_idx = 0;
    float best = llaisys::device::nvidia::toFloat(vals[0]);
    for (size_t index = threadIdx.x; index < numel; index += blockDim.x) {
        const float candidate = llaisys::device::nvidia::toFloat(vals[index]);
        if (isBetter(candidate, index, best, best_idx)) {
            best = candidate;
            best_idx = index;
        }
    }

    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_idx;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const float candidate = shared_values[threadIdx.x + offset];
            const size_t candidate_idx = shared_indices[threadIdx.x + offset];
            if (isBetter(
                    candidate,
                    candidate_idx,
                    shared_values[threadIdx.x],
                    shared_indices[threadIdx.x])) {
                shared_values[threadIdx.x] = candidate;
                shared_indices[threadIdx.x] = candidate_idx;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        max_idx[0] = static_cast<int64_t>(shared_indices[0]);
        max_val[0] = vals[shared_indices[0]];
    }
}

template <typename T>
void launchArgmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    size_t numel,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    argmaxKernel<<<1, threads, 0, stream>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);
    llaisys::device::nvidia::checkCuda(
        cudaGetLastError(), "Argmax kernel launch failed");
}

} // namespace

namespace llaisys::ops::nvidia {

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t type,
    size_t numel,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = device::nvidia::toCudaStream(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(max_idx, max_val, vals, numel, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(max_idx, max_val, vals, numel, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__nv_bfloat16>(max_idx, max_val, vals, numel, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
