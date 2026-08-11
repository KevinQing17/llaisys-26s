#pragma once

#include "llaisys.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

inline cudaStream_t toCudaStream(llaisysStream_t stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

inline void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
__device__ __forceinline__ float toFloat(T value);

template <>
__device__ __forceinline__ float toFloat<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

} // namespace llaisys::device::nvidia
