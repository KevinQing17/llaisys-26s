#include "linear_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cublas_v2.h>

#include <climits>
#include <stdexcept>
#include <string>

namespace {

void checkCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status "
            + std::to_string(static_cast<int>(status)));
    }
}

class CublasHandle {
private:
    cublasHandle_t _handle = nullptr;
    int _device = -1;

public:
    ~CublasHandle() {
        if (_handle != nullptr) {
            cublasDestroy(_handle);
        }
    }

    cublasHandle_t get() {
        int current_device = 0;
        llaisys::device::nvidia::checkCuda(cudaGetDevice(&current_device), "cudaGetDevice failed");
        if (_handle != nullptr && current_device != _device) {
            checkCublas(cublasDestroy(_handle), "cublasDestroy");
            _handle = nullptr;
        }
        if (_handle == nullptr) {
            checkCublas(cublasCreate(&_handle), "cublasCreate");
            _device = current_device;
        }
        return _handle;
    }
};

thread_local CublasHandle cublas_handle;

template <typename T>
__global__ void fillBiasKernel(T *out, const T *bias, size_t numel, size_t n) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; index < numel; index += stride) {
        out[index] = bias[index % n];
    }
}

template <typename T>
void launchBias(
    std::byte *out,
    const std::byte *bias,
    size_t numel,
    size_t n,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    constexpr size_t max_blocks = 65535;
    const size_t required_blocks = (numel + threads - 1) / threads;
    const auto blocks = static_cast<unsigned int>(
        required_blocks < max_blocks ? required_blocks : max_blocks);
    fillBiasKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        numel,
        n);
    llaisys::device::nvidia::checkCuda(
        cudaGetLastError(), "Linear bias fill kernel launch failed");
}

cudaDataType_t cudaType(llaisysDataType_t type) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

void fillBias(
    std::byte *out,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t numel,
    size_t n,
    cudaStream_t stream) {
    if (bias == nullptr) {
        return;
    }
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchBias<float>(out, bias, numel, n, stream);
    case LLAISYS_DTYPE_F16:
        return launchBias<__half>(out, bias, numel, n, stream);
    case LLAISYS_DTYPE_BF16:
        return launchBias<__nv_bfloat16>(out, bias, numel, n, stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k,
    llaisysStream_t stream) {
    if (m == 0 || n == 0) {
        return;
    }
    CHECK_ARGUMENT(
        m <= INT_MAX && n <= INT_MAX && k <= INT_MAX,
        "Linear: dimensions exceed cuBLAS limits.");

    const cudaStream_t cuda_stream = device::nvidia::toCudaStream(stream);
    const cudaDataType_t data_type = cudaType(type);
    const size_t numel = m * n;

    // Seed C with the broadcast bias so cuBLAS combines the bias and dot product
    // in the FP32 accumulator. Adding bias in a second low-precision kernel can
    // overflow when the dot product is outside F16 even if the final result is not.
    if (bias != nullptr) {
        fillBias(out, bias, type, numel, n, cuda_stream);
    } else if (k == 0) {
        llaisys::device::nvidia::checkCuda(
            cudaMemsetAsync(out, 0, numel * utils::dsize(type), cuda_stream),
            "Linear zero fill failed");
    }

    // cuBLAS need not accept null/zero-sized input pointers. For K == 0 the
    // prefill above is already the complete linear result.
    if (k == 0) {
        return;
    }

    cublasHandle_t handle = cublas_handle.get();
    checkCublas(cublasSetStream(handle, cuda_stream), "cublasSetStream");

    const float alpha = 1.0f;
    const float beta = bias == nullptr ? 0.0f : 1.0f;
#if CUBLAS_VERSION >= 11000
    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
#else
    const cudaDataType_t compute_type = CUDA_R_32F;
#endif
    checkCublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(n),
            static_cast<int>(m),
            static_cast<int>(k),
            &alpha,
            weight,
            data_type,
            static_cast<int>(k),
            in,
            data_type,
            static_cast<int>(k),
            &beta,
            out,
            data_type,
            static_cast<int>(n),
            compute_type,
            CUBLAS_GEMM_DEFAULT),
        "cublasGemmEx");
}

} // namespace llaisys::ops::nvidia
