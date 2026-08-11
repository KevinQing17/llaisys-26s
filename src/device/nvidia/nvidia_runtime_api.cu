#include "../runtime_api.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

namespace runtime_api {

namespace {

void checkCuda(cudaError_t status, const char *operation) {
    if (status == cudaSuccess) {
        return;
    }

    const char *description = cudaGetErrorString(status);
    std::cerr << "[ERROR] " << operation << " failed: " << description << std::endl;
    throw std::runtime_error(
        std::string(operation) + " failed: " + description);
}

cudaMemcpyKind toCudaMemcpyKind(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("Invalid CUDA memcpy kind.");
    }
}

cudaStream_t toCudaStream(llaisysStream_t stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

} // namespace

int getDeviceCount() {
    int count = 0;
    checkCuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    return count;
}

void setDevice(int device_id) {
    checkCuda(cudaSetDevice(device_id), "cudaSetDevice");
}

void deviceSynchronize() {
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream == nullptr) {
        return;
    }
    checkCuda(cudaStreamDestroy(toCudaStream(stream)), "cudaStreamDestroy");
}

void streamSynchronize(llaisysStream_t stream) {
    checkCuda(cudaStreamSynchronize(toCudaStream(stream)), "cudaStreamSynchronize");
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    void *ptr = nullptr;
    checkCuda(cudaMalloc(&ptr, size), "cudaMalloc");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    checkCuda(cudaFree(ptr), "cudaFree");
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    void *ptr = nullptr;
    checkCuda(cudaMallocHost(&ptr, size), "cudaMallocHost");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr == nullptr) {
        return;
    }
    checkCuda(cudaFreeHost(ptr), "cudaFreeHost");
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    if (size == 0) {
        return;
    }
    checkCuda(
        cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)),
        "cudaMemcpy");
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {
    if (size == 0) {
        return;
    }
    checkCuda(
        cudaMemcpyAsync(
            dst,
            src,
            size,
            toCudaMemcpyKind(kind),
            toCudaStream(stream)),
        "cudaMemcpyAsync");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
