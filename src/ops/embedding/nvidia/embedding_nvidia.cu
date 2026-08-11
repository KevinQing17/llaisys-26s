#include "embedding_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cstdint>

namespace {

template <typename T>
__global__ void embeddingKernel(
    T *out,
    const int64_t *index,
    const T *weight,
    size_t numel,
    size_t num_embeddings,
    size_t embedding_dim) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         output_index < numel;
         output_index += stride) {
        const size_t index_offset = output_index / embedding_dim;
        const size_t column = output_index % embedding_dim;
        const int64_t row = index[index_offset];
        if (row < 0 || static_cast<size_t>(row) >= num_embeddings) {
            out[output_index] = T{};
        } else {
            out[output_index] = weight[static_cast<size_t>(row) * embedding_dim + column];
        }
    }
}

template <typename T>
void launchEmbedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim,
    cudaStream_t stream) {
    const size_t numel = num_indices * embedding_dim;
    constexpr unsigned int threads = 256;
    constexpr size_t max_blocks = 65535;
    const size_t required_blocks = (numel - 1) / threads + 1;
    const auto blocks = static_cast<unsigned int>(
        required_blocks < max_blocks ? required_blocks : max_blocks);
    embeddingKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const int64_t *>(index),
        reinterpret_cast<const T *>(weight),
        numel,
        num_embeddings,
        embedding_dim);
    llaisys::device::nvidia::checkCuda(
        cudaGetLastError(), "Embedding kernel launch failed");
}

} // namespace

namespace llaisys::ops::nvidia {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim,
    llaisysStream_t stream) {
    if (num_indices == 0 || embedding_dim == 0) {
        return;
    }

    const cudaStream_t cuda_stream = device::nvidia::toCudaStream(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchEmbedding<float>(
            out, index, weight, num_indices, num_embeddings, embedding_dim, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchEmbedding<__half>(
            out, index, weight, num_indices, num_embeddings, embedding_dim, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchEmbedding<__nv_bfloat16>(
            out, index, weight, num_indices, num_embeddings, embedding_dim, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
