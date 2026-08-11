#include "rope_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

namespace {

template <typename T>
__global__ void ropeKernel(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t num_pairs,
    size_t num_heads,
    size_t head_dim,
    float theta) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t pair_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair_index < num_pairs;
         pair_index += stride) {
        const size_t half_dim = head_dim / 2;
        const size_t pair_dim = pair_index % half_dim;
        const size_t token_head = pair_index / half_dim;
        const size_t seq = token_head / num_heads;
        const size_t base = token_head * head_dim;
        const size_t a_index = base + pair_dim;
        const size_t b_index = a_index + half_dim;

        const float exponent = 2.0f * static_cast<float>(pair_dim) / static_cast<float>(head_dim);
        const float angle = static_cast<float>(pos_ids[seq]) / powf(theta, exponent);
        float sin_value = 0.0f;
        float cos_value = 0.0f;
        sincosf(angle, &sin_value, &cos_value);

        const float a = llaisys::device::nvidia::toFloat(in[a_index]);
        const float b = llaisys::device::nvidia::toFloat(in[b_index]);
        out[a_index] = llaisys::device::nvidia::fromFloat<T>(
            a * cos_value - b * sin_value);
        out[b_index] = llaisys::device::nvidia::fromFloat<T>(
            b * cos_value + a * sin_value);
    }
}

template <typename T>
void launchRope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta,
    cudaStream_t stream) {
    const size_t num_pairs = seq_len * num_heads * (head_dim / 2);
    constexpr unsigned int threads = 256;
    constexpr size_t max_blocks = 65535;
    const size_t required_blocks = (num_pairs - 1) / threads + 1;
    const auto blocks = static_cast<unsigned int>(
        required_blocks < max_blocks ? required_blocks : max_blocks);
    ropeKernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(pos_ids),
        num_pairs,
        num_heads,
        head_dim,
        theta);
    llaisys::device::nvidia::checkCuda(cudaGetLastError(), "RoPE kernel launch failed");
}

} // namespace

namespace llaisys::ops::nvidia {

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta,
    llaisysStream_t stream) {
    if (seq_len == 0 || num_heads == 0) {
        return;
    }

    const cudaStream_t cuda_stream = device::nvidia::toCudaStream(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchRope<float>(
            out, in, pos_ids, seq_len, num_heads, head_dim, theta, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchRope<__half>(
            out, in, pos_ids, seq_len, num_heads, head_dim, theta, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchRope<__nv_bfloat16>(
            out, in, pos_ids, seq_len, num_heads, head_dim, theta, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
