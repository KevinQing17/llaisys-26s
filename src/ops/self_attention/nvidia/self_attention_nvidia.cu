#include "self_attention_nvidia.cuh"

#include "../../../device/nvidia/cuda_utils.cuh"
#include "../../../utils.hpp"

#include <cmath>

namespace {

constexpr unsigned int ATTENTION_THREADS = 256;
constexpr unsigned int MAX_ATTENTION_BLOCKS = 65535;

template <typename T, unsigned int BLOCK_SIZE>
__global__ void selfAttentionKernel(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {
    // Shared-memory use is independent of the KV sequence length.
    __shared__ float scores[BLOCK_SIZE];
    __shared__ float reduction[BLOCK_SIZE];

    const size_t value_tiles = value_dim / BLOCK_SIZE
                             + static_cast<size_t>(value_dim % BLOCK_SIZE != 0);
    const size_t total_work_items = q_len * num_heads * value_tiles;

    for (size_t work_item = blockIdx.x;
         work_item < total_work_items;
         work_item += gridDim.x) {
        const size_t query_head = work_item / value_tiles;
        const size_t value_tile = work_item % value_tiles;
        const size_t query_index = query_head / num_heads;
        const size_t head = query_head % num_heads;
        const size_t heads_per_kv = num_heads / num_kv_heads;
        const size_t kv_head = head / heads_per_kv;
        const size_t past_len = kv_len - q_len;
        const size_t allowed_kv = past_len + query_index + 1;
        const size_t q_base = query_head * qk_dim;
        const size_t value_index = value_tile * BLOCK_SIZE + threadIdx.x;

        float running_max = -INFINITY;
        float running_sum = 0.0f;
        float result = 0.0f;

        for (size_t key_base = 0; key_base < allowed_kv; key_base += BLOCK_SIZE) {
            const size_t key_index = key_base + threadIdx.x;
            float score = -INFINITY;
            if (key_index < allowed_kv) {
                const size_t k_base =
                    (key_index * num_kv_heads + kv_head) * qk_dim;
                float dot = 0.0f;
                for (size_t dim = 0; dim < qk_dim; ++dim) {
                    dot += llaisys::device::nvidia::toFloat(q[q_base + dim])
                         * llaisys::device::nvidia::toFloat(k[k_base + dim]);
                }
                score = dot * scale;
            }
            scores[threadIdx.x] = score;
            reduction[threadIdx.x] = score;
            __syncthreads();

            for (unsigned int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
                if (threadIdx.x < offset) {
                    reduction[threadIdx.x] = fmaxf(
                        reduction[threadIdx.x], reduction[threadIdx.x + offset]);
                }
                __syncthreads();
            }

            const float tile_max = reduction[0];
            const float new_max = fmaxf(running_max, tile_max);
            const float previous_scale = running_sum == 0.0f
                ? 0.0f : expf(running_max - new_max);
            const float weight = key_index < allowed_kv
                ? expf(scores[threadIdx.x] - new_max) : 0.0f;
            __syncthreads();
            scores[threadIdx.x] = weight;
            reduction[threadIdx.x] = weight;
            __syncthreads();

            for (unsigned int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
                if (threadIdx.x < offset) {
                    reduction[threadIdx.x] += reduction[threadIdx.x + offset];
                }
                __syncthreads();
            }

            float tile_result = 0.0f;
            if (value_index < value_dim) {
                const size_t tile_size = allowed_kv - key_base < BLOCK_SIZE
                    ? allowed_kv - key_base : BLOCK_SIZE;
                for (size_t tile_key = 0; tile_key < tile_size; ++tile_key) {
                    const size_t v_offset =
                        ((key_base + tile_key) * num_kv_heads + kv_head)
                        * value_dim + value_index;
                    tile_result += scores[tile_key]
                                 * llaisys::device::nvidia::toFloat(v[v_offset]);
                }
            }
            result = result * previous_scale + tile_result;
            running_sum = running_sum * previous_scale + reduction[0];
            running_max = new_max;
            __syncthreads();
        }

        if (value_index < value_dim) {
            const size_t out_offset = query_head * value_dim + value_index;
            attn_val[out_offset] = llaisys::device::nvidia::fromFloat<T>(
                result / running_sum);
        }
        __syncthreads();
    }
}

template <typename T>
void launchSelfAttention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale,
    cudaStream_t stream) {
    const size_t value_tiles = value_dim / ATTENTION_THREADS
                             + static_cast<size_t>(value_dim % ATTENTION_THREADS != 0);
    const size_t total_work_items = q_len * num_heads * value_tiles;
    const auto blocks = static_cast<unsigned int>(
        total_work_items < MAX_ATTENTION_BLOCKS
            ? total_work_items : MAX_ATTENTION_BLOCKS);
    selfAttentionKernel<T, ATTENTION_THREADS>
        <<<blocks, ATTENTION_THREADS, 0, stream>>>(
            reinterpret_cast<T *>(attn_val),
            reinterpret_cast<const T *>(q),
            reinterpret_cast<const T *>(k),
            reinterpret_cast<const T *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);
    llaisys::device::nvidia::checkCuda(
        cudaGetLastError(), "Self-attention kernel launch failed");
}

} // namespace

namespace llaisys::ops::nvidia {

void selfAttention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale,
    llaisysStream_t stream) {
    const cudaStream_t cuda_stream = device::nvidia::toCudaStream(stream);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return launchSelfAttention<float>(
            attn_val, q, k, v, q_len, kv_len, num_heads, num_kv_heads,
            qk_dim, value_dim, scale, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchSelfAttention<__half>(
            attn_val, q, k, v, q_len, kv_len, num_heads, num_kv_heads,
            qk_dim, value_dim, scale, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchSelfAttention<__nv_bfloat16>(
            attn_val, q, k, v, q_len, kv_len, num_heads, num_kv_heads,
            qk_dim, value_dim, scale, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
