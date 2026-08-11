#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(
        index->ndim() == 1,
        "Embedding: index must be a 1D tensor.");

    CHECK_ARGUMENT(
        weight->ndim() == 2,
        "Embedding: weight must be a 2D tensor.");
    CHECK_ARGUMENT(
        out->ndim() == 2,
        "Embedding: out must be a 2D tensor. ");
    CHECK_ARGUMENT(
        index->dtype() == LLAISYS_DTYPE_I64,
        "Embedding: index must have Int64 dtype.");
    // 输出元素和 weight 元素的数据类型必须相同。
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());

    // 每个输入索引产生一行输出。
    CHECK_ARGUMENT(
        out->shape()[0] == index->shape()[0],
        "Embedding: out first dimension must match index length.");

    // 输出每行的长度必须等于 weight 每行的长度。
    CHECK_ARGUMENT(
        out->shape()[1] == weight->shape()[1],
        "Embedding: out second dimension must match embedding dimension.");
    // CPU kernel 使用一维下标访问连续内存。
    ASSERT(
        out->isContiguous() && index->isContiguous() && weight->isContiguous(),
        "Embedding: all tensors must be contiguous.");
    size_t num_indices = index->shape()[0];
    size_t num_embeddings = weight->shape()[0];
    size_t embedding_dim = weight->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            num_indices,
            num_embeddings,
            embedding_dim

        );
    }
    core::context().setDevice(
        out->deviceType(),
        out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(
            out->data(),
            index->data(),
            weight->data(),
            out->dtype(),
            num_indices,
            num_embeddings,
            embedding_dim);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(
            out->data(), index->data(), weight->data(), out->dtype(),
            num_indices, num_embeddings, embedding_dim,
            core::context().runtime().stream());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
