#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstdint>

template <typename T>
void embedding_(
    T *out,
    const int64_t *index,
    const T *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim) {

    for (size_t i = 0; i < num_indices; i++) {
        int64_t row = index[i];
        CHECK_ARGUMENT(
            row >= 0 && static_cast<size_t>(row) < num_embeddings,
            "Embedding: index is out of range.");

        for (size_t j = 0; j < embedding_dim; j++) {
            out[i * embedding_dim + j] = weight[static_cast<size_t>(row) * embedding_dim + j];
        }
    }
}

namespace llaisys::ops::cpu {

void embedding(
    std::byte *out,
    const std::byte *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t num_embeddings,
    size_t embedding_dim) {

    const auto *index_ptr = reinterpret_cast<const int64_t *>(index);
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(
            reinterpret_cast<float *>(out),
            index_ptr,
            reinterpret_cast<const float *>(weight),
            num_indices,
            num_embeddings,
            embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            index_ptr,
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            num_indices,
            num_embeddings,
            embedding_dim);

    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            index_ptr,
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            num_indices,
            num_embeddings,
            embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
