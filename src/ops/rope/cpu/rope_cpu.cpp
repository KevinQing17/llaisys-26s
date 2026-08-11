#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>

template <typename T>
void rope_(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta) {

    size_t half_dim = head_dim / 2;

    // 每个 seq 对应一个 token 位置。
    for (size_t seq = 0; seq < seq_len; seq++) {
        int64_t position = pos_ids[seq];

        CHECK_ARGUMENT(
            position >= 0,
            "RoPE: position id must not be negative.");

        // 前后两半中，每个 j 组成一对。
        for (size_t j = 0; j < half_dim; j++) {
            float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);

            float angle = static_cast<float>(position) / std::pow(theta, exponent);

            float sin_value = std::sin(angle);
            float cos_value = std::cos(angle);

            // 同一位置和维度的旋转角度由所有 head 共享。
            for (size_t head = 0; head < num_heads; head++) {
                size_t base = (seq * num_heads + head) * head_dim;

                size_t a_index = base + j;
                size_t b_index = base + half_dim + j;

                float a = llaisys::utils::cast<float>(
                    in[a_index]);

                float b = llaisys::utils::cast<float>(
                    in[b_index]);

                float rotated_a = a * cos_value - b * sin_value;

                float rotated_b = b * cos_value + a * sin_value;

                out[a_index] = llaisys::utils::cast<T>(rotated_a);

                out[b_index] = llaisys::utils::cast<T>(rotated_b);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void rope(
    std::byte *out,
    const std::byte *in,
    const std::byte *pos_ids,
    llaisysDataType_t type,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta) {

    const auto *position_ptr = reinterpret_cast<const int64_t *>(pos_ids);

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            position_ptr,
            seq_len,
            num_heads,
            head_dim,
            theta);

    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            position_ptr,
            seq_len,
            num_heads,
            head_dim,
            theta);

    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            position_ptr,
            seq_len,
            num_heads,
            head_dim,
            theta);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
