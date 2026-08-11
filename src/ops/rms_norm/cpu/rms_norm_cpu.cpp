#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t hidden_size,
    float eps) {

    // 每一行独立进行归一化。
    for (size_t row = 0; row < rows; row++) {
        float sum_square = 0.0f;

        // 第一遍：计算这一行的平方和。
        for (size_t col = 0; col < hidden_size; col++) {
            float value = llaisys::utils::cast<float>(
                in[row * hidden_size + col]);

            sum_square += value * value;
        }

        // mean_square = 平方和 / 每行元素数量
        float mean_square = sum_square / static_cast<float>(hidden_size);

        // 使用倒数后，后面可以用乘法代替每个元素的除法。
        float inv_rms = 1.0f / std::sqrt(mean_square + eps);

        // 第二遍：归一化并乘可学习权重。
        for (size_t col = 0; col < hidden_size; col++) {
            float input_value = llaisys::utils::cast<float>(
                in[row * hidden_size + col]);

            float weight_value = llaisys::utils::cast<float>(weight[col]);

            float result = input_value * inv_rms * weight_value;

            out[row * hidden_size + col] = llaisys::utils::cast<T>(result);
        }
    }
}

namespace llaisys::ops::cpu {

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t rows,
    size_t hidden_size,
    float eps) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            rows,
            hidden_size,
            eps);

    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            rows,
            hidden_size,
            eps);

    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            rows,
            hidden_size,
            eps);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
