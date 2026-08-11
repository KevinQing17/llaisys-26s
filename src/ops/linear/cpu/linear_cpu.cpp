#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(
    T *out,
    const T *in,
    const T *weight,
    const T *bias,
    size_t m,
    size_t n,
    size_t k) {

    // i：输入的第几行。
    for (size_t i = 0; i < m; i++) {
        // j：输出的第几个特征。
        for (size_t j = 0; j < n; j++) {
            // 有 bias 时从 bias[j] 开始；
            // 没有 bias 时从 0 开始。
            float sum = bias == nullptr
                          ? 0.0f
                          : llaisys::utils::cast<float>(bias[j]);

            // 计算输入第 i 行与 weight 第 j 行的点积。
            for (size_t p = 0; p < k; p++) {
                float in_value = llaisys::utils::cast<float>(
                    in[i * k + p]);

                float weight_value = llaisys::utils::cast<float>(
                    weight[j * k + p]);

                sum += in_value * weight_value;
            }

            out[i * n + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {

void linear(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    llaisysDataType_t type,
    size_t m,
    size_t n,
    size_t k) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            bias == nullptr
                ? nullptr
                : reinterpret_cast<const float *>(bias),
            m,
            n,
            k);

    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            bias == nullptr
                ? nullptr
                : reinterpret_cast<const llaisys::fp16_t *>(bias),
            m,
            n,
            k);

    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            bias == nullptr
                ? nullptr
                : reinterpret_cast<const llaisys::bf16_t *>(bias),
            m,
            n,
            k);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
