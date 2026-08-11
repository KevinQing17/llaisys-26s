#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.cuh"
#endif

namespace llaisys::ops {

void linear(
    tensor_t out,
    tensor_t in,
    tensor_t weight,
    tensor_t bias) {

    // 必须存在的三个张量需要位于同一设备。
    CHECK_SAME_DEVICE(out, in, weight);

    // 输入、权重、输出均为二维。
    CHECK_ARGUMENT(
        in->ndim() == 2,
        "Linear: input must be a 2D tensor.");

    CHECK_ARGUMENT(
        weight->ndim() == 2,
        "Linear: weight must be a 2D tensor.");

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "Linear: out must be a 2D tensor.");

    // 三个主要张量的数据类型必须相同。
    CHECK_SAME_DTYPE(
        out->dtype(),
        in->dtype(),
        weight->dtype());

    // X.shape = (M, K)
    // W.shape = (N, K)
    // 两者的 K 必须相同。
    CHECK_ARGUMENT(
        in->shape()[1] == weight->shape()[1],
        "Linear: input and weight dimensions do not match.");

    // out.shape = (M, N)
    CHECK_ARGUMENT(
        out->shape()[0] == in->shape()[0],
        "Linear: output row count must match input row count.");

    CHECK_ARGUMENT(
        out->shape()[1] == weight->shape()[0],
        "Linear: output column count must match weight row count.");

    ASSERT(
        out->isContiguous() && in->isContiguous() && weight->isContiguous(),
        "Linear: out, input and weight must be contiguous.");

    // bias 是可选的，因此只有它存在时才检查。
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);

        CHECK_SAME_DTYPE(
            out->dtype(),
            bias->dtype());

        CHECK_ARGUMENT(
            bias->ndim() == 1,
            "Linear: bias must be a 1D tensor.");

        CHECK_ARGUMENT(
            bias->shape()[0] == weight->shape()[0],
            "Linear: bias length must match output dimension.");

        ASSERT(
            bias->isContiguous(),
            "Linear: bias must be contiguous.");
    }

    size_t m = in->shape()[0];
    size_t k = in->shape()[1];
    size_t n = weight->shape()[0];

    // 没有 bias 时，向 CPU kernel 传递空指针。
    const std::byte *bias_data = bias == nullptr ? nullptr : bias->data();

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k);
    }

    core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            out->data(),
            in->data(),
            weight->data(),
            bias_data,
            out->dtype(),
            m,
            n,
            k);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(
            out->data(), in->data(), weight->data(), bias_data, out->dtype(), m, n, k,
            core::context().runtime().stream());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
