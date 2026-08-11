#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rms_norm_nvidia.cuh"
#endif

namespace llaisys::ops {

void rms_norm(
    tensor_t out,
    tensor_t in,
    tensor_t weight,
    float eps) {

    // 三个张量必须在同一设备。
    CHECK_SAME_DEVICE(out, in, weight);

    // 输入和输出都是二维张量。
    CHECK_ARGUMENT(
        in->ndim() == 2,
        "RMS Norm: input must be a 2D tensor.");

    CHECK_ARGUMENT(
        out->ndim() == 2,
        "RMS Norm: output must be a 2D tensor.");

    // weight 是一维张量。
    CHECK_ARGUMENT(
        weight->ndim() == 1,
        "RMS Norm: weight must be a 1D tensor.");

    // 输入和输出形状必须完全相同。
    CHECK_SAME_SHAPE(
        out->shape(),
        in->shape());

    // weight 长度等于每行长度。
    CHECK_ARGUMENT(
        weight->shape()[0] == in->shape()[1],
        "RMS Norm: weight length must match input row length.");

    // 每行必须至少包含一个元素，防止除以零。
    CHECK_ARGUMENT(
        in->shape()[1] > 0,
        "RMS Norm: input row length must be greater than zero.");

    // eps 不应为负数。
    CHECK_ARGUMENT(
        eps >= 0.0f,
        "RMS Norm: eps must not be negative.");

    // 三个张量的数据类型必须相同。
    CHECK_SAME_DTYPE(
        out->dtype(),
        in->dtype(),
        weight->dtype());

    // CPU kernel 使用连续内存。
    ASSERT(
        out->isContiguous() && in->isContiguous() && weight->isContiguous(),
        "RMS Norm: all tensors must be contiguous.");

    size_t rows = in->shape()[0];
    size_t hidden_size = in->shape()[1];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            rows,
            hidden_size,
            eps);
    }

    core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            out->dtype(),
            rows,
            hidden_size,
            eps);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rmsNorm(
            out->data(), in->data(), weight->data(), out->dtype(), rows, hidden_size,
            eps, core::context().runtime().stream());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
