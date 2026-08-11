#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

namespace llaisys::ops {

void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 三个张量必须位于同一设备。
    CHECK_SAME_DEVICE(out, gate, up);

    // 三个张量必须具有相同形状。
    CHECK_SAME_SHAPE(
        out->shape(),
        gate->shape(),
        up->shape());

    // 三个张量必须具有相同数据类型。
    CHECK_SAME_DTYPE(
        out->dtype(),
        gate->dtype(),
        up->dtype());

    // 当前作业要求输入输出为二维张量。
    CHECK_ARGUMENT(
        out->ndim() == 2,
        "SwiGLU: tensors must be 2D.");

    // CPU kernel 按一维连续数组处理。
    ASSERT(
        out->isContiguous() && gate->isContiguous() && up->isContiguous(),
        "SwiGLU: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            out->numel());
    }

    core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            out->dtype(),
            out->numel());

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(
            out->data(), gate->data(), up->data(), out->dtype(), out->numel(),
            core::context().runtime().stream());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
