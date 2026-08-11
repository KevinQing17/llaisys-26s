#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    ASSERT(
        max_idx->dtype() == LLAISYS_DTYPE_I64,
        "Argmax: max_idx must have Int64 dtype.");

    // 当前作业只要求处理一维输入。
    ASSERT(
        vals->ndim() == 1,
        "Argmax: vals must be a 1D tensor.");

    // 输入不能为空，因为 CPU kernel 会读取 vals[0]。
    ASSERT(
        vals->numel() > 0,
        "Argmax: vals must not be empty.");
    // 两个输出都必须是一维、单元素张量。
    ASSERT(
        max_idx->ndim() == 1 && max_idx->shape()[0] == 1,
        "Argmax: max_idx must have shape (1,).");

    ASSERT(
        max_val->ndim() == 1 && max_val->shape()[0] == 1,
        "Argmax: max_val must have shape (1,).");
    // CPU kernel 按连续数组访问内存。
    ASSERT(
        max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
        "Argmax: all tensors must be contiguous.");
    // CPU 张量调用 CPU 实现。
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel());
    }

    core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            vals->dtype(),
            vals->numel());

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(
            max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel(),
            core::context().runtime().stream());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
