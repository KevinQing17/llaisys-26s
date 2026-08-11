#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

namespace llaisys::ops {

void rope(
    tensor_t out,
    tensor_t in,
    tensor_t pos_ids,
    float theta) {

    // 三个张量必须位于同一设备。
    CHECK_SAME_DEVICE(out, in, pos_ids);

    // 输入和输出形状为：
    // [seq_len, num_heads, head_dim]
    CHECK_ARGUMENT(
        in->ndim() == 3,
        "RoPE: input must be a 3D tensor.");

    CHECK_ARGUMENT(
        out->ndim() == 3,
        "RoPE: output must be a 3D tensor.");

    // 输入输出形状必须完全一致。
    CHECK_SAME_SHAPE(
        out->shape(),
        in->shape());

    // 输入输出 dtype 必须一致。
    CHECK_SAME_DTYPE(
        out->dtype(),
        in->dtype());

    // 每个 token 对应一个位置编号。
    CHECK_ARGUMENT(
        pos_ids->ndim() == 1,
        "RoPE: pos_ids must be a 1D tensor.");

    CHECK_ARGUMENT(
        pos_ids->shape()[0] == in->shape()[0],
        "RoPE: pos_ids length must match sequence length.");

    CHECK_ARGUMENT(
        pos_ids->dtype() == LLAISYS_DTYPE_I64,
        "RoPE: pos_ids must have Int64 dtype.");

    // head_dim 必须能分成相同的前后两半。
    CHECK_ARGUMENT(
        in->shape()[2] > 0 && in->shape()[2] % 2 == 0,
        "RoPE: head dimension must be positive and even.");

    CHECK_ARGUMENT(
        theta > 0.0f,
        "RoPE: theta must be positive.");

    ASSERT(
        out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
        "RoPE: all tensors must be contiguous.");

    size_t seq_len = in->shape()[0];
    size_t num_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            out->dtype(),
            seq_len,
            num_heads,
            head_dim,
            theta);
    }

    core::context().setDevice(
        out->deviceType(),
        out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(),
            in->data(),
            pos_ids->data(),
            out->dtype(),
            seq_len,
            num_heads,
            head_dim,
            theta);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(
            out->data(), in->data(), pos_ids->data(), out->dtype(), seq_len, num_heads,
            head_dim, theta, core::context().runtime().stream());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
