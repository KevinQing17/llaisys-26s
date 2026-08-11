#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_ARGUMENT(out != nullptr && in != nullptr,
                   "Rearrange: input and output tensors must not be null.");
    CHECK_SAME_DEVICE(out, in);
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(out->isContiguous(),
                   "Rearrange: output tensor must be contiguous.");

    const size_t bytes = out->numel() * out->elementSize();
    if (bytes == 0 || out.get() == in.get()) {
        return;
    }

    // Tensor::contiguous performs the stride-aware packing for a view on both
    // CPU and accelerators.  Once packed, the final copy is a single D2D
    // transfer (D2D also denotes host-to-host for the CPU runtime).
    auto source = in->isContiguous() ? in : in->contiguous();
    core::context().setDevice(out->deviceType(), out->deviceId());
    core::context().runtime().api()->memcpy_sync(
        out->data(),
        source->data(),
        bytes,
        LLAISYS_MEMCPY_D2D);
}
} // namespace llaisys::ops
