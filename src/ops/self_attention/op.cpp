#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.cuh"
#endif

namespace llaisys::ops {

void self_attention(
    tensor_t attn_val,
    tensor_t q,
    tensor_t k,
    tensor_t v,
    float scale) {

    CHECK_SAME_DEVICE(attn_val, q, k, v);

    // 当前作业中的四个张量都是三维。
    CHECK_ARGUMENT(
        q->ndim() == 3,
        "Self Attention: q must be a 3D tensor.");

    CHECK_ARGUMENT(
        k->ndim() == 3,
        "Self Attention: k must be a 3D tensor.");

    CHECK_ARGUMENT(
        v->ndim() == 3,
        "Self Attention: v must be a 3D tensor.");

    CHECK_ARGUMENT(
        attn_val->ndim() == 3,
        "Self Attention: output must be a 3D tensor.");

    CHECK_SAME_DTYPE(
        attn_val->dtype(),
        q->dtype(),
        k->dtype(),
        v->dtype());

    size_t q_len = q->shape()[0];
    size_t kv_len = k->shape()[0];
    size_t num_heads = q->shape()[1];
    size_t num_kv_heads = k->shape()[1];
    size_t qk_dim = q->shape()[2];
    size_t value_dim = v->shape()[2];

    // Q 和 K 做点积，因此最后一维必须相同。
    CHECK_ARGUMENT(
        k->shape()[2] == qk_dim,
        "Self Attention: q and k dimensions must match.");

    // K 和 V 必须拥有相同的序列长度和 KV Head 数量。
    CHECK_ARGUMENT(
        v->shape()[0] == kv_len,
        "Self Attention: k and v lengths must match.");

    CHECK_ARGUMENT(
        v->shape()[1] == num_kv_heads,
        "Self Attention: k and v head counts must match.");

    // 当前 Query 被认为位于完整 KV 序列的末尾。
    CHECK_ARGUMENT(
        kv_len >= q_len,
        "Self Attention: kv length must not be smaller than q length.");

    CHECK_ARGUMENT(
        q_len > 0 && num_heads > 0 && qk_dim > 0 && value_dim > 0,
        "Self Attention: query dimensions must be positive.");

    // Query Head 必须能平均分配给 KV Head。
    CHECK_ARGUMENT(
        num_kv_heads > 0 && num_heads % num_kv_heads == 0,
        "Self Attention: query heads must be divisible by kv heads.");

    // 输出形状：[q_len, num_heads, value_dim]
    CHECK_ARGUMENT(
        attn_val->shape()[0] == q_len && attn_val->shape()[1] == num_heads && attn_val->shape()[2] == value_dim,
        "Self Attention: output shape is invalid.");

    ASSERT(
        attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
        "Self Attention: all tensors must be contiguous.");

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);
    }

    core::context().setDevice(
        attn_val->deviceType(),
        attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            attn_val->dtype(),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);

#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::selfAttention(
            attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(),
            q_len, kv_len, num_heads, num_kv_heads, qk_dim, value_dim, scale,
            core::context().runtime().stream());
#endif

    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
