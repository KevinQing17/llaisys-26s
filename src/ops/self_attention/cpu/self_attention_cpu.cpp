#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {

    size_t heads_per_kv = num_heads / num_kv_heads;

    size_t past_len = kv_len - q_len;

    // 重复利用这块临时内存，保存一个 Query 的注意力分数。
    std::vector<float> scores(kv_len);

    for (size_t query_index = 0;
         query_index < q_len;
         query_index++) {

        // 因果掩码允许访问的 Key 数量。
        size_t allowed_kv = past_len + query_index + 1;

        for (size_t head = 0;
             head < num_heads;
             head++) {

            // 多个 Query Head 可以共享同一个 KV Head。
            size_t kv_head = head / heads_per_kv;

            float max_score = -std::numeric_limits<float>::infinity();

            // 第一遍：计算 QK^T × scale，并找最大值。
            for (size_t key_index = 0;
                 key_index < allowed_kv;
                 key_index++) {

                float dot = 0.0f;

                for (size_t dim = 0;
                     dim < qk_dim;
                     dim++) {

                    size_t q_offset = (query_index * num_heads + head) * qk_dim + dim;

                    size_t k_offset = (key_index * num_kv_heads + kv_head) * qk_dim + dim;

                    float q_value = llaisys::utils::cast<float>(
                        q[q_offset]);

                    float k_value = llaisys::utils::cast<float>(
                        k[k_offset]);

                    dot += q_value * k_value;
                }

                float score = dot * scale;
                scores[key_index] = score;

                if (score > max_score) {
                    max_score = score;
                }
            }

            // 第二遍：稳定 Softmax 的指数与总和。
            float exp_sum = 0.0f;

            for (size_t key_index = 0;
                 key_index < allowed_kv;
                 key_index++) {

                float exp_value = std::exp(
                    scores[key_index] - max_score);

                scores[key_index] = exp_value;
                exp_sum += exp_value;
            }

            float inv_exp_sum = 1.0f / exp_sum;

            // 将 Softmax 权重乘以 V。
            for (size_t value_index = 0;
                 value_index < value_dim;
                 value_index++) {

                float result = 0.0f;

                for (size_t key_index = 0;
                     key_index < allowed_kv;
                     key_index++) {

                    float attention_weight = scores[key_index] * inv_exp_sum;

                    size_t v_offset = (key_index * num_kv_heads + kv_head) * value_dim + value_index;

                    float v_value = llaisys::utils::cast<float>(
                        v[v_offset]);

                    result += attention_weight * v_value;
                }

                size_t out_offset = (query_index * num_heads + head) * value_dim + value_index;

                attn_val[out_offset] = llaisys::utils::cast<T>(result);
            }
        }
    }
}

namespace llaisys::ops::cpu {

void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t type,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t qk_dim,
    size_t value_dim,
    float scale) {

    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);

    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);

    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            qk_dim,
            value_dim,
            scale);

    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::cpu
