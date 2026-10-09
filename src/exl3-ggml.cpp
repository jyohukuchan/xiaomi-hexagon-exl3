#include "exl3-block.h"
#include "exl3_codec.h"
#include "ggml-cpu-impl.h"
#include "ggml.h"
#include <vector>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>

static size_t row_offset(const ggml_tensor * t, size_t row) {
    const size_t i1 = row % t->ne[1];
    row /= t->ne[1];
    const size_t i2 = row % t->ne[2];
    const size_t i3 = row / t->ne[2];
    return i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
}

extern "C" void ggml_exl3_get_rows(const ggml_compute_params * params, ggml_tensor * dst) {
    const char * profile_option = std::getenv("EXL3_CPU_EMBED_PROFILE");
    const bool profile = profile_option && std::strcmp(profile_option, "1") == 0;
    const auto started = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    size_t completed_groups = 0;
    const auto * weight = dst->src[0];
    const auto * ids = dst->src[1];
    const size_t k = weight->ne[0], n = weight->ne[1];
    const unsigned bits = exl3_type_bits(weight->type);
    GGML_ASSERT(bits && k && n && k % 128 == 0 && n % 128 == 0 && weight->ne[2] == 1 && weight->ne[3] == 1);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32);
    const size_t count = ggml_nelements(ids);
    std::vector<uint16_t> decoded(128 * 128);
    for (size_t r = params->ith; r < count; r += params->nth) {
        int32_t id;
        std::memcpy(&id, (const char *) ids->data + row_offset(ids, r / ids->ne[0]) + (r % ids->ne[0]) * ids->nb[0], 4);
        GGML_ASSERT(id >= 0 && size_t(id) < n);
        float * result = (float *) ((char *) dst->data + row_offset(dst, r));
        for (size_t kb = 0; kb < k / 128; ++kb) {
            const auto * group = (const uint8_t *) weight->data + ((size_t(id) / 128) * (k / 128) + kb) * exl3_group_bytes(bits);
            exl3::reconstruct((const uint16_t *) group, exl3_group_su(group, bits), exl3_group_sv(group, bits),
                              128, 128, bits, exl3::Codebook::mul1, decoded.data());
            for (size_t i = 0; i < 128; ++i) result[kb * 128 + i] = exl3::half_to_float(decoded[i * 128 + id % 128]);
            ++completed_groups;
        }
    }
    if (profile) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        std::fprintf(stderr, "EXL3_EMBED thread=%d threads=%d ids=%zu groups=%zu ms=%.6f\n", params->ith, params->nth, count, completed_groups, ms);
    }
}

extern "C" void ggml_exl3_mul_mat(const ggml_compute_params * params, ggml_tensor * dst) {
    const auto * weight = dst->src[0];
    const auto * input = dst->src[1];
    const size_t k = weight->ne[0], n = weight->ne[1], rows = ggml_nrows(input);
    const unsigned bits = exl3_type_bits(weight->type);
    GGML_ASSERT(bits && k % 128 == 0 && n % 128 == 0 && weight->ne[2] == 1 && weight->ne[3] == 1);
    GGML_ASSERT(input->type == GGML_TYPE_F32 || input->type == GGML_TYPE_F16);
    std::vector<float> xh(rows * k), result(rows * 128);
    std::vector<uint16_t> decoded(128 * 128);
    for (size_t kb = 0; kb < k / 128; ++kb) {
        const auto * group = (const uint8_t *) weight->data + kb * exl3_group_bytes(bits);
        const auto * su = exl3_group_su(group, bits);
        for (size_t r = 0; r < rows; ++r) {
            const auto * row = (const uint8_t *) input->data + row_offset(input, r);
            for (size_t j = 0; j < 128; ++j) {
                float value;
                if (input->type == GGML_TYPE_F32) std::memcpy(&value, row + (kb * 128 + j) * 4, 4);
                else { uint16_t h; std::memcpy(&h, row + (kb * 128 + j) * 2, 2); value = exl3::half_to_float(h); }
                xh[r * k + kb * 128 + j] = value * exl3::half_to_float(su[j]);
            }
            exl3::hadamard128(xh.data() + r * k + kb * 128, 128);
        }
    }
    for (size_t nb = params->ith; nb < n / 128; nb += params->nth) {
        std::fill(result.begin(), result.end(), 0.0f);
        for (size_t kb = 0; kb < k / 128; ++kb) {
            const auto * group = (const uint8_t *) weight->data + (nb * (k / 128) + kb) * exl3_group_bytes(bits);
            exl3::decode_inner((const uint16_t *) group, 128, 128, bits, exl3::Codebook::mul1, decoded.data());
            for (size_t i = 0; i < 128; ++i) for (size_t j = 0; j < 128; ++j) {
                const float w = exl3::half_to_float(decoded[i * 128 + j]);
                for (size_t r = 0; r < rows; ++r) result[r * 128 + j] += xh[r * k + kb * 128 + i] * w;
            }
        }
        const auto * group = (const uint8_t *) weight->data + nb * (k / 128) * exl3_group_bytes(bits);
        const auto * sv = exl3_group_sv(group, bits);
        for (size_t r = 0; r < rows; ++r) {
            exl3::hadamard128(result.data() + r * 128, 128);
            auto * row = (float *) ((char *) dst->data + row_offset(dst, r));
            for (size_t j = 0; j < 128; ++j) row[nb * 128 + j] = result[r * 128 + j] * exl3::half_to_float(sv[j]);
        }
    }
}
