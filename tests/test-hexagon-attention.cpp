#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static std::vector<float> compute(ggml_backend_t backend, int sequences, int queries) {
    constexpr int dim = 256, heads = 8, kv_heads = 2, kv_length = 128, mask_rows = 32;
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    ggml_backend_buffer_t buffer = nullptr;
    try {
        auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, queries, heads, sequences);
        auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, kv_length, kv_heads, sequences);
        auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, dim, kv_length, kv_heads, sequences);
        auto * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv_length, mask_rows, 1, sequences);
        auto * output = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / std::sqrt(float(dim)), 0, 0);
        ggml_set_name(q, "test_query"); ggml_set_name(k, "test_key"); ggml_set_name(v, "test_value");
        ggml_set_name(mask, "test_mask"); ggml_set_name(output, "test_attention");
        if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("Backend rejected multi-sequence attention");
        auto * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, output);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("Buffer allocation failed");
        std::vector<float> query(ggml_nelements(q)), result(ggml_nelements(output));
        std::vector<ggml_fp16_t> key(ggml_nelements(k)), value(key.size()), masks(ggml_nelements(mask));
        for (size_t i = 0; i < query.size(); ++i) query[i] = std::sin(float(i * 13 % 997)) * 1.5f;
        for (size_t i = 0; i < key.size(); ++i) {
            key[i] = ggml_fp32_to_fp16(std::cos(float(i * 7 % 991)) * 1.5f);
            const int sequence = int(i / (dim * kv_length * kv_heads));
            value[i] = ggml_fp32_to_fp16(std::sin(float(i * 11 % 983)) * .5f + sequence * .2f);
        }
        for (int s = 0; s < sequences; ++s) for (int r = 0; r < mask_rows; ++r) for (int i = 0; i < kv_length; ++i) {
            masks[(s * mask_rows + r) * kv_length + i] = ggml_fp32_to_fp16(i < 24 + s * 9 + r ? 0.0f : -std::numeric_limits<float>::infinity());
        }
        ggml_backend_tensor_set(q, query.data(), 0, query.size() * 4);
        ggml_backend_tensor_set(k, key.data(), 0, key.size() * 2);
        ggml_backend_tensor_set(v, value.data(), 0, value.size() * 2);
        ggml_backend_tensor_set(mask, masks.data(), 0, masks.size() * 2);
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Attention graph failed");
        ggml_backend_synchronize(backend);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * 4);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        return result;
    } catch (...) {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw;
    }
}

int main() {
    ggml_backend_load_all();
    auto cpu = ggml_backend_init_by_name("CPU", nullptr);
    auto htp = ggml_backend_init_by_name("HTP0", nullptr);
    if (!cpu || !htp) { std::cerr << "CPU and HTP0 are required\n"; return 2; }
    bool pass = true;
    try {
        for (int sequences : {1, 4, 8}) for (int queries : {1, 3, 17}) {
            auto expected = compute(cpu, sequences, queries);
            auto actual = compute(htp, sequences, queries);
            double error = 0, norm = 0, max_error = 0;
            for (size_t i = 0; i < actual.size(); ++i) {
                const double delta = double(actual[i]) - expected[i];
                error += delta * delta; norm += double(expected[i]) * expected[i];
                max_error = std::max(max_error, std::abs(delta));
            }
            const double nmse = error / norm;
            pass &= std::isfinite(nmse) && nmse < 1e-4;
            std::cout << "sequences=" << sequences << " queries=" << queries << " nmse=" << nmse << " max_error=" << max_error << '\n';
        }
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; pass = false; }
    ggml_backend_free(htp); ggml_backend_free(cpu);
    return pass ? 0 : 1;
}
