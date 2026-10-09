#include "ggml.h"
#include "ggml-backend.h"
#include "exl3-block.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 4) { std::cerr << "Usage: test-exl3-graph backend fixture batch\n"; return 2; }
    ggml_backend_t backend = nullptr;
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    try {
        std::ifstream source(argv[2], std::ios::binary);
        char magic[8]; uint32_t fields[4];
        source.read(magic, 8); source.read((char *) fields, 16);
        if (!source || std::string(magic, 8) != "EXL3REF1") throw std::runtime_error("Invalid fixture");
        const size_t k = fields[0], n = fields[1], bits = fields[2], cb = fields[3];
        const size_t batch = std::stoul(argv[3]);
        if (k % 128 || n % 128 || !k || !n || k > 16384 || n > 16384 || k * n > 64 * 1024 * 1024 ||
            cb != 2 || (bits != 4 && bits != 6 && bits != 8) || !batch || batch > 512) throw std::runtime_error("Invalid fixture geometry");
        auto read = [&](size_t count) {
            std::vector<uint16_t> data(count);
            source.read((char *) data.data(), count * 2);
            if (!source) throw std::runtime_error("Truncated fixture");
            return data;
        };
        auto packed = read(k * n * bits / 16), su = read(k), sv = read(n);
        auto inner = read(k * n), original = read(k * n);
        std::vector<uint16_t> blocked(k * n * (16 * bits + 4) / 256);
        for (size_t nb = 0; nb < n / 128; ++nb) for (size_t kb = 0; kb < k / 128; ++kb) {
            auto * group = (uint8_t *) blocked.data() + (nb * (k / 128) + kb) * exl3_group_bytes(bits);
            for (size_t r = 0; r < 8; ++r) for (size_t c = 0; c < 8; ++c) {
                std::memcpy(group + (r * 8 + c) * bits * 32, packed.data() + ((kb * 8 + r) * (n / 16) + nb * 8 + c) * bits * 16, bits * 32);
            }
            std::memcpy(group + 2048 * bits, su.data() + kb * 128, 256);
            std::memcpy(group + 2048 * bits + 256, sv.data() + nb * 128, 256);
        }
        ggml_backend_load_all();
        backend = ggml_backend_init_by_name(argv[1], nullptr);
        if (!backend) throw std::runtime_error("Backend not available");
        ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
        const ggml_type type = bits == 4 ? GGML_TYPE_EXL3_4 : bits == 6 ? GGML_TYPE_EXL3_6 : GGML_TYPE_EXL3_8;
        auto * w = ggml_new_tensor_2d(ctx, type, k, n);
        auto * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, batch);
        auto * y = ggml_mul_mat(ctx, w, x);
        auto * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, batch);
        auto * added = ggml_add(ctx, y, residual);
        auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 3);
        auto * gathered = ggml_get_rows(ctx, w, ids);
        ggml_set_name(w, "exl3_weight"); ggml_set_name(x, "activation"); ggml_set_name(y, "result");
        if (!ggml_backend_supports_op(backend, y)) throw std::runtime_error("Backend rejected EXL3 matrix operation");
        auto * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, added);
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("Buffer allocation failed");
        std::vector<float> input(k * batch), actual(n * batch);
        std::vector<float> residual_data(n * batch);
        for (size_t i = 0; i < residual_data.size(); ++i) residual_data[i] = float(int(i % 23) - 11) / 7;
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 13) % 17) - 8) / 17;
        ggml_backend_tensor_set(w, blocked.data(), 0, blocked.size() * 2);
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4);
        ggml_backend_tensor_set(residual, residual_data.data(), 0, residual_data.size() * 4);
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Graph failed");
        ggml_backend_synchronize(backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        ggml_backend_tensor_get(added, actual.data(), 0, actual.size() * 4);
        double error = 0, norm = 0;
        size_t diagnostics = 0;
        for (size_t b = 0; b < batch; ++b) for (size_t c = 0; c < n; ++c) {
            double expected = residual_data[b * n + c];
            for (size_t r = 0; r < k; ++r) expected += double(input[b * k + r]) * ggml_fp16_to_fp32(original[r * n + c]);
            const double delta = actual[b * n + c] - expected;
            if ((!std::isfinite(delta) || std::abs(delta) > 0.1) && diagnostics++ < 12) {
                std::cout << "mismatch row=" << b << " col=" << c << " actual=" << actual[b * n + c] << " expected=" << expected << '\n';
            }
            error += delta * delta; norm += expected * expected;
        }
        const double nmse = norm ? error / norm : error;
        if (nmse > 1e-5) {
            for (size_t i = 0; i < 8; ++i) std::cout << "actual " << i << '=' << actual[i] << '\n';
        }
        std::cout << "backend=" << argv[1] << " k=" << k << " n=" << n << " bits=" << bits << " batch=" << batch << " ms=" << ms << " matmul_add_nmse=" << nmse << '\n';
        bool rows_ok = true;
        if (std::string(argv[1]) == "CPU") {
            const int32_t indices[3] = {0, int32_t(n / 2), int32_t(n - 1)};
            ggml_backend_tensor_set(ids, indices, 0, sizeof(indices));
            auto * rows_graph = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(rows_graph, gathered);
            if (ggml_backend_graph_compute(backend, rows_graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Embedding graph failed");
            std::vector<float> values(k * 3);
            ggml_backend_tensor_get(gathered, values.data(), 0, values.size() * 4);
            double e = 0, denominator = 0;
            for (size_t r = 0; r < 3; ++r) for (size_t i = 0; i < k; ++i) {
                const double expected = ggml_fp16_to_fp32(original[i * n + indices[r]]);
                const double delta = values[r * k + i] - expected;
                e += delta * delta; denominator += expected * expected;
            }
            const double row_nmse = denominator ? e / denominator : e;
            rows_ok = std::isfinite(row_nmse) && row_nmse < 1e-6;
            std::cout << "embedding_nmse=" << row_nmse << '\n';
        }
        ggml_backend_buffer_free(buffer); buffer = nullptr;
        ggml_free(ctx); ctx = nullptr;
        ggml_backend_free(backend); backend = nullptr;
        return rows_ok && std::isfinite(nmse) && nmse < 1e-5 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}
