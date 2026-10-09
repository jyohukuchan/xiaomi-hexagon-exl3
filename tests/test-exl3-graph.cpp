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
    if (argc != 4 && argc != 5) { std::cerr << "Usage: test-exl3-graph backend fixture batch [f32|f16]\n"; return 2; }
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
        const std::string input_name = argc == 5 ? argv[4] : "f32";
        if (input_name != "f32" && input_name != "f16") throw std::runtime_error("Invalid input type");
        const ggml_type input_type = input_name == "f16" ? GGML_TYPE_F16 : GGML_TYPE_F32;
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
        auto * x = ggml_new_tensor_2d(ctx, input_type, k, batch);
        auto * y = ggml_mul_mat(ctx, w, x);
        auto * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, batch);
        auto * added = ggml_add(ctx, y, residual);
        struct row_case { size_t count; ggml_tensor * ids; ggml_tensor * output; };
        std::vector<row_case> row_cases;
        if (std::string(argv[1]) == "CPU") for (size_t count : {1u, 3u, 17u}) {
            auto * ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, count);
            row_cases.push_back({count, ids, ggml_get_rows(ctx, w, ids)});
        }
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
        std::vector<ggml_fp16_t> half_input;
        if (input_type == GGML_TYPE_F16) {
            half_input.resize(input.size());
            for (size_t i = 0; i < input.size(); ++i) {
                half_input[i] = ggml_fp32_to_fp16(input[i]);
                input[i] = ggml_fp16_to_fp32(half_input[i]);
            }
        }
        ggml_backend_tensor_set(w, blocked.data(), 0, blocked.size() * 2);
        if (input_type == GGML_TYPE_F16) ggml_backend_tensor_set(x, half_input.data(), 0, half_input.size() * 2);
        else ggml_backend_tensor_set(x, input.data(), 0, input.size() * 4);
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
        std::cout << "backend=" << argv[1] << " k=" << k << " n=" << n << " bits=" << bits << " batch=" << batch << " input=" << input_name << " ms=" << ms << " matmul_add_nmse=" << nmse << '\n';
        bool rows_ok = true;
        if (std::string(argv[1]) == "CPU") {
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            const auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (!set_threads) throw std::runtime_error("CPU thread control unavailable");
            for (const auto & test : row_cases) {
                std::vector<int32_t> indices(test.count);
                for (size_t r = 0; r < test.count; ++r) indices[r] = test.count == 1 ? int32_t(n - 1) : r % 3 == 0 ? 0 : r % 3 == 1 ? int32_t(n / 2) : int32_t(n - 1);
                ggml_backend_tensor_set(test.ids, indices.data(), 0, indices.size() * 4);
                auto * rows_graph = ggml_new_graph_custom(ctx, 64, false);
                ggml_build_forward_expand(rows_graph, test.output);
                std::vector<float> baseline, values(k * test.count);
                for (int threads : {1, 2, 4, 8}) {
                    set_threads(backend, threads);
                    if (ggml_backend_graph_compute(backend, rows_graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("Embedding graph failed");
                    ggml_backend_tensor_get(test.output, values.data(), 0, values.size() * 4);
                    if (threads == 1) baseline = values;
                    else if (std::memcmp(baseline.data(), values.data(), values.size() * 4)) throw std::runtime_error("Parallel embedding differs bit for bit");
                    double e = 0, denominator = 0;
                    for (size_t r = 0; r < test.count; ++r) for (size_t i = 0; i < k; ++i) {
                        const double expected = ggml_fp16_to_fp32(original[i * n + indices[r]]);
                        const double delta = values[r * k + i] - expected;
                        e += delta * delta; denominator += expected * expected;
                    }
                    const double row_nmse = denominator ? e / denominator : e;
                    rows_ok &= std::isfinite(row_nmse) && row_nmse < 1e-6;
                    std::cout << "embedding_count=" << test.count << " threads=" << threads << " embedding_nmse=" << row_nmse << '\n';
                }
            }
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
