#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

using nlohmann::json;
static volatile sig_atomic_t stopped = 0;
static void stop_handler(int) { stopped = 1; }
static double boot_seconds() {
    timespec value{};
    if (clock_gettime(CLOCK_BOOTTIME, &value)) throw std::runtime_error("boottime clock");
    return value.tv_sec + value.tv_nsec / 1e9;
}
static void emit(json row) {
    row["boot_s"] = boot_seconds();
    std::cout << "BENCH " << row.dump() << '\n';
}
static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
static void compute(ggml_backend_t backend, ggml_cgraph * graph) {
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "backend graph compute failed");
    ggml_backend_synchronize(backend);
}
static float random_value(uint32_t & state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return float(int(state % 4096) - 2048) / 2048;
}

int main(int argc, char ** argv) try {
    std::cout.setf(std::ios::unitbuf);
    signal(SIGINT, stop_handler); signal(SIGTERM, stop_handler);
    require(argc >= 2, "Usage: accelerator_bench list | mat BACKEND f16|q8_0|q4_0 BATCH SECONDS | copy BACKEND contiguous|transpose SECONDS");
    ggml_backend_load_all_from_path("/data/local/tmp/llama.cpp/lib");
    if (std::string(argv[1]) == "list") {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            const auto dev = ggml_backend_dev_get(i);
            emit({{"event", "device"}, {"name", ggml_backend_dev_name(dev)}, {"description", ggml_backend_dev_description(dev)}});
        }
        return 0;
    }
    const bool mat = std::string(argv[1]) == "mat";
    require(mat ? argc == 6 || argc == 7 : (argc == 5 || argc == 6) && std::string(argv[1]) == "copy", "Invalid arguments");
    const std::string backend_name = argv[2], format = argv[3];
    require(backend_name == "GPUOpenCL" || backend_name == "HTP0", "Only GPUOpenCL/HTP0 are allowed; no CPU fallback");
    const int batch = mat ? std::stoi(argv[4]) : 1;
    const double duration = std::stod(argv[mat ? 5 : 4]);
    const double duty = argc == (mat ? 7 : 6) ? std::stod(argv[mat ? 6 : 5]) : 1;
    require(std::isfinite(duty) && duty >= 0.1 && duty <= 1, "Duty must be in [0.1,1]");
    require(batch > 0 && batch <= 512 && duration >= 0.1 && duration <= 120, "Invalid batch/duration");
    require(mat ? format == "f16" || format == "q8_0" || format == "q4_0" : format == "contiguous" || format == "transpose", "Invalid format");
    emit({{"event", "init"}, {"pid", getpid()}, {"backend", backend_name}, {"operation", mat ? "mat" : "copy"},
          {"format", format}, {"batch", batch}, {"duration_s", duration}, {"duty", duty}, {"seed", 2074350493}});
    const auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(
        ggml_backend_init_by_name(backend_name.c_str(), nullptr), ggml_backend_free);
    require(bool(backend), "Backend initialization failed");
    require(std::string(ggml_backend_dev_name(ggml_backend_get_device(backend.get()))) == backend_name, "Backend selection mismatch");
    ggml_init_params params{4 * 1024 * 1024, nullptr, true};
    const auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init(params), ggml_free);
    require(bool(ctx), "Context allocation failed");
    const int64_t m = 4096, k = mat ? 14336 : 4096;
    const ggml_type type = !mat || format == "f16" ? GGML_TYPE_F16 : format == "q8_0" ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_0;
    ggml_tensor * weights = ggml_new_tensor_2d(ctx.get(), type, k, m);
    ggml_tensor * input = mat ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, batch) : nullptr;
    ggml_tensor * destination = mat ? nullptr : ggml_new_tensor_2d(ctx.get(), type, k, m);
    ggml_tensor * output = mat ? ggml_mul_mat(ctx.get(), weights, input) :
        ggml_cpy(ctx.get(), format == "transpose" ? ggml_transpose(ctx.get(), weights) : weights, destination);
    require(ggml_backend_supports_op(backend.get(), output), "Operation not supported on selected backend");
    ggml_set_name(weights, "bench_weights");
    ggml_set_name(output, "bench_output");
    if (input) ggml_set_name(input, "bench_activations");
    const auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    require(bool(buffer), "Backend tensor allocation failed");
    std::vector<unsigned char> packed(ggml_nbytes(weights));
    std::vector<float> activations(mat ? size_t(k) * batch : 0);
    uint32_t state = 0x7ba4139d;
    if (mat) {
        std::vector<float> values(size_t(k) * m);
        for (auto & value : values) value = random_value(state);
        require(ggml_quantize_chunk(type, values.data(), packed.data(), 0, m, k, nullptr) == packed.size(), "Quantized payload size mismatch");
        for (auto & value : activations) value = random_value(state);
        ggml_backend_tensor_set(input, activations.data(), 0, activations.size() * 4);
    } else {
        auto * values = reinterpret_cast<ggml_fp16_t *>(packed.data());
        for (size_t i = 0; i < size_t(k) * m; ++i)
            values[i] = ggml_fp32_to_fp16(float(int(i % 1999) - 999) / 128);
    }
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
    auto * graph = ggml_new_graph_custom(ctx.get(), 2048, false);
    ggml_build_forward_expand(graph, output);
    const auto single_start = std::chrono::steady_clock::now();
    compute(backend.get(), graph);
    const double warmup_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - single_start).count();
    auto validate = [&]() {
        if (mat) {
            std::vector<float> actual(size_t(m) * batch), row(k);
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * 4);
            for (float value : actual) require(std::isfinite(value), "Nonfinite output");
            double error = 0, norm = 0, maximum = 0;
            const auto * traits = ggml_get_type_traits(type);
            require(traits && traits->to_float, "Missing weight decoder");
            for (size_t sample = 0; sample < 64; ++sample) {
                const size_t index = sample * (actual.size() - 1) / 63;
                const size_t column = index % m, b = index / m;
                traits->to_float(packed.data() + column * ggml_row_size(type, k), row.data(), k);
                double expected = 0;
                for (int64_t j = 0; j < k; ++j) expected += double(row[j]) * activations[b * k + j];
                const double delta = actual[index] - expected;
                error += delta * delta; norm += expected * expected;
                maximum = std::max(maximum, std::abs(delta));
            }
            const double nmse = norm ? error / norm : error;
            require(std::isfinite(nmse) && nmse <= 5e-4, "Sampled FP64 reference NMSE exceeds gate");
            emit({{"event", "validation"}, {"finite_elements", actual.size()}, {"reference_samples", 64}, {"nmse", nmse}, {"max_abs", maximum}});
        } else {
            std::vector<ggml_fp16_t> actual(size_t(k) * m);
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * 2);
            const auto * expected = reinterpret_cast<const ggml_fp16_t *>(packed.data());
            for (size_t i = 0; i < actual.size(); ++i) {
                const size_t j = format == "transpose" ? (i % k) * m + i / k : i;
                require(actual[i] == expected[j], "Copy/transpose bit mismatch");
            }
            emit({{"event", "validation"}, {"bit_exact_elements", actual.size()}});
        }
    };
    validate();
    if (stopped) { emit({{"event", "aborted"}, {"reason", "Stopped during setup"}}); return 2; }
    const int duplicates = std::clamp(int(50000 / std::max(1.0, warmup_us)), 1, 2048);
    for (int i = 1; i < duplicates; ++i) ggml_graph_add_node(graph, output);
    compute(backend.get(), graph);
    const double flops = mat ? 2.0 * m * k * batch : 0;
    const size_t traffic = mat ? packed.size() : packed.size() * 2;
    emit({{"event", "start"}, {"backend", backend_name}, {"operation", mat ? "mat" : "copy"}, {"format", format},
          {"batch", batch}, {"m", m}, {"k", k}, {"duplicates", duplicates}, {"warmup_single_us", warmup_us},
          {"flops_per_op", flops}, {"weight_bytes", packed.size()}, {"traffic_bytes_per_op", traffic}, {"duty", duty}});
    const auto begin = std::chrono::steady_clock::now();
    double total_seconds = 0;
    double total_active_seconds = 0;
    uint64_t total_ops = 0;
    unsigned window = 0;
    while (!stopped && std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() < duration) {
        const auto start = std::chrono::steady_clock::now();
        const double window_target = std::min(1.0, duration - std::chrono::duration<double>(start - begin).count());
        uint64_t ops = 0;
        double active_seconds = 0;
        do {
            const auto compute_start = std::chrono::steady_clock::now();
            compute(backend.get(), graph); ops += duplicates;
            const double active = std::chrono::duration<double>(std::chrono::steady_clock::now() - compute_start).count();
            active_seconds += active;
            if (duty < 1 && !stopped) std::this_thread::sleep_for(std::chrono::duration<double>(active * (1 / duty - 1)));
        } while (!stopped && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < window_target);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        total_seconds += seconds; total_ops += ops;
        total_active_seconds += active_seconds;
        emit({{"event", "window"}, {"index", window++}, {"seconds", seconds}, {"ops", ops},
              {"us_per_op", seconds * 1e6 / ops}, {"gflops", flops * ops / seconds / 1e9},
              {"effective_GBps", double(traffic) * ops / seconds / 1e9}, {"active_seconds", active_seconds},
              {"active_gflops", flops * ops / active_seconds / 1e9}});
    }
    validate();
    emit({{"event", "result"}, {"stopped", bool(stopped)}, {"seconds", total_seconds}, {"ops", total_ops},
          {"us_per_op", total_seconds * 1e6 / total_ops}, {"gflops", flops * total_ops / total_seconds / 1e9},
          {"effective_GBps", double(traffic) * total_ops / total_seconds / 1e9}, {"active_seconds", total_active_seconds},
          {"active_gflops", flops * total_ops / total_active_seconds / 1e9}});
    // The successful process exit, including backend teardown, is required by the runner.
    return stopped ? 2 : 0;
} catch (const std::exception & error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
