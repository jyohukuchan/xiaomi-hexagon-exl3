#include "llama.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static int bounded_number(const char * value, int maximum) {
    const std::string text(value);
    size_t consumed = 0;
    const long long number = std::stoll(text, &consumed);
    if (consumed != text.size() || number < 0 || number > maximum) throw std::runtime_error("Invalid numeric argument");
    return int(number);
}

int main(int argc, char ** argv) {
    if (argc != 5 && argc != 6) {
        std::fprintf(stderr, "Usage: context_probe MODEL EMPTY_DIRECTORY DEPTH STEPS [GPU_LAYERS]\n");
        return 2;
    }
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    bool backend_initialized = false;
    try {
        const int depth = bounded_number(argv[3], 65536), steps = bounded_number(argv[4], 1024);
        const int gpu_layers = argc == 6 ? bounded_number(argv[5], 99) : 99;
        if (!steps) throw std::runtime_error("At least one decode step is required");
        const std::filesystem::path directory(argv[2]);
        if (std::filesystem::exists(directory) && !std::filesystem::is_empty(directory)) throw std::runtime_error("Output directory must be empty");
        std::filesystem::create_directories(directory);
        llama_backend_init();
        backend_initialized = true;
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        model = llama_model_load_from_file(argv[1], mp);
        if (!model) throw std::runtime_error("Model loading failed");
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int vocabulary = llama_vocab_n_tokens(vocab);
        if (vocabulary < 2) throw std::runtime_error("Invalid vocabulary size");
        auto cp = llama_context_default_params();
        cp.n_ctx = std::max(512, depth + steps);
        cp.n_batch = 512; cp.n_ubatch = 128;
        cp.n_threads = 8; cp.n_threads_batch = 8;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("Context initialization failed");
        const unsigned context_capacity = llama_n_ctx_seq(ctx);
        if (context_capacity < unsigned(depth + steps)) throw std::runtime_error("Actual context capacity is too small");
        constexpr uint32_t seed = 0x7ba4139d;
        uint32_t random = seed;
        std::vector<llama_token> input(cp.n_batch), decoded;
        uint64_t checked_rows = 0;
        float min_logit = 0, max_logit = 0;
        auto inspect = [&]() {
            const float * logits = llama_get_logits_ith(ctx, -1);
            if (!logits) throw std::runtime_error("Missing logits");
            llama_token best = 0;
            for (int token = 0; token < vocabulary; ++token) {
                if (!std::isfinite(logits[token])) throw std::runtime_error("Non-finite model logits");
                if (logits[token] > logits[best]) best = token;
                if (!checked_rows && !token) min_logit = max_logit = logits[token];
                min_logit = std::min(min_logit, logits[token]);
                max_logit = std::max(max_logit, logits[token]);
            }
            ++checked_rows;
            return best;
        };
        auto verify_position = [&](int processed) {
            if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != processed - 1) throw std::runtime_error("Sequence position does not match processed tokens");
        };
        llama_token next = llama_vocab_get_add_bos(vocab) ? llama_vocab_bos(vocab) : 0;
        const auto prefill_start = std::chrono::steady_clock::now();
        for (int offset = 0; offset < depth; offset += cp.n_batch) {
            const int count = std::min<int>(cp.n_batch, depth - offset);
            for (int i = 0; i < count; ++i) {
                random ^= random << 13; random ^= random >> 17; random ^= random << 5;
                input[i] = llama_token(random % uint32_t(vocabulary));
            }
            if (llama_decode(ctx, llama_batch_get_one(input.data(), count))) throw std::runtime_error("Prefill decode failed");
            llama_synchronize(ctx);
            verify_position(offset + count);
            next = inspect();
            std::fprintf(stderr, "PREFILL checked=%d/%d position=%d\n", offset + count, depth, offset + count - 1);
        }
        const double prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prefill_start).count();
        const auto decode_start = std::chrono::steady_clock::now();
        for (int step = 0; step < steps; ++step) {
            decoded.push_back(next);
            if (llama_decode(ctx, llama_batch_get_one(&next, 1))) throw std::runtime_error("Extension decode failed");
            llama_synchronize(ctx);
            verify_position(depth + step + 1);
            next = inspect();
            std::fprintf(stderr, "DECODE checked=%d/%d position=%d\n", step + 1, steps, depth + step);
        }
        const double decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start).count();
        const auto memory = llama_get_memory(ctx);
        const auto position_min = llama_memory_seq_pos_min(memory, 0), position_max = llama_memory_seq_pos_max(memory, 0);
        const float * logits = llama_get_logits_ith(ctx, -1);
        std::ofstream binary(directory / "last_logits.bin", std::ios::binary);
        binary.write(reinterpret_cast<const char *>(logits), vocabulary * sizeof(float));
        binary.close();
        if (!binary) throw std::runtime_error("Logit artifact write failed");
        llama_free(ctx); ctx = nullptr;
        llama_model_free(model); model = nullptr;
        llama_backend_free(); backend_initialized = false;
        std::ofstream report(directory / "report.json");
        report << "{\"passed\":true,\"depth\":" << depth << ",\"decode_steps\":" << steps
               << ",\"context_capacity\":" << context_capacity << ",\"gpu_layers_requested\":" << gpu_layers
               << ",\"batch\":512,\"microbatch\":128,\"cpu_threads\":8,\"kv_requested\":\"f16\""
               << ",\"seed\":" << seed << ",\"extension_mode\":\"greedy_ignoring_eos\",\"vocabulary\":" << vocabulary
               << ",\"checked_logit_rows\":" << checked_rows << ",\"checked_logit_values\":" << checked_rows * vocabulary
               << ",\"hybrid_position_min\":" << position_min << ",\"hybrid_position_max\":" << position_max
               << ",\"prefill_ms_including_checks\":" << prefill_ms << ",\"decode_ms_including_checks\":" << decode_ms
               << ",\"diagnostic_decode_tokens_per_second\":" << steps * 1000.0 / decode_ms
               << ",\"logit_min\":" << min_logit << ",\"logit_max\":" << max_logit << ",\"decoded_token_ids\":[";
        for (size_t i = 0; i < decoded.size(); ++i) report << (i ? "," : "") << decoded[i];
        report << "]}\n";
        report.close();
        if (!report) throw std::runtime_error("Report write failed");
        std::puts("CONTEXT_PROBE_PASS");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "CONTEXT_PROBE_FAIL: %s\n", error.what());
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        if (backend_initialized) llama_backend_free();
        return 1;
    }
}
