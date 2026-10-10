#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>

using nlohmann::json;
static volatile sig_atomic_t stopped = 0;
static void handler(int) { stopped = 1; }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static double now() {
    timespec t{};
    require(clock_gettime(CLOCK_BOOTTIME, &t) == 0, "clock");
    return t.tv_sec + t.tv_nsec / 1e9;
}
static void emit(json row) {
    row["boot_s"] = now();
    std::cout << "BASELINE " << row.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
}
static void quiet(enum ggml_log_level level, const char * text, void *) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::cerr << text;
}

int main(int argc, char ** argv) try {
    require(argc == 5 || argc == 6, "Usage: homura_baseline MODEL PROMPT DECODE_STEPS REPEATS [teacher-forced]");
    const bool forced = argc == 6 && std::string(argv[5]) == "teacher-forced";
    require(argc == 5 || forced, "invalid decode mode");
    const int steps = std::stoi(argv[3]), repeats = std::stoi(argv[4]);
    require(steps >= 8 && steps <= 256 && repeats >= 1 && repeats <= 5, "invalid limits");
    std::cout.setf(std::ios::unitbuf);
    signal(SIGINT, handler); signal(SIGTERM, handler);
    emit({{"event", "init"}, {"pid", getpid()}, {"decode_steps", steps}, {"repeats", repeats}, {"teacher_forced", forced}});
    llama_log_set(quiet, nullptr);
    ggml_log_set(quiet, nullptr);
    ggml_backend_load_all_from_path("/data/local/tmp/llama.cpp/lib");
    llama_backend_init();
    auto device = ggml_backend_dev_by_name("HTP0");
    require(device != nullptr, "HTP0 unavailable");
    ggml_backend_dev_t devices[] = {device, nullptr};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = 99;
    const auto model = std::unique_ptr<llama_model, decltype(&llama_model_free)>(llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 64;
    cp.n_threads = 6; cp.n_threads_batch = 6;
    cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    const auto ctx = std::unique_ptr<llama_context, decltype(&llama_free)>(llama_init_from_model(model.get(), cp), llama_free);
    require(bool(ctx), "context load failed");
    const auto * vocab = llama_model_get_vocab(model.get());
    std::ifstream input(argv[2], std::ios::binary);
    require(bool(input), "prompt file missing");
    const std::string prompt{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    require(!prompt.empty() && prompt.size() < 100000, "invalid prompt");
    const int count = -llama_tokenize(vocab, prompt.data(), int(prompt.size()), nullptr, 0, true, true);
    require(count > 0 && count <= 512, "prompt token count outside bounds");
    std::vector<llama_token> tokens(count);
    require(llama_tokenize(vocab, prompt.data(), int(prompt.size()), tokens.data(), count, true, true) == count, "tokenization");
    std::vector<llama_token> forced_tokens;
    for (llama_token t : tokens) if (!llama_vocab_is_control(vocab, t) && !llama_vocab_is_eog(vocab, t)) forced_tokens.push_back(t);
    require(!forced_tokens.empty(), "no fixed input tokens");
    auto finite = [&]() {
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        require(logits != nullptr, "logits missing");
        for (int i = 0; i < llama_vocab_n_tokens(vocab); ++i) require(std::isfinite(logits[i]), "nonfinite logits");
    };
    auto compute = [&](llama_token * data, int n) {
        require(llama_decode(ctx.get(), llama_batch_get_one(data, n)) == 0, "llama_decode failed");
        llama_synchronize(ctx.get());
    };
    emit({{"event", "loaded"}, {"prompt_tokens", count}, {"backend", "HTP0"}, {"kv", "f16"}});
    for (int trial = -1; trial < repeats && !stopped; ++trial) {
        llama_memory_clear(llama_get_memory(ctx.get()), true);
        if (trial >= 0) {
            emit({{"event", "idle_start"}, {"trial", trial}});
            for (int i = 0; i < 150 && !stopped; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (stopped) break;
        const auto sampler = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>(llama_sampler_init_greedy(), llama_sampler_free);
        require(bool(sampler), "sampler allocation");
        if (trial >= 0) emit({{"event", "prefill_start"}, {"trial", trial}});
        const double p0 = now();
        compute(tokens.data(), count);
        const double p1 = now();
        finite();
        if (trial >= 0) emit({{"event", "prefill_end"}, {"trial", trial}, {"start_boot_s", p0}, {"end_boot_s", p1}, {"seconds", p1 - p0}, {"tokens", count}});
        llama_token next = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        std::vector<llama_token> generated{next};
        std::vector<llama_token> supplied;
        const int limit = trial < 0 ? 8 : steps;
        if (trial >= 0) emit({{"event", "decode_start"}, {"trial", trial}});
        const double d0 = now();
        int calls = 0;
        while (calls < limit && !stopped && (forced || !llama_vocab_is_eog(vocab, next))) {
            if (forced) next = forced_tokens[(size_t(calls) + 17) % forced_tokens.size()];
            supplied.push_back(next);
            compute(&next, 1);
            ++calls;
            next = llama_sampler_sample(sampler.get(), ctx.get(), -1);
            generated.push_back(next);
        }
        const double d1 = now();
        finite();
        if (trial >= 0) {
            std::string output;
            for (llama_token token : generated) {
                char piece[256];
                const int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
                require(n >= 0 && n <= int(sizeof(piece)), "token piece overflow");
                output.append(piece, n);
            }
            emit({{"event", "decode_end"}, {"trial", trial}, {"start_boot_s", d0}, {"end_boot_s", d1},
                  {"seconds", d1 - d0}, {"decode_calls", calls}, {"tokens_per_second", calls / (d1 - d0)},
                  {"stopped", bool(stopped)}, {"eog", llama_vocab_is_eog(vocab, next)}, {"teacher_forced", forced},
                  {"supplied_ids", supplied}, {"generated_ids", generated}, {"text", output}});
        }
    }
    emit({{"event", "finished"}, {"stopped", bool(stopped)}});
    return stopped ? 2 : 0;
} catch (const std::exception & e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
