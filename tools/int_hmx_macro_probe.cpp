// Project-authored adapter. HexKL is a separately licensed external dependency.
#define restrict __restrict
#include "remote.h"
#include "sdkl.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

using nlohmann::json;
static volatile sig_atomic_t stopped = 0;
static void handler(int) { stopped = 1; }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static void check(int status, const char * why) { if (status) throw std::runtime_error(std::string(why) + " status=" + std::to_string(status)); }
static double now() { timespec t{}; require(clock_gettime(CLOCK_BOOTTIME, &t) == 0, "clock"); return t.tv_sec + t.tv_nsec / 1e9; }
static void emit(json row) { row["boot_s"] = now(); std::cout << "INTEGER " << row.dump() << '\n'; }
static uint32_t random_u32(uint32_t & s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
struct Session {
    bool active = false;
    Session() { check(sdkl_npu_initialize(CDSP_DOMAIN_ID, nullptr, nullptr), "NPU initialize"); active = true; }
    void close() { check(sdkl_npu_finalize(CDSP_DOMAIN_ID), "NPU finalize"); active = false; }
    ~Session() { if (active) sdkl_npu_finalize(CDSP_DOMAIN_ID); }
};
struct Buffer {
    void * p = nullptr;
    explicit Buffer(size_t bytes) { check(sdkl_npu_alloc(bytes, &p), "shared allocation"); require(p != nullptr, "null allocation"); }
    ~Buffer() { if (p) sdkl_npu_free(p); }
};

int main(int argc, char ** argv) try {
    require(argc == 6 || argc == 7, "Usage: int_hmx_macro_probe BITS BATCH N K REPEATS [pad64]");
    const bool pad = argc == 7 && std::string(argv[6]) == "pad64";
    require(argc == 6 || pad, "invalid padding mode");
    const int bits = std::stoi(argv[1]), batch = std::stoi(argv[2]), n = std::stoi(argv[3]), k = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
    require((bits == 4 || bits == 8) && batch > 0 && batch <= 512 && n >= 32 && n <= 4096 && n % 32 == 0 &&
            k >= 32 && k <= 14336 && k % 32 == 0 && repeats > 0 && repeats <= 100, "invalid dimensions");
    std::cout.setf(std::ios::unitbuf);
    signal(SIGINT, handler); signal(SIGTERM, handler);
    const int api_batch = pad ? (batch + 63) / 64 * 64 : batch;
    emit({{"event", "init"}, {"pid", getpid()}, {"bits", bits}, {"batch", batch}, {"api_batch", api_batch},
          {"n", n}, {"k", k}, {"repeats", repeats}, {"pad64", pad}});
    Session session;
    char version[SDKL_VERSION_STR_LEN]{};
    check(sdkl_npu_get_version(CDSP_DOMAIN_ID, version), "NPU version");
    emit({{"event", "version"}, {"value", version}});
    {
        const size_t weights = size_t(n) * k, outputs = size_t(batch) * n, inputs = size_t(batch) * k;
        const size_t packed_bytes = weights * bits / 8;
        const size_t padded_rows = size_t((batch + 63) / 64) * 64;
        Buffer wb(packed_bytes + 4096), xb(padded_rows * k + 4096), yb(padded_rows * n * 4 + 4096);
        auto * w = static_cast<uint8_t *>(wb.p);
        auto * x = static_cast<uint8_t *>(xb.p);
        auto * y = static_cast<int32_t *>(yb.p);
        std::vector<int8_t> reference_weights(weights);
        uint32_t seed = 0x61d0b77a;
        for (auto & value : reference_weights) value = int8_t(int(random_u32(seed) % (bits == 4 ? 16 : 256)) - (bits == 4 ? 8 : 128));
        for (size_t i = 0; i < inputs; ++i) x[i] = uint8_t(random_u32(seed));
        std::memset(x + inputs, 0, padded_rows * k + 4096 - inputs);
        std::memset(w, 0, packed_bytes + 4096);
        const double r0 = now();
        if (bits == 8) {
            std::memcpy(w, reference_weights.data(), weights);
            check(sdkl_cpu_rm_to_wh_i8_inplace(n, k, reinterpret_cast<int8_t *>(w)), "i8 weight layout");
        } else {
            // The shipped API example uses inner dimension, then output columns.
            check(sdkl_cpu_rm_to_wh_i4(w, reference_weights.data(), k, n), "i4 weight layout");
        }
        emit({{"event", "layout"}, {"seconds", now() - r0}, {"packed_bytes", packed_bytes}});
        const bool full = weights * size_t(batch) <= 16777216;
        const size_t checks = full ? outputs : std::min(size_t(128), outputs);
        std::vector<size_t> positions(checks);
        std::vector<int32_t> expected(checks);
        for (size_t i = 0; i < checks; ++i) {
            const size_t pos = full ? i : i * (outputs - 1) / (checks - 1);
            positions[i] = pos;
            const size_t row = pos / n, col = pos % n;
            int64_t sum = 0;
            for (int j = 0; j < k; ++j) sum += int64_t(x[row * k + j]) * reference_weights[col * k + j];
            require(sum >= std::numeric_limits<int32_t>::min() && sum <= std::numeric_limits<int32_t>::max(), "reference overflow");
            expected[i] = int32_t(sum);
        }
        auto compute = [&]() {
            if (bits == 8) check(sdkl_npu_mm_u8i8_i32(CDSP_DOMAIN_ID, api_batch, n, k, y, x, reinterpret_cast<int8_t *>(w)), "u8i8 matrix");
            else check(sdkl_npu_mm_u8i4_i32(CDSP_DOMAIN_ID, api_batch, n, k, y, x, w), "u8i4 matrix");
        };
        auto validate = [&]() {
            for (size_t i = 0; i < checks; ++i) {
                if (y[positions[i]] != expected[i]) {
                    emit({{"event", "mismatch"}, {"position", positions[i]}, {"actual", y[positions[i]]}, {"expected", expected[i]}});
                    throw std::runtime_error("integer reference mismatch");
                }
            }
            for (size_t i = 0; i < outputs; ++i) require(y[i] != int32_t(0x51515151), "unwritten output");
            for (size_t i = outputs; i < size_t(api_batch) * n; ++i) require(y[i] == 0, "nonzero padded output");
            const size_t guard = size_t(api_batch) * n;
            for (size_t i = guard; i < guard + 1024; ++i) require(y[i] == int32_t(0x51515151), "output guard changed");
            emit({{"event", "validation"}, {"samples", checks}, {"full", full}, {"written_elements", outputs},
                  {"zero_padding_elements", guard - outputs}, {"guard_bytes", 4096}, {"mismatches", 0}});
        };
        std::fill(y, y + padded_rows * n + 1024, int32_t(0x51515151));
        compute(); validate();
        const double begin = now();
        int completed = 0;
        for (int i = 0; i < repeats && !stopped; ++i) { compute(); ++completed; }
        const double end = now();
        validate();
        emit({{"event", "result"}, {"seconds", end - begin}, {"calls", completed},
              {"us_per_call", (end - begin) * 1e6 / std::max(1, completed)},
              {"effective_GOPS", 2.0 * batch * n * k * completed / (end - begin) / 1e9},
              {"physical_GOPS", 2.0 * api_batch * n * k * completed / (end - begin) / 1e9},
              {"stopped", bool(stopped)}, {"scope", "SDKL call including dispatch/layout/transfer, not bare HMX peak"}});
    }
    session.close();
    emit({{"event", "finished"}, {"stopped", bool(stopped)}});
    return stopped ? 2 : 0;
} catch (const std::exception & e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
}
