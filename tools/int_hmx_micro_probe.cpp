// Project-authored FastRPC client. HexKL remains a separate external dependency.
#include "remote.h"
#include "rpcmem.h"
#include "int_hmx_iface.h"
#include "nlohmann/json.hpp"
#include "gguf.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <time.h>
#include <unistd.h>
#include <vector>

using nlohmann::json;
static volatile sig_atomic_t stopped = 0;
static void handler(int) { stopped = 1; }
static void require(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
static void check(int s, const char * why) { if (s) throw std::runtime_error(std::string(why) + " status=" + std::to_string(s)); }
static double now() { timespec t{}; require(clock_gettime(CLOCK_BOOTTIME, &t) == 0, "clock"); return t.tv_sec + t.tv_nsec / 1e9; }
static void emit(json row) { row["boot_s"] = now(); std::cout << "INTEGER " << row.dump() << '\n'; }
static uint32_t random_u32(uint32_t & s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
struct Buffer {
    unsigned char * p;
    int bytes;
    explicit Buffer(size_t n) : p(static_cast<unsigned char *>(rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_FLAG_CACHED, int(n)))), bytes(int(n)) {
        require(p != nullptr && (uintptr_t(p) & 127) == 0, "shared allocation");
    }
    ~Buffer() { rpcmem_free(p); }
};
struct Session {
    remote_handle64 handle = 0;
    Session() {
        remote_rpc_control_unsigned_module c{}; c.domain = CDSP_DOMAIN_ID; c.enable = 1;
        check(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &c, sizeof(c)), "unsigned session");
        check(int_hmx_iface_open("file:///libint_hmx_htp.so?int_hmx_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp", &handle), "DSP open");
    }
    void close() { check(int_hmx_iface_close(handle), "DSP close"); handle = 0; }
    ~Session() { if (handle) int_hmx_iface_close(handle); }
};
int main(int argc, char ** argv) try {
    require(argc == 7 || argc == 9, "Usage: int_hmx_micro_probe BITS BATCH N K REPEATS raw|scaled|ggml [GGUF TENSOR]");
    const int bits = std::stoi(argv[1]), batch = std::stoi(argv[2]), n = std::stoi(argv[3]), k = std::stoi(argv[4]), repeats = std::stoi(argv[5]);
    const bool baseline = std::string(argv[6]) == "ggml";
    const bool scaled = baseline || std::string(argv[6]) == "scaled";
    require(scaled || std::string(argv[6]) == "raw", "invalid mode");
    require(!baseline || argc == 9, "GGML baseline requires a real tensor");
    require((bits == 4 || bits == 8) && batch > 0 && batch <= 512 && n >= 32 && n <= 6144 && n % 32 == 0 &&
        k >= 32 && k <= 14336 && k % 32 == 0 && repeats > 0 && repeats <= 100, "invalid dimensions");
    const int m = (batch + 63) / 64 * 64, groups = k / 32;
    std::cout.setf(std::ios::unitbuf);
    signal(SIGINT, handler); signal(SIGTERM, handler);
    emit({{"event", "init"}, {"pid", getpid()}, {"bits", bits}, {"batch", batch}, {"api_batch", baseline ? batch : m},
        {"n", n}, {"k", k}, {"repeats", repeats}, {"scaled", scaled}, {"engine", baseline ? "ggml" : "micro"}, {"storage_batch", m},
        {"tensor", argc == 9 ? argv[8] : "synthetic"}});
    Buffer x(size_t(m) * k), w(size_t(n) * k), scales((size_t(n) * groups * 2 + size_t(m) * groups) * 4), y(size_t(m) * n * 4 + 4096);
    auto * wi = reinterpret_cast<int8_t *>(w.p);
    auto * ws = reinterpret_cast<float *>(scales.p), * wo = ws + size_t(n) * groups, * as = wo + size_t(n) * groups;
    uint32_t seed = 0x61d0b77a;
    for (size_t i = 0; i < size_t(n) * k; ++i) wi[i] = int8_t(int(random_u32(seed) % (bits == 4 ? 16 : 256)) - (bits == 4 ? 8 : 128));
    for (size_t i = 0; i < size_t(batch) * k; ++i) x.p[i] = uint8_t(random_u32(seed));
    std::memset(x.p + size_t(batch) * k, scaled ? 128 : 0, size_t(m - batch) * k);
    for (size_t i = 0; i < size_t(n) * groups; ++i) {
        ws[i] = float(1 + random_u32(seed) % 31) / (bits == 8 ? 1024 : 64);
        wo[i] = bits == 4 ? float(int(random_u32(seed) % 17) - 8) / 64 : 0;
    }
    for (size_t i = 0; i < size_t(m) * groups; ++i) as[i] = i / groups < size_t(batch) ? float(1 + random_u32(seed) % 31) / 1024 : 0;
    ggml_type tensor_type = GGML_TYPE_F32;
    std::vector<unsigned char> tensor_data;
    if (argc == 9) {
        ggml_context * metadata = nullptr;
        const auto file = std::unique_ptr<gguf_context, decltype(&gguf_free)>(gguf_init_from_file(argv[7], {true, &metadata}), gguf_free);
        const auto tensors = std::unique_ptr<ggml_context, decltype(&ggml_free)>(metadata, ggml_free);
        require(bool(file) && bool(tensors), "GGUF metadata");
        const int64_t index = gguf_find_tensor(file.get(), argv[8]);
        require(index >= 0, "Tensor not found");
        const auto * tensor = ggml_get_tensor(tensors.get(), argv[8]);
        require(tensor && tensor->ne[0] == k && tensor->ne[1] == n && tensor->ne[2] == 1 && tensor->ne[3] == 1, "Tensor dimensions mismatch");
        tensor_type = gguf_get_tensor_type(file.get(), index);
        require(tensor_type == (bits == 8 ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_K), "Expected Q8_0 or Q4_K tensor");
        tensor_data.resize(gguf_get_tensor_size(file.get(), index));
        std::ifstream stream(argv[7], std::ios::binary);
        stream.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), index));
        require(bool(stream.read(reinterpret_cast<char *>(tensor_data.data()), tensor_data.size())), "Read tensor payload");
        auto fp16 = [](const unsigned char * p) { ggml_fp16_t v; std::memcpy(&v, p, 2); return ggml_fp16_to_fp32(v); };
        const size_t row_bytes = ggml_row_size(tensor_type, k);
        for (int c = 0; c < n; ++c) for (int g = 0; g < groups; ++g) {
            const auto * block = tensor_data.data() + size_t(c) * row_bytes + (bits == 8 ? g * 34 : (g / 8) * 144);
            int sc = 1, mn = 0;
            if (bits == 4) {
                const auto * qs = block + 4; const int j = g % 8;
                sc = j < 4 ? qs[j] & 63 : (qs[j + 4] & 15) | ((qs[j - 4] >> 6) << 4);
                mn = j < 4 ? qs[j + 4] & 63 : (qs[j + 4] >> 4) | ((qs[j] >> 6) << 4);
            }
            const float d = fp16(block) * sc;
            ws[size_t(g) * n + c] = d;
            wo[size_t(g) * n + c] = bits == 4 ? 8 * d - fp16(block + 2) * mn : 0;
            for (int j = 0; j < 32; ++j) {
                const int q = bits == 8 ? int(int8_t(block[2 + j])) : int((block[16 + (g % 8 / 2) * 32 + j] >> ((g % 2) * 4)) & 15) - 8;
                wi[(size_t(g) * 32 + j) * n + c] = int8_t(q);
            }
        }
        // This fixture's activation is already representable in block-A8 form.
        // Both engines receive the same F32 values / exact A8 decomposition.
        for (int r = 0; r < batch; ++r) for (int g = 0; g < groups; ++g) {
            for (int j = 0; j < 32; ++j) if (x.p[size_t(r) * k + g * 32 + j] == 0) x.p[size_t(r) * k + g * 32 + j] = 1;
            x.p[size_t(r) * k + g * 32] = 255;
            x.p[size_t(r) * k + g * 32 + 1] = 1;
        }
        // Verify the new signed-nibble/offset mapping against the existing GGUF decoder.
        std::vector<float> decoded(k);
        const auto * traits = ggml_get_type_traits(tensor_type);
        require(traits && traits->to_float, "Weight decoder");
        double mapping_error = 0, mapping_norm = 0;
        for (int c = 0; c < n; ++c) {
            traits->to_float(tensor_data.data() + size_t(c) * row_bytes, decoded.data(), k);
            for (int j = 0; j < k; ++j) {
                const size_t s = size_t(j / 32) * n + c;
                const double v = double(wi[size_t(j) * n + c]) * ws[s] + wo[s];
                const double e = v - decoded[j]; mapping_error += e * e; mapping_norm += double(decoded[j]) * decoded[j];
            }
        }
        require(mapping_error / std::max(1e-30, mapping_norm) < 1e-12, "GGUF affine reconstruction error");
        emit({{"event", "gguf_mapping"}, {"weight_elements", size_t(n) * k}, {"nmse", mapping_error / mapping_norm}, {"format", ggml_type_name(tensor_type)}});
    }
    std::memset(y.p, 0x51, y.bytes);
    const size_t outputs = size_t(batch) * n;
    const bool full = size_t(n) * k * batch <= 16777216;
    const size_t count = full ? outputs : std::min(size_t(128), outputs);
    std::vector<size_t> positions(count);
    std::vector<double> expected(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t pos = full ? i : i * (outputs - 1) / (count - 1), row = pos / n, col = pos % n;
        positions[i] = pos;
        double sum = 0;
        for (int g = 0; g < groups; ++g) {
            int dot = 0, asum = 0;
            for (int j = 0; j < 32; ++j) {
                const int a = int(x.p[row * k + g * 32 + j]) - (scaled ? 128 : 0);
                dot += a * wi[(size_t(g) * 32 + j) * n + col]; asum += a;
            }
            sum += scaled ? (double(dot) * ws[size_t(g) * n + col] + double(asum) * wo[size_t(g) * n + col]) * as[row * groups + g] : dot;
        }
        expected[i] = sum;
    }
    uint64 pack = 0, compute = 0;
    uint32 arch = 0;
    double begin, end;
    if (baseline) {
        ggml_backend_load_all_from_path("/data/local/tmp/llama.cpp/lib");
        const auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_init_by_name("HTP0", nullptr), ggml_backend_free);
        require(bool(backend), "HTP0 unavailable");
        const auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
        auto * wt = ggml_new_tensor_2d(ctx.get(), tensor_type, k, n);
        auto * xt = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, batch);
        auto * yt = ggml_mul_mat(ctx.get(), wt, xt);
        require(ggml_backend_supports_op(backend.get(), yt), "HTP operation not supported");
        const auto buffer = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        require(bool(buffer), "HTP allocation");
        std::vector<float> xf(size_t(batch) * k);
        for (int r = 0; r < batch; ++r) for (int j = 0; j < k; ++j) xf[size_t(r) * k + j] = (int(x.p[size_t(r) * k + j]) - 128) * as[r * groups + j / 32];
        ggml_backend_tensor_set(wt, tensor_data.data(), 0, tensor_data.size());
        ggml_backend_tensor_set(xt, xf.data(), 0, xf.size() * 4);
        auto * graph = ggml_new_graph_custom(ctx.get(), 2048, false); ggml_build_forward_expand(graph, yt);
        auto execute = [&]() { require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "HTP compute"); ggml_backend_synchronize(backend.get()); };
        execute();
        begin = now();
        for (int rep = 0; rep < repeats && !stopped; ++rep) execute();
        end = now();
        require(!stopped, "Interrupted GGML trial");
        compute = uint64((end - begin) * 19200000);
        ggml_backend_tensor_get(yt, y.p, 0, outputs * 4);
        std::memset(y.p + outputs * 4, 0, size_t(m - batch) * n * 4);
        arch = 75;
    } else {
        Session session;
        begin = now();
        check(int_hmx_iface_mat(session.handle, bits, batch, n, k, scaled, repeats, x.p, x.bytes, w.p, w.bytes, scales.p, scales.bytes,
            y.p, y.bytes, &pack, &compute, &arch), "Micro matrix");
        end = now();
        session.close();
    }
    require(arch == 75, "Unexpected DSP architecture");
    const auto * yi = reinterpret_cast<const int32_t *>(y.p);
    const auto * yf = reinterpret_cast<const float *>(y.p);
    double error = 0, norm = 0, maximum = 0;
    for (size_t i = 0; i < count; ++i) {
        const double actual = scaled ? yf[positions[i]] : yi[positions[i]];
        require(std::isfinite(actual), "Nonfinite output");
        const double delta = actual - expected[i]; error += delta * delta; norm += expected[i] * expected[i]; maximum = std::max(maximum, std::abs(delta));
        require(scaled || delta == 0, "Integer reference mismatch");
    }
    const double nmse = error / std::max(1e-30, norm);
    const double tolerance = baseline ? 5e-4 : 1e-10;
    if (nmse >= tolerance) {
        for (size_t i = 0; i < std::min(count, size_t(8)); ++i)
            emit({{"event", "mismatch"}, {"position", positions[i]}, {"expected", expected[i]}, {"actual", yf[positions[i]]}, {"nmse", nmse}});
    }
    require(nmse < tolerance, "Scaled reference error");
    for (size_t i = 0; i < size_t(m) * n; ++i) {
        require(yi[i] != int32_t(0x51515151), "Unwritten output");
        if (scaled) require(std::isfinite(yf[i]), "Nonfinite unsampled output");
        if (i >= outputs) require(scaled ? yf[i] == 0 : yi[i] == 0, "Nonzero padding");
    }
    for (size_t i = size_t(m) * n * 4; i < size_t(y.bytes); ++i) require(y.p[i] == 0x51, "Output guard overwritten");
    emit({{"event", "validation"}, {"samples", count}, {"full", full}, {"nmse", nmse}, {"max_abs", maximum}, {"guard_bytes", 4096}, {"mismatches", 0}});
    const double seconds = double(compute) / 19200000;
    emit({{"event", "result"}, {"calls", repeats}, {"seconds", seconds}, {"pack_seconds", double(pack) / 19200000},
        {"host_seconds_including_prepack", end - begin}, {"us_per_call", seconds * 1e6 / repeats}, {"arch", arch},
        {"effective_GOPS", 2.0 * batch * n * k * repeats / seconds / 1e9},
        {"physical_GOPS", baseline ? json(nullptr) : json(2.0 * m * n * k * repeats / seconds / 1e9)},
        {"stopped", bool(stopped)}, {"scope", baseline ? "GGML graph compute plus host dispatch/synchronize; tensor upload and download excluded" : "DSP Micro loop, prepacked weights, transfer and groupwise scale epilogue included when scaled; RPC and prepack excluded"}});
    emit({{"event", "finished"}, {"stopped", bool(stopped)}});
    return stopped ? 2 : 0;
} catch (const std::exception & e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
