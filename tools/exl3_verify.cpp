#include "exl3_codec.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// Fixture: 8-byte magic, LE uint32 k,n,bits,codebook; packed,suh,svh,inner,original FP16 payloads.
int main(int argc, char ** argv) {
    if (argc != 2) { std::cerr << "Usage: exl3_verify fixture.bin\n"; return 2; }
    try {
        std::ifstream input(argv[1], std::ios::binary);
        char magic[8];
        input.read(magic, 8);
        if (!input || std::string(magic, 8) != "EXL3REF1") throw std::runtime_error("Invalid fixture magic");
        auto read_u32 = [&]() {
            unsigned char b[4]; input.read(reinterpret_cast<char *>(b), 4);
            if (!input) throw std::runtime_error("Truncated fixture header");
            return uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
        };
        const size_t k = read_u32(), n = read_u32();
        const unsigned bits = read_u32(), cb = read_u32();
        if (!k || !n || k > 16384 || n > 16384 || k * n > 64 * 1024 * 1024 || k % 128 || n % 128 ||
            bits < 1 || bits > 8 || cb > 2) throw std::runtime_error("Unsupported fixture geometry");
        auto read = [&](size_t count) {
            std::vector<uint16_t> data(count);
            input.read(reinterpret_cast<char *>(data.data()), std::streamsize(count * 2));
            if (!input) throw std::runtime_error("Truncated fixture payload");
            return data;
        };
        const auto packed = read(k * n * bits / 16), suh = read(k), svh = read(n);
        const auto inner = read(k * n), original = read(k * n);
        char extra;
        if (input.read(&extra, 1)) throw std::runtime_error("Trailing fixture bytes");
        std::vector<uint16_t> output(k * n);
        exl3::decode_inner(packed.data(), k, n, bits, static_cast<exl3::Codebook>(cb), output.data());
        size_t mismatches = 0;
        for (size_t i = 0; i < output.size(); ++i) mismatches += output[i] != inner[i];
        if (mismatches) throw std::runtime_error("CUDA inner-weight bit mismatch: " + std::to_string(mismatches));
        exl3::reconstruct(packed.data(), suh.data(), svh.data(), k, n, bits,
                          static_cast<exl3::Codebook>(cb), output.data());
        float max_error = 0, magnitude = 0;
        double mse = 0, norm = 0;
        for (size_t i = 0; i < output.size(); ++i) {
            const float expected = exl3::half_to_float(original[i]);
            const float actual = exl3::half_to_float(output[i]);
            if (!std::isfinite(actual) || !std::isfinite(expected)) throw std::runtime_error("Non-finite fixture value");
            const float error = actual - expected;
            max_error = std::max(max_error, std::fabs(error));
            magnitude = std::max(magnitude, std::fabs(expected));
            mse += double(error) * error; norm += double(expected) * expected;
        }
        const double nmse = norm ? mse / norm : mse;
        const float relative = magnitude ? max_error / magnitude : max_error;
        std::vector<uint16_t> slice(k * 128);
        for (size_t first = 0; first < n; first += 128) {
            exl3::reconstruct_slice(packed.data(), suh.data(), svh.data(), k, n, first, 128, bits,
                                    static_cast<exl3::Codebook>(cb), slice.data());
            for (size_t row = 0; row < k; ++row) for (size_t col = 0; col < 128; ++col) {
                if (slice[row * 128 + col] != output[row * n + first + col]) {
                    throw std::runtime_error("Embedding slice differs from full reconstruction");
                }
            }
        }
        std::cout << "k=" << k << " n=" << n << " bits=" << bits << " cb=" << cb
                  << " inner_bit_exact=yes slices_bit_exact=yes max_relative=" << relative << " nmse=" << nmse << '\n';
        return relative <= 0.002f && nmse <= 1e-6 ? 0 : 1;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
