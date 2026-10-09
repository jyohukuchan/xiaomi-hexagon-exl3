#include "exl3_model.h"
#include "exl3_codec.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    if (argc < 2 || argc > 5) { std::cerr << "Usage: exl3_model_info model_dir [token [fixture [fixture_column]]]\n"; return 2; }
    try {
        exl3::Model model(argv[1]);
        std::cout << "tensors=" << model.tensor_count() << " payload_bytes=" << model.payload_bytes()
                  << " bpw=" << model.bits_per_weight() << " hidden=" << model.hidden_size() << " vocab=" << model.vocab_size() << '\n';
        if (argc > 2) {
            const auto token_value = std::stoull(argv[2]);
            if (token_value >= model.vocab_size()) throw std::runtime_error("Invalid token ID");
            const auto start = std::chrono::steady_clock::now();
            const auto embedding = model.embedding(uint32_t(token_value));
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (model.embedding(uint32_t(token_value)) != embedding) throw std::runtime_error("Cache hit changed embedding");
            std::cout << "embedding_elements=" << embedding.size() << " first_lookup_ms=" << ms << " cache_hit_equal=yes\n";
            if (argc > 3) {
                std::ifstream fixture(argv[3], std::ios::binary);
                char magic[8]; uint32_t geometry[4];
                fixture.read(magic, 8); fixture.read(reinterpret_cast<char *>(geometry), 16);
                if (!fixture || std::string(magic, 8) != "EXL3REF1" || geometry[0] != model.hidden_size() ||
                    !geometry[1] || geometry[1] > 16384 || geometry[2] < 1 || geometry[2] > 8) throw std::runtime_error("Invalid embedding fixture");
                const size_t k = geometry[0], n = geometry[1], bits = geometry[2];
                const size_t column = argc > 4 ? std::stoull(argv[4]) : 0;
                if (column >= n) throw std::runtime_error("Invalid fixture column");
                fixture.seekg(std::streamoff(24 + k * n * bits / 8 + (k + n + k * n) * 2));
                std::vector<uint16_t> expected(k * n);
                fixture.read(reinterpret_cast<char *>(expected.data()), std::streamsize(expected.size() * 2));
                if (!fixture) throw std::runtime_error("Truncated embedding fixture");
                double error = 0, norm = 0;
                for (size_t row = 0; row < k; ++row) {
                    const double value = exl3::half_to_float(expected[row * n + column]);
                    const double delta = exl3::half_to_float(embedding[row]) - value;
                    error += delta * delta; norm += value * value;
                }
                const double nmse = norm ? error / norm : error;
                std::cout << "cuda_embedding_nmse=" << nmse << '\n';
                if (!std::isfinite(nmse) || nmse > 1e-6) return 1;
            }
        }
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
