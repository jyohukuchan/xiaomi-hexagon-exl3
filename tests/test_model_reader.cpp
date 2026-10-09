#include "exl3_model.h"
#include "exl3_codec.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include <iostream>

using json = nlohmann::json;

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main() {
    std::string pattern = (std::filesystem::temp_directory_path() / "exl3-reader-XXXXXX").string();
    char * created = mkdtemp(pattern.data());
    if (!created) { std::cerr << "Cannot create test directory\n"; return 1; }
    const std::filesystem::path directory(created);
    int result = 0;
    try {
        json header = {
            {"lm_head.trellis", {{"dtype", "I16"}, {"shape", {8, 8, 64}}, {"data_offsets", {0, 8192}}}},
            {"lm_head.suh", {{"dtype", "F16"}, {"shape", {128}}, {"data_offsets", {8192, 8448}}}},
            {"lm_head.svh", {{"dtype", "F16"}, {"shape", {128}}, {"data_offsets", {8448, 8704}}}},
            {"lm_head.mul1", {{"dtype", "I32"}, {"shape", json::array()}, {"data_offsets", {8704, 8708}}}}
        };
        json manifest = {{"format", "hexagon-exl3"}, {"version", 1}, {"kv_cache", "f16"},
            {"reference_parameter_count", 16384}, {"weight_payload_bytes", 8708},
            {"input_embedding", {{"type", "tied_linear"}, {"source", "lm_head"}, {"basis", "original"},
                                 {"token_axis", 1}, {"hidden_size", 128}, {"vocab_size", 128}}}};
        std::vector<uint8_t> data(8708, 0);
        for (size_t i = 8192; i < 8704; i += 2) { data[i] = 0; data[i + 1] = 0x3c; }
        const uint32_t codebook = 0x83dcd12du;
        for (unsigned i = 0; i < 4; ++i) data[8704 + i] = uint8_t(codebook >> (i * 8));
        auto write = [&]() {
            std::ofstream weights(directory / "model.safetensors", std::ios::binary);
            std::string text = header.dump(); text.append((8 - text.size() % 8) % 8, ' ');
            uint8_t length[8];
            for (unsigned i = 0; i < 8; ++i) length[i] = uint8_t(uint64_t(text.size()) >> (i * 8));
            weights.write(reinterpret_cast<char *>(length), 8);
            weights.write(text.data(), std::streamsize(text.size()));
            weights.write(reinterpret_cast<char *>(data.data()), std::streamsize(data.size()));
            std::ofstream config(directory / "hexagon-exl3.json"); config << manifest;
        };
        write();
        {
            exl3::Model model(directory.string());
            check(model.tensor_count() == 4 && model.payload_bytes() == 8708, "Model payload inventory");
            const auto row0 = model.embedding(0), row1 = model.embedding(1);
            check(row0 == model.embedding(0), "Cache preserves embedding");
            const float code = exl3::half_to_float(exl3::decode_codebook(0, exl3::Codebook::mul1));
            check(row0[0] == exl3::float_to_half(code * 128), "Tied head basis and scale");
            for (size_t i = 1; i < row0.size(); ++i) check(exl3::half_to_float(row0[i]) == 0, "H128 column reduction");
            for (auto value : row1) check(exl3::half_to_float(value) == 0, "H128 nonzero column isolation");
            bool rejected = false;
            try { model.embedding(128); } catch (const std::invalid_argument &) { rejected = true; }
            check(rejected, "Reject token outside vocabulary");
        }
        auto rejects = [&]() {
            write();
            bool rejected = false;
            try { exl3::Model model(directory.string()); } catch (const std::exception &) { rejected = true; }
            check(rejected, "Malformed model must be rejected");
        };
        manifest["reference_parameter_count"] = 1; rejects();
        manifest["reference_parameter_count"] = 16384;
        header["lm_head.suh"]["data_offsets"] = {8190, 8446}; rejects();
        header["lm_head.suh"]["data_offsets"] = {8192, 8448};
        header["lm_head.suh"]["shape"] = {-128}; rejects();
        header["lm_head.suh"]["shape"] = {128};
        data.resize(8707); rejects();
        std::cout << "PASS: tied embedding, cache, token bounds, budget, overlapping spans, negative dimensions, truncation\n";
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n'; result = 1;
    }
    std::filesystem::remove(directory / "model.safetensors");
    std::filesystem::remove(directory / "hexagon-exl3.json");
    std::filesystem::remove(directory);
    return result;
}
