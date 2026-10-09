#include "exl3_codec.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        for (unsigned bits = 0; bits < 65536; ++bits) {
            if ((bits & 0x7c00) != 0x7c00) {
                check(exl3::float_to_half(exl3::half_to_float(uint16_t(bits))) == bits, "FP16 roundtrip");
            }
        }
        check(exl3::float_to_half(1.00048828125f) == 0x3c00, "Round ties to even");
        check(exl3::float_to_half(1.00146484375f) == 0x3c02, "Round odd tie upward");
        check(exl3::float_to_half(65520.0f) == 0x7c00, "Overflow rounding");
        std::vector<float> values(256);
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int(i % 19) - 9) / 17;
        const auto original = values;
        exl3::hadamard128(values.data(), values.size());
        exl3::hadamard128(values.data(), values.size());
        for (size_t i = 0; i < values.size(); ++i) check(std::fabs(values[i] - original[i]) < 1e-6, "H128 inversion");
        uint16_t packed[128] = {};
        uint16_t tile[256];
        for (unsigned bits = 1; bits <= 8; ++bits) {
            for (unsigned cb = 0; cb < 3; ++cb) {
                auto codebook = static_cast<exl3::Codebook>(cb);
                exl3::decode_tile(packed, bits, codebook, tile);
                for (auto value : tile) check(value == exl3::decode_codebook(0, codebook), "Zero-ring decode");
            }
        }
        bool rejected = false;
        try { exl3::decode_tile(packed, 0, exl3::Codebook::mul1, tile); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "Reject invalid bitrate");
        std::cout << "PASS: FP16 exhaustive roundtrip, rounding, H128, integer bitrate validation\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
