#include "exl3_codec.h"
#include "exl3-block.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        check(exl3_row_chunk(4, 6144, 4, 8 * 1024 * 1024) == 32, "FFN row chunk capacity");
        check(exl3_row_chunk(4, 2048, 0, 8 * 1024 * 1024) == 0, "Reject no workers");
        check(exl3_row_chunk(4, 0, 4, 8 * 1024 * 1024) == 0, "Reject no input dimension");
        for (unsigned bits : {4u, 6u, 8u}) for (size_t k : {128u, 2048u, 6144u, 65536u}) {
            for (unsigned threads = 1; threads <= 4; ++threads) for (size_t budget : {131072u, 262144u, 4194304u, 8388608u}) {
                const size_t chunk = exl3_row_chunk(bits, k, threads, budget);
                check(chunk <= 32, "Row chunk bound");
                if (chunk) {
                    const size_t needed = 131072 + chunk * k * 4 + threads * (32768 + chunk * 512 + exl3_group_bytes(bits) + 640);
                    check(needed <= budget, "Row chunk fits VTCM");
                }
            }
        }
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
        rejected = false;
        try { exl3::decode_inner_slice(packed, 16, 16, 16, 16, 4, exl3::Codebook::mul1, tile); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "Reject out-of-bounds column slice");
        std::cout << "PASS: FP16 roundtrip, rounding, H128, bitrate validation, bounded VTCM row chunks\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
