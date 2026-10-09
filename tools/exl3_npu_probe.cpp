#include "exl3_codec.h"
#include "exl3_iface.h"
#include <remote.h>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

int main() {
    remote_rpc_control_unsigned_module control{};
    control.domain = 3;
    control.enable = 1;
    int status = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &control, sizeof(control));
    if (status) { std::cerr << "Unsigned module session setup failed: " << status << '\n'; return 1; }
    remote_handle64 handle;
    status = exl3_iface_open("file:///libexl3_htp.so?exl3_iface_skel_handle_invoke&_modver=1.0&_dom=cdsp", &handle);
    if (status) { std::cerr << "Hexagon open failed: " << status << '\n'; return 1; }
    constexpr unsigned k = 256, n = 384;
    uint32_t random = 0x78ef12ab;
    bool ok = true;
    for (unsigned bits = 1; bits <= 8; ++bits) {
        std::vector<uint16_t> packed(k * n * bits / 16), result(k * n), expected(k * n);
        for (auto & value : packed) {
            random ^= random << 13; random ^= random >> 17; random ^= random << 5;
            value = uint16_t(random);
        }
        for (unsigned cb = 0; cb < 3; ++cb) {
            exl3::decode_inner(packed.data(), k, n, bits, static_cast<exl3::Codebook>(cb), expected.data());
            uint64 cycles;
            const auto start = std::chrono::steady_clock::now();
            status = exl3_iface_decode(handle, bits, cb, k, n,
                reinterpret_cast<const unsigned char *>(packed.data()), int(packed.size() * 2),
                reinterpret_cast<unsigned char *>(result.data()), int(result.size() * 2), &cycles);
            const double elapsed = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
            size_t mismatches = 0;
            if (!status) for (size_t i = 0; i < expected.size(); ++i) mismatches += expected[i] != result[i];
            ok = ok && !status && !mismatches;
            std::cout << "bits=" << bits << " cb=" << cb << " status=" << status
                      << " mismatches=" << mismatches << " cycles=" << (status ? 0 : cycles)
                      << " rpc_us=" << elapsed << '\n';
        }
        if (bits == 4 || bits == 6 || bits == 8) {
            std::vector<uint16_t> suh(k), svh(n), weight(k * n);
            for (size_t i = 0; i < suh.size(); ++i) suh[i] = exl3::float_to_half(float(int(i % 7) - 3) / 7);
            for (size_t i = 0; i < svh.size(); ++i) svh[i] = exl3::float_to_half(float(int(i % 9) - 4) / 9);
            exl3::reconstruct(packed.data(), suh.data(), svh.data(), k, n, bits, exl3::Codebook::mul1, weight.data());
            for (unsigned batch : {1u, 4u, 8u}) {
                std::vector<float> input(batch * k), actual(batch * n);
                for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 13) % 17) - 8) / 17;
                uint64 cycles;
                status = exl3_iface_matmul(handle, bits, 2, k, n, batch,
                    reinterpret_cast<const unsigned char *>(packed.data()), int(packed.size() * 2),
                    reinterpret_cast<const unsigned char *>(suh.data()), int(suh.size() * 2),
                    reinterpret_cast<const unsigned char *>(svh.data()), int(svh.size() * 2),
                    reinterpret_cast<const unsigned char *>(input.data()), int(input.size() * 4),
                    reinterpret_cast<unsigned char *>(actual.data()), int(actual.size() * 4), &cycles);
                double error = 0, norm = 0;
                bool finite = true;
                for (unsigned b = 0; b < batch; ++b) for (unsigned col = 0; col < n; ++col) {
                    double expected_value = 0;
                    for (unsigned row = 0; row < k; ++row) expected_value += double(input[b * k + row]) * exl3::half_to_float(weight[row * n + col]);
                    const double delta = actual[b * n + col] - expected_value;
                    finite = finite && std::isfinite(actual[b * n + col]);
                    error += delta * delta; norm += expected_value * expected_value;
                }
                const double nmse = norm ? error / norm : error;
                ok = ok && !status && finite && nmse < 1e-6;
                std::cout << "matmul bits=" << bits << " batch=" << batch << " status=" << status
                          << " cycles=" << (status ? 0 : cycles) << " nmse=" << nmse << '\n';
            }
        }
    }
    exl3_iface_close(handle);
    std::cout << (ok ? "PASS" : "FAIL") << ": Hexagon scalar decoder vs portable reference\n";
    return ok ? 0 : 1;
}
