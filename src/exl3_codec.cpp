#include "exl3_codec.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace exl3 {

static void validate(unsigned bits, Codebook codebook) {
    if (bits < 1 || bits > 8) {
        throw std::invalid_argument("EXL3 integer bitrate must be in [1,8]");
    }
    if (static_cast<uint32_t>(codebook) > 2) {
        throw std::invalid_argument("Unknown EXL3 codebook");
    }
}

float half_to_float(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000u) << 16;
    const unsigned exponent = (value >> 10) & 31;
    const unsigned fraction = value & 1023;
    uint32_t word;
    if (exponent == 0) {
        const float magnitude = std::ldexp(float(fraction), -24);
        return (value & 0x8000u) ? -magnitude : magnitude;
    }
    if (exponent == 31) {
        word = sign | 0x7f800000u | (fraction << 13);
    } else {
        word = sign | ((exponent + 112) << 23) | (fraction << 13);
    }
    float result;
    std::memcpy(&result, &word, sizeof(result));
    return result;
}

static uint32_t round_even(uint32_t value, unsigned shift) {
    const uint32_t result = value >> shift;
    const uint32_t remainder = value & ((uint32_t(1) << shift) - 1);
    const uint32_t halfway = uint32_t(1) << (shift - 1);
    return result + (remainder > halfway || (remainder == halfway && (result & 1)));
}

uint16_t float_to_half(float value) {
    uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    const uint16_t sign = uint16_t((word >> 16) & 0x8000);
    const unsigned exponent = (word >> 23) & 255;
    const uint32_t fraction = word & 0x7fffff;
    if (exponent == 255) {
        return sign | (fraction ? 0x7e00 : 0x7c00);
    }
    if (exponent > 142) {
        return sign | 0x7c00;
    }
    if (exponent < 102) {
        return sign;
    }
    if (exponent < 113) {
        return sign | uint16_t(round_even(fraction | 0x800000u, 126 - exponent));
    }
    return sign | uint16_t(((exponent - 112) << 10) + round_even(fraction, 13));
}

uint16_t decode_codebook(uint16_t state, Codebook codebook) {
    uint32_t product = state;
    if (codebook == Codebook::mul1) {
        product *= 0x83dcd12du;
        const unsigned sum = (product & 255) + ((product >> 8) & 255) +
                             ((product >> 16) & 255) + (product >> 24);
        const float h = half_to_float(uint16_t(0x6400u + sum));
        return float_to_half(std::fma(h, half_to_float(0x1eee), half_to_float(0xc931)));
    }
    if (codebook == Codebook::mcg) {
        product *= 0xcbac1fedu;
    } else if (codebook == Codebook::three_inst) {
        product = product * 89226354u + 64248484u;
    } else {
        throw std::invalid_argument("Unknown EXL3 codebook");
    }
    product = (product & 0x8fff8fffu) ^ 0x3b603b60u;
    return float_to_half(half_to_float(uint16_t(product)) + half_to_float(uint16_t(product >> 16)));
}

static uint32_t word32(const uint16_t * packed, size_t index) {
    return uint32_t(packed[index * 2]) | (uint32_t(packed[index * 2 + 1]) << 16);
}

void decode_tile(const uint16_t * packed, unsigned bits, Codebook codebook, uint16_t * row_major) {
    validate(bits, codebook);
    const unsigned words = bits * 8;
    for (unsigned position = 0; position < 256; ++position) {
        const unsigned start = (position + 257) * bits - 16;
        const unsigned end = start + 16;
        const uint32_t a = word32(packed, (start / 32) % words);
        const uint32_t b = word32(packed, ((end - 1) / 32) % words);
        const unsigned shift = (((end - 1) / 32) + 1) * 32 - end;
        const uint16_t state = uint16_t(((uint64_t(a) << 32) | b) >> shift);
        const unsigned lane = position / 8;
        const unsigned element = position % 8;
        const unsigned row = (lane % 4) * 2 + (element & 1) + ((element & 2) ? 8 : 0);
        const unsigned col = lane / 4 + ((element & 4) ? 8 : 0);
        row_major[row * 16 + col] = decode_codebook(state, codebook);
    }
}

static void validate_shape(size_t k, size_t n, size_t multiple) {
    if (!k || !n || k % multiple || n % multiple || k > std::numeric_limits<size_t>::max() / n) {
        throw std::invalid_argument("Invalid EXL3 matrix dimensions");
    }
}

void decode_inner(const uint16_t * packed, size_t k, size_t n, unsigned bits, Codebook codebook,
                  uint16_t * output) {
    decode_inner_slice(packed, k, n, 0, n, bits, codebook, output);
}

void decode_inner_slice(const uint16_t * packed, size_t k, size_t n, size_t first_column,
                        size_t column_count, unsigned bits, Codebook codebook, uint16_t * output) {
    validate(bits, codebook);
    validate_shape(k, n, 16);
    if (!column_count || first_column % 16 || column_count % 16 ||
        first_column > n || column_count > n - first_column) {
        throw std::invalid_argument("Invalid EXL3 column slice");
    }
    uint16_t tile[256];
    for (size_t rk = 0; rk < k / 16; ++rk) {
        for (size_t cn = 0; cn < column_count / 16; ++cn) {
            decode_tile(packed + (rk * (n / 16) + first_column / 16 + cn) * 16 * bits, bits, codebook, tile);
            for (size_t row = 0; row < 16; ++row) {
                std::memcpy(output + (rk * 16 + row) * column_count + cn * 16, tile + row * 16, 32);
            }
        }
    }
}

void hadamard128(float * values, size_t count) {
    if (count % 128) {
        throw std::invalid_argument("H128 length must be a multiple of 128");
    }
    constexpr float scale = 0.08838834764831844055f;
    for (size_t block = 0; block < count; block += 128) {
        for (unsigned step = 1; step < 128; step *= 2) {
            for (unsigned base = 0; base < 128; base += step * 2) {
                for (unsigned j = 0; j < step; ++j) {
                    const float a = values[block + base + j];
                    const float b = values[block + base + j + step];
                    values[block + base + j] = a + b;
                    values[block + base + j + step] = a - b;
                }
            }
        }
        for (unsigned j = 0; j < 128; ++j) {
            values[block + j] *= scale;
        }
    }
}

void reconstruct(const uint16_t * packed, const uint16_t * suh, const uint16_t * svh,
                 size_t k, size_t n, unsigned bits, Codebook codebook, uint16_t * output) {
    reconstruct_slice(packed, suh, svh, k, n, 0, n, bits, codebook, output);
}

void reconstruct_slice(const uint16_t * packed, const uint16_t * suh, const uint16_t * svh,
                       size_t k, size_t n, size_t first_column, size_t column_count,
                       unsigned bits, Codebook codebook, uint16_t * output) {
    validate_shape(k, n, 128);
    if (first_column % 128 || column_count % 128) {
        throw std::invalid_argument("H128 slices must align to 128 columns");
    }
    decode_inner_slice(packed, k, n, first_column, column_count, bits, codebook, output);
    float block[128 * 128];
    float column[128];
    for (size_t rk = 0; rk < k; rk += 128) {
        for (size_t cn = 0; cn < column_count; cn += 128) {
            for (size_t row = 0; row < 128; ++row) {
                for (size_t col = 0; col < 128; ++col) {
                    block[row * 128 + col] = half_to_float(output[(rk + row) * column_count + cn + col]);
                }
            }
            hadamard128(block, 128 * 128);
            for (size_t col = 0; col < 128; ++col) {
                for (size_t row = 0; row < 128; ++row) {
                    column[row] = block[row * 128 + col];
                }
                hadamard128(column, 128);
                for (size_t row = 0; row < 128; ++row) {
                    const float value = column[row] * half_to_float(suh[rk + row]) * half_to_float(svh[first_column + cn + col]);
                    output[(rk + row) * column_count + cn + col] = float_to_half(value);
                }
            }
        }
    }
}

} // namespace exl3
