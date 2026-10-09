#pragma once

#include <cstddef>
#include <cstdint>

namespace exl3 {

enum class Codebook : uint32_t { three_inst = 0, mcg = 1, mul1 = 2 };

float half_to_float(uint16_t value);
uint16_t float_to_half(float value);
uint16_t decode_codebook(uint16_t state, Codebook codebook);

// Packed tiles use upstream CUDA lane order and little-endian uint16 storage.
void decode_tile(const uint16_t * packed, unsigned bits, Codebook codebook, uint16_t * row_major);
void decode_inner(const uint16_t * packed, size_t k, size_t n, unsigned bits, Codebook codebook,
                  uint16_t * output);

// In-place orthonormal H128 on each consecutive group of 128 values.
void hadamard128(float * values, size_t count);

// Original-basis matrix: diag(suh) H128 W_hat H128 diag(svh), laid out as [k,n].
void reconstruct(const uint16_t * packed, const uint16_t * suh, const uint16_t * svh,
                 size_t k, size_t n, unsigned bits, Codebook codebook, uint16_t * output);

} // namespace exl3
