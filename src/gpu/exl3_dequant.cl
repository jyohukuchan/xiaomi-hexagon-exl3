// Project-authored OpenCL reference probe, MIT. No GPU matrix multiplication.
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL FP_CONTRACT OFF

ushort codebook_value(uint state, uint cb) {
    uint x = state;
    if (cb == 2) {
        x *= 0x83dcd12du;
        uint sum = (x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24);
        // Exact FP32 numerator; convert_half_rte is the sole rounding operation.
        return as_ushort(convert_half_rte(convert_float(1774 * (int) sum - 905216) * 0x1p-18f));
    }
    x = cb == 1 ? x * 0xcbac1fedu : x * 89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    return as_ushort(convert_half_rte(convert_float(as_half((ushort) x)) +
                                      convert_float(as_half((ushort) (x >> 16)))));
}

__kernel void codebook_all(__global ushort * output, uint cb) {
    uint i = (uint) get_global_id(0);
    if (i < 65536) output[i] = codebook_value(i, cb);
}

__kernel void fill_pattern(__global uint * output, uint count, uint seed) {
    uint i = (uint) get_global_id(0);
    if (i < count) output[i] = i * 0x9e3779b9u + seed;
}

__kernel void check_pattern(__global const uint * input, __global uint * errors,
                            uint count, uint seed, uint mask) {
    uint i = (uint) get_global_id(0);
    if (i < count && input[i] != ((i * 0x9e3779b9u + seed) ^ mask)) atomic_inc(errors);
}

uint read_word(__global const ushort * input, uint index) {
    return (uint) input[index * 2] | ((uint) input[index * 2 + 1] << 16);
}

// Input: upstream packed 16x16 tiles, tile K-major/N-minor.
// Output layout 0: row-major [K,N]. Layout 1: current Hexagon HMX group layout.
__kernel void dequant_inner(__global const ushort * packed, __global ushort * output,
                            uint k, uint n, uint bits, uint cb, uint layout) {
    uint i = (uint) get_global_id(0);
    if (i >= k * n) return;
    uint tile = i >> 8, pos = i & 255u, words = bits * 8u;
    __global const ushort * src = packed + tile * (16u * bits);
    uint begin = (pos + 257u) * bits - 16u, end = begin + 16u;
    uint a = read_word(src, (begin / 32u) % words);
    uint b = read_word(src, ((end - 1u) / 32u) % words);
    uint shift = (((end - 1u) / 32u) + 1u) * 32u - end;
    uint state = (shift == 0 ? b : ((b >> shift) | (a << (32u - shift)))) & 65535u;
    uint lane = pos >> 3, element = pos & 7u;
    uint row = (lane & 3u) * 2u + (element & 1u) + ((element & 2u) ? 8u : 0u);
    uint col = lane / 4u + ((element & 4u) ? 8u : 0u);
    uint r = (tile / (n / 16u)) * 16u + row;
    uint c = (tile % (n / 16u)) * 16u + col;
    uint dst = r * n + c;
    if (layout) {
        uint group = (c / 128u) * (k / 128u) + r / 128u;
        uint rr = r & 127u, cc = c & 127u;
        dst = group * 16384u + ((cc / 32u) * 4u + rr / 32u) * 1024u +
              ((rr & 31u) / 2u) * 64u + (cc & 31u) * 2u + (rr & 1u);
    }
    output[dst] = codebook_value(state, cb);
}

// Lossless device repack: N128-major/K128-minor groups, 64 packed tiles followed
// by the existing 128+128 FP16 scales. Scales/Hadamards are not applied here;
// the HMX pipeline applies them to activations/results, as in the HVX path.
// One invocation writes one consecutive half in the HMX layout.
#define NATIVE_DEQUANT(BITS) \
__kernel void dequant_hmx_##BITS(__global const ushort * packed, __global ushort * output, uint count) { \
    uint i = (uint) get_global_id(0); \
    if (i >= count) return; \
    uint group = i >> 14, j = i & 16383u, tile32 = j >> 10, pair = (j & 1023u) >> 6; \
    uint r = (tile32 & 3u) * 32u + pair * 2u + (j & 1u); \
    uint c = (tile32 >> 2) * 32u + ((j & 63u) >> 1); \
    uint lane = (c & 7u) * 4u + ((r & 7u) >> 1); \
    uint element = (r & 1u) + ((r & 8u) >> 2) + ((c & 8u) >> 1); \
    uint pos = lane * 8u + element; \
    __global const ushort * src = packed + group * (1024u * BITS + 256u) + \
        ((r >> 4) * 8u + (c >> 4)) * (16u * BITS); \
    uint begin = (pos + 257u) * BITS - 16u, end = begin + 16u; \
    uint a = read_word(src, (begin / 32u) % (BITS * 8u)); \
    uint b = read_word(src, ((end - 1u) / 32u) % (BITS * 8u)); \
    uint shift = (((end - 1u) / 32u) + 1u) * 32u - end; \
    uint state = (shift == 0 ? b : ((b >> shift) | (a << (32u - shift)))) & 65535u; \
    output[i] = codebook_value(state, 2u); \
}
NATIVE_DEQUANT(4)
NATIVE_DEQUANT(6)
NATIVE_DEQUANT(8)
#undef NATIVE_DEQUANT
