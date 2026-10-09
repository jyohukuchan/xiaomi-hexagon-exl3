#include "exl3_iface.h"
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <HAP_compute_res.h>
#include <HAP_power.h>
#include "exl3-hvx.h"
#include "exl3-hadamard.h"

int exl3_iface_open(const char * uri, remote_handle64 * handle) {
    (void) uri;
    *handle = 1;
    return 0;
}

int exl3_iface_close(remote_handle64 handle) {
    (void) handle;
    return 0;
}

static uint16_t codebook_value(uint16_t state, unsigned cb) {
    uint32_t product = state;
    _Float16 decoded;
    if (cb == 2) {
        product *= 0x83dcd12du;
        const unsigned sum = (product & 255) + ((product >> 8) & 255) +
                             ((product >> 16) & 255) + (product >> 24);
        const uint16_t hbits = (uint16_t) (0x6400u + sum), invbits = 0x1eee, biasbits = 0xc931;
        _Float16 h, inv, bias;
        memcpy(&h, &hbits, 2); memcpy(&inv, &invbits, 2); memcpy(&bias, &biasbits, 2);
        decoded = (_Float16) fmaf((float) h, (float) inv, (float) bias);
    } else {
        product = cb == 1 ? product * 0xcbac1fedu : product * 89226354u + 64248484u;
        product = (product & 0x8fff8fffu) ^ 0x3b603b60u;
        const uint16_t lobits = (uint16_t) product, hibits = (uint16_t) (product >> 16);
        _Float16 lo, hi;
        memcpy(&lo, &lobits, 2); memcpy(&hi, &hibits, 2);
        decoded = (_Float16) ((float) lo + (float) hi);
    }
    uint16_t result;
    memcpy(&result, &decoded, 2);
    return result;
}

static uint32_t word32(const unsigned char * data, unsigned index) {
    uint32_t result;
    memcpy(&result, data + index * 4, 4);
    return result;
}

static uint16_t codebooks[3][65536];
static int initialized[3];
static qurt_mutex_t codebook_mutex = QURT_MUTEX_INIT;

static void prepare_codebook(unsigned cb) {
    qurt_mutex_lock(&codebook_mutex);
    if (!initialized[cb]) {
        for (unsigned state = 0; state < 65536; ++state) codebooks[cb][state] = codebook_value((uint16_t) state, cb);
        initialized[cb] = 1;
    }
    qurt_mutex_unlock(&codebook_mutex);
}

static inline __attribute__((always_inline)) void decode_all(unsigned bits, unsigned k, unsigned n,
        const unsigned char * packed, unsigned char * output, const uint16_t * table) {
    const unsigned words = bits * 8;
    for (unsigned rk = 0; rk < k / 16; ++rk) {
        for (unsigned cn = 0; cn < n / 16; ++cn) {
            const unsigned char * tile = packed + (rk * (n / 16) + cn) * 32 * bits;
            for (unsigned pos = 0; pos < 256; ++pos) {
                const unsigned begin = (pos + 257) * bits - 16, end = begin + 16;
                const uint32_t a = word32(tile, (begin / 32) % words);
                const uint32_t b = word32(tile, ((end - 1) / 32) % words);
                const unsigned shift = (((end - 1) / 32) + 1) * 32 - end;
                const uint16_t state = (uint16_t) ((((uint64_t) a << 32) | b) >> shift);
                const unsigned lane = pos / 8, element = pos % 8;
                const unsigned row = (lane % 4) * 2 + (element & 1) + ((element & 2) ? 8 : 0);
                const unsigned col = lane / 4 + ((element & 4) ? 8 : 0);
                const uint16_t value = table[state];
                memcpy(output + ((rk * 16 + row) * n + cn * 16 + col) * 2, &value, 2);
            }
        }
    }
}

int exl3_iface_decode(remote_handle64 handle, uint32 bits, uint32 cb, uint32 k, uint32 n,
                      const unsigned char * packed, int packed_len,
                      unsigned char * output, int output_len, uint64 * cycles) {
    (void) handle;
    if (bits < 1 || bits > 8 || cb > 2 || !k || !n || k % 16 || n % 16 ||
        (uint64_t) k * n > 64 * 1024 * 1024 || packed_len != (int) ((uint64_t) k * n * bits / 8) ||
        output_len != (int) ((uint64_t) k * n * 2)) return 2;
    prepare_codebook(cb);
    const uint64_t start = qurt_get_core_pcycles();
    switch (bits) {
        case 1: decode_all(1, k, n, packed, output, codebooks[cb]); break;
        case 2: decode_all(2, k, n, packed, output, codebooks[cb]); break;
        case 3: decode_all(3, k, n, packed, output, codebooks[cb]); break;
        case 4: decode_all(4, k, n, packed, output, codebooks[cb]); break;
        case 5: decode_all(5, k, n, packed, output, codebooks[cb]); break;
        case 6: decode_all(6, k, n, packed, output, codebooks[cb]); break;
        case 7: decode_all(7, k, n, packed, output, codebooks[cb]); break;
        case 8: decode_all(8, k, n, packed, output, codebooks[cb]); break;
    }
    *cycles = qurt_get_core_pcycles() - start;
    return 0;
}

struct hvx_resources { void * memory; unsigned context; };

static int hvx_acquire(struct hvx_resources * resources) {
    memset(resources, 0, sizeof(*resources));
    HAP_power_request_t power;
    memset(&power, 0, sizeof(power));
    power.type = HAP_power_set_HVX;
    power.hvx.power_up = 1;
    int status = HAP_power_set((void *) codebooks, &power);
    if (status) return 4000 + status;
    status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (status) return 5000 + status;
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, 256 * 1024, 256 * 1024, 256 * 1024);
    resources->context = HAP_compute_res_acquire(&attr, 1000000);
    unsigned size = 0;
    if (!resources->context || HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &resources->memory, &size) || size < 256 * 1024) {
        if (resources->context) HAP_compute_res_release(resources->context);
        qurt_hvx_unlock();
        return 5;
    }
    return 0;
}

static void hvx_release(struct hvx_resources * resources) {
    HAP_compute_res_release(resources->context);
    qurt_hvx_unlock();
}

int exl3_iface_hadamard_hvx(remote_handle64 handle, uint32 normalize, const unsigned char * input, int input_len,
        unsigned char * output, int output_len, uint64 * cycles) {
    (void) handle;
    if (normalize > 1 || !input || !output || !cycles || input_len < 512 || input_len > 65536 || input_len % 512 || input_len != output_len) return 2;
    struct hvx_resources resources;
    int status = hvx_acquire(&resources);
    if (status) return status;
    float * block = (float *) resources.memory;
    const uint64_t start = qurt_get_core_pcycles();
    for (int offset = 0; offset < input_len; offset += 512) {
        memcpy(block, input + offset, 512);
        exl3_had128_hvx(block, normalize);
        memcpy(output + offset, block, 512);
    }
    *cycles = qurt_get_core_pcycles() - start;
    hvx_release(&resources);
    return 0;
}

int exl3_iface_codebook_hvx(remote_handle64 handle, uint32 mode, const unsigned char * states, int states_len,
        unsigned char * output, int output_len, uint64 * cycles) {
    (void) handle;
    if (mode > 3 || !states || !output || !cycles || states_len < 128 || states_len > 131072 || states_len % 128 || states_len != output_len) return 2;
    struct hvx_resources resources;
    int status = hvx_acquire(&resources);
    if (status) return status;
    HVX_Vector * input = (HVX_Vector *) resources.memory;
    HVX_Vector * result = input + 4;
    HVX_Vector * gathered = result + 4;
    uint16_t * table = (uint16_t *) (gathered + 5);
    for (unsigned sum = 0; sum < 1024; ++sum) table[sum] = exl3_mul1_sum_value(sum);
    const uint64_t start = qurt_get_core_pcycles();
    for (int offset = 0; offset < states_len; offset += 512) {
        const unsigned vectors = (unsigned) (states_len - offset < 512 ? states_len - offset : 512) / 128;
        memcpy(input, states + offset, vectors * 128);
        // Repeat in VTCM so host copies do not dominate this comparison.
        for (unsigned repeat = 0; repeat < 64; ++repeat) {
            if (mode == 3) {
#pragma unroll 4
                for (unsigned v = 0; v < vectors; ++v) exl3_hvx_mul1_sum_gather(((volatile HVX_Vector *) input)[v], table, gathered + v);
                exl3_hvx_sync(gathered);
#pragma unroll 4
                for (unsigned v = 0; v < vectors; ++v) ((volatile HVX_Vector *) result)[v] = gathered[v];
            } else for (unsigned v = 0; v < vectors; ++v) {
                const HVX_Vector value = ((volatile HVX_Vector *) input)[v];
                ((volatile HVX_Vector *) result)[v] = mode == 2 ? exl3_hvx_mul1_packed(value) :
                    mode == 1 ? exl3_hvx_mul1_sum_lookup(value, table, gathered + v) : exl3_hvx_mul1_reference(value);
            }
        }
        memcpy(output + offset, result, vectors * 128);
    }
    *cycles = qurt_get_core_pcycles() - start;
    hvx_release(&resources);
    return 0;
}

int exl3_iface_decode_hvx(remote_handle64 handle, uint32 bits, uint32 cb, uint32 k, uint32 n,
        const unsigned char * packed, int packed_len, unsigned char * output, int output_len, uint64 * cycles) {
    (void) handle;
    if ((bits != 4 && bits != 6 && bits != 8) || cb > 2 || !k || !n || k % 16 || n % 16 ||
        (uint64_t) k * n > 64 * 1024 * 1024 || packed_len != (int) ((uint64_t) k * n * bits / 8) ||
        output_len != (int) ((uint64_t) k * n * 2)) return 2;
    prepare_codebook(cb);
    struct hvx_resources resources;
    int status = hvx_acquire(&resources);
    if (status) return status;
    uint16_t * table = (uint16_t *) resources.memory;
    memcpy(table, codebooks[cb], 65536 * 2);
    uint16_t * tile = table + 65536;
    HVX_Vector * gathered = (HVX_Vector *) (tile + 128);
    uint16_t * decoded = (uint16_t *) (gathered + 1);
    const uint64_t start = qurt_get_core_pcycles();
    for (unsigned rk = 0; rk < k / 16; ++rk) for (unsigned cn = 0; cn < n / 16; ++cn) {
        memcpy(tile, packed + (rk * (n / 16) + cn) * 32 * bits, 32 * bits);
        exl3_hvx_decode_tile(tile, bits, cb == 2 ? NULL : table, gathered, decoded);
        for (unsigned row = 0; row < 16; ++row) memcpy(output + ((rk * 16 + row) * n + cn * 16) * 2, decoded + row * 16, 32);
    }
    *cycles = qurt_get_core_pcycles() - start;
    hvx_release(&resources);
    return 0;
}

static float read_half(const unsigned char * data, unsigned index) {
    _Float16 value;
    memcpy(&value, data + index * 2, 2);
    return (float) value;
}

static void h128(float * values, unsigned count) {
    for (unsigned block = 0; block < count; block += 128) {
        for (unsigned step = 1; step < 128; step *= 2) {
            for (unsigned base = 0; base < 128; base += step * 2) {
                for (unsigned j = 0; j < step; ++j) {
                    const float a = values[block + base + j], b = values[block + base + j + step];
                    values[block + base + j] = a + b;
                    values[block + base + j + step] = a - b;
                }
            }
        }
        for (unsigned j = 0; j < 128; ++j) values[block + j] *= 0.08838834764831844055f;
    }
}

static inline __attribute__((always_inline)) void matmul_all(unsigned bits, unsigned k, unsigned n,
        unsigned batch, const unsigned char * packed, const float * xh, float * result, const uint16_t * table) {
    uint16_t tile[256];
    for (unsigned rk = 0; rk < k / 16; ++rk) {
        for (unsigned cn = 0; cn < n / 16; ++cn) {
            decode_all(bits, 16, 16, packed + (rk * (n / 16) + cn) * 32 * bits, (unsigned char *) tile, table);
            for (unsigned row = 0; row < 16; ++row) {
                for (unsigned col = 0; col < 16; ++col) {
                    _Float16 half;
                    memcpy(&half, tile + row * 16 + col, 2);
                    const float weight = (float) half;
                    for (unsigned b = 0; b < batch; ++b) {
                        result[b * n + cn * 16 + col] += xh[b * k + rk * 16 + row] * weight;
                    }
                }
            }
        }
    }
}

int exl3_iface_matmul(remote_handle64 handle, uint32 bits, uint32 cb, uint32 k, uint32 n, uint32 batch,
        const unsigned char * packed, int packed_len, const unsigned char * suh, int suh_len,
        const unsigned char * svh, int svh_len, const unsigned char * input, int input_len,
        unsigned char * output, int output_len, uint64 * cycles) {
    (void) handle;
    if (bits < 1 || bits > 8 || cb > 2 || !k || !n || k % 128 || n % 128 || !batch || batch > 8 ||
        (uint64_t) k * n > 64 * 1024 * 1024 || packed_len != (int) ((uint64_t) k * n * bits / 8) ||
        suh_len != (int) (k * 2) || svh_len != (int) (n * 2) ||
        input_len != (int) (batch * k * 4) || output_len != (int) (batch * n * 4)) return 2;
    prepare_codebook(cb);
    float * xh = (float *) malloc(batch * k * sizeof(float));
    float * result = (float *) calloc(batch * n, sizeof(float));
    if (!xh || !result) { free(xh); free(result); return 3; }
    const uint64_t start = qurt_get_core_pcycles();
    for (unsigned b = 0; b < batch; ++b) {
        for (unsigned i = 0; i < k; ++i) {
            float value;
            memcpy(&value, input + (b * k + i) * 4, 4);
            xh[b * k + i] = value * read_half(suh, i);
        }
        h128(xh + b * k, k);
    }
    switch (bits) {
        case 1: matmul_all(1, k, n, batch, packed, xh, result, codebooks[cb]); break;
        case 2: matmul_all(2, k, n, batch, packed, xh, result, codebooks[cb]); break;
        case 3: matmul_all(3, k, n, batch, packed, xh, result, codebooks[cb]); break;
        case 4: matmul_all(4, k, n, batch, packed, xh, result, codebooks[cb]); break;
        case 5: matmul_all(5, k, n, batch, packed, xh, result, codebooks[cb]); break;
        case 6: matmul_all(6, k, n, batch, packed, xh, result, codebooks[cb]); break;
        case 7: matmul_all(7, k, n, batch, packed, xh, result, codebooks[cb]); break;
        case 8: matmul_all(8, k, n, batch, packed, xh, result, codebooks[cb]); break;
    }
    for (unsigned b = 0; b < batch; ++b) {
        h128(result + b * n, n);
        for (unsigned i = 0; i < n; ++i) result[b * n + i] *= read_half(svh, i);
    }
    memcpy(output, result, output_len);
    *cycles = qurt_get_core_pcycles() - start;
    free(xh); free(result);
    return 0;
}
