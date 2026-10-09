#pragma once
#include <stdint.h>
#include <stddef.h>

#define EXL3_TYPE_4 43
#define EXL3_TYPE_6 44
#define EXL3_TYPE_8 45
#define EXL3_GROUP_DIM 128
#define EXL3_DECODE_PAD_BYTES 128

static inline unsigned exl3_type_bits(unsigned type) {
    return type == EXL3_TYPE_4 ? 4 : type == EXL3_TYPE_6 ? 6 : type == EXL3_TYPE_8 ? 8 : 0;
}
static inline size_t exl3_group_bytes(unsigned bits) { return 2048u * bits + 512u; }
static inline size_t exl3_thread_bytes(unsigned bits, size_t rows) {
    return 32768u + EXL3_DECODE_PAD_BYTES + rows * 512u + exl3_group_bytes(bits) + 640u;
}
static inline size_t exl3_hmx_extra(size_t k, unsigned threads) { return k * 64u + 256u + 4096u + threads * (8192u + 2048u); }
static inline size_t exl3_hmx_activation_offset(size_t k, size_t rows) {
    return (rows * k * 4u + 2047u) & ~(size_t) 2047u;
}
static inline size_t exl3_hmx_thread_bytes(unsigned bits, size_t rows) {
    return ((exl3_thread_bytes(bits, rows) + 2047u) & ~(size_t) 2047u) + 8192u;
}
static inline size_t exl3_hmx_prefix_bytes(size_t k, size_t rows) {
    return (exl3_hmx_activation_offset(k, rows) + k * 64u + 256u + 2047u) & ~(size_t) 2047u;
}
static inline size_t exl3_row_chunk(unsigned bits, size_t k, unsigned threads, size_t vtcm_size) {
    const size_t overhead = threads * exl3_thread_bytes(bits, 0);
    if (!threads || !k || vtcm_size <= overhead) return 0;
    const size_t rows = (vtcm_size - overhead) / (k * 4u + threads * 512u);
    return rows < 32u ? rows : 32u;
}
static inline const uint16_t * exl3_group_su(const void * group, unsigned bits) {
    return (const uint16_t *) ((const unsigned char *) group + 2048u * bits);
}
static inline const uint16_t * exl3_group_sv(const void * group, unsigned bits) {
    return (const uint16_t *) ((const unsigned char *) group + 2048u * bits + 256u);
}

struct ggml_compute_params;
struct ggml_tensor;
#ifdef __cplusplus
extern "C" {
#endif
void ggml_exl3_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_exl3_get_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);
#ifdef __cplusplus
}
#endif
