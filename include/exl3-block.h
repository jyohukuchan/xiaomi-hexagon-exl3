#pragma once
#include <stdint.h>
#include <stddef.h>

#define EXL3_TYPE_4 43
#define EXL3_TYPE_6 44
#define EXL3_TYPE_8 45
#define EXL3_GROUP_DIM 128

static inline unsigned exl3_type_bits(unsigned type) {
    return type == EXL3_TYPE_4 ? 4 : type == EXL3_TYPE_6 ? 6 : type == EXL3_TYPE_8 ? 8 : 0;
}
static inline size_t exl3_group_bytes(unsigned bits) { return 2048u * bits + 512u; }
static inline size_t exl3_row_chunk(unsigned bits, size_t k, unsigned threads, size_t vtcm_size) {
    const size_t overhead = 131072u + threads * (32768u + exl3_group_bytes(bits) + 640u);
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
