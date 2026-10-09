#pragma once
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <stddef.h>
typedef unsigned short exl3_v64h __attribute__((vector_size(128)));
typedef exl3_v64h exl3_v64h_unaligned __attribute__((aligned(1)));
static inline void exl3_hvx_sync(void * address);
static inline HVX_Vector exl3_hvx_mul1_sum(HVX_Vector low, HVX_Vector high) {
    const HVX_Vector product = Q6_Vw_vadd_VwVw(low, Q6_Vw_vasl_VwR(high, 16));
    return Q6_Vuw_vrmpy_VubRub(product, 0x01010101);
}
static inline HVX_Vector exl3_hvx_mul1_float(HVX_Vector low, HVX_Vector high) {
    // Dot four unsigned bytes with ones instead of shift/mask/add reduction.
    const HVX_Vector sum = exl3_hvx_mul1_sum(low, high);
    // The codebook is exactly (1774 * sum - 905216) / 2^18.
    const HVX_Vector numerator = Q6_Vw_vsub_VwVw(Q6_Vw_vmpyi_VwRh(sum, 0x06ee06ee), Q6_V_vsplat_R(905216));
    // The numerator magnitude fits 20 bits; exponent rebias scales it exactly.
    return Q6_Vw_vsub_VwVw(Q6_Vsf_equals_Vw(numerator), Q6_V_vsplat_R(18u << 23));
}
static inline HVX_Vector exl3_hvx_mul1_half(HVX_Vector low, HVX_Vector high) {
    const HVX_Vector f = exl3_hvx_mul1_float(low, high);
    const HVX_Vector sign = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(f, 16), Q6_V_vsplat_R(0x8000));
    // All mul1 values are normal FP16. Round halfway cases to an even mantissa.
    HVX_Vector magnitude = Q6_V_vand_VV(f, Q6_V_vsplat_R(0x7fffffff));
    const HVX_Vector tie = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(magnitude, 13), Q6_V_vsplat_R(1));
    magnitude = Q6_Vw_vadd_VwVw(magnitude, Q6_Vw_vadd_VwVw(tie, Q6_V_vsplat_R(0xfff)));
    magnitude = Q6_Vuw_vlsr_VuwR(Q6_Vw_vsub_VwVw(magnitude, Q6_V_vsplat_R(0x38000000)), 13);
    return Q6_V_vor_VV(sign, magnitude);
}
static inline HVX_Vector exl3_hvx_mul1_reference(HVX_Vector states) {
    const HVX_VectorPair low = Q6_Wuw_vmpy_VuhRuh(states, 0xd12dd12d);
    const HVX_VectorPair high = Q6_Wuw_vmpy_VuhRuh(states, 0x83dc83dc);
    const HVX_Vector even = exl3_hvx_mul1_half(Q6_V_lo_W(low), Q6_V_lo_W(high));
    const HVX_Vector odd = exl3_hvx_mul1_half(Q6_V_hi_W(low), Q6_V_hi_W(high));
    return Q6_Vh_vshuff_Vh(Q6_Vh_vpacke_VwVw(odd, even));
}
static inline HVX_Vector exl3_hvx_mul1_round_code(HVX_Vector f) {
    const HVX_Vector magnitude = Q6_V_vand_VV(f, Q6_V_vsplat_R(0x7fffffff));
    const HVX_Vector tie = Q6_V_vand_VV(Q6_Vuw_vlsr_VuwR(magnitude, 13), Q6_V_vsplat_R(1));
    // The rounded shift supplies 4096; subtract one for an even lower mantissa.
    return Q6_Vw_vadd_VwVw(Q6_Vw_vsub_VwVw(magnitude, Q6_V_vsplat_R(0x38000001)), tie);
}
static inline HVX_Vector exl3_hvx_mul1_packed(HVX_Vector states) {
    const HVX_VectorPair low = Q6_Wuw_vmpy_VuhRuh(states, 0xd12dd12d);
    const HVX_VectorPair high = Q6_Wuw_vmpy_VuhRuh(states, 0x83dc83dc);
    const HVX_Vector even = exl3_hvx_mul1_float(Q6_V_lo_W(low), Q6_V_lo_W(high));
    const HVX_Vector odd = exl3_hvx_mul1_float(Q6_V_hi_W(low), Q6_V_hi_W(high));
    HVX_Vector result = Q6_Vh_vasr_VwVwR_rnd_sat(exl3_hvx_mul1_round_code(odd), exl3_hvx_mul1_round_code(even), 13);
    const HVX_Vector signs = Q6_Vh_vpacke_VwVw(Q6_Vuw_vlsr_VuwR(odd, 16), Q6_Vuw_vlsr_VuwR(even, 16));
    return Q6_V_vor_VV(result, Q6_V_vand_VV(Q6_Vh_vshuff_Vh(signs), Q6_Vh_vsplat_R(0x8000)));
}
static inline HVX_Vector exl3_hvx_mul1(HVX_Vector states) {
    return exl3_hvx_mul1_packed(states);
}
static inline uint16_t exl3_mul1_sum_value(unsigned sum) {
    union { float f; uint32_t bits; } value;
    value.f = (1774 * (int) sum - 905216) * 0x1p-18f;
    uint32_t magnitude = value.bits & 0x7fffffff;
    magnitude += 0xfff + ((magnitude >> 13) & 1);
    return (uint16_t) (((value.bits >> 16) & 0x8000) | ((magnitude - 0x38000000) >> 13));
}
static inline void exl3_hvx_mul1_sum_gather(HVX_Vector states, const uint16_t * table, HVX_Vector * gathered) {
    const HVX_VectorPair low = Q6_Wuw_vmpy_VuhRuh(states, 0xd12dd12d);
    const HVX_VectorPair high = Q6_Wuw_vmpy_VuhRuh(states, 0x83dc83dc);
    const HVX_Vector even = exl3_hvx_mul1_sum(Q6_V_lo_W(low), Q6_V_lo_W(high));
    const HVX_Vector odd = exl3_hvx_mul1_sum(Q6_V_hi_W(low), Q6_V_hi_W(high));
    const HVX_VectorPair offsets = Q6_W_vcombine_VV(Q6_Vw_vasl_VwR(odd, 1), Q6_Vw_vasl_VwR(even, 1));
    Q6_vgather_ARMWw(gathered, (uint32_t) table, 2047, offsets);
}
static inline HVX_Vector exl3_hvx_mul1_sum_lookup(HVX_Vector states, const uint16_t * table, HVX_Vector * gathered) {
    exl3_hvx_mul1_sum_gather(states, table, gathered);
    exl3_hvx_sync(gathered);
    return *gathered;
}
// A null table selects the exact mul1 arithmetic codebook.
static inline HVX_Vector exl3_hvx_lookup(HVX_Vector states, const uint16_t * table, HVX_Vector * gathered) {
    if (!table) return exl3_hvx_mul1(states);
    HVX_VectorPair offsets = Q6_Wuw_vunpack_Vuh(states);
    offsets = Q6_W_vcombine_VV(Q6_Vw_vasl_VwR(Q6_V_hi_W(offsets), 1), Q6_Vw_vasl_VwR(Q6_V_lo_W(offsets), 1));
    Q6_vgather_ARMWw(gathered, (uint32_t) table, 131071, offsets);
    exl3_hvx_sync(gathered);
    return Q6_Vh_vdeal_Vh(*gathered);
}
static inline void exl3_hvx_scatter(uint16_t * output, HVX_Vector offsets, HVX_Vector values) {
#ifdef EXL3_HVX_HMX_OUTPUT
    const HVX_Vector rows = Q6_Vuh_vlsr_VuhR(offsets, 5);
    const HVX_Vector cols = Q6_V_vand_VV(offsets, Q6_V_vsplat_R(0x001f001f));
    const HVX_Vector pair = Q6_Vh_vasl_VhR(Q6_Vuh_vlsr_VuhR(rows, 1), 7);
    const HVX_Vector parity = Q6_Vh_vasl_VhR(Q6_V_vand_VV(rows, Q6_V_vsplat_R(0x00010001)), 1);
    offsets = Q6_Vh_vadd_VhVh(pair, Q6_Vh_vadd_VhVh(Q6_Vh_vasl_VhR(cols, 1), parity));
    Q6_vscatter_RMVhV((uint32_t) output, 1023, offsets, values);
#elif defined(EXL3_HVX_TILE32_OUTPUT)
    const HVX_Vector rows = Q6_Vuh_vlsr_VuhR(offsets, 5);
    const HVX_Vector cols = Q6_V_vand_VV(offsets, Q6_V_vsplat_R(0x001f001f));
    offsets = Q6_Vh_vadd_VhVh(Q6_Vh_vasl_VhR(rows, 6), cols);
    Q6_vscatter_RMVhV((uint32_t) output, 1023, offsets, values);
#elif defined(EXL3_HVX_MATRIX_OUTPUT)
    const HVX_Vector rows = Q6_Vuh_vlsr_VuhR(offsets, 5);
    const HVX_Vector cols = Q6_V_vand_VV(offsets, Q6_V_vsplat_R(0x001f001f));
    offsets = Q6_Vh_vadd_VhVh(Q6_Vh_vasl_VhR(rows, 8), cols);
    Q6_vscatter_RMVhV((uint32_t) output, 4095, offsets, values);
#else
    Q6_vscatter_RMVhV((uint32_t) output, 511, offsets, values);
#endif
}
static inline void exl3_hvx_sync(void * address) {
    asm volatile("vmem(%0+#0):scatter_release\n v0 = vmem(%0+#0)\n" :: "r"(address) : "v0", "memory");
}
static inline void exl3_hvx_part_4_0(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 62,62,62,1,1,1,1,0,0,0,0,3,3,3,3,2,2,2,2,5,5,5,5,4,4,4,4,7,7,7,7,6,6,6,6,9,9,9,9,8,8,8,8,11,11,11,11,10,10,10,10,13,13,13,13,12,12,12,12,15,15,15,15,14);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 1,1,1,1,0,0,0,0,3,3,3,3,2,2,2,2,5,5,5,5,4,4,4,4,7,7,7,7,6,6,6,6,9,9,9,9,8,8,8,8,11,11,11,11,10,10,10,10,13,13,13,13,12,12,12,12,15,15,15,15,14,14,14,14);
    const exl3_v64h left = {4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0}, right = {12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0}, mask = {65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {0,32,256,288,16,48,272,304,64,96,320,352,80,112,336,368,128,160,384,416,144,176,400,432,192,224,448,480,208,240,464,496,2,34,258,290,18,50,274,306,66,98,322,354,82,114,338,370,130,162,386,418,146,178,402,434,194,226,450,482,210,242,466,498};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_4_1(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 14,14,14,17,17,17,17,16,16,16,16,19,19,19,19,18,18,18,18,21,21,21,21,20,20,20,20,23,23,23,23,22,22,22,22,25,25,25,25,24,24,24,24,27,27,27,27,26,26,26,26,29,29,29,29,28,28,28,28,31,31,31,31,30);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 17,17,17,17,16,16,16,16,19,19,19,19,18,18,18,18,21,21,21,21,20,20,20,20,23,23,23,23,22,22,22,22,25,25,25,25,24,24,24,24,27,27,27,27,26,26,26,26,29,29,29,29,28,28,28,28,31,31,31,31,30,30,30,30);
    const exl3_v64h left = {4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0}, right = {12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0}, mask = {65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {4,36,260,292,20,52,276,308,68,100,324,356,84,116,340,372,132,164,388,420,148,180,404,436,196,228,452,484,212,244,468,500,6,38,262,294,22,54,278,310,70,102,326,358,86,118,342,374,134,166,390,422,150,182,406,438,198,230,454,486,214,246,470,502};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_4_2(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 30,30,30,33,33,33,33,32,32,32,32,35,35,35,35,34,34,34,34,37,37,37,37,36,36,36,36,39,39,39,39,38,38,38,38,41,41,41,41,40,40,40,40,43,43,43,43,42,42,42,42,45,45,45,45,44,44,44,44,47,47,47,47,46);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 33,33,33,33,32,32,32,32,35,35,35,35,34,34,34,34,37,37,37,37,36,36,36,36,39,39,39,39,38,38,38,38,41,41,41,41,40,40,40,40,43,43,43,43,42,42,42,42,45,45,45,45,44,44,44,44,47,47,47,47,46,46,46,46);
    const exl3_v64h left = {4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0}, right = {12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0}, mask = {65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {8,40,264,296,24,56,280,312,72,104,328,360,88,120,344,376,136,168,392,424,152,184,408,440,200,232,456,488,216,248,472,504,10,42,266,298,26,58,282,314,74,106,330,362,90,122,346,378,138,170,394,426,154,186,410,442,202,234,458,490,218,250,474,506};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_4_3(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 46,46,46,49,49,49,49,48,48,48,48,51,51,51,51,50,50,50,50,53,53,53,53,52,52,52,52,55,55,55,55,54,54,54,54,57,57,57,57,56,56,56,56,59,59,59,59,58,58,58,58,61,61,61,61,60,60,60,60,63,63,63,63,62);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 49,49,49,49,48,48,48,48,51,51,51,51,50,50,50,50,53,53,53,53,52,52,52,52,55,55,55,55,54,54,54,54,57,57,57,57,56,56,56,56,59,59,59,59,58,58,58,58,61,61,61,61,60,60,60,60,63,63,63,63,62,62,62,62);
    const exl3_v64h left = {4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0,4,8,12,0}, right = {12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0,12,8,4,0}, mask = {65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0,65535,65535,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {12,44,268,300,28,60,284,316,76,108,332,364,92,124,348,380,140,172,396,428,156,188,412,444,204,236,460,492,220,252,476,508,14,46,270,302,30,62,286,318,78,110,334,366,94,126,350,382,142,174,398,430,158,190,414,446,206,238,462,494,222,254,478,510};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_6_0(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 94,94,1,1,1,0,0,3,3,3,2,2,2,5,5,4,4,4,7,7,7,6,6,9,9,9,8,8,8,11,11,10,10,10,13,13,13,12,12,15,15,15,14,14,14,17,17,16,16,16,19,19,19,18,18,21,21,21,20,20,20,23,23,22);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 1,1,0,0,0,3,3,3,2,2,5,5,5,4,4,4,7,7,6,6,6,9,9,9,8,8,11,11,11,10,10,10,13,13,12,12,12,15,15,15,14,14,17,17,17,16,16,16,19,19,18,18,18,21,21,21,20,20,23,23,23,22,22,22);
    const exl3_v64h left = {6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0}, right = {10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0}, mask = {65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {0,32,256,288,16,48,272,304,64,96,320,352,80,112,336,368,128,160,384,416,144,176,400,432,192,224,448,480,208,240,464,496,2,34,258,290,18,50,274,306,66,98,322,354,82,114,338,370,130,162,386,418,146,178,402,434,194,226,450,482,210,242,466,498};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_6_1(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 22,22,25,25,25,24,24,27,27,27,26,26,26,29,29,28,28,28,31,31,31,30,30,33,33,33,32,32,32,35,35,34,34,34,37,37,37,36,36,39,39,39,38,38,38,41,41,40,40,40,43,43,43,42,42,45,45,45,44,44,44,47,47,46);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 25,25,24,24,24,27,27,27,26,26,29,29,29,28,28,28,31,31,30,30,30,33,33,33,32,32,35,35,35,34,34,34,37,37,36,36,36,39,39,39,38,38,41,41,41,40,40,40,43,43,42,42,42,45,45,45,44,44,47,47,47,46,46,46);
    const exl3_v64h left = {6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0}, right = {10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0}, mask = {65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {4,36,260,292,20,52,276,308,68,100,324,356,84,116,340,372,132,164,388,420,148,180,404,436,196,228,452,484,212,244,468,500,6,38,262,294,22,54,278,310,70,102,326,358,86,118,342,374,134,166,390,422,150,182,406,438,198,230,454,486,214,246,470,502};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_6_2(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 46,46,49,49,49,48,48,51,51,51,50,50,50,53,53,52,52,52,55,55,55,54,54,57,57,57,56,56,56,59,59,58,58,58,61,61,61,60,60,63,63,63,62,62,62,65,65,64,64,64,67,67,67,66,66,69,69,69,68,68,68,71,71,70);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 49,49,48,48,48,51,51,51,50,50,53,53,53,52,52,52,55,55,54,54,54,57,57,57,56,56,59,59,59,58,58,58,61,61,60,60,60,63,63,63,62,62,65,65,65,64,64,64,67,67,66,66,66,69,69,69,68,68,71,71,71,70,70,70);
    const exl3_v64h left = {6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0}, right = {10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0}, mask = {65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {8,40,264,296,24,56,280,312,72,104,328,360,88,120,344,376,136,168,392,424,152,184,408,440,200,232,456,488,216,248,472,504,10,42,266,298,26,58,282,314,74,106,330,362,90,122,346,378,138,170,394,426,154,186,410,442,202,234,458,490,218,250,474,506};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_6_3(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 70,70,73,73,73,72,72,75,75,75,74,74,74,77,77,76,76,76,79,79,79,78,78,81,81,81,80,80,80,83,83,82,82,82,85,85,85,84,84,87,87,87,86,86,86,89,89,88,88,88,91,91,91,90,90,93,93,93,92,92,92,95,95,94);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 73,73,72,72,72,75,75,75,74,74,77,77,77,76,76,76,79,79,78,78,78,81,81,81,80,80,83,83,83,82,82,82,85,85,84,84,84,87,87,87,86,86,89,89,89,88,88,88,91,91,90,90,90,93,93,93,92,92,95,95,95,94,94,94);
    const exl3_v64h left = {6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0,6,12,2,8,14,4,10,0}, right = {10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0,10,4,14,8,2,12,6,0}, mask = {65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0,65535,65535,65535,65535,65535,65535,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {12,44,268,300,28,60,284,316,76,108,332,364,92,124,348,380,140,172,396,428,156,188,412,444,204,236,460,492,220,252,476,508,14,46,270,302,30,62,286,318,78,110,334,366,94,126,350,382,142,174,398,430,158,190,414,446,206,238,462,494,222,254,478,510};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_8_0(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 126,1,1,0,0,3,3,2,2,5,5,4,4,7,7,6,6,9,9,8,8,11,11,10,10,13,13,12,12,15,15,14,14,17,17,16,16,19,19,18,18,21,21,20,20,23,23,22,22,25,25,24,24,27,27,26,26,29,29,28,28,31,31,30);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 1,1,0,0,3,3,2,2,5,5,4,4,7,7,6,6,9,9,8,8,11,11,10,10,13,13,12,12,15,15,14,14,17,17,16,16,19,19,18,18,21,21,20,20,23,23,22,22,25,25,24,24,27,27,26,26,29,29,28,28,31,31,30,30);
    const exl3_v64h left = {8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0}, right = {8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0}, mask = {65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {0,32,256,288,16,48,272,304,64,96,320,352,80,112,336,368,128,160,384,416,144,176,400,432,192,224,448,480,208,240,464,496,2,34,258,290,18,50,274,306,66,98,322,354,82,114,338,370,130,162,386,418,146,178,402,434,194,226,450,482,210,242,466,498};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_8_1(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 30,33,33,32,32,35,35,34,34,37,37,36,36,39,39,38,38,41,41,40,40,43,43,42,42,45,45,44,44,47,47,46,46,49,49,48,48,51,51,50,50,53,53,52,52,55,55,54,54,57,57,56,56,59,59,58,58,61,61,60,60,63,63,62);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 33,33,32,32,35,35,34,34,37,37,36,36,39,39,38,38,41,41,40,40,43,43,42,42,45,45,44,44,47,47,46,46,49,49,48,48,51,51,50,50,53,53,52,52,55,55,54,54,57,57,56,56,59,59,58,58,61,61,60,60,63,63,62,62);
    const exl3_v64h left = {8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0}, right = {8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0}, mask = {65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {4,36,260,292,20,52,276,308,68,100,324,356,84,116,340,372,132,164,388,420,148,180,404,436,196,228,452,484,212,244,468,500,6,38,262,294,22,54,278,310,70,102,326,358,86,118,342,374,134,166,390,422,150,182,406,438,198,230,454,486,214,246,470,502};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_8_2(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 62,65,65,64,64,67,67,66,66,69,69,68,68,71,71,70,70,73,73,72,72,75,75,74,74,77,77,76,76,79,79,78,78,81,81,80,80,83,83,82,82,85,85,84,84,87,87,86,86,89,89,88,88,91,91,90,90,93,93,92,92,95,95,94);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 65,65,64,64,67,67,66,66,69,69,68,68,71,71,70,70,73,73,72,72,75,75,74,74,77,77,76,76,79,79,78,78,81,81,80,80,83,83,82,82,85,85,84,84,87,87,86,86,89,89,88,88,91,91,90,90,93,93,92,92,95,95,94,94);
    const exl3_v64h left = {8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0}, right = {8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0}, mask = {65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {8,40,264,296,24,56,280,312,72,104,328,360,88,120,344,376,136,168,392,424,152,184,408,440,200,232,456,488,216,248,472,504,10,42,266,298,26,58,282,314,74,106,330,362,90,122,346,378,138,170,394,426,154,186,410,442,202,234,458,490,218,250,474,506};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_part_8_3(exl3_v64h lo, exl3_v64h hi, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_v64h a = __builtin_shufflevector(lo, hi, 94,97,97,96,96,99,99,98,98,101,101,100,100,103,103,102,102,105,105,104,104,107,107,106,106,109,109,108,108,111,111,110,110,113,113,112,112,115,115,114,114,117,117,116,116,119,119,118,118,121,121,120,120,123,123,122,122,125,125,124,124,127,127,126);
    exl3_v64h b = __builtin_shufflevector(lo, hi, 97,97,96,96,99,99,98,98,101,101,100,100,103,103,102,102,105,105,104,104,107,107,106,106,109,109,108,108,111,111,110,110,113,113,112,112,115,115,114,114,117,117,116,116,119,119,118,118,121,121,120,120,123,123,122,122,125,125,124,124,127,127,126,126);
    const exl3_v64h left = {8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0}, right = {8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0,8,0}, mask = {65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0,65535,0};
    HVX_Vector states = (HVX_Vector) (((a & mask) << left) | (b >> right));
    HVX_Vector values = exl3_hvx_lookup(states, table, gathered);
    const exl3_v64h scatter = {12,44,268,300,28,60,284,316,76,108,332,364,92,124,348,380,140,172,396,428,156,188,412,444,204,236,460,492,220,252,476,508,14,46,270,302,30,62,286,318,78,110,334,366,94,126,350,382,142,174,398,430,158,190,414,446,206,238,462,494,222,254,478,510};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, values);
}
static inline void exl3_hvx_decode4(const uint16_t * packed, uint16_t * output) {
    const HVX_Vector words = *(const HVX_Vector *) packed;
    const exl3_v64h b = (exl3_v64h) Q6_V_vor_VV(Q6_Vuw_vlsr_VuwR(words, 16), Q6_Vw_vasl_VwR(words, 16));
    const exl3_v64h a = __builtin_shufflevector(b, b, 63,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62);
    const exl3_v64h scatter = {0,16,64,80,128,144,192,208,2,18,66,82,130,146,194,210,4,20,68,84,132,148,196,212,6,22,70,86,134,150,198,214,8,24,72,88,136,152,200,216,10,26,74,90,138,154,202,218,12,28,76,92,140,156,204,220,14,30,78,94,142,158,206,222};
    exl3_hvx_scatter(output, (HVX_Vector) scatter, exl3_hvx_mul1((HVX_Vector) ((a << 4) | (b >> 12))));
    exl3_hvx_scatter(output, Q6_Vh_vadd_VhVh((HVX_Vector) scatter, Q6_Vh_vsplat_R(32)), exl3_hvx_mul1((HVX_Vector) ((a << 8) | (b >> 8))));
    exl3_hvx_scatter(output, Q6_Vh_vadd_VhVh((HVX_Vector) scatter, Q6_Vh_vsplat_R(256)), exl3_hvx_mul1((HVX_Vector) ((a << 12) | (b >> 4))));
    exl3_hvx_scatter(output, Q6_Vh_vadd_VhVh((HVX_Vector) scatter, Q6_Vh_vsplat_R(288)), exl3_hvx_mul1((HVX_Vector) b));
}
// The caller must synchronize before reading any scattered output.
static inline void exl3_hvx_decode_tile_async(const uint16_t * packed, unsigned bits, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    if (!table && bits == 4) {
        exl3_hvx_decode4(packed, output);
        return;
    }
    const exl3_v64h lo = *(const exl3_v64h_unaligned *) packed;
    const exl3_v64h hi = bits == 4 ? lo : *(const exl3_v64h_unaligned *) (packed + 64);
    switch (bits) {
        case 4:
            exl3_hvx_part_4_0(lo, hi, table, gathered, output);
            exl3_hvx_part_4_1(lo, hi, table, gathered, output);
            exl3_hvx_part_4_2(lo, hi, table, gathered, output);
            exl3_hvx_part_4_3(lo, hi, table, gathered, output);
            break;
        case 6:
            exl3_hvx_part_6_0(lo, hi, table, gathered, output);
            exl3_hvx_part_6_1(lo, hi, table, gathered, output);
            exl3_hvx_part_6_2(lo, hi, table, gathered, output);
            exl3_hvx_part_6_3(lo, hi, table, gathered, output);
            break;
        case 8:
            exl3_hvx_part_8_0(lo, hi, table, gathered, output);
            exl3_hvx_part_8_1(lo, hi, table, gathered, output);
            exl3_hvx_part_8_2(lo, hi, table, gathered, output);
            exl3_hvx_part_8_3(lo, hi, table, gathered, output);
            break;
    }
}
static inline void exl3_hvx_decode_tile(const uint16_t * packed, unsigned bits, const uint16_t * table, HVX_Vector * gathered, uint16_t * output) {
    exl3_hvx_decode_tile_async(packed, bits, table, gathered, output);
    exl3_hvx_sync(output);
}
