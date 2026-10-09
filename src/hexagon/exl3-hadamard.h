#pragma once
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef int exl3_had_v32w __attribute__((vector_size(128)));
#define EXL3_HAD_INLINE static inline __attribute__((always_inline))

EXL3_HAD_INLINE HVX_Vector exl3_had_add(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b));
}
EXL3_HAD_INLINE HVX_Vector exl3_had_sub(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(a, b));
}
EXL3_HAD_INLINE HVX_Vector exl3_had_mul(HVX_Vector a, HVX_Vector b) {
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b));
}

EXL3_HAD_INLINE HVX_Vector exl3_had_pass_1(HVX_Vector x, HVX_VectorPred lower) {
    const exl3_had_v32w v = (exl3_had_v32w) x;
    const HVX_Vector partner = (HVX_Vector) __builtin_shufflevector(v, v, 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16,19,18,21,20,23,22,25,24,27,26,29,28,31,30);
    return Q6_V_vmux_QVV(lower, exl3_had_add(x, partner), exl3_had_sub(partner, x));
}

EXL3_HAD_INLINE HVX_Vector exl3_had_pass_2(HVX_Vector x, HVX_VectorPred lower) {
    const exl3_had_v32w v = (exl3_had_v32w) x;
    const HVX_Vector partner = (HVX_Vector) __builtin_shufflevector(v, v, 2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13,18,19,16,17,22,23,20,21,26,27,24,25,30,31,28,29);
    return Q6_V_vmux_QVV(lower, exl3_had_add(x, partner), exl3_had_sub(partner, x));
}

EXL3_HAD_INLINE HVX_Vector exl3_had_pass_4(HVX_Vector x, HVX_VectorPred lower) {
    const exl3_had_v32w v = (exl3_had_v32w) x;
    const HVX_Vector partner = (HVX_Vector) __builtin_shufflevector(v, v, 4,5,6,7,0,1,2,3,12,13,14,15,8,9,10,11,20,21,22,23,16,17,18,19,28,29,30,31,24,25,26,27);
    return Q6_V_vmux_QVV(lower, exl3_had_add(x, partner), exl3_had_sub(partner, x));
}

EXL3_HAD_INLINE HVX_Vector exl3_had_pass_8(HVX_Vector x, HVX_VectorPred lower) {
    const exl3_had_v32w v = (exl3_had_v32w) x;
    const HVX_Vector partner = (HVX_Vector) __builtin_shufflevector(v, v, 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7,24,25,26,27,28,29,30,31,16,17,18,19,20,21,22,23);
    return Q6_V_vmux_QVV(lower, exl3_had_add(x, partner), exl3_had_sub(partner, x));
}

EXL3_HAD_INLINE HVX_Vector exl3_had_pass_16(HVX_Vector x, HVX_VectorPred lower) {
    const exl3_had_v32w v = (exl3_had_v32w) x;
    const HVX_Vector partner = (HVX_Vector) __builtin_shufflevector(v, v, 16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    return Q6_V_vmux_QVV(lower, exl3_had_add(x, partner), exl3_had_sub(partner, x));
}

// One aligned block; convert each butterfly to FP32 before the next stage.
static __attribute__((noinline)) void exl3_had128_hvx(float * values, unsigned normalize) {
    HVX_Vector * x = (HVX_Vector *) values;
    HVX_Vector a = x[0], b = x[1], c = x[2], d = x[3];
    const exl3_had_v32w indices = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
    HVX_VectorPred lower;
    lower = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV((HVX_Vector) indices, Q6_V_vsplat_R(1)), Q6_V_vzero());
    a = exl3_had_pass_1(a, lower); b = exl3_had_pass_1(b, lower);
    c = exl3_had_pass_1(c, lower); d = exl3_had_pass_1(d, lower);
    lower = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV((HVX_Vector) indices, Q6_V_vsplat_R(2)), Q6_V_vzero());
    a = exl3_had_pass_2(a, lower); b = exl3_had_pass_2(b, lower);
    c = exl3_had_pass_2(c, lower); d = exl3_had_pass_2(d, lower);
    lower = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV((HVX_Vector) indices, Q6_V_vsplat_R(4)), Q6_V_vzero());
    a = exl3_had_pass_4(a, lower); b = exl3_had_pass_4(b, lower);
    c = exl3_had_pass_4(c, lower); d = exl3_had_pass_4(d, lower);
    lower = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV((HVX_Vector) indices, Q6_V_vsplat_R(8)), Q6_V_vzero());
    a = exl3_had_pass_8(a, lower); b = exl3_had_pass_8(b, lower);
    c = exl3_had_pass_8(c, lower); d = exl3_had_pass_8(d, lower);
    lower = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV((HVX_Vector) indices, Q6_V_vsplat_R(16)), Q6_V_vzero());
    a = exl3_had_pass_16(a, lower); b = exl3_had_pass_16(b, lower);
    c = exl3_had_pass_16(c, lower); d = exl3_had_pass_16(d, lower);
    const HVX_Vector p0 = exl3_had_add(a, b), p1 = exl3_had_sub(a, b);
    const HVX_Vector p2 = exl3_had_add(c, d), p3 = exl3_had_sub(c, d);
    a = exl3_had_add(p0, p2); b = exl3_had_add(p1, p3);
    c = exl3_had_sub(p0, p2); d = exl3_had_sub(p1, p3);
    if (!normalize) {
        x[0] = a; x[1] = b; x[2] = c; x[3] = d;
        return;
    }
    const HVX_Vector norm = Q6_V_vsplat_R(0x3db504f3);
    x[0] = exl3_had_mul(a, norm); x[1] = exl3_had_mul(b, norm);
    x[2] = exl3_had_mul(c, norm); x[3] = exl3_had_mul(d, norm);
}
#undef EXL3_HAD_INLINE
