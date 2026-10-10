// Project-authored adapter; the separately licensed HexKL library is not vendored.
#include "int_hmx_iface.h"
#include "hexkl_micro.h"
#include <HAP_compute_res.h>
#include <HAP_power.h>
#include <HAP_farf.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdlib.h>
#include <string.h>

static int power_context;
int int_hmx_iface_open(const char * uri, remote_handle64 * handle) {
    (void) uri;
    HAP_power_request_t p;
    memset(&p, 0, sizeof(p));
    p.type = HAP_power_set_HVX; p.hvx.power_up = 1;
    int s = HAP_power_set(&power_context, &p);
    if (s) return s;
    memset(&p, 0, sizeof(p));
    p.type = HAP_power_set_HMX; p.hmx.power_up = 1;
    s = HAP_power_set(&power_context, &p);
    if (s) return s;
    // A normal performance vote, retaining DCVS and all system thermal policies.
    memset(&p, 0, sizeof(p));
    p.type = HAP_power_set_DCVS_v3;
    p.dcvs_v3.set_dcvs_enable = 1; p.dcvs_v3.dcvs_enable = 1;
    p.dcvs_v3.set_core_params = 1;
    p.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_SVS;
    p.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    p.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    p.dcvs_v3.set_bus_params = 1;
    p.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_SVS;
    p.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    p.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    s = HAP_power_set(&power_context, &p);
    if (!s) *handle = 1;
    return s;
}
int int_hmx_iface_close(remote_handle64 h) { (void) h; return 0; }

static inline HVX_Vector splat(float f) { int32_t b; memcpy(&b, &f, 4); return Q6_V_vsplat_R(b); }
typedef int32_t i32x32 __attribute__((vector_size(128)));
typedef float f32x32 __attribute__((vector_size(128)));
static inline HVX_Vector vfadd(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(a, b)); }
static inline HVX_Vector vfmul(HVX_Vector a, HVX_Vector b) { return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(a, b)); }

int int_hmx_iface_mat(remote_handle64 handle, uint32 bits, uint32 batch, uint32 n, uint32 k,
        uint32 scaled, uint32 repeats, const unsigned char * activation, int activation_len,
        const unsigned char * weights, int weights_len, const unsigned char * scales, int scales_len,
        unsigned char * output, int output_len, uint64 * pack_ticks, uint64 * compute_ticks, uint32 * arch) {
    (void) handle;
    FARF(ALWAYS, "int_hmx: enter mat");
    const uint32 m = (batch + 63) / 64 * 64, groups = k / 32;
    // Scales: group-major W scales, group-major W offsets, row-major A scales.
    const size_t scale_count = (size_t) n * groups * 2 + (size_t) m * groups;
    if ((bits != 4 && bits != 8) || !batch || batch > 512 || !n || n > 6144 || n % 32 || !k || k > 14336 || k % 32 ||
        scaled > 1 || !repeats || repeats > 100 || !activation || !weights || !scales || !output ||
        activation_len != (int) (m * k) || weights_len != (int) (n * k) || scales_len != (int) (scale_count * 4) ||
        output_len != (int) (m * n * 4 + 4096) || ((uintptr_t) output & 127) || ((uintptr_t) scales & 127)) return 2;
    int major, minor, patch, version_arch;
    char pre[HEXKL_PREREL_STR_LEN];
    int status = hexkl_micro_get_version(&major, &minor, &patch, pre, &version_arch);
    FARF(ALWAYS, "int_hmx: version status=%d arch=%d", status, version_arch);
    if (status) return status;
    *arch = version_arch;
    compute_res_attr_t attr;
    HAP_compute_res_attr_init(&attr);
    HAP_compute_res_attr_set_vtcm_param_v2(&attr, 1024 * 1024, 1024 * 1024, 1024 * 1024);
    HAP_compute_res_attr_set_hmx_param(&attr, 1);
    unsigned context = HAP_compute_res_acquire(&attr, 1000000), size = 0;
    FARF(ALWAYS, "int_hmx: VTCM context=%u", context);
    void * memory = NULL;
    if (!context) return 10;
    status = HAP_compute_res_attr_get_vtcm_ptr_v2(&attr, &memory, &size);
    if (status || size < 1024 * 1024) { HAP_compute_res_release(context); return 11; }
    uint8_t * v = memory;
    const uint32 woff = 64 * k, result = woff + 2048, flat = result + 8192, accum = flat + 8192;
    const uint32 cfg = accum + 8192;
    if (cfg + hexkl_micro_hmx_config_size() > size || ((uintptr_t) v & 2047)) { HAP_compute_res_release(context); return 12; }
    uint8_t * packed = memalign(128, (size_t) n * k * bits / 8);
    int32_t * sums = memalign(128, (size_t) n * groups * 4);
    int32_t * tile_flat = (int32_t *) (v + flat);
    float * tile_accum = (float *) (v + accum);
    int32_t * activation_sums = memalign(128, (size_t) m * groups * 4);
    int hmx_locked = 0, hvx_locked = 0;
    if (!packed || !sums || !activation_sums) { status = 13; goto done; }
    status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    FARF(ALWAYS, "int_hmx: HVX lock=%d", status);
    if (status) goto done;
    hvx_locked = 1;
    status = HAP_compute_res_hmx_lock(context);
    FARF(ALWAYS, "int_hmx: HMX lock=%d", status);
    if (status) goto done;
    hmx_locked = 1;
    status = hexkl_micro_hmx_setup_acc_read_int32(v, cfg);
    FARF(ALWAYS, "int_hmx: config=%d", status);
    if (status) goto done;
    const int8_t * w = (const int8_t *) weights;
    const unsigned tile_bytes = bits == 8 ? 1024 : 512;
    uint64 t0 = qurt_sysclock_get_hw_ticks();
    for (uint32 col = 0; col < n / 32; ++col) {
        for (uint32 g = 0; g < groups; ++g) {
            status = bits == 8 ? hexkl_micro_hmx_rm_to_wh_i8(v, woff, w, g, col, n) :
                                hexkl_micro_hmx_rm_to_wh_i4(v, woff, w, g, col, n);
            if (status) goto done;
            memcpy(packed + ((size_t) col * groups + g) * tile_bytes, v + woff, tile_bytes);
            for (uint32 c = 0; c < 32; ++c) {
                int sum = 0;
                for (uint32 j = 0; j < 32; ++j) sum += w[((size_t) g * 32 + j) * n + col * 32 + c];
                sums[g * n + col * 32 + c] = sum * 128;
            }
        }
    }
    *pack_ticks = qurt_sysclock_get_hw_ticks() - t0;
    FARF(ALWAYS, "int_hmx: packed");
    const float * ws = (const float *) scales, * wo = ws + (size_t) n * groups, * as = wo + (size_t) n * groups;
    t0 = qurt_sysclock_get_hw_ticks();
    for (uint32 rep = 0; rep < repeats; ++rep) {
        if (scaled) {
            if (batch < m) memset(output + (size_t) batch * n * 4, 0, (size_t) (m - batch) * n * 4);
            for (uint32 r = 0; r < m; ++r) for (uint32 g = 0; g < groups; ++g) {
                int sum = -4096;
                for (uint32 j = 0; j < 32; ++j) sum += activation[(size_t) r * k + g * 32 + j];
                activation_sums[r * groups + g] = sum;
            }
        }
        for (uint32 row = 0; row < m / 64; ++row) {
            const uint32 valid_rows = batch - row * 64 < 64 ? batch - row * 64 : 64;
            for (uint32 g = 0; g < groups; ++g) {
                status = hexkl_micro_hmx_copy_submatrix_to_8b_activation(v, g * 2048, activation, row, g, m, k);
                if (status) goto done;
            }
            for (uint32 col = 0; col < n / 32; ++col) {
                if (scaled) memset(tile_accum, 0, 8192);
                else hexkl_micro_hmx_acc_clear_int32();
                for (uint32 g = 0; g < groups; ++g) {
                    memcpy(v + woff, packed + ((size_t) col * groups + g) * tile_bytes, tile_bytes);
                    if (scaled) hexkl_micro_hmx_acc_clear_int32();
                    status = bits == 8 ? hexkl_micro_hmx_mm_u8i8(v, g * 2048, woff) : hexkl_micro_hmx_mm_u8i4(v, g * 2048, woff);
                    if (status) goto done;
                    if (scaled) {
                        status = hexkl_micro_hmx_acc_read_int32(v, cfg, result);
                        if (status) goto done;
                        status = hexkl_micro_hmx_copy_32b_to_submatrix(v, result, tile_flat, 0, 0, valid_rows, 32);
                        if (status) goto done;
                        asm volatile("syncht" ::: "memory");
                        const HVX_Vector sw = *(const HVX_Vector *) (ws + g * n + col * 32);
                        const HVX_Vector ow = *(const HVX_Vector *) (wo + g * n + col * 32);
                        const HVX_Vector correction = *(const HVX_Vector *) (sums + g * n + col * 32);
                        for (uint32 r = 0; r < valid_rows; ++r) {
                            const HVX_Vector dot = Q6_Vw_vsub_VwVw(*(const HVX_Vector *) (tile_flat + r * 32), correction);
                            const f32x32 converted = __builtin_convertvector((i32x32) dot, f32x32);
                            const HVX_Vector df = (HVX_Vector) converted;
                            HVX_Vector value = vfmul(df, sw);
                            const int asum = activation_sums[(row * 64 + r) * groups + g];
                            value = vfadd(value, vfmul(ow, splat((float) asum)));
                            value = vfmul(value, splat(as[(row * 64 + r) * groups + g]));
                            *(HVX_Vector *) (tile_accum + r * 32) = vfadd(*(const HVX_Vector *) (tile_accum + r * 32), value);
                        }
                        asm volatile("syncht" ::: "memory");
                    }
                }
                if (scaled) {
                    for (uint32 r = 0; r < valid_rows; ++r)
                        memcpy(output + (((size_t) row * 64 + r) * n + col * 32) * 4, tile_accum + r * 32, 128);
                } else {
                    status = hexkl_micro_hmx_acc_read_int32(v, cfg, result);
                    if (status) goto done;
                    status = hexkl_micro_hmx_copy_32b_to_submatrix(v, result, (int32_t *) output, row, col, m, n);
                    if (status) goto done;
                }
            }
        }
    }
    *compute_ticks = qurt_sysclock_get_hw_ticks() - t0;
done:
    if (hmx_locked) { int unlock = HAP_compute_res_hmx_unlock(context); if (!status) status = unlock; }
    if (hvx_locked) qurt_hvx_unlock();
    free(packed); free(sums); free(activation_sums);
    HAP_compute_res_release(context);
    return status;
}
