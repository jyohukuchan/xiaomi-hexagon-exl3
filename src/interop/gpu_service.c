#include "gpu_iface.h"
#include <HAP_power.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <qurt_memory.h>
#include <qurt_timer.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

static int power_context;

int gpu_iface_open(const char * uri, remote_handle64 * handle) {
    (void) uri;
    HAP_power_request_t request;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_DCVS_v3;
    request.dcvs_v3.set_dcvs_enable = 1;
    request.dcvs_v3.dcvs_enable = 0;
    request.dcvs_v3.set_bus_params = 1;
    request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.set_core_params = 1;
    request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
    request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
    int status = HAP_power_set(&power_context, &request);
    if (status) return status;
    memset(&request, 0, sizeof(request));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = 1;
    status = HAP_power_set(&power_context, &request);
    if (status) return status;
    *handle = 1;
    return 0;
}

int gpu_iface_close(remote_handle64 handle) {
    (void) handle;
    return 0;
}

int gpu_iface_transfer(remote_handle64 handle, uint32 mode, uint32 seed,
        const unsigned char * input, int input_len, unsigned char * output, int output_len,
        uint64 * cycles, uint64 * ticks, uint32 * mismatches) {
    (void) handle;
    if (mode > 3 || !input || !output || (mode == 3 ? output_len != 128 : input_len != output_len) || input_len < 128 ||
        input_len > 64 * 1024 * 1024 || input_len % 128 ||
        ((uintptr_t) input & 127) || ((uintptr_t) output & 127)) return 2;
    *mismatches = 0;
    *cycles = *ticks = 0;
    if (mode == 0) return 0;
    int status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (status) return status;
    status = qurt_mem_cache_clean((qurt_addr_t) input, input_len, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (status) { qurt_hvx_unlock(); return status; }
    if (mode == 2) {
        const uint32_t * data = (const uint32_t *) input;
        for (unsigned i = 0; i < (unsigned) input_len / 4; ++i)
            *mismatches += data[i] != (i * 0x9e3779b9u + seed);
    }
    const HVX_Vector mask = Q6_V_vsplat_R((int) 0xa5c39e17u);
    const uint64_t c0 = qurt_get_core_pcycles(), t0 = qurt_sysclock_get_hw_ticks();
    if (mode == 3) {
        HVX_Vector sum = Q6_V_vzero();
        for (int offset = 0; offset < input_len; offset += 128)
            sum = Q6_V_vxor_VV(sum, *(const HVX_Vector *) (input + offset));
        *(HVX_Vector *) output = sum;
    } else {
        for (int offset = 0; offset < input_len; offset += 128)
            *(HVX_Vector *) (output + offset) = Q6_V_vxor_VV(*(const HVX_Vector *) (input + offset), mask);
    }
    asm volatile("syncht" ::: "memory");
    *cycles = qurt_get_core_pcycles() - c0;
    *ticks = qurt_sysclock_get_hw_ticks() - t0;
    status = qurt_mem_cache_clean((qurt_addr_t) output, output_len, QURT_MEM_CACHE_FLUSH, QURT_MEM_DCACHE);
    qurt_hvx_unlock();
    return status;
}
