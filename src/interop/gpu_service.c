#include "gpu_iface.h"
#include <HAP_power.h>
#include <qurt.h>
#include <qurt_hvx.h>
#include <qurt_memory.h>
#include <qurt_timer.h>
#include <qurt_pmu.h>
#include "gpu-pmu.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>
#include <string.h>

static int power_context;
static int pmu_active;
static unsigned pmu_set;
static uint32_t pmu_initial[8], pmu_saved_config[3], pmu_saved_stid[2];
static uint32_t pmu_expected_config[3];
static uint64_t pmu_start_ticks, pmu_start_cycles;

static void pmu_counts(uint32_t * output) {
    // Explicit IDs: QURT_PMUCNT4 does not immediately follow QURT_PMUCNT3.
    output[0] = qurt_pmu_get(QURT_PMUCNT0);
    output[1] = qurt_pmu_get(QURT_PMUCNT1);
    output[2] = qurt_pmu_get(QURT_PMUCNT2);
    output[3] = qurt_pmu_get(QURT_PMUCNT3);
    output[4] = qurt_pmu_get(QURT_PMUCNT4);
    output[5] = qurt_pmu_get(QURT_PMUCNT5);
    output[6] = qurt_pmu_get(QURT_PMUCNT6);
    output[7] = qurt_pmu_get(QURT_PMUCNT7);
}

static int pmu_config_check(uint32_t * values) {
    values[0] = qurt_pmu_get(QURT_PMUCFG);
    values[1] = qurt_pmu_get(QURT_PMUEVTCFG);
    values[2] = qurt_pmu_get(QURT_PMUEVTCFG1);
    return (values[0] & 65535u) == pmu_expected_config[0] && values[1] == pmu_expected_config[1] &&
           values[2] == pmu_expected_config[2];
}

static void pmu_stop(void) {
    if (!pmu_active) return;
    uint32_t current[3];
    // Do not overwrite a different configuration installed by another actor.
    if (!pmu_config_check(current)) { pmu_active = 0; return; }
    qurt_pmu_enable(0);
    qurt_pmu_set(QURT_PMUCFG, pmu_saved_config[0]);
    qurt_pmu_set(QURT_PMUEVTCFG, pmu_saved_config[1]);
    qurt_pmu_set(QURT_PMUEVTCFG1, pmu_saved_config[2]);
    qurt_pmu_set(QURT_PMUSTID0, pmu_saved_stid[0]);
    qurt_pmu_set(QURT_PMUSTID1, pmu_saved_stid[1]);
    pmu_active = 0;
}

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
    pmu_stop();
    return 0;
}

int gpu_iface_pmu_begin(remote_handle64 handle, uint32 event_set) {
    (void) handle;
    if (event_set >= GPU_PMU_EVENT_SETS || pmu_active) return 2;
    pmu_saved_config[0] = qurt_pmu_get(QURT_PMUCFG);
    pmu_saved_config[1] = qurt_pmu_get(QURT_PMUEVTCFG);
    pmu_saved_config[2] = qurt_pmu_get(QURT_PMUEVTCFG1);
    pmu_saved_stid[0] = qurt_pmu_get(QURT_PMUSTID0);
    pmu_saved_stid[1] = qurt_pmu_get(QURT_PMUSTID1);
    uint32_t cfg = 0, low = 0, high = 0;
    for (unsigned i = 0; i < 8; ++i) {
        const unsigned event = gpu_pmu_events[event_set][i];
        cfg |= ((event >> 8) & 3u) << (i * 2);
        if (i < 4) low |= (event & 255u) << (i * 8);
        else high |= (event & 255u) << ((i - 4) * 8);
    }
    qurt_pmu_set(QURT_PMUCFG, cfg);
    qurt_pmu_set(QURT_PMUEVTCFG, low);
    qurt_pmu_set(QURT_PMUEVTCFG1, high);
    pmu_expected_config[0] = cfg;
    pmu_expected_config[1] = low;
    pmu_expected_config[2] = high;
    qurt_pmu_enable(1);
    pmu_active = 1;
    pmu_set = event_set;
    pmu_counts(pmu_initial);
    pmu_start_cycles = qurt_get_core_pcycles();
    pmu_start_ticks = qurt_sysclock_get_hw_ticks();
    return 0;
}

static __attribute__((noinline)) HVX_Vector pmu_reduce_hvx(const unsigned char * input, int size) {
    HVX_Vector sum = Q6_V_vzero();
    for (int offset = 0; offset < size; offset += 128)
        sum = Q6_V_vxor_VV(sum, *(const volatile HVX_Vector *) (input + offset));
    asm volatile("syncht" ::: "memory");
    return sum;
}

int gpu_iface_pmu_read(remote_handle64 handle, uint32 mode, uint32 seed,
        const unsigned char * input, int input_len, unsigned char * output, int output_len) {
    (void) handle;
    if (!pmu_active) return 2;
    if (mode > 3 || !input || !output || input_len < 128 || input_len > 32 * 1024 * 1024 ||
        input_len % 128 || ((uintptr_t) input & 127) || output_len != sizeof(gpu_pmu_report)) {
        pmu_stop(); return 2;
    }
    gpu_pmu_report report;
    memset(&report, 0, sizeof(report));
    report.version = GPU_PMU_VERSION; report.bytes = input_len; report.mode = mode; report.event_set = pmu_set;
    memcpy(report.events, gpu_pmu_events[pmu_set], sizeof(report.events));
    memcpy(report.saved_config, pmu_saved_config, sizeof(report.saved_config));
    if (!pmu_config_check(report.configured)) report.invalid_flags |= 1;
    uint32_t entry[8];
    pmu_counts(entry);
    report.cycles[0] = qurt_get_core_pcycles() - pmu_start_cycles;
    report.ticks[0] = qurt_sysclock_get_hw_ticks() - pmu_start_ticks;
    for (unsigned i = 0; i < 8; ++i) report.counters[0][i] = entry[i] - pmu_initial[i];
    int status = qurt_hvx_lock(QURT_HVX_MODE_128B);
    if (status) { pmu_stop(); return status; }
    if (mode != 3) status = qurt_mem_cache_clean((qurt_addr_t) input, input_len, QURT_MEM_CACHE_INVALIDATE, QURT_MEM_DCACHE);
    if (status) { qurt_hvx_unlock(); pmu_stop(); return status; }
    for (unsigned pass = 0; pass < 2; ++pass) {
        uint32_t before[8], after[8];
        HVX_Vector result = Q6_V_vzero();
        uint32_t scalar[32] __attribute__((aligned(128))) = {0};
        if (!pmu_config_check(report.final_config)) report.invalid_flags |= 2u << (pass * 2);
        pmu_counts(before);
        const uint64_t c0 = qurt_get_core_pcycles(), t0 = qurt_sysclock_get_hw_ticks();
        if (mode == 1 || mode == 3) result = pmu_reduce_hvx(input, input_len);
        if (mode == 2) {
            const volatile uint32_t * data = (const volatile uint32_t *) input;
            for (unsigned i = 0; i < (unsigned) input_len / 4; ++i) scalar[i & 31] ^= data[i];
        }
        asm volatile("syncht" ::: "memory");
        report.ticks[pass + 1] = qurt_sysclock_get_hw_ticks() - t0;
        report.cycles[pass + 1] = qurt_get_core_pcycles() - c0;
        pmu_counts(after);
        if (!pmu_config_check(report.final_config)) report.invalid_flags |= 4u << (pass * 2);
        for (unsigned i = 0; i < 8; ++i) report.counters[pass + 1][i] = after[i] - before[i];
        if (!pass) {
            report.ticks[3] = qurt_sysclock_get_hw_ticks() - pmu_start_ticks;
            report.cycles[3] = qurt_get_core_pcycles() - pmu_start_cycles;
            for (unsigned i = 0; i < 8; ++i) report.counters[3][i] = after[i] - pmu_initial[i];
        }
        if (mode == 1 || mode == 3) memcpy(report.checksum[pass], &result, 128);
        if (mode == 2) memcpy(report.checksum[pass], scalar, 128);
    }
    // Full validation after both measured reads; catches stale seeds/corrupt data
    // even where an XOR reduction could collide.
    const uint32_t * data = (const uint32_t *) input;
    for (unsigned i = 0; i < (unsigned) input_len / 4; ++i)
        report.mismatches += data[i] != i * 0x9e3779b9u + seed;
    qurt_hvx_unlock();
    pmu_stop();
    memcpy(output, &report, sizeof(report));
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
