#pragma once
#include <stdint.h>

#define GPU_PMU_VERSION 2u
#define GPU_PMU_COUNTERS 8u
#define GPU_PMU_PHASES 4u
#define GPU_PMU_EVENT_SETS 4u

// Raw V75 hardware event numbers, not itrace's architecture-neutral 0x80xx IDs.
static const uint32_t gpu_pmu_events[GPU_PMU_EVENT_SETS][GPU_PMU_COUNTERS] = {
    {0x3f, 0x40, 0x42, 0x46, 0x47, 0x48, 0x49, 0x118},
    {0x118, 0x81, 0x8a, 0x123, 0x124, 0x111, 0x7c, 0x7d},
    {0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40},
    {0x118, 0x118, 0x118, 0x118, 0x118, 0x118, 0x118, 0x118},
};

// Phases: 0=start RPC -> consume entry; 1=first read (mode 3 skips explicit
// invalidation); 2=immediate reread; 3=start RPC -> end of first read.
typedef struct gpu_pmu_report {
    uint32_t version, bytes, mode, event_set, mismatches, invalid_flags;
    uint32_t events[GPU_PMU_COUNTERS];
    uint32_t configured[3]; // PMUCFG, PMUEVTCFG, PMUEVTCFG1 readback
    uint32_t final_config[3];
    uint32_t saved_config[3];
    uint64_t ticks[GPU_PMU_PHASES], cycles[GPU_PMU_PHASES];
    uint32_t counters[GPU_PMU_PHASES][GPU_PMU_COUNTERS];
    uint32_t checksum[2][32];
} gpu_pmu_report;

#ifdef __cplusplus
static_assert(sizeof(gpu_pmu_report) == 544, "PMU report ABI");
#else
_Static_assert(sizeof(gpu_pmu_report) == 544, "PMU report ABI");
#endif
