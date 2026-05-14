#ifndef BENCH_UTILS_H
#define BENCH_UTILS_H

#include "device_profile.h"
#include <stdio.h>
#include <time.h>

static inline double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

#define HOST_REF_MHZ 2000.0
#define CORTEX_M4_PENALTY 8.0

static inline void fill_timing(BenchResult *r, double host_us)
{
    r->host_time_us   = host_us;
    r->device_cycles  = host_us * HOST_REF_MHZ * CORTEX_M4_PENALTY;
    r->device_time_ms = r->device_cycles / (DEVICE_CPU_MHZ * 1e3);
    r->energy_uj      = r->device_time_ms * DEVICE_ACTIVE_POWER_MW;
}

static inline void print_result(const BenchResult *r)
{
    printf("\n%s %s\n", r->algorithm, r->operation);
    printf("  Host time          : %8.2f us\n",  r->host_time_us);
    printf("  Device time (est.) : %8.2f ms\n", r->device_time_ms);
    printf("  Device cycles      : %8.0f\n",     r->device_cycles);
    printf("  Energy / op        : %8.2f uJ (budget %.0f uJ)\n",
        r->energy_uj, BUDGET_ENERGY_PER_OP_UJ);
    printf("  PK size            : %6zu B\n",    r->pk_bytes);
    printf("  SK size            : %6zu B\n",    r->sk_bytes);
    printf("  CT / Sig size      : %6zu B\n",    r->ct_or_sig_bytes);
    printf("  Total RAM (crypto) : %6zu B (budget %d KB)\n",
        r->total_ram_bytes, BUDGET_RAM_CRYPTO_KB);
    printf("  Timing OK          : %s\n", r->pass_timing ? "OK" : "FAIL");
    printf("  Memory OK          : %s\n", r->pass_memory ? "OK" : "FAIL");
    printf("  Energy OK          : %s\n", r->pass_energy ? "OK" : "FAIL");
    printf("  Overall            : %s\n",
        r->pass_overall ? "FEASIBLE on " DEVICE_NAME
                  : "EXCEEDS device constraints");
}

static inline void print_summary_header(void)
{
    printf("\nBenchmark summary - %s\n", DEVICE_NAME);
    printf("Heartbeat scenario: sign every packet @ 60 BPM; KEM once per session\n");
    printf("Algorithm     Op         Dev ms   uJ   Sig/CT   RAM KB  Feasible\n");
    printf("------------------------------------------------------------------\n");
}

static inline void print_summary_row(const BenchResult *r)
{
    printf("%-12s %-8s %8.2f %5.1f %7zu %7zu %s\n",
           r->algorithm,
           r->operation,
           r->device_time_ms,
           r->energy_uj,
           r->ct_or_sig_bytes,
           r->total_ram_bytes / 1024,
           r->pass_overall ? "YES" : "NO");
}

static inline void print_summary_footer(void)
{
    printf("\nNote: Device time is lower-bound estimate (host-scaled).\n");
    printf("Real Cortex-M4 without FPU optimization: ~5-10x slower.\n");
    printf("SLH-DSA signing may exceed budgets on real HW.\n\n");
}

#endif
