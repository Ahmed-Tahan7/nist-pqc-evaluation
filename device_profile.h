#ifndef DEVICE_PROFILE_H
#define DEVICE_PROFILE_H

#include <stddef.h>
#include <stdint.h>

#define DEVICE_NAME            "Nordic nRF52840 (Cortex-M4F @ 64 MHz)"

#define DEVICE_CPU_MHZ         64
#define DEVICE_RAM_KB          256
#define DEVICE_FLASH_KB        1024
#define DEVICE_ACTIVE_POWER_MW 14.4
#define DEVICE_BAUD_BPS        2000000

#define BUDGET_KEYGEN_MS       2000
#define BUDGET_ENCAPS_MS       500
#define BUDGET_DECAPS_MS       500
#define BUDGET_SIGN_MS         300
#define BUDGET_VERIFY_MS       300

#define BUDGET_RAM_CRYPTO_KB   128
#define BUDGET_KEY_BYTES       4096
#define BUDGET_SIG_BYTES       4096

#define HEARTBEAT_PAYLOAD_BYTES 71

#define BUDGET_ENERGY_PER_OP_UJ  3500.0
#define BATTERY_CAPACITY_MJ      2538000.0

// BLE 4.2 effective throughput (bytes/sec, accounting for ATT overhead).
#define BLE_EFFECTIVE_BPS        25000.0

static inline double estimate_ble_delay_ms(size_t bytes)
{
    return (bytes / BLE_EFFECTIVE_BPS) * 1000.0;
}

typedef struct {
    const char *algorithm;
    const char *operation;

    double host_time_us;
    double device_cycles;
    double device_time_ms;

    size_t pk_bytes;
    size_t sk_bytes;
    size_t ct_or_sig_bytes;
    size_t total_ram_bytes;

    double energy_uj;

    int pass_timing;
    int pass_memory;
    int pass_energy;
    int pass_overall;
} BenchResult;

static inline void evaluate_constraints(BenchResult *r,
                                        double budget_ms,
                                        size_t budget_sig_bytes)
{
    r->pass_timing = (r->device_time_ms <= budget_ms);
    r->pass_memory = (r->total_ram_bytes <= (size_t)(BUDGET_RAM_CRYPTO_KB * 1024))
                   && (r->ct_or_sig_bytes <= budget_sig_bytes);
    r->pass_energy = (r->energy_uj <= BUDGET_ENERGY_PER_OP_UJ);
    r->pass_overall = r->pass_timing && r->pass_memory && r->pass_energy;
}

#endif
