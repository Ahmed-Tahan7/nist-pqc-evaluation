#include "device_profile.h"
#include "bench_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <oqs/oqs.h>

void run_bench_mlkem  (BenchResult *out, int trials);
void run_bench_mldsa  (BenchResult *out, int trials);
void run_bench_slhdsa (BenchResult *out, int trials);

#define OPS_PER_ALG 3

static void resolve_csv_path(char *out, size_t out_size)
{
    const char *filename = "benchmark_results.csv";

    if (access("CMakeLists.txt", F_OK) == 0) {
        snprintf(out, out_size, "%s", filename);
        return;
    }

    if (access("../CMakeLists.txt", F_OK) == 0) {
        snprintf(out, out_size, "../%s", filename);
        return;
    }

    snprintf(out, out_size, "%s", filename);
}

int main(void) {
    srand((unsigned int)time(NULL));
    OQS_init();

    const int trials = 1000;

    printf("\nPQC Heartbeat IoT Benchmark - %s\n", DEVICE_NAME);
    printf("NIST FIPS 203 / 204 / 205\n");
    printf("Simulated ARM Cortex-M4 @ 80 MHz, 96 KB SRAM\n\n");
    printf("  Device   : %s\n", DEVICE_NAME);
    printf("  Scenario : Cardiac patch transmitting heartbeat data\n");
    printf("             over BLE to a hospital gateway.\n");
    printf("             HR = 60 BPM (1 packet/sec at rest)\n");
    printf("             Each packet = %d bytes, signed per transmission.\n",
           HEARTBEAT_PAYLOAD_BYTES);
    printf("             Session key established once every 60 min.\n\n");

    BenchResult mlkem_res[OPS_PER_ALG];
    BenchResult mldsa_res[OPS_PER_ALG];
    BenchResult slhdsa_res[OPS_PER_ALG];

    run_bench_mlkem(mlkem_res, trials);
    run_bench_mldsa(mldsa_res, trials);
    run_bench_slhdsa(slhdsa_res, trials);

    printf("\n\n");
    print_summary_header();

    print_summary_row(&mlkem_res[0]);
    print_summary_row(&mlkem_res[1]);
    print_summary_row(&mlkem_res[2]);

    print_summary_row(&mldsa_res[0]);
    print_summary_row(&mldsa_res[1]);
    print_summary_row(&mldsa_res[2]);

    print_summary_row(&slhdsa_res[0]);
    print_summary_row(&slhdsa_res[1]);
    print_summary_row(&slhdsa_res[2]);

    print_summary_footer();

    BenchResult all_results[OPS_PER_ALG * 3];
    size_t idx = 0;
    all_results[idx++] = mlkem_res[0];
    all_results[idx++] = mlkem_res[1];
    all_results[idx++] = mlkem_res[2];
    all_results[idx++] = mldsa_res[0];
    all_results[idx++] = mldsa_res[1];
    all_results[idx++] = mldsa_res[2];
    all_results[idx++] = slhdsa_res[0];
    all_results[idx++] = slhdsa_res[1];
    all_results[idx++] = slhdsa_res[2];

    char csv_path[256];
    resolve_csv_path(csv_path, sizeof(csv_path));
    if (write_results_csv(csv_path, all_results, idx) == 0)
        printf("CSV written: %s\n", csv_path);

    for (size_t i = 0; i < idx; i++) {
        free(all_results[i].trial_times_us);
    }

    OQS_destroy();
    return EXIT_SUCCESS;
}
