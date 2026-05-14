#include "device_profile.h"
#include "bench_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <oqs/oqs.h>

void run_bench_mlkem  (BenchResult *out, int trials);
void run_bench_mldsa  (BenchResult *out, int trials);
void run_bench_slhdsa (BenchResult *out, int trials);

#define OPS_PER_ALG 3

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

    OQS_destroy();
    return EXIT_SUCCESS;
}
