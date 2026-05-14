#include "device_profile.h"
#include "bench_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <oqs/oqs.h>

#define ALG_NAME  "ML-KEM-768"
#define ALG_SHORT "ML-KEM"

void run_bench_mlkem(BenchResult *out_results, int trials) {
    printf("\n----------------------------------------------\n");
    printf("  Algorithm  : %s\n", ALG_NAME);
    printf("  Device     : %s\n", DEVICE_NAME);
    printf("  Scenario   : Session key establishment (once/hour)\n");
    printf("  Trials     : %d\n", trials);
    printf("----------------------------------------------\n");

    OQS_KEM *kem = OQS_KEM_new(ALG_NAME);
    if (!kem) {
        fprintf(stderr, "[bench_mlkem] Failed to init %s\n", ALG_NAME);
        return;
    }

    uint8_t *pk   = malloc(kem->length_public_key);
    uint8_t *sk   = malloc(kem->length_secret_key);
    uint8_t *ct   = malloc(kem->length_ciphertext);
    uint8_t *ss_e = malloc(kem->length_shared_secret);
    uint8_t *ss_d = malloc(kem->length_shared_secret);

    if (!pk || !sk || !ct || !ss_e || !ss_d) {
        fprintf(stderr, "[bench_mlkem] malloc failed\n");
        goto cleanup;
    }

    size_t total_ram = kem->length_public_key
                     + kem->length_secret_key
                     + kem->length_ciphertext
                     + 2 * kem->length_shared_secret;

    OQS_KEM_keypair(kem, pk, sk);
    OQS_KEM_encaps(kem, ct, ss_e, pk);
    OQS_KEM_decaps(kem, ss_d, ct, sk);

    double t_start = now_us();
    for (int i = 0; i < trials; i++)
        OQS_KEM_keypair(kem, pk, sk);
    double t_keygen_us = (now_us() - t_start) / trials;

    BenchResult r_keygen = {
        .algorithm        = ALG_SHORT,
        .operation        = "KeyGen",
        .pk_bytes         = kem->length_public_key,
        .sk_bytes         = kem->length_secret_key,
        .ct_or_sig_bytes  = 0,
        .total_ram_bytes  = total_ram
    };
    fill_timing(&r_keygen, t_keygen_us);
    evaluate_constraints(&r_keygen, BUDGET_KEYGEN_MS, BUDGET_KEY_BYTES);
    print_result(&r_keygen);
    out_results[0] = r_keygen;

    OQS_KEM_keypair(kem, pk, sk);

    t_start = now_us();
    for (int i = 0; i < trials; i++)
        OQS_KEM_encaps(kem, ct, ss_e, pk);
    double t_encaps_us = (now_us() - t_start) / trials;

    BenchResult r_encaps = {
        .algorithm        = ALG_SHORT,
        .operation        = "Encaps",
        .pk_bytes         = kem->length_public_key,
        .sk_bytes         = 0,
        .ct_or_sig_bytes  = kem->length_ciphertext,
        .total_ram_bytes  = kem->length_public_key
                          + kem->length_ciphertext
                          + kem->length_shared_secret
    };
    fill_timing(&r_encaps, t_encaps_us);
    evaluate_constraints(&r_encaps, BUDGET_ENCAPS_MS, BUDGET_KEY_BYTES);
    print_result(&r_encaps);
    out_results[1] = r_encaps;

    t_start = now_us();
    for (int i = 0; i < trials; i++)
        OQS_KEM_decaps(kem, ss_d, ct, sk);
    double t_decaps_us = (now_us() - t_start) / trials;

    BenchResult r_decaps = {
        .algorithm        = ALG_SHORT,
        .operation        = "Decaps",
        .pk_bytes         = 0,
        .sk_bytes         = kem->length_secret_key,
        .ct_or_sig_bytes  = kem->length_ciphertext,
        .total_ram_bytes  = kem->length_secret_key
                          + kem->length_ciphertext
                          + kem->length_shared_secret
    };
    fill_timing(&r_decaps, t_decaps_us);
    evaluate_constraints(&r_decaps, BUDGET_DECAPS_MS, BUDGET_KEY_BYTES);
    print_result(&r_decaps);
    out_results[2] = r_decaps;

    OQS_KEM_keypair(kem, pk, sk);
    OQS_KEM_encaps(kem, ct, ss_e, pk);
    OQS_KEM_decaps(kem, ss_d, ct, sk);
    if (memcmp(ss_e, ss_d, kem->length_shared_secret) == 0)
        printf("\n  [OK] Session key agreement verified.\n");
    else
        printf("\n  [FAIL] Session key mismatch - implementation error.\n");

    double session_energy_uj = r_keygen.energy_uj
                             + r_encaps.energy_uj
                             + r_decaps.energy_uj;
    double sessions_per_hr   = 1.0;
    double battery_drain_pct = ((session_energy_uj / 1000.0) * sessions_per_hr)
                             / BATTERY_CAPACITY_MJ * 100.0;

    printf("\nSession energy budget\n");
    printf("  Total session energy : %.2f uJ\n", session_energy_uj);
    printf("  Battery drain/hr     : %.4f%% of CR2032\n", battery_drain_pct);
    printf("\n");

cleanup:
    free(pk); free(sk); free(ct); free(ss_e); free(ss_d);
    OQS_KEM_free(kem);
}
